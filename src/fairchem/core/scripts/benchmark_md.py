"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure fixed-cell ASE NVE steps, energy drift and allocator growth on CPU/XPU/CUDA.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet


def _backend(device):
    if device.type == "cpu":
        return None
    if device.type not in {"cuda", "xpu"}:
        raise ValueError("MD benchmark supports cpu, cuda and xpu")
    backend = getattr(torch, device.type)
    if not backend.is_available():
        raise RuntimeError(f"Requested {device.type} device is unavailable")
    return backend


def _memory(device):
    backend = _backend(device)
    if backend is None:
        return None
    return {
        "allocated_bytes": backend.memory_allocated(device),
        "reserved_bytes": backend.memory_reserved(device),
        "peak_allocated_bytes": backend.max_memory_allocated(device),
        "peak_reserved_bytes": backend.max_memory_reserved(device),
    }


def _nitro_runtime():
    module = sys.modules.get("fairchem.core.models.uma.uma_nitro._inference")
    function = getattr(module, "runtime_cache_info", None)
    return function() if function is not None else None


def _effective_settings(calculator, require_nitro):
    predictor = getattr(calculator, "predictor", None)
    settings = getattr(predictor, "inference_settings", None)
    if require_nitro and not getattr(settings, "nitro", False):
        raise RuntimeError("Nitro was requested but is not active; benchmark stopped")
    return repr(settings) if settings is not None else None


def _resolve_device(calculator, device):
    actual = getattr(getattr(calculator, "predictor", None), "device", None)
    actual = torch.device(actual) if actual is not None else None
    requested = torch.device(device if device is not None else actual or "cpu")
    if actual is not None and requested.type != actual.type:
        raise ValueError("Benchmark device must match calculator.predictor.device")
    backend = _backend(requested)
    if backend is not None:
        index = requested.index
        if index is None:
            index = backend.current_device()
        requested = torch.device(requested.type, index)
        if actual is not None:
            actual_index = actual.index
            if actual_index is None:
                actual_index = backend.current_device()
            if index != actual_index:
                raise ValueError(
                    "Benchmark device must match calculator.predictor.device"
                )
    return requested, backend


def _validate_configuration(
    steps, warmup, sample_interval, seed, timestep_fs, temperature_K
):
    for name, value, minimum in (
        ("steps", steps, 1),
        ("warmup", warmup, 0),
        ("sample_interval", sample_interval, 1),
        ("seed", seed, 0),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    for name, value, allow_zero in (
        ("timestep_fs", timestep_fs, False),
        ("temperature_K", temperature_K, True),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"Invalid {name}")


def run_md(
    atoms,
    calculator,
    *,
    device=None,
    steps=100,
    warmup=20,
    timestep_fs=0.1,
    temperature_K=300.0,
    seed=41,
    sample_interval=10,
    require_nitro=False,
):
    """
    Run a private structure without retaining trajectories or prediction tensors.

    Each timed interval includes one ASE MD step and finite-value/energy checks,
    synchronized on the selected accelerator. Memory queries are outside timing.
    Allocated/reserved memory samples describe the PyTorch allocator, not all
    driver allocations; growth alone does not prove a leak. CPU runs have no GPU
    allocator samples. The calculator remains owned by the caller.
    """
    _validate_configuration(
        steps, warmup, sample_interval, seed, timestep_fs, temperature_K
    )
    if not len(atoms):
        raise ValueError("MD benchmark requires at least one atom")
    device, backend = _resolve_device(calculator, device)
    _effective_settings(calculator, require_nitro)
    structure = atoms.copy()
    structure.info = copy.deepcopy(atoms.info)
    structure.calc = calculator
    momentum_source = "input" if "momenta" in structure.arrays else "maxwell_boltzmann"
    if momentum_source == "maxwell_boltzmann":
        MaxwellBoltzmannDistribution(
            structure, temperature_K=temperature_K, rng=np.random.default_rng(seed)
        )
    initial_temperature_K = float(structure.get_temperature())
    dynamics = VelocityVerlet(structure, timestep=timestep_fs * units.fs)

    def synchronize():
        if backend is not None:
            backend.synchronize(device)

    def energy():
        forces = structure.get_forces()
        total = float(structure.get_potential_energy() + structure.get_kinetic_energy())
        if (
            forces.shape != (len(structure), 3)
            or not np.isfinite(forces).all()
            or not np.isfinite(structure.positions).all()
            or not math.isfinite(total)
        ):
            raise FloatingPointError("Nonfinite or malformed MD state/output")
        _effective_settings(calculator, require_nitro)
        return total

    samples, durations = [], []
    runtime_before = _nitro_runtime()
    memory_before = _memory(device)
    try:
        synchronize()
        start = time.perf_counter()
        energy()
        synchronize()
        first_evaluation_seconds = time.perf_counter() - start
        for _ in range(warmup):
            dynamics.run(1)
            energy()
        synchronize()
        if backend is not None:
            backend.reset_peak_memory_stats(device)
        reference_energy = energy()
        max_drift = 0.0
        samples.append(
            {"step": 0, "energy_eV": reference_energy, "memory": _memory(device)}
        )
        for step in range(1, steps + 1):
            synchronize()
            start = time.perf_counter()
            dynamics.run(1)
            current_energy = energy()
            synchronize()
            durations.append((time.perf_counter() - start) * 1000)
            max_drift = max(max_drift, abs(current_energy - reference_energy))
            if step % sample_interval == 0 or step == steps:
                samples.append(
                    {
                        "step": step,
                        "energy_eV": current_energy,
                        "memory": _memory(device),
                    }
                )
        mean_ms = float(np.mean(durations))
        return {
            "status": "completed",
            "device": str(device),
            "natoms": len(structure),
            "steps": int(steps),
            "warmup": int(warmup),
            "timestep_fs": timestep_fs,
            "temperature_K": temperature_K,
            "momentum_source": momentum_source,
            "initial_temperature_K": initial_temperature_K,
            "seed": int(seed),
            "effective_settings": _effective_settings(calculator, require_nitro),
            "first_evaluation_seconds": first_evaluation_seconds,
            "step_mean_ms": mean_ms,
            "step_median_ms": float(np.median(durations)),
            "step_p95_ms": float(np.percentile(durations, 95)),
            "ns_per_day_at_timestep": 86.4 * timestep_fs / mean_ms,
            "max_energy_drift_eV_per_atom": max_drift / len(structure),
            "memory_before_evaluation": memory_before,
            "samples": samples,
            "nitro_runtime_before": runtime_before,
            "nitro_runtime_after": _nitro_runtime(),
            "scope": "ASE fixed-cell NVE with output validation; no trajectory retained. "
            "GPU allocator samples exclude non-PyTorch allocations; CPU samples are null. "
            "Drift is observational, not a model accuracy or long-time stability certificate.",
        }
    finally:
        structure.calc = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="uma-s-1p2p1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-settings", default="high-precision")
    parser.add_argument(
        "--input", type=Path, help="ASE-readable input; otherwise build cubic Si"
    )
    parser.add_argument("--repeat", type=int, nargs=3, default=(5, 5, 5))
    parser.add_argument("--task", default="omat")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--timestep-fs", type=float, default=0.1)
    parser.add_argument("--temperature-K", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--edge-margin", type=float, help="Nitro capacity margin, set before model load"
    )
    parser.add_argument("--output", type=Path, default=Path("md-benchmark.json"))
    args = parser.parse_args(argv)
    report = {
        "torch": torch.__version__,
        "requested_settings": args.inference_settings,
        "requested_model": args.model,
        "requested_device": args.device,
        "task": args.task,
        "requested_steps": args.steps,
    }
    try:
        _validate_configuration(
            args.steps,
            args.warmup,
            args.sample_interval,
            args.seed,
            args.timestep_fs,
            args.temperature_K,
        )
        _backend(torch.device(args.device))
        if args.edge_margin is not None:
            if not math.isfinite(args.edge_margin) or args.edge_margin < 1:
                raise ValueError("edge-margin must be finite and >= 1")
            if "fairchem.core.models.uma.uma_nitro._inference" in sys.modules:
                raise RuntimeError(
                    "Set edge-margin in a fresh process before Nitro import"
                )
            os.environ["UMA_NITRO_EDGE_MARGIN"] = str(args.edge_margin)
        from ase.build import bulk
        from ase.io import read

        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        if any(value < 1 for value in args.repeat):
            raise ValueError("repeat dimensions must be positive")
        atoms = (
            read(args.input)
            if args.input
            else bulk("Si", cubic=True).repeat(args.repeat)
        )
        predictor = pretrained_mlip.get_predict_unit(
            args.model,
            device=args.device,
            inference_settings=args.inference_settings,
            seed=args.seed,
        )
        calculator = FAIRChemCalculator(predictor, task_name=args.task)
        report.update(
            run_md(
                atoms,
                calculator,
                device=predictor.device,
                steps=args.steps,
                warmup=args.warmup,
                timestep_fs=args.timestep_fs,
                temperature_K=args.temperature_K,
                seed=args.seed,
                sample_interval=args.sample_interval,
                require_nitro=args.inference_settings == "nitro",
            )
        )
        report["nitro_edge_margin"] = os.environ.get("UMA_NITRO_EDGE_MARGIN", "1.3")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

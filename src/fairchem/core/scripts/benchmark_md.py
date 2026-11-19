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
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import replace
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


def _host_memory():
    # Linux reports process-local RSS/PSS in KiB. Other hosts leave them unset.
    try:
        fields = {}
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            name, _, value = line.partition(":")
            if name in ("Rss", "Pss"):
                fields[name.lower() + "_bytes"] = int(value.split()[0]) * 1024
        return fields
    except (OSError, ValueError):
        return None


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
    max_samples=4096,
    timing_block=1,
    sample_stream=None,
    aggregate_only=False,
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
    if (
        type(max_samples) is not int
        or max_samples < 2
        or type(timing_block) is not int
        or timing_block < 1
    ):
        raise ValueError("max_samples must be at least two and timing_block positive")
    if not len(atoms):
        raise ValueError("MD benchmark requires at least one atom")
    device, backend = _resolve_device(calculator, device)
    initial_settings = _effective_settings(calculator, require_nitro)
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
        if require_nitro and not getattr(
            getattr(calculator.predictor, "inference_settings", None), "nitro", False
        ):
            raise RuntimeError("Nitro was requested but is not active")
        return total

    samples, durations = [], []
    runtime_before = _nitro_runtime()
    memory_before = _memory(device)
    host_before = _host_memory()
    windows = {}
    timing_rng = random.Random(seed)
    retain_interval = sample_interval * max(
        1, math.ceil(math.ceil(steps / sample_interval) / (max_samples - 1))
    )

    def start_window():
        synchronize()
        if backend is not None:
            backend.reset_peak_memory_stats(device)
        return _memory(device)

    def sample(step, total, stream):
        entry = {
            "step": step,
            "time_ps": step * timestep_fs / 1000,
            "energy_eV": total,
            "energy_eV_per_atom": total / len(structure),
            "memory": _memory(device),
        }
        if stream is not None:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        if not aggregate_only and (step % retain_interval == 0 or step == steps):
            samples.append(entry)

    try:
        windows["first"] = {"before": start_window()}
        start = time.perf_counter()
        initial_energy = energy()
        synchronize()
        first_evaluation_seconds = time.perf_counter() - start
        windows["first"]["after"] = _memory(device)
        windows["warmup"] = {"before": start_window()}
        warmup_drift = 0.0
        for _ in range(warmup):
            dynamics.run(1)
            warmup_drift = max(warmup_drift, abs(energy() - initial_energy))
        synchronize()
        windows["warmup"]["after"] = _memory(device)
        reference_energy = energy()
        windows["steady"] = {"before": start_window()}
        max_drift = total_ms = integrator_seconds = 0.0
        # Center the trajectory time and subtract its initial energy before
        # accumulating the least-squares slope, to limit cancellation.
        covariance = 0.0
        blocks = 0
        if sample_stream is not None:
            Path(sample_stream).parent.mkdir(parents=True, exist_ok=True)
        stream_context = (
            Path(sample_stream).open("w")
            if sample_stream is not None
            else nullcontext(None)
        )
        with stream_context as stream:
            sample(0, reference_energy, stream)
            for first_step in range(1, steps + 1, timing_block):
                last_step = min(first_step + timing_block - 1, steps)
                elapsed = 0.0
                for step in range(first_step, last_step + 1):
                    start = time.perf_counter()
                    dynamics.run(1)
                    integrator_seconds += time.perf_counter() - start
                    current_energy = energy()
                    if step == last_step:
                        synchronize()
                    elapsed += time.perf_counter() - start
                    difference = current_energy - reference_energy
                    max_drift = max(max_drift, abs(difference))
                    covariance += (step - steps / 2) * difference
                    if step % sample_interval == 0 or step == steps:
                        sample(step, current_energy, stream)
                blocks += 1
                total_ms += elapsed * 1000
                per_step_ms = elapsed * 1000 / (last_step - first_step + 1)
                if len(durations) < max_samples:
                    durations.append(per_step_ms)
                else:
                    slot = timing_rng.randrange(blocks)
                    if slot < max_samples:
                        durations[slot] = per_step_ms
        mean_ms = total_ms / steps
        windows["steady"]["after"] = _memory(device)
        time_variance = steps * (steps + 1) * (steps + 2) / 12
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
            "integrator_host_mean_ms": integrator_seconds * 1000 / steps,
            "timing_block": timing_block,
            "quantile_scope": "Reservoir sample of synchronized block means; timing includes state checks and excludes trajectory sample serialization",
            "quantile_sample_size": len(durations),
            "initial_settings": initial_settings,
            "allocator_windows": windows,
            "host_memory_before": host_before,
            "host_memory_after": _host_memory(),
            "warmup_max_drift_eV_per_atom": warmup_drift / len(structure),
            "energy_slope_eV_per_atom_ps": covariance
            / time_variance
            * 1000
            / timestep_fs
            / len(structure),
            "elapsed_ps": steps * timestep_fs / 1000,
            "endpoint_drift_eV_per_atom": (current_energy - reference_energy)
            / len(structure),
            "retained_sample_interval": retain_interval,
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
        "--dtype",
        choices=("float32", "float64"),
        help="Override the preset's base precision; otherwise preserve the preset",
    )
    parser.add_argument(
        "--input", type=Path, help="ASE-readable input; otherwise build cubic Si"
    )
    parser.add_argument("--repeat", type=int, nargs=3, default=(5, 5, 5))
    parser.add_argument("--task", default="omat")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=4096,
        help="Maximum retained trajectory samples and timing observations",
    )
    parser.add_argument(
        "--timing-block",
        type=int,
        default=1,
        help="MD steps per synchronized timing interval",
    )
    parser.add_argument(
        "--sample-stream",
        type=Path,
        help="Stream all sample-interval observations to JSONL",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Omit trajectory samples from the report; optional JSONL streaming is independent",
    )
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
        "requested_dtype": args.dtype,
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
        if args.max_samples < 2 or args.timing_block < 1:
            raise ValueError(
                "max-samples must be at least two and timing-block positive"
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
        from fairchem.core.units.mlip_unit.api.inference import (
            guess_inference_settings,
        )

        settings = guess_inference_settings(args.inference_settings)
        if args.dtype is not None:
            settings = replace(settings, base_precision_dtype=args.dtype)
        report["base_precision_dtype"] = str(settings.base_precision_dtype)
        if settings.base_precision_dtype == torch.float64:
            if settings.nitro:
                raise ValueError(
                    "Nitro MD does not support base_precision_dtype=float64"
                )
            if torch.device(args.device).type == "xpu":
                from fairchem.core.units.mlip_unit.inference_runtime import (
                    probe_fp64_support,
                )

                report["fp64_compute_probe"] = probe_fp64_support(args.device)

        if any(value < 1 for value in args.repeat):
            raise ValueError("repeat dimensions must be positive")
        atoms = (
            read(args.input)
            if args.input
            else bulk("Si", cubic=True).repeat(args.repeat)
        )
        load_device = torch.device(args.device)
        load_backend = _backend(load_device)
        if load_backend is not None:
            load_backend.synchronize(load_device)
            load_backend.reset_peak_memory_stats(load_device)
        load_start = time.perf_counter()
        report["load_memory"] = {"before": _memory(load_device)}
        predictor = pretrained_mlip.get_predict_unit(
            args.model,
            device=args.device,
            inference_settings=settings,
            seed=args.seed,
        )
        if load_backend is not None:
            load_backend.synchronize(load_device)
        report["load_seconds"] = time.perf_counter() - load_start
        report["load_memory"]["after"] = _memory(load_device)
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
                max_samples=args.max_samples,
                timing_block=args.timing_block,
                sample_stream=args.sample_stream,
                aggregate_only=args.aggregate_only,
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

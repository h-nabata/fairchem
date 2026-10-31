"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare explicitly enabled Series B with strict FP32 on the same structure.

No error threshold or speedup claim is implied by successful execution. Native
GPU modes must be tested on the target driver/hardware; CPU reduced-GEMM
emulation is an arithmetic reference, not a measurement of XMX acceleration.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from numbers import Integral
from pathlib import Path

import torch
from ase.io import read
from ase.stress import voigt_6_to_full_3x3_stress

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    inference_settings_xpu,
    validate_uma_atoms_data,
)
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings

_PROPERTIES = frozenset({"energy", "forces", "stress", "hessian"})
_UNITS = {
    "energy": "eV",
    "forces": "eV/angstrom",
    "stress": "eV/angstrom^3",
    "hessian": "eV/angstrom^2",
}


def _count(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _properties(properties):
    if isinstance(properties, str):
        properties = tuple(name.strip() for name in properties.split(","))
    else:
        properties = tuple(properties)
    if not properties or any(name not in _PROPERTIES for name in properties):
        raise ValueError("properties must contain energy, forces, stress or hessian")
    if len(set(properties)) != len(properties):
        raise ValueError("properties must not contain duplicates")
    return properties


def _synchronize(device):
    device = torch.device(device)
    if device.type in ("cuda", "xpu"):
        getattr(torch, device.type).synchronize(device)


def _allocator(device, *, reset=False):
    device = torch.device(device)
    if device.type not in ("cuda", "xpu"):
        return None
    backend = getattr(torch, device.type)
    if reset:
        backend.reset_peak_memory_stats(device)
    return {
        "allocated_bytes": backend.memory_allocated(device),
        "reserved_bytes": backend.memory_reserved(device),
        "peak_allocated_bytes": backend.max_memory_allocated(device),
        "peak_reserved_bytes": backend.max_memory_reserved(device),
    }


def _snapshot(prediction, properties, natoms):
    values = {}
    for name in properties:
        value = prediction.get(name)
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"Missing or non-floating prediction: {name}")
        value = value.detach().cpu().double().clone()
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite prediction: {name}")
        if name == "energy" and value.numel() == 1:
            value = value.reshape(())
        elif name == "forces" and value.shape == (natoms, 3):
            pass
        elif name == "stress" and value.shape in ((3, 3), (1, 3, 3), (9,), (1, 9)):
            value = value.reshape(3, 3)
        elif name == "stress" and value.shape in ((6,), (1, 6)):
            value = torch.from_numpy(
                voigt_6_to_full_3x3_stress(value.reshape(6).numpy())
            )
        elif name == "hessian" and value.shape in (
            (3 * natoms, 3 * natoms),
            (1, 3 * natoms, 3 * natoms),
        ):
            value = value.reshape(3 * natoms, 3 * natoms)
        else:
            raise ValueError(f"Invalid {name} shape: {tuple(value.shape)}")
        values[name] = value
    return values


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    return value


def _measure_predictor(predictor, data, properties, warmup, repeats, measure_memory):
    device = torch.device(predictor.device)
    natoms = int(data.natoms.sum().item())
    if data.natoms.numel() != 1 or natoms < 1:
        raise ValueError("Comparison requires exactly one nonempty atomic system")
    timings = []
    snapshots = None
    first_seconds = None
    memory = None
    total = 1 + warmup + repeats
    for index in range(total):
        if index == 0 or index == 1 + warmup:
            _synchronize(device)
            if measure_memory:
                current = _allocator(device, reset=True)
                if current is not None:
                    memory = {} if memory is None else memory
                    memory["first" if index == 0 else "steady"] = {"before": current}
        _synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type, enabled=False):
            prediction = predictor.predict(
                data.clone(), requested_properties=properties
            )
        _synchronize(device)
        elapsed = time.perf_counter() - start
        if index == 0:
            first_seconds = elapsed
        elif index > warmup:
            timings.append(elapsed)
        if measure_memory and memory is not None:
            window = "first" if index == 0 else "steady"
            if index == 0 or index > warmup:
                memory[window]["after"] = _allocator(device)
        # CPU copies and validation are outside the timer. Release every GPU
        # output before the next evaluation so its buffers do not inflate peaks.
        snapshots = _snapshot(prediction, properties, natoms)
        del prediction
    settings = predictor.inference_settings
    summary = {
        "device": str(device),
        "settings": _json_safe(settings.to_omegaconf()),
        "approximation_manifest": _json_safe(
            getattr(predictor, "approximation_manifest", {})
        ),
        "cpu_low_precision_emulation": bool(settings.series_b.low_precision_emulation),
        "first_evaluation_seconds": first_seconds,
        "warmup_evaluations": warmup,
        "timing": {
            "repeats": repeats,
            "seconds": timings,
            "median_seconds": statistics.median(timings),
            "mean_seconds": statistics.mean(timings),
            "min_seconds": min(timings),
            "scope": "Private input copy, predictor call and selected-device synchronization; excludes loading, CPU output copies and validation",
        },
        "allocator_memory": memory,
    }
    return snapshots, summary


def _error_report(reference, candidate):
    metrics = {}
    for name, expected in reference.items():
        actual = candidate[name]
        if actual.shape != expected.shape:
            raise ValueError(f"Reference/candidate shape mismatch for {name}")
        difference = actual - expected
        reference_norm = torch.linalg.vector_norm(expected).item()
        error_norm = torch.linalg.vector_norm(difference).item()
        row = {
            "unit": _UNITS[name],
            "shape": list(expected.shape),
            "max_absolute_error": difference.abs().max().item(),
            "rms_error": difference.square().mean().sqrt().item(),
            "error_l2_norm": error_norm,
            "reference_l2_norm": reference_norm,
            "reference_rms": expected.square().mean().sqrt().item(),
            "relative_l2_error": error_norm / reference_norm
            if reference_norm
            else None,
        }
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in row.values()
        ):
            raise FloatingPointError(f"Nonfinite comparison metric: {name}")
        metrics[name] = row
    return metrics


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _report(reference_values, candidate_values, reference, candidate, memory_scope):
    device = torch.device(candidate["device"])
    name = (
        getattr(torch, device.type).get_device_name(device)
        if device.type in ("cuda", "xpu")
        else platform.processor() or platform.machine()
    )
    return {
        "success": True,
        "finite": True,
        "accuracy_qualified": False,
        "qualification_note": "No application-specific error thresholds, force finite differences or trajectory validation were applied; this is a single-structure comparison.",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "fairchem_core": _version("fairchem-core"),
            "ase": _version("ase"),
            "device": str(device),
            "device_name": name,
        },
        "reference": reference,
        "candidate": candidate,
        "errors": _error_report(reference_values, candidate_values),
        "reference_over_candidate_median_time": (
            reference["timing"]["median_seconds"]
            / candidate["timing"]["median_seconds"]
        ),
        "memory_scope": memory_scope,
    }


def compare_predictors(
    reference,
    candidate,
    data,
    *,
    properties=("energy", "forces"),
    warmup=2,
    repeats=5,
    measure_memory=False,
):
    """
    Compare independently constructed predictors on an owned AtomicData snapshot.

    Both supplied models remain resident: optional allocator figures include
    both models and all other process allocations. Use ``run_comparison`` for
    sequential model loading and release. Neither helper qualifies accuracy.
    """
    if reference is candidate:
        raise ValueError("Reference and candidate must be independent predictors")
    if reference.inference_settings.series_b.enabled:
        raise ValueError("The reference predictor must have Series B disabled")
    if reference.inference_settings.tf32 or reference.inference_settings.nitro:
        raise ValueError("The reference predictor must use strict FP32")
    if reference.inference_settings.base_precision_dtype != torch.float32:
        raise ValueError(
            "The reference predictor must use base_precision_dtype=float32"
        )
    if reference.inference_settings.compile or reference.inference_settings.xpu_graph:
        raise ValueError("The reference predictor must use eager execution")
    if torch.device(reference.device) != torch.device(candidate.device):
        raise ValueError("Reference and candidate must use the same device")
    properties = _properties(properties)
    warmup, repeats = _count("warmup", warmup, 0), _count("repeats", repeats, 1)
    snapshot = data.clone()
    expected, first = _measure_predictor(
        reference, snapshot, properties, warmup, repeats, measure_memory
    )
    actual, second = _measure_predictor(
        candidate, snapshot, properties, warmup, repeats, measure_memory
    )
    return _report(
        expected,
        actual,
        first,
        second,
        "Preloaded predictors: both models and unrelated process allocations are included. PyTorch device allocator only; CPU and external library allocations are excluded.",
    )


def run_comparison(
    checkpoint,
    atoms,
    *,
    task,
    device="xpu:0",
    series_b,
    properties=("energy", "forces"),
    warmup=2,
    repeats=5,
    measure_memory=False,
):
    """
    Load the same checkpoint sequentially for strict FP32 and optional Series B.

    Models are released and the selected accelerator cache is emptied between
    cases; unrelated process allocations can still contribute to reported
    peaks. Run in a dedicated process for memory comparisons.
    """
    if not isinstance(series_b, SeriesBSettings):
        series_b = SeriesBSettings(**series_b)
    properties = _properties(properties)
    warmup, repeats = _count("warmup", warmup, 0), _count("repeats", repeats, 1)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda", "xpu"):
        raise ValueError("Comparison device must be CPU, CUDA or XPU")
    if device.type != "cpu" and not getattr(torch, device.type).is_available():
        raise RuntimeError(f"Requested accelerator is unavailable: {device}")
    baseline = inference_settings_xpu()
    settings = [baseline, replace(baseline, series_b=series_b)]
    outputs, summaries = [], []
    structure = atoms.copy()
    validate_uma_atoms_data(structure, task)
    data = AtomicData.from_ase(structure, task_name=task, target_dtype=torch.float32)
    for configuration in settings:
        predictor = None
        try:
            predictor = pretrained_mlip.get_predict_unit(
                str(checkpoint), device=str(device), inference_settings=configuration
            )
            values, summary = _measure_predictor(
                predictor, data, properties, warmup, repeats, measure_memory
            )
            outputs.append(values)
            summaries.append(summary)
        finally:
            predictor = None
            gc.collect()
            if device.type in ("cuda", "xpu"):
                _synchronize(device)
                with getattr(torch, device.type).device(device):
                    getattr(torch, device.type).empty_cache()
    report = _report(
        *outputs,
        *summaries,
        "Sequential model loading and release. PyTorch selected-device allocator only; unrelated process allocations remain included, CPU/external allocations are excluded. First evaluation includes lazy model transfer/preparation, steady peaks are reset after warmup.",
    )
    report["checkpoint"] = str(checkpoint)
    report["task"] = task
    report["natoms"] = len(atoms)
    report["series_b_requested"] = asdict(series_b)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--structure", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument(
        "--series-b", required=True, help="JSON object or @path/to/settings.json"
    )
    parser.add_argument("--output", type=Path, default=Path("series-b-comparison.json"))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--properties", default="energy,forces")
    parser.add_argument("--measure-memory", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        raw = args.series_b
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text()
        options = json.loads(raw)
        if not isinstance(options, dict):
            raise ValueError("--series-b must contain a JSON object")
        atoms = read(args.structure)
        report = run_comparison(
            args.checkpoint,
            atoms,
            task=args.task,
            device=args.device,
            series_b=options,
            properties=args.properties,
            warmup=args.warmup,
            repeats=args.repeats,
            measure_memory=args.measure_memory,
        )
    except Exception as error:
        report = {
            "success": False,
            "accuracy_qualified": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Run FP32 correctness, synchronized ASE timing and allocator-memory checks on XPU.

Example:
    python -m fairchem.core.scripts.validate_xpu --device xpu:0 \
        --compare-merged --stress --output xpu-validation.json

The tolerances are configurable smoke-test thresholds, not an accuracy claim
for a checkpoint. No weights are downloaded unless this command is executed.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk, molecule


def validate_prediction(prediction, natoms, stress=False):
    """Reject malformed outputs before NumPy broadcasting can hide an error."""
    expected = {"energy": (), "forces": (natoms, 3)}
    if stress:
        expected["stress"] = (6,)
    if natoms < 1:
        raise ValueError("At least one atom is required")
    for name, shape in expected.items():
        if name not in prediction:
            raise ValueError(f"Missing prediction: {name}")
        actual = np.asarray(prediction[name])
        if actual.shape != shape:
            raise ValueError(
                f"Prediction {name} has shape {actual.shape}; expected {shape}"
            )


def compare_predictions(reference, candidate, natoms, tolerances):
    """
    Compare finite energy, force and optional stress values in ASE units.
    """
    stress = "stress" in reference or "stress" in candidate
    validate_prediction(reference, natoms, stress)
    validate_prediction(candidate, natoms, stress)
    errors = {
        "energy_eV_per_atom": abs(candidate["energy"] - reference["energy"]) / natoms,
        "forces_eV_per_A": float(
            np.max(np.abs(candidate["forces"] - reference["forces"]))
        ),
    }
    if "stress" in reference:
        errors["stress_eV_per_A3"] = float(
            np.max(np.abs(candidate["stress"] - reference["stress"]))
        )
    finite = all(
        np.isfinite(value).all()
        for prediction in (reference, candidate)
        for value in prediction.values()
    )
    return {
        "passed": bool(finite and all(v <= tolerances[k] for k, v in errors.items())),
        "finite": bool(finite),
        "max_absolute_errors": errors,
    }


def evaluate(atoms, stress=False):
    """
    Force a fresh ASE evaluation instead of timing cached calculator results.
    """
    atoms.calc.reset()
    result = {
        "energy": float(atoms.get_potential_energy()),
        "forces": np.asarray(atoms.get_forces()).copy(),
    }
    if stress:
        result["stress"] = np.asarray(atoms.get_stress()).copy()
    validate_prediction(result, len(atoms), stress)
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite {name} in ASE evaluation")
    return result


def finite_difference_check(atoms, force, step, tolerance):
    """
    Check the largest force component using central energy differences.

    The default 0.005 A step limits FP32 cancellation. Repeat with several
    steps before attributing a mismatch to a backend or using Hessians.
    """
    atom, axis = np.unravel_index(np.argmax(np.abs(force)), force.shape)
    original = atoms.positions.copy()
    energies = []
    try:
        for sign in (1, -1):
            positions = original.copy()
            positions[atom, axis] += sign * step
            atoms.set_positions(positions)
            energies.append(evaluate(atoms)["energy"])
    finally:
        atoms.set_positions(original)
        atoms.calc.reset()
    numeric = -(energies[0] - energies[1]) / (2 * step)
    analytic = float(force[atom, axis])
    error = abs(numeric - analytic)
    return {
        "passed": bool(np.isfinite(error) and error <= tolerance),
        "atom": int(atom),
        "axis": int(axis),
        "step_A": step,
        "analytic_eV_per_A": analytic,
        "finite_difference_eV_per_A": numeric,
        "absolute_error_eV_per_A": error,
        "tolerance_eV_per_A": tolerance,
    }


def finite_difference_stress_check(atoms, stress, step, tolerance):
    """Check all six ASE stress components against energy strain derivatives.

    Off-diagonal strains use half the engineering shear in each symmetric
    matrix entry, so their derivative equals one Voigt stress component.
    """
    original_cell = atoms.cell.copy()
    original_positions = atoms.positions.copy()
    volume = atoms.get_volume()
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Stress finite differences require a positive cell volume")
    numeric = []
    try:
        for row, col in ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)):
            energies = []
            for sign in (1, -1):
                strain = np.eye(3)
                if row == col:
                    strain[row, col] += sign * step
                else:
                    strain[row, col] += sign * step / 2
                    strain[col, row] += sign * step / 2
                # Restore first: each strain must be relative to the same cell.
                atoms.set_cell(original_cell, scale_atoms=False)
                atoms.set_positions(original_positions)
                atoms.set_cell(original_cell @ strain, scale_atoms=True)
                energies.append(evaluate(atoms)["energy"])
            numeric.append((energies[0] - energies[1]) / (2 * step * volume))
    finally:
        atoms.set_cell(original_cell, scale_atoms=False)
        atoms.set_positions(original_positions)
        atoms.calc.reset()
    numeric = np.asarray(numeric)
    analytic = np.asarray(stress)
    if analytic.shape != (6,):
        raise ValueError("Stress must contain six ASE Voigt components")
    errors = np.abs(numeric - analytic)
    return {
        "passed": bool(np.isfinite(errors).all() and np.max(errors) <= tolerance),
        "voigt_order": ["xx", "yy", "zz", "yz", "xz", "xy"],
        "strain_step": step,
        "analytic_eV_per_A3": analytic.tolist(),
        "finite_difference_eV_per_A3": numeric.tolist(),
        "absolute_errors_eV_per_A3": errors.tolist(),
        "tolerance_eV_per_A3": tolerance,
    }


def _xpu_memory_snapshot(device, *, reset_peak=False):
    """Read actual allocator counters after outstanding device work completes."""
    torch.xpu.synchronize(device)
    if reset_peak:
        torch.xpu.reset_peak_memory_stats(device)
    return {
        "allocated_bytes": torch.xpu.memory_allocated(device),
        "reserved_bytes": torch.xpu.memory_reserved(device),
        "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
        "peak_reserved_bytes": torch.xpu.max_memory_reserved(device),
    }


def synchronized_timings(
    atoms, device, warmup, repeats, stress=False, *, measure_memory=False
):
    """
    Time complete ASE energy/force calls including transfers and graph generation.
    """

    def synchronize():
        if device.type == "xpu":
            torch.xpu.synchronize(device)

    measure_memory = measure_memory and device.type == "xpu"
    memory = {}
    if measure_memory:
        memory["warmup"] = {"before": _xpu_memory_snapshot(device, reset_peak=True)}
    for _ in range(warmup):
        evaluate(atoms, stress)
    if measure_memory:
        memory["warmup"]["after"] = _xpu_memory_snapshot(device)
        memory["repeated_evaluation"] = {
            "before": _xpu_memory_snapshot(device, reset_peak=True)
        }
    else:
        synchronize()
    durations = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        evaluate(atoms, stress)
        synchronize()
        durations.append((time.perf_counter() - start) * 1000)
    result = {
        "median_ms": float(np.median(durations)),
        "minimum_ms": min(durations),
        "samples_ms": durations,
        "warmup_calls": warmup,
        "includes": (
            "ASE conversion, graph generation, energy/forces, transfers and output checks"
        ),
    }
    if measure_memory:
        memory["repeated_evaluation"]["after"] = _xpu_memory_snapshot(device)
        result["xpu_memory"] = memory
    return result


def model_tensor_inventory(predictor, device):
    """
    Reject unexpected model placement or floating-point precision.
    """
    inventory = {}
    count = 0
    for name, tensor in list(predictor.model.named_parameters()) + list(
        predictor.model.named_buffers()
    ):
        count += 1
        key = f"{tensor.device}/{tensor.dtype}"
        inventory[key] = inventory.get(key, 0) + tensor.numel()
        if tensor.device != device:
            raise RuntimeError(
                f"Model tensor {name} is on {tensor.device}, not {device}"
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise RuntimeError(f"Model tensor {name} uses {tensor.dtype}, not FP32")
    if not count:
        raise RuntimeError("No model tensors found; cannot verify device placement")
    return inventory


def probe_fp64_reduction(device):
    """
    Exercise FP64 index_add and autograd, needed by UMA's FP32 model path.
    """
    source = torch.tensor(
        [1.0, 2.0, 3.0], dtype=torch.float32, device=device, requires_grad=True
    )
    indices = torch.tensor([0, 1, 0], device=device)
    reduced = torch.zeros(2, dtype=torch.float64, device=device).index_add(
        0, indices, source.double()
    )
    gradient = torch.autograd.grad(reduced.square().sum(), source)[0]
    torch.testing.assert_close(
        reduced.cpu(), torch.tensor([4.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        gradient.cpu(), torch.tensor([8.0, 4.0, 8.0], dtype=torch.float32)
    )
    # Stress/virial accumulation uses native multi-dimensional index_add_.
    # Keep the source strided so the probe also covers the non-contiguous path.
    multidimensional_source = (
        torch.arange(1, 19, dtype=torch.float32, device=device)
        .reshape(3, 2, 3)
        .transpose(1, 2)
        .requires_grad_()
    )
    multidimensional_reduced = torch.zeros(
        (2, 3, 2), dtype=torch.float64, device=device
    ).index_add_(0, indices, multidimensional_source.double())
    multidimensional_gradient = torch.autograd.grad(
        multidimensional_reduced.square().sum(), multidimensional_source
    )[0]
    torch.testing.assert_close(
        multidimensional_reduced.cpu(),
        torch.tensor(
            [[[14, 20], [16, 22], [18, 24]], [[7, 10], [8, 11], [9, 12]]],
            dtype=torch.float64,
        ),
    )
    torch.testing.assert_close(
        multidimensional_gradient.cpu(),
        torch.tensor(
            [
                [[28, 40], [32, 44], [36, 48]],
                [[14, 20], [16, 22], [18, 24]],
                [[28, 40], [32, 44], [36, 48]],
            ],
            dtype=torch.float32,
        ),
    )
    return {
        "passed": True,
        "dtype": str(reduced.dtype),
        "device": str(reduced.device),
        "reduction": reduced.detach().cpu().tolist(),
        "gradient": gradient.detach().cpu().tolist(),
        "multidimensional": {
            "passed": True,
            "dtype": str(multidimensional_reduced.dtype),
            "device": str(multidimensional_reduced.device),
            "source_shape": list(multidimensional_source.shape),
            "source_contiguous": multidimensional_source.is_contiguous(),
            "reduction": multidimensional_reduced.detach().cpu().tolist(),
            "gradient": multidimensional_gradient.detach().cpu().tolist(),
        },
        "note": "FP32 model parameters still require FP64 energy/stress reductions.",
    }


def run_case(args, atoms, task, device, merge_mole, stress=False, progress=None):
    """
    Load a fresh predictor so each timing retains its chosen MoLE configuration.
    """
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

    def stage(name):
        if progress is not None:
            progress.update(device=str(device), merge_mole=merge_mole, stage=name)

    settings = InferenceSettings(
        base_precision_dtype=torch.float32,
        activation_checkpointing=False,
        tf32=False,
        nitro=False,
        execution_mode="general",
        internal_graph_gen_version=2,
        merge_mole=merge_mole,
        compile=args.compile and device.type == "xpu",
    )
    predictor = structure = None
    completed = False
    memory = {}
    try:
        stage("load_model")
        if device.type == "xpu":
            memory["load_model"] = {
                "before": _xpu_memory_snapshot(device, reset_peak=True)
            }
        predictor = pretrained_mlip.get_predict_unit(
            args.model, device=str(device), inference_settings=settings
        )
        if device.type == "xpu":
            memory["load_model"]["after"] = _xpu_memory_snapshot(device)
        structure = atoms.copy()
        structure.calc = FAIRChemCalculator(predictor, task_name=task)
        stage("first_evaluation")
        if device.type == "xpu":
            memory["first_evaluation"] = {
                "before": _xpu_memory_snapshot(device, reset_peak=True)
            }
        start = time.perf_counter()
        values = evaluate(structure, stress)
        if device.type == "xpu":
            torch.xpu.synchronize(device)
        first_evaluation_seconds = time.perf_counter() - start
        if device.type == "xpu":
            memory["first_evaluation"]["after"] = _xpu_memory_snapshot(device)
        stage("model_tensor_inventory")
        result = {
            "device": str(device),
            "model_tensor_elements": model_tensor_inventory(predictor, device),
            "settings": predictor.inference_settings.to_omegaconf(),
            "first_evaluation_seconds": first_evaluation_seconds,
            "energy_eV": values["energy"],
            "forces_eV_per_A": values["forces"].tolist(),
        }
        stage("force_finite_difference")
        result["finite_difference"] = finite_difference_check(
            structure, values["forces"], args.fd_step, args.fd_tolerance
        )
        if stress:
            result["stress_eV_per_A3"] = values["stress"].tolist()
            stage("stress_finite_difference")
            result["stress_finite_difference"] = finite_difference_stress_check(
                structure,
                values["stress"],
                args.stress_fd_step,
                args.stress_fd_tolerance,
            )
        stage("timing")
        result["timing"] = synchronized_timings(
            structure,
            device,
            args.warmup,
            args.repeats,
            stress,
            measure_memory=device.type == "xpu",
        )
        if device.type == "xpu":
            memory.update(result["timing"].pop("xpu_memory"))
            result["xpu_memory"] = {
                "scope": (
                    "PyTorch allocator on the selected XPU in this process; "
                    "excludes CPU and non-PyTorch allocations. Peaks include "
                    "existing allocations and reserved cached blocks at each "
                    "window's start and are reset before each window. First "
                    "evaluation includes lazy GPU transfer, model "
                    "preparation and optional compilation. Memory queries are "
                    "outside timed intervals."
                ),
                "windows": memory,
            }
        completed = True
    finally:
        # Detach the copied structure's calculator even when evaluation fails.
        # Only this case's references are released; the caller's Atoms is intact.
        if completed:
            stage("cleanup")
        if structure is not None:
            structure.calc = None
        predictor = structure = None
        gc.collect()
        if device.type == "xpu":
            try:
                with torch.xpu.device(device):
                    torch.xpu.empty_cache()
            except Exception as error:
                # A failing driver can also reject cleanup. Preserve the
                # original inference error and its stage in that situation.
                if completed:
                    raise
                if progress is not None:
                    progress["cleanup_error"] = {
                        "type": type(error).__name__,
                        "message": str(error),
                    }
    stage("complete")
    return values, result


def _version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="uma-s-1p2p1")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--output", type=Path, default=Path("xpu-validation.json"))
    parser.add_argument(
        "--case", choices=("all", "molecule", "periodic"), default="all"
    )
    parser.add_argument("--stress", action="store_true", help="Check periodic stress")
    parser.add_argument("--compare-merged", action="store_true")
    parser.add_argument(
        "--compile", action="store_true", help="Experimental XPU compile"
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--fd-step", type=float, default=0.005)
    parser.add_argument("--fd-tolerance", type=float, default=0.02)
    parser.add_argument("--energy-tolerance", type=float, default=1e-4)
    parser.add_argument("--force-tolerance", type=float, default=5e-4)
    parser.add_argument("--stress-tolerance", type=float, default=5e-5)
    parser.add_argument("--stress-fd-step", type=float, default=0.001)
    parser.add_argument("--stress-fd-tolerance", type=float, default=5e-4)
    return parser


def main(argv=None):
    """
    Return nonzero on absent XPU, exceptions, nonfinite outputs or failed checks.
    """
    args = build_parser().parse_args(argv)
    report = {
        "passed": False,
        "model": args.model,
        "requested_device": args.device,
        "versions": {
            name: _version(name) for name in ("torch", "fairchem-core", "ase", "e3nn")
        },
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cases": {},
        "options": vars(args),
        "note": "Small-system smoke checks; no claim about all operators or workloads.",
    }
    try:
        if args.warmup < 0 or args.repeats < 1:
            raise ValueError("warmup must be nonnegative and repeats positive")
        if any(
            not np.isfinite(value) or value <= 0
            for value in (
                args.fd_step,
                args.fd_tolerance,
                args.energy_tolerance,
                args.force_tolerance,
                args.stress_tolerance,
                args.stress_fd_step,
                args.stress_fd_tolerance,
            )
        ):
            raise ValueError("Finite-difference step and tolerances must be positive")
        device = torch.device(args.device)
        if device.type != "xpu":
            raise ValueError("--device must be an XPU device, such as xpu:0")
        if not torch.xpu.is_available():
            raise RuntimeError("No XPU available; CPU fallback is not permitted")
        device = torch.device("xpu", device.index if device.index is not None else 0)
        if device.index >= torch.xpu.device_count():
            raise ValueError(f"Requested {device}, but it is not an available XPU")
        report["device_name"] = torch.xpu.get_device_name(device)
        properties = torch.xpu.get_device_properties(device)
        report["device_properties"] = str(properties)
        report["has_fp64"] = getattr(properties, "has_fp64", "not exposed")
        report["torch_xpu_build"] = getattr(torch.version, "xpu", None)
        report["active_check"] = {
            "device": str(device),
            "stage": "fp64_reduction_probe",
        }
        report["fp64_reduction_probe"] = probe_fp64_reduction(device)
        tolerances = {
            "energy_eV_per_atom": args.energy_tolerance,
            "forces_eV_per_A": args.force_tolerance,
            "stress_eV_per_A3": args.stress_tolerance,
        }
        report["comparison_tolerances"] = tolerances
        water = molecule("H2O")
        water.positions[1, 0] += 0.04
        water.info.update(charge=0, spin=1)
        copper = bulk("Cu", "fcc", a=3.6, cubic=True)
        copper.positions[0, 0] += 0.04
        passed = True
        for name, atoms, task in (
            ("molecule", water, "omol"),
            ("periodic", copper, "omat"),
        ):
            if args.case not in ("all", name):
                continue
            case = {"natoms": len(atoms), "task": task}
            report["cases"][name] = case
            progress = {"case": name}
            report["active_check"] = progress
            stress = args.stress and name == "periodic"
            reference, case["cpu"] = run_case(
                args, atoms, task, torch.device("cpu"), False, stress, progress
            )
            candidate, case["xpu_unmerged"] = run_case(
                args, atoms, task, device, False, stress, progress
            )
            progress["stage"] = "cpu_vs_xpu_comparison"
            case["cpu_vs_xpu"] = compare_predictions(
                reference, candidate, len(atoms), tolerances
            )
            case["passed"] = all(
                (
                    case["cpu_vs_xpu"]["passed"],
                    case["cpu"]["finite_difference"]["passed"],
                    case["xpu_unmerged"]["finite_difference"]["passed"],
                )
            )
            if stress:
                case["passed"] &= (
                    case["cpu"]["stress_finite_difference"]["passed"]
                    and case["xpu_unmerged"]["stress_finite_difference"]["passed"]
                )
            if args.compare_merged:
                merged, case["xpu_merged"] = run_case(
                    args, atoms, task, device, True, stress, progress
                )
                progress["stage"] = "unmerged_vs_merged_comparison"
                case["unmerged_vs_merged"] = compare_predictions(
                    candidate, merged, len(atoms), tolerances
                )
                case["passed"] &= (
                    case["unmerged_vs_merged"]["passed"]
                    and case["xpu_merged"]["finite_difference"]["passed"]
                )
                if stress:
                    case["passed"] &= case["xpu_merged"]["stress_finite_difference"][
                        "passed"
                    ]
            passed &= case["passed"]
        report.pop("active_check", None)
        report["passed"] = passed
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False) + "\n"
    )
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

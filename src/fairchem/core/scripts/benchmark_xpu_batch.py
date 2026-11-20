"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure independent Xe inference processes and local serial/packed predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch
from ase.build import bulk
from ase.io import read

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.common.distutils import resolve_xpu_device
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu


def _parity(serial, packed, atom_counts, energy_atol, force_atol):
    """Compare the energy and every force vector of each input system in order."""
    errors = []
    per_system = []
    for key, size in (("energy", len(serial)), ("forces", sum(atom_counts))):
        expected = (size,) if key == "energy" else (size, 3)
        if key not in packed or tuple(packed[key].shape) != expected:
            errors.append(f"packed {key} must have shape {expected}")
    if not errors:
        energies = packed["energy"].detach().cpu().double()
        forces = packed["forces"].detach().cpu().double().split(atom_counts)
        for index, (row, count) in enumerate(zip(serial, atom_counts, strict=True)):
            if (
                "energy" not in row
                or row["energy"].numel() != 1
                or "forces" not in row
                or tuple(row["forces"].shape) != (count, 3)
            ):
                errors.append(f"malformed serial prediction for system {index}")
                continue
            energy = abs(energies[index].item() - row["energy"].item())
            force = (
                (forces[index] - row["forces"].detach().cpu().double())
                .abs()
                .max()
                .item()
            )
            # JSON cannot represent infinities, including a nonfinite reference.
            per_system.append(
                {
                    "system": index,
                    "energy": energy if math.isfinite(energy) else None,
                    "forces": force if math.isfinite(force) else None,
                }
            )
            if not (energy <= energy_atol and force <= force_atol):
                errors.append(
                    f"system {index} exceeds packed/serial tolerance or is nonfinite"
                )
    return {
        "passed": not errors,
        "errors": errors,
        "per_system": per_system,
        "energy_atol_eV": energy_atol,
        "force_atol_eV_per_A": force_atol,
    }


def _worker(
    checkpoint,
    device,
    structures,
    task,
    warmup,
    repeats,
    energy_atol=1e-3,
    force_atol=1e-4,
    barrier=None,
    failed=None,
):
    try:
        return _measure(
            checkpoint,
            device,
            structures,
            task,
            warmup,
            repeats,
            barrier,
            energy_atol,
            force_atol,
            failed,
        )
    except BaseException:
        if barrier is not None:
            barrier.abort()
        raise


def _measure(
    checkpoint,
    device,
    structures,
    task,
    warmup,
    repeats,
    barrier,
    energy_atol=1e-3,
    force_atol=1e-4,
    failed=None,
):
    device = resolve_xpu_device(device)
    predictor = pretrained_mlip.get_predict_unit(
        checkpoint,
        device=device,
        inference_settings=replace(inference_settings_xpu(), merge_mole=False),
    )
    for atoms in structures:
        predictor.validate_atoms_data(atoms, task)
    items = [
        AtomicData.from_ase(
            atoms,
            task_name=task,
            target_dtype=torch.float32,
            r_data_keys=["spin", "charge"],
        )
        for atoms in structures
    ]
    packed = atomicdata_list_to_batch(items)
    options = {"requested_properties": ("energy", "forces")}
    serial = [predictor.predict(item, **options) for item in items]
    batch = predictor.predict(packed, **options)
    atom_counts = [len(atoms) for atoms in structures]
    parity = _parity(serial, batch, atom_counts, energy_atol, force_atol)
    del serial, batch
    report = {
        "device": device,
        "device_name": torch.xpu.get_device_name(device),
        "systems_per_iteration": len(items),
        "natoms": atom_counts,
        "parity": parity,
        "max_packed_vs_serial_difference": {
            key: max(
                (
                    system[key]
                    for system in parity["per_system"]
                    if system[key] is not None
                ),
                default=None,
            )
            for key in ("energy", "forces")
        },
    }
    if barrier is not None:
        if not parity["passed"]:
            failed.set()
        barrier.wait(timeout=1800)
    if not parity["passed"] or (failed is not None and failed.is_set()):
        return report
    for name, inputs in (("serial", items), ("packed", (packed,))):
        for _ in range(warmup):
            for item in inputs:
                predictor.predict(item, **options)
        torch.xpu.synchronize(device)
        if barrier is not None:
            barrier.wait(timeout=1800)
        start = time.monotonic()
        torch.xpu.reset_peak_memory_stats(device)
        for _ in range(repeats):
            for item in inputs:
                predictor.predict(item, **options)
        torch.xpu.synchronize(device)
        end = time.monotonic()
        report[name] = {
            "start_monotonic": start,
            "end_monotonic": end,
            "systems_per_second": repeats * len(items) / (end - start),
            "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--devices", nargs="+", default=["xpu"])
    parser.add_argument(
        "--structure", type=Path, help="Read all frames of this ASE file"
    )
    parser.add_argument("--task", default="omat")
    parser.add_argument(
        "--energy-atol",
        type=float,
        default=1e-3,
        help="maximum absolute energy error per system (eV)",
    )
    parser.add_argument(
        "--force-atol",
        type=float,
        default=1e-4,
        help="maximum absolute force component error (eV/Angstrom)",
    )
    parser.add_argument("--systems", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("xpu-batch.json"))
    args = parser.parse_args(argv)
    if args.systems < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("systems and repeats must be positive and warmup nonnegative")
    if not all(
        math.isfinite(tol) and tol >= 0 for tol in (args.energy_atol, args.force_atol)
    ):
        parser.error("tolerances must be finite and nonnegative")
    devices = [resolve_xpu_device(d) for d in args.devices]
    if len(set(devices)) != len(devices):
        parser.error("each process requires a distinct XPU")
    frames = (
        read(args.structure, index=":")
        if args.structure
        else [
            bulk("Cu", cubic=True),
            bulk("Al", "fcc").repeat((2, 1, 1)),
            bulk("Si", "diamond", cubic=True),
            bulk("Fe", "bcc", cubic=True),
        ]
    )
    if not frames:
        parser.error("no input frames")
    structures = [frames[i % len(frames)].copy() for i in range(args.systems)]
    if not args.structure:
        for i, atoms in enumerate(structures):
            atoms.positions[0, 0] += 0.03 * (1 + i % 11)
    if any(len(atoms) == 0 for atoms in structures):
        parser.error("every system must contain at least one atom")
    common = (
        structures,
        args.task,
        args.warmup,
        args.repeats,
        args.energy_atol,
        args.force_atol,
    )
    # Every worker measures the identical workload. A spawn context prevents
    # the child from inheriting initialized accelerator state from its parent.
    if len(devices) == 1:
        rows = [_worker(args.checkpoint, devices[0], *common)]
    else:
        ctx = multiprocessing.get_context("spawn")
        with (
            ctx.Manager() as manager,
            ProcessPoolExecutor(len(devices), mp_context=ctx) as pool,
        ):
            barrier = manager.Barrier(len(devices))
            failed = manager.Event()
            pending = [
                pool.submit(_worker, args.checkpoint, device, *common, barrier, failed)
                for device in devices
            ]
            rows = [future.result() for future in pending]
    report = {
        "workers": rows,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "passed": all(row["parity"]["passed"] for row in rows),
    }
    if report["passed"]:
        report["systems_per_second"] = {
            name: len(devices)
            * args.repeats
            * len(structures)
            / (
                max(r[name]["end_monotonic"] for r in rows)
                - min(r[name]["start_monotonic"] for r in rows)
            )
            for name in ("serial", "packed")
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())

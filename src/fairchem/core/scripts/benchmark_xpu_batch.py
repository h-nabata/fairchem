"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure independent Xe inference processes and local serial/packed predictions.
"""

from __future__ import annotations

import argparse
import json
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


def _worker(checkpoint, device, structures, task, warmup, repeats, barrier=None):
    try:
        return _measure(checkpoint, device, structures, task, warmup, repeats, barrier)
    except BaseException:
        if barrier is not None:
            barrier.abort()
        raise


def _measure(checkpoint, device, structures, task, warmup, repeats, barrier):
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
    differences = {}
    for key in options["requested_properties"]:
        reference = torch.cat([row[key].reshape(-1) for row in serial])
        actual = batch[key].reshape(-1)
        if (
            actual.shape != reference.shape
            or not torch.isfinite(actual).all()
            or not torch.isfinite(reference).all()
        ):
            raise ValueError(f"Malformed or nonfinite packed/serial {key}")
        differences[key] = (actual - reference).abs().max().item()
    del serial, batch, actual, reference
    report = {
        "device": device,
        "device_name": torch.xpu.get_device_name(device),
        "systems_per_iteration": len(items),
        "natoms": [len(atoms) for atoms in structures],
        "max_packed_vs_serial_difference": differences,
    }
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
    parser.add_argument("--systems", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("xpu-batch.json"))
    args = parser.parse_args(argv)
    if args.systems < 1 or args.warmup < 0 or args.repeats < 1:
        parser.error("systems and repeats must be positive and warmup nonnegative")
    devices = [resolve_xpu_device(d) for d in args.devices]
    if len(set(devices)) != len(devices):
        parser.error("each process requires a distinct XPU")
    frames = (
        read(args.structure, index=":") if args.structure else [bulk("Cu", cubic=True)]
    )
    structures = [frames[i % len(frames)].copy() for i in range(args.systems)]
    common = (structures, args.task, args.warmup, args.repeats)
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
            pending = [
                pool.submit(_worker, args.checkpoint, device, *common, barrier)
                for device in devices
            ]
            rows = [future.result() for future in pending]
    report = {
        "workers": rows,
        "checkpoint": args.checkpoint,
        "task": args.task,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "systems_per_second": {
            name: len(devices)
            * args.repeats
            * len(structures)
            / (
                max(r[name]["end_monotonic"] for r in rows)
                - min(r[name]["start_monotonic"] for r in rows)
            )
            for name in ("serial", "packed")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

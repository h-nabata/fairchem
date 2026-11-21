"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.scripts import benchmark_xpu_batch as bench


def test_frame_reader_stops_at_limit_and_closes(monkeypatch, tmp_path):
    decoded = []
    closed = []

    def iread(path, index):
        assert path == tmp_path / "many.xyz"
        assert index == ":"
        try:
            for i in range(100):
                decoded.append(i)
                yield i
        finally:
            closed.append(True)

    monkeypatch.setattr(bench, "iread", iread)
    assert bench._read_frames(tmp_path / "many.xyz", 2) == [0, 1]
    assert decoded == [0, 1]
    assert closed == [True]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_workload_precision_and_timing(monkeypatch, precision):
    dtype = getattr(torch, precision)
    events = []
    monkeypatch.setattr(bench, "resolve_xpu_device", str)

    def predictor(checkpoint, device, inference_settings):
        assert inference_settings.base_precision_dtype == dtype
        assert not inference_settings.merge_mole
        return SimpleNamespace(
            validate_atoms_data=lambda *_: None,
            predict=lambda counts, **_: {
                "energy": torch.zeros(len(counts), dtype=dtype),
                "forces": torch.zeros(sum(counts), 3, dtype=dtype),
            },
        )

    def from_ase(atoms, *, target_dtype, **_):
        assert target_dtype == dtype
        return (len(atoms),)

    monkeypatch.setattr(bench.pretrained_mlip, "get_predict_unit", predictor)
    monkeypatch.setattr(bench.AtomicData, "from_ase", from_ase)
    monkeypatch.setattr(bench, "atomicdata_list_to_batch", lambda x: sum(x, ()))
    monkeypatch.setattr(bench.torch.xpu, "get_device_name", lambda *_: "stub")
    monkeypatch.setattr(
        bench.torch.xpu, "synchronize", lambda *_: events.append("sync")
    )
    monkeypatch.setattr(
        bench.torch.xpu, "reset_peak_memory_stats", lambda *_: events.append("reset")
    )
    monkeypatch.setattr(bench.torch.xpu, "max_memory_allocated", lambda *_: 47)

    def monotonic():
        events.append("time")
        return float(len(events))

    monkeypatch.setattr(bench, "time", SimpleNamespace(monotonic=monotonic))
    row = bench._measure(
        "unused.ckpt", "xpu:0", [[1], [1, 2]], "omat", 0, 1, None, dtype=precision
    )
    assert row["parity"]["passed"]
    assert row["dtype"] == precision
    assert row["serial"]["peak_allocated_bytes"] == 47
    assert events == ["sync", "reset", "time", "sync", "time"] * 2


def test_cli_threads_dtype_and_requested_frame_limit(monkeypatch, tmp_path):
    class AtomsStub(list):
        def copy(self):
            return type(self)(self)

    monkeypatch.setattr(bench, "resolve_xpu_device", str)

    def read_frames(path, limit):
        assert path == tmp_path / "one.xyz"
        assert limit == 3
        return [AtomsStub([1, 2])]

    def worker(checkpoint, device, structures, *_args, dtype):
        assert dtype == "float64"
        assert len(structures) == 3
        assert all(len(s) == 2 for s in structures)
        assert len({id(s) for s in structures}) == 3
        return {
            "parity": {"passed": True},
            **{
                key: {"start_monotonic": 1, "end_monotonic": 2}
                for key in ("serial", "packed")
            },
        }

    monkeypatch.setattr(bench, "_read_frames", read_frames)
    monkeypatch.setattr(bench, "_worker", worker)
    output = tmp_path / "report.json"
    assert (
        bench.main(
            [
                "--checkpoint",
                "unused.ckpt",
                "--structure",
                str(tmp_path / "one.xyz"),
                "--systems",
                "3",
                "--dtype",
                "float64",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["dtype"] == "float64"


def test_worker_does_not_mask_original_exception_if_barrier_is_gone(monkeypatch):
    def measure(*_args):
        raise ValueError("original calculation error")

    def abort():
        raise BrokenPipeError("manager exited")

    monkeypatch.setattr(bench, "_measure", measure)
    with pytest.raises(ValueError, match="original calculation error"):
        bench._worker(
            "unused.ckpt",
            "xpu:0",
            [],
            "omat",
            0,
            1,
            barrier=SimpleNamespace(abort=abort),
        )

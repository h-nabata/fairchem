"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import json
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ase.build import molecule
from ase.io import write

from fairchem.core.scripts import compare_series_b as script
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


def test_real_uma_comparison_reports_errors_without_claiming_qualification():
    reference, candidate = _public_predictor(), _public_predictor()
    candidate.inference_settings = replace(
        candidate.inference_settings,
        series_b=SeriesBSettings(gemm_precision="bf16", low_precision_emulation=True),
    )
    data = _data()
    before = data.clone()
    report = script.compare_predictors(
        reference,
        candidate,
        data,
        properties=("energy", "forces", "hessian"),
        warmup=0,
        repeats=2,
    )
    assert report["success"]
    assert report["finite"]
    assert not report["accuracy_qualified"]
    assert report["candidate"]["cpu_low_precision_emulation"]
    assert report["candidate"]["approximation_manifest"]["linear"]["converted"]
    assert len(report["candidate"]["timing"]["seconds"]) == 2
    assert report["candidate"]["allocator_memory"] is None
    assert report["errors"]["forces"]["shape"] == [3, 3]
    assert report["errors"]["hessian"]["shape"] == [9, 9]
    assert report["errors"]["energy"]["max_absolute_error"] > 0
    torch.testing.assert_close(data.pos, before.pos, rtol=0, atol=0)
    assert not data.pos.requires_grad
    json.dumps(report, allow_nan=False)


def test_cli_json_file_options_load_independent_models_sequentially(
    tmp_path, monkeypatch
):
    previous, settings_seen = [], []

    def load(name, *, device, inference_settings):
        assert name == "tiny-uma.pt"
        assert device == "cpu"
        if previous:
            assert previous[-1]() is None
        predictor = _public_predictor()
        predictor.inference_settings = copy.deepcopy(inference_settings)
        previous.append(weakref.ref(predictor))
        settings_seen.append(copy.deepcopy(inference_settings))
        return predictor

    monkeypatch.setattr(script.pretrained_mlip, "get_predict_unit", load)
    atoms = molecule("H2O")
    source, options, output = (
        tmp_path / "water.xyz",
        tmp_path / "options.json",
        tmp_path / "report.json",
    )
    write(source, atoms)
    options.write_text('{"weight_bits":8}')
    assert (
        script.main(
            [
                "--checkpoint",
                "tiny-uma.pt",
                "--structure",
                str(source),
                "--task",
                "omol",
                "--device",
                "cpu",
                "--series-b",
                f"@{options}",
                "--output",
                str(output),
                "--warmup",
                "0",
                "--repeats",
                "1",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text())
    assert report["success"]
    assert not report["accuracy_qualified"]
    assert not settings_seen[0].series_b.enabled
    assert settings_seen[1].series_b.weight_bits == 8
    assert all(reference() is None for reference in previous)
    assert "Sequential" in report["memory_scope"]
    assert report["natoms"] == 3
    assert report["task"] == "omol"
    assert "charge" not in atoms.info
    assert "spin" not in atoms.info


def test_cli_reports_failure_as_json_and_nonzero_status(tmp_path, monkeypatch):
    output = tmp_path / "error.json"
    monkeypatch.setattr(
        script.pretrained_mlip,
        "get_predict_unit",
        lambda *a, **k: pytest.fail("invalid options must fail before loading"),
    )
    assert (
        script.main(
            [
                "--checkpoint",
                "unused.pt",
                "--structure",
                "unused.xyz",
                "--task",
                "omol",
                "--series-b",
                "[]",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert not report["success"]
    assert not report["accuracy_qualified"]
    assert report["error"]["type"] == "ValueError"


def test_error_metrics_have_no_broadcasting_or_zero_reference_division():
    reference = {"forces": torch.zeros(2, 3, dtype=torch.float64)}
    candidate = {"forces": torch.ones(2, 3, dtype=torch.float64)}
    result = script._error_report(reference, candidate)["forces"]
    assert result["max_absolute_error"] == result["rms_error"] == 1
    assert result["reference_l2_norm"] == 0
    assert result["relative_l2_error"] is None
    with pytest.raises(ValueError, match="shape mismatch"):
        script._error_report(reference, {"forces": torch.ones(1, 3)})
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        script._snapshot({"energy": torch.tensor([float("nan")])}, ("energy",), 2)
    with pytest.raises(ValueError, match="shape"):
        script._snapshot({"forces": torch.ones(1, 3)}, ("forces",), 2)


def test_measurement_synchronizes_selected_device_and_resets_memory_windows(
    monkeypatch,
):
    events = []
    settings = inference_settings_xpu()
    predictor = SimpleNamespace(device="xpu:2", inference_settings=settings)

    def predict(data, requested_properties):
        events.append("predict")
        assert not torch.is_autocast_enabled("xpu")
        return {"energy": torch.zeros(1)}

    def allocator(device, *, reset=False):
        events.append(("memory", str(device), reset))
        return {"peak_allocated_bytes": 100}

    predictor.predict = predict
    monkeypatch.setattr(script, "_allocator", allocator)
    monkeypatch.setattr(
        script, "_synchronize", lambda device: events.append(("sync", str(device)))
    )
    _, result = script._measure_predictor(predictor, _data(), ("energy",), 1, 2, True)
    assert events.count("predict") == 4  # cold + warmup + two measured calls
    assert events.count(("memory", "xpu:2", True)) == 2
    for index, event in enumerate(events):
        if event == "predict":
            assert events[index - 1] == events[index + 1] == ("sync", "xpu:2")
    assert result["allocator_memory"]["steady"]["after"]["peak_allocated_bytes"] == 100
    assert result["timing"]["repeats"] == 2


def test_comparison_requires_independent_strict_reference_and_valid_counts():
    reference, candidate = _public_predictor(), _public_predictor()
    with pytest.raises(ValueError, match="independent"):
        script.compare_predictors(reference, reference, _data())
    for kwargs in (
        {"warmup": -1},
        {"repeats": 0},
        {"repeats": True},
        {"properties": "energy,energy"},
    ):
        with pytest.raises(ValueError):
            script.compare_predictors(reference, candidate, _data(), **kwargs)
    reference.inference_settings.tf32 = True
    with pytest.raises(ValueError, match="strict FP32"):
        script.compare_predictors(reference, candidate, _data())


def test_outer_cpu_autocast_does_not_reduce_reference_precision(monkeypatch):
    reference, candidate = _public_predictor(), _public_predictor()
    calls = []
    for predictor in (reference, candidate):
        original = predictor._run_inference

        def checked(*args, original=original, **kwargs):
            calls.append(torch.is_autocast_enabled("cpu"))
            return original(*args, **kwargs)

        monkeypatch.setattr(predictor, "_run_inference", checked)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        report = script.compare_predictors(
            reference, candidate, _data(), warmup=0, repeats=1
        )
        assert torch.is_autocast_enabled("cpu")
    assert calls == [False] * 4
    assert report["errors"]["energy"]["max_absolute_error"] < 1e-6
    assert report["errors"]["forces"]["max_absolute_error"] < 1e-6

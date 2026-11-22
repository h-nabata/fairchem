"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import molecule

from fairchem.core.scripts import benchmark_md, compare_series_b, validate_xpu
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from tests.core.scripts.test_benchmark_md import HarmonicCalculator
from tests.core.scripts.test_validate_xpu import mock_xpu
from tests.core.units.mlip_unit.test_xpu_preparation import _data


def test_current_device_and_direct_cpu_merged_accuracy(tmp_path, monkeypatch):
    mock_xpu(monkeypatch)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)

    def case(args, atoms, task, device, merged, stress, progress):
        error = 0.0 if device.type == "cpu" else 0.000075 * (1 + merged)
        return {"energy": len(atoms) * error, "forces": np.zeros((len(atoms), 3))}, {
            "finite_difference": {"passed": True}
        }

    monkeypatch.setattr(validate_xpu, "run_case", case)
    for selected, expected in (("xpu", "xpu:1"), ("xpu:0", "xpu:0")):
        output = tmp_path / "report.json"
        assert (
            validate_xpu.main(
                [
                    "--device",
                    selected,
                    "--compare-merged",
                    "--case",
                    "molecule",
                    "--output",
                    str(output),
                ]
            )
            == 1
        )
        report = json.loads(output.read_text())
        assert report["device"] == expected
        case_report = report["cases"]["molecule"]
        assert (
            case_report["cpu_vs_xpu"]["passed"]
            and case_report["unmerged_vs_merged"]["passed"]
        )
        assert not case_report["cpu_vs_merged"]["passed"]


def test_validation_rejects_equally_predicting_different_checkpoints(
    tmp_path, monkeypatch
):
    mock_xpu(monkeypatch)

    def case(args, atoms, task, device, merged, stress, progress):
        return {"energy": 1.0, "forces": np.zeros((len(atoms), 3))}, {
            "finite_difference": {"passed": True},
            "checkpoint_sha256": device.type,
        }

    monkeypatch.setattr(validate_xpu, "run_case", case)
    output = tmp_path / "validation.json"
    assert validate_xpu.main(["--case", "molecule", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["cases"]["molecule"]["cpu_vs_xpu"]["passed"]
    assert "different checkpoint" in report["error"]["message"]
    assert not report["passed"]


def test_energy_only_directional_checks_cover_other_components():
    atoms = molecule("H2O")
    atoms.calc = HarmonicCalculator()
    force = -atoms.positions.copy()
    wrong = force.copy()
    wrong[2, 0] += 0.1
    requests = []
    original = atoms.calc.calculate

    def record(atoms, properties, system_changes):
        requests.append(properties)
        original(atoms, properties, system_changes)

    atoms.calc.calculate = record
    result = validate_xpu.finite_difference_check(
        atoms, wrong, 1e-4, 1e-5, directions=3, scales=(1.0, 2.0)
    )
    assert len(result["checks"]) == 6 and not result["passed"]
    assert all(properties == ["energy"] for properties in requests)


def test_joint_batch_request_and_observed_p95(monkeypatch):
    from fairchem.core.scripts import benchmark_batch_server as script

    calls = []

    class Calculator:
        def calculate(self, atoms, properties, system_changes):
            calls.append(properties)
            self.results = {"energy": 1, "forces": np.zeros((len(atoms), 3))}

    monkeypatch.setattr(
        script, "FAIRChemCalculator", lambda *args, **kwargs: Calculator()
    )
    script.timed_calculate(molecule("H2"), "omol", None)
    assert calls == [["energy", "forces"]]
    for n in (1, 2, 8):
        metrics = script._summarize(list(range(1, n + 1)), 12.0, n)
        assert metrics["p95_latency_s"] == metrics["max_latency_s"] == n


def test_single_process_cpu_qps_without_distributed(monkeypatch):
    from fairchem.core.components.benchmark._single.uma_speed_benchmark import get_qps

    calls = []
    predictor = SimpleNamespace(device="cpu", predict=calls.append)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: pytest.fail("no group"))
    qps, per_day = get_qps("system", predictor, warmups=1, timeiters=2, repeats=3)
    assert len(calls) == 7 and qps > 0 and per_day > 0


def test_bounded_md_summary_and_stream(tmp_path):
    path = tmp_path / "samples.jsonl"
    result = benchmark_md.run_md(
        molecule("H2"),
        HarmonicCalculator(),
        steps=21,
        warmup=2,
        max_samples=4,
        sample_interval=1,
        sample_stream=path,
        timestep_fs=0.01,
        timing_block=3,
    )
    rows = [json.loads(row) for row in path.read_text().splitlines()]
    assert [row["step"] for row in rows] == list(range(22))
    assert len(result["samples"]) <= 4 and result["quantile_sample_size"] == 4
    slope = np.polyfit(
        [r["time_ps"] for r in rows], [r["energy_eV_per_atom"] for r in rows], 1
    )[0]
    assert result["energy_slope_eV_per_atom_ps"] == pytest.approx(slope, abs=1e-9)
    assert result["allocator_windows"].keys() == {"first", "warmup", "steady"}


def test_series_b_control_records_transient_errors_and_order():
    class Predictor:
        device = "cpu"

        def __init__(self, values):
            self.inference_settings = inference_settings_xpu()
            self.values = iter(values)

        def predict(self, data, requested_properties):
            return {"energy": torch.tensor(next(self.values), dtype=torch.float32)}

    report = compare_series_b.compare_predictors(
        Predictor([1, 2, 1]),
        Predictor([1, 4, 1]),
        _data(),
        properties=("energy",),
        warmup=0,
        repeats=2,
        candidate_first=True,
    )
    assert report["comparison_kind"] == "strict_control"
    assert report["evaluation_order"] == ["candidate", "reference"]
    assert report["errors"]["energy"]["max_absolute_error"] == 0
    assert report["candidate"]["max_deviation_from_own_first"]["energy"] == 3
    assert report["reference"]["deviation_from_first_run_by_call"][1]["energy"] == 1


def test_actual_model_hvp_and_geometry_variation(monkeypatch):
    import fairchem.core
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    def load(model, device, inference_settings):
        predictor = _public_predictor()
        predictor.inference_settings = inference_settings
        return predictor

    monkeypatch.setattr(fairchem.core.pretrained_mlip, "get_predict_unit", load)
    args = validate_xpu.build_parser().parse_args(
        [
            "--dtype",
            "float64",
            "--hvp",
            "--geometry-workloads",
            "subskin",
            "rebuild",
            "--warmup",
            "0",
            "--repeats",
            "2",
        ]
    )
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    _, result = validate_xpu.run_case(args, atoms, "omol", torch.device("cpu"), False)
    assert result["hvp_finite_difference"]["passed"]
    for mode in ("subskin", "rebuild"):
        row = result["geometry_workloads"][mode]
        assert row["physical_edges"] > 0 and row["p95_ms"] > 0


def test_counted_predictor_tracks_shared_call_after_replica_serialization():
    import pickle

    from fairchem.core.scripts.benchmark_batch_server import (
        _CountedPredictor,
        _request_metrics,
        _RequestObservation,
    )

    class Local:
        def predict(self, data):
            return {
                "energy": torch.zeros(data.num_graphs),
                "forces": torch.zeros(data.num_graphs, 3),
            }

    # Ray transfers the proxy between processes. Normal pickle must first be
    # able to reconstruct the wrapper before it delegates predictor attributes.
    wrapped = pickle.loads(pickle.dumps(_CountedPredictor(SimpleNamespace())))
    wrapped.predictor = Local()
    split = wrapped.predict(SimpleNamespace(num_graphs=2))
    records = []
    for i in range(2):
        raw = {key: value[i : i + 1] for key, value in split.items()}
        request = _RequestObservation(
            SimpleNamespace(predict=lambda data, raw=raw: raw)
        )
        output = request.predict(None)
        assert set(output) == {"energy", "forces"}
        records.append(
            {
                "latency_s": 0.01,
                "predictor_call_ids": request.call_ids,
                "returned_properties": request.returned,
            }
        )
    result = _request_metrics(records, 0.02)
    assert result["successful_predictor_calls"] == 1
    assert result["returned_property_sets"] == [("energy", "forces")]


def test_missing_runtime_execution_fails_requested_feature():
    settings = inference_settings_xpu()
    settings.xpu_graph = True
    runtime = SimpleNamespace(
        function=test_missing_runtime_execution_fails_requested_feature,
        call_count=3,
        compile_call_count=0,
        compile_failure_count=0,
        capture_count=0,
        capture_failure_count=1,
        replay_count=0,
        xpu_graph=True,
    )
    predictor = SimpleNamespace(_numeric_runtimes=[runtime])
    report = validate_xpu._execution_report(predictor, settings, 0)
    assert not report["passed"]
    assert report["blocks"][0]["graph_bypasses"] == 3
    runtime.capture_count = 1
    runtime.replay_count = 2
    assert validate_xpu._execution_report(predictor, settings, 0)["passed"]


@pytest.mark.parametrize("module", [validate_xpu, benchmark_md])
def test_benchmark_rejects_wrapped_device_before_torch_discovery(module, tmp_path):
    output = tmp_path / "report.json"
    assert module.main(["--device", "xpu:256", "--output", str(output)]) == 1
    error = json.loads(output.read_text())["error"]
    message = error["message"] if isinstance(error, dict) else error
    assert "Invalid xpu device index '256'" in message


def test_series_b_comparison_rejects_wrapped_device_before_loading():
    with pytest.raises(ValueError, match="Invalid cuda device index '257'"):
        compare_series_b.run_comparison(
            "unused.pt", molecule("H2O"), task="omol", device="cuda:257", series_b={}
        )

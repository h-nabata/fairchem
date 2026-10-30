"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.stress import full_3x3_to_voigt_6_stress

from fairchem.core.scripts.validate_xpu import (
    _json_safe,
    _xpu_memory_snapshot,
    build_parser,
    case_settings,
    compare_predictions,
    evaluate,
    finite_difference_check,
    finite_difference_stress_check,
    main,
    model_tensor_inventory,
    probe_fp64_reduction,
    run_case,
    synchronized_timings,
)


class HarmonicCalculator(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self):
        super().__init__()
        self.calls = 0

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        self.results = {
            "energy": float(np.sum(atoms.positions**2) / 2),
            "forces": -atoms.positions.copy(),
        }


def harmonic_atoms():
    atoms = Atoms("H", positions=[[1.0, 0.2, -0.1]])
    atoms.calc = HarmonicCalculator()
    return atoms


class HarmonicStressCalculator(HarmonicCalculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces", "stress"]

    def calculate(self, atoms, properties, system_changes):
        super().calculate(atoms, properties, system_changes)
        stress = atoms.positions.T @ atoms.positions / atoms.get_volume()
        self.results["stress"] = full_3x3_to_voigt_6_stress(stress)


def periodic_harmonic_atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.8, 0.3, 0.2], [0.2, 0.9, 0.5]],
        cell=[[2.0, 0.2, 0.0], [0.0, 2.1, 0.1], [0.1, 0.0, 2.2]],
        pbc=True,
    )
    atoms.calc = HarmonicStressCalculator()
    return atoms


@pytest.mark.parametrize("stress", [False, True])
def test_evaluate_requests_all_outputs_in_one_forward(stress):
    class SelectiveCalculator(HarmonicStressCalculator):
        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)
            self.requested = properties
            self.results = {name: self.results[name] for name in properties}

    atoms = periodic_harmonic_atoms()
    atoms.calc = SelectiveCalculator()
    result = evaluate(atoms, stress)
    assert atoms.calc.calls == 1
    assert set(atoms.calc.requested) == {
        "energy",
        "forces",
        *(["stress"] if stress else []),
    }
    assert set(result) == set(atoms.calc.requested)


def test_evaluate_converts_full_stress_tensor_to_ase_voigt_order():
    class MatrixStressCalculator(HarmonicStressCalculator):
        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)
            self.results["stress"] = (
                atoms.positions.T @ atoms.positions / atoms.get_volume()
            )

    atoms = periodic_harmonic_atoms()
    expected = evaluate(atoms, True)["stress"]
    atoms.calc = MatrixStressCalculator()
    np.testing.assert_allclose(evaluate(atoms, True)["stress"], expected)


def test_finite_difference_uses_energy_and_restores_structure():
    atoms = harmonic_atoms()
    before = atoms.positions.copy()
    force = evaluate(atoms)["forces"]
    result = finite_difference_check(atoms, force, step=0.005, tolerance=1e-9)
    assert result["passed"]
    assert result["finite_difference_eV_per_A"] == pytest.approx(-1.0)
    np.testing.assert_array_equal(atoms.positions, before)
    assert not atoms.calc.results
    assert not finite_difference_check(atoms, -force, 0.005, 1e-9)["passed"]


def test_benchmark_does_not_measure_ase_cache_and_synchronizes(monkeypatch):
    atoms = harmonic_atoms()
    synchronize = Mock()
    monkeypatch.setattr(torch.xpu, "synchronize", synchronize)
    result = synchronized_timings(atoms, torch.device("xpu:1"), 2, 3)
    assert atoms.calc.calls == 5
    assert synchronize.call_count == 7
    synchronize.assert_called_with(torch.device("xpu:1"))
    assert len(result["samples_ms"]) == 3
    assert result["minimum_ms"] >= 0


def test_xpu_memory_snapshot_synchronizes_resets_and_reads_selected_device(
    monkeypatch,
):
    events = []
    selected = torch.device("xpu:1")

    def record(name, result=None):
        def call(device):
            assert device == selected
            events.append(name)
            return result

        return call

    monkeypatch.setattr(torch.xpu, "synchronize", record("synchronize"))
    monkeypatch.setattr(torch.xpu, "reset_peak_memory_stats", record("reset_peak"))
    for name, value in (
        ("memory_allocated", 100),
        ("memory_reserved", 200),
        ("max_memory_allocated", 150),
        ("max_memory_reserved", 250),
    ):
        monkeypatch.setattr(torch.xpu, name, record(name, value))

    assert _xpu_memory_snapshot(selected, reset_peak=True) == {
        "allocated_bytes": 100,
        "reserved_bytes": 200,
        "peak_allocated_bytes": 150,
        "peak_reserved_bytes": 250,
    }
    assert events == [
        "synchronize",
        "reset_peak",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ]


def test_timing_memory_windows_exclude_queries_from_timed_calls(monkeypatch):
    from fairchem.core.scripts import validate_xpu

    atoms = harmonic_atoms()
    events = []
    original_evaluate = evaluate

    def prediction(*args):
        events.append("evaluate")
        return original_evaluate(*args)

    def snapshot(device, *, reset_peak=False):
        assert device == torch.device("xpu:1")
        events.append("reset_and_read" if reset_peak else "read")
        return {"allocated_bytes": 100, "peak_allocated_bytes": 150}

    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", snapshot)
    monkeypatch.setattr(validate_xpu, "evaluate", prediction)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda _: events.append("sync"))
    monkeypatch.setattr(
        validate_xpu.time, "perf_counter", lambda: events.append("clock") or 0.0
    )
    clear_cache = Mock()
    monkeypatch.setattr(torch.xpu, "empty_cache", clear_cache)
    report = synchronized_timings(
        atoms, torch.device("xpu:1"), 1, 2, measure_memory=True
    )
    assert events == [
        "reset_and_read",
        "evaluate",
        "read",
        "reset_and_read",
        "sync",
        "clock",
        "evaluate",
        "sync",
        "clock",
        "sync",
        "clock",
        "evaluate",
        "sync",
        "clock",
        "read",
    ]
    assert set(report["xpu_memory"]) == {"warmup", "repeated_evaluation"}
    assert report["xpu_memory"]["repeated_evaluation"]["after"] == {
        "allocated_bytes": 100,
        "peak_allocated_bytes": 150,
    }
    clear_cache.assert_not_called()


def test_cpu_timings_do_not_fabricate_xpu_memory(monkeypatch):
    from fairchem.core.scripts import validate_xpu

    read_memory = Mock(side_effect=AssertionError("XPU memory queried for CPU"))
    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", read_memory)
    report = synchronized_timings(
        harmonic_atoms(), torch.device("cpu"), 0, 1, measure_memory=True
    )
    assert "xpu_memory" not in report
    read_memory.assert_not_called()


def test_stress_finite_difference_checks_shear_and_restores_cell():
    atoms = periodic_harmonic_atoms()
    before_cell, before_positions = atoms.cell.copy(), atoms.positions.copy()
    stress = evaluate(atoms, stress=True)["stress"]
    report = finite_difference_stress_check(atoms, stress, 0.001, 1e-10)
    assert report["passed"]
    np.testing.assert_allclose(
        report["finite_difference_eV_per_A3"], stress, atol=1e-10
    )
    np.testing.assert_array_equal(atoms.cell, before_cell)
    np.testing.assert_array_equal(atoms.positions, before_positions)
    assert not atoms.calc.results
    wrong_stress = stress.copy()
    wrong_stress[3:] *= 2
    assert not finite_difference_stress_check(atoms, wrong_stress, 0.001, 1e-10)[
        "passed"
    ]


def test_stress_finite_difference_restores_structure_after_failure(monkeypatch):
    atoms = periodic_harmonic_atoms()
    before_cell, before_positions = atoms.cell.copy(), atoms.positions.copy()
    monkeypatch.setattr(
        atoms.calc, "calculate", Mock(side_effect=RuntimeError("synthetic failure"))
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        finite_difference_stress_check(atoms, np.zeros(6), 0.001, 1e-10)
    np.testing.assert_array_equal(atoms.cell, before_cell)
    np.testing.assert_array_equal(atoms.positions, before_positions)


def test_timings_reject_a_later_nonfinite_prediction(monkeypatch):
    atoms = harmonic_atoms()
    original_calculate = atoms.calc.calculate

    def calculate(*args, **kwargs):
        original_calculate(*args, **kwargs)
        if atoms.calc.calls >= 2:
            atoms.calc.results["forces"][0, 0] = np.nan

    monkeypatch.setattr(atoms.calc, "calculate", calculate)
    with pytest.raises(ValueError, match="Nonfinite forces"):
        synchronized_timings(atoms, torch.device("cpu"), warmup=1, repeats=2)


def test_comparison_uses_energy_per_atom_and_rejects_nonfinite():
    reference = {"energy": 10.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    candidate = {"energy": 10.1, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    tolerances = {
        "energy_eV_per_atom": 0.06,
        "forces_eV_per_A": 1e-3,
        "stress_eV_per_A3": 1e-3,
    }
    result = compare_predictions(reference, candidate, 2, tolerances)
    assert result["passed"]
    assert result["max_absolute_errors"]["energy_eV_per_atom"] == pytest.approx(0.05)
    candidate["forces"][0, 0] = np.nan
    assert not compare_predictions(reference, candidate, 2, tolerances)["passed"]


@pytest.mark.parametrize(
    "name,malformed",
    [("forces", np.zeros(3)), ("forces", np.zeros((1, 3))), ("stress", np.zeros(1))],
)
def test_comparison_rejects_broadcastable_wrong_shapes(name, malformed):
    reference = {"energy": 0.0, "forces": np.zeros((2, 3)), "stress": np.zeros(6)}
    candidate = {**reference, name: malformed}
    with pytest.raises(ValueError, match=f"Prediction {name} has shape"):
        compare_predictions(reference, candidate, 2, {})


def test_comparison_requires_stress_on_both_sides():
    reference = {"energy": 0.0, "forces": np.zeros((2, 3))}
    candidate = {**reference, "stress": np.zeros(6)}
    with pytest.raises(ValueError, match="Missing prediction: stress"):
        compare_predictions(reference, candidate, 2, {})


def test_model_inventory_rejects_wrong_device_and_dtype():
    predictor = SimpleNamespace(model=torch.nn.Linear(2, 1))
    assert model_tensor_inventory(predictor, torch.device("cpu"))
    with pytest.raises(RuntimeError, match="not xpu:0"):
        model_tensor_inventory(predictor, torch.device("xpu:0"))
    predictor.model.double()
    with pytest.raises(RuntimeError, match="not FP32"):
        model_tensor_inventory(predictor, torch.device("cpu"))


def test_fp64_probe_math_on_cpu():
    # This verifies the probe, not availability of XPU kernels.
    result = probe_fp64_reduction(torch.device("cpu"))
    assert result["passed"]
    assert result["gradient"] == [8.0, 4.0, 8.0]


def test_multidimensional_fp64_probe_math_on_cpu():
    # Known per-system sums and their squared-loss gradients verify this
    # diagnostic's arithmetic; running on CPU does not test XPU kernels.
    result = probe_fp64_reduction(torch.device("cpu"))["multidimensional"]
    assert result["passed"]
    assert result["dtype"] == "torch.float64"
    assert result["device"] == "cpu"
    assert result["source_shape"] == [3, 3, 2]
    assert not result["source_contiguous"]
    assert result["reduction"] == [
        [[14, 20], [16, 22], [18, 24]],
        [[7, 10], [8, 11], [9, 12]],
    ]
    assert result["gradient"] == [
        [[28, 40], [32, 44], [36, 48]],
        [[14, 20], [16, 22], [18, 24]],
        [[28, 40], [32, 44], [36, 48]],
    ]


def test_case_reporting_with_synthetic_calculator(monkeypatch):
    import fairchem.core

    def loader(model, device, inference_settings):
        return SimpleNamespace(
            model=torch.nn.Linear(2, 1), inference_settings=inference_settings
        )

    monkeypatch.setattr(fairchem.core.pretrained_mlip, "get_predict_unit", loader)
    monkeypatch.setattr(
        fairchem.core, "FAIRChemCalculator", lambda *a, **kw: HarmonicCalculator()
    )
    args = SimpleNamespace(
        model="synthetic",
        compile=False,
        fd_step=0.005,
        fd_tolerance=1e-9,
        warmup=0,
        repeats=1,
    )
    values, report = run_case(
        args, harmonic_atoms(), "omol", torch.device("cpu"), False
    )
    assert values["energy"] == pytest.approx(0.525)
    assert report["settings"]["base_precision_dtype"] == "float32"
    assert report["finite_difference"]["passed"]
    assert report["first_evaluation_seconds"] >= 0
    assert "xpu_memory" not in report


def _case_test_setup(monkeypatch):
    import fairchem.core
    from fairchem.core.scripts import validate_xpu

    monkeypatch.setattr(
        fairchem.core.pretrained_mlip,
        "get_predict_unit",
        lambda model, device, inference_settings: SimpleNamespace(
            model=torch.nn.Linear(2, 1), inference_settings=inference_settings
        ),
    )
    monkeypatch.setattr(
        fairchem.core, "FAIRChemCalculator", lambda *a, **kw: HarmonicCalculator()
    )
    monkeypatch.setattr(validate_xpu, "model_tensor_inventory", lambda *a: {})
    monkeypatch.setattr(torch.xpu, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.xpu, "device", lambda _: nullcontext())
    return SimpleNamespace(
        model="synthetic",
        compile=False,
        fd_step=0.005,
        fd_tolerance=1e-9,
        warmup=1,
        repeats=2,
    )


def test_case_reports_distinct_load_first_warmup_and_repeated_memory(monkeypatch):
    from fairchem.core.scripts import validate_xpu

    args = _case_test_setup(monkeypatch)
    snapshots = []

    def snapshot(device, *, reset_peak=False):
        assert device == torch.device("xpu:1")
        snapshots.append(reset_peak)
        return {"allocated_bytes": len(snapshots)}

    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", snapshot)
    clear_cache = Mock()
    monkeypatch.setattr(torch.xpu, "empty_cache", clear_cache)
    atoms = harmonic_atoms()
    original_calculator = atoms.calc
    _, report = run_case(args, atoms, "omol", torch.device("xpu:1"), False)
    assert atoms.calc is original_calculator
    assert snapshots == [True, False] * 4
    windows = report["xpu_memory"]["windows"]
    assert list(windows) == [
        "load_model",
        "first_evaluation",
        "warmup",
        "repeated_evaluation",
    ]
    assert windows["load_model"]["after"]["allocated_bytes"] == 2
    assert windows["first_evaluation"]["after"]["allocated_bytes"] == 4
    assert windows["warmup"]["after"]["allocated_bytes"] == 6
    assert windows["repeated_evaluation"]["after"]["allocated_bytes"] == 8
    assert "xpu_memory" not in report["timing"]
    assert "excludes CPU and non-PyTorch allocations" in report["xpu_memory"]["scope"]
    clear_cache.assert_called_once_with()


@pytest.mark.parametrize(
    "failure_stage",
    ["load_model", "first_evaluation", "force_finite_difference", "timing"],
)
def test_case_failure_releases_owned_calculator_and_runs_cleanup(
    monkeypatch, failure_stage
):
    import fairchem.core
    from fairchem.core.scripts import validate_xpu

    args = _case_test_setup(monkeypatch)
    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", lambda *a, **kw: {})
    clear_cache = Mock()
    monkeypatch.setattr(torch.xpu, "empty_cache", clear_cache)
    collected = Mock()
    monkeypatch.setattr(validate_xpu.gc, "collect", collected)
    copied_structures = []
    original_evaluate = evaluate

    def prediction(atoms, *args):
        copied_structures.append(atoms)
        if failure_stage == "first_evaluation":
            raise RuntimeError("synthetic failure")
        return original_evaluate(atoms, *args)

    monkeypatch.setattr(validate_xpu, "evaluate", prediction)
    failing = Mock(side_effect=RuntimeError("synthetic failure"))
    if failure_stage == "load_model":
        monkeypatch.setattr(fairchem.core.pretrained_mlip, "get_predict_unit", failing)
    elif failure_stage == "force_finite_difference":
        monkeypatch.setattr(validate_xpu, "finite_difference_check", failing)
    elif failure_stage == "timing":
        monkeypatch.setattr(validate_xpu, "synchronized_timings", failing)
    atoms = harmonic_atoms()
    original_calculator = atoms.calc
    progress = {}
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_case(args, atoms, "omol", torch.device("xpu:1"), False, progress=progress)
    assert atoms.calc is original_calculator
    assert all(structure.calc is None for structure in copied_structures)
    assert progress["stage"] == failure_stage
    clear_cache.assert_called_once_with()
    collected.assert_called_once_with()


def test_cleanup_failure_preserves_original_error_and_stage(monkeypatch):
    import fairchem.core
    from fairchem.core.scripts import validate_xpu

    args = _case_test_setup(monkeypatch)
    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", lambda *a, **kw: {})
    monkeypatch.setattr(
        fairchem.core.pretrained_mlip,
        "get_predict_unit",
        Mock(side_effect=RuntimeError("original load error")),
    )
    monkeypatch.setattr(
        torch.xpu, "empty_cache", Mock(side_effect=RuntimeError("cleanup failure"))
    )
    progress = {}
    with pytest.raises(RuntimeError, match="original load error"):
        run_case(
            args,
            harmonic_atoms(),
            "omol",
            torch.device("xpu:1"),
            False,
            progress=progress,
        )
    assert progress["stage"] == "load_model"
    assert progress["cleanup_error"] == {
        "type": "RuntimeError",
        "message": "cleanup failure",
    }


def test_cleanup_failure_after_success_is_reported_as_cleanup(monkeypatch):
    from fairchem.core.scripts import validate_xpu

    args = _case_test_setup(monkeypatch)
    monkeypatch.setattr(validate_xpu, "_xpu_memory_snapshot", lambda *a, **kw: {})
    monkeypatch.setattr(
        torch.xpu, "empty_cache", Mock(side_effect=RuntimeError("cleanup failure"))
    )
    progress = {}
    with pytest.raises(RuntimeError, match="cleanup failure"):
        run_case(
            args,
            harmonic_atoms(),
            "omol",
            torch.device("xpu:1"),
            False,
            progress=progress,
        )
    assert progress["stage"] == "cleanup"


def test_periodic_case_reports_stress_finite_differences(monkeypatch):
    import fairchem.core

    monkeypatch.setattr(
        fairchem.core.pretrained_mlip,
        "get_predict_unit",
        lambda model, device, inference_settings: SimpleNamespace(
            model=torch.nn.Linear(2, 1), inference_settings=inference_settings
        ),
    )
    monkeypatch.setattr(
        fairchem.core, "FAIRChemCalculator", lambda *a, **kw: HarmonicStressCalculator()
    )
    args = SimpleNamespace(
        model="synthetic",
        compile=False,
        fd_step=0.005,
        fd_tolerance=1e-9,
        stress_fd_step=0.001,
        stress_fd_tolerance=1e-9,
        warmup=0,
        repeats=1,
    )
    _, report = run_case(
        args, periodic_harmonic_atoms(), "omat", torch.device("cpu"), False, stress=True
    )
    assert report["stress_finite_difference"]["passed"]
    assert report["finite_difference"]["passed"]


def mock_xpu(monkeypatch):
    from fairchem.core.scripts import validate_xpu

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda _: "synthetic XPU")
    monkeypatch.setattr(
        torch.xpu, "get_device_properties", lambda _: SimpleNamespace(has_fp64=True)
    )
    monkeypatch.setattr(
        validate_xpu, "probe_fp64_reduction", lambda _: {"passed": True}
    )


@pytest.mark.parametrize("failing_backend", ["cpu", "xpu_unmerged", "xpu_merged"])
def test_main_rejects_wrong_stress_even_when_cpu_and_xpu_agree(
    tmp_path, monkeypatch, failing_backend
):
    from fairchem.core.scripts import validate_xpu

    mock_xpu(monkeypatch)

    def run(*args):
        device, merge_mole = args[3:5]
        backend = (
            "cpu"
            if device.type == "cpu"
            else "xpu_merged"
            if merge_mole
            else "xpu_unmerged"
        )
        values = {"energy": 0.0, "forces": np.zeros((4, 3)), "stress": np.zeros(6)}
        return values, {
            "finite_difference": {"passed": True},
            "stress_finite_difference": {"passed": backend != failing_backend},
        }

    monkeypatch.setattr(validate_xpu, "run_case", run)
    output = tmp_path / "validation.json"
    assert (
        main(
            [
                "--case",
                "periodic",
                "--stress",
                "--compare-merged",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    report = json.loads(output.read_text())
    assert report["cases"]["periodic"]["cpu_vs_xpu"]["passed"]
    assert not report["cases"]["periodic"]["passed"]


def test_main_retains_failure_stage_and_completed_cpu_report(tmp_path, monkeypatch):
    from fairchem.core.scripts import validate_xpu

    mock_xpu(monkeypatch)

    def run(args, atoms, task, device, merge_mole, stress, progress):
        progress.update(device=str(device), stage="force_finite_difference")
        if device.type == "xpu":
            raise RuntimeError("synthetic kernel failure")
        return {}, {"finite_difference": {"passed": True}}

    monkeypatch.setattr(validate_xpu, "run_case", run)
    output = tmp_path / "validation.json"
    assert main(["--case", "molecule", "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["active_check"] == {
        "case": "molecule",
        "device": "xpu:0",
        "stage": "force_finite_difference",
    }
    assert report["cases"]["molecule"]["cpu"]["finite_difference"]["passed"]
    assert report["error"]["message"] == "synthetic kernel failure"


def test_no_xpu_fails_before_loading_model_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    output = tmp_path / "validation.json"
    assert main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["passed"]
    assert "CPU fallback is not permitted" in report["error"]["message"]
    assert report["cases"] == {}


def test_series_a_options_preserve_cpu_reference_and_scope_merged_operations():
    args = build_parser().parse_args(
        [
            "--neighbor-skin",
            "0.4",
            "--neighbor-cache-entries",
            "2",
            "--graph-search-method",
            "bvh",
            "--graph-edge-order",
            "spatial",
            "--edge-chunk-size",
            "128",
            "--compile",
            "--compile-scope",
            "blocks",
            "--execution-mode",
            "umas_fast_portable",
            "--compare-merged",
            "--xpu-graph",
            "--xpu-graph-cache-size",
            "2",
        ]
    )
    cpu = case_settings(args, torch.device("cpu"), False, True)
    unmerged = case_settings(args, torch.device("xpu:1"), False, True)
    merged = case_settings(args, torch.device("xpu:1"), True, True)
    for settings in (cpu, unmerged, merged):
        assert settings.base_precision_dtype == torch.float32
        assert not settings.tf32
        assert not settings.nitro
        assert settings.requested_properties == ("energy", "forces", "stress")
    assert cpu.execution_mode == unmerged.execution_mode == "general"
    assert not cpu.compile
    assert not cpu.xpu_graph
    assert cpu.neighbor_skin == 0
    assert cpu.graph_edge_order == "original"
    assert cpu.graph_search_method == "cell"
    assert cpu.edge_chunk_size is None
    for settings in (unmerged, merged):
        assert settings.neighbor_skin == 0.4
        assert settings.neighbor_cache_entries == 2
        assert settings.graph_search_method == "bvh"
        assert settings.graph_edge_order == "spatial"
        assert settings.edge_chunk_size == 128
        assert settings.compile
        assert settings.compile_scope == "blocks"
    assert not unmerged.xpu_graph
    assert merged.xpu_graph
    assert merged.xpu_graph_cache_size == 2
    assert merged.execution_mode == "umas_fast_portable"


@pytest.mark.parametrize(
    "options,message",
    [
        (["--neighbor-skin", "nan"], "finite and nonnegative"),
        (["--neighbor-skin", "-1"], "finite and nonnegative"),
        (["--neighbor-cache-entries", "0"], "positive integer"),
        (["--edge-chunk-size", "0"], "positive"),
        (["--xpu-graph-cache-size", "0"], "positive integer"),
        (["--execution-mode", "umas_fast_portable"], "--compare-merged"),
        (["--xpu-graph"], "--compare-merged"),
        (["--compile-scope", "blocks"], "requires --compile"),
    ],
)
def test_invalid_series_a_options_fail_before_loading(
    tmp_path, monkeypatch, options, message
):
    from fairchem.core import pretrained_mlip

    load = Mock(side_effect=AssertionError("No checkpoint should be loaded"))
    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", load)
    output = tmp_path / "validation.json"
    assert main([*options, "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert report["error"]["type"] == "ValueError"
    assert message in report["error"]["message"]
    load.assert_not_called()


@pytest.mark.parametrize(
    "flag,value",
    [("--repeats", "0"), ("--fd-step", "nan"), ("--stress-fd-step", "nan")],
)
def test_invalid_options_fail_with_report(tmp_path, flag, value):
    output = tmp_path / "validation.json"
    assert main([flag, value, "--output", str(output)]) == 1
    assert json.loads(output.read_text())["error"]["type"] == "ValueError"


def test_nonfinite_json_is_standards_compliant():
    result = json.dumps(_json_safe({"energy": float("nan")}), allow_nan=False)
    assert json.loads(result) == {"energy": "nan"}

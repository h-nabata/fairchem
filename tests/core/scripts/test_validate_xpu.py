"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
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

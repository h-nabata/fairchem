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

from fairchem.core.scripts.validate_xpu import (
    _json_safe,
    compare_predictions,
    evaluate,
    finite_difference_check,
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


def test_no_xpu_fails_before_loading_model_and_writes_json(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    output = tmp_path / "validation.json"
    assert main(["--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report["passed"]
    assert "CPU fallback is not permitted" in report["error"]["message"]
    assert report["cases"] == {}


@pytest.mark.parametrize("flag,value", [("--repeats", "0"), ("--fd-step", "nan")])
def test_invalid_options_fail_with_report(tmp_path, flag, value):
    output = tmp_path / "validation.json"
    assert main([flag, value, "--output", str(output)]) == 1
    assert json.loads(output.read_text())["error"]["type"] == "ValueError"


def test_nonfinite_json_is_standards_compliant():
    result = json.dumps(_json_safe({"energy": float("nan")}), allow_nan=False)
    assert json.loads(result) == {"energy": "nan"}

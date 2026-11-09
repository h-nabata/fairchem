"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes

from fairchem.core.scripts import benchmark_md


class HarmonicCalculator(Calculator):
    implemented_properties = ("energy", "forces")

    def calculate(
        self, atoms=None, properties=("energy", "forces"), system_changes=all_changes
    ):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float(0.5 * np.sum(atoms.positions**2)),
            "forces": -atoms.positions.copy(),
        }


def test_md_records_drift_without_mutating_structure_rng_or_retaining_trajectory():
    atoms = Atoms("H2", positions=[[0.3, 0, 0], [0, 0.7, 0]])
    atoms.info["job"] = {"values": [1, 2]}
    before, rng = atoms.copy(), np.random.get_state()
    result = benchmark_md.run_md(
        atoms,
        HarmonicCalculator(),
        steps=20,
        warmup=2,
        sample_interval=7,
        timestep_fs=0.01,
        temperature_K=10,
    )
    assert result["status"] == "completed"
    assert result["max_energy_drift_eV_per_atom"] < 1e-6
    assert [sample["step"] for sample in result["samples"]] == [0, 7, 14, 20]
    assert all(sample["memory"] is None for sample in result["samples"])
    assert result["ns_per_day_at_timestep"] == pytest.approx(
        86.4 * 0.01 / result["step_mean_ms"]
    )
    np.testing.assert_array_equal(atoms.positions, before.positions)
    assert atoms.info == before.info
    assert "momenta" not in atoms.arrays
    np.testing.assert_array_equal(np.random.get_state()[1], rng[1])
    json.dumps(result)  # no tensors, dynamics objects or calculator references


@pytest.mark.parametrize(
    "argument",
    [
        {"steps": 0},
        {"warmup": -1},
        {"sample_interval": 0},
        {"timestep_fs": float("nan")},
        {"temperature_K": -1},
    ],
)
def test_md_rejects_invalid_configuration_before_calculation(argument):
    calculator = HarmonicCalculator()
    with pytest.raises(ValueError):
        benchmark_md.run_md(molecule("H2"), calculator, **argument)
    assert calculator.results == {}


def test_md_rejects_nonfinite_output_and_silent_nitro_fallback():
    calculator = HarmonicCalculator()
    with pytest.raises(RuntimeError, match="Nitro.*not active"):
        benchmark_md.run_md(molecule("H2"), calculator, require_nitro=True)

    class Invalid(HarmonicCalculator):
        def calculate(self, *args, **kwargs):
            super().calculate(*args, **kwargs)
            self.results["forces"][:] = float("nan")

    with pytest.raises(FloatingPointError, match="Nonfinite"):
        benchmark_md.run_md(molecule("H2"), Invalid(), steps=1, warmup=0)


def test_accelerator_memory_and_synchronization_use_requested_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):

        def operation(device, name=name):
            calls.append((name, str(device)))
            return 123

        monkeypatch.setattr(torch.cuda, name, operation)
    result = benchmark_md.run_md(
        molecule("H2"), HarmonicCalculator(), device="cuda:2", steps=2, warmup=1
    )
    assert all(device == "cuda:2" for _, device in calls)
    assert result["samples"][-1]["memory"]["allocated_bytes"] == 123


def test_cli_records_failure_before_checkpoint_download(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    destination = tmp_path / "failed.json"
    assert benchmark_md.main(["--device", "xpu:0", "--output", str(destination)]) == 1
    report = json.loads(destination.read_text())
    assert report["status"] == "failed"
    assert "unavailable" in report["error"]
    assert report["requested_device"] == "xpu:0"


def test_device_inferred_from_predictor_and_mismatch_rejected(monkeypatch):
    calculator = HarmonicCalculator()
    calculator.predictor = SimpleNamespace(device="cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    device, backend = benchmark_md._resolve_device(calculator, None)
    assert str(device) == "cuda:2"
    assert backend is torch.cuda
    for wrong in ("cpu", "cuda", "cuda:1"):
        with pytest.raises(ValueError, match="must match"):
            benchmark_md._resolve_device(calculator, wrong)


def test_supplied_momenta_have_explicit_provenance():
    atoms = molecule("H2")
    atoms.set_momenta(np.full((2, 3), 0.1))
    initial_temperature = atoms.get_temperature()
    result = benchmark_md.run_md(
        atoms, HarmonicCalculator(), steps=1, warmup=0, temperature_K=0
    )
    assert result["momentum_source"] == "input"
    assert result["initial_temperature_K"] == pytest.approx(initial_temperature)
    assert result["initial_temperature_K"] > 0
    np.testing.assert_array_equal(atoms.get_momenta(), np.full((2, 3), 0.1))


@pytest.mark.parametrize("argument", [["--steps", "0"], ["--timestep-fs", "nan"]])
def test_cli_rejects_invalid_run_before_model_load(tmp_path, monkeypatch, argument):
    from fairchem.core import pretrained_mlip

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid benchmark configuration must not load/download a model")

    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", unexpected_load)
    destination = tmp_path / "invalid.json"
    assert benchmark_md.main([*argument, "--output", str(destination)]) == 1
    assert json.loads(destination.read_text())["status"] == "failed"


def test_real_small_uma_md_has_owned_inputs_and_finite_outputs():
    from fairchem.core import FAIRChemCalculator
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    atoms = molecule("H2O")
    atoms.set_cell([10, 10, 10])
    before = atoms.copy()
    predictor = _public_predictor()
    calculator = FAIRChemCalculator(predictor, task_name="omol")
    result = benchmark_md.run_md(
        atoms, calculator, steps=4, warmup=1, timestep_fs=0.001, temperature_K=0
    )
    assert result["status"] == "completed"
    assert np.isfinite(result["max_energy_drift_eV_per_atom"])
    np.testing.assert_array_equal(atoms.positions, before.positions)
    assert atoms.calc is None
    assert not predictor.inference_settings.nitro


def test_runtime_diagnostics_are_owned_numeric_snapshots(monkeypatch):
    state = {"captures": 2, "active_capacities": {"edge_capacity": 128}}
    module = SimpleNamespace(runtime_cache_info=lambda: copy.deepcopy(state))
    monkeypatch.setitem(
        benchmark_md.sys.modules,
        "fairchem.core.models.uma.uma_nitro._inference",
        module,
    )
    snapshot = benchmark_md._nitro_runtime()
    state["active_capacities"]["edge_capacity"] = 256
    assert snapshot["active_capacities"]["edge_capacity"] == 128


def test_md_cli_runs_actual_fp64_small_uma_and_records_precision(tmp_path, monkeypatch):
    from ase.io import write

    from fairchem.core import pretrained_mlip
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    settings_seen = []

    def load(model, *, device, inference_settings, seed):
        settings_seen.append(inference_settings)
        predictor = _public_predictor()
        predictor.inference_settings = inference_settings
        return predictor

    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", load)
    source, destination = tmp_path / "water.extxyz", tmp_path / "md64.json"
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    write(source, atoms)
    assert (
        benchmark_md.main(
            [
                "--device",
                "cpu",
                "--inference-settings",
                "xpu",
                "--dtype",
                "float64",
                "--input",
                str(source),
                "--task",
                "omol",
                "--steps",
                "2",
                "--warmup",
                "0",
                "--timestep-fs",
                "0.001",
                "--output",
                str(destination),
            ]
        )
        == 0
    )
    report = json.loads(destination.read_text())
    assert report["base_precision_dtype"] == "torch.float64"
    assert settings_seen[0].base_precision_dtype == torch.float64
    assert not settings_seen[0].compile
    assert np.isfinite(report["max_energy_drift_eV_per_atom"])


def test_md_fp64_probe_precedes_checkpoint_download(tmp_path, monkeypatch):
    from fairchem.core import pretrained_mlip
    from fairchem.core.units.mlip_unit import inference_runtime

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        inference_runtime,
        "probe_fp64_support",
        lambda device: (_ for _ in ()).throw(RuntimeError("unsupported FP64")),
    )
    monkeypatch.setattr(
        pretrained_mlip,
        "get_predict_unit",
        lambda *a, **kw: pytest.fail("FP64 support must be checked before loading"),
    )
    destination = tmp_path / "failed-fp64.json"
    assert (
        benchmark_md.main(
            [
                "--device",
                "xpu:0",
                "--inference-settings",
                "xpu",
                "--dtype",
                "float64",
                "--output",
                str(destination),
            ]
        )
        == 1
    )
    assert "unsupported FP64" in json.loads(destination.read_text())["error"]


def test_md_rejects_fp64_nitro_before_loading_a_checkpoint(tmp_path, monkeypatch):
    from fairchem.core import pretrained_mlip

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        pretrained_mlip,
        "get_predict_unit",
        lambda *a, **kw: pytest.fail("FP64 Nitro must fail before loading"),
    )
    destination = tmp_path / "nitro64.json"
    assert (
        benchmark_md.main(
            [
                "--device",
                "cuda:0",
                "--inference-settings",
                "nitro",
                "--dtype",
                "float64",
                "--output",
                str(destination),
            ]
        )
        == 1
    )
    assert "Nitro" in json.loads(destination.read_text())["error"]

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from dataclasses import asdict

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.optimize import BFGS, FIRE

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor


def _atomic_data(atoms):
    return AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["charge", "spin"])


@pytest.mark.parametrize(
    "optimizer_cls,options",
    [
        (BFGS, SeriesBSettings(gemm_precision="bf16", low_precision_emulation=True)),
        (FIRE, SeriesBSettings(weight_bits=8)),
    ],
)
def test_real_uma_ase_optimizer_from_startup_settings(
    monkeypatch, optimizer_cls, options
):
    # Only checkpoint acquisition is substituted. Calculator dispatch, UMA
    # preparation/forward/autograd, output conversion and ASE optimizers are real.
    reference = _public_predictor()
    candidate = copy.deepcopy(reference)
    settings = copy.deepcopy(reference.inference_settings)
    settings.series_b = options
    settings_before = asdict(settings)
    candidate.inference_settings = copy.deepcopy(settings)
    oracle = copy.deepcopy(candidate)

    def load(name, **kwargs):
        assert name == "local-small-uma.pt"
        assert kwargs["device"] == "cpu"
        assert kwargs["inference_settings"] is settings
        return candidate

    monkeypatch.setattr(pretrained_mlip, "get_predict_unit", load)
    source = molecule("H2O")
    source.set_cell([8, 8, 8])
    source.info.update(charge=0, spin=1)
    source_before = source.positions.copy()
    strict_atoms = source.copy()
    strict_atoms.calc = FAIRChemCalculator(reference, task_name="omol")
    strict_energy = strict_atoms.get_potential_energy()
    strict_forces = strict_atoms.get_forces()
    strict_settings = asdict(reference.inference_settings)

    working = source.copy()
    working.calc = FAIRChemCalculator.from_model_checkpoint(
        "local-small-uma.pt",
        inference_settings=settings,
        task_name="omol",
        device="cpu",
    )
    with optimizer_cls(working, logfile=None) as optimizer:
        optimizer.run(fmax=0, steps=2)
        assert optimizer.nsteps == 2
    energy = working.get_potential_energy()
    forces = working.get_forces()
    assert np.isfinite(energy)
    assert forces.shape == (3, 3)
    assert np.isfinite(forces).all()
    assert np.count_nonzero(forces)
    assert not np.array_equal(working.positions, source_before)

    data = _atomic_data(working)
    before = data.clone()
    expected = oracle.predict(data, requested_properties=("energy", "forces"))
    np.testing.assert_allclose(energy, expected["energy"].item(), atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(
        forces, expected["forces"].detach().numpy(), atol=2e-5, rtol=2e-4
    )
    for name in ("pos", "cell", "atomic_numbers", "charge", "spin"):
        torch.testing.assert_close(data[name], before[name], atol=0, rtol=0)
    np.testing.assert_array_equal(source.positions, source_before)
    assert asdict(settings) == settings_before
    assert asdict(reference.inference_settings) == strict_settings
    assert candidate.approximation_manifest["enabled"]

    # Force a fresh FP32 calculation after the approximate job, so this checks
    # execution isolation rather than simply reading an unchanged ASE cache.
    strict_atoms.calc.reset()
    np.testing.assert_allclose(
        strict_atoms.get_potential_energy(), strict_energy, atol=2e-8, rtol=2e-5
    )
    np.testing.assert_allclose(
        strict_atoms.get_forces(), strict_forces, atol=2e-8, rtol=2e-5
    )
    assert reference.approximation_manifest == {"enabled": False}


def test_real_int8_uma_torchsim_fire_preserves_fp32_predictor_and_input(monkeypatch):
    pytest.importorskip("torch_sim")
    from torch_sim.optimizers.fire import fire_init, fire_step

    from fairchem.core.calculate import torchsim_interface as interface
    from tests.core.calculate.test_torchsim_interop import _state, _wrapper

    reference = _public_predictor()
    candidate = copy.deepcopy(reference)
    candidate.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    oracle = copy.deepcopy(candidate)
    settings_before = asdict(candidate.inference_settings)
    state = _state()
    before = state.clone()
    original_data = interface._simstate_to_atomicdata_batch(state, "omol")
    strict_before = reference.predict(
        original_data, requested_properties=("energy", "forces")
    )
    model = _wrapper(monkeypatch, predictor=candidate, dtype=torch.float32)
    initial = model(state)
    initial_snapshot = {name: value.clone() for name, value in initial.items()}
    optimized = fire_init(state.clone(), model)
    optimized = fire_step(optimized, model)
    actual = model(optimized)
    expected = oracle.predict(
        interface._simstate_to_atomicdata_batch(optimized, "omol"),
        requested_properties=("energy", "forces"),
    )
    for name in ("energy", "forces"):
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == torch.float32
        assert torch.isfinite(actual[name]).all()
        torch.testing.assert_close(
            actual[name], expected[name].to(torch.float32), atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(
            initial[name], initial_snapshot[name], atol=0, rtol=0
        )
    assert not torch.equal(optimized.positions, before.positions)
    for name in ("positions", "cell", "atomic_numbers"):
        torch.testing.assert_close(
            getattr(state, name), getattr(before, name), atol=0, rtol=0
        )
    assert asdict(candidate.inference_settings) == settings_before
    strict_after = reference.predict(
        original_data, requested_properties=("energy", "forces")
    )
    for name in strict_before:
        torch.testing.assert_close(
            strict_after[name], strict_before[name], atol=2e-8, rtol=2e-5
        )
    assert candidate.approximation_manifest["enabled"]
    assert reference.approximation_manifest == {"enabled": False}

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from ase.build import molecule

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


class RecordingPredictor:
    supports_requested_properties = True

    def __init__(self):
        self.inference_settings = InferenceSettings()
        self.dataset_to_tasks = {
            "omol": [
                SimpleNamespace(property=p) for p in ("energy", "forces", "stress")
            ]
        }
        self.requests = []

    def validate_atoms_data(self, atoms, task_name):
        pass

    def predict(self, data, requested_properties=None):
        self.requests.append(requested_properties)
        results = {
            "energy": data.pos.square().sum().reshape(1),
            "forces": -2 * data.pos,
            "stress": torch.eye(3).unsqueeze(0),
        }
        return {
            k: v
            for k, v in results.items()
            if requested_properties is None or k in requested_properties
        }


def test_ase_requests_only_needed_derivatives_and_retains_valid_results():
    predictor = RecordingPredictor()
    atoms = molecule("H2O")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    energy = atoms.get_potential_energy()
    assert predictor.requests == [("energy",)]
    forces = atoms.get_forces()
    assert predictor.requests[-1] == ("energy", "forces")
    atoms.get_stress()
    assert predictor.requests[-1] == ("energy", "stress")
    assert atoms.get_potential_energy() == energy
    np.testing.assert_array_equal(atoms.get_forces(), forces)
    assert len(predictor.requests) == 3
    atoms.positions += 0.2
    atoms.get_potential_energy(force_consistent=True)
    assert predictor.requests[-1] == ("energy",)
    assert "forces" not in atoms.calc.results
    assert "stress" not in atoms.calc.results


def test_ase_legacy_predictor_signature_remains_supported():
    predictor = RecordingPredictor()
    predictor.supports_requested_properties = False
    predict = predictor.predict
    predictor.predict = lambda data: predict(data)
    atoms = molecule("H2O")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.get_forces()
    atoms.get_stress()
    atoms.get_potential_energy()
    assert predictor.requests == [None]

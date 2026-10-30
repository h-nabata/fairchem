"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.calculators.lj import LennardJones
from ase.calculators.mixing import SumCalculator
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.mep import NEB
from ase.optimize import BFGS, FIRE
from ase.vibrations import Vibrations

from fairchem.core.calculate.ase_calculator import (
    FAIRChemCalculator,
    FormationEnergyCalculator,
)
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


class BufferPredictor:
    """
    A coordinate-dependent potential with deliberately reusable output storage.
    """

    supports_requested_properties = True

    def __init__(self):
        self.inference_settings = InferenceSettings()
        self.dataset_to_tasks = {
            "omol": [
                SimpleNamespace(property=p)
                for p in ("energy", "forces", "stress", "hessian")
            ]
        }
        self.calls = 0
        self.fail = False
        self.override = {}
        self.forces = torch.empty(3, 3)

    def validate_atoms_data(self, atoms, task_name):
        pass

    def predict(self, data, requested_properties=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("deliberate prediction failure")
        self.forces.copy_(-2 * data.pos)
        results = {
            "energy": data.pos.square().sum().reshape(1) + data.charge,
            "forces": self.forces,
            "stress": torch.eye(3).unsqueeze(0),
            "hessian": 2 * torch.eye(9).unsqueeze(0),
        }
        results.update(self.override)
        return {
            k: v
            for k, v in results.items()
            if requested_properties is None or k in requested_properties
        }


def _atoms():
    atoms = molecule("H2O")
    atoms.set_cell([7.0, 7.5, 8.0])
    atoms.info.update(charge=0, spin=1)
    return atoms


def test_metadata_arrays_and_nested_mutation_do_not_contaminate_cache():
    predictor = BufferPredictor()
    atoms = _atoms()
    atoms.info["job"] = {
        "vector": np.array([1.0, 2.0]),
        "frames": [{"tag": np.array([3, 4])}],
    }
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    atoms.get_potential_energy()
    atoms.get_potential_energy()
    assert predictor.calls == 1
    atoms.info["job"]["frames"][0]["tag"][0] = 10
    assert atoms.calc.atoms.info["job"]["frames"][0]["tag"][0] == 3
    atoms.get_potential_energy()
    assert predictor.calls == 2
    atoms.info["charge"] = 1
    before = atoms.calc.results["energy"]
    assert atoms.get_potential_energy() == pytest.approx(before + 1)


def test_calculators_own_cpu_prediction_arrays_across_shared_predictor_jobs():
    predictor = BufferPredictor()
    first, second = _atoms(), _atoms()
    second.positions += 2
    for atoms in (first, second):
        atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    expected = first.get_forces()
    second.get_forces()
    predictor.forces.add_(100)
    np.testing.assert_array_equal(first.get_forces(), expected)
    assert predictor.calls == 2
    assert first.calc.results["forces"].flags.owndata


def test_direct_calculate_defaults_and_failure_do_not_reuse_previous_job():
    predictor = BufferPredictor()
    calc = FAIRChemCalculator(predictor, task_name="omol")
    first, second = _atoms(), _atoms()
    second.positions += 1
    calc.calculate(first)
    calc.calculate()
    predictor.fail = True
    with pytest.raises(RuntimeError, match="deliberate"):
        calc.calculate_properties(second, ["energy"])
    assert calc.atoms is None
    assert not calc.results
    predictor.fail = False
    second.calc = calc
    assert second.get_potential_energy() == pytest.approx(
        float((second.positions**2).sum()), rel=2e-6
    )


@pytest.mark.parametrize("formation", [False, True])
def test_cancelled_calculation_does_not_reuse_previous_job(formation):
    predictor = BufferPredictor()
    calc = FAIRChemCalculator(predictor, task_name="omol")
    if formation:
        calc = FormationEnergyCalculator(
            calc, {"H": 0, "O": 0}, apply_corrections=False
        )
    calc.calculate(_atoms())

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    predictor.predict = interrupted
    moved = _atoms()
    moved.positions += 1
    with pytest.raises(KeyboardInterrupt):
        calc.calculate(moved)
    assert calc.atoms is None
    assert not calc.results


@pytest.mark.parametrize(
    "name,value",
    [
        ("energy", torch.zeros(2)),
        ("forces", torch.zeros(9)),
        ("stress", torch.zeros(4)),
        ("hessian", torch.zeros(3, 3)),
        ("energy", np.array([np.nan])),
        ("forces", np.full((3, 3), "bad")),
    ],
)
def test_malformed_wrapper_outputs_fail_without_poisoning_results(name, value):
    predictor = BufferPredictor()
    calc = FAIRChemCalculator(predictor, task_name="omol")
    calc.calculate(_atoms())
    predictor.override[name] = value
    with pytest.raises(ValueError):
        calc.calculate(_atoms(), [name])
    assert calc.atoms is None
    assert not calc.results


@pytest.mark.parametrize("shape", [(6,), (1, 6), (3, 3), (1, 3, 3), (9,), (1, 9)])
def test_standard_stress_layouts_preserve_voigt_order(shape):
    predictor = BufferPredictor()
    stress = np.array([[1.0, 4.0, 5.0], [4.0, 2.0, 6.0], [5.0, 6.0, 3.0]])
    expected = np.array([1.0, 2.0, 3.0, 6.0, 5.0, 4.0])
    predictor.override["stress"] = (
        expected if np.prod(shape) == 6 else stress
    ).reshape(shape)
    atoms = _atoms()
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
    np.testing.assert_array_equal(atoms.get_stress(), expected)


def test_formation_wrapper_caches_owns_references_and_invalidates_parameters():
    predictor = BufferPredictor()
    base = FAIRChemCalculator(predictor, task_name="omol")
    references = {"H": 1.0, "O": 2.0}
    wrapper = FormationEnergyCalculator(base, references, apply_corrections=False)
    atoms = _atoms()
    atoms.calc = wrapper
    forces = atoms.get_forces()
    energy = atoms.get_potential_energy()
    assert wrapper.atoms is not None
    assert predictor.calls == 1
    references["H"] = 100
    base.results["forces"][:] = 100
    np.testing.assert_array_equal(atoms.get_forces(), forces)
    assert atoms.get_potential_energy() == energy
    wrapper.element_references["H"] = 2
    assert atoms.get_potential_energy() == pytest.approx(energy - 2)
    assert predictor.calls == 2
    atoms.info["charge"] = 1
    assert atoms.get_potential_energy() == pytest.approx(energy - 1)
    wrapper.element_references.pop("O")
    with pytest.raises(ValueError, match="Missing reference"):
        atoms.get_potential_energy()
    assert wrapper.atoms is None
    assert not wrapper.results


def _real_calculator():
    from tests.core.units.mlip_unit.test_predict_memory import (
        _IdentityNormalizer,
        _task,
    )
    from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor

    predictor = _small_predictor()
    predictor.inference_settings.merge_mole = False
    predictor.inference_settings.activation_checkpointing = False
    backbone = predictor.model.module.backbone
    backbone.neighbor_skin = 0.4
    backbone.edge_chunk_size = 16
    backbone.graph_edge_order = "spatial"
    tasks = []
    for prop in ("energy", "forces", "stress", "hessian"):
        task = _task(f"omol_{prop}", "atom" if prop == "forces" else "system", prop)
        task.normalizer = _IdentityNormalizer()
        task.element_references = None
        tasks.append(task)
    predictor.model.module._tasks = {task.name: task for task in tasks}
    predictor.model.module._dataset_to_tasks = {"omol": tasks}
    return FAIRChemCalculator(predictor, task_name="omol")


@pytest.mark.parametrize("optimizer", [BFGS, FIRE])
def test_real_uma_optimizer_and_constraint_jobs(optimizer):
    atoms = _atoms()
    atoms.calc = _real_calculator()
    atoms.set_constraint(FixAtoms(indices=[0]))
    initial = atoms.positions.copy()
    forces = atoms.get_forces()
    assert np.count_nonzero(forces[1:]) > 0
    optimizer(atoms, logfile=None).run(fmax=1e-10, steps=2)
    np.testing.assert_array_equal(atoms.positions[0], initial[0])
    assert not np.array_equal(atoms.positions[1:], initial[1:])
    assert np.isfinite(atoms.get_forces()).all()
    assert atoms.calc.predictor.model.module.backbone._neighbor_cache is not None


def test_real_uma_neb_and_sum_calculator_job_outputs_are_independent():
    calculator = _real_calculator()
    images = [_atoms() for _ in range(3)]
    for i, atoms in enumerate(images):
        atoms.positions[1, 0] += 0.07 * i
        atoms.calc = FAIRChemCalculator(calculator.predictor, task_name="omol")
    neb = NEB(images, climb=True)
    assert np.isfinite(neb.get_forces()).all()
    first_force = images[0].get_forces()
    images[-1].get_forces()
    np.testing.assert_array_equal(images[0].get_forces(), first_force)
    atoms = _atoms()
    base, correction = _real_calculator(), LennardJones()
    expected_energy = base.get_potential_energy(
        atoms
    ) + correction.get_potential_energy(atoms)
    expected_forces = base.get_forces(atoms) + correction.get_forces(atoms)
    atoms.calc = SumCalculator([base, correction])
    assert atoms.get_potential_energy() == pytest.approx(expected_energy)
    np.testing.assert_allclose(atoms.get_forces(), expected_forces)


def test_real_uma_cell_filter_and_finite_difference_vibration_jobs(tmp_path):
    atoms = _atoms()
    atoms.pbc = True
    atoms.calc = _real_calculator()
    cell_filter = FrechetCellFilter(atoms, mask=[1, 1, 1, 0, 0, 0])
    assert cell_filter.get_forces().shape == (len(atoms) + 3, 3)
    assert np.isfinite(cell_filter.get_forces()).all()
    FIRE(cell_filter, logfile=None, dt=0.01).run(fmax=1e-12, steps=1)
    positions = atoms.positions.copy()
    vibrations = Vibrations(atoms, indices=[1], name=str(tmp_path / "vib"), delta=0.005)
    vibrations.run()
    hessian = vibrations.get_vibrations().get_hessian_2d()
    assert hessian.shape == (3, 3)
    assert np.isfinite(hessian).all()
    np.testing.assert_array_equal(atoms.positions, positions)
    vibrations.clean()


def test_real_uma_sella_job_accepts_numpy_hessian_callback():
    sella = pytest.importorskip("sella")
    atoms = _atoms()
    atoms.calc = _real_calculator()

    def hessian(current):
        value = current.calc.get_property("hessian", current)
        assert value.shape == (9, 9)
        assert np.isfinite(value).all()
        return value

    optimizer = sella.Sella(
        atoms, order=1, logfile=None, hessian_function=hessian, max_cpu_threads=1
    )
    optimizer.run(fmax=1e-12, steps=1)
    assert np.isfinite(atoms.positions).all()
    assert np.isfinite(atoms.get_forces()).all()

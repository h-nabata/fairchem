"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import units
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.verlet import VelocityVerlet

from fairchem.core.calculate import torchsim_interface as interface
from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
from fairchem.core.calculate.batched_inference import MicrobatchPredictor
from fairchem.core.calculate.device_buffers import DeviceBatchStager
from fairchem.core.datasets.atomic_data import AtomicData
from tests.core.components.test_batch_server_interop import _data, _server
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor


class _FusedPredictor:
    """
    CPU oracle for Nitro's single-system/fused-output protocol, not its kernels.
    """

    device = "cpu"
    supports_requested_properties = True

    def __init__(self, nitro=True):
        self.inference_settings = SimpleNamespace(
            nitro=nitro, external_graph_gen=False, base_precision_dtype=torch.float32
        )
        self.dataset_to_tasks = {
            "omol": [
                SimpleNamespace(property=name, level=level)
                for name, level in (
                    ("energy", "system"),
                    ("forces", "atom"),
                    ("stress", "system"),
                )
            ]
        }
        self.calls = []
        self.composition = None

    def validate_atoms_data(self, atoms, task_name):
        pass

    def predict(self, data, undo_element_references=True, requested_properties=None):
        composition = tuple(data.atomic_numbers.tolist())
        if data.num_graphs != 1 or (
            self.composition is not None and self.composition != composition
        ):
            self.inference_settings.nitro = False
        self.composition = composition
        self.calls.append((data.num_graphs, requested_properties))
        self.latest = {
            "energy": data.pos.new_zeros(data.num_graphs).index_add(
                0, data.batch, 0.5 * data.pos.square().sum(-1)
            ),
            "forces": -data.pos,
            "stress": torch.eye(3, device=data.pos.device)
            .expand(data.num_graphs, -1, -1)
            .clone(),
        }
        return {
            name: value
            for name, value in self.latest.items()
            if requested_properties is None or name in requested_properties
        }


def _atoms(cell=True):
    atoms = molecule("H2O")
    if cell:
        atoms.cell = [8, 9, 10]
    atoms.info.update(charge=0, spin=1)
    return atoms


@pytest.mark.parametrize("first", ["energy", "stress"])
@pytest.mark.parametrize("nitro", [False, True])
def test_ase_reuses_nitro_fused_outputs_without_changing_generic_requests(first, nitro):
    predictor = _FusedPredictor(nitro)
    atoms = _atoms()
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    getters = {
        "energy": atoms.get_potential_energy,
        "forces": atoms.get_forces,
        "stress": atoms.get_stress,
    }
    getters[first]()
    results = {name: method() for name, method in getters.items()}
    assert len(predictor.calls) == (1 if nitro else 2 if first == "stress" else 3)
    if nitro:
        assert predictor.calls[0][1] == ("energy", "forces", "stress")
    expected = {name: np.array(value, copy=True) for name, value in results.items()}
    for value in predictor.latest.values():
        value.zero_()
    for name, method in getters.items():
        np.testing.assert_array_equal(method(), expected[name])


def test_nitro_energy_without_cell_does_not_add_optional_stress():
    predictor = _FusedPredictor()
    atoms = _atoms(cell=False)
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    atoms.get_potential_energy()
    assert predictor.calls[0][1] == ("energy", "forces")


def test_nitro_energy_in_inference_mode_does_not_add_derivative_requests():
    predictor = _FusedPredictor()
    atoms = _atoms()
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    with torch.inference_mode():
        atoms.get_potential_energy()
    assert predictor.calls[0][1] == ("energy",)


def test_ase_after_nitro_fallback_stops_requesting_extra_derivatives():
    predictor = _FusedPredictor()
    atoms = _atoms()
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    atoms.get_potential_energy()
    atoms.numbers[0] = 7
    atoms.get_potential_energy()
    assert not predictor.inference_settings.nitro
    atoms.positions += 0.1
    atoms.get_potential_energy()
    assert predictor.calls[-1][1] == ("energy",)


def test_ase_velocity_verlet_with_stress_observer_uses_one_fused_call_per_step():
    predictor = _FusedPredictor()
    atoms = _atoms()
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    atoms.set_velocities(np.full((3, 3), 0.01))
    energies = []

    def observe():
        energies.append(atoms.get_total_energy())
        assert np.isfinite(atoms.get_stress()).all()

    dynamics = VelocityVerlet(atoms, timestep=0.01 * units.fs, logfile=None)
    dynamics.attach(observe, interval=1)
    dynamics.run(20)
    assert len(predictor.calls) == 21
    assert max(energies) - min(energies) < 1e-5
    assert predictor.inference_settings.nitro


@pytest.mark.parametrize("slots", [None, 1, 2])
def test_microbatch_does_not_disable_nitro_by_collating_independent_steps(slots):
    predictor = _FusedPredictor()
    stager = None if slots is None else DeviceBatchStager("cpu", slots=slots)
    runner = MicrobatchPredictor(predictor, stager=stager)
    inputs = [_data(0.7), _data(0.9), _data(1.1)]
    output = runner.predict_many(inputs)
    assert [systems for systems, _ in predictor.calls] == [1, 1, 1]
    assert predictor.inference_settings.nitro
    for data, result in zip(inputs, output):
        torch.testing.assert_close(result["forces"], -data.pos)
    if stager is not None:
        stager.close()


def test_server_does_not_disable_nitro_by_collating_independent_steps():
    predictor = _FusedPredictor()
    inputs = [_data(0.7), _data(0.9), _data(1.1)]
    results = _server()._run_batched_inference(inputs, predictor, False)
    assert [systems for systems, _ in predictor.calls] == [1, 1, 1]
    assert predictor.inference_settings.nitro
    for data, result in zip(inputs, results):
        torch.testing.assert_close(result["forces"], -data.pos)


@pytest.mark.parametrize("kind", ["microbatch", "server"])
def test_scheduler_uses_effective_backend_after_composition_fallback(kind):
    predictor = _FusedPredictor()
    if kind == "microbatch":
        run = MicrobatchPredictor(predictor).predict_many
    else:

        def run(data):
            return _server()._run_batched_inference(data, predictor, False)

    inputs = [_data(0.7), _data(0.9)]
    inputs[1].atomic_numbers[0] = 6
    run(inputs)
    assert not predictor.inference_settings.nitro
    predictor.calls.clear()
    run(inputs)
    assert [systems for systems, _ in predictor.calls] == [2]


@pytest.mark.parametrize("periodic", [False, True])
def test_torchsim_equivalent_clones_reuse_metadata_and_release_previous_state(
    monkeypatch, periodic
):
    pytest.importorskip("torch_sim")
    from tests.core.calculate.test_torchsim_resident_metadata import _state

    state = _state(periodic)
    converter = interface._SimStateConverter()
    original = interface._simstate_to_atomicdata_batch
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(interface, "_simstate_to_atomicdata_batch", counted)
    previous_owner = weakref.ref(state.atomic_numbers)
    previous = converter(state, "omol", torch.float32)
    previous_positions = previous.pos.clone()
    for _ in range(5):
        state = state.clone()
        state.positions += 0.01
        actual = converter(state, "omol", torch.float32)
        expected = original(state, "omol", torch.float32)
        for field in ("pos", "cell", "atomic_numbers", "pbc", "charge", "spin"):
            torch.testing.assert_close(actual[field], expected[field])
    gc.collect()
    assert previous_owner() is None
    assert len(calls) == 1
    torch.testing.assert_close(previous.pos, previous_positions)
    state.atomic_numbers[0] = 7
    converter(state, "omol", torch.float32)
    assert len(calls) == 2


@pytest.mark.parametrize("field", ["charge", "spin"])
@pytest.mark.parametrize("mutation", ["none", "atom_extra", "atom_none"])
@pytest.mark.parametrize("clone_state", [False, True])
def test_torchsim_metadata_none_transitions_revalidate(field, mutation, clone_state):
    pytest.importorskip("torch_sim")
    from tests.core.calculate.test_torchsim_resident_metadata import _state

    state = _state()
    if mutation == "none":
        state.system_extras[field] = torch.tensor([4.0])
    else:
        del state.system_extras[field]
    converter = interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    if clone_state:
        state = state.clone()
    if mutation == "none":
        state.system_extras[field] = None
        match = f"{field} must be a tensor with shape"
    else:
        state.atom_extras[field] = (
            None if mutation == "atom_none" else torch.zeros(len(state.positions))
        )
        match = f"{field} must be a per-system extra"
    with pytest.raises(ValueError, match=match):
        interface._simstate_to_atomicdata_batch(state, "omol", torch.float32)
    with pytest.raises(ValueError, match=match):
        converter(state, "omol", torch.float32)


def test_real_uma_ase_evaluation_ignores_foreign_factory_device():
    predictor = _public_predictor()
    oracle = copy.deepcopy(predictor)
    atoms = _atoms()
    reference = atoms.copy()
    reference.calc = FAIRChemCalculator(oracle, "omol")
    expected = reference.get_forces()
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    with torch.device("meta"):
        actual = atoms.get_forces()
        assert torch.empty(0).device.type == "meta"
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_real_uma_torchsim_cloned_nve_steps_match_uncached_conversion(monkeypatch):
    pytest.importorskip("torch_sim")
    from torch_sim.integrators.nve import nve_init, nve_step

    from tests.core.calculate.test_torchsim_interop import _state, _wrapper

    predictor = _public_predictor()
    oracle = copy.deepcopy(predictor)
    cached = _wrapper(monkeypatch, predictor=predictor, task_name="omol")
    uncached = _wrapper(
        monkeypatch, predictor=oracle, task_name="omol", cache_state_metadata=False
    )
    state = _state()
    before = state.clone()
    actual = nve_init(state.clone(), cached, kT=0)
    expected = nve_init(state.clone(), uncached, kT=0)
    original = interface._simstate_to_atomicdata_batch
    rebuilds = []

    def counted(*args, **kwargs):
        rebuilds.append(True)
        return original(*args, **kwargs)

    for _ in range(10):
        with monkeypatch.context() as patch:
            patch.setattr(interface, "_simstate_to_atomicdata_batch", counted)
            actual = nve_step(actual.clone(), cached, dt=0.001)
        expected = nve_step(expected.clone(), uncached, dt=0.001)
    assert not rebuilds
    for name in ("positions", "momenta", "energy", "forces"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))
    torch.testing.assert_close(state.positions, before.positions, rtol=0, atol=0)


@pytest.mark.parametrize("periodic", [False, True])
def test_atomic_data_accepts_owned_cpu_tensor_metadata(periodic):
    class TensorAtoms:
        def __init__(self, atoms):
            self.atoms = atoms

        def __getattr__(self, name):
            return getattr(self.atoms, name)

        def __len__(self):
            return len(self.atoms)

        def copy(self):
            return TensorAtoms(self.atoms.copy())

        @property
        def pbc(self):
            return torch.from_numpy(self.atoms.pbc)

        def get_tags(self):
            return torch.from_numpy(self.atoms.arrays["tags"])

    atoms = _atoms()
    atoms.pbc = periodic
    atoms.set_tags([1, 2, 3])
    wrapped = TensorAtoms(atoms)
    actual = AtomicData.from_ase(wrapped, task_name="omol")
    expected = AtomicData.from_ase(atoms, task_name="omol")
    for field in ("pos", "cell", "atomic_numbers", "pbc", "tags"):
        torch.testing.assert_close(actual[field], expected[field])
    actual.tags.zero_()
    np.testing.assert_array_equal(atoms.get_tags(), [1, 2, 3])
    predictor = _FusedPredictor()
    calc = FAIRChemCalculator(predictor, "omol")
    calc.calculate(wrapped)
    assert np.isfinite(calc.results["energy"])


@pytest.mark.parametrize("source", ["none", "info", "singlepoint"])
def test_atomic_data_ase_targets_are_cpu_under_foreign_factory_context(source):
    atoms = _atoms()
    targets = {"energy": 2.0, "forces": np.ones((3, 3)), "stress": np.eye(3)}
    if source == "info":
        atoms.info.update(targets)
    elif source == "singlepoint":
        atoms.calc = SinglePointCalculator(atoms, **targets)
    expected = AtomicData.from_ase(atoms, task_name="omol")
    with torch.device("meta"):
        actual = AtomicData.from_ase(atoms, task_name="omol")
    for name, value in actual:
        if torch.is_tensor(value):
            assert value.device.type == "cpu"
            torch.testing.assert_close(value, expected[name])


@pytest.mark.parametrize("cell", [[0, 0, 0], [8, 0, 0]])
def test_nonperiodic_external_graph_completes_only_owned_cell(cell):
    pytest.importorskip("pymatgen")
    atoms = _atoms(cell=False)
    atoms.cell = cell
    before = atoms.copy()
    with torch.device("meta"):
        data = AtomicData.from_ase(atoms, r_edges=True, max_neigh=30, task_name="omol")
    assert data.nedges.tolist() == [6]
    assert set(map(tuple, data.edge_index.T.tolist())) == {
        (i, j) for i in range(3) for j in range(3) if i != j
    }
    assert not data.cell_offsets.any()
    assert not data.pbc.any()
    for _, value in data:
        if torch.is_tensor(value):
            assert value.device.type == "cpu"
    np.testing.assert_array_equal(atoms.cell, before.cell)
    np.testing.assert_array_equal(atoms.pbc, before.pbc)
    np.testing.assert_array_equal(atoms.positions, before.positions)
    predictor = _FusedPredictor(nitro=False)
    predictor.inference_settings.external_graph_gen = True
    atoms.calc = FAIRChemCalculator(predictor, "omol")
    np.testing.assert_allclose(atoms.get_forces(), -atoms.positions, atol=1e-7)

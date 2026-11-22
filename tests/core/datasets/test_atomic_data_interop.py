"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import weakref
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixConstraint

from fairchem.core.datasets.atomic_data import (
    AtomicData,
    _clone_tensor_tree,
    atomicdata_list_to_batch,
    tensor_or_int_to_tensor,
)


def _data():
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["charge", "spin"])


@pytest.mark.parametrize("form", ["keyword", "dtype", "reference"])
def test_conversion_preserves_index_types_nested_tuples_and_tensor_aliases(form):
    data = _data()
    data.alias = data.pos
    data.metadata = {"coordinates": (data.pos,), "labels": ["job"]}
    data._requested_properties = ("energy", "forces")
    if form == "keyword":
        result = data.to("cpu", dtype=torch.float64, copy=True)
    elif form == "dtype":
        result = data.to(torch.float64, copy=True)
    else:
        result = data.to(torch.empty(0, dtype=torch.float64), copy=True)
    assert result is data
    assert data.pos.dtype == data.cell.dtype == data.cell_offsets.dtype == torch.float64
    for name in (
        "atomic_numbers",
        "edge_index",
        "natoms",
        "nedges",
        "charge",
        "spin",
        "batch",
        "tags",
        "fixed",
    ):
        assert data[name].dtype == torch.long
    assert data.pbc.dtype == torch.bool
    assert data.alias is data.pos is data.metadata["coordinates"][0]
    assert isinstance(data.metadata["coordinates"], tuple)
    assert data._requested_properties == ("energy", "forces")
    data.validate()


def test_clone_keeps_nonleaf_gradients_and_owns_nested_metadata():
    data = _data()
    leaf = data.pos.clone().requires_grad_()
    data.pos = leaf * 2
    data.alias = data.pos
    data.metadata = {"nested": (data.pos,), "job": ["original"]}
    cloned = data.clone()
    assert cloned.pos is cloned.alias is cloned.metadata["nested"][0]
    assert cloned.pos.data_ptr() != data.pos.data_ptr()
    cloned.metadata["job"][0] = "changed"
    assert data.metadata["job"] == ["original"]
    torch.testing.assert_close(
        torch.autograd.grad(cloned.pos.sum(), leaf)[0], torch.full_like(leaf, 2)
    )


def test_clone_supports_nonleaf_tensors_in_custom_metadata_objects():
    data = _data()
    leaf = data.pos.clone().requires_grad_()
    data.metadata = SimpleNamespace(coordinates=leaf * 2, names=["first"])
    cloned = data.clone()
    cloned.metadata.names.append("second")
    assert data.metadata.names == ["first"]
    assert cloned.metadata.coordinates is not data.metadata.coordinates
    torch.testing.assert_close(
        torch.autograd.grad(cloned.metadata.coordinates.sum(), leaf)[0],
        torch.full_like(leaf, 2),
    )


@pytest.mark.parametrize("operation", ["clone", "metadata_conversion"])
def test_released_clones_do_not_wait_for_cyclic_gc(operation):
    data = _data()
    data.metadata = SimpleNamespace(coordinates=data.pos, names=["first"])
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(4):
            cloned = data.clone()
            if operation == "metadata_conversion":
                cloned.to(dtype=torch.float64, copy=True)
            assert cloned.pos is cloned.metadata.coordinates
            ref = weakref.ref(cloned.pos)
            owner_ref = weakref.ref(cloned)
            del cloned
            assert ref() is None
            assert owner_ref() is None
    finally:
        if enabled:
            gc.enable()


def test_clone_cleanup_preserves_external_memo_aliases_and_cycles():
    tensor = torch.ones(3, requires_grad=True) * 2
    source = [tensor, tensor]
    source.append(source)
    memo = {}
    cloned = _clone_tensor_tree(source, memo=memo)
    assert cloned[0] is cloned[1] is memo[id(tensor)]
    assert cloned[2] is cloned is memo[id(source)]
    assert cloned[0].grad_fn is not None
    assert cloned[0].data_ptr() != tensor.data_ptr()


def test_clone_transform_failure_does_not_retain_earlier_copies_until_gc():
    references = []

    def transform(tensor):
        if references:
            raise RuntimeError("conversion failed")
        result = tensor.clone()
        references.append(weakref.ref(result))
        return result

    enabled = gc.isenabled()
    gc.disable()
    try:
        try:
            _clone_tensor_tree([torch.ones(2), torch.ones(3)], transform)
        except RuntimeError:
            # Do not retain an ExceptionInfo/traceback: that itself would own
            # conversion frames and their memo during this lifetime check.
            pass
        else:
            pytest.fail("the conversion should fail")
        assert len(references) == 1
        assert references[0]() is None
    finally:
        if enabled:
            gc.enable()


def test_conversion_keeps_original_field_ids_alive_until_memo_is_released():
    class WeakList(list):
        pass

    class SidFirst(set):
        def __iter__(self):
            return iter(sorted(super().__iter__(), key=lambda key: key != "sid"))

    data = _data()
    data.sid = WeakList(data.sid)
    original = weakref.ref(data.sid)
    data.__keys__ = SidFirst(data.__keys__)
    converted = []

    def convert(tensor):
        # sid has already been replaced. Its original id must remain reserved
        # while the same conversion memo can still be used by deepcopy below.
        assert original() is not None
        converted.append(True)
        return tensor.clone()

    data.apply(convert)
    assert converted
    assert original() is None  # The temporary roots must not survive apply.
    data.validate()


def test_conversion_preserves_tuple_list_cycles_and_tensor_aliases():
    data = _data()
    items = []
    data.metadata = (items, data.pos)
    items.append(data.metadata)
    data.to(torch.float64)
    assert data.metadata[0][0] is data.metadata
    assert data.metadata[1] is data.pos


def test_namedtuple_metadata_roundtrips_without_losing_wrapper_type():
    options_type = namedtuple("Options", ["energy", "forces"])
    first, second = _data(), _data()
    first.options = options_type(True, False)
    second.options = options_type(False, True)
    batch = atomicdata_list_to_batch([first, second])
    restored = batch.batch_to_atomicdata_list()
    assert type(restored[0].options) is options_type
    assert restored[0].options == first.options
    assert restored[1].options == second.options


def test_batch_roundtrip_preserves_arbitrary_metadata_and_does_not_shift_job_ids():
    structures = [_data(), _data()]
    for i, data in enumerate(structures):
        data.job_index = torch.tensor([17 + i])
        data.scalar = torch.tensor(i, dtype=torch.int32)
        data.labels = [f"item-{i}"] if i else []
        data.options = ("energy", "forces")
        data.enabled = bool(i)
        data.metadata = {"sequence": [i]}
        data.edge_index = torch.tensor([[0, 1], [1, 0]])
        data.cell_offsets = torch.zeros(2, 3)
        data.nedges = torch.tensor([2])
    batch = atomicdata_list_to_batch(structures)
    assert batch.job_index.tolist() == [17, 18]
    assert batch.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    structures[0].metadata["sequence"][0] = 99
    for i in range(2):
        restored = batch[i]
        assert restored.job_index.item() == 17 + i
        assert restored.scalar.ndim == 0
        assert restored.scalar.item() == i
        assert restored.labels == (["item-1"] if i else [])
        assert restored.options == ("energy", "forces")
        assert restored.enabled is bool(i)
        assert restored.metadata["sequence"] == [i]
        torch.testing.assert_close(restored.edge_index, structures[i].edge_index)
        restored.metadata["sequence"][0] = 20
        restored.pos.add_(100)
        assert batch[i].metadata["sequence"] == [i]
        assert batch[i].pos.abs().max() < 10


def test_single_example_is_an_owned_job_input():
    data = _data()
    data.metadata = {"job": [1]}
    job = data[0]
    job.pos.add_(50)
    job.metadata["job"][0] = 2
    assert data.pos.abs().max() < 10
    assert data.metadata["job"] == [1]


def test_missing_fields_are_not_silently_lost_and_exclusions_work():
    first, second = _data(), _data()
    second.energy = torch.tensor([3.0])
    with pytest.raises(ValueError, match="different fields"):
        atomicdata_list_to_batch([first, second])
    assert "energy" not in atomicdata_list_to_batch(
        [first, second], exclude_keys=["energy"]
    )
    with pytest.raises(ValueError, match="required"):
        atomicdata_list_to_batch([first], exclude_keys=["pos"])
    with pytest.raises(ValueError, match="empty"):
        atomicdata_list_to_batch([])


def test_batching_refuses_silent_float_precision_promotion():
    with pytest.raises(ValueError, match="dtype"):
        atomicdata_list_to_batch([_data(), _data().to(dtype=torch.float64)])


@pytest.mark.parametrize("shape", [(1, 3, 3), (1, 9)])
def test_ase_export_includes_stress_and_detaches_autograd(shape):
    data = _data()
    data.pos.requires_grad_()
    data.cell.requires_grad_()
    data.energy = data.pos.square().sum().reshape(1)
    data.forces = -2 * data.pos
    data.stress = torch.eye(3).reshape(shape).requires_grad_()
    atoms = data.to_ase_single()
    assert np.isscalar(atoms.get_potential_energy())
    np.testing.assert_allclose(atoms.get_forces(), data.forces.detach().numpy())
    np.testing.assert_allclose(atoms.get_stress(), [1, 1, 1, 0, 0, 0])
    data.stress.detach().zero_()
    np.testing.assert_allclose(atoms.get_stress(), [1, 1, 1, 0, 0, 0])
    atoms.info["sid"].append("new")
    assert data.sid == [""]


@pytest.mark.parametrize("value", [1.5, True, np.float64(2), float("nan")])
def test_ase_integer_metadata_does_not_silently_truncate(value):
    atoms = molecule("H2O")
    atoms.info["charge"] = value
    with pytest.raises(TypeError, match="integer"):
        AtomicData.from_ase(atoms, r_data_keys=["charge"])


def test_scalar_integer_helper_constructs_values_instead_of_uninitialized_shape():
    for value in (0, -2, np.int64(3)):
        result = tensor_or_int_to_tensor(value, torch.long)
        assert result.shape == (1,)
        assert result.item() == value


def test_numpy_index_selection_and_boolean_mask_validation():
    batch = atomicdata_list_to_batch([_data(), _data()])
    assert len(batch[np.array([1, 0])]) == 2
    assert len(batch[np.array([True, False])]) == 1
    with pytest.raises(IndexError, match="one entry"):
        batch[np.array([True])]
    with pytest.raises(IndexError):
        batch[2]


def test_invalid_positions_and_periodic_cells_fail_before_graph_construction():
    atoms = molecule("H2O")
    atoms.positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        AtomicData.from_ase(atoms)
    atoms = molecule("H2O")
    atoms.pbc = True
    with pytest.raises(ValueError, match="nonsingular"):
        AtomicData.from_ase(atoms)


def test_interleaved_batch_membership_is_rejected_instead_of_mixing_systems():
    data = atomicdata_list_to_batch([_data(), _data()])
    data.batch = torch.tensor([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="contiguously"):
        data.validate()


def test_empty_system_fails_with_explicit_diagnostic():
    data = _data()
    data.natoms.zero_()
    with pytest.raises(ValueError, match="at least one atom"):
        data.validate()


def test_external_graph_wrapping_does_not_reapply_fixed_atom_constraints():
    atoms = molecule("H2")
    atoms.cell = [5, 6, 7]
    atoms.pbc = True
    atoms.positions += [5, 0, 0]
    expected = AtomicData.from_ase(atoms, r_edges=True, radius=2, max_neigh=20)
    atoms.set_constraint(FixAtoms(indices=[0]))
    original_positions = atoms.positions.copy()

    actual = AtomicData.from_ase(atoms, r_edges=True, radius=2, max_neigh=20)

    torch.testing.assert_close(actual.pos, expected.pos)
    torch.testing.assert_close(actual.edge_index, expected.edge_index)
    torch.testing.assert_close(actual.cell_offsets, expected.cell_offsets)
    torch.testing.assert_close(actual.fixed, torch.tensor([1, 0]))
    np.testing.assert_array_equal(atoms.positions, original_positions)


class _UnchangedCellConstraint(FixConstraint):
    def adjust_cell(self, atoms, cell):
        # Cell constraints are actions of an optimizer/MD integrator, not a
        # coordinate-conversion operation. Preserve the caller's original cell.
        cell[:] = atoms.cell


def test_external_graph_cell_completion_bypasses_ase_cell_constraints():
    atoms = molecule("H2")
    expected = AtomicData.from_ase(atoms, r_edges=True, radius=2, max_neigh=20)
    atoms.set_constraint(_UnchangedCellConstraint())

    actual = AtomicData.from_ase(atoms, r_edges=True, radius=2, max_neigh=20)

    torch.testing.assert_close(actual.cell, expected.cell)
    torch.testing.assert_close(actual.edge_index, expected.edge_index)
    torch.testing.assert_close(actual.cell_offsets, expected.cell_offsets)
    assert not atoms.cell.array.any()


@pytest.mark.parametrize("name", ["forces", "stress"])
@pytest.mark.parametrize("source", ["info", "calculator"])
def test_explicitly_disabled_ase_labels_are_not_read(name, source):
    atoms = molecule("H2")
    malformed_label = np.zeros((7, 7))
    if source == "info":
        atoms.info[name] = malformed_label
    else:
        atoms.calc = SinglePointCalculator(atoms, **{name: malformed_label})

    data = AtomicData.from_ase(atoms, **{f"r_{name}": False})

    assert name not in data


def test_ase_isotope_roundtrip_mixed_batch_and_stale_labels():
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.stress import voigt_6_to_full_3x3_stress

    heavy = Atoms("H", masses=[2.014], info={"stress": list(range(6)), "sid": 42})
    native = AtomicData.from_ase(Atoms("H"))
    isotope = AtomicData.from_ase(heavy, target_dtype=native.pos.dtype)
    torch.testing.assert_close(
        isotope.stress[0],
        torch.as_tensor(voigt_6_to_full_3x3_stress(list(range(6)))).to(isotope.pos),
    )
    assert isotope.sid == ["42"]
    batch = atomicdata_list_to_batch([native, isotope], exclude_keys=["stress"])
    assert batch.masses.tolist() == pytest.approx([1.008, 2.014])
    assert [a.get_masses()[0] for a in batch.to_ase()] == pytest.approx([1.008, 2.014])
    assert "atomic_masses" not in native
    heavy.calc = SinglePointCalculator(heavy, energy=3)
    heavy.positions[0, 0] += 1
    with pytest.raises(ValueError, match="Stale"):
        AtomicData.from_ase(heavy)


def test_invalid_graph_replacement_is_atomic():
    from ase import Atoms

    data = AtomicData.from_ase(Atoms("HH"))
    edges = torch.tensor([[0, 1], [1, 0]])
    offsets = data.pos.new_zeros(2, 3)
    data.update_batch_edges(edges, offsets, torch.tensor([2]))
    with pytest.raises(ValueError):
        data.update_batch_edges(edges + 1, offsets, torch.tensor([2]))
    assert data.edge_index is edges
    assert data.cell_offsets is offsets
    assert data.nedges.tolist() == [2]


def test_from_dict_preserves_normalized_sid_and_batch():
    from ase import Atoms

    data = AtomicData.from_ase(Atoms("H")).to_dict()
    rebuilt = AtomicData.from_dict({**data, "sid": 12, "batch": None})
    assert rebuilt.sid == ["12"]
    torch.testing.assert_close(rebuilt.batch, torch.tensor([0]))
    fields = atomicdata_list_to_batch([rebuilt, rebuilt]).to_dict()
    assert AtomicData.from_dict({**fields, "sid": None}).sid == ["", ""]


def test_ase_isotope_bits_and_implicit_masses_survive_model_dtype_changes():
    from ase import Atoms

    natural = Atoms("HO")
    heavy = Atoms("HO", masses=[2.01410177812, 17.99915961286])
    data = atomicdata_list_to_batch([AtomicData.from_ase(a) for a in (natural, heavy)])
    for dtype in (torch.float64, torch.float32, torch.float64):
        data.to(dtype=dtype)
        expected = np.concatenate((natural.get_masses(), heavy.get_masses()))
        torch.testing.assert_close(
            data.masses, torch.tensor(expected, dtype=dtype), rtol=0, atol=0
        )
        for restored, original in zip(data.to_ase(), (natural, heavy), strict=True):
            assert restored.has("masses") == original.has("masses")
            np.testing.assert_array_equal(restored.get_masses(), original.get_masses())


def test_singleton_batch_roundtrip_preserves_scalar_and_object_fields():
    source = _data()
    source.extra = torch.tensor(1.5)
    source.info = {"labels": ["old"]}
    source.label = "one"
    for _ in range(2):
        batch = atomicdata_list_to_batch([source])
        result = batch.get_example(0)
        assert result.extra.shape == ()
        assert result.label == "one"
        assert result.info == {"labels": ["old"]}
        result.info["labels"].append("new")
        assert batch.info == [{"labels": ["old"]}]
        result.info["labels"].pop()
        source = result


def test_failed_field_conversion_keeps_original_tensors():
    data = _data()
    original = {key: value for key, value in data if torch.is_tensor(value)}
    calls = 0

    def fail(tensor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("conversion failed")
        return tensor.clone()

    with pytest.raises(RuntimeError, match="conversion failed"):
        data.apply(fail)
    assert all(data[key] is value for key, value in original.items())

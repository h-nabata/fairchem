"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import (
    AtomicData,
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

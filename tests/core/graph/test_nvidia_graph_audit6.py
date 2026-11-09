"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import inspect
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph import radius_graph_pbc_nvidia as nvidia
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2


def _data(positions, cell=None, pbc=False, dtype=torch.float64):
    pos = torch.tensor(positions, dtype=dtype).reshape(-1, 3)
    return SimpleNamespace(
        pos=pos,
        cell=torch.as_tensor(np.eye(3) * 10 if cell is None else cell, dtype=dtype)[
            None
        ],
        pbc=torch.full((1, 3), pbc, dtype=torch.bool),
        natoms=torch.tensor([len(pos)]),
        batch=torch.zeros(len(pos), dtype=torch.long),
    )


@pytest.mark.parametrize("dtype,edge_count", [(torch.float32, 2), (torch.float64, 0)])
def test_ase_external_graph_uses_requested_coordinate_precision(dtype, edge_count):
    if not nvidia.nvalchemiops_installed():
        pytest.skip("nvalchemiops is not installed")
    atoms = Atoms("H2", positions=[[0, 0, 0], [1 + 2**-25, 0, 0]])
    original = atoms.positions.copy()
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=1.0,
        max_neigh=10,
        target_dtype=dtype,
        external_graph_method="nvidia",
    )
    assert data.pos.dtype == dtype
    assert data.edge_index.shape[1] == edge_count
    assert data.nedges.item() == edge_count
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ase_native_neighbor_distances_retain_coordinate_dtype(dtype):
    if not nvidia.nvalchemiops_installed():
        pytest.skip("nvalchemiops is not installed")
    atoms = Atoms("H2", positions=[[0, 0, 0], [1 + 2**-25, 0, 0]])
    _, _, distances, _ = nvidia.get_neighbors_nvidia_atoms(
        atoms, 1.1, 10, target_dtype=dtype
    )
    expected = torch.tensor(1 + 2**-25, dtype=dtype).numpy()
    assert distances.dtype == expected.dtype
    np.testing.assert_array_equal(distances, np.full(2, expected, dtype=expected.dtype))


def _tuples(edge_index, offsets):
    return sorted(
        (int(source), int(target), *map(int, offset))
        for (source, target), offset in zip(edge_index.T, offsets)
    )


def _search(data, cap=0, **kwargs):
    # Exercise the Python adapter even when the optional native extension is
    # absent. The test double models its output buffers, not the adapter logic.
    return inspect.unwrap(nvidia.get_neighbors_nvidia)(
        data.pos,
        data.cell,
        data.pbc,
        max_neigh=cap,
        batch=data.batch,
        natoms=data.natoms,
        **kwargs,
    )


@pytest.fixture()
def native_double(monkeypatch):
    calls = []

    def fill(**kwargs):
        calls.append(kwargs["neighbor_matrix"].shape[1])
        pos = kwargs["positions"].numpy()
        cell = kwargs["cell"]
        batch = kwargs["batch_idx"].numpy()
        for system in np.unique(batch):
            ids = np.flatnonzero(batch == system)
            lattice = None if cell is None else cell[system].numpy()
            if lattice is not None:
                assert np.linalg.det(lattice) > 0
            atoms = Atoms(
                numbers=np.ones(len(ids), dtype=int),
                positions=pos[ids],
                cell=lattice,
                pbc=False if cell is None else kwargs["pbc"][system].numpy(),
            )
            target, source, shifts = neighbor_list("ijS", atoms, kwargs["cutoff"])
            for local_target, global_target in enumerate(ids):
                rows = np.flatnonzero(target == local_target)
                # Reverse the native traversal so truncation cannot accidentally
                # retain the closest neighbors in the capacity regression.
                rows = rows[::-1].copy()
                kwargs["num_neighbors"][global_target] = len(rows)
                rows = rows[: calls[-1]]
                kwargs["neighbor_matrix"][global_target, : len(rows)] = torch.tensor(
                    ids[source[rows]], dtype=torch.int32
                )
                kwargs["neighbor_matrix_shifts"][global_target, : len(rows)] = (
                    torch.tensor(shifts[rows], dtype=torch.int32)
                )

    monkeypatch.setattr(nvidia, "neighbor_list", fill)
    monkeypatch.setattr(nvidia, "estimate_max_neighbors", lambda **_: 2)
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cap", [0, 2])
def test_buffer_growth_preserves_complete_nearest_neighbors(native_double, dtype, cap):
    data = _data([[0, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [0.83, 0, 0]], dtype=dtype)
    target, source, offsets, distances = _search(
        data, cap, cutoff=1.0, enforce_max_neighbors_strictly=True
    )
    expected = radius_graph_pbc_v2(data, 1.0, cap, True)
    assert native_double == [2, 3]
    assert _tuples(torch.stack((source, target)), offsets) == _tuples(*expected[:2])
    assert distances is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cutoff_expansion_is_removed_before_returning_edges(native_double, dtype):
    data = _data([[0, 0, 0], [1, 0, 0], [1 + 5e-7, 0, 0]], dtype=dtype)
    target, source, offsets, distances = _search(
        data, cutoff=1.0, return_distances_sq=True
    )
    expected = radius_graph_pbc_v2(data, 1.0, 0)
    assert _tuples(torch.stack((source, target)), offsets) == _tuples(*expected[:2])
    assert bool((distances <= 1.0).all())
    assert bool((distances > 0).all())
    assert distances.dtype == dtype


def test_periodic_distance_keeps_float64_cell_precision(native_double):
    data = _data(
        [[0, 0, 0], [10, 0, 0]],
        cell=np.diag([10 + 5e-8, 10, 10]),
        pbc=True,
    )
    target, source, offsets, distances = _search(
        data, cutoff=1e-6, return_distances_sq=True
    )
    expected = radius_graph_pbc_v2(data, 1e-6, 0, method="all_pairs")
    assert _tuples(torch.stack((source, target)), offsets) == _tuples(*expected[:2])
    torch.testing.assert_close(distances, data.pos.new_full((2,), (5e-8) ** 2))


@pytest.mark.parametrize("real_backend", [False, True])
def test_left_handed_mixed_batch_preserves_offsets_and_input(real_backend, request):
    if real_backend:
        if not nvidia.nvalchemiops_installed():
            pytest.skip("nvalchemiops is not installed")
    else:
        request.getfixturevalue("native_double")
    data = _data([[0.2, 0, 0], [4.8, 0, 0], [0.3, 0, 0], [4.7, 0, 0]])
    data.cell = torch.diag(torch.tensor([5.0, 5.0, 5.0], dtype=data.pos.dtype)).repeat(
        2, 1, 1
    )
    data.cell[0, 0, 0] = -5
    data.pbc = torch.ones((2, 3), dtype=torch.bool)
    data.natoms = torch.tensor([2, 2])
    data.batch = torch.tensor([0, 0, 1, 1])
    # Use wrapped coordinates, as required by the portable comparison backend.
    data.pos[:2, 0] -= 5
    original = {key: value.clone() for key, value in vars(data).items()}
    target, source, offsets, distances = _search(
        data, cutoff=0.7, return_distances_sq=True
    )
    expected = radius_graph_pbc_v2(data, 0.7, 0)
    assert _tuples(torch.stack((source, target)), offsets) == _tuples(*expected[:2])
    assert len(offsets) == 4
    torch.testing.assert_close(
        distances.sort().values, data.pos.new_tensor([0.16] * 2 + [0.36] * 2)
    )
    for name, value in original.items():
        torch.testing.assert_close(getattr(data, name), value, atol=0, rtol=0)


def test_empty_graph_returns_without_native_launch(native_double):
    data = _data([])
    target, source, offsets, distances = _search(
        data, cutoff=1.0, return_distances_sq=True
    )
    assert target.shape == source.shape == distances.shape == (0,)
    assert offsets.shape == (0, 3)
    assert native_double == []


def test_invalid_periodic_cell_is_rejected_before_native_launch(native_double):
    data = _data([[0, 0, 0]], cell=np.zeros((3, 3)), pbc=True)
    with pytest.raises(ValueError, match="nonsingular"):
        _search(data, cutoff=1.0)
    assert native_double == []


def test_shared_cell_and_pbc_support_inferred_batch(native_double):
    data = _data([[0.1, 0, 0], [9.9, 0, 0], [0.2, 0, 0], [9.8, 0, 0]])
    data.pbc[:] = True
    data.natoms = torch.tensor([2, 2])
    target, source, offsets, _ = inspect.unwrap(nvidia.get_neighbors_nvidia)(
        data.pos,
        data.cell,
        data.pbc[0],
        cutoff=0.5,
        max_neigh=0,
        natoms=data.natoms,
    )
    assert len(target) == 4
    assert torch.equal(target // 2, source // 2)
    assert torch.equal(offsets[:, 0].abs(), torch.ones(4, dtype=offsets.dtype))
    assert data.cell.shape == (1, 3, 3)


@pytest.mark.parametrize("kind", ["negative", "growing"])
def test_invalid_native_counts_fail_without_returning_partial_graph(monkeypatch, kind):
    data = _data([[0, 0, 0]])
    capacities = []

    def fill(**kwargs):
        capacity = kwargs["neighbor_matrix"].shape[1]
        capacities.append(capacity)
        kwargs["num_neighbors"].fill_(-1 if kind == "negative" else capacity + 1)

    monkeypatch.setattr(nvidia, "neighbor_list", fill)
    monkeypatch.setattr(nvidia, "estimate_max_neighbors", lambda **_: 2)
    with pytest.raises(RuntimeError, match="overflowed|three attempts"):
        _search(data, cutoff=1.0)
    assert capacities == ([2] if kind == "negative" else [2, 3, 4])


def test_dense_native_buffers_are_released_before_edge_geometry(
    native_double, monkeypatch
):
    from fairchem.core.graph import compute

    native = nvidia.neighbor_list
    geometry = compute.edge_cell_offsets
    buffers = []

    def record_buffers(**kwargs):
        buffers.extend(
            weakref.ref(kwargs[name])
            for name in ("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors")
        )
        return native(**kwargs)

    def check_lifetime(*args):
        assert buffers
        assert all(reference() is None for reference in buffers)
        return geometry(*args)

    monkeypatch.setattr(nvidia, "neighbor_list", record_buffers)
    monkeypatch.setattr(compute, "edge_cell_offsets", check_lifetime)
    data = _data([[0, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [0.83, 0, 0]])
    assert len(_search(data, cutoff=1.0)[0]) == 12


@pytest.mark.gpu()
def test_cuda_capping_accepts_host_structure_counts():
    if not nvidia.nvalchemiops_installed():
        pytest.skip("nvalchemiops is not installed")
    data = _data(
        [[0, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [0.83, 0, 0]], dtype=torch.float32
    )
    expected = radius_graph_pbc_v2(data, 1.0, 1, True)
    for name in ("pos", "cell", "pbc", "batch"):
        setattr(data, name, getattr(data, name).cuda())
    target, source, offsets, _ = _search(
        data, cap=1, cutoff=1.0, enforce_max_neighbors_strictly=True
    )
    assert _tuples(torch.stack((source, target)), offsets) == _tuples(*expected[:2])
    assert data.natoms.device.type == "cpu"

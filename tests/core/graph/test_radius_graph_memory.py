"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch.utils._python_dispatch import TorchDispatchMode

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph.radius_graph_pbc import (
    get_max_neighbors_mask,
    radius_graph_pbc_v2,
)


def _data(atoms):
    return atomicdata_list_to_batch(
        [AtomicData.from_ase(system, task_name="omol") for system in atoms]
    )


def _edges_as_tuples(edges, cells):
    return sorted(
        (source, target, *shift)
        for (source, target), shift in zip(edges.T.tolist(), cells.long().tolist())
    )


def _ase_edges(systems, cutoff, partition=None):
    result = []
    offset = 0
    for atoms in systems:
        targets, sources, shifts = neighbor_list("ijS", atoms, cutoff)
        result.extend(
            (int(source + offset), int(target + offset), *shift.tolist())
            for target, source, shift in zip(targets, sources, shifts)
            if partition is None or int(target + offset) in partition
        )
        offset += len(atoms)
    return sorted(result)


@pytest.mark.parametrize("pbc", [False, True, [True, False, True]])
@pytest.mark.parametrize("partitioned", [False, True])
def test_sparse_grid_matches_ase_for_skew_and_mixed_batches(pbc, partitioned):
    random = np.random.default_rng(17)
    cell = np.array([[3.1, 0.2, 0.3], [0.1, 3.5, 0.4], [0.3, 0.2, 4.1]])
    systems = [
        Atoms("H7", positions=random.random((7, 3)) @ cell, cell=cell, pbc=pbc),
        Atoms(
            "H3",
            positions=random.random((3, 3)) + [14, -13, 12],
            cell=[40, 40, 40],
            pbc=False,
        ),
    ]
    data = _data(systems)
    partition = {1, 4, 7, 9} if partitioned else None
    if partitioned:
        data["node_partition"] = torch.tensor(sorted(partition))
    edges, cells, neighbors = radius_graph_pbc_v2(data, 2.2, 0)
    assert _edges_as_tuples(edges, cells) == _ase_edges(systems, 2.2, partition)
    assert neighbors.tolist() == [int((edges[1] < 7).sum()), int((edges[1] >= 7).sum())]


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("translation", [20.0, 1000.0])
def test_sparse_grid_with_translated_unequal_nonperiodic_systems(sign, translation):
    systems = [
        Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[2, 2, 2]),
        Atoms(
            "H4",
            positions=(
                np.array([[0, 0, 0], [0.2, 0.1, 0], [5, 5, 5], [5.1, 5, 5]])
                + translation
            )
            * sign,
            cell=[60, 60, 60],
        ),
    ]
    edges, cells, neighbors = radius_graph_pbc_v2(_data(systems), 1.0, 0)
    assert _edges_as_tuples(edges, cells) == _ase_edges(systems, 1.0)
    assert neighbors.tolist() == [2, 4]


def test_empty_partition_returns_empty_graph_for_every_system():
    data = _data(
        [
            Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True),
            Atoms("H2", positions=[[0, 0, 0], [0.5, 0, 0]], cell=[2, 2, 2]),
        ]
    )
    data["node_partition"] = torch.empty(0, dtype=torch.long)
    edges, cells, neighbors = radius_graph_pbc_v2(data, 3.0, 1)
    assert edges.shape == (2, 0)
    assert cells.shape == (0, 3)
    assert edges.dtype == torch.long
    assert cells.dtype == data.cell.dtype
    assert neighbors.tolist() == [0, 0]


class _LargestTensor(TorchDispatchMode):
    """Check temporary tensor sizes without relying on allocator/RSS noise."""

    def __init__(self):
        super().__init__()
        self.largest_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        leaves, _ = torch.utils._pytree.tree_flatten(result)
        self.largest_numel = max(
            [self.largest_numel]
            + [x.numel() for x in leaves if isinstance(x, torch.Tensor)]
        )
        return result


def test_sparse_grid_does_not_pad_vacuum_to_densest_bucket():
    generator = torch.Generator().manual_seed(7)
    positions = torch.cat(
        [
            torch.rand((64, 3), generator=generator) * 0.2,
            torch.tensor([[30.0, 30.0, 30.0]]),
        ]
    )
    atoms = Atoms("H65", positions=positions.numpy(), cell=[60, 60, 60])
    data = _data([atoms])
    with _LargestTensor() as sizes:
        edges, cells, neighbors = radius_graph_pbc_v2(data, 3.0, 100)
    assert _edges_as_tuples(edges, cells) == _ase_edges([atoms], 3.0)
    assert neighbors.tolist() == [64 * 63]
    # All actual candidates fit well below this limit. The old representation
    # allocates 65 * 125 * 64 = 520,000 int64 entries for padded candidates.
    assert sizes.largest_numel < 100_000


@pytest.mark.parametrize("sparse", [False, True])
def test_bounded_direct_and_sparse_cell_lookups_match_ase(monkeypatch, sparse):
    positions = [[0, 0, 0], [0.2, 0.1, 0], [0.4, 0, 0.1]]
    if sparse:
        positions.append([30, 30, 30])
    atoms = Atoms(f"H{len(positions)}", positions=positions, cell=[60, 60, 60])
    searchsorted = torch.searchsorted
    searches = []

    def record_search(*args, **kwargs):
        searches.append(True)
        return searchsorted(*args, **kwargs)

    monkeypatch.setattr(torch, "searchsorted", record_search)
    edges, cells, _ = radius_graph_pbc_v2(_data([atoms]), 3.0, 0)
    assert bool(searches) is sparse
    assert _edges_as_tuples(edges, cells) == _ase_edges([atoms], 3.0)


@pytest.mark.parametrize("strict", [False, True])
def test_unsorted_partition_matches_filtered_full_graph_without_mutation(strict):
    random = np.random.default_rng(29)
    systems = [
        Atoms("H5", positions=random.random((5, 3)), cell=[2, 2, 2], pbc=True),
        Atoms("H4", positions=random.random((4, 3)), cell=[3, 3, 3]),
    ]
    data = _data(systems)
    full_edges, full_cells, _ = radius_graph_pbc_v2(data, 2.1, 2, strict)
    partition = torch.tensor([8, 1, 6, 0])
    original_partition = partition.clone()
    data["node_partition"] = partition
    edges, cells, neighbors = radius_graph_pbc_v2(data, 2.1, 2, strict)
    selected = torch.isin(full_edges[1], partition)
    torch.testing.assert_close(edges, full_edges[:, selected], rtol=0, atol=0)
    torch.testing.assert_close(cells, full_cells[selected], rtol=0, atol=0)
    torch.testing.assert_close(data.node_partition, original_partition)
    assert neighbors.tolist() == [int((edges[1] < 5).sum()), int((edges[1] >= 5).sum())]


@pytest.mark.parametrize("cap", [0, 1, 3, 10])
@pytest.mark.parametrize("strict", [False, True])
def test_only_overfull_rows_are_capped_against_scalar_reference(cap, strict):
    degrees = [0, 1, 3, 8, 2, 6]
    index = torch.repeat_interleave(torch.arange(6), torch.tensor(degrees))
    # Distinct distances make strict selection unambiguous; zero is included.
    distances = ((torch.arange(len(index)) * 17) % 101).float() / 100
    expected = torch.zeros(len(index), dtype=torch.bool)
    start = 0
    per_atom = []
    for degree in degrees:
        positions = list(range(start, start + degree))
        ordered = sorted(positions, key=lambda i: float(distances[i]))
        if cap <= 0 or degree <= cap:
            selected = positions
        elif strict:
            selected = ordered[:cap]
        else:
            # Preserve the established non-strict cutoff at zero-based `cap`,
            # including all neighbors within the extra degeneracy tolerance.
            cutoff = float(distances[ordered[cap]]) + 0.01
            selected = [i for i in positions if float(distances[i]) <= cutoff]
        expected[selected] = True
        per_atom.append(len(selected))
        start += degree
    mask, counts = get_max_neighbors_mask(
        torch.tensor([3, 3]), index, distances, cap, enforce_max_strictly=strict
    )
    torch.testing.assert_close(mask, expected)
    assert counts.tolist() == [sum(per_atom[:3]), sum(per_atom[3:])]


@pytest.mark.parametrize("strict", [False, True])
def test_overfull_degenerate_distances_keep_existing_cutoff_rule(strict):
    degrees = torch.tensor([1, 2, 9])
    index = torch.repeat_interleave(torch.arange(3), degrees)
    mask, counts = get_max_neighbors_mask(
        torch.tensor([3]),
        index,
        torch.ones(len(index)),
        2,
        enforce_max_strictly=strict,
    )
    assert mask[:3].all()
    assert int(mask[3:].sum()) == (2 if strict else 9)
    assert counts.tolist() == [5 if strict else 12]


@pytest.mark.parametrize("strict", [False, True])
def test_underfull_rows_preserve_nonfinite_filtering_when_capping(strict):
    index = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3, 3])
    distances = torch.tensor(
        [0.0, 1.0, 3.0, 5.0, torch.nan, torch.nan, torch.inf, -torch.inf, 1.0]
    )
    mask, counts = get_max_neighbors_mask(
        torch.tensor([4]), index, distances, 2, enforce_max_strictly=strict
    )
    assert mask.tolist() == [
        True,
        True,
        not strict,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    # Preserve legacy counts as well as masks: this memory optimization does
    # not introduce a new validation policy for nonfinite helper inputs.
    assert counts.tolist() == [7 if strict else 6]


def test_neighbor_cap_does_not_pad_low_degree_atoms_to_largest_degree():
    index = torch.cat([torch.arange(1999), torch.full((256,), 1999)])
    distances = torch.linspace(0.1, 1.0, len(index))
    with _LargestTensor() as sizes:
        mask, counts = get_max_neighbors_mask(
            torch.tensor([2000]), index, distances, 16
        )
    assert mask[:1999].all()
    assert counts.tolist() == [int(mask.sum())]
    # The old distance/index sorting matrices have 2000 * 256 entries each.
    # Only the last atom needs sorting; all other temporary sizes stay linear.
    assert sizes.largest_numel < 20_000

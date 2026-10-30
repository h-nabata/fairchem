"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph.compute import (
    edge_cell_offsets,
    generate_graph,
    spatially_order_graph,
)
from fairchem.core.graph.neighbor_cache import VerletNeighborCache
from fairchem.core.graph.portable_search import integer_cartesian_table, source_cloud
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2


def _data(atoms):
    return atomicdata_list_to_batch(
        [AtomicData.from_ase(a, task_name="omol") for a in atoms]
    )


def _tuples(result):
    edges, offsets = result[:2]
    return sorted(
        (int(a), int(b), *map(int, shift))
        for (a, b), shift in zip(edges.T.tolist(), offsets.tolist())
    )


@pytest.mark.parametrize("method", ["cell", "all_pairs", "bvh", "auto"])
@pytest.mark.parametrize("pbc", [False, True, [True, False, True]])
def test_search_methods_match_independent_ase(method, pbc):
    rng = np.random.default_rng(41)
    cell = np.array([[2.2, 0.2, 0.3], [0.4, 2.6, 0.1], [0.3, 0.2, 2.1]])
    systems = [
        Atoms("H11", positions=rng.random((11, 3)) @ cell, cell=cell, pbc=pbc),
        Atoms("H7", positions=rng.random((7, 3)) * 3 + 100, cell=[200] * 3),
    ]
    data = _data(systems)
    result = radius_graph_pbc_v2(data, 2.8, 0, method=method)
    expected = []
    offset = 0
    for atoms in systems:
        targets, sources, shifts = neighbor_list("ijS", atoms, 2.8)
        expected += [
            (int(s + offset), int(t + offset), *map(int, shift))
            for t, s, shift in zip(targets, sources, shifts)
        ]
        offset += len(atoms)
    assert _tuples(result) == sorted(expected)
    assert result[2].tolist() == [
        int((result[0][1] < 11).sum()),
        int((result[0][1] >= 11).sum()),
    ]


@pytest.mark.parametrize("method", ["all_pairs", "bvh", "auto"])
@pytest.mark.parametrize("strict", [False, True])
def test_alternative_search_preserves_neighbor_cap(method, strict):
    rng = np.random.default_rng(23)
    data = _data(
        [Atoms("H24", positions=rng.random((24, 3)) * 2, cell=[3] * 3, pbc=True)]
    )
    expected = radius_graph_pbc_v2(data, 1.7, 4, strict)
    actual = radius_graph_pbc_v2(data, 1.7, 4, strict, method=method)
    assert _tuples(actual) == _tuples(expected)
    torch.testing.assert_close(actual[2], expected[2])


@pytest.mark.parametrize("method", ["cell", "all_pairs", "bvh"])
def test_cache_tracks_cutoff_and_new_neighbors_without_capping_outer_list(method):
    data = _data(
        [Atoms("H3", positions=[[0, 0, 0], [0.9, 0, 0], [1.1, 0, 0]], cell=[4] * 3)]
    )
    cache = VerletNeighborCache(0.5)
    first = cache.get_graph(data, 1.0, 1, True, method=method)
    assert (1, 0, 0, 0, 0) in _tuples(first)
    data.pos[1, 0] = 1.01
    data.pos[2, 0] = 0.99
    second = cache.get_graph(data, 1.0, 1, True, method=method)
    reference = radius_graph_pbc_v2(data, 1.0, 1, True)
    assert _tuples(second) == _tuples(reference)
    assert (2, 0, 0, 0, 0) in _tuples(second)
    assert cache.stats["rebuilds"] == 1
    assert cache.stats["hits"] == 1


def test_cache_includes_initially_coincident_distinct_atoms():
    data = _data([Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]], cell=[4] * 3)])
    cache = VerletNeighborCache(0.5)
    assert cache.get_graph(data, 1.0, 0)[0].shape[1] == 0
    data.pos[1, 0] = 0.1
    assert cache.get_graph(data, 1.0, 0)[0].shape[1] == 2
    assert cache.stats["rebuilds"] == 1


@pytest.mark.parametrize(
    "mutation", ["cell", "pbc", "atomic_numbers", "partition", "large_move", "dtype"]
)
def test_cache_invalidates_owned_metadata_and_displacement(mutation):
    data = _data(
        [
            Atoms(
                "H3",
                positions=[[0, 0, 0], [0.6, 0, 0], [1.4, 0.2, 0]],
                cell=[3] * 3,
                pbc=True,
            )
        ]
    )
    cache = VerletNeighborCache(0.4)
    cache.get_graph(data, 1.1, 0)
    if mutation == "cell":
        data.cell[0, 0, 0] += 0.01
    elif mutation == "pbc":
        data.pbc[0, 1] = False
    elif mutation == "atomic_numbers":
        data.atomic_numbers[0] = 8
    elif mutation == "partition":
        data["node_partition"] = torch.tensor([2, 0])
    elif mutation == "large_move":
        data.pos[0, 0] += 0.3
    else:
        data.pos = data.pos.double()
        data.cell = data.cell.double()
    assert _tuples(cache.get_graph(data, 1.1, 0)) == _tuples(
        radius_graph_pbc_v2(data, 1.1, 0)
    )
    assert cache.stats["rebuilds"] == 2


def test_cache_preserves_strict_degenerate_selection():
    data = _data(
        [
            Atoms(
                "H5",
                positions=[[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]],
                cell=[5] * 3,
            )
        ]
    )
    cache = VerletNeighborCache(0.5)
    for _ in range(2):
        actual = cache.get_graph(data, 1.1, 2, True)
        expected = radius_graph_pbc_v2(data, 1.1, 2, True)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert cache.stats["tie_fallbacks"] == 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cache_uses_reference_at_cutoff_boundary(dtype):
    data = _data([Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[4] * 3)])
    data.pos = data.pos.to(dtype)
    data.cell = data.cell.to(dtype)
    cache = VerletNeighborCache(0.5)
    for delta in [0.0, 1.0, -1.0]:
        data.pos[1, 0] = (
            1.0
            if delta == 0
            else torch.nextafter(
                torch.tensor(1.0, dtype=dtype), torch.tensor(1.0 + delta, dtype=dtype)
            )
        )
        actual = cache.get_graph(data, 1.0, 0)
        expected = radius_graph_pbc_v2(data, 1.0, 0)
        assert _tuples(actual) == _tuples(expected)
    assert cache.stats["rebuilds"] == 1
    assert cache.stats["boundary_fallbacks"] == 3


@pytest.mark.parametrize("strict", [False, True])
def test_cache_guards_nearly_degenerate_neighbor_cap_thresholds(strict):
    positions = (
        [[0, 0, 0], [0.3, 0, 0], [1.0, 0, 0], [1.0000001, 0, 0]]
        if strict
        else [[0, 0, 0], [0.5, 0, 0], [1.0, 0, 0], [np.sqrt(1.01), 0, 0]]
    )
    data = _data([Atoms("H4", positions=positions, cell=[4] * 3)])
    cap = 2 if strict else 1
    cache = VerletNeighborCache(0.5)
    for _ in range(2):
        actual = cache.get_graph(data, 2.0, cap, strict)
        expected = radius_graph_pbc_v2(data, 2.0, cap, strict)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert cache.stats["cap_fallbacks"] == 2


def test_bounded_cache_reuses_multiple_structures_and_deepcopy_is_empty():
    first = _data([Atoms("H2", positions=[[0, 0, 0], [0.6, 0, 0]], cell=[3] * 3)])
    second = first.clone()
    second.pos += 10
    cache = VerletNeighborCache(0.4, max_entries=2)
    cache.get_graph(first, 1.0, 0)
    cache.get_graph(second, 1.0, 0)
    cache.get_graph(first, 1.0, 0)
    assert cache.stats["rebuilds"] == 2
    assert cache.stats["hits"] == 1
    third = first.clone()
    third.pos += 20
    cache.get_graph(third, 1.0, 0)
    assert cache.stats["entries"] == 2
    assert copy.deepcopy(cache).stats["entries"] == 0
    cache.clear()
    assert cache.stats["entries"] == 0


@pytest.mark.parametrize("operation", ["copy", "deepcopy", "pickle"])
def test_cache_serialization_drops_device_events_and_owned_snapshots(operation):
    class UnpickleableEvent:
        def __reduce__(self):
            raise AssertionError("Derived device events must never be serialized")

    data = _data([Atoms("H2", positions=[[0, 0, 0], [0.6, 0, 0]], cell=[3] * 3)])
    cache = VerletNeighborCache(0.4, max_entries=2)
    cache.get_graph(data, 1.0, 0)
    cache._entry["event"] = UnpickleableEvent()
    restored = (
        pickle.loads(pickle.dumps(cache))
        if operation == "pickle"
        else getattr(copy, operation)(cache)
    )
    assert restored.skin == cache.skin
    assert restored.max_entries == cache.max_entries
    assert restored._entry is None
    assert restored._lock is not cache._lock
    assert all(value == 0 for value in restored.stats.values())
    assert cache.stats["entries"] == 1
    actual = restored.get_graph(data, 1.0, 0)
    expected = radius_graph_pbc_v2(data, 1.0, 0)
    assert _tuples(actual) == _tuples(expected)
    assert restored.stats["rebuilds"] == 1


@pytest.mark.parametrize("systems", [1, 3])
def test_cell_offsets_match_reference_and_higher_derivatives(systems):
    torch.manual_seed(10)
    counts = torch.tensor([5] * systems)
    cell = torch.randn(systems, 3, 3, dtype=torch.double, requires_grad=True)
    offsets = torch.randn(5 * systems, 3, dtype=torch.double, requires_grad=True)

    def function(c, o):
        return edge_cell_offsets(c, o, counts)

    reference = offsets[:, None].bmm(cell.repeat_interleave(counts, 0)).squeeze(1)
    torch.testing.assert_close(function(cell, offsets), reference)
    assert torch.autograd.gradcheck(function, (cell, offsets))
    assert torch.autograd.gradgradcheck(function, (cell, offsets))


def test_cached_graph_keeps_current_geometry_and_second_derivatives():
    data = _data(
        [
            Atoms(
                "H3",
                positions=[[0.2, 0.2, 0.1], [0.8, 0.1, 0.2], [1.7, 0.2, 0.1]],
                cell=[2.3] * 3,
                pbc=True,
            )
        ]
    )
    data.pos = data.pos.double().requires_grad_()
    data.cell = data.cell.double().requires_grad_()
    cache = VerletNeighborCache(0.5)

    def evaluate(cached):
        result = generate_graph(
            data, 1.5, 0, False, 2, data.pbc, neighbor_cache=cache if cached else None
        )
        energy = result["edge_distance"].square().exp().sum()
        first = torch.autograd.grad(energy, (data.pos, data.cell), create_graph=True)
        second = torch.autograd.grad(
            sum(v.square().sum() for v in first), (data.pos, data.cell)
        )
        return energy, first, second

    actual, expected = evaluate(True), evaluate(False)
    torch.testing.assert_close(actual, expected)
    assert not cache._entry["pos"].requires_grad
    assert cache._entry["pos"].data_ptr() != data.pos.data_ptr()


def test_boundary_pruning_omits_unneeded_images_and_integer_tables_stay_immutable():
    pos = torch.tensor([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0]])
    table = integer_cartesian_table(1, 1, 1, pos.device)
    saved = table.clone()
    offset = table.float() * 10
    all_sources = source_cloud(
        pos,
        pos,
        torch.zeros(2, dtype=torch.long),
        offset,
        table,
        torch.tensor([27]),
        torch.tensor([2]),
    )
    pruned = source_cloud(
        pos,
        pos,
        torch.zeros(2, dtype=torch.long),
        offset,
        table,
        torch.tensor([27]),
        torch.tensor([2]),
        prune=1.0,
    )
    assert len(all_sources[0]) == 54
    assert len(pruned[0]) == 2
    torch.testing.assert_close(table, saved)
    assert integer_cartesian_table(1, 1, 1, pos.device).data_ptr() != table.data_ptr()


def test_spatial_edge_order_preserves_geometry_system_groups_and_gradients():
    atoms = [
        Atoms(
            "H4",
            positions=[[1, 0, 0], [0, 0, 0], [0.5, 0.2, 0], [0.2, 0.1, 0.3]],
            cell=[3] * 3,
        ),
        Atoms("H2", positions=[[5, 0, 0], [5.5, 0, 0]], cell=[10] * 3),
    ]
    data = _data(atoms)
    data.pos.requires_grad_()
    graph = generate_graph(data, 1.2, 0, False, 2, data.pbc)
    ordered = spatially_order_graph(data, graph)
    assert _tuples((graph["edge_index"], graph["cell_offsets"])) == _tuples(
        (ordered["edge_index"], ordered["cell_offsets"])
    )
    torch.testing.assert_close(ordered["neighbors"], graph["neighbors"])
    assert torch.all(
        data.batch[ordered["edge_index"][1]][1:]
        >= data.batch[ordered["edge_index"][1]][:-1]
    )
    original_grad = torch.autograd.grad(
        graph["edge_distance"].square().sum(), data.pos, retain_graph=True
    )[0]
    ordered_grad = torch.autograd.grad(
        ordered["edge_distance"].square().sum(), data.pos
    )[0]
    torch.testing.assert_close(original_grad, ordered_grad)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires a real Intel XPU")
def test_integer_and_neighbor_cache_cross_xpu_streams():
    device = torch.device("xpu:0")
    data = _data(
        [Atoms("H3", positions=[[0, 0, 0], [0.6, 0, 0], [1.4, 0.2, 0]], cell=[3] * 3)]
    ).to(device)
    first, second = torch.xpu.Stream(device=device), torch.xpu.Stream(device=device)
    first.wait_stream(torch.xpu.current_stream(device))
    second.wait_stream(torch.xpu.current_stream(device))
    cache = VerletNeighborCache(0.4)
    with torch.xpu.stream(first):
        table = integer_cartesian_table(1, 2, 1, device)
        cache.get_graph(data, 1.1, 0)
    with torch.xpu.stream(second):
        read_table = integer_cartesian_table(1, 2, 1, device).clone()
        actual = cache.get_graph(data, 1.1, 0)
        expected = radius_graph_pbc_v2(data, 1.1, 0)
        cache.clear()
    torch.xpu.synchronize(device)
    torch.testing.assert_close(read_table, table)
    assert _tuples(actual) == _tuples(expected)

"""Regression cases for periodic images, bounded searches and cache retention."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import fairchem.core.graph.portable_search as portable
import fairchem.core.graph.radius_graph_pbc as radius_graph
from fairchem.core.graph.compute import edge_cell_offsets, generate_graph
from fairchem.core.graph.geometry import minimum_image_displacement, wrap_positions
from fairchem.core.graph.neighbor_cache import VerletNeighborCache


def data_for(pos, cell, pbc, counts):
    counts = torch.tensor(counts)
    return SimpleNamespace(
        pos=pos,
        cell=cell,
        pbc=torch.tensor(pbc),
        natoms=counts,
        batch=torch.arange(len(counts)).repeat_interleave(counts),
    )


@pytest.mark.parametrize("method", ["all_pairs", "bvh", "cell", "auto"])
@pytest.mark.parametrize("cached", [False, True])
def test_atom_specific_image_shifts_preserve_vectors_and_derivatives(method, cached):
    cell = torch.tensor(
        [[[5.0, 0.5, 0], [0, 4.0, 0.25], [0, 0, 6.0]]],
        dtype=torch.double,
        requires_grad=True,
    )
    frac = torch.tensor(
        [[0.02, 0.04, 0.11], [0.96, 0.02, 0.11], [0.11, 0.07, 0.1]], dtype=torch.double
    )
    pos = (frac @ cell[0]).detach().requires_grad_()
    images = torch.tensor([[0.0, 0, 0], [3.0, -2, 0], [-4.0, 5, 0]], dtype=torch.double)

    def evaluate(shifts):
        data = data_for(pos + shifts @ cell[0], cell, [[True, True, False]], [3])
        graph = generate_graph(
            data,
            1.0,
            0,
            False,
            2,
            data.pbc,
            neighbor_cache=VerletNeighborCache(0.4) if cached else None,
            graph_search_method=method,
        )
        edge = graph["edge_index"]
        order = (edge[0] * 3 + edge[1]).argsort()
        assert len(order) == 6
        corrected = graph["cell_offsets"] + shifts[edge[0]] - shifts[edge[1]]
        energy = graph["edge_distance"].pow(3).sum()
        forces, stress = torch.autograd.grad(energy, (pos, cell), create_graph=True)
        hessian = torch.autograd.grad(forces.square().sum(), pos)[0]
        return (
            edge[:, order],
            corrected[order],
            graph["edge_distance_vec"][order],
            energy,
            forces,
            stress,
            hessian,
        )

    for observed, expected in zip(evaluate(images), evaluate(images * 0), strict=True):
        torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-10)


def test_v1_respects_override_when_checking_mixed_pbc():
    pos = torch.tensor([[0.0, 0, 0], [4.8, 0, 0]]).repeat(2, 1)
    data = data_for(
        pos,
        torch.eye(3).repeat(2, 1, 1) * 5,
        [[True, False, False], [False] * 3],
        [2, 2],
    )
    assert radius_graph.radius_graph_pbc(
        data, 0.5, 0, pbc=torch.tensor([True, False, False])
    )[2].tolist() == [2, 2]
    override = data.pbc.clone()
    data.pbc.zero_()
    with pytest.raises(RuntimeError, match="mixed PBC"):
        radius_graph.radius_graph_pbc(data, 0.5, 0, pbc=override)


def test_mixed_nonperiodic_zero_cell_and_skew_partial_periodic_sublattice():
    cell = torch.tensor([[[1.0, 0, 0], [0.9, 1, 0], [0, 0, 1]], [[0.0, 0, 0]] * 3])
    pbc = torch.tensor([[True, False, False], [False] * 3])
    dr = torch.tensor([[9.0, 10.0, 0], [7.0, -5.0, 12]], requires_grad=True)
    result = minimum_image_displacement(dr, cell, pbc, torch.tensor([0, 1]))
    torch.testing.assert_close(result, torch.tensor([[0.0, 10, 0], [7.0, -5, 12]]))
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), dr)[0], torch.ones_like(dr)
    )
    torch.testing.assert_close(wrap_positions(dr[1:], cell[1], False), dr[1:])
    np.testing.assert_array_equal(
        wrap_positions(dr[1:].detach().numpy(), cell[1].numpy(), False),
        dr[1:].detach().numpy(),
    )


def test_exact_image_count_before_allocation():
    assert radius_graph._image_counts([[128.0, 128.0, 128.0]]) == [16974593]
    with pytest.raises(ValueError, match="index range"):
        radius_graph._image_counts([[2**32] * 3])


@pytest.mark.parametrize("method", ["cell", "all_pairs"])
@pytest.mark.parametrize("strict", [True, False])
def test_tiny_tiles_preserve_order_and_cap_ties(method, strict, monkeypatch):
    pos = torch.arange(18, dtype=torch.float).reshape(6, 3) % 4
    data = data_for(
        pos, torch.eye(3).repeat(2, 1, 1) * 4, [[False] * 3, [True] * 3], [2, 4]
    )
    data.node_partition = torch.tensor([5, 2, 0, 4])
    expected = radius_graph.radius_graph_pbc_v2(data, 4, 1, strict, method=method)
    monkeypatch.setattr(radius_graph, "_STENCIL_TILE", 1)
    monkeypatch.setattr(radius_graph, "_CAP_TILE", 1)
    monkeypatch.setattr(portable, "_PAIR_TILE", 1)
    actual = radius_graph.radius_graph_pbc_v2(data, 4, 1, strict, method=method)
    for observed, reference in zip(actual, expected, strict=True):
        assert torch.equal(observed, reference)


def test_cache_byte_budget_aggregates_entries_and_does_not_retain_oversized_graph():
    cache = VerletNeighborCache(0.3, max_entries=3)
    pos = torch.tensor([[0.0, 0, 0], [0.2, 0, 0]])
    data = data_for(pos, torch.eye(3)[None], [[False] * 3], [2])
    cache.get_graph(data, 1, 0)
    one = cache.stats
    data.pos = pos + 3
    cache.get_graph(data, 1, 0)
    assert cache.stats["retained_bytes"] == 2 * one["retained_bytes"]
    assert cache.stats["candidate_edges"] == 2 * one["candidate_edges"]
    cache.get_graph(data, 1, 0)
    assert cache.stats["reused_graphs"] == 1
    budgeted = VerletNeighborCache(0.3, max_entries=3, max_bytes=one["retained_bytes"])
    for i in range(3):
        data.pos = pos + 3 * i
        budgeted.get_graph(data, 1, 0)
        assert budgeted.stats["entries"] == 1
    tight = VerletNeighborCache(0.3, max_bytes=1)
    assert tight.get_graph(data, 1, 0)[0].shape == (2, 2)
    assert tight.stats["retained_bytes"] == tight.stats["candidate_edges"] == 0


def test_segmented_cell_matmul_preserves_second_derivatives_and_zero_edge_systems():
    cell = torch.randn(3, 3, 3, dtype=torch.double, requires_grad=True)
    offsets = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    counts = torch.tensor([3, 0, 4])

    def eval_method(segmented):
        shifts = edge_cell_offsets(cell, offsets, counts, segmented=segmented)
        grads = torch.autograd.grad(
            shifts.square().sum(), (cell, offsets), create_graph=True
        )
        return shifts, *grads, torch.autograd.grad(grads[0].square().sum(), offsets)[0]

    for actual, expected in zip(eval_method(True), eval_method(False), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", ["all_pairs", "bvh"])
def test_interleaved_standalone_search_stays_within_systems(method):
    positions = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.5, 0, 0], [0.75, 0, 0]])
    systems = torch.tensor([1, 0, 1, 0])
    targets = torch.tensor([3, 0, 1])
    sources, local_targets = portable.alternative_candidates(
        positions, positions[targets], systems, systems[targets], 2, method
    )
    observed = sorted(zip(sources.tolist(), local_targets.tolist(), strict=True))
    assert observed == sorted(
        (s, t)
        for t, i in enumerate(targets)
        for s in range(4)
        if systems[s] == systems[i]
    )


def test_auto_dispatch_mixed_small_dense_and_sparse_systems():
    generator = torch.Generator().manual_seed(42)
    counts = [4, 258, 261]
    pos = torch.cat(
        [
            torch.rand(n, 3, generator=generator) * scale
            for n, scale in zip(counts, [1, 3, 500], strict=True)
        ]
    )
    data = data_for(pos, torch.zeros(3, 3, 3), [[False] * 3] * 3, counts)
    # Partition contains interleaved systems. Local candidate counts make
    # auto choose all_pairs, cell, and BVH in one call.
    data.node_partition = torch.arange(len(pos)).flip(0)
    expected = radius_graph.radius_graph_pbc_v2(data, 1.0, 4, method="all_pairs")
    actual = radius_graph.radius_graph_pbc_v2(data, 1.0, 4, method="auto")
    for edges, cells, counts in (expected, actual):
        assert (cells == 0).all()
        assert int(counts.sum()) == edges.shape[1]

    def sorted_ids(result):
        edges = result[0]
        return (edges[1] * len(pos) + edges[0]).sort().values

    torch.testing.assert_close(sorted_ids(expected), sorted_ids(actual))
    torch.testing.assert_close(expected[2], actual[2])


def test_v1_reports_original_frame_offsets_for_unwrapped_positions():
    pos = torch.tensor([[0.1, 0, 0], [10.3, 0, 0]])
    data = data_for(pos, torch.eye(3)[None] * 5, [[True, False, False]], [2])
    edges, offsets, counts = radius_graph.radius_graph_pbc(data, 0.5, 0)
    assert counts.tolist() == [2]
    vectors = pos[edges[0]] - pos[edges[1]] + offsets @ data.cell[0]
    torch.testing.assert_close(vectors.norm(dim=1), torch.full((2,), 0.2))


@pytest.mark.parametrize("strict", [False, True])
def test_skewed_degrees_segmented_cap_matches_legacy_tie_selection(strict, monkeypatch):
    counts = torch.tensor([0, 2, 4, 4, 3, 150, 4, 4, 3, 2, 0])
    targets = torch.arange(len(counts)).repeat_interleave(counts)
    distances = (torch.arange(len(targets)) % 13).float()
    distances[targets == 3] = 2
    args = (torch.tensor([len(counts)]), targets, distances, 2)
    monkeypatch.setattr(radius_graph, "_CAP_TILE", 1000000)
    expected = radius_graph.get_max_neighbors_mask(*args, enforce_max_strictly=strict)
    monkeypatch.setattr(radius_graph, "_CAP_TILE", 3)
    actual = radius_graph.get_max_neighbors_mask(*args, enforce_max_strictly=strict)
    for observed, reference in zip(actual, expected, strict=True):
        assert torch.equal(observed, reference)

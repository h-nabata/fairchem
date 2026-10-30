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
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph import portable_search
from fairchem.core.graph.compute import generate_graph
from fairchem.core.graph.neighbor_cache import VerletNeighborCache
from fairchem.core.graph.radius_graph_pbc import canonical_pbc, radius_graph_pbc_v2


def _data():
    return AtomicData.from_ase(
        Atoms(
            "H3", positions=[[0, 0, 0], [0.7, 0.1, 0], [1.4, 0.2, 0]], cell=[2, 3, 3]
        ),
        task_name="omol",
    )


def _edges(result):
    edge_index, offsets = result[:2]
    return sorted(
        (int(source), int(target), *map(int, offset))
        for (source, target), offset in zip(edge_index.T.tolist(), offsets.tolist())
    )


@pytest.mark.parametrize("cached", [False, True])
def test_partition_is_per_call_and_does_not_contaminate_later_jobs(cached):
    data = _data()
    cache = VerletNeighborCache(0.4) if cached else None
    keys = set(data.keys())
    partition = torch.tensor([1])
    partial = generate_graph(data, 1.1, 8, False, 2, data.pbc, partition, cache)
    assert set(partial["edge_index"][1].tolist()) == {1}
    assert not hasattr(data, "node_partition")
    assert set(data.keys()) == keys
    whole = generate_graph(data, 1.1, 8, False, 2, data.pbc, neighbor_cache=cache)
    expected = radius_graph_pbc_v2(data, 1.1, 8)
    assert _edges((whole["edge_index"], whole["cell_offsets"])) == _edges(expected)
    assert set(whole["edge_index"][1].tolist()) == {0, 1, 2}

    # Existing metadata is not an implicit per-call partition either.
    data.node_partition = partition
    whole = generate_graph(data, 1.1, 8, False, 2, data.pbc, neighbor_cache=cache)
    assert set(whole["edge_index"][1].tolist()) == {0, 1, 2}
    assert data.node_partition is partition


@pytest.mark.parametrize("kind", ["tensor", "numpy", "list"])
def test_pbc_override_normalization_preserves_inputs(kind):
    data = _data()
    original = data.pbc
    flags = {
        "tensor": torch.tensor([True, False, False]),
        "numpy": np.array([True, False, False]),
        "list": [True, False, False],
    }[kind]
    expected = radius_graph_pbc_v2(data, 1.1, 8, pbc=torch.tensor([True, False, False]))
    cache = VerletNeighborCache(0.4)
    actual = cache.get_graph(data, 1.1, 8, pbc=flags)
    assert _edges(actual) == _edges(expected)
    assert data.pbc is original
    data.pbc = torch.tensor([True, False, False])
    flat = data.pbc
    assert canonical_pbc(data, None) == [True, False, False]
    assert data.pbc is flat
    assert data.pbc.shape == (3,)
    assert canonical_pbc(SimpleNamespace(pbc=flat), None) == [True, False, False]


@pytest.mark.parametrize("field", ["pos", "cell", "pbc", "atomic_numbers"])
def test_numpy_writes_bypassing_tensor_versions_do_not_reuse_stale_topology(field):
    data = _data()
    cache = VerletNeighborCache(0.4)
    cache.get_graph(data, 1.1, 8)
    tensor = getattr(data, field)
    version = tensor._version
    array = tensor.numpy()
    if field == "pos":
        array[1, 0] += 0.35
    elif field == "cell":
        array[0, 0, 0] += 0.2
    elif field == "pbc":
        array[0, 0] = True
    else:
        array[1] = 8
    assert tensor._version == version
    actual = cache.get_graph(data, 1.1, 8)
    expected = radius_graph_pbc_v2(data, 1.1, 8)
    assert _edges(actual) == _edges(expected)
    assert cache.rebuilds == 2


def test_cache_skin_change_rebuilds_outer_candidates():
    data = _data()
    data.pos[2] = torch.tensor([2.0, 0.0, 0.0])
    cache = VerletNeighborCache(0.2)
    cache.get_graph(data, 1.1, 8)
    cache.skin = 1.0
    data.pos[2, 0] = 1.65
    actual = cache.get_graph(data, 1.1, 8)
    assert _edges(actual) == _edges(radius_graph_pbc_v2(data, 1.1, 8))
    assert cache.rebuilds == 2


def test_returned_topology_and_integer_tables_cannot_contaminate_other_jobs():
    data = _data()
    cache = VerletNeighborCache(0.4)
    expected = _edges(cache.get_graph(data, 1.1, 8))
    edges, cells, counts = cache.get_graph(data, 1.1, 8)
    edges.numpy()[:] = 0
    cells.zero_()
    counts.zero_()
    assert _edges(cache.get_graph(data, 1.1, 8)) == expected
    builder = portable_search._integer_cartesian_entry
    builder.cache_clear()
    try:
        table = portable_search.integer_cartesian_table(2, 2, 2, "cpu")
        pristine = table.clone()
        table.numpy()[:] = 1000
        restored = portable_search.integer_cartesian_table(2, 2, 2, torch.device("cpu"))
        torch.testing.assert_close(restored, pristine)
        assert restored.data_ptr() != table.data_ptr()
        assert _edges(radius_graph_pbc_v2(data, 1.1, 8)) == expected
    finally:
        builder.cache_clear()


def test_implicit_accelerator_ordinal_is_resolved_before_table_cache_lookup(
    monkeypatch,
):
    seen = []

    def fake_entry(x, y, z, device):
        seen.append(device)
        return torch.zeros((1, 3), dtype=torch.long), None

    monkeypatch.setattr(portable_search, "_integer_cartesian_entry", fake_entry)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)
    portable_search.integer_cartesian_table(0, 0, 0, "xpu")
    portable_search.integer_cartesian_table(0, 0, 0, "xpu:0")
    assert seen == [torch.device("xpu:1"), torch.device("xpu:0")]


@pytest.mark.parametrize("value", [-1, 0.5, True, np.bool_(False)])
def test_integer_tables_reject_lossy_extent_conversions(value):
    with pytest.raises(ValueError, match="nonnegative integers"):
        portable_search.integer_cartesian_table(value, 0, 0, "cpu")


@pytest.mark.parametrize("radius", [-0.1, 0, float("nan"), float("inf"), True, "1.1"])
def test_cached_and_fresh_graphs_reject_invalid_radius_consistently(radius):
    data = _data()
    with pytest.raises(ValueError, match="radius"):
        radius_graph_pbc_v2(data, radius, 8)
    cache = VerletNeighborCache(0.4)
    with pytest.raises(ValueError, match="radius"):
        cache.get_graph(data, radius, 8)
    assert cache.rebuilds == 0


@pytest.mark.parametrize("threshold", [2.5, True, np.bool_(False), "2"])
def test_graph_caps_are_not_silently_cast_to_another_value(threshold):
    with pytest.raises(ValueError, match="integer"):
        VerletNeighborCache().get_graph(_data(), 1.1, threshold)


@pytest.mark.parametrize("partition", [[-1], [3], [1, 1], [[1]], [0.5]])
def test_partition_indices_are_validated_before_indexing(partition):
    data = _data()
    data.node_partition = torch.as_tensor(partition)
    with pytest.raises(ValueError, match="node_partition"):
        radius_graph_pbc_v2(data, 1.1, 8)


@pytest.mark.parametrize(
    "pbc", [[True, False], [[True] * 3] * 2, [True, False, 2], [0, float("nan"), 0]]
)
def test_pbc_shape_and_values_are_validated_before_graph_generation(pbc):
    with pytest.raises(ValueError, match="PBC"):
        VerletNeighborCache().get_graph(_data(), 1.1, 8, pbc=pbc)


@pytest.mark.parametrize("method", ["all_pairs", "bvh"])
@pytest.mark.parametrize("empty_source", [False, True])
def test_alternative_search_handles_empty_candidate_sides(method, empty_source):
    source = torch.zeros((0 if empty_source else 2, 3))
    target = torch.zeros((2 if empty_source else 0, 3))
    result = portable_search.alternative_candidates(
        source,
        target,
        torch.zeros(len(source), dtype=torch.long),
        torch.zeros(len(target), dtype=torch.long),
        1.1,
        method,
    )
    assert all(value.shape == (0,) and value.dtype == torch.long for value in result)


@pytest.mark.parametrize("field", ["pos", "cell"])
@pytest.mark.parametrize("method", ["cell", "all_pairs", "bvh"])
@pytest.mark.parametrize("cached", [False, True])
def test_nonfinite_mutated_job_geometry_is_rejected_instead_of_losing_edges(
    field, method, cached
):
    data = _data()
    cache = VerletNeighborCache(0.4)
    cache.get_graph(data, 1.1, 8, method=method)
    tensor = getattr(data, field)
    tensor.reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        if cached:
            cache.get_graph(data, 1.1, 8, method=method)
        else:
            radius_graph_pbc_v2(data, 1.1, 8, method=method)


def test_zero_cell_is_allowed_only_without_periodic_directions():
    data = _data()
    data.cell.zero_()
    assert radius_graph_pbc_v2(data, 1.1, 8)[0].shape[1] > 0
    data.pbc[0, 0] = True
    with pytest.raises(ValueError, match="nonsingular"):
        radius_graph_pbc_v2(data, 1.1, 8)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("invalid", ["interleaved", "count", "range", "natoms"])
def test_mutated_batch_layout_cannot_mix_coordinates_between_systems(cached, invalid):
    data = atomicdata_list_to_batch([_data(), _data()])
    cache = VerletNeighborCache(0.4)
    cache.get_graph(data, 1.1, 8)
    if invalid == "interleaved":
        data.batch.numpy()[:] = [0, 1, 0, 1, 0, 1]
    elif invalid == "count":
        data.batch.numpy()[:] = [0, 0, 1, 1, 1, 1]
    elif invalid == "range":
        data.batch[0] = -1
    else:
        data.natoms[0] += 1
    with pytest.raises(ValueError, match="batch|natoms"):
        if cached:
            cache.get_graph(data, 1.1, 8)
        else:
            radius_graph_pbc_v2(data, 1.1, 8)

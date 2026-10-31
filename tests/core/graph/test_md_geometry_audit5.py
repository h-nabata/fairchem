"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import weakref
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ase import Atoms
from torch.utils._python_dispatch import TorchDispatchMode

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph.compute import get_pbc_distances
from fairchem.core.graph.neighbor_cache import VerletNeighborCache
from fairchem.core.graph.radius_graph_pbc import radius_graph_pbc_v2
from tests.core.graph.test_portable_search_cache import _tuples
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


class _Operations(TorchDispatchMode):
    def __init__(self):
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("systems", [1, 2])
def test_nonzero_md_geometry_avoids_dynamic_selection_and_preserves_derivatives(
    dtype, systems
):
    pos = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.6, 0.8], [1.4, 0.3, 0.2], [1.8, 0.7, 0.9]],
        dtype=dtype,
        requires_grad=True,
    )
    cell = torch.eye(3, dtype=dtype).repeat(systems, 1, 1).requires_grad_()
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    coefficients = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    counts = torch.tensor([4] if systems == 1 else [2, 2])
    with _Operations() as operations:
        actual = get_pbc_distances(
            pos,
            edges,
            cell,
            coefficients,
            counts,
            return_offsets=True,
            return_distance_vec=True,
        )
    assert not operations.counts["aten.nonzero.default"]
    if systems == 1:
        assert not operations.counts["aten.arange.default"]
    # A previous returned graph must remain independent of later input reuse.
    assert actual["edge_index"].data_ptr() != edges.data_ptr()
    reference_offsets = torch.stack(
        [
            coefficients[i].to(dtype) @ cell[0 if systems == 1 else i // 2]
            for i in range(4)
        ]
    )
    reference_vectors = pos[edges[0]] - pos[edges[1]] + reference_offsets
    reference_distances = reference_vectors.norm(dim=-1)
    torch.testing.assert_close(actual["offsets"], reference_offsets, rtol=0, atol=0)
    torch.testing.assert_close(
        actual["distance_vec"], reference_vectors, rtol=0, atol=0
    )
    torch.testing.assert_close(actual["distances"], reference_distances, rtol=0, atol=0)
    first = torch.autograd.grad(
        actual["distances"].exp().sum(), (pos, cell), create_graph=True
    )
    expected_first = torch.autograd.grad(
        reference_distances.exp().sum(), (pos, cell), create_graph=True
    )
    second = torch.autograd.grad(
        sum(value.square().sum() for value in first), (pos, cell)
    )
    expected_second = torch.autograd.grad(
        sum(value.square().sum() for value in expected_first), (pos, cell)
    )
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second, expected_second)
    saved = actual["edge_index"].clone()
    edges.zero_()
    torch.testing.assert_close(actual["edge_index"], saved, rtol=0, atol=0)


@pytest.mark.parametrize(
    "flags", [(False, False), (True, False), (False, True), (True, True)]
)
def test_empty_geometry_preserves_requested_output_shapes(flags):
    offsets, vectors = flags
    result = get_pbc_distances(
        torch.empty(0, 3),
        torch.empty(2, 0, dtype=torch.long),
        torch.eye(3).unsqueeze(0),
        torch.empty(0, 3),
        torch.zeros(1, dtype=torch.long),
        return_offsets=offsets,
        return_distance_vec=vectors,
    )
    assert result["edge_index"].shape == (2, 0)
    assert result["distances"].shape == (0,)
    assert ("offsets" in result) == offsets
    assert ("distance_vec" in result) == vectors
    if offsets:
        assert result["offsets"].shape == (0, 3)
    if vectors:
        assert result["distance_vec"].shape == (0, 3)


@pytest.mark.parametrize("method", ["all_pairs", "bvh", "auto"])
@pytest.mark.parametrize("reason", ["boundary", "cap"])
def test_neighbor_fallback_preserves_requested_search_method(method, reason):
    positions = (
        [[0, 0, 0], [1, 0, 0]]
        if reason == "boundary"
        else [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [np.sqrt(1.01), 0, 0]]
    )
    data = atomicdata_list_to_batch(
        [
            AtomicData.from_ase(
                Atoms("H" * len(positions), positions=positions, cell=[4] * 3),
                task_name="omol",
            )
        ]
    )
    cache = VerletNeighborCache(0.5)
    cutoff, cap = (1.0, 0) if reason == "boundary" else (2.0, 1)
    with patch(
        "fairchem.core.graph.neighbor_cache.radius_graph_pbc_v2",
        wraps=radius_graph_pbc_v2,
    ) as builder:
        actual = cache.get_graph(data, cutoff, cap, method=method)
        assert len(builder.call_args_list) == 2
        assert all(call.kwargs["method"] == method for call in builder.call_args_list)
    expected = radius_graph_pbc_v2(data, cutoff, cap, method=method)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_repeated_real_uma_geometry_updates_keep_cache_bounded_and_outputs_owned():
    reference = _public_predictor()
    # Exercise the configured backbone directly: this fixture does not load a
    # checkpoint configuration for the inference-time MoLE replacement.
    reference.inference_settings.merge_mole = False
    cached = copy.deepcopy(reference)
    cached.inference_settings.neighbor_skin = 0.5
    # This small-model fixture bypasses checkpoint loading, where the public
    # constructor normally applies the setting to the backbone configuration.
    cached.model.module.backbone.neighbor_skin = 0.5
    data = _data()
    data.pos += 2.0
    data.cell = torch.eye(3).unsqueeze(0) * 8
    data.pbc[:] = True
    snapshots = []
    for step in range(16):
        data.pos[1, 0] += 0.001
        if step == 8:
            data.cell[0, 0, 0] += 0.001
        position_before = data.pos.clone()
        actual = cached.predict(data, requested_properties=("energy", "forces"))
        expected = reference.predict(data, requested_properties=("energy", "forces"))
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(data.pos, position_before, atol=0, rtol=0)
        snapshots.append(
            (actual, {name: value.clone() for name, value in actual.items()})
        )
    cache = cached.model.module.backbone._neighbor_cache
    assert cache.stats["entries"] == 1
    assert cache.stats["rebuilds"] == 2
    assert cache.stats["hits"] == 14
    assert not cache._entry["pos"].requires_grad
    for actual, saved in snapshots:
        torch.testing.assert_close(actual, saved, atol=0, rtol=0)


@pytest.mark.parametrize("cap", [0, 1, 300])
def test_neighbor_rebuild_retains_only_owned_topology_and_releases_replaced_entry(cap):
    data = _data()
    cache = VerletNeighborCache(0.5)
    expected = radius_graph_pbc_v2(data, 3.0, cap)
    actual = cache.get_graph(data, 3.0, cap)
    assert _tuples(actual) == _tuples(expected)
    torch.testing.assert_close(actual[2], expected[2])
    saved = tuple(value.clone() for value in actual)
    # Caller results and its source metadata are independent of retained state.
    actual[0].fill_(0)
    actual[1].fill_(91)
    actual[2].fill_(0)
    repeated = cache.get_graph(data, 3.0, cap)
    torch.testing.assert_close(repeated, saved)
    old_edges = weakref.ref(cache._entry["edges"])
    old_cells = weakref.ref(cache._entry["cells"])
    data.pos[1, 0] += 0.3
    rebuilt = cache.get_graph(data, 3.0, cap)
    assert cache.stats["entries"] == 1
    assert cache.stats["rebuilds"] == 2
    assert old_edges() is None
    assert old_cells() is None
    fresh = radius_graph_pbc_v2(data, 3.0, cap)
    assert _tuples(rebuilt) == _tuples(fresh)
    torch.testing.assert_close(rebuilt[2], fresh[2])

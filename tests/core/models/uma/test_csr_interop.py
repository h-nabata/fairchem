"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fairchem.core.models.uma.nn.portable_wigner import CSRTopologyCache, segmented_sum


def test_csr_cpu_cache_observes_numpy_changes_without_version_increments():
    array = np.array([0, 1, 0], dtype=np.int64)
    target = torch.from_numpy(array)
    cache = CSRTopologyCache()
    first = cache.get(target, 2)
    version = target._version
    array[:] = [1, 0, 1]
    assert target._version == version
    second = cache.get(target, 2)
    assert second is not first
    assert second.order.tolist() == [1, 0, 2]
    assert second.lengths.tolist() == [1, 2]
    assert cache.get(target, 2) is second
    assert cache.hits == 1
    array[0] = 2
    with pytest.raises(ValueError, match="outside"):
        cache.get(target, 2)


def test_segment_backward_owns_forward_topology_across_numpy_jobs():
    array = np.array([0, 1, 0], dtype=np.int64)
    target = torch.from_numpy(array)
    original = target.clone()
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.double, requires_grad=True)
    first = segmented_sum(values.square(), target, 2)
    array[:] = [1, 0, 1]
    second = segmented_sum(values.square(), target, 2)
    expected_first = values.new_zeros(2, 1).index_add(0, original, values.square())
    expected_second = values.new_zeros(2, 1).index_add(0, target, values.square())
    torch.testing.assert_close(first, expected_first)
    torch.testing.assert_close(second, expected_second)
    assert first.data_ptr() != second.data_ptr()
    seed = torch.tensor([[3.0], [7.0]], dtype=torch.double)
    gradient = torch.autograd.grad((first * seed).sum(), values, create_graph=True)[0]
    torch.testing.assert_close(gradient, 2 * values * seed[original])
    curvature = torch.autograd.grad(gradient.sum(), values)[0]
    torch.testing.assert_close(curvature, 2 * seed[original])


def test_csr_snapshot_counts_towards_the_memory_budget():
    target = torch.tensor([0, 1, 0])
    cache = CSRTopologyCache()
    topology = cache.get(target, 2)
    assert topology.owner_snapshot.data_ptr() != target.data_ptr()
    tensor_bytes = sum(
        value.untyped_storage().nbytes()
        for value in (
            target,
            topology.order,
            topology.lengths,
            topology.offsets,
            topology.owner_snapshot,
        )
    )
    assert cache._size(topology) == tensor_bytes
    bounded = CSRTopologyCache(max_bytes=tensor_bytes - 1)
    bounded.get(target, 2)
    assert not bounded._entries


@pytest.mark.parametrize("length", [0, 2, 4])
def test_segment_length_mismatch_never_silently_drops_values(length):
    with pytest.raises(ValueError, match="same leading length"):
        segmented_sum(torch.ones(length, 2), torch.tensor([0, 1, 0]), 2)

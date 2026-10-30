"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from fairchem.core.models.uma.outputs import (
    compute_forces,
    compute_hessian,
    reduce_node_to_system,
)


def test_strided_reduction_preserves_values_and_derivatives():
    base = torch.arange(24, dtype=torch.float64, requires_grad=True)
    values = base.reshape(4, 2, 3).transpose(1, 2)
    assert not values.is_contiguous()
    actual, _ = reduce_node_to_system(values, torch.tensor([0, 0, 1, 1]), 2)
    expected = torch.stack((values[:2].sum(0), values[2:].sum(0)))
    torch.testing.assert_close(actual, expected)
    first_actual = torch.autograd.grad(actual.square().sum(), base, create_graph=True)[
        0
    ]
    first_expected = torch.autograd.grad(
        expected.square().sum(), base, create_graph=True
    )[0]
    torch.testing.assert_close(first_actual, first_expected)
    torch.testing.assert_close(
        torch.autograd.grad(first_actual.sum(), base)[0],
        torch.autograd.grad(first_expected.sum(), base)[0],
    )


@pytest.mark.parametrize("trailing_shape", [(), (3,), (3, 3)])
def test_empty_reduction_preserves_fp64_accumulation(trailing_shape):
    values = torch.empty((0, *trailing_shape), requires_grad=True)
    result, _ = reduce_node_to_system(values, torch.empty(0, dtype=torch.long), 2)
    torch.testing.assert_close(
        result, torch.zeros((2, *trailing_shape), dtype=torch.float64)
    )
    torch.testing.assert_close(torch.autograd.grad(result.sum(), values)[0], values)


@pytest.mark.parametrize("vmap", [False, True])
def test_hessian_retains_graph_when_third_derivatives_are_requested(vmap):
    positions = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    forces = compute_forces(positions.square().sin().sum(), positions, training=True)
    hessian = compute_hessian(forces, positions, vmap=vmap, training=True)
    expected_diagonal = (
        2 * positions.square().cos() - 4 * positions.square() * positions.square().sin()
    )
    torch.testing.assert_close(hessian[0], expected_diagonal.flatten().diag())
    expected_third = (
        -12 * positions * positions.square().sin()
        - 8 * positions.pow(3) * positions.square().cos()
    )
    torch.testing.assert_close(
        torch.autograd.grad(hessian.sum(), positions)[0], expected_third
    )


class _SavedTensor:
    def __init__(self, tensor):
        self.tensor = tensor


@pytest.mark.parametrize("vmap", [False, True])
def test_inference_hessian_releases_saved_intermediates(vmap):
    saved_references = []

    def pack(tensor):
        saved = _SavedTensor(tensor)
        saved_references.append(weakref.ref(saved))
        return saved

    positions = torch.tensor([[0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda saved: saved.tensor):
        forces = compute_forces(
            positions.square().sin().sum(), positions, training=True
        )
        hessian = compute_hessian(forces, positions, vmap=vmap, training=False)
    # Keep returned forces/Hessian alive: releasing only the local function
    # variables would hide retained model intermediates in normal prediction.
    assert torch.isfinite(forces).all()
    assert torch.isfinite(hessian).all()
    gc.collect()
    assert saved_references
    assert all(reference() is None for reference in saved_references)

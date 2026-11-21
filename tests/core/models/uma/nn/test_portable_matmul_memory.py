"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from fairchem.core.models.uma.nn.matmul import linear_with_folded_batch
from fairchem.core.models.uma.nn.mole import MOLE, MOLEGlobals


def _layer(sizes, offset=0, *, bias=True, frozen=False, channels=(4, 3)):
    generator = torch.Generator().manual_seed(732)
    globals_ = MOLEGlobals(
        torch.randn(len(sizes), 3, dtype=torch.float64, generator=generator),
        torch.tensor(sizes),
        ac_start_idx=offset,
    )
    layer = MOLE(3, *channels, globals_, bias=bias).double()
    layer.requires_grad_(not frozen)
    globals_.expert_mixing_coefficients.requires_grad_(not frozen)
    return layer


def _reference(layer, x):
    globals_ = layer.global_mole_tensors
    weights = torch.einsum(
        "eoi,be->boi", layer.weights, globals_.expert_mixing_coefficients
    )
    owners = torch.repeat_interleave(
        torch.arange(len(globals_.mole_sizes)), globals_.mole_sizes
    )[globals_.ac_start_idx : globals_.ac_start_idx + len(x)]
    output = torch.einsum("n...i,noi->n...o", x, weights[owners])
    return output if layer.bias is None else output + layer.bias


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize(
    ("sizes", "offset", "length", "bias"),
    [
        ([3, 4, 2], 0, 9, True),
        ([3, 4, 2], 3, 4, False),
        ([3, 4, 2], 2, 6, True),
        ([0, 3, 0, 4, 0], 1, 5, False),
        ([3, 4], 3, 0, True),
        ([0, 0], 0, 0, True),
    ],
)
def test_chunked_mole_matches_per_row_values_and_two_derivatives(
    sizes, offset, length, bias, rank, frozen
):
    layer = _layer(sizes, offset, bias=bias, frozen=frozen)
    # A permuted input also exercises the frozen-weight folded GEMM path.
    x = (
        torch.randn(length, 4, 2, dtype=torch.float64).transpose(1, 2)
        if rank == 3
        else torch.randn(length, 8, dtype=torch.float64)[:, ::2]
    ).requires_grad_()
    actual, expected = layer(x), _reference(layer, x)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    variables = [x]
    if not frozen:
        variables += [
            layer.weights,
            layer.global_mole_tensors.expert_mixing_coefficients,
        ]
        if layer.bias is not None:
            variables.append(layer.bias)
    gradients = []
    for output in (actual, expected):
        first = torch.autograd.grad(
            output.sin().square().sum(), variables, create_graph=True
        )
        second = torch.autograd.grad(sum(g.square().sum() for g in first), variables)
        gradients.append((first, second))
    for actual_order, expected_order in zip(*gradients):
        for actual_value, expected_value in zip(actual_order, expected_order):
            torch.testing.assert_close(
                actual_value, expected_value, atol=1e-10, rtol=1e-10
            )


def test_single_system_chunk_limits_mixed_weight_allocation_and_avoids_cat():
    layer = _layer([5] * 40, offset=50, frozen=True, channels=(64, 96)).float()
    layer.global_mole_tensors.expert_mixing_coefficients = (
        layer.global_mole_tensors.expert_mixing_coefficients.float()
    )
    x = torch.randn(5, 2, 64, requires_grad=True)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as trace:
        output = layer(x)
    torch.testing.assert_close(output, _reference(layer, x))
    # The expert contraction must allocate one system's [out, in] matrix,
    # not 40 matrices for a chunk that touches only one system.
    matrix_allocations = [
        event.cpu_memory_usage for event in trace.events() if event.name == "aten::bmm"
    ]
    assert matrix_allocations == [96 * 64 * x.element_size()]
    assert not any(event.name == "aten::cat" for event in trace.events())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_merged_linear_preserves_precision_and_independent_parameters(dtype, bias):
    layer = _layer([2, 3, 4], bias=bias).to(dtype=dtype)
    layer.global_mole_tensors.expert_mixing_coefficients = (
        layer.global_mole_tensors.expert_mixing_coefficients.to(dtype=dtype)
    )
    expected = torch.einsum(
        "eoi,be->boi",
        layer.weights,
        layer.global_mole_tensors.expert_mixing_coefficients,
    )[0]
    generator_state = torch.random.get_rng_state()
    merged = layer.merged_linear_layer()
    assert merged.weight.dtype == dtype
    assert merged.weight.device == layer.weights.device
    assert merged.weight.is_leaf
    assert merged.weight.requires_grad
    assert torch.equal(torch.random.get_rng_state(), generator_state)
    torch.testing.assert_close(merged.weight, expected)
    if bias:
        assert merged.bias.data_ptr() != layer.bias.data_ptr()
        torch.testing.assert_close(merged.bias, layer.bias)
    with torch.no_grad():
        layer.weights.add_(1)
    torch.testing.assert_close(merged.weight, expected)


@pytest.mark.parametrize("shape", [(0, 2, 4), (2, 0, 4), (2, 3, 4)])
def test_folded_linear_preserves_empty_and_strided_shapes(shape):
    x = torch.randn(*shape, 2, dtype=torch.float64)[..., 0].requires_grad_()
    weights = torch.randn(3, 4, dtype=torch.float64)
    actual = linear_with_folded_batch(x, weights)
    expected = torch.einsum("...i,oi->...o", x, weights)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (*shape[:-1], 3)


@pytest.mark.parametrize(("offset", "length"), [(-1, 2), (3, 3), (6, 0)])
def test_mole_rejects_chunks_outside_system_boundaries(offset, length):
    layer = _layer([2, 3], offset)
    with pytest.raises(AssertionError, match="outside MoLE sizes"):
        layer(torch.randn(length, 4, dtype=torch.float64))

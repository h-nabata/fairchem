"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from fairchem.core.models.uma.nn.matmul import linear_with_folded_batch


def _input(kind, dtype=torch.float64):
    # Match the order-m slice of nine SO(2) coefficients. Flattening its leading
    # dimensions needs a copy; this is the broadcast-BMM regression trigger.
    x = torch.randn(16, 9, 8, dtype=dtype)[:, 3:7].view(16, 2, 16)
    if kind == "rank4":
        return x.reshape(4, 4, 2, 16).transpose(0, 1)
    if kind == "empty":
        return x[:0]
    return x[2:13]


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode, nullcontext])
@pytest.mark.parametrize("kind", ["strided", "rank4", "empty"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_frozen_inference_uses_one_gemm_without_changing_values(
    context, kind, use_bias
):
    x = _input(kind)
    weight = torch.randn(12, 16, dtype=x.dtype)
    bias = torch.randn(12, dtype=x.dtype) if use_bias else None
    with context():
        reference = F.linear(x, weight, bias)
        with profile(activities=[ProfilerActivity.CPU]) as trace:
            actual = linear_with_folded_batch(x, weight, bias)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)
    assert actual.shape == reference.shape
    assert not actual.requires_grad
    operators = {event.key for event in trace.key_averages()}
    assert "aten::bmm" not in operators
    assert "aten::mm" in operators or "aten::addmm" in operators


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("kind", ["strided", "rank4"])
def test_folded_and_trainable_paths_preserve_first_and_second_derivatives(
    trainable, kind
):
    x = _input(kind).requires_grad_()
    weight = torch.randn(12, 16, dtype=x.dtype, requires_grad=trainable)
    # Bias gradients remain supported even if only weights are frozen.
    bias = torch.randn(12, dtype=x.dtype, requires_grad=True)
    variables = [x, bias, *([weight] if trainable else [])]
    results = []
    for function in (linear_with_folded_batch, F.linear):
        output = function(x, weight, bias)
        first = torch.autograd.grad(output.sin().sum(), variables, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in first), variables
        )
        results.append((output, first, second))
    for actual, reference in zip(*results, strict=True):
        torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-10)


def test_folded_inference_preserves_ambient_autocast_and_compile_contract():
    x = _input("strided", torch.float32)
    weight = torch.randn(12, 16)
    compiled = torch.compile(linear_with_folded_batch, backend="eager", fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16), torch.no_grad():
        actual = compiled(x, weight)
        reference = F.linear(x.contiguous(), weight)
        assert torch.is_autocast_enabled("cpu")
    assert actual.dtype == reference.dtype == torch.bfloat16
    torch.testing.assert_close(actual, reference)

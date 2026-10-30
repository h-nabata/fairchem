"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.models.uma.common.rotation import EPS, Safeacos
from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.execution_backends import (
    ExecutionBackend,
    UMASFastPytorchBackend,
)


def _dense_reference(wigner, inverse, mapping, coefficient_index):
    if coefficient_index is not None:
        wigner = wigner.index_select(1, coefficient_index)
        inverse = inverse.index_select(2, coefficient_index)
    return (
        torch.einsum("mk,nkj->nmj", mapping.to_m.to(wigner.dtype), wigner),
        torch.einsum("njk,mk->njm", inverse, mapping.to_m.to(inverse.dtype)),
    )


@pytest.mark.parametrize(("lmax", "mmax"), [(0, 0), (2, 2), (4, 2), (6, 4)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("edges", [0, 7])
def test_wigner_permutation_matches_dense_with_strided_inputs(lmax, mmax, dtype, edges):
    torch.manual_seed(17)
    mapping = CoefficientMapping(lmax, mmax).to(dtype=dtype)
    full = CoefficientMapping(lmax, lmax)
    coefficient_index = full.coefficient_idx(lmax, mmax) if mmax != lmax else None
    size = (lmax + 1) ** 2
    wigner = torch.randn(edges, size, size * 2, dtype=dtype)[..., ::2]
    inverse = wigner.transpose(1, 2)
    expected = _dense_reference(wigner, inverse, mapping, coefficient_index)
    for backend in (ExecutionBackend, UMASFastPytorchBackend):
        actual = backend.prepare_wigner(wigner, inverse, mapping, coefficient_index)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
    assert mapping.to_m_index.dtype == torch.int64
    assert "to_m_index" not in mapping.state_dict()


@pytest.mark.parametrize(("lmax", "mmax"), [(2, 2), (4, 2)])
def test_wigner_permutation_preserves_first_and_second_derivatives(lmax, mmax):
    torch.manual_seed(23)
    mapping = CoefficientMapping(lmax, mmax).double()
    full = CoefficientMapping(lmax, lmax)
    coefficient_index = full.coefficient_idx(lmax, mmax) if mmax != lmax else None
    size = (lmax + 1) ** 2
    actual_inputs = tuple(
        torch.randn(3, size, size, dtype=torch.double, requires_grad=True)
        for _ in range(2)
    )
    reference_inputs = tuple(
        value.detach().clone().requires_grad_() for value in actual_inputs
    )

    def derivatives(function, inputs):
        output = function(*inputs, mapping, coefficient_index)
        loss = sum(value.sin().square().sum() for value in output)
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(sum(value.sum() for value in first), inputs)
        return (*output, *first, *second)

    actual = derivatives(ExecutionBackend.prepare_wigner, actual_inputs)
    expected = derivatives(_dense_reference, reference_inputs)
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)


def test_wigner_permutation_has_no_dynamo_graph_break():
    mapping = CoefficientMapping(2, 2)

    def prepare(wigner, inverse):
        return ExecutionBackend.prepare_wigner(wigner, inverse, mapping, None)

    compiled = torch.compile(prepare, backend="eager", fullgraph=True, dynamic=True)
    for edges in (3, 5):
        wigner = torch.randn(edges, 9, 9)
        inverse = wigner.transpose(1, 2)
        for result, reference in zip(
            compiled(wigner, inverse), prepare(wigner, inverse), strict=True
        ):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_wigner_permutation_keeps_fake_xpu_placement():
    # FakeTensor checks metadata only, not Intel kernel execution.
    with FakeTensorMode(allow_fallback_kernels=False):
        wigner = torch.empty(4, 25, 25, device="xpu:0")
        mapping = SimpleNamespace(
            to_m_index=torch.empty(19, dtype=torch.long, device="xpu:0")
        )
        coefficient_index = torch.empty(19, dtype=torch.long, device="xpu:0")
        output = ExecutionBackend.prepare_wigner(
            wigner, wigner.transpose(1, 2), mapping, coefficient_index
        )
        assert output[0].shape == (4, 19, 25)
        assert output[1].shape == (4, 25, 19)
        assert all(value.device == wigner.device for value in output)


def test_safeacos_first_and_second_derivatives_match_analytic_reference():
    x = torch.tensor(
        [-0.8, -0.2, 0.0, 0.4, 0.9], dtype=torch.double, requires_grad=True
    )
    value = Safeacos.apply(x)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    torch.testing.assert_close(value, torch.acos(x))
    torch.testing.assert_close(first, -(1 - x.square()).rsqrt())
    torch.testing.assert_close(second, -x / (1 - x.square()).pow(1.5))
    assert torch.autograd.gradcheck(Safeacos.apply, (x,))
    assert torch.autograd.gradgradcheck(Safeacos.apply, (x,))


def test_safeacos_preserves_regularized_endpoint_derivatives():
    x = torch.tensor([-1.0, 1.0], dtype=torch.double, requires_grad=True)
    value = Safeacos.apply(x)
    first = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), x)[0]
    torch.testing.assert_close(value, torch.acos(x))
    expected = -1 / (1 - x.clamp(-1 + EPS, 1 - EPS).square()).sqrt().clamp(min=EPS)
    torch.testing.assert_close(first, expected)
    assert torch.isfinite(first).all()
    torch.testing.assert_close(second, torch.zeros_like(x))

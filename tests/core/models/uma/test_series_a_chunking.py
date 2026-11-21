"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.escn_md_block import Edgewise
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend


def test_streaming_checkpoint_chunks_match_full_values_and_second_derivatives(
    monkeypatch,
):
    base = Edgewise(
        sphere_channels=2,
        hidden_channels=3,
        lmax=2,
        mmax=2,
        edge_channels_list=[10, 5, 5],
        mappingReduced=CoefficientMapping(2, 2),
        SO3_grid=None,
        cutoff=5.0,
        activation_checkpoint_chunk_size=None,
        backend=ExecutionBackend(),
    ).double()
    chunked = copy.deepcopy(base)
    chunked.activation_checkpoint_chunk_size = 2
    x = torch.randn(5, 9, 2, dtype=torch.float64, requires_grad=True)
    radial = torch.randn(19, 10, dtype=torch.float64, requires_grad=True)
    edge_index = torch.randint(0, 5, (2, 19))
    wigner = torch.randn(19, 9, 9, dtype=torch.float64, requires_grad=True)
    inv = torch.randn(19, 9, 9, dtype=torch.float64, requires_grad=True)
    residual = torch.randn_like(x, requires_grad=True)

    def forbidden_stack(*args, **kwargs):
        raise AssertionError("chunking must not stack full-node outputs")

    expected = base(x, radial, edge_index, wigner, inv, 5, residual=residual)
    monkeypatch.setattr(torch, "stack", forbidden_stack)
    # No explicit target also exercises the documented no-GP default.
    actual = chunked(x, radial, edge_index, wigner, inv, 5, residual=residual)
    torch.testing.assert_close(actual, expected)
    inputs = (x, radial, wigner, inv, residual)
    ga = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), inputs, create_graph=True)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b)
    ha = torch.autograd.grad(sum(a.square().sum() for a in ga), inputs)
    hb = torch.autograd.grad(sum(b.square().sum() for b in gb), inputs)
    for a, b in zip(ha, hb):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-8)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.nn.approximate_ops import (
    ApproximateGaussian,
    apply_feature_approximations,
    approximate_ops_context,
    compensated_node_sum,
    compensated_reduction_enabled,
    local_fast_exp,
)
from fairchem.core.models.uma.nn.radial import GaussianSmearing
from fairchem.core.models.uma.outputs import reduce_node_to_system


def _derivatives(function, values):
    output = function(values)
    first = torch.autograd.grad(output.sum(), values, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), values)[0]
    return output, first, second


def test_compensated_sum_recovers_cancellation_and_keeps_systems_separate():
    values = (
        torch.tensor(
            [[1e8, 1], [10, 2], [1, 3], [-1e8, 4], [-10, 5]], dtype=torch.float32
        )
        .T.contiguous()
        .T
    )
    assert not values.is_contiguous()
    batch = torch.tensor([0, 2, 0, 0, 2])
    actual = compensated_node_sum(values, batch, 3)
    expected = values.new_zeros(3, 2, dtype=torch.float64).index_add(
        0, batch, values.double()
    )
    ordinary = torch.zeros(3, 2).index_add(0, batch, values)
    torch.testing.assert_close(actual.double(), expected)
    assert actual[0, 0] == 1
    assert ordinary[0, 0] == 0


@pytest.mark.parametrize("shape", [(0,), (0, 3), (0, 3, 3)])
@pytest.mark.parametrize("systems", [0, 2])
def test_compensated_empty_shapes_and_gradients(shape, systems):
    values = torch.empty(shape, requires_grad=True)
    result = compensated_node_sum(values, torch.empty(0, dtype=torch.long), systems)
    assert result.shape == (systems, *shape[1:])
    assert result.dtype == torch.float32
    assert torch.count_nonzero(result) == 0
    torch.testing.assert_close(torch.autograd.grad(result.sum(), values)[0], values)


def test_compensated_reduction_retains_forces_and_hessian():
    positions = torch.tensor(
        [[0.3, 0.7, -0.2], [0.1, -0.5, 0.8], [0.9, 0.2, 0.4]], requires_grad=True
    )
    batch = torch.tensor([0, 1, 0])

    def compute(approximate):
        with approximate_ops_context(
            SimpleNamespace(compensated_reduction=approximate)
        ):
            output, unreduced = reduce_node_to_system(
                positions.pow(4).sum(-1), batch, 3
            )
        assert output.dtype == unreduced.dtype == torch.float64
        first = torch.autograd.grad(output.sum(), positions, create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), positions)[0]
        return output, first, second

    for actual, expected in zip(compute(True), compute(False)):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


def test_reduction_context_nesting_exception_and_threads_do_not_contaminate_jobs():
    enabled = SimpleNamespace(compensated_reduction=True)
    disabled = SimpleNamespace(compensated_reduction=False)
    assert not compensated_reduction_enabled()
    with approximate_ops_context(enabled):
        assert compensated_reduction_enabled()
        with (
            pytest.raises(RuntimeError, match="test"),
            approximate_ops_context(disabled),
        ):
            assert not compensated_reduction_enabled()
            raise RuntimeError("test")
        assert compensated_reduction_enabled()
    assert not compensated_reduction_enabled()
    barrier = Barrier(2)

    def check(settings):
        with approximate_ops_context(settings):
            barrier.wait(timeout=5)
            return compensated_reduction_enabled()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(check, [enabled, disabled]))
    assert results == [True, False]
    assert not compensated_reduction_enabled()


def test_default_reduction_still_compiles_without_contextvar_graph_breaks():
    compiled = torch.compile(reduce_node_to_system, backend="eager", fullgraph=True)
    values, batch = (
        torch.arange(6, dtype=torch.float32),
        torch.tensor([0, 0, 1, 1, 1, 1]),
    )
    for actual, expected in zip(
        compiled(values, batch, 2), reduce_node_to_system(values, batch, 2)
    ):
        torch.testing.assert_close(actual, expected)


def test_local_fast_exp_value_and_first_two_derivative_error_bounds():
    values = torch.linspace(-8, 0, 4001, dtype=torch.float64, requires_grad=True)
    outputs = _derivatives(local_fast_exp, values)
    exact = values.detach().exp()
    # Measured scalar-function bounds; these are not UMA E/F/S guarantees.
    for actual in outputs:
        assert ((actual.detach() - exact).abs() / exact).max() < 0.002
    outside = torch.tensor(
        [-30.0, -9.0, 0.5, 2.0], dtype=torch.float64, requires_grad=True
    )
    for actual in _derivatives(local_fast_exp, outside):
        torch.testing.assert_close(actual, outside.exp(), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("boundary", [-8.0, -6.0, 0.0])
def test_local_fast_exp_is_c2_at_blend_and_fallback_boundaries(boundary):
    values = torch.tensor(
        [boundary - 1e-7, boundary, boundary + 1e-7],
        dtype=torch.float64,
        requires_grad=True,
    )
    for actual in _derivatives(local_fast_exp, values):
        torch.testing.assert_close(
            actual[[0, 2]], actual[1].expand(2), rtol=3e-6, atol=1e-10
        )


@pytest.mark.parametrize("points", [17, 257])
def test_quintic_table_matches_value_slope_curvature_at_every_node(points):
    original = GaussianSmearing(0, 5, 8).double()
    table = ApproximateGaussian(original, points=points)
    normalized = torch.linspace(-8, 8, points, dtype=torch.float64)
    distances = (normalized / math.sqrt(-original.coeff)).requires_grad_()
    for actual, expected in zip(
        _derivatives(lambda x: table(x)[:, 0], distances),
        _derivatives(lambda x: original(x)[:, 0], distances),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-12)
    # In particular, the center must retain its nonzero second derivative.
    center = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    assert _derivatives(lambda x: table(x)[:, 0], center)[2].abs().item() > 0


def test_quintic_table_error_bounds_and_exact_outside_fallback():
    original = GaussianSmearing(0, 5, 8).double()
    table = ApproximateGaussian(original, points=257)
    distances = torch.linspace(-10, 15, 4001, dtype=torch.float64, requires_grad=True)
    actual = _derivatives(table, distances)
    expected = _derivatives(original, distances)
    for candidate, reference, tolerance in zip(actual, expected, [2e-9, 2e-7, 5e-5]):
        assert (candidate - reference).abs().max() < tolerance
    scale = math.sqrt(-original.coeff)
    outside = torch.tensor(
        [-9 / scale, 5 + 9 / scale], dtype=torch.float64, requires_grad=True
    )
    for candidate, reference in zip(
        _derivatives(table, outside), _derivatives(original, outside)
    ):
        torch.testing.assert_close(candidate, reference, rtol=1e-12, atol=1e-200)


def test_quintic_table_is_c2_across_internal_nodes_and_fallback_boundaries():
    original = GaussianSmearing(0, 5, 8).double()
    table = ApproximateGaussian(original, points=17)
    knots = torch.tensor([-8.0, -4.0, 0.0, 4.0, 8.0], dtype=torch.float64)
    normalized = knots[:, None] + torch.tensor([-1e-10, 0, 1e-10], dtype=torch.float64)
    distances = (normalized.flatten() / math.sqrt(-original.coeff)).requires_grad_()
    for derivative in _derivatives(lambda x: table(x)[:, 0], distances):
        values = derivative.view(-1, 3)
        torch.testing.assert_close(
            values[:, [0, 2]], values[:, 1:2].expand(-1, 2), rtol=3e-6, atol=1e-9
        )


@pytest.mark.parametrize("options", [{"points": 257}, {"fast_exp": True}])
def test_radial_approximations_preserve_gradgrad_and_empty_batches(options):
    approximation = ApproximateGaussian(GaussianSmearing(0, 3, 4).double(), **options)
    values = torch.tensor([0.37, 1.29, 2.31], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(approximation, (values,), fast_mode=True)
    assert torch.autograd.gradgradcheck(approximation, (values,), fast_mode=True)
    empty = torch.empty(0, dtype=torch.float64, requires_grad=True)
    result = approximation(empty)
    assert result.shape == (0, 4)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), empty)[0], empty)


def test_application_is_explicit_json_safe_and_disabled_is_unchanged():
    model = torch.nn.Module()
    model.distance_expansion = GaussianSmearing(0, 5, 8)
    original = model.distance_expansion
    assert apply_feature_approximations(model, SimpleNamespace()) == {}
    assert model.distance_expansion is original
    settings = SimpleNamespace(radial_table_points=257, compensated_reduction=True)
    manifest = apply_feature_approximations(model, settings)
    json.dumps(manifest)
    assert isinstance(model.distance_expansion, ApproximateGaussian)
    assert model.distance_expansion.original is original
    assert apply_feature_approximations(model, settings) == manifest
    assert model.distance_expansion.original is original
    assert not model.state_dict()
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_feature_approximations(
            model, SimpleNamespace(radial_table_points=17, radial_fast_exp=True)
        )
    assert apply_feature_approximations(model, SimpleNamespace()) == {}
    assert model.distance_expansion is original


@pytest.mark.parametrize("points", [0, 1, True, 2.5])
def test_invalid_table_resolution_is_rejected(points):
    with pytest.raises(ValueError, match="integer >= 2"):
        ApproximateGaussian(GaussianSmearing(0, 5, 8), points=points)

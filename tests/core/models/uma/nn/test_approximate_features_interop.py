"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
from types import MethodType, SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.nn.approximate_ops import (
    ApproximateGaussian,
    apply_feature_approximations,
)
from fairchem.core.models.uma.nn.approximate_saved_features import (
    _pack_feature,
    _unpack_feature,
)
from fairchem.core.models.uma.nn.radial import GaussianSmearing


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_saved_features_scale_finite_entries_independently_of_nonfinite(
    dtype, nonfinite
):
    limit = torch.finfo(torch.float32).max
    source = torch.tensor([limit, -limit, nonfinite], requires_grad=True)
    restored = _unpack_feature(_pack_feature(source, dtype, 1))
    assert torch.isfinite(restored[:2]).all()
    torch.testing.assert_close(restored, source, rtol=0.004, atol=0, equal_nan=True)


def test_radial_table_preparation_ignores_default_device_and_preserves_derivatives():
    original = GaussianSmearing(0, 5, 8).double().eval()
    values = torch.tensor([0.37, 1.29, 2.31], dtype=torch.float64, requires_grad=True)
    # A meta default exposes accidental factory allocations without GPU hardware.
    with torch.device("meta"):
        table = ApproximateGaussian(original, points=257)
        actual = table(values)
        gradient = torch.autograd.grad(actual.sum(), values, create_graph=True)[0]
        hessian = torch.autograd.grad(gradient.sum(), values)[0]
    reference = original(values)
    expected_gradient = torch.autograd.grad(reference.sum(), values, create_graph=True)[
        0
    ]
    expected_hessian = torch.autograd.grad(expected_gradient.sum(), values)[0]
    assert table.table_coefficients.device.type == "cpu"
    assert table.table_coefficients.dtype == torch.float64
    assert table.training is False
    for candidate, expected, tolerance in zip(
        [actual, gradient, hessian],
        [reference, expected_gradient, expected_hessian],
        [2e-9, 2e-7, 5e-5],
    ):
        torch.testing.assert_close(candidate, expected, rtol=0, atol=tolerance)


class _CustomGaussian(GaussianSmearing):
    def forward(self, values):
        return super().forward(values) + 2


def _custom_forward(self, values):
    return GaussianSmearing.forward(self, values) + 2


@pytest.mark.parametrize(
    "customization", ["subclass", "forward", "pre", "post", "backward", "backward_pre"]
)
@pytest.mark.parametrize(
    "options", [{"radial_table_points": 257}, {"radial_fast_exp": True}]
)
def test_radial_replacement_rejects_custom_semantics_without_mutation(
    customization, options
):
    radial = (_CustomGaussian if customization == "subclass" else GaussianSmearing)(
        0, 5, 8
    )
    if customization == "forward":
        radial.forward = MethodType(_custom_forward, radial)
    elif customization == "pre":
        radial.register_forward_pre_hook(lambda module, args: (args[0] + 1,))
    elif customization == "post":
        radial.register_forward_hook(lambda module, args, output: output + 2)
    elif customization == "backward":
        radial.register_full_backward_hook(lambda module, inputs, outputs: None)
    elif customization == "backward_pre":
        radial.register_full_backward_pre_hook(lambda module, outputs: None)
    model = torch.nn.Module()
    model.distance_expansion = radial
    values = torch.tensor([0.3], requires_grad=True)
    expected = radial(values)
    with pytest.raises(ValueError, match="customized|hooks"):
        apply_feature_approximations(model, SimpleNamespace(**options))
    assert model.distance_expansion is radial
    torch.testing.assert_close(model.distance_expansion(values), expected)


def test_reapplying_radial_policy_preserves_identity_and_hooks():
    model = torch.nn.Module()
    model.distance_expansion = GaussianSmearing(0, 5, 8).eval()
    settings = SimpleNamespace(radial_table_points=257)
    apply_feature_approximations(model, settings)
    radial = model.distance_expansion
    values = torch.tensor([0.3])
    expected = radial(values) + 2
    handle = radial.register_forward_hook(lambda module, args, output: output + 2)
    apply_feature_approximations(model, settings)
    assert model.distance_expansion is radial
    torch.testing.assert_close(radial(values), expected)
    for policy in [
        SimpleNamespace(radial_table_points=257.0),
        SimpleNamespace(radial_table_points=257, radial_fast_exp=0),
    ]:
        with pytest.raises(ValueError, match="integer|boolean"):
            apply_feature_approximations(model, policy)
        assert model.distance_expansion is radial
    for policy in [SimpleNamespace(), SimpleNamespace(radial_fast_exp=True)]:
        with pytest.raises(ValueError, match="hooks"):
            apply_feature_approximations(model, policy)
        assert model.distance_expansion is radial
    handle.remove()
    apply_feature_approximations(model, SimpleNamespace())
    assert model.distance_expansion is radial.original


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda x: pickle.loads(pickle.dumps(x))]
)
@pytest.mark.parametrize("options", [{"points": 257}, {"fast_exp": True}])
def test_radial_restoration_keeps_buffer_ownership_and_live_gaussian_parameters(
    restore, options
):
    radial = ApproximateGaussian(GaussianSmearing(0, 5, 8).double(), **options)
    restored = restore(radial)
    values = torch.tensor([0.37, 1.29, 2.31], dtype=torch.float64)
    original_values = radial(values)
    torch.testing.assert_close(restored(values), original_values)
    assert restored.offset.data_ptr() != radial.offset.data_ptr()
    if restored.points is not None:
        assert (
            restored.table_coefficients.data_ptr()
            != radial.table_coefficients.data_ptr()
        )
    restored.original.offset.add_(0.4)
    restored.original.coeff *= 0.7
    torch.testing.assert_close(radial(values), original_values)
    expected = ApproximateGaussian(copy.deepcopy(restored.original), **options)
    torch.testing.assert_close(restored(values), expected(values))
    assert not restored.state_dict()

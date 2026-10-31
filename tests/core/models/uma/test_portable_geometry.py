"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.common.quaternion.quaternion_wigner_utils import (
    create_wigner_data_module,
)
from fairchem.core.models.uma.common.quaternion.wigner_d_hybrid import (
    axis_angle_wigner_hybrid,
)
from fairchem.core.models.uma.nn.portable_geometry import PortableGeometry
from fairchem.core.models.uma.nn.radial import GaussianSmearing, PolynomialEnvelope
from fairchem.core.models.uma.nn.xpu_wigner_benchmark import benchmark


def _model(dtype):
    return SimpleNamespace(
        wigner_data=create_wigner_data_module(lmax=2, lmin=5).to(dtype=dtype),
        distance_expansion=GaussianSmearing(0, 5, 8).to(dtype=dtype),
        envelope=PolynomialEnvelope(5),
        cutoff=5.0,
    )


def _reference(model, vectors, gamma):
    distance = vectors.norm(dim=-1)
    w, inv = axis_angle_wigner_hybrid(
        vectors,
        2,
        gamma=gamma,
        coeffs=model.wigner_data.coeffs,
        U_blocks=model.wigner_data.U_blocks,
        custom_kernels=model.wigner_data.custom_kernels,
        compact_l2=True,
    )
    return (
        model.distance_expansion(distance),
        model.envelope(distance / model.cutoff)[:, None],
        w,
        inv,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("empty", [False, True])
def test_geometry_matches_existing_formula_at_axes_and_cutoff(dtype, empty):
    model = _model(dtype)
    geometry = PortableGeometry(model)
    vectors = torch.tensor(
        [
            [0, 1, 0],
            [0, -1, 0],
            [0.1, 1, -0.2],
            [0.2, -1, 0.3],
            [3.0, 0, 4.0],
            [5.1, 0, 0],
        ],
        dtype=dtype,
    )
    if empty:
        vectors = vectors[:0]
    gamma = torch.linspace(0, 6, len(vectors), dtype=dtype)
    for actual, expected in zip(
        geometry(vectors, gamma), _reference(model, vectors, gamma)
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert geometry.quartic_coefficients.shape == (25, 35)
    assert not geometry.state_dict()


def test_joint_geometry_first_and_second_derivatives_match_original():
    model = _model(torch.double)
    geometry = PortableGeometry(model)
    torch.manual_seed(17)
    vectors = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
    gamma = torch.randn(3, dtype=torch.double, requires_grad=True)
    inputs = (vectors, gamma)

    def derivatives(function):
        outputs = function(*inputs)
        loss = sum(value.sin().square().sum() for value in outputs)
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        second = torch.autograd.grad(sum(value.sum() for value in first), inputs)
        return (*outputs, *first, *second)

    for a, b in zip(
        derivatives(geometry), derivatives(lambda v, g: _reference(model, v, g))
    ):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(geometry, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(geometry, inputs, fast_mode=True)


def test_fp64_analytic_coefficients_follow_fp32_network_geometry_dtype():
    model = _model(torch.float32)
    model.wigner_data.double()
    geometry = PortableGeometry(model)
    vectors = torch.tensor([[0.2, 0.8, -0.4], [1.0, 0.2, 0.5]], requires_grad=True)
    gamma = torch.tensor([0.5, 1.0])
    assert geometry.quartic_coefficients.dtype == torch.float32
    assert geometry.quartic_coefficients.device == vectors.device
    for actual, expected in zip(
        geometry(vectors, gamma), _reference(model, vectors, gamma)
    ):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_joint_geometry_compiles_and_handles_multiple_shapes():
    model = _model(torch.float32)
    geometry = PortableGeometry(model)
    geometry = torch.compile(geometry, backend="eager", fullgraph=True, dynamic=True)
    for count in (2, 5):
        vectors, gamma = torch.randn(count, 3), torch.randn(count)
        for a, b in zip(geometry(vectors, gamma), _reference(model, vectors, gamma)):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


def test_kernel_benchmark_rejects_missing_hardware_without_fabricated_results():
    with pytest.raises(ValueError, match="available Intel XPU"):
        benchmark(device="cpu")


@pytest.mark.parametrize("condition", ["tf32", "autocast", "hidden"])
def test_kernel_benchmark_rejects_unsafe_precision_or_shape_before_import(
    monkeypatch, condition
):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.backends, "mkldnn", SimpleNamespace(allow_tf32=condition == "tf32")
    )
    monkeypatch.setattr(
        torch, "is_autocast_enabled", lambda device="cuda": condition == "autocast"
    )
    message = {
        "tf32": "allow_tf32=False",
        "autocast": "autocast",
        "hidden": "hidden <= 512",
    }[condition]
    with pytest.raises(ValueError, match=message):
        benchmark(hidden=513 if condition == "hidden" else 128)
    assert torch.backends.mkldnn.allow_tf32 == (condition == "tf32")

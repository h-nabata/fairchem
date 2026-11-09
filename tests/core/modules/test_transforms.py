"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.models.uma.escn_md import compose_tensor
from fairchem.core.modules.transforms import decompose_tensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_tensor_decomposition_preserves_input_dtype_and_device(dtype, device):
    # A foreign process default must not choose the coefficient precision.
    original_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32 if dtype == torch.float64 else torch.float64)
    stress = torch.tensor(
        [[1.0 + 2**-30, 0.25, -0.5], [0.25, 2.0, 0.75], [-0.5, 0.75, 3.0]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    data = {"stress": stress}
    config = {
        "tensor": "stress",
        "rank": 2,
        "decomposition": {
            "stress_isotropic": {"irrep_dim": 0},
            "stress_antisymmetric": {"irrep_dim": 1},
            "stress_anisotropic": {"irrep_dim": 2},
        },
    }
    try:
        assert decompose_tensor(data, config) is data
    finally:
        torch.set_default_dtype(original_default)
    assert data["stress"] is stress
    for name, width in (
        ("stress_isotropic", 1),
        ("stress_antisymmetric", 3),
        ("stress_anisotropic", 5),
    ):
        assert data[name].dtype == dtype
        assert data[name].device == stress.device
        assert data[name].shape == (1, width)
    if device == "meta":
        return

    reconstructed = compose_tensor(
        data["stress_isotropic"], data["stress_anisotropic"]
    ).reshape(3, 3)
    tolerance = 8 * torch.finfo(dtype).eps
    torch.testing.assert_close(reconstructed, stress, rtol=tolerance, atol=tolerance)
    first = torch.autograd.grad(
        reconstructed.square().sum(), stress, create_graph=True
    )[0]
    second = torch.autograd.grad(first.sum(), stress)[0]
    torch.testing.assert_close(first, 2 * stress, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        second, torch.full_like(stress, 2), rtol=tolerance, atol=tolerance
    )

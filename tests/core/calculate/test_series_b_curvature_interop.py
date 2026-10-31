"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.calculate.series_b_curvature import (
    HutchinsonOptions,
    LanczosOptions,
    hutchinson_estimates,
    lanczos_modes,
)


@pytest.mark.parametrize(
    "dtype,scale",
    [
        (torch.float32, 1e30),
        (torch.float32, 1e-30),
        (torch.float64, 1e200),
        (torch.float64, 1e-200),
    ],
)
def test_finite_operator_scale_does_not_overflow_or_erase_residuals(dtype, scale):
    diagonal = torch.tensor([-2.0, 1.0, 3.0], dtype=dtype) * scale
    result = lanczos_modes(
        lambda vector: diagonal * vector,
        torch.zeros_like(diagonal),
        options=LanczosOptions(
            num_modes=3,
            max_iterations=3,
            absolute_tolerance=0,
            relative_tolerance=1e-5,
        ),
    )
    torch.testing.assert_close(result.eigenvalues / scale, diagonal / scale)
    assert torch.isfinite(result.residual_norms).all()
    assert result.converged.all()
    # Measure in scaled coordinates so this independent check does not itself
    # square numbers that overflow/underflow the working dtype.
    expected = torch.linalg.vector_norm(
        (diagonal / scale)[:, None] * result.eigenvectors
        - (result.eigenvalues / scale) * result.eigenvectors,
        dim=0,
    )
    torch.testing.assert_close(
        result.residual_norms / scale, expected, atol=1e-6, rtol=1e-5
    )


def test_small_operator_does_not_report_a_truncated_subspace_as_converged():
    diagonal = torch.tensor([-2.0, 1.0, 3.0]) * 1e-30
    result = lanczos_modes(
        lambda vector: diagonal * vector,
        torch.zeros_like(diagonal),
        options=LanczosOptions(
            max_iterations=1, absolute_tolerance=0, relative_tolerance=1e-5
        ),
    )
    assert result.residual_norms.item() > 1e-31
    assert not result.converged.any()


@pytest.mark.parametrize("overflow", ["trace", "variance"])
def test_finite_products_cannot_return_nonfinite_hutchinson_statistics(overflow):
    if overflow == "trace":
        diagonal = torch.full((3,), 8e307, dtype=torch.float64)

        def operator(vector):
            return diagonal * vector

    else:
        # Each product and trace is finite, but the sample variance is not
        # representable in FP64. This must be reported, not returned as inf.
        matrix = torch.tensor([[0.0, 1e200], [1e200, 0.0]], dtype=torch.float64)

        def operator(vector):
            return matrix @ vector

        diagonal = matrix[0]
    with pytest.raises(ValueError, match="statistics overflowed.*rescale"):
        hutchinson_estimates(
            operator,
            torch.zeros_like(diagonal),
            options=HutchinsonOptions(samples=16),
        )


def test_zero_operator_retains_zero_eigenvalues_and_residuals():
    result = lanczos_modes(lambda vector: vector * 0, torch.zeros(3))
    assert result.eigenvalues.item() == 0
    assert result.residual_norms.item() == 0
    assert result.converged.all()

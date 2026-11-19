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

from fairchem.core.calculate import series_b_curvature
from fairchem.core.calculate.series_b_curvature import (
    HutchinsonOptions,
    LanczosOptions,
    hutchinson_estimates,
    lanczos_modes,
)


def test_lanczos_releases_product_copies_before_the_projected_eigensolve(monkeypatch):
    references = []
    original_product = series_b_curvature._product
    original_eigh = torch.linalg.eigh

    def product(*args):
        result = original_product(*args)
        references.append(weakref.ref(result))
        return result

    def eigh(matrix):
        assert len(references) == 4
        assert all(reference() is None for reference in references)
        return original_eigh(matrix)

    monkeypatch.setattr(series_b_curvature, "_product", product)
    monkeypatch.setattr(torch.linalg, "eigh", eigh)
    diagonal = torch.tensor([-2.0, 1.0, 3.0, 5.0], dtype=torch.float64)
    result = lanczos_modes(
        lambda vector: diagonal * vector,
        torch.zeros_like(diagonal),
        options=LanczosOptions(num_modes=4, max_iterations=4),
    )
    torch.testing.assert_close(result.eigenvalues, diagonal)
    assert result.converged.all()


@pytest.mark.parametrize("workflow", ["lanczos", "hutchinson"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_retained_curvature_failure_releases_private_vectors(workflow, error_type):
    references = []
    template = torch.zeros(8, dtype=torch.float64)
    diagonal = torch.arange(1.0, 9.0, dtype=template.dtype)
    rng = torch.get_rng_state().clone()

    def operator(vector):
        references.append(weakref.ref(vector))
        if len(references) == 3:
            raise error_type("HVP interrupted")
        return diagonal * vector

    function = lanczos_modes if workflow == "lanczos" else hutchinson_estimates
    with pytest.raises(error_type, match="HVP interrupted") as recorded:
        function(operator, template)
    gc.collect()
    assert recorded.value is not None
    assert all(reference() is None for reference in references)
    torch.testing.assert_close(template, torch.zeros_like(template))
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lanczos_releases_basis_matrix_before_residual_products(monkeypatch, dtype):
    matrices = []
    empty = torch.Tensor.new_empty
    empty_like = torch.empty_like
    calls = 0

    def track_empty(tensor, size, **kwargs):
        result = empty(tensor, size, **kwargs)
        if size == (4, 4):
            matrices.append(weakref.ref(result))
        return result

    def track_like(tensor, **kwargs):
        result = empty_like(tensor, **kwargs)
        if tensor.shape == (4, 4):
            matrices.append(weakref.ref(result))
        return result

    diagonal = torch.tensor([-2.0, 1.0, 3.0, 5.0], dtype=dtype)

    def product(vector):
        nonlocal calls
        calls += 1
        if calls > 4:
            assert len(matrices) == 2
            assert all(reference() is None for reference in matrices)
        return diagonal * vector

    monkeypatch.setattr(torch.Tensor, "new_empty", track_empty)
    monkeypatch.setattr(torch, "empty_like", track_like)
    result = lanczos_modes(
        product,
        torch.zeros_like(diagonal),
        options=LanczosOptions(num_modes=2, max_iterations=4),
    )
    torch.testing.assert_close(result.eigenvalues, diagonal[:2])
    assert result.converged.all()


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

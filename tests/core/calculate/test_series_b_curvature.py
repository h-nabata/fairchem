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
    PredictorHVP,
    hutchinson_estimates,
    lanczos_modes,
)
from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


def _matrix(dtype=torch.float64):
    generator = torch.Generator().manual_seed(19)
    rotation, _ = torch.linalg.qr(torch.randn(8, 8, dtype=dtype, generator=generator))
    eigenvalues = torch.tensor([-9, -2, -2, 0, 1, 3, 8, 15], dtype=dtype)
    return (rotation * eigenvalues) @ rotation.T, eigenvalues


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lanczos_lowest_algebraic_modes_match_indefinite_dense_matrix(dtype):
    matrix, eigenvalues = _matrix(dtype)
    result = lanczos_modes(
        lambda vector: matrix @ vector,
        torch.zeros(8, dtype=dtype),
        options=LanczosOptions(num_modes=4, max_iterations=8, absolute_tolerance=2e-5),
    )
    torch.testing.assert_close(
        result.eigenvalues, eigenvalues[:4], atol=2e-5, rtol=2e-5
    )
    torch.testing.assert_close(
        result.eigenvectors.T @ result.eigenvectors,
        torch.eye(4, dtype=dtype),
        atol=2e-6,
        rtol=2e-6,
    )
    residuals = torch.linalg.vector_norm(
        matrix @ result.eigenvectors - result.eigenvalues * result.eigenvectors, dim=0
    )
    torch.testing.assert_close(result.residual_norms, residuals, atol=2e-6, rtol=2e-6)
    assert result.converged.all()
    assert result.iterations == 8
    assert result.hvp_evaluations == 12


def test_lanczos_exact_breakdown_retains_all_degenerate_modes():
    result = lanczos_modes(
        lambda vector: 2 * vector,
        torch.zeros(6, dtype=torch.float64),
        options=LanczosOptions(num_modes=6, max_iterations=6),
    )
    torch.testing.assert_close(
        result.eigenvalues, torch.full((6,), 2.0, dtype=torch.float64)
    )
    torch.testing.assert_close(
        result.eigenvectors.T @ result.eigenvectors, torch.eye(6, dtype=torch.float64)
    )
    assert result.converged.all()


def test_lanczos_reports_nonconvergence_with_insufficient_subspace():
    diagonal = torch.linspace(-7, 50, 30, dtype=torch.float64)
    result = lanczos_modes(
        lambda vector: diagonal * vector,
        torch.zeros_like(diagonal),
        options=LanczosOptions(
            num_modes=1,
            max_iterations=2,
            absolute_tolerance=1e-10,
            relative_tolerance=0,
        ),
    )
    assert not result.converged.any()
    assert result.residual_norms.item() > 1e-3


def test_lanczos_rejects_nonsymmetric_projected_operator():
    matrix = torch.tensor([[1.0, 4.0], [-2.0, 3.0]], dtype=torch.float64)
    for scale in (1, 1e-9):
        with pytest.raises(ValueError, match="not symmetric"):
            lanczos_modes(
                lambda x, scale=scale: scale * (matrix @ x),
                torch.zeros(2, dtype=torch.float64),
            )


def test_lanczos_breakdown_is_relative_to_operator_scale():
    matrix, _ = _matrix(torch.float32)
    template = torch.zeros(8)
    options = LanczosOptions(max_iterations=4)
    reference = lanczos_modes(lambda x: matrix @ x, template, options=options)
    scaled = lanczos_modes(lambda x: 1e-9 * (matrix @ x), template, options=options)
    torch.testing.assert_close(scaled.eigenvalues * 1e9, reference.eigenvalues)


def test_hutchinson_indefinite_diagonal_and_zero_trace_have_zero_variance():
    diagonal = torch.tensor([-5.0, -1.0, 0.0, 1.0, 2.0, 3.0]).reshape(2, 3)
    result = hutchinson_estimates(
        lambda vector: diagonal * vector,
        torch.zeros_like(diagonal),
        options=HutchinsonOptions(samples=8, seed=5),
    )
    torch.testing.assert_close(result.diagonal, diagonal.double())
    assert result.trace.item() == 0
    assert result.trace_standard_error.item() == 0
    assert not torch.count_nonzero(result.diagonal_standard_error)
    assert result.samples == result.hvp_evaluations == 8


def test_hutchinson_variance_and_standard_error_match_explicit_probe_samples():
    matrix, _ = _matrix()
    samples, seed = 50, 19
    generator = torch.Generator().manual_seed(seed)
    probes = torch.stack(
        [2 * torch.randint(2, (8,), generator=generator) - 1 for _ in range(samples)]
    ).double()
    draws = probes * (probes @ matrix.T)
    result = hutchinson_estimates(
        lambda vector: matrix @ vector,
        torch.zeros(8, dtype=torch.float64),
        options=HutchinsonOptions(samples=samples, seed=seed),
    )
    torch.testing.assert_close(result.diagonal, draws.mean(0))
    torch.testing.assert_close(result.diagonal_sample_variance, draws.var(0))
    torch.testing.assert_close(result.trace_sample_variance, draws.sum(1).var())
    torch.testing.assert_close(
        result.trace_standard_error, draws.sum(1).std() / samples**0.5
    )
    torch.testing.assert_close(
        result.diagonal_standard_error, draws.std(0) / samples**0.5
    )


def test_private_random_generators_preserve_global_rng_and_reproduce_results():
    before = torch.get_rng_state().clone()
    matrix, _ = _matrix()
    template = torch.zeros(8, dtype=torch.float64)
    for algorithm in (lanczos_modes, hutchinson_estimates):
        first = algorithm(lambda x: matrix @ x, template)
        second = algorithm(lambda x: matrix @ x, template)
        name = "eigenvalues" if algorithm is lanczos_modes else "diagonal"
        torch.testing.assert_close(
            getattr(first, name), getattr(second, name), rtol=0, atol=0
        )
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_curvature_ignores_an_unrelated_default_device():
    diagonal = torch.tensor([-3.0, -1.0, 2.0, 7.0], dtype=torch.float64, device="cpu")
    template = torch.zeros_like(diagonal, device="cpu")
    with torch.device("meta"):
        modes = lanczos_modes(lambda x: diagonal * x, template)
        estimates = hutchinson_estimates(lambda x: diagonal * x, template)
    assert modes.eigenvalues.device.type == "cpu"
    assert modes.eigenvalues.item() == pytest.approx(-3)
    assert estimates.diagonal.device.type == "cpu"
    torch.testing.assert_close(estimates.diagonal, diagonal)


def test_operator_input_mutation_and_shared_output_do_not_corrupt_basis_or_probes():
    matrix, _ = _matrix()
    buffer = torch.empty(8, dtype=torch.float64)

    def shared(vector):
        buffer.copy_(matrix @ vector)
        vector.zero_()
        return buffer

    for algorithm in (lanczos_modes, hutchinson_estimates):
        expected = algorithm(lambda x: matrix @ x, torch.zeros_like(buffer))
        actual = algorithm(shared, torch.zeros_like(buffer))
        name = "eigenvalues" if algorithm is lanczos_modes else "diagonal"
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("algorithm", [lanczos_modes, hutchinson_estimates])
def test_curvature_rejects_bad_operator_outputs_and_invalid_templates(algorithm):
    template = torch.zeros(3)
    for operator in (
        lambda x: x.double(),
        lambda x: x[:, None],
        lambda x: x * float("nan"),
    ):
        with pytest.raises(ValueError, match="HVP output"):
            algorithm(operator, template)
    with pytest.raises(ValueError, match="nonempty"):
        algorithm(lambda x: x, torch.empty(0))
    with pytest.raises(ValueError, match="float32 or float64"):
        algorithm(lambda x: x, template.half())
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        algorithm(lambda x: x, template)


def test_curvature_options_reject_ambiguous_and_nonfinite_values():
    for cls, kwargs in (
        (LanczosOptions, {"num_modes": True}),
        (LanczosOptions, {"max_iterations": 0}),
        (LanczosOptions, {"absolute_tolerance": float("nan")}),
        (LanczosOptions, {"relative_tolerance": -1}),
        (LanczosOptions, {"symmetry_tolerance": False}),
        (HutchinsonOptions, {"samples": 1}),
        (HutchinsonOptions, {"samples": 1.5}),
        (HutchinsonOptions, {"seed": -1}),
        (HutchinsonOptions, {"seed": 2**63}),
    ):
        with pytest.raises(ValueError):
            cls(**kwargs)
    with pytest.raises(ValueError, match="num_modes"):
        lanczos_modes(lambda x: x, torch.zeros(2), options=LanczosOptions(num_modes=3))


def test_actual_uma_adapter_snapshot_modes_and_estimates_match_dense_hessian():
    predictor = _public_predictor()
    data = _data()
    before = data.clone()
    expected = predictor.predict(data, requested_properties=("hessian",))["hessian"][0]
    adapter = PredictorHVP(predictor, data)
    vector = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 9
    # A caller's later trajectory update cannot change this HVP's geometry.
    data.pos.add_(4)
    data["hessian_vector"] = vector * 100
    torch.testing.assert_close(
        adapter(vector).flatten(), expected @ vector.flatten(), atol=1e-7, rtol=2e-4
    )
    assert "hessian_vector" not in before
    assert not before.pos.requires_grad
    modes = lanczos_modes(
        adapter,
        options=LanczosOptions(num_modes=3, max_iterations=9, absolute_tolerance=1e-7),
    )
    torch.testing.assert_close(
        modes.eigenvalues, torch.linalg.eigvalsh(expected)[:3], atol=1e-7, rtol=1e-3
    )
    assert modes.converged.all()
    options = HutchinsonOptions(samples=6)
    estimate = hutchinson_estimates(adapter, options=options)
    dense_estimate = hutchinson_estimates(
        lambda x: (expected @ x.flatten()).reshape(x.shape),
        adapter.template,
        options=options,
    )
    torch.testing.assert_close(
        estimate.diagonal, dense_estimate.diagonal, atol=1e-7, rtol=1e-3
    )
    torch.testing.assert_close(
        estimate.trace, dense_estimate.trace, atol=1e-7, rtol=1e-3
    )
    with pytest.raises(ValueError, match="template"):
        adapter(vector.double())
    with pytest.raises(ValueError, match="one nonempty"):
        PredictorHVP(predictor, atomicdata_list_to_batch([before, before]))

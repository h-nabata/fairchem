"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _tolerance(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class LanczosOptions:
    """
    Explicit B13 work and residual limits; no inference setting enables this.
    """

    num_modes: int = 1
    max_iterations: int = 64
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 1e-5
    symmetry_tolerance: float = 1e-4
    seed: int = 41

    def __post_init__(self):
        for name in ("num_modes", "max_iterations", "seed"):
            object.__setattr__(
                self,
                name,
                _integer(name, getattr(self, name), 0 if name == "seed" else 1),
            )
        if self.seed >= 2**63:
            raise ValueError("seed must be smaller than 2**63")
        for name in (
            "absolute_tolerance",
            "relative_tolerance",
            "symmetry_tolerance",
        ):
            object.__setattr__(self, name, _tolerance(name, getattr(self, name)))


@dataclass(frozen=True)
class LanczosResult:
    """
    Ritz pairs sorted by algebraic eigenvalue, with explicitly measured residuals.

    Columns of ``eigenvectors`` have flattened coordinate order. ``converged``
    reports only the supplied residual criterion: a limited subspace cannot
    certify that an unobserved lower eigenmode does not exist. This is neither
    a complete negative-mode count nor a transition-state certificate. Modes
    are Cartesian, without mass weighting or projection of rigid motions.
    """

    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    residual_norms: torch.Tensor
    converged: torch.Tensor
    iterations: int
    hvp_evaluations: int
    seed: int


@dataclass(frozen=True)
class HutchinsonOptions:
    """
    Explicit B17 independent Rademacher probe count and private random seed.
    """

    samples: int = 64
    seed: int = 41

    def __post_init__(self):
        object.__setattr__(self, "samples", _integer("samples", self.samples, 2))
        object.__setattr__(self, "seed", _integer("seed", self.seed, 0))
        if self.seed >= 2**63:
            raise ValueError("seed must be smaller than 2**63")


@dataclass(frozen=True)
class HutchinsonResult:
    """
    Trace and coordinate diagonal estimates with sample variances and SEs.

    Statistics are accumulated in CPU FP64 using online updates; no collection
    of all probe products is retained. Standard errors are estimates, not
    guaranteed confidence intervals or relative-error bounds. Indefinite
    Hessians and zero/negative traces are permitted. These estimates do not
    identify eigenvalues, their signs, or a transition state.
    """

    trace: torch.Tensor
    diagonal: torch.Tensor
    trace_sample_variance: torch.Tensor
    diagonal_sample_variance: torch.Tensor
    trace_standard_error: torch.Tensor
    diagonal_standard_error: torch.Tensor
    samples: int
    seed: int
    hvp_evaluations: int


def _template(hvp, template):
    if template is None:
        template = getattr(hvp, "template", None)
    if not isinstance(template, torch.Tensor):
        raise TypeError("Provide a floating-point tensor template for the HVP shape")
    if template.dtype not in (torch.float32, torch.float64):
        raise ValueError("Curvature vectors require float32 or float64")
    if template.layout != torch.strided or template.numel() == 0:
        raise ValueError("Curvature vectors must be nonempty strided tensors")
    if not torch.isfinite(template).all():
        raise ValueError("Curvature template must be finite")
    # Shape/type are the template's only meaning. Do not keep a caller's graph.
    return torch.empty_like(
        template, device=template.device, memory_format=torch.contiguous_format
    )


def _product(hvp, vector, template):
    # An operator is allowed to retain or modify its input, or reuse an output
    # buffer. Neither action may corrupt the Krylov basis or a saved probe.
    argument = vector.reshape(template.shape).detach().clone()
    with torch.enable_grad(), torch.autocast(template.device.type, enabled=False):
        result = hvp(argument)
    if not isinstance(result, torch.Tensor) or result.shape != template.shape:
        raise ValueError("HVP output must be a tensor with the template shape")
    if result.dtype != template.dtype or result.device != template.device:
        raise ValueError("HVP output must preserve the template dtype and device")
    if not torch.isfinite(result).all():
        raise ValueError("HVP output contains nonfinite values")
    return result.detach().flatten().clone()


def _orthogonalize(vector, basis):
    # Two modified Gram-Schmidt passes avoid duplicate/ghost Ritz vectors,
    # especially for degenerate eigenvalues and near-breakdown in FP32.
    for _ in range(2):
        for q in basis:
            vector = vector - torch.dot(q, vector) * q
    return vector


def _scaled_norm(tensor, dim=None, keepdim=False):
    # Squaring finite FP32 coordinates can overflow (or underflow to zero)
    # even when their norm is representable. Keep the accelerator dtype;
    # a CPU FP64 copy of every Krylov vector would defeat matrix-free usage.
    scale = tensor.abs().amax(dim=dim, keepdim=True)
    divisor = torch.where(scale == 0, torch.ones_like(scale), scale)
    norm = torch.linalg.vector_norm(tensor / divisor, dim=dim, keepdim=True) * scale
    if not torch.isfinite(norm).all():
        raise ValueError("Curvature arithmetic overflowed; rescale the HVP operator")
    if keepdim:
        return norm
    return norm.squeeze() if dim is None else norm.squeeze(dim)


def _random_direction(size, template, generator, basis):
    for _ in range(4):
        vector = torch.randn(
            size, dtype=template.dtype, generator=generator, device="cpu"
        ).to(template.device)
        vector = _orthogonalize(vector, basis)
        norm = _scaled_norm(vector)
        if norm.item() > 32 * torch.finfo(template.dtype).eps:
            return vector / norm
    # Deterministic fallback when a nearly complete basis leaves a very small
    # complement. It also avoids interpreting a lucky random breakdown as a
    # proof that all requested degeneracies have been found.
    for index in range(size):
        vector = torch.zeros(size, dtype=template.dtype, device=template.device)
        vector[index] = 1
        vector = _orthogonalize(vector, basis)
        norm = _scaled_norm(vector)
        if norm.item() > 32 * torch.finfo(template.dtype).eps:
            return vector / norm
    raise RuntimeError("Unable to construct an independent Krylov direction")


def lanczos_modes(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    template: torch.Tensor | None = None,
    *,
    options: LanczosOptions | None = None,
) -> LanczosResult:
    """
    Approximate the lowest algebraic Hessian modes using matrix-free products.

    This fully reorthogonalized Lanczos/Rayleigh-Ritz implementation uses at
    most ``max_iterations`` basis vectors plus one fresh HVP per returned
    eigenpair to measure its actual residual. Exact Krylov breakdown starts
    an orthogonal block, retaining eigenvalue multiplicities. The full work
    budget is used; a converged Ritz pair alone does not justify prematurely
    abandoning unexplored directions. Storage is O(dimension * iterations),
    becoming quadratic if the caller explicitly requests a full-size basis.

    The callable must be a fixed, linear, symmetric HVP at one geometry.
    Projected asymmetry is rejected, but this check cannot prove that an
    arbitrary callable is linear or symmetric on unexplored directions.
    Returned residuals/convergence must be checked before using the modes.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Curvature workflows must run outside torch.inference_mode")
    if options is None:
        options = LanczosOptions()
    if not isinstance(options, LanczosOptions):
        raise TypeError("options must be LanczosOptions")
    reference = _template(hvp, template)
    dimension = reference.numel()
    iterations = min(dimension, options.max_iterations)
    if options.num_modes > iterations:
        raise ValueError("num_modes cannot exceed the dimension or max_iterations")
    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    basis, products = [], []
    q = _random_direction(dimension, reference, generator, basis)
    with torch.no_grad(), torch.autocast(reference.device.type, enabled=False):
        for index in range(iterations):
            basis.append(q)
            hq = _product(hvp, q, reference)
            products.append(hq)
            if index + 1 == iterations:
                break
            residual = _orthogonalize(hq, basis)
            norm = _scaled_norm(residual)
            scale = max(torch.finfo(reference.dtype).tiny, _scaled_norm(hq).item())
            if norm.item() <= 32 * torch.finfo(reference.dtype).eps * scale:
                q = _random_direction(dimension, reference, generator, basis)
            else:
                q = residual / norm
        q_matrix = torch.stack(basis, dim=1)
        hq_matrix = torch.stack(products, dim=1)
        # Only the small projected eigensolve runs on CPU FP64. There is no
        # requirement for an accelerator FP64 eigh kernel.
        projected = (q_matrix.T @ hq_matrix).to(device="cpu", dtype=torch.float64)
        if not torch.isfinite(projected).all():
            raise ValueError(
                "Curvature projection overflowed; rescale the HVP operator"
            )
        # Solve a scaled small matrix to protect both the symmetry test and
        # the eigensolver from squaring extreme, but finite, operator scales.
        projection_scale = projected.abs().amax()
        if projection_scale.item() != 0:
            projected = projected / projection_scale
        asymmetry = torch.linalg.matrix_norm(projected - projected.T).item()
        scale = max(
            torch.finfo(reference.dtype).tiny,
            torch.linalg.matrix_norm(projected).item(),
        )
        if asymmetry > options.symmetry_tolerance * scale:
            raise ValueError("HVP is not symmetric within symmetry_tolerance")
        eigenvalues, rotations = torch.linalg.eigh((projected + projected.T) * 0.5)
        values = (eigenvalues[: options.num_modes] * projection_scale).to(reference)
        vectors = q_matrix @ rotations[:, : options.num_modes].to(reference)
        vectors = vectors / _scaled_norm(vectors, dim=0, keepdim=True)
        if not torch.isfinite(values).all():
            raise ValueError(
                "Curvature eigenvalues overflowed; rescale the HVP operator"
            )
        residuals = torch.stack(
            [
                _scaled_norm(
                    _product(hvp, vectors[:, i], reference) - values[i] * vectors[:, i]
                )
                for i in range(options.num_modes)
            ]
        )
        converged = residuals <= (
            options.absolute_tolerance + options.relative_tolerance * values.abs()
        )
    return LanczosResult(
        eigenvalues=values,
        eigenvectors=vectors,
        residual_norms=residuals,
        converged=converged,
        iterations=iterations,
        hvp_evaluations=iterations + options.num_modes,
        seed=options.seed,
    )


def hutchinson_estimates(
    hvp: Callable[[torch.Tensor], torch.Tensor],
    template: torch.Tensor | None = None,
    *,
    options: HutchinsonOptions | None = None,
) -> HutchinsonResult:
    """
    Estimate Cartesian Hessian trace and diagonal with independent sign probes.

    ``z * H(z)`` estimates the diagonal and its sum estimates the trace. A
    private CPU generator makes the chosen probes reproducible across devices
    without reseeding the application's RNG. HVPs stay on the template device;
    one O(dimension) result is copied to CPU per sample for FP64 online
    statistics. All returned estimates and uncertainties are CPU FP64 tensors.
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("Curvature workflows must run outside torch.inference_mode")
    if options is None:
        options = HutchinsonOptions()
    if not isinstance(options, HutchinsonOptions):
        raise TypeError("options must be HutchinsonOptions")
    reference = _template(hvp, template)
    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    mean = torch.zeros(reference.numel(), dtype=torch.float64, device="cpu")
    m2 = torch.zeros_like(mean, device="cpu")
    trace_mean = torch.zeros((), dtype=torch.float64, device="cpu")
    trace_m2 = torch.zeros_like(trace_mean, device="cpu")
    with torch.no_grad():
        for index in range(1, options.samples + 1):
            probe = torch.randint(
                2, (reference.numel(),), generator=generator, device="cpu"
            )
            probe = (2 * probe - 1).to(reference)
            sample = (_product(hvp, probe, reference) * probe).to(
                device="cpu", dtype=torch.float64
            )
            delta = sample - mean
            mean += delta / index
            m2 += delta * (sample - mean)
            trace_sample = sample.sum()
            trace_delta = trace_sample - trace_mean
            trace_mean += trace_delta / index
            trace_m2 += trace_delta * (trace_sample - trace_mean)
            if not all(
                torch.isfinite(statistic).all()
                for statistic in (mean, m2, trace_mean, trace_m2)
            ):
                raise ValueError(
                    "Hutchinson statistics overflowed in FP64; rescale the HVP operator"
                )
        variance = (m2 / (options.samples - 1)).clamp_min(0)
        trace_variance = (trace_m2 / (options.samples - 1)).clamp_min(0)
    return HutchinsonResult(
        trace=trace_mean,
        diagonal=mean.reshape(reference.shape),
        trace_sample_variance=trace_variance,
        diagonal_sample_variance=variance.reshape(reference.shape),
        trace_standard_error=(trace_variance / options.samples).sqrt(),
        diagonal_standard_error=(variance / options.samples)
        .sqrt()
        .reshape(reference.shape),
        samples=options.samples,
        seed=options.seed,
        hvp_evaluations=options.samples,
    )


class PredictorHVP:
    """
    Own one geometry snapshot and expose UMA's requested-property HVP interface.

    Construct a new adapter when coordinates, cell, metadata, weights, or model
    settings change. Model ownership remains with the caller: do not update
    its weights/settings concurrently with a curvature calculation. Only a
    single system and conservative-force model are supported. This adapter
    does not build a dense Hessian or modify calculator/predictor settings.
    """

    def __init__(self, predictor: MLIPPredictUnitProtocol, data: AtomicData):
        if not getattr(predictor, "supports_requested_properties", False):
            raise TypeError("Predictor must support requested_properties")
        if data.natoms.numel() != 1 or data.pos.numel() == 0:
            raise ValueError("PredictorHVP requires one nonempty system")
        settings = predictor.inference_settings
        dtype = settings.base_precision_dtype
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("PredictorHVP requires float32 or float64 base precision")
        self.predictor = predictor
        self._data = data.clone()
        for key in ("hessian_vector", "_requested_properties"):
            if key in self._data:
                del self._data[key]
        self._template = torch.zeros_like(
            data.pos, dtype=dtype, device=torch.device(predictor.device)
        )

    @property
    def template(self):
        """
        Return independent shape/dtype/device metadata for curvature algorithms.
        """
        return self._template.clone()

    def __call__(self, vector: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(vector, torch.Tensor)
            or vector.shape != self._template.shape
            or vector.dtype != self._template.dtype
            or vector.device != self._template.device
        ):
            raise ValueError("HVP vector must match the adapter template")
        if not torch.isfinite(vector).all():
            raise ValueError("HVP vector must be finite")
        data = self._data.clone()
        data["hessian_vector"] = vector.detach().clone()
        output = self.predictor.predict(
            data, requested_properties=("hessian_vector_product",)
        )
        result = output.get("hessian_vector_product")
        if not isinstance(result, torch.Tensor) or result.shape != vector.shape:
            raise ValueError("Predictor did not return a coordinate-shaped HVP")
        if result.dtype != vector.dtype or result.device != vector.device:
            raise ValueError("Predictor HVP dtype/device differs from the vector")
        if not torch.isfinite(result).all():
            raise ValueError("Predictor HVP contains nonfinite values")
        return result.detach().clone()

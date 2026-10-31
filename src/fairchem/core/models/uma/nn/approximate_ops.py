"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch

from fairchem.core.models.uma.nn.radial import GaussianSmearing

_COMPENSATED_REDUCTION = ContextVar("fairchem_compensated_reduction", default=False)


@contextmanager
def approximate_ops_context(settings):
    """
    Scope approximate reduction to this predictor call, including nested calls.
    """
    enabled = bool(getattr(settings, "compensated_reduction", False))
    token = _COMPENSATED_REDUCTION.set(enabled)
    try:
        yield
    finally:
        _COMPENSATED_REDUCTION.reset(token)


def compensated_reduction_enabled() -> bool:
    # Series B rejects compile at settings validation. Preserve the existing
    # compiled Series A graph without introducing a ContextVar graph break.
    if torch.compiler.is_compiling():
        return False
    return _COMPENSATED_REDUCTION.get()


def compensated_node_sum(values, batch, num_systems):
    """
    Sum each system with a differentiable FP32 Neumaier pairwise tree.

    This is an opt-in arithmetic approximation, not an FP64 equivalent. Per-
    system selection bounds temporary storage by the largest system, without
    padding every system to its largest atom count. A tree avoids one launch per
    atom, but sorting/masking and compensation can still cost more than FP64.
    """
    if (
        isinstance(num_systems, bool)
        or not isinstance(num_systems, int)
        or num_systems < 0
    ):
        raise ValueError("num_systems must be a nonnegative integer")
    if (
        values.ndim < 1
        or batch.shape != (len(values),)
        or batch.dtype not in (torch.int32, torch.int64)
        or batch.device != values.device
        or not values.is_floating_point()
    ):
        raise ValueError(
            "Expected floating node values and matching integer batch indices"
        )
    if bool(((batch < 0) | (batch >= num_systems)).any()):
        raise ValueError("batch index outside num_systems")
    # Keep a zero connected to inputs for empty systems and higher derivatives.
    empty = values[:0].sum(0, dtype=torch.float32)
    if num_systems == 0:
        return empty.unsqueeze(0)[:0]
    outputs = []
    for system in range(num_systems):
        total = values[batch == system].float()
        if len(total) == 0:
            outputs.append(empty)
            continue
        correction = torch.zeros_like(total)
        while len(total) > 1:
            if len(total) % 2:
                total = torch.cat((total, torch.zeros_like(total[:1])))
                correction = torch.cat((correction, torch.zeros_like(correction[:1])))
            left, right = total[::2], total[1::2]
            joined = left + right
            error = torch.where(
                left.abs() >= right.abs(),
                (left - joined) + right,
                (right - joined) + left,
            )
            correction = correction[::2] + correction[1::2] + error
            total = joined
        outputs.append(total[0] + correction[0])
    return torch.stack(outputs)


def _exp_polynomial(exponent):
    # Fifth-order Taylor at x/16, followed by four squarings. Domain [-8, 0].
    z = exponent / 16
    value = 1 + z * (1 + z * (0.5 + z * (1 / 6 + z * (1 / 24 + z / 120))))
    for _ in range(4):
        value = value.square()
    return value


def local_fast_exp(exponent):
    """
    Approximate exp locally, with C2 blending and exact fallback outside [-8, 0].

    The polynomial is used on [-6, 0]. A quintic smoothstep blends its value to
    exact exp across [-8, -6]; value and the first two derivatives meet at the
    boundaries. Only the blend/fallback entries evaluate exp. This eager tensor
    implementation is an experiment; extra indexing may outweigh exp savings.
    """
    original_shape = exponent.shape
    flat = exponent.reshape(-1)
    middle = (flat >= -6) & (flat <= 0)
    blend = (flat > -8) & (flat < -6)
    outside = ~(middle | blend)
    result = torch.empty_like(flat)
    result[middle] = _exp_polynomial(flat[middle])
    blend_x = flat[blend]
    t = (blend_x + 8) / 2
    weight = t.pow(3) * (10 + t * (-15 + 6 * t))
    exact = blend_x.exp()
    result[blend] = exact + weight * (_exp_polynomial(blend_x) - exact)
    result[outside] = flat[outside].exp()
    return result.reshape(original_shape)


class ApproximateGaussian(torch.nn.Module):
    """
    Gaussian radial approximation; geometry, normalization and weights stay FP32.

    The signed normalized coordinate avoids the nondifferentiable abs() at a
    Gaussian center. Table coefficients are universal for exp(-x*x), so moving
    the model or changing Gaussian offsets does not invalidate them.
    """

    def __init__(self, original: GaussianSmearing, *, points=None, fast_exp=False):
        super().__init__()
        if points is not None and (
            isinstance(points, bool) or not isinstance(points, int) or points < 2
        ):
            raise ValueError("radial_table_points must be an integer >= 2")
        if points is not None and fast_exp:
            raise ValueError(
                "radial_table_points and radial_fast_exp are mutually exclusive"
            )
        if not isinstance(fast_exp, bool):
            raise ValueError("radial_fast_exp must be a boolean")
        if points is None and not fast_exp:
            raise ValueError("Choose a Gaussian approximation explicitly")
        self.original = original
        self.points = points
        self.fast_exp = fast_exp
        self.num_output = original.num_output
        if points is not None:
            # Build once on the CPU in FP64; runtime buffers follow model dtype
            # and device, so Intel GPU setup does not require FP64 arithmetic.
            grid = torch.linspace(-8, 8, points, dtype=torch.float64)
            values = (-grid.square()).exp()
            first = -2 * grid * values
            second = (4 * grid.square() - 2) * values
            self.step = 16 / (points - 1)
            a0 = values[:-1]
            a1 = self.step * first[:-1]
            a2 = 0.5 * self.step**2 * second[:-1]
            b0 = values[1:] - a0 - a1 - a2
            b1 = self.step * first[1:] - a1 - 2 * a2
            b2 = self.step**2 * second[1:] - 2 * a2
            coefficients = torch.stack(
                (
                    a0,
                    a1,
                    a2,
                    10 * b0 - 4 * b1 + 0.5 * b2,
                    -15 * b0 + 7 * b1 - b2,
                    6 * b0 - 3 * b1 + 0.5 * b2,
                ),
                dim=-1,
            ).to(original.offset)
            self.register_buffer("table_coefficients", coefficients, persistent=False)

    @property
    def offset(self):
        return self.original.offset

    @property
    def coeff(self):
        return self.original.coeff

    def forward(self, distance):
        displacement = distance.reshape(-1, 1) - self.offset.reshape(1, -1)
        if self.fast_exp:
            return local_fast_exp(self.coeff * displacement.square())
        normalized = displacement * math.sqrt(-self.coeff)
        inside = (normalized >= -8) & (normalized <= 8)
        selected = normalized[inside]
        coordinate = (selected + 8) / self.step
        interval = coordinate.floor().long().clamp(0, self.points - 2)
        t = coordinate - interval
        coefficients = self.table_coefficients[interval]
        value = coefficients[:, 5]
        for power in range(4, -1, -1):
            value = value * t + coefficients[:, power]
        result = torch.empty_like(normalized)
        result[inside] = value
        result[~inside] = (-normalized[~inside].square()).exp()
        return result


def apply_feature_approximations(backbone, settings) -> dict:
    """
    Apply requested Gaussian options after backend preparation and MOLE merging.

    Disabled settings leave an unmodified model untouched, and remove a prior
    radial approximation if this helper is explicitly applied again. The
    JSON-safe manifest does not claim measured speed or accuracy guarantees.
    """
    points = getattr(settings, "radial_table_points", None)
    fast_exp = getattr(settings, "radial_fast_exp", False)
    manifest = {}
    current = getattr(backbone, "distance_expansion", None)
    if points is not None or fast_exp:
        if isinstance(current, ApproximateGaussian):
            current = current.original
        if not isinstance(current, GaussianSmearing):
            raise ValueError("Radial approximation requires UMA GaussianSmearing")
        backbone.distance_expansion = ApproximateGaussian(
            current, points=points, fast_exp=fast_exp
        )
        manifest["radial_approximation"] = {
            "target": "distance_expansion",
            "kind": "quintic_gaussian_table" if points is not None else "local_exp",
            "points": points,
            "exact_fallback": True,
            "smoothness": "C2",
        }
    elif isinstance(current, ApproximateGaussian):
        backbone.distance_expansion = current.original
    if getattr(settings, "compensated_reduction", False):
        manifest["compensated_reduction"] = {
            "kind": "pairwise_neumaier_fp32",
            "output_dtype": "float64",
            "fp64_equivalent": False,
        }
    return manifest

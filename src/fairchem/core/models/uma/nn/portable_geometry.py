"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch

from fairchem.core.models.uma.common.quaternion.quaternion_utils import (
    quaternion_edge_to_y_stable,
    quaternion_multiply,
    quaternion_y_rotation,
)
from fairchem.core.models.uma.common.quaternion.wigner_d_custom_kernels import (
    quaternion_to_rotation_matrix,
)

_EXPONENTS = tuple(
    (a, b, c, 4 - a - b - c)
    for a in range(5)
    for b in range(5 - a)
    for c in range(5 - a - b)
)


def compress_quartic_coefficients(coefficients):
    """
    Collect symmetric degree-four terms without changing the polynomial.
    """
    flat = coefficients.reshape(25, 256)
    groups = {exponent: [] for exponent in _EXPONENTS}
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    exponent = tuple((a, b, c, d).count(i) for i in range(4))
                    groups[exponent].append(64 * a + 16 * b + 4 * c + d)
    return torch.stack([flat[:, groups[e]].sum(-1) for e in _EXPONENTS], dim=1)


def quartic_monomials(quaternion):
    """
    Produce the 35 unique quartic terms instead of all 256 tensor products.
    """
    powers = []
    for coordinate in quaternion.unbind(-1):
        square = coordinate.square()
        powers.append(
            (
                None,
                coordinate,
                square,
                square * coordinate,
                square.square(),
            )
        )
    terms = []
    for exponents in _EXPONENTS:
        factors = [powers[i][power] for i, power in enumerate(exponents) if power]
        term = factors[0]
        for factor in factors[1:]:
            term = term * factor
        terms.append(term)
    return torch.stack(terms, dim=-1)


def geometry_core(
    vectors,
    gamma,
    quartic_coefficients,
    offsets,
    gaussian_coeff,
    cutoff,
    p,
    a,
    b,
    c,
    distance=None,
    fold_envelope=False,
    valid=None,
):
    """
    Joint geometry graph suitable for torch.compile pointwise fusion.

    This retains the existing smooth two-chart quaternion construction and
    mathematical Gaussian/envelope functions. It introduces no fast-math
    approximation. Inductor decides the actual kernel partition on each device.
    """
    if distance is None:
        distance = vectors.norm(dim=-1)
    unit = vectors / distance.clamp_min(1e-12).unsqueeze(-1)
    quaternion = quaternion_multiply(
        quaternion_y_rotation(gamma), quaternion_edge_to_y_stable(unit)
    )
    l1 = quaternion_to_rotation_matrix(quaternion)
    l2 = (quartic_monomials(quaternion) @ quartic_coefficients.T).reshape(-1, 5, 5)
    ones = vectors.new_ones((vectors.shape[0], 1))
    wigner = torch.cat((ones, l1.flatten(1), l2.flatten(1)), dim=1)
    scaled = distance / cutoff
    envelope = torch.where(
        scaled < 1, 1 + scaled**p * (a + scaled * (b + c * scaled)), 0
    )[:, None]
    if valid is not None:
        envelope = torch.where(valid[:, None], envelope, 0)
    inv1 = l1.transpose(1, 2).flatten(1)
    inv2 = l2.transpose(1, 2).flatten(1)
    if fold_envelope:
        inverse = torch.cat((envelope, inv1 * envelope, inv2 * envelope), dim=1)
    else:
        inverse = torch.cat((ones, inv1, inv2), dim=1)
    radial = torch.exp(gaussian_coeff * (distance[:, None] - offsets[None]).square())
    return radial, envelope, wigner, inverse


class PortableGeometry(torch.nn.Module):
    """
    Own immutable l=2 and radial coefficients for the portable execution mode.
    """

    def __init__(self, model):
        super().__init__()
        coefficients = model.wigner_data.custom_kernels.C_l2
        self.register_buffer(
            "quartic_coefficients",
            # Analytic Wigner coefficients originate in FP64 even when the
            # network was constructed directly in FP32. Match its radial
            # buffer rather than requiring a later model.to(dtype) call.
            compress_quartic_coefficients(coefficients).to(
                model.distance_expansion.offset
            ),
            persistent=False,
        )
        self.register_buffer(
            "offsets",
            model.distance_expansion.offset.detach().clone(),
            persistent=False,
        )
        self.gaussian_coeff = model.distance_expansion.coeff
        self.cutoff = model.cutoff
        self.p = model.envelope.p
        self.a, self.b, self.c = model.envelope.a, model.envelope.b, model.envelope.c

    def forward(
        self, vectors, gamma=None, *, distance=None, fold_envelope=False, valid=None
    ):
        if gamma is None:
            gamma = torch.rand(
                vectors.shape[0], device=vectors.device, dtype=vectors.dtype
            ) * (2 * math.pi)
        return geometry_core(
            vectors,
            gamma,
            self.quartic_coefficients,
            self.offsets,
            self.gaussian_coeff,
            self.cutoff,
            self.p,
            self.a,
            self.b,
            self.c,
            distance=distance,
            fold_envelope=fold_envelope,
            valid=valid,
        )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _normalized_silu_parts(x, weight, bias, eps):
    centered = x - x.mean(dim=-1, keepdim=True)
    inv_std = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normalized = centered * inv_std
    affine = normalized * weight + bias
    sigmoid = affine.sigmoid()
    derivative = sigmoid * (1 + affine * (1 - sigmoid))
    return normalized, inv_std, derivative


class _LayerNormSiLU(torch.autograd.Function):
    """
    Recompute a cheap epilogue instead of saving a full normalized activation.

    Forward uses the native normalization and SiLU operations. The pure tensor
    backward is compiler-fusible and supports higher derivatives and vmap;
    no reduced-precision or approximate transcendental operation is selected.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, weight, bias, eps):
        return torch.nn.functional.silu(
            torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, bias, eps = inputs
        ctx.save_for_backward(x, weight, bias)
        ctx.save_for_forward(x, weight, bias)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        normalized, inv_std, derivative = _normalized_silu_parts(
            x, weight, bias, ctx.eps
        )
        grad_affine = grad_output * derivative
        grad_normalized = grad_affine * weight
        grad_x = inv_std * (
            grad_normalized
            - grad_normalized.mean(dim=-1, keepdim=True)
            - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True)
        )
        batch_dims = tuple(range(x.ndim - 1))
        grad_weight = grad_affine * normalized
        grad_bias = grad_affine
        if batch_dims:
            grad_weight = grad_weight.sum(dim=batch_dims)
            grad_bias = grad_bias.sum(dim=batch_dims)
        return grad_x, grad_weight, grad_bias, None

    @staticmethod
    def jvp(ctx, tangent_x, tangent_weight, tangent_bias, tangent_eps):
        x, weight, bias = ctx.saved_tensors
        normalized, inv_std, derivative = _normalized_silu_parts(
            x, weight, bias, ctx.eps
        )
        tangent_affine = torch.zeros_like(x)
        if tangent_x is not None:
            tangent_affine = (
                weight
                * inv_std
                * (
                    tangent_x
                    - tangent_x.mean(dim=-1, keepdim=True)
                    - normalized * (tangent_x * normalized).mean(dim=-1, keepdim=True)
                )
            )
        if tangent_weight is not None:
            tangent_affine = tangent_affine + normalized * tangent_weight
        if tangent_bias is not None:
            tangent_affine = tangent_affine + tangent_bias
        return tangent_affine * derivative


def layer_norm_silu(x, weight, bias, eps):
    """
    Normalize and activate while retaining only the input for force backward.
    """
    # Dynamo does not capture custom jvp; capturing a custom reverse adjoint
    # also loses double-backward support on some Torch releases. Let the
    # compiler fuse native operators and manage saved activations instead.
    if torch.compiler.is_compiling() or x.dtype not in (torch.float32, torch.float64):
        return torch.nn.functional.silu(
            torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
        )
    return _LayerNormSiLU.apply(x, weight, bias, eps)


@torch.jit.script
def gaussian(x: torch.Tensor, mean, std) -> torch.Tensor:
    a = (2 * math.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)


class PolynomialEnvelope(torch.nn.Module):
    """
    Polynomial envelope function that ensures a smooth cutoff.
    """

    def __init__(self, exponent: int = 5) -> None:
        super().__init__()
        assert exponent > 0
        self.p: float = float(exponent)
        self.a: float = -(self.p + 1) * (self.p + 2) / 2
        self.b: float = self.p * (self.p + 2)
        self.c: float = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + (d_scaled**self.p) * (
            self.a + d_scaled * (self.b + self.c * d_scaled)
        )
        return torch.where(d_scaled < 1, env_val, 0)


class GaussianSmearing(torch.nn.Module):
    def __init__(
        self,
        start: float = -5.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        basis_width_scalar: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_output = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (basis_width_scalar * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(self, dist) -> torch.Tensor:
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class RadialMLP(nn.Module):
    """
    Construct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    """

    def __init__(self, channels_list) -> None:
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue

            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]

            if i == len(channels_list) - 1:
                break

            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _LayerNormSiLU(torch.autograd.Function):
    """
    Recompute a cheap epilogue instead of saving a full normalized activation.

    Recompute the same native mean and reciprocal standard deviation in backward.
    Native LayerNorm also supplies its registered second derivative.
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
        ctx.eps = eps

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        shape = (x.shape[-1],)
        affine, mean, rstd = torch.native_layer_norm(x, shape, weight, bias, ctx.eps)
        if torch.is_grad_enabled():
            # The fused first derivative has no second-derivative registration.
            sigmoid = affine.sigmoid()
            grad_affine = grad_output * sigmoid * (1 + affine * (1 - sigmoid))
        else:
            grad_affine = torch.ops.aten.silu_backward(grad_output, affine)
        grad_x, grad_weight, grad_bias = torch.ops.aten.native_layer_norm_backward(
            grad_affine, x, shape, mean, rstd, weight, bias, ctx.needs_input_grad[:3]
        )
        return grad_x, grad_weight, grad_bias, None


def layer_norm_silu(x, weight, bias, eps):
    """
    Normalize and activate while retaining only the input for force backward.
    """
    # Capturing a custom reverse adjoint loses double-backward support on some
    # Torch releases. Let the compiler fuse native operators instead.
    # Native operations propagate an active forward-AD level directly. Starting
    # another jvp inside a custom Function fails for make_dual callers.
    # Nested torch.func transforms can hide the dual inside an outer wrapper.
    if (
        torch.compiler.is_compiling()
        or x.dtype not in (torch.float32, torch.float64)
        or torch._C._are_functorch_transforms_active()
        or any(
            torch.autograd.forward_ad.unpack_dual(value).tangent is not None
            for value in (x, weight, bias)
        )
    ):
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

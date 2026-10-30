"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Experimental strict-FP32 Linear + LayerNorm + SiLU for Triton-XPU.

The forward kernel removes two activation buffers. Its differentiable Torch
adjoint recomputes the linear output, so end-to-end force/Hessian benchmarks
must establish a benefit before enabling it by default. No hardware speed or
accuracy validation is implied by the CPU reference tests.
"""

from __future__ import annotations

import math
import os
from functools import cache

import torch
import torch.nn.functional as F

# Resolved lazily so importing this module never requires Triton or an XPU.
tl = None
libdevice = None


def linear_norm_silu_reference(x, weight, bias, norm_weight, norm_bias, eps):
    """
    Native reference, also used by the differentiable adjoint tests.
    """
    return F.silu(
        F.layer_norm(
            F.linear(x, weight, bias), (weight.shape[0],), norm_weight, norm_bias, eps
        )
    )


def _recomputed_parts(x, weight, bias, norm_weight, norm_bias, eps):
    linear = F.linear(x, weight, bias)
    centered = linear - linear.mean(dim=-1, keepdim=True)
    inv_std = (centered.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    normalized = centered * inv_std
    affine = normalized * norm_weight + norm_bias
    sigmoid = affine.sigmoid()
    derivative = sigmoid * (1 + affine * (1 - sigmoid))
    return normalized, inv_std, derivative


def linear_norm_silu_adjoint(
    x, weight, bias, norm_weight, norm_bias, eps, grad_output, needs=None
):
    """
    Recompute the exact operation's adjoint in differentiable Torch operations.

    Saved original inputs, rather than detached intermediate activations, make
    the adjoint itself differentiable for forces, Hessians and force training.
    Skip unneeded parameter gradients for frozen-model inference.
    """
    if needs is None:
        needs = (True,) * 5
    normalized, inv_std, derivative = _recomputed_parts(
        x, weight, bias, norm_weight, norm_bias, eps
    )
    grad_affine = grad_output * derivative
    grad_normalized = grad_affine * norm_weight
    grad_linear = inv_std * (
        grad_normalized
        - grad_normalized.mean(dim=-1, keepdim=True)
        - normalized * (grad_normalized * normalized).mean(dim=-1, keepdim=True)
    )
    return (
        grad_linear @ weight if needs[0] else None,
        grad_linear.mT @ x if needs[1] else None,
        grad_linear.sum(dim=0) if needs[2] else None,
        (grad_affine * normalized).sum(dim=0) if needs[3] else None,
        grad_affine.sum(dim=0) if needs[4] else None,
    )


def _linear_norm_silu_kernel(
    X,
    WEIGHT,
    BIAS,
    NORM_WEIGHT,
    NORM_BIAS,
    OUTPUT,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    accumulator = tl.full((BLOCK_M, BLOCK_N), 0, tl.float32)
    for start in range(tl.cdiv(K, BLOCK_K)):
        k = start * BLOCK_K + inner
        lhs = tl.load(
            X + row[:, None] * K + k[None, :],
            mask=(row[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        rhs = tl.load(
            WEIGHT + col[None, :] * K + k[:, None],
            mask=(k[:, None] < K) & (col[None, :] < N),
            other=0,
        )
        # Intel Triton's default is TF32; never inherit that default here.
        accumulator = tl.dot(lhs, rhs, accumulator, input_precision="ieee")
    linear = accumulator + tl.load(BIAS + col, col < N, other=0)[None, :]
    count = tl.full((), N, tl.float32)
    mean = libdevice.div_rn(tl.sum(tl.where(col[None, :] < N, linear, 0), 1), count)
    centered = tl.where(col[None, :] < N, linear - mean[:, None], 0)
    variance = libdevice.div_rn(tl.sum(centered * centered, 1), count)
    normalized = libdevice.div_rn(centered, libdevice.sqrt(variance + EPS)[:, None])
    affine = (
        normalized * tl.load(NORM_WEIGHT + col, col < N, other=0)[None, :]
        + tl.load(NORM_BIAS + col, col < N, other=0)[None, :]
    )
    # Use Intel's non-fast libdevice operations, not approximate tl.exp/rsqrt.
    value = libdevice.div_rn(affine, 1.0 + libdevice.exp(-affine))
    tl.store(
        OUTPUT + row[:, None] * N + col[None, :],
        value,
        mask=(row[:, None] < M) & (col[None, :] < N),
    )


@cache
def _load_kernel():
    # Triton resolves globals when JIT-compiling the function above.
    global tl, libdevice  # noqa: PLW0603
    try:
        import triton
        import triton.language as triton_language
        from triton.language.extra.intel import libdevice as intel_libdevice
    except ImportError:
        return None
    tl = triton_language
    libdevice = intel_libdevice
    return triton, triton.jit(_linear_norm_silu_kernel)


def _fast_math_enabled(triton):
    # In current Triton-XPU the explicit Intel flag overrides the Inductor flag.
    # Respect the live knob too: callers can override it without setting env vars.
    intel_knobs = getattr(getattr(triton, "knobs", None), "intel", None)
    if intel_knobs is not None and hasattr(intel_knobs, "fast_math"):
        return bool(intel_knobs.fast_math)
    value = os.environ.get(
        "TRITON_INTEL_FAST_MATH", os.environ.get("TORCHINDUCTOR_USE_FAST_MATH", "0")
    )
    return value.lower() in {"1", "true", "on", "yes"}


def _supported_layout(x, weight, bias, norm_weight, norm_bias, eps):
    tensors = (x, weight, bias, norm_weight, norm_bias)
    if (
        x.ndim != 2
        or weight.ndim != 2
        or any(t.dtype != torch.float32 or t.device != x.device for t in tensors)
        or any(not t.is_contiguous() for t in tensors)
        or not isinstance(eps, (float, int))
        or not math.isfinite(eps)
        or eps <= 0
    ):
        return False
    rows, inner = x.shape
    width = weight.shape[0]
    # A full-width normalization tile keeps all reductions in one kernel, but
    # wider tiles can spill registers. Unsupported shapes use the native path.
    return (
        rows > 0
        and 1 <= inner <= 2048
        and 16 <= width <= 512
        and weight.shape[1] == inner
        and all(t.shape == (width,) for t in (bias, norm_weight, norm_bias))
        and rows * width <= 2**31 - 1
        and all(t.numel() <= 2**31 - 1 for t in tensors)
    )


def _launch_linear_norm_silu(x, weight, bias, norm_weight, norm_bias, eps):
    triton, kernel = _load_kernel()
    output = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    kernel[(triton.cdiv(x.shape[0], 16),)](
        x,
        weight,
        bias,
        norm_weight,
        norm_bias,
        output,
        M=x.shape[0],
        K=x.shape[1],
        N=weight.shape[0],
        EPS=float(eps),
        BLOCK_M=16,
        BLOCK_K=32,
        BLOCK_N=triton.next_power_of_2(weight.shape[0]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


class _FusedLinearNormSiLU(torch.autograd.Function):
    @staticmethod
    def forward(x, weight, bias, norm_weight, norm_bias, eps):
        return _launch_linear_norm_silu(x, weight, bias, norm_weight, norm_bias, eps)

    @staticmethod
    def setup_context(ctx, inputs, output):
        *tensors, ctx.eps = inputs
        ctx.save_for_backward(*tensors)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            *linear_norm_silu_adjoint(
                *ctx.saved_tensors, ctx.eps, grad_output, needs=ctx.needs_input_grad[:5]
            ),
            None,
        )


def xpu_fused_linear_norm_silu(x, weight, bias, norm_weight, norm_bias, eps):
    """
    Try the experimental FP32 kernel, or return None before any kernel launch.

    Explicit opt-in is required. Compiled graphs, transforms, autocast,
    unsupported layouts and missing Intel Triton use the caller's native path.
    Compilation/execution failures propagate; they are never hidden by rerunning
    a failed or potentially corrupted computation on the native path.
    """
    if (
        os.environ.get("FAIRCHEM_XPU_FUSED_KERNELS") != "1"
        or x.device.type != "xpu"
        or torch.compiler.is_compiling()
        or torch._C._are_functorch_transforms_active()
        or torch.is_autocast_enabled("xpu")
        or not _supported_layout(x, weight, bias, norm_weight, norm_bias, eps)
    ):
        return None
    # Forward AD outside torch.func also wraps tensors; use native JVP support.
    if any(
        torch.autograd.forward_ad.unpack_dual(t).tangent is not None
        for t in (x, weight, bias, norm_weight, norm_bias)
    ):
        return None
    loaded = _load_kernel()
    if loaded is None or _fast_math_enabled(loaded[0]):
        return None
    return _FusedLinearNormSiLU.apply(x, weight, bias, norm_weight, norm_bias, eps)

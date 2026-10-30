"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Optional l=2 Triton-XPU forward kernels with differentiable PyTorch adjoints.

Import only when FAIRCHEM_XPU_FUSED_KERNELS=1 is explicitly requested. No dot,
reduced-precision math, or approximate transcendental instructions are used.
Forward kernels have not been hardware-validated by the CPU test suite.
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError(
        "FAIRCHEM_XPU_FUSED_KERNELS=1 requires the Triton-XPU distribution "
        "matched to the installed PyTorch XPU wheel"
    ) from exc

from fairchem.core.models.uma.nn.portable_wigner import (
    consumer_adjoint,
    consumer_reference,
    producer_adjoint,
    producer_projected_adjoint,
    producer_projected_reference,
    producer_reference,
)
from fairchem.core.models.uma.nn.unified_radial import RadialProjection


def _device_key(device):
    properties = torch.xpu.get_device_properties(device)
    return ":".join(
        map(
            str,
            (
                device,
                properties.name,
                getattr(properties, "driver_version", ""),
                torch.__version__,
                triton.__version__,
            ),
        )
    )


def _wide_offsets(*tensors):
    # Wrappers pass contiguous tensors, so numel bounds every pointer offset.
    return any(t is not None and t.numel() > 2**31 - 1 for t in tensors)


def strict_math_available():
    """Respect the effective Intel compiler knob, including its environment fallback."""
    intel_knobs = getattr(getattr(triton, "knobs", None), "intel", None)
    if intel_knobs is not None and hasattr(intel_knobs, "fast_math"):
        return not bool(intel_knobs.fast_math)
    value = os.environ.get(
        "TRITON_INTEL_FAST_MATH", os.environ.get("TORCHINDUCTOR_USE_FAST_MATH", "0")
    )
    return value.lower() not in {"1", "true", "on", "yes"}


def _require_strict_math():
    if not strict_math_available():
        raise ValueError("Series A XPU kernels require Intel fast_math=False")


_CONFIGS = [
    triton.Config({"BLOCK_C": block}, num_warps=warps)
    for block, warps in ((32, 4), (64, 4), (128, 4), (64, 8))
]


@triton.autotune(
    configs=_CONFIGS, key=["E", "C", "H", "PROJECTED", "WIDE", "DEVICE_KEY"]
)
@triton.jit
def _producer_kernel(
    X,
    INDEX,
    W,
    RADIAL,
    WEIGHT,
    BIAS,
    M0,
    M1,
    M2,
    E: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    PROJECTED: tl.constexpr,
    WIDE: tl.constexpr,
    DEVICE_KEY: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    edge = tl.program_id(0)
    m = tl.program_id(1)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    side = channel // C
    valid = channel < 2 * C
    if WIDE:
        edge = edge.to(tl.int64)
        channel = channel.to(tl.int64)
        side = side.to(tl.int64)
    atom = tl.load(INDEX + side * E + edge, mask=valid, other=0)
    if WIDE:
        atom = atom.to(tl.int64)
    else:
        atom = atom.to(tl.int32)
    # M-major coefficient -> L-major coefficient.
    row = tl.where(
        m == 0,
        0,
        tl.where(
            m == 1,
            2,
            tl.where(
                m == 2,
                6,
                tl.where(
                    m == 3,
                    3,
                    tl.where(
                        m == 4,
                        7,
                        tl.where(
                            m == 5, 1, tl.where(m == 6, 5, tl.where(m == 7, 8, 4))
                        ),
                    ),
                ),
            ),
        ),
    )
    start = tl.where(row == 0, 0, tl.where(row < 4, 1, 4))
    width = tl.where(row == 0, 1, tl.where(row < 4, 3, 5))
    offset = tl.where(row == 0, 0, tl.where(row < 4, 1, 10))
    value = tl.full((BLOCK_C,), 0, tl.float32)
    for j in tl.static_range(5):
        weight = tl.load(
            W + edge * 35 + offset + (row - start) * width + j, mask=j < width, other=0
        )
        feature = tl.load(
            X + (atom * 9 + start + j) * C + channel % C,
            mask=valid & (j < width),
            other=0,
        )
        value = value + weight * feature
    radial_slot = tl.where(m < 3, m, tl.where(m < 7, 3 + (m - 3) % 2, 5))
    radial_row = radial_slot * (2 * C) + channel
    if PROJECTED:
        # The last radial MLP layer is evaluated where its result is consumed.
        # Keep all products and the reduction in FP32, including on Xe hardware.
        hidden_channel = tl.arange(0, BLOCK_H)
        hidden = tl.load(
            RADIAL + edge * H + hidden_channel, mask=hidden_channel < H, other=0
        )
        weight = tl.load(
            WEIGHT + radial_row[:, None] * H + hidden_channel[None, :],
            mask=valid[:, None] & (hidden_channel[None, :] < H),
            other=0,
        )
        scale = tl.sum(hidden[None, :] * weight, 1) + tl.load(
            BIAS + radial_row, mask=valid, other=0
        )
    else:
        scale = tl.load(RADIAL + edge * (12 * C) + radial_row, mask=valid, other=0)
    value = value * scale
    tl.store(M0 + edge * (6 * C) + m * (2 * C) + channel, value, mask=valid & (m < 3))
    tl.store(
        M1 + edge * (8 * C) + (m - 3) * (2 * C) + channel,
        value,
        mask=valid & (m >= 3) & (m < 7),
    )
    tl.store(
        M2 + edge * (4 * C) + (m - 7) * (2 * C) + channel, value, mask=valid & (m >= 7)
    )


@triton.autotune(
    configs=_CONFIGS, key=["N", "E", "C", "WIDE", "DEVICE_KEY", "HAS_RESIDUAL"]
)
@triton.jit
def _consumer_kernel(
    G0,
    G1,
    G2,
    W,
    ORDER,
    OFFSETS,
    RESIDUAL,
    OUTPUT,
    N: tl.constexpr,
    E: tl.constexpr,
    C: tl.constexpr,
    WIDE: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    DEVICE_KEY: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    atom = tl.program_id(0)
    row = tl.program_id(1)
    channel = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    if WIDE:
        atom = atom.to(tl.int64)
        channel = channel.to(tl.int64)
    mask = channel < C
    start = tl.where(row == 0, 0, tl.where(row < 4, 1, 4))
    width = tl.where(row == 0, 1, tl.where(row < 4, 3, 5))
    offset = tl.where(row == 0, 0, tl.where(row < 4, 1, 10))
    if HAS_RESIDUAL:
        value = tl.load(RESIDUAL + (atom * 9 + row) * C + channel, mask=mask, other=0)
    else:
        value = tl.full((BLOCK_C,), 0, tl.float32)
    begin = tl.load(OFFSETS + atom)
    end = tl.load(OFFSETS + atom + 1)
    for cursor in range(begin, end):
        edge = tl.load(ORDER + cursor)
        if WIDE:
            edge = edge.to(tl.int64)
        rotated = tl.full((BLOCK_C,), 0, tl.float32)
        for j in tl.static_range(5):
            coefficient = start + j
            weight = tl.load(
                W + edge * 35 + offset + (row - start) * width + j,
                mask=j < width,
                other=0,
            )
            is0 = (coefficient == 0) | (coefficient == 2) | (coefficient == 6)
            is1 = (
                (coefficient == 1)
                | (coefficient == 3)
                | (coefficient == 5)
                | (coefficient == 7)
            )
            is2 = (coefficient == 4) | (coefficient == 8)
            p0 = tl.where(coefficient == 0, 0, tl.where(coefficient == 2, 1, 2))
            p1 = tl.where(
                coefficient == 3,
                0,
                tl.where(coefficient == 7, 1, tl.where(coefficient == 1, 2, 3)),
            )
            p2 = tl.where(coefficient == 8, 0, 1)
            feature = tl.load(
                G0 + edge * (3 * C) + p0 * C + channel,
                mask=mask & (j < width) & is0,
                other=0,
            )
            feature += tl.load(
                G1 + edge * (4 * C) + p1 * C + channel,
                mask=mask & (j < width) & is1,
                other=0,
            )
            feature += tl.load(
                G2 + edge * (2 * C) + p2 * C + channel,
                mask=mask & (j < width) & is2,
                other=0,
            )
            rotated += weight * feature
        value += rotated
    # Every output element has one writer; autotune may safely rerun candidates.
    tl.store(OUTPUT + (atom * 9 + row) * C + channel, value, mask=mask)


class FusedProducer(torch.autograd.Function):
    """
    Strict-FP32 gather/rotate/radial packing, with an analytic Torch adjoint.
    """

    @staticmethod
    def vmap(info, in_dims, *inputs):
        # Triton cannot receive functorch's BatchedTensor pointers. Preserve the
        # transform with native operations; ordinary calls still use the kernel.
        values = torch.vmap(
            producer_reference, in_dims=in_dims, randomness=info.randomness
        )(*inputs)
        return values, (0, 0, 0)

    @staticmethod
    def forward(x, index, wigner, radial, channels):
        _require_strict_math()
        x, index, wigner, radial = (t.contiguous() for t in (x, index, wigner, radial))
        edges = index.shape[1]
        outputs = [x.new_empty((edges, width * channels)) for width in (6, 8, 4)]

        def grid(meta):
            return (edges, 9, triton.cdiv(2 * channels, meta["BLOCK_C"]))

        _producer_kernel[grid](
            x,
            index,
            wigner,
            radial,
            radial,
            radial,
            *outputs,
            edges,
            channels,
            1,
            1,
            False,
            _wide_offsets(x, wigner, radial, *outputs),
            _device_key(x.device),
            enable_fp_fusion=False,
        )
        return tuple(outputs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:4])
        ctx.save_for_forward(*inputs[:4])
        ctx.channels = inputs[4]

    @staticmethod
    def backward(ctx, g0, g1, g2):
        x, index, wigner, radial = ctx.saved_tensors
        shapes = (6, 8, 4)
        gradients = [
            g if g is not None else x.new_zeros((index.shape[1], n * ctx.channels))
            for g, n in zip((g0, g1, g2), shapes)
        ]
        dx, dw, dr = producer_adjoint(
            *gradients, x, index, wigner, radial, ctx.channels
        )
        return dx, None, dw, dr, None

    @staticmethod
    def jvp(ctx, dx, _index, dw, dr, _channels):
        x, index, wigner, radial = ctx.saved_tensors
        result = [x.new_zeros((index.shape[1], n * ctx.channels)) for n in (6, 8, 4)]
        for changed, tangent in ((0, dx), (1, dw), (2, dr)):
            if tangent is None:
                continue
            values = [x, wigner, radial]
            values[changed] = tangent
            part = producer_reference(
                values[0], index, values[1], values[2], ctx.channels
            )
            result = [a + b for a, b in zip(result, part)]
        return tuple(result)


class FusedProjectedProducer(torch.autograd.Function):
    """FP32 final radial projection, gather, rotation and packing in one kernel."""

    @staticmethod
    def vmap(info, in_dims, *inputs):
        def reference(x, index, wigner, hidden, weight, bias, channels):
            return producer_projected_reference(
                x, index, wigner, RadialProjection(hidden, weight, bias), channels
            )

        values = torch.vmap(reference, in_dims=in_dims, randomness=info.randomness)(
            *inputs
        )
        return values, (0, 0, 0)

    @staticmethod
    def forward(x, index, wigner, hidden, weight, bias, channels):
        _require_strict_math()
        if not 1 <= hidden.shape[1] <= 512:
            raise ValueError(
                "The projected XPU kernel supports hidden widths 1 through 512"
            )
        x, index, wigner, hidden, weight, bias = (
            t.contiguous() for t in (x, index, wigner, hidden, weight, bias)
        )
        edges = index.shape[1]
        outputs = [x.new_empty((edges, width * channels)) for width in (6, 8, 4)]

        def grid(meta):
            return (edges, 9, triton.cdiv(2 * channels, meta["BLOCK_C"]))

        _producer_kernel[grid](
            x,
            index,
            wigner,
            hidden,
            weight,
            bias,
            *outputs,
            edges,
            channels,
            hidden.shape[1],
            triton.next_power_of_2(hidden.shape[1]),
            True,
            _wide_offsets(x, wigner, hidden, weight, *outputs),
            _device_key(x.device),
            enable_fp_fusion=False,
        )
        return tuple(outputs)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:6])
        ctx.save_for_forward(*inputs[:6])
        ctx.channels = inputs[6]

    @staticmethod
    def backward(ctx, g0, g1, g2):
        x, index, wigner, hidden, weight, bias = ctx.saved_tensors
        gradients = [
            g if g is not None else x.new_zeros((index.shape[1], n * ctx.channels))
            for g, n in zip((g0, g1, g2), (6, 8, 4))
        ]
        dx, dw, dh, dweight, dbias = producer_projected_adjoint(
            *gradients, x, index, wigner, hidden, weight, bias, ctx.channels
        )
        return dx, None, dw, dh, dweight, dbias, None

    @staticmethod
    def jvp(ctx, dx, _index, dw, dh, dweight, dbias, _channels):
        x, index, wigner, hidden, weight, bias = ctx.saved_tensors
        radial = torch.nn.functional.linear(hidden, weight, bias)
        dr = torch.zeros_like(radial)
        if dh is not None:
            dr = dr + torch.nn.functional.linear(dh, weight)
        if dweight is not None:
            dr = dr + torch.nn.functional.linear(hidden, dweight)
        if dbias is not None:
            dr = dr + dbias
        result = producer_reference(x, index, wigner, dr, ctx.channels)
        for changed, tangent in ((0, dx), (1, dw)):
            if tangent is None:
                continue
            values = [x, wigner]
            values[changed] = tangent
            part = producer_reference(values[0], index, values[1], radial, ctx.channels)
            result = tuple(a + b for a, b in zip(result, part))
        return result


class FusedConsumer(torch.autograd.Function):
    """
    Compact inverse rotation and CSR row-owned sum, including the residual.
    """

    @staticmethod
    def vmap(info, in_dims, *inputs):
        def reference(
            g0, g1, g2, wigner, target, order, offsets, nodes, channels, residual
        ):
            return consumer_reference(
                g0, g1, g2, wigner, target, nodes, channels, residual
            )

        value = torch.vmap(reference, in_dims=in_dims, randomness=info.randomness)(
            *inputs
        )
        return value, 0

    @staticmethod
    def forward(
        g0, g1, g2, wigner, target, order, offsets, num_nodes, channels, residual
    ):
        _require_strict_math()
        g0, g1, g2, wigner, order, offsets = (
            t.contiguous() for t in (g0, g1, g2, wigner, order, offsets)
        )
        if residual is not None:
            residual = residual.contiguous()
        output = g0.new_empty((num_nodes, 9, channels))

        def grid(meta):
            return (num_nodes, 9, triton.cdiv(channels, meta["BLOCK_C"]))

        _consumer_kernel[grid](
            g0,
            g1,
            g2,
            wigner,
            order,
            offsets,
            output if residual is None else residual,
            output,
            num_nodes,
            target.numel(),
            channels,
            _wide_offsets(g0, g1, g2, wigner, output),
            residual is not None,
            _device_key(g0.device),
            enable_fp_fusion=False,
        )
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs[:5])
        ctx.save_for_forward(*inputs[:5])
        ctx.num_nodes, ctx.channels = inputs[7:9]
        ctx.has_residual = inputs[9] is not None

    @staticmethod
    def backward(ctx, grad):
        g0, g1, g2, wigner, target = ctx.saved_tensors
        grads = consumer_adjoint(grad, g0, g1, g2, wigner, target, ctx.channels)
        return (
            *grads,
            None,
            None,
            None,
            None,
            None,
            grad if ctx.has_residual else None,
        )

    @staticmethod
    def jvp(ctx, dg0, dg1, dg2, dw, _target, _order, _offsets, _nodes, _channels, dr):
        g0, g1, g2, wigner, target = ctx.saved_tensors

        def zero(value, tangent):
            return torch.zeros_like(value) if tangent is None else tangent

        result = consumer_reference(
            zero(g0, dg0),
            zero(g1, dg1),
            zero(g2, dg2),
            wigner,
            target,
            ctx.num_nodes,
            ctx.channels,
            dr,
        )
        if dw is not None:
            result = result + consumer_reference(
                g0, g1, g2, dw, target, ctx.num_nodes, ctx.channels
            )
        return result

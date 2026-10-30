"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compact l=2 operations with ordinary PyTorch higher derivatives.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass

import torch

# Coefficient order used by CoefficientMapping(lmax=2, mmax=2).
M_TO_L = (0, 2, 6, 3, 7, 1, 5, 8, 4)
L_TO_M = (0, 5, 1, 3, 8, 6, 2, 4, 7)
_BLOCKS = ((0, 1, 0), (1, 3, 1), (4, 5, 10))


def compact_wigner(value: torch.Tensor) -> torch.Tensor:
    """
    Pack diagonal l blocks without detaching coordinate derivatives.
    """
    if value.ndim == 2 and value.shape[1] == 35:
        return value
    if value.ndim != 3 or value.shape[1:] != (9, 9):
        raise ValueError("portable l=2 Wigner requires [E,35] or [E,9,9]")
    return torch.cat(
        [
            value[:, start : start + width, start : start + width].flatten(1)
            for start, width, _ in _BLOCKS
        ],
        dim=1,
    )


def _rotate_blocks(x: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
    """
    Rotate each nonzero diagonal block; never expand Wigner to 9x9.
    """
    edges = w.shape[0]
    blocks = []
    for start, width, offset in _BLOCKS:
        matrix = w[:, offset : offset + width * width].reshape(edges, width, width)
        blocks.extend(torch.bmm(matrix, x[:, start : start + width]).unbind(1))
    return blocks


def _pack_coefficients(coeffs: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return (
        torch.cat([coeffs[i] for i in M_TO_L[:3]], dim=-1),
        torch.cat([coeffs[i] for i in M_TO_L[3:7]], dim=-1),
        torch.cat([coeffs[i] for i in M_TO_L[7:]], dim=-1),
    )


def _unpack_coefficients(g0, g1, g2, channels):
    m_coeffs = list(g0.split(channels, -1))
    m_coeffs.extend(g1.split(channels, -1))
    m_coeffs.extend(g2.split(channels, -1))
    return [m_coeffs[index] for index in L_TO_M]


def producer_reference(x, edge_index, wigner, radial, channels):
    """
    Gather, rotate and form SO2 GEMM inputs without a dense Wigner matrix.
    """
    wigner = compact_wigner(wigner)
    edge_features = torch.cat(
        (x.index_select(0, edge_index[0]), x.index_select(0, edge_index[1])), dim=-1
    )
    m0, m1, m2 = _pack_coefficients(_rotate_blocks(edge_features, wigner))
    r0, r1, r2 = radial.split((6 * channels, 4 * channels, 2 * channels), dim=-1)
    return m0 * r0, m1 * r1.repeat(1, 2), m2 * r2.repeat(1, 2)


def producer_projected_reference(x, edge_index, wigner, radial, channels):
    """
    Consume one radial projection block at a time, avoiding its concatenation.
    """
    wigner = compact_wigner(wigner)
    edge_features = torch.cat(
        (x.index_select(0, edge_index[0]), x.index_select(0, edge_index[1])), dim=-1
    )
    m0, m1, m2 = _pack_coefficients(_rotate_blocks(edge_features, wigner))
    return (
        m0 * radial.block(0, 6 * channels),
        m1 * radial.block(6 * channels, 4 * channels).repeat(1, 2),
        m2 * radial.block(10 * channels, 2 * channels).repeat(1, 2),
    )


@dataclass
class CSRTopology:
    """
    Own the source index as well as its stable CSR permutation.
    """

    owner: torch.Tensor
    order: torch.Tensor
    lengths: torch.Tensor
    offsets: torch.Tensor
    index32: bool
    ready_event: object | None = None

    def wait_on_current_stream(self):
        if self.ready_event is None:
            return
        accelerator = getattr(torch, self.owner.device.type)
        stream = accelerator.current_stream(self.owner.device)
        stream.wait_event(self.ready_event)
        for value in (self.owner, self.order, self.lengths, self.offsets):
            value.record_stream(stream)


class CSRTopologyCache:
    """
    Bounded cache whose source ownership prevents allocator-address reuse hits.

    Only integer topology is cached. In-place Tensor operations invalidate
    entries through the version counter. As with autograd, writes through
    ``.data`` or foreign memory that bypass version tracking are unsupported.
    Inference tensors without a version counter are deliberately not cached.
    """

    def __init__(self, max_entries=8, max_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def clear(self):
        with self._lock:
            self._entries.clear()

    @staticmethod
    def _size(topology):
        return sum(
            t.untyped_storage().nbytes()
            for t in (
                topology.owner,
                topology.order,
                topology.lengths,
                topology.offsets,
            )
        )

    def get(self, target, num_nodes):
        if target.ndim != 1 or target.dtype not in (torch.int32, torch.int64):
            raise ValueError("scatter targets must be a one-dimensional integer tensor")
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        try:
            version = target._version
        except RuntimeError:
            version = None
        key = (
            id(target),
            version,
            num_nodes,
            target.device,
            target.dtype,
            target.shape,
            target.stride(),
            target.storage_offset(),
        )
        with self._lock:
            if version is not None and key in self._entries:
                self.hits += 1
                self._entries.move_to_end(key)
                topology = self._entries[key]
                topology.wait_on_current_stream()
                return topology
            self.misses += 1
            if target.numel() and not bool(
                ((target >= 0) & (target < num_nodes)).all()
            ):
                raise ValueError("scatter target is outside [0,num_nodes)")
            order = torch.argsort(target, stable=True)
            lengths = torch.bincount(target.to(torch.int64), minlength=num_nodes)
            offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            index32 = max(num_nodes, target.numel()) <= torch.iinfo(torch.int32).max
            if index32:
                order = order.to(torch.int32)
                offsets = offsets.to(torch.int32)
            topology = CSRTopology(target, order, lengths, offsets, index32)
            if target.device.type in {"cuda", "xpu"}:
                accelerator = getattr(torch, target.device.type)
                topology.ready_event = accelerator.Event()
                topology.ready_event.record(accelerator.current_stream(target.device))
                topology.wait_on_current_stream()
            if version is not None and self.max_entries > 0:
                # Drop stale versions without retaining each old owner forever.
                for old in [k for k in self._entries if k[0] == id(target)]:
                    del self._entries[old]
                self._entries[key] = topology
                while self._entries and (
                    len(self._entries) > self.max_entries
                    or sum(self._size(v) for v in self._entries.values())
                    > self.max_bytes
                ):
                    self._entries.popitem(last=False)
            return topology


_CSR_CACHE = CSRTopologyCache()


@torch.compiler.disable
def get_csr_topology(target, num_nodes):
    return _CSR_CACHE.get(target, num_nodes)


class _SegmentSum(torch.autograd.Function):
    """
    Native segmented forward with a gather adjoint supporting double backward.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(values, target, order, lengths, num_nodes):
        sorted_values = values.index_select(0, order)
        return torch.segment_reduce(sorted_values, "sum", lengths=lengths)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])
        ctx.save_for_forward(inputs[1])
        ctx.num_nodes = inputs[4]

    @staticmethod
    def backward(ctx, grad_output):
        (target,) = ctx.saved_tensors
        return grad_output.index_select(0, target), None, None, None, None

    @staticmethod
    def jvp(ctx, grad_values, _target, _order, _lengths, _num_nodes):
        (target,) = ctx.saved_tensors
        return grad_values.new_zeros(
            (ctx.num_nodes,) + grad_values.shape[1:]
        ).index_add(0, target, grad_values)


def segmented_sum(values, target, num_nodes):
    """
    Sum existing dtype values by target without a global-prefix subtraction.
    """
    if torch.compiler.is_compiling() or values.device.type == "xpu":
        # segment_reduce kernel coverage varies across XPU releases. The XPU
        # fused consumer below owns CSR rows; this reference retains index_add.
        return values.new_zeros((num_nodes,) + values.shape[1:]).index_add(
            0, target, values
        )
    topology = get_csr_topology(target, num_nodes)
    if not values.shape[0]:
        return values.new_zeros((num_nodes,) + values.shape[1:]) + values.sum() * 0
    return _SegmentSum.apply(
        values, target, topology.order, topology.lengths, num_nodes
    )


def consumer_reference(g0, g1, g2, wigner, target, num_nodes, channels, residual=None):
    """
    Inverse rotation and segmented scatter with compact block matrices.
    """
    wigner = compact_wigner(wigner)
    coefficients = _unpack_coefficients(g0, g1, g2, channels)
    result = []
    for start, width, offset in _BLOCKS:
        features = torch.stack(coefficients[start : start + width], dim=1)
        matrix = wigner[:, offset : offset + width * width].reshape(-1, width, width)
        reduced = segmented_sum(torch.bmm(matrix, features), target, num_nodes)
        if residual is not None:
            reduced = reduced + residual[:, start : start + width]
        result.append(reduced)
    return torch.cat(result, dim=1)


_VALIDATED_INDICES = OrderedDict()
_VALIDATION_LOCK = threading.RLock()


@torch.compiler.disable
def _validate_kernel_indices(index, num_nodes):
    """Check raw-kernel bounds once per immutable topology, with bounded ownership."""
    try:
        version = index._version
    except RuntimeError:
        version = None
    key = (
        id(index),
        version,
        num_nodes,
        index.device,
        tuple(index.shape),
        tuple(index.stride()),
    )
    with _VALIDATION_LOCK:
        if version is not None and key in _VALIDATED_INDICES:
            _VALIDATED_INDICES.move_to_end(key)
            return
        if index.numel() and bool(((index < 0) | (index >= num_nodes)).any()):
            raise ValueError("edge indices are outside the node range")
        size = index.untyped_storage().nbytes()
        if version is None or size > 32 * 1024 * 1024:
            return
        for stale in [k for k in _VALIDATED_INDICES if k[0] == id(index)]:
            del _VALIDATED_INDICES[stale]
        _VALIDATED_INDICES[key] = index
        while (
            len(_VALIDATED_INDICES) > 8
            or sum(
                value.untyped_storage().nbytes()
                for value in _VALIDATED_INDICES.values()
            )
            > 32 * 1024 * 1024
        ):
            _VALIDATED_INDICES.popitem(last=False)


def _want_xpu_kernel(*values):
    if (
        torch.compiler.is_compiling()
        or os.environ.get("FAIRCHEM_XPU_FUSED_KERNELS", "0") != "1"
        or getattr(torch._C, "_are_functorch_transforms_active", lambda: False)()
        or torch.is_autocast_enabled("xpu")
        or not all(
            v.device.type == "xpu"
            and v.device == values[0].device
            and v.dtype == torch.float32
            for v in values
        )
    ):
        return False
    if any(
        torch.autograd.forward_ad.unpack_dual(v).tangent is not None for v in values
    ):
        return False
    from fairchem.core.models.uma.nn.xpu_wigner_kernels import strict_math_available

    return strict_math_available()


def producer(x, edge_index, wigner, radial, channels):
    if (
        channels <= 0
        or x.shape[1:] != (9, channels)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype not in (torch.int32, torch.int64)
        or edge_index.device != x.device
    ):
        raise ValueError("invalid portable producer shape, index dtype or device")
    wigner = compact_wigner(wigner)
    if wigner.shape[0] != edge_index.shape[1] or radial.shape != (
        edge_index.shape[1],
        12 * channels,
    ):
        raise ValueError("Wigner/radial sizes must match the edge count and channels")
    if _want_xpu_kernel(x, wigner, radial) and edge_index.shape[1]:
        from fairchem.core.models.uma.nn.xpu_wigner_kernels import FusedProducer

        _validate_kernel_indices(edge_index, x.shape[0])
        return FusedProducer.apply(x, edge_index, wigner, radial, channels)
    return producer_reference(x, edge_index, wigner, radial, channels)


def producer_projected(x, edge_index, wigner, radial, channels):
    """Dispatch the lazy radial producer without constructing its full projection."""
    wigner = compact_wigner(wigner)
    if channels <= 0 or x.shape[1:] != (9, channels):
        raise ValueError("l=2 node features must have shape [N, 9, channels]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if (
        edge_index.dtype not in (torch.int32, torch.int64)
        or edge_index.device != x.device
    ):
        raise ValueError("edge_index must use integer indices on the feature device")
    edges = edge_index.shape[1]
    if (
        wigner.shape != (edges, 35)
        or radial.hidden.ndim != 2
        or radial.hidden.shape[0] != edges
        or radial.weight.shape != (12 * channels, radial.hidden.shape[1])
        or radial.bias.shape != (12 * channels,)
    ):
        raise ValueError("incompatible Wigner or radial projection shape")
    if (
        edges
        and 1 <= radial.hidden.shape[1] <= 512
        and _want_xpu_kernel(x, wigner, *radial)
    ):
        from fairchem.core.models.uma.nn.xpu_wigner_kernels import (
            FusedProjectedProducer,
        )

        _validate_kernel_indices(edge_index, x.shape[0])
        return FusedProjectedProducer.apply(x, edge_index, wigner, *radial, channels)
    return producer_projected_reference(x, edge_index, wigner, radial, channels)


def consumer(g0, g1, g2, wigner, target, num_nodes, channels, residual=None):
    wigner = compact_wigner(wigner)
    edges = target.numel()
    if (
        channels <= 0
        or target.ndim != 1
        or num_nodes < 0
        or target.device != g0.device
        or any(
            g.shape != (edges, n * channels) for g, n in zip((g0, g1, g2), (3, 4, 2))
        )
        or wigner.shape[0] != edges
        or (residual is not None and residual.shape != (num_nodes, 9, channels))
    ):
        raise ValueError("invalid portable consumer shapes or target device")
    residual_compatible = residual is None or (
        residual.device == g0.device and residual.dtype == g0.dtype
    )
    if _want_xpu_kernel(g0, g1, g2, wigner) and edges and residual_compatible:
        from fairchem.core.models.uma.nn.xpu_wigner_kernels import FusedConsumer

        topology = get_csr_topology(target, num_nodes)
        return FusedConsumer.apply(
            g0,
            g1,
            g2,
            wigner,
            target,
            topology.order,
            topology.offsets,
            num_nodes,
            channels,
            residual,
        )
    return consumer_reference(g0, g1, g2, wigner, target, num_nodes, channels, residual)


def producer_adjoint(g0, g1, g2, x, edge_index, wigner, radial, channels):
    """
    Analytic differentiable adjoint, also used by the optional XPU kernels.
    """
    edges = edge_index.shape[1]
    double_channels = 2 * channels
    x_edge = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
    rotated = _rotate_blocks(x_edge, wigner)
    packed = _pack_coefficients(rotated)
    r0, r1, r2 = radial.split((6 * channels, 4 * channels, 2 * channels), -1)
    d_packed = (g0 * r0, g1 * r1.repeat(1, 2), g2 * r2.repeat(1, 2))
    d_coeffs = _unpack_coefficients(*d_packed, double_channels)
    dx_blocks, dw_blocks = [], []
    for start, width, offset in _BLOCKS:
        dw = torch.stack(d_coeffs[start : start + width], dim=1)
        matrix = wigner[:, offset : offset + width * width].reshape(edges, width, width)
        dx_blocks.append(torch.bmm(matrix.transpose(1, 2), dw))
        dw_blocks.append(
            torch.bmm(dw, x_edge[:, start : start + width].transpose(1, 2)).flatten(1)
        )
    dx_edge = torch.cat(dx_blocks, dim=1)
    dx = torch.zeros_like(x).index_add(0, edge_index[0], dx_edge[:, :, :channels])
    dx = dx.index_add(0, edge_index[1], dx_edge[:, :, channels:])
    dr = torch.cat(
        (
            g0 * packed[0],
            (g1 * packed[1]).reshape(edges, 2, 4 * channels).sum(dim=1),
            (g2 * packed[2]).reshape(edges, 2, 2 * channels).sum(dim=1),
        ),
        dim=1,
    )
    return dx, torch.cat(dw_blocks, dim=1), dr


def producer_projected_adjoint(
    g0, g1, g2, x, edge_index, wigner, hidden, weight, bias, channels
):
    """Differentiate the fused radial projection using native FP32 operations."""
    radial = torch.nn.functional.linear(hidden, weight, bias)
    dx, dw, dr = producer_adjoint(g0, g1, g2, x, edge_index, wigner, radial, channels)
    return dx, dw, dr @ weight, dr.transpose(0, 1) @ hidden, dr.sum(dim=0)


def consumer_adjoint(grad, g0, g1, g2, wigner, target, channels):
    """
    Gather the node adjoint, then differentiate compact inverse rotations.
    """
    coefficients = _unpack_coefficients(g0, g1, g2, channels)
    grad_edges = grad.index_select(0, target)
    d_coefficients, d_wigner = [], []
    edges = target.numel()
    for start, width, offset in _BLOCKS:
        matrix = wigner[:, offset : offset + width * width].reshape(edges, width, width)
        local_grad = grad_edges[:, start : start + width]
        features = torch.stack(coefficients[start : start + width], dim=1)
        d_coefficients.extend(torch.bmm(matrix.transpose(1, 2), local_grad).unbind(1))
        d_wigner.append(torch.bmm(local_grad, features.transpose(1, 2)).flatten(1))
    return (*_pack_coefficients(d_coefficients), torch.cat(d_wigner, dim=1))

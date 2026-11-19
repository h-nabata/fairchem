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
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass

import torch

# Coefficient order used by CoefficientMapping(lmax=2, mmax=2).
M_TO_L = (0, 2, 6, 3, 7, 1, 5, 8, 4)
L_TO_M = (0, 5, 1, 3, 8, 6, 2, 4, 7)
_BLOCKS = ((0, 1, 0), (1, 3, 1), (4, 5, 10))


_CAPTURE_SAFE = ContextVar("fairchem_capture_safe_wigner", default=False)


@contextmanager
def capture_safe_kernels():
    """Capture dynamic indices using native gathers and scatter operations."""
    token = _CAPTURE_SAFE.set(True)
    try:
        yield
    finally:
        _CAPTURE_SAFE.reset(token)


def fused_kernel_enabled(family):
    return (
        os.environ.get(
            f"FAIRCHEM_XPU_FUSED_{family}",
            os.environ.get("FAIRCHEM_XPU_FUSED_KERNELS", "0"),
        )
        == "1"
    )


def csr_chunk_edges():
    """Maximum edges per CSR partial. Zero preserves the one-stage consumer."""
    value = os.environ.get("FAIRCHEM_XPU_WIGNER_CSR_CHUNK_EDGES", "0")
    try:
        chunk = int(value)
    except ValueError as exc:
        raise ValueError(
            "FAIRCHEM_XPU_WIGNER_CSR_CHUNK_EDGES must be a nonnegative integer"
        ) from exc
    if chunk < 0 or chunk > 2**31 - 1:
        raise ValueError(
            "FAIRCHEM_XPU_WIGNER_CSR_CHUNK_EDGES must be between 0 and 2**31-1"
        )
    return chunk


def split_csr_rows(offsets, chunk, output_row_bytes, *, budget=64 << 20):
    """Partition ordered CSR edges and their partials into contiguous segments."""
    nodes = offsets.numel() - 1
    # Budget includes room for integer indexing temporaries. Check the bound
    # before allocating per-node or per-group vectors.
    if not chunk or _CAPTURE_SAFE.get() or nodes == 0 or 64 * (nodes + 1) > budget:
        return None
    if torch._C._are_functorch_transforms_active():
        return None
    lengths = offsets.diff().long()
    counts = torch.div(lengths - 1, chunk, rounding_mode="floor") + 1
    groups, largest = torch.stack((counts.sum(), counts.max())).tolist()
    if (
        largest < 2
        or 64 * (nodes + 1) + (groups + 1) * (64 + output_row_bytes) > budget
    ):
        return None
    zero = counts.new_zeros(1)
    node_offsets = torch.cat((zero, counts.cumsum(0)))
    node_ids = torch.arange(nodes, device=offsets.device)
    owner = torch.repeat_interleave(node_ids, counts, output_size=groups)
    relative = torch.arange(groups, device=offsets.device) - node_offsets[owner]
    start = offsets[owner] + relative * chunk
    partial_offsets = torch.cat((start, offsets[-1:])).to(offsets.dtype)
    return partial_offsets, node_offsets.to(offsets.dtype)


def fused_policy_key():
    from fairchem.core.models.uma.nn.xpu_radial_kernels import _fast_math_enabled

    wigner, radial = fused_kernel_enabled("WIGNER"), fused_kernel_enabled("RADIAL")
    triton_module = None
    if wigner or radial:
        with suppress(ImportError):
            import triton as triton_module
    return (
        wigner,
        radial,
        os.environ.get("FAIRCHEM_XPU_WIGNER_CSR", "1") == "1",
        _fast_math_enabled(triton_module),
        csr_chunk_edges(),
    )


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


def _scale_pair(packed, radial):
    # The real/imaginary SO2 blocks share the same radial coefficients. Use a
    # broadcast view instead of allocating their repeated edge-sized tensor.
    return (
        packed.reshape(packed.shape[0], 2, radial.shape[-1]) * radial[:, None, :]
    ).flatten(1)


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
    return m0 * r0, _scale_pair(m1, r1), _scale_pair(m2, r2)


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
        _scale_pair(m1, radial.block(6 * channels, 4 * channels)),
        _scale_pair(m2, radial.block(10 * channels, 2 * channels)),
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
    owner_snapshot: torch.Tensor | None = None

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
    entries through the version counter. CPU topology also owns a value snapshot
    because a NumPy view can change integer indices without updating that counter.
    Foreign writes to accelerator memory still require caller synchronization and
    the normal PyTorch mutation/version protocol.
    Inference tensors without a version counter are deliberately not cached.
    """

    def __init__(self, max_entries=8, max_bytes=32 * 1024 * 1024):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._generation = 0
        self.hits = 0
        self.misses = 0

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._generation += 1

    @staticmethod
    def _size(topology):
        return sum(
            t.untyped_storage().nbytes()
            for t in (
                topology.owner,
                topology.order,
                topology.lengths,
                topology.offsets,
                topology.owner_snapshot,
            )
            if t is not None
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
            generation = self._generation
            topology = self._entries.get(key) if version is not None else None
        if topology is not None and (
            topology.owner_snapshot is None
            or torch.equal(topology.owner_snapshot, target)
        ):
            with self._lock:
                self.hits += 1
                if self._entries.get(key) is topology:
                    self._entries.move_to_end(key)
            topology.wait_on_current_stream()
            return topology
        with self._lock:
            self.misses += 1
        source = target.detach().clone() if target.device.type == "cpu" else target
        if source.numel() and not bool(((source >= 0) & (source < num_nodes)).all()):
            raise ValueError("scatter target is outside [0,num_nodes)")
        order = torch.argsort(source, stable=True)
        lengths = torch.bincount(source.to(torch.int64), minlength=num_nodes)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        index32 = max(num_nodes, target.numel()) <= torch.iinfo(torch.int32).max
        if index32:
            order = order.to(torch.int32)
            offsets = offsets.to(torch.int32)
        topology = CSRTopology(target, order, lengths, offsets, index32)
        if target.device.type == "cpu":
            topology.owner_snapshot = source
        if target.device.type in {"cuda", "xpu"}:
            accelerator = getattr(torch, target.device.type)
            topology.ready_event = accelerator.Event()
            topology.ready_event.record(accelerator.current_stream(target.device))
            topology.wait_on_current_stream()
        with self._lock:
            if (
                version is not None
                and self.max_entries > 0
                and target._version == version
                and self._generation == generation
            ):
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
    def forward(values, target, order, lengths, offsets, num_nodes):
        sorted_values = values.index_select(0, order)
        chunk = csr_chunk_edges()
        if chunk:
            plan = split_csr_rows(
                offsets, chunk, values[0].numel() * values.element_size()
            )
            if plan is not None:
                partial_offsets, node_offsets = plan
                partials = torch.segment_reduce(
                    sorted_values, "sum", offsets=partial_offsets
                )
                return torch.segment_reduce(partials, "sum", offsets=node_offsets)
        return torch.segment_reduce(sorted_values, "sum", lengths=lengths)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])
        ctx.save_for_forward(inputs[1])
        ctx.num_nodes = inputs[5]

    @staticmethod
    def backward(ctx, grad_output):
        (target,) = ctx.saved_tensors
        return grad_output.index_select(0, target), None, None, None, None, None

    @staticmethod
    def jvp(ctx, grad_values, _target, _order, _lengths, _offsets, _num_nodes):
        (target,) = ctx.saved_tensors
        return grad_values.new_zeros(
            (ctx.num_nodes,) + grad_values.shape[1:]
        ).index_add(0, target, grad_values)


def segmented_sum(values, target, num_nodes):
    """
    Sum existing dtype values by target without a global-prefix subtraction.
    """
    if target.ndim != 1 or values.ndim < 1 or values.shape[0] != target.numel():
        raise ValueError(
            "Segment values and target indices must have the same leading length"
        )
    if (
        torch.compiler.is_compiling()
        or values.device.type == "xpu"
        or _CAPTURE_SAFE.get()
    ):
        # segment_reduce kernel coverage varies across XPU releases. The XPU
        # fused consumer below owns CSR rows; this reference retains index_add.
        return values.new_zeros((num_nodes,) + values.shape[1:]).index_add(
            0, target, values
        )
    topology = get_csr_topology(target, num_nodes)
    if not values.shape[0]:
        return values.new_zeros((num_nodes,) + values.shape[1:]) + values.sum() * 0
    return _SegmentSum.apply(
        values,
        topology.owner_snapshot if topology.owner_snapshot is not None else target,
        topology.order,
        topology.lengths,
        topology.offsets,
        num_nodes,
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
    with _VALIDATION_LOCK:
        if index._version != version:
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
        or not fused_kernel_enabled("WIGNER")
        or _CAPTURE_SAFE.get()
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
    if (
        edges
        and residual_compatible
        and os.environ.get("FAIRCHEM_XPU_WIGNER_CSR", "1") == "1"
        and _want_xpu_kernel(g0, g1, g2, wigner)
    ):
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


def _producer_edge_adjoint(g0, g1, g2, x, edge_index, wigner, radial, channels, needs):
    """Differentiate only the requested operands before accumulating node gradients."""
    need_x, need_w, need_r = needs
    edges = edge_index.shape[1]
    dx_edge = dw = dr = None
    if need_w or need_r:
        x_edge = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=-1)
    if need_x or need_w:
        r0, r1, r2 = radial.split((6 * channels, 4 * channels, 2 * channels), -1)
        d_coeffs = _unpack_coefficients(
            g0 * r0, _scale_pair(g1, r1), _scale_pair(g2, r2), 2 * channels
        )
        dx_blocks, dw_blocks = [], []
        for start, width, offset in _BLOCKS:
            adj = torch.stack(d_coeffs[start : start + width], dim=1)
            if need_x:
                matrix = wigner[:, offset : offset + width * width].reshape(
                    edges, width, width
                )
                dx_blocks.append(torch.bmm(matrix.transpose(1, 2), adj))
            if need_w:
                dw_blocks.append(
                    torch.bmm(
                        adj, x_edge[:, start : start + width].transpose(1, 2)
                    ).flatten(1)
                )
        if need_x:
            dx_edge = torch.cat(dx_blocks, dim=1)
        if need_w:
            dw = torch.cat(dw_blocks, dim=1)
    if need_r:
        packed = _pack_coefficients(_rotate_blocks(x_edge, wigner))
        dr = torch.cat(
            (
                g0 * packed[0],
                (g1 * packed[1]).reshape(edges, 2, 4 * channels).sum(dim=1),
                (g2 * packed[2]).reshape(edges, 2, 2 * channels).sum(dim=1),
            ),
            dim=1,
        )
    return dx_edge, dw, dr


def _accumulate_node_gradient(dx, dx_edge, index, channels):
    dx = dx.index_add(0, index[0], dx_edge[:, :, :channels])
    return dx.index_add(0, index[1], dx_edge[:, :, channels:])


def producer_adjoint(
    g0, g1, g2, x, edge_index, wigner, radial, channels, needs=(True, True, True)
):
    dx_edge, dw, dr = _producer_edge_adjoint(
        g0, g1, g2, x, edge_index, wigner, radial, channels, needs
    )
    dx = (
        _accumulate_node_gradient(torch.zeros_like(x), dx_edge, edge_index, channels)
        if needs[0]
        else None
    )
    return dx, dw, dr


def producer_projected_adjoint(
    g0,
    g1,
    g2,
    x,
    edge_index,
    wigner,
    hidden,
    weight,
    bias,
    channels,
    needs=(True, True, True, True, True),
    *,
    chunk_edges=None,
):
    """Stream projected radial and its adjoint by edge, including higher derivatives."""
    need_x, need_w, need_h, need_weight, need_bias = needs
    need_r = need_h or need_weight or need_bias
    # Bound each [chunk, 12*C] radial temporary to about 4 MiB in FP32.
    if chunk_edges is None:
        chunk_edges = max(1, (4 << 20) // (12 * channels * x.element_size()))
    dx = torch.zeros_like(x) if need_x else None
    dweight = dbias = None
    dw_chunks, dh_chunks = [], []
    # Include an empty slice to preserve shape, dtype and derivative connectivity.
    for start in range(0, max(1, hidden.shape[0]), chunk_edges):
        sl = slice(start, start + chunk_edges)
        h = hidden[sl]
        index = edge_index[:, sl]
        radial = (
            torch.nn.functional.linear(h, weight, bias) if need_x or need_w else None
        )
        dx_edge, dw, dr = _producer_edge_adjoint(
            g0[sl],
            g1[sl],
            g2[sl],
            x,
            index,
            wigner[sl],
            radial,
            channels,
            (need_x, need_w, need_r),
        )
        if need_x:
            dx = _accumulate_node_gradient(dx, dx_edge, index, channels)
        if need_w:
            dw_chunks.append(dw)
        if need_h:
            dh_chunks.append(dr @ weight)
        if need_weight:
            part = dr.transpose(0, 1) @ h
            dweight = part if dweight is None else dweight + part
        if need_bias:
            part = dr.sum(dim=0)
            dbias = part if dbias is None else dbias + part
    return (
        dx,
        torch.cat(dw_chunks) if need_w else None,
        torch.cat(dh_chunks) if need_h else None,
        dweight,
        dbias,
    )


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

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import accumulate, pairwise
from typing import NamedTuple

import torch
import torch.nn as nn

from .matmul import linear_with_folded_batch


@lru_cache(maxsize=1)
def _load_fairchem_cpp():
    # A binary extension built for another PyTorch/CUDA version can fail with
    # ImportError or OSError. Do not load it for the portable PyTorch MoLE path.
    try:
        import fairchem_cpp
    except (ImportError, OSError) as exc:
        raise ImportError(
            "DGL MoLE requires a working fairchem_cpp extension matching "
            "PyTorch and the target device. Use mole_layer_type='pytorch' "
            "to use the portable implementation."
        ) from exc
    return fairchem_cpp


def interval_intersection(interval1, interval2):
    """
    Compute intersection of two intervals [a, b] and [c, d]
    Returns None if no intersection, otherwise returns [start, end]
    """
    a, b = interval1
    c, d = interval2

    start = max(a, c)
    end = min(b, d)

    if start <= end:
        return [start, end]
    else:
        return None  # No intersection


def _softmax(x):
    return torch.softmax(x, dim=1) + 0.005


def _pnorm(x):
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[0], p=1.0, dim=1)


def _pnorm_per_system(x):
    """Batch-independent L1 routing with a total smoothing weight of two per system."""
    return torch.nn.functional.normalize(x.abs() + 2 / x.shape[1], p=1.0, dim=1)


def norm_str_to_fn(act):
    if act == "softmax":
        return _softmax
    elif act == "pnorm":
        return _pnorm
    elif act == "pnorm_per_system":
        return _pnorm_per_system
    else:
        raise ValueError


@dataclass
class MOLEGlobals:
    # the linear coefficient for each expert
    expert_mixing_coefficients: torch.Tensor
    # if the input contains N separate systems, then the sizes represent the number of atoms in each system
    # this is used to for the MoLE to assign the correct parameters for each system
    mole_sizes: torch.Tensor
    # when using activation checkpointing, the inputs are chunked and given piecemeal so the start idx must be
    # updated each time the chunked operation happens. It's better to make this an input but in order for
    # the MolE interface to maintain functional equivalence to the Linear layer interface, this extra info
    # needs to be added here instead. (TODO: is there a cleaner way to do this?)
    ac_start_idx: int = 0

    # Padded layouts for MOLE.forward, valid for one mole_sizes tensor
    _padded_plans: dict = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def row_layout(self, start: int, rows: int, device):
        sizes = self.mole_sizes
        if not hasattr(self, "_padded_plans"):
            self._padded_plans = {}
        cache = self._padded_plans
        version = None if sizes.is_inference() else sizes._version
        if (
            sizes is not cache.get("sizes")
            or version is None
            or version != cache.get("version")
        ):
            cache.clear()
            counts = sizes.tolist()
            cache.update(
                sizes=sizes,
                version=version,
                counts=counts,
                boundaries=[0, *accumulate(counts)],
            )
        device = torch.device(device)
        stream = (
            getattr(torch, device.type).current_stream(device)
            if device.type in {"cuda", "xpu"}
            else None
        )
        key = (start, rows, device, stream)
        if key not in cache:
            cache[key] = _build_padded_plan(cache["counts"], start, rows, device)
        return cache["boundaries"], cache[key]

    def __getstate__(self):
        return {**self.__dict__, "_padded_plans": {}}


class PaddedGroup(NamedTuple):
    """
    Systems of similar size that share one bmm.
    """

    # System index of each block in the group, used to select its mixed weights
    systems: torch.Tensor
    # Rows per block (the largest system in the group)
    block_rows: int
    # First padded row of the group
    offset: int


class PaddedPlan(NamedTuple):
    """
    Layout for running one MOLE layer on several systems with a few bmm calls.
    """

    groups: tuple[PaddedGroup, ...]
    # Total padded rows over all groups
    padded_rows: int
    # Padded position of each input row, in input order
    index: torch.Tensor


@torch.inference_mode(False)
@torch.no_grad()
def _build_padded_plan(
    sizes: list[int], start: int, num_rows: int, device: torch.device
) -> PaddedPlan | None:
    """
    Build the padded layout for input rows [start, start + num_rows).

    Systems are sorted by size and grouped greedily so that each group's padded
    rows stay within twice its real rows.

    Args:
        sizes: Rows per system for the whole batch.
        start: Global index of the first input row (activation checkpointing).
        num_rows: Number of input rows.
        device: Device for the index tensors.

    Returns:
        The layout, or None when the per-system loop is as good (at most one
        system has rows, or every system needs its own group) or the sizes do
        not cover the input.
    """
    end = start + num_rows
    segments = []  # (system, first input row, rows), in input order
    offset = 0
    for system, size in enumerate(sizes):
        first, last = max(offset, start), min(offset + size, end)
        if last > first:
            segments.append((system, first - start, last - first))
        offset += size
    if len(segments) < 2 or sum(seg[2] for seg in segments) != num_rows:
        return None

    grouped = []  # each group sorted by rows, largest first
    group_rows = 0
    for segment in sorted(segments, key=lambda seg: -seg[2]):
        if grouped:
            group = grouped[-1]
            if (len(group) + 1) * group[0][2] <= 2 * (group_rows + segment[2]):
                group_rows += segment[2]
                group.append(segment)
                continue
        grouped.append([segment])
        group_rows = segment[2]
    if len(grouped) == len(segments):
        return None

    groups = []
    index = torch.empty(num_rows, dtype=torch.long, device="cpu")
    padded_rows = 0
    for group in grouped:
        block_rows = group[0][2]
        for block, (_, first, rows) in enumerate(group):
            begin = padded_rows + block * block_rows
            index[first : first + rows] = torch.arange(
                begin, begin + rows, device="cpu"
            )
        systems = torch.tensor([seg[0] for seg in group], device="cpu")
        groups.append(PaddedGroup(systems.to(device), block_rows, padded_rows))
        padded_rows += len(group) * block_rows
    return PaddedPlan(tuple(groups), padded_rows, index.to(device))


def init_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


# A checkpoint can outlive later forward calls on the same module. Keep its
# routing local to the invocation, including when autograd replays the chunk.
_checkpoint_routes = ContextVar("mole_checkpoint_routes", default=None)


@contextmanager
def checkpoint_routes(routes):
    token = _checkpoint_routes.set(routes)
    try:
        yield
    finally:
        _checkpoint_routes.reset(token)


class MOLEDGL(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors,
        bias: bool,
    ):
        super().__init__()

        assert global_mole_tensors is not None
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors

    def forward(self, x):
        if x.device.type == "xpu":
            raise NotImplementedError(
                "The fairchem_cpp DGL MoLE kernels do not support XPU. "
                "Use mole_layer_type='pytorch' for Intel GPU inference."
            )
        fairchem_cpp = _load_fairchem_cpp()
        state = self.global_mole_tensors
        routes = _checkpoint_routes.get()
        route = routes.get(id(state)) if routes else None
        sizes = state.mole_sizes
        if route is not None:
            state = route
            # segment_mm consumes one length per system for this chunk.
            ends = (route.mole_sizes.cumsum(0) - route.ac_start_idx).clamp(0, len(x))
            sizes = ends.diff(prepend=ends.new_zeros(1)).to(route.mole_sizes.dtype)
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                state.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = fairchem_cpp.ops.segment_mm(x, weights, sizes)
        elif x.ndim == 3:
            r = fairchem_cpp.ops.segment_mm(
                x.reshape(-1, x_shape[-1]),
                weights,
                sizes * x_shape[1],
            ).reshape(*x_shape[:-1], -1)
        else:
            raise ValueError("x.ndim not in (2,3) not allowed")
        if self.bias is not None:
            r += self.bias
        return r


class MOLE(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        global_mole_tensors: MOLEGlobals,
        bias: bool,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features

        self.weights, self.bias = init_linear(
            num_experts, bias, out_features, in_features
        )

        self.global_mole_tensors = global_mole_tensors

    def merged_linear_layer(self):
        # The merged layer uses the first system's coefficients. Avoid mixing
        # every system and allocating/randomizing a second real weight matrix
        # merely to overwrite it. Explicit dtype also preserves FP64 sources.
        linear = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.bias is not None,
            device="meta",
            dtype=self.weights.dtype,
        )

        with (
            torch.no_grad(),
            torch.autocast(device_type=self.weights.device.type, enabled=False),
        ):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients[:1],
            )
            linear.weight = torch.nn.Parameter(weights[0])
            if self.bias is not None:
                linear.bias = torch.nn.Parameter(self.bias.clone())
        return linear

    def forward(self, x):
        state = self.global_mole_tensors
        routes = _checkpoint_routes.get()
        if routes:
            state = routes.get(id(state), state)
        ac_start_idx = state.ac_start_idx
        assert len(state.mole_sizes) > 0
        start_idxs, plan = state.row_layout(ac_start_idx, x.shape[0], x.device)
        mole_intervals = pairwise(start_idxs)

        # Because activation checkpointing can chunk the inputs, we need to only compute
        # the mole_size intervals that overlap with the current chunks
        # for example if mole_sizes = [10,10,15]
        # start_idxs -> [0,10,20,35]
        # mole_intervals -> [(0,10),(10,20),(20,35)]
        # if the input segment is (5,15) then we compute the following 2 segments
        # (5,10),(10,15)
        input_segment = (ac_start_idx, ac_start_idx + x.shape[0])
        assert (
            0 <= input_segment[0] <= input_segment[1] <= start_idxs[-1]
        ), f"Input segment {input_segment} is outside MoLE sizes {start_idxs[-1]}"

        segments = []
        for n, mole_segment in enumerate(mole_intervals):
            interval_overlap = interval_intersection(input_segment, mole_segment)
            # Intervals are half-open for tensor slicing. A touching boundary
            # or a system with zero edges must not launch an empty GEMM.
            if (
                interval_overlap is not None
                and interval_overlap[0] < interval_overlap[1]
            ):
                start = interval_overlap[0] - ac_start_idx
                end = interval_overlap[1] - ac_start_idx
                segments.append((n, start, end))

        first_system = segments[0][0] if segments else 0
        last_system = segments[-1][0] + 1 if segments else 1
        # Checkpoint chunks often touch only one or two systems. Mixing all
        # systems would retain an unnecessary [batch, out, in] allocation in
        # every layer/chunk. A slice also preserves gradients to the routing
        # coefficients without constructing a device-side index tensor.
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->boi",
                self.weights,
                state.expert_mixing_coefficients[first_system:last_system],
            )

        if plan is not None:
            return self._padded_forward(x, weights, plan, first_system)

        if not segments:
            # Preserve an empty input's shape and autograd connections to the
            # parameters/routing coefficients, including all-zero edge batches.
            return linear_with_folded_batch(x, weights[0], bias=self.bias)

        out = [
            linear_with_folded_batch(
                x[start:end], weights[n - first_system], bias=self.bias
            )
            for n, start, end in segments
        ]

        result = out[0] if len(out) == 1 else torch.concatenate(out, dim=0)
        assert (
            result.shape[0] == x.shape[0]
        ), f"result shape {result.shape}, does not match input shape {x.shape} at dim 0"
        return result

    def _padded_forward(self, x, weights, plan: PaddedPlan, first_system: int):
        """
        Apply each system's mixed weights to its rows with one bmm per group.

        Rows are scattered into zero-padded per-system blocks, each group of
        blocks is multiplied by its weights in one batched matmul, and rows are
        gathered back in input order. This replaces one matmul per system per
        layer, which made batched inference bound by CPU-side kernel launches.

        Args:
            x: Input of shape (rows, ..., in_features).
            weights: Mixed weights per system, (systems, out_features, in_features).
            plan: Padded layout from MOLEGlobals.padded_plan.

        Returns:
            Output of shape (rows, ..., out_features).
        """
        rows = x.reshape(x.shape[0], -1, x.shape[-1])
        width, in_features = rows.shape[1], rows.shape[2]
        padded = rows.new_zeros(plan.padded_rows, width, in_features)
        padded.index_copy_(0, plan.index, rows)
        outputs = []
        for group in plan.groups:
            blocks = group.systems.shape[0]
            chunk = padded[group.offset : group.offset + blocks * group.block_rows]
            out = torch.bmm(
                chunk.reshape(blocks, group.block_rows * width, in_features),
                weights[group.systems - first_system].transpose(1, 2),
            )
            outputs.append(out.reshape(blocks * group.block_rows, width, -1))
        out = (outputs[0] if len(outputs) == 1 else torch.cat(outputs)).index_select(
            0, plan.index
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*x.shape[:-1], out.shape[-1])

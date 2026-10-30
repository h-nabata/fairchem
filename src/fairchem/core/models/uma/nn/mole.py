"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

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


def norm_str_to_fn(act):
    if act == "softmax":
        return _softmax
    elif act == "pnorm":
        return _pnorm
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


def init_linear(num_experts, use_bias, out_features, in_features):
    k = math.sqrt(1.0 / in_features)
    weights = nn.Parameter(
        k * 2 * (torch.rand(num_experts, out_features, in_features) - 0.5)
    )
    bias = nn.Parameter(k * 2 * (torch.rand(out_features) - 0.5)) if use_bias else None
    return weights, bias


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
        with torch.autocast(device_type=self.weights.device.type, enabled=False):
            weights = torch.einsum(
                "eoi, be->bio",
                self.weights,
                self.global_mole_tensors.expert_mixing_coefficients,
            )
        x_shape = x.shape
        if x.ndim == 2:
            r = fairchem_cpp.ops.segment_mm(
                x, weights, self.global_mole_tensors.mole_sizes
            )
        elif x.ndim == 3:
            r = fairchem_cpp.ops.segment_mm(
                x.reshape(-1, x_shape[-1]),
                weights,
                self.global_mole_tensors.mole_sizes * x_shape[1],
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
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        assert len(self.global_mole_tensors.mole_sizes) > 0
        # TODO: precompute these if needed but they should be small and on cpu
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        mole_intervals = zip(start_idxs, start_idxs[1:])

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
                self.global_mole_tensors.expert_mixing_coefficients[
                    first_system:last_system
                ],
            )

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

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Unified Radial MLP: Computes all layers' radial functions in a single
batched operation.

Instead of running N separate RadialMLP forward passes:
    for layer in layers:
        radial_out = layer.so2_conv_1.rad_func(x_edge)  # Sequential

We run one batched first layer, then each tail:
    all_radial_outs = unified_radial_mlp(x_edge)  # list of [E, out]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn

from .radial import layer_norm_silu

if TYPE_CHECKING:
    from .radial import RadialMLP

__all__ = ["RadialProjection", "UnifiedRadialMLP", "create_unified_radial_mlp"]


class RadialProjection(NamedTuple):
    """
    Deferred final radial linear, consumed one angular-order block at a time.

    The payload retains the smaller hidden activation instead of materializing
    every radial coefficient. All fields remain differentiable tensors.
    """

    hidden: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def block(self, start: int, width: int) -> torch.Tensor:
        return torch.nn.functional.linear(
            self.hidden,
            self.weight.narrow(0, start, width),
            self.bias.narrow(0, start, width),
        )

    def materialize(self) -> torch.Tensor:
        return torch.nn.functional.linear(self.hidden, self.weight, self.bias)

    def split(self, split_size: int, dim: int = 0) -> tuple[RadialProjection, ...]:
        if dim != 0:
            raise ValueError("RadialProjection can only be split along edges")
        return tuple(
            type(self)(h, self.weight, self.bias)
            for h in self.hidden.split(split_size, dim=0)
        )


def _tensor_cache_key(tensor: torch.Tensor) -> tuple | None:
    # The owners are retained separately: id/storage reuse must never hit a
    # previous tensor's cache entry. Inference tensors have no version counter.
    try:
        version = tensor._version
    except RuntimeError:
        return None
    return (id(tensor), version, tensor.device, tensor.dtype, tuple(tensor.shape))


def _record_cache_event(tensor: torch.Tensor):
    if tensor.device.type not in {"cuda", "xpu"}:
        return None
    backend = getattr(torch, tensor.device.type)
    event = backend.Event()
    event.record(backend.current_stream(tensor.device))
    return event


def _wait_cache_event(event, *tensors: torch.Tensor):
    if event is None:
        return
    device = tensors[0].device
    stream = getattr(torch, device.type).current_stream(device)
    stream.wait_event(event)
    for tensor in tensors:
        tensor.record_stream(stream)


# Expected structure of RadialMLP.net Sequential
_EXPECTED_NET_STRUCTURE = (
    nn.Linear,  # 0: first linear
    nn.LayerNorm,  # 1
    nn.SiLU,  # 2
    nn.Linear,  # 3: second linear
    nn.LayerNorm,  # 4
    nn.SiLU,  # 5
    nn.Linear,  # 6: third linear
)


def _validate_radial_mlp(mlp: RadialMLP, idx: int, reference: RadialMLP | None) -> None:
    """
    Validate a single RadialMLP has expected structure and matches reference.

    Args:
        mlp: The RadialMLP to validate.
        idx: Index in the list (for error messages).
        reference: First RadialMLP to compare dimensions against (None for first).
    """
    # Check layer count
    if len(mlp.net) != 7:
        raise ValueError(f"RadialMLP[{idx}]: expected 7 layers, got {len(mlp.net)}")

    # Check layer types
    for j, expected_type in enumerate(_EXPECTED_NET_STRUCTURE):
        if not isinstance(mlp.net[j], expected_type):
            raise TypeError(
                f"RadialMLP[{idx}].net[{j}]: expected {expected_type.__name__}, "
                f"got {type(mlp.net[j]).__name__}"
            )

    # Check feature dimensions match reference (all MLPs must be identical)
    if reference is not None:
        for j in (0, 3, 6):  # Linear layers
            if mlp.net[j].in_features != reference.net[j].in_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: in_features mismatch "
                    f"({mlp.net[j].in_features} vs {reference.net[j].in_features})"
                )
            if mlp.net[j].out_features != reference.net[j].out_features:
                raise ValueError(
                    f"RadialMLP[{idx}].net[{j}]: out_features mismatch "
                    f"({mlp.net[j].out_features} vs {reference.net[j].out_features})"
                )


class UnifiedRadialMLP(nn.Module):
    """
    Unified radial MLP that batches the first linear layer across N RadialMLPs.

    The first layer uses concatenated weights for a single GEMM (all N layers
    share the same input). Layers 2+ use stacked weight buffers for fast
    indexed functional calls.
    """

    def __init__(self, radial_mlps: list[RadialMLP]) -> None:
        """
        Initialize from a list of RadialMLP modules.

        Args:
            radial_mlps: List of RadialMLP modules with identical architecture.
        """
        super().__init__()

        assert len(radial_mlps) > 0, "Need at least one RadialMLP"

        # Validate all MLPs have expected structure and match each other
        for i, mlp in enumerate(radial_mlps):
            _validate_radial_mlp(mlp, i, radial_mlps[0] if i > 0 else None)

        self.num_layers = len(radial_mlps)
        self.hidden_features = radial_mlps[0].net[0].out_features
        self.ln_eps = radial_mlps[0].net[1].eps
        self.ln1_eps = tuple(mlp.net[1].eps for mlp in radial_mlps)
        self.ln2_eps = tuple(mlp.net[4].eps for mlp in radial_mlps)
        self.hidden2_features = radial_mlps[0].net[3].out_features
        self.register_buffer("_source_table", None, persistent=False)
        self.register_buffer("_target_table", None, persistent=False)
        self._element_cache_key = None
        self._element_cache_owners = ()
        self._element_cache_event = None

        # First layer: concatenated for single GEMM
        self.register_buffer(
            "W1_cat",
            torch.cat([mlp.net[0].weight.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "b1_cat",
            torch.cat([mlp.net[0].bias.detach() for mlp in radial_mlps], dim=0),
        )

        # Remaining layers: stacked [N, ...] for indexed access
        self.register_buffer(
            "ln1_weight",
            torch.stack([mlp.net[1].weight.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln1_bias",
            torch.stack([mlp.net[1].bias.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_weight",
            torch.stack([mlp.net[3].weight.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc2_bias",
            torch.stack([mlp.net[3].bias.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_weight",
            torch.stack([mlp.net[4].weight.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "ln2_bias",
            torch.stack([mlp.net[4].bias.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_weight",
            torch.stack([mlp.net[6].weight.detach() for mlp in radial_mlps], dim=0),
        )
        self.register_buffer(
            "fc3_bias",
            torch.stack([mlp.net[6].bias.detach() for mlp in radial_mlps], dim=0),
        )

    def _apply(self, fn, recurse=True):
        # Do not retain references to tensors on the old device after .to().
        self._clear_element_cache()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        # Accelerator events cannot be pickled/deep-copied. Derived tables
        # belong to the live instance and are rebuilt in the copied model.
        state = super().__getstate__()
        state["_buffers"] = dict(state["_buffers"])
        state["_buffers"]["_source_table"] = None
        state["_buffers"]["_target_table"] = None
        state["_element_cache_key"] = None
        state["_element_cache_owners"] = ()
        state["_element_cache_event"] = None
        return state

    def _clear_element_cache(self) -> None:
        self._source_table = None
        self._target_table = None
        self._element_cache_key = None
        self._element_cache_owners = ()
        self._element_cache_event = None

    def umas_radial_mlp(
        self, h: torch.Tensor, i: int, lazy: bool = False
    ) -> torch.Tensor | RadialProjection:
        """
        Apply each layer's own normalization and activation specifications.
        """
        h = layer_norm_silu(
            h,
            self.ln1_weight[i],
            self.ln1_bias[i],
            self.ln1_eps[i],
        )
        fused = None
        if h.device.type == "xpu":
            from .xpu_radial_kernels import xpu_fused_linear_norm_silu

            fused = xpu_fused_linear_norm_silu(
                h,
                self.fc2_weight[i],
                self.fc2_bias[i],
                self.ln2_weight[i],
                self.ln2_bias[i],
                self.ln2_eps[i],
            )
        if fused is None:
            h = torch.nn.functional.linear(h, self.fc2_weight[i], self.fc2_bias[i])
            h = layer_norm_silu(
                h,
                self.ln2_weight[i],
                self.ln2_bias[i],
                self.ln2_eps[i],
            )
        else:
            h = fused
        projection = RadialProjection(h, self.fc3_weight[i], self.fc3_bias[i])
        return projection if lazy else projection.materialize()

    def _tails(self, h_all, lazy):
        h_per_layer = h_all.split(self.hidden_features, dim=1)
        return [self.umas_radial_mlp(h, i, lazy) for i, h in enumerate(h_per_layer)]

    def forward(self, x: torch.Tensor, lazy: bool = False) -> list:
        """
        Compute all radial functions, optionally deferring the final linear.
        """
        h_all = torch.nn.functional.linear(x, self.W1_cat, self.b1_cat)
        return self._tails(h_all, lazy)

    def forward_decomposed(
        self,
        distance_features: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        source_embedding: torch.Tensor,
        target_embedding: torch.Tensor,
        lazy: bool = False,
    ) -> list:
        """
        Precompute the fixed elemental parts of the first radial linear.

        Only frozen embeddings and frozen first-layer weights are cached. The
        cache is owned, versioned and device/dtype-specific; differentiable
        weights take a fresh path, so no autograd graph survives between calls.
        """
        D = distance_features.shape[1]
        S = source_embedding.shape[1]
        T = target_embedding.shape[1]
        if self.W1_cat.shape[1] != D + S + T:
            raise ValueError("Radial distance and element feature widths do not match")
        if (
            source_indices.shape != target_indices.shape
            or source_indices.numel() != distance_features.shape[0]
        ):
            raise ValueError("Radial element indices must have one entry per edge")
        owners = (self.W1_cat, source_embedding, target_embedding)
        cacheable = not torch.compiler.is_compiling() and not any(
            t.requires_grad for t in owners
        )
        key = tuple(_tensor_cache_key(t) for t in owners) if cacheable else None
        cacheable = cacheable and all(k is not None for k in key)
        key = (key, D, S, T)
        if not cacheable or key != self._element_cache_key:
            source_table = torch.nn.functional.linear(
                source_embedding,
                self.W1_cat[:, D : D + S],
            )
            target_table = torch.nn.functional.linear(
                target_embedding,
                self.W1_cat[:, D + S :],
            )
            if cacheable:
                self._source_table = source_table
                self._target_table = target_table
                self._element_cache_key = key
                self._element_cache_owners = owners
                self._element_cache_event = _record_cache_event(source_table)
        else:
            source_table, target_table = self._source_table, self._target_table
            _wait_cache_event(self._element_cache_event, source_table, target_table)
        h_all = torch.nn.functional.linear(
            distance_features,
            self.W1_cat[:, :D],
            self.b1_cat,
        )
        h_all = h_all + source_table.index_select(0, source_indices)
        h_all = h_all + target_table.index_select(0, target_indices)
        return self._tails(h_all, lazy)


def create_unified_radial_mlp(radial_mlps: list) -> UnifiedRadialMLP:
    """
    Factory function to create a UnifiedRadialMLP from a list of RadialMLPs.

    Args:
        radial_mlps: List of RadialMLP modules

    Returns:
        UnifiedRadialMLP instance with shared first layer weights
    """
    return UnifiedRadialMLP(radial_mlps)

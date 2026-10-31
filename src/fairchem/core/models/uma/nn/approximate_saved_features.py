"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from numbers import Integral
from types import MethodType

import torch

from fairchem.core.models.uma.nn.activation import (
    GateActivation,
    ScaledSigmoid,
    ScaledSiLU,
)
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}
_ACTIVATIONS = (
    GateActivation,
    ScaledSigmoid,
    ScaledSiLU,
    torch.nn.SiLU,
    torch.nn.Sigmoid,
)


@dataclass(frozen=True)
class _PackedFeature:
    compressed: torch.Tensor
    scale: torch.Tensor
    original_dtype: torch.dtype
    source: weakref.ReferenceType
    version: int | None


@dataclass(frozen=True)
class _UncompressedFeature:
    tensor: torch.Tensor
    version: int | None


def _pack_feature(tensor, dtype, min_elements):
    # Frozen parameters, their views and buffers never require gradients.
    # Returning detached tensors also avoids saved-output reference cycles.
    value = tensor.detach()
    version = None if tensor.is_inference() else tensor._version
    if (
        isinstance(tensor, torch.nn.Parameter)
        or not tensor.requires_grad
        or tensor.dtype != torch.float32
        or tensor.layout != torch.strided
        or tensor.numel() < min_elements
    ):
        return _UncompressedFeature(value, version)

    # A block scale avoids introducing infinities for finite FP32 features that
    # exceed FP16's range (or round above BF16's largest finite value). The scale
    # stays on the same device, without scalar reads or host synchronization.
    maximum = value.abs().amax()
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.ones_like(maximum))
    scale = (maximum / torch.finfo(dtype).max).clamp_min(1)
    packed = (value / scale).to(dtype=dtype, memory_format=torch.contiguous_format)
    return _PackedFeature(packed, scale, tensor.dtype, weakref.ref(tensor), version)


def _unpack_feature(packed):
    source = packed.source() if isinstance(packed, _PackedFeature) else packed.tensor
    # Installing saved-tensor hooks disables PyTorch's usual version check.
    # Retain it for pass-through storage, and for compressed source objects
    # while they remain alive, without retaining the full-precision tape.
    if (
        source is not None
        and packed.version is not None
        and source._version != packed.version
    ):
        raise RuntimeError(
            "A tensor saved by feature compression was modified in place"
        )
    if isinstance(packed, _PackedFeature):
        restored = packed.compressed.to(dtype=packed.original_dtype) * packed.scale
        # Quantization can round the reconstructed largest finite FP32 value
        # above FP32's range. Saturate only finite packed values; pre-existing
        # NaN/Inf must remain visible to the predictor's finite-value checks.
        limit = torch.finfo(packed.original_dtype).max
        return torch.where(
            torch.isfinite(packed.compressed), restored.clamp(-limit, limit), restored
        )
    return packed.tensor


class _CompressedFeatureForward:
    """
    Limit lossy saved-tensor hooks to one parameter-free feature activation.

    Forward values stay unchanged. Backward and double backward use rounded
    saved features, so forces and Hessians are approximate. A forward inside
    torch.func/vmap is unsupported; vmap over an existing backward graph works.
    """

    def __init__(self, function, dtype, min_elements):
        self.function = function
        self.dtype = dtype
        self.min_elements = min_elements

    def _pack(self, tensor):
        return _pack_feature(tensor, _DTYPES[self.dtype], self.min_elements)

    def __call__(self, *args, **kwargs):
        if getattr(torch._C, "_are_functorch_transforms_active", lambda: False)():
            raise RuntimeError(
                "Saved-feature compression does not support a forward inside "
                "torch.func/vmap. Disable saved-feature compression for this transform."
            )
        if not torch.is_grad_enabled():
            return self.function(*args, **kwargs)
        with torch.autograd.graph.saved_tensors_hooks(self._pack, _unpack_feature):
            return self.function(*args, **kwargs)

    def __getstate__(self):
        # Pickling a bound method by its name would resolve to the replacement
        # forward and recursively invoke this wrapper after loading.
        return {
            "owner": self.function.__self__,
            "method": self.function.__func__,
            "dtype": self.dtype,
            "min_elements": self.min_elements,
        }

    def __setstate__(self, state):
        self.function = MethodType(state["method"], state["owner"])
        self.dtype = state["dtype"]
        self.min_elements = state["min_elements"]


def apply_saved_feature_compression(backbone, dtype="bf16", min_elements=1024):
    """
    Compress FP32 learned activation features saved for inference derivatives.

    This opt-in, lossy transformation wraps only parameter-free gate/SiLU/
    sigmoid modules. Geometry, normalization, weights and FP64 tensors are not
    compressed. It preserves module types and state_dict keys. Compilation,
    graph capture and activation checkpointing must be disabled by the caller.

    Returns a serializable manifest of wrapped modules. Reapplying the same
    configuration is idempotent; changing it requires a fresh backbone.
    """
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError("saved-feature dtype must be 'bf16' or 'fp16'")
    if (
        isinstance(min_elements, bool)
        or not isinstance(min_elements, Integral)
        or min_elements < 1
    ):
        raise ValueError("saved-feature min_elements must be a positive integer")
    min_elements = int(min_elements)
    if backbone.training or any(p.requires_grad for p in backbone.parameters()):
        raise ValueError(
            "Saved-feature compression requires eval mode and frozen weights"
        )
    backend = getattr(backbone, "backend", None)
    if backend is not None and type(backend) is not ExecutionBackend:
        raise ValueError("Saved-feature compression requires the general backend")

    targets = []
    visited = set()

    def visit(module, name):
        if id(module) in visited:
            return
        visited.add(id(module))
        if type(module) in _ACTIVATIONS:
            if getattr(module, "inplace", False) or any(
                True for _ in module.parameters()
            ):
                raise ValueError(
                    "Saved-feature activations must be parameter-free and out of place"
                )
            current = module.forward
            if isinstance(current, _CompressedFeatureForward):
                if (current.dtype, current.min_elements) != (dtype, min_elements):
                    raise ValueError(
                        "Saved-feature compression is already configured differently"
                    )
            elif not isinstance(current, MethodType) or current.__self__ is not module:
                raise ValueError("Cannot replace a customized activation forward")
            targets.append((name or "<root>", module, current))
            return  # A gate's children are covered by its single hook context.
        for child_name, child in module.named_children():
            visit(child, f"{name}.{child_name}" if name else child_name)

    visit(backbone, "")
    if not targets:
        raise ValueError("No supported learned-feature activations were found")
    # Validate every target before changing any module.
    for _, module, current in targets:
        if not isinstance(current, _CompressedFeatureForward):
            module.forward = _CompressedFeatureForward(current, dtype, min_elements)
    return {
        "dtype": dtype,
        "min_elements": min_elements,
        "module_count": len(targets),
        "modules": [name for name, _, _ in targets],
    }

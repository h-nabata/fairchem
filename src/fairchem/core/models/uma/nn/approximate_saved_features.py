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

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
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
    block_size: int | None = None


@dataclass(frozen=True)
class _UncompressedFeature:
    tensor: torch.Tensor
    version: int | None


def _pack_feature(tensor, dtype, min_elements, block_size=None):
    # Frozen parameters, their views and buffers never require gradients.
    # Returning detached tensors also avoids saved-output reference cycles.
    value = tensor.detach()
    version = None if tensor.is_inference() else tensor._version
    source_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    if (
        isinstance(tensor, torch.nn.Parameter)
        or not tensor.requires_grad
        or tensor.dtype != source_dtype
        or tensor.layout != torch.strided
        or tensor.numel() < min_elements
    ):
        return _UncompressedFeature(value, version)

    # A block scale avoids introducing infinities for finite source features
    # outside the storage dtype's range. The scale
    # stays on the same device, without scalar reads or host synchronization.
    # Nonfinite entries must remain visible, but must not disable scaling for
    # the finite entries sharing this activation tensor.
    # abs() owns this temporary, so replacing its nonfinite values in place
    # avoids a full mask and second floating-point tensor without touching the
    # saved source. The finite maximum (and hence the packed values) is unchanged.
    def pack_blocks(blocks):
        maximum = blocks.abs().nan_to_num_(nan=0, posinf=0, neginf=0).amax(dim=1)
        scale = (maximum / torch.finfo(dtype).max).clamp_min_(1)
        return (blocks / scale[:, None]).to(dtype), scale

    if block_size is None or tensor.numel() <= block_size:
        maximum = value.abs().nan_to_num_(nan=0, posinf=0, neginf=0).amax()
        scale = (maximum / torch.finfo(dtype).max).clamp_min_(1)
        packed = (value / scale).to(dtype=dtype, memory_format=torch.contiguous_format)
        return _PackedFeature(packed, scale, tensor.dtype, weakref.ref(tensor), version)
    flat = value.reshape(-1)
    full = flat.numel() // block_size * block_size
    packed, scale = pack_blocks(flat[:full].view(-1, block_size))
    packed = packed.flatten()
    if full < flat.numel():
        tail, tail_scale = pack_blocks(flat[full:].unsqueeze(0))
        packed = torch.cat((packed, tail.flatten()))
        scale = torch.cat((scale, tail_scale))
    return _PackedFeature(
        packed.reshape(value.shape),
        scale,
        tensor.dtype,
        weakref.ref(tensor),
        version,
        block_size,
    )


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
        # This upcast owns its storage. Reuse it for scaling and saturation
        # instead of retaining several full-precision temporary activations.
        restored = packed.compressed.to(dtype=packed.original_dtype, copy=True)
        block_size = getattr(packed, "block_size", None)
        if block_size is None:
            restored.mul_(packed.scale)
        else:
            flat = restored.view(-1)
            blocks = flat.numel() // block_size
            flat[: blocks * block_size].view(blocks, block_size).mul_(
                packed.scale[:blocks, None]
            )
            if blocks * block_size < flat.numel():
                flat[blocks * block_size :].mul_(packed.scale[-1])
        # Quantization can round the reconstructed largest finite source value
        # above its dtype's range. Saturate only finite packed values; pre-existing
        # NaN/Inf must remain visible to the predictor's finite-value checks.
        limit = torch.finfo(packed.original_dtype).max
        return torch.where(
            torch.isfinite(packed.compressed),
            restored.clamp_(-limit, limit),
            packed.compressed,
            out=restored,
        )
    return packed.tensor


@torch.inference_mode(False)
def saved_feature_roundtrip_stats(tensor, dtype="fp16", block_size=None):
    """Measure compression error and actual packed bytes for a representative activation."""
    if dtype not in _DTYPES:
        raise ValueError("Unsupported storage dtype")
    if block_size is not None and (type(block_size) is not int or block_size <= 0):
        raise ValueError("block_size must be a positive integer")
    source_dtype = torch.float64 if dtype == "fp32" else torch.float32
    if tensor.dtype != source_dtype or not tensor.numel():
        raise ValueError(
            "Expected a nonempty activation of the compression source dtype"
        )
    source = tensor.detach()
    if source.is_inference():
        source = source.clone()
    source.requires_grad_(True)
    packed = _pack_feature(source, _DTYPES[dtype], 1, block_size)
    restored = _unpack_feature(packed)
    finite = source.isfinite()
    nonzero = finite & (source != 0)
    eligible, lost = torch.stack(
        (nonzero.sum(), (nonzero & (restored == 0)).sum())
    ).tolist()
    return {
        "finite_nonzero_elements": eligible,
        "zeroed_elements": lost,
        "zeroed_fraction": lost / eligible if eligible else 0.0,
        "max_abs_error_finite": float(
            torch.where(finite, (restored - source).abs(), 0).amax().detach()
        ),
        "original_bytes": source.nbytes,
        "packed_bytes": packed.compressed.nbytes + packed.scale.nbytes,
    }


class _CompressedFeatureForward:
    """
    Limit lossy saved-tensor hooks to one parameter-free feature activation.

    Forward values stay unchanged. Backward and double backward use rounded
    saved features, so forces and Hessians are approximate. A forward inside
    torch.func/vmap is unsupported; vmap over an existing backward graph works.
    """

    def __init__(self, function, dtype, min_elements, block_size=None):
        self.function = function
        self.dtype = dtype
        self.min_elements = min_elements
        self.block_size = block_size

    def _pack(self, tensor):
        if self.block_size is None:
            return _pack_feature(tensor, _DTYPES[self.dtype], self.min_elements)
        return _pack_feature(
            tensor, _DTYPES[self.dtype], self.min_elements, self.block_size
        )

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
            "block_size": self.block_size,
            "dtype": self.dtype,
            "min_elements": self.min_elements,
        }

    def __setstate__(self, state):
        self.function = MethodType(state["method"], state["owner"])
        self.dtype = state["dtype"]
        self.min_elements = state["min_elements"]
        self.block_size = state.get("block_size")


def apply_saved_feature_compression(
    backbone, dtype="bf16", min_elements=1024, block_size=None
):
    """
    Compress learned activation features saved for inference derivatives.

    This opt-in, lossy transformation wraps only parameter-free gate/SiLU/
    sigmoid modules. BF16/FP16 storage applies only to FP32 activations; FP32
    storage applies only to FP64 activations. Geometry, normalization and weights
    are not compressed. Forward arithmetic and values stay at the original
    precision; derivative values are approximate. Module types and state_dict
    keys are preserved. Compilation,
    graph capture and activation checkpointing must be disabled by the caller.

    Returns a serializable manifest of wrapped modules. Reapplying the same
    configuration is idempotent; changing it requires a fresh backbone.
    """
    if not isinstance(dtype, str) or dtype not in _DTYPES:
        raise ValueError("saved-feature dtype must be 'bf16', 'fp16' or 'fp32'")
    if (
        isinstance(min_elements, bool)
        or not isinstance(min_elements, Integral)
        or min_elements < 1
    ):
        raise ValueError("saved-feature min_elements must be a positive integer")
    min_elements = int(min_elements)
    if block_size is not None and (type(block_size) is not int or block_size <= 0):
        raise ValueError("saved-feature block_size must be a positive integer")
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
                if (current.dtype, current.min_elements, current.block_size) != (
                    dtype,
                    min_elements,
                    block_size,
                ):
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
            module.forward = _CompressedFeatureForward(
                current, dtype, min_elements, block_size
            )
    return {
        "dtype": dtype,
        "source_dtype": "float64" if dtype == "fp32" else "float32",
        "forward_unchanged": True,
        "derivatives_approximate": True,
        "min_elements": min_elements,
        "block_size": block_size,
        "module_count": len(targets),
        "modules": [name for name, _, _ in targets],
    }

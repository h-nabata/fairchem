"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Explicit, inference-only approximations for ordinary learned Linear layers.

These experimental B-series operations are disabled unless requested. Geometry,
SO3 contractions, functional linears and fused backends are outside this module's
coverage. Reduced GEMMs round activations, so their straight-through derivatives
are approximations; energy conservation and accuracy require application tests.

Native reduced GEMMs use FP16/BF16 operands and FP32 outputs. CPU emulation rounds
the operands but computes in FP32: it is a numerical reference, not acceleration.
Backward deliberately uses FP32 GEMM and reconstructed rounded frozen weights.
Finite checks synchronize accelerators on every forward call. No speedup is
promised, including for weight dequantization and unfused split GEMMs.
"""

from __future__ import annotations

import fnmatch
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

GEMM_PRECISIONS = frozenset({"fp32", "bf16", "fp16", "bf16_split2", "bf16_split3"})


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"ApproximateLinear: non-finite {name}")


def _components(tensor: torch.Tensor, precision: str) -> list[torch.Tensor]:
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    count = {"bf16_split2": 2, "bf16_split3": 3}.get(precision, 1)
    residual = tensor
    components = []
    for index in range(count):
        component = residual.to(dtype)
        components.append(component)
        if index + 1 < count:
            residual = residual - component.float()
    return components


def _reconstructed_weight(weight: torch.Tensor, precision: str) -> torch.Tensor:
    components = _components(weight, precision)
    reconstructed = components[-1].float()
    for component in reversed(components[:-1]):
        reconstructed = reconstructed + component.float()
    return reconstructed


def _mm_fp32_output(
    left: torch.Tensor, right: torch.Tensor, emulation: bool
) -> torch.Tensor:
    if left.device.type == "cpu":
        if not emulation:
            raise RuntimeError(
                "Reduced GEMM on CPU requires low_precision_emulation=True; "
                "this computes rounded operands in FP32 without a speed claim."
            )
        result = torch.mm(left.float(), right.float())
    elif left.device.type in {"xpu", "cuda"}:
        if left.device.type == "cuda":
            dtype_name = "fp16" if left.dtype == torch.float16 else "bf16"
            flag_name = f"allow_{dtype_name}_reduced_precision_reduction"
            flag = getattr(torch.backends.cuda.matmul, flag_name, None)
            enabled = any(flag) if isinstance(flag, tuple) else flag
            if enabled is not False:
                raise RuntimeError(
                    "Reduced GEMM requires FP32 partial reductions: set "
                    f"torch.backends.cuda.matmul.{flag_name}=False before execution. "
                    "ApproximateLinear does not mutate process precision settings."
                )
            if left.dtype == torch.float16 and getattr(
                torch.backends.cuda.matmul, "allow_fp16_accumulation", False
            ):
                raise RuntimeError(
                    "Reduced GEMM requires FP32 accumulation: set "
                    "torch.backends.cuda.matmul.allow_fp16_accumulation=False "
                    "before execution."
                )
        # This overload has no native autograd derivative in supported PyTorch
        # versions. _ReducedGEMM supplies the input derivative explicitly.
        result = torch.mm(left, right, out_dtype=torch.float32)
    else:
        raise RuntimeError("Reduced GEMM requires XPU, CUDA, or explicit CPU emulation")
    if result.dtype != torch.float32:
        raise RuntimeError("Reduced GEMM backend did not honor FP32 output dtype")
    return result


class _ReducedGEMM(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x, weight, precision, emulation):
        xs = _components(x, precision)
        ws = _components(weight, precision)
        for component in (*xs, *ws):
            _require_finite(component, "reduced-precision operand (cast overflow)")
        # All 4 / 9 products, smallest component pairs first. These names
        # describe this algorithm, not Intel oneMKL's bf16x2 / bf16x3 modes.
        pairs = sorted(
            ((i, j) for i in range(len(xs)) for j in range(len(ws))),
            key=lambda pair: sum(pair),
            reverse=True,
        )
        result = None
        for i, j in pairs:
            term = _mm_fp32_output(xs[i], ws[j].T, emulation)
            result = term if result is None else result + term
        return result

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, weight, precision, _ = inputs
        ctx.precision = precision
        ctx.output_shape = (x.shape[0], weight.shape[0])
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(weight)
        ctx.save_for_forward(weight)

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None
        (weight,) = ctx.saved_tensors
        # No downcast of grad_output. The weights are frozen, so this expression
        # preserves higher coordinate derivatives through grad_output. No item()
        # checks here: Hessian code vmaps this backward over cotangents.
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            effective_weight = _reconstructed_weight(weight, ctx.precision)
            return grad_output @ effective_weight, None, None, None

    @staticmethod
    def jvp(ctx, x_tangent, weight_tangent, precision_tangent, emulation_tangent):
        (weight,) = ctx.saved_tensors
        if weight_tangent is not None:
            raise RuntimeError("ApproximateLinear supports frozen weights only")
        if x_tangent is None:
            return weight.new_zeros(ctx.output_shape)
        with torch.autocast(device_type=x_tangent.device.type, enabled=False):
            return x_tangent @ _reconstructed_weight(weight, ctx.precision).T


def _unpack_weight(packed, scale, rows, columns, bits):
    if scale.dtype != torch.float32:
        raise TypeError("ApproximateLinear quantization scales must remain FP32")
    if packed.dtype != (torch.uint8 if bits == 4 else torch.int8):
        raise TypeError("ApproximateLinear packed weight has the wrong integer dtype")
    if bits == 4:
        packed = packed.to(torch.int16)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=1).flatten()
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
        quantized = signed[: rows * columns].reshape(rows, columns)
    else:
        quantized = packed
    return quantized.float() * scale[:, None]


class _PackedGEMM(torch.autograd.Function):
    """
    Retain packed tensors, not an expanded FP32 weight, for force backward.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x, packed, scale, rows, columns, bits, precision, emulation):
        weight = _unpack_weight(packed, scale, rows, columns, bits)
        _require_finite(weight, "dequantized weight")
        return (
            F.linear(x, weight)
            if precision == "fp32"
            else _ReducedGEMM.forward(x, weight, precision, emulation)
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, packed, scale, rows, columns, bits, precision, _ = inputs
        ctx.weight_spec = (rows, columns, bits)
        ctx.precision = precision
        ctx.output_shape = (x.shape[0], rows)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(packed, scale)
        ctx.save_for_forward(packed, scale)

    @staticmethod
    def _weight(ctx):
        packed, scale = ctx.saved_tensors
        weight = _unpack_weight(packed, scale, *ctx.weight_spec)
        if ctx.precision != "fp32":
            weight = _reconstructed_weight(weight, ctx.precision)
        return weight

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 8
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            gradient = grad_output @ _PackedGEMM._weight(ctx)
        return gradient, None, None, None, None, None, None, None

    @staticmethod
    def jvp(ctx, x_tangent, *other_tangents):
        if any(tangent is not None for tangent in other_tangents):
            raise RuntimeError("ApproximateLinear supports frozen weights only")
        weight = _PackedGEMM._weight(ctx)
        if x_tangent is None:
            return weight.new_zeros(ctx.output_shape)
        with torch.autocast(device_type=x_tangent.device.type, enabled=False):
            return x_tangent @ weight.T


class _StoredWeight(nn.Module):
    """
    FP32 or symmetric per-row integer weight storage, with no FP32 shadow.
    """

    def __init__(self, weight: torch.Tensor, bits: int | None) -> None:
        super().__init__()
        self.rows, self.columns = weight.shape
        self.bits = bits
        if bits is None:
            self.register_buffer("values", weight.detach().clone())
            self.register_buffer("packed", None)
            self.register_buffer("scale", None)
        else:
            qmax = (1 << (bits - 1)) - 1
            maximum = weight.abs().amax(dim=1)
            scale = torch.where(
                maximum == 0,
                torch.ones_like(maximum),
                (maximum / qmax).clamp_min(torch.finfo(torch.float32).tiny),
            )
            quantized = (weight / scale[:, None]).round().clamp(-qmax, qmax)
            quantized = quantized.to(torch.int8)
            if bits == 4:
                nibbles = quantized.flatten().to(torch.int16) & 15
                if nibbles.numel() % 2:
                    nibbles = F.pad(nibbles, (0, 1))
                packed = (nibbles[0::2] | (nibbles[1::2] << 4)).to(torch.uint8)
            else:
                packed = quantized
            self.register_buffer("values", None)
            self.register_buffer("packed", packed)
            self.register_buffer("scale", scale)

    def dense(self) -> torch.Tensor:
        if self.bits is None:
            return self.values
        return _unpack_weight(
            self.packed, self.scale, self.rows, self.columns, self.bits
        )

    def get_extra_state(self):
        return {
            "version": 1,
            "rows": self.rows,
            "columns": self.columns,
            "bits": self.bits,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError(
                "ApproximateLinear packed weight metadata does not match"
            )

    def storage_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers())


class ApproximateLinear(nn.Module):
    """
    Frozen learned linear, with optional factorization and packed weights.

    Create through :func:`apply_linear_approximations`. Changing its dtype away
    from FP32 is unsupported and fails at execution. Moving device is supported;
    integer buffers retain their packed integer dtype under ``Module.to``.
    Backward vmap and directional JVP are supported. Forward vmap/jacfwd and
    compilation are unsupported because finite checks use host decisions.
    """

    def __init__(
        self,
        source: nn.Linear,
        *,
        gemm_precision: str,
        weight_bits: int | None,
        low_rank: int | None,
        low_precision_emulation: bool,
    ) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.gemm_precision = gemm_precision
        self.low_precision_emulation = low_precision_emulation
        self.low_rank = low_rank
        self.retained_singular_energy = None
        weight = source.weight.detach()
        if low_rank is None:
            factors = [weight]
        else:
            # Offline preparation only. FP64 SVD is on CPU even for GPU callers.
            u, s, vh = torch.linalg.svd(weight.cpu().double(), full_matrices=False)
            total = s.square().sum()
            self.retained_singular_energy = (
                float(s[:low_rank].square().sum() / total) if total > 0 else 1.0
            )
            factors = [
                vh[:low_rank].to(device=weight.device, dtype=torch.float32),
                (u[:, :low_rank] * s[:low_rank]).to(
                    device=weight.device, dtype=torch.float32
                ),
            ]
        self.factors = nn.ModuleList(
            [_StoredWeight(factor, weight_bits) for factor in factors]
        )
        for factor in factors:
            _require_finite(factor, "prepared weight factor")
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().clone()
        )
        self.eval()

    def materialized_weight(self) -> torch.Tensor:
        """
        Return an FP32 diagnostic copy/view of the represented full weight.
        """
        if len(self.factors) == 1:
            return self.factors[0].dense()
        right = self.factors[0].dense()
        with torch.autocast(device_type=right.device.type, enabled=False):
            return self.factors[1].dense() @ right

    def get_extra_state(self):
        return {
            "version": 1,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "gemm_precision": self.gemm_precision,
            "low_precision_emulation": self.low_precision_emulation,
            "low_rank": self.low_rank,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError(
                "ApproximateLinear architecture or policy metadata does not match"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x)

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("ApproximateLinear is inference-only; call eval()")
        if x.dtype != torch.float32:
            raise TypeError("ApproximateLinear requires FP32 inputs")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                "ApproximateLinear input has an incompatible last dimension"
            )
        _require_finite(x, "input")
        leading = x.shape[:-1]
        value = x.reshape(-1, self.in_features)
        for factor in self.factors:
            if factor.bits is not None:
                if factor.packed.device != x.device or factor.scale.device != x.device:
                    raise ValueError(
                        "ApproximateLinear input and weights must share a device"
                    )
                value = _PackedGEMM.apply(
                    value,
                    factor.packed,
                    factor.scale,
                    factor.rows,
                    factor.columns,
                    factor.bits,
                    self.gemm_precision,
                    self.low_precision_emulation,
                )
                _require_finite(value, "GEMM output")
                continue
            weight = factor.dense()
            if weight.dtype != torch.float32:
                raise TypeError("ApproximateLinear weight storage must remain FP32")
            if weight.device != x.device:
                raise ValueError(
                    "ApproximateLinear input and weights must share a device"
                )
            _require_finite(weight, "weight")
            value = (
                F.linear(value, weight)
                if self.gemm_precision == "fp32"
                else _ReducedGEMM.apply(
                    value, weight, self.gemm_precision, self.low_precision_emulation
                )
            )
            _require_finite(value, "GEMM output")
        if self.bias is not None:
            if self.bias.dtype != torch.float32:
                raise TypeError("ApproximateLinear bias storage must remain FP32")
            _require_finite(self.bias, "bias")
            value = value + self.bias
        _require_finite(value, "output")
        return value.reshape(*leading, self.out_features)


def apply_linear_approximations(
    model: nn.Module,
    *,
    gemm_precision: str = "fp32",
    weight_bits: int | None = None,
    low_rank_fraction: float | None = None,
    low_precision_emulation: bool = False,
    exclude_patterns: tuple[str, ...] = (),
) -> dict:
    """
    Replace ordinary frozen FP32 ``nn.Linear`` children; return a manifest.

    Apply after composition-specific MoLE merging and parameter freezing, before
    compilation/device migration. The caller must select a general backend: no
    functional, fused, SO3/einsum, or custom Linear-subclass calls are rewritten.
    Reapplying to an already transformed model raises rather than silently mixing
    policies. Exclusions use case-sensitive shell patterns on qualified names.
    Low-rank plus integer quantization is deliberately unsupported initially.
    """
    if not isinstance(gemm_precision, str) or gemm_precision not in GEMM_PRECISIONS:
        raise ValueError(f"Unsupported gemm_precision: {gemm_precision!r}")
    if weight_bits is not None and (
        isinstance(weight_bits, bool)
        or not isinstance(weight_bits, int)
        or weight_bits not in (4, 8)
    ):
        raise ValueError("weight_bits must be None, 4, or 8")
    if low_rank_fraction is not None and (
        isinstance(low_rank_fraction, bool)
        or not isinstance(low_rank_fraction, Real)
        or not math.isfinite(low_rank_fraction)
        or not 0 < low_rank_fraction <= 1
    ):
        raise ValueError("low_rank_fraction must be finite and in (0, 1]")
    if weight_bits is not None and low_rank_fraction is not None:
        raise ValueError("Combining low-rank and integer quantization is unsupported")
    if not isinstance(low_precision_emulation, bool):
        raise ValueError("low_precision_emulation must be a boolean")
    if isinstance(exclude_patterns, (str, bytes)) or not isinstance(
        exclude_patterns, (tuple, list)
    ):
        raise ValueError("exclude_patterns must be a sequence of nonempty strings")
    if any(not isinstance(p, str) or not p for p in exclude_patterns):
        raise ValueError("exclude_patterns must be a sequence of nonempty strings")
    active = (
        gemm_precision != "fp32"
        or weight_bits is not None
        or low_rank_fraction is not None
    )
    manifest = {
        "active": active,
        "effective": False,
        "gemm_precision": gemm_precision,
        "weight_bits": weight_bits,
        "low_rank_fraction": (
            None if low_rank_fraction is None else float(low_rank_fraction)
        ),
        "low_precision_emulation": low_precision_emulation,
        "exclude_patterns": list(exclude_patterns),
        "coverage": "ordinary nn.Linear children only; no functional, fused or SO3 contractions",
        "backward_precision": "fp32",
        "converted": [],
        "skipped": [],
        "original_weight_bytes": 0,
        "stored_weight_bytes": 0,
        "storage_accounting": (
            "Per unique Linear module; excludes bias/workspaces and does not deduplicate "
            "tied parameter storage across distinct Linear modules."
        ),
        "warnings": [],
    }
    if not active:
        return manifest
    if isinstance(model, (nn.Linear, ApproximateLinear)):
        raise ValueError("Pass a parent module containing Linear children")
    if any(isinstance(module, ApproximateLinear) for module in model.modules()):
        raise ValueError("Linear approximations were already applied; reload the model")

    # Collect aliases too, preserving shared-module identity and honoring an
    # exclusion on any alias. Stage all replacements before mutating the model.
    groups: dict[int, list] = {}

    def collect(parent, prefix, ancestors=frozenset()):
        if id(parent) in ancestors:
            return
        ancestors = ancestors | {id(parent)}
        for name, child in parent._modules.items():
            if child is None:
                continue
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear):
                groups.setdefault(id(child), []).append((parent, name, path, child))
            else:
                collect(child, path, ancestors)

    collect(model, "")
    replacements = []
    for aliases in groups.values():
        source = aliases[0][3]
        paths = [alias[2] for alias in aliases]
        reason = None
        rank = None
        if type(source) is not nn.Linear:
            reason = "custom Linear subclass is outside coverage"
        elif (
            source._forward_hooks
            or source._forward_pre_hooks
            or source._backward_hooks
            or source._backward_pre_hooks
        ):
            reason = "module hooks require the original Linear implementation"
        elif any(
            fnmatch.fnmatchcase(path, pattern)
            for path in paths
            for pattern in exclude_patterns
        ):
            reason = "excluded (including shared aliases)"
        elif not source.in_features or not source.out_features:
            reason = "empty weight"
        elif low_rank_fraction is not None:
            rank = math.ceil(
                low_rank_fraction * min(source.in_features, source.out_features)
            )
            if (
                rank * (source.in_features + source.out_features)
                >= source.weight.numel()
            ):
                rank = None
                if gemm_precision == "fp32":
                    reason = "requested rank does not reduce weight or GEMM size"
        if reason is not None:
            manifest["skipped"].append({"names": paths, "reason": reason})
            continue
        for name, tensor in source.named_parameters():
            if tensor.requires_grad:
                raise ValueError(
                    f"{paths[0]}.{name} must be frozen before approximation"
                )
            if tensor.dtype != torch.float32:
                raise ValueError(f"{paths[0]}.{name} must use FP32")
            _require_finite(tensor, f"{paths[0]}.{name}")
        replacement = ApproximateLinear(
            source,
            gemm_precision=gemm_precision,
            weight_bits=weight_bits,
            low_rank=rank,
            low_precision_emulation=low_precision_emulation,
        )
        replacements.append((aliases, replacement))
        original_bytes = source.weight.numel() * source.weight.element_size()
        stored_bytes = sum(factor.storage_bytes() for factor in replacement.factors)
        manifest["original_weight_bytes"] += original_bytes
        manifest["stored_weight_bytes"] += stored_bytes
        manifest["converted"].append(
            {
                "names": paths,
                "shape": [source.out_features, source.in_features],
                "rank": rank,
                "retained_singular_energy": replacement.retained_singular_energy,
                "original_weight_bytes": original_bytes,
                "stored_weight_bytes": stored_bytes,
            }
        )
    for aliases, replacement in replacements:
        for parent, name, _, _ in aliases:
            parent._modules[name] = replacement
    manifest["effective"] = bool(replacements)
    manifest["warnings"] = [
        "Experimental approximation; application energy/force/stress/curvature validation is required.",
        "Forward finite checks synchronize accelerators; speedup is not guaranteed.",
    ]
    if gemm_precision != "fp32":
        manifest["warnings"].append(
            "Reduced activation casts use surrogate coordinate derivatives; backward GEMMs stay FP32."
        )
        if low_precision_emulation:
            manifest["warnings"].append(
                "CPU uses rounded-input FP32 emulation, not native reduced-precision acceleration."
            )
    if low_rank_fraction is not None:
        manifest["warnings"].append(
            "SVD truncation is not a retrained compressed checkpoint."
        )
    if not replacements:
        manifest["warnings"].append(
            "No eligible Linear was transformed; the requested approximation has no effect."
        )
    return manifest

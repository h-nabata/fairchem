"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class SeriesBSettings:
    """
    Explicit, default-off approximations for an independently owned predictor.

    Workflow approximations (distillation, RESPA, relaxation and curvature
    estimation) have separate entry points; selecting these settings never
    changes a model architecture or starts an optimization or simulation.
    """

    gemm_precision: str = "fp32"
    low_precision_emulation: bool = False
    weight_bits: int | None = None
    low_rank_fraction: float | None = None
    saved_feature_dtype: str | None = None
    saved_feature_min_elements: int = 1024
    compensated_reduction: bool = False
    radial_table_points: int | None = None
    radial_fast_exp: bool = False
    onednn_tf32: bool = False

    def __post_init__(self):
        if self.gemm_precision not in {
            "fp32",
            "bf16",
            "fp16",
            "bf16_split2",
            "bf16_split3",
        }:
            raise ValueError("Unsupported series_b.gemm_precision")
        for name in (
            "low_precision_emulation",
            "compensated_reduction",
            "radial_fast_exp",
            "onednn_tf32",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"series_b.{name} must be a boolean")
        if self.weight_bits is not None and (
            isinstance(self.weight_bits, bool)
            or not isinstance(self.weight_bits, Integral)
            or self.weight_bits not in (4, 8)
        ):
            raise ValueError("series_b.weight_bits must be 4, 8 or None")
        if self.low_rank_fraction is not None:
            value = self.low_rank_fraction
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or not 0 < value <= 1
            ):
                raise ValueError("series_b.low_rank_fraction must be in (0, 1]")
            if self.weight_bits is not None:
                raise ValueError("Select weight quantization or low rank separately")
        if self.saved_feature_dtype not in (None, "bf16", "fp16"):
            raise ValueError("series_b.saved_feature_dtype must be bf16, fp16 or None")
        for name, minimum in (
            ("saved_feature_min_elements", 1),
            ("radial_table_points", 2),
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(f"series_b.{name} must be an integer >= {minimum}")
        if self.saved_feature_min_elements is None:
            raise ValueError("series_b.saved_feature_min_elements cannot be None")
        if self.radial_fast_exp and self.radial_table_points is not None:
            raise ValueError(
                "Select radial fast exp or radial interpolation separately"
            )
        if self.low_precision_emulation and self.gemm_precision == "fp32":
            raise ValueError("low_precision_emulation requires a reduced GEMM mode")
        for name in (
            "weight_bits",
            "saved_feature_min_elements",
            "radial_table_points",
        ):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, int(getattr(self, name)))
        if self.low_rank_fraction is not None:
            object.__setattr__(self, "low_rank_fraction", float(self.low_rank_fraction))

    @property
    def linear_enabled(self) -> bool:
        return (
            self.gemm_precision != "fp32"
            or self.weight_bits is not None
            or self.low_rank_fraction is not None
        )

    @property
    def enabled(self) -> bool:
        return (
            self.linear_enabled
            or self.saved_feature_dtype is not None
            or self.compensated_reduction
            or self.radial_table_points is not None
            or self.radial_fast_exp
            or self.onednn_tf32
        )

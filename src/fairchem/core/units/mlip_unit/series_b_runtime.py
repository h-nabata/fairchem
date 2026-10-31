"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch


def validate_series_b(settings, device, backbone=None):
    """
    Fail before approximation reaches an unsupported backend or graph policy.
    """
    settings.__post_init__()
    options = settings.series_b
    if not options.enabled:
        return
    device_type = torch.device(device).type
    if options.onednn_tf32 and device_type != "xpu":
        raise ValueError("series_b.onednn_tf32 is only available on XPU")
    if options.gemm_precision != "fp32":
        if device_type == "cpu" and not options.low_precision_emulation:
            raise ValueError("CPU reduced GEMM requires low_precision_emulation=True")
        if device_type != "cpu" and options.low_precision_emulation:
            raise ValueError("low_precision_emulation is only a CPU reference mode")
        if device_type not in ("cpu", "cuda", "xpu"):
            raise ValueError("Reduced GEMM supports only CPU reference, CUDA or XPU")
    if backbone is not None:
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone
        from fairchem.core.models.uma.nn.execution_backends import get_execution_backend

        if not isinstance(backbone, eSCNMDBackbone):
            raise ValueError(
                "Series B model transformations currently support UMA only"
            )
        if type(backbone.backend) is not type(get_execution_backend("general")):
            raise ValueError(
                "Series B cannot be combined with an overridden fast backend"
            )
        if any(
            getattr(module, "activation_checkpoint_chunk_size", None) is not None
            for module in backbone.modules()
        ):
            raise ValueError("Series B cannot use overridden activation checkpointing")


def prepare_series_b(model, settings, device):
    """
    Transform the private, prepared backbone; keep geometry and heads at FP32.
    """
    validate_series_b(settings, device, model.backbone)
    options = settings.series_b
    manifest = {"enabled": options.enabled, "backbone_only": True}
    if not options.enabled:
        return manifest
    if options.linear_enabled:
        from fairchem.core.models.uma.nn.approximate_linear import (
            apply_linear_approximations,
        )

        manifest["linear"] = apply_linear_approximations(
            model.backbone,
            gemm_precision=options.gemm_precision,
            weight_bits=options.weight_bits,
            low_rank_fraction=options.low_rank_fraction,
            low_precision_emulation=options.low_precision_emulation,
        )
    if options.radial_table_points is not None or options.radial_fast_exp:
        from fairchem.core.models.uma.nn.approximate_ops import (
            apply_feature_approximations,
        )

        manifest["radial"] = apply_feature_approximations(model.backbone, options)
    if options.saved_feature_dtype is not None:
        from fairchem.core.models.uma.nn.approximate_saved_features import (
            apply_saved_feature_compression,
        )

        manifest["saved_features"] = apply_saved_feature_compression(
            model.backbone,
            dtype=options.saved_feature_dtype,
            min_elements=options.saved_feature_min_elements,
        )
    manifest["compensated_reduction"] = options.compensated_reduction
    manifest["onednn_tf32"] = options.onednn_tf32
    return manifest

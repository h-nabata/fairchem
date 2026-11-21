"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch


def onednn_matmul_policy():
    """Read PyTorch's effective oneDNN matmul policy on modern and legacy builds."""
    backend = torch.backends.mkldnn
    return (
        getattr(getattr(backend, "matmul", None), "fp32_precision", None),
        getattr(backend, "allow_tf32", None),
    )


def matmul_precision_key(device_type):
    """Return hashable policy fields for precomputed tensors and captured kernels."""
    if device_type in {"cpu", "xpu"}:
        modern, legacy = onednn_matmul_policy()
    elif device_type == "cuda":
        modern = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
        legacy = None
    else:
        modern = legacy = None

    # When modern per-backend flags are set, the legacy global getter raises
    # even when the flag belongs to a different device from this operation.
    try:
        global_precision = torch.get_float32_matmul_precision()
    except RuntimeError:
        if not hasattr(torch.backends, "fp32_precision"):
            raise
        global_precision = torch.backends.fp32_precision
    return global_precision, modern, legacy

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn import functional as F

if TYPE_CHECKING:
    import torch


def linear_with_folded_batch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply a frozen linear operation with the leading dimensions folded together.

    Non-contiguous SO(2) slices with frozen weights can dispatch one small matrix
    multiplication per edge, even when no gradients are recorded. Folding the
    leading dimensions gives one two-dimensional GEMM, and also avoids saving
    expanded weights for input derivatives. A viewable input needs no copy.
    Keep the native path for trainable weights. This also covers energy-only and
    direct-force inference, as proposed in upstream fairchem PR #2264.

    Args:
        x: Input to the linear operation.
        weight: Linear weight used to determine whether gradients are required.
        bias: Optional linear bias.

    Returns:
        The linear result with the original leading dimensions restored.
    """
    should_fold = x.ndim > 2 and not weight.requires_grad
    x_shape = x.shape
    linear_input = x.flatten(0, -2) if should_fold else x
    output = F.linear(linear_input, weight, bias)
    return output.reshape(*x_shape[:-1], output.shape[-1]) if should_fold else output

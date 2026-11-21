"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from threading import RLock

# Scoped model construction and distillation both temporarily change the CPU
# RNG; construction also changes the default dtype. They must share one lock so
# that overlapping fairchem jobs cannot restore each other's temporary state.
# External code using these process globals still needs caller coordination.
MODEL_STATE_LOCK = RLock()

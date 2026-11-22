"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from traceback import format_tb


def clear_exception_tracebacks(
    error: BaseException, *, preserve_locations: bool = False
) -> None:
    """Drop references to model frames, optionally retaining their formatted locations."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if preserve_locations and current.__traceback__ is not None:
            stack = "".join(format_tb(current.__traceback__))
            current.add_note(
                f"Worker traceback (most recent call last):\n{stack.rstrip()}"
            )
        current.__traceback__ = None
        pending.extend(
            cause
            for cause in (current.__cause__, current.__context__)
            if cause is not None
        )
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)

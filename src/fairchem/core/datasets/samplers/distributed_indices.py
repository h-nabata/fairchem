"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from itertools import chain

import torch


def distributed_indices(size, total_size, rank, num_replicas, shuffle, seed, start):
    """Return the remaining rank-local slice of a PyTorch distributed sampling epoch."""
    first = rank + start * num_replicas
    if first >= total_size:
        return iter(())
    regular = range(first, min(size, total_size), num_replicas)
    padding = range(first + len(regular) * num_replicas, total_size, num_replicas)
    if not shuffle:
        return chain(regular, (position % size for position in padding))

    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(size, generator=generator, device="cpu")
    local = permutation[regular.start : regular.stop : regular.step].tolist()
    local.extend(permutation[position % size].item() for position in padding)
    return iter(local)

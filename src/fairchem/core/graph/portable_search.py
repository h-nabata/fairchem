"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _integer_cartesian_entry(x: int, y: int, z: int, device: torch.device):
    """
    Own immutable image/stencil tables; callers must not modify the result.
    """
    table = torch.cartesian_prod(
        torch.arange(-x, x + 1, device=device),
        torch.arange(-y, y + 1, device=device),
        torch.arange(-z, z + 1, device=device),
    )
    event = None
    if device.type in {"cuda", "xpu"}:
        backend = getattr(torch, device.type)
        event = backend.Event()
        event.record(backend.current_stream(device))
    return table, event


def integer_cartesian_table(x: int, y: int, z: int, device: torch.device):
    """
    Read an immutable table with cross-stream readiness and lifetime tracking.
    """
    # Entry count alone does not bound memory for tiny periodic cells with
    # millions of images. Large tables are valid, but are not retained.
    builder = _integer_cartesian_entry
    if (2 * x + 1) * (2 * y + 1) * (2 * z + 1) > 32768:
        builder = _integer_cartesian_entry.__wrapped__
    table, event = builder(x, y, z, device)
    if event is not None:
        stream = getattr(torch, device.type).current_stream(device)
        stream.wait_event(event)
        table.record_stream(stream)
    return table


def source_cloud(
    atom_pos,
    target_pos,
    target_images,
    pbc_offsets,
    unit_cell,
    cells_per_image,
    natoms,
    prune=None,
):
    """
    Generate source images, optionally discarding those outside target bounds.

    Bounded image tiles avoid the full atoms-times-images position allocation.
    The output keeps atom-major ordering after conservative bounding-box pruning.
    """
    device = atom_pos.device
    if prune is None:
        counts = torch.repeat_interleave(cells_per_image, natoms)
        sources = torch.repeat_interleave(
            torch.arange(len(atom_pos), device=device), counts
        )
        images = []
        start = 0
        for count, n in zip(cells_per_image.tolist(), natoms.tolist()):
            images.append(torch.arange(start, start + count, device=device).repeat(n))
            start += count
        images = torch.cat(images)
        return sources, images, atom_pos[sources] + pbc_offsets[images]
    sources, images, keys = [], [], []
    atom_start = image_start = 0
    for system, (n, count) in enumerate(zip(natoms.tolist(), cells_per_image.tolist())):
        target = target_pos[target_images == system]
        if n and len(target):
            lower, upper = target.amin(0), target.amax(0)
            atoms = atom_pos[atom_start : atom_start + n]
            margin = (
                32
                * torch.finfo(atom_pos.dtype).eps
                * (
                    atoms.abs().amax()
                    + pbc_offsets[image_start : image_start + count].abs().amax()
                    + prune
                    + 1
                )
            )
            for chunk in range(0, count, 16):
                image = torch.arange(
                    image_start + chunk,
                    image_start + min(chunk + 16, count),
                    device=device,
                )
                positions = atoms[:, None] + pbc_offsets[image]
                delta = (lower - positions).clamp(min=0) + (positions - upper).clamp(
                    min=0
                )
                selected = delta.square().sum(-1) <= (prune + margin) ** 2
                atom, tile = torch.where(selected)
                sources.append(atom + atom_start)
                images.append(image[tile])
                keys.append((atom + atom_start) * len(unit_cell) + image[tile])
        atom_start += n
        image_start += count
    if not sources:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, atom_pos.new_empty((0, 3))
    sources, images, keys = torch.cat(sources), torch.cat(images), torch.cat(keys)
    order = torch.argsort(keys)
    sources, images = sources[order], images[order]
    return sources, images, atom_pos[sources] + pbc_offsets[images]


def alternative_candidates(
    source_pos, target_pos, source_images, target_images, radius, method
):
    """
    Return exact candidates from tiled all-pairs or a balanced bounding-volume tree.
    """
    device = source_pos.device
    if method == "all_pairs":
        source_parts, target_parts = [], []
        tile = max(1, 262144 // max(1, len(source_pos)))
        for start in range(0, len(target_pos), tile):
            target = target_pos[start : start + tile]
            distance2 = (target[:, None] - source_pos).square().sum(-1)
            keep = (distance2 <= radius * radius) & (
                target_images[start : start + tile, None] == source_images
            )
            targets, sources = torch.where(keep)
            source_parts.append(sources)
            target_parts.append(targets + start)
        return torch.cat(source_parts), torch.cat(target_parts)

    # Median-split tree with eight sources per leaf. Exact bounding boxes, not
    # the longest-axis sorting heuristic, decide which leaves may be discarded.
    leaf_size = 8
    axis = int((source_pos.amax(0) - source_pos.amin(0)).argmax())
    order = torch.argsort(source_pos[:, axis], stable=True)
    leaves = 1 << max(
        0, math.ceil(math.log2(max(1, math.ceil(len(order) / leaf_size))))
    )
    padded = source_pos.new_full((leaves * leaf_size, 3), torch.inf)
    padded[: len(order)] = source_pos[order]
    lower = source_pos.new_full((2 * leaves, 3), torch.inf)
    upper = source_pos.new_full((2 * leaves, 3), -torch.inf)
    lower[leaves:] = padded.view(leaves, leaf_size, 3).amin(1)
    padded[len(order) :] = -torch.inf
    upper[leaves:] = padded.view(leaves, leaf_size, 3).amax(1)
    width = leaves
    while width > 1:
        lower[width // 2 : width] = lower[width : 2 * width].view(-1, 2, 3).amin(1)
        upper[width // 2 : width] = upper[width : 2 * width].view(-1, 2, 3).amax(1)
        width //= 2
    targets = torch.arange(len(target_pos), device=device)
    nodes = torch.ones_like(targets)
    margin = (
        32
        * torch.finfo(source_pos.dtype).eps
        * (source_pos.abs().amax() + target_pos.abs().amax() + radius + 1)
    )
    for _ in range(leaves.bit_length()):
        positions = target_pos[targets]
        delta = (lower[nodes] - positions).clamp(min=0) + (
            positions - upper[nodes]
        ).clamp(min=0)
        keep = delta.square().sum(-1) <= (radius + margin) ** 2
        targets, nodes = targets[keep], nodes[keep]
        if width == leaves:
            break
        width *= 2
        targets = targets.repeat_interleave(2)
        nodes = (nodes[:, None] * 2 + torch.tensor([0, 1], device=device)).flatten()
    leaf_indices = (nodes[:, None] - leaves) * leaf_size + torch.arange(
        leaf_size, device=device
    )
    valid = leaf_indices < len(order)
    expanded_targets = targets[:, None].expand(-1, leaf_size)[valid]
    expanded_sources = order[leaf_indices[valid]]
    same_system = target_images[expanded_targets] == source_images[expanded_sources]
    expanded_targets, expanded_sources = (
        expanded_targets[same_system],
        expanded_sources[same_system],
    )
    sort = torch.argsort(expanded_targets, stable=True)
    return expanded_sources[sort], expanded_targets[sort]

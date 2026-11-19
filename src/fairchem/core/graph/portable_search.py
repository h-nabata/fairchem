"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math
import operator
from functools import lru_cache
from itertools import groupby

import numpy as np
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


def _borrow_integer_cartesian_table(x: int, y: int, z: int, device: torch.device):
    """
    Borrow immutable storage for internal graph operations that never mutate it.
    """
    dimensions = []
    for value in (x, y, z):
        if isinstance(value, (bool, np.bool_)) or (
            torch.is_tensor(value) and value.dtype == torch.bool
        ):
            raise ValueError("Image repetitions must be nonnegative integers")
        try:
            dimension = operator.index(value)
        except TypeError as error:
            raise ValueError(
                "Image repetitions must be nonnegative integers"
            ) from error
        if dimension < 0:
            raise ValueError("Image repetitions must be nonnegative integers")
        dimensions.append(dimension)
    x, y, z = dimensions
    device = torch.device(device)
    if device.type in {"cuda", "xpu"} and device.index is None:
        device = torch.device(device.type, getattr(torch, device.type).current_device())
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


def integer_cartesian_table(x: int, y: int, z: int, device: torch.device):
    """
    Return owned storage so an external NumPy/in-place consumer cannot poison it.
    """
    return _borrow_integer_cartesian_table(x, y, z, device).clone()


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
        for count, n in zip(cells_per_image.tolist(), natoms.tolist(), strict=True):
            images.append(torch.arange(start, start + count, device=device).repeat(n))
            start += count
        images = torch.cat(images)
        return sources, images, atom_pos[sources] + pbc_offsets[images]
    sources, images = [], []
    # Reduce target bounds once, including arbitrary target subsets.
    indices = target_images.long()[:, None].expand(-1, 3)
    lower_bounds = atom_pos.new_full((len(natoms), 3), math.inf)
    upper_bounds = atom_pos.new_full((len(natoms), 3), -math.inf)
    lower_bounds.scatter_reduce_(0, indices, target_pos, reduce="amin")
    upper_bounds.scatter_reduce_(0, indices, target_pos, reduce="amax")
    sizes = torch.stack(
        (natoms, cells_per_image, lower_bounds[:, 0].isfinite()), dim=1
    ).tolist()
    atom_start = image_start = 0
    for system, (n, count, has_targets) in enumerate(sizes):
        if n and has_targets:
            lower, upper = lower_bounds[system], upper_bounds[system]
            atoms = atom_pos[atom_start : atom_start + n]
            offsets = pbc_offsets[image_start : image_start + count]
            margin = (
                32
                * torch.finfo(atom_pos.dtype).eps
                * (atoms.abs().amax() + offsets.abs().amax() + prune + 1)
            )
            # A flattened, atom-major walk bounds both image and atom
            # dimensions and already emits the required traversal order.
            for start in range(0, n * count, 32768):
                flat = torch.arange(start, min(start + 32768, n * count), device=device)
                atom = flat // count
                image = flat % count
                positions = atoms[atom] + offsets[image]
                delta = (lower - positions).clamp_min(0) + (
                    positions - upper
                ).clamp_min(0)
                keep = delta.square().sum(-1) <= (prune + margin) ** 2
                sources.append(atom[keep] + atom_start)
                images.append(image[keep] + image_start)
                del positions, delta, keep
        atom_start += n
        image_start += count
    if not sources:
        empty = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty, atom_pos.new_empty((0, 3))
    sources, images = torch.cat(sources), torch.cat(images)
    return sources, images, atom_pos[sources] + pbc_offsets[images]


_PAIR_TILE = 262144


def alternative_candidates(
    source_pos,
    target_pos,
    source_images,
    target_images,
    radius,
    method,
    *,
    return_distances=False,
    num_systems=None,
):
    """Search system-local source clouds, returning target-grouped candidates."""
    if method not in {"all_pairs", "bvh"}:
        raise ValueError(f"Unsupported alternative search method: {method}")
    empty = source_images.new_empty(0, dtype=torch.long)
    if not len(source_pos) or not len(target_pos):
        return (
            (empty, empty, source_pos.new_empty(0))
            if return_distances
            else (empty, empty)
        )
    if num_systems is None:
        num_systems = int(torch.maximum(source_images.max(), target_images.max())) + 1
        if num_systems > 1:
            # The graph builder passes validated, contiguous system runs.
            # Standalone callers may provide interleaved point clouds.
            source_order = source_images.argsort(stable=True)
            target_order = target_images.argsort(stable=True)
            result = alternative_candidates(
                source_pos[source_order],
                target_pos[target_order],
                source_images[source_order],
                target_images[target_order],
                radius,
                method,
                return_distances=return_distances,
                num_systems=num_systems,
            )
            sources = source_order[result[0]]
            targets = target_order[result[1]]
            order = targets.argsort(stable=True)
            pair = (sources[order], targets[order])
            return (*pair, result[2][order]) if return_distances else pair
    if num_systems == 1:
        sizes = [(len(source_pos), len(target_pos))]
    else:
        counts = torch.zeros(
            (2, num_systems), device=source_pos.device, dtype=torch.long
        )
        for row, ids in zip(counts, (source_images, target_images), strict=True):
            row.scatter_add_(0, ids.long(), torch.ones_like(ids, dtype=torch.long))
        sizes = counts.T.tolist()
    source_parts, target_parts, distances = [], [], []
    source_start = target_start = 0
    for (ns, nt), same_size in groupby(map(tuple, sizes)):
        repeats = sum(1 for _ in same_size)
        # Batch independent, equally sized small systems in one distance
        # kernel without introducing cross-system pairs or padding.
        if (
            method == "all_pairs"
            and repeats > 1
            and ns
            and nt
            and ns * nt <= _PAIR_TILE
        ):
            batch_width = _PAIR_TILE // (ns * nt)
            for start in range(0, repeats, batch_width):
                length = min(batch_width, repeats - start)
                source = source_pos[source_start : source_start + length * ns].view(
                    length, ns, 3
                )
                target = target_pos[target_start : target_start + length * nt].view(
                    length, nt, 3
                )
                d2 = (target[:, :, None] - source[:, None]).square().sum(-1)
                system, ti, si = torch.where(d2 <= radius * radius)
                source_parts.append(source_start + system * ns + si)
                target_parts.append(target_start + system * nt + ti)
                if return_distances:
                    distances.append(d2[system, ti, si])
                source_start += length * ns
                target_start += length * nt
            continue
        for _ in range(repeats):
            source = source_pos[source_start : source_start + ns]
            target = target_pos[target_start : target_start + nt]
            if ns and nt and method == "bvh":
                si, ti = _bvh_candidates(source, target, radius)
                source_parts.append(si + source_start)
                target_parts.append(ti + target_start)
                if return_distances:
                    distances.append((target[ti] - source[si]).square().sum(-1))
            elif ns and nt:
                source_width = min(ns, _PAIR_TILE)
                target_width = max(1, _PAIR_TILE // source_width)
                for t in range(0, nt, target_width):
                    for s in range(0, ns, source_width):
                        d2 = (
                            (
                                target[t : t + target_width, None]
                                - source[s : s + source_width]
                            )
                            .square()
                            .sum(-1)
                        )
                        ti, si = torch.where(d2 <= radius * radius)
                        source_parts.append(si + s + source_start)
                        target_parts.append(ti + t + target_start)
                        if return_distances:
                            distances.append(d2[ti, si])
            source_start += ns
            target_start += nt
    pair = (
        (torch.cat(source_parts), torch.cat(target_parts))
        if source_parts
        else (empty, empty)
    )
    if return_distances:
        return (*pair, torch.cat(distances) if distances else source_pos.new_empty(0))
    return pair


def _bvh_candidates(source_pos, target_pos, radius):
    device = source_pos.device
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
    del padded
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
        del positions, delta, keep
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
    sort = torch.argsort(expanded_targets, stable=True)
    return expanded_sources[sort], expanded_targets[sort]

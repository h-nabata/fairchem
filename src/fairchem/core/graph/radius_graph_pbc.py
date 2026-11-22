"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math
import operator
from types import SimpleNamespace

import numpy as np
import torch

from fairchem.core.graph.geometry import needs_wide_image_math, wide_image_device
from fairchem.core.graph.portable_search import (
    _borrow_integer_cartesian_table,
    alternative_candidates,
    source_cloud,
)


def is_mixed_pbc(data) -> bool:
    """
    Check if a batch has mixed PBC (some systems periodic, others not).

    Returns True if the batch contains systems with different PBC settings
    across any of the 3 dimensions, which is not supported by radius_graph_pbc.
    """
    if hasattr(data, "pbc") and torch.atleast_2d(data.pbc).shape[0] > 1:
        sys_pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            col = sys_pbc[:, i]
            if torch.any(col).item() and not torch.all(col).item():
                return True
    return False


def sum_partitions(x: torch.Tensor, partition_idxs: torch.Tensor) -> torch.Tensor:
    """
    Sum values within partitions defined by indices.
    """
    num_partitions = partition_idxs.shape[0] - 1
    if num_partitions == 0:
        return torch.zeros(0, device=x.device, dtype=x.dtype)

    # Use cumsum-based approach for vectorization
    cumsum = torch.zeros(len(x) + 1, device=x.device, dtype=x.dtype)
    cumsum[1:] = torch.cumsum(x, dim=0)

    # Gather cumsum at partition boundaries and compute differences
    starts = cumsum[partition_idxs[:-1]]
    ends = cumsum[partition_idxs[1:]]
    return ends - starts


def get_counts(x: torch.Tensor, length: int):
    dtype = x.dtype
    device = x.device
    return torch.zeros(length, device=device, dtype=dtype).scatter_reduce(
        dim=0,
        index=x,
        src=torch.ones(x.shape[0], device=device, dtype=dtype),
        reduce="sum",
    )


def compute_neighbors(data, edge_index):
    # Get number of neighbors
    num_neighbors = get_counts(edge_index[1], data.natoms.sum())

    # Get number of neighbors per image
    image_indptr = torch.zeros(
        data.natoms.shape[0] + 1, device=data.pos.device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(data.natoms, dim=0)
    return sum_partitions(num_neighbors, image_indptr)


_CAP_TILE = 262144


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Strict selection retains up to k nearest neighbors. The historical
    non-strict rule includes all squared distances up to the (k+1)th
    nearest squared distance plus degeneracy_tolerance (in squared length
    units). In particular even nondegenerate rows may retain k+1 neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    num_neighbors = get_counts(index, num_atoms)

    image_indptr = torch.cat((natoms.new_zeros(1), natoms.cumsum(0)))
    if max_num_neighbors_threshold <= 0 or not len(num_neighbors):
        return torch.ones(1, dtype=torch.bool, device=device).expand_as(
            index
        ), sum_partitions(num_neighbors, image_indptr)

    width = int(num_neighbors.max())
    cap = max_num_neighbors_threshold
    if width <= cap:
        return torch.ones(1, dtype=torch.bool, device=device).expand_as(
            index
        ), sum_partitions(num_neighbors, image_indptr)
    overfull = num_neighbors > cap
    atoms = torch.where(overfull)[0]
    offsets = num_neighbors.cumsum(0) - num_neighbors
    mask = torch.isfinite(atom_distance) & ~overfull[index]
    selected_counts = num_neighbors.clamp_max(cap)
    if not enforce_max_strictly and len(atoms) < len(num_neighbors):
        # Preserve the historical finite filtering and counts on rows with
        # NaNs when another row makes capping active.
        nan_counts = torch.zeros_like(num_neighbors).scatter_add_(
            0, index, atom_distance.isnan().long()
        )
        valid = nan_counts < (width - cap)
        if math.isnan(math.inf + degeneracy_tolerance):
            valid.zero_()
        mask &= valid[index]
        selected_counts.masked_fill_(~overfull & ~valid, 0)

    if len(atoms) * width > max(_CAP_TILE, 2 * len(index)) and bool(
        atom_distance.isfinite().all()
    ):
        # A few extremely high-degree atoms would make most padded slots
        # empty even when chunked. Sort only real edges, then order by target.
        order = atom_distance.argsort(stable=True)
        order = order[index[order].argsort(stable=True)]
        kth = offsets[atoms] + cap
        thresholds = atom_distance.new_full((len(num_neighbors),), -math.inf)
        if not enforce_max_strictly:
            thresholds[atoms] = atom_distance[order[kth]] + degeneracy_tolerance
            mask |= atom_distance <= thresholds[index]
            return mask, sum_partitions(
                get_counts(index[mask], len(num_neighbors)), image_indptr
            )
        unique_boundary = atom_distance[order[kth - 1]] < atom_distance[order[kth]]
        thresholds[atoms[unique_boundary]] = atom_distance[order[kth - 1]][
            unique_boundary
        ]
        mask |= atom_distance <= thresholds[index]
        # Only exact ties at the strict cap require the historical padded
        # unstable sort to reproduce its arbitrary choice between equal edges.
        atoms = atoms[~unique_boundary]
        if not len(atoms):
            return mask, sum_partitions(selected_counts, image_indptr)

    # Keep the historical padded row width: unstable strict sorts can choose
    # different indices among ties at different widths. Bound simultaneous
    # row count so a single outlier does not inflate the entire allocation.
    column = torch.arange(width, device=device)
    rows_per_tile = max(1, _CAP_TILE // width)
    for rows in atoms.split(rows_per_tile):
        positions = offsets[rows, None] + column
        distances = atom_distance[positions.clamp_max(len(index) - 1)]
        distances.masked_fill_(column >= num_neighbors[rows, None], math.inf)
        sorted_distances, order = distances.sort(dim=1)
        if enforce_max_strictly:
            chosen = sorted_distances[:, :cap].isfinite()
            order = order[:, :cap]
        else:
            included = (
                sorted_distances
                <= sorted_distances[:, cap : cap + 1] + degeneracy_tolerance
            )
            selected_counts[rows] = torch.minimum(num_neighbors[rows], included.sum(-1))
            chosen = included & sorted_distances.isfinite()
        mask[(offsets[rows, None] + order)[chosen]] = True
    return mask, sum_partitions(selected_counts, image_indptr)


def complete_periodic_cell(cell, flags):
    """Choose orthogonal vacuum directions for searches in a periodic subspace.

    Only translations along flagged axes can contribute neighbors. Including
    a nearly parallel vacuum vector in the reciprocal basis can otherwise
    inflate the number of trial images arbitrarily.
    """
    flags = flags.expand(len(cell), -1)
    partial = flags.any(-1) & ~flags.all(-1)
    if not bool(partial.any()):
        return cell
    # Keep existing orthogonal vacuum rows (and their lengths). No completion
    # is necessary for the usual nonzero diagonal slab or wire cell.
    if bool((cell == torch.diag_embed(cell.diagonal(dim1=-2, dim2=-1))).all()) and bool(
        (cell.diagonal(dim1=-2, dim2=-1) != 0).all()
    ):
        return cell
    completed = cell.clone()
    for pattern in flags[partial].unique(dim=0).tolist():
        axes = [i for i, periodic in enumerate(pattern) if periodic]
        missing = [i for i, periodic in enumerate(pattern) if not periodic]
        systems = (flags == flags.new_tensor(pattern)).all(-1).nonzero().flatten()
        basis = cell[systems][:, axes]
        if len(axes) == 2:
            normal = torch.cross(basis[:, 0], basis[:, 1], dim=-1)
            length = normal.norm(dim=-1, keepdim=True)
            if bool((length == 0).any()):
                raise ValueError(
                    "Periodic lattice vectors must span a nonsingular subspace"
                )
            completed[systems, missing[0]] = normal / length
        else:
            direction = basis[:, 0]
            length = direction.norm(dim=-1, keepdim=True)
            if bool((length == 0).any()):
                raise ValueError(
                    "Periodic lattice vectors must span a nonsingular subspace"
                )
            direction = direction / length
            pivot = torch.zeros_like(direction)
            pivot.scatter_(1, direction.abs().argmin(dim=-1, keepdim=True), 1)
            normal = torch.cross(direction, pivot, dim=-1)
            normal = normal / normal.norm(dim=-1, keepdim=True)
            completed[systems, missing[0]] = normal
            completed[systems, missing[1]] = torch.cross(direction, normal, dim=-1)
    return completed


def public_image_offsets(images, dtype):
    """Use the historical floating dtype whenever it represents every image ID."""
    converted = images.to(dtype)
    return converted if torch.equal(converted.long(), images) else images


def periodic_search_frame(data, flags, *, search_cell=None):
    """Return differentiable local positions, integer image IDs and reciprocal vectors."""
    cell = data.cell
    with torch.no_grad():
        if search_cell is None:
            search_cell = complete_periodic_cell(cell, flags)
        cross = torch.cross(search_cell.roll(-1, 1), search_cell.roll(-2, 1), dim=-1)
        volume = (search_cell[:, 0] * cross[:, 0]).sum(-1)
        periodic = flags.any(1)
        if bool((periodic & ((volume == 0) | ~volume.isfinite())).any()):
            raise ValueError(
                "Periodic graph generation requires a nonsingular finite-volume cell"
            )
        reciprocal = cross / torch.where(periodic, volume, 1)[:, None, None]
        batch = getattr(data, "batch", None)
        if batch is None and len(cell) > 1:
            batch = torch.arange(len(cell), device=cell.device).repeat_interleave(
                data.natoms, output_size=len(data.pos)
            )
        pos = data.pos.to(torch.promote_types(data.pos.dtype, cell.dtype))
        # Center the search frame on the origin. Wrapping a tiny negative
        # coordinate to one whole cell can round the displacement away in FP32.
        if len(cell) == 1:
            frac = pos @ reciprocal[0].T.to(pos)
            fractional_images = torch.where(flags, frac.round(), 0)
        else:
            frac = (reciprocal[batch].to(pos) @ pos.unsqueeze(-1)).squeeze(-1)
            fractional_images = torch.where(
                flags.expand(len(cell), -1)[batch], frac.round(), 0
            )
    if pos.dtype == torch.float32 and bool(
        needs_wide_image_math(fractional_images, cell)
    ):
        # FP32 cannot reliably subtract large integer multiples of a nonbinary
        # lattice vector. Reduce only such unwrapped inputs in double. Xe devices
        # without FP64 can still run the model in FP32 after a CPU reduction.
        device = wide_image_device(pos.device)
        wide = SimpleNamespace(
            pos=data.pos.to(device=device, dtype=torch.float64),
            cell=cell.to(device=device, dtype=torch.float64),
            batch=None if batch is None else batch.to(device),
            natoms=data.natoms.to(device),
        )
        local, exact_images, _ = periodic_search_frame(wide, flags.to(device))
        return local.to(pos), exact_images.to(pos.device), reciprocal
    if pos.dtype == torch.float64 and bool((fractional_images.abs() >= 2**62).any()):
        raise ValueError("Periodic image difference exceeds supported integer range")
    images = fractional_images.long()
    pos = data.pos.to(torch.promote_types(data.pos.dtype, cell.dtype))
    if len(cell) == 1:
        pos = pos - images.to(pos) @ cell[0].to(pos)
    else:
        pos = pos - (images.to(pos).unsqueeze(1) @ cell[batch].to(pos)).squeeze(1)
    return pos, images, reciprocal


# @torch.no_grad() # breaks torch compile test with OOM
def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
    *,
    _frame=None,
):
    flags = system_pbc_flags(data, pbc)
    pbc = flags.any(0).tolist()

    # v1 uses a single global PBC for the whole batch and cannot produce
    # correct graphs for batches where some systems are periodic and others
    # are not.  Use radius_graph_pbc_v2 for mixed-PBC batches.
    if len(flags) > 1 and bool((flags != flags[:1]).any()):
        raise RuntimeError(
            "radius_graph_pbc does not support batches with mixed PBC "
            "(some systems periodic, others not). Use radius_graph_pbc_v2 instead."
        )

    device = data.pos.device
    if _frame is None:
        validate_graph_geometry(data)
    batch_size = len(data.natoms)
    if not len(data.pos):
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            data.cell.new_empty((0, 3)),
            torch.zeros_like(data.natoms, dtype=torch.long),
        )
    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = num_atoms_per_image.long().square()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this approach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    images = None
    if any(pbc):
        atom_pos, images, reciprocal = (
            periodic_search_frame(data, flags) if _frame is None else _frame
        )
        repetitions = torch.where(flags, (radius * reciprocal.norm(dim=-1)).ceil(), 0)
        max_rep = repetitions.amax(dim=0)
    else:
        max_rep = data.cell.new_zeros(3)
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=data.cell.dtype)
        for rep in max_rep.tolist()
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if max_num_neighbors_threshold > 0 and not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    if images is not None:
        original_dtype = unit_cell.dtype
        unit_cell = unit_cell.long() + (images[index1] - images[index2])
        if _frame is None:
            unit_cell = public_image_offsets(unit_cell, original_dtype)
    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def validate_graph_geometry(data):
    """
    Reject invalid coordinates before they can silently disappear from a graph.
    """
    if not data.pos.is_floating_point() or not data.cell.is_floating_point():
        raise ValueError("Graph positions and cells must have floating-point dtypes")
    if data.cell.device != data.pos.device:
        raise ValueError("Graph positions and cells must be on the same device")
    if data.pos.ndim != 2 or data.pos.shape[1] != 3:
        raise ValueError("Graph positions must have shape [number of atoms, 3]")
    if data.natoms.ndim != 1 or data.natoms.dtype not in (torch.int32, torch.int64):
        raise ValueError("Graph natoms must be a one-dimensional integer tensor")
    if data.cell.shape != (len(data.natoms), 3, 3):
        raise ValueError("Graph cells must have shape [number of systems, 3, 3]")
    natoms = data.natoms.to(data.pos.device)
    checks = [
        (
            (natoms >= 0).all() & (natoms.sum() == len(data.pos)),
            "Graph natoms must be nonnegative and sum to the atom count",
        ),
        (
            torch.isfinite(data.pos).all() & torch.isfinite(data.cell).all(),
            "Graph positions and cells must contain only finite values",
        ),
    ]
    batch = getattr(data, "batch", None)
    if batch is None:
        if hasattr(data, "batch") and len(natoms) > 1:
            raise ValueError("Multiple systems require an explicit graph batch index")
    else:
        if batch.shape != (len(data.pos),) or batch.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Graph batch must be a one-dimensional integer index")
        if batch.device != data.pos.device:
            raise ValueError("Graph batch and positions must be on the same device")
        in_range = ((batch >= 0) & (batch < len(natoms))).all()
        checks.append((in_range, "Graph batch contains an out-of-range system index"))
        # With monotone IDs, checking both endpoints of every nonempty
        # segment validates exact counts. Clamp invalid offsets until reported.
        ends = natoms.cumsum(0)
        if len(batch):
            ids = torch.arange(len(natoms), device=batch.device)
            start_id = batch[(ends - natoms).clamp(0, len(batch) - 1)]
            end_id = batch[(ends - 1).clamp(0, len(batch) - 1)]
            grouped = ((natoms == 0) | ((start_id == ids) & (end_id == ids))).all()
            checks.append(
                (
                    grouped & (batch[1:] >= batch[:-1]).all(),
                    "Graph batch must group atoms contiguously according to natoms",
                )
            )
    for passed, (_, message) in zip(
        torch.stack([flag for flag, _ in checks]).tolist(), checks, strict=True
    ):
        if not passed:
            raise ValueError(message)


def system_pbc_flags(data, pbc=None):
    """
    Normalize per-system PBC without replacing or mutating caller metadata.
    """
    value = getattr(data, "pbc", None) if pbc is None else pbc
    if value is None:
        raise ValueError("PBC flags are required")
    flags = torch.as_tensor(
        value, device=getattr(getattr(data, "pos", None), "device", None)
    )
    systems = getattr(data, "natoms", None)
    if (
        flags.ndim not in (1, 2)
        or flags.shape[-1] != 3
        or (
            flags.ndim == 2
            and systems is not None
            and flags.shape[0] not in (1, len(systems))
        )
    ):
        raise ValueError("PBC must have shape [3], [1, 3] or [number of systems, 3]")
    if flags.dtype != torch.bool and not bool(((flags == 0) | (flags == 1)).all()):
        raise ValueError("PBC flags must be booleans or zero/one values")
    return torch.atleast_2d(flags).to(dtype=torch.bool)


def validate_graph_parameters(radius, max_neighbors, strict, method="cell"):
    """
    Validate scalars before a cache key or graph conversion can truncate them.
    """
    if not isinstance(method, str) or method not in {
        "cell",
        "auto",
        "all_pairs",
        "bvh",
    }:
        raise ValueError(f"Unknown graph search method: {method}")
    if isinstance(radius, (bool, np.bool_, str)) or (
        torch.is_tensor(radius) and radius.dtype == torch.bool
    ):
        raise ValueError("Graph radius must be finite and positive")
    try:
        radius = float(radius)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Graph radius must be finite and positive") from error
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Graph radius must be finite and positive")
    if isinstance(max_neighbors, (bool, np.bool_)) or (
        torch.is_tensor(max_neighbors) and max_neighbors.dtype == torch.bool
    ):
        raise ValueError("max_neighbors must be an integer")
    try:
        max_neighbors = operator.index(max_neighbors)
    except TypeError as error:
        raise ValueError("max_neighbors must be an integer") from error
    if not isinstance(strict, (bool, np.bool_)):
        raise ValueError("enforce_max_neighbors_strictly must be a boolean")
    return radius, max_neighbors, bool(strict)


def canonical_pbc(data, pbc: torch.Tensor | None) -> list[bool]:
    """
    Resolve and normalize the periodic boundary condition (PBC) flags for a batch.

    This helper produces a canonical 1D PBC specification (a list of three booleans,
    one per lattice direction) that downstream graph construction code can rely on.

    Behavior:
      - If ``pbc`` is None, derive it from ``data.pbc``: a dimension is considered
        periodic only if at least one system in the batch is periodic along that
        dimension. Caller metadata is unchanged.
      - If ``pbc`` has more than one dimension, collapse it across systems with a
        logical OR so that a dimension is periodic if any system requests it.

    Args:
        data: An AtomicData-like object that must expose a ``pbc`` attribute
            describing per-system PBC flags.
        pbc: Optional explicit PBC override. May be a 1D tensor of length 3 or a
            2D tensor of shape (num_systems, 3). If None, the value is taken from
            ``data.pbc``.

    Returns:
        A length-3 Python list of booleans ``[pbc_x, pbc_y, pbc_z]`` indicating
        which lattice directions are periodic for the batch.
    """
    return system_pbc_flags(data, pbc).any(dim=0).tolist()


def box_size_warning(cell, pos, pbc):
    if hasattr(box_size_warning, "already_printed"):
        return
    box_size_warning.already_printed = True
    logging.warning(
        f"PBCv2: graph generation encountered a very large box. The size of the cell is {cell} and min/max positions are {pos.min(), pos.max()}. Performance will be slower than optimal."
    )
    if any(pbc):
        logging.warning(
            "PBCv2: Does this system require PBC=True or will PBC=False work?"
        )


def _image_counts(repetitions):
    """Check index capacity in exact integer arithmetic before allocating tables."""
    counts = []
    for row in repetitions:
        if not all(math.isfinite(value) for value in row):
            raise ValueError("Periodic image count exceeds the supported index range")
        count = math.prod(2 * int(value) + 1 for value in row)
        if count > torch.iinfo(torch.long).max // 3:
            raise ValueError("Periodic image count exceeds the supported index range")
        counts.append(count)
    if sum(counts) > torch.iinfo(torch.long).max // 3:
        raise ValueError("Periodic image count exceeds the supported index range")
    return counts


_STENCIL_TILE = 262144
_CELL_PAIR_TILE = 32768


def _cell_candidates(
    atom_pos,
    data_batch_idxs,
    source_atom_pos,
    source_atom_image,
    target_atom_pos,
    target_atom_image,
    radius,
    legacy_order,
    include_zero_distance,
    batch_size,
):
    device = atom_pos.device
    grid_resolution = radius / 1.99
    # Given the positions of all the atoms has been computed, we
    # split them up using a cubic grid to make pairwise comparisons
    # more computationally efficient. The resolution of the grid
    # is grid_resolution.

    # Compute the grid index for each dimension for each atom
    # The source tiles include the unshifted atoms and therefore every target.
    # One reduction suffices; Python max(tensor, tensor) also synchronizes an
    # accelerator unnecessarily to compare the two scalar tensors.
    if legacy_order:
        max_coordinate = source_atom_pos.abs().max()
        if max_coordinate / grid_resolution > 200:
            grid_resolution = max_coordinate / 200
        origin = atom_pos.new_zeros((batch_size, 3))
    else:
        origin = atom_pos.new_zeros((batch_size, 3))
        origin.scatter_reduce_(
            0,
            data_batch_idxs[:, None].expand(-1, 3),
            atom_pos,
            reduce="amin",
            include_self=False,
        )
        # The sparse integer hash must fit int64, including batch offsets.
        max_extent = (source_atom_pos - origin[source_atom_image]).abs().amax(0)
        safe_axis = min(1000000, int((2**59 / max(1, batch_size)) ** (1 / 3)))
        # Widen only the overflowing axes so vacuum in x cannot crowd y/z bins.
        grid_resolution = (max_extent / safe_axis).clamp_min(grid_resolution)
    source_atom_grid = torch.floor(
        (source_atom_pos - origin[source_atom_image]) / grid_resolution
    ).long()
    target_grid_float = (target_atom_pos - origin[target_atom_image]) / grid_resolution
    target_atom_grid = torch.floor(target_grid_float).long()

    # Reduce directly into one row per system. Padding all systems to the
    # largest tiled system wastes memory and includes spurious zero grid
    # coordinates in the bounds of smaller systems.
    if batch_size == 1:
        grid_min = source_atom_grid.amin(dim=0, keepdim=True)
        grid_max = source_atom_grid.amax(dim=0, keepdim=True)
    else:
        image_index = source_atom_image[:, None].expand(-1, 3)
        grid_min = torch.zeros((batch_size, 3), device=device, dtype=torch.long)
        grid_max = torch.zeros_like(grid_min)
        grid_min.scatter_reduce_(
            0, image_index, source_atom_grid, reduce="amin", include_self=False
        )
        grid_max.scatter_reduce_(
            0, image_index, source_atom_grid, reduce="amax", include_self=False
        )

    # Size of grid in each dimension for each image
    grid_size = grid_max - grid_min + 1
    grid_length = grid_size[:, 0] * grid_size[:, 1] * grid_size[:, 2]
    # Offset between grids for each image
    grid_offset = torch.cat(
        [torch.tensor([0], device=device), torch.cumsum(grid_length, dim=0)], dim=0
    )

    num_grid_cells = torch.sum(grid_length)

    # Subtract the minimum grid index so they are zero indexed
    source_atom_grid = source_atom_grid - grid_min[source_atom_image]
    target_atom_grid = target_atom_grid - grid_min[target_atom_image]

    # Compute the grid id which is a combination of the grid indices in the x,y,z directions
    # Grid id is x + y*grid_size[0] + z*grid_size[0]*grid_size[1] + offset
    source_atom_grid_size = grid_size[source_atom_image]
    source_atom_grid_offset = grid_offset[source_atom_image]
    source_atom_grid_id = (
        source_atom_grid[:, 0]
        + source_atom_grid[:, 1] * source_atom_grid_size[:, 0]
        + source_atom_grid[:, 2]
        * source_atom_grid_size[:, 0]
        * source_atom_grid_size[:, 1]
    )
    source_atom_grid_id = source_atom_grid_id + source_atom_grid_offset

    target_atom_grid_size = grid_size[target_atom_image]
    target_atom_grid_offset = grid_offset[target_atom_image]
    target_atom_grid_id = (
        target_atom_grid[:, 0]
        + target_atom_grid[:, 1] * target_atom_grid_size[:, 0]
        + target_atom_grid[:, 2]
        * target_atom_grid_size[:, 0]
        * target_atom_grid_size[:, 1]
    )
    target_atom_grid_id = target_atom_grid_id + target_atom_grid_offset

    # Keep only occupied cells. A dense [all cells, maximum occupancy]
    # table scales with vacuum volume and the densest cell in any system.
    # Sorting is unchanged so ties retain the previous source traversal order.
    sort_grid_id, sort_indices = torch.sort(source_atom_grid_id)
    occupied_grid_ids, grid_cell_atom_count = torch.unique_consecutive(
        sort_grid_id, return_counts=True
    )
    grid_cell_start = torch.cumsum(grid_cell_atom_count, 0) - grid_cell_atom_count

    # How many grid cells do we need to include in each direction given the search radius?
    padding_size = math.floor(radius / grid_resolution) + 1 if legacy_order else 2
    num_padding_grid_cells = (
        (2 * padding_size + 1) * (2 * padding_size + 1) * (2 * padding_size + 1)
    )

    padding_offsets = (
        _borrow_integer_cartesian_table(
            padding_size, padding_size, padding_size, device
        )
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
    )

    strides = torch.stack(
        (
            torch.ones_like(grid_size[:, 0]),
            grid_size[:, 0],
            grid_size[:, 0] * grid_size[:, 1],
        ),
        dim=-1,
    )
    padding_ids = (padding_offsets * strides[:, None]).sum(dim=2)
    cell_to_occupied = None
    if num_grid_cells <= 4 * len(source_atom_grid_id):
        cell_to_occupied = torch.full(
            (num_grid_cells + 1,), -1, device=device, dtype=torch.long
        )
        cell_to_occupied[occupied_grid_ids] = torch.arange(
            len(occupied_grid_ids), device=device
        )
    tile = max(1, _STENCIL_TILE // num_padding_grid_cells)
    source_parts, target_parts, squared_parts = [], [], []
    for start in range(0, len(target_atom_pos), tile):
        rows = slice(start, start + tile)
        systems = target_atom_image[rows]
        grid_index = target_atom_grid[rows, None] + padding_offsets[systems]
        in_bounds = (
            (grid_index >= 0) & (grid_index < target_atom_grid_size[rows, None])
        ).all(-1)
        if not legacy_order:
            relative = target_grid_float[rows] - grid_min[systems]
            delta = (grid_index - relative[:, None]).clamp_min(0) + (
                relative[:, None] - grid_index - 1
            ).clamp_min(0)
            margin = 64 * torch.finfo(atom_pos.dtype).eps * (relative.abs() + 1)
            nearest = (delta - margin[:, None]).clamp_min_(0) * grid_resolution
            in_bounds &= nearest.square().sum(-1) <= radius * radius
            del delta
        del grid_index
        keys = target_atom_grid_id[rows, None] + padding_ids[systems]
        keys.masked_fill_(~in_bounds, num_grid_cells)
        keys = keys.flatten()
        if cell_to_occupied is None:
            occupied = torch.searchsorted(occupied_grid_ids, keys).clamp_max(
                len(occupied_grid_ids) - 1
            )
        else:
            occupied = cell_to_occupied[keys].clamp_min(0)
        counts = torch.where(
            occupied_grid_ids[occupied] == keys, grid_cell_atom_count[occupied], 0
        )
        ends = counts.cumsum(0)
        total = int(ends[-1])
        # Bound atom pairs even when one occupied bin contains a dense cluster.
        # Prefix sums let a single target/bin run span several pair tiles.
        for first in range(0, total, _CELL_PAIR_TILE):
            pair = torch.arange(
                first, min(first + _CELL_PAIR_TILE, total), device=device
            )
            if total <= _CELL_PAIR_TILE:
                group = torch.arange(len(counts), device=device).repeat_interleave(
                    counts, output_size=total
                )
            else:
                group = torch.searchsorted(ends, pair, right=True)
            source = sort_indices[
                grid_cell_start[occupied[group]] + pair - (ends[group] - counts[group])
            ]
            target = start + group // num_padding_grid_cells
            squared = (
                (target_atom_pos[target] - source_atom_pos[source]).square().sum(-1)
            )
            keep = squared <= radius * radius
            if not include_zero_distance:
                keep &= squared != 0
            source_parts.append(source[keep])
            target_parts.append(target[keep])
            squared_parts.append(squared[keep])
    if not source_parts:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, atom_pos.new_empty(0)
    return tuple(
        torch.cat(parts) if len(parts) > 1 else parts[0]
        for parts in (source_parts, target_parts, squared_parts)
    )


@torch.no_grad()
def radius_graph_pbc_v2(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
    *,
    method: str = "cell",
    include_zero_distance: bool = False,
    _frame=None,
):
    radius, max_num_neighbors_threshold, enforce_max_neighbors_strictly = (
        validate_graph_parameters(
            radius, max_num_neighbors_threshold, enforce_max_neighbors_strictly, method
        )
    )
    # Strict tie selection is historically tied to the old cell traversal.
    # Preserve that path instead of changing which degenerate neighbors survive.
    legacy_order = enforce_max_neighbors_strictly and max_num_neighbors_threshold > 0
    if legacy_order:
        method = "cell"
    # Preserve the per-system flags before canonical_pbc collapses them. An
    # explicit override (including always_use_pbc) must also control image
    # repetitions, not only the global warning/check flags.
    sys_pbc = system_pbc_flags(data, pbc)
    pbc = sys_pbc.any(dim=0).tolist()

    device = data.pos.device
    if _frame is None:
        validate_graph_geometry(data)
    batch_size = len(data.natoms)
    node_partition = None
    if hasattr(data, "node_partition"):
        node_partition = torch.as_tensor(data.node_partition, device=device)
        if node_partition.ndim != 1 or node_partition.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("node_partition must be a one-dimensional integer index")
        if bool(((node_partition < 0) | (node_partition >= len(data.pos))).any()):
            raise ValueError("node_partition contains an out-of-range atom index")
        if node_partition.unique().numel() != node_partition.numel():
            raise ValueError("node_partition must not contain duplicate atom indices")
        node_partition = node_partition.long()
    if len(data.pos) == 0 or (node_partition is not None and len(node_partition) == 0):
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            data.cell.new_empty((0, 3)),
            torch.zeros_like(data.natoms, dtype=torch.long),
        )
    data_batch_idxs = (
        data.batch.long()
        if getattr(data, "batch", None) is not None
        else torch.arange(batch_size, device=device).repeat_interleave(
            data.natoms, output_size=len(data.pos)
        )
    )

    atom_pos = data.pos
    image_numbers = None
    if any(pbc):
        cell = data.cell
        atom_pos, image_numbers, reciprocal = (
            periodic_search_frame(data, sys_pbc) if _frame is None else _frame
        )
        repetitions = torch.where(sys_pbc, (radius * reciprocal.norm(dim=-1)).ceil(), 0)
        # Compute image counts in Python integer arithmetic. FP32 cannot
        # represent all products above 2**24 and int64 products may overflow.
        reps = repetitions.tolist()
        counts = _image_counts(reps)
        unit_cells = [
            _borrow_integer_cartesian_table(*map(int, row), device).to(cell.dtype)
            for row in reps
        ]
        cells_per_image = torch.tensor(counts, device=device, dtype=torch.long)
        if batch_size == 1:
            unit_cell = unit_cells[0]
            pbc_cell_offsets = unit_cell @ cell[0]
        else:
            unit_cell = torch.cat(unit_cells)
            pbc_cell_offsets = torch.cat(
                [
                    offsets @ system_cell
                    for offsets, system_cell in zip(unit_cells, cell, strict=True)
                ]
            )
    # If node_partition exists, this means we want to generate only a partial graph
    # from a subset of target atoms to the entire set of source atoms
    if hasattr(data, "node_partition"):
        # Neighbor capping expects edges grouped by global target index.
        # Canonicalize a private index tensor without changing the caller's
        # partition; arbitrary input order otherwise corrupts capped graphs.
        node_partition, _ = torch.sort(node_partition)
        target_atom_pos = atom_pos[node_partition]
        target_atom_image = data_batch_idxs[node_partition]
    else:
        target_atom_pos = atom_pos
        target_atom_image = data_batch_idxs

    if any(pbc):
        source_atom_index, pbc_cell_index, source_atom_pos = source_cloud(
            atom_pos,
            target_atom_pos if method in ("all_pairs", "bvh") else atom_pos,
            target_atom_image if method in ("all_pairs", "bvh") else data_batch_idxs,
            pbc_cell_offsets,
            unit_cell,
            cells_per_image,
            data.natoms,
            prune=radius if not legacy_order else None,
        )
        source_atom_image = data_batch_idxs[source_atom_index]
    else:
        source_atom_pos = atom_pos
        source_atom_image = data_batch_idxs
    selections = [(method, None)]
    if method == "auto":
        names = ("all_pairs", "cell", "bvh")
        # CPU measurements for dense 256-atom systems favor tiled distances.
        # Preserve the accelerator cutoff until device-specific measurements.
        pair_limit = 65536 if device.type == "cpu" else 16384
        # Select by each system's actual local cloud. A batch of many tiny
        # systems must not resemble a single dense or high-vacuum system.
        if batch_size == 1:
            small = len(source_atom_pos) * len(target_atom_pos) <= pair_limit
            if small:
                selections = [("all_pairs", None)]
            else:
                extent = source_atom_pos.amax(0) - source_atom_pos.amin(0)
                sparse = bool((1 + extent / radius).prod() > 64 * len(source_atom_pos))
                selections = [("bvh" if sparse else "cell", None)]
        else:
            counts = get_counts(source_atom_image.long(), batch_size)
            targets = get_counts(target_atom_image.long(), batch_size)
            idx = source_atom_image.long()[:, None].expand(-1, 3)
            bounds = source_atom_pos.new_zeros((batch_size, 3))
            lower = bounds.scatter_reduce(
                0, idx, source_atom_pos, reduce="amin", include_self=False
            )
            upper = bounds.scatter_reduce(
                0, idx, source_atom_pos, reduce="amax", include_self=False
            )
            sparse = (1 + (upper - lower) / radius).prod(1) > 64 * counts
            methods = torch.where(counts * targets <= pair_limit, 0, sparse.long() + 1)
            active = methods[targets > 0].unique().tolist()
            selections = (
                [(names[active[0]], None)]
                if len(active) == 1
                else [
                    (names[code], torch.where(methods[target_atom_image] == code)[0])
                    for code in active
                ]
            )

    parts = []
    for algorithm, selected in selections:
        positions = target_atom_pos if selected is None else target_atom_pos[selected]
        systems = target_atom_image if selected is None else target_atom_image[selected]
        if algorithm == "cell":
            si, ti, d2 = _cell_candidates(
                atom_pos,
                data_batch_idxs,
                source_atom_pos,
                source_atom_image,
                positions,
                systems,
                radius,
                legacy_order,
                include_zero_distance,
                batch_size,
            )
        else:
            si, ti, d2 = alternative_candidates(
                source_atom_pos,
                positions,
                source_atom_image,
                systems,
                radius,
                algorithm,
                return_distances=True,
                num_systems=batch_size,
            )
            keep = d2 <= radius * radius
            if not include_zero_distance:
                keep &= d2 != 0
            si, ti, d2 = si[keep], ti[keep], d2[keep]
        parts.append((si, ti if selected is None else selected[ti], d2))
    if len(parts) == 1:
        source_atom_edge_index, target_atom_edge_index, atom_distance_sqr = parts[0]
    else:
        si, ti, d2 = (torch.cat(tensors) for tensors in zip(*parts, strict=True))
        order = ti.argsort(stable=True)
        source_atom_edge_index, target_atom_edge_index, atom_distance_sqr = (
            si[order],
            ti[order],
            d2[order],
        )
    del source_atom_pos, target_atom_pos
    # Map targets once, before capping, so global per-atom degrees also
    # remain correct for an unsorted caller partition.
    target_idx = (
        target_atom_edge_index
        if node_partition is None
        else node_partition[target_atom_edge_index]
    )
    mask, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=target_idx,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )
    # The cap helper returns a broadcast scalar for under-cap graphs. Its
    # stride is host metadata, so this fast path requires no device readback.
    if max_num_neighbors_threshold > 0 and mask.stride(0) != 0:
        target_idx = target_idx[mask]
        source_atom_edge_index = source_atom_edge_index[mask]
    if any(pbc):
        source_idx = source_atom_index[source_atom_edge_index]
        source_cell = unit_cell[pbc_cell_index[source_atom_edge_index]].long() + (
            image_numbers[target_idx] - image_numbers[source_idx]
        )
        if _frame is None:
            source_cell = public_image_offsets(source_cell, data.cell.dtype)
    else:
        source_idx = source_atom_edge_index
        source_cell = data.cell.new_zeros((len(source_idx), 3))
    return torch.stack((source_idx, target_idx)), source_cell, num_neighbors_image

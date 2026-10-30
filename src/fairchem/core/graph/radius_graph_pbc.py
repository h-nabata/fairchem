"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch


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


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
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

    max_num_neighbors = num_neighbors.max()
    if max_num_neighbors_threshold > 0:
        num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)
    else:
        num_neighbors_thresholded = num_neighbors

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Only atoms above the cap need sorting. Padding every atom to the
    # largest degree can be much larger than the edge list in heterogeneous
    # batches (for example, many isolated atoms and one dense neighborhood).
    overfull = num_neighbors > max_num_neighbors_threshold
    overfull_atoms = torch.nonzero(overfull).flatten()
    has_uncapped_atoms = len(overfull_atoms) < len(num_neighbors)
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    if has_uncapped_atoms:
        overfull_edges = overfull[index]
        edge_positions = torch.nonzero(overfull_edges).flatten()
        atom_rows = torch.cumsum(overfull.long(), dim=0) - 1
        index_sort_map = (
            atom_rows[index[edge_positions]] * max_num_neighbors
            + edge_positions
            - index_neighbor_offset[index[edge_positions]]
        )
        capped_counts = num_neighbors[overfull_atoms]
        capped_offsets = index_neighbor_offset[overfull_atoms]
        capped_distances = atom_distance[edge_positions]
        # Preserve the existing finite filtering when capping is active,
        # including under-cap rows that no longer enter the padded matrix.
        mask_num_neighbors = torch.isfinite(atom_distance) & ~overfull_edges
    else:
        # Homogeneous over-cap batches still need every row. Avoid filtering
        # their entire edge list only to select the same distances again.
        index_sort_map = (
            index * max_num_neighbors
            + torch.arange(len(index), device=device)
            - torch.repeat_interleave(
                index_neighbor_offset, num_neighbors, output_size=len(index)
            )
        )
        capped_counts = num_neighbors
        capped_offsets = index_neighbor_offset
        capped_distances = atom_distance
        mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    distance_sort = torch.full(
        [len(overfull_atoms) * max_num_neighbors],
        np.inf,
        device=device,
        dtype=atom_distance.dtype,
    )
    distance_sort.index_copy_(0, index_sort_map, capped_distances)
    distance_sort = distance_sort.view(len(overfull_atoms), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute capped rows, retaining under-cap rows without sorting.
        capped_neighbors = torch.minimum(capped_counts, num_included_per_atom)
        if has_uncapped_atoms:
            num_neighbors_thresholded[overfull_atoms] = capped_neighbors
            # NaNs sort after infinity padding. An under-cap row's old cutoff
            # is NaN exactly when enough NaNs reach column `cap`; the old
            # comparison then excludes that whole row. Preserve that case
            # without its padded distances or a host-side finite check.
            nan_counts = torch.zeros_like(num_neighbors)
            nan_counts.scatter_add_(0, index, torch.isnan(atom_distance).long())
            valid_uncapped = nan_counts < (
                max_num_neighbors - max_num_neighbors_threshold
            )
            if math.isnan(math.inf + degeneracy_tolerance):
                valid_uncapped.zero_()
            mask_num_neighbors &= valid_uncapped[index]
            num_neighbors_thresholded.masked_fill_(~overfull & ~valid_uncapped, 0)
        else:
            num_neighbors_thresholded = capped_neighbors
        num_neighbors_image = sum_partitions(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + capped_offsets.view(-1, 1).expand(-1, max_num_included)
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Add selected capped edges to the already retained under-cap edges.
    mask_num_neighbors.index_fill_(0, index_sort, True)
    return mask_num_neighbors, num_neighbors_image


# @torch.no_grad() # breaks torch compile test with OOM
def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
    pbc = canonical_pbc(data, pbc)

    # v1 uses a single global PBC for the whole batch and cannot produce
    # correct graphs for batches where some systems are periodic and others
    # are not.  Use radius_graph_pbc_v2 for mixed-PBC batches.
    if is_mixed_pbc(data):
        raise RuntimeError(
            "radius_graph_pbc does not support batches with mixed PBC "
            "(some systems periodic, others not). Use radius_graph_pbc_v2 instead."
        )

    device = data.pos.device
    batch_size = len(data.natoms)

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

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
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep.item(), rep.item() + 1, device=device, dtype=data.cell.dtype)
        for rep in max_rep
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

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def canonical_pbc(data, pbc: torch.Tensor | None) -> list[bool]:
    """
    Resolve and normalize the periodic boundary condition (PBC) flags for a batch.

    This helper produces a canonical 1D PBC specification (a list of three booleans,
    one per lattice direction) that downstream graph construction code can rely on.

    Behavior:
      - If ``pbc`` is None, derive it from ``data.pbc``: a dimension is considered
        periodic only if at least one system in the batch is periodic along that
        dimension. ``data.pbc`` is also normalized to be at least 2D in-place.
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
    assert hasattr(data, "pbc"), "AtomicData does not have pbc set"

    if pbc is None:
        data.pbc = torch.atleast_2d(data.pbc)
        pbc = torch.any(data.pbc, dim=0)

    # If a per-system PBC tensor was passed in, collapse it via logical OR
    # so a dimension is periodic when any system requests it.
    if pbc.ndim > 1:
        pbc = pbc.any(dim=0)

    assert isinstance(pbc, torch.Tensor)
    assert pbc.ndim == 1
    assert pbc.shape[0] == 3

    return pbc.tolist()


def box_size_warning(cell, pos, pbc):
    if hasattr(box_size_warning, "already_printed"):
        return
    box_size_warning.already_printed = True
    logging.warning(
        f"PBCv2: graph generation encountered a very large box. The size of the cell is {cell} and min/max positions are {pos.min(),pos.max()}. Performance will be slower than optimal."
    )
    if any(pbc):
        logging.warning(
            "PBCv2: Does this system require PBC=True or will PBC=False work?"
        )


@torch.no_grad()
def radius_graph_pbc_v2(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
):
    # Preserve the per-system flags before canonical_pbc collapses them. An
    # explicit override (including always_use_pbc) must also control image
    # repetitions, not only the global warning/check flags.
    sys_pbc = torch.atleast_2d(data.pbc if pbc is None else pbc).to(
        device=data.pos.device, dtype=torch.bool
    )
    pbc = canonical_pbc(data, pbc)

    device = data.pos.device
    batch_size = len(data.natoms)
    if len(data.pos) == 0 or (
        hasattr(data, "node_partition") and len(data.node_partition) == 0
    ):
        return (
            torch.empty((2, 0), device=device, dtype=torch.long),
            data.cell.new_empty((0, 3)),
            torch.zeros_like(data.natoms, dtype=torch.long),
        )
    data_batch_idxs = (
        data.batch
        if data.batch is not None
        else data.pos.new_zeros(data.natoms[0], dtype=torch.int)
    )
    # Resolution of the grid cells, should be less than the radius
    grid_resolution = radius / 1.99

    # This function assumes that all atoms are within the unti cell. If atoms
    # are outside of the unit cell, it will not work correctly.
    # position of the atoms
    atom_pos = data.pos
    num_atoms = len(data.pos)
    num_atoms_per_image = data.natoms

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    # Use per-system PBC so mixed-PBC batches are handled correctly.
    # Each system's rep (number of periodic image repetitions) is computed from
    # its own cell and masked to zero for non-periodic dimensions.
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    rep_a1 = torch.where(
        sys_pbc[:, 0],
        torch.ceil(radius * inv_min_dist_a1),
        data.cell.new_zeros(batch_size),
    )

    cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    rep_a2 = torch.where(
        sys_pbc[:, 1],
        torch.ceil(radius * inv_min_dist_a2),
        data.cell.new_zeros(batch_size),
    )

    cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    rep_a3 = torch.where(
        sys_pbc[:, 2],
        torch.ceil(radius * inv_min_dist_a3),
        data.cell.new_zeros(batch_size),
    )

    rep = torch.cat([rep_a1.view(-1, 1), rep_a2.view(-1, 1), rep_a3.view(-1, 1)], dim=1)
    cells_per_image = (
        (rep[:, 0] * 2 + 1.0) * (rep[:, 1] * 2 + 1.0) * (rep[:, 2] * 2 + 1.0)
    ).long()

    # Create a tensor of unit cells for each image
    unit_cell = torch.zeros(
        torch.sum(cells_per_image), 3, device=device, dtype=data.cell.dtype
    )
    offset = 0
    for i in range(batch_size):
        cells_x = torch.arange(
            -rep[i][0], rep[i][0] + 1, device=device, dtype=data.cell.dtype
        )
        cells_y = torch.arange(
            -rep[i][1], rep[i][1] + 1, device=device, dtype=data.cell.dtype
        )
        cells_z = torch.arange(
            -rep[i][2], rep[i][2] + 1, device=device, dtype=data.cell.dtype
        )
        unit_cell[offset : cells_per_image[i] + offset] = torch.cartesian_prod(
            cells_x, cells_y, cells_z
        )
        offset = offset + cells_per_image[i]

    # Compute the x, y, z positional offsets for each cell in each image
    cell_matrix = torch.transpose(data.cell, 1, 2)
    cell_matrix = torch.repeat_interleave(cell_matrix, cells_per_image, dim=0)
    pbc_cell_offsets = torch.bmm(cell_matrix, unit_cell.view(-1, 3, 1)).squeeze(-1)

    # If node_partition exists, this means we want to generate only a partial graph
    # from a subset of target atoms to the entire set of source atoms
    if hasattr(data, "node_partition"):
        # Neighbor capping expects edges grouped by global target index.
        # Canonicalize a private index tensor without changing the caller's
        # partition; arbitrary input order otherwise corrupts capped graphs.
        node_partition, _ = torch.sort(data.node_partition)
        target_atom_pos = atom_pos[node_partition]
        target_atom_image = data_batch_idxs[node_partition]
    else:
        target_atom_pos = atom_pos
        target_atom_image = data_batch_idxs

    # Compute the position of the source atoms for the edges. There are
    # more source atoms than target atoms, since the source atoms are
    # tiled by the PBC cells.
    num_cells_per_atom = torch.repeat_interleave(cells_per_image, num_atoms_per_image)
    source_atom_index = torch.repeat_interleave(
        torch.arange(num_atoms, device=device).long(), num_cells_per_atom
    )
    source_atom_image = data_batch_idxs[source_atom_index]

    # Concatenate once: repeatedly growing this tensor copies all earlier
    # systems for every new system in a heterogeneous batch.
    pbc_cell_indices = []
    offset = 0
    for i in range(batch_size):
        cell_indices = (
            torch.arange(offset, offset + cells_per_image[i], device=device)
            .repeat(num_atoms_per_image[i])
            .long()
        )
        pbc_cell_indices.append(cell_indices)
        offset = offset + cells_per_image[i]

    pbc_cell_index = torch.cat(pbc_cell_indices)
    del pbc_cell_indices, cell_indices
    source_atom_pos = atom_pos[source_atom_index] + pbc_cell_offsets[pbc_cell_index]

    # Given the positions of all the atoms has been computed, we
    # split them up using a cubic grid to make pairwise comparisons
    # more computationally efficient. The resolution of the grid
    # is grid_resolution.

    # Compute the grid index for each dimension for each atom
    # The source tiles include the unshifted atoms and therefore every target.
    # One reduction suffices; Python max(tensor, tensor) also synchronizes an
    # accelerator unnecessarily to compare the two scalar tensors.
    max_coordinate = source_atom_pos.abs().max()
    if max_coordinate / grid_resolution > 200:
        box_size_warning(data.cell, data.pos, pbc)
        grid_resolution = max_coordinate / 200
    source_atom_grid = torch.floor(source_atom_pos / grid_resolution).long()
    target_atom_grid = torch.floor(target_atom_pos / grid_resolution).long()

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
    padding_size = math.floor(radius / grid_resolution) + 1
    num_padding_grid_cells = (
        (2 * padding_size + 1) * (2 * padding_size + 1) * (2 * padding_size + 1)
    )

    padding_offsets = torch.arange(
        -padding_size, padding_size + 1, device=device, dtype=torch.long
    )
    padding_offsets = padding_offsets.view(1, -1).repeat(3, 1)
    padding_offsets = torch.cartesian_prod(
        padding_offsets[0, :], padding_offsets[1, :], padding_offsets[2, :]
    )
    padding_offsets = padding_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    grid_index = target_atom_grid[:, None, :] + padding_offsets[target_atom_image]
    max_index = target_atom_grid_size.view(-1, 1, 3).expand(
        -1, num_padding_grid_cells, -1
    )
    out_of_bounds = torch.logical_and(grid_index.ge(0), grid_index.lt(max_index))
    out_of_bounds = torch.all(out_of_bounds, dim=2).ne(True)

    padding_offsets[:, :, 1] = padding_offsets[:, :, 1] * grid_size.unsqueeze(-1)[:, 0]
    padding_offsets[:, :, 2] = (
        padding_offsets[:, :, 2]
        * grid_size.unsqueeze(-1)[:, 0]
        * grid_size.unsqueeze(-1)[:, 1]
    )
    padding_offsets = torch.sum(padding_offsets, dim=2)

    # For each target, preserve cell traversal order while expanding only
    # actual source atoms, rather than padding every cell to the maximum
    # occupancy. Out-of-bounds cells use a key above every valid grid id.
    neighboring_grid_cells = (
        target_atom_grid_id[:, None] + padding_offsets[target_atom_image]
    )
    neighboring_grid_cells.masked_fill_(out_of_bounds, num_grid_cells)
    neighboring_grid_cells = neighboring_grid_cells.reshape(-1)
    # A direct lookup is faster for compact grids, but cap its size by the
    # number of tiled atoms so vacuum cannot turn it into a volume allocation.
    if num_grid_cells <= 4 * len(source_atom_grid_id):
        cell_to_occupied = torch.full(
            (num_grid_cells + 1,), -1, device=device, dtype=torch.long
        )
        cell_to_occupied[occupied_grid_ids] = torch.arange(
            len(occupied_grid_ids), device=device
        )
        occupied_index = cell_to_occupied[neighboring_grid_cells].clamp(min=0)
    else:
        occupied_index = torch.searchsorted(occupied_grid_ids, neighboring_grid_cells)
        occupied_index = occupied_index.clamp(max=len(occupied_grid_ids) - 1)
    candidate_counts = torch.where(
        occupied_grid_ids[occupied_index] == neighboring_grid_cells,
        grid_cell_atom_count[occupied_index],
        0,
    )
    candidate_starts = torch.cumsum(candidate_counts, 0) - candidate_counts
    source_offsets = torch.repeat_interleave(
        grid_cell_start[occupied_index] - candidate_starts, candidate_counts
    )
    source_atom_edge_index = sort_indices[
        torch.arange(len(source_offsets), device=device) + source_offsets
    ]
    target_atom_edge_index = torch.repeat_interleave(
        torch.arange(len(target_atom_pos), device=device),
        candidate_counts.view(-1, num_padding_grid_cells).sum(dim=1),
        output_size=len(source_atom_edge_index),
    )

    # Keep the same distance arithmetic, but release gathered positions after
    # subtraction instead of retaining both candidate-sized arrays to return.
    atom_distance_sqr = torch.sum(
        (
            target_atom_pos[target_atom_edge_index]
            - source_atom_pos[source_atom_edge_index]
        )
        ** 2,
        dim=1,
    )
    # Is the distance within the radius and not 0 (the same atom)
    within_radius_mask = torch.logical_and(
        atom_distance_sqr.le(radius * radius), atom_distance_sqr.ne(0.0)
    )
    source_atom_edge_index = torch.masked_select(
        source_atom_edge_index, within_radius_mask
    )
    target_atom_edge_index = torch.masked_select(
        target_atom_edge_index, within_radius_mask
    )
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, within_radius_mask)

    # Get the return values
    # The indices of the atoms for each edge
    source_idx = source_atom_index[source_atom_edge_index]
    target_idx = target_atom_edge_index
    # The cell for each source atom
    source_cell = unit_cell[pbc_cell_index[source_atom_edge_index]]

    # Reduce the number of neighbors for each atom to the
    # desired threshold max_num_neighbors_threshold

    # need to map the target_idx back to the reference frame of the original data before indexing
    # otherwise the neighbor count will be wrong
    # this will fail the test test_generate_graph_batch_partition in test_radius_graph_pbc.py
    if hasattr(data, "node_partition"):
        target_idx_for_num_neighbors = node_partition[target_idx]
    else:
        target_idx_for_num_neighbors = target_idx

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=target_idx_for_num_neighbors,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        target_idx = torch.masked_select(target_idx, mask_num_neighbors)
        source_idx = torch.masked_select(source_idx, mask_num_neighbors)
        source_cell = torch.masked_select(
            source_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        source_cell = source_cell.view(-1, 3)

    edge_index = torch.stack((source_idx, target_idx))

    # if we used node_partition, we need to map target indices back to original indices
    if hasattr(data, "node_partition"):
        edge_index[1] = node_partition[edge_index[1]]

    return edge_index, source_cell, num_neighbors_image

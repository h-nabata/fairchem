"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import torch
from monty.dev import requires

from fairchem.core.graph.radius_graph_pbc import (
    get_max_neighbors_mask,
    system_pbc_flags,
    validate_graph_geometry,
    validate_graph_parameters,
)

try:
    from nvalchemiops.torch.neighbors import neighbor_list
    from nvalchemiops.torch.neighbors.neighbor_utils import estimate_max_neighbors

    def nvalchemiops_installed() -> bool:
        return True

except ImportError as e:
    logging.debug(
        f"nvalchemiops not available: {e}. "
        "Install with `pip install nvalchemi-toolkit-ops`"
    )
    estimate_max_neighbors = None
    neighbor_list = None

    def nvalchemiops_installed() -> bool:
        return False


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.no_grad()
def get_neighbors_nvidia(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_neigh: int,
    method: str = "cell_list",
    enforce_max_neighbors_strictly: bool = False,
    batch: torch.Tensor | None = None,
    natoms: torch.Tensor | None = None,
    return_distances_sq: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Performs nearest neighbor search using NVIDIA nvalchemiops and returns edge index, distances,
    and cell offsets as tensors. Supports both single structure and batched inputs.

    Args:
        positions: Atomic positions tensor (N, 3) - device and dtype determine computation
        cell: Unit cell tensor (B, 3, 3) or (3, 3) - must be on same device as positions
        pbc: Periodic boundary conditions (B, 3) or (3,) boolean tensor - must be on same device as positions
        cutoff: Cutoff radius in Angstroms
        max_neigh: Maximum number of neighbors per atom; nonpositive disables the cap
        method: NVIDIA method to use ("naive" or "cell_list")
        enforce_max_neighbors_strictly: If True, strictly limit to max neighbors;
            if False, include additional neighbors within degeneracy tolerance
        batch: Optional batch tensor (N,) indicating which structure each atom belongs to
        natoms: Optional tensor (B,) with number of atoms per structure. If not provided,
            inferred as single structure with all atoms.
        return_distances_sq: If True, compute and return pairwise distances squared; if False, return None for distances

    Returns:
        c_index: Center atom indices (tensor, int32) - global indices if batched
        n_index: Neighbor atom indices (tensor, int32) - global indices if batched
        offsets: Cell offsets (tensor, int32, shape [num_edges, 3])
        distances_sq: Pairwise distances squared (tensor) accounting for PBC if distances_sq is True, otherwise None
    """
    # Validate input shapes
    assert (
        positions.ndim == 2
    ), f"positions must have shape (N, 3), got {positions.shape}"
    assert (
        positions.shape[1] == 3
    ), f"positions must have shape (N, 3), got {positions.shape}"
    assert cell.ndim in (
        2,
        3,
    ), f"cell must have shape (3, 3) or (B, 3, 3), got {cell.shape}"
    if cell.ndim == 2:
        assert cell.shape == (3, 3), f"cell must have shape (3, 3), got {cell.shape}"
    else:
        assert cell.shape[1:] == (
            3,
            3,
        ), f"cell must have shape (B, 3, 3), got {cell.shape}"
    assert pbc.ndim in (1, 2), f"pbc must have shape (3,) or (B, 3), got {pbc.shape}"
    if pbc.ndim == 1:
        assert pbc.shape == (3,), f"pbc must have shape (3,), got {pbc.shape}"
    else:
        assert pbc.shape[1] == 3, f"pbc must have shape (B, 3), got {pbc.shape}"
    if batch is not None:
        assert (
            batch.ndim == 1
        ), f"batch must have shape (N,) where N={positions.shape[0]}, got {batch.shape}"
        assert (
            batch.shape[0] == positions.shape[0]
        ), f"batch must have shape (N,) where N={positions.shape[0]}, got {batch.shape}"
    if natoms is not None:
        assert natoms.ndim == 1, f"natoms must have shape (B,), got {natoms.shape}"
    cutoff, max_neigh, enforce_max_neighbors_strictly = validate_graph_parameters(
        cutoff, max_neigh, enforce_max_neighbors_strictly
    )

    device = positions.device
    total_atoms = positions.shape[0]

    # Normalize inputs to batched format
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    if pbc.ndim == 1:
        pbc = pbc.unsqueeze(0)
    if batch is None:
        batch = (
            torch.zeros(total_atoms, dtype=torch.long, device=device)
            if natoms is None
            else torch.repeat_interleave(
                torch.arange(len(natoms), device=device), natoms.to(device)
            )
        )
    if natoms is None:
        natoms = torch.bincount(batch.long(), minlength=len(cell))
    else:
        # Host-side structure counts are common in wrappers. Capping allocates
        # on natoms.device, so normalize this small metadata tensor as well.
        natoms = natoms.to(device=device)

    # Normalize shared flags and cells before entering a native extension.
    # In particular, a one-dimensional PBC override applies to every system.
    if len(cell) == 1 and len(natoms) > 1:
        cell = cell.expand(len(natoms), -1, -1).contiguous()
    data = SimpleNamespace(pos=positions, cell=cell, natoms=natoms, batch=batch)
    validate_graph_geometry(data)
    if (
        positions.dtype not in {torch.float32, torch.float64}
        or cell.dtype != positions.dtype
    ):
        raise ValueError(
            "NVIDIA graph positions and cells must share float32/float64 dtype"
        )
    pbc = system_pbc_flags(data, pbc).expand(len(natoms), -1).contiguous()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("NVIDIA graph generation supports CPU/CUDA only")
    if total_atoms == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty((0, 3), dtype=torch.int32, device=device),
            positions.new_empty(0) if return_distances_sq else None,
        )

    # Request more neighbors to handle degeneracy to allow our max neighbors filter to work properly
    # The NVIDIA neighbor list doesn't prioritize closest neighbors, so we need
    # a large buffer to ensure we capture all neighbors within the cutoff.
    # This allows the mask to correctly include degenerate edges.
    # note estimate_max_neighbors(cutoff=6.0, safety_factor=2.0) = 640 which should be overly safe.
    buffer_max_neigh = estimate_max_neighbors(cutoff=cutoff, safety_factor=2.0)

    # Small epsilon to ensure atoms at exactly cutoff distance are included
    nvidia_cutoff = cutoff + 1e-6

    # nvalchemi wants cell=None and pbc=None for non-periodic systems and only
    # accepts right-handed cells. Reflect a basis vector, then transform the
    # returned integer shifts back to the caller's original lattice basis.
    cell_basis_signs = None
    neighbor_cell = None
    if bool(pbc.any()):
        volume = (cell[:, 0] * torch.cross(cell[:, 1], cell[:, 2], dim=-1)).sum(-1)
        if bool((pbc.any(-1) & ((volume == 0) | ~torch.isfinite(volume))).any()):
            raise ValueError(
                "Periodic graph generation requires a nonsingular finite-volume cell"
            )
        cell_basis_signs = torch.ones((len(cell), 3), dtype=torch.int32, device=device)
        cell_basis_signs[:, 0] = torch.where(volume < 0, -1, 1)
        neighbor_cell = cell * cell_basis_signs.unsqueeze(-1)
    neighbor_pbc = None if neighbor_cell is None else pbc

    # Counts report the complete degree, even if the caller-owned buffers are
    # too small. Never cap a truncated, unordered list: its missing neighbors
    # may be closer than every retained one. Retry with the measured capacity.
    for _ in range(3):
        neighbor_matrix = torch.full(
            (total_atoms, buffer_max_neigh),
            total_atoms,
            dtype=torch.int32,
            device=device,
        )
        neighbor_matrix_shifts = torch.zeros(
            (total_atoms, buffer_max_neigh, 3), dtype=torch.int32, device=device
        )
        num_neighbors = torch.zeros(total_atoms, dtype=torch.int32, device=device)
        neighbor_list(
            positions=positions,
            cutoff=nvidia_cutoff,
            cell=neighbor_cell,
            pbc=neighbor_pbc,
            batch_idx=batch.int(),
            method=f"batch_{method}",
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            num_neighbors=num_neighbors,
            half_fill=False,
        )
        if bool((num_neighbors < 0).any()):
            raise RuntimeError("NVIDIA neighbor counts overflowed their integer dtype")
        required_capacity = int(num_neighbors.max())
        if required_capacity <= buffer_max_neigh:
            break
        buffer_max_neigh = required_capacity
        neighbor_matrix = neighbor_matrix_shifts = num_neighbors = None
    else:
        raise RuntimeError(
            "NVIDIA neighbor counts exceeded capacity after three attempts"
        )

    # Conversion from neighbor matrix to edge list
    atom_indices = torch.arange(total_atoms, device=device).unsqueeze(1)
    neigh_indices = torch.arange(buffer_max_neigh, device=device).unsqueeze(0)
    valid_mask = neigh_indices < num_neighbors.unsqueeze(1)

    c_index = atom_indices.expand(-1, buffer_max_neigh)[valid_mask]
    n_index = neighbor_matrix[valid_mask]
    offsets = neighbor_matrix_shifts[valid_mask]
    if cell_basis_signs is not None:
        offsets = offsets * cell_basis_signs[batch[c_index]]
    # Boolean selection owns these edge arrays. Release the dense native
    # buffers before allocating edge geometry, especially for large MD systems.
    del neighbor_matrix, neighbor_matrix_shifts, num_neighbors, valid_mask

    # We used to sort the edges here but the ordering of the edges is no longer required, leaving this comment here for reference

    from fairchem.core.graph.compute import edge_cell_offsets

    # The native search uses an expanded cutoff to include equality. Remove
    # that expansion again before capping, with the input floating-point dtype.
    # The shared geometry helper avoids one 3x3 cell allocation per edge.
    edge_counts = (
        natoms.new_full((1,), len(c_index))
        if len(cell) == 1
        else torch.bincount(batch[c_index], minlength=len(cell))
    )
    offsets_cartesian = edge_cell_offsets(cell, offsets, edge_counts)
    distances_sq = (
        (positions[c_index] - (positions[n_index] + offsets_cartesian)).square().sum(-1)
    )
    keep = (distances_sq <= cutoff * cutoff) & (distances_sq != 0)
    if not bool(keep.all()):
        c_index, n_index = c_index[keep], n_index[keep]
        offsets, distances_sq = offsets[keep], distances_sq[keep]

    filter_max_neighbors = max_neigh > 0 and required_capacity > max_neigh

    if filter_max_neighbors and c_index.shape[0] > 0:
        # Apply max neighbors mask to handle degeneracy properly
        mask_num_neighbors, _ = get_max_neighbors_mask(
            natoms=natoms,
            index=c_index,
            atom_distance=distances_sq,
            max_num_neighbors_threshold=max_neigh,
            enforce_max_strictly=enforce_max_neighbors_strictly,
        )

        c_index = c_index[mask_num_neighbors]
        n_index = n_index[mask_num_neighbors]
        distances_sq = distances_sq[mask_num_neighbors]
        offsets = offsets[mask_num_neighbors]

    return c_index, n_index, offsets, distances_sq if return_distances_sq else None


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.no_grad()
def radius_graph_pbc_nvidia(
    data,
    radius: float,
    max_num_neighbors_threshold: int,
    enforce_max_neighbors_strictly: bool = False,
    pbc: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVIDIA-accelerated radius graph generation with PBC support.

    This function has the same interface as radius_graph_pbc and radius_graph_pbc_v2,
    allowing it to be used as a drop-in replacement in generate_graph.

    Args:
        data: Data object with pos, cell, natoms, pbc, and optionally batch attributes
        radius: Cutoff radius for neighbor search
        max_num_neighbors_threshold: Maximum number of neighbors per atom
        enforce_max_neighbors_strictly: If True, strictly limit to max neighbors;
            if False, include additional neighbors within degeneracy tolerance
        pbc: Periodic boundary conditions tensor (optional, uses data.pbc if not provided)

    Returns:
        edge_index: (2, num_edges) tensor with [source, target] indices
        cell_offsets: (num_edges, 3) tensor with integer cell offsets
        neighbors: (batch_size,) tensor with number of edges per structure
    """
    pos = data.pos
    natoms = data.natoms
    cell = data.cell
    device = pos.device
    batch_size = len(natoms)

    # Get batch tensor
    if hasattr(data, "batch") and data.batch is not None:
        batch = data.batch
    else:
        batch = torch.repeat_interleave(torch.arange(batch_size, device=device), natoms)

    # Get PBC tensor
    if pbc is None:
        pbc_tensor = (
            data.pbc
            if hasattr(data, "pbc")
            else torch.tensor([True, True, True], device=device)
        )
    else:
        pbc_tensor = pbc

    c_index, n_index, offsets, _ = get_neighbors_nvidia(
        positions=pos,
        cell=cell,
        pbc=pbc_tensor,
        cutoff=radius,
        max_neigh=max_num_neighbors_threshold,
        method="cell_list",
        enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
        batch=batch,
        natoms=natoms,
    )

    edge_batch = batch[c_index]
    num_neighbors_image = torch.zeros(batch_size, dtype=torch.long, device=device)
    num_neighbors_image.scatter_add_(0, edge_batch, torch.ones_like(edge_batch))

    edge_index = torch.stack([n_index, c_index], dim=0)

    return edge_index, offsets, num_neighbors_image


@requires(nvalchemiops_installed(), message="Requires `nvalchemiops` to be installed")
@torch.inference_mode()
def get_neighbors_nvidia_atoms(
    atoms,
    cutoff: float,
    max_neigh: int,
    *,
    target_dtype: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs nearest neighbor search using NVIDIA nvalchemiops and returns edge index, distances,
    and cell offsets.

    Args:
        atoms: ASE Atoms object
        cutoff: Cutoff radius in Angstroms
        max_neigh: Maximum number of neighbors per atom
        target_dtype: Coordinate precision used to select the neighbor topology.

    Returns:
        c_index: Center atom indices (numpy array)
        n_index: Neighbor atom indices (numpy array)
        distances: Pairwise distances (numpy array) accounting for PBC
        offsets: Cell offsets (numpy array)
    """
    if target_dtype not in (torch.float32, torch.float64):
        raise ValueError("NVIDIA ASE graph requires float32 or float64 coordinates")
    positions = torch.from_numpy(atoms.get_positions()).to(target_dtype)
    cell = torch.from_numpy(np.array(atoms.get_cell(complete=True))).to(target_dtype)
    pbc = torch.from_numpy(np.array(atoms.pbc)).bool()

    c_index, n_index, offsets, distances_sq = get_neighbors_nvidia(
        positions=positions,
        cell=cell,
        pbc=pbc,
        cutoff=cutoff,
        max_neigh=max_neigh,
        method="cell_list",
        enforce_max_neighbors_strictly=True,
        return_distances_sq=True,
    )

    return (
        c_index.numpy(),
        n_index.numpy(),
        np.sqrt(distances_sq.numpy()),
        offsets.numpy(),
    )

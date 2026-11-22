"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import torch
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce

# FP32 spacing reaches roughly 0.002 Å for Cartesian values at this scale.
WIDE_TRANSLATION_SCALE = 16384


def image_precision_limits(lattice_widths):
    """Per-axis image magnitudes at which FP32 can lose local precision."""
    return WIDE_TRANSLATION_SCALE / lattice_widths.clamp_min(1)


def needs_wide_image_math(images, lattice_rows):
    """Detect image IDs or Cartesian translations that lose local FP32 precision."""
    # Bound each periodic axis separately: a long vacuum axis with zero
    # translation should not send an ordinary slab calculation to FP64.
    widths = (
        lattice_rows.detach().abs().reshape(-1, images.shape[-1], 3).amax(dim=(0, 2))
    )
    # Cast integer IDs before abs: the most negative signed integer has no
    # positive representation. All IDs near the threshold are exact in FP32.
    values = images if images.is_floating_point() else images.float()
    return (values.abs() >= image_precision_limits(widths)).any()


def wide_image_device(device):
    # Portable inference must work on XPU devices without native FP64.
    return device if device.type in {"cpu", "cuda"} else torch.device("cpu")


def wrap_positions(
    pos: np.ndarray | torch.Tensor,
    cell: np.ndarray | torch.Tensor,
    pbc: np.ndarray | torch.Tensor | bool,
) -> np.ndarray | torch.Tensor:
    """
    Wrap positions into the unit cell along periodic dimensions.

    Equivalent to ase.geometry.wrap_positions with eps=0.  Dispatches to a
    pure-torch path for tensor inputs (preserves gradients) and
    a pure-numpy path for array inputs.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Unit cell with rows as lattice vectors, shape (3, 3).
        pbc: Periodic boundary flags — scalar bool or shape (3,).

    Returns:
        Wrapped positions, same type and shape as pos.
    """
    if isinstance(pos, torch.Tensor):
        return _wrap_torch(pos, cell, pbc)
    pos_np = np.asarray(pos, dtype=np.float64)
    cell_np = np.asarray(cell, dtype=np.float64)
    pbc_np = np.broadcast_to(np.asarray(pbc, dtype=bool), (3,)).copy()
    if not pbc_np.any():
        return np.array(pos, copy=True)

    if (pbc_np & (cell_np == 0).all(axis=1)).any():
        raise ValueError("Periodic lattice vectors must be nonzero")
    cell_np = complete_cell(cell_np)
    frac = np.linalg.solve(cell_np.T, pos_np.T).T
    for i in range(3):
        if pbc_np[i]:
            frac[:, i] %= 1.0
    return (frac @ cell_np).astype(np.asarray(pos).dtype)


def wrap_positions_torch(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Wrap positions into the unit cell (torch, batched per-atom cells).

    Each atom carries its own cell matrix — the natural layout after
    ``data.cell[data.batch]``.  Non-periodic dimensions are left unchanged.

    Args:
        pos: Cartesian positions, shape (n_atoms, 3).
        cell: Per-atom cells of shape (n_atoms, 3, 3), or (1, 3, 3) when shared.
        pbc: Per-atom flags of shape (n_atoms, 3), or (1, 3) for a shared cell.

    Returns:
        Wrapped positions, shape (n_atoms, 3).
    """
    if len(cell) == len(pbc) == 1:
        if not bool(pbc.any()):
            return pos.clone()
        basis = cell[0]
        with torch.no_grad():
            missing = (basis == 0).all(dim=-1)
            if bool(missing.any()):
                if bool((missing & pbc[0]).any()):
                    raise ValueError("Periodic lattice vectors must be nonzero")
                active = basis[~missing]
                q, r = torch.linalg.qr(active.T, mode="reduced")
                frac = torch.zeros_like(pos)
                frac[:, ~missing] = torch.linalg.solve(r, q.T @ pos.T).T
            else:
                # One factorization with N right-hand sides for a shared cell.
                frac = torch.linalg.solve(basis.T, pos.T).T
            images = torch.where(pbc, frac.floor(), 0)
        if pos.dtype == torch.float32 and bool(needs_wide_image_math(images, basis)):
            device = wide_image_device(pos.device)
            return wrap_positions_torch(
                pos.to(device, torch.float64),
                cell.to(device, torch.float64),
                pbc.to(device),
            ).to(pos)
        return pos - images @ basis
    periodic = pbc.any(dim=1)
    if not bool(periodic.any()):
        return pos.clone()
    active_cell = cell[periodic]
    active_pos = pos[periodic]
    # Lattice image numbers are piecewise constant. Only the final lattice
    # translation needs the autograd graph.
    with torch.no_grad():
        missing = (active_cell == 0).all(dim=-1)
        if bool(missing.any()):
            if bool((pbc[periodic] & missing).any()):
                raise ValueError("Periodic lattice vectors must be nonzero")
            # Solve full cells directly. For a missing vacuum vector, solve
            # for the projection onto the span of the present rows. Thin QR
            # avoids squaring the condition number of narrow skew cells.
            patterns = (
                missing.long() * missing.new_tensor([1, 2, 4], dtype=torch.long)
            ).sum(1)
            frac = torch.zeros_like(active_pos)
            for pattern in patterns.unique().tolist():
                systems = torch.where(patterns == pattern)[0]
                if pattern == 0:
                    frac[systems] = torch.linalg.solve(
                        active_cell[systems].transpose(1, 2),
                        active_pos[systems].unsqueeze(-1),
                    ).squeeze(-1)
                    continue
                axes = [i for i in range(3) if not (pattern & (1 << i))]
                basis = active_cell[systems][:, axes]
                q, r = torch.linalg.qr(basis.transpose(1, 2), mode="reduced")
                projected = q.transpose(1, 2) @ active_pos[systems].unsqueeze(-1)
                frac[systems[:, None], axes] = torch.linalg.solve(r, projected).squeeze(
                    -1
                )
        else:
            frac = torch.linalg.solve(
                active_cell.transpose(1, 2), active_pos.unsqueeze(-1)
            ).squeeze(-1)
        images = torch.where(pbc[periodic], frac.floor(), 0)
    if pos.dtype == torch.float32 and bool(needs_wide_image_math(images, active_cell)):
        device = wide_image_device(pos.device)
        return wrap_positions_torch(
            pos.to(device, torch.float64),
            cell.to(device, torch.float64),
            pbc.to(device),
        ).to(pos)
    result = pos.clone()
    result[periodic] = active_pos - (images.unsqueeze(1) @ active_cell).squeeze(1)
    return result


def compute_minkowski_op(
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Minkowski reduction operator for a single cell.

    Args:
        cell: Unit cell, shape (3, 3) or (1, 3, 3).
        pbc: Periodic flags, shape (3,) or (1, 3).

    Returns:
        Integer reduction operator, shape matching input (3, 3) or (1, 3, 3).
    """
    squeeze = cell.dim() == 2
    c = cell.squeeze(0) if not squeeze else cell
    p = pbc.squeeze(0) if pbc.dim() == 2 else pbc

    _, op = minkowski_reduce(c.detach().cpu().numpy(), pbc=p.detach().cpu().numpy())
    op_t = torch.tensor(op, dtype=cell.dtype, device=cell.device)

    if not squeeze:
        return op_t.unsqueeze(0)
    return op_t


def minimum_image_displacement(
    dr: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch: torch.Tensor | None = None,
    minkowski_op: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply minimum image convention to displacement vectors.

    Uses Minkowski reduction to transform the cell into a compact basis where
    a 27-neighbour search is guaranteed to find the true shortest image.  This
    is correct for arbitrary cell shapes, including highly skewed cells.
    Non-periodic dimensions are unchanged.

    Args:
        dr: Displacement vectors, shape (n_atoms, 3).
        cell: Unit cells, per system when batch is supplied, per atom, or
            one shared cell of shape (3, 3) or (1, 3, 3).
        pbc: Periodic flags — either per-system (n_systems, 3) when ``batch``
            is provided, or per-atom (n_atoms, 3).
        batch: System index for each atom, shape (n_atoms,).  When provided,
            ``cell`` and ``pbc`` are per-system and will be expanded
            internally.
        minkowski_op: Precomputed Minkowski reduction operators, per-system
            (n_systems, 3, 3). Recompute when the cell or periodic axes
            change. If ``None``, computed on the fly.

    Returns:
        Minimum-image displacements, shape (n_atoms, 3).
    """
    if not len(dr):
        return dr
    if cell.ndim == 2:
        cell = cell.unsqueeze(0)
    pbc = torch.atleast_2d(pbc)
    if batch is None:
        if len(cell) == 1 and len(pbc) == 1:
            # One shared lattice: every displacement belongs to system zero.
            batch = torch.zeros(len(dr), dtype=torch.long, device=dr.device)
        elif len(cell) in (1, len(dr)):
            cell = cell.expand(len(dr), -1, -1)
        else:
            raise ValueError(
                "Supply a shared cell or one cell per displacement when batch is omitted"
            )
    pbc = pbc.expand(len(cell), 3)
    if minkowski_op is not None:
        minkowski_op = minkowski_op.expand(len(cell), -1, -1)
    # Eight patterns suffice. A zero cell in a nonperiodic system must never
    # enter a solve or division. Distinct axis-aligned cells share the fast path.
    powers = torch.tensor([1, 2, 4], device=pbc.device)
    patterns = (pbc.long() * powers).sum(-1)
    diagonal = cell - torch.diag_embed(cell.diagonal(dim1=-2, dim2=-1))
    patterns = patterns + 8 * (diagonal == 0).all(dim=(1, 2)).long()
    groups = torch.unique(patterns).tolist()
    if all(pattern % 8 == 0 for pattern in groups):
        return dr
    dtype = torch.promote_types(dr.dtype, cell.dtype)
    dr, cell = dr.to(dtype), cell.to(dtype)
    result = dr.clone()
    for pattern in groups:
        if pattern % 8 == 0:
            continue
        sys_indices = torch.where(patterns == pattern)[0]
        if batch is None:
            atom_indices = sys_indices
            local = torch.arange(len(sys_indices), device=dr.device)
        else:
            atom_indices = torch.where(patterns[batch] == pattern)[0]
            local = torch.searchsorted(sys_indices, batch[atom_indices])
        axes = [i for i in range(3) if pattern & (1 << i)]
        cells = cell[sys_indices]
        vectors = dr[atom_indices]
        if pattern >= 8:
            lengths = cells.diagonal(dim1=-2, dim2=-1)[local][:, axes]
            if bool((lengths == 0).any()):
                raise ValueError("Periodic lattice vectors must be nonzero")
            coordinates = vectors[:, axes]
            frac = coordinates / lengths
        else:
            if len(axes) > 1:
                if minkowski_op is None:
                    cells = _minkowski_reduce_batched(cells, pbc[sys_indices])
                else:
                    cells = minkowski_op[sys_indices].to(cells) @ cells
            basis = cells[:, axes]
            # Project onto the periodic subspace. Full-cell fractional
            # coordinates depend on arbitrary vacuum vectors for partial PBC.
            if len(axes) == 3:
                dual = torch.linalg.inv(basis).transpose(1, 2)
            else:
                dual = torch.linalg.solve(basis @ basis.transpose(1, 2), basis)
            frac = (dual[local] @ vectors.unsqueeze(-1)).squeeze(-1)
        if dtype == torch.float32 and bool(needs_wide_image_math(frac, cells[:, axes])):
            device = wide_image_device(dr.device)
            # Reduce from the original cell, before any FP32 rounding.
            ops = (
                None
                if minkowski_op is None
                else minkowski_op[sys_indices].to(device, torch.float64)
            )
            wrapped = minimum_image_displacement(
                vectors.to(device, torch.float64),
                cell[sys_indices].to(device, torch.float64),
                pbc[sys_indices].to(device),
                local.to(device),
                ops,
            ).to(dr)
        elif pattern >= 8:
            wrapped = vectors.clone()
            wrapped[:, axes] = coordinates - frac.round() * lengths
        else:
            image = (frac + 0.5).floor()
            centered = vectors - (image.unsqueeze(1) @ basis[local]).squeeze(1)
            wrapped = centered
            if len(axes) > 1:
                signs = dr.new_tensor([-1, 0, 1])
                shifts = torch.cartesian_prod(*([signs] * len(axes)))
                translations = shifts @ basis
                # All images have the same displacement normal to the periodic
                # subspace. Including its squared norm can erase the entire
                # difference between neighboring images in floating point.
                projected = (
                    centered
                    if len(axes) == 3
                    else ((frac - image).unsqueeze(1) @ basis[local]).squeeze(1)
                )
                candidates = projected[:, None] + translations[local]
                best = candidates.square().sum(-1).argmin(-1)
                wrapped = centered + translations[local, best]
        result[atom_indices] = wrapped
    return result


def _minkowski_reduce_batched(
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """
    Minkowski-reduce each cell in a batch.

    Computes integer unimodular operators ``op`` via ASE (cheap integer
    arithmetic on 3x3 matrices), then applies ``rcell = op @ cell`` in
    torch so that ``cell`` stays in the autograd graph.

    Args:
        cell: Unit cells, shape (n_systems, 3, 3).
        pbc: Periodic flags, shape (n_systems, 3).

    Returns:
        Reduced cells, shape (n_systems, 3, 3), same dtype/device as cell.
    """
    cell_cpu = cell.detach().cpu()
    pbc_cpu = pbc.detach().cpu()
    n = cell.shape[0]

    ops = np.stack(
        [
            minkowski_reduce(cell_cpu[i].numpy(), pbc=pbc_cpu[i].numpy())[1]
            for i in range(n)
        ]
    )
    return cell.new_tensor(ops) @ cell


def _wrap_torch(
    pos: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor | bool,
) -> torch.Tensor:
    """Single-system torch wrap with one shared lattice."""
    shared = torch.as_tensor(cell, dtype=pos.dtype, device=pos.device).reshape(1, 3, 3)
    flags = torch.as_tensor(pbc, dtype=torch.bool, device=pos.device)
    if flags.ndim == 0:
        flags = flags.expand(3)
    flags = torch.atleast_2d(flags)
    if len(flags) > 1:
        shared = shared.expand(len(pos), -1, -1)
    return wrap_positions_torch(pos, shared, flags)

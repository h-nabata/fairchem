"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.nn.approximate_ops import (
    compensated_node_sum,
    compensated_reduction_enabled,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


def resolve_requested_properties(
    regress_config,
    requested_properties: Iterable[str] | None = None,
) -> frozenset[str]:
    """
    Resolve an output selection without adding internal derivative dependencies.

    ``None`` retains the historical EFS-head output selection. Explicit requests
    select returned properties only: a Hessian still needs forces internally,
    but does not require returning them. Model/head capability checks belong to
    the caller; this helper validates the public property names.
    """
    supported = frozenset(
        {
            "energy",
            "forces",
            "stress",
            "hessian",
            "hessian_vector_product",
            "embeddings",
        }
    )
    if requested_properties is not None:
        if isinstance(requested_properties, str):
            raise TypeError(
                "requested_properties must be an iterable of property names"
            )
        properties = frozenset(requested_properties)
        unknown = properties - supported
        if unknown:
            raise ValueError(
                f"Unknown requested properties: {sorted(unknown, key=str)}"
            )
        return properties

    properties = {"energy", "embeddings"}
    if regress_config.forces or (
        regress_config.stress and not regress_config.direct_stress
    ):
        properties.add("forces")
    if regress_config.stress:
        properties.add("stress")
    if regress_config.hessian:
        properties.add("hessian")
    return frozenset(properties)


def _energy_gradients(
    energy: torch.Tensor,
    inputs: tuple[torch.Tensor, ...],
    create_graph: bool,
) -> tuple[torch.Tensor, ...]:
    """
    Differentiate energy, including constant/unused coordinate components.
    """
    if any(not value.requires_grad for value in inputs):
        raise ValueError("Derivative inputs must have requires_grad=True")
    if energy.requires_grad:
        gradients = torch.autograd.grad(
            energy.sum(), inputs, create_graph=create_graph, allow_unused=True
        )
    else:
        gradients = (None,) * len(inputs)
    result = []
    for value, input_gradient in zip(inputs, gradients):
        if input_gradient is None:
            gradient = value * 0 if create_graph else torch.zeros_like(value)
        elif create_graph and not input_gradient.requires_grad:
            # A linear energy has constant forces. Preserve its exact zero
            # curvature without treating arbitrary detached force inputs as
            # valid Hessian inputs later.
            gradient = input_gradient + value * 0
        else:
            gradient = input_gradient
        result.append(gradient)
    return tuple(result)


def get_l_component_range(x: torch.Tensor, l_min: int, l_max: int) -> torch.Tensor:
    """Extract spherical harmonic components for L in [l_min, l_max] from node embeddings.

    The node embeddings are assumed to be organized as [N, (lmax+1)^2, C] where the
    second dimension contains spherical harmonic coefficients ordered by L:
    - L=0: index 0 (1 component)
    - L=1: indices 1-3 (3 components)
    - L=2: indices 4-8 (5 components)
    - etc.

    Args:
        x: Node embeddings tensor of shape [N, (lmax+1)^2, C].
        l_min: Lowest angular momentum quantum number to include (0, 1, 2, ...).
        l_max: Highest angular momentum quantum number to include (>= l_min).

    Returns:
        Tensor of shape [N, (l_max+1)^2 - l_min^2, C] containing the concatenated
        spherical harmonic components for all L in [l_min, l_max].
    """
    start_idx = l_min * l_min
    num_components = (l_max + 1) ** 2 - l_min**2
    return x.narrow(1, start_idx, num_components)


def reduce_node_to_system(
    node_values: torch.Tensor,
    batch: torch.Tensor,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce node-level values to system-level by summing over nodes in each system.

    Handles graph-parallel (GP) reduction when GP is initialized.

    Args:
        node_values: Node-level values of shape [N, ...] where N is the number of nodes.
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.

    Returns:
        A tuple of (reduced, unreduced) where:
        - reduced: System-level values after GP reduction (if applicable), shape [num_systems, ...].
        - unreduced: System-level values before GP reduction, useful for autograd, shape [num_systems, ...].
    """
    if compensated_reduction_enabled():
        system_values = compensated_node_sum(node_values, batch, num_systems).double()
    else:
        output_shape = (num_systems,) + node_values.shape[1:]
        system_values = torch.zeros(
            output_shape,
            device=node_values.device,
            dtype=torch.float64,
        )

        # index_add_ supports arbitrary trailing dimensions directly, including
        # strided and empty inputs. Flattening with view() rejects both cases and
        # using reshape() instead would introduce an unnecessary contiguous copy.
        system_values.index_add_(0, batch, node_values.to(system_values.dtype))

    if gp_utils.initialized():
        reduced = gp_utils.reduce_from_model_parallel_region(system_values)
    else:
        reduced = system_values

    return reduced, system_values


# Compile produces the wrong values using index_add with float64 precision :(
@torch.compiler.disable
def compute_energy(
    emb: dict[str, torch.Tensor],
    energy_block: torch.nn.Module,
    batch: torch.Tensor,
    num_systems: int,
    natoms: torch.Tensor | None = None,
    reduce: Literal["sum", "mean"] = "sum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute system-level energy from node embeddings and an energy block.

    Extracts the L=0 (scalar) component from node embeddings, applies the energy
    block to produce per-node energies, reduces to system-level energies, and
    optionally normalizes by the number of atoms per system.

    Args:
        emb: Embedding dictionary containing "node_embedding" of shape [N, (lmax+1)^2, C].
        energy_block: Module that maps scalar node features [N, C] to per-node energies [N, 1].
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.
        natoms: Number of atoms per system, shape [num_systems]. Required when reduce="mean".
        reduce: How to aggregate node energies into system energies. "sum" returns the total
            energy; "mean" divides by natoms to return the average energy per atom.

    Returns:
        A tuple of (energy, energy_part) where:
        - energy: System-level energy after GP reduction and reduce, shape [num_systems].
        - energy_part: System-level energy before GP reduction, with the same
            sum/mean normalization as energy (for autograd), shape [num_systems].
    """
    scalar_embedding = get_l_component_range(
        emb["node_embedding"], l_min=0, l_max=0
    ).squeeze(1)
    node_energy = energy_block(scalar_embedding)
    node_energy_flat = node_energy.view(-1)
    energy, energy_part = reduce_node_to_system(
        node_energy_flat,
        batch,
        num_systems,
    )

    if reduce == "sum":
        pass
    elif reduce == "mean":
        if natoms is None:
            raise ValueError("natoms must be provided when reduce='mean'")
        energy = energy / natoms
        # Derivatives must describe the reported mean energy too. Under GP,
        # natoms is the global atom count, not the local partition size.
        energy_part = energy_part / natoms
    else:
        raise ValueError(f"reduce can only be sum or mean, got: {reduce}")

    return energy, energy_part


def compute_forces(
    energy_part: torch.Tensor,
    pos: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute forces as negative gradient of energy with respect to positions.

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        training: Whether to create graph for higher-order gradients.

    Returns:
        Forces tensor of shape [N, 3].
    """
    (grad,) = _energy_gradients(energy_part, (pos,), create_graph=training)
    forces = torch.neg(grad)

    if gp_utils.initialized():
        forces = gp_utils.reduce_from_model_parallel_region(forces)

    return forces


def compute_forces_and_stress(
    energy_part: torch.Tensor,
    pos: torch.Tensor,
    cell: torch.Tensor,
    batch: torch.Tensor,
    training: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute forces and stress from energy using autograd.

    Forces are computed as the negative gradient of energy with respect to positions.
    Stress is computed by reconstructing the virial tensor from the position and cell
    gradients, equivalent to the strain-derivative approach.

    The virial is:
        V = (g_r^T r + r^T g_r) / 2 + (cell^T g_h + g_h^T cell) / 2

    where g_r = dE/dpos and g_h = dE/dcell. This matches dE/dε for a symmetric
    strain ε applied as r' = r(I + ε), h' = h(I + ε).

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        cell: Unit cell vectors, shape [num_systems, 3, 3].
        batch: Batch indices mapping each node to its system, shape [N].
        training: Whether to create graph for higher-order gradients.

    Returns:
        A tuple of (forces, stress) where:
        - forces: Shape [N, 3].
        - stress: Shape [num_systems, 9] (flattened 3x3 tensor).
    """
    grads = _energy_gradients(energy_part, (pos, cell), create_graph=training)

    num_systems = cell.shape[0]

    # Compute pos_virial from LOCAL (unreduced) gradients so that
    # reduce_node_to_system performs the single correct GP all-reduce.
    # Using already-reduced gradients here would cause double-reduction
    # since reduce_node_to_system also calls reduce_from_model_parallel_region.
    pos_virial_per_atom = grads[0].unsqueeze(2) * pos.unsqueeze(1)  # [N, 3, 3]
    pos_virial, _ = reduce_node_to_system(pos_virial_per_atom, batch, num_systems)

    # Now all-reduce gradients for forces and cell_virial
    if gp_utils.initialized():
        grad_pos = gp_utils.reduce_from_model_parallel_region(grads[0])
        grad_cell = gp_utils.reduce_from_model_parallel_region(grads[1])
    else:
        grad_pos, grad_cell = grads[0], grads[1]

    forces = torch.neg(grad_pos)
    cell_virial = cell.mT @ grad_cell  # [B, 3, 3]

    virial = (pos_virial + pos_virial.mT + cell_virial + cell_virial.mT) / 2
    volume = torch.det(cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = stress.view(-1, 9)

    return forces, stress


def compute_hessian_vmap(
    forces_flat: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool,
    chunk_size: int | None = 64,
) -> torch.Tensor:
    """
    Compute Hessian rows in bounded vectorized blocks.

    The final dense Hessian still requires quadratic storage. Unlike a full
    identity/vmap, the basis and simultaneous backward intermediates scale with
    ``chunk_size`` rather than the number of coordinates. ``None`` explicitly
    requests one unbounded block.

    Args:
        forces_flat: Flattened forces tensor, shape [N*3].
        pos: Atomic positions, shape [N, 3].
        create_graph: Whether to create graph for third-order derivatives.
        chunk_size: Maximum rows per backward block, or None for all rows.

    Returns:
        Hessian matrix of shape [N*3, N*3].
    """

    _validate_curvature_inputs(forces_flat, pos)
    n_dof = forces_flat.numel()
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise ValueError("hessian chunk_size must be a positive integer or None")
    if n_dof == 0:
        return _empty_hessian(forces_flat, pos, create_graph)
    block_size = min(chunk_size or n_dof, n_dof)
    # In inference, write directly into final storage rather than retaining all
    # block tensors and concatenating another complete Hessian at the end.
    hessian = None if create_graph else forces_flat.new_empty((n_dof, n_dof))
    blocks = []
    negative_forces = -forces_flat
    for start in range(0, n_dof, block_size):
        stop = min(start + block_size, n_dof)
        basis = forces_flat.new_zeros((stop - start, n_dof))
        rows = torch.arange(stop - start, device=forces_flat.device)
        basis[rows, rows + start] = 1

        def compute_grad_component(vec, stop=stop):
            gradient = torch.autograd.grad(
                negative_forces,
                pos,
                grad_outputs=vec,
                retain_graph=create_graph or stop < n_dof,
                create_graph=create_graph,
                allow_unused=True,
            )[0]
            if gradient is None:
                return _zero_curvature(pos, forces_flat, create_graph)
            if create_graph and not gradient.requires_grad:
                gradient = gradient + pos * 0
            return gradient

        block = torch.vmap(compute_grad_component)(basis).reshape(stop - start, n_dof)
        if create_graph:
            blocks.append(block)
        else:
            hessian[start:stop].copy_(block)
    return torch.cat(blocks, dim=0) if create_graph else hessian


def compute_hessian_loop(
    forces_flat: torch.Tensor,
    pos: torch.Tensor,
    create_graph: bool,
) -> torch.Tensor:
    """Compute Hessian using a loop over force components.

    Iteratively computes gradients of each force component with respect to
    positions. This is a fallback when vmap is not desired or unavailable.

    Args:
        forces_flat: Flattened forces tensor, shape [N*3].
        pos: Atomic positions, shape [N, 3].
        create_graph: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [N*3, N*3].
    """
    _validate_curvature_inputs(forces_flat, pos)
    n_forces = forces_flat.numel()
    if n_forces == 0:
        return _empty_hessian(forces_flat, pos, create_graph)
    hessian = torch.zeros(
        (n_forces, n_forces),
        device=forces_flat.device,
        dtype=forces_flat.dtype,
        requires_grad=False,
    )

    for i in range(n_forces):
        gradient = torch.autograd.grad(
            -forces_flat[i],
            pos,
            retain_graph=create_graph or i < n_forces - 1,
            create_graph=create_graph,
            allow_unused=True,
        )[0]
        if gradient is None:
            gradient = _zero_curvature(pos, forces_flat, create_graph)
        elif create_graph and not gradient.requires_grad:
            gradient = gradient + pos * 0
        hessian[i] = gradient.flatten()

    return hessian


def compute_hessian(
    forces: torch.Tensor,
    pos: torch.Tensor,
    vmap: bool = True,
    training: bool = False,
    chunk_size: int | None = 64,
) -> torch.Tensor:
    """Compute Hessian matrix as second derivative of energy w.r.t. positions.

    The Hessian is computed as the negative gradient of forces with respect to
    positions: H = -∇_pos(forces) = ∇²_pos(energy).

    Args:
        forces: Force tensor, shape [N, 3].
        pos: Atomic positions, shape [N, 3].
        vmap: Whether to use vectorized mapping (faster but higher memory).
        training: Whether to create graph for third-order derivatives.
        chunk_size: Maximum vmap rows per backward block. None selects all rows.

    Returns:
        Hessian matrix of shape [1, N*3, N*3] (batch dim always 1 since
        hessian requires single-system batches).

    Note:
        Graph parallel (GP) mode is not fully supported. The Hessian should
        be computed after forces have been reduced across GP ranks.
    """
    if gp_utils.initialized():
        raise NotImplementedError(
            "Hessian computation is not currently supported with graph parallel mode. "
            "Please compute Hessian on a single rank after force reduction."
        )

    forces_flat = forces.flatten()

    if vmap:
        hessian = compute_hessian_vmap(
            forces_flat, pos, create_graph=training, chunk_size=chunk_size
        )
    else:
        hessian = compute_hessian_loop(forces_flat, pos, create_graph=training)

    return hessian.unsqueeze(0)


def _validate_curvature_inputs(forces: torch.Tensor, pos: torch.Tensor) -> None:
    """
    Reject missing derivative graphs instead of silently returning zero curvature.
    """
    if forces.numel() != pos.numel():
        raise ValueError("forces and positions must have the same number of elements")
    if forces.device != pos.device:
        raise ValueError("forces and positions must be on the same device")
    if forces.numel() and (not pos.requires_grad or not forces.requires_grad):
        raise ValueError(
            "Curvature requires differentiable forces and positions; "
            "compute forces with training=True/create_graph=True"
        )


def _zero_curvature(
    pos: torch.Tensor, forces: torch.Tensor, create_graph: bool
) -> torch.Tensor:
    """
    Materialize zero curvature while preserving zero higher derivatives.
    """
    if create_graph:
        return pos * 0 + forces.sum() * 0
    return torch.zeros_like(pos)


def _empty_hessian(
    forces: torch.Tensor, pos: torch.Tensor, create_graph: bool
) -> torch.Tensor:
    result = forces.new_empty((0, 0))
    if create_graph:
        result = result + (pos.sum() + forces.sum()) * 0
    return result


def compute_hessian_vector_product(
    forces: torch.Tensor,
    pos: torch.Tensor,
    vector: torch.Tensor,
    training: bool = False,
    retain_graph: bool | None = None,
) -> torch.Tensor:
    """
    Compute an exact Hessian-vector product without materializing the Hessian.

    Forces must be the differentiable negative gradient of the same scalar
    energy with respect to ``pos``. The reverse-over-reverse calculation gives
    ``H.T @ vector``, equal to ``H @ vector`` for this conservative energy.
    Direct/nonconservative force models do not satisfy this contract.

    Args:
        forces: Differentiable conservative forces, shape [N, 3].
        pos: Positions used to compute forces, shape [N, 3].
        vector: A vector with shape pos.shape, or flattened shape [N*3]. For
            independent systems in one batch, the product applies their block
            diagonal Hessian to the corresponding slices of this vector.
        training: Whether to retain differentiability of the product.
        retain_graph: Keep the force graph for additional products or a full
            Hessian. Defaults to training, as in torch.autograd.grad.

    Returns:
        Hessian-vector product with the same shape as vector.
    """
    if gp_utils.initialized():
        raise NotImplementedError(
            "Hessian-vector products are not supported with graph parallel mode"
        )
    _validate_curvature_inputs(forces, pos)
    if vector.shape not in (pos.shape, torch.Size([pos.numel()])):
        raise ValueError("vector must have position shape or flattened position shape")
    if vector.device != pos.device or vector.dtype != forces.dtype:
        raise ValueError("vector must have the force dtype and position device")
    if pos.numel() == 0:
        result = _zero_curvature(pos, forces, training).reshape(vector.shape)
        return result + vector * 0 if training else result
    gradient = torch.autograd.grad(
        -forces.reshape(-1),
        pos,
        grad_outputs=vector.reshape(-1),
        create_graph=training,
        retain_graph=training if retain_graph is None else retain_graph,
        allow_unused=True,
    )[0]
    if gradient is None:
        gradient = _zero_curvature(pos, forces, training)
        if training:
            gradient = gradient + vector.reshape(pos.shape) * 0
    elif training and not gradient.requires_grad:
        gradient = gradient + pos * 0 + vector.reshape(pos.shape) * 0
    return gradient.reshape(vector.shape)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from fairchem.core.graph.radius_graph_pbc import (
    radius_graph_pbc,
    radius_graph_pbc_v2,
    validate_graph_parameters,
)

if TYPE_CHECKING:
    from fairchem.core.graph.neighbor_cache import VerletNeighborCache


def edge_cell_offsets(cell, cell_offsets, neighbors):
    """
    Apply each system's cell without materializing one 3x3 matrix per edge.

    The operations remain differentiable with respect to the cell, including
    double backward. The single-system path uses one ordinary matrix product.
    """
    coefficients = cell_offsets.to(dtype=cell.dtype, device=cell.device)
    if len(cell) == 1:
        return coefficients @ cell[0]
    systems = torch.repeat_interleave(
        torch.arange(len(cell), device=cell.device),
        neighbors.to(cell.device),
        output_size=len(cell_offsets),
    )
    return (
        coefficients[:, 0:1] * cell[:, 0].index_select(0, systems)
        + coefficients[:, 1:2] * cell[:, 1].index_select(0, systems)
        + coefficients[:, 2:3] * cell[:, 2].index_select(0, systems)
    )


def spatially_order_graph(data, graph):
    """
    Order selected edges spatially without changing atom IDs or cap decisions.

    Ten-bit Morton coordinates are only sorting keys: physical positions and
    floating point computations are unchanged. Systems remain contiguous so
    the per-system edge counts still describe the returned tensors.
    """
    edges = graph["edge_index"]
    if edges.shape[1] < 2:
        return graph
    with torch.no_grad():
        systems = data.batch
        if systems is None:
            systems = torch.zeros(
                len(data.pos), device=data.pos.device, dtype=torch.long
            )
        lower = data.pos.new_zeros((len(data.natoms), 3))
        upper = torch.zeros_like(lower)
        indices = systems[:, None].expand(-1, 3)
        lower.scatter_reduce_(0, indices, data.pos, reduce="amin", include_self=False)
        upper.scatter_reduce_(0, indices, data.pos, reduce="amax", include_self=False)
        extent = (upper - lower).clamp_min(torch.finfo(data.pos.dtype).tiny)
        coordinates = (
            ((data.pos - lower[systems]) / extent[systems] * 1023).clamp(0, 1023).long()
        )
        # Vectorized bit spreading avoids thirty per-coordinate/per-bit loops.
        spread = (coordinates | (coordinates << 16)) & 0x030000FF
        spread = (spread | (spread << 8)) & 0x0300F00F
        spread = (spread | (spread << 4)) & 0x030C30C3
        spread = (spread | (spread << 2)) & 0x09249249
        morton = spread[:, 0] | (spread[:, 1] << 1) | (spread[:, 2] << 2)
        order = torch.argsort(morton[edges[0]], stable=True)
        order = order[torch.argsort(morton[edges[1, order]], stable=True)]
        order = order[torch.argsort(systems[edges[1, order]], stable=True)]
    result = dict(graph)
    result["edge_index"] = edges[:, order]
    for key in (
        "edge_distance",
        "edge_distance_vec",
        "cell_offsets",
        "offset_distances",
    ):
        if key in result:
            result[key] = result[key][order]
    return result


def filter_edges_by_node_partition(
    node_partition: torch.Tensor,
    edge_index: torch.Tensor,
    cell_offsets: torch.Tensor,
    neighbors: torch.Tensor,
    num_atoms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Filter edges to keep only those where the target atom belongs to the node partition.
    edge_index is shape (2, num_edges) where the first row is the source atom index and the second row is the target atom index for each edge
    cell_offsets is shape (num_edges, 3)
    neighbors is cardinality of the edge_index per system in the batch

    Args:
        node_partition: Tensor of atom indices belonging to the current rank's partition.
        edge_index: Edge index tensor of shape (2, num_edges), where row 0 is the source and 1 is the target atom.
        cell_offsets: Cell offsets tensor of shape (num_edges, 3).
        neighbors: Tensor with edge count per system in the batch (length = num_systems).
        num_atoms: Total number of atoms across all batches. Used to create a boolean mask for filtering.

    Returns:
        Filtered edge_index, cell_offsets, and neighbors tensors.
    """
    target_atoms = edge_index[1]
    node_mask = torch.zeros(num_atoms, dtype=torch.bool, device=target_atoms.device)
    node_mask[node_partition] = True
    local_edge_mask = node_mask[target_atoms]

    # Create system index for each edge to track which system each edge belongs to
    num_systems = neighbors.shape[0]
    edge_system_idx = torch.repeat_interleave(
        torch.arange(num_systems, device=neighbors.device), neighbors
    )

    # Filter edges
    edge_index = edge_index[:, local_edge_mask]
    cell_offsets = cell_offsets[local_edge_mask]
    if neighbors.shape[0] == 1:
        # If there's only one system, we can skip the scatter_add step and just return the count of remaining edges
        new_neighbors = local_edge_mask.sum(dtype=neighbors.dtype).unsqueeze(0)
        return edge_index, cell_offsets, new_neighbors

    filtered_edge_system_idx = edge_system_idx[local_edge_mask]

    # Count edges per system after filtering
    new_neighbors = torch.zeros(
        num_systems, device=neighbors.device, dtype=neighbors.dtype
    )
    new_neighbors.scatter_add_(
        0,
        filtered_edge_system_idx,
        torch.ones_like(filtered_edge_system_idx, dtype=neighbors.dtype),
    )

    return edge_index, cell_offsets, new_neighbors


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    offsets = edge_cell_offsets(cell, cell_offsets, neighbors)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # Graph-search and geometry arithmetic associate the periodic additions
    # differently. A nonzero candidate can therefore round to zero here. Keep
    # topology metadata aligned when removing it, including in mixed batches.
    nonzero_mask = distances != 0
    nonzero_idx = torch.arange(len(distances), device=distances.device)[nonzero_mask]
    if len(nonzero_idx) != len(distances):
        cell_offsets = cell_offsets[nonzero_idx]
        kept_prefix = torch.cat(
            (
                neighbors.new_zeros(1),
                nonzero_mask.to(neighbors.device).cumsum(0, dtype=neighbors.dtype),
            )
        )
        kept_per_system = kept_prefix[neighbors.cumsum(0)]
        neighbors = torch.diff(kept_per_system, prepend=neighbors.new_zeros(1))
        # Selecting an already evaluated zero norm leaves its undefined second
        # derivative in the graph (0 * NaN). Recompute only surviving norms.
        distances = distance_vectors[nonzero_idx].norm(dim=-1)
    else:
        distances = distances[nonzero_idx]
    edge_index = edge_index[:, nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
        "cell_offsets": cell_offsets,
        "neighbors": neighbors,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


# TODO: compiling internal graph gen is not supported right now
@torch.compiler.disable()
def generate_graph(
    data: dict,  # this is still a torch geometric batch object currently, turn this into a dict
    cutoff: float,
    max_neighbors: int,
    enforce_max_neighbors_strictly: bool,
    radius_pbc_version: int,
    pbc: torch.Tensor,
    node_partition: torch.Tensor | None = None,
    neighbor_cache: VerletNeighborCache | None = None,
    graph_search_method: str = "cell",
) -> dict:
    """Generate a graph representation from atomic structure data.

    Args:
        data (dict): A dictionary containing a batch of molecular structures.
            It should have the following keys:
                - 'pos' (torch.Tensor): Positions of the atoms.
                - 'cell' (torch.Tensor): Cell vectors of the molecular structures.
                - 'natoms' (torch.Tensor): Number of atoms in each molecular structure.
        cutoff (float): The maximum distance between atoms to consider them as neighbors.
        max_neighbors (int): The maximum number of neighbors to consider for each atom.
        enforce_max_neighbors_strictly (bool): Whether to strictly enforce the maximum number of neighbors.
        radius_pbc_version: the version of radius_pbc impl (1, 2, or 3 for NVIDIA)
        pbc (list[bool]): The periodic boundary conditions in 3 dimensions, defaults to [True,True,True] for 3D pbc
        node_partition (torch.Tensor | None): The partitioning of the nodes (atoms) for distributed inference. If provided, returned graph will be filtered to keep only edges where the target atom (edge_index[1,:]) belongs to the current rank's partition.

    Returns:
        dict: A dictionary containing the generated graph with the following keys:
            - 'edge_index' (torch.Tensor): Indices of the edges in the graph.
            - 'edge_distance' (torch.Tensor): Distances between the atoms connected by the edges.
            - 'edge_distance_vec' (torch.Tensor): Vectors representing the distances between the atoms connected by the edges.
            - 'cell_offsets' (torch.Tensor): Offsets of the cell vectors for each edge.
            - 'offset_distances' (torch.Tensor): Distances between the atoms connected by the edges, including the cell offsets.
            - 'neighbors' (torch.Tensor): Number of neighbors for each atom.
    """
    cutoff, max_neighbors, enforce_max_neighbors_strictly = validate_graph_parameters(
        cutoff, max_neighbors, enforce_max_neighbors_strictly, graph_search_method
    )
    if radius_pbc_version == 1:
        radius_graph_pbc_fn = radius_graph_pbc
    elif radius_pbc_version == 2:
        radius_graph_pbc_fn = radius_graph_pbc_v2
        if node_partition is not None or hasattr(data, "node_partition"):
            # A per-call partition must never become persistent metadata on a
            # shared AtomicData (or a PyG-style attribute/store wrapper).
            data = SimpleNamespace(
                **{
                    name: getattr(data, name)
                    for name in (
                        "pos",
                        "cell",
                        "natoms",
                        "batch",
                        "pbc",
                        "atomic_numbers",
                    )
                    if hasattr(data, name)
                }
            )
            if node_partition is not None:
                data.node_partition = node_partition
    elif radius_pbc_version == 3:
        if data.pos.device.type not in {"cpu", "cuda"}:
            raise ValueError(
                "NVIDIA graph generation (version 3) supports CPU/CUDA only. "
                "Use radius_pbc_version=2 for other devices, including XPU."
            )
        from fairchem.core.graph.radius_graph_pbc_nvidia import radius_graph_pbc_nvidia

        radius_graph_pbc_fn = radius_graph_pbc_nvidia
    else:
        raise ValueError(f"Invalid radius_pbc version {radius_pbc_version}")

    if neighbor_cache is not None:
        if radius_pbc_version != 2:
            raise ValueError("Verlet caching requires graph version 2")
        edge_index, cell_offsets, neighbors = neighbor_cache.get_graph(
            data,
            cutoff,
            max_neighbors,
            enforce_max_neighbors_strictly,
            pbc=pbc,
            method=graph_search_method,
        )
    elif radius_pbc_version == 2:
        edge_index, cell_offsets, neighbors = radius_graph_pbc_fn(
            data,
            cutoff,
            max_neighbors,
            enforce_max_neighbors_strictly,
            pbc=pbc,
            method=graph_search_method,
        )
    else:
        if graph_search_method != "cell":
            raise ValueError("Graph search method selection requires graph version 2")
        edge_index, cell_offsets, neighbors = radius_graph_pbc_fn(
            data,
            cutoff,
            max_neighbors,
            enforce_max_neighbors_strictly,
            pbc=pbc,
        )

    # for v2 it is still faster right now to not do this post filtering, need to investigate further
    if node_partition is not None and radius_pbc_version != 2:
        edge_index, cell_offsets, neighbors = filter_edges_by_node_partition(
            node_partition,
            edge_index,
            cell_offsets,
            neighbors,
            num_atoms=data.pos.shape[0],
        )

    out = get_pbc_distances(
        data.pos,
        edge_index,
        data.cell,
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )

    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offset_distances = out["offsets"]
    distance_vec = out["distance_vec"]

    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": distance_vec,
        "cell_offsets": out["cell_offsets"],
        "offset_distances": cell_offset_distances,
        "neighbors": out["neighbors"],
    }

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, NamedTuple

import torch

from fairchem.core.graph.geometry import (
    WIDE_TRANSLATION_SCALE,
    image_precision_limits,
    needs_wide_image_math,
    wide_image_device,
)
from fairchem.core.graph.portable_search import morton_keys
from fairchem.core.graph.radius_graph_pbc import (
    periodic_search_frame,
    public_image_offsets,
    radius_graph_pbc,
    radius_graph_pbc_v2,
    system_pbc_flags,
    validate_graph_geometry,
    validate_graph_parameters,
)

if TYPE_CHECKING:
    from fairchem.core.graph.neighbor_cache import VerletNeighborCache


def edge_cell_offsets(cell, cell_offsets, neighbors, *, segmented=None):
    """
    Apply each system's cell without materializing one 3x3 matrix per edge.

    The operations remain differentiable with respect to the cell, including
    double backward. The single-system path uses one ordinary matrix product.
    """
    coefficients = cell_offsets.to(dtype=cell.dtype, device=cell.device)
    if len(cell) == 1:
        return coefficients @ cell[0]
    if segmented is None:
        # Host count access is cheap on CPU. A large contiguous run per
        # system amortizes Python dispatch. Compiled calls retain tensor ops.
        segmented = (
            cell.device.type == "cpu"
            and neighbors.device.type == "cpu"
            and not torch.compiler.is_compiling()
            and len(cell_offsets) >= 128 * len(cell)
        )
    if segmented and len(cell):
        # One matmul per contiguous system. Callers can trade device launches
        # and a count transfer for avoiding all edge-sized cell-row gathers.
        return torch.cat(
            [
                tile @ system_cell
                for tile, system_cell in zip(
                    coefficients.split(neighbors.tolist()), cell, strict=True
                )
            ]
        )
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
        morton = morton_keys(coordinates)
        # Morton codes use only 30 bits. Packing target and source gives
        # the same stable lexicographic order in a signed int64.
        pair_key = (morton[edges[1]] << 30) | morton[edges[0]]
        order = torch.argsort(pair_key, stable=True)
        if len(data.natoms) > 1:
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


def canonicalize_external_edges(batch, edge_index, cell_offsets, nedges):
    """Validate edge ownership and group the topology by system, stably."""
    if (
        edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or cell_offsets.shape != (edge_index.shape[1], 3)
        or nedges.ndim != 1
        or nedges.dtype != torch.long
        or any(
            value.device != batch.device for value in (edge_index, cell_offsets, nedges)
        )
    ):
        raise ValueError(
            "Invalid external edge indices, cell offsets, counts or device"
        )
    if torch.compiler.is_compiling():
        # Keep external full-graph compilation free of data-dependent Python branches.
        torch._assert_async(
            ((edge_index >= 0) & (edge_index < len(batch))).all(), "Invalid edge index"
        )
        owners, targets = batch[edge_index]
        torch._assert_async((owners == targets).all(), "Cross-system external edge")
        actual = torch.zeros_like(nedges).scatter_add_(
            0, targets, torch.ones_like(targets)
        )
        torch._assert_async((actual == nedges).all(), "Incorrect external edge counts")
        order = targets.argsort(stable=True)
        return edge_index[:, order], cell_offsets[order]
    if edge_index.shape[1] == 0:
        if bool(nedges.any()):
            raise ValueError("nedges must match actual edge counts per system")
        return edge_index, cell_offsets
    if bool(((edge_index < 0) | (edge_index >= len(batch))).any()):
        raise ValueError("External edge index outside the atom range")
    if len(nedges) == 1:
        if int(nedges[0]) != edge_index.shape[1]:
            raise ValueError("nedges must match actual edge counts per system")
        return edge_index, cell_offsets
    source, target = batch[edge_index]
    if not torch.equal(source, target):
        raise ValueError("External edges must connect atoms in the same system")
    counts = torch.zeros_like(nedges)
    counts.scatter_add_(0, target, torch.ones_like(target))
    if not torch.equal(counts, nedges):
        raise ValueError("nedges must match actual edge counts per system")
    if bool((target[1:] < target[:-1]).any()):
        order = torch.argsort(target, stable=True)
        return edge_index[:, order], cell_offsets[order]
    return edge_index, cell_offsets


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

    # Filter edges
    edge_index = edge_index[:, local_edge_mask]
    cell_offsets = cell_offsets[local_edge_mask]
    if neighbors.shape[0] == 1:
        # If there's only one system, we can skip the scatter_add step and just return the count of remaining edges
        new_neighbors = local_edge_mask.sum(dtype=neighbors.dtype).unsqueeze(0)
        return edge_index, cell_offsets, new_neighbors

    # Create system index for each edge to track which system each edge belongs to
    num_systems = neighbors.shape[0]
    edge_system_idx = torch.repeat_interleave(
        torch.arange(num_systems, device=neighbors.device),
        neighbors,
        output_size=local_edge_mask.numel(),
    )

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


class EdgeVirial(NamedTuple):
    """Real edges before padding or reordering, for the zero-strain derivative."""

    vectors: torch.Tensor
    systems: torch.Tensor | None
    risk: torch.Tensor | None  # None means every system uses its local edges.


def edge_strain_for_stress(
    graph,
    pos,
    cell,
    atom_batch,
    *,
    cell_offsets=None,
    stress_risk=None,
    selective=False,
):
    """Prepare the symmetric strain derivative for potentially cancelling virials.

    The selective single-device path retains the real Cartesian edge vectors so
    the head can contract their adjoints only for affected structures. The
    explicit-strain path also exposes a differentiable strain tensor to callers.
    """
    if not len(cell) or stress_risk is False:
        return None
    if stress_risk is None:
        if selective:
            _, stress_risk = stress_geometry_risks(
                pos, cell, atom_batch, graph["edge_index"], cell_offsets
            )
            if stress_risk is False:
                return None
        else:
            # All GP ranks must see the full inputs, even ranks without edges.
            with torch.no_grad():
                large = (pos.abs() >= WIDE_TRANSLATION_SCALE).any() | (
                    cell.abs() >= WIDE_TRANSLATION_SCALE
                ).any()
                if cell_offsets is not None and cell_offsets.numel():
                    large = large | needs_wide_image_math(cell_offsets, cell)
                if not bool(large):
                    return None
    if selective:
        vectors = graph["edge_distance_vec"]
        if not vectors.requires_grad:
            vectors = vectors.detach().requires_grad_()
            graph["edge_distance_vec"] = vectors
        # Both radial and angular derivatives must pass through these exact
        # vectors, including for filtered, cached and externally supplied edges.
        graph["edge_distance"] = vectors.norm(dim=-1)
        systems = atom_batch[graph["edge_index"][1]].long() if len(cell) > 1 else None
        return EdgeVirial(
            vectors,
            systems,
            None if stress_risk is True or len(cell) == 1 else stress_risk,
        )

    strain = torch.zeros_like(cell, dtype=torch.float64, requires_grad=True)
    vectors = graph["edge_distance_vec"]
    wide = vectors.double()
    if len(cell) == 1 and cell.device.type in {"cpu", "cuda"}:
        delta = wide @ strain[0]
    else:
        # Keep the FP64 products and gradient sums without requiring an XPU
        # FP64 GEMM or materializing one 3x3 tensor per edge in a batch.
        systems = atom_batch[graph["edge_index"][1]].long() if len(cell) > 1 else None
        delta = 0
        for axis in range(3):
            row = strain[:, axis]
            if systems is not None:
                row = row.index_select(0, systems)
            delta = delta + wide[:, axis : axis + 1] * row
    deformed = vectors + delta.to(vectors)
    graph["edge_distance_vec"] = deformed
    graph["edge_distance"] = deformed.norm(dim=-1)
    return strain


def _external_image_risks(cell_abs, atom_batch, edge_index, offsets):
    limits = image_precision_limits(cell_abs.amax(-1))
    wide = None
    if offsets.device.type == "cpu" and not torch.compiler.is_compiling():
        # A signed range per axis avoids an [edges, 3] temporary and handles
        # the most negative signed image ID. Ordinary CPU batches stop here.
        low, high = torch.aminmax(offsets, dim=0)
        bound = limits.amin(0)
        wide = ((low <= -bound) | (high >= bound)).any()
        if len(limits) == 1 or not bool(wide):
            return wide, wide

    # Group by the edge's actual owner; an empty system keeps zero peaks.
    # Integer IDs near the maximum threshold of 16384 are exact in FP32.
    images = offsets if offsets.is_floating_point() else offsets.float()
    images = images.abs()
    if len(limits) == 1:
        wide = (images.amax(0) >= limits[0]).any()
        return wide, wide
    owner = atom_batch[edge_index[1]].long()
    peaks = images.new_zeros((len(limits), 3))
    peaks.scatter_reduce_(0, owner[:, None].expand_as(images), images, "amax")
    risk = (peaks >= limits).any(1)
    if wide is None:
        # Preserve the existing batch-wide precision gate for edge geometry.
        wide = (peaks.amax(0) >= limits.amin(0)).any()
    return wide, risk


@torch.no_grad()
def stress_geometry_risks(pos, cell, atom_batch, edge_index=None, cell_offsets=None):
    """Classify image arithmetic and the stress route for this forward pass.

    Image translation uses a batch-wide bound. Each stress virial uses the
    actual lattice of the edge's system. A bool denotes a homogeneous batch.
    """
    large_coords = pos.detach().abs() >= WIDE_TRANSLATION_SCALE
    cell_abs = cell.detach().abs()
    large_cell = cell_abs >= WIDE_TRANSLATION_SCALE
    any_stress = large_cell.any() | large_coords.any()
    wide_images = None
    image_risk = None
    if cell_offsets is not None and cell_offsets.numel():
        wide_images, image_risk = _external_image_risks(
            cell_abs, atom_batch, edge_index, cell_offsets.detach()
        )
        any_stress = any_stress | image_risk.any()

    if cell.device.type == "cpu" or torch.compiler.is_compiling():
        if not bool(any_stress):
            return wide_images, False
        if len(cell) == 1:
            return wide_images, True

    if len(cell) == 1:
        cell_risk = any_stress.reshape(1)
    else:
        cell_risk = large_cell.any(dim=(1, 2))
        if image_risk is not None:
            cell_risk = cell_risk | image_risk
        cell_risk.scatter_reduce_(0, atom_batch.long(), large_coords.any(-1), "amax")
    if cell.device.type != "cpu" and not torch.compiler.is_compiling():
        # One transfer selects native/wide edge geometry and ordinary/local
        # virials for the batch, including the all-local fast path.
        count = cell_risk.sum()
        if wide_images is None:
            num_local = int(count)
        else:
            flag, num_local = torch.stack((wide_images.to(count.dtype), count)).tolist()
            wide_images = bool(flag)
        if num_local == 0 or num_local == len(cell):
            return wide_images, num_local != 0
    elif bool(cell_risk.all()):
        return wide_images, True
    return wide_images, cell_risk


def external_edge_geometry(pos, edges, cell, cells, counts, *, _wide_images=None):
    """Cancel large image translations before rounding to the model dtype."""
    if pos.dtype != cell.dtype:
        dtype = torch.promote_types(pos.dtype, cell.dtype)
        vectors, shifts = external_edge_geometry(
            pos.to(dtype),
            edges,
            cell.to(dtype),
            cells,
            counts,
            _wide_images=_wide_images if dtype == cell.dtype else None,
        )
        return vectors, shifts.to(cell)
    if pos.dtype == torch.float32 and cells.numel():
        # Even an exactly representable floating image ID can produce a
        # rounded Cartesian shift when multiplied by a nonbinary cell vector.
        large = (
            needs_wide_image_math(cells, cell) if _wide_images is None else _wide_images
        )
        if torch.compiler.is_compiling():
            # torch.cond does not propagate higher-order derivatives on the
            # supported PyTorch versions. Choose the forward value without AD;
            # both formulas have the same linear coordinate/cell Jacobian.
            with torch.no_grad():
                selected, offsets = torch.cond(
                    large,
                    _wide_external_geometry,
                    _native_external_geometry,
                    (pos, edges, cell, cells, counts),
                )
            if torch.is_grad_enabled() and (
                pos.requires_grad or cell.requires_grad or cells.requires_grad
            ):
                native, shifts = _native_external_geometry(
                    pos, edges, cell, cells, counts
                )
                selected = selected + (native - native.detach())
                offsets = offsets + (shifts - shifts.detach())
            return selected, offsets
        if bool(large):
            return _wide_external_geometry(pos, edges, cell, cells, counts)
    return _native_external_geometry(pos, edges, cell, cells, counts)


def _wide_external_geometry(pos, edges, cell, cells, counts):
    device = wide_image_device(pos.device)
    wide_pos = pos.to(device=device, dtype=torch.float64)
    row, col = edges.to(device)
    shifts = edge_cell_offsets(
        cell.to(device=device, dtype=torch.float64), cells.to(device), counts.to(device)
    )
    return (wide_pos[row] - wide_pos[col] + shifts).to(pos), shifts.to(cell)


def _native_external_geometry(pos, edges, cell, cells, counts):
    shifts = edge_cell_offsets(cell, cells, counts)
    source, target = pos[edges[0]], pos[edges[1]]
    # Keep the original operation order except when the endpoint difference
    # and image shift cancel to a small residual.
    direct = source - target + shifts
    translated = torch.where(
        source.abs() > target.abs(),
        (source + shifts) - target,
        source - (target - shifts),
    )
    return torch.where(direct.abs() < shifts.abs() * 0.01, translated, direct), shifts


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
    *,
    _owned_topology: bool = False,
    _frame=None,
):
    row, col = edge_index

    if _frame is None:
        distance_vectors, offsets = external_edge_geometry(
            pos, edge_index, cell, cell_offsets, neighbors
        )
    else:
        local_pos, images, _ = _frame
        local_cells = cell_offsets.long() + (images[row] - images[col])
        local_offsets = edge_cell_offsets(cell, local_cells, neighbors)
        distance_vectors = (local_pos[row] + local_offsets) - local_pos[col]
        offsets = (
            edge_cell_offsets(cell, cell_offsets, neighbors) if return_offsets else None
        )
        if not _owned_topology:
            cell_offsets = public_image_offsets(cell_offsets, cell.dtype)

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # Graph-search and geometry arithmetic associate the periodic additions
    # differently. A nonzero candidate can therefore round to zero here. Keep
    # topology metadata aligned when removing it, including in mixed batches.
    nonzero_mask = distances != 0
    if bool(nonzero_mask.all()):
        # Normal MD graphs contain no zero edges. Avoid creating a dynamic-size
        # index list and gathering every already-owned geometry tensor. On an
        # accelerator the previous boolean indexing also synchronized to learn
        # the selected count; this replaces that with one scalar reduction.
        # Topology may be caller-owned, so preserve its independent output copy.
        if not _owned_topology:
            edge_index = edge_index.clone()
    else:
        nonzero_idx = torch.nonzero(nonzero_mask).flatten()
        edge_index = edge_index[:, nonzero_idx]
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
        distance_vectors = distance_vectors[nonzero_idx]
        distances = distance_vectors.norm(dim=-1)
        if return_offsets:
            offsets = offsets[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
        "cell_offsets": cell_offsets,
        "neighbors": neighbors,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

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
    *,
    geometry_only: bool = False,
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

    frame = None
    if radius_pbc_version in (1, 2):
        flags = system_pbc_flags(data, pbc)
        if bool(flags.any()):
            validate_graph_geometry(data)
            frame = periodic_search_frame(data, flags)

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
            _frame=frame,
        )
    elif radius_pbc_version == 2:
        edge_index, cell_offsets, neighbors = radius_graph_pbc_fn(
            data,
            cutoff,
            max_neighbors,
            enforce_max_neighbors_strictly,
            pbc=pbc,
            method=graph_search_method,
            _frame=frame,
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
            **({"_frame": frame} if radius_pbc_version == 1 else {}),
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
        return_offsets=not geometry_only,
        return_distance_vec=True,
        _owned_topology=geometry_only,
        _frame=frame,
    )

    graph = {
        "edge_index": out["edge_index"],
        "edge_distance": out["distances"],
        "edge_distance_vec": out["distance_vec"],
    }
    if not geometry_only:
        graph.update(
            cell_offsets=out["cell_offsets"],
            offset_distances=out["offsets"],
            neighbors=out["neighbors"],
        )
    return graph

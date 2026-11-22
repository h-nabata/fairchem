"""Regression tests for external graph ownership, periodic frames, and bounded searches."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list

from fairchem.core.common import gp_utils
from fairchem.core.common.parallelism import graph_parallel_a2a
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph import (
    portable_search,
    radius_graph_pbc,
    radius_graph_pbc_nvidia,
)
from fairchem.core.graph.compute import (
    edge_strain_for_stress,
    external_edge_geometry,
    generate_graph,
)
from fairchem.core.graph.neighbor_cache import VerletNeighborCache
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.outputs import compute_forces_and_stress
from tests.core.graph.test_radius_graph_memory import _LargestTensor


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_large_unwrapped_image_preserves_distance_and_derivatives(cached, dtype):
    cell = (
        torch.diag(torch.tensor([3.0, 6.0, 6.0], dtype=dtype))
        .unsqueeze(0)
        .requires_grad_()
    )
    # In FP32 100,000,000 is exactly representable but the integer image
    # 33,333,333 and its product by 3 cannot both be represented in FP32.
    pos = torch.tensor(
        [[0.25, 0, 0], [100000000.0, 0, 0]], dtype=dtype, requires_grad=True
    )
    data = SimpleNamespace(
        pos=pos,
        cell=cell,
        batch=torch.zeros(2, dtype=torch.long),
        natoms=torch.tensor([2]),
        pbc=torch.tensor([[True, False, False]]),
    )
    cache = VerletNeighborCache(0.3) if cached else None
    for _ in range(2):
        graph = generate_graph(data, 1.1, 8, False, 2, data.pbc, neighbor_cache=cache)
        torch.testing.assert_close(graph["edge_distance"], pos.new_full((2,), 0.75))
        assert graph["cell_offsets"][:, 0].abs().long().tolist() == [33333333] * 2
        external, _ = external_edge_geometry(
            pos, graph["edge_index"], cell, graph["cell_offsets"], graph["neighbors"]
        )
        torch.testing.assert_close(external.norm(dim=-1), graph["edge_distance"])
        energy = graph["edge_distance"].pow(3).sum()
        grad, cell_grad = torch.autograd.grad(energy, (pos, cell), create_graph=True)
        torch.testing.assert_close(grad[:, 0], pos.new_tensor([-3.375, 3.375]))
        torch.testing.assert_close(
            cell_grad[0, 0, 0], pos.new_tensor(-3.375 * 33333333)
        )
        hessian = torch.autograd.grad(grad[0, 0], pos)[0]
        torch.testing.assert_close(hessian[:, 0], pos.new_tensor([9.0, -9.0]))


@pytest.mark.parametrize("dtype, expected", [(torch.float32, 2), (torch.float64, 0)])
def test_pymatgen_cutoff_uses_model_coordinates(dtype, expected):
    atoms = Atoms("HH", positions=[[0, 0, 0], [6.0000001, 0, 0]])
    data = AtomicData.from_ase(
        atoms,
        target_dtype=dtype,
        r_edges=True,
        radius=6,
        max_neigh=3,
        enforce_max_neighbors_strictly=False,
    )
    assert data.nedges.item() == expected
    internal = generate_graph(data, 6.0, 3, False, 2, data.pbc)
    assert torch.equal(data.edge_index, internal["edge_index"])


@pytest.mark.parametrize(
    "cell,pbc",
    [
        ([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [True, False, False]),
        ([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 2.0]], [True, False, True]),
    ],
)
def test_partial_periodic_graph_uses_physical_subspace(cell, pbc):
    atoms = Atoms(
        "HH", positions=[[0.0, 0.0, 0.0], [1.8, 0.9, 0.05]], cell=cell, pbc=pbc
    )
    center, source, shift = neighbor_list("ijS", atoms, 0.4)
    expected = sorted(
        (s, t, *off) for s, t, off in zip(source, center, shift, strict=True)
    )
    data = AtomicData.from_ase(
        atoms,
        r_edges=True,
        radius=0.4,
        max_neigh=8,
        enforce_max_neighbors_strictly=False,
        target_dtype=torch.double,
    )
    torch.testing.assert_close(data.cell[0], torch.tensor(cell, dtype=torch.double))
    for version in (1, 2):
        graph = generate_graph(data, 0.4, 8, False, version, data.pbc)
        for edges, offsets in (
            (data.edge_index, data.cell_offsets),
            (graph["edge_index"], graph["cell_offsets"]),
        ):
            assert (
                sorted(
                    (*edge, *off)
                    for edge, off in zip(
                        edges.T.tolist(), offsets.tolist(), strict=True
                    )
                )
                == expected
            )


@pytest.mark.parametrize(
    "strict,expected", [(None, True), (True, True), (False, False)]
)
def test_nvidia_ase_adapter_forwards_shell_policy(monkeypatch, strict, expected):
    module = radius_graph_pbc_nvidia

    def record(positions, *, enforce_max_neighbors_strictly, **kwargs):
        assert enforce_max_neighbors_strictly is expected
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty.reshape(0, 3), positions.new_empty(0)

    monkeypatch.setattr(module, "get_neighbors_nvidia", record)
    monkeypatch.setattr(
        module,
        "get_neighbors_nvidia_atoms",
        inspect.unwrap(module.get_neighbors_nvidia_atoms),
    )
    AtomicData.from_ase(
        Atoms("H"),
        r_edges=True,
        radius=1.0,
        max_neigh=1,
        enforce_max_neighbors_strictly=strict,
        external_graph_method="nvidia",
    )


@pytest.mark.parametrize(
    "mode", [gp_utils.GPMode.ALLGATHER, gp_utils.GPMode.ALL_TO_ALL]
)
def test_external_graph_partition_matches_internal_and_rejects_empty_ranks(
    monkeypatch, mode
):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    monkeypatch.setattr(gp_utils, "initialized", lambda: True)
    monkeypatch.setattr(
        gp_utils,
        "get_gp_config",
        lambda: gp_utils.GraphParallelConfig(group_size=2, mode=mode),
    )
    monkeypatch.setattr(gp_utils, "get_gp_world_size", lambda: 2)
    params = SimpleNamespace(
        always_use_pbc=False,
        cutoff=1.4,
        max_neighbors=8,
        radius_pbc_version=2,
        enforce_max_neighbors_strictly=False,
    )
    for rank in range(2):
        monkeypatch.setattr(gp_utils, "get_gp_rank", lambda rank=rank: rank)
        monkeypatch.setattr(
            graph_parallel_a2a,
            "_sparse_index_exchange",
            lambda recv_counts, rank=rank, **kwargs: (
                recv_counts.clone(),
                torch.tensor([rank]),
            ),
        )
        for external in (True, False):
            params.otf_graph = not external
            data = AtomicData.from_ase(atoms, r_edges=external, radius=1.4, max_neigh=8)
            data["batch_full"] = data.batch
            data["atomic_numbers_full"] = data.atomic_numbers
            data.pos.requires_grad_()
            graph = eSCNMDBackbone._generate_graph(params, data)
            assert graph["edge_index"].tolist() == [[1 - rank], [rank]]
            assert data["scatter_target"].tolist() == [0]
            torch.testing.assert_close(
                torch.autograd.grad(graph["edge_distance"].square().sum(), data.pos)[0][
                    :, 0
                ],
                torch.tensor([-2.0, 2.0]),
            )
            data["atomic_numbers_full"] = data.atomic_numbers[:1]
            with pytest.raises(ValueError, match="at least one atom per rank"):
                eSCNMDBackbone._generate_graph(params, data)


def test_external_edge_ownership_and_aligned_offsets():
    atoms = AtomicData.from_ase(
        Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]]),
        r_edges=True,
        radius=1.4,
        max_neigh=2,
    )
    data = atomicdata_list_to_batch([atoms, atoms])
    edges = data.edge_index.clone()
    offsets = torch.arange(12).to(data.pos).view(4, 3)
    order = torch.tensor([0, 2, 1, 3])
    data.edge_index, data.cell_offsets = edges[:, order], offsets[order]
    data.validate()
    torch.testing.assert_close(data.edge_index, edges)
    torch.testing.assert_close(data.cell_offsets, offsets)
    data.nedges = torch.tensor([1, 3])
    with pytest.raises(ValueError, match="counts per system"):
        data.validate()
    data.nedges = torch.tensor([2, 2])
    data.edge_index[0, 0] = 2
    with pytest.raises(ValueError, match="same system"):
        data.validate()


@pytest.mark.parametrize("method", ["cell", "bvh"])
def test_sparse_outlier_search_bounds_scratch_and_preserves_pairs(method, monkeypatch):
    size = 257
    y = torch.arange(size, dtype=torch.float64).mul(2)
    y[1:-1:2] = y[1:-1:2].flip((0,))
    pos = torch.zeros((size, 3), dtype=torch.float64)
    pos[:, 1] = y
    pos[-1, 0] = 1e12
    data = SimpleNamespace(
        pos=pos,
        natoms=torch.tensor([size]),
        batch=torch.zeros(size, dtype=torch.long),
        cell=torch.eye(3, dtype=pos.dtype)[None],
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    monkeypatch.setattr(radius_graph_pbc, "_STENCIL_TILE", 1000)
    monkeypatch.setattr(radius_graph_pbc, "_CELL_PAIR_TILE", 128)
    monkeypatch.setattr(portable_search, "_PAIR_TILE", 4096)
    with _LargestTensor() as scratch:
        edges, _, _ = radius_graph_pbc.radius_graph_pbc_v2(data, 1.1, 0, method=method)
    assert edges.shape[1] == 0
    assert scratch.largest_numel <= 3 * 4096


def test_exact_external_images_batch_with_ordinary_offsets():
    atoms = Atoms(
        "H2",
        positions=[[0.25, 0, 0], [1.0, 0, 0]],
        cell=[3, 6, 6],
        pbc=[True, False, False],
    )
    a = AtomicData.from_ase(atoms, r_edges=True, radius=1.1, max_neigh=8)
    b = a.clone()
    b.pos[1, 0] = 100000000
    graph = generate_graph(b, 1.1, 8, False, 2, b.pbc)
    b.update_batch_edges(graph["edge_index"], graph["cell_offsets"], graph["neighbors"])
    mixed = atomicdata_list_to_batch([a, b])
    assert a.cell_offsets.dtype == torch.float32
    assert mixed.cell_offsets.dtype == torch.long
    mixed.validate()
    vectors, _ = external_edge_geometry(
        mixed.pos, mixed.edge_index, mixed.cell, mixed.cell_offsets, mixed.nedges
    )
    torch.testing.assert_close(vectors.norm(dim=-1), torch.full((4,), 0.75))


@pytest.mark.parametrize("offset_dtype", [torch.float32, torch.long])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("period,image", [(3.0, -33333334), (33333334.0, -3)])
def test_external_wide_shift_preserves_local_energy_derivatives(
    offset_dtype, compiled, period, image
):
    # The image ID itself fits in FP32, but its product by 3 does not.
    pos = torch.tensor([[1e8, 0, 0], [0.25, 0, 0]], requires_grad=True)
    cell = torch.diag(torch.tensor([period, 6, 6])).unsqueeze(0).requires_grad_()
    edges = torch.tensor([[0], [1]])
    cells = torch.tensor([[image, 0, 0]], dtype=offset_dtype)
    evaluate = external_edge_geometry
    if compiled:
        evaluate = torch.compile(evaluate, backend="eager", fullgraph=True)
    vec, _ = evaluate(pos, edges, cell, cells, torch.tensor([1]))
    torch.testing.assert_close(vec, pos.new_tensor([[-2.25, 0, 0]]))
    forces, stress = torch.autograd.grad(
        vec.square().sum(), (pos, cell), create_graph=True
    )
    torch.testing.assert_close(forces[:, 0], pos.new_tensor([-4.5, 4.5]))
    torch.testing.assert_close(stress[0, 0, 0], pos.new_tensor(-4.5 * image))
    torch.testing.assert_close(
        torch.autograd.grad(forces[0, 0], pos)[0][:, 0], pos.new_tensor([2, -2])
    )


@pytest.mark.parametrize("pbc", [[True, False, False], [True, False, True]])
def test_skew_vacuum_does_not_inflate_search_or_change_edges(pbc):
    flags = torch.tensor([pbc])
    pos = torch.tensor([[0.2, 0.4, 0.1], [3.6, 0.4, 0.1]])
    cell = torch.diag(torch.tensor([4.0, 4.0, 6.0])).unsqueeze(0)
    data = SimpleNamespace(
        pos=pos,
        batch=torch.zeros(2, dtype=torch.long),
        natoms=torch.tensor([2]),
        cell=cell,
        pbc=flags,
    )
    reference = generate_graph(data, 1.1, 8, False, 2, flags)
    data.cell = cell.clone()
    data.cell[0, 1] = pos.new_tensor([4, 0.001, 0])
    _, _, reciprocal = radius_graph_pbc.periodic_search_frame(data, flags)
    torch.testing.assert_close(
        reciprocal[0, flags[0]].norm(dim=-1),
        cell[0, flags[0]].norm(dim=-1).reciprocal(),
    )
    for version, cache in ((1, None), (2, None), (2, VerletNeighborCache(0.3))):
        for _ in range(2):
            graph = generate_graph(
                data, 1.1, 8, False, version, flags, neighbor_cache=cache
            )
            for field in (
                "edge_index",
                "cell_offsets",
                "edge_distance",
                "edge_distance_vec",
            ):
                torch.testing.assert_close(graph[field], reference[field])


@pytest.mark.parametrize(
    "period,first,second,cutoff,dist",
    [
        (1e7, 0.0, -0.25, 0.5, 0.25),
        (33333334.0, 0.25, 1e8, 2.2, None),
        (33333334.0, 0.25, 1e8, 2.3, 2.25),
    ],
)
@pytest.mark.parametrize("version", [1, 2, 3])
def test_centered_local_frame_preserves_both_cutoff_directions(
    period, first, second, cutoff, dist, version
):
    if version == 3 and not radius_graph_pbc_nvidia.nvalchemiops_installed():
        pytest.skip("nvalchemiops is not installed")
    pos = torch.tensor([[first, 0.0, 0.0], [second, 0.0, 0.0]])
    data = SimpleNamespace(
        pos=pos,
        cell=torch.diag(pos.new_tensor([period, 6.0, 6.0])).unsqueeze(0),
        batch=torch.zeros(2, dtype=torch.long),
        natoms=torch.tensor([2]),
        pbc=torch.tensor([[True, False, False]]),
    )
    graph = generate_graph(data, cutoff, 8, False, version, data.pbc)
    expected = [] if dist is None else [dist, dist]
    torch.testing.assert_close(
        graph["edge_distance"], pos.new_tensor(expected), atol=1e-12, rtol=1e-6
    )


def test_compact_and_long_axes_choose_independent_periodic_origins():
    # Ordinary compact-cell coordinates retain their historical frame and edge
    # traversal for FP32 model reductions. Only the long (y) axis needs centering.
    pos = torch.tensor([[17.0, -0.25, 30.0], [12.1, 0.25, -1e-8]])
    cell = torch.diag(pos.new_tensor([24.0, 1e7, 24.0])).unsqueeze(0)
    data = SimpleNamespace(pos=pos, cell=cell, natoms=torch.tensor([2]))
    local, images, _ = radius_graph_pbc.periodic_search_frame(
        data, torch.ones((1, 3), dtype=torch.bool)
    )
    torch.testing.assert_close(images, torch.tensor([[0, 0, 1], [0, 0, 0]]))
    torch.testing.assert_close(
        local, pos.new_tensor([[17.0, -0.25, 6.0], [12.1, 0.25, -1e-8]]), atol=0, rtol=0
    )


@pytest.mark.parametrize("mode", ["otf", "cached", "external"])
def test_local_edge_strain_preserves_virial_for_mixed_translated_systems(mode):
    # Identical wires at different image origins, plus a molecule translated
    # as a whole. The correct xx strain derivative is 2 * separation**2.
    pos = torch.tensor(
        [
            [0.25, 0.0, 0.0],
            [1e8, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [32768.0, 0.0, 0.0],
            [32769.0, 0.0, 0.0],
        ],
        requires_grad=True,
    )
    cell = (
        torch.diag(pos.new_tensor([3.0, 6.0, 6.0]))
        .expand(3, -1, -1)
        .clone()
        .requires_grad_()
    )
    counts = torch.tensor([2, 2, 2])
    data = SimpleNamespace(
        pos=pos,
        cell=cell,
        batch=torch.tensor([0, 0, 1, 1, 2, 2]),
        natoms=counts,
        pbc=torch.tensor(
            [[True, False, False], [True, False, False], [False, False, False]]
        ),
    )
    raw_offsets = None
    cache = VerletNeighborCache(0.3) if mode == "cached" else None
    for _ in range(2 if cache is not None else 1):
        if mode == "external":
            edges = torch.tensor([[1, 0, 3, 2, 5, 4], [0, 1, 2, 3, 4, 5]])
            raw_offsets = torch.zeros((6, 3), dtype=torch.long)
            raw_offsets[:2, 0] = torch.tensor([-33333333, 33333333])
            vec, _ = external_edge_geometry(pos, edges, cell, raw_offsets, counts)
            graph = {
                "edge_index": edges,
                "edge_distance_vec": vec,
                "edge_distance": vec.norm(dim=-1),
            }
        else:
            graph = generate_graph(
                data, 1.1, 8, False, 2, data.pbc, neighbor_cache=cache
            )
        strain = edge_strain_for_stress(
            graph, pos, cell, data.batch, cell_offsets=raw_offsets
        )
        energy = graph["edge_distance"].square().sum() / 2
        gpos, gcell, virial = torch.autograd.grad(
            energy, (pos, cell, strain), create_graph=True
        )
        expected = strain.new_zeros(strain.shape)
        expected[:, 0, 0] = strain.new_tensor([1.125, 1.125, 2.0])
        torch.testing.assert_close(virial, expected, atol=0, rtol=0)
        # Model energy still differentiates against the caller's original
        # Cartesian position and cell arguments, including the large image.
        torch.testing.assert_close(
            gpos[:, 0], pos.new_tensor([-1.5, 1.5, -1.5, 1.5, -2.0, 2.0])
        )
        torch.testing.assert_close(gcell[0, 0, 0], pos.new_tensor(-1.5 * 33333333))
        (hrow,) = torch.autograd.grad(gpos[0, 0], pos)
        torch.testing.assert_close(
            hrow[:, 0], pos.new_tensor([2.0, -2.0, 0.0, 0.0, 0.0, 0.0])
        )
    if cache is not None:
        assert cache.hits == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
@pytest.mark.parametrize("device", ["cpu", "xpu"])
def test_external_stress_batch_matches_independent_virials_and_training_gradients(
    dtype, device
):
    if device == "xpu" and not torch.xpu.is_available():
        pytest.skip("Requires a real Intel XPU")
    # The two dimers have identical local vectors, including shear components.
    # The first crosses a normal cell boundary. The second spans 33,333,333
    # cells. The isolated atom has no real edges and also triggers the guard.
    cases = ((2.75, 0.5, 1), (32768.0, None, 0), (0.25, 1e8, -33333333))

    def evaluate(systems):
        coordinates = []
        forward, offsets = [], []
        counts = []
        for left, right, image in systems:
            first = len(coordinates)
            coordinates.append([left, 0.125, 0.25])
            counts.append(1 if right is None else 2)
            if right is not None:
                coordinates.append([right, 0.5, 0.75])
                forward.append([first + 1, first])
                offsets.append([image, 0, 0])
        pos = torch.tensor(coordinates, requires_grad=True, dtype=dtype, device=device)
        cell = (
            torch.diag(pos.new_tensor([3, 4, 6]))
            .repeat(len(systems), 1, 1)
            .requires_grad_()
        )
        counts = torch.tensor(counts, device=device)
        batch = torch.arange(len(systems), device=device).repeat_interleave(counts)
        # Interleave edges belonging to different systems at the external API.
        edges = torch.tensor(forward, dtype=torch.long, device=device).reshape(-1, 2).T
        images = torch.tensor(offsets, dtype=torch.long, device=device).reshape(-1, 3)
        graph = eSCNMDBackbone._generate_graph(
            SimpleNamespace(
                otf_graph=False,
                cutoff=2,
                regress_config=SimpleNamespace(direct_stress=False),
            ),
            {
                "pos": pos,
                "cell": cell,
                "batch": batch,
                "edge_index": torch.cat([edges, edges.flip(0)], dim=1),
                "cell_offsets": torch.cat([images, -images]),
                "nedges": (counts - 1) * 2,
                "_resolved_requested_properties": {"stress"},
            },
        )
        if len(systems) > 1:
            # The compact dimer must retain the ordinary position/cell route.
            assert graph["edge_strain"].risk.tolist() == [False, True, True]
        vectors = graph["edge_distance_vec"]
        energy = graph["edge_distance"].square() * vectors.square().sum(dim=1) / 2
        if "edge_valid" in graph:
            energy = energy[graph["edge_valid"]]
        forces, stress = compute_forces_and_stress(
            energy.sum(),
            pos,
            cell,
            batch,
            training=True,
            edge_strain=graph.get("edge_strain"),
        )
        # Reverse through the force and stress graphs independently.
        curvature = torch.autograd.grad(forces.square().sum(), pos, retain_graph=True)[
            0
        ]
        stress_gradient = torch.autograd.grad(stress.sum(), pos)[0]
        return forces, stress.reshape(-1, 3, 3), curvature, stress_gradient

    mixed = evaluate(cases)
    independent = [evaluate([case]) for case in cases]
    for actual, separate in zip(mixed, zip(*independent, strict=True), strict=True):
        torch.testing.assert_close(actual, torch.cat(separate))

    local = mixed[1].new_tensor([0.75, 0.375, 0.5])
    # E = |d|⁴ per dimer, so dE/dε = 4 |d|² (d⊗d).
    expected = 4 * local.square().sum() * torch.outer(local, local) / 72
    torch.testing.assert_close(
        mixed[1], torch.stack([expected, 0 * expected, expected])
    )


def test_edge_strain_keeps_small_graph_and_supports_empty_large_partition():
    cell = torch.eye(3).unsqueeze(0)
    pos = torch.zeros((2, 3))
    batch = torch.zeros(2, dtype=torch.long)
    vec = pos.new_empty((0, 3))
    distances = vec.new_empty(0)
    graph = {
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "edge_distance_vec": vec,
        "edge_distance": distances,
    }
    assert edge_strain_for_stress(graph, pos, cell, batch) is None
    assert graph["edge_distance"] is distances
    assert graph["edge_distance_vec"] is vec
    # The trigger is global across graph-parallel ranks, including a rank with
    # no edges, and it can originate solely in the external lattice shifts.
    strain = edge_strain_for_stress(
        graph, pos, cell, batch, cell_offsets=torch.tensor([[100000, 0, 0]])
    )
    (gradient,) = torch.autograd.grad(graph["edge_distance"].sum(), strain)
    torch.testing.assert_close(gradient, torch.zeros_like(strain))


def test_local_strain_matches_fixed_fractional_shear_and_training_derivative():
    pos = torch.tensor([[0.25, 0.0, 0.0], [1e8, 0.5, 0.25]], requires_grad=True)
    cell = torch.diag(pos.new_tensor([3.0, 6.0, 6.0])).unsqueeze(0).requires_grad_()
    data = SimpleNamespace(
        pos=pos,
        cell=cell,
        natoms=torch.tensor([2]),
        batch=torch.zeros(2, dtype=torch.long),
        pbc=torch.tensor([[True, False, False]]),
    )
    graph = generate_graph(data, 1.1, 8, False, 2, data.pbc)
    strain = edge_strain_for_stress(graph, pos, cell, data.batch)
    scale = torch.tensor(1.5, requires_grad=True)
    energy = scale * graph["edge_distance_vec"].square().sum() / 2
    (virial,) = torch.autograd.grad(energy, strain, create_graph=True)
    direction = strain.new_tensor([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    shear = (virial[0] * direction).sum()
    # Independently strain the exact, wrapped Cartesian separation. Both atoms
    # and the lattice follow the affine shear (fixed fractional coordinates).
    local = strain.new_tensor([0.75, 0.5, 0.25])
    step = 1e-4

    def strained_energy(t):
        return scale.double() * (local + t * (local @ direction)).square().sum()

    finite_difference = (strained_energy(step) - strained_energy(-step)) / (2 * step)
    torch.testing.assert_close(shear, finite_difference, rtol=1e-10, atol=1e-10)
    (training_gradient,) = torch.autograd.grad(shear, scale)
    torch.testing.assert_close(training_gradient, scale.new_tensor(0.75))


def test_external_stress_reuses_image_decision_and_updates_it_each_forward(monkeypatch):
    from fairchem.core.models.uma import escn_md

    backbone = SimpleNamespace(
        otf_graph=False,
        regress_config=SimpleNamespace(direct_stress=False),
    )
    pos = torch.tensor([[0.25, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True)
    cell = torch.diag(torch.tensor([3.0, 6.0, 6.0])).unsqueeze(0).requires_grad_()
    offsets = torch.zeros(2, 3, dtype=torch.long)
    data = {
        "pos": pos,
        "cell": cell,
        "edge_index": torch.tensor([[1, 0], [0, 1]]),
        "cell_offsets": offsets,
        "nedges": torch.tensor([2]),
        "batch": torch.zeros(2, dtype=torch.long),
        "_resolved_requested_properties": {"stress"},
    }
    geometry = escn_md.external_edge_geometry
    wide_inputs = []

    def observe(*args, **kwargs):
        wide_inputs.append(bool(kwargs["_wide_images"]))
        return geometry(*args, **kwargs)

    monkeypatch.setattr(escn_md, "external_edge_geometry", observe)
    # Reuse the same input tensors as in an MD loop. The vacuum axis can
    # require a stable virial even when no edge crosses that axis.
    for height, x, image in (
        (6, 1, 0),
        (65536, 1, 0),
        (6, 100000000, -33333333),
        (6, 1, 0),
    ):
        with torch.no_grad():
            cell[0, 2, 2] = height
            pos[1, 0] = x
            offsets[:, 0] = offsets.new_tensor([image, -image])
        graph = escn_md.eSCNMDBackbone._generate_graph(backbone, data)
        torch.testing.assert_close(
            graph["edge_distance_vec"][:, 0],
            pos.new_tensor([0.75, -0.75]),
            rtol=0,
            atol=0,
        )
    assert wide_inputs == [False, False, True, False]

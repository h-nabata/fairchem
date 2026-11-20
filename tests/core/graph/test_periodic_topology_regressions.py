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
from fairchem.core.graph.compute import external_edge_geometry, generate_graph
from fairchem.core.graph.neighbor_cache import VerletNeighborCache
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
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

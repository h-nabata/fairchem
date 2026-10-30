"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from ase.build import bulk, molecule
from ase.neighborlist import neighbor_list
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.graph.compute import generate_graph
from fairchem.core.models.uma.escn_md import MLP_EFS_Head
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend
from fairchem.core.models.uma.nn.mole import MOLEDGL, MOLEGlobals
from fairchem.core.models.uma.outputs import reduce_node_to_system
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

DEVICES = [
    "cpu",
    pytest.param(
        "xpu:0",
        marks=pytest.mark.skipif(
            not torch.xpu.is_available(), reason="Requires a real Intel XPU"
        ),
    ),
]


def _atoms(periodic):
    if periodic:
        atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
        atoms.positions[0] += [0.03, 0.02, 0.01]
    else:
        atoms = molecule("H2O")
        atoms.center(vacuum=8.0)
    return atoms


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("periodic", [False, True])
def test_xpu_graph_v2_against_ase(device, periodic):
    atoms = _atoms(periodic)
    data = AtomicData.from_ase(atoms, task_name="omat").to(device)
    cutoff = 3.0
    graph = generate_graph(data, cutoff, 100, False, 2, data.pbc)
    i, j, offsets = neighbor_list("ijS", atoms, cutoff)
    reference = {tuple(row) for row in np.column_stack((j, i, offsets))}
    actual = {
        tuple(row)
        for row in torch.cat(
            (graph["edge_index"].T, graph["cell_offsets"].long()), dim=1
        )
        .cpu()
        .tolist()
    }
    assert actual == reference
    assert graph["edge_distance"].dtype == torch.float32
    assert graph["edge_distance"].device.type == torch.device(device).type


def _small_model():
    torch.manual_seed(42)
    backbone = eSCNMDMoeBackbone(
        sphere_channels=8,
        hidden_channels=16,
        edge_channels=8,
        num_distance_basis=8,
        num_layers=1,
        num_experts=2,
        lmax=2,
        mmax=2,
        otf_graph=True,
        cutoff=3.0,
        use_dataset_embedding=False,
        use_composition_embedding=True,
        always_use_pbc=False,
        direct_forces=False,
        regress_forces=True,
        regress_stress=True,
        model_version=1.2,
        execution_mode="general",
    )
    backbone.model_id = "UMA-1.2.1"
    head = MLP_EFS_Head(backbone, wrap_property=False)
    backbone.eval().requires_grad_(False)
    head.eval().requires_grad_(False)
    return backbone, head


def _evaluate(backbone, head, atoms, device):
    data = AtomicData.from_ase(atoms, task_name="omat").to(device)
    output = head(data, backbone(data))
    return {key: output[key] for key in ("energy", "forces", "stress")}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("periodic", [False, True])
def test_xpu_synthetic_uma_merged_energy_forces_stress(device, periodic):
    """
    Exercise actual UMA operations with random weights, without gated checkpoints.
    """
    atoms = _atoms(periodic)
    backbone, head = _small_model()
    reference = _evaluate(backbone, head, atoms, "cpu")
    candidate = copy.deepcopy(backbone).to(device)
    candidate_head = copy.deepcopy(head).to(device)
    data = AtomicData.from_ase(atoms, task_name="omat").to(device)
    settings = InferenceSettings(
        merge_mole=True,
        compile=False,
        tf32=False,
        activation_checkpointing=False,
        execution_mode="general",
        internal_graph_gen_version=2,
    )
    candidate = candidate.prepare_for_inference(data, settings).to(device)
    candidate.requires_grad_(False)
    result = _evaluate(candidate, candidate_head, atoms, device)
    for key in reference:
        assert torch.isfinite(result[key]).all()
        assert result[key].device.type == torch.device(device).type
        torch.testing.assert_close(
            result[key].cpu(), reference[key], rtol=2e-4, atol=2e-5
        )
    assert result["forces"].dtype == torch.float32
    # FP64 sums are deliberate even with FP32 network parameters.
    assert result["energy"].dtype == torch.float64
    assert result["stress"].dtype == torch.float64
    torch.testing.assert_close(
        result["forces"].sum(0), torch.zeros(3, device=device), atol=2e-5, rtol=0
    )


def test_synthetic_uma_force_matches_energy_finite_difference():
    atoms = _atoms(False)
    backbone, head = _small_model()
    result = _evaluate(backbone, head, atoms, "cpu")
    delta = 1e-3
    plus, minus = atoms.copy(), atoms.copy()
    plus.positions[1, 2] += delta
    minus.positions[1, 2] -= delta
    e_plus = _evaluate(backbone, head, plus, "cpu")["energy"]
    e_minus = _evaluate(backbone, head, minus, "cpu")["energy"]
    finite_difference = -(e_plus - e_minus) / (2 * delta)
    torch.testing.assert_close(
        result["forces"][1, 2].double(),
        finite_difference.squeeze(),
        atol=3e-5,
        rtol=3e-3,
    )


def test_synthetic_uma_stress_matches_strain_finite_difference():
    atoms = _atoms(True)
    backbone, head = _small_model()
    result = _evaluate(backbone, head, atoms, "cpu")
    delta = 1e-3
    plus, minus = atoms.copy(), atoms.copy()
    strain = np.diag([delta, 0.0, 0.0])
    plus.set_cell(atoms.cell @ (np.eye(3) + strain), scale_atoms=True)
    minus.set_cell(atoms.cell @ (np.eye(3) - strain), scale_atoms=True)
    e_plus = _evaluate(backbone, head, plus, "cpu")["energy"]
    e_minus = _evaluate(backbone, head, minus, "cpu")["energy"]
    finite_difference = (e_plus - e_minus) / (2 * delta * atoms.get_volume())
    torch.testing.assert_close(
        result["stress"][0, 0], finite_difference.squeeze(), atol=3e-6, rtol=3e-3
    )


def test_xpu_fake_tensor_general_backend_metadata():
    """
    Check XPU dispatch metadata only; this does not execute GPU kernels.
    """
    with FakeTensorMode(allow_fallback_kernels=False):
        device = "xpu:0"
        messages = torch.empty((6, 9, 8), device=device)
        targets = torch.empty(6, dtype=torch.long, device=device)
        wigner = torch.empty((6, 9, 9), device=device)
        out = ExecutionBackend.permute_wigner_inv_edge_to_node(
            messages, wigner, targets, 4
        )
        assert out.device == torch.device(device)
        assert out.shape == (4, 9, 8)
        values = torch.empty(4, device=device)
        batch = torch.empty(4, dtype=torch.long, device=device)
        energy, _ = reduce_node_to_system(values, batch, 1)
        assert energy.device == torch.device(device)
        assert energy.dtype == torch.float64


def test_xpu_dgl_mole_fails_before_entering_cpp_kernels():
    globals_ = MOLEGlobals(torch.ones(1, 2), torch.tensor([4]))
    layer = MOLEDGL(2, 8, 8, globals_, bias=True)
    with FakeTensorMode(allow_fallback_kernels=False):
        x = torch.empty((4, 8), device="xpu:0")
        with pytest.raises(NotImplementedError, match="mole_layer_type='pytorch'"):
            layer(x)

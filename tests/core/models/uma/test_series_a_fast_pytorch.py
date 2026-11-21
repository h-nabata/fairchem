"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.nn.execution_backends import get_execution_backend
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


@pytest.mark.parametrize("pbc", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "xpu:0"])
def test_fast_pytorch_geometry_energy_force_stress_hvp(pbc, dtype, device):
    if device.startswith("xpu") and not torch.xpu.is_available():
        pytest.skip("Requires an Intel XPU")
    baseline = (
        eSCNMDBackbone(
            sphere_channels=4,
            hidden_channels=8,
            edge_channels=5,
            num_distance_basis=7,
            num_layers=2,
            lmax=2,
            mmax=2,
            cutoff=3.0,
            use_dataset_embedding=False,
            always_use_pbc=False,
            otf_graph=True,
            direct_forces=False,
            regress_forces=True,
            regress_stress=True,
            execution_mode="general",
        )
        .to(device=device, dtype=dtype)
        .eval()
        .requires_grad_(False)
    )
    candidate = copy.deepcopy(baseline)
    candidate.backend = get_execution_backend("umas_fast_pytorch")
    for block in candidate.blocks:
        block.edge_wise.backend = candidate.backend
    atoms = molecule("H2O")
    atoms.set_cell([3.9, 4.3, 4.7])
    atoms.pbc = pbc
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol").to(device)
    for name, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            data[name] = value.to(dtype)
    candidate.prepare_for_inference(
        data.clone(),
        InferenceSettings(
            execution_mode="umas_fast_pytorch",
            merge_mole=True,
            activation_checkpointing=False,
        ),
    )
    direction = torch.randn_like(data.pos)

    def evaluate(model):
        sample = data.clone()
        emb = model(sample)["node_embedding"]
        assert sample["scatter_target"].numel() > 0
        energy = emb[:, 0].square().sum()
        gradient, cell_gradient = torch.autograd.grad(
            energy,
            (sample.pos, sample.cell),
            create_graph=True,
        )
        hvp = torch.autograd.grad((gradient * direction).sum(), sample.pos)[0]
        assert gradient.abs().max() > 0
        return energy, -gradient, cell_gradient, hvp

    actual, expected = evaluate(candidate), evaluate(baseline)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(
            a,
            b,
            rtol=2e-4 if dtype == torch.float32 else 1e-9,
            atol=2e-5 if dtype == torch.float32 else 1e-10,
        )

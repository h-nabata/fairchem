"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.escn_md import GradRegressConfig, MLP_EFS_Head


@pytest.mark.parametrize("hessian_vmap", [False, True])
def test_eval_forces_only_head_retains_graph_for_hessian(hessian_vmap):
    backbone = SimpleNamespace(
        sphere_channels=1,
        hidden_channels=4,
        regress_config=GradRegressConfig(
            forces=True, stress=False, hessian=True, hessian_vmap=hessian_vmap
        ),
    )
    head = MLP_EFS_Head(backbone, wrap_property=False)
    # An exact quadratic energy provides an independent reference for both
    # first and second derivatives through the real EFS head/autograd path.
    head.energy_block = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        head.energy_block.weight.fill_(1.0)
    head.eval().requires_grad_(False)
    positions = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], requires_grad=True)
    data = {
        "pos": positions,
        "batch": torch.zeros(2, dtype=torch.long),
        "natoms": torch.tensor([2]),
    }
    embeddings = {"node_embedding": positions.square().sum(-1).view(2, 1, 1)}

    output = head(data, embeddings)

    torch.testing.assert_close(
        output["energy"], positions.square().sum().double().view(1)
    )
    torch.testing.assert_close(output["forces"], -2 * positions)
    torch.testing.assert_close(output["hessian"], 2 * torch.eye(6).unsqueeze(0))
    assert torch.isfinite(output["hessian"]).all()

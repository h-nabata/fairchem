"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma import escn_md
from fairchem.core.models.uma.escn_md import GradRegressConfig, MLP_EFS_Head


def _head(**regress_options):
    backbone = SimpleNamespace(
        sphere_channels=1,
        hidden_channels=4,
        regress_config=GradRegressConfig(**regress_options),
    )
    head = MLP_EFS_Head(backbone, wrap_property=False).double().eval()
    head.energy_block = torch.nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        head.energy_block.weight.fill_(1)
    head.requires_grad_(False)
    return head


def _data(requested, requires_grad=True):
    pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float64)
    pos.requires_grad_(requires_grad)
    return {
        "pos": pos,
        "batch": torch.zeros(2, dtype=torch.long),
        "batch_full": torch.zeros(2, dtype=torch.long),
        "natoms": torch.tensor([2]),
        "_requested_properties": requested,
    }


def _forbid(*args, **kwargs):
    raise AssertionError("Unrequested derivative computation was executed")


def test_energy_only_omits_force_stress_hessian_and_embeddings(monkeypatch):
    head = _head(forces=True, stress=True, hessian=True)
    data = _data(("energy",), requires_grad=False)
    monkeypatch.setattr(escn_md, "compute_forces", _forbid)
    monkeypatch.setattr(escn_md, "compute_forces_and_stress", _forbid)
    monkeypatch.setattr(escn_md, "compute_hessian", _forbid)
    output = head(data, {"node_embedding": data["pos"].square().sum(-1).view(2, 1, 1)})
    assert set(output) == {"energy"}
    torch.testing.assert_close(output["energy"], data["pos"].square().sum().reshape(1))


def test_forces_only_skips_stress_even_when_checkpoint_requests_it(monkeypatch):
    head = _head(forces=True, stress=True)
    data = _data(("forces",))
    monkeypatch.setattr(escn_md, "compute_forces_and_stress", _forbid)
    output = head(data, {"node_embedding": data["pos"].square().sum(-1).view(2, 1, 1)})
    assert set(output) == {"forces"}
    torch.testing.assert_close(output["forces"], -2 * data["pos"])


@pytest.mark.parametrize("property_name", ["hessian", "hessian_vector_product"])
def test_explicit_curvature_enables_grad_without_configured_forces(property_name):
    head = _head(forces=False, stress=False, hessian=False)
    data = _data((property_name,))
    data["hessian_vector"] = torch.ones_like(data["pos"])
    embeddings = {"node_embedding": data["pos"].pow(4).sum(-1).view(2, 1, 1)}
    with torch.no_grad():
        output = head(data, embeddings)
    assert set(output) == {property_name}
    diagonal = 12 * data["pos"].square()
    expected = (
        diagonal.flatten().diag().unsqueeze(0)
        if property_name == "hessian"
        else diagonal
    )
    torch.testing.assert_close(output[property_name], expected)


def test_hvp_does_not_construct_dense_hessian(monkeypatch):
    head = _head(forces=True)
    data = _data(("hessian_vector_product",))
    data["hessian_vector"] = torch.ones_like(data["pos"])
    monkeypatch.setattr(escn_md, "compute_hessian", _forbid)
    output = head(data, {"node_embedding": data["pos"].pow(4).sum(-1).view(2, 1, 1)})
    torch.testing.assert_close(
        output["hessian_vector_product"], 12 * data["pos"].square()
    )


def test_hvp_and_hessian_share_the_force_graph():
    head = _head(forces=True)
    head.regress_config.hessian_chunk_size = 2
    data = _data(("hessian", "hessian_vector_product"))
    data["hessian_vector"] = torch.full_like(data["pos"], 0.5)
    output = head(data, {"node_embedding": data["pos"].pow(4).sum(-1).view(2, 1, 1)})
    assert set(output) == {"hessian", "hessian_vector_product"}
    torch.testing.assert_close(
        output["hessian_vector_product"].flatten(),
        output["hessian"][0] @ data["hessian_vector"].flatten(),
    )


def test_constant_embeddings_give_zero_force_stress_and_hessian():
    head = _head(forces=True, stress=True)
    data = _data(("forces", "stress", "hessian", "hessian_vector_product"))
    data["cell"] = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_()
    data["hessian_vector"] = torch.ones_like(data["pos"])
    output = head(data, {"node_embedding": torch.ones(2, 1, 1, dtype=torch.float64)})
    for tensor in output.values():
        torch.testing.assert_close(tensor, torch.zeros_like(tensor))


def test_embeddings_only_skips_energy_head(monkeypatch):
    head = _head(forces=True, stress=True)
    data = _data(("embeddings",), requires_grad=False)
    embeddings = {"node_embedding": torch.ones(2, 1, 1, dtype=torch.float64)}
    monkeypatch.setattr(escn_md, "compute_energy", _forbid)
    output = head(data, embeddings)
    assert set(output) == {"embeddings"}
    torch.testing.assert_close(output["embeddings"], embeddings["node_embedding"])

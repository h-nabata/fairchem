"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fairchem.core.common import distutils
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.modules.loss import (
    DDPLoss,
    L2NormLoss,
    MAELoss,
    MSELoss,
    PerAtomMAELoss,
)


@pytest.fixture()
def energy():
    # batch size = 4
    pred = torch.rand([4, 1])
    target = torch.rand([4, 1])
    return pred, target


@pytest.fixture()
def forces():
    # batch size = 4
    # total atoms = 100
    pred = torch.rand(100, 3)
    target = torch.rand(100, 3)
    return pred, target


@pytest.fixture()
def anisotropic_stress():
    # batch size = 4
    pred = torch.rand([4, 5])
    target = torch.rand([4, 5])
    return pred, target


@pytest.fixture()
def natoms():
    # batch size = 4
    # total atoms = 100
    return torch.tensor([25, 34, 21, 20])


def test_mae(energy, forces, natoms):
    loss = MAELoss()
    ref_loss = nn.L1Loss(reduction="none")
    pred, target = energy
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))
    pred, target = forces
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))


def test_mse(energy, forces):
    loss = MSELoss()
    ref_loss = nn.MSELoss(reduction="none")
    pred, target = energy
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))
    pred, target = forces
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))


def test_per_atom_mae(energy, natoms):
    loss = PerAtomMAELoss()
    ref_loss = nn.L1Loss(reduction="none")
    pred, target = energy
    _natoms = torch.reshape(natoms, target.shape)
    assert target.shape == (target / _natoms).shape
    assert torch.allclose(
        loss(pred, target, natoms), ref_loss(pred / _natoms, target / _natoms)
    )


def test_l2norm(forces, natoms):
    loss = L2NormLoss()
    pred, target = forces
    ref_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    assert torch.allclose(loss(pred, target, natoms), ref_norm)


def test_energy_mae_reduction(energy, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="mae", reduction="mean")
    ref_loss = nn.L1Loss(reduction="mean")
    pred, target = energy
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))
    # sum reduction
    loss = DDPLoss(loss_name="mae", reduction="sum")
    ref_loss = nn.L1Loss(reduction="sum")
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))


def test_stress_mae_reduction(anisotropic_stress, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="mae", reduction="mean")
    ref_loss = nn.L1Loss(reduction="mean")
    pred, target = anisotropic_stress
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))
    # sum reduction
    loss = DDPLoss(loss_name="mae", reduction="sum")
    ref_loss = nn.L1Loss(reduction="sum")
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))


def test_l2norm_reduction(forces, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="l2norm", reduction="mean")
    pred, target = forces
    ref_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    ref_loss = ref_norm.mean()
    assert torch.allclose(loss(pred, target, natoms), ref_loss)
    # sum reduction
    loss = DDPLoss(loss_name="l2norm", reduction="sum")
    ref_loss = ref_norm.sum()
    assert torch.allclose(loss(pred, target, natoms), ref_loss)


def test_mse_reduction(forces, natoms):
    # this is testing on a single process i.e. world_size=1
    # mean reduction
    loss = DDPLoss(loss_name="mse", reduction="mean")
    ref_loss = nn.MSELoss(reduction="mean")
    pred, target = forces
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))
    # sum reduction
    loss = DDPLoss(loss_name="mse", reduction="sum")
    ref_loss = nn.MSELoss(reduction="sum")
    assert torch.allclose(loss(pred, target, natoms), ref_loss(pred, target))


def split_batch_for_ddp(
    task: str, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
):
    if task == "energy":
        return list(torch.split(pred, 1)), list(torch.split(target, 1))
    elif task == "forces":
        split_shape = natoms.tolist()
        return list(torch.split(pred, split_shape, dim=0)), list(
            torch.split(target, split_shape, dim=0)
        )
    else:
        raise ValueError(f"Invalid task: {task}")


def run_ddp_loss(pred, target, natoms, loss_name, reduction):
    loss = DDPLoss(loss_name=loss_name, reduction=reduction)
    local_rank = distutils.get_rank()
    return loss(pred[int(local_rank)], target[int(local_rank)], natoms)


@pytest.fixture()
def world_size():
    # batch size = 4
    return 4


@pytest.mark.serial()
def test_ddp_mae(energy, natoms, world_size):
    pred, target = energy
    ddp_pred, ddp_target = split_batch_for_ddp("energy", pred, target, natoms)
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=1, use_gp=False
    )
    output = spawn_multi_process(
        config,
        run_ddp_loss,
        init_pg_and_rank_and_launch_test,
        ddp_pred,
        ddp_target,
        natoms,
        "mae",
        "mean",
    )
    # this mocks what ddp does when averaging gradients
    ddp_loss = torch.sum(torch.tensor(output)) / float(world_size)
    ref_loss = nn.L1Loss(reduction="mean")
    assert torch.allclose(ddp_loss, ref_loss(pred, target))


@pytest.mark.serial()
def test_ddp_l2norm(forces, natoms, world_size):
    pred, target = forces
    ddp_pred, ddp_target = split_batch_for_ddp("forces", pred, target, natoms)
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=1, use_gp=False
    )
    output = spawn_multi_process(
        config,
        run_ddp_loss,
        init_pg_and_rank_and_launch_test,
        ddp_pred,
        ddp_target,
        natoms,
        "l2norm",
        "mean",
    )
    # this mocks what ddp does when averaging gradients
    ddp_loss = torch.sum(torch.tensor(output)) / float(world_size)
    ref_norm = torch.linalg.vector_norm(pred - target, ord=2, dim=-1)
    ref_loss = ref_norm.mean()
    assert torch.allclose(ddp_loss, ref_loss)


@pytest.mark.serial()
def test_ddp_mse(forces, natoms, world_size):
    pred, target = forces
    ddp_pred, ddp_target = split_batch_for_ddp("forces", pred, target, natoms)
    config = PGConfig(
        backend="gloo", world_size=world_size, gp_group_size=1, use_gp=False
    )
    output = spawn_multi_process(
        config,
        run_ddp_loss,
        init_pg_and_rank_and_launch_test,
        ddp_pred,
        ddp_target,
        natoms,
        "mse",
        "mean",
    )
    # this mocks what ddp does when averaging gradients
    ddp_loss = torch.sum(torch.tensor(output)) / float(world_size)
    ref_loss = nn.MSELoss(reduction="mean")
    assert torch.allclose(ddp_loss, ref_loss(pred, target))


@pytest.mark.parametrize("mode", ["sum", "mean", "per_structure"])
def test_masked_loss_device_counts_and_gradients(mode):
    from fairchem.core.modules.loss import DDPMTLoss

    loss_fn = DDPMTLoss(MSELoss(), reduction=mode)
    pred = torch.tensor([2.0, 1.0, 3.0], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, False, True])
    natoms = torch.tensor([1, 2])
    weight = 1 if mode == "sum" else 0.5
    result = loss_fn(pred, torch.zeros_like(pred), mask, natoms)
    torch.testing.assert_close(result, pred.new_tensor(13 * weight))
    torch.testing.assert_close(
        torch.autograd.grad(result, pred)[0], pred.new_tensor([4, 0, 6]) * weight
    )
    empty = loss_fn(pred, torch.zeros_like(pred), ~torch.ones_like(mask), natoms)
    assert empty.item() == 0


def test_masked_loss_unequal_replica_counts(monkeypatch):
    from fairchem.core.modules.loss import DDPMTLoss

    # Three valid components on this rank and six on its peer.
    monkeypatch.setattr(distutils, "get_world_size", lambda: 2)

    def add_peer_count(count, **kwargs):
        assert count.device.type == "cpu" and count.item() == 3
        return count + 6

    monkeypatch.setattr(distutils, "all_reduce", add_peer_count)
    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    out = DDPMTLoss(MSELoss())(
        pred, torch.zeros_like(pred), torch.tensor([[True], [False]]), torch.tensor([2])
    )
    torch.testing.assert_close(out, pred.new_tensor(28 / 9))
    torch.testing.assert_close(
        torch.autograd.grad(out, pred)[0],
        pred.new_tensor([[4 / 9, 8 / 9, 12 / 9], [0.0, 0.0, 0.0]]),
    )

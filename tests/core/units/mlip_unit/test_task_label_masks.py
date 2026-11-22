"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.modules.loss import DDPMTLoss, MSELoss
from fairchem.core.modules.normalization.element_references import ElementReferences
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit._metrics import Metrics
from fairchem.core.units.mlip_unit.mlip_unit import (
    OutputSpec,
    Task,
    compute_loss,
    compute_metrics,
    get_output_mask,
)


def _batch():
    structures = [AtomicData.from_ase(Atoms(symbols)) for symbols in ("He", "H2")]
    for sample, name in zip(structures, ("one", "two")):
        sample.dataset_name = name
    return atomicdata_list_to_batch(structures)


def _task(name, level):
    return Task(
        name="custom_label",
        level=level,
        property=name,
        out_spec=OutputSpec(dim=[], dtype="float32"),
        normalizer=Normalizer(),
        datasets=["one", "two"],
        metrics=["mae", "per_atom_mae"],
        loss_fn=DDPMTLoss(MSELoss()),
    )


def test_matrix_stress_alias_masks_only_selected_system_and_finite_components():
    batch = _batch()
    batch.custom_label = torch.zeros(2, 3, 3)
    batch.custom_label[1, 0, 1] = torch.nan
    pred = torch.arange(18, dtype=torch.float32).reshape(2, 9).requires_grad_()
    task = _task("stress", "system")
    predictions = {task.name: {task.property: pred}}
    masks = get_output_mask(batch, task)
    assert masks["two.custom_label"].shape == (2, 3, 3)
    assert masks["two.custom_label"][0].sum() == 0
    metrics = compute_metrics(task, predictions, batch, "two")
    assert metrics["mae"].numel == 8
    assert metrics["mae"].metric == pytest.approx((sum(range(9, 18)) - 10) / 8)
    assert metrics["per_atom_mae"].metric == metrics["mae"].metric / 2
    # Training only needs validity, independent of dataset-specific metric masks.
    del batch["dataset_name"]
    loss = compute_loss([task], predictions, batch)[task.name]
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad[1, 1] == 0


def test_column_energy_with_references_does_not_mix_per_atom_denominators():
    batch = _batch()
    batch.custom_label = torch.tensor([[2.0], [4.0]])
    task = _task("energy", "system")
    task.element_references = ElementReferences(torch.tensor([0.0, 1.0, 1.0]))
    batch.batch_full = batch.batch
    batch.atomic_numbers_full = batch.atomic_numbers
    pred = torch.zeros(2, requires_grad=True)
    predictions = {task.name: {task.property: pred}}
    metrics = compute_metrics(task, predictions, batch)
    assert metrics["per_atom_mae"] == Metrics(metric=1, total=2, numel=2)
    assert metrics["mae"] == Metrics(metric=1.5, total=3, numel=2)
    loss = compute_loss([task], predictions, batch)[task.name]
    assert loss.item() == 2.5
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([-1.0, -2.0]))


def test_force_alias_uses_complete_mobile_vectors_and_empty_metric_is_accumulable():
    batch = _batch()
    batch.custom_label = torch.zeros(3, 3)
    batch.custom_label[1, 2] = torch.nan
    batch.fixed[0] = 1
    task = _task("forces", "atom")
    pred = torch.ones(3, 3, requires_grad=True)
    predictions = {task.name: {task.property: pred}}
    metrics = compute_metrics(task, predictions, batch, "two")
    assert metrics["mae"] == Metrics(metric=1, total=3, numel=3)
    assert metrics["per_atom_mae"] == Metrics(metric=0.5, total=1.5, numel=3)
    empty = Metrics()
    empty += compute_metrics(task, predictions, batch, "one")["mae"]
    assert empty == Metrics()
    empty += metrics["mae"]
    assert empty == metrics["mae"]
    loss = compute_loss([task], predictions, batch)[task.name]
    loss.backward()
    assert torch.all(pred.grad[:2] == 0)
    torch.testing.assert_close(pred.grad[2], torch.full((3,), 2 / 3))


@pytest.mark.parametrize("empty_local_rank", [False, True])
def test_rmse_pools_component_errors_across_batches_and_ranks(
    monkeypatch, empty_local_rank
):
    from types import SimpleNamespace

    from fairchem.core.units.mlip_unit import mlip_unit
    from fairchem.core.units.mlip_unit._metrics import rmse

    pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    label = torch.zeros_like(pred)
    expected = pred.double().square().mean().sqrt().item()
    split = 0 if empty_local_rank else 1
    local = Metrics()
    for row in pred[:split]:
        local += rmse(row, torch.zeros_like(row))
    remote = rmse(pred[split:], label[split:])
    combined = Metrics()
    combined += local
    combined += remote
    assert combined.metric == pytest.approx(expected)
    assert combined.numel == pred.numel()

    unit = SimpleNamespace(
        running_metrics={"custom": {"one.val": {"rmse": local}}},
        total_atoms=3,
        total_runtime=1.0,
        total_loss_metrics=Metrics(),
        eval_progress=SimpleNamespace(num_epochs_completed=1),
        logger=None,
    )
    # Feed the rank-local accumulation through the normal epoch finalizer.
    remote_values = iter([remote.total, remote.numel, 1.0, 2])
    monkeypatch.setattr(mlip_unit, "get_device_for_local_rank", lambda: "cpu")
    monkeypatch.setattr(
        mlip_unit.distutils,
        "all_reduce",
        lambda value, **kwargs: value + next(remote_values),
    )
    log = mlip_unit.MLIPEvalUnit.on_eval_epoch_end(unit, None)
    assert log["val/one.val,custom,rmse"] == pytest.approx(expected)

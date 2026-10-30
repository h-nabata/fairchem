"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.units.mlip_unit.api.inference import (
    MergeMoleConsistencyError,
    inference_settings_xpu,
)


def _model(model_id, use_composition_embedding=True):
    model = eSCNMDMoeBackbone(
        sphere_channels=4,
        hidden_channels=8,
        edge_channels=4,
        num_distance_basis=4,
        num_layers=1,
        num_experts=2,
        use_dataset_embedding=False,
        use_composition_embedding=use_composition_embedding,
        lmax=2,
        mmax=2,
        execution_mode="general",
    ).eval()
    model.model_id = model_id
    return model


def _data(copies=1):
    atoms = molecule("H2O")
    for i in range(1, copies):
        extra = molecule("H2O")
        extra.translate([5 * i, 0, 0])
        atoms += extra
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(atoms, task_name="omol")


def test_uma_1p2_router_really_depends_on_absolute_composition():
    """A deterministic router exposes the trained generation's size dependence."""
    model = _model("UMA-S-1.2")
    model.routing_mlp = torch.nn.Linear(8, 2, bias=False)
    with torch.no_grad():
        model.composition_embedding.weight.fill_(1)
        model.routing_mlp.weight.zero_()
        model.routing_mlp.weight[0, 0] = 1
        model.routing_mlp.weight[1, 0] = -1
    data = atomicdata_list_to_batch([_data(), _data(2)])
    model.set_MOLE_coefficients(data.atomic_numbers, data.batch, torch.zeros(2, 4))
    # The production normalization adds a small offset to softmax.
    expected = torch.tensor([[3 / 4, -3 / 4], [6 / 7, -6 / 7]]).softmax(-1) + 0.005
    actual = model.global_mole_tensors.expert_mixing_coefficients
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual[0], actual[1])


@pytest.mark.parametrize(
    ("model_id", "use_composition_embedding", "must_reject"),
    [
        ("UMA-S-1.2", True, True),
        ("UMA-S-1.2.1", True, False),
        ("UMA-S-1.1", True, False),
        (None, True, False),
        ("UMA-S-1.2", False, False),
    ],
)
def test_scaled_composition_initial_batch_merge(
    model_id, use_composition_embedding, must_reject
):
    model = _model(model_id, use_composition_embedding)
    data = atomicdata_list_to_batch([_data(), _data(2)])
    if must_reject:
        with pytest.raises(MergeMoleConsistencyError, match="absolute compositions"):
            model.prepare_for_inference(data, inference_settings_xpu())
        assert model.num_experts == 2
    else:
        merged = model.prepare_for_inference(data, inference_settings_xpu())
        merged.on_predict_check(data)


@pytest.mark.parametrize(
    ("model_id", "use_composition_embedding", "must_reject"),
    [
        ("UMA-S-1.2", True, True),
        ("UMA-S-1.2.1", True, False),
        (None, True, False),
        ("UMA-S-1.2", False, False),
    ],
)
def test_scaled_composition_after_merge(
    model_id, use_composition_embedding, must_reject
):
    model = _model(model_id, use_composition_embedding)
    merged = model.prepare_for_inference(_data(), inference_settings_xpu())
    # The bound guard runs on the replacement plain backbone, not the MoE.
    merged.on_predict_check(_data())
    if must_reject:
        with pytest.raises(MergeMoleConsistencyError, match="absolute composition"):
            merged.on_predict_check(_data(2))
    else:
        merged.on_predict_check(_data(2))


def test_uma_1p2_allows_same_counts_with_different_atom_order_and_geometry():
    data = _data()
    permuted = data.clone()
    permuted.atomic_numbers = data.atomic_numbers.flip(0)
    permuted.pos = data.pos.flip(0) + 10
    model = _model("UMA-S-1.2")
    batch = atomicdata_list_to_batch([data, permuted])
    merged = model.prepare_for_inference(batch, inference_settings_xpu())
    merged.on_predict_check(permuted)


def test_uma_1p2_rejects_changed_element_counts_even_with_same_size():
    data = _data()
    changed = data.clone()
    changed.atomic_numbers[0] = 6
    model = _model("UMA-S-1.2")
    merged = model.prepare_for_inference(data, inference_settings_xpu())
    with pytest.raises(MergeMoleConsistencyError, match="absolute composition"):
        merged.on_predict_check(changed)

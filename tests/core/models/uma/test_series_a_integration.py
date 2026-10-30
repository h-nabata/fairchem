"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone
from fairchem.core.models.uma.nn.execution_backends import get_execution_backend
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def _data(periodic=True):
    atoms = Atoms(
        numbers=[1, 8, 1],
        positions=[[0.2, 0.1, 0.3], [2.9, 0.3, 0.4], [1.4, 1.7, 0.8]],
        cell=[4, 4.2, 4.4],
        pbc=periodic,
    )
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol")
    for key, value in data.to_dict().items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            data[key] = value.double()
    return data


def _model(**kwargs):
    backbone = (
        eSCNMDBackbone(
            sphere_channels=4,
            hidden_channels=8,
            edge_channels=4,
            num_distance_basis=4,
            num_layers=1,
            lmax=2,
            mmax=2,
            regress_forces=True,
            regress_stress=True,
            direct_forces=False,
            cutoff=2.6,
            always_use_pbc=False,
            use_dataset_embedding=False,
            otf_graph=True,
            **kwargs,
        )
        .double()
        .eval()
    )
    head = MLP_EFS_Head(backbone, wrap_property=False).double().eval()
    backbone.requires_grad_(False)
    head.requires_grad_(False)
    return backbone, head


def _evaluate(model, data, requested=None):
    backbone, head = model
    data = data.clone()
    if requested is not None:
        data["_requested_properties"] = requested
    return head(data, backbone(data))


@pytest.mark.parametrize("periodic", [False, True])
def test_edge_bucketing_preserves_energy_force_stress_and_hessian(periodic):
    model = _model()
    padded = copy.deepcopy(model)
    padded[0].edge_chunk_size = 32
    data = _data(periodic)
    properties = ("energy", "forces", "stress", "hessian")
    reference = _evaluate(model, data, properties)
    assert torch.count_nonzero(reference["forces"]) > 0
    assert torch.count_nonzero(reference["hessian"]) > 0
    actual = _evaluate(padded, data, properties)
    for name in properties:
        torch.testing.assert_close(actual[name], reference[name], atol=2e-9, rtol=2e-8)
    graph = padded[0]._generate_graph(data)
    assert graph["edge_index"].shape[1] % 32 == 0
    assert graph["edge_valid"].sum() < graph["edge_valid"].numel()


def test_energy_only_avoids_derivative_work_and_caller_mutation():
    model = _model()
    data = _data()
    with patch(
        "fairchem.core.models.uma.escn_md.compute_forces", side_effect=AssertionError
    ), patch(
        "fairchem.core.models.uma.escn_md.compute_forces_and_stress",
        side_effect=AssertionError,
    ):
        output = _evaluate(model, data, ("energy",))
    assert set(output) == {"energy"}
    assert not data.pos.requires_grad
    assert not data.cell.requires_grad


def test_hvp_matches_full_hessian_and_emits_only_requested_properties():
    model = _model(hessian_chunk_size=2)
    data = _data()
    data["hessian_vector"] = torch.arange(1, 10, dtype=data.pos.dtype).reshape(3, 3) / 9
    reference = _evaluate(model, data, ("hessian",))
    actual = _evaluate(model, data, ("hessian_vector_product",))
    assert set(actual) == {"hessian_vector_product"}
    expected = (reference["hessian"][0] @ data["hessian_vector"].flatten()).reshape(
        3, 3
    )
    torch.testing.assert_close(
        actual["hessian_vector_product"], expected, atol=2e-9, rtol=2e-8
    )
    both = _evaluate(model, data, ("hessian", "hessian_vector_product"))
    torch.testing.assert_close(both["hessian"], reference["hessian"])
    torch.testing.assert_close(
        both["hessian_vector_product"], expected, atol=2e-9, rtol=2e-8
    )


def test_explicit_derivatives_work_without_regression_flags_under_no_grad():
    model = _model()
    model[0].regress_config.forces = False
    model[0].regress_config.stress = False
    with torch.no_grad():
        output = _evaluate(model, _data(), ("hessian",))
    assert output["hessian"].shape == (1, 9, 9)
    assert torch.isfinite(output["hessian"]).all()


@pytest.mark.parametrize("bucket", [1, 16])
def test_isolated_atoms_have_zero_force_and_curvature(bucket):
    model = _model(edge_chunk_size=bucket)
    data = _data(False)
    data.pos = data.pos * 20
    output = _evaluate(model, data, ("energy", "forces", "stress", "hessian"))
    for name in ("forces", "stress", "hessian"):
        torch.testing.assert_close(output[name], torch.zeros_like(output[name]))


@pytest.mark.parametrize("mode", ["general", "umas_fast_portable"])
def test_combined_cache_spatial_padding_and_portable_geometry(mode):
    model = _model()
    candidate = copy.deepcopy(model)
    backbone = candidate[0]
    backbone.neighbor_skin = 0.4
    backbone.graph_search_method = "auto"
    backbone.graph_edge_order = "spatial"
    backbone.edge_chunk_size = 16
    backbone.backend = get_execution_backend(mode)
    backbone.edge_degree_embedding.backend = backbone.backend
    for block in backbone.blocks:
        block.edge_wise.backend = backbone.backend
    data = _data()
    backbone.prepare_for_inference(
        data.clone(),
        InferenceSettings(
            execution_mode=mode, merge_mole=True, activation_checkpointing=False
        ),
    )
    properties = ("energy", "forces", "stress", "hessian")
    for step in range(3):
        moved = data.clone()
        moved.pos[0, 0] += step * 0.025
        if step == 2:
            moved.cell[0, 0, 0] += 0.1
        # The same random gauge makes the numerical comparison more sensitive.
        torch.manual_seed(24)
        expected = _evaluate(model, moved, properties)
        torch.manual_seed(24)
        actual = _evaluate(candidate, moved, properties)
        for name in properties:
            torch.testing.assert_close(
                actual[name], expected[name], atol=2e-9, rtol=2e-8
            )
    assert backbone._neighbor_cache.hits >= 1
    assert backbone._neighbor_cache.rebuilds >= 2


def test_padding_preserves_mixed_dataset_unmerged_mole_routing():
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
    from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor

    reference = _small_predictor()
    candidate = copy.deepcopy(reference)
    for predictor in (reference, candidate):
        predictor.inference_settings.merge_mole = False
        predictor._process_outputs = lambda data, output, undo_refs: output
    backbone = candidate.model.module.backbone
    backbone.edge_chunk_size = 32
    backbone.graph_edge_order = "spatial"
    backbone.neighbor_skin = 0.4
    first, second = _data(), _data()
    second.dataset = ["omat"]
    second.charge += 1
    second.spin += 1
    second.pos[1, 1] += 0.15
    data = atomicdata_list_to_batch([first, second])
    properties = ("energy", "forces", "stress")
    outputs = [
        MLIPPredictUnit.predict.__wrapped__(
            predictor, data, requested_properties=properties
        )
        for predictor in (reference, candidate)
    ]
    for index, dataset in enumerate(data.dataset):
        for prop in properties:
            key = f"{dataset}_{prop}"
            values = [out[key][prop] for out in outputs]
            selected = [
                value[data.batch == index] if prop == "forces" else value[index]
                for value in values
            ]
            assert all(torch.isfinite(value).all() for value in selected)
            torch.testing.assert_close(*selected, atol=3e-5, rtol=2e-4)

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
from fairchem.core.models.base import HydraModel, HydraModelV2, supports_graph_retention
from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.mole import MOLE, norm_str_to_fn
from fairchem.core.models.uma.nn.portable_wigner import segmented_sum
from fairchem.core.models.uma.nn.radial import GaussianSmearing
from fairchem.core.models.uma.outputs import compute_forces, compute_forces_and_stress
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MergeMoleConsistencyError,
)
from tests.core.models.uma.uma_nitro.nitro_test_utils import batch

_SMALL = dict(
    num_layers=1,
    sphere_channels=8,
    hidden_channels=16,
    edge_channels=8,
    num_distance_basis=8,
    lmax=2,
    mmax=2,
    ff_type="spectral",
    otf_graph=True,
    cutoff=3.0,
    always_use_pbc=False,
    direct_forces=False,
)
_MOE = dict(num_experts=2, use_composition_embedding=True)


def test_head_autocast_follows_input_device_after_migration():
    from torch._subclasses.fake_tensor import FakeTensorMode

    class Head(torch.nn.Module):
        use_amp = True

        def forward(self, data, embeddings):
            return torch.is_autocast_enabled(data["pos"].device.type)

    model = HydraModelV2(torch.nn.Identity(), {"head": Head()})
    with FakeTensorMode():
        for device in ("cpu", "xpu", "cpu"):
            assert model({"pos": torch.empty(2, 3, device=device)})["head"]


@pytest.mark.parametrize("num_heads", [1, 2, 3])
def test_head_graph_retention_compiles_as_one_graph(num_heads):
    class Head(torch.nn.Module):
        use_amp = False

        @supports_graph_retention
        def forward(self, data, embeddings, *, retain_graph=False):
            x = embeddings["pos"]
            return x.square() if retain_graph else x.sin()

    class EnergyOnly(Head):
        def forward(self, data, embeddings):
            return embeddings["pos"].sin()

    heads = {str(n): Head() for n in range(num_heads)}
    if num_heads == 3:
        heads["1"] = EnergyOnly()
    model = HydraModelV2(torch.nn.Identity(), heads)
    data = {"pos": torch.tensor([1.0, 2.0])}
    compiled = torch.compile(model, fullgraph=True, backend="eager")
    with torch.no_grad():
        actual = compiled(data)
        assert not torch.is_grad_enabled()
    for index in range(num_heads):
        retained = index < num_heads - 1 and not isinstance(
            heads[str(index)], EnergyOnly
        )
        value = data["pos"].square() if retained else data["pos"].sin()
        torch.testing.assert_close(actual[str(index)], value)


@pytest.mark.parametrize(
    "properties",
    [
        ("energy", "forces"),
        ("energy", "stress"),
        ("hessian_vector_product",),
        ("hessian", "hessian_vector_product"),
    ],
)
def test_conservative_heads_share_backbone_in_evaluation(properties):
    torch.manual_seed(401)
    model = HydraModel(
        backbone=dict(
            model="fairchem.core.models.uma.escn_md.eSCNMDBackbone",
            **_SMALL,
            regress_stress=True,
            use_dataset_embedding=False,
        ),
        heads={
            name: {
                "module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "wrap_property": False,
            }
            for name in ("first", "second")
        },
    ).eval()
    water = molecule("H2O")
    water.set_cell([8, 8, 8])
    inputs = batch((water, "omol"))
    inputs["_requested_properties"] = properties
    inputs["hessian_vector"] = torch.rand_like(inputs.pos)

    def run(module):
        torch.manual_seed(402)
        return module(inputs.clone())

    with torch.no_grad():
        outputs = run(model)
        assert not torch.is_grad_enabled()
    for name in model.output_heads:
        independent = copy.deepcopy(model)
        independent.output_heads = torch.nn.ModuleDict(
            {name: independent.output_heads[name]}
        )
        expected = run(independent)[name]
        assert set(outputs[name]) == set(properties)
        for prop in properties:
            torch.testing.assert_close(outputs[name][prop], expected[prop])


def test_overlapping_checkpointed_forwards_keep_input_and_parameter_gradients():
    torch.manual_seed(42)
    model = eSCNMDMoeBackbone(
        **_SMALL, **_MOE, use_dataset_embedding=False, activation_checkpointing=True
    ).eval()
    model.blocks[0].edge_wise.activation_checkpoint_chunk_size = 3

    def gradients(backbone, overlap):
        layer = next(m for m in backbone.blocks[0].modules() if isinstance(m, MOLE))
        parameters = (layer.weights, backbone.routing_mlp[0].weight)
        losses, positions, derivatives = [], [], []
        torch.manual_seed(406)
        for name in ("H2O", "CH4"):
            data = batch((molecule(name), "omol"))
            positions.append(data.pos.requires_grad_())
            losses.append(backbone(data)["node_embedding"][:, 0, 0].sum())
            if not overlap:
                derivatives.append(
                    torch.autograd.grad(losses[-1], (positions[-1], *parameters))
                )
        if overlap:
            return torch.autograd.grad(sum(losses), (*positions, *parameters))
        return (
            *[g[0] for g in derivatives],
            *[
                a + b
                for a, b in zip(derivatives[0][1:], derivatives[1][1:], strict=True)
            ],
        )

    reference = gradients(copy.deepcopy(model), overlap=False)
    observed = gradients(model, overlap=True)
    for expected, actual in zip(reference, observed, strict=True):
        torch.testing.assert_close(actual, expected)


def test_external_interleaved_edges_preserve_uma_energy_forces_and_stress():
    torch.manual_seed(43)
    backbone = eSCNMDMoeBackbone(
        **(_SMALL | {"otf_graph": False}),
        **_MOE,
        use_dataset_embedding=False,
        regress_stress=True,
    ).eval()
    head = MLP_EFS_Head(backbone, wrap_property=False).eval()
    edges = torch.tensor([[1, 0, 3, 2], [0, 1, 2, 3]])
    offsets = torch.tensor([[-1, 0, 0], [1, 0, 0], [-1, 0, 0], [1, 0, 0]]).float()

    def run(order):
        data = AtomicData(
            pos=torch.tensor(
                [[0.1, 0.0, 0.0], [4.6, 0.1, 0.2], [0.2, 0.0, 0.0], [6.5, 0.2, 0.1]]
            ),
            atomic_numbers=torch.tensor([1, 8, 6, 1]),
            cell=torch.stack((torch.eye(3) * 5, torch.eye(3) * 7)),
            pbc=torch.ones(2, 3, dtype=torch.bool),
            natoms=torch.tensor([2, 2]),
            edge_index=edges[:, order],
            cell_offsets=offsets[order],
            nedges=torch.tensor([2, 2]),
            charge=torch.tensor([0, 0]),
            spin=torch.tensor([1, 1]),
            tags=torch.zeros(4, dtype=torch.long),
            fixed=torch.zeros(4, dtype=torch.long),
            batch=torch.tensor([0, 0, 1, 1]),
            dataset=["omol", "omol"],
            sid=["a", "b"],
        )
        torch.manual_seed(44)
        return head(data, backbone(data))

    reference, actual = run([0, 1, 2, 3]), run([0, 2, 1, 3])
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], reference[key], rtol=0, atol=0)


@pytest.mark.parametrize("with_moe_backbone", [True, False])
def test_merged_model_checks_all_head_datasets(with_moe_backbone):
    datasets = {"omol": "omol", "omat": "omat"}
    model_class = eSCNMDMoeBackbone if with_moe_backbone else eSCNMDBackbone
    model = HydraModel(
        backbone=dict(
            model=f"{model_class.__module__}.{model_class.__name__}",
            **_SMALL,
            **(_MOE if with_moe_backbone else {}),
            dataset_mapping=datasets,
            use_dataset_embedding=True,
        ),
        heads={
            "efs": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "head_kwargs": {"wrap_property": False},
                "dataset_mapping": datasets,
            }
        },
        pass_through_head_outputs=True,
        model_id="UMA-S-1.2.1",
    ).eval()
    model._tasks = {}
    water = molecule("H2O")
    first, other = (batch((water, name)) for name in ("omol", "omat"))
    mixed = batch((water, "omol"), (water, "omat"))
    settings = InferenceSettings(merge_mole=True, execution_mode="general")
    with pytest.raises(MergeMoleConsistencyError, match="dataset"):
        copy.deepcopy(model).prepare_for_inference(mixed, settings)
    first["_requested_properties"] = ("energy",)
    first["_present_datasets_only"] = True
    model.prepare_for_inference(first, settings)
    model.on_predict_check(first)
    assert set(model(first)) == {"omol_energy"}
    for invalid in (mixed, other):
        with pytest.raises(MergeMoleConsistencyError, match=r"(?i)dataset|merged"):
            model.on_predict_check(invalid)


@pytest.mark.parametrize("inference_indices", [True, False])
def test_segmented_sum_inference_warmup_followed_by_double_backward(inference_indices):
    with torch.inference_mode(inference_indices):
        target = torch.tensor([0, 1, 0])
    values = torch.tensor([1.0, 2.0, 4.0], requires_grad=True)
    with torch.inference_mode():
        segmented_sum(values, target, 2)
    first = torch.autograd.grad(
        segmented_sum(values, target, 2).square().sum(), values, create_graph=True
    )[0]
    torch.testing.assert_close(first, torch.tensor([10.0, 4.0, 10.0]))
    torch.testing.assert_close(
        torch.autograd.grad(first.sum(), values)[0], torch.tensor([4.0, 2.0, 4.0])
    )
    # A subsequent host write through numpy doesn't bump the version counter.
    target.numpy()[2] = 1
    with torch.inference_mode():
        torch.testing.assert_close(
            segmented_sum(values, target, 2), torch.tensor([1.0, 6.0])
        )


@pytest.mark.parametrize("systems", [1, 2])
def test_strict_virial_after_large_translation(systems):
    pos = (
        torch.tensor([[1e6, 0.0, 0.0], [1e6 + 1, 0.0, 0.0]])
        .repeat(systems, 1)
        .requires_grad_()
    )
    cell = torch.eye(3).repeat(systems, 1, 1).requires_grad_()
    indices = torch.arange(systems).repeat_interleave(2)
    k = torch.tensor(1.1)
    energy = (pos[1::2, 0] - pos[::2, 0]) * k
    forces, stress = compute_forces_and_stress(energy, pos, cell, indices)
    torch.testing.assert_close(forces[:, 0], torch.stack((k, -k)).repeat(systems))
    torch.testing.assert_close(stress[:, 0], k.double().expand(systems), rtol=0, atol=0)
    assert not stress[:, 1:].any()


def test_singular_cell_identifies_system_and_allows_force_only():
    pos = torch.randn(2, 3, requires_grad=True)
    cell = torch.eye(3).repeat(2, 1, 1)
    cell[1, 2] = 0
    cell.requires_grad_()
    energy = pos.square().sum(1)
    with pytest.raises(ValueError, match=r"nonzero cell volume for systems \[1\]"):
        compute_forces_and_stress(energy, pos, cell, torch.arange(2))
    torch.testing.assert_close(compute_forces(energy, pos), -2 * pos)


def test_partial_periodic_structure_needs_physical_volume_only_for_stress():
    water = molecule("H2O")
    water.set_pbc((True, True, False))
    water.set_cell((5, 6, 0))
    model = eSCNMDBackbone(
        **_SMALL,
        use_dataset_embedding=False,
        regress_stress=True,
    ).eval()
    head = MLP_EFS_Head(model, wrap_property=False).eval()

    def predict(properties):
        data = batch((water, "omol"))
        data["_requested_properties"] = properties
        return head(data, model(data))

    assert predict(("energy", "forces"))["forces"].isfinite().all()
    with pytest.raises(ValueError, match=r"nonzero cell volume for systems \[0\]"):
        predict(("energy", "stress"))
    water.set_cell((5, 6, 8))
    assert predict(("energy", "stress"))["stress"].isfinite().all()


@pytest.mark.parametrize("customization", ["standard", "subclass", "hook", "late_hook"])
def test_portable_geometry_uses_live_radial_and_envelope(customization):
    class ScaledGaussian(GaussianSmearing):
        def forward(self, dist):
            return super().forward(dist) * 2

    model = eSCNMDBackbone(
        **_SMALL, execution_mode="umas_fast_portable", use_dataset_embedding=False
    ).eval()
    if customization == "subclass":
        model.distance_expansion = ScaledGaussian(0, model.cutoff, 8)

    def attach():
        model.distance_expansion.register_forward_hook(lambda _m, _x, y: 2 * y)
        model.envelope.register_forward_hook(lambda _m, _x, y: 3 * y)

    if customization == "hook":
        attach()
    model.backend.prepare_model_for_inference(model)
    if customization == "late_hook":
        attach()
    vectors = torch.tensor([[0.3, 1.0, 0.2], [1.0, -0.4, 0.3]], requires_grad=True)
    distance = vectors.norm(dim=-1)
    expected = (
        model.distance_expansion(distance),
        model.envelope(distance / model.cutoff)[:, None],
    )
    actual = model._compute_geometry(vectors, distance)[:2]

    def derivatives(outputs):
        return torch.autograd.grad(
            sum(t.square().sum() for t in outputs), vectors, retain_graph=True
        )[0]

    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(
        derivatives(actual), derivatives(expected), rtol=0, atol=0
    )


def test_per_system_pnorm_is_batch_independent_and_legacy_pnorm_is_preserved():
    x = torch.tensor([[1.0, 3.0]])
    legacy, independent = (
        norm_str_to_fn(name) for name in ("pnorm", "pnorm_per_system")
    )
    torch.testing.assert_close(legacy(x), torch.tensor([[0.375, 0.625]]))
    repeated = x.expand(3, -1)
    torch.testing.assert_close(
        legacy(repeated), torch.tensor([[5 / 16, 11 / 16]]).expand_as(repeated)
    )
    torch.testing.assert_close(independent(x), torch.tensor([[1 / 3, 2 / 3]]))
    torch.testing.assert_close(independent(repeated)[0], independent(x)[0])

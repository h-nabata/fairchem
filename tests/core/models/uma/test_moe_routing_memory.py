"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from ase.build import molecule
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.base import HydraModel
from fairchem.core.models.uma.escn_md import GradRegressConfig
from fairchem.core.models.uma.escn_moe import (
    DatasetSpecificMoEWrapper,
    DatasetSpecificSingleHeadWrapper,
    eSCNMDMoeBackbone,
)
from fairchem.core.models.uma.nn.mole import MOLEGlobals
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu


class _OutputsHead(torch.nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, data, emb):
        return self.outputs


def _wrapper(wrapper_type, outputs, wrap_property=False):
    # Exercise the real wrapper forward methods independently of head weights.
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    wrapper.regress_config = GradRegressConfig(forces=True)
    wrapper.head = _OutputsHead(outputs)
    wrapper.dataset_names = ["a", "b", "absent"]
    wrapper.dataset_name_to_exp = {"a": 0, "b": 1, "absent": 2}
    wrapper.global_mole_tensors = MOLEGlobals(None, None)
    wrapper.merged_on_dataset = None
    wrapper.wrap_property = wrap_property
    return wrapper


def _batch(device="cpu"):
    batch = torch.tensor([0, 0, 1, 1, 1, 2], device=device)
    data = SimpleNamespace(
        dataset=["a", "b", "a"],
        natoms=torch.tensor([2, 3, 1], device=device),
        pos=torch.empty((6, 3), device=device),
        batch_full=batch,
    )
    return data, {"batch": batch}


@pytest.mark.parametrize(
    "wrapper_type", [DatasetSpecificMoEWrapper, DatasetSpecificSingleHeadWrapper]
)
@pytest.mark.parametrize("wrap_property", [False, True])
def test_dataset_routing_matches_values_and_two_derivatives(
    wrapper_type, wrap_property
):
    data, emb = _batch()
    x = torch.linspace(-0.7, 0.8, 18, dtype=torch.float64).reshape(6, 3)
    x.requires_grad_()
    outputs = {
        "energy": x.square().reshape(3, 6).sum(1),
        "forces": x.sin(),
        "stress": x.cos().reshape(3, 3, 2),
        "atom_scalar": x.square().sum(1),
    }
    result = _wrapper(wrapper_type, outputs, wrap_property)(data, emb)

    expected = {}
    for dataset, systems, atoms in (
        ("a", [0, 2], [0, 1, 5]),
        ("b", [1], [2, 3, 4]),
        ("absent", [], []),
    ):
        for name, value in outputs.items():
            target = torch.zeros_like(value)
            if systems:
                indices = systems if value.shape[0] == 3 else atoms
                target[indices] = value[indices]
            key = f"{dataset}_{name}"
            actual = result[key][name] if wrap_property else result[key]
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
            assert actual.data_ptr() != value.data_ptr()
            expected[key] = target

    actual_loss = sum(
        (value[next(iter(value))] if wrap_property else value).square().sum()
        for value in result.values()
    )
    reference_loss = sum(value.square().sum() for value in expected.values())
    actual_grad = torch.autograd.grad(actual_loss, x, create_graph=True)[0]
    reference_grad = torch.autograd.grad(reference_loss, x, create_graph=True)[0]
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual_grad.sum(), x, retain_graph=True)[0],
        torch.autograd.grad(reference_grad.sum(), x)[0],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "wrapper_type", [DatasetSpecificMoEWrapper, DatasetSpecificSingleHeadWrapper]
)
def test_dataset_routing_excludes_nonfinite_unselected_rows(wrapper_type):
    data, emb = _batch()
    outputs = {
        "energy": torch.tensor([1.0, float("nan"), 3.0]),
        "forces": torch.tensor([1.0, 2.0, float("nan"), float("inf"), 5.0, 6.0]),
    }
    result = _wrapper(wrapper_type, outputs)(data, emb)
    torch.testing.assert_close(result["a_energy"], torch.tensor([1.0, 0.0, 3.0]))
    torch.testing.assert_close(
        result["a_forces"], torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 6.0])
    )
    assert not result["absent_energy"].any()
    assert not result["absent_forces"].any()
    assert torch.isnan(result["b_energy"][1])
    assert torch.isinf(result["b_forces"][3])


@pytest.mark.parametrize(
    "wrapper_type", [DatasetSpecificMoEWrapper, DatasetSpecificSingleHeadWrapper]
)
@pytest.mark.parametrize("device", ["cpu", "xpu:0"])
def test_dataset_routing_avoids_cpu_transfer_and_membership_search(
    wrapper_type, device, monkeypatch
):
    # FakeTensor verifies dispatch and shapes, not Intel numerical kernels.
    with FakeTensorMode(allow_fallback_kernels=False):
        data, emb = _batch(device)
        outputs = {
            "energy": torch.empty(3, device=device),
            "forces": torch.empty((6, 3), device=device),
            "atom_scalar": torch.empty(6, device=device),
        }
        wrapper = _wrapper(wrapper_type, outputs)

        def forbidden(*args, **kwargs):
            pytest.fail("Dataset routing must not transfer atoms to CPU or use isin")

        monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
        monkeypatch.setattr(torch, "isin", forbidden)
        result = wrapper(data, emb)
        for dataset in wrapper.dataset_names:
            for name, value in outputs.items():
                assert result[f"{dataset}_{name}"].shape == value.shape
                assert result[f"{dataset}_{name}"].device == value.device


@pytest.mark.parametrize(
    "wrapper_type", [DatasetSpecificMoEWrapper, DatasetSpecificSingleHeadWrapper]
)
def test_merged_head_does_not_read_atom_routing(wrapper_type):
    class DataWithoutAtomRouting:
        dataset = ("a",)

        @property
        def batch_full(self):
            pytest.fail("Merged head does not need atom-to-system routing")

    outputs = {"energy": torch.tensor([1.0]), "forces": torch.ones(3, 3)}
    wrapper = _wrapper(wrapper_type, outputs)
    wrapper.merge_MOLE_model(SimpleNamespace(dataset=["a"], pos=torch.ones(3, 3)))
    result = wrapper(DataWithoutAtomRouting(), {})
    for name, value in outputs.items():
        torch.testing.assert_close(result[f"a_{name}"], value)
        assert torch.isnan(result[f"b_{name}"]).all()


@pytest.mark.parametrize("constructor_dtype", [torch.float32, torch.float64])
def test_backbone_merge_preserves_mixed_and_nonpersistent_buffer_precision(
    constructor_dtype,
):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(constructor_dtype)
        model = eSCNMDMoeBackbone(
            sphere_channels=4,
            hidden_channels=8,
            edge_channels=4,
            num_distance_basis=4,
            num_layers=1,
            num_experts=2,
            use_dataset_embedding=False,
            use_composition_embedding=True,
            lmax=2,
            mmax=2,
        ).eval()
    finally:
        torch.set_default_dtype(original_dtype)
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol")
    merged = model.merge_MOLE_model(data)
    # The source has already replaced experts by linear layers. Compare its
    # exact tensors with the plain replacement, including derived buffers.
    source = dict(model.named_parameters()) | dict(model.named_buffers())
    target = dict(merged.named_parameters()) | dict(merged.named_buffers())
    assert source.keys() == target.keys()
    assert any(tensor.dtype == torch.float64 for tensor in target.values())
    for name, value in source.items():
        torch.testing.assert_close(target[name], value, rtol=0, atol=0)
        assert target[name].device == value.device
    assert torch.get_default_dtype() == original_dtype


def test_full_uma_merge_preserves_loaded_fp64_weights_and_predictions():
    torch.manual_seed(42)
    model = HydraModel(
        backbone={
            "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
            "sphere_channels": 8,
            "hidden_channels": 16,
            "edge_channels": 8,
            "num_distance_basis": 8,
            "num_layers": 1,
            "num_experts": 2,
            "lmax": 2,
            "mmax": 2,
            "otf_graph": True,
            "cutoff": 3.0,
            "use_dataset_embedding": True,
            "dataset_mapping": {"omol": "omol", "omat": "omat"},
            "use_composition_embedding": True,
            "always_use_pbc": False,
            "direct_forces": False,
            "regress_forces": True,
            "regress_stress": True,
            "model_version": 1.2,
            "execution_mode": "general",
        },
        heads={
            "efs": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "head_kwargs": {"wrap_property": False},
                "dataset_mapping": {"omol": "omol", "omat": "omat"},
            }
        },
        pass_through_head_outputs=True,
        model_id="UMA-S-1.2.1",
    ).double()
    model._tasks = {}
    model.eval().requires_grad_(False)
    # Values beyond FP32 resolution expose an unintended downcast during merge.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1e-9)
    candidate = copy.deepcopy(model)
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol")
    for name, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            data[name] = value.double()
    expected = model(data.clone())

    default_dtype = torch.get_default_dtype()
    candidate.prepare_for_inference(data.clone(), inference_settings_xpu())
    assert torch.get_default_dtype() == default_dtype
    assert all(
        tensor.dtype == torch.float64
        for tensor in candidate.state_dict().values()
        if tensor.is_floating_point()
    )
    actual = candidate(data.clone())
    for name in ("energy", "forces", "stress"):
        torch.testing.assert_close(
            actual[f"omol_{name}"][name],
            expected[f"omol_{name}"][name],
            rtol=1e-10,
            atol=1e-12,
        )

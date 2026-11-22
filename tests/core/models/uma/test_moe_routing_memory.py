"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from dataclasses import replace
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
from fairchem.core.models.uma.nn.mole import MOLE, MOLEGlobals, _pnorm
from fairchem.core.units.mlip_unit.api.inference import (
    MergeMoleConsistencyError,
    inference_settings_xpu,
)
from fairchem.core.units.mlip_unit.inference_runtime import model_storage_bytes
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from tests.core.models.uma.uma_nitro.nitro_test_utils import random_checkpoint


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
    original = {
        name: value.detach().clone()
        for name, value in (
            dict(model.named_parameters()) | dict(model.named_buffers())
        ).items()
    }
    gaussian_coeff = model.distance_expansion.coeff
    merged = model.merge_MOLE_model(data)
    target = dict(merged.named_parameters()) | dict(merged.named_buffers())
    assert not any(isinstance(module, MOLE) for module in merged.modules())
    assert any(tensor.dtype == torch.float64 for tensor in target.values())
    assert merged.distance_expansion.coeff == gaussian_coeff
    # The unmixed parameters and all derived buffers retain their exact
    # pre-merge state. Expert mixing creates new linear weight names.
    for name in original.keys() & target.keys():
        torch.testing.assert_close(target[name], original[name], rtol=0, atol=0)
        assert target[name].device == original[name].device
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


@pytest.mark.parametrize("cache_size", [0, 2])
def test_portable_checkpoint_energy_then_forces_with_lazy_weights(tmp_path, cache_size):
    path = tmp_path / "small.pt"
    torch.save(
        random_checkpoint(
            num_experts=2,
            num_layers=2,
            sphere_channels=8,
            hidden_channels=16,
            edge_channels=8,
            num_distance_basis=8,
            otf_graph=True,
        ),
        path,
    )
    settings = replace(
        inference_settings_xpu(),
        execution_mode="umas_fast_portable",
        mole_cache_size=cache_size,
    )
    unit = MLIPPredictUnit(
        str(path),
        device="cpu",
        inference_settings=settings,
        atom_refs={},
        form_elem_refs={},
    )
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol")
    rng_before = torch.get_rng_state()
    unit._initialize(data)
    torch.testing.assert_close(torch.get_rng_state(), rng_before)
    with torch.inference_mode():
        energy = unit.predict(data, requested_properties=["energy"])
    assert torch.isfinite(energy["energy"]).all()
    forces = unit.predict(data, requested_properties=["forces"])
    assert forces["forces"].isfinite().all()
    if not cache_size:
        compact = MLIPPredictUnit(
            str(path),
            device="cpu",
            inference_settings=replace(
                settings, compact_angular_weights=True, radial_layer_group_size=1
            ),
            atom_refs={},
            form_elem_refs={},
        )
        torch.manual_seed(45)
        packed = unit.predict(data, requested_properties=["energy", "forces"])
        torch.manual_seed(45)
        streamed = compact.predict(data, requested_properties=["energy", "forces"])
        for prop in packed:
            torch.testing.assert_close(
                streamed[prop], packed[prop], atol=1e-6, rtol=1e-5
            )
        assert model_storage_bytes(compact.model) < model_storage_bytes(unit.model)
    if cache_size:
        assert unit._prepared_models.bytes == model_storage_bytes(unit.model)
        # Physical nonpersistent buffers must continue to trigger mutation checks.
        unit.model.module.backbone.coefficient_index.add_(0)
        with pytest.raises(RuntimeError, match="weights/buffers changed"):
            unit.predict(data, requested_properties=["energy"])


def test_merge_requires_exact_element_ratio_and_pnorm_system_count():
    def batch(counts):
        numbers = torch.cat(
            [
                torch.repeat_interleave(torch.tensor([1, 6]), torch.tensor(pair))
                for pair in counts
            ]
        )
        sizes = torch.tensor([sum(pair) for pair in counts])
        return SimpleNamespace(
            natoms=sizes,
            batch=torch.repeat_interleave(torch.arange(len(sizes)), sizes),
            atomic_numbers=numbers,
            charge=sizes * 0,
            spin=sizes * 0 + 1,
            dataset=["omol"] * len(sizes),
        )

    model = eSCNMDMoeBackbone(
        use_dataset_embedding=False,
        num_layers=1,
        sphere_channels=4,
        hidden_channels=8,
        num_experts=2,
        lmax=2,
        mmax=2,
    )
    first = (377, 233)
    second = (610, 377)
    model.model_id = "UMA-S-1.2.1"
    model._merged_composition = model._get_merged_mole_consistency_info(batch([first]))
    with pytest.raises(MergeMoleConsistencyError, match="Compositions differ"):
        model._assert_merged_mole_consistency(
            model._get_merged_mole_consistency_info(batch([second]))
        )
    with pytest.raises(MergeMoleConsistencyError, match="same reduced composition"):
        model._assert_all_mole_info_consistent(batch([first, second]))
    model.mole_expert_coefficient_norm = _pnorm
    with pytest.raises(MergeMoleConsistencyError, match="Batch size"):
        model._assert_merged_mole_consistency(
            model._get_merged_mole_consistency_info(batch([first, first]))
        )


def test_grouped_mole_chunk_preserves_routing_gradients():
    sizes = torch.tensor([4, 2, 0, 5, 3])
    coefficients = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    state = MOLEGlobals(coefficients, sizes, ac_start_idx=3)
    layer = MOLE(3, 4, 5, state, bias=True).double()
    x = torch.randn(8, 2, 4, dtype=torch.float64, requires_grad=True)
    with torch.inference_mode():
        layer(x)
    ids = torch.repeat_interleave(torch.arange(5), sizes)[3:11]
    weights = torch.einsum("be,eoi->boi", coefficients, layer.weights)
    expected = torch.einsum("bki,boi->bko", x, weights[ids]) + layer.bias
    actual = layer(x)
    _, plan = state.row_layout(3, len(x), x.device)
    assert plan is not None
    torch.testing.assert_close(actual, expected)
    variables = (x, coefficients, layer.weights, layer.bias)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), variables),
        torch.autograd.grad(expected.square().sum(), variables),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    sizes[0] += 1
    assert state.row_layout(3, len(x), x.device)[1] is not plan


def test_mole_chunk_invalidates_cpu_counts_mutated_through_numpy():
    sizes = torch.tensor([3, 0, 2, 2])
    routing = torch.eye(4)
    state = MOLEGlobals(routing, sizes, ac_start_idx=2)
    layer = MOLE(4, 1, 1, state, bias=False)
    with torch.no_grad():
        layer.weights[:, 0, 0].copy_(torch.arange(1, 5))
    x = torch.ones(4, 1)
    assert layer(x).flatten().tolist() == [1, 3, 3, 4]
    sizes.numpy()[:] = [2, 0, 4, 1]
    assert layer(x).flatten().tolist() == [3, 3, 3, 3]

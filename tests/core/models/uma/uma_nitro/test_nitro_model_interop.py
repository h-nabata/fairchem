"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import pytest
import torch
from ase.build import molecule

from fairchem.core.units.mlip_unit.api.inference import (
    MergeMoleConsistencyError,
    UMANitroNotSupportedError,
)
from tests.core.models.uma.uma_nitro.nitro_test_utils import batch, nitro_settings

pytest.importorskip("triton")

from fairchem.core.models.uma.uma_nitro import _inference as nitro  # noqa: E402


def _water():
    return batch((molecule("H2O"), "omol"))


def test_requested_edge_chunking_requires_backend_fallback():
    nitro._validate_settings(nitro_settings())
    with pytest.raises(UMANitroNotSupportedError, match="edge_chunk_size"):
        nitro._validate_settings(nitro_settings(edge_chunk_size=1024))


def test_model_owns_retained_checkpoint_experts(nitro_random_checkpoint):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    key = next(iter(model._experts))
    checkpoint_tensor = nitro_random_checkpoint.ema_state_dict[f"module.{key}"]
    expected = checkpoint_tensor.clone()
    try:
        checkpoint_tensor.add_(5)
        torch.testing.assert_close(model._experts[key], expected, rtol=0, atol=0)
        assert model._experts[key].data_ptr() != checkpoint_tensor.data_ptr()
    finally:
        checkpoint_tensor.copy_(expected)


def test_standalone_construction_is_cpu_fp32_frozen_and_context_independent(
    nitro_random_checkpoint,
):
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"), torch.inference_mode():
            model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
            assert torch.empty(0).device.type == "meta"
            assert torch.is_inference_mode_enabled()
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(original_dtype)
    assert not model.training
    for value in (*model.parameters(), *model.buffers(), *model._experts.values()):
        assert value.device.type == "cpu"
        if value.is_floating_point():
            expected_dtype = (
                torch.float64 if value is model.backbone.wigner_c_l2 else torch.float32
            )
            assert value.dtype == expected_dtype
        assert not value.requires_grad
        assert not value.is_inference()
    with torch.device("meta"):
        model.prepare_for_inference(_water(), nitro_settings())
    assert not model._experts


@pytest.mark.parametrize(
    "copy_model", [copy.deepcopy, lambda model: pickle.loads(pickle.dumps(model))]
)
def test_serialization_omits_runtime_caches_without_mutating_the_source(
    nitro_random_checkpoint,
    copy_model,
):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    model.prepare_for_inference(_water(), nitro_settings())
    leaf = torch.ones(3, requires_grad=True)
    transient = leaf * 2
    model._uma_nitro_graph_token = 711
    model.backbone._uma_nitro_merged_radial = (transient,)
    linear = model.head.energy_block[0]
    linear._uma_nitro_wt = transient
    norm = model.backbone.norm
    norm._uma_nitro_wexp = transient
    so3 = model.backbone.blocks[0].atom_wise.so3_linear_1
    so3._uma_nitro_bias9 = transient
    so2 = model.backbone.blocks[0].edge_wise.so2_conv_1.so2_m_conv[0]
    so2._build_w_block()
    source_block = so2._w_block
    clone = copy_model(model)
    assert model._uma_nitro_graph_token == 711
    assert linear._uma_nitro_wt is transient
    assert so2._w_block is source_block
    assert not hasattr(clone, "_uma_nitro_graph_token")
    for module in clone.modules():
        assert not any(name.startswith("_uma_nitro_") for name in vars(module))
        if isinstance(module, nitro._SO2mConv):
            assert module._w_block is None
    for original, copied in zip(model.parameters(), clone.parameters(), strict=True):
        torch.testing.assert_close(original, copied, atol=0, rtol=0)
        assert original.data_ptr() != copied.data_ptr()
    torch.testing.assert_close(
        clone.backbone._unified_radial_mlp.W1_cat,
        model.backbone._unified_radial_mlp.W1_cat,
        atol=0,
        rtol=0,
    )
    clone.on_predict_check(_water())
    transient.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))


def test_dtype_move_and_weight_load_discard_derived_layouts(
    nitro_random_checkpoint,
    monkeypatch,
):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    releases = []
    monkeypatch.setattr(
        nitro, "_release_runtime", lambda instance: releases.append(instance)
    )
    linear = model.head.energy_block[0]
    so2 = model.backbone.blocks[0].edge_wise.so2_conv_1.so2_m_conv[0]
    for operation in (
        lambda: model.to(dtype=torch.float64),
        lambda: model.load_state_dict(model.state_dict()),
    ):
        model._uma_nitro_graph_token = 712
        linear._uma_nitro_wt = linear.weight.t().contiguous()
        so2._build_w_block()
        operation()
        assert not hasattr(model, "_uma_nitro_graph_token")
        assert not hasattr(linear, "_uma_nitro_wt")
        assert so2._w_block is None
    assert releases == [model, model]


def test_repeated_preparation_checks_the_original_merged_contract(
    nitro_random_checkpoint,
):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    data = _water()
    model.prepare_for_inference(data, nitro_settings())
    assert model.prepare_for_inference(data.clone(), nitro_settings()) is model
    with pytest.raises(MergeMoleConsistencyError, match="Compositions"):
        model.prepare_for_inference(batch((molecule("CO2"), "omol")), nitro_settings())


def test_direct_preparation_preserves_weights_under_ambient_autocast(
    nitro_random_checkpoint,
):
    reference = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    candidate = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    data = _water()
    reference.prepare_for_inference(data, nitro_settings())
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        candidate.prepare_for_inference(data, nitro_settings())
        assert torch.is_inference_mode_enabled()
        assert torch.is_autocast_enabled("cpu")
    for expected, actual in zip(
        reference.parameters(), candidate.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert not actual.is_inference()


def test_md_metadata_snapshot_avoids_recount_and_detects_numpy_writes(
    nitro_random_checkpoint,
    monkeypatch,
):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    data = _water()
    model.prepare_for_inference(data, nitro_settings())
    calls = []
    original = model._merge_key

    def count(data):
        calls.append(True)
        return original(data)

    monkeypatch.setattr(model, "_merge_key", count)
    for _ in range(30):
        current = data.clone()
        current.pos.add_(0.1)
        model.on_predict_check(current)
    assert not calls
    version = data.atomic_numbers._version
    data.atomic_numbers.numpy()[0] = 6
    assert data.atomic_numbers._version == version
    with pytest.raises(MergeMoleConsistencyError, match="Compositions"):
        model.on_predict_check(data)
    assert calls == [True]


def test_composition_comparison_does_not_accept_close_but_different_ratios():
    model = nitro.UMANitroModel.__new__(nitro.UMANitroModel)
    torch.nn.Module.__init__(model)
    data = _water()
    key = model._merge_key(data)
    model._merged_key = (key[0].clone(), *key[1:])
    model._merged_key[0][1] += 1e-7
    with pytest.raises(MergeMoleConsistencyError, match="Compositions"):
        model.on_predict_check(data)


@pytest.mark.parametrize(
    "field,value",
    [("charge", 0.5), ("spin", float("nan")), ("charge", 101), ("spin", -1)],
)
def test_merge_metadata_rejects_lossy_integer_conversion(field, value):
    model = nitro.UMANitroModel.__new__(nitro.UMANitroModel)
    torch.nn.Module.__init__(model)
    data = _water()
    data[field] = torch.tensor([value])
    with pytest.raises(ValueError, match=field):
        model._merge_key(data)


@pytest.mark.parametrize("method", ["add_tasks", "setup_tasks"])
def test_rejected_tasks_leave_both_registries_unchanged(
    nitro_random_checkpoint,
    monkeypatch,
    method,
):
    import hydra

    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    before_tasks = dict(model.tasks)
    before_map = dict(model.dataset_to_tasks)
    valid = SimpleNamespace(name="extra", property="energy", datasets=["omol"])
    invalid = SimpleNamespace(name="bad", property="hessian", datasets=["omol"])
    if method == "setup_tasks":
        monkeypatch.setattr(hydra.utils, "instantiate", lambda value: value)
    with pytest.raises(ValueError, match="energy, forces and stress only"):
        getattr(model, method)([valid, invalid])
    assert model.tasks == before_tasks
    assert model.dataset_to_tasks == before_map


@pytest.mark.parametrize(
    "property_name", ["hessian", "hessian_vector_product", "embeddings"]
)
def test_direct_forward_rejects_unsupported_properties_before_running_kernels(
    nitro_random_checkpoint,
    monkeypatch,
    property_name,
):
    model = nitro.UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    data = _water()
    data["_requested_properties"] = (property_name,)
    monkeypatch.setattr(
        nitro, "_nitro_forward", lambda *args: pytest.fail("kernel ran")
    )
    with pytest.raises(
        UMANitroNotSupportedError, match="energy, forces and stress only"
    ):
        model(data)

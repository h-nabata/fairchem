"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import hydra
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    inference_settings_xpu,
)
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from fairchem.core.units.mlip_unit.series_b_runtime import validate_series_b
from fairchem.core.units.mlip_unit.utils import tf32_context_manager
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


@pytest.mark.parametrize(
    "options",
    [
        {"gemm_precision": "float8"},
        {"weight_bits": True},
        {"weight_bits": 4.0},
        {"low_rank_fraction": float("nan")},
        {"low_rank_fraction": 0},
        {"weight_bits": 8, "low_rank_fraction": 0.5},
        {"saved_feature_dtype": "float64"},
        {"saved_feature_min_elements": None},
        {"radial_table_points": 1.5},
        {"radial_table_points": True},
        {"radial_fast_exp": True, "radial_table_points": 257},
        {"onednn_tf32": "false"},
        {"low_precision_emulation": True},
    ],
)
def test_series_b_rejects_ambiguous_configuration(options):
    with pytest.raises((ValueError, TypeError)):
        SeriesBSettings(**options)


def test_series_b_defaults_serialization_and_server_identity_are_isolated():
    assert not InferenceSettings().series_b.enabled
    options = {"weight_bits": 8}
    settings = replace(inference_settings_xpu(), series_b=options)
    options["weight_bits"] = 4
    assert settings.series_b.weight_bits == 8
    restored = hydra.utils.instantiate(OmegaConf.create(settings.to_omegaconf()))
    assert restored == settings
    reference = ModelSpec("local.pt", source="path", inference_settings="xpu")
    candidate = ModelSpec("local.pt", source="path", inference_settings=settings)
    assert reference.model_id != candidate.model_id
    assert (
        candidate.model_id
        == ModelSpec("local.pt", source="path", inference_settings=restored).model_id
    )
    settings.series_b = SeriesBSettings(weight_bits=4)
    assert candidate.loader_settings().series_b.weight_bits == 8
    assert asdict(SeriesBSettings()) == asdict(InferenceSettings().series_b)


@pytest.mark.parametrize(
    "changes",
    [
        {"compile": True},
        {"xpu_graph": True},
        {"nitro": True},
        {"execution_mode": "umas_fast_portable"},
        {"activation_checkpointing": True},
        {"base_precision_dtype": "float64"},
        {"tf32": True},
    ],
)
def test_series_b_rejects_incompatible_runtime(changes):
    with pytest.raises(ValueError):
        replace(inference_settings_xpu(), series_b={"weight_bits": 8}, **changes)


def test_device_precision_modes_do_not_silently_fall_back():
    settings = replace(inference_settings_xpu(), series_b={"gemm_precision": "bf16"})
    with pytest.raises(ValueError, match="emulation"):
        validate_series_b(settings, "cpu")
    settings.series_b = replace(settings.series_b, low_precision_emulation=True)
    with pytest.raises(ValueError, match="CPU"):
        validate_series_b(settings, "xpu")
    settings.series_b = SeriesBSettings(onednn_tf32=True)
    with pytest.raises(ValueError, match="XPU"):
        validate_series_b(settings, "cpu")


def test_xpu_onednn_context_restores_after_error_without_touching_cuda(monkeypatch):
    monkeypatch.setattr(torch.backends, "mkldnn", SimpleNamespace(allow_tf32=False))
    precision = torch.get_float32_matmul_precision()
    cudnn = torch.backends.cudnn.allow_tf32
    with pytest.raises(RuntimeError, match="deliberate"), tf32_context_manager(
        True, device_type="xpu"
    ):
        assert torch.backends.mkldnn.allow_tf32
        with pytest.raises(ValueError, match="Strict FP32"), tf32_context_manager(
            False, device_type="xpu"
        ):
            pass
        raise RuntimeError("deliberate")
    assert not torch.backends.mkldnn.allow_tf32
    assert precision == torch.get_float32_matmul_precision()
    assert cudnn == torch.backends.cudnn.allow_tf32


def test_series_b_numpy_scalars_survive_json_and_yaml():
    options = SeriesBSettings(
        weight_bits=np.int64(8),
        radial_table_points=np.int64(257),
        saved_feature_min_elements=np.int64(32),
    )
    settings = replace(inference_settings_xpu(), series_b=options)
    spec = ModelSpec("local.pt", source="path", inference_settings=settings)
    assert isinstance(spec.model_id, str)
    assert (
        hydra.utils.instantiate(OmegaConf.create(settings.to_omegaconf())) == settings
    )
    assert (
        type(SeriesBSettings(low_rank_fraction=np.float32(0.25)).low_rank_fraction)
        is float
    )


@pytest.mark.parametrize("initial", [SeriesBSettings(), SeriesBSettings(weight_bits=8)])
@pytest.mark.parametrize("cache_size", [0, 2])
def test_prepared_approximation_policy_cannot_be_relabelled(initial, cache_size):
    unit = _public_predictor(cache_size)
    unit.inference_settings.series_b = initial
    unit.predict(_data(), requested_properties=("energy",))
    unit.inference_settings.series_b = (
        SeriesBSettings() if initial.enabled else SeriesBSettings(weight_bits=8)
    )
    with pytest.raises(ValueError, match="construct a new predictor"):
        unit.predict(_data(), requested_properties=("energy",))


def test_effective_checkpoint_override_is_rejected():
    unit = _public_predictor()
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    backbone = unit.model.module.backbone
    module = next(
        m for m in backbone.modules() if hasattr(m, "activation_checkpoint_chunk_size")
    )
    module.activation_checkpoint_chunk_size = 8
    with pytest.raises(ValueError, match="overridden activation"):
        validate_series_b(unit.inference_settings, "cpu", backbone)


def test_default_predictor_masks_outer_approximate_reduction_context(monkeypatch):
    from fairchem.core.models.uma.nn import approximate_ops

    unit = _public_predictor()

    def forbidden(*args, **kwargs):
        pytest.fail("Strict predictor inherited another predictor's reduction setting")

    monkeypatch.setattr(approximate_ops, "compensated_node_sum", forbidden)
    with approximate_ops.approximate_ops_context(
        SeriesBSettings(compensated_reduction=True)
    ):
        unit.predict(_data(), requested_properties=("energy", "forces"))
        assert approximate_ops.compensated_reduction_enabled()
    assert not approximate_ops.compensated_reduction_enabled()


@pytest.mark.parametrize(
    "options",
    [
        {"gemm_precision": "bf16", "low_precision_emulation": True},
        {"gemm_precision": "fp16", "low_precision_emulation": True},
        {"gemm_precision": "bf16_split2", "low_precision_emulation": True},
        {"gemm_precision": "bf16_split3", "low_precision_emulation": True},
        {"weight_bits": 8},
        {"weight_bits": 4},
        {"low_rank_fraction": 0.25},
        {"compensated_reduction": True},
        {"radial_fast_exp": True},
        {"radial_table_points": 257},
        {"saved_feature_dtype": "bf16", "saved_feature_min_elements": 1},
        {"saved_feature_dtype": "fp16", "saved_feature_min_elements": 1},
        {
            "gemm_precision": "bf16",
            "low_precision_emulation": True,
            "weight_bits": 8,
            "saved_feature_dtype": "bf16",
            "saved_feature_min_elements": 1,
            "radial_table_points": 257,
            "compensated_reduction": True,
        },
        {
            "low_rank_fraction": 0.25,
            "radial_fast_exp": True,
            "saved_feature_dtype": "fp16",
            "saved_feature_min_elements": 1,
        },
    ],
)
def test_real_uma_optional_precision_energy_force_hessian_and_job_isolation(options):
    reference = _public_predictor()
    candidate = copy.deepcopy(reference)
    candidate.inference_settings.series_b = SeriesBSettings(**options)
    candidate.inference_settings.__post_init__()
    data = _data()
    data.cell = torch.eye(3).unsqueeze(0) * 8
    original = data.clone()
    request = ("energy", "forces", "stress", "hessian")
    expected = reference.predict(data, requested_properties=request)
    actual = candidate.predict(data, requested_properties=request)
    assert set(expected) == set(actual) == set(request)
    for name in request:
        assert actual[name].shape == expected[name].shape
        assert torch.isfinite(actual[name]).all()
    # Test repeated job ownership and restored default behavior, rather than
    # pretending one tolerance qualifies every approximate model for science.
    saved = {key: value.clone() for key, value in actual.items()}
    changed = data.clone()
    changed.pos[0, 0] += 0.02
    candidate.predict(changed, requested_properties=request)
    for name in request:
        torch.testing.assert_close(actual[name], saved[name], atol=0, rtol=0)
    again = reference.predict(data, requested_properties=request)
    for name in request:
        torch.testing.assert_close(again[name], expected[name], atol=2e-8, rtol=2e-5)
    torch.testing.assert_close(data.pos, original.pos, atol=0, rtol=0)
    assert candidate.approximation_manifest["enabled"]
    assert reference.approximation_manifest == {"enabled": False}
    owned = candidate.approximation_manifest
    owned.clear()
    assert candidate.approximation_manifest["enabled"]

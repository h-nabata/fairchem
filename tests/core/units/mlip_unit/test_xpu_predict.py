"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

These CPU tests simulate XPU discovery and exercise device/configuration dispatch.
They do not execute Intel GPU kernels or establish numerical XPU correctness.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.units import mlip_unit
from fairchem.core.units.mlip_unit import MLIPPredictUnit, load_predict_unit
from fairchem.core.units.mlip_unit import predict as predict_module
from fairchem.core.units.mlip_unit.api.inference import (
    NAME_TO_INFERENCE_SETTING,
    InferenceSettings,
    describe_inference_settings,
    inference_settings_xpu,
)
from fairchem.core.units.mlip_unit.predict import _ModelWrapper


@pytest.fixture()
def mock_xpu(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)


@pytest.mark.parametrize("device", ["xpu", "xpu:1"])
def test_xpu_device_preserves_index(device, mock_xpu):
    predictor = SimpleNamespace()
    MLIPPredictUnit._setup_device(predictor, device)
    assert predictor.device == "xpu:1"


def test_unavailable_xpu_fails_before_checkpoint_load(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    checkpoint_load = Mock(side_effect=AssertionError("checkpoint must not be loaded"))
    monkeypatch.setattr(torch, "load", checkpoint_load)
    with pytest.raises(RuntimeError, match="XPU was requested"):
        MLIPPredictUnit("does-not-exist.pt", device="xpu")
    checkpoint_load.assert_not_called()


def test_invalid_xpu_index_is_rejected(mock_xpu):
    with pytest.raises(ValueError, match="Requested xpu:2"):
        MLIPPredictUnit._setup_device(SimpleNamespace(), "xpu:2")


@pytest.mark.parametrize("device", ["xpu", "xpu:1"])
def test_loader_uses_xpu_preset_without_changing_cuda_default(device, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(mlip_unit, "MLIPPredictUnit", constructor)
    before = copy.deepcopy(NAME_TO_INFERENCE_SETTING["default"])
    load_predict_unit("unused.pt", device=device)
    settings = constructor.call_args.kwargs["inference_settings"]
    assert settings == inference_settings_xpu()
    assert NAME_TO_INFERENCE_SETTING["default"] == before
    assert constructor.call_args.kwargs["device"] == device


@pytest.mark.parametrize(
    "cuda_available,xpu_available,expected",
    [(True, True, "cuda"), (False, True, "xpu"), (False, False, "cpu")],
)
def test_automatic_device_priority(
    cuda_available, xpu_available, expected, monkeypatch
):
    constructor = Mock()
    monkeypatch.setattr(mlip_unit, "MLIPPredictUnit", constructor)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu_available)
    load_predict_unit("unused.pt")
    assert constructor.call_args.kwargs["device"] == expected


@pytest.mark.parametrize("workers,gp_config", [(2, None), (1, object())])
def test_xpu_parallel_configuration_is_rejected_before_construction(
    workers, gp_config, monkeypatch
):
    constructor = Mock()
    monkeypatch.setattr(mlip_unit, "MLIPPredictUnit", constructor)
    with pytest.raises(ValueError, match="workers=1 without graph parallelism"):
        load_predict_unit(
            "unused.pt", device="xpu:1", workers=workers, gp_config=gp_config
        )
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"nitro": True}, "require CUDA"),
        ({"execution_mode": "umas_fast_gpu"}, "require CUDA"),
        ({"internal_graph_gen_version": 3}, "NVIDIA-specific"),
        ({"tf32": True}, "tf32=False"),
    ],
)
def test_xpu_rejects_unsupported_modes(kwargs, error):
    predictor = SimpleNamespace(inference_settings=InferenceSettings(**kwargs))
    with pytest.raises(ValueError, match=error):
        MLIPPredictUnit._validate_xpu_settings(predictor)


def test_xpu_compile_is_explicit_opt_in():
    predictor = SimpleNamespace(inference_settings=InferenceSettings(compile=True))
    MLIPPredictUnit._validate_xpu_settings(predictor)
    assert predictor.inference_settings.compile
    assert predictor.inference_settings.execution_mode == "general"


def test_xpu_fp64_checks_selected_device_before_loading_model(mock_xpu, monkeypatch):
    def failed_probe(device):
        assert device == "xpu:1"
        raise RuntimeError("FP64 execution probe failed")

    monkeypatch.setattr(predict_module, "probe_fp64_support", failed_probe)
    loader = Mock(side_effect=AssertionError("checkpoint must not be loaded"))
    monkeypatch.setattr(MLIPPredictUnit, "_load_model", loader)
    with pytest.raises(RuntimeError, match="FP64 execution probe"):
        MLIPPredictUnit(
            "unused.pt",
            device="xpu:1",
            inference_settings=InferenceSettings(base_precision_dtype=torch.float64),
        )
    loader.assert_not_called()


def test_xpu_fp64_keeps_probe_evidence_and_fp64_policy(monkeypatch):
    evidence = {"passed": True, "device": "xpu:1", "dtype": "float64"}
    probe = Mock(return_value=evidence)
    monkeypatch.setattr(predict_module, "probe_fp64_support", probe)
    predictor = SimpleNamespace(
        device="xpu:1",
        inference_settings=InferenceSettings(base_precision_dtype=torch.float64),
    )
    MLIPPredictUnit._validate_xpu_settings(predictor)
    probe.assert_called_once_with("xpu:1")
    assert predictor.fp64_probe is evidence
    assert predictor.inference_settings.base_precision_dtype == torch.float64
    assert predictor.inference_settings.execution_mode == "general"


@pytest.mark.parametrize(
    "settings", ["fp64", "xpu-fp64", InferenceSettings(base_precision_dtype="float64")]
)
@pytest.mark.parametrize(
    "factory",
    [pretrained_mlip.get_predict_unit, FAIRChemCalculator.from_model_checkpoint],
)
def test_public_xpu_fp64_probe_fails_before_download(
    settings, factory, mock_xpu, monkeypatch
):
    probe = Mock(side_effect=RuntimeError("FP64 execution probe failed"))
    download = Mock(side_effect=AssertionError("checkpoint must not be downloaded"))
    monkeypatch.setattr(pretrained_mlip, "probe_fp64_support", probe)
    monkeypatch.setattr(pretrained_mlip, "resolve_checkpoint", download)
    with pytest.raises(RuntimeError, match="FP64 execution probe"):
        factory("uma-s-1p2p1", inference_settings=settings, device="xpu")
    probe.assert_called_once_with("xpu:1")
    download.assert_not_called()


@pytest.mark.parametrize("settings", ["default", "xpu", "fp64", "xpu-fp64"])
def test_public_xpu_fp64_preflight_order_and_fp32_unchanged(
    settings, mock_xpu, monkeypatch
):
    operations = Mock()
    operations.download.return_value = pretrained_mlip.ResolvedCheckpoint("local.pt")
    monkeypatch.setattr(pretrained_mlip, "probe_fp64_support", operations.probe)
    monkeypatch.setattr(pretrained_mlip, "resolve_checkpoint", operations.download)
    monkeypatch.setattr(pretrained_mlip, "load_predict_unit", operations.load)
    result = pretrained_mlip.get_predict_unit(
        "uma-s-1p2p1", inference_settings=settings, device="xpu"
    )
    expected = (
        ["probe", "download", "load"] if "fp64" in settings else ["download", "load"]
    )
    assert [call[0] for call in operations.mock_calls] == expected
    assert operations.load.call_args.args[1] == settings
    assert operations.load.call_args.args[3] == "xpu:1"
    assert result is operations.load.return_value


def test_public_xpu_preflight_validates_private_settings_copy(mock_xpu, monkeypatch):
    settings = InferenceSettings()
    settings.base_precision_dtype = "float64"
    before = copy.deepcopy(settings)
    probe = Mock(side_effect=RuntimeError("FP64 execution probe failed"))
    monkeypatch.setattr(pretrained_mlip, "probe_fp64_support", probe)
    with pytest.raises(RuntimeError, match="FP64 execution probe"):
        pretrained_mlip.get_predict_unit(
            "uma-s-1p2p1", inference_settings=settings, device="xpu"
        )
    assert settings == before

    settings.base_precision_dtype = "not-a-dtype"
    with pytest.raises(ValueError, match="base_precision_dtype"):
        pretrained_mlip.get_predict_unit(
            "uma-s-1p2p1", inference_settings=settings, device="xpu"
        )


def test_direct_constructor_has_safe_xpu_defaults(mock_xpu, monkeypatch):
    def load_fake_model(predictor):
        model = torch.nn.Module()
        model.backbone = SimpleNamespace(
            regress_config=SimpleNamespace(direct_forces=False)
        )
        predictor.model = _ModelWrapper(model)

    monkeypatch.setattr(MLIPPredictUnit, "_load_model", load_fake_model)
    monkeypatch.setattr(MLIPPredictUnit, "set_seed", lambda self, seed: None)
    predictor = MLIPPredictUnit("unused.pt", device="xpu:1")
    assert predictor.device == "xpu:1"
    assert predictor.inference_settings.execution_mode == "general"
    assert predictor.inference_settings.base_precision_dtype == torch.float32
    assert not predictor.inference_settings.compile
    assert not predictor.inference_settings.nitro


def test_xpu_preset_reports_its_name():
    assert describe_inference_settings(inference_settings_xpu()) == "xpu"


def test_xpu_inference_disables_and_restores_callers_autocast(monkeypatch):
    precision = Mock(wraps=predict_module.tf32_context_manager)
    monkeypatch.setattr(predict_module, "tf32_context_manager", precision)
    cuda_precision = Mock(side_effect=AssertionError("XPU must not touch CUDA TF32"))
    monkeypatch.setattr(torch, "set_float32_matmul_precision", cuda_precision)

    class AutocastCheckingModel:
        module = SimpleNamespace(
            backbone=SimpleNamespace(regress_config=SimpleNamespace(direct_forces=True))
        )

        def __call__(self, data):
            assert not torch.is_autocast_enabled("xpu")
            return {"energy": torch.tensor([1.0])}

    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    predictor.device = "xpu:0"
    predictor.inference_settings = inference_settings_xpu()
    predictor.model = AutocastCheckingModel()
    predictor._process_outputs = lambda data, output, undo_refs: output
    with torch.autocast("xpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("xpu")
        result = predictor._run_inference(None, False)
        assert torch.is_autocast_enabled("xpu")
    assert result["energy"].dtype == torch.float32
    precision.assert_called_once_with(False, device_type="xpu")
    cuda_precision.assert_not_called()


@pytest.mark.parametrize(
    "backbone_override,error",
    [
        ({"execution_mode": "umas_fast_gpu"}, "requires CUDA"),
        ({"radius_pbc_version": 3}, "radius_pbc_version=1 or 2"),
    ],
)
def test_user_overrides_cannot_enable_cuda_only_paths(
    backbone_override, error, monkeypatch
):
    # Use the actual UMA builder: it supplies general/v2 defaults that must not
    # overwrite user overrides before the unsupported-path validation runs.
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone

    monkeypatch.setattr(
        predict_module,
        "get_backbone_class_from_checkpoint",
        lambda checkpoint: eSCNMDBackbone,
    )
    predictor = SimpleNamespace(_requested_device="xpu:1")
    with pytest.raises(ValueError, match=error):
        MLIPPredictUnit._build_overrides_from_settings(
            predictor, None, {"backbone": backbone_override}, inference_settings_xpu()
        )


def test_backbone_user_overrides_win_without_mutating_caller(monkeypatch):
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone

    monkeypatch.setattr(
        predict_module,
        "get_backbone_class_from_checkpoint",
        lambda checkpoint: eSCNMDBackbone,
    )
    user = {"backbone": {"radius_pbc_version": 1, "edge_chunk_size": 64}}
    before = copy.deepcopy(user)
    predictor = SimpleNamespace(_requested_device="xpu:1")
    actual = MLIPPredictUnit._build_overrides_from_settings(
        predictor, None, user, inference_settings_xpu()
    )
    assert actual["backbone"]["radius_pbc_version"] == 1
    assert actual["backbone"]["edge_chunk_size"] == 64
    assert actual["backbone"]["execution_mode"] == "general"
    assert user == before


def test_xpu_derivatives_enable_grad_and_restore_callers_no_grad():
    class StressModel:
        module = SimpleNamespace(
            backbone=SimpleNamespace(
                regress_config=SimpleNamespace(
                    direct_forces=False, forces=False, stress=True
                )
            )
        )

        def __call__(self, cell):
            energy = cell.square().sum()
            return {"stress": torch.autograd.grad(energy, cell)[0]}

    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    predictor.device = "xpu:0"
    predictor.inference_settings = inference_settings_xpu()
    predictor.model = StressModel()
    predictor._process_outputs = lambda data, output, undo_refs: output
    cell = torch.eye(3).requires_grad_()
    with torch.no_grad():
        result = predictor._run_inference(cell, False)
        assert not torch.is_grad_enabled()
    torch.testing.assert_close(result["stress"], 2 * torch.eye(3))

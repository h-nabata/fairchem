"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Test wrapper dispatch without executing torch-sim or Intel GPU kernels.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fairchem.core.calculate import torchsim_interface


@pytest.fixture()
def predictor_loader(monkeypatch):
    # Only the constructor's routing is under test. The optional torch-sim
    # runtime and model download are not needed to reproduce the lost index.
    monkeypatch.setattr(torchsim_interface, "ts", object())
    monkeypatch.setattr(torchsim_interface, "ModelInterface", object)
    monkeypatch.setattr(torchsim_interface, "setup_imports", lambda: None)
    monkeypatch.setattr(torchsim_interface, "setup_logging", lambda: None)
    loader = Mock(
        side_effect=lambda *args, **kwargs: SimpleNamespace(
            device="xpu:1" if kwargs["device"] == "xpu" else kwargs["device"]
        )
    )
    monkeypatch.setattr(torchsim_interface.pretrained_mlip, "get_predict_unit", loader)
    return loader


@pytest.mark.parametrize("device", ["xpu:1", torch.device("xpu:1")])
@pytest.mark.parametrize("use_cache", [False, True])
def test_xpu_wrapper_retains_requested_gpu(
    device, use_cache, tmp_path, predictor_loader
):
    model = torchsim_interface.FairChemModel(
        "unused-checkpoint",
        device=device,
        model_cache_dir=tmp_path if use_cache else None,
    )
    assert model.device == torch.device("xpu:1")
    assert predictor_loader.call_args.kwargs["device"] == "xpu:1"
    assert predictor_loader.call_args.kwargs["inference_settings"] == "batch"


@pytest.mark.parametrize(
    "cuda_available,xpu_available,expected",
    [(True, True, "cuda"), (False, True, "xpu"), (False, False, "cpu")],
)
def test_wrapper_detects_xpu_without_cuda(
    cuda_available, xpu_available, expected, monkeypatch, predictor_loader
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu_available)
    model = torchsim_interface.FairChemModel("unused-checkpoint")
    assert model.device.type == expected
    assert predictor_loader.call_args.kwargs["device"] == expected


def test_explicit_cpu_stays_on_cpu(monkeypatch, predictor_loader):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    model = torchsim_interface.FairChemModel("unused-checkpoint", device="cpu")
    assert model.device == torch.device("cpu")
    assert predictor_loader.call_args.kwargs["device"] == "cpu"


@pytest.mark.parametrize("device", [None, "xpu"])
def test_wrapper_uses_predictors_resolved_xpu_ordinal(
    device, monkeypatch, predictor_loader
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    model = torchsim_interface.FairChemModel("unused-checkpoint", device=device)
    assert predictor_loader.call_args.kwargs["device"] == "xpu"
    assert model.device == torch.device("xpu:1")

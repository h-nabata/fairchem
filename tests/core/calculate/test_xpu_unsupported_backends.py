"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from ase.calculators.emt import EMT

from fairchem.core.calculate import dftd3_calculator
from fairchem.core.components import batch_server
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec


@pytest.mark.parametrize("device", ["xpu", "xpu:1", "mps"])
def test_d3_rejects_unsupported_device_before_import_or_download(device, monkeypatch):
    backend_import = Mock(side_effect=AssertionError("must not import GPU backend"))
    parameter_load = Mock(side_effect=AssertionError("must not download parameters"))
    monkeypatch.setattr(dftd3_calculator, "_import_nvalchemiops", backend_import)
    monkeypatch.setattr(dftd3_calculator, "_load_dftd3_parameters", parameter_load)
    with pytest.raises(ValueError, match="supports only CPU and CUDA"):
        dftd3_calculator.DFTD3Calculator(EMT(), "pbe", device=device)
    backend_import.assert_not_called()
    parameter_load.assert_not_called()


@pytest.mark.parametrize("device", ["xpu", "xpu:0", "xpu:1"])
def test_ray_model_spec_rejects_xpu_before_loading(device):
    with pytest.raises(ValueError, match="XPU model serving through Ray"):
        ModelSpec("unused-model", device=device).resolve_device()


@pytest.mark.skipif(not batch_server.ray_installed, reason="Ray is optional")
@pytest.mark.parametrize("num_gpus", [0, 1, 2])
def test_ray_server_rejects_xpu_before_starting_ray(num_gpus, monkeypatch):
    init_ray = Mock(side_effect=AssertionError("must not start Ray"))
    monkeypatch.setattr(batch_server, "_init_ray_and_serve", init_ray)
    with pytest.raises(ValueError, match="XPU model serving through Ray"):
        batch_server.setup_batch_predict_server(
            SimpleNamespace(device="xpu:1"), num_gpus=num_gpus
        )
    init_ray.assert_not_called()


@pytest.mark.skipif(not batch_server.ray_installed, reason="Ray is optional")
def test_existing_ray_server_cannot_be_replaced_with_xpu(monkeypatch):
    record = SimpleNamespace(options_kwargs={"ray_actor_options": {"num_gpus": 1}})
    monkeypatch.setitem(batch_server._DEPLOYED_APPS, "xpu-test-server", record)
    ray_put = Mock(side_effect=AssertionError("must not serialize the XPU model"))
    monkeypatch.setattr(batch_server.ray, "put", ray_put)
    with pytest.raises(ValueError, match="XPU model serving through Ray"):
        batch_server.update_served_predict_unit(
            "xpu-test-server", SimpleNamespace(device="xpu:1")
        )
    ray_put.assert_not_called()


@pytest.mark.skipif(not batch_server.ray_installed, reason="Ray is optional")
def test_xpu_autobatch_does_not_return_unmeasured_cpu_defaults():
    with pytest.raises(ValueError, match="only for CUDA"):
        batch_server.probe_optimal_batch_size(
            SimpleNamespace(device="xpu:0"), probe_data=[object()]
        )

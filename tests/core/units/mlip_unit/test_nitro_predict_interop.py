"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.units.mlip_unit import predict as predict_module
from fairchem.core.units.mlip_unit.api.inference import (
    MergeMoleConsistencyError,
    UMANitroNotSupportedError,
)
from fairchem.core.units.mlip_unit.predict import _ModelWrapper
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


class _NitroBoundary(torch.nn.Module):
    """
    Stand in only for CUDA kernels; exercise the real public predictor dispatch.
    """

    def __init__(self, reference):
        super().__init__()
        self.backbone = reference.backbone
        self.tasks = reference.tasks
        self.dataset_to_tasks = reference.dataset_to_tasks
        self.supports_single_atoms = True
        self.calls = 0

    def prepare_for_inference(self, data, settings):
        return self

    def on_predict_check(self, data):
        return None

    def forward(self, data):
        self.calls += 1
        return {
            "omol_energy": {"energy": data.pos.new_zeros(1)},
            "omol_forces": {"forces": torch.zeros_like(data.pos)},
        }


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "property_name", ["hessian", "hessian_vector_product", "embeddings"]
)
def test_per_call_properties_leave_nitro_before_unsupported_forward(
    warm, property_name
):
    reference = _public_predictor()
    candidate = copy.deepcopy(reference)
    pristine = copy.deepcopy(reference.model)
    boundary = _NitroBoundary(candidate.model.module)
    candidate.model = _ModelWrapper(boundary)
    candidate.inference_settings.nitro = True
    candidate.inference_settings.execution_mode = None
    candidate._overrides = None
    loads = []

    def reload():
        loads.append(candidate.inference_settings.execution_mode)
        candidate.model = copy.deepcopy(pristine)

    candidate._load_model = reload
    data = _data()
    data.hessian_vector = torch.ones_like(data.pos)
    if warm:
        candidate.predict(data, requested_properties=("energy",))
    calls_before = boundary.calls
    expected = reference.predict(data, requested_properties=(property_name,))
    actual = candidate.predict(data, requested_properties=(property_name,))
    assert boundary.calls == calls_before
    assert len(loads) == 1
    assert loads[0] == "general"
    assert not candidate.inference_settings.nitro
    assert not candidate.inference_settings.compile
    assert set(actual) == {property_name}
    torch.testing.assert_close(actual[property_name], expected[property_name])
    candidate.predict(data, requested_properties=("energy", "forces"))
    assert len(loads) == 1


def test_explicit_model_overrides_are_not_silently_discarded_by_nitro(monkeypatch):
    unit = _public_predictor()
    unit._requested_device = "cuda:0"
    unit._inference_model_path = "unused.pt"
    unit._overrides = {"backbone": {"cutoff": 7.0}}
    calls = []

    class Nitro:
        @staticmethod
        def validate(*args):
            calls.append("validate")

        @staticmethod
        def from_checkpoint(checkpoint):
            calls.append("build")
            return torch.nn.Linear(1, 1)

    monkeypatch.setitem(
        sys.modules,
        "fairchem.core.models.uma.uma_nitro",
        SimpleNamespace(UMANitroModel=Nitro),
    )
    checkpoint = SimpleNamespace(
        model_config={
            "model_id": "UMA-S-1.2.1",
            "backbone": {"model": "escnmd_moe_backbone", "num_experts": 32},
        }
    )
    with pytest.raises(UMANitroNotSupportedError, match="overrides"):
        unit._load_nitro_model(checkpoint)
    assert calls == []
    assert unit._overrides == {"backbone": {"cutoff": 7.0}}


def test_repeated_md_does_not_deepcopy_settings_and_detects_later_policy_change(
    monkeypatch,
):
    unit = _public_predictor()
    data = _data()
    expected = unit.predict(data, requested_properties=("energy", "forces"))
    settings_type = type(unit.inference_settings)

    def forbid_copy(self, memo):
        raise AssertionError(
            "Prepared MD must compare settings without allocating a copy"
        )

    monkeypatch.setattr(settings_type, "__deepcopy__", forbid_copy, raising=False)
    for _ in range(4):
        actual = unit.predict(data, requested_properties=("energy", "forces"))
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=2e-5, atol=2e-6
            )
    unit.inference_settings.requested_properties = ("energy",)
    assert set(unit.predict(data)) == {"energy"}
    unit.inference_settings.predict_untrained_forces.add("omat")
    with pytest.raises(ValueError, match="settings cannot change"):
        unit.predict(data)


@pytest.mark.parametrize(
    "property_name", ["hessian", "hessian_vector_product", "embeddings"]
)
def test_startup_requested_properties_skip_nitro_allocation(monkeypatch, property_name):
    unit = _public_predictor()
    unit.inference_settings.nitro = True
    unit.inference_settings.compile = True
    unit.inference_settings.execution_mode = None
    unit.inference_settings.requested_properties = (property_name,)
    unit._inference_model_path = "unused.pt"
    checkpoint = object()
    calls = []

    def build_hydra(loaded):
        assert loaded is checkpoint
        calls.append("hydra")
        assert not unit.inference_settings.nitro
        assert not unit.inference_settings.compile
        assert unit.inference_settings.execution_mode == "general"

    def build_nitro(loaded):
        raise AssertionError("An unsupported startup request must not allocate Nitro")

    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(unit, "_load_hydra_model", build_hydra)
    monkeypatch.setattr(unit, "_load_nitro_model", build_nitro)
    unit._load_model()
    assert calls == ["hydra"]


@pytest.mark.parametrize("fail", [False, True])
def test_nitro_load_uses_resolved_gpu_and_restores_foreign_current_device(
    monkeypatch, fail
):
    unit = _public_predictor()
    unit._requested_device = "cuda"
    unit.device = "cuda:1"
    unit._inference_model_path = "unused.pt"
    unit._overrides = None
    active = ["cuda:0"]
    observed = []

    @contextmanager
    def selected(device):
        before = active[0]
        active[0] = str(device)
        try:
            yield
        finally:
            active[0] = before

    class Nitro:
        @staticmethod
        def validate(settings, path, device):
            observed.append((device, active[0]))

        @staticmethod
        def from_checkpoint(checkpoint):
            observed.append(active[0])
            if fail:
                raise RuntimeError("model construction failed")
            return torch.nn.Linear(1, 1)

    monkeypatch.setattr(torch.cuda, "device", selected)
    monkeypatch.setitem(
        sys.modules,
        "fairchem.core.models.uma.uma_nitro",
        SimpleNamespace(UMANitroModel=Nitro),
    )
    checkpoint = SimpleNamespace(
        model_config={
            "model_id": "UMA-S-1.2.1",
            "backbone": {"model": "escnmd_moe_backbone", "num_experts": 32},
        }
    )
    if fail:
        with pytest.raises(RuntimeError, match="model construction failed"):
            unit._load_nitro_model(checkpoint)
    else:
        unit._load_nitro_model(checkpoint)
    assert observed == [("cuda:1", "cuda:1"), "cuda:1"]
    assert active == ["cuda:0"]


@pytest.mark.parametrize("changed", [False, True])
def test_nitro_contract_is_checked_before_owned_device_transfer(monkeypatch, changed):
    reference = _public_predictor()
    unit = copy.deepcopy(reference)
    pristine = copy.deepcopy(reference.model)
    boundary = _NitroBoundary(unit.model.module)
    unit.model = _ModelWrapper(boundary)
    unit.inference_settings.nitro = True
    unit._overrides = None
    data = _data()
    unit.predict(data, requested_properties=("energy",))
    if changed:
        data.charge += 1
    snapshot = data.clone()
    events = []
    original_copy = predict_module._copy_data_to_device

    def check(original):
        assert original is data
        events.append("check")
        if changed:
            raise MergeMoleConsistencyError("charge changed")

    def transfer(*args, **kwargs):
        assert events[0] == "check"
        events.append("copy")
        return original_copy(*args, **kwargs)

    def reload():
        assert events == ["check"]
        events.append("reload")
        unit.model = copy.deepcopy(pristine)

    monkeypatch.setattr(boundary, "on_predict_check", check)
    monkeypatch.setattr(unit, "_load_model", reload)
    monkeypatch.setattr(predict_module, "_copy_data_to_device", transfer)
    actual = unit.predict(data, requested_properties=("energy", "forces"))
    assert events.count("check") == 1
    if changed:
        monkeypatch.setattr(predict_module, "_copy_data_to_device", original_copy)
        expected = reference.predict(data, requested_properties=("energy", "forces"))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])
    else:
        assert events == ["check", "copy"]
    for name in ("pos", "cell", "atomic_numbers", "charge", "spin"):
        torch.testing.assert_close(data[name], snapshot[name], rtol=0, atol=0)
    assert not data.pos.requires_grad


@pytest.mark.parametrize("fail", [False, True])
def test_nitro_execution_preserves_fixed_precision_under_ambient_autocast(
    monkeypatch, fail
):
    unit = _public_predictor()
    boundary = _NitroBoundary(unit.model.module)
    unit.model = _ModelWrapper(boundary)
    unit.inference_settings.nitro = True
    original = boundary.forward

    def forward(data):
        assert not torch.is_autocast_enabled("cpu")
        if fail:
            raise RuntimeError("kernel failed")
        return original(data)

    monkeypatch.setattr(boundary, "forward", forward)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if fail:
            with pytest.raises(RuntimeError, match="kernel failed"):
                unit.predict(_data(), requested_properties=("energy",))
        else:
            output = unit.predict(_data(), requested_properties=("energy",))
            assert output["energy"].dtype == torch.float32
        assert torch.is_autocast_enabled("cpu")


@pytest.mark.parametrize("periodic", [False, True])
def test_cpu_geometry_key_precedes_transfer_and_refreshes_every_md_step(
    monkeypatch, periodic
):
    nitro = pytest.importorskip("fairchem.core.models.uma.uma_nitro._inference")
    unit = _public_predictor()
    boundary = _NitroBoundary(unit.model.module)
    unit.model = _ModelWrapper(boundary)
    unit.inference_settings.nitro = True
    unit.predict(_data(), requested_properties=("energy",))
    data = _data()
    data.pos = data.pos.double()
    data.cell = torch.eye(3, dtype=torch.float64).mul(8).unsqueeze(0)
    data.pbc.fill_(periodic)
    original_copy = predict_module._copy_data_to_device
    original_forward = boundary.forward
    events = []
    keys = []

    def key(source, dtype):
        assert source is data
        assert dtype == torch.float32
        events.append("key")
        return nitro.UMANitroModel._cpu_geometry_key(source, dtype)

    def transfer(source, device, dtype):
        assert events[-1] == "key"
        events.append("copy")
        return original_copy(source, device, dtype)

    def forward(owned, *, geometry_key):
        assert owned is not data
        assert owned.pos.data_ptr() != data.pos.data_ptr()
        assert owned.pos.dtype == torch.float32
        assert geometry_key == nitro._geometry_key(owned)
        keys.append(geometry_key)
        return original_forward(owned)

    monkeypatch.setattr(boundary, "_cpu_geometry_key", key, raising=False)
    monkeypatch.setattr(boundary, "forward", forward)
    monkeypatch.setattr(predict_module, "_copy_data_to_device", transfer)
    for step in range(2):
        if step:
            data.pos.mul_(2)
            data.cell.mul_(2)
        before = data.clone()
        unit.predict(data, requested_properties=("energy", "forces"))
        for name in ("pos", "cell", "atomic_numbers", "pbc"):
            torch.testing.assert_close(data[name], before[name], rtol=0, atol=0)
        assert not any("geometry_key" in name for name, _ in data)
    assert events == ["key", "copy", "key", "copy"]
    assert keys[0] != keys[1]


def test_nitro_geometry_key_is_dropped_on_forward_fallback(monkeypatch):
    nitro = pytest.importorskip("fairchem.core.models.uma.uma_nitro._inference")
    reference = _public_predictor()
    unit = copy.deepcopy(reference)
    pristine = copy.deepcopy(reference.model)
    boundary = _NitroBoundary(unit.model.module)
    unit.model = _ModelWrapper(boundary)
    unit.inference_settings.nitro = True
    unit._overrides = None
    keys = []

    def forward(data, *, geometry_key):
        keys.append(geometry_key)
        raise UMANitroNotSupportedError("capacity exceeded")

    def reload():
        unit.model = copy.deepcopy(pristine)

    monkeypatch.setattr(
        boundary,
        "_cpu_geometry_key",
        nitro.UMANitroModel._cpu_geometry_key,
        raising=False,
    )
    monkeypatch.setattr(boundary, "forward", forward)
    monkeypatch.setattr(unit, "_load_model", reload)
    data = _data()
    expected = reference.predict(data, requested_properties=("energy", "forces"))
    actual = unit.predict(data, requested_properties=("energy", "forces"))
    assert len(keys) == 1
    assert keys[0] == nitro._geometry_key(data)
    assert not unit.inference_settings.nitro
    # Real Hydra accepts no geometry-key argument; successful fallback therefore
    # checks the boundary as well as ordinary output normalization and collation.
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name])

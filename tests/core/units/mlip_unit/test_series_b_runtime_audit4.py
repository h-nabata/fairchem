"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from ase.build import molecule

from fairchem.core.common._model_state import MODEL_STATE_LOCK
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MLIPInferenceCheckpoint,
)
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit
from fairchem.core.units.mlip_unit.utils import load_inference_model, load_tasks
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


class VersionedLinear(torch.nn.Linear):
    _version = 7

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if local_metadata.get("version") != self._version:
            error_msgs.append("missing original module version metadata")
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


@pytest.mark.parametrize("use_ema", [False, True])
def test_checkpoint_load_ignores_foreign_default_device(use_ema):
    source = torch.nn.Linear(2, 1)
    ema = torch.optim.swa_utils.AveragedModel(source)
    checkpoint = MLIPInferenceCheckpoint(
        model_config={
            "_target_": "torch.nn.Linear",
            "in_features": 2,
            "out_features": 1,
        },
        model_state_dict=source.state_dict(),
        ema_state_dict=ema.state_dict(),
        tasks_config={},
    )
    with torch.device("meta"), torch.inference_mode():
        model, _ = load_inference_model(
            "unused", preloaded_checkpoint=checkpoint, use_ema=use_ema
        )
        assert torch.empty(0).device.type == "meta"
        assert torch.is_inference_mode_enabled()
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    assert not any(value.is_inference() for value in model.state_dict().values())
    values = torch.randn(3, 2, requires_grad=True)
    torch.testing.assert_close(model(values), source(values), atol=0, rtol=0)
    actual_grad = torch.autograd.grad(model(values).sum(), values)[0]
    expected_grad = torch.autograd.grad(source(values).sum(), values)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("use_ema", [False, True])
def test_checkpoint_load_preserves_module_version_metadata(use_ema):
    source = VersionedLinear(2, 1)
    ema = torch.optim.swa_utils.AveragedModel(source)
    checkpoint = MLIPInferenceCheckpoint(
        model_config={
            "_target_": f"{__name__}.VersionedLinear",
            "in_features": 2,
            "out_features": 1,
        },
        model_state_dict=source.state_dict(),
        ema_state_dict=ema.state_dict(),
        tasks_config={},
    )
    before = copy.deepcopy(checkpoint.ema_state_dict._metadata)
    model, _ = load_inference_model(
        "unused", preloaded_checkpoint=checkpoint, use_ema=use_ema
    )
    values = torch.randn(3, 2)
    torch.testing.assert_close(model(values), source(values))
    assert checkpoint.ema_state_dict._metadata == before


def test_task_load_owns_cpu_statistics_outside_caller_inference_context(tmp_path):
    checkpoint = MLIPInferenceCheckpoint(
        model_config={},
        model_state_dict={},
        ema_state_dict={},
        tasks_config=[
            {
                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                "name": "omol_energy",
                "level": "system",
                "property": "energy",
                "out_spec": {
                    "_target_": "fairchem.core.units.mlip_unit.mlip_unit.OutputSpec",
                    "dim": [1],
                    "dtype": "float32",
                },
                "normalizer": {
                    "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                    "mean": 1.0,
                    "rmsd": 2.0,
                },
                "datasets": ["omol"],
            }
        ],
    )
    path = tmp_path / "tasks.pt"
    torch.save(checkpoint, path)
    with torch.device("meta"), torch.inference_mode():
        tasks = load_tasks(path)
        assert torch.empty(0).device.type == "meta"
        assert torch.is_inference_mode_enabled()
    for value in tasks[0].normalizer.buffers():
        assert value.device.type == "cpu"
        assert not value.is_inference()


@pytest.mark.parametrize(
    "copy_spec", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_model_spec_identity_survives_mutation_and_serialization(copy_spec):
    strict = ModelSpec("checkpoint", InferenceSettings(activation_checkpointing=False))
    approximate = ModelSpec(
        "checkpoint",
        InferenceSettings(
            activation_checkpointing=False, series_b=SeriesBSettings(weight_bits=8)
        ),
    )
    strict_copy = copy_spec(strict)
    strict.inference_settings.series_b = approximate.inference_settings.series_b
    strict.inference_settings.execution_mode = (
        approximate.inference_settings.execution_mode
    )
    assert strict != approximate
    assert strict == strict_copy
    assert hash(strict) == hash(strict_copy)
    assert len({strict, strict_copy, approximate}) == 2
    assert not strict.loader_settings().series_b.enabled


def test_model_spec_validates_and_normalizes_owned_settings_before_hashing():
    settings = InferenceSettings(activation_checkpointing=False)
    settings.series_b = {"weight_bits": 8}
    mapped = ModelSpec("checkpoint", settings)
    typed = ModelSpec(
        "checkpoint",
        InferenceSettings(
            activation_checkpointing=False, series_b=SeriesBSettings(weight_bits=8)
        ),
    )
    assert mapped == typed
    assert mapped.model_id == typed.model_id
    assert isinstance(mapped.loader_settings().series_b, SeriesBSettings)
    assert settings.series_b == {"weight_bits": 8}
    settings.compile = True
    with pytest.raises(ValueError, match="requires compile"):
        ModelSpec("checkpoint", settings)


def test_model_spec_equality_preserves_serialized_override_types():
    integer = ModelSpec("checkpoint", overrides={"backbone": {"max_neighbors": 1}})
    floating = ModelSpec("checkpoint", overrides={"backbone": {"max_neighbors": 1.0}})
    assert integer.model_id != floating.model_id
    assert integer != floating


@pytest.mark.parametrize("cache_size", [0, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_series_b_preparation_retries_from_pristine_model(
    monkeypatch, cache_size, error_type
):
    unit = _public_predictor(cache_size)
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    pristine = copy.deepcopy(unit.model)
    reference = copy.deepcopy(unit)
    expected = reference.predict(_data())
    move = unit.move_to_device
    moves, loads = 0, 0

    def fail_once():
        nonlocal moves
        moves += 1
        if moves == 1:
            raise error_type("transient device allocation failure")
        move()

    def reload_model():
        nonlocal loads
        loads += 1
        unit.model = copy.deepcopy(pristine)

    monkeypatch.setattr(unit, "move_to_device", fail_once)
    monkeypatch.setattr(unit, "_load_model", reload_model)
    with pytest.raises(error_type, match="transient device allocation failure"):
        unit.predict(_data())
    assert not unit.lazy_model_intialized
    actual = unit.predict(_data())
    assert unit.lazy_model_intialized
    assert loads == (1 if cache_size == 0 else 0)
    for property_name, value in expected.items():
        torch.testing.assert_close(actual[property_name], value, rtol=1e-6, atol=1e-7)
    assert unit.approximation_manifest == reference.approximation_manifest


@pytest.mark.parametrize("interrupted", [False, True])
def test_prepared_template_clones_preserve_autograd_under_outer_inference_mode(
    monkeypatch, interrupted
):
    unit = _public_predictor(2)
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    data = _data()
    if interrupted:
        move = unit.move_to_device

        def fail_move():
            raise RuntimeError("allocation failure")

        monkeypatch.setattr(unit, "move_to_device", fail_move)
        with pytest.raises(RuntimeError, match="allocation failure"):
            unit.predict(data)
        monkeypatch.setattr(unit, "move_to_device", move)
    else:
        unit.predict(data)
        data.atomic_numbers[0] = 6
    with torch.inference_mode():
        result = unit.predict(data, requested_properties=("energy",))
        assert torch.is_inference_mode_enabled()
    assert torch.isfinite(result["energy"]).all()
    assert not any(parameter.is_inference() for parameter in unit.model.parameters())


def test_series_b_preparation_ignores_foreign_default_device():
    unit = _public_predictor()
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    reference = copy.deepcopy(unit)
    data = _data()
    expected = reference.predict(data)
    with torch.device("meta"):
        actual = unit.predict(data)
        assert torch.empty(0).device.type == "meta"
    for property_name, value in expected.items():
        assert actual[property_name].device.type == "cpu"
        torch.testing.assert_close(actual[property_name], value, rtol=1e-6, atol=1e-7)


def test_public_constructor_preserves_caller_context_and_later_force_inference(
    monkeypatch,
):
    def load_model(unit):
        unit.model = _public_predictor().model

    monkeypatch.setattr(MLIPPredictUnit, "_load_model", load_model)
    settings = InferenceSettings(
        activation_checkpointing=False,
        merge_mole=True,
        series_b=SeriesBSettings(weight_bits=8),
    )
    with torch.device("meta"), torch.inference_mode():
        unit = MLIPPredictUnit("unused", "cpu", inference_settings=settings)
        assert torch.is_inference_mode_enabled()
        assert torch.empty(0).device.type == "meta"
    result = unit.predict(_data())
    assert result["forces"].shape == (3, 3)
    assert torch.isfinite(result["forces"]).all()


def test_preparation_retry_keeps_failed_state_after_checkpoint_read_error(monkeypatch):
    unit = _public_predictor()
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    pristine = copy.deepcopy(unit.model)
    move = unit.move_to_device
    loads = 0

    def fail_move():
        raise RuntimeError("allocation failure")

    def reload_model():
        nonlocal loads
        loads += 1
        if loads == 1:
            raise OSError("checkpoint temporarily unavailable")
        unit.model = copy.deepcopy(pristine)

    monkeypatch.setattr(unit, "move_to_device", fail_move)
    monkeypatch.setattr(unit, "_load_model", reload_model)
    with pytest.raises(RuntimeError, match="allocation failure"):
        unit.predict(_data())
    monkeypatch.setattr(unit, "move_to_device", move)
    with pytest.raises(OSError, match="checkpoint temporarily unavailable"):
        unit.predict(_data())
    assert unit._preparation_failed
    assert not unit.lazy_model_intialized
    result = unit.predict(_data())
    assert loads == 2
    assert not unit._preparation_failed
    assert torch.isfinite(result["energy"]).all()


@pytest.mark.parametrize("validate_before_retry", [False, True])
def test_fallback_checkpoint_read_error_can_be_retried(
    monkeypatch, validate_before_retry
):
    unit = _public_predictor()
    unit.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    pristine = copy.deepcopy(unit.model)
    unit.predict(_data())
    loads = 0

    def reload_model():
        nonlocal loads
        loads += 1
        if loads == 1:
            raise OSError("checkpoint temporarily unavailable")
        unit.model = copy.deepcopy(pristine)

    monkeypatch.setattr(unit, "_load_model", reload_model)
    with pytest.raises(OSError, match="checkpoint temporarily unavailable"):
        unit._fall_back_with("test composition fallback", merge_mole=False)
    if validate_before_retry:
        atoms = molecule("H2O")
        unit.validate_atoms_data(atoms, "omol")
        assert atoms.info["charge"] == 0
    result = unit.predict(_data())
    assert loads == 2
    assert not unit.inference_settings.merge_mole
    assert torch.isfinite(result["forces"]).all()


def test_checkpoint_reload_conflict_fails_without_waiting_for_callback(monkeypatch):
    unit = _public_predictor()
    entered, finished = threading.Event(), threading.Event()
    reads = []

    def read_checkpoint(*args, **kwargs):
        reads.append(True)
        raise AssertionError("conflicting reload must fail before reading")

    def reload_model():
        entered.set()
        try:
            unit._load_model()
        finally:
            finished.set()

    monkeypatch.setattr(torch, "load", read_checkpoint)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with MODEL_STATE_LOCK:
            future = executor.submit(reload_model)
            assert entered.wait(timeout=5)
            completed_while_state_owned = finished.wait(timeout=5)
        assert completed_while_state_owned
        with pytest.raises(RuntimeError, match="retry after that operation"):
            future.result(timeout=5)
    assert reads == []

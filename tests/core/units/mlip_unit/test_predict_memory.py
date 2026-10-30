"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import traceback
import weakref
from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import predict as predict_module
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MergeMoleConsistencyError,
    UMANitroNotSupportedError,
)
from fairchem.core.units.mlip_unit.predict import (
    MLIPPredictUnit,
    _copy_data_to_device,
    _ModelWrapper,
    collate_predictions,
)
from tests.core.units.mlip_unit.test_xpu_preparation import _data, _small_predictor


class _CopyRecorder(TorchDispatchMode):
    def __init__(self):
        self.calls = Counter()
        self.copy_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[func] += 1
        result = func(*args, **(kwargs or {}))
        if func in (torch.ops.aten.clone.default, torch.ops.aten._to_copy.default):
            self.copy_bytes += result.numel() * result.element_size()
        return result


def _legacy_copy(data, device, dtype):
    copied = data.to(device).clone()
    for key, value in copied:
        if torch.is_tensor(value) and value.is_floating_point():
            copied[key] = value.to(dtype)
    return copied


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_transfer_copies_each_tensor_once_without_scalar_validation(dtype):
    data = _data()
    for key, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.double()
    with _CopyRecorder() as legacy:
        expected = _legacy_copy(data, "cpu", dtype)
    with _CopyRecorder() as current:
        actual = _copy_data_to_device(data, "cpu", dtype)
    for key, value in expected:
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value)
            assert actual[key].data_ptr() != data[key].data_ptr() or not value.numel()
    tensors = [value for _, value in data if torch.is_tensor(value)]
    assert current.calls[torch.ops.aten._to_copy.default] == len(tensors)
    assert not current.calls[torch.ops.aten.clone.default]
    assert not current.calls[torch.ops.aten._local_scalar_dense.default]
    assert legacy.calls[torch.ops.aten._local_scalar_dense.default] > 0
    assert current.copy_bytes < legacy.copy_bytes


def test_transfer_preserves_nested_aliases_metadata_and_second_derivatives():
    data = atomicdata_list_to_batch([_data(), _data()])
    leaf = data.pos.detach().double().requires_grad_(True)
    data.pos = leaf * 1.5
    data.extra = {"same_positions": data.pos, "metadata": ["original"]}
    copied = _copy_data_to_device(data, "cpu", torch.float64)
    assert copied.extra["same_positions"] is copied.pos
    assert copied.extra is not data.extra
    assert copied.get_batch_stats() == data.get_batch_stats()
    copied.get_batch_stats()[0]["pos"][0] = 123
    assert data.get_batch_stats()[0]["pos"][0] == 0
    copied.extra["metadata"][0] = "changed"
    assert data.extra["metadata"] == ["original"]
    copied.dataset[0] = "changed"
    assert data.dataset == ["omol", "omol"]
    first = torch.autograd.grad(copied.pos.pow(3).sum(), leaf, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), leaf)[0]
    torch.testing.assert_close(first, 3 * 1.5**3 * leaf.square())
    torch.testing.assert_close(second, 6 * 1.5**3 * leaf)


def test_transfer_to_fake_xpu_keeps_caller_on_cpu():
    data = _data()
    with FakeTensorMode(allow_non_fake_inputs=True):
        copied = _copy_data_to_device(data, "xpu:1", torch.float32)
    for _, value in copied:
        if torch.is_tensor(value):
            assert value.device == torch.device("xpu:1")
    assert all(
        value.device.type == "cpu" for _, value in data if torch.is_tensor(value)
    )


def _task(name, level, prop):
    return SimpleNamespace(name=name, level=level, property=prop)


@pytest.mark.parametrize("same_property", [False, True])
def test_single_system_collation_preserves_values_and_gradients(same_property):
    data = _data()
    energy = torch.tensor([2.0], requires_grad=True)
    forces = torch.arange(9.0).view(3, 3).requires_grad_(True)
    tasks = [_task("e", "system", "energy"), _task("f", "atom", "forces")]
    predictions = {"e": energy, "f": forces}
    if same_property:
        tasks.append(_task("f2", "atom", "forces"))
        predictions["f2"] = forces * 2
    unit = SimpleNamespace(dataset_to_tasks={"omol": tasks})

    @collate_predictions
    def run(unit, data, undo_refs):
        return predictions

    with _CopyRecorder() as trace:
        actual = run(unit, data)
    expected_forces = torch.cat([forces, forces * 2]) if same_property else forces
    torch.testing.assert_close(actual["forces"], expected_forces)
    torch.testing.assert_close(actual["energy"], energy)
    if not same_property:
        assert actual["forces"] is forces
        assert not trace.calls[torch.ops.aten.index.Tensor]
        assert not trace.calls[torch.ops.aten.cat.default]
    gradient = torch.autograd.grad(actual["forces"].sum(), forces)[0]
    torch.testing.assert_close(
        gradient, torch.full_like(forces, 3 if same_property else 1)
    )


def test_collation_keeps_interleaved_multidataset_atom_order():
    data = SimpleNamespace(dataset=["a", "b"], batch=torch.tensor([1, 0, 1, 0, 1]))
    a = torch.arange(15.0).view(5, 3).requires_grad_(True)
    b = (torch.arange(15.0).view(5, 3) + 20).requires_grad_(True)
    unit = SimpleNamespace(
        dataset_to_tasks={
            "a": [_task("a", "atom", "forces")],
            "b": [_task("b", "atom", "forces")],
        }
    )

    @collate_predictions
    def run(unit, data, undo_refs):
        return {"a": a, "b": b}

    actual = run(unit, data)["forces"]
    torch.testing.assert_close(actual, torch.cat([a[[1, 3]], b[[0, 2, 4]]]))
    grad_a, grad_b = torch.autograd.grad(actual.sum(), (a, b))
    torch.testing.assert_close(grad_a[:, 0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(grad_b[:, 0], torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0]))


def test_collation_does_not_retain_unselected_system_storage():
    data = SimpleNamespace(dataset=["a", "b"])
    prediction = torch.ones(2, 8, 8)
    unit = SimpleNamespace(
        dataset_to_tasks={"a": [_task("h", "system", "hessian")], "b": []}
    )

    @collate_predictions
    def run(unit, data, undo_refs):
        return {"h": prediction}

    actual = run(unit, data)["hessian"]
    torch.testing.assert_close(actual, prediction[:1])
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()


class _IdentityNormalizer:
    def denorm(self, value):
        return value

    def to(self, device):
        return self


@pytest.mark.parametrize("merge_mole", [False, True])
def test_real_uma_prediction_matches_legacy_copy(merge_mole, monkeypatch):
    candidate = _small_predictor()
    candidate.inference_settings.merge_mole = merge_mole
    tasks = []
    for prop in ("energy", "forces", "stress"):
        task = _task(f"omol_{prop}", "atom" if prop == "forces" else "system", prop)
        task.normalizer = _IdentityNormalizer()
        task.element_references = None
        tasks.append(task)
    candidate.model.module._tasks = {task.name: task for task in tasks}
    candidate.model.module._dataset_to_tasks = {"omol": tasks}
    reference = copy.deepcopy(candidate)
    data = _data()
    before = data.clone()
    rng_state = torch.random.get_rng_state()
    with monkeypatch.context() as context:
        context.setattr(predict_module, "_copy_data_to_device", _legacy_copy)
        expected = reference.predict(data.clone())
    torch.random.set_rng_state(rng_state)
    actual = candidate.predict(data)
    for prop in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[prop], expected[prop], rtol=0, atol=0)
    for key, value in before:
        if torch.is_tensor(value):
            torch.testing.assert_close(data[key], value)
            assert data[key].requires_grad == value.requires_grad


class _FallbackModule(torch.nn.Module):
    def __init__(self, released):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(64))
        self.released = released

    def release_inference_resources(self):
        self.released.append(True)

    def fail(self, chained):
        temporary = self.weight.square()
        self.temporary_ref = weakref.ref(temporary)
        if chained:
            try:
                self._kernel_failure(temporary)
            except RuntimeError as cause:
                raise MergeMoleConsistencyError("changed composition") from cause
        raise MergeMoleConsistencyError("changed composition")

    def _kernel_failure(self, temporary):
        raise RuntimeError("recoverable kernel failure")


@pytest.mark.parametrize("chained", [False, True])
def test_fallback_releases_model_and_intermediates_before_reload(chained):
    released = []
    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    super(MLIPPredictUnit, predictor).__init__()
    predictor.model = _ModelWrapper(_FallbackModule(released))
    predictor.inference_settings = InferenceSettings(merge_mole=True)
    predictor._requested_device = "cpu"
    predictor.lazy_model_intialized = True
    old = weakref.ref(predictor.model.module)
    temporary_ref = None

    def reload():
        assert old() is None
        assert temporary_ref() is None
        predictor.model = _ModelWrapper(torch.nn.Identity())

    predictor._load_model = reload
    try:
        predictor.model.module.fail(chained)
    except MergeMoleConsistencyError as error:
        temporary_ref = predictor.model.module.temporary_ref
        predictor._fall_back(error)
    assert released == [True]
    assert not predictor.inference_settings.merge_mole
    assert not predictor.lazy_model_intialized


class _PredictionFallbackModule(torch.nn.Module):
    def __init__(self, failure_stage=None):
        super().__init__()
        self.backbone = torch.nn.Linear(1, 1)
        self.backbone.regress_config = SimpleNamespace(direct_forces=True)
        self.failure_stage = failure_stage
        self.supports_single_atoms = True
        self.tasks = {}
        self.dataset_to_tasks = {"omol": []}

    def on_predict_check(self, data):
        if self.failure_stage == "check":
            raise MergeMoleConsistencyError("changed composition")

    def prepare_for_inference(self, data, settings):
        pass

    def forward(self, data):
        if self.failure_stage == "forward":
            raise UMANitroNotSupportedError("kernel capacity exceeded")
        return {}


@pytest.mark.parametrize("failure_stage", ["check", "forward"])
def test_predict_does_not_retain_old_backbone_during_fallback(failure_stage):
    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    super(MLIPPredictUnit, predictor).__init__()
    predictor.model = _ModelWrapper(_PredictionFallbackModule(failure_stage))
    predictor.inference_settings = InferenceSettings(
        merge_mole=True, nitro=failure_stage == "forward"
    )
    predictor._requested_device = predictor.device = "cpu"
    predictor.lazy_model_intialized = True
    predictor._warned_upcast = predictor._nitro_announced = False
    old_model = weakref.ref(predictor.model.module)
    old_backbone = weakref.ref(predictor.model.module.backbone)

    def reload():
        assert old_model() is None
        assert old_backbone() is None
        predictor.model = _ModelWrapper(_PredictionFallbackModule())

    predictor._load_model = reload
    assert predictor.predict(_data()) == {}


def test_terminal_fallback_keeps_model_and_original_traceback():
    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    super(MLIPPredictUnit, predictor).__init__()
    predictor.model = _ModelWrapper(_FallbackModule([]))
    predictor.inference_settings = InferenceSettings()
    predictor._requested_device = "cpu"
    original_model = predictor.model

    def unexpected_reload():
        raise AssertionError("An unchanged general model must not be reloaded")

    predictor._load_model = unexpected_reload
    with pytest.raises(MergeMoleConsistencyError) as original:
        predictor.model.module.fail(chained=True)
    with pytest.raises(MergeMoleConsistencyError) as raised:
        predictor._fall_back(original.value)
    error = original.value
    assert raised.value is error
    assert "fail" in [frame.name for frame in traceback.extract_tb(error.__traceback__)]
    assert "_kernel_failure" in [
        frame.name for frame in traceback.extract_tb(error.__cause__.__traceback__)
    ]
    assert predictor.model is original_model


def test_load_model_releases_partial_nitro_model_before_hydra(monkeypatch):
    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    super(MLIPPredictUnit, predictor).__init__()
    predictor.inference_settings = InferenceSettings(
        nitro=True, auto_add_default_untrained_tasks=False
    )
    predictor._inference_model_path = "unused.pt"
    predictor._requested_device = "cpu"
    partial_model = None
    checkpoint = object()
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)

    def load_nitro(checkpoint):
        # The real Nitro constructor builds its backbone/head before validating
        # checkpoint tensor names. A rejected tensor retains this model in the
        # exception traceback even though predictor.model was never assigned.
        nonlocal partial_model
        module = torch.nn.Linear(64, 64)
        partial_model = weakref.ref(module)
        raise UMANitroNotSupportedError("unexpected checkpoint tensor")

    def load_hydra(checkpoint):
        assert partial_model() is None
        predictor.model = _ModelWrapper(_PredictionFallbackModule())

    predictor._load_nitro_model = load_nitro
    predictor._load_hydra_model = load_hydra
    predictor._load_model()
    assert not predictor.inference_settings.nitro

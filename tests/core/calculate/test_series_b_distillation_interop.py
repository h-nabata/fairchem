"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from threading import Event, RLock
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from fairchem.core.calculate.series_b_distillation import (
    DistillationDataset,
    DistillationResult,
    DistillationSample,
    DistillationWeights,
    PropertyOutputAdapter,
    collect_teacher_labels,
    distill_student,
    export_distilled_checkpoint,
)
from fairchem.core.models.base import HydraModel
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.predict import (
    MLIPPredictUnit,
    _isolated_model_initialization,
    _predictor_state,
    _serialized_prediction,
)
from fairchem.core.units.mlip_unit.utils import load_inference_model
from tests.core.calculate.test_series_b_distillation import (
    HarmonicStudent,
    Teacher,
    _data,
)
from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor


class VersionedStudent(torch.nn.Linear):
    _version = 7

    def __init__(self):
        super().__init__(3, 1)
        self.extra = {"scale": torch.tensor(2.0), "labels": ["trained"]}
        self.extra["cycle"] = self.extra

    def get_extra_state(self):
        return self.extra

    def set_extra_state(self, state):
        self.extra = state

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, *args, **kwargs
    ):
        if local_metadata.get("version") != self._version:
            raise RuntimeError("Student state version metadata was lost")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, *args, **kwargs
        )

    def forward(self, inputs):
        return super().forward(inputs) * self.extra["scale"]


def test_recursive_metadata_is_owned_detached_and_cast_for_each_training_step():
    data = _data()
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    nested = SimpleNamespace(tensor=source * 2, integers=torch.tensor([1, 2]))
    nested.cycle = nested
    data.metadata = {nested.tensor: nested, "alias": nested.tensor}
    sample = DistillationSample(data, {"energy": torch.ones(1)})
    owned_key = next(iter(sample.data.metadata))
    assert not owned_key.requires_grad
    assert sample.data.metadata[owned_key].cycle is sample.data.metadata[owned_key]
    assert sample.data.metadata["alias"] is owned_key
    assert owned_key.data_ptr() != nested.tensor.data_ptr()

    class MetadataStudent(HarmonicStudent):
        def forward(self, batch):
            key = next(iter(batch.metadata))
            metadata = batch.metadata[key]
            assert metadata.tensor is key
            assert metadata.cycle is metadata
            assert not key.requires_grad
            assert key.dtype == self.coefficient.dtype
            assert key.device == self.coefficient.device
            assert metadata.integers.dtype == torch.int64
            # Each invocation must see an independent copy, including aliases.
            torch.testing.assert_close(key, key.new_tensor([2.0, 4.0]))
            key.add_(100)
            return super().forward(batch)

    dataset = DistillationDataset((sample,))
    distill_student(
        MetadataStudent().double(),
        dataset,
        steps=2,
        weights=DistillationWeights(forces=0),
    )
    torch.testing.assert_close(nested.tensor, source * 2)
    torch.testing.assert_close(sample.data.metadata["alias"], source.detach() * 2)
    torch.testing.assert_close(
        dataset.samples[0].data.metadata["alias"], source.detach() * 2
    )


@pytest.mark.parametrize(
    "context", ["meta", "inference", "no_grad_autocast", "float64"]
)
def test_training_is_independent_of_callers_torch_context(context):
    student = HarmonicStudent()
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [_data()], teacher_id="analytic"
    )
    reference = distill_student(student, dataset, steps=2)
    original_dtype = torch.get_default_dtype()
    rng = torch.get_rng_state().clone()
    with ExitStack() as stack:
        if context == "meta":
            stack.enter_context(torch.device("meta"))
        elif context == "inference":
            stack.enter_context(torch.inference_mode())
        elif context == "no_grad_autocast":
            stack.enter_context(torch.no_grad())
            stack.enter_context(torch.autocast("cpu", dtype=torch.bfloat16))
        else:
            stack.callback(torch.set_default_dtype, original_dtype)
            torch.set_default_dtype(torch.float64)
        candidate = distill_student(student, dataset, steps=2)
        if context == "meta":
            assert torch.empty(0).device.type == "meta"
        elif context == "inference":
            assert torch.is_inference_mode_enabled()
        elif context == "no_grad_autocast":
            assert not torch.is_grad_enabled()
            assert torch.is_autocast_enabled("cpu")
        else:
            assert torch.get_default_dtype() == torch.float64
    assert candidate.loss_history == reference.loss_history
    torch.testing.assert_close(candidate.model.coefficient, reference.model.coefficient)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert torch.get_default_dtype() == original_dtype


def test_training_exception_restores_rng_and_caller_model():
    class FailingStudent(HarmonicStudent):
        def forward(self, data):
            torch.rand(8)
            data.cell.zero_()
            raise RuntimeError("student failure")

    student = FailingStudent()
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [_data()], teacher_id="analytic"
    )
    cell = dataset.samples[0].data.cell.clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="student failure"):
        distill_student(student, dataset, steps=1)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(dataset.samples[0].data.cell, cell)
    assert student.coefficient.item() == 0.5
    assert student.training


def test_training_and_predictor_construction_serialize_global_rng_scopes():
    training_started, release_training = Event(), Event()
    construction_attempted, construction_started = Event(), Event()

    class BlockingStudent(HarmonicStudent):
        def forward(self, data):
            training_started.set()
            if not release_training.wait(10):
                raise RuntimeError("Training was not released")
            return super().forward(data)

    @_isolated_model_initialization
    def construct():
        construction_started.set()
        torch.random.default_generator.manual_seed(123)
        return torch.nn.Linear(2, 1)

    def attempt_construction():
        construction_attempted.set()
        return construct()

    student = BlockingStudent()
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [_data()], teacher_id="analytic"
    )
    rng = torch.get_rng_state().clone()
    with ThreadPoolExecutor(max_workers=2) as executor:
        training = executor.submit(distill_student, student, dataset, steps=1)
        try:
            assert training_started.wait(10)
            construction = executor.submit(attempt_construction)
            assert construction_attempted.wait(10)
            assert not construction_started.wait(0.1)
        finally:
            release_training.set()
        training.result(timeout=10)
        construction.result(timeout=10)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


def test_export_preserves_extra_state_versions_and_cpu_ema_under_default_meta(tmp_path):
    model = VersionedStudent()
    values = torch.tensor([[1.0, -2.0, 3.0]])
    expected = model(values).detach()
    result = DistillationResult(model, (1.0,), {"teacher_id": "analytic"})
    config = {"_target_": f"{__name__}.VersionedStudent"}
    path = tmp_path / "student.pt"
    rng = torch.get_rng_state().clone()
    with torch.device("meta"):
        export_distilled_checkpoint(
            result, path, model_config=config, tasks_config=[], architecture_note="test"
        )
        assert torch.empty(0).device.type == "meta"
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint.model_state_dict._metadata[""]["version"] == 7
    assert checkpoint.ema_state_dict._metadata["module"]["version"] == 7
    assert checkpoint.ema_state_dict["n_averaged"].device.type == "cpu"
    state = checkpoint.model_state_dict["_extra_state"]
    ema_state = checkpoint.ema_state_dict["module._extra_state"]
    assert state["cycle"] is state
    assert ema_state["cycle"] is ema_state
    assert state is not ema_state
    model.extra["scale"].zero_()
    for use_ema in (False, True):
        loaded, _ = load_inference_model(str(path), use_ema=use_ema, strict=True)
        torch.testing.assert_close(loaded(values), expected)


@pytest.mark.parametrize("overwrite", [False, True])
def test_export_serialization_failure_is_atomic(tmp_path, monkeypatch, overwrite):
    model = torch.nn.Linear(3, 1)
    config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 1}
    destination = tmp_path / "student.pt"
    if overwrite:
        destination.write_bytes(b"original checkpoint")

    def fail_after_partial_write(checkpoint, handle):
        handle.write(b"partial new checkpoint")
        raise OSError("serialization failed")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    rng = torch.get_rng_state().clone()
    with pytest.raises(OSError, match="serialization failed"):
        export_distilled_checkpoint(
            DistillationResult(model, (), {}),
            destination,
            model_config=config,
            tasks_config=[],
            architecture_note="test",
            overwrite=overwrite,
        )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    if overwrite:
        assert destination.read_bytes() == b"original checkpoint"
    else:
        assert not destination.exists()
    assert not list(tmp_path.glob("*.tmp"))


def _tiny_uma_with_tasks():
    with patch(
        "tests.core.units.mlip_unit.test_xpu_preparation.HydraModel", wraps=HydraModel
    ) as constructor:
        student = _small_predictor().model.module
    model_config = {
        "_target_": "fairchem.core.models.base.HydraModel",
        **copy.deepcopy(constructor.call_args.kwargs),
    }
    tasks = []
    properties = ("energy", "forces", "stress")
    for name, level, dim in zip(
        properties, ("system", "atom", "system"), ([1], [3], [9])
    ):
        task = {
            "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
            "name": f"omol_{name}",
            "level": level,
            "property": name,
            "out_spec": {
                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.OutputSpec",
                "dim": dim,
                "dtype": "float32",
            },
            "normalizer": {
                "_target_": "fairchem.core.modules.normalization.normalizer.Normalizer",
                "mean": 3.0 if name == "energy" else 0.0,
                "rmsd": 2.0,
            },
            "datasets": ["omol"],
        }
        if name == "energy":
            task["element_references"] = {
                "_target_": "fairchem.core.modules.normalization.element_references.ElementReferences",
                "element_references": {"_target_": "torch.tensor", "data": [0.0, 4.0]},
            }
        tasks.append(task)
    student.setup_tasks(tasks)
    return student, model_config, tasks


def test_real_uma_distill_export_predict_preserves_normalization_and_references(
    tmp_path,
):
    student, model_config, tasks = _tiny_uma_with_tasks()
    properties = ("energy", "forces", "stress")
    student.train()
    data = _data()
    data["_requested_properties"] = properties
    adapter = PropertyOutputAdapter(
        {name: (f"omol_{name}", name) for name in properties},
        energy_scale=2.0,
        energy_offset=3.0,
        element_references={1: 4.0},
    )
    source_output = student(data.clone())
    source_coefficients = (
        student.backbone.global_mole_tensors.expert_mixing_coefficients
    )
    targets = adapter(source_output, data)
    targets["energy"] = targets["energy"] + 0.01
    dataset = DistillationDataset((DistillationSample(data, targets),))
    with torch.device("meta"):
        result = distill_student(
            student,
            dataset,
            steps=2,
            learning_rate=1e-5,
            weights=DistillationWeights(stress=1.0),
            output_adapter=adapter,
        )
    assert student.training
    assert (
        student.backbone.global_mole_tensors.expert_mixing_coefficients
        is source_coefficients
    )
    source_gradient = torch.autograd.grad(
        source_output["omol_energy"]["energy"].sum(),
        next(student.backbone.routing_mlp.parameters()),
    )[0]
    assert torch.isfinite(source_gradient).all()
    assert all(parameter.grad is None for parameter in student.parameters())
    reference = adapter(result.model(data.clone()), data)
    path = export_distilled_checkpoint(
        result,
        tmp_path / "tiny-uma.pt",
        model_config=model_config,
        tasks_config=tasks,
        architecture_note="Randomly initialized tiny UMA: serialization test only",
    )
    # MLIPPredictUnit loads the exported EMA state and applies task conventions.
    predictor = MLIPPredictUnit(
        str(path),
        device="cpu",
        inference_settings=InferenceSettings(
            activation_checkpointing=False, execution_mode="general"
        ),
    )
    actual = predictor.predict(data, requested_properties=properties)
    for name in properties:
        torch.testing.assert_close(
            actual[name].reshape(reference[name].shape),
            reference[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_training_callback_and_predictor_reload_fail_fast_and_recover(tmp_path):
    student, model_config, tasks = _tiny_uma_with_tasks()
    path = export_distilled_checkpoint(
        DistillationResult(student, (), {}),
        tmp_path / "callback-uma.pt",
        model_config=model_config,
        tasks_config=tasks,
        architecture_note="Random tiny UMA for concurrent reload testing",
    )
    predictor = MLIPPredictUnit(
        str(path),
        inference_settings=InferenceSettings(
            activation_checkpointing=False, execution_mode="general"
        ),
    )
    data = _data()
    expected = predictor.predict(data, requested_properties=("energy",))["energy"]
    adapter_started, predictor_locked, callback_attempted = Event(), Event(), Event()

    class BoundedPredictionLock:
        # A regressed lock order must fail this test instead of hanging pytest.
        def __init__(self):
            self.lock = RLock()

        def __enter__(self):
            if not self.lock.acquire(timeout=5):
                raise TimeoutError("Predictor and distillation locks deadlocked")
            return self

        def __exit__(self, *args):
            self.lock.release()

    predictor_state = _predictor_state(predictor)
    original_lock = predictor_state[0]
    predictor_state[0] = BoundedPredictionLock()

    def adapter(output, batch):
        adapter_started.set()
        assert predictor_locked.wait(10)
        callback_attempted.set()
        # The conflicting reload leaves the predictor dirty. This call must
        # recover it on the training thread, which already owns the global lock.
        actual = predictor.predict(batch, requested_properties=("energy",))["energy"]
        torch.testing.assert_close(actual, expected)
        return output

    def conflicting_reload():
        assert adapter_started.wait(10)
        with _serialized_prediction(predictor):
            predictor_locked.set()
            assert callback_attempted.wait(10)
            with pytest.raises(RuntimeError, match="retry after that operation"):
                predictor._fall_back_with("Test reload during distillation")

    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [data], teacher_id="analytic"
    )
    harmonic = HarmonicStudent()
    rng = torch.get_rng_state().clone()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            training = executor.submit(
                distill_student, harmonic, dataset, steps=1, output_adapter=adapter
            )
            reloading = executor.submit(conflicting_reload)
            reloading.result(timeout=15)
            result = training.result(timeout=15)
    finally:
        predictor_state[0] = original_lock
    assert len(result.loss_history) == 1
    assert not predictor._preparation_failed
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(
        predictor.predict(data, requested_properties=("energy",))["energy"], expected
    )

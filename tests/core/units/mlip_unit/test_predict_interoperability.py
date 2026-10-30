"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import gc
import os
import random
import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.common.distutils import CURRENT_DEVICE_TYPE_STR
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MLIPInferenceCheckpoint,
    validate_uma_atoms_data,
)
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task
from fairchem.core.units.mlip_unit.predict import (
    BatchServerPredictUnit,
    MLIPPredictUnit,
    _copy_data_to_device,
    _ModelWrapper,
    _serialized_prediction,
    _validation_info_snapshot,
    collate_predictions,
)
from fairchem.core.units.mlip_unit.utils import (
    load_inference_model,
    tf32_context_manager,
    update_configs,
)
from tests.core.units.mlip_unit.test_xpu_preparation import _data, _small_predictor


@pytest.mark.parametrize(
    "options",
    [
        {"neighbor_skin": float("nan")},
        {"neighbor_skin": float("inf")},
        {"neighbor_skin": True},
        {"edge_chunk_size": 1.5},
        {"edge_chunk_size": True},
        {"mole_cache_max_bytes": 1.5},
        {"hessian_chunk_size": True},
        {"xpu_graph_cache_size": True},
        {"mole_cache_size": True},
        {"torch_num_threads": 1.5},
        {"compile": "False"},
        {"tf32": 0},
        {"max_atoms": 1.5},
        {"base_precision_dtype": torch.int64},
        {"base_precision_dtype": "missing_dtype"},
        {"predict_untrained_forces": "omol"},
        {"requested_properties": {"energy": 1}},
    ],
)
def test_settings_reject_ambiguous_types_and_nonfinite_limits(options):
    with pytest.raises(ValueError):
        InferenceSettings(**options)


def test_settings_copy_dataset_collections_and_normalize_numpy_integers():
    datasets = ["omol"]
    settings = InferenceSettings(
        edge_chunk_size=np.int64(32),
        neighbor_skin=np.float32(0.4),
        predict_untrained_forces=datasets,
    )
    datasets.append("omat")
    assert settings.predict_untrained_forces == {"omol"}
    assert type(settings.edge_chunk_size) is int
    assert type(settings.neighbor_skin) is float


@pytest.mark.parametrize("field", ["charge", "spin"])
def test_boolean_charge_and_spin_are_not_integer_metadata(field):
    atoms = Atoms("H")
    atoms.info.update(charge=0, spin=1)
    atoms.info[field] = True
    with pytest.raises(TypeError, match="Invalid type"):
        validate_uma_atoms_data(atoms, "omol")


@pytest.mark.parametrize("fail", [False, True])
def test_constructor_preserves_callers_rng_environment_and_reference_ownership(
    monkeypatch, fail
):
    draws = []

    def load(unit):
        draws.append((random.random(), np.random.rand(), torch.rand(3)))
        assert os.environ[CURRENT_DEVICE_TYPE_STR] == "cpu"
        if fail:
            raise RuntimeError("checkpoint read failed")
        model = torch.nn.Module()
        model.backbone = SimpleNamespace(
            regress_config=SimpleNamespace(direct_forces=False)
        )
        unit.model = _ModelWrapper(model)

    monkeypatch.setattr(MLIPPredictUnit, "_load_model", load)
    monkeypatch.setenv(CURRENT_DEVICE_TYPE_STR, "xpu:1")
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda seed: pytest.fail("constructor must not seed another accelerator"),
    )
    py_state, np_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    refs = {"omol_elem_refs": {1: 2.0}}
    for _ in range(2):
        if fail:
            with pytest.raises(RuntimeError, match="checkpoint read failed"):
                MLIPPredictUnit("unused", "cpu", seed=17)
        else:
            unit = MLIPPredictUnit(
                "unused",
                torch.device("cpu"),
                seed=17,
                atom_refs=refs,
                form_elem_refs=refs,
            )
            assert unit.atom_refs["omol"] is not refs["omol_elem_refs"]
            assert unit.form_elem_refs is not refs
    assert draws[0][:2] == draws[1][:2]
    torch.testing.assert_close(draws[0][2], draws[1][2], rtol=0, atol=0)
    assert random.getstate() == py_state
    after = np.random.get_state()
    assert after[0] == np_state[0]
    assert after[2:] == np_state[2:]
    np.testing.assert_array_equal(after[1], np_state[1])
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert os.environ[CURRENT_DEVICE_TYPE_STR] == "xpu:1"


def test_repeated_matching_thread_configuration_does_not_reset_interop(monkeypatch):
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 2)
    monkeypatch.setattr(
        torch, "set_num_threads", lambda n: pytest.fail("unnecessary reset")
    )
    monkeypatch.setattr(
        torch,
        "set_num_interop_threads",
        lambda n: pytest.fail("cannot reset after parallel work"),
    )
    MLIPPredictUnit._setup_threads(
        SimpleNamespace(), InferenceSettings(torch_num_threads=2)
    )


def test_explicit_cuda_ordinal_is_preserved_and_validated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    unit = SimpleNamespace()
    MLIPPredictUnit._setup_device(unit, torch.device("cuda:1"))
    assert unit.device == "cuda:1"
    with pytest.raises(ValueError, match="index"):
        MLIPPredictUnit._setup_device(unit, "cuda:2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        MLIPPredictUnit._setup_device(unit, "cuda:1")


def _public_predictor(cache_size=0):
    unit = _small_predictor()
    unit.inference_settings.mole_cache_size = cache_size
    tasks = [
        Task(
            name=f"omol_{prop}",
            property=prop,
            level=level,
            out_spec=OutputSpec(dim=[1] if level == "system" else [3], dtype="float32"),
            datasets=["omol"],
            normalizer=Normalizer(mean=0.0, rmsd=1.0),
        )
        for prop, level in (("energy", "system"), ("forces", "atom"))
    ]
    unit.model.module._tasks = {task.name: task for task in tasks}
    unit.model.module._dataset_to_tasks = {"omol": tasks}
    return unit


def test_shared_predictor_serializes_real_jobs_and_preserves_cached_results():
    unit = _public_predictor(2)
    reference = _public_predictor()
    reference.inference_settings.merge_mole = False
    first, second = _data(), _data()
    second.charge += 1
    expected = [
        reference.predict(data, requested_properties=("energy", "forces"))
        for data in (first, second)
    ]
    entered, release, second_entered = Event(), Event(), Event()
    original = unit._run_inference
    calls = []

    def blocked(data, undo_refs):
        calls.append(int(data.charge.item()))
        if len(calls) == 1:
            entered.set()
            assert release.wait(10)
        else:
            second_entered.set()
        return original(data, undo_refs)

    unit._run_inference = blocked
    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(
            unit.predict, first, requested_properties=("energy", "forces")
        )
        assert entered.wait(10)
        two = executor.submit(
            unit.predict, second, requested_properties=("energy", "forces")
        )
        try:
            assert not second_entered.wait(0.05)
        finally:
            release.set()
        actual = [one.result(20), two.result(20)]
    assert calls == [0, 1]
    for outputs, oracle in zip(actual, expected):
        for prop in oracle:
            torch.testing.assert_close(
                outputs[prop], oracle[prop], atol=2e-5, rtol=2e-4
            )
    assert torch.count_nonzero(actual[0]["forces"]) > 0
    snapshot = actual[0]["forces"].clone()
    unit._run_inference = original
    unit.predict(second, requested_properties=("forces",))
    torch.testing.assert_close(actual[0]["forces"], snapshot, atol=0, rtol=0)


def test_copied_prepared_predictor_has_its_own_cache_generation():
    unit = _public_predictor(2)
    expected = unit.predict(_data(), requested_properties=("energy", "forces"))
    cloned = copy.deepcopy(unit)
    assert cloned._prepared_models is None
    actual = cloned.predict(_data(), requested_properties=("energy", "forces"))
    for prop in expected:
        torch.testing.assert_close(actual[prop], expected[prop], atol=2e-5, rtol=2e-4)


def test_clearing_prepared_cache_waits_for_active_prediction():
    unit = _public_predictor(2)
    entered, release, cleared = Event(), Event(), Event()
    original = unit._mole_key

    def blocked(data):
        entered.set()
        assert release.wait(10)
        return original(data)

    def clear():
        unit.clear_prepared_model_cache()
        cleared.set()

    unit._mole_key = blocked
    with ThreadPoolExecutor(max_workers=2) as executor:
        prediction = executor.submit(
            unit.predict, _data(), requested_properties=("energy", "forces")
        )
        assert entered.wait(10)
        clearing = executor.submit(clear)
        try:
            assert not cleared.wait(0.05)
        finally:
            release.set()
        expected = prediction.result(20)
        clearing.result(20)
    assert cleared.is_set()
    assert unit._prepared_models is None
    unit._mole_key = original
    actual = unit.predict(_data(), requested_properties=("energy", "forces"))
    for prop in expected:
        torch.testing.assert_close(actual[prop], expected[prop], atol=2e-5, rtol=2e-4)


def test_collation_rejects_missing_properties_and_wrong_row_counts():
    task = SimpleNamespace(name="energy", property="energy", level="system")
    unit = SimpleNamespace(dataset_to_tasks={"omol": [task]})
    outputs = {"energy": torch.ones(1)}

    @collate_predictions
    def predict(unit, data, undo_refs, **kwargs):
        return outputs

    with pytest.raises(ValueError, match="unavailable"):
        predict(unit, _data(), requested_properties=("forces",))
    outputs["energy"] = torch.ones(2)
    with pytest.raises(ValueError, match="expected 1"):
        predict(unit, _data())


def test_tf32_contexts_from_concurrent_jobs_restore_caller_policy():
    entered, release, other_entered = Event(), Event(), Event()
    precision, cudnn = (
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
    )

    def first():
        with tf32_context_manager(True):
            entered.set()
            assert release.wait(10)
            assert torch.get_float32_matmul_precision() == "high"

    def second():
        with tf32_context_manager(False):
            other_entered.set()
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("job failed")

    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(first)
        assert entered.wait(10)
        two = executor.submit(second)
        try:
            assert not other_entered.wait(0.05)
        finally:
            release.set()
        one.result(10)
        with pytest.raises(RuntimeError, match="job failed"):
            two.result(10)
    assert torch.get_float32_matmul_precision() == precision
    assert torch.backends.cudnn.allow_tf32 == cudnn


def test_preloaded_ema_checkpoint_can_be_loaded_twice_without_mutation():
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
    first, returned = load_inference_model(
        "unused", preloaded_checkpoint=checkpoint, use_ema=True
    )
    second, _ = load_inference_model(
        "unused", preloaded_checkpoint=checkpoint, use_ema=True
    )
    assert "n_averaged" in checkpoint.ema_state_dict
    assert returned is not checkpoint
    values = torch.randn(3, 2)
    torch.testing.assert_close(first(values), second(values))
    original, override = {"value": 1}, {"new": {"items": [2]}}
    merged = update_configs(original, override)
    merged["new"]["items"].append(3)
    assert original == {"value": 1}
    assert override == {"new": {"items": [2]}}


def test_remote_validation_rechecks_mutations_and_owns_returned_metadata():
    calls = []

    def validate(info, task, spec=None):
        calls.append(copy.deepcopy(info))
        validate_uma_atoms_data(SimpleNamespace(info=info), task)
        return SimpleNamespace(result=lambda **kwargs: info)

    unit = BatchServerPredictUnit(
        SimpleNamespace(validate_atoms_data=SimpleNamespace(remote=validate))
    )
    atoms = Atoms("H")
    unit.validate_atoms_data(atoms, "omol")
    unit.validate_atoms_data(atoms, "omol")
    assert len(calls) == 1
    atoms.info["charge"] = "invalid"
    with pytest.raises(TypeError):
        unit.validate_atoms_data(atoms, "omol")
    atoms.info["charge"] = 0
    atoms.info["extra"] = np.array([1])
    unit.validate_atoms_data(atoms, "omol")
    atoms.info["extra"][0] = 2
    unit.validate_atoms_data(atoms, "omol")
    assert len(calls) == 4
    assert calls[-1]["extra"].tolist() == [2]
    assert unit._validated_info is None


def test_device_copy_preserves_cyclic_metadata_aliases_and_autograd():
    data = _data()
    source = data.pos.clone().requires_grad_(True)
    data.pos = source * 2
    data.metadata = {"positions": data.pos}
    data.metadata["self"] = data.metadata
    data.metadata["items"] = [data.metadata]
    data.metadata["object"] = SimpleNamespace(positions=data.pos, separate=source.sin())
    tuple_cycle = ([],)
    tuple_cycle[0].append(tuple_cycle)
    data.metadata["tuple_cycle"] = tuple_cycle
    copied = _copy_data_to_device(data, "cpu", torch.float64)
    assert copied.pos.dtype == torch.float64
    assert copied.metadata["positions"] is copied.pos
    assert copied.metadata["self"] is copied.metadata
    assert copied.metadata["items"][0] is copied.metadata
    assert copied.metadata["tuple_cycle"][0][0] is copied.metadata["tuple_cycle"]
    assert copied.metadata["object"] is not data.metadata["object"]
    assert copied.metadata["object"].positions is copied.pos
    assert copied.metadata["object"].separate.dtype == torch.float64
    assert copied.metadata is not data.metadata
    torch.testing.assert_close(
        torch.autograd.grad(copied.pos.sum(), source)[0], torch.full_like(source, 2)
    )
    torch.testing.assert_close(
        torch.autograd.grad(copied.metadata["object"].separate.sum(), source)[0],
        source.cos(),
    )
    assert _validation_info_snapshot(data.metadata) is None
    cyclic = []
    cyclic.append(cyclic)
    assert _validation_info_snapshot(cyclic) is None


def test_failed_gpu_tail_event_does_not_retain_predictor(monkeypatch):
    class FailedEvent:
        def record(self, stream):
            raise RuntimeError("event recording failed")

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device: object())
    monkeypatch.setattr(torch.xpu, "Event", FailedEvent)

    def attempt():
        unit = MLIPPredictUnit.__new__(MLIPPredictUnit)
        unit.device = "xpu:0"
        reference = weakref.ref(unit)
        with (
            pytest.raises(RuntimeError, match="event recording failed"),
            _serialized_prediction(unit),
        ):
            pass
        with (
            pytest.raises(RuntimeError, match="construct a new predictor"),
            _serialized_prediction(unit),
        ):
            pass
        return reference

    reference = attempt()
    gc.collect()
    assert reference() is None

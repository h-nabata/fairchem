"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.calculate import _batch
from fairchem.core.components.batch_server import BatchConfig, BatchPredictServerMixin
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


def _data(value=1.0, task="omol", dtype=torch.float32):
    return AtomicData.from_ase(
        Atoms("H2", positions=[[0, 0, 0], [value, 0, 0]], cell=[8, 8, 8]),
        task_name=task,
        target_dtype=dtype,
    )


def _server():
    server = BatchPredictServerMixin()
    server.split_oom_batch = True
    return server


def test_split_outputs_own_storage_and_preserve_atom_system_shapes():
    batch = atomicdata_list_to_batch([_data(), _data(2)])
    predictions = {
        "energy": torch.tensor([1.0, 2.0]),
        "forces": torch.arange(12.0).reshape(4, 3),
        "stress": torch.arange(18.0).reshape(2, 3, 3),
    }
    results = _server()._split_predictions(predictions, batch)
    torch.testing.assert_close(results[1]["forces"], predictions["forces"][2:])
    assert results[0]["energy"].shape == (1,)
    assert results[0]["stress"].shape == (1, 3, 3)
    for tensor in predictions.values():
        tensor.fill_(-1)
    assert results[0]["energy"].item() == 1
    assert results[1]["stress"][0, 2, 2].item() == 17
    results[0]["energy"].fill_(100)
    assert results[1]["energy"].item() == 2


@pytest.mark.parametrize("prediction", [torch.tensor(1.0), [1.0], {"energy": 1}])
def test_split_invalid_output_type_has_actionable_error(prediction):
    with pytest.raises(TypeError, match="leading batch dimension"):
        _server()._split_predictions({"energy": prediction}, _data())


def test_split_bad_shape_and_declared_level_are_validated():
    batch = atomicdata_list_to_batch([_data(), _data(2)])
    with pytest.raises(ValueError, match="num_atoms 4"):
        _server()._split_predictions({"forces": torch.zeros(5, 3)}, batch)
    predictor = SimpleNamespace(
        dataset_to_tasks={"omol": [SimpleNamespace(property="forces", level="atom")]}
    )
    # This shape used to be silently interpreted as a per-system result.
    with pytest.raises(ValueError, match="Cannot split prediction"):
        _server()._split_predictions({"forces": torch.zeros(2, 3)}, batch, predictor)


class MixedTaskPredictor:
    device = torch.device("cpu")
    dataset_to_tasks: ClassVar[dict] = {
        task: [
            SimpleNamespace(property=prop, level=level) for prop, level in properties
        ]
        for task, properties in {
            "omol": [("energy", "system"), ("forces", "atom")],
            "omat": [("energy", "system"), ("stress", "system")],
        }.items()
    }

    def __init__(self):
        self.calls = []

    def predict(self, batch, undo_element_references):
        self.calls.append((tuple(batch.dataset), batch.pos.dtype))
        assert len(set(batch.dataset)) == 1
        energy = torch.zeros(batch.num_graphs, dtype=batch.pos.dtype)
        energy.index_add_(0, batch.batch, batch.pos.square().sum(-1))
        results = {"energy": energy}
        if batch.dataset[0] == "omol":
            results["forces"] = -2 * batch.pos
        else:
            results["stress"] = torch.eye(3).expand(batch.num_graphs, 3, 3)
        return results


def test_mixed_tasks_and_dtypes_restore_request_order_without_partial_outputs():
    predictor = MixedTaskPredictor()
    items = [
        _data(1),
        _data(2, "omat"),
        _data(3),
        _data(4, dtype=torch.float64),
    ]
    results = _server()._run_batched_inference(items, predictor, False)
    assert [r["energy"].item() for r in results] == [1, 4, 9, 16]
    assert [set(r) for r in results] == [
        {"energy", "forces"},
        {"energy", "stress"},
        {"energy", "forces"},
        {"energy", "forces"},
    ]
    assert results[-1]["energy"].dtype == torch.float64
    assert len(predictor.calls) == 3


@pytest.mark.parametrize("device", ["cpu", "xpu:2", "cuda:3"])
def test_oom_retry_releases_failed_frames_and_uses_predictor_device(
    monkeypatch, device
):
    # Virtual backend dispatch only: prediction and all tensors remain on CPU.
    events = []
    failed_allocations = []

    @contextmanager
    def device_scope(selected):
        assert str(selected) == device
        events.append("enter")
        yield
        events.append("exit")

    def empty_cache():
        assert all(ref() is None for ref in failed_allocations)
        events.append("cache")

    def forbidden():
        pytest.fail("CPU inference must not clear a CUDA allocator")

    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    if device != "cpu":
        backend = getattr(torch, torch.device(device).type)
        monkeypatch.setattr(backend, "device", device_scope)
        monkeypatch.setattr(backend, "empty_cache", empty_cache)

    class OOMPredictor(MixedTaskPredictor):
        def predict(self, batch, undo_element_references):
            if batch.num_graphs > 1:
                temporary = torch.ones(10)
                failed_allocations.append(weakref.ref(temporary))
                raise torch.OutOfMemoryError("synthetic OOM")
            assert all(ref() is None for ref in failed_allocations)
            return super().predict(batch, undo_element_references)

    predictor = OOMPredictor()
    predictor.device = torch.device(device)
    results = _server()._run_batched_inference([_data(1), _data(2)], predictor, False)
    assert [r["energy"].item() for r in results] == [1, 4]
    assert events == ([] if device == "cpu" else ["enter", "cache", "exit"])


def test_server_rejects_multi_system_requests_and_accepts_empty_queue():
    server = _server()
    predictor = MixedTaskPredictor()
    assert server._run_batched_inference([], predictor, False) == []
    batch = atomicdata_list_to_batch([_data(), _data()])
    with pytest.raises(ValueError, match="exactly one system"):
        server._run_batched_inference([batch], predictor, False)


@pytest.mark.parametrize("flags", ["false", 0, [False, 1], ["false", True]])
def test_reference_flags_do_not_accept_truthy_non_booleans(flags):
    with pytest.raises(TypeError, match="boolean"):
        _server()._normalize_undo_flags(flags, 2)


def test_group_results_must_match_request_count():
    server = _server()
    server._run_batched_inference = Mock(return_value=[{}])
    with pytest.raises(ValueError, match="shorter"):
        server._run_grouped_inference([_data(), _data()], [False, False], None)


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_size": 1.5},
        {"max_batch_size": True},
        {"max_batch_size": 0},
        {"batch_wait_timeout_s": np.nan},
        {"batch_wait_timeout_s": -1},
        {"batch_wait_timeout_s": "0.1"},
        {"split_oom_batch": "false"},
    ],
)
def test_batch_configuration_rejects_lossy_or_ambiguous_values(options):
    with pytest.raises(ValueError):
        BatchConfig(**options)


def test_reconfigure_validates_before_mutating_existing_config():
    server = _server()
    server.predict = Mock()
    with pytest.raises(ValueError, match="split_oom_batch"):
        server.reconfigure({"max_batch_size": 1, "split_oom_batch": "false"})
    server.predict.set_max_batch_size.assert_not_called()
    assert server.split_oom_batch is True


@pytest.mark.parametrize(
    "options",
    [
        {"concurrency_backend": "processes"},
        {"concurrency_backend_options": {"max_workers": 0}},
        {"max_batch_size": 0},
    ],
)
def test_batcher_invalid_local_options_never_start_remote_deployment(
    monkeypatch, options
):
    setup = Mock()
    monkeypatch.setattr(_batch, "setup_batch_predict_server", setup)
    with pytest.raises(ValueError):
        _batch.InferenceBatcher(None, **options)
    setup.assert_not_called()


def test_batcher_setup_failure_releases_executor_without_mutating_caller(monkeypatch):
    executor = Mock()
    monkeypatch.setattr(_batch, "_get_concurrency_backend", Mock(return_value=executor))
    setup = Mock(side_effect=RuntimeError("setup failed"))
    monkeypatch.setattr(_batch, "setup_batch_predict_server", setup)
    options = {}
    with pytest.raises(RuntimeError, match="setup failed"):
        _batch.InferenceBatcher(None, concurrency_backend_options=options)
    assert options == {}
    executor.shutdown.assert_called_once_with(wait=False)


@pytest.mark.parametrize("change", ["tasks", "dtype", "references", None])
def test_checkpoint_update_preserves_cached_client_interface(monkeypatch, change):
    def predictor():
        return SimpleNamespace(
            dataset_to_tasks={
                "omol": [SimpleNamespace(property="energy", level="system")]
            },
            inference_settings=InferenceSettings(),
            atom_refs={"omol": torch.tensor([1.0, 2.0])},
            form_elem_refs={"omol": {"H": 1.0}},
        )

    batcher = _batch.InferenceBatcher.__new__(_batch.InferenceBatcher)
    old, new = predictor(), predictor()
    batcher.predict_unit = old
    batcher.deployment_name = "test"
    batcher.__dict__["batch_predict_unit"] = object()
    update = Mock()
    monkeypatch.setattr(_batch, "update_served_predict_unit", update)
    if change == "tasks":
        new.dataset_to_tasks["omol"][0].property = "forces"
    elif change == "dtype":
        new.inference_settings.base_precision_dtype = "float64"
    elif change == "references":
        new.form_elem_refs["omol"]["H"] = 2.0
    if change is None:
        batcher.update_checkpoint(new)
        update.assert_called_once_with("test", new)
        assert batcher.predict_unit is new
        assert "batch_predict_unit" not in batcher.__dict__
    else:
        with pytest.raises(ValueError, match="cached client metadata"):
            batcher.update_checkpoint(new)
        update.assert_not_called()
        assert batcher.predict_unit is old

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import asyncio
import weakref
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.calculate import _batch
from fairchem.core.components import batch_server
from fairchem.core.components.batch_server import (
    BatchConfig,
    BatchPredictServerMixin,
    ModelSpec,
    MultiplexedBatchPredictServer,
)
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


@pytest.mark.parametrize("initial_gpu_count", [0, 1])
def test_deployment_defaults_do_not_leak_between_apps(initial_gpu_count):
    actor_options = {"num_cpus": 0.5}
    config = batch_server.DeploymentConfig(ray_actor_options=actor_options)
    first = batch_server._prepare_deployment_config(config, initial_gpu_count, "first")
    second = batch_server._prepare_deployment_config(
        config, 1 - initial_gpu_count, "second"
    )

    assert config.ray_actor_options is actor_options
    assert actor_options == {"num_cpus": 0.5}
    assert first.ray_actor_options["num_gpus"] == initial_gpu_count
    assert second.ray_actor_options["num_gpus"] == 1 - initial_gpu_count


@pytest.mark.parametrize("key", ["num_cpus", "num_gpus"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.5, "1", True])
def test_invalid_actor_resources_fail_before_startup(key, value):
    with pytest.raises(ValueError, match=key):
        batch_server._prepare_deployment_config(
            {"ray_actor_options": {key: value}}, 0, "test"
        )


@pytest.mark.parametrize(
    "autoscaling",
    [
        dict(min_replicas=3, max_replicas=8),
        SimpleNamespace(min_replicas=3, max_replicas=8),
    ],
)
def test_startup_uses_actual_autoscaling_minimum(autoscaling):
    assert batch_server._effective_replicas({"autoscaling_config": autoscaling}) == 3


def test_batch_size_counts_nodes_without_tensor_values():
    class Sample:
        def __init__(self, num_nodes):
            self.num_nodes = num_nodes

        @property
        def natoms(self):
            raise AssertionError("Batch sizing must not inspect tensor values")

    assert batch_server._batch_atom_count([Sample(2), Sample(7)]) == 9
    assert batch_server._batch_atom_count([]) == 0


def test_concurrent_batch_apps_initialize_ray_once(monkeypatch):
    entered_init = Event()
    entered_second_call = Event()
    release_init = Event()
    state = {"initialized": False, "init_calls": 0, "start_calls": 0}

    def init(**kwargs):
        assert kwargs["num_cpus"] == 1
        assert type(kwargs["num_cpus"]) is int
        state["init_calls"] += 1
        entered_init.set()
        assert release_init.wait(timeout=5)
        state["initialized"] = True

    def start(**kwargs):
        assert kwargs["proxy_location"] == "Disabled"
        state["start_calls"] += 1

    monkeypatch.setattr(
        batch_server,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: state["initialized"],
            LoggingConfig=dict,
            init=init,
            available_resources=lambda: {"CPU": 4},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        batch_server,
        "serve",
        SimpleNamespace(schema=SimpleNamespace(LoggingConfig=dict), start=start),
    )

    class StartupLock:
        # Observe that the second app reaches the shared startup lock while
        # the first app is still inside ray.init.
        def __init__(self):
            self.lock = Lock()

        def __enter__(self):
            if entered_init.is_set():
                entered_second_call.set()
            self.lock.acquire()

        def __exit__(self, *args):
            self.lock.release()

    monkeypatch.setattr(batch_server, "_RAY_STARTUP_LOCK", StartupLock())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(batch_server._init_ray_and_serve, {"num_cpus": 0.5}, 1)
        try:
            assert entered_init.wait(timeout=5)
            second = executor.submit(
                batch_server._init_ray_and_serve, {"num_cpus": 0.5}, 1
            )
            assert entered_second_call.wait(timeout=5)
        finally:
            release_init.set()
        first.result()
        second.result()

    assert state == {"initialized": True, "init_calls": 1, "start_calls": 2}


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


def test_split_heterogeneous_outputs_avoids_atom_masks_and_owns_storage():
    structures = [
        AtomicData.from_ase(Atoms("H" * size), task_name="omol") for size in (1, 3, 2)
    ]
    batch = atomicdata_list_to_batch(structures)
    predictions = {
        "energy": torch.arange(3.0),
        # Exercise noncontiguous atom outputs, too.
        "forces": torch.arange(36.0).reshape(3, 12)[:, ::2].T,
    }
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        results = _server()._split_predictions(predictions, batch)
    operations = {event.key for event in p.key_averages()}
    # Boolean indexing scans the whole batch for each request, and its dynamic
    # nonzero output synchronizes accelerators. Contiguous system slices avoid it.
    assert "aten::nonzero" not in operations
    assert "aten::index" not in operations
    start = 0
    for result, size in zip(results, (1, 3, 2), strict=True):
        torch.testing.assert_close(
            result["forces"], predictions["forces"][start : start + size]
        )
        assert result["forces"].untyped_storage().nbytes() == size * 3 * 4
        start += size
    predictions["forces"].fill_(-1)
    assert results[1]["forces"].min() >= 0


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


@pytest.mark.parametrize("native,groups", [(False, [1, 1, 2]), (True, [4])])
def test_server_isolates_only_reference_lookup_single_atoms(native, groups):
    predictor = MixedTaskPredictor()
    predictor.model = SimpleNamespace(
        module=SimpleNamespace(supports_single_atoms=native)
    )
    items = [
        AtomicData.from_ase(Atoms("He", cell=[8] * 3, pbc=pbc), task_name="omol")
        for pbc in (False, True, False)
    ]
    items.append(_data(2))
    results = _server()._run_batched_inference(items, predictor, False)
    assert sorted(len(datasets) for datasets, _ in predictor.calls) == groups
    assert [r["energy"].item() for r in results] == [0, 0, 0, 4]


def test_actual_uma_hessians_are_isolated_per_server_request():
    from tests.core.calculate.test_batch_interoperability import _uma

    predictor = _uma()
    predictor.inference_settings.requested_properties = ("hessian",)
    items = [_data(1), _data(2)]
    expected = [predictor.predict(item, False)["hessian"] for item in items]
    actual = _server()._run_batched_inference(items, predictor, False)
    for result, oracle in zip(actual, expected, strict=True):
        assert result["hessian"].shape == (1, 6, 6)
        torch.testing.assert_close(result["hessian"], oracle)


@pytest.mark.parametrize("default_requested", [None, ("energy",)])
def test_actual_uma_per_request_properties_override_defaults_without_cross_talk(
    default_requested,
):
    from tests.core.calculate.test_batch_interoperability import _uma

    predictor = _uma()
    predictor.inference_settings.requested_properties = default_requested
    items = [_data(1), _data(2), _data(0.8)]
    for data, properties in zip(
        items, [("hessian",), ("hessian",), ("energy",)], strict=True
    ):
        data._requested_properties = properties
    expected = [
        predictor.predict(data, False, requested_properties=data._requested_properties)
        for data in items
    ]

    actual = _server()._run_batched_inference(items, predictor, False)

    for result, oracle in zip(actual, expected, strict=True):
        assert result.keys() == oracle.keys()
        for name in oracle:
            torch.testing.assert_close(result[name], oracle[name])
    assert items[0]._requested_properties == ("hessian",)


def test_server_rejects_property_selection_for_predictors_without_support():
    predictor = MixedTaskPredictor()
    data = _data()
    data._requested_properties = ("energy",)
    with pytest.raises(ValueError, match="does not support requested_properties"):
        _server()._run_batched_inference([data], predictor, False)
    assert not predictor.calls


def test_server_none_property_field_uses_defaults_without_leaking_metadata():
    data = _data()
    data._requested_properties = None

    def predict(batch, undo_element_references):
        assert "_requested_properties" not in batch
        return {"energy": torch.ones(batch.num_graphs)}

    predictor = SimpleNamespace(predict=predict)
    result = _server()._run_batched_inference([data], predictor, False)
    assert result[0]["energy"].item() == 1
    assert data._requested_properties is None


def test_declared_hessians_use_singletons_but_explicit_energy_can_batch():
    calls = []

    def predict(batch, undo_element_references):
        calls.append(batch.num_graphs)
        return {"energy": torch.zeros(batch.num_graphs)}

    predictor = SimpleNamespace(
        predict=predict,
        inference_settings=InferenceSettings(),
        dataset_to_tasks={
            "omol": [SimpleNamespace(property="hessian", level="system")]
        },
    )
    _server()._run_batched_inference([_data(), _data(2)], predictor, False)
    assert calls == [1, 1]
    calls.clear()
    predictor.inference_settings.requested_properties = ("energy",)
    _server()._run_batched_inference([_data(), _data(2)], predictor, False)
    assert calls == [2]


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


@pytest.mark.parametrize("device", ["cpu", "cuda:3", "xpu:2"])
def test_server_releases_collated_storage_before_cleanup_and_next_allocation(
    monkeypatch, device
):
    # Accelerator cleanup is virtual; ownership is checked with real CPU data.
    batch_refs = []
    original_collate = batch_server.atomicdata_list_to_batch

    def collate(*args, **kwargs):
        assert all(ref() is None for ref in batch_refs)
        batch = original_collate(*args, **kwargs)
        batch_refs.extend((weakref.ref(batch), weakref.ref(batch.pos)))
        return batch

    monkeypatch.setattr(batch_server, "atomicdata_list_to_batch", collate)

    @contextmanager
    def device_scope(selected):
        assert str(selected) == device
        yield

    def empty_cache():
        assert all(ref() is None for ref in batch_refs)

    if device != "cpu":
        backend = getattr(torch, torch.device(device).type)
        monkeypatch.setattr(backend, "device", device_scope)
        monkeypatch.setattr(backend, "empty_cache", empty_cache)

    class OOMPredictor(MixedTaskPredictor):
        def predict(self, batch, undo_element_references):
            if batch.num_graphs > 1:
                raise torch.OutOfMemoryError("synthetic OOM")
            return super().predict(batch, undo_element_references)

    predictor = OOMPredictor()
    predictor.device = torch.device(device)
    items = [_data(1), _data(2)]
    results = _server()._run_batched_inference(items, predictor, False)
    assert [r["energy"].item() for r in results] == [1, 4]
    assert len(batch_refs) == 6  # Failed full batch, then two successful singletons.
    assert all(ref() is None for ref in batch_refs)
    assert [data.pos[1, 0].item() for data in items] == [1, 2]


def _multiplexed_server():
    server_class = MultiplexedBatchPredictServer.func_or_class
    server = object.__new__(server_class)
    server._specs = OrderedDict()
    server._active_spec_counts = defaultdict(int)
    server._spec_capacity = 12
    server.split_oom_batch = True
    return server, server_class.predict.__wrapped__


def test_multiplexed_switch_does_not_pin_an_evicted_previous_predictor():
    server, predict = _multiplexed_server()
    owners = []
    refs = []

    async def get_model(model_id):
        # Emulate eviction during the next asynchronous load. Other Serve
        # requests may have changed the LRU order since the preceding group.
        owners.clear()
        assert all(ref() is None for ref in refs)
        model = MixedTaskPredictor()
        owners.append(model)
        refs.append(weakref.ref(model))
        return model

    server.get_model = get_model
    specs = [ModelSpec(f"model-{index}", device="cpu") for index in range(2)]
    results = asyncio.run(
        predict(
            server,
            [_data(1), _data(2)],
            [spec.model_id for spec in specs],
            specs,
            [False, False],
        )
    )
    owners.clear()
    assert all(ref() is None for ref in refs)
    assert [r["energy"].item() for r in results] == [1, 4]


@pytest.mark.parametrize("invalid_spec", ["different_model", "not_a_spec"])
def test_multiplexed_validates_every_spec_before_running_any_group(invalid_spec):
    server, predict = _multiplexed_server()
    original = ModelSpec("same-model", device="cpu")
    replacement = (
        ModelSpec("different-model", device="cpu")
        if invalid_spec == "different_model"
        else object()
    )
    loaded = []

    async def get_model(model_id):
        loaded.append(model_id)
        return MixedTaskPredictor()

    server.get_model = get_model
    with pytest.raises((ValueError, TypeError), match="ModelSpec"):
        asyncio.run(
            predict(
                server,
                [_data(1), _data(2)],
                [original.model_id, original.model_id],
                [original, replacement],
                False,
            )
        )
    assert not loaded


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


def test_server_can_split_an_oom_during_collation(monkeypatch):
    sizes, refs = [], []
    original = batch_server.atomicdata_list_to_batch

    def collate(items, **kwargs):
        assert all(ref() is None for ref in refs)
        sizes.append(len(items))
        if len(items) > 1:
            temporary = torch.zeros(128)
            refs.append(weakref.ref(temporary))
            raise torch.OutOfMemoryError("collate")
        return original(items, **kwargs)

    monkeypatch.setattr(batch_server, "atomicdata_list_to_batch", collate)
    output = _server()._run_batched_inference(
        [_data(1), _data(2)], MixedTaskPredictor(), False
    )
    assert sizes == [2, 1, 1]
    assert [result["energy"].item() for result in output] == [1, 4]


@pytest.mark.parametrize("failure", ["model", "small", "large", None])
def test_autobatch_uses_predictor_ordinal_and_only_successful_measurements(
    monkeypatch, failure
):
    device = torch.device("cuda:3")
    active = []

    @contextmanager
    def scope(ordinal):
        assert ordinal == device
        active.append(device)
        yield
        active.pop()

    monkeypatch.setattr(torch.cuda, "device", scope)

    def exact(ordinal):
        assert active == [device]
        assert ordinal == device
        return 0

    def empty_cache():
        assert active == [device]

    monkeypatch.setattr(torch.cuda, "synchronize", exact)
    monkeypatch.setattr(torch.cuda, "memory_reserved", exact)
    monkeypatch.setattr(torch.cuda, "memory_allocated", exact)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda ordinal: (exact(ordinal), 2**30)
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    def predict(data, **kwargs):
        if failure == "model":
            raise RuntimeError("bad weights")
        if failure == "small" or (failure == "large" and data.num_nodes > 2):
            raise torch.OutOfMemoryError("allocator")

    predictor = SimpleNamespace(device=device, predict=predict)
    config = batch_server.AutobatchConfig(
        min_batch_size=4, max_batch_size_cap=8, probe_steps=2
    )
    if failure is None:
        result = batch_server.probe_optimal_batch_size(predictor, [_data()], config)
        assert result.max_batch_size == int(8 * config.backoff_factor)
        assert result.median_latency_s > 0
    else:
        error = RuntimeError if failure == "model" else torch.OutOfMemoryError
        message = {
            "model": "bad weights",
            "small": "single-request",
            "large": "configured probe range",
        }[failure]
        with pytest.raises(error, match=message):
            batch_server.probe_optimal_batch_size(predictor, [_data()], config)


@pytest.mark.parametrize("first", ["weights", "config"])
def test_registry_serializes_concurrent_model_and_batch_updates(monkeypatch, first):
    entered, release, attempted = Event(), Event(), Event()
    deployed = []
    name = "concurrent"
    original = batch_server.BatchConfig(max_batch_size=16)
    new = batch_server.BatchConfig(max_batch_size=32)

    def run(app, **kwargs):
        deployed.append(app)
        if len(deployed) == 2:
            entered.set()
            assert release.wait(5)
        return object()

    class Deployment:
        def options(self, **options):
            return SimpleNamespace(
                bind=lambda ref, **kwargs: (
                    ref,
                    options["user_config"]["max_batch_size"],
                )
            )

    monkeypatch.setattr(batch_server, "_DEPLOYED_APPS", {})
    monkeypatch.setattr(batch_server.ray, "put", lambda unit: unit.name)
    monkeypatch.setattr(batch_server.serve, "run", run)
    batch_server._deploy_app(Deployment(), {}, ("old",), {}, original, name, "/")
    model = SimpleNamespace(device="cpu", name="new")
    update = {
        "weights": lambda: batch_server.update_served_predict_unit(name, model),
        "config": lambda: batch_server.update_batch_config(name, new),
    }
    last = "config" if first == "weights" else "weights"

    def second():
        attempted.set()
        update[last]()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_job = executor.submit(update[first])
        try:
            assert entered.wait(5)
            second_job = executor.submit(second)
            assert attempted.wait(5)
        finally:
            release.set()
        first_job.result(timeout=5)
        second_job.result(timeout=5)
    assert deployed[-1] == ("new", 32)
    assert batch_server._DEPLOYED_APPS[name].bind_args == ("new",)
    assert batch_server._DEPLOYED_APPS[name].batch_config == new


def test_batcher_delete_releases_registry_after_success_and_preserves_retry(
    monkeypatch,
):
    name = "test-delete"
    owner = object()
    handle = object()
    record = SimpleNamespace(owner=owner)
    registry = {name: record}
    delete = Mock(side_effect=[RuntimeError("temporarily unavailable"), None])
    monkeypatch.setattr(batch_server, "_DEPLOYED_APPS", registry)
    monkeypatch.setattr(batch_server.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(batch_server.serve, "delete", delete)
    batcher = _batch.InferenceBatcher.__new__(_batch.InferenceBatcher)
    batcher.deployment_name = name
    batcher._deployment_owner = owner
    batcher.predict_server_handle = handle
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        batcher.delete()
    assert batcher.predict_server_handle is handle
    assert registry[name] is record
    batcher.delete()
    assert name not in registry
    assert batcher.predict_server_handle is None
    registry[name] = SimpleNamespace(owner=object())
    batch_server.delete_batch_predict_server(name, owner=owner)
    assert name in registry
    assert delete.call_count == 2

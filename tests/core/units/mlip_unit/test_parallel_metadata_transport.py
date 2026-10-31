"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit import predict as module
from fairchem.core.units.mlip_unit.predict import (
    MLIPWorkerLocal,
    ParallelMLIPPredictUnit,
)
from tests.core.units.mlip_unit.test_xpu_preparation import _data


class _Reference:
    """
    Simple metadata-sensitive calculation; the real transport runs unchanged.
    """

    def __init__(self, barrier, merge):
        self.barrier = barrier
        self.inference_settings = InferenceSettings(merge_mole=merge)
        self.requests = []
        self.fail = False

    def predict(self, data):
        self.requests.append(data.clone())
        if self.barrier is not None:
            self.barrier.wait(timeout=10)
        if self.fail:
            raise RuntimeError("model failure")
        if data.charge.any():
            self.inference_settings.merge_mole = False
        return {
            "energy": (
                (data.pos * data.atomic_numbers[:, None]).sum()
                + data.charge.sum()
                + data.spin.sum()
                + data.pbc.sum()
            ).reshape(1)
        }


@pytest.fixture()
def transport(monkeypatch):
    units = []
    executor = ThreadPoolExecutor(max_workers=2)
    root_thread = threading.get_ident()
    messages = queue.Queue()
    events = []
    actors = []
    killed = []
    removed = []
    group = object()

    def broadcast(value, src, group=None):
        assert src == 0
        if threading.get_ident() == root_thread:
            events.append(
                ("control" if group is not None else "geometry", value.clone())
            )
            if actors:
                messages.put(("tensor", value.clone()))
        else:
            kind, received = messages.get(timeout=10)
            assert kind == "tensor"
            value.copy_(received)

    def objects(value, src, group=None):
        assert src == 0
        assert group is not None
        if threading.get_ident() == root_thread:
            events.append(("refresh", value[0]))
            if actors:
                messages.put(("object", copy.deepcopy(value[0])))
        else:
            kind, received = messages.get(timeout=10)
            assert kind == "object"
            value[0] = received

    def get(value, timeout=None):
        if isinstance(value, list):
            return [get(item, timeout) for item in value]
        return value.result(timeout=10 if timeout is None else timeout)

    def kill(actor, no_restart):
        killed.append(actor)
        if actor in actors and actor.future is not None and not actor.future.done():
            messages.put(("tensor", torch.tensor([MLIPWorkerLocal._STOP])))

    monkeypatch.setattr(module.dist, "broadcast", broadcast)
    monkeypatch.setattr(module.dist, "broadcast_object_list", objects)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(module.gp_utils, "initialized", lambda: False)
    monkeypatch.setattr(module.ray, "put", lambda value: value.clone())
    monkeypatch.setattr(module.ray, "get", get)
    monkeypatch.setattr(
        module.ray,
        "wait",
        lambda values, **kwargs: (
            [value for value in values if value.done()],
            [value for value in values if not value.done()],
        ),
    )
    monkeypatch.setattr(module.ray, "kill", kill)
    monkeypatch.setattr(module.ray.util, "remove_placement_group", removed.append)

    def make(merge=True, workers=2):
        barrier = threading.Barrier(workers) if workers > 1 else None
        unit = ParallelMLIPPredictUnit.__new__(ParallelMLIPPredictUnit)
        unit.inference_settings = InferenceSettings(merge_mole=merge)
        unit.atomic_data_on_device = None
        unit._input_snapshot = None
        unit._use_broadcast = merge
        unit._broadcast_command = torch.empty(1, dtype=torch.int64)
        unit._worker_futures = []
        unit._closed = False
        unit._resources_released = False
        unit._prediction_lock = threading.RLock()
        unit._rank0_reservation = object()
        unit._placement_groups = [object()]
        unit.workers = []
        references = []
        for rank in range(workers):
            worker = MLIPWorkerLocal(rank, workers, {}, 12345, "127.0.0.1")
            worker.is_setup = True
            worker.device = "cpu"
            worker._broadcast_group = group
            worker.predict_unit = _Reference(barrier, merge)
            references.append(worker.predict_unit)
            if rank == 0:
                unit.local_rank0 = worker
            else:
                actor = SimpleNamespace(worker=worker, future=None)

                def remote(data, use_nccl, actor=actor):
                    actor.future = executor.submit(actor.worker.predict, data, use_nccl)
                    return actor.future

                actor.predict = SimpleNamespace(remote=remote)
                unit.workers.append(actor)
                actors.append(actor)
        units.append(unit)
        return unit, references, events, killed, removed

    yield make
    for unit in units:
        unit.close()
    executor.shutdown(wait=True)


def _assert_request(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected:
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value)
        else:
            assert ParallelMLIPPredictUnit._same_metadata(actual[key], value)


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("merge", [False, True])
def test_parallel_original_paths_and_owned_geometry(transport, merge, workers):
    unit, references, events, killed, removed = transport(merge, workers)
    data = _data()
    first = data.clone()
    unit.predict(data)
    data.pos.add_(0.1)
    expected = data.clone()
    result = unit.predict(data)
    for reference in references:
        _assert_request(reference.requests[0], first)
        _assert_request(reference.requests[1], expected)
    _assert_request(data, expected)
    torch.testing.assert_close(
        result["energy"], _Reference(None, False).predict(data)["energy"]
    )
    assert unit.atomic_data_on_device.pos.data_ptr() != data.pos.data_ptr()
    if merge:
        assert [kind for kind, _ in events] == ["control", "geometry", "geometry"]
        assert unit._input_snapshot["pos"] == unit._geometry_signature(data.pos)
    else:
        assert not events
        assert not unit._worker_futures
    unit.close()
    unit.close()
    assert len(killed) == workers
    assert len(removed) == 1
    assert unit.local_rank0.predict_unit is None
    with pytest.raises(RuntimeError, match="closed"):
        unit.predict(data)


@pytest.mark.parametrize(
    "change",
    ["order", "charge", "spin", "dataset", "pbc", "count", "dtype", "extra", "edges"],
)
def test_stream_refreshes_all_metadata_and_keeps_streaming_after_fallback(
    transport, change
):
    unit, references, events, _, _ = transport()
    data = _data()
    unit.predict(data)
    if change == "order":
        data.atomic_numbers = data.atomic_numbers[[1, 0, 2]]
        data.pos = data.pos[[1, 0, 2]]
    elif change == "charge":
        data.charge.add_(1)
    elif change == "spin":
        data.spin.add_(2)
    elif change == "dataset":
        data.dataset[0] = "omat"
    elif change == "pbc":
        data.pbc.fill_(True)
        data.cell = torch.eye(3).reshape(1, 3, 3) * 10
    elif change == "count":
        atoms = Atoms("H4O2", positions=np.arange(18).reshape(6, 3), cell=[20] * 3)
        atoms.info.update(charge=0, spin=1)
        data = AtomicData.from_ase(atoms, task_name="omol")
    elif change == "dtype":
        data.to(dtype=torch.float64)
    elif change == "extra":
        data.options = {"nested": [torch.tensor([3]), np.array([1, 2])]}
    elif change == "edges":
        data.edge_index = torch.tensor([[0, 1], [1, 0]])
        data.cell_offsets = torch.zeros(2, 3)
        data.nedges.fill_(2)
    expected = data.clone()
    unit.predict(data)
    for reference in references:
        _assert_request(reference.requests[-1], expected)
    _assert_request(data, expected)
    assert [kind for kind, _ in events] == ["control", "refresh"]
    assert events[0][1].item() == MLIPWorkerLocal._REFRESH
    data.pos.add_(0.02)
    unit.predict(data)
    assert [kind for kind, _ in events[-3:]] == ["control", "geometry", "geometry"]
    assert len(references[1].requests) == 3
    if change == "charge":
        assert not unit.inference_settings.merge_mole
        assert unit._use_broadcast


@pytest.mark.parametrize("invalid", ["count", "nan", "pickle"])
def test_rejected_request_does_not_announce_collective_and_can_recover(
    transport, invalid
):
    unit, references, events, _, _ = transport()
    data = _data()
    unit.predict(data)
    invalid_data = data.clone()
    if invalid == "count":
        invalid_data.natoms.add_(1)
    elif invalid == "nan":
        invalid_data.pos[0, 0] = float("nan")
    else:
        invalid_data.options = lambda: None
    with pytest.raises(
        (AssertionError, ValueError, AttributeError, pickle.PicklingError)
    ):
        unit.predict(invalid_data)
    assert not events
    assert not unit._closed
    unit.predict(data)
    assert len(references[1].requests) == 2
    assert [kind for kind, _ in events] == ["control", "geometry", "geometry"]


def test_in_place_nested_metadata_changes_are_not_hidden_by_snapshot(transport):
    unit, references, events, _, _ = transport()
    data = _data()
    data.options = {"list": [torch.tensor([2]), np.array([3])]}
    unit.predict(data)
    data.options["list"][0].add_(1)
    data.options["list"][1] += 1
    unit.predict(data)
    assert [kind for kind, _ in events] == ["control", "refresh"]
    _assert_request(references[1].requests[-1], data)


def test_model_failure_closes_stream_and_releases_resources(transport):
    unit, references, _, killed, removed = transport()
    data = _data()
    unit.predict(data)
    for reference in references:
        reference.fail = True
    with pytest.raises(RuntimeError, match="model failure"):
        unit.predict(data)
    assert unit._closed
    assert unit._resources_released
    assert unit.atomic_data_on_device is None
    assert unit._input_snapshot is None
    assert len(killed) == 2
    assert len(removed) == 1
    with pytest.raises(RuntimeError, match="closed"):
        unit.predict(data)


def test_close_before_first_request_releases_actors(transport):
    unit, _, events, killed, removed = transport()
    unit.close()
    assert not events
    assert len(killed) == 2
    assert len(removed) == 1


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_control_group_is_created_once_on_cpu(monkeypatch, backend):
    worker = MLIPWorkerLocal(0, 1, {}, 12345, "127.0.0.1")
    worker.is_setup = True
    worker.device = "cpu"
    worker.predict_unit = _Reference(None, True)
    group = object()
    creations = []
    monkeypatch.setattr(module.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(module.dist, "group", SimpleNamespace(WORLD=group))

    def new_group(backend):
        creations.append(backend)
        return group

    monkeypatch.setattr(module.dist, "new_group", new_group)
    worker.predict(_data(), use_nccl=True)
    worker.predict(_data(), use_nccl=True)
    assert worker._broadcast_group is group
    assert creations == (["gloo"] if backend == "nccl" else [])


def test_noncontiguous_positions_use_matching_owned_transport_buffers(transport):
    unit, references, _, _, _ = transport()
    data = _data()
    data.pos = data.pos.T.contiguous().T
    assert not data.pos.is_contiguous()
    unit.predict(data)
    data.pos.add_(0.04)
    unit.predict(data)
    for reference in references:
        _assert_request(reference.requests[-1], data)
        assert reference.requests[-1].pos.is_contiguous()
    assert not data.pos.is_contiguous()


def test_completed_worker_error_is_observed_before_next_collective(transport):
    from concurrent.futures import Future

    unit, _, events, killed, _ = transport()
    data = _data()
    unit.predict(data)
    failed = Future()
    failed.set_exception(RuntimeError("remote worker failure"))
    unit._worker_futures = [failed]
    with pytest.raises(RuntimeError, match="remote worker failure"):
        unit.predict(data)
    assert not events
    assert len(killed) == 2


@pytest.mark.parametrize("custom_validator", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cpu:0", "cuda", "cuda:3"])
def test_constructor_releases_standard_uma_model_but_keeps_custom_validator(
    monkeypatch, custom_validator, device
):
    import gc
    import weakref
    from types import MethodType

    from fairchem.core.units.mlip_unit.api.inference import validate_uma_atoms_data
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    references = {}
    reservations = []

    def make_predictor(**kwargs):
        predictor = _public_predictor()
        model = predictor.model.module
        if custom_validator:

            def validate(self, atoms, task_name):
                atoms.info["custom"] = self.model_id
                validate_uma_atoms_data(atoms, task_name)

            model.validate_atoms_data = MethodType(validate, model)
        references["predictor"] = weakref.ref(predictor)
        references["model"] = weakref.ref(model)
        return predictor

    actor = SimpleNamespace(
        get_device_for_local_rank=SimpleNamespace(remote=lambda: "cpu"),
        get_cuda_placement=SimpleNamespace(
            remote=lambda: {"node_id": "node0", "device_uuid": "GPU-0"}
        ),
    )
    monkeypatch.setattr(
        module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node0"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="GPU-0"),
    )
    monkeypatch.setattr(module, "MLIPPredictUnit", make_predictor)

    def options(**kwargs):
        reservations.append(kwargs["num_gpus"])
        return SimpleNamespace(remote=lambda *args, **kwargs: actor)

    monkeypatch.setattr(module, "MLIPWorker", SimpleNamespace(options=options))
    monkeypatch.setattr(module, "get_free_port", lambda: 12345)
    monkeypatch.setattr(module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "kill", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(
        module.ray.util,
        "placement_group",
        lambda *args, **kwargs: SimpleNamespace(ready=lambda: "ready"),
    )
    monkeypatch.setattr(module.ray.util, "remove_placement_group", lambda group: None)
    monkeypatch.setenv(module.CURRENT_DEVICE_TYPE_STR, "cpu")
    unit = ParallelMLIPPredictUnit("unused.pt", device=device, num_workers=1)
    device_type = torch.device(device).type
    assert reservations == [int(device_type == "cuda")]
    assert unit.local_rank0.predictor_config["device"] == device_type
    gc.collect()
    assert references["predictor"]() is None
    assert (references["model"]() is not None) == custom_validator
    for task in ("omol", "omat"):
        atoms = Atoms("H2")
        expected = atoms.copy()
        validate_uma_atoms_data(expected, task)
        unit.validate_atoms_data(atoms, task)
        if custom_validator:
            assert atoms.info.pop("custom") == references["model"]().model_id
        assert atoms.info == expected.info
    with pytest.raises(TypeError, match="charge"):
        unit.validate_atoms_data(Atoms("H2", info={"charge": True}), "omol")
    unit.close()
    gc.collect()
    assert references["model"]() is None


def test_cleanup_attempts_all_resources_and_allows_retry(transport, monkeypatch):
    unit, _, _, _, removed = transport()
    attempts = []

    def kill(actor, no_restart):
        attempts.append(actor)
        if len(attempts) == 1:
            raise RuntimeError("temporary Ray failure")

    monkeypatch.setattr(module.ray, "kill", kill)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        unit.close()
    assert len(attempts) == 2
    assert len(removed) == 1
    assert unit.local_rank0.predict_unit is None
    assert not unit._resources_released
    unit.close()
    assert unit._resources_released


def test_sparse_optional_metadata_uses_full_refresh(transport):
    unit, references, events, _, _ = transport()
    data = _data()
    data.sparse_options = torch.eye(3).to_sparse()
    unit.predict(data)
    unit.predict(data)
    assert [kind for kind, _ in events] == ["control", "refresh"]
    torch.testing.assert_close(
        references[1].requests[-1].sparse_options, data.sparse_options
    )


@pytest.mark.parametrize("device", ["cpu:0", "cuda:4"])
def test_worker_setup_uses_ray_local_ordinal(monkeypatch, device):
    config = {"device": device}
    worker = MLIPWorkerLocal(1, 2, config, 12345, "127.0.0.1")
    device_type = torch.device(device).type
    assignment = []
    backends = []
    instantiated = []
    monkeypatch.setattr(module, "setup_env_local_multi_gpu", lambda *args: None)
    monkeypatch.setattr(
        module,
        "assign_device_for_local_rank",
        lambda cpu, rank: assignment.append((cpu, rank)),
    )
    monkeypatch.setattr(
        module,
        "get_device_for_local_rank",
        lambda: "cpu" if device_type == "cpu" else "cuda:0",
    )
    monkeypatch.setattr(
        module.dist, "init_process_group", lambda **kwargs: backends.append(kwargs)
    )
    monkeypatch.setattr(module.ray, "get_gpu_ids", lambda: [4])
    monkeypatch.setattr(
        module.hydra.utils, "instantiate", lambda value: instantiated.append(value)
    )
    worker._distributed_setup()
    assert assignment == [(device_type == "cpu", 0)]
    assert backends[0]["backend"] == ("gloo" if device_type == "cpu" else "nccl")
    assert instantiated == [{"device": device_type}]
    assert config["device"] == device
    assert worker.device == ("cpu" if device_type == "cpu" else "cuda:0")


@pytest.mark.parametrize(
    "failure", ["node", "uuid", "missing_uuid", "ready", "actor", "later_actor"]
)
def test_constructor_rejects_unsafe_cuda_placement_and_unwinds(monkeypatch, failure):
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    groups, killed, removed, launches = [], [], [], []
    placement = {"node_id": "node0", "device_uuid": "GPU-0"}
    if failure == "node":
        placement["node_id"] = "node1"
    if failure == "uuid":
        placement["device_uuid"] = "GPU-1"
    actor = SimpleNamespace(
        get_cuda_placement=SimpleNamespace(remote=lambda: placement),
        get_device_for_local_rank=SimpleNamespace(remote=lambda: "cuda:0"),
    )

    def remote(*args, **kwargs):
        launches.append(args[0])
        if failure == "actor" or (failure == "later_actor" and args[0] == 1):
            raise RuntimeError("actor launch failed")
        return actor

    def placement_group(*args, **kwargs):
        def ready():
            if failure == "ready":
                raise RuntimeError("placement ready failed")
            return "ready"

        group = SimpleNamespace(ready=ready)
        groups.append(group)
        return group

    monkeypatch.setattr(module, "MLIPPredictUnit", lambda **kwargs: _public_predictor())
    monkeypatch.setattr(
        module,
        "MLIPWorker",
        SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=remote)),
    )
    monkeypatch.setattr(module, "get_free_port", lambda: 12345)
    monkeypatch.setattr(module.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(
        module.ray, "kill", lambda actor, **kwargs: killed.append(actor)
    )
    monkeypatch.setattr(
        module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node0"),
    )
    monkeypatch.setattr(module.ray.util, "placement_group", placement_group)
    monkeypatch.setattr(module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(module.ray.util, "remove_placement_group", removed.append)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            uuid=None if failure == "missing_uuid" else "GPU-0"
        ),
    )
    monkeypatch.setenv(module.CURRENT_DEVICE_TYPE_STR, "cpu")
    with pytest.raises(RuntimeError):
        ParallelMLIPPredictUnit("unused.pt", device="cuda:3", num_workers=2)
    assert groups == removed
    assert len(killed) == (0 if failure in {"ready", "actor"} else 1)
    if failure in {"node", "uuid", "missing_uuid"}:
        assert launches == [0]


def test_worker_placement_reports_local_cuda_zero_uuid(monkeypatch):
    calls = []
    worker = MLIPWorkerLocal(0, 1, {}, 12345, "127.0.0.1")
    monkeypatch.setattr(
        module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: "node0"),
    )

    def properties(device):
        calls.append(device)
        return SimpleNamespace(uuid="GPU-0123")

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    assert worker.get_cuda_placement() == {
        "node_id": "node0",
        "device_uuid": "GPU-0123",
    }
    assert calls == [0]


def test_failed_model_initialization_releases_already_owned_process_group(
    transport, monkeypatch
):
    unit, _, _, _, _ = transport(workers=1)
    unit.local_rank0.is_setup = False
    initialized, destroyed = [False], []
    monkeypatch.setattr(module, "setup_env_local_multi_gpu", lambda *args: None)
    monkeypatch.setattr(module, "assign_device_for_local_rank", lambda *args: None)
    monkeypatch.setattr(
        module.dist,
        "init_process_group",
        lambda **kwargs: initialized.__setitem__(0, True),
    )
    monkeypatch.setattr(module.dist, "is_initialized", lambda: initialized[0])

    def destroy():
        destroyed.append(True)
        initialized[0] = False

    def fail(config):
        raise RuntimeError("model initialization failed")

    monkeypatch.setattr(module.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(module.hydra.utils, "instantiate", fail)
    with pytest.raises(RuntimeError, match="model initialization failed"):
        unit.predict(_data())
    assert destroyed == [True]
    assert not unit.local_rank0._owns_process_group
    assert unit._resources_released


def test_gp_cleanup_failure_still_destroys_default_process_group(
    transport, monkeypatch
):
    unit, _, _, _, _ = transport(workers=1)
    initialized, destroyed = [True], []
    unit.local_rank0._owns_process_group = True
    unit.local_rank0.gp_config = object()
    monkeypatch.setattr(module.dist, "is_initialized", lambda: initialized[0])

    def fail():
        raise RuntimeError("partial GP cleanup failed")

    def destroy():
        destroyed.append(True)
        initialized[0] = False

    monkeypatch.setattr(module.gp_utils, "cleanup_gp", fail)
    monkeypatch.setattr(module.dist, "destroy_process_group", destroy)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        unit.close()
    assert destroyed == [True]
    assert unit.local_rank0.predict_unit is None
    assert not unit.local_rank0._owns_process_group


@pytest.mark.parametrize(
    ("location", "failure"),
    [
        ("gp", KeyboardInterrupt("cancel GP cleanup")),
        ("actor", SystemExit(12)),
        (
            "gp",
            BaseExceptionGroup(
                "interrupted group cleanup",
                [KeyboardInterrupt("cancel"), RuntimeError("group failed")],
            ),
        ),
    ],
)
def test_cleanup_cancellation_attempts_all_resources_and_preserves_exception(
    transport, monkeypatch, location, failure
):
    unit, _, _, _, _ = transport(workers=1)
    unit.local_rank0._owns_process_group = True
    unit.local_rank0.gp_config = object()
    initialized, pending, calls = [True], [True], []
    monkeypatch.setattr(module.dist, "is_initialized", lambda: initialized[0])

    def release(name):
        calls.append(name)
        if pending[0] and name == location:
            pending[0] = False
            raise failure

    def destroy():
        calls.append("default")
        initialized[0] = False

    monkeypatch.setattr(module.ray, "kill", lambda *args, **kwargs: release("actor"))
    monkeypatch.setattr(
        module.ray.util, "remove_placement_group", lambda group: release("placement")
    )
    monkeypatch.setattr(module.gp_utils, "cleanup_gp", lambda: release("gp"))
    monkeypatch.setattr(module.dist, "destroy_process_group", destroy)
    with pytest.raises(type(failure)) as caught:
        unit.close()
    assert caught.value is failure
    assert calls == ["actor", "placement", "gp", "default"]
    assert unit.local_rank0.predict_unit is None
    assert not unit.local_rank0._owns_process_group
    assert unit._closed
    assert not unit._resources_released
    unit.close()
    assert unit._resources_released

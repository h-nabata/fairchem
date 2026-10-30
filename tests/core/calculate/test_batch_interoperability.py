"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.calculate import batched_inference, device_buffers
from fairchem.core.calculate.batched_inference import (
    IndependentDevicePool,
    InferenceBatchLimits,
    MicrobatchPredictor,
    plan_microbatches,
    run_resident_steps,
)
from fairchem.core.calculate.device_buffers import DeviceBatchStager
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task
from tests.core.calculate.test_batched_inference import HarmonicPredictor, _structure
from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor


@pytest.mark.parametrize("count", [1, 2])
def test_singleton_and_batched_predictors_cannot_mutate_caller_inputs(count):
    structures = [_structure(offset=i) for i in range(count)]
    snapshots = [data.clone() for data in structures]

    def mutate(data):
        data.pos.add_(10)
        data.charge.add_(2)
        data.dataset[0] = "changed"
        return {"forces": data.pos}

    results = MicrobatchPredictor(SimpleNamespace(predict=mutate)).predict_many(
        structures
    )
    for result, data, before in zip(results, structures, snapshots):
        torch.testing.assert_close(result["forces"], before.pos + 10)
        torch.testing.assert_close(data.pos, before.pos)
        torch.testing.assert_close(data.charge, before.charge)
        assert data.dataset == before.dataset


def test_private_input_clone_retains_nonleaf_gradients_and_higher_derivatives():
    data = _structure()
    leaf = data.pos.clone().requires_grad_()
    data.pos = leaf * 1.25
    result = MicrobatchPredictor(HarmonicPredictor()).predict_many([data])[0]
    gradient = torch.autograd.grad(result["energy"].sum(), leaf, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(), leaf)[0]
    torch.testing.assert_close(gradient, 2 * 1.25**2 * leaf)
    torch.testing.assert_close(curvature, torch.full_like(leaf, 2 * 1.25**2))


def test_per_structure_requests_do_not_become_nested_or_cross_jobs():
    calls = []

    class Predictor(HarmonicPredictor):
        supports_requested_properties = True

        def predict(self, data, requested_properties=None):
            calls.append((data.num_graphs, requested_properties))
            output = super().predict(data)
            return {name: output[name] for name in requested_properties}

    structures = [_structure(offset=i) for i in range(3)]
    for data, props in zip(structures, [("energy",), ("forces",), ("energy",)]):
        data._requested_properties = props
    output = MicrobatchPredictor(Predictor()).predict_many(structures)
    assert [set(value) for value in output] == [{"energy"}, {"forces"}, {"energy"}]
    assert (2, ("energy",)) in calls
    assert (1, ("forces",)) in calls
    assert structures[0]._requested_properties == ("energy",)
    override = MicrobatchPredictor(Predictor(), requested_properties=("forces",))
    assert all(set(value) == {"forces"} for value in override.predict_many(structures))


@pytest.mark.parametrize(
    "properties", ["forces", b"forces", {"energy": True}, 1, (), [""], [1]]
)
def test_invalid_requested_properties_fail_before_running_jobs(properties):
    with pytest.raises(ValueError, match="requested_properties"):
        MicrobatchPredictor(HarmonicPredictor(), requested_properties=properties)
    with pytest.raises(ValueError, match="requested_properties"):
        IndependentDevicePool(
            HarmonicPredictor, ["cpu"], requested_properties=properties
        )


def test_extra_tensor_dtypes_are_grouped_without_implicit_promotion():
    first, second = _structure(), _structure()
    first.hessian_vector = torch.ones_like(first.pos)
    second.hessian_vector = torch.ones_like(second.pos, dtype=torch.float64)
    assert len(plan_microbatches([first, second], InferenceBatchLimits())) == 2
    limits = InferenceBatchLimits(max_atoms=np.int64(32), max_edges=np.int64(64))
    plan_microbatches([first, second], limits, edge_counts=np.array([6, 6]))
    assert type(limits.max_atoms) is int


def _uma():
    predictor = _small_predictor()
    predictor.inference_settings.merge_mole = False
    task = Task(
        name="omol_energy",
        property="energy",
        level="system",
        out_spec=OutputSpec(dim=[1], dtype="float32"),
        datasets=["omol"],
        normalizer=Normalizer(mean=0.0, rmsd=2.0),
    )
    predictor.model.module._tasks = {task.name: task}
    predictor.model.module._dataset_to_tasks = {"omol": [task]}
    return predictor


@pytest.mark.parametrize(
    "property_name", ["hessian", "hessian_vector_product", "embeddings"]
)
@pytest.mark.parametrize("staged", [False, True])
def test_actual_uma_curvature_and_embeddings_match_single_jobs(property_name, staged):
    structures = [_structure("H2O"), _structure("CH4", offset=0.1)]
    for data in structures:
        data.hessian_vector = (
            torch.arange(data.pos.numel()).reshape_as(data.pos).to(data.pos) / 10
        )
    predictor = _uma()
    reference = copy.deepcopy(predictor)
    expected = [
        reference.predict(data, requested_properties=(property_name,))
        for data in structures
    ]
    stager = DeviceBatchStager("cpu") if staged else None
    wrapper = MicrobatchPredictor(
        predictor, stager=stager, requested_properties=(property_name,)
    )
    actual = wrapper.predict_many(structures)
    for result, oracle in zip(actual, expected):
        assert set(result) == {property_name}
        assert torch.isfinite(result[property_name]).all()
        assert torch.count_nonzero(result[property_name]) > 0
        torch.testing.assert_close(
            result[property_name], oracle[property_name], atol=3e-5, rtol=3e-4
        )
    if stager is not None:
        stager.close()


def test_queued_jobs_snapshot_tensor_and_metadata_values_at_submission():
    entered, release = Event(), Event()

    class Predictor(HarmonicPredictor):
        def predict(self, data):
            entered.set()
            assert release.wait(timeout=10)
            output = super().predict(data)
            output["energy"] += data.metadata["value"]
            return output

    first, second = _structure(), _structure(offset=1)
    first.metadata = {"value": 1}
    second.metadata = {"value": 2}
    expected = second.pos.square().sum() + 2
    with IndependentDevicePool(Predictor, ["cpu"]) as pool:
        running = pool.submit([first])
        assert entered.wait(timeout=10)
        queued = pool.submit([second])
        second.pos.add_(100)
        second.metadata["value"] = 1000
        release.set()
        running.result(timeout=10)
        torch.testing.assert_close(
            queued.result(timeout=10)[0]["energy"], expected[None]
        )


def test_failed_predictor_setup_releases_created_stager():
    class Stager(DeviceBatchStager):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    stager = Stager("cpu")
    with IndependentDevicePool(
        HarmonicPredictor,
        ["cpu"],
        stager_factory=lambda device: stager,
        requested_properties=("energy",),
    ) as pool:
        with pytest.raises(ValueError, match="does not support"):
            pool.submit([_structure()]).result(timeout=10)
        assert stager.closed


@pytest.mark.parametrize(
    "kwargs",
    [
        {"slots": True},
        {"slots": 1.5},
        {"max_batch_bytes": None},
        {"max_batch_bytes": False},
    ],
)
def test_stager_configuration_rejects_noninteger_values(kwargs):
    with pytest.raises(ValueError, match="positive integers"):
        DeviceBatchStager("cpu", **kwargs)


def test_staging_cyclic_metadata_has_no_recursion_error_or_caller_alias():
    data = _structure()
    data.metadata = {"value": torch.ones(1)}
    data.metadata["self"] = data.metadata
    stager = DeviceBatchStager("cpu")
    with stager.stage(data) as owned:
        assert owned.metadata["self"] is owned.metadata
        owned.metadata["value"].zero_()
        assert data.metadata["value"].item() == 1
    stager.close()


def test_concurrent_staging_cannot_lease_same_slot_during_metadata_copy():
    entered, release, second_entered = Event(), Event(), Event()

    class BlockingMetadata:
        def __deepcopy__(self, memo):
            entered.set()
            assert release.wait(timeout=10)
            return "copied"

    first, second = _structure(), _structure(offset=1)
    first.metadata = BlockingMetadata()
    stager = DeviceBatchStager("cpu", slots=1)

    def stage_second():
        second_entered.set()
        return stager.stage(second)

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending = executor.submit(stager.stage, first)
        assert entered.wait(timeout=10)
        other = executor.submit(stage_second)
        assert second_entered.wait(timeout=10)
        release.set()
        lease = pending.result(timeout=10)
        with pytest.raises(RuntimeError, match="leased"):
            other.result(timeout=10)
        lease.release()
    stager.close()


def test_release_failure_preserves_primary_prediction_error_and_attempts_all_leases():
    released = []

    class Lease:
        def __init__(self, batch):
            self.batch = batch

        def __enter__(self):
            return self.batch

        def __exit__(self, *args):
            pass

        def release(self):
            released.append(self)
            raise RuntimeError("cleanup failure")

    class Stager:
        device = torch.device("cpu")
        slots = 2

        def stage(self, batch):
            return Lease(batch)

    def fail(data):
        raise ValueError("primary prediction failure")

    wrapper = MicrobatchPredictor(
        SimpleNamespace(predict=fail),
        InferenceBatchLimits(max_systems=1),
        stager=Stager(),
    )
    with pytest.raises(ValueError, match="primary prediction failure") as info:
        wrapper.predict_many([_structure(), _structure()])
    assert len(released) == 2
    assert any("cleanup failure" in note for note in info.value.__notes__)


def test_staged_context_and_pool_cleanup_do_not_hide_job_error(monkeypatch):
    stager = DeviceBatchStager("cpu")
    lease = stager.stage(_structure())

    def cleanup_failure():
        raise RuntimeError("cleanup failure")

    with monkeypatch.context() as patch:
        patch.setattr(lease, "release", cleanup_failure)
        with pytest.raises(ValueError, match="job failure") as info, lease:
            raise ValueError("job failure")
        assert "cleanup failure" in info.value.__notes__[0]
    lease.release()
    stager.close()
    pool = IndependentDevicePool(HarmonicPredictor, ["cpu"])
    with monkeypatch.context() as patch:
        patch.setattr(pool, "close", cleanup_failure)
        with pytest.raises(ValueError, match="job failure") as info, pool:
            raise ValueError("job failure")
        assert "cleanup failure" in info.value.__notes__[0]
    pool.close()


def test_virtual_gpu_submission_waits_for_source_snapshot_before_worker_use(
    monkeypatch,
):
    # Dispatch/lifetime check only; these fake tensors do not execute GPU math.
    data = _structure()
    with FakeTensorMode(allow_non_fake_inputs=True):
        data.to("xpu:0")
    events = []

    class Completion:
        def record(self, stream):
            events.append(("record", self))

        def synchronize(self):
            events.append(("wait", self))

    def factory(device):
        events.append(("factory", device))
        return SimpleNamespace(
            device=device,
            predict=lambda batch: {"energy": torch.ones(batch.num_graphs)},
        )

    monkeypatch.setattr(batched_inference, "device_scope", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "Event", Completion)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device: device)
    monkeypatch.setattr(
        torch.Tensor,
        "record_stream",
        lambda tensor, stream: events.append(("source-use", stream)),
    )
    with IndependentDevicePool(factory, ["xpu:0"]) as pool:
        result = pool.submit([data]).result(timeout=10)
    torch.testing.assert_close(result[0]["energy"], torch.ones(1))
    assert events[0][0] == "record"
    assert events[1] == ("wait", events[0][1])
    assert events[2] == ("source-use", torch.device("xpu:0"))
    assert ("factory", torch.device("xpu:0")) in events[3:]
    assert events[-1] == ("wait", events[-2][1])


@pytest.mark.parametrize(
    "transfer", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_copied_runner_has_independent_locks_and_empty_staging_storage(transfer):
    original = MicrobatchPredictor(HarmonicPredictor(), stager=DeviceBatchStager("cpu"))
    expected = original.predict_many([_structure()])
    assert original.stager.allocated_bytes > 0
    cloned = transfer(original)
    assert cloned._lock is not original._lock
    assert cloned.stager.allocated_bytes == 0
    actual = cloned.predict_many([_structure()])
    torch.testing.assert_close(actual[0]["forces"], expected[0]["forces"])
    original.stager.close()
    cloned.stager.close()


def test_copying_an_active_staging_lease_is_rejected():
    stager = DeviceBatchStager("cpu")
    with stager.stage(_structure()), pytest.raises(RuntimeError, match="Release all"):
        copy.deepcopy(stager)
    stager.close()


def test_metadata_copy_cannot_reenter_and_overwrite_its_own_staging_slot():
    stager = DeviceBatchStager("cpu", slots=1)
    source, other = _structure(), _structure(offset=20)

    class ReentrantMetadata:
        def __deepcopy__(self, memo):
            with pytest.raises(RuntimeError, match="leased"):
                stager.stage(other)
            return "copied"

    source.metadata = ReentrantMetadata()
    with stager.stage(source) as batch:
        torch.testing.assert_close(batch.pos, source.pos)
    stager.close()


def test_failed_metadata_copy_does_not_leak_staging_lease():
    class BrokenMetadata:
        def __deepcopy__(self, memo):
            raise ValueError("invalid metadata")

    stager = DeviceBatchStager("cpu", slots=1)
    source = _structure()
    source.metadata = BrokenMetadata()
    with pytest.raises(ValueError, match="invalid metadata"):
        stager.stage(source)
    with stager.stage(_structure()) as batch:
        assert batch.pos.shape == (3, 3)
    stager.close()


def test_pool_cleanup_attempts_all_workers_and_allows_failed_worker_retry():
    attempts = []

    class Stager:
        def __init__(self, name):
            self.name = name
            self.fail = name == "first"

        def close(self):
            attempts.append(self.name)
            if self.fail:
                raise RuntimeError("first worker cleanup failed")

    pool = IndependentDevicePool(HarmonicPredictor, ["cpu", "cpu:0"])
    first, second = Stager("first"), Stager("second")
    pool._workers = [SimpleNamespace(stager=first), SimpleNamespace(stager=second)]
    with pytest.raises(RuntimeError, match="first worker cleanup failed"):
        pool.close()
    assert attempts == ["first", "second"]
    assert pool._workers[0] is not None
    assert pool._workers[1] is None
    first.fail = False
    pool.close()
    assert attempts == ["first", "second", "first"]


@pytest.mark.parametrize("steps,interval", [(1.5, 1), (1, 1.5), (True, 1), (1, True)])
def test_resident_steps_rejects_fractional_or_boolean_counts(steps, interval):
    with pytest.raises(ValueError, match="steps"):
        run_resident_steps(None, None, None, steps, observe_every=interval)


def test_resident_steps_accepts_numpy_integer_configuration():
    seen = []

    class Model:
        device = torch.device("cpu")

        def __call__(self, state):
            return state.value + 1

    def update(state, value):
        return SimpleNamespace(device=state.device, value=value)

    final = run_resident_steps(
        Model(),
        SimpleNamespace(device="cpu", value=0),
        update,
        np.int64(3),
        observe_every=np.int64(2),
        observer=lambda step, state, prediction: seen.append((step, prediction)),
    )
    assert final.value == 3
    assert seen == [(0, 1), (2, 3)]


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("failed_sync", [False, True])
def test_virtual_staging_interruption_finishes_copies_or_quarantines_slot(
    monkeypatch, interruption, failed_sync
):
    # Control-flow check only: allocations/copies use CPU storage, and pinning
    # and stream operations are stubbed. No accelerator execution is claimed.
    source = _structure()
    stager = DeviceBatchStager("cpu", slots=1)
    events = []
    primary = interruption("cancelled during ready-event recording")

    class Stream:
        def synchronize(self):
            events.append("synchronize")
            if failed_sync:
                raise KeyboardInterrupt("interrupted synchronization")

    class Ready:
        def record(self, stream):
            events.append("record")
            raise primary

    allocate, copy_tensor = torch.empty, torch.Tensor.copy_

    def allocate_on_cpu(*args, device=None, **kwargs):
        return allocate(*args, device="cpu", **kwargs)

    def record_copy(tensor, other, non_blocking=False):
        if non_blocking:
            events.append("copy")
        return copy_tensor(tensor, other, non_blocking=non_blocking)

    monkeypatch.setattr(torch, "empty", allocate_on_cpu)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(torch.Tensor, "copy_", record_copy)
    monkeypatch.setattr(device_buffers, "device_scope", lambda device: nullcontext())
    stager.device = torch.device("xpu:0")
    stager._copy_stream = Stream()
    stager._backend = SimpleNamespace(stream=lambda stream: nullcontext(), Event=Ready)
    with pytest.raises(interruption) as error:
        stager.stage(source)
    assert error.value is primary
    assert "copy" in events[:-2]
    assert events[-2:] == ["record", "synchronize"]
    slot = stager._slots[0]
    assert slot.leased is failed_sync
    if failed_sync:
        with pytest.raises(RuntimeError, match="leased"):
            stager.stage(source)
    else:
        # A partially recorded event must not survive a successful stream join.
        assert slot.ready is None
        stager.device = torch.device("cpu")
        stager._backend = None
        stager._copy_stream = None
        with stager.stage(source) as restored:
            torch.testing.assert_close(restored.pos, source.pos)
        stager.close()


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_metadata_copy_interruption_does_not_leak_staging_lease(interruption):
    primary = interruption("cancelled metadata copy")

    class Metadata:
        def __deepcopy__(self, memo):
            raise primary

    source = _structure()
    source.metadata = Metadata()
    stager = DeviceBatchStager("cpu", slots=1)
    with pytest.raises(interruption) as error:
        stager.stage(source)
    assert error.value is primary
    with stager.stage(_structure()) as restored:
        assert restored.pos.shape == (3, 3)
    stager.close()

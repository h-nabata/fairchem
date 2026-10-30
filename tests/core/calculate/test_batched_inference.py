"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from threading import Event, Lock, RLock
from types import SimpleNamespace

import pytest
import torch
from ase.build import molecule

from fairchem.core.calculate import batched_inference, device_buffers
from fairchem.core.calculate.batched_inference import (
    IndependentDevicePool,
    InferenceBatchLimits,
    MicrobatchPredictor,
    plan_microbatches,
    run_resident_steps,
)
from fairchem.core.calculate.device_buffers import DeviceBatchStager
from fairchem.core.datasets.atomic_data import AtomicData


def _structure(name="H2O", offset=0, task="omol", dtype=torch.float32):
    atoms = molecule(name)
    atoms.positions += offset
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(
        atoms, task_name=task, target_dtype=dtype, r_data_keys=["charge", "spin"]
    )


class HarmonicPredictor:
    """
    A genuine tensor-only independent-system potential used as a CPU oracle.
    """

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.calls = []

    def predict(self, data):
        self.calls.append((data.num_graphs, data.pos.shape[0], tuple(data.dataset)))
        energy = data.pos.new_zeros(data.num_graphs).index_add(
            0, data.batch, data.pos.square().sum(-1)
        )
        stress = data.pos.new_zeros(data.num_graphs, 3, 3).index_add(
            0, data.batch, data.pos.unsqueeze(-1) * data.pos.unsqueeze(-2)
        )
        return {"energy": energy, "forces": -2 * data.pos, "stress": stress}


@pytest.mark.parametrize("slots", [None, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_microbatches_preserve_independent_values_order_and_owned_storage(slots, dtype):
    structures = [
        _structure("H2O", 0.1, dtype=dtype),
        _structure("CH4", 0.2, dtype=dtype),
        _structure("H2", 0.3, task="omat", dtype=dtype),
        _structure("H2O", 0.4, dtype=dtype),
    ]
    predictor = HarmonicPredictor()
    stager = DeviceBatchStager("cpu", slots=slots) if slots is not None else None
    wrapper = MicrobatchPredictor(
        predictor,
        InferenceBatchLimits(max_atoms=8, max_systems=2, max_edges=30),
        stager=stager,
    )
    actual = wrapper.predict_many(structures, edge_counts=[6, 20, 2, 6])
    oracle = HarmonicPredictor()
    for data, result in zip(structures, actual):
        expected = oracle.predict(data)
        for key in expected:
            torch.testing.assert_close(result[key], expected[key])
            assert result[key].dtype == dtype
            assert result[key].untyped_storage().nbytes() == (
                result[key].numel() * result[key].element_size()
            )
    assert all(atoms <= 8 and systems <= 2 for systems, atoms, _ in predictor.calls)
    assert all(len(set(datasets)) == 1 for _, _, datasets in predictor.calls)
    snapshots = [{name: value.clone() for name, value in r.items()} for r in actual]
    wrapper.predict_many([_structure("CH4", 99, dtype=dtype)], edge_counts=[20])
    for before, after in zip(snapshots, actual):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    if stager is not None:
        stager.close()
        assert stager.allocated_bytes == 0


def test_microbatch_force_gradients_survive_batch_assembly():
    structures = [_structure(offset=0.1), _structure(offset=0.5)]
    for data in structures:
        data.pos.requires_grad_()
    results = MicrobatchPredictor(HarmonicPredictor()).predict_many(structures)
    grad = torch.autograd.grad(
        sum(r["energy"].sum() for r in results), [d.pos for d in structures]
    )
    for derivative, data in zip(grad, structures):
        torch.testing.assert_close(derivative, 2 * data.pos)


def test_batch_planner_limits_do_not_truncate_graphs():
    data = [_structure(), _structure("CH4")]
    with pytest.raises(ValueError, match="edge_counts"):
        plan_microbatches(data, InferenceBatchLimits(max_edges=20))
    with pytest.raises(ValueError, match="Structure 1"):
        plan_microbatches(data, InferenceBatchLimits(max_atoms=4))
    with pytest.raises(ValueError, match="Structure 0"):
        plan_microbatches(
            data, InferenceBatchLimits(max_edges=20), edge_counts=[21, 10]
        )
    assert plan_microbatches([], InferenceBatchLimits()) == []
    assert MicrobatchPredictor(HarmonicPredictor()).predict_many([]) == []
    plan = plan_microbatches(
        data, InferenceBatchLimits(max_edges=20), edge_counts=[6, 20]
    )
    assert sorted(i for batch in plan for i in batch) == [0, 1]
    assert len(plan) == 2


def test_unknown_property_requires_explicit_output_layout():
    predictor = SimpleNamespace(predict=lambda data: {"descriptor": data.pos.square()})
    with pytest.raises(ValueError, match="descriptor"):
        MicrobatchPredictor(predictor).predict_many([_structure()])
    result = MicrobatchPredictor(
        predictor, property_levels={"descriptor": "atom"}
    ).predict_many([_structure()])
    torch.testing.assert_close(result[0]["descriptor"], _structure().pos.square())


def test_staging_owns_inputs_preserves_aliases_and_reuses_capacity():
    data = _structure(dtype=torch.float64)
    data.alias = data.pos
    data.metadata = {"names": ["before"]}
    stager = DeviceBatchStager("cpu", slots=1, max_batch_bytes=4096)
    expected = data.pos.clone()
    with stager.stage(data) as staged:
        pointer = staged.pos.data_ptr()
        assert staged.pos is staged.alias
        data.pos.add_(100)
        data.metadata["names"].append("after")
        torch.testing.assert_close(staged.pos, expected)
        assert staged.metadata == {"names": ["before"]}
        with pytest.raises(RuntimeError, match="leased"):
            stager.stage(data)
        with pytest.raises(RuntimeError, match="Release"):
            stager.close()
    with stager.stage(data) as staged:
        assert staged.pos.data_ptr() == pointer
        torch.testing.assert_close(staged.pos, data.pos)
    assert stager.allocated_bytes <= 4096
    stager.close()


def test_staging_rejects_oversize_and_differentiable_inputs():
    data = _structure()
    with pytest.raises(ValueError, match="max_batch_bytes"):
        DeviceBatchStager("cpu", max_batch_bytes=1).stage(data)
    data.pos.requires_grad_()
    with pytest.raises(ValueError, match="differentiable"):
        DeviceBatchStager("cpu").stage(data)
    with pytest.raises(ValueError, match="ordinal"):
        DeviceBatchStager("xpu")


def test_prediction_failure_releases_prefetched_staging_slots():
    stager = DeviceBatchStager("cpu")

    def fail(data):
        raise RuntimeError("prediction failed")

    wrapper = MicrobatchPredictor(
        SimpleNamespace(predict=fail),
        InferenceBatchLimits(max_systems=1),
        stager=stager,
    )
    with pytest.raises(RuntimeError, match="prediction failed"):
        wrapper.predict_many([_structure(), _structure()])
    # Both current and prefetched leases were released, so close must succeed.
    stager.close()


def test_real_cpu_worker_reuses_predictor_and_preserves_order():
    calls = []

    def factory(device):
        calls.append(device)
        return HarmonicPredictor(device)

    structures = [_structure(offset=i) for i in range(6)]
    with IndependentDevicePool(
        factory,
        ["cpu"],
        limits=InferenceBatchLimits(max_systems=2),
        stager_factory=lambda device: DeviceBatchStager(device),
    ) as pool:
        actual = pool.predict_many(structures)
        future = pool.submit([structures[0]])
        torch.testing.assert_close(future.result()[0]["energy"], actual[0]["energy"])
        assert calls == [torch.device("cpu")]
    for data, result in zip(structures, actual):
        torch.testing.assert_close(result["forces"], -2 * data.pos)
    with pytest.raises(RuntimeError, match="closed"):
        pool.submit([structures[0]])


def test_full_uma_microbatch_and_cpu_worker_match_single_structure_predictions():
    from tests.core.units.mlip_unit.test_predict_memory import (
        _IdentityNormalizer,
        _task,
    )
    from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor

    prototype = _small_predictor()
    prototype.inference_settings.merge_mole = False
    tasks = []
    for prop in ("energy", "forces", "stress"):
        task = _task(f"omol_{prop}", "atom" if prop == "forces" else "system", prop)
        task.normalizer = _IdentityNormalizer()
        task.element_references = None
        tasks.append(task)
    prototype.model.module._tasks = {task.name: task for task in tasks}
    prototype.model.module._dataset_to_tasks = {"omol": tasks}
    structures = [_structure(), _structure("CH4", 0.25), _structure(offset=0.5)]
    reference = copy.deepcopy(prototype)
    expected = [reference.predict(data) for data in structures]
    actual = MicrobatchPredictor(
        copy.deepcopy(prototype), InferenceBatchLimits(max_atoms=8)
    ).predict_many(structures)
    with IndependentDevicePool(
        lambda device: copy.deepcopy(prototype),
        ["cpu"],
        limits=InferenceBatchLimits(max_atoms=8),
    ) as pool:
        pooled = pool.predict_many(structures)
    for outputs in (actual, pooled):
        for result, oracle in zip(outputs, expected):
            for name in ("energy", "forces", "stress"):
                torch.testing.assert_close(
                    result[name], oracle[name], rtol=2e-5, atol=2e-5
                )


def test_cancelled_queued_job_releases_scheduling_load():
    entered, release = Event(), Event()

    class BlockingPredictor(HarmonicPredictor):
        def predict(self, data):
            entered.set()
            assert release.wait(timeout=10)
            return super().predict(data)

    with IndependentDevicePool(
        lambda device: BlockingPredictor(device), ["cpu"]
    ) as pool:
        first = pool.submit([_structure()])
        assert entered.wait(timeout=10)
        queued = pool.submit([_structure()])
        assert queued.cancel()
        release.set()
        first.result(timeout=10)
        assert pool._pending == [0]


def test_explicit_gpu_ordinals_are_scheduled_on_distinct_serial_workers(monkeypatch):
    monkeypatch.setattr(batched_inference, "device_scope", lambda device: nullcontext())
    waited = []

    class Completion:
        def record(self, stream):
            self.stream = stream

        def synchronize(self):
            waited.append(self.stream)

    monkeypatch.setattr(torch.xpu, "Event", Completion)
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device: device)
    entered, release = Event(), Event()
    calls, active, maximum = [], {}, {}
    lock = Lock()

    class DevicePredictor(HarmonicPredictor):
        def predict(self, data):
            with lock:
                active[self.device] = active.get(self.device, 0) + 1
                maximum[self.device] = max(
                    maximum.get(self.device, 0), active[self.device]
                )
            entered.set()
            assert release.wait(timeout=10)
            result = super().predict(data)
            with lock:
                active[self.device] -= 1
            return result

    def factory(device):
        calls.append(device)
        return DevicePredictor(device)

    with IndependentDevicePool(factory, ["xpu:0", "xpu:1"]) as pool:
        first = pool.submit([_structure()])
        assert entered.wait(timeout=10)
        second = pool.submit([_structure()])
        third = pool.submit([_structure()])
        release.set()
        for future in (first, second, third):
            future.result(timeout=10)
    assert set(calls) == {torch.device("xpu:0"), torch.device("xpu:1")}
    assert set(maximum.values()) == {1}
    assert len(waited) == 3
    assert set(waited) == {torch.device("xpu:0"), torch.device("xpu:1")}
    with pytest.raises(ValueError, match="ordinal"):
        IndependentDevicePool(factory, ["xpu"])


@pytest.mark.parametrize("name", ["max_atoms", "max_systems"])
def test_required_microbatch_limits_cannot_be_none(name):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        InferenceBatchLimits(**{name: None})


def test_resident_loop_uses_callers_update_without_changing_algorithm():
    initial = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    state = SimpleNamespace(positions=initial.clone(), device=torch.device("cpu"))

    class Model:
        device = torch.device("cpu")

        def __call__(self, current):
            return {"forces": -2 * current.positions}

    def update(current, prediction):
        current.positions = current.positions + 0.125 * prediction["forces"]
        return current

    observed = []
    result = run_resident_steps(
        Model(),
        state,
        update,
        5,
        observer=lambda step, current, prediction: observed.append(step),
        observe_every=2,
    )
    torch.testing.assert_close(result.positions, initial * 0.75**5, rtol=0, atol=0)
    assert result.positions.dtype == torch.float64
    assert observed == [0, 2, 4]


def test_staged_lease_records_allocator_lifetime_and_joins_consumer_streams(
    monkeypatch,
):
    # This checks scheduling invariants without pretending CPU tensors are XPU
    # tensors; the real-device test below checks actual copies and computation.
    events = []

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_event(self, event):
            events.append((self.name, "wait", event))

    class Completion:
        def record(self, stream):
            events.append((stream.name, "record", self))

    first, second, release_stream = (
        Stream(name) for name in ("first", "second", "release")
    )
    current = first
    backend = SimpleNamespace(current_stream=lambda device: current, Event=Completion)
    owner = SimpleNamespace(
        device=torch.device("xpu:1"), _backend=backend, _lock=RLock()
    )
    recorded = []
    slot = device_buffers._Slot(
        target={"pos": SimpleNamespace(record_stream=recorded.append)},
        ready=object(),
        leased=True,
    )
    monkeypatch.setattr(device_buffers, "device_scope", lambda device: nullcontext())
    lease = device_buffers.StagedBatch(owner, slot, object())
    lease.wait()
    current = second
    lease.wait()
    current = release_stream
    lease.release()
    assert recorded == [first, second, release_stream]
    assert events[:3] == [
        ("first", "wait", slot.ready),
        ("second", "wait", slot.ready),
        ("release", "wait", slot.ready),
    ]
    assert events[3][0:2] == ("first", "record")
    assert events[4] == ("release", "wait", events[3][2])
    assert events[5][0:2] == ("second", "record")
    assert events[6] == ("release", "wait", events[5][2])
    assert events[7] == ("release", "record", slot.consumed)
    assert not slot.leased
    lease.release()
    assert len(events) == 8


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires a real Intel XPU")
def test_xpu_staging_copy_and_consumer_order():
    device = torch.device("xpu", torch.xpu.current_device())
    stager = DeviceBatchStager(device)
    saved = []
    for i in range(6):
        data = _structure(offset=i)
        expected = data.pos.square().clone()
        lease = stager.stage(data)
        data.pos.zero_()
        with lease as batch:
            result = batch.pos.square()
        saved.append((result, expected))
    stager.close()
    for result, expected in saved:
        torch.testing.assert_close(result.cpu(), expected)

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import sys
from collections import OrderedDict
from collections.abc import Collection, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from numbers import Integral
from threading import Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from fairchem.core.calculate.device_buffers import (
    DeviceBatchStager,
    _tensors,
    device_scope,
)
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass(frozen=True)
class InferenceBatchLimits:
    """
    Admission limits for independent structures, never graph truncation limits.

    max_edges requires explicit edge counts for internally generated graphs.
    Supplied estimates limit planning cost but do not guarantee peak VRAM; use
    actual counts or a conservative bound when a hard bound is required.
    """

    max_atoms: int = 4096
    max_systems: int = 32
    max_edges: int | None = None

    def __post_init__(self):
        for name in ("max_atoms", "max_systems", "max_edges"):
            value = getattr(self, name)
            if value is None and name == "max_edges":
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))


def _normalize_requested_properties(properties):
    if properties is None:
        return None
    if isinstance(properties, (str, bytes, Mapping)) or not isinstance(
        properties, Collection
    ):
        raise ValueError("requested_properties must be a sequence of names")
    properties = tuple(properties)
    if not properties or any(not isinstance(p, str) or not p for p in properties):
        raise ValueError("requested_properties must contain property names")
    return tuple(sorted(set(properties)))


def _field_signature(data, name, value):
    if torch.is_tensor(value):
        shape = list(value.shape)
        axis = data.__cat_dim__(name, value)
        if shape and axis is not None:
            shape.pop(axis)
        return (name, value.device, value.dtype, value.ndim, tuple(shape))
    return (name, type(value))


def _compatibility_key(data: AtomicData) -> tuple:
    # A property may exist for only some datasets. Keep each microbatch on one
    # dataset so output splitting cannot assign an omitted property to a system.
    return (
        tuple(getattr(data, "dataset", ())),
        frozenset(data.keys()),
        data.pos.device,
        data.pos.dtype,
        tuple(_field_signature(data, name, value) for name, value in data),
        _normalize_requested_properties(data.get("_requested_properties", None)),
    )


def plan_microbatches(
    structures: Sequence[AtomicData],
    limits: InferenceBatchLimits,
    *,
    edge_counts: Sequence[int] | None = None,
) -> list[list[int]]:
    """
    Pack independent structures deterministically and return original indices.

    Largest structures are packed first within compatible datasets; output
    indices make the caller's original ordering independent of execution order.
    A structure exceeding a limit is rejected instead of silently truncating its
    neighbors or exceeding the requested batch budget.
    """
    if edge_counts is not None and len(edge_counts) != len(structures):
        raise ValueError("edge_counts must contain one value per structure")
    if edge_counts is not None and any(
        isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in edge_counts
    ):
        raise ValueError("edge_counts must be nonnegative integers")
    costs = []
    for index, data in enumerate(structures):
        if data.num_graphs != 1:
            raise ValueError("Each input must contain exactly one structure")
        atoms = data.pos.shape[0]
        if atoms < 1:
            raise ValueError("Empty structures are not supported by UMA batching")
        edges = (
            int(edge_counts[index])
            if edge_counts is not None
            else data.edge_index.shape[1]
        )
        if limits.max_edges is not None and edge_counts is None and edges == 0:
            raise ValueError(
                "Supply edge_counts when max_edges is used with an internal graph"
            )
        if atoms > limits.max_atoms or (
            limits.max_edges is not None and edges > limits.max_edges
        ):
            raise ValueError(f"Structure {index} exceeds the microbatch limits")
        costs.append((atoms, edges))

    bins = []
    open_bins = {}
    for index in sorted(
        range(len(structures)), key=lambda i: (-costs[i][1], -costs[i][0], i)
    ):
        atoms, edges = costs[index]
        if limits.max_systems == 1:
            _normalize_requested_properties(
                structures[index].get("_requested_properties", None)
            )
            bins.append([index])
            continue
        compatible = open_bins.setdefault(_compatibility_key(structures[index]), {})
        for key, entry in compatible.items():
            members, used_atoms, used_edges = entry
            if used_atoms + atoms <= limits.max_atoms and (
                limits.max_edges is None or used_edges + edges <= limits.max_edges
            ):
                break  # key and members identify the first fit below.
        else:
            key = len(bins)
            members, used_atoms, used_edges = [], 0, 0
            bins.append(members)
        members.append(index)
        used_atoms += atoms
        if len(members) == limits.max_systems or used_atoms == limits.max_atoms:
            compatible.pop(key, None)
        else:
            compatible[key] = members, used_atoms, used_edges + edges
    return bins


_PROPERTY_LEVELS = {
    "energy": "system",
    "free_energy": "system",
    "forces": "atom",
    "stress": "system",
    "magmoms": "atom",
    "charges": "atom",
    "hessian_vector_product": "atom",
    "hessian": "system",
    "embeddings": "atom",
}


def _requires_single_system(predictor, data, requested):
    """Resolve forward constraints after a worker has loaded its predictor."""
    if getattr(predictor, "requires_single_system", False) is True:
        return True
    settings = getattr(predictor, "inference_settings", None)
    if getattr(settings, "nitro", False):
        return True
    module = getattr(getattr(predictor, "model", None), "module", None)
    if (
        not getattr(
            module,
            "supports_single_atoms",
            getattr(predictor, "supports_single_atoms", True),
        )
        and data.num_nodes == 1
        and not data.pbc.any()
    ):
        return True
    if requested is not None:
        return "hessian" in requested
    datasets = data.get("dataset", ())
    if tasks := getattr(predictor, "dataset_to_tasks", {}):
        selected = (task for dataset in datasets for task in tasks.get(dataset, ()))
    else:
        tasks = getattr(predictor, "tasks", {})
        selected = tasks.values() if isinstance(tasks, Mapping) else ()
    return any(
        getattr(task, "property", None) == "hessian" for task in selected
    ) or bool(
        set(datasets).intersection(getattr(settings, "predict_untrained_hessian", ()))
    )


class MicrobatchPredictor:
    """
    Evaluate independent structures with bounded batches and owned outputs.

    This changes scheduling only. NEB spring forces, finite-difference formulas,
    AFIR bias potentials and convergence criteria remain in the calling program.
    A DeviceBatchStager optionally overlaps CPU input transfer with computation.
    The wrapped predictor must be single-device and must not retain or mutate
    the input after predict returns, except for work queued on the current stream.
    """

    def __init__(
        self,
        predictor,
        limits: InferenceBatchLimits | None = None,
        *,
        property_levels: Mapping[str, str] | None = None,
        stager: DeviceBatchStager | None = None,
        requested_properties: Sequence[str] | None = None,
    ):
        self.predictor = predictor
        self.limits = limits or InferenceBatchLimits()
        self.property_levels = _PROPERTY_LEVELS | dict(property_levels or {})
        if any(
            level not in {"atom", "system"} for level in self.property_levels.values()
        ):
            raise ValueError("Property levels must be 'atom' or 'system'")
        self.stager = stager
        self.requested_properties = _normalize_requested_properties(
            requested_properties
        )
        self._lock = Lock()
        if self.requested_properties is not None and not getattr(
            predictor, "supports_requested_properties", False
        ):
            raise ValueError("The predictor does not support requested_properties")
        if (
            stager is not None
            and hasattr(predictor, "device")
            and torch.device(predictor.device) != stager.device
        ):
            raise ValueError("Predictor and staging device must match exactly")

    def _requests(self, data):
        if self.requested_properties is not None:
            return self.requested_properties
        request = data.get("_requested_properties", None)
        if request is None:
            request = getattr(
                getattr(self.predictor, "inference_settings", None),
                "requested_properties",
                None,
            )
        return _normalize_requested_properties(request)

    def __getstate__(self):
        with self._lock:
            return {key: value for key, value in vars(self).items() if key != "_lock"}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()

    def _predict(self, batch, requested):
        if requested is not None and not getattr(
            self.predictor, "supports_requested_properties", False
        ):
            raise ValueError("The predictor does not support requested_properties")
        kwargs = {} if requested is None else {"requested_properties": requested}
        predict = getattr(
            self.predictor, "_predict_owned_batch", self.predictor.predict
        )
        return predict(batch, **kwargs)

    def _split(self, predictions, structures):
        sizes = [data.pos.shape[0] for data in structures]
        results = [{} for _ in sizes]
        for name, values in predictions.items():
            if name not in self.property_levels:
                raise ValueError(f"Specify the atom/system level of output {name!r}")
            if not isinstance(values, torch.Tensor) or values.ndim == 0:
                raise ValueError(f"Output {name!r} must have a leading batch dimension")
            lengths = (
                sizes if self.property_levels[name] == "atom" else [1] * len(sizes)
            )
            if values.shape[0] != sum(lengths):
                raise ValueError(f"Output {name!r} has an incompatible leading size")
            for result, value in zip(results, values.split(lengths, dim=0)):
                # A returned single structure must not keep every other system's
                # output allocation alive, or alias reusable staging/model storage.
                result[name] = value.clone()
        return results

    def predict_many(
        self,
        structures: Sequence[AtomicData],
        *,
        edge_counts: Sequence[int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        # Predictor state and reusable staging slots belong to this runner.
        # Separate calling threads must not run their jobs through it at once.
        with self._lock:
            plan = plan_microbatches(structures, self.limits, edge_counts=edge_counts)
            return self._predict_plan(structures, plan)

    def _predict_plan(self, structures, plan, *, owned=False):
        groups = []
        for indices in plan:
            if len(indices) == 1:
                groups.append(indices)
                continue
            regular = []
            for i in indices:
                data = structures[i]
                if _requires_single_system(self.predictor, data, self._requests(data)):
                    groups.append([i])
                else:
                    regular.append(i)
            if regular:
                groups.append(regular)
        plan = groups
        results = [None] * len(structures)
        current = None
        next_lease = None

        def assemble(indices):
            selected = [structures[i] for i in indices]
            if len(selected) > 1:
                return atomicdata_list_to_batch(
                    selected, exclude_keys=["_requested_properties"]
                )
            item = selected[0]
            if not owned and self.stager is None:
                item = item.clone()
            if "_requested_properties" in item:
                # Staging takes its own snapshot. Keep the caller's field set intact.
                item = copy.copy(item)
                item.__keys__ = item.keys().copy()
                del item["_requested_properties"]
            return item

        try:
            if self.stager is not None and plan:
                current = self.stager.stage(assemble(plan[0]))
            for position, indices in enumerate(plan):
                selected = [structures[i] for i in indices]
                requested = self._requests(selected[0])
                if current is None:
                    predictions = self._predict(assemble(indices), requested)
                    split = self._split(predictions, selected)
                    # Split outputs own their storage. Do not retain the large
                    # combined outputs while allocating the next input/forward.
                    del predictions
                else:
                    with current as batch:
                        if position + 1 < len(plan) and self.stager.slots > 1:
                            next_lease = self.stager.stage(assemble(plan[position + 1]))
                        predictions = self._predict(batch, requested)
                        split = self._split(predictions, selected)
                        del predictions
                    del batch
                    if position + 1 < len(plan) and self.stager.slots == 1:
                        next_lease = self.stager.stage(assemble(plan[position + 1]))
                    current = next_lease
                    next_lease = None
                for index, prediction in zip(indices, split):
                    results[index] = prediction
        finally:
            primary_error = sys.exception()
            cleanup_error = None
            for lease in (current, next_lease):
                if lease is not None:
                    try:
                        lease.release()
                    except Exception as error:
                        if primary_error is not None:
                            primary_error.add_note(f"Staging release failed: {error}")
                        elif cleanup_error is None:
                            cleanup_error = error
                        else:
                            cleanup_error.add_note(
                                f"Additional staging release failure: {error}"
                            )
            if cleanup_error is not None:
                raise cleanup_error
        return results


class IndependentDevicePool:
    """
    Schedule independent jobs on one serial predictor worker per explicit device.

    Each worker lazily creates and retains its own predictor. No distributed
    process group, peer-memory pooling, model sharding or CPU model fallback is
    involved. Callers should use the context manager or close() after all jobs.
    GPU tensors returned by workers stay on their respective devices.
    """

    def __init__(
        self,
        predictor_factory: Callable[[torch.device], Any],
        devices: Sequence[str | torch.device],
        *,
        limits: InferenceBatchLimits | None = None,
        property_levels: Mapping[str, str] | None = None,
        stager_factory: Callable[[torch.device], DeviceBatchStager] | None = None,
        requested_properties: Sequence[str] | None = None,
    ) -> None:
        self.devices = tuple(torch.device(device) for device in devices)
        if not self.devices or len(set(self.devices)) != len(self.devices):
            raise ValueError("Supply distinct devices")
        for device in self.devices:
            if device.type not in {"cpu", "cuda", "xpu"}:
                raise ValueError("Workers support CPU, CUDA and XPU")
            if device.type != "cpu" and device.index is None:
                raise ValueError("Each accelerator requires an explicit device ordinal")
        self.limits = limits or InferenceBatchLimits()
        self._factory = predictor_factory
        self._levels = dict(property_levels) if property_levels is not None else None
        self._stager_factory = stager_factory
        self._requested_properties = _normalize_requested_properties(
            requested_properties
        )
        self._workers = [None] * len(self.devices)
        self._executors = [
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"fairchem-{device}")
            for device in self.devices
        ]
        self._pending = [0] * len(self.devices)
        self._rates = OrderedDict()
        self._default_rate = None
        self._lock = Lock()
        self._closed = False

    def _run(self, worker, structures, plan, ready_events, profile, units):
        device = self.devices[worker]
        with device_scope(device):
            for source, event in ready_events:
                if device == source:
                    getattr(torch, source.type).current_stream(source).wait_event(event)
                else:
                    event.synchronize()
            recorded = set()
            for data in structures:
                for _, tensor in _tensors(vars(data)):
                    if (
                        tensor.device.type in {"cuda", "xpu"}
                        and id(tensor) not in recorded
                    ):
                        backend = getattr(torch, tensor.device.type)
                        tensor.record_stream(backend.current_stream(tensor.device))
                        recorded.add(id(tensor))
            if self._workers[worker] is None:
                predictor = self._factory(device)
                if (
                    hasattr(predictor, "device")
                    and torch.device(predictor.device) != device
                ):
                    raise ValueError(
                        "The factory returned a predictor on another device"
                    )
                stager = (
                    self._stager_factory(device)
                    if self._stager_factory is not None
                    else None
                )
                try:
                    self._workers[worker] = MicrobatchPredictor(
                        predictor,
                        self.limits,
                        property_levels=self._levels,
                        stager=stager,
                        requested_properties=self._requested_properties,
                    )
                except Exception as error:
                    if stager is not None:
                        try:
                            stager.close()
                        except Exception as cleanup_error:
                            error.add_note(f"Staging cleanup failed: {cleanup_error}")
                    raise
            start = perf_counter()
            results = self._workers[worker]._predict_plan(structures, plan, owned=True)
            if device.type != "cpu":
                # Future completion promises usable outputs even when the
                # caller subsequently selects a different stream. Wait only
                # for this worker's submitted work, not the entire device.
                backend = getattr(torch, device.type)
                completed = backend.Event()
                completed.record(backend.current_stream(device))
                completed.synchronize()
            rate = max(1e-9, perf_counter() - start) / units
            with self._lock:
                previous_rate = self._rates.get(profile, rate)
                self._rates[profile] = 0.75 * previous_rate + 0.25 * rate
                self._rates.move_to_end(profile)
                if len(self._rates) > 128:
                    self._rates.popitem(last=False)
                if self._default_rate is None:
                    self._default_rate = rate
            return results

    def submit(
        self,
        structures: Sequence[AtomicData],
        *,
        edge_counts: Sequence[int] | None = None,
    ) -> Future:
        """
        Queue an owned snapshot; later caller mutations cannot change this job.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("The device pool is closed")
        inputs = list(structures)
        counts = list(edge_counts) if edge_counts is not None else None
        plan = plan_microbatches(inputs, self.limits, edge_counts=counts)
        return self._submit_plan(inputs, counts, plan)

    def _submit_plan(self, inputs, counts, plan):
        inputs = [data.clone() for data in inputs]
        devices = {
            tensor.device
            for data in inputs
            for _, tensor in _tensors(vars(data))
            if tensor.device.type in {"cuda", "xpu"}
        }
        ready_events = []
        for device in devices:
            with device_scope(device):
                backend = getattr(torch, device.type)
                ready = backend.Event()
                ready.record(backend.current_stream(device))
                ready_events.append((device, ready))
        with self._lock:
            if self._closed:
                raise RuntimeError("The device pool is closed")
            worker = min(
                range(len(self.devices)),
                key=lambda index: (
                    self._pending[index],
                    self.devices[index] not in devices,
                ),
            )
            # Learn a wall-time-per-work-unit estimate for each property mix.
            # Include actual external edges or the caller's conservative graph counts.
            profile = frozenset(
                self._requested_properties
                or _normalize_requested_properties(
                    data.get("_requested_properties", None)
                )
                for data in inputs
            )
            units = max(
                1,
                sum(
                    data.num_nodes
                    + (data.num_edges if counts is None else int(counts[index]))
                    for index, data in enumerate(inputs)
                ),
            )
            # User estimates are arbitrary Python integers. Bound scheduling
            # arithmetic independently of admission (which retains exact counts).
            units = min(units, 2**53)
            rate = self._rates.get(profile, self._default_rate)
            relative_rate = (
                1 if self._default_rate is None else rate / self._default_rate
            )
            cost = max(1, round(units * relative_rate))
            self._pending[worker] += cost
            try:
                future = self._executors[worker].submit(
                    self._run, worker, inputs, plan, ready_events, profile, units
                )
            except Exception:
                self._pending[worker] -= cost
                raise

        def completed(_future):
            # A cancelled queued job never enters _run, but must release its
            # scheduling load just like a completed or failed prediction.
            with self._lock:
                self._pending[worker] -= cost

        future.add_done_callback(completed)
        return future

    def predict_many(
        self,
        structures: Sequence[AtomicData],
        *,
        edge_counts: Sequence[int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        plan = plan_microbatches(structures, self.limits, edge_counts=edge_counts)
        jobs = {}
        results = [None] * len(structures)
        batches = iter(plan)
        try:
            while True:
                for indices in batches:
                    inputs = [structures[i] for i in indices]
                    counts = (
                        None
                        if edge_counts is None
                        else [edge_counts[i] for i in indices]
                    )
                    future = self._submit_plan(
                        inputs, counts, [list(range(len(inputs)))]
                    )
                    jobs[future] = indices
                    if len(jobs) >= 2 * len(self.devices):
                        break
                if not jobs:
                    break
                done, _ = wait(jobs, return_when=FIRST_COMPLETED)
                for future in done:
                    indices = jobs.pop(future)
                    for index, prediction in zip(indices, future.result(), strict=True):
                        results[index] = prediction
        except BaseException:
            for future in jobs:
                future.cancel()
            raise
        return results

    def close(self) -> None:
        with self._lock:
            self._closed = True
        for executor in self._executors:
            executor.shutdown(wait=True)
        cleanup_error = None
        for index, worker in enumerate(self._workers):
            try:
                if worker is not None and worker.stager is not None:
                    worker.stager.close()
                self._workers[index] = None
            except Exception as error:
                # Leave the failed worker available for a later cleanup retry,
                # but do not strand the remaining workers' buffers and models.
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    cleanup_error.add_note(
                        f"Additional worker cleanup failure: {error}"
                    )
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.close()
        except Exception as error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Device pool cleanup failed: {error}")


def run_resident_steps(
    model,
    state,
    update: Callable,
    steps: int,
    *,
    observer: Callable | None = None,
    observe_every: int = 1,
):
    """
    Run a caller-selected TorchSim update rule with simulation state on-device.

    update(state, predictions) returns the next state. This function introduces
    no new integration scheme, timestep, precision conversion or convergence
    tolerance. An optional observer sees the pre-update state and matching
    prediction; only that callback controls any CPU logging or trajectory copies.
    The returned state is after exactly ``steps`` updates.
    """
    if (
        any(
            isinstance(v, bool) or not isinstance(v, Integral)
            for v in (steps, observe_every)
        )
        or steps < 0
        or observe_every < 1
    ):
        raise ValueError("steps must be nonnegative and observe_every positive")
    steps, observe_every = int(steps), int(observe_every)
    device = torch.device(model.device)
    if torch.device(state.device) != device:
        state = state.to(device)
    for step in range(steps):
        predictions = model(state)
        if observer is not None and step % observe_every == 0:
            observer(step, state, predictions)
        state = update(state, predictions)
        # The integrator/observer owns anything it retains. Avoid overlapping
        # otherwise completed force/stress outputs with the next model graph.
        del predictions
        if torch.device(state.device) != device:
            raise ValueError("The update moved simulation state off the model device")
    return state

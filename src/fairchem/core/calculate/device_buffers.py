"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, field
from numbers import Integral
from threading import RLock
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


def device_scope(device: torch.device):
    """
    Select the explicit accelerator ordinal for this thread.
    """
    if device.type == "cpu":
        return nullcontext()
    return getattr(torch, device.type).device(device)


def _tensors(value, path=(), seen=None):
    if seen is None:
        seen = set()
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (dict, list, tuple)):
        if id(value) in seen:
            return
        seen.add(id(value))
        items = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in items:
            yield from _tensors(item, (*path, key), seen)


@dataclass
class _Slot:
    host: dict = field(default_factory=dict)
    target: dict = field(default_factory=dict)
    ready: object | None = None
    consumed: object | None = None
    leased: bool = False


class StagedBatch:
    """
    An explicitly leased input batch whose storage is reusable after release.

    Use as a context manager around prediction. Do not retain the batch or its
    tensor views after leaving the context. Prediction outputs must own their
    storage. Release records the consuming stream, so later staging cannot
    overwrite buffers still being read by asynchronous kernels.
    """

    def __init__(self, owner, slot, batch):
        self._owner = owner
        self._slot = slot
        self._batch = batch
        self._released = False
        self._consumer_streams = []

    def wait(self) -> AtomicData:
        with self._owner._lock:
            return self._wait()

    def _wait(self):
        if self._released:
            raise RuntimeError("The staged batch has already been released")
        if self._owner.device.type != "cpu":
            with device_scope(self._owner.device):
                stream = self._owner._backend.current_stream(self._owner.device)
                stream.wait_event(self._slot.ready)
                # The allocator must also know about consumers if the lease and
                # stager are dropped without close(). Waiting on the copy event
                # alone does not extend allocation lifetime on another stream.
                for tensor in self._slot.target.values():
                    tensor.record_stream(stream)
                if stream not in self._consumer_streams:
                    self._consumer_streams.append(stream)
        return self._batch

    def release(self) -> None:
        with self._owner._lock:
            self._release()

    def _release(self):
        if self._released:
            return
        # Also order release after the copy when the caller never entered/waited.
        self.wait()
        if self._owner.device.type != "cpu":
            with device_scope(self._owner.device):
                stream = self._owner._backend.current_stream(self._owner.device)
                # A caller may wait and consume on several streams. Join all
                # their submitted work before allowing this slot to be reused.
                for consumer in self._consumer_streams:
                    if consumer != stream:
                        consumed = self._owner._backend.Event()
                        consumed.record(consumer)
                        stream.wait_event(consumed)
                event = self._owner._backend.Event()
                event.record(stream)
                self._slot.consumed = event
        self._slot.leased = False
        self._released = True
        self._batch = None

    def __enter__(self) -> AtomicData:
        return self.wait()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.release()
        except Exception as error:
            if exc_value is None:
                raise
            exc_value.add_note(f"Staged batch release failed: {error}")


class DeviceBatchStager:
    """
    Reuse bounded, owned CPU and device buffers for independent inference inputs.

    CPU inputs are copied synchronously into owned pinned buffers before an
    asynchronous accelerator transfer. Callers can therefore immediately reuse
    or mutate the original input. Two slots allow the next input transfer to
    overlap the current computation. Each slot holds at most max_batch_bytes of
    tensor payload on each side (CPU mode needs only the target buffers).

    This utility is inference-only: input tensors requiring gradients are
    rejected. Conservative model forces remain supported because the predictor
    creates its own differentiable position/cell copies. GPU-to-GPU inputs
    should use the resident prediction path instead of this staging utility.

    Args:
        device: CPU or an explicit CUDA/XPU device ordinal.
        slots: Number of independent reusable input slots.
        max_batch_bytes: Maximum payload capacity per slot and memory side.
    """

    def __init__(
        self,
        device: str | torch.device,
        *,
        slots: int = 2,
        max_batch_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError("Staging supports CPU, CUDA and XPU devices")
        if self.device.type != "cpu" and self.device.index is None:
            raise ValueError("Specify an explicit accelerator ordinal, e.g. xpu:0")
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 1
            for v in (slots, max_batch_bytes)
        ):
            raise ValueError("slots and max_batch_bytes must be positive integers")
        self.max_batch_bytes = int(max_batch_bytes)
        self._slots = [_Slot() for _ in range(int(slots))]
        self._lock = RLock()
        self._next = 0
        self._backend = None
        self._copy_stream = None
        if self.device.type != "cpu":
            self._backend = getattr(torch, self.device.type)
            if not self._backend.is_available():
                raise RuntimeError(f"{self.device.type} is unavailable")
            with device_scope(self.device):
                self._copy_stream = self._backend.Stream(device=self.device)

    @property
    def slots(self) -> int:
        return len(self._slots)

    def __getstate__(self):
        with self._lock:
            if any(slot.leased for slot in self._slots):
                raise RuntimeError(
                    "Release all staged batches before copying staging configuration"
                )
            # Streams, events and reusable storage belong to the originating
            # worker. A copied/pickled stager starts with independent slots.
            return {
                "device": self.device,
                "slots": self.slots,
                "max_batch_bytes": self.max_batch_bytes,
            }

    def __setstate__(self, state):
        self.__init__(**state)

    @property
    def allocated_bytes(self) -> int:
        with self._lock:
            return sum(
                tensor.numel() * tensor.element_size()
                for slot in self._slots
                for buffers in (slot.host, slot.target)
                for tensor in buffers.values()
            )

    def _slot(self) -> _Slot:
        for offset in range(len(self._slots)):
            index = (self._next + offset) % len(self._slots)
            slot = self._slots[index]
            if not slot.leased:
                if slot.consumed is not None:
                    slot.consumed.synchronize()
                elif slot.ready is not None:
                    slot.ready.synchronize()
                self._next = (index + 1) % len(self._slots)
                return slot
        raise RuntimeError("All staging slots are leased; release a batch first")

    def stage(self, batch: AtomicData) -> StagedBatch:
        """
        Own the CPU input and enqueue its transfer without changing its dtype.
        """
        with self._lock:
            return self._stage(batch)

    def _stage(self, batch):
        unique = {}
        for path, tensor in _tensors(vars(batch)):
            unique.setdefault(id(tensor), (path, tensor))
        tensors = list(unique.values())
        if any(t.device.type != "cpu" for _, t in tensors):
            raise ValueError("Staging inputs must be on CPU")
        if any(t.requires_grad for _, t in tensors):
            raise ValueError("Staging does not accept differentiable input tensors")
        if any(t.layout != torch.strided for _, t in tensors):
            raise ValueError("Staging supports strided tensors only")
        payload = sum(t.numel() * t.element_size() for _, t in tensors)
        if payload > self.max_batch_bytes:
            raise ValueError("Input exceeds max_batch_bytes; reduce the microbatch")
        slot = self._slot()
        slot.consumed = None
        slot.ready = None
        keys = {path for path, _ in tensors}
        for buffers in (slot.host, slot.target):
            for path in buffers.keys() - keys:
                del buffers[path]
        capacity = 0
        for path, tensor in tensors:
            old = slot.target.get(path)
            capacity += (
                max(old.numel(), tensor.numel())
                if old is not None and old.dtype == tensor.dtype
                else tensor.numel()
            ) * tensor.element_size()
        if capacity > self.max_batch_bytes:
            slot.host.clear()
            slot.target.clear()

        memo = {}
        copy_context = (
            nullcontext()
            if self._backend is None
            else self._backend.stream(self._copy_stream)
        )
        try:
            with device_scope(self.device), copy_context, torch.no_grad():
                for path, tensor in tensors:
                    target = slot.target.get(path)
                    if (
                        target is None
                        or target.dtype != tensor.dtype
                        or target.numel() < tensor.numel()
                    ):
                        target = torch.empty(
                            tensor.numel(), dtype=tensor.dtype, device=self.device
                        )
                        slot.target[path] = target
                    target_view = target[: tensor.numel()].view(tensor.shape)
                    if self._backend is None:
                        target_view.copy_(tensor)
                    else:
                        host = slot.host.get(path)
                        if (
                            host is None
                            or host.dtype != tensor.dtype
                            or host.numel() < tensor.numel()
                        ):
                            # Explicit type matters on hosts exposing multiple backends.
                            host = torch.empty(
                                tensor.numel(), dtype=tensor.dtype
                            ).pin_memory(device=self.device.type)
                            slot.host[path] = host
                        host_view = host[: tensor.numel()].view(tensor.shape)
                        host_view.copy_(tensor)
                        target_view.copy_(host_view, non_blocking=True)
                    memo[id(tensor)] = target_view
                if self._backend is not None:
                    slot.ready = self._backend.Event()
                    slot.ready.record(self._copy_stream)
        except BaseException:
            if self._copy_stream is not None:
                try:
                    self._copy_stream.synchronize()
                    slot.ready = None
                except BaseException:
                    # A failed stream cannot establish safe reuse. Quarantine
                    # this slot rather than overwriting an in-flight input.
                    slot.leased = True
            raise
        slot.leased = True
        try:
            result = copy.deepcopy(batch, memo)
        except BaseException:
            # A failed metadata copy has no consumer. Keep the ready event so
            # later reuse still waits for the already queued transfer.
            slot.leased = False
            raise
        return StagedBatch(self, slot, result)

    def close(self) -> None:
        """
        Wait for pending consumers and release cached buffer capacity.
        """
        with self._lock:
            self._close()

    def _close(self):
        if any(slot.leased for slot in self._slots):
            raise RuntimeError("Release all staged batches before closing")
        for slot in self._slots:
            if slot.consumed is not None:
                slot.consumed.synchronize()
            elif slot.ready is not None:
                slot.ready.synchronize()
            slot.host.clear()
            slot.target.clear()

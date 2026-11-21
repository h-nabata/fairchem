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

from fairchem.core.common.distutils import parse_device
from fairchem.core.datasets.atomic_data import _tensors

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


def device_scope(device: torch.device):
    """
    Select the explicit accelerator ordinal for this thread.
    """
    if device.type == "cpu":
        return nullcontext()
    return getattr(torch, device.type).device(device)


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
        pack_small_tensors: Coalesce small integer metadata of each dtype into
            one host-to-device copy. Enable after profiling the target device.
    """

    def __init__(
        self,
        device: str | torch.device,
        *,
        slots: int = 2,
        max_batch_bytes: int = 256 * 1024 * 1024,
        pack_small_tensors: bool = False,
    ) -> None:
        self.device = parse_device(device)
        if self.device.type not in {"cpu", "cuda", "xpu"}:
            raise ValueError("Staging supports CPU, CUDA and XPU devices")
        if self.device.type != "cpu" and self.device.index is None:
            raise ValueError("Specify an explicit accelerator ordinal, e.g. xpu:0")
        if any(
            isinstance(v, bool) or not isinstance(v, Integral) or v < 1
            for v in (slots, max_batch_bytes)
        ):
            raise ValueError("slots and max_batch_bytes must be positive integers")
        if not isinstance(pack_small_tensors, bool):
            raise TypeError("pack_small_tensors must be boolean")
        self.pack_small_tensors = pack_small_tensors
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
                "pack_small_tensors": self.pack_small_tensors,
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
        pending = None
        host_ready = None
        for offset in range(self.slots):
            index = (self._next + offset) % self.slots
            slot = self._slots[index]
            if slot.leased:
                continue
            event = slot.consumed if slot.consumed is not None else slot.ready
            if event is None or event.query():
                self._next = (index + 1) % self.slots
                return slot
            if (
                self._copy_stream is not None
                and host_ready is None
                and slot.ready is not None
                and slot.ready.query()
            ):
                host_ready = index
            if pending is None:
                pending = index
        if pending is None:
            raise RuntimeError("All staging slots are leased; release a batch first")
        index = pending if host_ready is None else host_ready
        slot = self._slots[index]
        if self._copy_stream is not None and slot.ready is not None:
            if host_ready is None:
                slot.ready.synchronize()
            # Pinned host bytes are now reusable. Target writes stay ordered
            # after all earlier consumers, without blocking this host thread.
            if slot.consumed is not None:
                with device_scope(self.device):
                    self._copy_stream.wait_event(slot.consumed)
        else:
            event = slot.consumed if slot.consumed is not None else slot.ready
            event.synchronize()
        self._next = (index + 1) % self.slots
        return self._slots[index]

    def stage(self, batch: AtomicData) -> StagedBatch:
        """
        Own the CPU input and enqueue its transfer without changing its dtype.
        """
        with self._lock:
            return self._stage(batch)

    def _stage(self, batch):
        tensors = list(_tensors(batch, strict=True))
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
        slot.ready = None
        groups = {}
        for path, tensor in tensors:
            pack = (
                self.pack_small_tensors
                and tensor.numel() <= 256
                and not tensor.is_floating_point()
                and not tensor.is_complex()
            )
            key = tensor.dtype if pack else path
            groups.setdefault(key, []).append(tensor)
        sizes = {key: sum(t.numel() for t in group) for key, group in groups.items()}
        for buffers in (slot.host, slot.target):
            for key in buffers.keys() - groups.keys():
                del buffers[key]
        capacity = 0
        for key, group in groups.items():
            old = slot.target.get(key)
            size = sizes[key]
            if old is not None and old.dtype == group[0].dtype:
                size = max(size, old.numel())
            capacity += size * group[0].element_size()
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
            # Reusable buffers must remain writable by later calls outside an
            # ambient inference_mode context. no_grad alone still creates
            # inference tensors when the caller has inference_mode enabled.
            with (
                device_scope(self.device),
                copy_context,
                torch.inference_mode(False),
                torch.no_grad(),
            ):
                for key, group in groups.items():
                    size, dtype = sizes[key], group[0].dtype
                    target = slot.target.get(key)
                    if target is None or target.dtype != dtype or target.numel() < size:
                        target = torch.empty(size, dtype=dtype, device=self.device)
                        slot.target[key] = target
                    buffer = target
                    if self._backend is not None:
                        buffer = slot.host.get(key)
                        if (
                            buffer is None
                            or buffer.dtype != dtype
                            or buffer.numel() < size
                        ):
                            buffer = torch.empty(
                                size, dtype=dtype, device="cpu"
                            ).pin_memory(device=self.device.type)
                            slot.host[key] = buffer
                    offset = 0
                    for tensor in group:
                        end = offset + tensor.numel()
                        buffer[offset:end].view(tensor.shape).copy_(tensor)
                        memo[id(tensor)] = target[offset:end].view(tensor.shape)
                        offset = end
                    if self._backend is not None:
                        target[:size].copy_(buffer[:size], non_blocking=True)
                if self._backend is not None:
                    slot.ready = self._backend.Event()
                    slot.ready.record(self._copy_stream)
                slot.consumed = None
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
            # Custom __deepcopy__ implementations must substitute the admitted
            # views from memo, just like standard Python containers do.
            staged_ids = {id(view) for view in memo.values()}
            result = copy.deepcopy(batch, memo)
            if any(
                id(tensor) not in staged_ids
                for _, tensor in _tensors(result, strict=True)
            ):
                raise ValueError(
                    "Metadata copy introduced a tensor outside the staged buffers"
                )
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

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from numbers import Integral
from types import MethodType

import torch
from torch.utils import _pytree


@dataclass(eq=False)
class _TensorIdentity:
    """
    Retain replaced tensors so a recycled Python id cannot match old weights.
    """

    tensor: torch.Tensor

    def __eq__(self, other):
        return isinstance(other, _TensorIdentity) and self.tensor is other.tensor


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def model_tensor_signature(model):
    """
    Track in-place edits and replacement of parameters or buffers.

    References are owned by the cache entry. Pointer reuse cannot impersonate a
    previously cached model, and each new model receives its own signature.
    """
    signature = []
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        if value.is_inference():
            raise ValueError(
                "Cached/compiled models require parameter mutation counters. "
                "Construct the predictor outside torch.inference_mode()."
            )
        signature.append(
            (
                name,
                _TensorIdentity(value),
                value._version,
                value.device,
                value.dtype,
                tuple(value.shape),
                value.requires_grad,
            )
        )
    return tuple(signature)


def model_storage_bytes(model):
    """
    Count unique parameter/buffer storage; this excludes backend workspaces.
    """
    storages = {}
    for value in list(model.parameters()) + list(model.buffers()):
        storage = value.untyped_storage()
        storages[(value.device, storage.data_ptr())] = storage.nbytes()
    return sum(storages.values())


class ModelStreamGuard:
    """Protect immutable prepared weights when a caller changes XPU streams.

    The preparation event orders first use on another stream. record_stream
    prevents cache eviction from recycling storage while that stream uses it.
    Neither operation synchronizes the host or changes the caller's stream.
    """

    def __init__(self, model):
        self._restore = False
        self._initialize(model)

    def _initialize(self, model):
        devices = {
            tensor.device
            for tensor in list(model.parameters()) + list(model.buffers())
            if tensor.device.type == "xpu"
        }
        self.ready = {}
        for device in devices:
            stream = torch.xpu.current_stream(device)
            event = torch.xpu.Event()
            event.record(stream)
            self.ready[device] = (stream, event)

    def wait_and_record(self, model):
        if self._restore:
            # Copy/load can change device or creation stream. Original events
            # must not be reused with restored tensors. Order all prior copies
            # once before establishing the new immutable-weight event.
            devices = {
                tensor.device
                for tensor in list(model.parameters()) + list(model.buffers())
                if tensor.device.type == "xpu"
            }
            for device in devices:
                torch.xpu.synchronize(device)
            self._initialize(model)
            self._restore = False
        streams = {}
        for device, (origin, ready) in self.ready.items():
            stream = torch.xpu.current_stream(device)
            if stream != origin:
                stream.wait_event(ready)
                streams[device] = stream
        if streams:
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.device in streams:
                    tensor.record_stream(streams[tensor.device])

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.ready = {}
        self._restore = True


def mole_composition_key(data, *, device, dtype, model_id, settings_key):
    """
    Build a collision-free routing key including absolute elemental counts.

    Geometry is intentionally absent: it never changes merged expert weights.
    Counts are exact integers, including for the size-dependent UMA-S-1.2
    composition embedding. Each batch member is checked, not only the first.
    """
    numbers = data.atomic_numbers.detach().to(device="cpu", dtype=torch.int64)
    batch = data.batch.detach().to(device="cpu", dtype=torch.int64)
    charges = data.charge.detach().cpu().reshape(-1).tolist()
    spins = data.spin.detach().cpu().reshape(-1).tolist()
    systems = []
    for index, dataset in enumerate(data.dataset):
        elements, counts = numbers[batch == index].unique(return_counts=True)
        systems.append(
            (
                tuple(zip(elements.tolist(), counts.tolist())),
                charges[index],
                spins[index],
                dataset,
            )
        )
    return (
        model_id,
        str(torch.device(device)),
        str(dtype),
        settings_key,
        tuple(systems),
    )


class PreparedModelCache:
    """
    Bounded LRU of independently prepared, immutable inference models.

    The caller separately owns the active model; eviction never changes it.
    The byte budget covers retained parameter/buffer storage, not allocator
    reservations, graph pools, or the current inference's autograd tensors.
    """

    def __init__(self, max_entries, max_bytes=None):
        self.max_entries = _positive_integer(max_entries, "max_entries")
        self.max_bytes = (
            None if max_bytes is None else _positive_integer(max_bytes, "max_bytes")
        )
        self._entries = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0

    def clear(self):
        self._entries.clear()
        self.bytes = 0

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        model, signature, size = entry
        if signature != model_tensor_signature(model):
            del self._entries[key]
            self.bytes -= size
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return model

    def put(self, key, model):
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes -= previous[2]
        size = model_storage_bytes(model)
        if self.max_bytes is not None and size > self.max_bytes:
            return
        self._entries[key] = (model, model_tensor_signature(model), size)
        self.bytes += size
        while len(self._entries) > self.max_entries or (
            self.max_bytes is not None and self.bytes > self.max_bytes
        ):
            _, (_, _, evicted_size) = self._entries.popitem(last=False)
            self.bytes -= evicted_size

    def __len__(self):
        return len(self._entries)


class _OwnedGraphInput(torch.autograd.Function):
    """
    Keep captured backward workspaces out of externally returned gradients.
    """

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return None if gradient is None else gradient.clone()


class NumericCoreRuntime:
    """
    Compile or capture a tensor-only numeric block, keeping an eager reference.

    Captures own their sample tensors and return cloned outputs so graph replay
    cannot overwrite a previously returned result. A bounded shape LRU limits
    retained capture workspaces. Double backward and functional transforms use
    the ordinary callable because XPU make_graphed_callables does not support
    higher-order differentiation.
    """

    def __init__(
        self,
        function,
        *,
        compile=False,
        dynamic=True,
        xpu_graph=False,
        graph_cache_size=1,
    ):
        self.function = function
        self._compile_enabled = compile
        self._dynamic = dynamic
        self.compiled = torch.compile(function, dynamic=dynamic) if compile else None
        self.xpu_graph = xpu_graph
        self.graph_cache_size = _positive_integer(graph_cache_size, "graph_cache_size")
        self.higher_order = False
        self._graphs = OrderedDict()
        self._failed_captures = OrderedDict()
        owner = getattr(function, "__self__", None)
        self.owner = owner if isinstance(owner, torch.nn.Module) else None
        self._owner_signature = self._execution_signature()
        self._lock = threading.RLock()
        self.capture_count = 0
        self.replay_count = 0

    def clear(self):
        with self._lock:
            self._graphs.clear()
            self._failed_captures.clear()

    def _execution_signature(self):
        if self.owner is None:
            return None
        return (
            model_tensor_signature(self.owner),
            tuple(
                (name, module.training) for name, module in self.owner.named_modules()
            ),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Capture closures and compiled bound methods can refer to the original
        # module. Recreate them against the copied function on its first call.
        state.update(
            compiled=None,
            _graphs=OrderedDict(),
            _failed_captures=OrderedDict(),
            _owner_signature=None,
            capture_count=0,
            replay_count=0,
            higher_order=False,
        )
        state.pop("_lock", None)
        if isinstance(self.function, MethodType) and self.owner is not None:
            # Pickling a bound method by name would look up the replacement
            # ``module.forward`` and bind the runtime recursively to itself.
            state["_method_function"] = self.function.__func__
            state["function"] = None
        return state

    def __setstate__(self, state):
        method_function = state.pop("_method_function", None)
        self.__dict__.update(state)
        if method_function is not None:
            self.function = MethodType(method_function, self.owner)
        self._lock = threading.RLock()

    def _graph_arguments(self, args, kwargs):
        if not (
            hasattr(torch._C._autograd, "_get_current_graph_task_keep_graph")
            and hasattr(torch.autograd.Variable._execution_engine, "queue_callback")
        ):
            return None
        # The closure passed to make_graphed_callables does not expose module
        # parameters. In inference these are frozen; a trainable owner must use
        # the ordinary callable so parameter gradients are never lost.
        if self.owner is not None and any(
            tensor.requires_grad
            for tensor in list(self.owner.parameters()) + list(self.owner.buffers())
        ):
            return None
        leaves, spec = _pytree.tree_flatten((args, kwargs))
        tensors = []
        description = []
        for index, value in enumerate(leaves):
            if torch.is_tensor(value):
                if value.device.type != "xpu" or value.layout != torch.strided:
                    return None
                tensors.append(value)
                description.append(
                    (
                        index,
                        tuple(value.shape),
                        tuple(value.stride()),
                        value.dtype,
                        value.device,
                        value.requires_grad,
                    )
                )
            elif value is None or isinstance(value, (str, int, float, bool)):
                description.append((index, type(value), value))
            else:
                return None
        if not tensors or len({tensor.device for tensor in tensors}) != 1:
            return None
        # Captured workspaces are never shared across submission streams.
        stream = torch.xpu.current_stream(tensors[0].device)
        return (
            leaves,
            spec,
            tensors,
            (repr(spec), tuple(description), stream, torch.is_grad_enabled()),
        )

    def __call__(self, *args, **kwargs):
        # The Python submission and output clone must form one transaction:
        # otherwise another thread can replay into a shared output in between.
        with self._lock:
            return self._call_locked(*args, **kwargs)

    def _call_locked(self, *args, **kwargs):
        if self.owner is not None:
            signature = self._execution_signature()
            if signature != self._owner_signature:
                self.clear()
                self._owner_signature = signature
        transforms_active = getattr(
            torch._C, "_are_functorch_transforms_active", lambda: False
        )()
        if self.higher_order or transforms_active:
            return self.function(*args, **kwargs)
        if self._compile_enabled and self.compiled is None:
            self.compiled = torch.compile(self.function, dynamic=self._dynamic)
        function = self.compiled if self.compiled is not None else self.function
        arguments = self._graph_arguments(args, kwargs) if self.xpu_graph else None
        if arguments is not None:
            leaves, spec, tensors, key = arguments
            entry = self._graphs.get(key)
            if entry is not None and entry.busy:
                # A caller may delay backward. Replaying here would overwrite
                # the saved static inputs needed by that earlier backward.
                return self._run_uncaptured(*args, **kwargs)
            if entry is None and key not in self._failed_captures:
                indices = [
                    index
                    for index, value in enumerate(leaves)
                    if torch.is_tensor(value)
                ]
                # Never retain the caller's tensors or autograd tape in a
                # cached closure; make_graphed_callables owns detached samples.
                static_leaves = [
                    None if torch.is_tensor(value) else value for value in leaves
                ]

                def tensor_function(*values):
                    current = list(static_leaves)
                    for index, value in zip(indices, values):
                        current[index] = value
                    positional, named = _pytree.tree_unflatten(current, spec)
                    output = function(*positional, **named)
                    output_leaves = _pytree.tree_leaves(output)
                    if not output_leaves or any(
                        not torch.is_tensor(value) for value in output_leaves
                    ):
                        raise TypeError("XPU graph outputs must contain only tensors")
                    return output

                samples = tuple(
                    t.detach().clone().requires_grad_(t.requires_grad) for t in tensors
                )
                try:
                    with torch.xpu.device(tensors[0].device):
                        graphed = torch.xpu.make_graphed_callables(
                            tensor_function, samples, allow_unused_input=True
                        )
                    entry = _GraphEntry(graphed)
                    self._graphs[key] = entry
                    self.capture_count += 1
                    while len(self._graphs) > self.graph_cache_size:
                        self._graphs.popitem(last=False)
                except (
                    RuntimeError,
                    AssertionError,
                    ValueError,
                    TypeError,
                    NotImplementedError,
                ) as error:
                    # Device faults and OOM are not unsupported-capture cases.
                    if any(
                        term in str(error).lower()
                        for term in ("out of memory", "device_lost", "device lost")
                    ):
                        raise
                    self._failed_captures[key] = None
                    while len(self._failed_captures) > 2 * self.graph_cache_size:
                        self._failed_captures.popitem(last=False)
                    logging.warning(
                        "XPU graph capture unavailable for this block/shape; using eager execution: %s",
                        error,
                    )
                    entry = None
            if entry is not None:
                self._graphs.move_to_end(key)
                self.replay_count += 1
                grad_inputs = tuple(t for t in tensors if t.requires_grad)
                entry.busy = bool(grad_inputs) and torch.is_grad_enabled()
                try:
                    # PyTorch's graphed backward returns views of its static
                    # gradient buffers. Clone at the input's backward boundary
                    # so a subsequent replay cannot overwrite autograd.grad's
                    # previously returned tensors.
                    owned_inputs = tuple(
                        _OwnedGraphInput.apply(value)
                        if value.requires_grad and torch.is_grad_enabled()
                        else value
                        for value in tensors
                    )
                    output = _pytree.tree_map(
                        lambda value: value.clone(), entry.function(*owned_inputs)
                    )
                except Exception:
                    entry.busy = False
                    raise
                output_leaves, _ = _pytree.tree_flatten(output)
                if entry.busy and any(value.requires_grad for value in output_leaves):

                    def queue_release(gradient):
                        # Output hooks run before the captured backward. Release
                        # its inputs only after the complete graph task finishes.
                        # Unlike register_multi_grad_hook on leaf inputs, this
                        # also supports torch.autograd.grad used by force heads.
                        retained = (
                            torch._C._autograd._get_current_graph_task_keep_graph()
                        )

                        def release():
                            if not retained:
                                with self._lock:
                                    entry.busy = False
                                    for handle in entry.hooks:
                                        handle.remove()
                                    entry.hooks.clear()

                        torch.autograd.Variable._execution_engine.queue_callback(
                            release
                        )
                        return gradient

                    entry.hooks = [
                        value.register_hook(queue_release)
                        for value in output_leaves
                        if value.requires_grad
                    ]
                else:
                    entry.busy = False
                return output
        return self._run_uncaptured(*args, **kwargs)

    def _run_uncaptured(self, *args, **kwargs):
        function = self.compiled if self.compiled is not None else self.function
        try:
            return function(*args, **kwargs)
        except (
            torch._dynamo.exc.BackendCompilerFailed,
            torch._dynamo.exc.Unsupported,
        ) as error:
            if self.compiled is None:
                raise
            logging.warning(
                "Numeric block compilation unavailable; using eager execution: %s",
                error,
            )
            self.compiled = None
            self._compile_enabled = False
            self.clear()
            return self.function(*args, **kwargs)


@dataclass
class _GraphEntry:
    function: object
    busy: bool = False
    hooks: object = None


def configure_numeric_runtime(backbone, settings):
    """
    Install per-block runtimes without changing parameter names or state_dict.
    """
    enabled_compile = settings.compile and settings.compile_scope == "blocks"
    enabled = enabled_compile or settings.xpu_graph
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        if not enabled:
            return []
        raise ValueError(
            "Numeric-block compile/graph execution requires a block-based backbone"
        )
    if enabled and settings.activation_checkpointing:
        raise ValueError(
            "Numeric-block compile/graph execution requires activation_checkpointing=False"
        )
    if settings.xpu_graph and getattr(backbone, "num_experts", 0):
        raise ValueError("XPU graph replay requires merged MoLE weights")
    targets = [(block, "forward") for block in blocks]
    if hasattr(backbone, "_compute_geometry"):
        targets.append((backbone, "_compute_geometry"))
    elif hasattr(backbone, "_portable_geometry"):
        targets.append((backbone._portable_geometry, "forward"))
    radial = getattr(backbone, "_unified_radial_mlp", None)
    if radial is not None:
        targets.append((radial, "forward"))
        if hasattr(radial, "forward_decomposed"):
            targets.append((radial, "forward_decomposed"))
    runtimes = []
    replacements = []
    for module, method_name in targets:
        current = getattr(module, method_name)
        if (
            enabled
            and isinstance(current, NumericCoreRuntime)
            and not isinstance(current.function, NumericCoreRuntime)
            and current._compile_enabled == enabled_compile
            and current._dynamic == settings.compile_dynamic
            and current.xpu_graph == settings.xpu_graph
            and current.graph_cache_size == settings.xpu_graph_cache_size
        ):
            runtimes.append(current)
            continue
        function = current
        while isinstance(function, NumericCoreRuntime):
            function = function.function
        if not enabled:
            replacements.append((module, method_name, current, function))
            continue
        runtime = NumericCoreRuntime(
            function,
            compile=enabled_compile,
            dynamic=settings.compile_dynamic,
            xpu_graph=settings.xpu_graph,
            graph_cache_size=settings.xpu_graph_cache_size,
        )
        replacements.append((module, method_name, current, runtime))
        runtimes.append(runtime)
    # Construct every replacement first. A failing compile configuration must
    # not leave only a subset of the backbone wrapped.
    for module, method_name, previous, replacement in replacements:
        if isinstance(previous, NumericCoreRuntime):
            previous.clear()
        setattr(module, method_name, replacement)
    return runtimes

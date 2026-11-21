"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import gc
import pickle
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils import _pytree

from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.inference_runtime import (
    ModelStreamGuard,
    NumericCoreRuntime,
    PreparedModelCache,
    configure_numeric_runtime,
    model_tensor_signature,
)


class _ModeModule(torch.nn.Module):
    def forward(self, x):
        return x * (1 if self.training else 2)


class _Uncopyable:
    def __getstate__(self):
        raise RuntimeError("A device capture/event must not be serialized")


def _backbone():
    module = torch.nn.Module()
    module.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 2)])
    return module.eval().requires_grad_(False)


def _virtual_arguments(args, kwargs):
    leaves, spec = _pytree.tree_flatten((args, kwargs))
    tensors = [value for value in leaves if torch.is_tensor(value)]
    return leaves, spec, tensors, "fixed"


def _fake_capture_api(monkeypatch, runtime, capture):
    # This emulates capture ownership only; it does not execute GPU kernels.
    monkeypatch.setattr(runtime, "_graph_arguments", _virtual_arguments)
    monkeypatch.setattr(torch.xpu, "device", lambda selected: nullcontext())
    monkeypatch.setattr(torch.xpu, "make_graphed_callables", capture)


def test_reconfigure_is_idempotent_and_disabling_restores_original_forward():
    backbone = _backbone()
    block = backbone.blocks[0]
    original = block.forward
    state_names = tuple(backbone.state_dict())
    settings = InferenceSettings(activation_checkpointing=False, xpu_graph=True)
    first = configure_numeric_runtime(backbone, settings)
    second = configure_numeric_runtime(backbone, settings)
    assert first == second
    assert block.forward is first[0]
    assert not isinstance(first[0].function, NumericCoreRuntime)
    value = torch.randn(4, 3)
    torch.testing.assert_close(block(value), original(value))
    disabled = configure_numeric_runtime(backbone, InferenceSettings())
    assert disabled == []
    assert block.forward == original
    assert tuple(backbone.state_dict()) == state_names
    torch.testing.assert_close(block(value), original(value))


def test_failed_reconfiguration_does_not_leave_partially_wrapped_modules(monkeypatch):
    backbone = _backbone()
    backbone.blocks.append(torch.nn.Linear(2, 2))
    originals = [block.forward for block in backbone.blocks]
    compiled = []

    def compile_function(function, **kwargs):
        compiled.append(function)
        if len(compiled) == 2:
            raise RuntimeError("unsupported compiler configuration")
        return function

    monkeypatch.setattr(torch, "compile", compile_function)
    with pytest.raises(RuntimeError, match="unsupported compiler"):
        configure_numeric_runtime(
            backbone,
            InferenceSettings(
                compile=True, compile_scope="blocks", activation_checkpointing=False
            ),
        )
    assert [block.forward for block in backbone.blocks] == originals


@pytest.mark.parametrize(
    "operation", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_copy_and_pickle_drop_capture_state_and_bind_the_restored_model(operation):
    backbone = _backbone()
    runtime = configure_numeric_runtime(
        backbone, InferenceSettings(activation_checkpointing=False, xpu_graph=True)
    )[0]
    runtime._graphs["unpickleable"] = _Uncopyable()
    runtime._failed_captures["old_shape"] = None
    runtime.compiled = _Uncopyable()
    runtime.capture_count = 3
    copied = operation(backbone)
    copied_runtime = copied.blocks[0].forward
    assert copied_runtime.owner is copied.blocks[0]
    assert copied_runtime.function.__self__ is copied.blocks[0]
    assert copied_runtime.function.__func__ is torch.nn.Linear.forward
    assert not copied_runtime._graphs
    assert not copied_runtime._failed_captures
    assert copied_runtime.capture_count == 0
    assert copied_runtime.compiled is None
    with torch.no_grad():
        copied.blocks[0].weight.add_(2)
    value = torch.randn(4, 3)
    torch.testing.assert_close(
        copied.blocks[0](value),
        torch.nn.functional.linear(
            value, copied.blocks[0].weight, copied.blocks[0].bias
        ),
    )
    assert not torch.equal(copied.blocks[0].weight, backbone.blocks[0].weight)
    assert runtime._graphs["unpickleable"].__class__ is _Uncopyable


def test_deepcopied_compiled_runtime_recompiles_against_copied_weights(monkeypatch):
    calls = []

    def compiler(function, **kwargs):
        calls.append(function.__self__)
        return function

    monkeypatch.setattr(torch, "compile", compiler)
    backbone = _backbone()
    configure_numeric_runtime(
        backbone,
        InferenceSettings(
            compile=True, compile_scope="blocks", activation_checkpointing=False
        ),
    )
    copied = copy.deepcopy(backbone)
    assert len(calls) == 1
    value = torch.randn(4, 3)
    copied.blocks[0](value)
    assert calls == [backbone.blocks[0], copied.blocks[0]]


def test_graph_capture_invalidates_train_eval_mode_changes(monkeypatch):
    module = _ModeModule()
    runtime = NumericCoreRuntime(module.forward, xpu_graph=True)

    def capture(function, samples, **kwargs):
        fixed_multiplier = function(torch.ones_like(samples[0])).clone()
        return lambda x: x * fixed_multiplier

    _fake_capture_api(monkeypatch, runtime, capture)
    value = torch.tensor([3.0])
    torch.testing.assert_close(runtime(value), value)
    module.eval()
    torch.testing.assert_close(runtime(value), 2 * value)
    assert runtime.capture_count == 2


def test_trainable_buffers_decline_capture_and_keep_parameter_derivatives(monkeypatch):
    module = torch.nn.Module()
    module.register_buffer("gain", torch.ones(2, requires_grad=True))
    module.forward = lambda x: x * module.gain
    # Bind an actual method so the runtime can inspect the owner buffers.
    runtime = NumericCoreRuntime(module.__call__, xpu_graph=True)
    with FakeTensorMode(allow_fallback_kernels=False):
        value = torch.empty(2, device="xpu:0")
        assert runtime._graph_arguments((value,), {}) is None
    actual = runtime(torch.tensor([2.0, 3.0], requires_grad=True))
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), module.gain)[0], torch.tensor([2.0, 3.0])
    )


def test_signature_retains_replaced_parameters_and_tracks_gradient_policy():
    module = torch.nn.Linear(3, 2)
    signature = model_tensor_signature(module)
    previous = weakref.ref(module.weight)
    module.weight = torch.nn.Parameter(torch.ones_like(module.weight))
    assert previous() is not None
    assert signature != model_tensor_signature(module)
    del signature
    gc.collect()
    assert previous() is None
    signature = model_tensor_signature(module)
    module.requires_grad_(False)
    assert signature != model_tensor_signature(module)


def test_metadata_outputs_fall_back_instead_of_failing_on_requires_grad(monkeypatch):
    runtime = NumericCoreRuntime(
        lambda x: {"value": x.square(), "unit": "eV"}, xpu_graph=True
    )

    def capture(function, samples, **kwargs):
        function(*samples)
        pytest.fail("Non-tensor output must reject capture before replay")

    _fake_capture_api(monkeypatch, runtime, capture)
    result = runtime(torch.tensor([2.0]))
    assert result["unit"] == "eV"
    torch.testing.assert_close(result["value"], torch.tensor([4.0]))
    assert runtime.capture_count == 0
    assert len(runtime._failed_captures) == 1


def test_concurrent_replays_cannot_overwrite_another_jobs_output(monkeypatch):
    runtime = NumericCoreRuntime(lambda x: x.square(), xpu_graph=True)
    first_entered, release_first, second_started, second_entered = (
        Event() for _ in range(4)
    )

    def capture(function, samples, **kwargs):
        buffer = torch.empty_like(samples[0])

        def replay(x):
            buffer.copy_(function(x))
            if x[0] == 2:
                first_entered.set()
                assert release_first.wait(timeout=5)
            else:
                second_entered.set()
            return buffer

        return replay

    _fake_capture_api(monkeypatch, runtime, capture)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(runtime, torch.tensor([2.0, 3.0]))
        assert first_entered.wait(timeout=5)

        def second_call():
            second_started.set()
            return runtime(torch.tensor([4.0, 5.0]))

        second = pool.submit(second_call)
        assert second_started.wait(timeout=5)
        try:
            assert not second_entered.wait(timeout=0.1)
        finally:
            release_first.set()
        torch.testing.assert_close(first.result(timeout=5), torch.tensor([4.0, 9.0]))
        torch.testing.assert_close(second.result(timeout=5), torch.tensor([16.0, 25.0]))


def test_captured_backward_gradients_own_storage_across_later_jobs(monkeypatch):
    runtime = NumericCoreRuntime(lambda x: x.square(), xpu_graph=True)

    def capture(function, samples, **kwargs):
        static_input = samples[0].detach().clone()
        static_gradient = torch.empty_like(static_input)

        class Captured(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                static_input.copy_(value)
                return function(value)

            @staticmethod
            def backward(ctx, gradient):
                static_gradient.copy_(2 * static_input * gradient)
                return static_gradient

        return Captured.apply

    _fake_capture_api(monkeypatch, runtime, capture)
    first = torch.tensor([2.0, 3.0], requires_grad=True)
    first_gradient = torch.autograd.grad(runtime(first).sum(), first)[0]
    second = torch.tensor([4.0, 5.0], requires_grad=True)
    second_gradient = torch.autograd.grad(runtime(second).sum(), second)[0]
    torch.testing.assert_close(first_gradient, 2 * first)
    torch.testing.assert_close(second_gradient, 2 * second)
    assert runtime.replay_count == 2


@pytest.mark.parametrize(
    "operation", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_model_stream_guard_restores_without_reusing_device_events(operation):
    guard = ModelStreamGuard(torch.nn.Linear(3, 2))
    guard.ready["old_gpu"] = _Uncopyable()
    restored = operation(guard)
    assert not restored.ready
    restored.wait_and_record(torch.nn.Linear(3, 2))
    assert not restored.ready
    assert not restored._restore
    assert "old_gpu" in guard.ready


def test_restored_stream_guard_orders_copied_weights_once(monkeypatch):
    device = torch.device("xpu:1")
    calls = []
    stream = object()
    event = SimpleNamespace(record=lambda selected: calls.append(("record", selected)))
    tensor = SimpleNamespace(device=device)
    model = SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]))
    monkeypatch.setattr(torch.xpu, "current_stream", lambda selected: stream)
    monkeypatch.setattr(torch.xpu, "Event", lambda: event)
    monkeypatch.setattr(
        torch.xpu, "synchronize", lambda selected: calls.append(("sync", selected))
    )
    guard = ModelStreamGuard.__new__(ModelStreamGuard)
    guard.__setstate__({})
    guard.wait_and_record(model)
    guard.wait_and_record(model)
    assert calls == [("sync", device), ("record", stream)]


@pytest.mark.parametrize(
    "value", [True, False, 0, -1, 1.5, float("nan"), float("inf"), "2", None]
)
def test_runtime_cache_limits_reject_noninteger_or_nonpositive_values(value):
    with pytest.raises(ValueError, match="positive integer"):
        NumericCoreRuntime(lambda x: x, graph_cache_size=value)
    with pytest.raises(ValueError, match="positive integer"):
        PreparedModelCache(value)
    if value is not None:
        with pytest.raises(ValueError, match="positive integer"):
            PreparedModelCache(1, value)

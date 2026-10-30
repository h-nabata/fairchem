"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import sys
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from torch.utils import _pytree

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.inference_runtime import (
    ModelStreamGuard,
    NumericCoreRuntime,
    PreparedModelCache,
    configure_numeric_runtime,
    model_storage_bytes,
    mole_composition_key,
)
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit, collate_predictions
from tests.core.units.mlip_unit.test_xpu_preparation import _data, _small_predictor


def _key(data, device="xpu:0"):
    return mole_composition_key(
        data,
        device=device,
        dtype=torch.float32,
        model_id="UMA-S-1.2",
        settings_key="strict",
    )


def test_mole_key_uses_exact_counts_all_systems_metadata_and_device():
    one = _data()
    two = one.clone()
    two.atomic_numbers = one.atomic_numbers.flip(0)
    two.pos = one.pos.flip(0) + 50
    assert _key(one) == _key(two)
    two.charge += 1
    assert _key(one) != _key(two)
    assert _key(one, "xpu:0") != _key(one, "xpu:1")
    batch = atomicdata_list_to_batch([one, one])
    changed_batch = atomicdata_list_to_batch([one, two])
    assert _key(batch) != _key(changed_batch)
    # Same reduced composition, twice as many atoms: a different routing key.
    doubled = one.clone()
    doubled.atomic_numbers = one.atomic_numbers.repeat(2)
    doubled.batch = one.batch.repeat(2)
    doubled.natoms = one.natoms * 2
    assert _key(one) != _key(doubled)


def test_prepared_cache_lru_budget_and_changed_parameters():
    models = [torch.nn.Linear(3, 2) for _ in range(3)]
    size = model_storage_bytes(models[0])
    cache = PreparedModelCache(5, 2 * size)
    cache.put("a", models[0])
    cache.put("b", models[1])
    assert cache.get("a") is models[0]
    cache.put("c", models[2])
    assert cache.get("b") is None
    assert len(cache) == 2
    assert cache.bytes == 2 * size
    with torch.no_grad():
        models[0].weight.add_(1)
    assert cache.get("a") is None
    assert len(cache) == 1
    assert cache.bytes == size
    cache.clear()
    assert len(cache) == cache.bytes == 0


def test_cache_counts_shared_storage_once_and_skips_oversize():
    module = torch.nn.Module()
    weight = torch.randn(4, 4)
    module.register_buffer("first", weight)
    module.register_buffer("second", weight[:, :2])
    assert model_storage_bytes(module) == weight.untyped_storage().nbytes()
    cache = PreparedModelCache(1, 1)
    cache.put("too-large", module)
    assert cache.get("too-large") is None
    assert cache.bytes == 0


def test_prepared_model_stream_guard_orders_creation_and_retains_cross_stream_storage(
    monkeypatch,
):
    device = torch.device("xpu:1")
    calls = []
    origin = SimpleNamespace(
        wait_event=lambda event: calls.append(("unexpected", event))
    )
    consumer = SimpleNamespace(wait_event=lambda event: calls.append(("wait", event)))
    current = [origin]
    event = SimpleNamespace(record=lambda stream: calls.append(("record", stream)))
    tensor = SimpleNamespace(
        device=device, record_stream=lambda stream: calls.append(("storage", stream))
    )
    model = SimpleNamespace(parameters=lambda: iter([tensor]), buffers=lambda: iter([]))
    monkeypatch.setattr(torch.xpu, "current_stream", lambda selected: current[0])
    monkeypatch.setattr(torch.xpu, "Event", lambda: event)
    guard = ModelStreamGuard(model)
    guard.wait_and_record(model)
    assert calls == [("record", origin)]
    current[0] = consumer
    guard.wait_and_record(model)
    assert calls == [("record", origin), ("wait", event), ("storage", consumer)]


def _raw_predictor(cache_size=0):
    predictor = _small_predictor()
    predictor.inference_settings.mole_cache_size = cache_size
    # Use actual UMA outputs, without checkpoint task normalization fixtures.
    predictor._process_outputs = lambda data, output, undo_refs: output
    return predictor


def _raw_predict(predictor, data):
    return MLIPPredictUnit.predict.__wrapped__(predictor, data)


def test_real_predictor_switches_cached_mole_models_and_matches_unmerged():
    candidate = _raw_predictor(2)
    reference = _raw_predictor()
    reference.inference_settings.merge_mole = False
    inputs = [_data(), _data(), _data()]
    for data in inputs:
        data.cell = torch.eye(3).unsqueeze(0) * 10
    inputs[1].charge += 1
    inputs[2].spin += 1
    original_model = None
    for index in (0, 1, 0, 2):
        actual = _raw_predict(candidate, inputs[index])
        expected = _raw_predict(reference, inputs[index])
        for name, value in actual.items():
            # Merged heads intentionally put NaN in absent datasets; only the
            # selected dataset is a prediction exposed by the public API.
            if not name.startswith("omol_"):
                continue
            if torch.is_tensor(value):
                torch.testing.assert_close(value, expected[name], atol=2e-5, rtol=2e-4)
            else:
                for prop, tensor in value.items():
                    torch.testing.assert_close(
                        tensor, expected[name][prop], atol=2e-5, rtol=2e-4
                    )
        if index == 0:
            if original_model is None:
                original_model = candidate.model
            else:
                assert candidate.model is original_model
        assert candidate.inference_settings.merge_mole
        assert getattr(candidate.model.module.backbone, "num_experts", 0) == 0
        assert len(candidate._prepared_models) <= 2
    assert candidate._prepared_models.hits == 1
    assert candidate._mole_template.module.backbone.num_experts == 2


def test_real_cache_detects_weight_edit_and_can_clear_without_disabling_merging():
    predictor = _raw_predictor(2)
    _raw_predict(predictor, _data())
    predictor.clear_prepared_model_cache()
    changed = _data()
    changed.charge += 1
    _raw_predict(predictor, changed)
    assert predictor.inference_settings.merge_mole
    with torch.no_grad():
        next(predictor.model.parameters()).add_(0.1)
    with pytest.raises(RuntimeError, match="weights/buffers changed"):
        _raw_predict(predictor, changed)


def test_numeric_compile_keeps_higher_derivatives_on_eager_path(monkeypatch):
    original_compile = torch.compile
    calls = []

    def compile_for_test(fn, **kwargs):
        calls.append(kwargs)
        return original_compile(fn, backend="eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_for_test)
    runtime = NumericCoreRuntime(
        lambda x: x.sin().square(), compile=True, dynamic=False
    )
    x = torch.randn(7, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(runtime(x), x.sin().square())
    runtime.higher_order = True
    assert torch.autograd.gradcheck(runtime, (x,))
    assert torch.autograd.gradgradcheck(runtime, (x,))
    assert calls == [{"dynamic": False}]


def test_energy_only_initialization_in_inference_mode_keeps_trainable_inputs_usable():
    predictor = _raw_predictor(2)
    predictor.inference_settings.xpu_graph = True
    data = _data()
    with torch.inference_mode():
        output = MLIPPredictUnit.predict.__wrapped__(
            predictor, data, requested_properties=("energy",)
        )
    assert torch.isfinite(output["omol_energy"]["energy"]).all()
    assert all(
        not parameter.is_inference() for parameter in predictor.model.parameters()
    )
    output = MLIPPredictUnit.predict.__wrapped__(
        predictor, data, requested_properties=("forces",)
    )
    assert torch.isfinite(output["omol_forces"]["forces"]).all()


def test_real_uma_regional_compile_installs_geometry_and_bypasses_for_hvp(monkeypatch):
    original_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: original_compile(fn, backend="eager", **kwargs),
    )
    predictor = _raw_predictor()
    predictor.inference_settings.compile = True
    predictor.inference_settings.compile_scope = "blocks"
    data = _data()
    actual = MLIPPredictUnit.predict.__wrapped__(
        predictor, data, requested_properties=("energy",)
    )
    reference = _raw_predictor()
    expected = MLIPPredictUnit.predict.__wrapped__(
        reference, data, requested_properties=("energy",)
    )
    torch.testing.assert_close(
        actual["omol_energy"]["energy"], expected["omol_energy"]["energy"]
    )
    assert len(predictor._numeric_runtimes) == 2  # One block and joint geometry.
    for runtime in predictor._numeric_runtimes:
        runtime.compiled = lambda *args, **kwargs: pytest.fail(
            "HVP used compiled callable"
        )
    data["hessian_vector"] = torch.ones_like(data.pos)
    actual = MLIPPredictUnit.predict.__wrapped__(
        predictor, data, requested_properties=("hessian_vector_product",)
    )
    expected = MLIPPredictUnit.predict.__wrapped__(
        reference, data, requested_properties=("hessian_vector_product",)
    )
    torch.testing.assert_close(
        actual["omol_hessian_vector_product"]["hessian_vector_product"],
        expected["omol_hessian_vector_product"]["hessian_vector_product"],
    )


def test_runtime_does_not_capture_cpu_and_preserves_parameter_names():
    backbone = torch.nn.Module()
    backbone.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 2)])
    before = set(backbone.state_dict())
    settings = InferenceSettings(activation_checkpointing=False, xpu_graph=True)
    runtimes = configure_numeric_runtime(backbone, settings)
    x = torch.randn(4, 3)
    expected = torch.nn.functional.linear(
        x, backbone.blocks[0].weight, backbone.blocks[0].bias
    )
    torch.testing.assert_close(backbone.blocks[0](x), expected)
    assert set(backbone.state_dict()) == before
    assert runtimes[0].capture_count == 0


def test_virtual_graph_replay_owns_samples_bounds_shapes_and_copies_outputs(
    monkeypatch,
):
    runtime = NumericCoreRuntime(
        lambda x: x.square(), xpu_graph=True, graph_cache_size=1
    )
    calls = []

    def arguments(args, kwargs):
        leaves, spec = _pytree.tree_flatten((args, kwargs))
        return leaves, spec, list(args), tuple(tuple(value.shape) for value in args)

    def capture(fn, samples, **kwargs):
        calls.append(weakref.ref(samples[0]))
        output = fn(*samples).detach().clone()

        def replay(*values):
            output.copy_(fn(*values))
            return output

        return replay

    monkeypatch.setattr(runtime, "_graph_arguments", arguments)
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "make_graphed_callables", capture)
    x = torch.tensor([2.0, 3.0])
    owner = weakref.ref(x)
    first = runtime(x)
    second = runtime(torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(first, torch.tensor([4.0, 9.0]))
    torch.testing.assert_close(second, torch.tensor([16.0, 25.0]))
    del x
    gc.collect()
    assert owner() is None
    runtime(torch.ones(3))
    assert len(runtime._graphs) == 1
    assert runtime.capture_count == 2
    runtime.higher_order = True
    x = torch.ones(3, requires_grad=True)
    assert torch.autograd.gradgradcheck(runtime, (x.double(),))


def test_hvp_postprocessing_scales_without_energy_mean_and_collates():
    task = SimpleNamespace(
        name="omol_energy",
        property="energy",
        level="system",
        normalizer=SimpleNamespace(rmsd=torch.tensor(2.0), mean=torch.tensor(17.0)),
    )
    predictor = SimpleNamespace(
        model=SimpleNamespace(module=SimpleNamespace(tasks={task.name: task})),
        assert_on_nans=False,
        dataset_to_tasks={"omol": [task]},
    )
    data = atomicdata_list_to_batch([_data(), _data()])
    value = torch.arange(18.0).reshape(6, 3)
    output = {"omol_hessian_vector_product": {"hessian_vector_product": value}}
    processed = MLIPPredictUnit._process_outputs(predictor, data, output, True)

    @collate_predictions
    def run(unit, data, undo_refs, **kwargs):
        return processed

    torch.testing.assert_close(
        run(predictor, data)["hessian_vector_product"], 2 * value
    )


def test_xpu_rejects_relaxed_onednn_flag_before_any_gpu_work(monkeypatch):
    predictor = _small_predictor("xpu:0")
    monkeypatch.setattr(torch.backends, "mkldnn", SimpleNamespace(allow_tf32=True))
    with pytest.raises(ValueError, match="mkldnn.allow_tf32=False"):
        predictor._run_inference(_data(), True)
    assert torch.backends.mkldnn.allow_tf32


@pytest.mark.parametrize("source", ["inductor", "intel_env", "intel_knob"])
def test_xpu_compilation_rejects_effective_fast_math_without_mutating_it(
    monkeypatch, source
):
    from torch._inductor import config as inductor_config

    predictor = _small_predictor("xpu:0")
    predictor.inference_settings.compile = True
    monkeypatch.setattr(torch.backends, "mkldnn", SimpleNamespace(allow_tf32=False))
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "0")
    if source == "intel_env":
        monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "1")
        monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    elif source == "intel_knob":
        monkeypatch.setitem(
            sys.modules,
            "triton",
            SimpleNamespace(
                knobs=SimpleNamespace(intel=SimpleNamespace(fast_math=True))
            ),
        )
    with inductor_config.patch(use_fast_math=source == "inductor"):
        with pytest.raises(ValueError, match="Inductor use_fast_math=False"):
            predictor._run_inference(_data(), True)
        assert inductor_config.use_fast_math == (source == "inductor")
    assert not torch.backends.mkldnn.allow_tf32
    if source == "intel_knob":
        assert sys.modules["triton"].knobs.intel.fast_math


def test_single_atom_lookup_filters_requests_and_rejects_unavailable_outputs():
    predictor = _small_predictor()
    tasks = [
        Task(
            name=f"omol_{prop}",
            property=prop,
            level=level,
            out_spec=OutputSpec(dim=[dim], dtype="float32"),
            datasets=["omol"],
            normalizer=Normalizer(),
        )
        for prop, level, dim in (("energy", "system", 1), ("forces", "atom", 3))
    ]
    predictor.model.module._tasks = {task.name: task for task in tasks}
    predictor.model.module._dataset_to_tasks = {"omol": tasks}
    predictor.atom_refs = {"omol": {1: {0: 1.5}}}
    data = AtomicData.from_ase(
        Atoms("H", positions=[[0, 0, 0]], info={"charge": 0, "spin": 1}),
        task_name="omol",
    )
    result = predictor.predict(data, requested_properties=("energy",))
    assert set(result) == {"energy"}
    torch.testing.assert_close(result["energy"], torch.tensor([1.5]))
    with pytest.raises(ValueError, match="lookup cannot provide"):
        predictor.predict(data, requested_properties=("embeddings",))


def test_virtual_graph_delayed_autograd_grad_and_retained_graph_keep_inputs(
    monkeypatch,
):
    runtime = NumericCoreRuntime(lambda x: x.square(), xpu_graph=True)

    def arguments(args, kwargs):
        leaves, spec = _pytree.tree_flatten((args, kwargs))
        return leaves, spec, list(args), "fixed"

    def capture(fn, samples, **kwargs):
        static_input = samples[0].detach().clone()

        class Captured(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                static_input.copy_(x)
                ctx.save_for_backward(static_input)
                return static_input.square()

            @staticmethod
            def backward(ctx, gradient):
                (saved,) = ctx.saved_tensors
                return 2 * saved * gradient

        return Captured.apply

    monkeypatch.setattr(runtime, "_graph_arguments", arguments)
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "make_graphed_callables", capture)
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    first = runtime(x)
    entry = runtime._graphs["fixed"]
    assert entry.busy
    y = torch.tensor([4.0, 5.0], requires_grad=True)
    runtime(y)
    assert runtime.replay_count == 1
    torch.testing.assert_close(
        torch.autograd.grad(first.sum(), x, retain_graph=True)[0], 2 * x
    )
    assert entry.busy
    runtime(y)
    assert runtime.replay_count == 1
    torch.testing.assert_close(torch.autograd.grad(first.sum(), x)[0], 2 * x)
    assert not entry.busy
    second = runtime(y)
    assert runtime.replay_count == 2
    torch.testing.assert_close(torch.autograd.grad(second.sum(), y)[0], 2 * y)
    torch.testing.assert_close(first, x.square())


@pytest.mark.parametrize(
    "merge,datasets", [(True, ("omol",)), (True, ("omat",)), (False, ("omat", "omol"))]
)
def test_public_prediction_dynamic_derivatives_and_embeddings_selected_datasets(
    merge, datasets
):
    candidate = _small_predictor()
    reference = _raw_predictor()
    for predictor in (candidate, reference):
        predictor.inference_settings.merge_mole = merge
    tasks = [
        Task(
            name=f"{name}_energy",
            property="energy",
            level="system",
            out_spec=OutputSpec(dim=[1], dtype="float32"),
            datasets=[name],
            normalizer=Normalizer(mean=17.0, rmsd=2.0 if name == "omol" else 3.0),
        )
        for name in ("omol", "omat")
    ]
    candidate.model.module._tasks = {task.name: task for task in tasks}
    candidate.model.module._dataset_to_tasks = {
        task.datasets[0]: [task] for task in tasks
    }
    candidate.assert_on_nans = True
    items = []
    for name in datasets:
        item = _data()
        item.dataset = [name]
        item.cell = torch.eye(3).unsqueeze(0) * 10
        items.append(item)
    data = atomicdata_list_to_batch(items)
    requested = ("forces", "stress", "embeddings")
    actual = candidate.predict(data, requested_properties=requested)
    raw = MLIPPredictUnit.predict.__wrapped__(
        reference, data, requested_properties=requested
    )
    assert set(actual) == set(requested)
    assert "_requested_properties" not in data
    assert not data.pos.requires_grad
    for prop in requested:
        parts = []
        for i, dataset in enumerate(datasets):
            value = raw[f"{dataset}_{prop}"][prop]
            value = value[i : i + 1] if prop == "stress" else value[data.batch == i]
            scale = 1 if prop == "embeddings" else (2 if dataset == "omol" else 3)
            parts.append(value * scale)
        torch.testing.assert_close(actual[prop], torch.cat(parts), atol=2e-5, rtol=2e-4)
    energy = candidate.predict(data, requested_properties=("energy",))
    assert set(energy) == {"energy"}


@pytest.mark.skipif(
    not torch.xpu.is_available(), reason="Requires real Intel XPU graph support"
)
def test_xpu_numeric_graph_values_gradients_and_two_submission_streams():
    device = torch.device("xpu:0")
    runtime = NumericCoreRuntime(
        lambda x: x.sin().square(), xpu_graph=True, graph_cache_size=2
    )
    streams = [torch.xpu.Stream(device=device), torch.xpu.Stream(device=device)]
    for stream in streams:
        with torch.xpu.stream(stream):
            x = torch.linspace(0.2, 1.0, 16, device=device, requires_grad=True)
            actual = runtime(x)
            grad = torch.autograd.grad(actual.sum(), x)[0]
            torch.testing.assert_close(actual, x.sin().square())
            torch.testing.assert_close(grad, 2 * x.sin() * x.cos())
    assert runtime.capture_count == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"compile_scope": "all"},
        {"xpu_graph_cache_size": 0},
        {"mole_cache_size": -1},
        {"mole_cache_max_bytes": 0},
        {"neighbor_skin": -1},
        {"neighbor_cache_entries": 0},
        {"graph_search_method": "unknown"},
        {"graph_edge_order": "unknown"},
        {"edge_chunk_size": 0},
        {"hessian_chunk_size": 0},
        {"requested_properties": ()},
        {"requested_properties": "energy"},
    ],
)
def test_runtime_settings_reject_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        InferenceSettings(**kwargs)

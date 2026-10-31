"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")

nitro = pytest.importorskip("fairchem.core.models.uma.uma_nitro._inference")


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch):
    for name in ("_GRAPH_STATE", "_LIVE_EDGES", "_SIDE_STREAMS", "_WARMUP_STREAM"):
        monkeypatch.setattr(nitro, name, {})
    monkeypatch.setattr(nitro, "_GRAPH_POOL", None)
    monkeypatch.setattr(nitro, "_GRAPH_DEVICE", None)
    monkeypatch.setattr(nitro, "_GRAPH_REFERENCES", ())
    monkeypatch.setattr(nitro, "_GRAPH_INPUTS", ())
    monkeypatch.setattr(nitro, "_GRAPH_OUTPUT_READY", None)
    monkeypatch.setattr(nitro, "_CAPTURE_REFERENCES", None)
    monkeypatch.setattr(nitro, "_MAX_FORK", [0])
    monkeypatch.setattr(
        nitro, "_RUNTIME_COUNTERS", dict.fromkeys(nitro._RUNTIME_COUNTERS, 0)
    )
    buffers = (nitro._device_buffer, nitro._pinned_buffer, nitro._staging_ready)
    for function in buffers:
        function.cache_clear()
    for cache in nitro._constant_caches():
        cache.clear()
    yield
    for function in buffers:
        function.cache_clear()
    for cache in nitro._constant_caches():
        cache.clear()


def _data(device="cpu"):
    return SimpleNamespace(
        atomic_numbers=torch.ones(3, dtype=torch.long, device=device),
        pos=torch.ones(3, 3, device=device),
        cell=torch.ones(1, 3, 3, device=device),
        natoms=torch.full((1,), 3, dtype=torch.long, device=device),
        batch=torch.zeros(3, dtype=torch.long, device=device),
        pbc=torch.zeros(1, 3, dtype=torch.bool, device=device),
        charge=torch.zeros(1, device=device),
        spin=torch.ones(1, device=device),
        dataset=["omol"],
    )


@pytest.mark.parametrize(
    "environment", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"]
)
def test_allocator_respects_an_explicit_external_policy(monkeypatch, environment):
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(environment, "max_split_size_mb:64,expandable_segments:False")
    monkeypatch.setattr(nitro, "EXPANDABLE_SEGMENTS", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    calls = []
    monkeypatch.setattr(
        torch._C, "_accelerator_setAllocatorSettings", calls.append, raising=False
    )
    assert not nitro._configure_allocator()
    assert calls == []


def test_allocator_does_not_change_a_cpu_process(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert not nitro._configure_allocator()
    assert "PYTORCH_CUDA_ALLOC_CONF" not in nitro.os.environ


def test_allocator_extends_the_primary_environment_alias(monkeypatch):
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "max_split_size_mb:64")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setattr(nitro, "EXPANDABLE_SEGMENTS", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert nitro._configure_allocator()
    assert (
        nitro.os.environ["PYTORCH_ALLOC_CONF"]
        == "max_split_size_mb:64,expandable_segments:True"
    )
    assert "PYTORCH_CUDA_ALLOC_CONF" not in nitro.os.environ


def test_input_storage_is_bounded_and_old_shapes_are_released():
    first = nitro._device_buffer("pos", (1, 3), torch.float32, "cpu")
    reference = weakref.ref(first)
    del first
    for size in range(2, 100):
        nitro._device_buffer("pos", (size, 3), torch.float32, "cpu")
    gc.collect()
    assert nitro._device_buffer.cache_info().currsize <= 32
    assert reference() is None


def test_host_staging_is_bounded_and_ignores_ambient_device(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    with torch.device("meta"):
        for size in range(100):
            staged = nitro._pinned_buffer(
                "pos", size % 2, (size, 3), torch.float32, "cuda:0"
            )
            assert staged.device.type == "cpu"
    assert nitro._pinned_buffer.cache_info().currsize <= 32


def test_gpu_resident_inputs_skip_host_staging_events(monkeypatch):
    def unexpected_event(*args):
        raise AssertionError("GPU-resident input waited for host staging")

    class Buffer:
        # Exercise the real host dispatch without pretending a CPU build can
        # perform CUDA copies or autograd on CUDA storage.
        def __init__(self, shape, dtype):
            self.shape, self.dtype = shape, dtype
            self.device = torch.device("cuda:0")
            self.requires_grad = False

        def is_floating_point(self):
            return self.dtype.is_floating_point

        def to(self, dtype):
            return self

        def copy_(self, source, **kwargs):
            assert source.device == self.device

        def requires_grad_(self, value):
            self.requires_grad = value

    data = SimpleNamespace(
        **{
            name: Buffer(value.shape, value.dtype)
            for name, value in vars(_data()).items()
            if isinstance(value, torch.Tensor)
        }
    )
    monkeypatch.setattr(torch.cuda, "Event", unexpected_event)
    monkeypatch.setattr(
        nitro, "_device_buffer", lambda name, shape, dtype, device: Buffer(shape, dtype)
    )
    result = nitro._move_inputs(data, torch.device("cuda:0"), torch.float32)
    assert result["pos"].device == torch.device("cuda:0")
    assert result["pos"].requires_grad


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_staging_failure_finishes_previous_copies(monkeypatch, error_type):
    events = []

    class Event:
        def synchronize(self):
            events.append("wait")

        def record(self):
            events.append("record")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    original = nitro._device_buffer

    def allocate(name, *args):
        if name == "cell":
            raise error_type("allocation failed after earlier copies")
        return original(name, *args)

    monkeypatch.setattr(nitro, "_device_buffer", allocate)
    with pytest.raises(error_type, match="after earlier copies"):
        nitro._move_inputs(_data(), "cpu", torch.float32)
    assert events == ["wait", "record", "wait"]
    # Restore the cache-bearing callable for the fixture's cleanup.
    monkeypatch.setattr(nitro, "_device_buffer", original)


def test_streams_and_staging_events_are_device_specific(monkeypatch):
    current = [0]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda: SimpleNamespace(device=current[0])
    )
    monkeypatch.setattr(torch.cuda, "Event", lambda: object())
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda tensor: tensor)
    first = nitro._side_streams(2)
    warm_first = nitro._warmup_stream()
    event_first = nitro._staging_ready(0, "cuda:0")
    staging_first = nitro._pinned_buffer("pos", 0, (3, 3), torch.float32, "cuda:0")
    current[0] = 1
    second = nitro._side_streams(2)
    assert all(stream.device == 1 for stream in second)
    assert second[0] is not first[0]
    assert nitro._warmup_stream() is not warm_first
    assert nitro._staging_ready(0, "cuda:1") is not event_first
    staging_second = nitro._pinned_buffer("pos", 0, (3, 3), torch.float32, "cuda:1")
    assert staging_first.data_ptr() != staging_second.data_ptr()
    current[0] = 0
    assert nitro._side_streams(2) == first


def test_compiled_handle_cache_is_device_specific_and_bounded(monkeypatch):
    current = [0]
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])

    @nitro._device_lru_cache(maxsize=2)
    def compile_handle(shape):
        return current[0], shape, object()

    first = compile_handle(32)
    assert compile_handle(32) is first
    current[0] = 1
    second = compile_handle(32)
    assert second is not first
    assert second[0] == 1
    current[0] = 0
    assert compile_handle(32) is first
    compile_handle(64)
    assert compile_handle.cache_info().currsize == 2
    compile_handle.cache_clear()
    assert compile_handle.cache_info().currsize == 0


@pytest.mark.parametrize("failure_at", [1, 4])
def test_capture_failures_join_streams_and_release_partial_graph(
    monkeypatch, failure_at
):
    events = []
    side = SimpleNamespace(wait_stream=lambda stream: events.append("side_wait"))
    main = SimpleNamespace(wait_stream=lambda stream: events.append("main_wait"))
    graph = SimpleNamespace(reset=lambda: events.append("reset"), pool=lambda: object())
    monkeypatch.setattr(nitro, "_warmup_stream", lambda: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: main)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        if builds == failure_at:
            raise RuntimeError("capture interrupted")
        return {}

    with pytest.raises(RuntimeError, match="capture interrupted"):
        nitro._capture(build)
    assert events[-2:] == (
        ["sync", "reset"] if failure_at == 4 else ["main_wait", "sync"]
    )
    assert not nitro._SEQ_FORK
    assert nitro._GRAPH_POOL is None
    assert nitro._GRAPH_DEVICE is None


def test_live_graph_retains_evicted_constants_until_synchronized_reset(monkeypatch):
    stream = SimpleNamespace(wait_stream=lambda stream: None)
    references = []
    calls = 0

    def reset():
        assert references[0]() is not None

    graph = SimpleNamespace(reset=reset, pool=lambda: object())
    monkeypatch.setattr(nitro, "_warmup_stream", lambda: stream)
    monkeypatch.setattr(nitro, "_TENSOR_CACHE_LIMIT", 1)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", lambda *args, **kwargs: nullcontext())

    def build():
        nonlocal calls
        calls += 1
        if calls == 4:
            first = nitro._repeat_arange(3, 1, "cpu")
            references.append(weakref.ref(first))
            nitro._repeat_arange(6, 1, "cpu")
        return {}

    nitro._GRAPH_STATE["current"] = nitro._capture(build)
    assert len(nitro._REPEAT_ARANGE) == 1
    assert references[0]() is not None
    assert nitro._CAPTURE_REFERENCES is None
    nitro._flush_graphs()
    assert references[0]() is None
    assert not nitro._GRAPH_REFERENCES


def test_graph_reset_synchronizes_its_owner_device(monkeypatch):
    current = [0]
    observed = []

    @contextmanager
    def scope(device):
        previous = current[0]
        current[0] = device
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "device", scope)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: observed.append(("sync", current[0]))
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: observed.append(("empty", current[0]))
    )
    monkeypatch.setattr(nitro, "_GRAPH_DEVICE", 1)
    nitro._GRAPH_STATE["current"] = (
        SimpleNamespace(reset=lambda: observed.append(("reset", current[0]))),
        {},
    )
    nitro._flush_graphs()
    assert observed == [("sync", 1), ("reset", 1), ("empty", 1)]
    assert current == [0]


def _forward_stub(monkeypatch):
    model = torch.nn.Module()
    model.backbone = SimpleNamespace(
        sphere_embedding=SimpleNamespace(weight=torch.ones(1))
    )
    model.head = None
    data = _data()
    tensors = {
        name: value
        for name, value in vars(data).items()
        if isinstance(value, torch.Tensor)
    }
    tensors["pos"].requires_grad_(True)
    tensors["cell"].requires_grad_(True)
    monkeypatch.setattr(nitro, "_move_inputs", lambda *args: tensors)
    monkeypatch.setattr(nitro, "_geometry_key", lambda *args: (0, b"H3"))
    monkeypatch.setattr(nitro, "_measured_caps", lambda *args: {"edge_capacity": 64})
    monkeypatch.setattr(nitro, "_apply_regime", lambda *args: None)
    monkeypatch.setattr(nitro, "FINITE_GUARD", False)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None)
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(wait_event=lambda event: None),
    )
    return model, data, tensors


def test_capture_rebuilds_if_an_input_buffer_address_changes(monkeypatch):
    model, data, tensors = _forward_stub(monkeypatch)
    monkeypatch.setattr(nitro, "_USE_GRAPH", True)
    keys = []

    def capture(key, build):
        keys.append(key)
        return SimpleNamespace(replay=lambda: None), {
            "energy": torch.ones(1),
            "forces": torch.ones(3, 3),
            "stress": torch.ones(1, 9),
            "_overflow": torch.zeros(1),
        }

    monkeypatch.setattr(nitro, "_get_or_capture_graph", capture)
    nitro._forward_once(model, data, None)
    tensors["cell"] = tensors["cell"].clone()
    nitro._forward_once(model, data, None)
    assert keys[0] != keys[1]


def test_next_stream_waits_for_previous_output_clones_before_buffer_reuse(monkeypatch):
    model, data, tensors = _forward_stub(monkeypatch)
    monkeypatch.setattr(nitro, "_USE_GRAPH", True)
    current = [0]
    order = []
    outputs = {
        "energy": torch.ones(1),
        "forces": torch.ones(3, 3),
        "stress": torch.ones(1, 9),
        "_overflow": torch.zeros(1),
    }
    monkeypatch.setattr(
        nitro,
        "_get_or_capture_graph",
        lambda *args: (
            SimpleNamespace(replay=lambda: order.append(("replay", current[0]))),
            outputs,
        ),
    )
    monkeypatch.setattr(
        nitro,
        "_move_inputs",
        lambda *args: (order.append(("copy_inputs", current[0])), tensors)[1],
    )
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda: SimpleNamespace(
            record=lambda: order.append(("copies_ready", current[0]))
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(
            wait_event=lambda event: order.append(("wait_copies", current[0]))
        ),
    )
    clone = torch.Tensor.clone

    def clone_output(tensor, *args, **kwargs):
        order.append(("clone", current[0]))
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone_output)
    first = nitro._forward_once(model, data, None)
    current[0] = 1
    nitro._forward_once(model, data, None)
    assert order == [
        ("copy_inputs", 0),
        ("replay", 0),
        ("clone", 0),
        ("clone", 0),
        ("clone", 0),
        ("copies_ready", 0),
        ("wait_copies", 1),
        ("copy_inputs", 1),
        ("replay", 1),
        ("clone", 1),
        ("clone", 1),
        ("clone", 1),
        ("copies_ready", 1),
    ]
    assert first["forces"].data_ptr() != outputs["forces"].data_ptr()


def test_eager_outputs_do_not_retain_autograd_or_internal_storage(monkeypatch):
    model, data, tensors = _forward_stub(monkeypatch)
    monkeypatch.setattr(nitro, "_USE_GRAPH", False)
    monkeypatch.setattr(
        nitro,
        "_build_graph",
        lambda *args, **kwargs: (None, None, None, None, torch.zeros(1)),
    )
    energy = tensors["pos"].sum().reshape(1)
    raw = {
        "energy": energy,
        "forces": tensors["pos"] * 2,
        "stress": tensors["cell"].flatten().reshape(1, 9) * 3,
    }
    monkeypatch.setattr(nitro, "_head_derivatives", lambda *args: raw.copy())
    output = nitro._forward_once(model, data, lambda *args: energy)
    for name, value in output.items():
        assert value.grad_fn is None
        assert not value.requires_grad
        assert value.data_ptr() != raw[name].data_ptr()


def test_md_density_changes_reuse_measured_capacity_until_overflow(monkeypatch):
    backbone = SimpleNamespace(cutoff=6.0)
    tensors = vars(_data())
    probes = []

    def build(*args):
        probes.append(args[-1])
        return None, None, None, None, torch.zeros(1), torch.tensor([0, 30])

    monkeypatch.setattr(nitro, "_graph_padded", build)
    first = nitro._measured_caps(backbone, tensors, "omol", (1, b"H3"))
    for density in range(2, 500):
        assert (
            nitro._measured_caps(backbone, tensors, "omol", (density, b"H3")) == first
        )
    assert len(probes) == 1
    assert len(nitro._LIVE_EDGES) == 1
    # An overflow invalidates both the density record and the fitting allocation.
    model = SimpleNamespace(backbone=backbone)
    calls = []

    def forward(*args):
        calls.append(True)
        if len(calls) == 1:
            raise nitro._GraphCapacityOverflow((3, "omol", (500, b"H3")), 3)
        return nitro._measured_caps(backbone, tensors, "omol", (500, b"H3"))

    monkeypatch.setattr(nitro, "_forward_once", forward)
    assert nitro._nitro_forward(model, _data()) == first
    assert len(probes) == 2


def test_capacity_records_have_a_fixed_upper_bound():
    for index in range(300):
        nitro._remember_capacity(nitro._LIVE_EDGES, index, (index, 24, 1))
    assert len(nitro._LIVE_EDGES) == nitro._CAPACITY_CACHE_LIMIT


@pytest.mark.gpu()
@pytest.mark.parametrize(
    ("initial", "value", "expected"),
    [
        (0, 1.0, 0),
        (0, float("nan"), 2),
        (0, float("inf"), 2),
        (1, float("nan"), 1),
        (1, float("inf"), 1),
        (1, 1.0, 1),
    ],
)
def test_finite_guard_preserves_retriable_capacity_overflow(initial, value, expected):
    forces = torch.full((3, 3), value, device="cuda")
    energy = torch.ones(1, device="cuda")
    stress = torch.ones(1, 9, device="cuda")
    flag = torch.tensor([initial], dtype=torch.int64, device="cuda")
    torch.ops.uma_nitro.finite_guard(forces, energy, stress, flag)
    assert flag.item() == expected


@pytest.mark.parametrize("first_flag", [1, 2])
def test_runtime_retries_incomplete_graphs_but_rejects_bad_complete_results(
    monkeypatch, first_flag
):
    calls = []

    def forward(*args):
        calls.append(True)
        flag = first_flag if len(calls) == 1 else 0
        nitro._check_graph_flag(torch.tensor([flag]), (3, "omol", (0, b"H3")), 3)
        return "complete"

    monkeypatch.setattr(nitro, "_forward_once", forward)
    if first_flag == 1:
        assert nitro._nitro_forward(object(), None) == "complete"
        assert len(calls) == 2
    else:
        with pytest.raises(RuntimeError, match="non-finite model output"):
            nitro._nitro_forward(object(), None)
        assert len(calls) == 1


@pytest.mark.parametrize("kind", ["images", "neighbors", "repeat", "beta"])
def test_geometry_constant_caches_release_old_layouts(kind):
    calls = {
        "images": (
            nitro._IMAGE_SCAFFOLD,
            lambda size: nitro._image_scaffold(size, 1, "cpu", torch.float32),
        ),
        "neighbors": (
            nitro._NEIGHBOR_GRID,
            lambda size: nitro._neighbor_grid(size, "cpu"),
        ),
        "repeat": (
            nitro._REPEAT_ARANGE,
            lambda size: nitro._repeat_arange(size, 3, "cpu"),
        ),
        "beta": (nitro._DBETA_VEC, lambda size: nitro._dbeta_vec(size, "cpu")),
    }
    cache, generate = calls[kind]
    for size in range(1, 16):
        generate(size)
    assert len(cache) <= nitro._TENSOR_CACHE_LIMIT


def test_tensor_cache_respects_byte_budget_and_keeps_current_layout(monkeypatch):
    monkeypatch.setattr(nitro, "_TENSOR_CACHE_BYTES", 64)
    first = nitro._repeat_arange(8, 1, "cpu")
    reference = weakref.ref(first)
    del first
    latest = nitro._repeat_arange(100, 1, "cpu")
    assert list(nitro._REPEAT_ARANGE.values()) == [latest]
    assert reference() is None
    assert nitro._repeat_arange(100, 1, "cpu") is latest


@pytest.mark.parametrize("function", [nitro._d_colsum, nitro._dm_colsums])
def test_weight_bound_cache_tracks_mutation_and_does_not_own_weights(function):
    weights = [torch.ones(2, 2) for _ in range(3)]
    function(*weights)
    weights[0].mul_(3)
    actual = function(*weights)
    expected = [6.0, 2.0, 2.0] if function is nitro._dm_colsums else 6.0
    assert actual == expected
    references = [weakref.ref(value) for value in weights]
    del weights
    gc.collect()
    assert all(reference() is None for reference in references)


def test_coefficient_cache_tracks_mutation(monkeypatch):
    monkeypatch.setattr(nitro, "_assert_d55_pattern", lambda value: None)
    coefficients = torch.ones(25, 256)
    initial = nitro._coefficients(coefficients)
    coefficients.mul_(2)
    actual = nitro._coefficients(coefficients)
    for before, after in zip(initial, actual, strict=True):
        torch.testing.assert_close(after, before * 2)


def test_same_force_precision_does_not_flush_another_models_capture(monkeypatch):
    for name in nitro._FORCE_PATH_FP16_FLAGS:
        monkeypatch.setattr(nitro, name, False)
    monkeypatch.setattr(nitro, "_GRAPH_STATE", {"existing": object()})
    flushes = []
    monkeypatch.setattr(nitro, "_flush_graphs", lambda: flushes.append(True))
    nitro.set_force_precision(True)
    assert flushes == []
    nitro.set_force_precision(False)
    assert flushes == [True]


def test_runtime_diagnostics_do_not_expose_mutable_state(monkeypatch):
    key = ((3, 3), (1,), "omol", (("edge_capacity", 64),), 7)
    monkeypatch.setattr(nitro, "_GRAPH_STATE", {key: (object(), {})})
    result = nitro.runtime_cache_info()
    result["active_capacities"]["edge_capacity"] = 999
    result["captures"] = 999
    assert nitro.runtime_cache_info()["active_capacities"]["edge_capacity"] == 64
    assert nitro.runtime_cache_info()["captures"] == 0


def test_model_finalization_releases_only_its_owned_graph(monkeypatch):
    released = []

    def flush():
        released.append(True)
        nitro._GRAPH_STATE.clear()

    monkeypatch.setattr(nitro, "_flush_graphs", flush)
    first, second = torch.nn.Module(), torch.nn.Module()
    first_token = nitro._model_token(first)
    second_token = nitro._model_token(second)
    key = ((3, 3), (1,), "omol", (), second_token)
    nitro._GRAPH_STATE[key] = (object(), {})
    del first
    gc.collect()
    assert released == []
    assert first_token != second_token
    del second
    gc.collect()
    assert released == [True]


def test_explicit_release_detaches_the_finalizer(monkeypatch):
    model = torch.nn.Module()
    nitro._model_token(model)
    finalizer = model._uma_nitro_runtime_finalizer
    nitro._release_runtime(model)
    assert not finalizer.alive


@pytest.mark.parametrize("scope", [torch.no_grad, torch.inference_mode])
def test_runtime_temporarily_enables_conservative_derivatives(monkeypatch, scope):
    def forward(*args):
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()
        position = torch.ones(3, requires_grad=True)
        return torch.autograd.grad(position.square().sum(), position)[0]

    monkeypatch.setattr(nitro, "_forward_once", forward)
    with scope():
        expected_mode = torch.is_inference_mode_enabled()
        torch.testing.assert_close(
            nitro._nitro_forward(object(), None), torch.full((3,), 2.0)
        )
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled() == expected_mode


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_geometry_key_matches_effective_precision_without_mutating_input(
    periodic, dtype
):
    data = _data()
    # This density lies across a capacity bucket boundary after FP32 rounding.
    side = 1.4161139200038488
    data.pos = torch.tensor(
        [[0, 0, 0], [side, side, side], [0.5, 0.5, 0.5]], dtype=torch.float64
    )
    data.cell = torch.eye(3, dtype=torch.float64).mul(side).unsqueeze(0)
    data.pbc.fill_(periodic)
    before = {
        name: value.clone()
        for name, value in vars(data).items()
        if torch.is_tensor(value)
    }
    effective = SimpleNamespace(**vars(data))
    effective.pos, effective.cell = data.pos.to(dtype), data.cell.to(dtype)
    actual = nitro.UMANitroModel._cpu_geometry_key(data, dtype)
    assert actual == nitro._geometry_key(effective)
    assert isinstance(actual[0], int)
    assert isinstance(actual[1], bytes)
    if dtype == torch.float32:
        assert actual != nitro._geometry_key(data)
    for name, value in before.items():
        torch.testing.assert_close(getattr(data, name), value, rtol=0, atol=0)


@pytest.mark.parametrize("field", ["pos", "cell", "pbc", "atomic_numbers"])
def test_device_origin_geometry_uses_existing_engine_path(monkeypatch, field):
    data = _data()
    setattr(data, field, getattr(data, field).to("meta"))
    monkeypatch.setattr(
        nitro, "_geometry_key", lambda *args, **kwargs: pytest.fail("host key read")
    )
    assert nitro.UMANitroModel._cpu_geometry_key(data, torch.float32) is None


@pytest.mark.parametrize("supplied", [False, True])
def test_forward_uses_request_key_without_device_to_host_reconstruction(
    monkeypatch, supplied
):
    model, data, _ = _forward_stub(monkeypatch)
    key = (11, b"owned-composition")
    reconstructed = []
    observed = []

    def geometry_key(value):
        assert not supplied, "The supplied CPU key must avoid device readbacks"
        assert value is data
        reconstructed.append(True)
        return key

    def caps(backbone, tensors, dataset, request_key):
        observed.append(request_key)
        return {"edge_capacity": 64}

    monkeypatch.setattr(nitro, "_geometry_key", geometry_key)
    monkeypatch.setattr(nitro, "_measured_caps", caps)
    monkeypatch.setattr(nitro, "_USE_GRAPH", True)
    monkeypatch.setattr(
        nitro,
        "_get_or_capture_graph",
        lambda *args: (
            SimpleNamespace(replay=lambda: None),
            {
                "energy": torch.ones(1),
                "forces": torch.ones(3, 3),
                "stress": torch.ones(1, 9),
                "_overflow": torch.zeros(1),
            },
        ),
    )
    nitro._forward_once(model, data, None, key if supplied else None)
    assert observed == [key]
    assert len(reconstructed) == (0 if supplied else 1)


def test_capacity_retry_reuses_the_same_immutable_request_key(monkeypatch):
    key = (11, b"owned-composition")
    seen = []

    def forward(model, data, core, geometry_key):
        seen.append(geometry_key)
        if len(seen) == 1:
            raise nitro._GraphCapacityOverflow((3, "omol", geometry_key), 3)
        return "complete"

    monkeypatch.setattr(nitro, "_forward_once", forward)
    assert nitro._nitro_forward(object(), None, geometry_key=key) == "complete"
    assert len(seen) == 2
    assert all(value is key for value in seen)


def test_public_nitro_forward_passes_request_key_without_storing_it(monkeypatch):
    class Data(dict):
        __getattr__ = dict.__getitem__

    data = Data(vars(_data()))
    task = SimpleNamespace(name="omol_energy", property="energy")
    model = SimpleNamespace(_dataset_to_tasks={"omol": [task]})
    key = (11, b"owned-composition")

    def forward(received_model, received_data, *, geometry_key):
        assert received_model is model
        assert received_data is data
        assert geometry_key is key
        return {"energy": torch.ones(1)}

    monkeypatch.setattr(nitro, "_nitro_forward", forward)
    result = nitro.UMANitroModel.forward(model, data, geometry_key=key)
    assert set(result) == {"omol_energy"}
    assert set(vars(model)) == {"_dataset_to_tasks"}
    assert not any("geometry_key" in name for name in data)

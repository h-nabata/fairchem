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
from threading import Barrier
from types import MethodType

import pytest
import torch

from fairchem.core.models.uma.nn import approximate_saved_features
from fairchem.core.models.uma.nn.activation import GateActivation
from fairchem.core.models.uma.nn.approximate_saved_features import (
    _CompressedFeatureForward,
    _pack_feature,
    _PackedFeature,
    _UncompressedFeature,
    _unpack_feature,
    apply_saved_feature_compression,
)
from fairchem.core.models.uma.nn.execution_backends import get_execution_backend
from tests.core.models.uma.test_series_a_integration import _data, _evaluate, _model


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_features_save_storage_restore_dtype_and_keep_finite_range(dtype):
    source = torch.linspace(-2e20, 2e20, 2048, requires_grad=True)
    packed = _pack_feature(source, dtype, 1024)
    assert isinstance(packed, _PackedFeature)
    assert packed.compressed.dtype == dtype
    assert not packed.compressed.requires_grad
    assert (
        packed.compressed.untyped_storage().nbytes() + packed.scale.element_size()
        < (source.untyped_storage().nbytes())
    )
    restored = _unpack_feature(packed)
    assert restored.dtype == source.dtype
    assert restored.device == source.device
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, source, rtol=0.004, atol=0)
    # Each unpack owns its output; consumers cannot corrupt later backward use.
    restored.zero_()
    assert torch.count_nonzero(_unpack_feature(packed)) > 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_compression_preserves_finite_extremes_and_preexisting_nonfinite_values(dtype):
    limit = torch.finfo(torch.float32).max
    source = torch.tensor([limit, -limit], requires_grad=True)
    restored = _unpack_feature(_pack_feature(source, dtype, 1))
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, source, rtol=0.004, atol=0)
    invalid = torch.tensor(
        [float("inf"), float("nan"), -float("inf")], requires_grad=True
    )
    restored = _unpack_feature(_pack_feature(invalid, dtype, 1))
    assert torch.isposinf(restored[0])
    assert torch.isnan(restored[1])
    assert torch.isneginf(restored[2])


def test_pack_excludes_weights_buffers_fp64_and_small_values():
    candidates = [
        torch.randn(2048),
        torch.randn(2048, dtype=torch.float64, requires_grad=True),
        torch.nn.Parameter(torch.randn(2048)),
        torch.randn(16, requires_grad=True),
        torch.arange(2048),
    ]
    weight_view = torch.nn.Parameter(torch.randn(2048), requires_grad=False)[::2]
    candidates.append(weight_view)
    for source in candidates:
        packed = _pack_feature(source, torch.bfloat16, 1024)
        assert isinstance(packed, _UncompressedFeature)
        assert packed.tensor.data_ptr() == source.data_ptr()
        assert packed.tensor.dtype == source.dtype


@pytest.mark.parametrize("size", [16, 2048])
def test_saved_feature_mutation_does_not_silently_change_backward(size):
    activation = torch.nn.SiLU().eval()
    apply_saved_feature_compression(activation)
    source = torch.randn(size, requires_grad=True)
    output = activation(source)
    with torch.no_grad():
        source.add_(1)
    with pytest.raises(RuntimeError, match="modified in place"):
        output.sum().backward()


def test_pack_does_not_retain_full_precision_input_or_autograd_tape():
    source = torch.randn(2048, requires_grad=True).sin()
    source_ref = weakref.ref(source)
    packed = _pack_feature(source, torch.bfloat16, 1024)
    del source
    gc.collect()
    assert source_ref() is None
    assert packed.compressed.grad_fn is None
    assert packed.scale.grad_fn is None


def _derivatives(module, values):
    values = values.detach().clone().requires_grad_(True)
    output = module(values)
    gradient = torch.autograd.grad(output.sum(), values, create_graph=True)[0]
    hessian_diagonal = torch.autograd.grad(gradient.sum(), values)[0]
    return output, gradient, hessian_diagonal


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_compressed_silu_keeps_forward_and_defined_double_backward(dtype):
    reference = torch.nn.SiLU().eval()
    candidate = copy.deepcopy(reference)
    manifest = apply_saved_feature_compression(candidate, dtype=dtype, min_elements=1)
    assert manifest["module_count"] == 1
    values = torch.linspace(-4, 4, 2048)
    expected = _derivatives(reference, values)
    actual = _derivatives(candidate, values)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    for approximate, exact in zip(actual[1:], expected[1:]):
        assert approximate.dtype == torch.float32
        assert torch.isfinite(approximate).all()
        torch.testing.assert_close(approximate, exact, atol=0.004, rtol=0.02)


def test_gate_is_wrapped_once_without_changing_norm_or_state_dict():
    model = (
        torch.nn.Sequential(
            torch.nn.Linear(8, 8),
            torch.nn.LayerNorm(8),
            torch.nn.SiLU(),
            GateActivation(2, 2, 4),
        )
        .eval()
        .requires_grad_(False)
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    manifest = apply_saved_feature_compression(model, min_elements=1)
    assert manifest["modules"] == ["2", "3"]
    assert manifest["module_count"] == 2
    assert not isinstance(model[1].forward, _CompressedFeatureForward)
    assert not isinstance(model[3].gate_act.forward, _CompressedFeatureForward)
    assert apply_saved_feature_compression(model, min_elements=1) == manifest
    assert set(model.state_dict()) == set(before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    "restore", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))]
)
def test_copied_and_pickled_activation_owns_its_forward(restore):
    original = torch.nn.SiLU().eval()
    apply_saved_feature_compression(original, min_elements=1)
    restored = restore(original)
    assert restored.forward.function.__self__ is restored
    assert restored.forward is not original.forward
    values = torch.linspace(-2, 2, 32)
    for actual, expected in zip(
        _derivatives(restored, values), _derivatives(original, values)
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _fail_after_silu(self, value):
    torch.nn.functional.silu(value)
    raise RuntimeError("activation failure")


def test_hook_context_restores_outer_hooks_after_an_exception():
    activation = torch.nn.SiLU().eval()
    activation.forward = MethodType(_fail_after_silu, activation)
    apply_saved_feature_compression(activation, min_elements=1)
    outer = []

    def pack(value):
        outer.append(value.dtype)
        return value.detach()

    values = torch.randn(32, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        with pytest.raises(RuntimeError, match="activation failure"):
            activation(values)
        assert not outer
        values.sin()
    assert outer == [torch.float32]


def test_independent_predictor_threads_keep_separate_hook_precision():
    barrier = Barrier(2)
    values = torch.linspace(-3, 3, 512)

    def work(dtype):
        module = torch.nn.SiLU().eval()
        apply_saved_feature_compression(module, dtype, 1)
        barrier.wait(timeout=10)
        return _derivatives(module, values)

    with ThreadPoolExecutor(2) as executor:
        bf16 = executor.submit(work, "bf16")
        fp16 = executor.submit(work, "fp16")
        results = [bf16.result(timeout=15), fp16.result(timeout=15)]
    for dtype, actual in zip(("bf16", "fp16"), results):
        module = torch.nn.SiLU().eval()
        apply_saved_feature_compression(module, dtype, 1)
        for value, expected in zip(actual, _derivatives(module, values)):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "fp32"},
        {"min_elements": True},
        {"min_elements": 0},
        {"min_elements": 1.5},
    ],
)
def test_invalid_compression_options_fail_without_modifying_forward(kwargs):
    activation = torch.nn.SiLU().eval()
    with pytest.raises(ValueError):
        apply_saved_feature_compression(activation, **kwargs)
    assert isinstance(activation.forward, MethodType)


def test_unsupported_models_and_conflicting_settings_fail_explicitly():
    with pytest.raises(ValueError, match="frozen"):
        apply_saved_feature_compression(
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.SiLU()).eval()
        )
    with pytest.raises(ValueError, match="eval"):
        apply_saved_feature_compression(torch.nn.SiLU())
    with pytest.raises(ValueError, match="No supported"):
        apply_saved_feature_compression(torch.nn.Identity().eval())
    activation = torch.nn.SiLU().eval()
    activation.backend = get_execution_backend("umas_fast_pytorch")
    with pytest.raises(ValueError, match="general backend"):
        apply_saved_feature_compression(activation)
    sequence = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.SiLU(inplace=True)).eval()
    with pytest.raises(ValueError, match="out of place"):
        apply_saved_feature_compression(sequence)
    assert isinstance(sequence[0].forward, MethodType)
    activation = torch.nn.SiLU().eval()
    apply_saved_feature_compression(activation)
    with pytest.raises(ValueError, match="differently"):
        apply_saved_feature_compression(activation, dtype="fp16")


def test_forward_vmap_has_explicit_limitation():
    activation = torch.nn.SiLU().eval()
    apply_saved_feature_compression(activation, min_elements=1)
    with pytest.raises(RuntimeError, match="forward inside"):
        torch.vmap(activation)(torch.randn(4, 8, requires_grad=True))


@pytest.mark.parametrize("vectorized", [False, True])
def test_real_uma_energy_force_stress_and_hessian_with_compressed_features(
    vectorized, monkeypatch
):
    recorded = []
    original_pack = _pack_feature

    def record_pack(tensor, dtype, min_elements):
        packed = original_pack(tensor, dtype, min_elements)
        if isinstance(packed, _PackedFeature):
            recorded.append(
                (
                    tensor.numel() * tensor.element_size(),
                    packed.compressed.untyped_storage().nbytes()
                    + packed.scale.element_size(),
                )
            )
        return packed

    monkeypatch.setattr(approximate_saved_features, "_pack_feature", record_pack)
    reference = tuple(module.float() for module in _model())
    reference[0].regress_config.hessian_vmap = vectorized
    candidate = copy.deepcopy(reference)
    manifest = apply_saved_feature_compression(candidate[0], min_elements=1)
    assert manifest["module_count"] > 0
    data = _data()
    for name, value in data.to_dict().items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[name] = value.float()
    properties = ("energy", "forces", "stress", "hessian")
    torch.manual_seed(19)
    expected = _evaluate(reference, data, properties)
    torch.manual_seed(19)
    actual = _evaluate(candidate, data, properties)
    assert recorded
    assert sum(packed for _, packed in recorded) < sum(
        original for original, _ in recorded
    )
    torch.testing.assert_close(actual["energy"], expected["energy"], rtol=0, atol=0)
    for name in properties[1:]:
        assert torch.isfinite(actual[name]).all()
        assert torch.count_nonzero(actual[name]) > 0
        torch.testing.assert_close(actual[name], expected[name], rtol=0.08, atol=0.003)

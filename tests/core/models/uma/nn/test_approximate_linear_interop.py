"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import pickle
from types import MethodType

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fairchem.core.models.uma.nn.approximate_linear import (
    apply_linear_approximations,
)


def _model(**options):
    model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 3))
    model.eval().requires_grad_(False)
    apply_linear_approximations(model, **options)
    return model


def _assert_state_equal(model, expected):
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
        else:
            assert actual[key] == value


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize("direct", [False, True])
def test_bad_packed_metadata_does_not_partially_overwrite_live_state(assign, direct):
    model = _model(weight_bits=4)
    model.register_buffer("unrelated", torch.ones(1))
    if direct:
        model = model[1]
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    for key, value in checkpoint.items():
        if isinstance(value, torch.Tensor) and not key.endswith("scale"):
            value.zero_()
    prefix = "" if direct else "1."
    checkpoint[prefix + "factors.0._extra_state"]["columns"] += 1
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_state_dict(checkpoint, assign=assign)
    _assert_state_equal(model, previous)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize(
    "bad_kind",
    [
        "packed_float",
        "packed_shape",
        "packed_meta",
        "packed_sparse",
        "scale_double",
        "scale_nan",
        "scale_inf",
        "scale_zero",
        "scale_negative",
        "scale_grad",
        "scale_parameter",
        "bias_double",
        "metadata_type",
    ],
)
def test_malformed_packed_checkpoint_is_rejected_before_any_copy(bad_kind, assign):
    model = _model(weight_bits=4)
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    checkpoint["0.bias"].add_(7)
    packed_key, scale_key = "1.factors.0.packed", "1.factors.0.scale"
    if bad_kind == "packed_float":
        checkpoint[packed_key] = checkpoint[packed_key].float() + 0.75
    elif bad_kind == "packed_shape":
        checkpoint[packed_key] = checkpoint[packed_key][:-1]
    elif bad_kind == "packed_meta":
        checkpoint[packed_key] = checkpoint[packed_key].to("meta")
    elif bad_kind == "packed_sparse":
        checkpoint[packed_key] = checkpoint[packed_key].to_sparse()
    elif bad_kind == "scale_double":
        checkpoint[scale_key] = checkpoint[scale_key].double()
    elif bad_kind == "scale_grad":
        checkpoint[scale_key].requires_grad_(True)
    elif bad_kind == "scale_parameter":
        checkpoint[scale_key] = nn.Parameter(checkpoint[scale_key], requires_grad=False)
    elif bad_kind == "bias_double":
        checkpoint["1.bias"] = checkpoint["1.bias"].double()
    elif bad_kind == "metadata_type":
        checkpoint["1._extra_state"]["in_features"] = torch.tensor([5, 5])
    else:
        value = {
            "scale_nan": float("nan"),
            "scale_inf": float("inf"),
            "scale_zero": 0.0,
            "scale_negative": -1.0,
        }[bad_kind]
        checkpoint[scale_key][0] = value
    with pytest.raises(RuntimeError, match="ApproximateLinear"):
        model.load_state_dict(checkpoint, assign=assign)
    _assert_state_equal(model, previous)


@pytest.mark.parametrize("missing", ["_extra_state", "factors.0._extra_state", "bias"])
def test_non_strict_load_still_requires_complete_supplied_converted_layer(missing):
    model = _model(weight_bits=8)
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    checkpoint["0.bias"].add_(7)
    del checkpoint["1." + missing]
    with pytest.raises(RuntimeError, match="ApproximateLinear"):
        model.load_state_dict(checkpoint, strict=False)
    _assert_state_equal(model, previous)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("extra", ["factors.2.packed", "factors.0.values", "bias"])
def test_unexpected_converted_state_keys_cannot_cause_partial_loading(strict, extra):
    model = nn.Sequential(nn.Linear(7, 5), nn.Linear(5, 3, bias=False))
    model.eval().requires_grad_(False)
    apply_linear_approximations(model, weight_bits=4)
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    checkpoint["0.bias"].add_(7)
    checkpoint["1." + extra] = torch.zeros(3)
    with pytest.raises(RuntimeError, match="unexpected state keys"):
        model.load_state_dict(checkpoint, strict=strict)
    _assert_state_equal(model, previous)


def test_partial_load_of_unrelated_component_can_omit_converted_layers():
    model = _model(weight_bits=8)
    previous = copy.deepcopy(model.state_dict())
    model.register_buffer("unrelated", torch.zeros(1))
    result = model.load_state_dict({"unrelated": torch.ones(1)}, strict=False)
    assert "0._extra_state" in result.missing_keys
    assert model.unrelated.item() == 1
    del model.unrelated
    _assert_state_equal(model, previous)


def test_shared_aliases_are_all_preflighted_before_loading_any_alias():
    shared = nn.Linear(7, 5).eval().requires_grad_(False)
    model = nn.ModuleDict({"first": shared, "last": shared})
    apply_linear_approximations(model, weight_bits=4)
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    checkpoint["first.bias"].add_(7)
    checkpoint["last.factors.0._extra_state"]["columns"] += 1
    with pytest.raises(RuntimeError, match="metadata"):
        model.load_state_dict(checkpoint)
    assert model["first"] is model["last"]
    _assert_state_equal(model, previous)


@pytest.mark.parametrize(
    "roundtrip", [copy.deepcopy, lambda model: pickle.loads(pickle.dumps(model))]
)
def test_copied_model_preflight_is_independent_of_original(roundtrip):
    model = _model(weight_bits=4)
    restored = roundtrip(model)
    checkpoint = copy.deepcopy(restored.state_dict())
    previous = copy.deepcopy(checkpoint)
    checkpoint["0.bias"].add_(7)
    checkpoint["1.factors.0._extra_state"]["columns"] += 1
    with pytest.raises(RuntimeError, match="metadata"):
        restored.load_state_dict(checkpoint)
    _assert_state_equal(model, previous)
    _assert_state_equal(restored, previous)


@pytest.mark.parametrize("assign", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"weight_bits": 4},
        {"weight_bits": 8},
        {"gemm_precision": "bf16", "low_precision_emulation": True},
        {"low_rank_fraction": 0.2},
    ],
)
def test_existing_version_one_states_load_and_recover_after_error(options, assign):
    source, target = _model(**options), _model(**options)
    checkpoint = copy.deepcopy(source.state_dict())
    bad = copy.deepcopy(checkpoint)
    bad["1._extra_state"]["in_features"] += 1
    with pytest.raises(RuntimeError, match="metadata"):
        target.load_state_dict(bad, assign=assign)
    target.load_state_dict(checkpoint, assign=assign)
    assert not list(target.parameters())
    _assert_state_equal(target, checkpoint)
    x = torch.randn(3, 7)
    torch.testing.assert_close(target(x), source(x), rtol=0, atol=0)


def test_frozen_keep_vars_checkpoint_buffers_remain_supported():
    source, target = _model(weight_bits=8), _model(weight_bits=8)
    checkpoint = source.state_dict(keep_vars=True)
    target.load_state_dict(checkpoint, assign=True)
    _assert_state_equal(target, checkpoint)
    assert not list(target.parameters())


def test_direct_factor_load_preflights_before_overwriting_packed_buffer():
    model = _model(weight_bits=4)[0].factors[0]
    previous = copy.deepcopy(model.state_dict())
    checkpoint = copy.deepcopy(previous)
    checkpoint["packed"].zero_()
    checkpoint["scale"] = checkpoint["scale"].double()
    with pytest.raises(RuntimeError, match="scale"):
        model.load_state_dict(checkpoint)
    _assert_state_equal(model, previous)


def test_instance_forward_wrapper_is_preserved_with_original_behavior():
    model = _model()
    original = model[0]

    def wrapped_forward(self, x):
        return F.linear(x, self.weight, self.bias) + 7

    original.forward = MethodType(wrapped_forward, original)
    x = torch.randn(2, 7)
    expected = original(x)
    manifest = apply_linear_approximations(model, weight_bits=4)
    assert model[0] is original
    torch.testing.assert_close(model[0](x), expected, rtol=0, atol=0)
    assert manifest["skipped"][0]["names"] == ["0"]
    assert "custom forward" in manifest["skipped"][0]["reason"]


def test_in_place_compiled_linear_is_preserved():
    model = _model()
    original = model[0]
    original.compile(backend="eager")
    manifest = apply_linear_approximations(model, weight_bits=8)
    assert model[0] is original
    assert "custom forward" in manifest["skipped"][0]["reason"]


@pytest.mark.parametrize(
    "register",
    [
        "register_state_dict_pre_hook",
        "register_state_dict_post_hook",
        "register_load_state_dict_pre_hook",
        "register_load_state_dict_post_hook",
    ],
)
def test_external_serialization_hooks_are_preserved(register):
    model = _model()
    original = model[0]

    def hook(*args):
        return None

    getattr(original, register)(hook)
    manifest = apply_linear_approximations(model, weight_bits=8)
    assert model[0] is original
    assert manifest["skipped"][0]["names"] == ["0"]
    assert "module hooks" in manifest["skipped"][0]["reason"]

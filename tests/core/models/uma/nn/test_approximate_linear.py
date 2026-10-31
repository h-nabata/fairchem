"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Numerical and integration contracts of explicitly enabled B-series Linear.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fairchem.core.models.uma.nn.approximate_linear import (
    ApproximateLinear,
    _mm_fp32_output,
    apply_linear_approximations,
)


def _model(in_features=7, out_features=5, bias=True):
    generator = torch.Generator().manual_seed(1902)
    model = nn.Sequential(nn.Linear(in_features, out_features, bias=bias))
    with torch.no_grad():
        model[0].weight.copy_(
            torch.randn(out_features, in_features, generator=generator) / 3
        )
        if bias:
            model[0].bias.copy_(torch.randn(out_features, generator=generator) / 3)
    return model.eval().requires_grad_(False)


def _rounded_components(value, mode):
    dtype = torch.float16 if mode == "fp16" else torch.bfloat16
    count = {"bf16_split2": 2, "bf16_split3": 3}.get(mode, 1)
    parts = []
    for _ in range(count):
        part = value.to(dtype).float()
        parts.append(part)
        value = value - part
    return parts


def test_disabled_preserves_module_and_predictions():
    model = _model()
    original = model[0]
    x = torch.randn(3, 7)
    expected = model(x)
    manifest = apply_linear_approximations(model)
    assert model[0] is original
    assert not manifest["active"]
    assert manifest["converted"] == []
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
    json.dumps(manifest, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gemm_precision": "tf32"},
        {"gemm_precision": []},
        {"weight_bits": True},
        {"weight_bits": 3},
        {"weight_bits": 8.0},
        {"low_rank_fraction": True},
        {"low_rank_fraction": 0},
        {"low_rank_fraction": -0.1},
        {"low_rank_fraction": 1.1},
        {"low_rank_fraction": float("nan")},
        {"low_rank_fraction": float("inf")},
        {"low_rank_fraction": 0.25, "weight_bits": 8},
        {"low_precision_emulation": 1},
        {"exclude_patterns": "0"},
        {"exclude_patterns": [""]},
    ],
)
def test_invalid_options_fail_without_mutation(kwargs):
    model = _model()
    original = model[0]
    with pytest.raises(ValueError):
        apply_linear_approximations(model, **kwargs)
    assert model[0] is original


@pytest.mark.parametrize("mode", ["bf16", "fp16", "bf16_split2", "bf16_split3"])
@pytest.mark.parametrize("shape", [(7,), (3, 7), (2, 3, 7), (0, 7)])
def test_reduced_forward_and_fp32_output(mode, shape):
    model = _model()
    weight = model[0].weight.detach().clone()
    bias = model[0].bias.detach().clone()
    manifest = apply_linear_approximations(
        model, gemm_precision=mode, low_precision_emulation=True
    )
    x = torch.randn(shape)
    xs = _rounded_components(x.reshape(-1, 7), mode)
    ws = _rounded_components(weight, mode)
    expected = sum(a @ b.T for a in xs for b in ws) + bias
    actual = model(x)
    assert actual.dtype == torch.float32
    assert actual.shape == (*shape[:-1], 5)
    torch.testing.assert_close(
        actual, expected.reshape(*shape[:-1], 5), atol=4e-7, rtol=4e-6
    )
    assert manifest["converted"][0]["names"] == ["0"]
    assert "surrogate" in " ".join(manifest["warnings"])
    json.dumps(manifest, allow_nan=False)


@pytest.mark.parametrize("mode", ["bf16", "fp16", "bf16_split2", "bf16_split3"])
def test_reduced_backward_and_vmapped_hessian(mode):
    model = _model()
    weight = model[0].weight.detach().clone()
    apply_linear_approximations(
        model, gemm_precision=mode, low_precision_emulation=True
    )
    # A slice exercises folded, non-contiguous input and coordinate gradients.
    original_x = torch.randn(2, 14, requires_grad=True)
    x = original_x[:, ::2]
    y = model(x)
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    effective_weight = sum(_rounded_components(weight, mode))
    torch.testing.assert_close(gradient, 2 * y @ effective_weight, atol=4e-7, rtol=4e-6)
    vectors = torch.randn(4, *x.shape)
    actual_hvp = torch.vmap(
        lambda v: torch.autograd.grad(gradient, x, v, retain_graph=True)[0]
    )(vectors)
    expected_hvp = 2 * (vectors @ effective_weight.T) @ effective_weight
    torch.testing.assert_close(actual_hvp, expected_hvp, atol=2e-6, rtol=5e-6)
    assert list(model.parameters()) == []


def test_cpu_reduced_requires_explicit_emulation():
    model = _model()
    apply_linear_approximations(model, gemm_precision="bf16")
    with pytest.raises(RuntimeError, match="low_precision_emulation=True"):
        model(torch.randn(3, 7))


def test_native_dispatch_requests_fp32_and_rejects_backend_dtype(monkeypatch):
    class Operand:
        device = torch.device("xpu")

    left, right = Operand(), Operand()
    calls = []

    def mm(a, b, *, out_dtype):
        calls.append((a, b, out_dtype))
        return torch.ones(2, 2)

    monkeypatch.setattr(torch, "mm", mm)
    result = _mm_fp32_output(left, right, False)
    assert result.dtype == torch.float32
    assert calls == [(left, right, torch.float32)]
    monkeypatch.setattr(
        torch, "mm", lambda *args, **kwargs: torch.ones(2, 2, dtype=torch.bfloat16)
    )
    with pytest.raises(RuntimeError, match="honor FP32"):
        _mm_fp32_output(left, right, False)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_reduction_flags_are_checked_without_global_mutation(monkeypatch, dtype):
    left = SimpleNamespace(device=torch.device("cuda"), dtype=dtype)
    right = SimpleNamespace(device=torch.device("cuda"), dtype=dtype)
    flag_name = (
        "allow_fp16_reduced_precision_reduction"
        if dtype == torch.float16
        else "allow_bf16_reduced_precision_reduction"
    )
    flags = SimpleNamespace(**{flag_name: True})
    monkeypatch.setattr(torch.backends.cuda, "matmul", flags)
    calls = []

    def mm(*args, **kwargs):
        calls.append(kwargs)
        return torch.ones(2, 2)

    monkeypatch.setattr(torch, "mm", mm)
    with pytest.raises(RuntimeError, match=flag_name):
        _mm_fp32_output(left, right, False)
    assert getattr(flags, flag_name) is True
    assert calls == []
    setattr(flags, flag_name, False)
    assert _mm_fp32_output(left, right, False).dtype == torch.float32
    assert calls == [{"out_dtype": torch.float32}]
    if dtype == torch.float16:
        flags.allow_fp16_accumulation = True
        with pytest.raises(RuntimeError, match="allow_fp16_accumulation"):
            _mm_fp32_output(left, right, False)
        assert flags.allow_fp16_accumulation is True
        assert len(calls) == 1


@pytest.mark.parametrize("bits", [4, 8])
def test_integer_storage_quantization_error_and_input_hessian(bits):
    model = _model(in_features=9, out_features=7, bias=False)
    with torch.no_grad():
        model[0].weight[0].zero_()
    original = model[0].weight.clone()
    manifest = apply_linear_approximations(model, weight_bits=bits)
    module = model[0]
    weight = module.materialized_weight()
    storage = module.factors[0]
    assert storage.values is None  # noqa: PD011  (a torch buffer, not a dataframe)
    assert storage.packed.dtype == (torch.uint8 if bits == 4 else torch.int8)
    assert storage.packed.numel() == (32 if bits == 4 else 63)
    assert manifest["stored_weight_bytes"] == storage.packed.numel() + 7 * 4
    assert manifest["stored_weight_bytes"] < manifest["original_weight_bytes"]
    assert torch.count_nonzero(weight[0]) == 0
    assert bool(((weight - original).abs() <= storage.scale[:, None] * 0.501).all())
    assert torch.equal(weight.sign(), original.sign()) or bool(
        ((weight.sign() == original.sign()) | (weight == 0)).all()
    )
    x = torch.randn(3, 9, requires_grad=True)
    y = module(x)
    torch.testing.assert_close(y, x @ weight.T)
    gradient = torch.autograd.grad(y.square().sum(), x, create_graph=True)[0]
    v = torch.randn_like(x)
    hvp = torch.autograd.grad(gradient, x, v)[0]
    torch.testing.assert_close(hvp, 2 * (v @ weight.T) @ weight)


@pytest.mark.parametrize("bits", [4, 8])
def test_integer_state_roundtrip_deepcopy_and_dtype_guard(bits):
    model = _model()
    apply_linear_approximations(model, weight_bits=bits)
    restored = _model()
    apply_linear_approximations(restored, weight_bits=bits)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    copied = copy.deepcopy(model).to("cpu")
    x = torch.randn(2, 7)
    torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)
    torch.testing.assert_close(copied(x), model(x), atol=0, rtol=0)
    assert (
        copied[0].factors[0].packed.data_ptr() != model[0].factors[0].packed.data_ptr()
    )
    copied.to(torch.bfloat16)
    assert copied[0].factors[0].packed.dtype == (
        torch.uint8 if bits == 4 else torch.int8
    )
    with pytest.raises(TypeError, match="scales.*FP32"):
        copied(x)


def test_low_rank_reconstructs_known_rank_and_retains_coordinate_derivatives():
    model = _model(12, 10)
    with torch.no_grad():
        model[0].weight.copy_(torch.randn(10, 2) @ torch.randn(2, 12))
    reference = copy.deepcopy(model)
    manifest = apply_linear_approximations(model, low_rank_fraction=0.2)
    assert model[0].low_rank == 2
    assert manifest["stored_weight_bytes"] == 2 * (12 + 10) * 4
    assert manifest["converted"][0]["retained_singular_energy"] > 0.999999
    x = torch.randn(3, 12, requires_grad=True)
    torch.testing.assert_close(model(x), reference(x), atol=3e-6, rtol=3e-6)
    v = torch.randn_like(x)
    actual_gradient = torch.autograd.grad(
        model(x).square().sum(), x, create_graph=True
    )[0]
    expected_gradient = torch.autograd.grad(
        reference(x).square().sum(), x, create_graph=True
    )[0]
    actual_hvp = torch.autograd.grad(actual_gradient, x, v)[0]
    expected_hvp = torch.autograd.grad(expected_gradient, x, v)[0]
    torch.testing.assert_close(actual_hvp, expected_hvp, atol=2e-4, rtol=1e-5)


def test_nonbeneficial_rank_skips_but_can_still_enable_reduced_gemm():
    model = _model(2, 2)
    original = model[0]
    manifest = apply_linear_approximations(model, low_rank_fraction=0.9)
    assert model[0] is original
    assert manifest["converted"] == []
    assert "does not reduce" in manifest["skipped"][0]["reason"]
    apply_linear_approximations(
        model,
        low_rank_fraction=0.9,
        gemm_precision="bf16",
        low_precision_emulation=True,
    )
    assert isinstance(model[0], ApproximateLinear)
    assert model[0].low_rank is None


def test_shared_nested_alias_exclusion_and_replacement_preserve_identity():
    shared = _model()
    model = nn.ModuleDict({"a": shared, "b": shared})
    manifest = apply_linear_approximations(
        model, weight_bits=8, exclude_patterns=("b.*",)
    )
    assert not manifest["converted"]
    assert manifest["skipped"][0]["names"] == ["a.0", "b.0"]
    manifest = apply_linear_approximations(model, weight_bits=8)
    assert model["a"][0] is model["b"][0]
    assert manifest["converted"][0]["names"] == ["a.0", "b.0"]
    assert manifest["original_weight_bytes"] == 7 * 5 * 4
    with pytest.raises(ValueError, match="already applied"):
        apply_linear_approximations(model, weight_bits=8)


def test_hooked_and_custom_linears_are_reported_and_preserved():
    class CustomLinear(nn.Linear):
        def forward(self, x):
            return super().forward(x) + 7

    model = nn.Sequential(nn.Linear(3, 3), CustomLinear(3, 3))
    model.eval().requires_grad_(False)
    model[0].register_forward_hook(lambda module, args, output: output + 2)
    original = list(model)
    manifest = apply_linear_approximations(model, weight_bits=4)
    assert list(model) == original
    assert len(manifest["skipped"]) == 2


def test_validation_is_transactional_and_rejects_trainable_or_non_fp32():
    model = nn.Sequential(_model()[0], nn.Linear(5, 4)).eval()
    original = list(model)
    with pytest.raises(ValueError, match="frozen"):
        apply_linear_approximations(model, weight_bits=8)
    assert list(model) == original
    model.requires_grad_(False).double()
    with pytest.raises(ValueError, match="FP32"):
        apply_linear_approximations(model, weight_bits=8)
    assert list(model) == original


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_and_weight_rejected(bad):
    model = _model()
    apply_linear_approximations(model, weight_bits=8)
    with pytest.raises(FloatingPointError, match="input"):
        model(torch.full((2, 7), bad))
    original = _model()
    original[0].weight[0, 0] = bad
    with pytest.raises(FloatingPointError, match="weight"):
        apply_linear_approximations(original, weight_bits=8)


def test_fp16_cast_overflow_and_fp32_output_overflow_are_rejected():
    model = _model()
    apply_linear_approximations(
        model, gemm_precision="fp16", low_precision_emulation=True
    )
    with pytest.raises(FloatingPointError, match="cast overflow"):
        model(torch.full((2, 7), 1e6))
    model = _model(2, 2)
    model[0].weight.fill_(1e30)
    apply_linear_approximations(model, weight_bits=8)
    with pytest.raises(FloatingPointError, match="GEMM output"):
        model(torch.full((2, 2), 1e30))


def test_input_dtype_shape_and_training_mode_are_explicit():
    model = _model()
    apply_linear_approximations(model, weight_bits=4)
    with pytest.raises(TypeError, match="FP32 inputs"):
        model(torch.randn(2, 7, dtype=torch.float64))
    with pytest.raises(ValueError, match="last dimension"):
        model(torch.randn(2, 6))
    model.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        model(torch.randn(2, 7))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gemm_precision": "bf16"},
        {"weight_bits": 4},
        {"weight_bits": 8, "gemm_precision": "fp16"},
    ],
)
def test_autocast_does_not_contaminate_forward_backward_or_jvp(kwargs):
    model = _model()
    apply_linear_approximations(model, low_precision_emulation=True, **kwargs)
    x = torch.randn(2, 7, requires_grad=True)
    reference = model(x)
    reference_gradient = torch.autograd.grad(reference.square().sum(), x)[0]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model(x)
        actual_gradient = torch.autograd.grad(actual.square().sum(), x)[0]
    assert actual.dtype == actual_gradient.dtype == torch.float32
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradient, reference_gradient, rtol=0, atol=0)
    tangent = torch.randn_like(x)
    _, jvp = torch.func.jvp(model, (x,), (tangent,))
    weight = model[0].materialized_weight()
    if kwargs.get("gemm_precision", "fp32") != "fp32":
        weight = sum(_rounded_components(weight, kwargs["gemm_precision"]))
    torch.testing.assert_close(jvp, tangent @ weight.T)


def test_integer_backward_saves_packed_weights_not_fp32_expansions():
    model = _model(65, 33, bias=False)
    apply_linear_approximations(model, weight_bits=4)
    saved = []

    def pack(tensor):
        saved.append(
            (tuple(tensor.shape), tensor.dtype, tensor.numel() * tensor.element_size())
        )
        return tensor.detach()

    x = torch.randn(2, 65, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(x)
    assert ((33, 65), torch.float32, 33 * 65 * 4) not in saved
    assert any(dtype == torch.uint8 for _, dtype, _ in saved)
    assert sum(size for _, _, size in saved) < 33 * 65 * 4
    gradient = torch.autograd.grad(output.sum(), x)[0]
    torch.testing.assert_close(
        gradient, model[0].materialized_weight().sum(0).expand_as(x)
    )


def test_packed_state_dict_rejects_same_storage_shape_different_architecture():
    source = _model(3, 1)
    target = _model(4, 1)
    apply_linear_approximations(source, weight_bits=4)
    apply_linear_approximations(target, weight_bits=4)
    assert source[0].factors[0].packed.shape == target[0].factors[0].packed.shape
    with pytest.raises(RuntimeError, match="metadata"):
        target.load_state_dict(source.state_dict(), strict=True)


def test_synthetic_general_uma_energy_force_stress_and_hessian_all_variants():
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone

    torch.manual_seed(1902)
    backbone = (
        eSCNMDBackbone(
            sphere_channels=4,
            hidden_channels=8,
            edge_channels=4,
            num_distance_basis=4,
            num_layers=1,
            lmax=2,
            mmax=2,
            regress_forces=True,
            regress_stress=True,
            direct_forces=False,
            cutoff=2.6,
            always_use_pbc=False,
            use_dataset_embedding=False,
            otf_graph=True,
            execution_mode="general",
        )
        .eval()
        .requires_grad_(False)
    )
    head = MLP_EFS_Head(backbone, wrap_property=False).eval().requires_grad_(False)
    atoms = Atoms(
        numbers=[1, 8, 1],
        positions=[[0.2, 0.1, 0.3], [1.2, 0.3, 0.4], [1.4, 1.7, 0.8]],
        cell=[4, 4.2, 4.4],
        pbc=True,
    )
    atoms.info.update(charge=0, spin=1)

    def evaluate(model):
        data = AtomicData.from_ase(atoms, task_name="omol")
        data["_requested_properties"] = ("energy", "forces", "stress", "hessian")
        torch.manual_seed(42)
        return head(data, model(data))

    reference = evaluate(backbone)
    variants = [
        ({"gemm_precision": "bf16"}, 0.003),
        ({"gemm_precision": "fp16"}, 0.001),
        ({"gemm_precision": "bf16_split2"}, 1e-5),
        ({"gemm_precision": "bf16_split3"}, 1e-6),
        ({"weight_bits": 8}, 0.003),
        ({"weight_bits": 4}, 0.03),
        ({"low_rank_fraction": 0.25}, 0.15),
    ]
    for kwargs, absolute_limit in variants:
        candidate = copy.deepcopy(backbone)
        manifest = apply_linear_approximations(
            candidate, low_precision_emulation=True, **kwargs
        )
        assert manifest["effective"]
        output = evaluate(candidate)
        for name, baseline in reference.items():
            assert output[name].shape == baseline.shape
            assert bool(torch.isfinite(output[name]).all())
            torch.testing.assert_close(
                output[name],
                baseline,
                rtol=0,
                atol=absolute_limit,
                msg=f"{kwargs}: {name}",
            )

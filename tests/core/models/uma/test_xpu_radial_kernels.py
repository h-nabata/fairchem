"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.uma.nn import xpu_radial_kernels as kernels


def _inputs(rows=3, inner=4, width=5, dtype=torch.float64, device="cpu"):
    return tuple(
        torch.randn(*shape, dtype=dtype, device=device, requires_grad=True)
        for shape in ((rows, inner), (width, inner), (width,), (width,), (width,))
    )


def test_module_import_does_not_import_triton(monkeypatch):
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        assert not name.startswith("triton")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    importlib.reload(kernels)


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_cpu_fallback_does_not_load_compiler(monkeypatch, enabled):
    monkeypatch.setenv("FAIRCHEM_XPU_FUSED_KERNELS", enabled)

    def unexpected_load():
        pytest.fail("CPU fallback must not import the compiler")

    monkeypatch.setattr(kernels, "_load_kernel", unexpected_load)
    assert kernels.xpu_fused_linear_norm_silu(*_inputs(), 1e-5) is None


@pytest.mark.parametrize(
    "rows,inner,width,expected",
    [
        (1, 1, 16, True),
        (17, 31, 33, True),
        (2, 2048, 512, True),
        (0, 4, 16, False),
        (2, 0, 16, False),
        (2, 2049, 16, False),
        (2, 4, 15, False),
        (2, 4, 513, False),
    ],
)
def test_supported_tile_limits(rows, inner, width, expected):
    assert (
        kernels._supported_layout(
            *_inputs(rows, inner, width, dtype=torch.float32), 1e-5
        )
        is expected
    )


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_eps_fallback(eps):
    assert not kernels._supported_layout(*_inputs(width=16, dtype=torch.float32), eps)


def test_layout_fallbacks():
    inputs = list(_inputs(rows=4, inner=4, width=16, dtype=torch.float32))
    assert kernels._supported_layout(*inputs, 1e-5)
    inputs[0] = inputs[0].mT
    assert not kernels._supported_layout(*inputs, 1e-5)
    inputs[0] = inputs[0].contiguous().double()
    assert not kernels._supported_layout(*inputs, 1e-5)
    inputs[0] = inputs[0].float()
    inputs[2] = inputs[2].unsqueeze(0)
    assert not kernels._supported_layout(*inputs, 1e-5)


@pytest.mark.parametrize(
    "intel,inductor,expected",
    [(None, "0", False), (None, "1", True), ("0", "1", False), ("1", "0", True)],
)
def test_fast_math_env_precedence(monkeypatch, intel, inductor, expected):
    monkeypatch.delenv("TRITON_INTEL_FAST_MATH", raising=False)
    if intel is not None:
        monkeypatch.setenv("TRITON_INTEL_FAST_MATH", intel)
    monkeypatch.setenv("TORCHINDUCTOR_USE_FAST_MATH", inductor)
    assert kernels._fast_math_enabled(SimpleNamespace()) is expected


def test_fast_math_live_knob_precedence(monkeypatch):
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "0")
    fake_triton = SimpleNamespace(
        knobs=SimpleNamespace(intel=SimpleNamespace(fast_math=True))
    )
    assert kernels._fast_math_enabled(fake_triton)


def test_missing_intel_compiler_falls_back_before_launch(monkeypatch):
    original_import = builtins.__import__

    def no_triton(name, *args, **kwargs):
        if name.startswith("triton"):
            raise ImportError("unavailable")
        return original_import(name, *args, **kwargs)

    kernels._load_kernel.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", no_triton)
        assert kernels._load_kernel() is None
    kernels._load_kernel.cache_clear()


@pytest.mark.parametrize("frozen", [False, True])
def test_recomputed_adjoint_matches_native(frozen):
    inputs = _inputs()
    output = kernels.linear_norm_silu_reference(*inputs, 1e-5)
    cotangent = torch.randn_like(output)
    expected = torch.autograd.grad(output, inputs, cotangent, create_graph=True)
    needs = (True, not frozen, not frozen, not frozen, not frozen)
    result = kernels.linear_norm_silu_adjoint(*inputs, 1e-5, cotangent, needs)
    for index, (actual, reference) in enumerate(zip(result, expected, strict=True)):
        if needs[index]:
            torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)
        else:
            assert actual is None


def test_recomputed_adjoint_gradcheck():
    inputs = _inputs(rows=2, inner=2, width=3)
    cotangent = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)

    def adjoint(*args):
        return kernels.linear_norm_silu_adjoint(*args[:5], 1e-4, args[5])

    assert torch.autograd.gradcheck(adjoint, (*inputs, cotangent))


def test_custom_function_gradcheck_and_double_backward(monkeypatch):
    # This validates the autograd bridge, not the unavailable hardware kernel.
    monkeypatch.setattr(
        kernels, "_launch_linear_norm_silu", kernels.linear_norm_silu_reference
    )
    inputs = _inputs(rows=2, inner=2, width=3)

    def fused(*args):
        return kernels._FusedLinearNormSiLU.apply(*args, 1e-4)

    assert torch.autograd.gradcheck(fused, inputs)
    assert torch.autograd.gradgradcheck(fused, inputs)


def test_custom_function_hessian_batched_reverse(monkeypatch):
    monkeypatch.setattr(
        kernels, "_launch_linear_norm_silu", kernels.linear_norm_silu_reference
    )
    inputs = _inputs(rows=2, inner=2, width=3)
    x, *parameters = inputs

    def fused(value):
        return (
            kernels._FusedLinearNormSiLU.apply(value, *parameters, 1e-5).square().sum()
        )

    def native(value):
        return (
            kernels.linear_norm_silu_reference(value, *parameters, 1e-5).square().sum()
        )

    expected = torch.autograd.functional.hessian(native, x)
    actual = torch.autograd.functional.hessian(fused, x, vectorize=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_launch_errors_are_not_swallowed(monkeypatch):
    def broken_launch(*args):
        raise RuntimeError("kernel launch failed")

    monkeypatch.setattr(kernels, "_launch_linear_norm_silu", broken_launch)
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        kernels._FusedLinearNormSiLU.apply(*_inputs(), 1e-5)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires an Intel XPU")
@pytest.mark.parametrize("shape", [(1, 16, 16), (17, 31, 33), (32, 128, 128)])
def test_actual_xpu_forward_force_and_hvp(monkeypatch, shape):
    pytest.importorskip("triton.language.extra.intel.libdevice")
    monkeypatch.setenv("FAIRCHEM_XPU_FUSED_KERNELS", "1")
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "0")
    inputs = _inputs(*shape, dtype=torch.float32, device="xpu")
    actual = kernels.xpu_fused_linear_norm_silu(*inputs, 1e-5)
    assert actual is not None, "supported XPU inputs must execute the opt-in kernel"
    expected = kernels.linear_norm_silu_reference(*inputs, 1e-5)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    actual_grad = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
    expected_grad = torch.autograd.grad(
        expected.square().sum(), inputs, create_graph=True
    )
    for result, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(result, reference, atol=2e-4, rtol=2e-4)
    vector = torch.randn_like(inputs[0])
    actual_hvp = torch.autograd.grad(actual_grad[0], inputs[0], vector)[0]
    expected_hvp = torch.autograd.grad(expected_grad[0], inputs[0], vector)[0]
    torch.testing.assert_close(actual_hvp, expected_hvp, atol=5e-4, rtol=5e-4)

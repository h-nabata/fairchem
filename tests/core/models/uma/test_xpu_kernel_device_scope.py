"""Check launch placement without claiming native GPU kernel validation."""

from __future__ import annotations

import math
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


class _Tensor:
    device = torch.device("xpu:1")
    dtype = torch.float32

    def __init__(self, shape):
        self.shape = shape

    def contiguous(self):
        return self

    def numel(self):
        return math.prod(self.shape)

    def new_empty(self, shape):
        return _Tensor(shape)


@pytest.mark.parametrize("operation", ["radial", "producer", "projected", "consumer"])
@pytest.mark.parametrize("fail", [False, True])
def test_xpu_launch_uses_input_ordinal_and_restores_caller(
    monkeypatch, operation, fail
):
    current = torch.device("xpu:0")
    calls = []

    @contextmanager
    def scope(device):
        nonlocal current
        previous, current = current, device
        try:
            yield
        finally:
            current = previous

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                assert current == torch.device("xpu:1")
                calls.append(operation)
                if fail:
                    raise RuntimeError("injected launch failure")

            return launch

    monkeypatch.setattr(torch.xpu, "device", scope)
    if operation == "radial":
        from fairchem.core.models.uma.nn import xpu_radial_kernels as kernels

        compiler = SimpleNamespace(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        )
        monkeypatch.setattr(kernels, "_load_kernel", lambda: (compiler, Kernel()))
        monkeypatch.setattr(torch, "empty", lambda shape, **kw: _Tensor(shape))

        def call():
            return kernels._launch_linear_norm_silu(
                *(_Tensor(s) for s in ((2, 16), (32, 16), (32,), (32,), (32,))),
                1e-5,
            )
    else:
        pytest.importorskip("triton")
        from fairchem.core.models.uma.nn import xpu_wigner_kernels as kernels

        monkeypatch.setattr(kernels, "_require_strict_math", lambda: None)
        monkeypatch.setattr(kernels, "_device_key", str)
        monkeypatch.setattr(kernels, "_producer_kernel", Kernel())
        monkeypatch.setattr(kernels, "_consumer_kernel", Kernel())
        x, index, wigner = _Tensor((3, 9, 4)), _Tensor((2, 6)), _Tensor((6, 35))

        def call():
            if operation == "producer":
                return kernels.FusedProducer.forward(
                    x, index, wigner, _Tensor((6, 48)), 4
                )
            if operation == "projected":
                return kernels.FusedProjectedProducer.forward(
                    x,
                    index,
                    wigner,
                    _Tensor((6, 8)),
                    _Tensor((48, 8)),
                    _Tensor((48,)),
                    4,
                )
            return kernels.FusedConsumer.forward(
                *(_Tensor((6, n * 4)) for n in (3, 4, 2)),
                wigner,
                _Tensor((6,)),
                _Tensor((6,)),
                _Tensor((4,)),
                3,
                4,
                x,
            )

    if fail:
        with pytest.raises(RuntimeError, match="injected launch failure"):
            call()
    else:
        call()
    assert current == torch.device("xpu:0")
    assert calls == [operation]


@pytest.mark.skipif(torch.xpu.device_count() < 2, reason="requires two Intel XPUs")
@pytest.mark.parametrize("operation", ["radial", "producer", "projected", "consumer"])
def test_native_xpu1_kernels_with_xpu0_current(monkeypatch, operation):
    pytest.importorskip("triton.language.extra.intel.libdevice")
    from fairchem.core.models.uma.nn import portable_wigner as reference
    from fairchem.core.models.uma.nn import xpu_radial_kernels as radial
    from fairchem.core.models.uma.nn import xpu_wigner_kernels as wigner

    monkeypatch.setenv("FAIRCHEM_XPU_FUSED_KERNELS", "1")
    monkeypatch.setenv("TRITON_INTEL_FAST_MATH", "0")
    generator = torch.Generator(device="xpu:1").manual_seed(1611)

    def tensor(*shape):
        return torch.randn(
            *shape, device="xpu:1", generator=generator, requires_grad=True
        )

    with torch.xpu.device(0), torch.autocast("xpu", enabled=False):
        index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device="xpu:1")
        rotation = tensor(4, 35)
        if operation == "radial":
            inputs = (tensor(4, 16), tensor(32, 16), *(tensor(32) for _ in range(3)))
            actual = (radial.xpu_fused_linear_norm_silu(*inputs, 1e-5),)
            expected = (radial.linear_norm_silu_reference(*inputs, 1e-5),)
        elif operation == "producer":
            inputs = (tensor(3, 9, 4), rotation, tensor(4, 48))
            x, rotation, coefficients = inputs
            actual = wigner.FusedProducer.apply(x, index, rotation, coefficients, 4)
            expected = reference.producer_reference(x, index, rotation, coefficients, 4)
        elif operation == "projected":
            inputs = (
                tensor(3, 9, 4),
                rotation,
                tensor(4, 8),
                tensor(48, 8),
                tensor(48),
            )
            x, rotation, hidden, weight, bias = inputs
            actual = wigner.FusedProjectedProducer.apply(
                x, index, rotation, hidden, weight, bias, 4
            )
            projection = reference.RadialProjection(hidden, weight, bias)
            expected = reference.producer_projected_reference(
                x, index, rotation, projection, 4
            )
        else:
            inputs = (*(tensor(4, n * 4) for n in (3, 4, 2)), rotation, tensor(3, 9, 4))
            g0, g1, g2, rotation, residual = inputs
            topology = reference.get_csr_topology(index[1], 3)
            actual = (
                wigner.FusedConsumer.apply(
                    g0,
                    g1,
                    g2,
                    rotation,
                    index[1],
                    topology.order,
                    topology.offsets,
                    3,
                    4,
                    residual,
                ),
            )
            expected = (
                reference.consumer_reference(
                    g0, g1, g2, rotation, index[1], 3, 4, residual
                ),
            )
        assert torch.xpu.current_device() == 0
        for left, right in zip(actual, expected):
            assert left.device == torch.device("xpu:1")
            torch.testing.assert_close(left, right, atol=5e-5, rtol=5e-5)
        actual_grad = torch.autograd.grad(
            sum(x.square().sum() for x in actual), inputs, create_graph=True
        )
        expected_grad = torch.autograd.grad(
            sum(x.square().sum() for x in expected), inputs, create_graph=True
        )
        for left, right in zip(actual_grad, expected_grad):
            torch.testing.assert_close(left, right, atol=5e-4, rtol=5e-4)
        direction = torch.ones_like(inputs[0])
        actual_hvp = torch.autograd.grad(actual_grad[0], inputs[0], direction)[0]
        expected_hvp = torch.autograd.grad(expected_grad[0], inputs[0], direction)[0]
        torch.testing.assert_close(actual_hvp, expected_hvp, atol=1e-3, rtol=1e-3)
        assert torch.xpu.current_device() == 0

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import weakref
from dataclasses import replace

import pytest
import torch
from ase.build import bulk, molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.escn_md import MLP_EFS_Head
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
from fairchem.core.models.uma.nn.execution_backends import (
    UMASFastPortableBackend,
    get_execution_backend,
)
from fairchem.core.models.uma.nn.portable_wigner import (
    CSRTopologyCache,
    _validate_kernel_indices,
    compact_wigner,
    consumer,
    consumer_adjoint,
    consumer_reference,
    producer,
    producer_adjoint,
    producer_projected,
    producer_projected_adjoint,
    producer_projected_reference,
    producer_reference,
    segmented_sum,
)
from fairchem.core.models.uma.nn.unified_radial import RadialProjection
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu


def _dense(w):
    result = w.new_zeros((len(w), 9, 9))
    result[:, :1, :1] = w[:, :1].reshape(-1, 1, 1)
    result[:, 1:4, 1:4] = w[:, 1:10].reshape(-1, 3, 3)
    result[:, 4:, 4:] = w[:, 10:].reshape(-1, 5, 5)
    return result


def _producer_dense(x, edges, w, radial, channels):
    mapping = CoefficientMapping(2, 2).to(device=x.device, dtype=x.dtype)
    rotated = torch.bmm(
        torch.einsum("ij,ejk->eik", mapping.to_m, _dense(w)),
        torch.cat((x[edges[0]], x[edges[1]]), dim=2),
    )
    r0, r1, r2 = radial.split((6 * channels, 4 * channels, 2 * channels), -1)
    return (
        rotated[:, :3].flatten(1) * r0,
        (rotated[:, 3:7].reshape(len(w), 2, 4 * channels) * r1[:, None]).flatten(1),
        (rotated[:, 7:].reshape(len(w), 2, 2 * channels) * r2[:, None]).flatten(1),
    )


def _consumer_dense(g0, g1, g2, w, target, nodes, channels, residual=None):
    mapping = CoefficientMapping(2, 2).to(device=w.device, dtype=w.dtype)
    message = torch.cat((g0, g1, g2), dim=1).reshape(len(w), 9, channels)
    rotation = torch.einsum("eij,kj->eik", _dense(w), mapping.to_m)
    values = torch.bmm(rotation, message)
    result = w.new_zeros((nodes, 9, channels)).index_add(0, target, values)
    return result if residual is None else result + residual


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [0, 7])
def test_compact_forward_matches_dense_with_strided_inputs(dtype, count):
    torch.manual_seed(4)
    x = torch.randn(5, 9, 6, dtype=dtype)[..., ::2]
    edges = torch.randint(0, 5, (2, count))
    w = torch.randn(count, 70, dtype=dtype)[:, ::2]
    radial = torch.randn(count, 72, dtype=dtype)[:, ::2]
    torch.testing.assert_close(compact_wigner(_dense(w)), w)
    for actual, expected in zip(
        producer(x, edges, w, radial, 3), _producer_dense(x, edges, w, radial, 3)
    ):
        torch.testing.assert_close(actual, expected)
    packed = [torch.randn(count, n * 6, dtype=dtype)[:, ::2] for n in (3, 4, 2)]
    residual = torch.randn_like(x)
    torch.testing.assert_close(
        consumer(*packed, w, edges[1], 5, 3, residual),
        _consumer_dense(*packed, w, edges[1], 5, 3, residual),
    )


def test_producer_gradcheck_and_gradgradcheck():
    torch.manual_seed(9)
    edges = torch.tensor([[0, 1], [1, 2]])
    inputs = tuple(
        torch.randn(shape, dtype=torch.double, requires_grad=True)
        for shape in ((3, 9, 1), (2, 35), (2, 12))
    )

    def function(x, w, radial):
        return producer_reference(x, edges, w, radial, 1)

    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)
    grad = [torch.randn_like(v, requires_grad=True) for v in function(*inputs)]
    analytic = producer_adjoint(*grad, inputs[0], edges, inputs[1], inputs[2], 1)
    expected = torch.autograd.grad(function(*inputs), inputs, grad, create_graph=True)
    for a, b in zip(analytic, expected):
        torch.testing.assert_close(a, b)
    # The adjoint used by XPU kernels must itself retain cross derivatives.
    actual_second = torch.autograd.grad(sum(v.square().sum() for v in analytic), inputs)
    expected_second = torch.autograd.grad(
        sum(v.square().sum() for v in expected), inputs
    )
    for a, b in zip(actual_second, expected_second):
        torch.testing.assert_close(a, b)


def test_consumer_gradcheck_and_gradgradcheck():
    torch.manual_seed(13)
    target = torch.tensor([2, 0, 2])
    inputs = tuple(
        torch.randn(shape, dtype=torch.double, requires_grad=True)
        for shape in ((3, 3), (3, 4), (3, 2), (3, 35), (4, 9, 1))
    )

    def function(g0, g1, g2, w, residual):
        return consumer_reference(g0, g1, g2, w, target, 4, 1, residual)

    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)
    grad = torch.randn(4, 9, 1, dtype=torch.double, requires_grad=True)
    analytic = consumer_adjoint(grad, *inputs[:4], target, 1)
    expected = torch.autograd.grad(function(*inputs), inputs, grad, create_graph=True)
    for a, b in zip(analytic, expected[:4]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(expected[4], grad)
    actual_second = torch.autograd.grad(
        sum(v.square().sum() for v in analytic), inputs[:4]
    )
    expected_second = torch.autograd.grad(
        sum(v.square().sum() for v in expected[:4]), inputs[:4]
    )
    for a, b in zip(actual_second, expected_second):
        torch.testing.assert_close(a, b)


def test_lazy_producer_adjoint_preserves_second_derivatives():
    torch.manual_seed(114)
    edges = torch.tensor([[0, 1], [1, 2]])
    inputs = tuple(
        torch.randn(shape, dtype=torch.double, requires_grad=True)
        for shape in ((3, 9, 1), (2, 35), (2, 5), (12, 5), (12,))
    )

    def function(x, w, h, weight, bias):
        return producer_projected(x, edges, w, RadialProjection(h, weight, bias), 1)

    assert torch.autograd.gradcheck(function, inputs, fast_mode=True)
    assert torch.autograd.gradgradcheck(function, inputs, fast_mode=True)
    seeds = [torch.randn_like(v, requires_grad=True) for v in function(*inputs)]
    expected = torch.autograd.grad(function(*inputs), inputs, seeds, create_graph=True)
    analytic = producer_projected_adjoint(*seeds, inputs[0], edges, *inputs[1:], 1)
    for a, b in zip(analytic, expected):
        torch.testing.assert_close(a, b)
    actual_second = torch.autograd.grad(sum(v.square().sum() for v in analytic), inputs)
    expected_second = torch.autograd.grad(
        sum(v.square().sum() for v in expected), inputs
    )
    for a, b in zip(actual_second, expected_second):
        torch.testing.assert_close(a, b)


def test_segment_sum_vmap_backward_and_forward_ad():
    target = torch.tensor([2, 0, 2, 1])
    x = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    y = segmented_sum(x.sin(), target, 5)
    seed = torch.randn(2, 5, 3, dtype=torch.double)
    actual = torch.autograd.grad(y, x, seed, is_grads_batched=True)[0]
    torch.testing.assert_close(actual, seed[:, target] * x.cos()[None])
    tangent = torch.randn_like(x)
    value, jvp = torch.func.jvp(
        lambda a: segmented_sum(a.sin(), target, 5), (x,), (tangent,)
    )
    torch.testing.assert_close(value, y)
    torch.testing.assert_close(
        jvp, torch.zeros_like(y).index_add(0, target, tangent * x.cos())
    )


def test_csr_cache_owns_source_invalidates_mutation_and_bounds_memory():
    cache = CSRTopologyCache(max_entries=1, max_bytes=4096)
    target = torch.tensor([2, 0, 2, 1])
    owner = weakref.ref(target)
    first = cache.get(target, 4)
    assert cache.get(target, 4) is first
    assert first.index32
    assert first.order.dtype == torch.int32
    assert first.order.tolist() == [1, 3, 0, 2]
    target[0] = 0
    second = cache.get(target, 4)
    assert second is not first
    assert second.order.tolist() == [0, 1, 3, 2]
    del target, first, second
    assert owner() is not None
    cache.clear()
    assert owner() is None
    cache = CSRTopologyCache(max_bytes=1)
    cache.get(torch.tensor([0]), 1)
    assert not cache._entries


def test_csr_cache_checks_stride_device_size_and_inference_version():
    cache = CSRTopologyCache()
    source = torch.tensor([1, 2, 0, 2, 1, 0])
    view = source[::2]
    assert cache.get(view, 3).order.tolist() == [1, 0, 2]
    assert cache.get(view, 4).lengths.numel() == 4
    with pytest.raises(ValueError, match="outside"):
        cache.get(torch.tensor([-1]), 3)
    with torch.inference_mode():
        inference_target = torch.tensor([0, 1])
    assert cache.get(inference_target, 2) is not cache.get(inference_target, 2)


def test_raw_kernel_index_validation_invalidates_changed_topology():
    indices = torch.tensor([[0, 1], [2, 2]])
    _validate_kernel_indices(indices, 3)
    _validate_kernel_indices(indices, 3)
    indices[0, 0] = -1
    with pytest.raises(ValueError, match="outside"):
        _validate_kernel_indices(indices, 3)
    indices[0, 0] = 0
    with pytest.raises(ValueError, match="outside"):
        _validate_kernel_indices(indices, 2)
    with torch.inference_mode():
        no_version = torch.tensor([[0], [1]])
        _validate_kernel_indices(no_version, 2)
        no_version[0, 0] = 2
        with pytest.raises(ValueError, match="outside"):
            _validate_kernel_indices(no_version, 2)


@pytest.mark.parametrize("periodic", [False, True])
def test_portable_backend_full_uma_energy_force_stress(periodic):
    torch.manual_seed(27)
    atoms = bulk("Cu", "fcc", a=3.7, cubic=True) if periodic else molecule("H2O")
    atoms.center(vacuum=4)
    model = (
        eSCNMDMoeBackbone(
            sphere_channels=4,
            hidden_channels=8,
            edge_channels=4,
            num_distance_basis=4,
            num_layers=1,
            num_experts=2,
            lmax=2,
            mmax=2,
            cutoff=3.0,
            otf_graph=True,
            use_dataset_embedding=False,
            use_composition_embedding=True,
            always_use_pbc=False,
            direct_forces=False,
            regress_forces=True,
            regress_stress=True,
            model_version=1.2,
        )
        .eval()
        .requires_grad_(False)
    )
    model.model_id = "UMA-S-1.2.1"
    head = MLP_EFS_Head(model, wrap_property=False).eval().requires_grad_(False)
    reference_model = copy.deepcopy(model).prepare_for_inference(
        AtomicData.from_ase(atoms, task_name="omat"), inference_settings_xpu()
    )
    settings = replace(inference_settings_xpu(), execution_mode="umas_fast_portable")
    candidate = model.prepare_for_inference(
        AtomicData.from_ase(atoms, task_name="omat"), settings
    )

    def evaluate(backbone):
        torch.manual_seed(123)
        data = AtomicData.from_ase(atoms, task_name="omat")
        return head(data, backbone(data))

    expected = evaluate(reference_model)
    actual = evaluate(candidate)
    assert torch.count_nonzero(expected["forces"]) > 0
    assert torch.isfinite(expected["forces"]).all()
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-4, atol=2e-5)
    assert actual["energy"].dtype == torch.float64
    assert actual["stress"].dtype == torch.float64


def test_portable_backend_configuration_and_compile_reference():
    settings = replace(inference_settings_xpu(), execution_mode="umas_fast_portable")
    backend = get_execution_backend(settings.execution_mode)
    assert isinstance(backend, UMASFastPortableBackend)
    backend.validate(2, 2, settings)
    for bad in (replace(settings, tf32=True), replace(settings, merge_mole=False)):
        with pytest.raises(ValueError):
            backend.validate(2, 2, bad)
    with pytest.raises(ValueError):
        backend.validate(3, 2, settings)
    compiled = torch.compile(producer_reference, backend="eager", fullgraph=True)
    x, edges, w, radial = (
        torch.randn(4, 9, 2),
        torch.tensor([[0, 1], [1, 2]]),
        torch.randn(2, 35),
        torch.randn(2, 24),
    )
    for a, b in zip(
        compiled(x, edges, w, radial, 2), producer_reference(x, edges, w, radial, 2)
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(
    not torch.xpu.is_available(), reason="Requires real Intel XPU and Triton-XPU"
)
def test_xpu_fused_kernels_values_and_second_derivatives(monkeypatch):
    monkeypatch.setenv("FAIRCHEM_XPU_FUSED_KERNELS", "1")
    torch.manual_seed(33)
    edges = torch.tensor([[0, 1, 2], [2, 2, 0]], device="xpu")
    inputs = tuple(
        torch.randn(shape, device="xpu", requires_grad=True)
        for shape in ((3, 9, 5), (3, 35), (3, 60))
    )
    actual = producer(inputs[0], edges, inputs[1], inputs[2], 5)
    expected = producer_reference(inputs[0], edges, inputs[1], inputs[2], 5)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for outputs in (actual, expected):
        first = torch.autograd.grad(
            sum(v.square().sum() for v in outputs), inputs, create_graph=True
        )
        second = torch.autograd.grad(
            sum(v.sum() for v in first), inputs, retain_graph=True
        )
        if outputs is actual:
            reference_first, reference_second = first, second
        else:
            for a, b in zip((*first, *second), (*reference_first, *reference_second)):
                torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    blocks = tuple(
        torch.randn(3, n * 5, device="xpu", requires_grad=True) for n in (3, 4, 2)
    )
    residual = torch.randn(3, 9, 5, device="xpu", requires_grad=True)
    actual = consumer(*blocks, inputs[1], edges[1], 3, 5, residual)
    expected = consumer_reference(*blocks, inputs[1], edges[1], 3, 5, residual)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    arguments = (*blocks, inputs[1], residual)
    actual_first = torch.autograd.grad(
        actual.square().sum(), arguments, create_graph=True
    )
    expected_first = torch.autograd.grad(
        expected.square().sum(), arguments, create_graph=True
    )
    actual_second = torch.autograd.grad(sum(v.sum() for v in actual_first), arguments)
    expected_second = torch.autograd.grad(
        sum(v.sum() for v in expected_first), arguments
    )
    for a, b in zip(
        (*actual_first, *actual_second), (*expected_first, *expected_second)
    ):
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    projection = RadialProjection(
        *(
            torch.randn(shape, device="xpu", requires_grad=True)
            for shape in ((3, 7), (60, 7), (60,))
        )
    )
    actual = producer_projected(inputs[0], edges, inputs[1], projection, 5)
    expected = producer_projected_reference(inputs[0], edges, inputs[1], projection, 5)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
    arguments = (inputs[0], inputs[1], *projection)
    actual_first = torch.autograd.grad(
        sum(v.square().sum() for v in actual), arguments, create_graph=True
    )
    expected_first = torch.autograd.grad(
        sum(v.square().sum() for v in expected), arguments, create_graph=True
    )
    actual_second = torch.autograd.grad(sum(v.sum() for v in actual_first), arguments)
    expected_second = torch.autograd.grad(
        sum(v.sum() for v in expected_first), arguments
    )
    for a, b in zip(
        (*actual_first, *actual_second), (*expected_first, *expected_second)
    ):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(
    not torch.xpu.is_available(), reason="Requires real Intel XPU streams"
)
def test_xpu_csr_cache_cross_stream_reuse():
    cache = CSRTopologyCache()
    first_stream, second_stream = torch.xpu.Stream(), torch.xpu.Stream()
    with torch.xpu.stream(first_stream):
        target = torch.randint(0, 13, (3000,), device="xpu")
        topology = cache.get(target, 13)
    with torch.xpu.stream(second_stream):
        reused = cache.get(target, 13)
        assert reused is topology
        sorted_target = target[reused.order]
        expected = torch.sort(target, stable=True)[0]
    second_stream.synchronize()
    torch.testing.assert_close(sorted_target, expected)

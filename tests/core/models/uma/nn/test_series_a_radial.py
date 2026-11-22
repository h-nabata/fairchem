"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.radial import RadialMLP, layer_norm_silu
from fairchem.core.models.uma.nn.so2_layers import (
    SO2_Convolution,
    SO2_m_Conv,
    SO2_m_Conv_Block,
    convert_so2_conv1,
    convert_so2_conv2,
)
from fairchem.core.models.uma.nn.unified_radial import (
    RadialProjection,
    UnifiedRadialMLP,
)


def _radial(dtype=torch.float64):
    mlps = [RadialMLP([10, 5, 7, 12]).to(dtype) for _ in range(2)]
    for i, mlp in enumerate(mlps):
        mlp.net[1].eps = 0.003 * (i + 1)
        mlp.net[4].eps = 0.007 * (i + 1)
    return mlps, UnifiedRadialMLP(mlps)


@pytest.mark.parametrize("edges", [0, 1, 9])
def test_unified_preserves_each_layer_eps_and_hidden_width(edges):
    mlps, unified = _radial()
    x = torch.randn(edges, 10, dtype=torch.float64, requires_grad=True)
    for expected, actual in zip((m(x) for m in mlps), unified(x)):
        torch.testing.assert_close(actual, expected)
    projected = unified(x, lazy=True)
    for i, value in enumerate(projected):
        assert isinstance(value, RadialProjection)
        assert value.hidden.shape == (edges, 7)
        torch.testing.assert_close(value.materialize(), mlps[i](x))
        torch.testing.assert_close(value.block(2, 4), mlps[i](x)[:, 2:6])


def _decomposed(unified, distance, source, target):
    src = torch.tensor([0, 3, 1], dtype=torch.long)
    dst = torch.tensor([1, 2, 0], dtype=torch.long)
    actual = unified.forward_decomposed(distance, src, dst, source, target)
    reference = unified(torch.cat((distance, source[src], target[dst]), dim=1))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
    return actual, reference


def test_element_precomputation_preserves_value_first_and_second_derivatives():
    _, unified = _radial()
    distance = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    source = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    actual, reference = _decomposed(unified, distance, source, target)
    inputs = (distance, source, target)
    ga = torch.autograd.grad(
        sum(x.square().sum() for x in actual), inputs, create_graph=True
    )
    gb = torch.autograd.grad(
        sum(x.square().sum() for x in reference), inputs, create_graph=True
    )
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    ha = torch.autograd.grad(sum(x.square().sum() for x in ga), inputs)
    hb = torch.autograd.grad(sum(x.square().sum() for x in gb), inputs)
    for a, b in zip(ha, hb):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)
    assert unified._element_cache_key is None


def test_element_cache_invalidates_on_mutation_and_dtype_conversion():
    _, unified = _radial()
    distance = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    source = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    _decomposed(unified, distance, source, target)
    first = unified._source_table
    cloned = copy.deepcopy(unified)
    assert cloned._source_table is None
    _decomposed(cloned, distance, source, target)
    _decomposed(unified, distance, source, target)
    assert unified._source_table is first
    source.add_(0.25)
    _decomposed(unified, distance, source, target)
    assert unified._source_table is not first
    first = unified._source_table
    unified.W1_cat.mul_(1.1)
    _decomposed(unified, distance, source, target)
    assert unified._source_table is not first
    unified.float()
    assert unified._source_table is None
    assert unified._element_cache_owners == ()
    output = unified.forward_decomposed(
        distance.float(),
        torch.tensor([0, 3, 1]),
        torch.tensor([1, 2, 0]),
        source.float(),
        target.float(),
    )
    assert all(x.dtype == torch.float32 for x in output)


def test_upstream_packed_radial_state_restores_shared_epsilon_and_cache():
    radial_mlps = [RadialMLP([10, 6, 6, 12]).double() for _ in range(2)]
    for mlp in radial_mlps:
        mlp.net[1].eps = mlp.net[4].eps = 0.035
    radial = UnifiedRadialMLP(radial_mlps)
    # These fields and persistent buffers are the complete upstream module.
    state = torch.nn.Module.__getstate__(radial)
    state["_buffers"] = dict(radial.named_buffers())
    state["_non_persistent_buffers_set"] = set()
    for key in (
        "ln1_eps",
        "ln2_eps",
        "hidden2_features",
        "_element_cache_key",
        "_element_cache_owners",
        "_element_cache_event",
    ):
        del state[key]
    restored = UnifiedRadialMLP.__new__(UnifiedRadialMLP)
    restored.__setstate__(state)

    distance = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    source, target = (torch.randn(4, 3, dtype=torch.float64) for _ in range(2))
    actual, expected = _decomposed(restored, distance, source, target)
    for i, (a, b) in enumerate(zip(actual, expected)):
        direct = radial_mlps[i](
            torch.cat((distance, source[[0, 3, 1]], target[[1, 2, 0]]), dim=-1)
        )
        torch.testing.assert_close(a, direct)
        torch.testing.assert_close(b, direct)
    assert restored.state_dict().keys() == radial.state_dict().keys()
    assert copy.deepcopy(restored)._source_table is None


def test_element_cache_never_retains_a_differentiable_tape():
    _, unified = _radial()
    source = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    target = torch.randn(4, 3, dtype=torch.float64)
    for _ in range(2):
        distance = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        actual, _ = _decomposed(unified, distance, source, target)
        torch.autograd.grad(sum(x.sum() for x in actual), (source, distance))
    assert unified._source_table is None


@pytest.mark.parametrize(
    ("warm_dtype", "next_dtype"),
    [
        (torch.bfloat16, None),
        (None, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.bfloat16),
    ],
)
def test_element_cache_does_not_mix_autocast_policies(warm_dtype, next_dtype):
    _, unified = _radial(torch.float32)
    fresh = copy.deepcopy(unified)
    source, target = torch.randn(4, 3), torch.randn(4, 3)
    src, dst = torch.tensor([0, 3, 1]), torch.tensor([1, 2, 0])
    distance = torch.randn(3, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=warm_dtype, enabled=warm_dtype is not None):
        unified.forward_decomposed(distance, src, dst, source, target)
    with torch.autocast("cpu", dtype=next_dtype, enabled=next_dtype is not None):
        actual = unified.forward_decomposed(distance, src, dst, source, target)
        expected = fresh.forward_decomposed(distance, src, dst, source, target)
        # A second call in the same policy must still reuse the ready table.
        table = unified._source_table
        unified.forward_decomposed(distance, src, dst, source, target)
        assert unified._source_table is table
    assert table.dtype == (next_dtype or torch.float32)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    derivatives = []
    for output in (actual, expected):
        first = torch.autograd.grad(
            sum(value.square().sum() for value in output), distance, create_graph=True
        )[0]
        second = torch.autograd.grad(first.square().sum(), distance)[0]
        derivatives.append((first, second))
    for a, b in zip(*derivatives):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("assign", [False, True])
def test_parent_state_reload_releases_derived_radial_tables(assign):
    _, unified = _radial()
    parent = torch.nn.Module()
    parent.radial = unified
    distance = torch.randn(3, 4, dtype=torch.float64)
    source, target = (torch.randn(4, 3, dtype=torch.float64) for _ in range(2))
    checkpoint = {name: value.clone() for name, value in parent.state_dict().items()}
    _decomposed(unified, distance, source, target)
    table_ref = weakref.ref(unified._source_table)
    weight_ref = weakref.ref(unified.W1_cat)
    parent.load_state_dict(checkpoint, assign=assign)
    assert unified._source_table is unified._target_table is None
    assert unified._element_cache_owners == ()
    assert table_ref() is None
    if assign:
        assert weight_ref() is None
    _decomposed(unified, distance, source, target)


def test_element_cache_distinguishes_fp32_matmul_policy(monkeypatch):
    _, unified = _radial(torch.float32)
    distance, source, target = torch.randn(3, 4), torch.randn(4, 3), torch.randn(4, 3)
    src, dst = torch.tensor([0, 3, 1]), torch.tensor([1, 2, 0])
    previous = None
    for policy in ("high", "highest", "medium", "highest"):
        # Exercise policy invalidation on CPU without changing process-global
        # TF32 state or pretending that this host implements CUDA arithmetic.
        monkeypatch.setattr(
            torch, "get_float32_matmul_precision", lambda policy=policy: policy
        )
        actual = unified.forward_decomposed(distance, src, dst, source, target)
        table = unified._source_table
        assert table is not previous
        expected = copy.deepcopy(unified).forward_decomposed(
            distance, src, dst, source, target
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        unified.forward_decomposed(distance, src, dst, source, target)
        assert unified._source_table is table
        previous = table


def test_decomposed_radial_supports_meta_shape_evaluation():
    _, unified = _radial(torch.float32)
    unified.to("meta")
    outputs = unified.forward_decomposed(
        torch.empty(3, 4, device="meta"),
        torch.empty(3, dtype=torch.long, device="meta"),
        torch.empty(3, dtype=torch.long, device="meta"),
        torch.empty(4, 3, device="meta"),
        torch.empty(4, 3, device="meta"),
    )
    assert len(outputs) == 2
    assert all(output.shape == (3, 12) and output.is_meta for output in outputs)


def test_element_cache_distinguishes_virtual_xpu_onednn_policy(monkeypatch):
    from fairchem.core.models.uma.nn import unified_radial

    # Substitute only policy metadata; numerical operations stay on the CPU.
    # This verifies cache isolation, not Intel kernel execution.
    backend = SimpleNamespace(allow_tf32=False)
    weight_metadata = SimpleNamespace(device=torch.device("xpu:0"), dtype=torch.float32)
    precision_key = unified_radial._precision_cache_key
    monkeypatch.setattr(torch.backends, "mkldnn", backend)
    monkeypatch.setattr(
        unified_radial,
        "_precision_cache_key",
        lambda tensor: precision_key(weight_metadata),
    )
    initialized = torch.xpu.is_initialized()
    _, unified = _radial(torch.float32)
    distance, source, target = torch.randn(3, 4), torch.randn(4, 3), torch.randn(4, 3)
    src, dst = torch.tensor([0, 3, 1]), torch.tensor([1, 2, 0])
    previous = None
    for policy in (False, True, False, None):
        backend.allow_tf32 = policy
        actual = unified.forward_decomposed(distance, src, dst, source, target)
        table = unified._source_table
        assert table is not previous
        expected = copy.deepcopy(unified).forward_decomposed(
            distance, src, dst, source, target
        )
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        unified.forward_decomposed(distance, src, dst, source, target)
        assert unified._source_table is table
        previous = table
    del backend.allow_tf32
    assert precision_key(weight_metadata)[-1] is None
    assert torch.xpu.is_initialized() == initialized


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires an Intel XPU")
def test_cached_radial_tables_and_so2_weights_cross_xpu_streams():
    mlps, _ = _radial(torch.float32)
    unified = UnifiedRadialMLP([mlp.to("xpu") for mlp in mlps])
    block = SO2_m_Conv_Block(1, 3, 2, 2, 2).to("xpu").requires_grad_(False)
    distance = torch.randn(1024, 4, device="xpu")
    source = torch.randn(4, 3, device="xpu")
    target = torch.randn(4, 3, device="xpu")
    src = torch.randint(4, (1024,), device="xpu")
    dst = torch.randint(4, (1024,), device="xpu")
    features = torch.randn(1024, 2, 6, device="xpu")
    producer, consumer = torch.xpu.Stream(), torch.xpu.Stream()
    producer.wait_stream(torch.xpu.current_stream())
    consumer.wait_stream(torch.xpu.current_stream())
    with torch.no_grad(), torch.xpu.stream(producer):
        unified.forward_decomposed(distance, src, dst, source, target)
        block(features)
    # No consumer wait on producer: each hidden cache must establish readiness.
    with torch.no_grad(), torch.xpu.stream(consumer):
        actual = unified.forward_decomposed(distance, src, dst, source, target)
        actual_block = block(features)
        expected = unified(torch.cat((distance, source[src], target[dst]), dim=1))
        expected_block = F.linear(features.flatten(1), block._make_w_block())
        # Drop ownership while the consumer may still be reading cached tensors.
        unified._clear_element_cache()
        block._w_block = None
    consumer.synchronize()
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(
        torch.cat(actual_block, dim=1).flatten(1), expected_block
    )


def test_decomposed_radial_fullgraph_capture_preserves_gradients():
    _, unified = _radial()
    compiled = torch.compile(
        unified.forward_decomposed, backend="eager", fullgraph=True
    )
    source = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    src = torch.tensor([0, 3, 1])
    dst = torch.tensor([1, 2, 0])
    distance = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
    expected = unified.forward_decomposed(distance, src, dst, source, target)
    actual = compiled(distance, src, dst, source, target)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(
        sum(x.square().sum() for x in actual), distance, create_graph=True
    )[0]
    gb = torch.autograd.grad(
        sum(x.square().sum() for x in expected), distance, create_graph=True
    )[0]
    torch.testing.assert_close(ga, gb)
    torch.testing.assert_close(
        torch.autograd.grad(ga.square().sum(), distance)[0],
        torch.autograd.grad(gb.square().sum(), distance)[0],
    )


def test_norm_silu_gradients_jvp_and_saved_activation_storage():
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(5, dtype=torch.float64, requires_grad=True)

    def fn(a, b, c):
        return layer_norm_silu(a, b, c, 0.003)

    assert torch.autograd.gradcheck(fn, (x, weight, bias))
    assert torch.autograd.gradgradcheck(fn, (x, weight, bias))

    def native(a):
        return F.silu(F.layer_norm(a, (5,), weight, bias, 0.003))

    direction = torch.randn_like(x)
    _, actual_jvp = torch.func.jvp(lambda a: fn(a, weight, bias), (x,), (direction,))
    _, expected_jvp = torch.func.jvp(native, (x,), (direction,))
    torch.testing.assert_close(actual_jvp, expected_jvp)
    gradient = torch.func.grad(lambda a: fn(a, weight, bias).square().sum())
    hvp = torch.func.jvp(gradient, (x,), (direction,))[1]
    reference = torch.autograd.functional.hvp(
        lambda a: native(a).square().sum(), x, direction
    )[1]
    torch.testing.assert_close(hvp, reference)
    torch.testing.assert_close(torch.vmap(lambda a: fn(a, weight, bias))(x), native(x))

    def saved_bytes(func):
        saved = []

        def pack(t):
            saved.append(t.numel() * t.element_size())
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            func(x)
        return sum(saved)

    assert saved_bytes(lambda a: fn(a, weight, bias)) < saved_bytes(native)


@pytest.mark.parametrize("active", [(0,), (1,), (2,), (0, 1, 2)])
def test_norm_silu_raw_forward_ad_and_mixed_derivatives(active):
    inputs = tuple(
        torch.randn(shape, dtype=torch.float64, requires_grad=True)
        for shape in ((2, 3, 5), (5,), (5,))
    )
    tangents = [torch.randn_like(value) for value in inputs]
    with torch.autograd.forward_ad.dual_level():
        duals = [
            torch.autograd.forward_ad.make_dual(value, tangents[i])
            if i in active
            else value
            for i, value in enumerate(inputs)
        ]
        actual = layer_norm_silu(*duals, 0.003)
        expected = F.silu(F.layer_norm(duals[0], (5,), *duals[1:], 0.003))
        actual, expected = map(
            torch.autograd.forward_ad.unpack_dual, (actual, expected)
        )
        torch.testing.assert_close(actual.primal, expected.primal)
        torch.testing.assert_close(actual.tangent, expected.tangent)
        for left, right in zip(
            torch.autograd.grad(actual.tangent.square().sum(), inputs),
            torch.autograd.grad(expected.tangent.square().sum(), inputs),
        ):
            torch.testing.assert_close(left, right)


@pytest.mark.parametrize("shape", [(7,), (2, 3, 7)])
@pytest.mark.parametrize("requires", range(1, 8))
def test_norm_silu_preserves_derivatives_for_partially_frozen_inputs(shape, requires):
    inputs = tuple(
        torch.randn(size, dtype=torch.float64, requires_grad=bool(requires & (1 << i)))
        for i, size in enumerate((shape, (shape[-1],), (shape[-1],)))
    )
    variables = tuple(value for value in inputs if value.requires_grad)
    actual = layer_norm_silu(*inputs, 0.003)
    expected = F.silu(F.layer_norm(inputs[0], (shape[-1],), *inputs[1:], 0.003))
    torch.testing.assert_close(actual, expected)
    derivatives = []
    for output in (actual, expected):
        first = torch.autograd.grad(output.square().sum(), variables, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in first), variables
        )
        derivatives.append((first, second))
    for actual_order, expected_order in zip(*derivatives):
        for a, b in zip(actual_order, expected_order):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)


def test_upstream_cached_angular_block_rebuilds_from_loaded_weights():
    block = SO2_m_Conv_Block(1, 3, 2, 2, 2).double().requires_grad_(False)
    state = torch.nn.Module.__getstate__(block)
    state["_buffers"] = {"_w_block": torch.zeros_like(block._make_w_block())}
    for key in ("compact_weights", "_w_block_key", "_w_block_owner", "_w_block_event"):
        del state[key]
    restored = SO2_m_Conv_Block.__new__(SO2_m_Conv_Block)
    restored.__setstate__(state)
    x = torch.randn(4, 2, 6, dtype=torch.float64, requires_grad=True)
    actual = torch.cat(restored(x), dim=1)
    expected = torch.cat(block(x), dim=1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
    )


def test_block_gemm_preserves_parameter_gradients_and_refreshes_frozen_cache():
    old = SO2_m_Conv(1, 3, 2, 2, 2).double()
    new = SO2_m_Conv_Block(1, 3, 2, 2, 2).double()
    new.load_state_dict(old.state_dict())
    x = torch.randn(4, 2, 6, dtype=torch.float64, requires_grad=True)
    a = torch.cat(new(x), dim=1)
    b = torch.cat(old(x), dim=1)
    torch.testing.assert_close(a, b)
    ga = torch.autograd.grad(a.square().sum(), (x, new.fc.weight), create_graph=True)
    gb = torch.autograd.grad(b.square().sum(), (x, old.fc.weight), create_graph=True)
    for aa, bb in zip(ga, gb):
        torch.testing.assert_close(aa, bb)
    new.requires_grad_(False)
    before = new(x)[0]
    new.fc.weight.add_(1)
    after = new(x)[0]
    assert not torch.allclose(before, after)
    with torch.no_grad():
        old.fc.weight.copy_(new.fc.weight)
    torch.testing.assert_close(after, old(x)[0])
    assert new._w_block_owner[0] is new.fc.weight
    cloned = copy.deepcopy(new)
    assert cloned._w_block is None
    torch.testing.assert_close(cloned(x)[0], new(x)[0])
    new.float()
    assert new._w_block is None
    assert new._w_block_owner == ()


@pytest.mark.parametrize("conv", [1, 2])
def test_so2_conversion_preserves_fp64_training_state_and_ownership(conv):
    old = (
        SO2_Convolution(
            sphere_channels=4,
            m_output_channels=3,
            lmax=2,
            mmax=2,
            mappingReduced=CoefficientMapping(2, 2),
            internal_weights=conv == 2,
            edge_channels_list=[10, 5, 7] if conv == 1 else None,
            extra_m0_output_channels=6 if conv == 1 else None,
        )
        .double()
        .eval()
        .requires_grad_(False)
    )
    if conv == 1:
        old.rad_func.net[1].eps = 0.004
    new = (convert_so2_conv1 if conv == 1 else convert_so2_conv2)(old)
    assert not new.training
    for (name, source), (_, target) in zip(
        old.named_parameters(), new.named_parameters()
    ):
        assert target.dtype == torch.float64, name
        assert not target.requires_grad
        assert source.data_ptr() != target.data_ptr()
        torch.testing.assert_close(source, target, rtol=0, atol=0)
    if conv == 1:
        assert new.rad_func.net[1].eps == 0.004


@pytest.mark.parametrize("edges", [0, 3])
def test_lazy_radial_conv1_matches_materialized_with_second_derivatives(edges):
    old = (
        SO2_Convolution(
            sphere_channels=4,
            m_output_channels=3,
            lmax=2,
            mmax=2,
            mappingReduced=CoefficientMapping(2, 2),
            internal_weights=False,
            edge_channels_list=[10, 5, 7],
            extra_m0_output_channels=6,
        )
        .double()
        .eval()
    )
    conv = convert_so2_conv1(old)
    radial = UnifiedRadialMLP([conv.rad_func])
    x = torch.randn(edges, 9, 4, dtype=torch.float64, requires_grad=True)
    edge = torch.randn(edges, 10, dtype=torch.float64, requires_grad=True)
    payload = radial(edge, lazy=True)[0]
    actual = conv(x, payload)
    expected = conv(x, payload.materialize())
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    if edges:
        ga = torch.autograd.grad(
            sum(v.square().sum() for v in actual), (x, edge), create_graph=True
        )
        gb = torch.autograd.grad(
            sum(v.square().sum() for v in expected), (x, edge), create_graph=True
        )
        ha = torch.autograd.grad(
            sum(v.square().sum() for v in ga), (x, edge), retain_graph=True
        )
        hb = torch.autograd.grad(sum(v.square().sum() for v in gb), (x, edge))
        for a, b in zip(ha, hb):
            torch.testing.assert_close(a, b)


def test_frozen_compact_so2_and_radial_groups_match_packed_execution():
    radial = UnifiedRadialMLP([RadialMLP([11, 6, 8, 12]).double() for _ in range(3)])
    distance = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    table = torch.randn(6, 4, dtype=torch.float64)
    src = torch.arange(9) % 6
    args = (distance, src, src.flip(0), table, table)
    full = radial.forward_decomposed(*args, lazy=True)
    groups = radial.forward_decomposed(*args, lazy=True, end_layer=2)
    groups += radial.forward_decomposed(*args, lazy=True, first_layer=2)
    for packed, grouped in zip(full, groups, strict=True):
        torch.testing.assert_close(packed.materialize(), grouped.materialize())
    block = SO2_m_Conv_Block(1, 3, 2, 2, 2).double().requires_grad_(False)
    x = torch.randn(9, 2, 6, dtype=torch.float64, requires_grad=True)
    compact = copy.deepcopy(block)
    compact.compact_weights = True
    for out, ref in zip(compact(x), block(x), strict=True):
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(
            torch.autograd.grad(out.square().sum(), x, retain_graph=True)[0],
            torch.autograd.grad(ref.square().sum(), x, retain_graph=True)[0],
        )
    assert compact._w_block is None


def test_first_force_norm_silu_matches_native_with_large_common_offset():
    torch.manual_seed(77)
    x = (torch.randn(5, 32) * 0.04 + 10000).requires_grad_()
    w, b = torch.randn(32), torch.randn(32)
    upstream = torch.randn_like(x)
    actual = layer_norm_silu(x, w, b, 1e-5)
    expected = F.silu(F.layer_norm(x, (32,), w, b, 1e-5))
    torch.testing.assert_close(
        torch.autograd.grad(actual, x, upstream)[0],
        torch.autograd.grad(expected, x, upstream)[0],
        rtol=0,
        atol=0,
    )

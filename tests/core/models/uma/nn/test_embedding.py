"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.embedding import (
    ChgSpinEmbedding,
    DatasetEmbedding,
    EdgeDegreeEmbedding,
)
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend


class TestDatasetEmbedding:
    """Test the DatasetEmbedding class."""

    def test_embeddings_trainable_when_grad_true(self):
        """Test that embeddings have requires_grad=True when grad=True."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=True,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=True
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is True
                ), f"Expected requires_grad=True for dataset '{dataset}'"

    def test_embeddings_not_trainable_when_grad_false(self):
        """Test that embeddings have requires_grad=False when grad=False."""
        dataset_mapping = {"oc20": "oc20", "omat": "omat", "omol": "omol"}
        embedding_size = 64

        layer = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )

        # Check all embedding parameters have requires_grad=False
        for dataset in dataset_mapping:
            for param in layer.dataset_emb_dict[dataset].parameters():
                assert (
                    param.requires_grad is False
                ), f"Expected requires_grad=False for dataset '{dataset}'"

    def test_dataset_mapping(self):
        """Test that dataset_mapping correctly maps one dataset to another's embedding."""
        dataset_mapping = {
            "oc20_subset": "oc20",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        no_dataset_mapping = {
            "oc20_subset": "oc20_subset",
            "oc20": "oc20",
            "omat": "omat",
            "omol": "omol",
        }
        embedding_size = 64

        # Instance 1: no mapping
        torch.manual_seed(42)
        layer_no_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=no_dataset_mapping,
        )
        layer_no_mapping.eval()

        # Instance 2: with mapping
        torch.manual_seed(42)
        layer_with_mapping = DatasetEmbedding(
            embedding_size=embedding_size,
            enable_grad=False,
            dataset_mapping=dataset_mapping,
        )
        layer_with_mapping.eval()

        # Test 1: layer_with_mapping(["oc20_subset"]) == layer_with_mapping(["oc20"])
        # Both resolve to oc20's embedding when mapping is active
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_with_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as 'oc20'"

        # Test 2: layer_with_mapping(["oc20_subset"]) == layer_no_mapping(["oc20"])
        # With mapping, oc20_subset uses oc20's embedding
        assert torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20"])
        ), "With mapping, 'oc20_subset' should return same embedding as unmapped 'oc20'"

        # Test 3: layer_with_mapping(["oc20_subset"]) != layer_no_mapping(["oc20_subset"])
        # Without mapping, oc20_subset uses its own embedding
        assert not torch.allclose(
            layer_with_mapping(["oc20_subset"]), layer_no_mapping(["oc20_subset"])
        ), "With mapping, 'oc20_subset' should differ from unmapped 'oc20_subset'"

        # Test 4: oc20 should not equal omol or omat
        oc20_embedding = layer_with_mapping(["oc20"])
        omol_embedding = layer_with_mapping(["omol"])
        omat_embedding = layer_with_mapping(["omat"])

        assert not torch.allclose(
            oc20_embedding, omol_embedding
        ), "'oc20' should not equal 'omol' embedding"
        assert not torch.allclose(
            oc20_embedding, omat_embedding
        ), "'oc20' should not equal 'omat' embedding"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("target", ["charge", "spin"])
@pytest.mark.parametrize("input_dtype", [torch.long, torch.float64])
def test_linear_charge_spin_preserves_model_dtype_and_caller_input(
    dtype, target, input_dtype
):
    layer = ChgSpinEmbedding("lin_emb", target, 8, grad=False).to(dtype)
    owner = torch.tensor([0, 1, 0, 3], dtype=input_dtype)
    values = owner[::2]  # alias the caller's storage
    before = owner.clone()
    version = owner._version
    inputs = torch.where(values == 0, -100, values) if target == "spin" else values
    expected = F.linear(
        inputs.unsqueeze(-1).to(dtype), layer.lin_emb.weight, layer.lin_emb.bias
    )
    for _ in range(2):
        actual = layer(values)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(owner, before, rtol=0, atol=0)
        assert owner._version == version


@pytest.mark.parametrize("embedding_type", ["pos_emb", "lin_emb"])
def test_charge_spin_fp64_preserves_first_and_second_derivatives(embedding_type):
    layer = ChgSpinEmbedding(embedding_type, "spin", 8, grad=True).double()
    values = torch.tensor([0.0, 0.25, 1.5], dtype=torch.float64, requires_grad=True)

    def reference(x):
        if embedding_type == "lin_emb":
            mapped = torch.where(x == 0, -100, x)
            return F.linear(mapped[:, None], layer.lin_emb.weight, layer.lin_emb.bias)
        projected = x[:, None] * layer.W[None, :] * 2 * torch.pi
        result = torch.cat((projected.sin(), projected.cos()), dim=-1)
        # Original positional-embedding algorithm, including its null-token
        # derivative policy, is the numerical reference for the fixed mask.
        result[torch.where(x == 0)[0]] = 0
        return result

    variables = (values, *layer.parameters())
    results = []
    for fn in (layer, reference):
        output = fn(values)
        first = torch.autograd.grad(output.square().sum(), variables, create_graph=True)
        second = torch.autograd.grad(sum(t.square().sum() for t in first), variables)
        results.append((output, first, second))
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert values[0] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positional_null_spin_mask_does_not_extract_dynamic_indices(dtype):
    layer = ChgSpinEmbedding("pos_emb", "spin", 8, grad=False).to(dtype)
    values = torch.tensor([0, 1, 0, 3])
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        actual = layer(values)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual[[0, 2]], torch.zeros_like(actual[[0, 2]]))
    operators = {event.key for event in trace.key_averages()}
    assert "aten::nonzero" not in operators
    assert "aten::index_put_" not in operators


def _edge_degree_layer(chunk_size, dtype):
    return EdgeDegreeEmbedding(
        sphere_channels=4,
        lmax=2,
        mmax=2,
        edge_channels_list=[6, 8],
        rescale_factor=5.0,
        mappingReduced=CoefficientMapping(2, 2),
        activation_checkpoint_chunk_size=chunk_size,
        backend=ExecutionBackend(),
    ).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("trainable", [False, True])
def test_checkpointed_edge_degree_preserves_values_and_two_derivatives(
    dtype, trainable
):
    layer = _edge_degree_layer(3, dtype).requires_grad_(trainable)
    reference = copy.deepcopy(layer)
    x = torch.randn(5, 9, 4, dtype=dtype, requires_grad=True)
    edge = torch.randn(11, 6, dtype=dtype, requires_grad=True)
    wigner = torch.randn(11, 9, 3, dtype=dtype, requires_grad=True)
    target = torch.tensor([0, 1, 1, 4, 3, 1, 2, 0, 4, 4, 1])
    original = x.detach().clone()
    results = []
    for model in (layer, reference):
        variables = (x, edge, wigner, *(model.parameters() if trainable else ()))
        if model is reference:
            # Preserve the former chunking and GEMM shapes, changing only the
            # checkpoint accumulation algorithm under test.
            output = x
            for radial, destinations, rotation in zip(
                edge.split(3), target.split(3), wigner.split(3), strict=True
            ):
                output = torch.utils.checkpoint.checkpoint(
                    model.forward_chunk,
                    output,
                    radial,
                    destinations,
                    rotation,
                    use_reentrant=False,
                )
        else:
            output = model(x, edge, target, wigner)
        first = torch.autograd.grad(output.square().sum(), variables, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in first), variables
        )
        results.append((output, first, second))
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    for actual, expected in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(x, original, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_checkpointed_edge_degree_retains_one_node_buffer_across_chunks(dtype):
    layer = _edge_degree_layer(4, dtype).requires_grad_(False)
    x = torch.randn(64, 9, 4, dtype=dtype, requires_grad=True)
    edge = torch.randn(19, 6, dtype=dtype, requires_grad=True)
    wigner = torch.randn(19, 9, 3, dtype=dtype, requires_grad=True)
    target = torch.arange(19) % len(x)
    saved_node_storage = set()

    def pack(tensor):
        if tensor.shape == x.shape:
            saved_node_storage.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = layer(x, edge, target, wigner)
    # Five chunks must share the same zero-node input. The former chain saved
    # five distinct full node buffers, growing with the chunk count and dtype.
    assert len(saved_node_storage) == 1
    torch.autograd.grad(output.square().sum(), (x, edge, wigner))


@pytest.mark.parametrize("edges", [0, 1])
def test_checkpointed_edge_degree_handles_empty_and_single_edge_fp64(edges):
    layer = _edge_degree_layer(3, torch.float64).requires_grad_(False)
    x = torch.randn(5, 9, 4, dtype=torch.float64)
    edge = torch.randn(edges, 6, dtype=torch.float64)
    wigner = torch.randn(edges, 9, 3, dtype=torch.float64)
    target = torch.zeros(edges, dtype=torch.long)
    torch.testing.assert_close(
        layer(x, edge, target, wigner), layer.forward_chunk(x, edge, target, wigner)
    )


def test_fp64_uma_checkpointed_embedding_preserves_energy_forces_stress_and_hvp():
    from tests.core.units.mlip_unit.test_xpu_preparation import _data, _small_predictor

    model = _small_predictor().model.module.double().eval().requires_grad_(False)
    reference = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, EdgeDegreeEmbedding):
            module.activation_checkpoint_chunk_size = 2

    def evaluate(backbone):
        data = _data().to(dtype=torch.float64)
        data["_requested_properties"] = (
            "energy",
            "forces",
            "stress",
            "hessian_vector_product",
        )
        data["hessian_vector"] = torch.arange(9, dtype=torch.float64).reshape(3, 3) / 9
        torch.manual_seed(17)
        return backbone(data)

    actual, expected = evaluate(model), evaluate(reference)
    for name in ("energy", "forces", "stress", "hessian_vector_product"):
        key = f"omol_{name}"
        value = actual[key][name]
        assert value.dtype == torch.float64
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected[key][name], rtol=1e-10, atol=1e-11)

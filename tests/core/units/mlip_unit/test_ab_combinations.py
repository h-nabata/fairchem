"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from itertools import combinations, product

import hydra
import pytest
import torch

from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec
from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data

# Deterministic pairwise covering array, rather than the 23,040-case Cartesian
# product. Every pair of values in the ten selected axes occurs at least once.
# This does not establish all higher-order combinations or native GPU support.
_LEVELS = (5, 4, 3, 2, 3, 4, 2, 2, 2, 2)
_CASES = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, 2, 2, 0, 2, 2, 0, 0, 1, 1),
    (2, 3, 2, 1, 2, 3, 1, 1, 0, 0),
    (3, 0, 0, 1, 0, 2, 1, 1, 1, 1),
    (3, 1, 1, 0, 1, 3, 0, 0, 0, 0),
    (4, 2, 0, 0, 0, 1, 0, 1, 0, 0),
    (4, 3, 1, 1, 1, 0, 1, 0, 1, 1),
    (1, 2, 0, 1, 1, 3, 1, 1, 0, 1),
    (2, 3, 1, 0, 0, 2, 0, 0, 1, 0),
    (1, 0, 1, 0, 2, 1, 1, 0, 0, 0),
    (2, 1, 0, 1, 2, 0, 0, 1, 0, 1),
    (4, 0, 2, 0, 0, 3, 0, 0, 1, 0),
    (0, 1, 2, 0, 1, 2, 0, 0, 0, 0),
    (3, 2, 2, 0, 2, 0, 0, 0, 0, 0),
    (2, 0, 2, 0, 1, 1, 0, 0, 0, 0),
    (0, 2, 1, 0, 2, 3, 0, 0, 0, 0),
    (1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
    (3, 3, 0, 0, 0, 1, 0, 0, 0, 0),
    (4, 1, 0, 0, 2, 2, 0, 0, 0, 0),
    (0, 3, 0, 0, 0, 0, 0, 0, 0, 0),
    (1, 3, 0, 0, 0, 0, 0, 0, 0, 0),
    (2, 2, 0, 0, 0, 0, 0, 0, 0, 0),
)


def _combined_settings(row):
    gemm, weight, saved, reduction, radial, search, order, skin, padding, cache = row
    policy = SeriesBSettings(
        gemm_precision=("fp32", "bf16", "fp16", "bf16_split2", "bf16_split3")[gemm],
        low_precision_emulation=bool(gemm),
        weight_bits=(None, 4, 8, None)[weight],
        low_rank_fraction=0.5 if weight == 3 else None,
        saved_feature_dtype=(None, "bf16", "fp16")[saved],
        saved_feature_min_elements=1,
        compensated_reduction=bool(reduction),
        radial_fast_exp=radial == 1,
        radial_table_points=65 if radial == 2 else None,
    )
    return replace(
        inference_settings_xpu(),
        series_b=policy,
        graph_search_method=("cell", "all_pairs", "bvh", "auto")[search],
        graph_edge_order=("original", "spatial")[order],
        neighbor_skin=0.4 if skin else 0.0,
        edge_chunk_size=8 if padding else None,
        mole_cache_size=2 if cache else 0,
        hessian_chunk_size=3,
    )


def test_combination_matrix_covers_every_selected_pair():
    for left, right in combinations(range(len(_LEVELS)), 2):
        assert {(row[left], row[right]) for row in _CASES} == set(
            product(range(_LEVELS[left]), range(_LEVELS[right]))
        )


@pytest.mark.parametrize(
    "row", _CASES, ids=[f"pairwise-{i:02d}" for i in range(len(_CASES))]
)
def test_real_uma_pairwise_ab_reuse_derivatives_and_policy_isolation(row):
    settings = _combined_settings(row)
    candidate = _public_predictor()
    candidate.inference_settings = settings
    reference = copy.deepcopy(candidate)
    reference.inference_settings = replace(
        settings,
        neighbor_skin=0,
        graph_search_method="cell",
        graph_edge_order="original",
        edge_chunk_size=None,
    )
    restored = hydra.utils.instantiate(settings.to_omegaconf())
    assert restored == settings
    spec = ModelSpec("unused.pt", source="path", inference_settings=settings)
    assert spec.loader_settings() == settings
    data = _data()
    data.cell = torch.eye(3).unsqueeze(0) * 8
    data.hessian_vector = torch.arange(9, dtype=data.pos.dtype).reshape_as(data.pos) / 9
    original = data.clone()
    requested = ("energy", "forces", "stress", "hessian")
    expected = reference.predict(data, requested_properties=requested)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = candidate.predict(data, requested_properties=requested)
        assert torch.is_autocast_enabled("cpu")
    saved = {name: value.clone() for name, value in actual.items()}
    assert set(actual) == set(requested)
    for name in requested:
        assert torch.isfinite(actual[name]).all()
        assert actual[name].shape == expected[name].shape
        torch.testing.assert_close(actual[name], expected[name], rtol=3e-4, atol=3e-6)
    hvp = candidate.predict(data, requested_properties=("hessian_vector_product",))[
        "hessian_vector_product"
    ]
    torch.testing.assert_close(
        hvp.flatten(),
        actual["hessian"].reshape(9, 9) @ data.hessian_vector.flatten(),
        rtol=5e-4,
        atol=5e-6,
    )
    changed = data.clone()
    changed.pos[0, 0] += 0.02
    if settings.mole_cache_size:
        changed.charge += 1
    candidate.predict(changed, requested_properties=("energy", "forces"))
    repeated = candidate.predict(data, requested_properties=("energy", "forces"))
    for name in ("energy", "forces"):
        torch.testing.assert_close(repeated[name], saved[name], rtol=3e-4, atol=3e-6)
    for name in requested:
        torch.testing.assert_close(actual[name], saved[name], rtol=0, atol=0)
    for name in ("pos", "cell", "charge", "hessian_vector"):
        torch.testing.assert_close(data[name], original[name], rtol=0, atol=0)
    assert candidate.inference_settings == spec.loader_settings()


@pytest.mark.parametrize(
    "policy",
    [
        {"gemm_precision": "bf16"},
        {"gemm_precision": "fp16"},
        {"gemm_precision": "bf16_split2"},
        {"gemm_precision": "bf16_split3"},
        {"weight_bits": 4},
        {"weight_bits": 8},
        {"low_rank_fraction": 0.5},
        {"saved_feature_dtype": "bf16"},
        {"saved_feature_dtype": "fp16"},
        {"compensated_reduction": True},
        {"radial_fast_exp": True},
        {"radial_table_points": 65},
        {"onednn_tf32": True},
    ],
)
@pytest.mark.parametrize(
    "incompatible",
    [
        {"compile": True},
        {"xpu_graph": True},
        {"nitro": True},
        {"activation_checkpointing": True},
        {"execution_mode": "umas_fast_portable"},
    ],
)
def test_each_b_family_rejects_incompatible_a_backends(policy, incompatible):
    with pytest.raises(ValueError, match="Series B"):
        replace(inference_settings_xpu(), series_b=policy, **incompatible)

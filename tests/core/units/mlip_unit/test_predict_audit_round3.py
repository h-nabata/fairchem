"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.models.uma.nn.execution_backends import (
    ExecutionMode,
    get_execution_backend,
)
from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


@pytest.mark.parametrize(
    "mode",
    [
        ExecutionMode.GENERAL,
        ExecutionMode.UMAS_FAST_PYTORCH,
        ExecutionMode.UMAS_FAST_PORTABLE,
    ],
)
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_composition_fallback_reprepares_portable_model(mode, capture, override):
    candidate = _public_predictor()
    candidate.inference_settings.execution_mode = (
        ExecutionMode.GENERAL if override else mode
    )
    candidate._overrides = {"backbone": {"execution_mode": mode}} if override else {}
    candidate.inference_settings.xpu_graph = capture
    candidate.model.module.backbone.parent_kwargs["execution_mode"] = mode
    for module in candidate.model.modules():
        if hasattr(module, "backend"):
            module.backend = get_execution_backend(mode)
    reference = _public_predictor()
    reference.inference_settings.merge_mole = False
    pristine = copy.deepcopy(reference.model)

    # Substitute the checkpoint I/O boundary; the merged models, fallback,
    # preparation, real UMA forward and force differentiation run normally.
    def reload():
        candidate.model = copy.deepcopy(pristine)
        candidate.model.module.backbone.backend = get_execution_backend(
            candidate._overrides.get("backbone", {}).get(
                "execution_mode", candidate.inference_settings.execution_mode
            )
        )

    candidate._load_model = reload
    candidate.predict(_data(), requested_properties=("energy", "forces"))
    changed = _data()
    changed.charge += 1
    expected = reference.predict(changed, requested_properties=("energy", "forces"))
    actual = candidate.predict(changed, requested_properties=("energy", "forces"))

    assert candidate.inference_settings.execution_mode == ExecutionMode.GENERAL
    assert not candidate.inference_settings.merge_mole
    assert not candidate.inference_settings.compile
    assert not candidate.inference_settings.xpu_graph
    for name in ("energy", "forces"):
        torch.testing.assert_close(actual[name], expected[name], atol=2e-6, rtol=2e-5)


def test_named_preset_customization_cannot_contaminate_other_jobs():
    expected = guess_inference_settings("xpu")
    customized = guess_inference_settings("xpu")
    customized.tf32 = True
    customized.predict_untrained_hessian.add("omol")

    assert guess_inference_settings("xpu") == expected
    assert customized is not expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("merge_mole", False),
        ("base_precision_dtype", "float64"),
        ("execution_mode", "umas_fast_portable"),
        ("neighbor_skin", 0.5),
    ],
)
def test_prepared_settings_cannot_disable_checks_without_rebuilding(field, value):
    candidate = _public_predictor()
    candidate.predict(_data(), requested_properties=("energy", "forces"))
    setattr(candidate.inference_settings, field, value)
    changed = _data()
    changed.charge += 1
    with pytest.raises(ValueError, match="construct a new predictor"):
        candidate.predict(changed, requested_properties=("energy", "forces"))


def test_prepared_settings_allow_per_call_output_selection():
    candidate = _public_predictor()
    candidate.predict(_data(), requested_properties=("energy",))
    candidate.inference_settings.requested_properties = ("forces",)
    assert set(candidate.predict(_data())) == {"forces"}

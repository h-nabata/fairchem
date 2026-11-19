"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fairchem.core.regression_suite.execution import settings as settings_module
from fairchem.core.regression_suite.execution.settings import selected_suites
from fairchem.core.regression_suite.spec.profiles import (
    build_inference_settings,
    json_settings,
    profile_config,
    unsupported_suites_for_profile,
)
from fairchem.core.regression_suite.suites import load_suite

_PROFILES = sorted(
    path.stem
    for path in Path(settings_module.__file__).parents[1].glob("configs/profile/*.yaml")
)


@pytest.mark.parametrize("profile", _PROFILES)
def test_profiles_build_valid_settings_and_declare_unsupported_suites(profile):
    config = profile_config(profile)
    unsupported = unsupported_suites_for_profile(profile)

    assert config.label == profile.replace("_", "-")
    for suite in selected_suites(None, profile):
        overrides = load_suite(suite).inference_settings_overrides
        settings = build_inference_settings(config, overrides)
        json.dumps(json_settings(settings))
        fast = settings.execution_mode in ("umas_fast_gpu", "umas_fast_pytorch")
        # Fast backends reject activation checkpointing; umas_fast_gpu needs merging.
        assert not (fast and settings.activation_checkpointing)
        assert settings.merge_mole or settings.execution_mode != "umas_fast_gpu"
        # A merged MOLE model serves one composition, so structures cannot share it.
        assert (config.get("predictor_scope") == "structure") == settings.merge_mole
    elastic = build_inference_settings(
        config, load_suite("elastic").inference_settings_overrides
    )
    # Autograd Hessians need eager, vmap-capable backward operators.
    assert ("autograd-frequencies" in unsupported) == (
        elastic.compile or elastic.execution_mode == "umas_fast_gpu"
    )
    assert "finite-difference-frequencies" not in unsupported


def test_xpu_ordinal_and_cross_device_reference(monkeypatch):
    import torch

    from fairchem.core.regression_suite.cli import _parser
    from fairchem.core.regression_suite.execution.executors import LocalExecutor
    from fairchem.core.regression_suite.execution.settings import (
        DeviceUnavailableError,
        resolve_device,
    )

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)
    assert resolve_device("xpu", "local") == "xpu:1"
    assert resolve_device("xpu:0", "local") == "xpu:0"
    with pytest.raises(DeviceUnavailableError):
        resolve_device("xpu:2", "local")
    assert _parser().parse_args(["full", "xpu", "--device", "xpu:1"]).device == "xpu:1"
    worker = LocalExecutor(".", name="test", gpus_per_node=1, device="xpu:1")
    assert worker._slots(3) == [{}]


def test_cross_device_qualification_keeps_launch_identity_and_archived_settings(
    monkeypatch,
):
    from fairchem.core.regression_suite.reporting import report
    from fairchem.core.regression_suite.spec.schema import RegressionSchemaError

    # A pinned CUDA artifact predates the fork's additional precision options.
    historical = {"base_precision_dtype": "float64"}
    expected = {
        "predictor_scope": "suite",
        "inference_settings": {**historical, "compact_angular_weights": False},
    }
    archived = {
        "run_id": "verify",
        "git_commit": "upstream",
        "inference_settings": historical,
        "predictor_scope": "suite",
    }
    entry = {
        "plan_hash": "plan",
        "suite_contract_hash": "contract",
        "reference": {"reference_hash": "ref"},
        "profiles": {"fp64-eager": {"completion": {"requested_cases": 1}}},
        "verifications": {"fp64-eager": archived},
    }
    baseline = {
        "suites": {"elastic": entry},
        "seed": 42,
        "checkpoint": "test",
        "device": "cuda",
    }
    identity = {
        **expected,
        "suite": "elastic",
        "role": "comparison",
        "profile": "xpu-fp64",
        "plan_hash": "plan",
        "suite_contract_hash": "contract",
        "reference_hash": "ref",
        "checkpoint": "test",
        "seed": 42,
        "git_commit": "fork",
        "git_dirty": False,
        "device": "xpu:1",
        "execution_environment": {"device_type": "xpu"},
        "case_count": 1,
    }
    monkeypatch.setattr(report, "qualify_suite", lambda *_: {"status": "validated"})

    def qualify(data, **kwargs):
        return report._qualify_candidate(
            {"identity": data},
            suite="elastic",
            profile=data["profile"],
            role=data["role"],
            baseline=baseline,
            expectation=expected,
            **kwargs,
        )

    launch = {"device": "xpu:1", "git_commit": "fork"}
    assert qualify(identity, launch_identity=launch)["status"] == "validated"
    with pytest.raises(RegressionSchemaError, match="device"):
        qualify({**identity, "device": "xpu:0"}, launch_identity=launch)
    with pytest.raises(RegressionSchemaError, match="inference_settings"):
        qualify({**identity, "inference_settings": historical}, launch_identity=launch)
    assert (
        qualify(
            {
                **identity,
                **archived,
                "role": "verification",
                "profile": "fp64-eager",
                "device": "cuda",
                "execution_environment": {"device_type": "cuda"},
            }
        )["status"]
        == "validated"
    )


def test_runner_stops_before_cases_if_loading_changed_profile(monkeypatch, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from fairchem.core.regression_suite.execution import runner
    from fairchem.core.regression_suite.spec.schema import RegressionSchemaError

    settings = build_inference_settings(profile_config("fp64_eager"), {})
    identity = {
        "device": "cpu",
        "seed": 42,
        "scope": "full",
        "predictor_scope": "suite",
        "inference_settings": json_settings(settings),
    }
    monkeypatch.setattr(
        runner,
        "_prepare_run",
        lambda _: (
            None,
            settings,
            SimpleNamespace(cases=()),
            identity,
            tmp_path / "out",
        ),
    )
    monkeypatch.setattr(
        runner.pretrained_mlip,
        "get_predict_unit",
        lambda *a, **kw: SimpleNamespace(
            device="cpu", inference_settings=replace(settings, merge_mole=True)
        ),
    )
    monkeypatch.setattr(
        runner,
        "_records_for_cases",
        lambda *a: pytest.fail("profile fallback reached the benchmark"),
    )
    with pytest.raises(RegressionSchemaError, match="merge_mole"):
        runner.run_suite({"checkpoint": "synthetic"})
    assert not (tmp_path / "out").exists()

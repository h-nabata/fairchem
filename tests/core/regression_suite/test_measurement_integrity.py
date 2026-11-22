"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import dataclasses
import os
import zipfile

import numpy as np
import pytest

from fairchem.core.regression_suite.artifacts.storage import (
    load_reference,
    staged_directory,
    write_json_atomic,
    write_reference,
)
from fairchem.core.regression_suite.evaluation.comparison import compare_records
from fairchem.core.regression_suite.evaluation.decision import qualify_suite
from fairchem.core.regression_suite.reporting.report import _pinned_run, _run_sidecar
from fairchem.core.regression_suite.spec.hashing import file_sha256, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    HistogramSpec,
    MeasurementRecord,
    ObservableSpec,
    RegressionSchemaError,
)


def _record(case, values, comparison="absolute"):
    return MeasurementRecord(
        case,
        "identity",
        "energy",
        np.asarray(values),
        "eV",
        comparison,
        0.0,
        "c",
        False,
    )


def _compare(reference, candidate):
    spec = ObservableSpec("eV", reference[0].comparison, HistogramSpec(1e-9, 10))
    return compare_records(
        reference, candidate, {"energy": spec}, identity={"case_count": len(reference)}
    )["observables"]["energy"]


@pytest.mark.parametrize(
    "changes",
    [
        {"units": "meV"},
        {"floor": 1.0},
        {"report_only": True},
        {"cluster_id": "other"},
        {"comparison": "log_ratio"},
    ],
)
def test_pairing_rejects_mismatched_contract_and_cluster(changes):
    reference = _record("a", [1.0])
    with pytest.raises(
        RegressionSchemaError, match=next(iter(changes)).removesuffix("_id")
    ):
        _compare([reference], [dataclasses.replace(reference, **changes)])


def test_cluster_means_weight_values_in_unequal_records():
    reference = [_record("a", [1.0]), _record("b", [1.0] * 3)]
    summary = _compare(reference, [_record("a", [1.0]), _record("b", [5.0] * 3)])
    assert summary["population"] == {"clusters": 1, "cases": 2, "values": 4}
    assert summary["decision_clusters"] == [
        {"cluster_id": "c", "primary_mean": 3.0, "absolute_mean": 3.0}
    ]


@pytest.mark.parametrize("small_first", [True, False])
def test_extreme_finite_log_ratio(small_first):
    a, b = (1e-300, 1e300) if small_first else (1e300, 1e-300)
    summary = _compare([_record("a", a, "log_ratio")], [_record("a", b, "log_ratio")])
    assert summary["decision_clusters"][0]["primary_mean"] == pytest.approx(
        600 * np.log(10)
    )


def test_trailing_nonfinite_full_statistic_fails_qualification():
    measured = _compare([_record("a", [1.0])], [_record("a", [1.0])])
    measured["statistics"]["p95"] = np.nan
    limits = {
        tier: {"primary": {"median_absolute": 1.0, "p95": 1.0}}
        for tier in ("FP64", "FP32", "TF32")
    }
    baseline = {
        "energy": {"limits": limits, "comparison": "absolute", "report_only": False}
    }
    result = qualify_suite(
        {"observables": {"energy": measured}},
        {"observables": baseline},
        minimum_tier="TF32",
    )
    assert not result["passed"]
    assert result["limiting"]["statistic"] == "p95"


def test_empty_smoke_is_not_a_pass():
    with pytest.raises(RegressionSchemaError, match="decision observable"):
        qualify_suite({"observables": {}}, {"observables": {}}, smoke=True)


def test_measurements_must_fit_storage_precision():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("no extended-range floating point on this platform")
    with pytest.raises(RegressionSchemaError, match="finite"):
        _record("a", np.finfo(np.longdouble).max)


def test_nested_archive_reads_outer_values_once(tmp_path, monkeypatch):
    records = [_record(str(i), np.arange(150) + i) for i in range(12)]
    manifest = write_reference(tmp_path / "records", records, identity={})
    with zipfile.ZipFile(tmp_path / "sealed", "w", zipfile.ZIP_LZMA) as archive:
        for path in (tmp_path / "records").iterdir():
            archive.write(path, path.name)

    with zipfile.ZipFile(tmp_path / "sealed") as archive:
        original = archive.open
        reads = []

        def opened(name, *args, **kwargs):
            handle = original(name, *args, **kwargs)
            if name == "values.npz":
                reads.append(name)
                monkeypatch.setattr(
                    handle, "seek", lambda *_: pytest.fail("rewound outer NPZ member")
                )
            return handle

        monkeypatch.setattr(archive, "open", opened)
        _, restored = load_reference(
            zipfile.Path(archive), expected_hash=manifest["reference_hash"]
        )
    assert reads == ["values.npz"]
    for expected, actual in zip(sorted(records, key=lambda item: item.key), restored):
        np.testing.assert_array_equal(expected.values, actual.values)


def test_finite_mean_when_record_total_exceeds_storage_range():
    summary = _compare([_record("a", [0, 0])], [_record("a", [1.6e308, 1.6e308])])
    assert summary["decision_clusters"][0]["primary_mean"] == 1.6e308


def test_report_rejects_unrelated_performance_sidecar(tmp_path):
    run = {
        "run_id": "verification-run",
        "execution_environment": {"device_type": "xpu"},
    }
    payload = {"identity": run, "environment": run["execution_environment"]}
    path = tmp_path / "performance.json"
    write_json_atomic(path, {**payload, "sidecar_id": stable_hash(payload)})
    assert _run_sidecar(path, run)["identity"] == run
    with pytest.raises(RegressionSchemaError, match="run_id"):
        _run_sidecar(path, {**run, "run_id": "calibration-run"})


def test_report_rejects_replaced_calibration_fragment(tmp_path):
    path = tmp_path / "manifest.json"
    write_json_atomic(path, {"run": "original"})
    entry = {"fragment": path.name, "fragment_sha256": file_sha256(path)}
    assert _pinned_run(tmp_path, entry) == tmp_path
    write_json_atomic(path, {"run": "replacement"})
    with pytest.raises(RegressionSchemaError, match="pinned baseline"):
        _pinned_run(tmp_path, entry)


def test_failed_directory_swap_and_restore_preserve_previous_artifact(
    tmp_path, monkeypatch
):
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "original").write_text("data")
    rename = os.rename

    def fail_publication(source, destination):
        if destination == output:
            raise PermissionError("cannot publish or restore")
        rename(source, destination)

    monkeypatch.setattr(os, "rename", fail_publication)
    with pytest.raises(
        PermissionError, match="cannot publish or restore"
    ), staged_directory(output, replace=True) as stage:
        (stage / "replacement").write_text("new data")
    (backup,) = tmp_path.glob("*.previous")
    assert (backup / "original").read_text() == "data"
    assert not stage.exists()

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.constraints import FixedLine, FixedPlane, Hookean

from fairchem.core.regression_suite import cli
from fairchem.core.regression_suite.artifacts.baselines import baseline_provenance
from fairchem.core.regression_suite.artifacts.storage import (
    pack_baseline,
    write_json_atomic,
    write_reference,
)
from fairchem.core.regression_suite.evaluation.decision import qualify_suite
from fairchem.core.regression_suite.evaluation.performance import _incomparability
from fairchem.core.regression_suite.execution import (
    environment,
    executors,
    runner,
    smoke,
)
from fairchem.core.regression_suite.spec.hashing import stable_hash
from fairchem.core.regression_suite.spec.schema import (
    BASELINE_SCHEMA_VERSION,
    CommandError,
    HistogramSpec,
    MeasurementRecord,
    ObservableSpec,
    RegressionSchemaError,
)
from fairchem.core.regression_suite.spec.structures import atoms_hash
from fairchem.core.regression_suite.spec.transforms import apply_case_transform
from fairchem.core.scripts import benchmark_xpu_batch as benchmark


def test_fast_checks_tier_and_rejects_substituted_reference(tmp_path, monkeypatch):
    case = {"case_id": "a", "transform_id": "i"}
    identity = {
        "suite": "test",
        "seed": 1,
        "checkpoint": "u",
        "case_count": 1,
        "plan_hash": "p",
        "suite_contract_hash": "c",
    }

    def make_reference(name, value, **fields):
        record = MeasurementRecord(
            "a", "i", "energy", np.array([value]), "eV", "absolute", 0, "c", False
        )
        return write_reference(
            tmp_path / name, [record], identity={**identity, "run_id": name, **fields}
        )

    pinned = make_reference("reference", 0.0)
    make_reference("candidate", 0.5, scope="smoke", selected_cases=[case])
    limits = {
        tier: {"primary": {"median_absolute": limit, "p95": limit}}
        for tier, limit in (("FP64", 0.001), ("FP32", 0.01), ("TF32", 1.0))
    }
    reference = {
        "manifest": "reference/manifest.json",
        "values": "reference/values.npz",
        "reference_hash": pinned["reference_hash"],
    }
    baseline = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "seed": 1,
        "checkpoint": "u",
        "suites": {
            "test": {
                "reference": reference,
                "profiles": {},
                "verifications": {},
                "observables": {
                    "energy": {
                        "report_only": False,
                        "comparison": "absolute",
                        "limits": limits,
                    }
                },
            }
        },
    }
    write_json_atomic(
        tmp_path / "baseline.json", {**baseline, "baseline_id": stable_hash(baseline)}
    )
    sealed = pack_baseline(tmp_path, tmp_path / "valid.baseline")
    monkeypatch.setattr(
        smoke, "_run_smoke_cases", lambda *a, **kw: (tmp_path / "candidate", "xpu-fp64")
    )
    monkeypatch.setattr(
        smoke,
        "load_suite",
        lambda _: SimpleNamespace(
            observables={
                "energy": ObservableSpec("eV", "absolute", HistogramSpec(1e-9, 10))
            },
            group_values=lambda *_: {},
        ),
    )
    args = dict(
        suite="test",
        output_dir=tmp_path,
        baseline_dir=sealed,
        profile="xpu-fp64",
        device="cpu",
        checkpoint="u",
        checkpoint_id="u",
    )
    check = smoke._fast_suite(**args)
    assert not check["passed"]
    assert check["qualification"]["tightest_tier"] == "TF32"
    full = qualify_suite(
        check["summary"], baseline["suites"]["test"], minimum_tier="FP64"
    )
    assert full["status"] == "FAIL"
    assert full["classification"] == "PASS@TF32"
    assert full["limiting"]["tier"] == "FP64"

    assert smoke._fast_suite(**args, minimum_tier="TF32")["passed"]

    (tmp_path / "reference").rename(tmp_path / "original")
    make_reference("reference", 0.5)
    with pytest.raises(RegressionSchemaError, match="pinned baseline"):
        smoke._fast_suite(**{**args, "baseline_dir": tmp_path})
    with pytest.raises(RegressionSchemaError, match="pinned baseline"):
        pack_baseline(tmp_path, tmp_path / "changed.baseline")
    assert not (tmp_path / "changed.baseline").exists()


def test_checkpoint_identity_requires_matching_bytes_and_describes_legacy(tmp_path):
    checkpoint = tmp_path / "weights"
    checkpoint.write_bytes(b"a")
    baseline = {
        "checkpoint": "model",
        "checkpoint_sha256": runner.checkpoint_digest(str(checkpoint)),
    }
    assert runner.resolve_baseline_checkpoint(baseline, str(checkpoint), "model") == (
        str(checkpoint),
        "model",
    )
    checkpoint.write_bytes(b"b")
    with pytest.raises(RegressionSchemaError, match="SHA-256"):
        runner.resolve_baseline_checkpoint(baseline, str(checkpoint), "model")
    del baseline["checkpoint_sha256"]
    assert runner.resolve_baseline_checkpoint(baseline, str(checkpoint), "model") == (
        str(checkpoint),
        "model",
    )
    provenance = baseline_provenance(
        {
            **baseline,
            "baseline_id": "id",
            "campaign": "test",
            "device": "cpu",
            "scope": "full",
            "suites": {},
        },
        tmp_path,
    )
    assert provenance["checkpoint_sha256"] is None
    assert provenance["checkpoint_verification"] == "name_only"


@pytest.mark.parametrize("failure", ["offset", "order", "shape", "nan"])
def test_batch_cli_saves_failed_parity_and_omits_throughput(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(benchmark, "resolve_xpu_device", lambda d: d)

    def evaluate(
        checkpoint, device, structures, task, warmup, repeats, energy_atol, force_atol
    ):
        assert len({len(atoms) for atoms in structures}) > 1
        serial = [
            {"energy": torch.tensor([float(i)]), "forces": torch.zeros(len(s), 3)}
            for i, s in enumerate(structures)
        ]
        packed = {k: torch.cat([s[k] for s in serial]) for k in ("energy", "forces")}
        assert benchmark._parity(
            serial, packed, [len(a) for a in structures], energy_atol, force_atol
        )["passed"]
        if failure == "offset":
            packed["forces"] += 10
        elif failure == "order":
            packed["energy"] = packed["energy"].flip(0)
        elif failure == "shape":
            packed["forces"] = packed["forces"].flatten()
        else:
            packed["energy"][0] = float("nan")
        return {
            "parity": benchmark._parity(
                serial, packed, [len(a) for a in structures], energy_atol, force_atol
            )
        }

    monkeypatch.setattr(benchmark, "_worker", evaluate)
    out = tmp_path / "results.json"
    assert benchmark.main(["--checkpoint", "weights", "--output", str(out)]) == 1
    report = json.loads(out.read_text())
    assert not report["passed"]
    assert "systems_per_second" not in report
    assert report["workers"][0]["parity"]["errors"]


def test_batch_worker_stops_before_timing_failed_predictions(monkeypatch):
    class Predictor:
        def validate_atoms_data(self, atoms, task):
            pass

        def predict(self, data, **kwargs):
            counts = data if isinstance(data, list) else [data]
            return {
                "energy": torch.full(
                    (len(counts),), 100.0 if isinstance(data, list) else 0.0
                ),
                "forces": torch.zeros(sum(counts), 3),
            }

    monkeypatch.setattr(benchmark, "resolve_xpu_device", lambda device: device)
    monkeypatch.setattr(
        benchmark.AtomicData, "from_ase", lambda atoms, **kw: len(atoms)
    )
    monkeypatch.setattr(benchmark, "atomicdata_list_to_batch", list)
    monkeypatch.setattr(
        benchmark.pretrained_mlip, "get_predict_unit", lambda *args, **kw: Predictor()
    )
    monkeypatch.setattr(torch.xpu, "get_device_name", lambda _: "synthetic")
    monkeypatch.setattr(
        torch.xpu, "synchronize", lambda _: pytest.fail("timed invalid predictions")
    )
    result = benchmark._worker(
        "test", "xpu:0", [Atoms("Cu"), Atoms("Al2")], "omat", 1, 1
    )
    assert not result["parity"]["passed"]
    assert "packed" not in result
    assert result["max_packed_vs_serial_difference"] == {"energy": 100.0, "forces": 0.0}


@pytest.mark.parametrize("constraint", [FixedLine, FixedPlane])
def test_rotation_round_trip_preserves_vector_conditions(constraint):
    atoms = Atoms(
        "H",
        positions=[[1.0, 0, 0]],
        momenta=[[2.0, 0, 0]],
        constraint=constraint(0, [1.0, 0, 0]),
    )
    matrix = np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])

    def rotate(a, m):
        return apply_case_transform(
            a, SimpleNamespace(transform=SimpleNamespace(matrix=m))
        ).transformed_atoms

    transformed = rotate(atoms, matrix)
    np.testing.assert_allclose(transformed.get_momenta(), [[0, 2, 0]])
    np.testing.assert_allclose(transformed.constraints[0].dir, [0, 1, 0])
    restored = rotate(transformed, matrix.T)
    for a, b in (
        (atoms.positions, restored.positions),
        (atoms.get_momenta(), restored.get_momenta()),
        (atoms.constraints[0].dir, restored.constraints[0].dir),
    ):
        np.testing.assert_allclose(a, b)
    atoms.set_constraint(Hookean(0, (0.0, 0, 0), k=1, rt=1))
    with pytest.raises(RegressionSchemaError, match="Hookean"):
        rotate(atoms, matrix)


def test_isotopes_affect_mass_dependent_plan_only():
    h = Atoms("H")
    d = h.copy()
    d.set_masses([2.014])
    assert atoms_hash(h) == atoms_hash(d)
    assert atoms_hash(h, mass_dependent=True) == atoms_hash(h)
    assert atoms_hash(h, mass_dependent=True) != atoms_hash(d, mass_dependent=True)


@pytest.mark.parametrize("seed", [True, False])
def test_node_local_caches_with_optional_cuda_seed(tmp_path, monkeypatch, seed):
    for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    seeds = tmp_path / "seeds"
    monkeypatch.setenv("FAIRCHEM_TRITON_SEED_CACHE_DIR", str(seeds))
    binary = "cuda_utils" + sysconfig.get_config_var("EXT_SUFFIX")
    if seed:
        (seeds / "target").mkdir(parents=True)
        (seeds / "target" / binary).write_bytes(b"cache")
    environment.prepare_compilation_cache()
    triton = Path(os.environ["TRITON_CACHE_DIR"])
    assert triton.is_dir()
    assert triton.is_relative_to(tmp_path)
    assert (triton / "target" / binary).is_file() == seed
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(explicit))
    environment.prepare_compilation_cache()
    assert explicit.is_dir()
    assert not list(explicit.iterdir())


def test_parallel_runs_get_independent_paths(tmp_path):
    def persist(index):
        destination = tmp_path / cli._timestamp()
        destination.mkdir()
        write_json_atomic(destination / "result.json", {"index": index})
        return destination

    with ThreadPoolExecutor(4) as pool:
        paths = list(pool.map(persist, range(16)))
    assert len(set(paths)) == len(paths)
    for index, path in enumerate(paths):
        assert json.loads((path / "result.json").read_text()) == {"index": index}


def test_cpu_model_and_thread_changes_reject_performance_comparison():
    env = environment.execution_environment("cpu")
    assert env["cpu_count"] > 0
    assert env["torch_num_threads"] > 0
    run = {"identity": {}, "environment": env}
    for field in ("cpu_model", "torch_num_threads", "cpu_count"):
        assert field in _incomparability(
            run, {**run, "environment": {**env, field: "changed"}}, "test"
        )


@pytest.mark.parametrize("failure", ["write", "spawn"])
def test_local_executor_propagates_worker_errors(tmp_path, monkeypatch, failure):
    if failure == "write":
        (tmp_path / "job_0.pkl").mkdir()
    else:

        def failed_start(*args, **kwargs):
            raise OSError("spawn failed")

        monkeypatch.setattr(executors.subprocess, "run", failed_start)
    executor = executors.LocalExecutor(tmp_path, name="job", gpus_per_node=0)
    with pytest.raises(CommandError, match=r"unfinished task indices \[0, 1\]"):
        executor.map_array(str, [1, 2])

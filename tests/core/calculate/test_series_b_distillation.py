"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch
from ase import Atoms

from fairchem.core.calculate.series_b_distillation import (
    DistillationDataset,
    DistillationResult,
    DistillationSample,
    DistillationWeights,
    PropertyOutputAdapter,
    collect_teacher_labels,
    distill_student,
    export_distilled_checkpoint,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.utils import load_inference_model


class HarmonicStudent(torch.nn.Module):
    def __init__(self, coefficient=0.5):
        super().__init__()
        self.coefficient = torch.nn.Parameter(torch.tensor(float(coefficient)))

    def forward(self, data):
        pos = data.pos.requires_grad_(True)
        energy = (self.coefficient * pos.square().sum()).reshape(1)
        force = -torch.autograd.grad(energy.sum(), pos, create_graph=self.training)[0]
        stress = (
            self.coefficient * (pos.T @ pos) / torch.linalg.det(data.cell[0])
        ).unsqueeze(0)
        return {"energy": energy, "forces": force, "stress": stress}


class Teacher:
    def __init__(self, model):
        self.model = model.eval()

    def predict(self, data, requested_properties=None):
        with torch.enable_grad():
            output = self.model(data)
        return {name: output[name] for name in requested_properties}


def _data(scale=1.0):
    atoms = Atoms(
        "HH",
        positions=[[0.3 * scale, 0.2, 0.1], [-0.4, 0.1 * scale, 0.2]],
        cell=[3, 3, 3],
    )
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(atoms, task_name="omol")


@pytest.mark.parametrize("force_only", [False, True])
def test_real_student_training_reduces_loss_and_preserves_caller_state(force_only):
    student = HarmonicStudent()
    teacher = Teacher(HarmonicStudent(2.0))
    structures = [_data(0.8), _data(1.2)]
    positions = [data.pos.clone() for data in structures]
    dataset = collect_teacher_labels(
        teacher,
        structures,
        teacher_id="analytic-k2",
        properties=("energy", "forces", "stress"),
    )
    rng = torch.get_rng_state().clone()
    weights = DistillationWeights(
        energy=0 if force_only else 1, forces=1, stress=0 if force_only else 1
    )
    result = distill_student(
        student, dataset, steps=100, learning_rate=0.05, weights=weights, seed=9
    )
    assert result.loss_history[-1] < result.loss_history[0] / 1000
    assert result.model.coefficient.item() == pytest.approx(2.0, abs=0.015)
    assert student.coefficient.item() == 0.5
    assert teacher.model.coefficient.item() == 2.0
    assert not teacher.model.training
    assert student.training
    assert not result.model.training
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    for data, reference in zip(structures, positions):
        torch.testing.assert_close(data.pos, reference)
        assert not data.pos.requires_grad
    repeat = distill_student(
        student, dataset, steps=100, learning_rate=0.05, weights=weights, seed=9
    )
    assert repeat.loss_history == result.loss_history


def test_labels_own_inputs_and_outputs_and_canonicalize_stress():
    data = _data()
    output = Teacher(HarmonicStudent(2)).predict(
        data.clone(), ("energy", "forces", "stress")
    )
    output["stress"] = output["stress"].reshape(1, 9)
    sample = DistillationSample(data, output)
    data.pos.add_(10)
    with torch.no_grad():
        output["forces"].zero_()
    assert sample.data.pos.abs().max() < 1
    assert sample.targets["forces"].abs().sum() > 0
    assert sample.targets["stress"].shape == (1, 3, 3)
    assert not any(value.requires_grad for value in sample.targets.values())
    dataset = DistillationDataset((sample,))
    sample.targets["forces"].zero_()
    assert dataset.samples[0].targets["forces"].abs().sum() > 0


def test_explicit_hydra_adapter_preserves_units_and_derivative_scaling():
    data = _data()
    model = HarmonicStudent()
    raw = model(data)
    nested = {f"omol_{name}": {name: value} for name, value in raw.items()}
    adapter = PropertyOutputAdapter(
        {name: (f"omol_{name}", name) for name in raw},
        energy_scale=2,
        energy_offset=3,
        element_references={1: 4},
    )
    actual = adapter(nested, data)
    torch.testing.assert_close(actual["energy"], raw["energy"] * 2 + 11)
    for name in ("forces", "stress"):
        torch.testing.assert_close(actual[name], raw[name] * 2)
    derivative = -torch.autograd.grad(actual["energy"].sum(), data.pos)[0]
    torch.testing.assert_close(derivative, actual["forces"])
    with pytest.raises(ValueError, match="Missing element"):
        PropertyOutputAdapter({"energy": ("energy",)}, element_references={8: 1})(
            raw, data
        )


def test_detached_derivative_is_rejected_without_mutating_student_or_rng():
    student = HarmonicStudent()
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [_data()], teacher_id="teacher"
    )
    rng = torch.get_rng_state().clone()

    def detached(output, data):
        return {
            name: value.detach() if name == "forces" else value
            for name, value in output.items()
        }

    with pytest.raises(ValueError, match="detached"):
        distill_student(student, dataset, output_adapter=detached)
    assert student.coefficient.item() == 0.5
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)


@pytest.mark.parametrize(
    "targets",
    [
        {"energy": torch.tensor([float("nan")])},
        {"energy": torch.ones(2)},
        {"forces": torch.ones(1, 3)},
        {"stress": torch.ones(1, 6)},
        {"unknown": torch.ones(1)},
        {},
    ],
)
def test_invalid_label_data_cannot_broadcast_into_loss(targets):
    with pytest.raises(ValueError):
        DistillationSample(_data(), targets)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 0},
        {"seed": -1},
        {"learning_rate": float("nan")},
        {"learning_rate": -1},
        {"weights": {"forces": 1}},
    ],
)
def test_training_settings_are_validated(kwargs):
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)), [_data()], teacher_id="teacher"
    )
    with pytest.raises((ValueError, TypeError)):
        distill_student(HarmonicStudent(), dataset, **kwargs)


def test_training_requires_available_labels_and_trainable_parameters():
    dataset = collect_teacher_labels(
        Teacher(HarmonicStudent(2)),
        [_data()],
        teacher_id="teacher",
        properties=("energy",),
    )
    with pytest.raises(ValueError, match="positively weighted"):
        distill_student(HarmonicStudent(), dataset)
    with pytest.raises(ValueError, match="no trainable"):
        distill_student(
            HarmonicStudent().requires_grad_(False),
            dataset,
            weights=DistillationWeights(forces=0),
        )
    with pytest.raises(ValueError, match="At least one"):
        DistillationWeights(0, 0, 0)
    with pytest.raises(ValueError, match="nonempty"):
        collect_teacher_labels(Teacher(HarmonicStudent(2)), [], teacher_id="teacher")


def test_export_strict_load_and_provenance_roundtrip(tmp_path):
    model = torch.nn.Linear(3, 1)
    result = DistillationResult(
        model, (1.0, 0.5), {"teacher_id": "teacher", "validation_status": "unvalidated"}
    )
    config = {"_target_": "torch.nn.Linear", "in_features": 3, "out_features": 1}
    path = tmp_path / "student.pt"
    rng = torch.get_rng_state().clone()
    export_distilled_checkpoint(
        result,
        path,
        model_config=config,
        tasks_config=[],
        architecture_note="Independent small three-feature linear student",
    )
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    loaded, checkpoint = load_inference_model(str(path), strict=True)
    values = torch.tensor([[1.0, -2.0, 3.0]])
    torch.testing.assert_close(loaded(values), model(values))
    assert checkpoint.distillation_metadata["teacher_id"] == "teacher"
    assert checkpoint.distillation_metadata["validation_status"] == "unvalidated"
    ema, _ = load_inference_model(str(path), strict=True, use_ema=True)
    torch.testing.assert_close(ema(values), model(values))
    with pytest.raises(FileExistsError):
        export_distilled_checkpoint(
            result, path, model_config=config, tasks_config=[], architecture_note="same"
        )
    wrong = {**config, "in_features": 4}
    with pytest.raises(RuntimeError, match="size mismatch"):
        export_distilled_checkpoint(
            result,
            tmp_path / "bad.pt",
            model_config=wrong,
            tasks_config=[],
            architecture_note="wrong dimensions",
        )
    assert not (tmp_path / "bad.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_real_uma_student_force_training_step():
    from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor

    student = _small_predictor().model.module
    teacher = copy.deepcopy(student).eval()
    data = _data()
    data["_requested_properties"] = ("energy", "forces")
    output = teacher(data.clone())
    adapter = PropertyOutputAdapter(
        {name: (f"omol_{name}", name) for name in ("energy", "forces")}
    )
    targets = adapter(output, data)
    targets["energy"] = targets["energy"] + 0.01
    targets["forces"] = targets["forces"] + 0.001
    dataset = DistillationDataset((DistillationSample(data, targets),))
    result = distill_student(
        student, dataset, steps=2, learning_rate=1e-5, output_adapter=adapter
    )
    assert all(torch.isfinite(torch.tensor(result.loss_history)))
    assert any(
        not torch.equal(before, after)
        for before, after in zip(student.parameters(), result.model.parameters())
    )

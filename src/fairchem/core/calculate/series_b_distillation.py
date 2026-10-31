"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

import hydra
import torch

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


_TRAINING_RNG_LOCK = RLock()
_PROPERTIES = frozenset({"energy", "forces", "stress"})


def _positive_number(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or (value < 0 if zero else value <= 0)
    ):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )
    return float(value)


def _property_tensor(value, name, data):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a real floating tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")
    systems, atoms = data.natoms.numel(), data.pos.shape[0]
    if name == "energy":
        if value.shape not in ((systems,), (systems, 1)):
            raise ValueError(f"energy must have shape ({systems},) or ({systems}, 1)")
        return value.reshape(systems)
    if name == "forces" and value.shape == (atoms, 3):
        return value
    if name == "stress" and value.shape in ((systems, 9), (systems, 3, 3)):
        return value.reshape(systems, 3, 3)
    raise ValueError(f"Invalid {name} shape {tuple(value.shape)} for this AtomicData")


def _owned_data(data):
    if not isinstance(data, AtomicData):
        raise TypeError("Distillation inputs must be AtomicData")
    owned = data.clone().apply(lambda tensor: tensor.detach()).cpu()
    if owned.pos.numel() == 0 or not torch.isfinite(owned.pos).all():
        raise ValueError("Distillation inputs require finite, nonempty positions")
    if not torch.isfinite(owned.cell).all():
        raise ValueError("Distillation inputs require finite cells")
    return owned


@dataclass(frozen=True)
class DistillationWeights:
    """
    Explicit relative weights for mean squared errors in physical units.

    These are not dimensionless universal defaults: choose weights and held-out
    error limits appropriate to the chemical domain and property units.
    """

    energy: float = 1.0
    forces: float = 1.0
    stress: float = 0.0

    def __post_init__(self):
        for name in _PROPERTIES:
            object.__setattr__(
                self, name, _positive_number(getattr(self, name), name, zero=True)
            )
        if not any(getattr(self, name) for name in _PROPERTIES):
            raise ValueError("At least one distillation loss weight must be positive")

    @property
    def active(self):
        return tuple(
            name for name in ("energy", "forces", "stress") if getattr(self, name)
        )


@dataclass(frozen=True)
class DistillationSample:
    """
    Owned CPU inputs and physical-unit labels; no teacher autograd graph is kept.
    """

    data: AtomicData
    targets: dict[str, torch.Tensor]

    def __post_init__(self):
        data = _owned_data(self.data)
        if (
            not isinstance(self.targets, Mapping)
            or not self.targets
            or set(self.targets) - _PROPERTIES
        ):
            raise ValueError("targets must be a nonempty energy/forces/stress mapping")
        targets = {
            name: _property_tensor(value, name, data).detach().cpu().clone()
            for name, value in self.targets.items()
        }
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True)
class DistillationDataset:
    samples: tuple[DistillationSample, ...]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        samples = tuple(self.samples)
        if not samples or any(
            not isinstance(sample, DistillationSample) for sample in samples
        ):
            raise ValueError("A nonempty sequence of DistillationSample is required")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        object.__setattr__(
            self,
            "samples",
            tuple(
                DistillationSample(sample.data, sample.targets) for sample in samples
            ),
        )
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))


def collect_teacher_labels(
    teacher,
    structures: Iterable[AtomicData],
    *,
    teacher_id: str,
    properties: tuple[str, ...] = ("energy", "forces"),
) -> DistillationDataset:
    """
    Evaluate an explicit predictor to build an owned physical-unit dataset.

    ``teacher.predict`` must follow MLIPPredictUnit's property-dictionary API.
    The teacher's normal lazy preparation/cache may run; its weights, precision
    settings and training mode are not changed by this helper. Include domain-
    relevant TS, compressed, and nonequilibrium structures in the supplied data.
    A finite training set is not evidence of transferability to unseen systems.
    """
    if not isinstance(teacher_id, str) or not teacher_id.strip():
        raise ValueError("teacher_id must identify the teacher/checkpoint")
    if (
        isinstance(properties, str)
        or not properties
        or set(properties) - _PROPERTIES
        or len(set(properties)) != len(properties)
    ):
        raise ValueError("properties must be unique energy/forces/stress names")
    samples = []
    for structure in structures:
        data = _owned_data(structure)
        output = teacher.predict(data.clone(), requested_properties=tuple(properties))
        if not isinstance(output, Mapping) or any(
            name not in output for name in properties
        ):
            raise ValueError("Teacher did not return all requested physical properties")
        samples.append(
            DistillationSample(data, {name: output[name] for name in properties})
        )
    return DistillationDataset(
        tuple(samples),
        {
            "teacher_id": teacher_id,
            "properties": list(properties),
            "label_units": "teacher predictor physical units",
        },
    )


@dataclass(frozen=True)
class PropertyOutputAdapter:
    """
    Explicitly map nested student outputs to physical E/F/stress units.

    ``paths`` gives dictionary keys, e.g. ``{'energy': ('omol_energy', 'energy')}``.
    A shared energy scale also scales forces/stress, preserving derivative units.
    The energy offset is per system. Optional per-element energy references are
    added only to energy. This does not guess checkpoint task normalization.
    """

    paths: dict[str, tuple[str, ...]]
    energy_scale: float = 1.0
    energy_offset: float = 0.0
    element_references: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not isinstance(self.paths, Mapping)
            or not self.paths
            or set(self.paths) - _PROPERTIES
        ):
            raise ValueError("paths must map physical property names to key sequences")
        paths = {}
        for name, path in self.paths.items():
            if (
                isinstance(path, str)
                or not path
                or any(not isinstance(key, str) or not key for key in path)
            ):
                raise ValueError("Each output path must be a nonempty sequence of keys")
            paths[name] = tuple(path)
        object.__setattr__(self, "paths", paths)
        object.__setattr__(
            self, "energy_scale", _positive_number(self.energy_scale, "energy_scale")
        )
        if (
            isinstance(self.energy_offset, bool)
            or not isinstance(self.energy_offset, Real)
            or not math.isfinite(self.energy_offset)
        ):
            raise ValueError("energy_offset must be finite")
        references = dict(self.element_references)
        for number, reference in references.items():
            if (
                isinstance(number, bool)
                or not isinstance(number, Integral)
                or not 1 <= number <= 118
            ):
                raise ValueError(
                    "Element references require atomic numbers in [1, 118]"
                )
            if (
                isinstance(reference, bool)
                or not isinstance(reference, Real)
                or not math.isfinite(reference)
            ):
                raise ValueError("Element references must be finite")
        object.__setattr__(self, "element_references", references)

    def __call__(self, output: Mapping, data: AtomicData) -> dict[str, torch.Tensor]:
        result = {}
        for name, path in self.paths.items():
            value = output
            for key in path:
                value = value[key]
            value = _property_tensor(value, name, data) * self.energy_scale
            if name == "energy":
                value = value + self.energy_offset
                if self.element_references:
                    numbers = data.atomic_numbers.detach().cpu().tolist()
                    missing = set(numbers) - self.element_references.keys()
                    if missing:
                        raise ValueError(
                            f"Missing element references for {sorted(missing)}"
                        )
                    atomic = value.new_tensor(
                        [self.element_references[number] for number in numbers]
                    )
                    value = value + torch.zeros_like(value).index_add(
                        0, data.batch, atomic
                    )
            result[name] = value
        return result


@dataclass
class DistillationResult:
    model: torch.nn.Module
    loss_history: tuple[float, ...]
    provenance: dict


def distill_student(
    student: torch.nn.Module,
    dataset: DistillationDataset,
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    weights: DistillationWeights | None = None,
    output_adapter: Callable | None = None,
    seed: int = 0,
) -> DistillationResult:
    """
    Train an independent student copy with differentiable E/F/stress losses.

    B09 uses an explicitly supplied student. For B10, construct a new model with
    the desired cutoff/channels/lmax and train it here; no checkpoint dimensions
    or neighbour limits are silently truncated. ``student(data)`` must return
    physical property tensors, or supply an explicit ``output_adapter``.

    Inputs are cloned each step because conservative heads mutate coordinates.
    Training mode retains force/stress derivative graphs for weight gradients.
    Seeded sampling and the selected device's PyTorch RNG are scoped/restored;
    unrelated threads using global random generators need caller coordination.
    The returned model is in evaluation mode; its weights remain trainable.
    Loss history is training loss, not a held-out accuracy or speed guarantee.
    """
    if not isinstance(student, torch.nn.Module):
        raise TypeError("student must be a torch.nn.Module")
    if not isinstance(dataset, DistillationDataset):
        raise TypeError("dataset must be a DistillationDataset")
    for name, value in (("steps", steps), ("seed", seed)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < (1 if name == "steps" else 0)
        ):
            raise ValueError(
                f"{name} must be a {'positive' if name == 'steps' else 'nonnegative'} integer"
            )
    learning_rate = _positive_number(learning_rate, "learning_rate")
    weights = DistillationWeights() if weights is None else weights
    if not isinstance(weights, DistillationWeights):
        raise TypeError("weights must be DistillationWeights")
    # Revalidate/copy so a caller's post-construction edits cannot inject bad
    # values midway through a job or alias the training examples.
    dataset = DistillationDataset(dataset.samples, dataset.provenance)
    if any(set(weights.active) - sample.targets.keys() for sample in dataset.samples):
        raise ValueError("Every sample must contain all positively weighted properties")
    model = copy.deepcopy(student)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError(
            "Student has no trainable parameters; construct an unfrozen student"
        )
    device, dtype = parameters[0].device, parameters[0].dtype
    if dtype not in (torch.float32, torch.float64) or any(
        parameter.device != device or parameter.dtype != dtype
        for parameter in parameters
    ):
        raise ValueError(
            "Student training requires one device and FP32 or FP64 parameters"
        )
    if device.type not in {"cpu", "cuda", "xpu"}:
        raise ValueError("Student training supports CPU, CUDA, or XPU")
    model.train()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    devices = [] if device.type == "cpu" else [device.index or 0]
    rng_backend = "cuda" if device.type == "cpu" else device.type
    history = []
    with _TRAINING_RNG_LOCK, torch.random.fork_rng(
        devices=devices, device_type=rng_backend
    ), torch.enable_grad(), torch.autocast(device.type, enabled=False):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            getattr(torch, device.type).default_generators[devices[0]].manual_seed(
                int(seed)
            )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        for _ in range(int(steps)):
            sample = dataset.samples[
                int(torch.randint(len(dataset.samples), (), generator=generator))
            ]
            data = sample.data.clone().to(device, dtype=dtype)
            data["_requested_properties"] = weights.active
            data.pos.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            if output_adapter is not None:
                output = output_adapter(output, data)
            if not isinstance(output, Mapping):
                raise TypeError("Student adapter must return a property mapping")
            loss = torch.zeros((), device=device, dtype=dtype)
            for name in weights.active:
                if name not in output:
                    raise ValueError(f"Student did not return {name}")
                prediction = _property_tensor(output[name], name, data)
                if not prediction.requires_grad:
                    raise ValueError(
                        f"Student {name} is detached; derivative training needs create_graph=True"
                    )
                target = sample.targets[name].to(device=device, dtype=prediction.dtype)
                if prediction.shape != target.shape:
                    raise ValueError(f"Student/teacher {name} shapes differ")
                loss = (
                    loss
                    + getattr(weights, name) * (prediction - target).square().mean()
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite distillation loss")
            loss.backward()
            gradients = [
                parameter.grad for parameter in parameters if parameter.grad is not None
            ]
            if not gradients or any(
                not torch.isfinite(gradient).all() for gradient in gradients
            ):
                raise ValueError("Missing or nonfinite student parameter gradients")
            optimizer.step()
            if any(not torch.isfinite(parameter).all() for parameter in parameters):
                raise ValueError("Nonfinite student parameters after optimizer step")
            history.append(float(loss.detach().cpu()))
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    provenance = copy.deepcopy(dataset.provenance)
    provenance.update(
        {
            "method": "series_b_domain_distillation",
            "steps": int(steps),
            "seed": int(seed),
            "learning_rate": learning_rate,
            "loss_weights": {name: getattr(weights, name) for name in _PROPERTIES},
            "training_samples": len(dataset.samples),
            "validation_status": "unvalidated: evaluate held-out E/F/stress and dynamics before deployment",
        }
    )
    return DistillationResult(model, tuple(history), provenance)


def export_distilled_checkpoint(
    result: DistillationResult,
    path: str | Path,
    *,
    model_config: dict,
    tasks_config: list,
    architecture_note: str,
    overwrite: bool = False,
) -> Path:
    """
    Export a standard checkpoint only after strict architecture/state matching.

    Configurations and task normalizers must describe the trained student's
    actual output convention. This validates serialization/shape compatibility,
    not physical accuracy. Construct the resulting predictor with this checkpoint
    path explicitly; production UMA checkpoints are never rewritten here.
    """
    if not isinstance(result, DistillationResult):
        raise TypeError("result must be DistillationResult")
    if not isinstance(architecture_note, str) or not architecture_note.strip():
        raise ValueError("architecture_note must describe the trained architecture")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if not isinstance(model_config, Mapping) or "_target_" not in model_config:
        raise ValueError("model_config must be an explicit Hydra target config")
    if not isinstance(tasks_config, list):
        raise TypeError("tasks_config must be a list of task configurations")
    state = {
        name: value.detach().cpu().clone()
        for name, value in result.model.state_dict().items()
    }
    if any(
        value.is_floating_point() and not torch.isfinite(value).all()
        for value in state.values()
    ):
        raise ValueError("Cannot export nonfinite student state")
    config, tasks = copy.deepcopy(dict(model_config)), copy.deepcopy(tasks_config)
    with _TRAINING_RNG_LOCK, torch.random.fork_rng(devices=[]):
        reconstructed = hydra.utils.instantiate(config)
        reconstructed.load_state_dict(state, strict=True)
        if hasattr(reconstructed, "setup_tasks"):
            reconstructed.setup_tasks(tasks)
    checkpoint = MLIPInferenceCheckpoint(
        model_config=config,
        model_state_dict=state,
        ema_state_dict={
            "n_averaged": torch.tensor(0),
            **{f"module.{name}": value.clone() for name, value in state.items()},
        },
        tasks_config=tasks,
    )
    checkpoint.distillation_metadata = {
        **copy.deepcopy(result.provenance),
        "architecture_note": architecture_note,
    }
    destination = Path(path)
    # A same-directory temporary file allows atomic publication; hard linking
    # implements no-overwrite without a check-then-write race.
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(checkpoint, handle)
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination

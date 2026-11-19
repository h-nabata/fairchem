"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import gc
import math
import operator
import os
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import torch

from fairchem.core.calculate.series_b_workflows import _release_failed_workflow
from fairchem.core.common._model_state import MODEL_STATE_LOCK
from fairchem.core.datasets.atomic_data import AtomicData, _clone_tensor_tree
from fairchem.core.models.uma.nn.mole import MOLEGlobals
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


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
    owned = _clone_tensor_tree(
        data,
        tensor_transform=lambda tensor: tensor.detach().to(device="cpu", copy=True),
    )
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
        self._initialize(data, self.targets)

    def _initialize(self, data, targets):
        if (
            not isinstance(targets, Mapping)
            or not targets
            or set(targets) - _PROPERTIES
        ):
            raise ValueError("targets must be a nonempty energy/forces/stress mapping")
        targets = {
            name: _property_tensor(value, name, data)
            .detach()
            .to(device="cpu", copy=True)
            for name, value in targets.items()
        }
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "targets", targets)

    @classmethod
    def _from_owned_data(cls, data, targets):
        sample = cls.__new__(cls)
        sample._initialize(data, targets)
        return sample


class _ShardedSamples(Sequence):
    """Private, job-owned shards. Reads return independent CPU samples."""

    def __init__(self, samples, directory, shard_size):
        self._directory = tempfile.TemporaryDirectory(
            dir=directory, prefix="fairchem-labels-"
        )
        self._shard_size = shard_size
        self._count = 0
        shard = []
        try:
            for sample in samples:
                shard.append(sample)
                self._count += 1
                if len(shard) == shard_size:
                    torch.save(shard, self._path((self._count - 1) // shard_size))
                    shard.clear()
            if shard:
                torch.save(shard, self._path((self._count - 1) // shard_size))
        except BaseException:
            self._directory.cleanup()
            raise

    def _path(self, number):
        return Path(self._directory.name) / f"{number}.pt"

    def _read(self, number):
        # Only read data written into this instance's private temporary directory.
        return torch.load(self._path(number), map_location="cpu", weights_only=False)

    def __len__(self):
        return self._count

    def __iter__(self):
        for start in range(0, len(self), self._shard_size):
            yield from self._read(start // self._shard_size)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("distillation sample index out of range")
        shard, offset = divmod(index, self._shard_size)
        return self._read(shard)[offset]


@dataclass(frozen=True)
class DistillationDataset:
    samples: Sequence[DistillationSample]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        # Consume generators one sample at a time. Materializing their original
        # samples before making ownership copies would keep two whole datasets.
        owned = []
        for sample in self.samples:
            if not isinstance(sample, DistillationSample):
                raise ValueError(
                    "A nonempty sequence of DistillationSample is required"
                )
            owned.append(DistillationSample(sample.data, sample.targets))
            del sample
        if not owned:
            raise ValueError("A nonempty sequence of DistillationSample is required")
        object.__setattr__(self, "samples", tuple(owned))
        object.__setattr__(self, "provenance", copy.deepcopy(self.provenance))

    @classmethod
    def _from_owned_samples(cls, samples, provenance):
        if not len(samples):
            raise ValueError("A nonempty sequence of DistillationSample is required")
        dataset = cls.__new__(cls)
        object.__setattr__(dataset, "samples", samples)
        object.__setattr__(dataset, "provenance", provenance)
        return dataset


@_release_failed_workflow
def collect_teacher_labels(
    teacher,
    structures: Iterable[AtomicData],
    *,
    teacher_id: str,
    properties: tuple[str, ...] = ("energy", "forces"),
    shard_directory: str | Path | None = None,
    shard_size: int = 32,
) -> DistillationDataset:
    """
    Evaluate an explicit predictor to build an owned physical-unit dataset.

    ``teacher.predict`` must follow MLIPPredictUnit's property-dictionary API.
    The teacher's normal lazy preparation/cache may run; its weights, precision
    settings and training mode are not changed by this helper. Include domain-
    relevant TS, compressed, and nonequilibrium structures in the supplied data.
    When shard_directory is supplied, hold at most one shard of new labels in
    memory. Private temporary files live as long as the returned dataset.
    Its samples must have serializable metadata. Each file read owns its tensors.
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

    if (
        isinstance(shard_size, bool)
        or not isinstance(shard_size, Integral)
        or shard_size < 1
    ):
        raise ValueError("shard_size must be a positive integer")

    def samples():
        for structure in structures:
            data = _owned_data(structure)
            output = teacher.predict(
                data.clone(), requested_properties=tuple(properties)
            )
            if not isinstance(output, Mapping) or any(
                name not in output for name in properties
            ):
                raise ValueError(
                    "Teacher did not return all requested physical properties"
                )
            sample = DistillationSample._from_owned_data(
                data, {name: output[name] for name in properties}
            )
            # Release the predictor's output/graph before yielding the CPU
            # sample, and consume that sample before evaluating the next one.
            del output, data
            yield sample
            del sample

    owned = (
        tuple(samples())
        if shard_directory is None
        else _ShardedSamples(samples(), shard_directory, int(shard_size))
    )
    return DistillationDataset._from_owned_samples(
        owned,
        {
            "teacher_id": teacher_id,
            "properties": list(properties),
            "label_units": "teacher predictor physical units",
        },
    )


@lru_cache(maxsize=16)
def _element_reference_table(entries, device, dtype):
    # Index zero and the last index also catch invalid atomic numbers.
    table = [math.nan] * 120
    for number, value in entries:
        table[number] = value
    return torch.tensor(table, device=device, dtype=dtype)


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
                    table = _element_reference_table(
                        tuple(sorted(self.element_references.items())),
                        value.device,
                        value.dtype,
                    )
                    numbers = data.atomic_numbers.long()
                    atomic = table[numbers.clamp(0, 119)]
                    unknown = torch.isnan(atomic)
                    if unknown.any():
                        missing = numbers[unknown].unique().tolist()
                        raise ValueError(
                            f"Missing element references for {sorted(missing)}"
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


@_release_failed_workflow
@torch.inference_mode(False)
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
    if isinstance(dataset.samples, _ShardedSamples):
        dataset = DistillationDataset._from_owned_samples(
            dataset.samples, copy.deepcopy(dataset.provenance)
        )
    else:
        dataset = DistillationDataset(dataset.samples, dataset.provenance)
    if any(set(weights.active) - sample.targets.keys() for sample in dataset.samples):
        raise ValueError("Every sample must contain all positively weighted properties")
    # A warm UMA stores the previous forward's routing graph in shared MoLE
    # globals. These transient coefficients are recomputed before each forward;
    # snapshot them without their graph so deepcopy never follows caller weights.
    # Keep ordinary parameters and their tied aliases under deepcopy's control.
    memo = {}
    for module in student.modules():
        mole_globals = getattr(module, "global_mole_tensors", None)
        if isinstance(mole_globals, MOLEGlobals) and id(mole_globals) not in memo:
            _clone_tensor_tree(
                mole_globals,
                tensor_transform=lambda tensor: tensor.detach().clone(),
                memo=memo,
            )
    model = copy.deepcopy(student, memo)
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
    with (
        MODEL_STATE_LOCK,
        torch.random.fork_rng(devices=devices, device_type=rng_backend),
        torch.enable_grad(),
        torch.autocast(device.type, enabled=False),
    ):
        torch.random.default_generator.manual_seed(int(seed))
        if devices:
            getattr(torch, device.type).default_generators[devices[0]].manual_seed(
                int(seed)
            )
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(int(steps)):
            sample = dataset.samples[
                int(
                    torch.randint(
                        len(dataset.samples), (), generator=generator, device="cpu"
                    )
                )
            ]
            data = _clone_tensor_tree(
                sample.data,
                tensor_transform=lambda tensor: tensor.to(
                    device=device,
                    dtype=dtype if tensor.is_floating_point() else None,
                    copy=True,
                ),
            )
            data["_requested_properties"] = weights.active
            data.pos.requires_grad_(True)
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
            if (
                not gradients
                or not torch.stack([torch.isfinite(g).all() for g in gradients]).all()
            ):
                raise ValueError("Missing or nonfinite student parameter gradients")
            optimizer.step()
            if not torch.stack([torch.isfinite(p).all() for p in parameters]).all():
                raise ValueError("Nonfinite student parameters after optimizer step")
            history.append(float(loss.detach().cpu()))
            optimizer.zero_grad(set_to_none=True)
            # These tensors can retain a conservative derivative graph and
            # coordinate storage even after backward has freed saved values.
            # Do not overlap them with the next forward's allocations.
            del output, prediction, target, loss, gradients, data
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


@_release_failed_workflow
def export_distilled_checkpoint(
    result: DistillationResult,
    path: str | Path,
    *,
    model_config: dict,
    tasks_config: list,
    architecture_note: str,
    overwrite: bool = False,
    share_ema_storage: bool = False,
) -> Path:
    """
    Export a standard checkpoint only after strict architecture/state matching.

    Configurations and task normalizers must describe the trained student's
    actual output convention. This validates serialization/shape compatibility,
    not physical accuracy. Construct the resulting predictor with this checkpoint
    path explicitly; production UMA checkpoints are never rewritten here.
    With share_ema_storage=True the file stores each raw tensor once and its
    EMA entry references the same storage. Treat both serialized states as read-only.
    """
    if not isinstance(result, DistillationResult):
        raise TypeError("result must be DistillationResult")
    if not isinstance(architecture_note, str) or not architecture_note.strip():
        raise ValueError("architecture_note must describe the trained architecture")
    if not isinstance(share_ema_storage, bool):
        raise TypeError("share_ema_storage must be a boolean")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    if not isinstance(model_config, Mapping) or "_target_" not in model_config:
        raise ValueError("model_config must be an explicit Hydra target config")
    if not isinstance(tasks_config, list):
        raise TypeError("tasks_config must be a list of task configurations")

    def cpu_state_tensor(value):
        value = value.detach().to(device="cpu", copy=True)
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise ValueError("Cannot export nonfinite student state")
        return value

    # PyTorch state_dicts also carry version metadata and may contain a
    # module's arbitrary get_extra_state() payload, rather than only tensors.
    # With keep_vars, tied parameters are the same Python object in both
    # entries. The clone memo preserves that tie when storage sharing is selected.
    raw = (
        result.model.state_dict(keep_vars=True)
        if share_ema_storage
        else result.model.state_dict()
    )
    state = _clone_tensor_tree(raw, tensor_transform=cpu_state_tensor)
    del raw
    config, tasks = copy.deepcopy(dict(model_config)), copy.deepcopy(tasks_config)
    with MODEL_STATE_LOCK, torch.random.fork_rng(devices=[]), torch.device("cpu"):
        reconstructed = hydra.utils.instantiate(config)
        reconstructed.load_state_dict(state, strict=True)
        if hasattr(reconstructed, "setup_tasks"):
            reconstructed.setup_tasks(tasks)
    # The validation-only model is no longer needed. Release it before copying
    # EMA state and serializing; some models contain bound-method cycles.
    del reconstructed
    gc.collect()
    # The serialized raw and EMA weights are identical snapshots. Share their
    # read-only tensor storage; deepcopy still separates mutable extra-state containers.
    ema_values = _clone_tensor_tree(
        state, tensor_transform=(lambda value: value) if share_ema_storage else None
    )
    ema_state = OrderedDict(
        [("n_averaged", torch.tensor(0, device="cpu"))]
        + [(f"module.{name}", value) for name, value in ema_values.items()]
    )
    if hasattr(state, "_metadata"):
        ema_state._metadata = {
            "": {"version": torch.optim.swa_utils.AveragedModel._version},
            **{
                "module" if not name else f"module.{name}": copy.deepcopy(value)
                for name, value in state._metadata.items()
            },
        }
    checkpoint = MLIPInferenceCheckpoint(
        model_config=config,
        model_state_dict=state,
        ema_state_dict=ema_state,
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

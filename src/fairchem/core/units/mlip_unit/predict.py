"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import gc
import logging
import math
import os
import random
import sys
import threading
import weakref
from collections import defaultdict
from collections.abc import Collection, Mapping
from contextlib import contextmanager, nullcontext
from functools import cached_property, wraps
from typing import TYPE_CHECKING, ClassVar, Protocol

import hydra
import numpy as np
import torch
import torch.distributed as dist
from monty.dev import requires
from torch.distributed.elastic.utils.distributed import get_free_port
from torchtnt.framework import PredictUnit, State

try:
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ray_installed = True
except ImportError:
    ray_installed = False

from fairchem.core.common import gp_utils
from fairchem.core.common.distutils import (
    CURRENT_DEVICE_TYPE_STR,
    assign_device_for_local_rank,
    get_device_for_local_rank,
    resolve_xpu_device,
    setup_env_local_multi_gpu,
)
from fairchem.core.common.utils import get_commit_hash_for_repo
from fairchem.core.components.serve_utils import get_app_handle_with_retry
from fairchem.core.datasets.atomic_data import (
    AtomicData,
    _clone_tensor_tree,
    warn_if_upcasting,
)
from fairchem.core.models.uma.compat import get_uma_version
from fairchem.core.models.uma.nn.execution_backends import (
    ExecutionMode,
    maybe_update_settings_backend,
)
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.api.inference import (
    NITRO_CHECKPOINT,
    NITRO_MODEL_ID,
    MergeMoleConsistencyError,
    UMANitroNotSupportedError,
    describe_inference_settings,
)
from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec
from fairchem.core.units.mlip_unit.inference_runtime import (
    ModelStreamGuard,
    PreparedModelCache,
    configure_numeric_runtime,
    model_tensor_signature,
    mole_composition_key,
)
from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task
from fairchem.core.units.mlip_unit.series_b_runtime import (
    prepare_series_b,
    validate_series_b,
)
from fairchem.core.units.mlip_unit.single_atom_patch import (
    single_atom_prediction_from_lookup,
)
from fairchem.core.units.mlip_unit.utils import (
    get_backbone_class_from_checkpoint,
    load_inference_model,
    tf32_context_manager,
)

if TYPE_CHECKING:
    from ase import Atoms
    from ray.serve.handle import DeploymentHandle

    from fairchem.core.common.gp_utils import GraphParallelConfig
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint


_AUXILIARY_LEVELS = {
    "forces": "atom",
    "stress": "system",
    "hessian": "system",
    "hessian_vector_product": "atom",
    "embeddings": "atom",
}

_PREDICTOR_STATES = weakref.WeakKeyDictionary()
_PREDICTOR_STATE_LOCK = threading.RLock()
_MODEL_CONSTRUCTION_LOCK = threading.RLock()


def _predictor_state(predictor):
    with _PREDICTOR_STATE_LOCK:
        return _PREDICTOR_STATES.setdefault(predictor, [threading.RLock(), None])


def _validation_info_snapshot(value, active=None):
    """
    Snapshot small immutable metadata; mutable arrays require fresh validation.
    """
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        if isinstance(value, np.generic):
            scalar = value.item()
            return (
                (type(value), scalar)
                if scalar is None
                or type(scalar) in (str, bytes, bool, int, float, complex)
                else None
            )
        return (type(value), value)
    if isinstance(value, (list, tuple, dict)):
        active = set() if active is None else active
        if id(value) in active:
            return None
        active.add(id(value))
        try:
            if isinstance(value, dict):
                children = [
                    (
                        _validation_info_snapshot(key, active),
                        _validation_info_snapshot(item, active),
                    )
                    for key, item in value.items()
                ]
                return (
                    None
                    if any(key is None or item is None for key, item in children)
                    else (dict, tuple(children))
                )
            children = [_validation_info_snapshot(item, active) for item in value]
            return (
                None
                if any(item is None for item in children)
                else (type(value), tuple(children))
            )
        finally:
            active.remove(id(value))
    return None


def _isolated_model_initialization(initialize):
    @wraps(initialize)
    def isolated(*args, **kwargs):
        # Checkpoints and their newly created parameters are initialized on CPU.
        # Do not seed an accelerator here: manual_seed() would enqueue lazy
        # seed changes even on devices this predictor never uses.
        with _MODEL_CONSTRUCTION_LOCK, torch.random.fork_rng(devices=[]):
            python_state, numpy_state = random.getstate(), np.random.get_state()
            prior_device = os.environ.get(CURRENT_DEVICE_TYPE_STR)
            try:
                return initialize(*args, **kwargs)
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
                if prior_device is None:
                    os.environ.pop(CURRENT_DEVICE_TYPE_STR, None)
                else:
                    os.environ[CURRENT_DEVICE_TYPE_STR] = prior_device

    return isolated


@contextmanager
def _serialized_prediction(predictor):
    """
    Serialize mutable predictor state through postprocessing and GPU submission.

    Locks/events live outside predictor objects so copying or pickling a model
    does not copy a lock or an event associated with another model's execution.
    """
    if not isinstance(predictor, MLIPPredictUnit):
        yield
        return
    state = _predictor_state(predictor)
    with state[0]:
        device = torch.device(predictor.device)
        accelerator = getattr(torch, device.type, None)
        stream = None
        if device.type in {"cuda", "xpu"} and accelerator.is_available():
            stream = accelerator.current_stream(device)
            if state[1] is not None:
                if isinstance(state[1], str):
                    raise RuntimeError(
                        f"Previous prediction's GPU completion could not be established; construct a new predictor ({state[1]})"
                    )
                stream.wait_event(state[1])
        try:
            yield
        finally:
            # A failed call may have enqueued work before raising. Subsequent
            # calls still must not overwrite buffers used by that submission.
            if stream is not None:
                failed = sys.exc_info()[0] is not None
                try:
                    event = accelerator.Event()
                    event.record(stream)
                    state[1] = event
                except Exception as error:
                    state[1] = f"{type(error).__name__}: {error}"
                    if not failed:
                        raise
                    logging.warning(
                        "Unable to record failed prediction's completion: %s", error
                    )


def collate_predictions(predict_fn):
    @wraps(predict_fn)
    def collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True, **kwargs
    ):
        with _serialized_prediction(predict_unit):
            return _collated_predict(
                predict_unit, data, undo_element_references, **kwargs
            )

    def _collated_predict(
        predict_unit, data: AtomicData, undo_element_references: bool = True, **kwargs
    ):
        # Get the full prediction dictionary from the original predict method
        preds = predict_fn(predict_unit, data, undo_element_references, **kwargs)
        collated_preds = defaultdict(list)
        batch_by_device = {}
        provided = [set() for _ in data.dataset]

        def validate_output(name, value, level):
            if not torch.is_tensor(value) or value.ndim == 0:
                raise ValueError(
                    f"Prediction {name!r} must be a tensor with a {level} dimension"
                )
            expected = len(data.dataset) if level == "system" else data.batch.numel()
            if value.shape[0] != expected:
                raise ValueError(
                    f"Prediction {name!r} has {value.shape[0]} {level} rows; expected {expected}"
                )

        for i, dataset in enumerate(data.dataset):
            if dataset not in predict_unit.dataset_to_tasks:
                raise ValueError(
                    f"Unknown prediction dataset {dataset!r} at system {i}"
                )
            for task in predict_unit.dataset_to_tasks[dataset]:
                if task.name not in preds:
                    continue
                prediction = preds[task.name]
                validate_output(task.name, prediction, task.level)
                provided[i].add(task.property)
                if task.level == "system":
                    collated_preds[task.property].append(prediction[i].unsqueeze(0))
                elif task.level == "atom":
                    if len(data.dataset) == 1:
                        # A valid single-system batch already has the required
                        # atom order. Boolean indexing would copy it and can
                        # require a device synchronization for the output size.
                        collated_preds[task.property].append(prediction)
                    else:
                        # Prediction no longer moves the caller's batch in place.
                        # Share one index transfer across properties on a device.
                        if prediction.device not in batch_by_device:
                            batch_by_device[prediction.device] = data.batch.to(
                                prediction.device
                            )
                        batch = batch_by_device[prediction.device]
                        collated_preds[task.property].append(prediction[batch == i])
                else:
                    raise RuntimeError(
                        f"Unrecognized task level={task.level} found in data batch at position {i}"
                    )

            # Requested derivatives/embeddings need not be learned checkpoint
            # tasks. Their energy task supplies the head prefix and dataset.
            task_properties = {
                task.property for task in predict_unit.dataset_to_tasks[dataset]
            }
            for task in predict_unit.dataset_to_tasks[dataset]:
                if task.property != "energy":
                    continue
                for prop, level in _AUXILIARY_LEVELS.items():
                    name = task.name.removesuffix("energy") + prop
                    if prop in task_properties or name not in preds:
                        continue
                    value = preds[name]
                    validate_output(name, value, level)
                    provided[i].add(prop)
                    if level == "system":
                        value = value[i].unsqueeze(0)
                    elif len(data.dataset) > 1:
                        if value.device not in batch_by_device:
                            batch_by_device[value.device] = data.batch.to(value.device)
                        value = value[batch_by_device[value.device] == i]
                    collated_preds[prop].append(value)

        requested = kwargs.get("requested_properties")
        if requested is None:
            requested = getattr(
                getattr(predict_unit, "inference_settings", None),
                "requested_properties",
                None,
            )
        if requested is not None:
            for i, properties in enumerate(provided):
                if missing := set(requested) - properties:
                    raise ValueError(
                        f"Requested properties {sorted(missing)} are unavailable for system {i} (dataset={data.dataset[i]!r})"
                    )

        return {
            prop: values[0]
            if len(data.dataset) == 1 and len(values) == 1
            else torch.cat(values)
            for prop, values in collated_preds.items()
        }

    return collated_predict


def _copy_data_to_device(
    data: AtomicData, device: str | torch.device, dtype: torch.dtype
) -> AtomicData:
    """Copy a batch once per tensor, combining device and dtype conversion.

    Deep-copying with a tensor memo preserves metadata and non-leaf autograd
    inputs without re-running AtomicData's scalar validation on an accelerator.
    """
    return _clone_tensor_tree(
        data,
        tensor_transform=lambda value: value.to(
            device=device,
            dtype=dtype if value.is_floating_point() else value.dtype,
            copy=True,
        ),
    )


def _prepare_inference_gradients(backbone, data: AtomicData) -> None:
    regress_config = getattr(backbone, "regress_config", None)
    if regress_config is None or regress_config.direct_forces:
        return
    requested = data.get("_requested_properties", None)
    derivatives = {"forces", "stress", "hessian", "hessian_vector_product"}
    need_derivatives = (
        bool(derivatives.intersection(requested))
        if requested is not None
        else any(getattr(regress_config, name, False) for name in derivatives)
    )
    if need_derivatives:
        data["pos"].requires_grad_(True)
    if regress_config.stress and (requested is None or "stress" in requested):
        data["cell"].requires_grad_(True)


def _clear_exception_tracebacks(error: Exception) -> None:
    """Release failed-model frames before allocating a fallback model."""
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        current.__traceback__ = None
        pending.extend(
            cause
            for cause in (current.__cause__, current.__context__)
            if cause is not None
        )


_REDUCED_PRECISION_WARNING = (
    "Running in a reduced-precision mode: some values, such as higher-order "
    "derivatives and phonon calculations, will be less accurate. For standard "
    "FP32 precision, use inference_settings='high-precision'."
)

_NITRO_BANNER = r"""
    _   ________________  ____
   / | / /  _/_  __/ __ \/ __ \
  /  |/ // /  / / / /_/ / / / /
 / /|  // /  / / / _, _/ /_/ /
/_/ |_/___/ /_/ /_/ |_|\____/

          >>> ENGAGED >>>
"""


def _uses_reduced_precision(settings: InferenceSettings, device: str) -> bool:
    """
    Whether inference opts into Series B, or CUDA TF32/UMA Nitro.
    """
    if settings.series_b.enabled:
        return True
    if torch.device(device).type != "cuda":
        return False
    return settings.nitro or (
        settings.tf32 and settings.base_precision_dtype == torch.float32
    )


def _describe_settings_change(
    before: InferenceSettings, after: InferenceSettings, device: str
) -> str:
    """
    Name the settings a fallback changed and the precision now in effect.
    """
    changed = []
    for field in dataclasses.fields(after):
        value = getattr(after, field.name)
        if value != getattr(before, field.name):
            if isinstance(value, enum.Enum):  # e.g. ExecutionMode
                value = value.value
            changed.append(f"{field.name}={value!r}")
    if _uses_reduced_precision(after, device):
        precision = "TF32"
    elif after.base_precision_dtype in (torch.float64, "float64"):
        precision = "FP64"
    else:
        precision = "FP32"
    return (
        f"Now running with {', '.join(changed) or 'unchanged settings'} ({precision})."
    )


def _checkpoint_model_name(model_config: dict) -> str:
    """
    Name a checkpoint's model for messages, e.g. "UMA 1.1" or "UMA-S-1.2".
    """
    version = get_uma_version(model_config)
    if version == "1.1":
        return "UMA 1.1"
    if version == "tagged":
        return model_config["model_id"]
    return "This checkpoint"


class _ModelWrapper(torch.nn.Module):
    """
    Expose a model as ``.module``, like the ``AveragedModel`` that
    ``load_inference_model`` returns.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, data: AtomicData) -> dict:
        return self.module(data)


class MLIPPredictUnitProtocol(Protocol):
    def predict(self, data: AtomicData, undo_element_references: bool) -> dict: ...

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None: ...

    @property
    def dataset_to_tasks(self) -> dict[str, list]: ...


class MLIPPredictUnit(PredictUnit[AtomicData], MLIPPredictUnitProtocol):
    supports_requested_properties = True

    @_isolated_model_initialization
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        form_elem_refs: dict | None = None,
        assert_on_nans: bool = False,
    ):
        super().__init__()
        device = str(device)
        os.environ[CURRENT_DEVICE_TYPE_STR] = device
        # Fail before loading a potentially large checkpoint if the requested
        # accelerator is unavailable; never silently substitute CPU inference.
        self._setup_device(device)
        os.environ[CURRENT_DEVICE_TYPE_STR] = self.device

        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        self._setup_refs(atom_refs, form_elem_refs)

        if inference_settings is None:
            inference_settings = InferenceSettings()
        if not isinstance(inference_settings, InferenceSettings):
            raise ValueError("inference_settings must be an InferenceSettings instance")

        # Named inference modes (e.g., default, multigpu) are shared module-level
        # instances. We keep a private copy so a change driven by a contract break
        # does not alter other predictors.
        self.inference_settings = copy.deepcopy(inference_settings)
        self.inference_settings.__post_init__()
        self._inference_model_path = inference_model_path
        self._overrides = copy.deepcopy(overrides)
        self._requested_device = device
        validate_series_b(self.inference_settings, device)
        if torch.device(device).type == "xpu":
            self._validate_xpu_settings()
        self._setup_threads(self.inference_settings)

        if self.inference_settings.wigner_cuda:
            logging.warning(
                "The wigner_cuda flag is deprecated and will be removed in future versions."
            )

        self._load_model()
        # After _load_model, so a request UMA Nitro could not serve is judged by
        # the mode that actually runs.
        if _uses_reduced_precision(self.inference_settings, device):
            logging.warning(_REDUCED_PRECISION_WARNING)

        self.model.eval()
        self.lazy_model_intialized = False
        self.assert_on_nans = assert_on_nans
        self._warned_upcast = False
        self._nitro_announced = False
        self._prepared_models = None
        self._mole_template = None
        self._active_mole_key = None
        self._active_model_signature = None
        self._numeric_runtimes = []

        if self.model.module.backbone.regress_config.direct_forces:
            logging.warning(
                "This is a direct-force model. Direct force predictions may lead to "
                "discontinuities in the potential energy surface and energy conservation errors."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Cache signatures describe original tensor identities and generations.
        # Copies must establish their own observations and prepared-model LRU.
        for name in ("_prepared_models", "_active_mole_key", "_active_model_signature"):
            state[name] = None
        return state

    def _load_model(self) -> None:
        """Load a fresh, unprepared model from the inference checkpoint."""
        # Load checkpoint first to get model type; UMA compat fixups run downstream
        # in load_inference_model.
        checkpoint = torch.load(
            self._inference_model_path, map_location="cpu", weights_only=False
        )

        if self.inference_settings.nitro:
            before = copy.deepcopy(self.inference_settings)
            try:
                self._load_nitro_model(checkpoint)
            except UMANitroNotSupportedError as error:
                logging.warning(f"UMA Nitro cannot serve this request: {error}.")
                # Nitro can reject checkpoint tensors after constructing its
                # backbone. Release that partial model before building Hydra.
                _clear_exception_tracebacks(error)
                self.inference_settings.nitro = False
                self._load_hydra_model(checkpoint)
                self._log_settings_change(before)
        else:
            self._load_hydra_model(checkpoint)

        # Get backbone's default untrained tasks (if supported and enabled)
        default_backbone_tasks = []
        if self.inference_settings.auto_add_default_untrained_tasks:
            backbone = self.model.module.backbone
            if hasattr(backbone, "get_default_untrained_tasks"):
                default_backbone_tasks = backbone.get_default_untrained_tasks(
                    self.model.module.tasks,
                    self.inference_settings,
                )

        # Create explicitly requested untrained tasks
        untrained_tasks = self._create_untrained_tasks(
            self.inference_settings, self.model.module.tasks
        )

        explicit_task_names = {t.name for t in untrained_tasks}
        checkpoint_task_names = set(self.model.module.tasks.keys())

        for task in default_backbone_tasks:
            if (
                task.name not in explicit_task_names
                and task.name not in checkpoint_task_names
            ):
                untrained_tasks.append(task)

        if untrained_tasks:
            logging.info(
                f"Adding {len(untrained_tasks)} untrained task(s): "
                f"{[t.name for t in untrained_tasks]}"
            )
            self.model.module.add_tasks(untrained_tasks)
        self.model.eval()

    def _load_nitro_model(self, checkpoint: MLIPInferenceCheckpoint) -> None:
        """
        Build the standalone UMA Nitro model from the checkpoint weights.

        Raises:
            UMANitroNotSupportedError: If Nitro cannot serve this request.
        """
        # Checked here, before importing Nitro, so these requests get a clear
        # reason even where its kernels cannot be imported (e.g. no Triton). A
        # fine-tune added to Nitro's MD5 allowlist must keep model_id
        # UMA-S-1.2.1, or this check rejects it before the allowlist is read.
        model_name = _checkpoint_model_name(checkpoint.model_config)
        if model_name != NITRO_MODEL_ID:
            raise UMANitroNotSupportedError(
                f"{model_name} is not supported in nitro mode; use UMA 1.2.1 "
                f"({NITRO_CHECKPOINT}) for the nitro speedup"
            )
        if torch.device(self._requested_device).type != "cuda":
            raise UMANitroNotSupportedError("nitro mode requires a CUDA device")
        try:
            from fairchem.core.models.uma.uma_nitro import UMANitroModel
        except ImportError as error:
            raise UMANitroNotSupportedError(
                f"nitro mode's kernels could not be imported ({error})"
            ) from error

        UMANitroModel.validate(
            self.inference_settings, self._inference_model_path, self._requested_device
        )
        self.model = _ModelWrapper(UMANitroModel.from_checkpoint(checkpoint))

    def _fall_back_with(self, reason: str, **changes) -> None:
        """
        Warn with the reason, change settings, reload, and warn with what runs.

        The reason is logged first so it survives a reload that fails.

        Args:
            reason: Why the current model cannot serve the request.
            changes: The ``InferenceSettings`` fields to change.
        """
        logging.warning(reason)
        before = copy.deepcopy(self.inference_settings)
        for name, value in changes.items():
            setattr(self.inference_settings, name, value)
        self.clear_prepared_model_cache(release_template=True)
        release = getattr(self.model.module, "release_inference_resources", None)
        if release is not None:
            release()
        # A bound callback keeps its module alive even after deleting self.model.
        # Release that reference before collecting and loading the replacement.
        del release
        self.lazy_model_intialized = False
        del self.model
        gc.collect()
        device_type = torch.device(self._requested_device).type
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "xpu":
            torch.xpu.empty_cache()
        self._load_model()
        self._log_settings_change(before)

    def _log_settings_change(self, before: InferenceSettings) -> None:
        """
        Warn with the settings a fallback changed and the precision now in use.
        """
        logging.warning(
            _describe_settings_change(
                before, self.inference_settings, self._requested_device
            )
        )

    def _fall_back(self, error: Exception) -> None:
        """
        Take the next step down: nitro to multigpu, or multigpu to the general model.
        """
        settings = self.inference_settings
        if not (
            settings.nitro
            or settings.merge_mole
            or settings.compile
            or settings.execution_mode == ExecutionMode.UMAS_FAST_GPU
        ):
            # Reloading the same general model cannot resolve the failure.
            # Retain the original traceback for errors that must propagate.
            raise error
        # Recoverable exceptions may retain the failed model and its large
        # intermediates through forward/prepare traceback frames. They must be
        # dropped before the replacement model is allocated, including chained
        # kernel errors. The error messages remain available for the warning.
        _clear_exception_tracebacks(error)
        if self.inference_settings.nitro:
            # The same settings without nitro: multigpu, for the default preset.
            self._fall_back_with(
                f"UMA Nitro cannot serve this request: {error}.", nitro=False
            )
        else:
            self._fall_back_from_fast_path(error)

    def _initialize(self, data: AtomicData) -> None:
        """
        Lazily initialize, falling back until a model accepts ``data``.

        A heterogeneous batch, for example, falls back from nitro to multigpu,
        whose MOLE merge rejects it too, and then to the general model.
        """
        for _ in range(2):  # nitro -> multigpu -> general
            try:
                self._lazy_init(data)
                return
            except (MergeMoleConsistencyError, UMANitroNotSupportedError) as error:
                self._fall_back(error)
        self._lazy_init(data)

    @_isolated_model_initialization
    def _load_hydra_model(self, checkpoint: MLIPInferenceCheckpoint) -> None:
        """Build the composable HydraModel from the checkpoint."""
        # if the model is uma-s and the execution mode is not explicitly set, default to the optimized uma-s gpu execution mode.
        # only for CUDA predict units: the fast backend uses Triton kernels that cannot run on CPU tensors.
        if torch.device(self._requested_device).type == "cuda":
            self.inference_settings = maybe_update_settings_backend(
                self.inference_settings, checkpoint.model_config
            )

        # Build model-specific overrides
        final_overrides = self._build_overrides_from_settings(
            checkpoint, self._overrides, self.inference_settings
        )

        # Set default dtype during model construction so that non-persistent
        # buffers (SO3_Grid matrices, CoefficientMapping) are created at the
        # requested precision rather than being cast from float32 later.
        prev_dtype = torch.get_default_dtype()
        if prev_dtype != self.inference_settings.base_precision_dtype:
            torch.set_default_dtype(self.inference_settings.base_precision_dtype)

        try:
            # Load model with overrides, passing pre-loaded checkpoint
            self.model, checkpoint = load_inference_model(
                self._inference_model_path,
                use_ema=True,
                overrides=final_overrides,
                preloaded_checkpoint=checkpoint,
            )

            # Model sets up tasks
            self.model.module.setup_tasks(checkpoint.tasks_config)
        finally:
            if torch.get_default_dtype() != prev_dtype:
                torch.set_default_dtype(prev_dtype)

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self.model.module.dataset_to_tasks

    @property
    def tasks(self) -> dict:
        return self.model.module.tasks

    def set_seed(self, seed: int) -> None:
        """
        Initialize random seeds.
        """
        logging.debug(f"Setting random seed to {seed}")
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)
        # torch.manual_seed seeds all available device backends, including XPU.
        torch.manual_seed(seed)

    def _setup_refs(self, atom_refs: dict | None, form_elem_refs: dict | None) -> None:
        """
        Setup element references.
        """
        self.atom_refs = (
            {
                task.replace("_elem_refs", ""): copy.deepcopy(refs)
                for task, refs in atom_refs.items()
            }
            if atom_refs
            else {}
        )
        self.form_elem_refs = copy.deepcopy(form_elem_refs) if form_elem_refs else {}

    def _setup_threads(self, settings: InferenceSettings) -> None:
        """
        Configure thread settings.
        """
        if settings.torch_num_threads is not None:
            if torch.get_num_interop_threads() != settings.torch_num_threads:
                torch.set_num_interop_threads(settings.torch_num_threads)
            if torch.get_num_threads() != settings.torch_num_threads:
                torch.set_num_threads(settings.torch_num_threads)

    def _setup_device(self, device: str) -> None:
        """
        Setup inference device.
        """
        device = str(device)
        if device in ("cpu", "cuda"):
            self.device = get_device_for_local_rank() if device == "cuda" else "cpu"
            return
        requested = torch.device(device)
        if requested.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Requested CUDA device {requested}, but CUDA is unavailable"
                )
            if requested.index is None or requested.index >= torch.cuda.device_count():
                raise ValueError(f"CUDA device index {requested.index} is unavailable")
            self.device = str(requested)
            return
        if requested.type != "xpu":
            raise ValueError(
                "device must be 'cpu', 'cuda', 'cuda:<index>', 'xpu', or 'xpu:<index>'"
            )
        self.device = resolve_xpu_device(requested)

    def _validate_xpu_settings(self) -> None:
        """
        Reject CUDA-only paths instead of reaching unsupported kernels on XPU.
        """
        settings = self.inference_settings
        if settings.nitro or settings.execution_mode == ExecutionMode.UMAS_FAST_GPU:
            raise ValueError(
                "UMA Nitro and umas_fast_gpu require CUDA. Use inference_settings='xpu' "
                "or execution_mode='general' for Intel GPUs."
            )
        if settings.internal_graph_gen_version not in (1, 2):
            raise ValueError(
                "XPU requires internal_graph_gen_version=1 or 2; "
                "version 3 uses NVIDIA-specific neighbor-list kernels."
            )
        if settings.tf32:
            raise ValueError("XPU inference requires tf32=False.")
        if settings.base_precision_dtype != torch.float32:
            raise ValueError(
                "XPU inference currently supports base_precision_dtype=float32."
            )
        if settings.execution_mode is None:
            settings.execution_mode = ExecutionMode.GENERAL

    def clear_prepared_model_cache(self, *, release_template: bool = False) -> None:
        """
        Release cached models, optionally also the CPU preparation template.

        Keep the template for subsequent exact re-preparation by default.
        Reload paths discard it because their settings can change.
        """
        with _predictor_state(self)[0]:
            cache = getattr(self, "_prepared_models", None)
            if cache is not None:
                cache.clear()
            self._prepared_models = None
            if release_template:
                self._mole_template = None
            self._active_mole_key = None
            self._active_model_signature = None

    def _mole_key(self, data):
        return mole_composition_key(
            data,
            device=self.device,
            dtype=self.inference_settings.base_precision_dtype,
            model_id=getattr(self.model.module, "model_id", None),
            settings_key=repr(self.inference_settings),
        )

    def _select_prepared_model(self, data) -> None:
        """
        Select or prepare an exact-composition model without disabling merging.
        """
        settings = self.inference_settings
        if not settings.mole_cache_size or not settings.merge_mole or settings.nitro:
            return
        if getattr(self, "_prepared_models", None) is None:
            self._prepared_models = PreparedModelCache(
                settings.mole_cache_size, settings.mole_cache_max_bytes
            )
        signature = getattr(self, "_active_model_signature", None)
        if signature is not None and signature != model_tensor_signature(self.model):
            self.clear_prepared_model_cache(release_template=True)
            raise RuntimeError(
                "Prepared model weights/buffers changed after MoLE caching. "
                "Construct a predictor from the updated checkpoint before reusing merged models."
            )
        key = self._mole_key(data)
        if key == getattr(self, "_active_mole_key", None):
            return
        cached = self._prepared_models.get(key)
        if cached is not None:
            self.model = cached
            self.lazy_model_intialized = True
            self._active_mole_key = key
            self._active_model_signature = model_tensor_signature(self.model)
            self._numeric_runtimes = getattr(self.model.module, "_numeric_runtimes", [])
            return
        template = getattr(self, "_mole_template", None)
        if self.lazy_model_intialized and template is not None:
            self.model = copy.deepcopy(template)
            self.lazy_model_intialized = False
            self._numeric_runtimes = []
            self._active_model_signature = None
        self._active_mole_key = key

    def _build_overrides_from_settings(
        self,
        checkpoint: MLIPInferenceCheckpoint,
        user: dict | None,
        settings: InferenceSettings,
    ) -> dict:
        """Build backbone config overrides by delegating to model-specific logic."""
        # The backbone mapping is updated below. A shallow copy would overwrite
        # the caller's nested values before they can take precedence, and would
        # conceal unsupported XPU overrides from the validation below.
        overrides = {} if user is None else copy.deepcopy(user)
        if "backbone" not in overrides:
            overrides["backbone"] = {}

        # Delegate to model-specific classmethod
        backbone_cls = get_backbone_class_from_checkpoint(checkpoint)
        backbone_overrides = backbone_cls.build_inference_settings(settings)

        overrides["backbone"].update(backbone_overrides)

        # User overrides take precedence
        if user is not None and "backbone" in user:
            overrides["backbone"].update(user["backbone"])

        # User overrides take precedence, so validate the effective settings as
        # well; otherwise an override can re-enable a CUDA-only implementation.
        if torch.device(self._requested_device).type == "xpu":
            backbone = overrides["backbone"]
            if backbone.get("execution_mode") == ExecutionMode.UMAS_FAST_GPU:
                raise ValueError("The umas_fast_gpu backbone override requires CUDA.")
            if backbone.get("radius_pbc_version", 2) not in (1, 2):
                raise ValueError(
                    "XPU backbone overrides require radius_pbc_version=1 or 2."
                )

        return overrides

    def _create_untrained_tasks(
        self,
        settings: InferenceSettings,
        checkpoint_tasks: dict[str, Task],
    ) -> list[Task]:
        """
        Generate Task objects for untrained derivative properties.

        For each requested property+dataset combination:
        1. Verify an energy task exists for that dataset
        2. Create a new Task with:
           - inference_only=True (exclude from training/eval)
           - Normalizer copied from energy task
           - element_references=None (derivatives don't use elem refs)
           - Appropriate out_spec for the property type

        Args:
            settings: InferenceSettings with compute_untrained_* flags
            checkpoint_tasks: Dictionary of Task objects from checkpoint

        Returns:
            List of Task objects for untrained properties
        """
        untrained_tasks = []
        energy_task_by_dataset = {}

        for task in checkpoint_tasks.values():
            if task.property == "energy":
                for dataset in task.datasets:
                    energy_task_by_dataset[dataset] = task

        # Generate forces tasks
        for dataset in settings.predict_untrained_forces:
            if dataset not in energy_task_by_dataset:
                logging.warning(
                    f"Cannot create forces task for dataset '{dataset}': "
                    f"no energy task found. Skipping."
                )
                continue

            energy_task = energy_task_by_dataset[dataset]
            # Infer task name prefix from energy task naming convention
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            untrained_tasks.append(
                Task(
                    name=f"{task_prefix}forces",
                    level="atom",
                    property="forces",
                    out_spec=OutputSpec(
                        dim=[3], dtype=self.inference_settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,  # Copy from energy
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,  # Forces are derivatives, no elem refs
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,  # KEY: Skip training/eval
                )
            )

        # Generate stress tasks
        for dataset in settings.predict_untrained_stress:
            if dataset not in energy_task_by_dataset:
                logging.warning(
                    f"Cannot create stress task for dataset '{dataset}': "
                    f"no energy task found. Skipping."
                )
                continue

            energy_task = energy_task_by_dataset[dataset]
            # Infer task name prefix from energy task naming convention
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            untrained_tasks.append(
                Task(
                    name=f"{task_prefix}stress",
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(
                        dim=[1, 9], dtype=self.inference_settings.base_precision_dtype
                    ),
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )

        # Generate hessian tasks
        for dataset in settings.predict_untrained_hessian:
            if dataset not in energy_task_by_dataset:
                logging.warning(
                    f"Cannot create hessian task for dataset '{dataset}': "
                    f"no energy task found. Skipping."
                )
                continue

            energy_task = energy_task_by_dataset[dataset]
            # Infer task name prefix from energy task naming convention
            task_prefix = "" if energy_task.name == "energy" else f"{dataset}_"
            untrained_tasks.append(
                Task(
                    name=f"{task_prefix}hessian",
                    level="system",
                    property="hessian",
                    out_spec=OutputSpec(
                        dim=[None, None],
                        dtype=self.inference_settings.base_precision_dtype,
                    ),  # [N*3, N*3]
                    normalizer=energy_task.normalizer,
                    datasets=[dataset],
                    loss_fn=None,
                    element_references=None,
                    metrics=[],
                    train_on_free_atoms=True,
                    eval_on_free_atoms=True,
                    inference_only=True,
                )
            )

        return untrained_tasks

    def move_to_device(self):
        self.model.to(self.device)
        for task in self.model.module.tasks.values():
            task.normalizer.to(self.device)
            if task.element_references is not None:
                task.element_references.to(self.device)

    def _fall_back_from_fast_path(
        self, consistency_error: MergeMoleConsistencyError
    ) -> None:
        reason = (
            "The UMA fast path (merge_mole + compile) is only available for "
            "fixed composition, task, charge, and spin. This is optimized for "
            "MD applications. Falling back to a less optimized version for "
            "subsequent evaluations. "
            f"Reason: '{consistency_error}'.\n"
            "Use inference_settings='batch' for heterogeneous batched evaluations."
        )
        if self.inference_settings.series_b.low_rank_fraction is not None:
            raise ValueError(
                "Series B low-rank weights belong to one merged composition. "
                "Use mole_cache_size>0 or construct a predictor for the new composition."
            )
        # Fall back to the unmerged and uncompiled model.
        changes = {"merge_mole": False, "compile": False, "xpu_graph": False}
        # Every specialized UMA-S backend uses merged expert weights. Keeping
        # a portable/PyTorch backend here would make the replacement model fail
        # its own preparation validation; XPU graph capture also needs merging.
        backbone_overrides = (getattr(self, "_overrides", None) or {}).get(
            "backbone", {}
        )
        execution_mode = backbone_overrides.get(
            "execution_mode", self.inference_settings.execution_mode
        )
        if execution_mode in (
            ExecutionMode.UMAS_FAST_GPU,
            ExecutionMode.UMAS_FAST_PYTORCH,
            ExecutionMode.UMAS_FAST_PORTABLE,
        ):
            changes["execution_mode"] = ExecutionMode.GENERAL
            if "execution_mode" in backbone_overrides:
                # The predictor owns this copy. An explicit override otherwise
                # takes precedence and silently restores the incompatible mode.
                backbone_overrides["execution_mode"] = ExecutionMode.GENERAL
        self._fall_back_with(reason, **changes)

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Delegates to the model's backbone for model-specific validation.
        """
        self.model.module.validate_atoms_data(atoms, task_name)

    def predict_step(self, state: State, data: AtomicData) -> dict[str, torch.tensor]:
        return self.predict(data)

    @collate_predictions
    def predict(
        self,
        data: AtomicData,
        undo_element_references: bool = True,
        requested_properties: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, torch.tensor]:
        if requested_properties is not None and (
            isinstance(requested_properties, (str, bytes, Mapping))
            or not isinstance(requested_properties, Collection)
        ):
            raise ValueError(
                "requested_properties must be a sequence of names, not a string"
            )
        requested = (
            self.inference_settings.requested_properties
            if requested_properties is None
            else tuple(requested_properties)
        )
        if requested is not None and (
            not requested
            or any(not isinstance(prop, str) or not prop for prop in requested)
        ):
            raise ValueError("requested_properties must contain property names")
        if requested is not None:
            supported = {"energy", *_AUXILIARY_LEVELS}
            supported.update(task.property for task in self.tasks.values())
            if unknown := set(requested) - supported:
                raise ValueError(f"Unknown requested properties: {sorted(unknown)}")
        prepared_options = getattr(self.model.module, "_prepared_series_b", None)
        if (
            self.lazy_model_intialized
            and prepared_options is not None
            and prepared_options != self.inference_settings.series_b
        ):
            raise ValueError(
                "Series B settings cannot change after preparation; construct a new predictor"
            )
        prepared_settings = getattr(
            self.model.module, "_prepared_inference_settings", None
        )
        if self.lazy_model_intialized and prepared_settings is not None:
            current = copy.deepcopy(self.inference_settings)
            # Output selection is per-call and does not change prepared weights.
            current.requested_properties = prepared_settings.requested_properties
            if current != prepared_settings:
                raise ValueError(
                    "Inference settings cannot change after preparation; construct a new predictor. "
                    "Use requested_properties for per-call output selection."
                )
        self._select_prepared_model(data)
        regress_config = getattr(self.model.module.backbone, "regress_config", None)
        if (
            torch.is_inference_mode_enabled()
            and regress_config is not None
            and not regress_config.direct_forces
            and (
                bool(
                    {
                        "forces",
                        "stress",
                        "hessian",
                        "hessian_vector_product",
                    }.intersection(requested)
                )
                if requested is not None
                else any(
                    getattr(regress_config, property_name, False)
                    for property_name in ("forces", "stress", "hessian")
                )
            )
        ):
            raise RuntimeError(
                "Energy-derived forces, stress, and Hessians require autograd. "
                "Call predict() outside torch.inference_mode(); torch.no_grad() "
                "can be used instead."
            )
        if not self.lazy_model_intialized:
            self._initialize(data)  # may fall back: nitro -> multigpu -> general
            revision = get_commit_hash_for_repo(os.path.dirname(__file__))
            logging.warning(
                f"Running ({revision or 'unknown commit'}) with inference settings: "
                f"{describe_inference_settings(self.inference_settings)}"
            )

        # Handle single-atom systems (natoms==1 and pbc all False)
        # Skip this check if the model natively supports single atoms
        if not self.model.module.supports_single_atoms:
            single_atom_result = single_atom_prediction_from_lookup(
                data=data,
                atom_refs=self.atom_refs,
                tasks=self.tasks,
                device=self.device,
            )
            if single_atom_result is not None:
                if requested is not None:
                    single_atom_result = {
                        name: value
                        for name, value in single_atom_result.items()
                        if self.tasks[name].property in requested
                    }
                    available = {
                        self.tasks[name].property for name in single_atom_result
                    }
                    if missing := set(requested) - available:
                        raise ValueError(
                            "Single-atom reference lookup cannot provide requested "
                            f"properties: {sorted(missing)}"
                        )
                return single_atom_result

        dtype = self.inference_settings.base_precision_dtype
        if not self._warned_upcast:
            self._warned_upcast = warn_if_upcasting(data.pos.dtype, dtype)
        # Keep the caller isolated from graph-parallel/MOLE mutations without
        # first retaining another full copy on the accelerator.
        data_device = _copy_data_to_device(data, self.device, dtype)
        if requested is not None:
            data_device["_requested_properties"] = tuple(requested)

        _prepare_inference_gradients(self.model.module.backbone, data_device)

        # Model handles any per-prediction checks (e.g., MOLE consistency)
        try:
            self.model.module.on_predict_check(data_device)
        except MergeMoleConsistencyError as error:
            self._fall_back(error)
            # The multigpu model re-merges for this input, or falls back further.
            # _lazy_init prepares its own CPU copy even if data is already on GPU.
            self._initialize(data)
            self.model.module.on_predict_check(data_device)

        try:
            return self._run_inference(data_device, undo_element_references)
        except UMANitroNotSupportedError as error:
            # e.g. a geometry whose radius graph outgrows Nitro's capacity.
            self._fall_back(error)
            self._initialize(data)
            return self._run_inference(data_device, undo_element_references)

    @torch.inference_mode(False)
    def _lazy_init(self, data: AtomicData) -> None:
        """
        Lazy initialization on first predict call.
        """
        validate_series_b(
            self.inference_settings, self.device, self.model.module.backbone
        )
        # Checkpoints are loaded on CPU. Preparing MOLE with an already-device-
        # resident batch would mix accelerator inputs and CPU weights; float64
        # inputs also break the head's FP32 expert contraction before predict()
        # gets a chance to cast them. Prepare from a private CPU copy at the
        # loaded model's precision, preserving the caller's batch and placement.
        # The final requested precision is applied after preparation, as before.
        if (
            self.inference_settings.mole_cache_size
            and self.inference_settings.merge_mole
            and not self.inference_settings.nitro
            and getattr(self, "_mole_template", None) is None
        ):
            self._mole_template = copy.deepcopy(self.model).cpu()
        dtype = next(
            (
                parameter.dtype
                for parameter in self.model.parameters()
                if parameter.is_floating_point()
            ),
            self.inference_settings.base_precision_dtype,
        )
        preparation_data = _copy_data_to_device(data, "cpu", dtype)
        # Model handles its own preparation (MOLE merge, eval mode, etc.).
        # CPU autocast must not permanently round the merged FP32 parameters.
        with torch.autocast("cpu", enabled=False):
            self.model.module.prepare_for_inference(
                preparation_data, self.inference_settings
            )
        # Inference differentiates outputs with respect to inputs, not weights.
        self.model.requires_grad_(False)

        self.model.to(self.inference_settings.base_precision_dtype)

        if self.inference_settings.series_b.enabled:
            self.model.module._series_b_manifest = prepare_series_b(
                self.model.module, self.inference_settings, self.device
            )
        self.model.module._prepared_series_b = self.inference_settings.series_b
        self.model.module._prepared_inference_settings = copy.deepcopy(
            self.inference_settings
        )

        self.move_to_device()
        self.model.module._inference_stream_guard = ModelStreamGuard(self.model)

        self._numeric_runtimes = configure_numeric_runtime(
            self.model.module.backbone, self.inference_settings
        )
        self.model.module._numeric_runtimes = self._numeric_runtimes

        # torch.compile is for the HydraModel; Nitro has its own kernels.
        if (
            self.inference_settings.compile
            and self.inference_settings.compile_scope == "model"
            and not self.inference_settings.nitro
        ):
            logging.warning(
                "Model is being compiled. This might take a while the first time."
            )
            # Dynamo configuration is process-global and may be shared with
            # other models/jobs. Keep the caller's compiler policy unchanged.
            self.model = torch.compile(
                self.model, dynamic=self.inference_settings.compile_dynamic
            )

        self.lazy_model_intialized = True
        cache = getattr(self, "_prepared_models", None)
        if cache is not None and self.inference_settings.merge_mole:
            key = self._mole_key(data)
            cache.put(key, self.model)
            self._active_mole_key = key
            self._active_model_signature = model_tensor_signature(self.model)

    def _run_inference(self, data: AtomicData, undo_refs: bool) -> dict:
        """
        Execute model inference.
        """
        inference_context = (
            torch.no_grad()
            if self.model.module.backbone.regress_config.direct_forces
            else torch.enable_grad()
        )
        device_type = torch.device(getattr(self, "device", "cpu")).type
        if device_type == "xpu" and self.inference_settings.compile:
            from torch._inductor import config as inductor_config

            # Inductor's live setting can differ from its import-time env var;
            # Intel Triton also has a separate live knob/environment override.
            intel_knobs = getattr(
                getattr(sys.modules.get("triton"), "knobs", None), "intel", None
            )
            intel_fast_math = (
                bool(intel_knobs.fast_math)
                if intel_knobs is not None and hasattr(intel_knobs, "fast_math")
                else os.environ.get(
                    "TRITON_INTEL_FAST_MATH",
                    os.environ.get("TORCHINDUCTOR_USE_FAST_MATH", "0"),
                ).lower()
                in {"1", "true", "on", "yes"}
            )
            if inductor_config.use_fast_math or intel_fast_math:
                raise ValueError(
                    "Strict FP32 XPU compilation requires Inductor use_fast_math=False "
                    "and Intel Triton fast_math=False. The predictor does not change "
                    "these process-global settings."
                )
        stream_guard = getattr(self.model.module, "_inference_stream_guard", None)
        if stream_guard is not None:
            stream_guard.wait_and_record(self.model)
        requested = (
            data.get("_requested_properties", None) if hasattr(data, "get") else None
        )
        higher_order = (
            bool({"hessian", "hessian_vector_product"}.intersection(requested))
            if requested is not None
            else getattr(self.model.module.backbone.regress_config, "hessian", False)
        )
        for runtime in getattr(self, "_numeric_runtimes", ()):
            runtime.higher_order = higher_order
        inference_model = (
            getattr(self.model, "_orig_mod", self.model) if higher_order else self.model
        )
        # XPU's oneDNN precision control is separate from CUDA TF32 controls.
        # Strict XPU leaves it unchanged; B03 scopes and restores its opt-in flag.
        precision_context = nullcontext()
        if device_type == "cuda":
            precision_context = tf32_context_manager(self.inference_settings.tf32)
        elif device_type == "xpu":
            precision_context = tf32_context_manager(
                self.inference_settings.series_b.onednn_tf32, device_type="xpu"
            )
        # Disable ambient autocast. Approximate modules choose their own operand
        # precision, leaving geometry, normalization and heads in FP32.
        autocast_context = (
            torch.autocast(device_type, enabled=False)
            if device_type == "xpu" or self.inference_settings.series_b.enabled
            else nullcontext()
        )
        from fairchem.core.models.uma.nn.approximate_ops import approximate_ops_context

        approximation_context = approximate_ops_context(
            self.inference_settings.series_b
        )
        with (
            inference_context,
            precision_context,
            autocast_context,
            approximation_context,
        ):
            output = inference_model(data)
            predictions = self._process_outputs(data, output, undo_refs)

        if self.inference_settings.series_b.enabled:
            for name, value in predictions.items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.is_floating_point()
                    and not torch.isfinite(value).all()
                ):
                    raise FloatingPointError(f"Series B produced nonfinite {name}")

        if self.inference_settings.nitro and not self._nitro_announced:
            print(_NITRO_BANNER, file=sys.stderr, flush=True)
            self._nitro_announced = True
        return predictions

    @property
    def approximation_manifest(self) -> dict:
        """
        Return the applied approximation coverage after the first prediction.
        """
        with _predictor_state(self)[0]:
            return copy.deepcopy(
                getattr(self.model.module, "_series_b_manifest", {"enabled": False})
            )

    def _process_outputs(self, data: AtomicData, output: dict, undo_refs: bool) -> dict:
        """
        Denormalize and post-process model outputs.
        """
        pred_output = {}
        datasets = set(data.dataset)
        active_tasks = [
            task
            for task in self.model.module.tasks.values()
            if datasets.intersection(getattr(task, "datasets", datasets))
        ]

        def check_finite(task, value, prop, level):
            if not self.assert_on_nans:
                return
            selected = [
                dataset in getattr(task, "datasets", datasets)
                for dataset in data.dataset
            ]
            if not all(selected):
                mask = torch.tensor(selected, dtype=torch.bool, device=value.device)
                if level == "atom":
                    mask = mask[data.batch.to(value.device)]
                value = value[mask]
            assert torch.isfinite(
                value
            ).all(), f"NaNs/Infs found in prediction for task {task.name}.{prop}"

        for task in active_tasks:
            task_name = task.name
            requested = data.get("_requested_properties", None)
            if requested is not None and task.property not in requested:
                continue
            # A model may omit tasks for datasets outside the current batch.
            if task_name not in output:
                continue
            value = output[task_name][task.property]
            pred_output[task_name] = (
                value * task.normalizer.rmsd
                if getattr(task, "inference_only", False)
                and task.property in _AUXILIARY_LEVELS.keys() - {"embeddings"}
                else task.normalizer.denorm(value)
            )
            check_finite(task, pred_output[task_name], task.property, task.level)
            if undo_refs and task.element_references is not None:
                pred_output[task_name] = task.element_references.undo_refs(
                    data, pred_output[task_name]
                )
        requested = data.get("_requested_properties", None)
        for task in active_tasks:
            if task.property != "energy":
                continue
            for prop in _AUXILIARY_LEVELS:
                key = task.name.removesuffix("energy") + prop
                source = (
                    "embeddings" if prop == "embeddings" and key not in output else key
                )
                if key in pred_output or source not in output:
                    continue
                if prop != "hessian_vector_product" and (
                    requested is None or prop not in requested
                ):
                    continue
                value = output[source]
                if isinstance(value, dict):
                    value = value[prop]
                # Derivatives scale by rmsd only; the energy mean and element
                # reference constants have zero coordinate derivative.
                pred_output[key] = (
                    value if prop == "embeddings" else value * task.normalizer.rmsd
                )
                check_finite(task, pred_output[key], prop, _AUXILIARY_LEVELS[prop])
        return pred_output


def move_tensors_to_cpu(data):
    """
    Recursively move all PyTorch tensors in a nested data structure to CPU.

    Args:
        data: Input data structure (dict, list, tuple, tensor, or other)

    Returns:
        Data structure with all tensors moved to CPU
    """
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: move_tensors_to_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_tensors_to_cpu(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_tensors_to_cpu(item) for item in data)
    else:
        # Return as-is for non-tensor types (str, int, float, etc.)
        return data


class MLIPWorkerLocal:
    def __init__(
        self,
        worker_id: int,
        world_size: int,
        predictor_config: dict,
        master_port: int | None = None,
        master_address: str | None = None,
        gp_config: GraphParallelConfig | None = None,
    ):
        self.worker_id = worker_id
        self.world_size = world_size
        self.predictor_config = predictor_config
        self.gp_config = gp_config
        self.master_address = (
            ray.util.get_node_ip_address() if master_address is None else master_address
        )
        self.master_port = get_free_port() if master_port is None else master_port
        self.is_setup = False
        self.last_received_atomic_data = None

    def get_master_address_and_port(self):
        return (self.master_address, self.master_port)

    def get_device_for_local_rank(self):
        return get_device_for_local_rank()

    def _distributed_setup(
        self,
    ):
        # initialize distributed environment
        # TODO, this won't work for multi-node, need to fix master addr
        logging.info(f"Initializing worker {self.worker_id}...")
        setup_env_local_multi_gpu(self.worker_id, self.master_port, self.master_address)

        device = self.predictor_config.get("device", "cpu")
        assign_device_for_local_rank(device == "cpu", 0)
        backend = "gloo" if device == "cpu" else "nccl"
        dist.init_process_group(
            backend=backend,
            rank=self.worker_id,
            world_size=self.world_size,
        )
        if self.gp_config is not None:
            gp_utils.setup_graph_parallel_groups(self.world_size, backend)
            gp_utils.set_gp_config(self.gp_config)
        self.predict_unit = hydra.utils.instantiate(self.predictor_config)
        self.device = get_device_for_local_rank()
        logging.info(
            f"Worker {self.worker_id}, gpu_id: {ray.get_gpu_ids()}, loaded predict unit: {self.predict_unit}, "
            f"on port {self.master_port}, with device: {self.device}, config: {self.predictor_config}"
        )
        self.is_setup = True

    def predict(
        self, data: AtomicData, use_nccl: bool = False
    ) -> dict[str, torch.tensor] | None:
        if not self.is_setup:
            self._distributed_setup()

        out = self.predict_unit.predict(data)
        if self.worker_id == 0:
            return move_tensors_to_cpu(out)

        if self.worker_id != 0 and use_nccl:
            self.last_received_atomic_data = data.to(self.device)
            while True:
                torch.distributed.broadcast(self.last_received_atomic_data.pos, src=0)
                torch.distributed.broadcast(self.last_received_atomic_data.cell, src=0)
                self.predict_unit.predict(self.last_received_atomic_data)

        return None


class MLIPWorker(MLIPWorkerLocal):
    pass


if ray_installed:
    MLIPWorker = ray.remote(MLIPWorker)
else:
    MLIPWorker = requires(
        ray_installed, message="Requires `ray[serve]` to be installed"
    )(MLIPWorker)


class ParallelMLIPPredictUnit(MLIPPredictUnitProtocol):
    @requires(ray_installed, message="Requires `ray[serve]` to be installed")
    def __init__(
        self,
        inference_model_path: str,
        device: str = "cpu",
        overrides: dict | None = None,
        inference_settings: InferenceSettings | None = None,
        seed: int = 41,
        atom_refs: dict | None = None,
        form_elem_refs: dict | None = None,
        assert_on_nans: bool = False,
        num_workers: int = 1,
        num_workers_per_node: int = 8,
        log_level: int = logging.INFO,
        gp_config: GraphParallelConfig | None = None,
    ):
        super().__init__()

        if torch.device(device).type == "xpu":
            raise ValueError("ParallelMLIPPredictUnit does not support XPU devices.")

        if inference_settings is not None and inference_settings.series_b.enabled:
            raise ValueError(
                "Series B currently requires independent single-worker predictors"
            )

        gp_config = gp_utils.resolve_gp_config_for_workers(gp_config, num_workers)

        if inference_settings is not None and inference_settings.nitro:
            without_nitro = dataclasses.replace(inference_settings, nitro=False)
            change = _describe_settings_change(
                inference_settings, without_nitro, device
            )
            logging.warning(
                "UMA Nitro does not support multi-worker or graph-parallel "
                f"inference. {change}"
            )
            inference_settings = without_nitro

        _mlip_pred_unit = MLIPPredictUnit(
            inference_model_path=inference_model_path,
            device="cpu",
            overrides=overrides,
            inference_settings=inference_settings,
            seed=seed,
            atom_refs=atom_refs,
            form_elem_refs=form_elem_refs,
        )
        if inference_settings is None:
            inference_settings = InferenceSettings()
        self.inference_settings = inference_settings
        self._dataset_to_tasks = copy.deepcopy(_mlip_pred_unit.dataset_to_tasks)
        self._validate_atoms_data_fn = _mlip_pred_unit.model.module.validate_atoms_data

        predict_unit_config = {
            "_target_": "fairchem.core.units.mlip_unit.predict.MLIPPredictUnit",
            "inference_model_path": inference_model_path,
            "device": device,
            "overrides": overrides,
            "inference_settings": inference_settings.to_omegaconf(),
            "seed": seed,
            "atom_refs": atom_refs,
            "form_elem_refs": form_elem_refs,
            "assert_on_nans": assert_on_nans,
        }

        logging.basicConfig(
            level=log_level,
            force=True,
            stream=sys.stdout,
            format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
        )
        # Optional: keep Ray/uvicorn chatty logs in check
        logging.getLogger("ray").setLevel(log_level)
        logging.getLogger("uvicorn").setLevel(log_level)
        if not ray.is_initialized():
            # in CI environment, we want to control the number of CPUs allocated to limit the pool of IDLE ray workers
            if os.environ.get("CI"):
                logging.info(
                    f"CI environment detected, initializing ray with limited CPUs: {num_workers_per_node}"
                )
                ray.init(
                    logging_level=log_level,
                    num_cpus=num_workers_per_node,
                    runtime_env={
                        "env_vars": {"RAY_DEBUG": "1"},
                    },
                )
            else:
                ray.init(logging_level=log_level)

        self.atomic_data_on_device = None

        num_nodes = math.ceil(num_workers / num_workers_per_node)
        num_workers_on_node_array = [num_workers_per_node] * num_nodes
        if num_workers % num_workers_per_node > 0:
            num_workers_on_node_array[-1] = num_workers % num_workers_per_node
        logging.info(
            f"Creating placement groups with {num_workers_on_node_array} workers on {device}"
        )

        # first create one placement group for each node
        num_gpu_per_worker = 1 if device == "cuda" else 0
        placement_groups = []
        for workers in num_workers_on_node_array:
            bundle = {"CPU": workers}
            if device == "cuda":
                bundle["GPU"] = workers
            pg = ray.util.placement_group([bundle], strategy="STRICT_PACK")
            placement_groups.append(pg)
        ray.get(pg.ready())  # Wait for each placement group to be scheduled

        # Need to still place worker to occupy space, otherwise ray double books this GPU
        rank0_worker = MLIPWorker.options(
            num_gpus=num_gpu_per_worker,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
            ),
        ).remote(0, num_workers, predict_unit_config, gp_config=gp_config)

        local_gpu_or_cpu = ray.get(rank0_worker.get_device_for_local_rank.remote())
        os.environ[CURRENT_DEVICE_TYPE_STR] = local_gpu_or_cpu

        self.workers = []
        self.local_rank0 = MLIPWorkerLocal(
            worker_id=0,
            world_size=num_workers,
            predictor_config=predict_unit_config,
            gp_config=gp_config,
        )
        master_addr, master_port = self.local_rank0.get_master_address_and_port()
        logging.info(f"Started rank0 on {master_addr}:{master_port}")

        # next place all ranks in order and pack them on placement groups
        # ie: rank0-7 -> placement group 0, 8->15 -> placement group 1 etc.
        worker_id = 0
        for pg_idx, pg in enumerate(placement_groups):
            workers = num_workers_on_node_array[pg_idx]
            logging.info(
                f"Launching workers for placement group {pg_idx} (Node {pg_idx}), workers={workers}"
            )

            for i in range(workers):
                # skip the first one because it's already been initialized above
                if pg_idx == 0 and i == 0:
                    worker_id += 1
                    continue
                # Each actor requests 1 worker worth of resources and uses the specific placement group
                actor = MLIPWorker.options(
                    num_gpus=num_gpu_per_worker,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=0,  # Use the first (and only) bundle in the PG
                        placement_group_capture_child_tasks=True,  # Ensure child tasks also run in this PG
                    ),
                ).remote(
                    worker_id,
                    num_workers,
                    predict_unit_config,
                    master_port,
                    master_addr,
                    gp_config=gp_config,
                )
                self.workers.append(actor)
                worker_id += 1

    def predict(self, data: AtomicData) -> dict[str, torch.tensor]:
        # put the reference in the object store only once
        if not self.inference_settings.merge_mole or self.atomic_data_on_device is None:
            data_ref = ray.put(data)
            # this will put the ray works into an infinite loop listening for broadcasts
            _futures = [
                w.predict.remote(data_ref, use_nccl=self.inference_settings.merge_mole)
                for w in self.workers
            ]
            self.atomic_data_on_device = data.clone()
        else:
            self.atomic_data_on_device.pos = data.pos.to(self.local_rank0.device)
            self.atomic_data_on_device.cell = data.cell.to(self.local_rank0.device)
            torch.distributed.broadcast(self.atomic_data_on_device.pos, src=0)
            torch.distributed.broadcast(self.atomic_data_on_device.cell, src=0)

        return self.local_rank0.predict(self.atomic_data_on_device)

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Delegates to the model's validate_atoms_data captured at init time.
        """
        self._validate_atoms_data_fn(atoms, task_name)

    @property
    def dataset_to_tasks(self) -> dict[str, list]:
        return self._dataset_to_tasks


_DEFAULT_BATCH_SERVER_TIMEOUT_S = 600.0


def _resolve_batch_server_timeout() -> float | None:
    """Read ``FAIRCHEM_BATCH_SERVER_TIMEOUT_S`` once per construction.

    Returns the float seconds to pass as ``DeploymentResponse.result``'s
    ``timeout_s`` (Ray Serve's blocking wait kwarg). ``"none"`` /
    ``"0"`` / negative values disable the bound. Malformed values fall
    back to the default with a warning so a typo can't silently revert
    to an indefinite wait.
    """
    raw = os.environ.get("FAIRCHEM_BATCH_SERVER_TIMEOUT_S")
    if raw is None:
        return _DEFAULT_BATCH_SERVER_TIMEOUT_S
    cleaned = raw.strip().lower()
    if cleaned in ("none", ""):
        return None
    try:
        val = float(cleaned)
    except ValueError:
        logging.warning(
            "FAIRCHEM_BATCH_SERVER_TIMEOUT_S=%r is not a number; "
            "falling back to default %.1fs.",
            raw,
            _DEFAULT_BATCH_SERVER_TIMEOUT_S,
        )
        return _DEFAULT_BATCH_SERVER_TIMEOUT_S
    if val <= 0:
        return None
    return val


class BatchServerPredictUnit(MLIPPredictUnitProtocol):
    """
    PredictUnit wrapper that uses Ray Serve for batched inference.

    This provides a clean interface compatible with MLIPPredictUnitProtocol
    while leveraging Ray Serve's batching capabilities under the hood.

    Works with both ``BatchPredictServer`` (single model) and
    ``MultiplexedBatchPredictServer`` (on-demand model loading). For
    multiplexed deployments, pass a ``ModelSpec`` to
    ``from_deployment_connection_info``. Its deterministic ``model_id`` is
    bound to the Ray Serve handle for replica-affinity routing, while the
    complete spec travels with every request.

    Can be constructed directly with a server handle, or via
    ``from_deployment_connection_info`` to connect to an already-running deployment.

    Every blocking ``.result()`` on a Ray Serve future is bounded by a
    per-call timeout sourced from the ``FAIRCHEM_BATCH_SERVER_TIMEOUT_S``
    environment variable (default ``600.0`` seconds). A wedged handle
    or unresponsive replica surfaces as a clean ``TimeoutError`` instead
    of an indefinite hang.  Use the env var to override; set to a
    floating-point number of seconds, or to ``"none"`` / ``"0"`` to
    disable the timeout entirely.
    """

    _handle_cache: ClassVar[dict[str, DeploymentHandle]] = {}

    @requires(ray_installed, message="Requires `ray[serve]` to be installed")
    def __init__(
        self,
        server_handle: DeploymentHandle,
        model_spec: ModelSpec | None = None,
    ):
        """
        Args:
            server_handle: Ray Serve deployment handle for a
                ``BatchPredictServer`` or ``MultiplexedBatchPredictServer``.
            model_spec: Typed model configuration for a multiplexed deployment.
                Its derived ``model_id`` is bound to the handle for routing, and
                the full spec is sent with each remote call.
        """
        self._request_timeout_s = _resolve_batch_server_timeout()
        if model_spec is not None:
            if not isinstance(model_spec, ModelSpec):
                raise TypeError(
                    f"model_spec must be a ModelSpec, got {type(model_spec).__name__}"
                )
            if not server_handle.is_multiplexed.remote().result(
                timeout_s=self._request_timeout_s
            ):
                raise ValueError(
                    f"model_spec={model_spec!r} was provided but the deployment "
                    "is not a multiplexed server. Use "
                    "MultiplexedBatchPredictServer or remove the model_spec argument."
                )
            server_handle = server_handle.options(
                multiplexed_model_id=model_spec.model_id
            )
        self.server_handle = server_handle
        self._model_spec = model_spec
        # A bounded, owned snapshot detects in-place metadata edits and prevents
        # a newly allocated dictionary from matching an old id. Array/tensor
        # metadata bypasses this optimization because external writes can occur.
        self._validated_info = None
        # Per-call ``.result()`` timeout (seconds). Sourced from
        # ``FAIRCHEM_BATCH_SERVER_TIMEOUT_S`` so callers can override
        # without code changes (tests can shrink it to fail fast;
        # users on slow CPUs can lengthen it). ``None`` disables the
        # bound entirely so behavior matches the pre-timeout default
        # if someone needs it.

    @property
    def model_spec(self) -> ModelSpec | None:
        """The model configuration sent to a multiplexed deployment."""
        return self._model_spec

    @property
    def multiplexed_model_id(self) -> str | None:
        """The identity token derived from ``model_spec`` for Serve routing."""
        return self._model_spec.model_id if self._model_spec is not None else None

    @classmethod
    def from_deployment_connection_info(
        cls,
        deployment_name: str = "predict-server",
        ray_address: str | None = None,
        namespace: str | None = None,
        model_spec: ModelSpec | None = None,
    ) -> BatchServerPredictUnit:
        """
        Connect to an already-running server by deployment name.

        Args:
            deployment_name: Name of the Ray Serve application.
            ray_address: Ray client address (e.g. ``ray://host:10001``).
                Falls back to ``RAY_ADDRESS`` env var. If *None* and env var
                is unset, assumes Ray is already initialised locally.
            namespace: Ray namespace. Falls back to
                ``RAY_NAMESPACE_SERVE_FAIRCHEM`` env var.
            model_spec: Typed model configuration for a multiplexed deployment.

        Returns:
            A ``BatchServerPredictUnit`` connected to the remote deployment.
        """
        if ray_address is None:
            ray_address = os.environ.get("RAY_ADDRESS")
        if namespace is None:
            namespace = os.environ.get("RAY_NAMESPACE_SERVE_FAIRCHEM")

        cache_key = f"{ray_address}|{namespace}|{deployment_name}"
        if cache_key not in cls._handle_cache:
            if ray_address and not ray.is_initialized():
                ray.init(ray_address, namespace=namespace)

            handle = get_app_handle_with_retry(deployment_name)
            cls._handle_cache[cache_key] = handle

        return cls(
            cls._handle_cache[cache_key],
            model_spec=model_spec,
        )

    def predict(self, data: AtomicData, undo_element_references: bool = True) -> dict:
        """
        Args:
            data: AtomicData object (single system)
            undo_element_references: Whether to undo element references

        Returns:
            Prediction dictionary
        """
        result = self.server_handle.remote(
            data,
            spec=self._model_spec,
            undo_element_references=undo_element_references,
        ).result(timeout_s=self._request_timeout_s)
        return result

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        """
        Validate and set defaults for calculator input data.

        Delegates to the server's predict unit so that validation is
        model-specific rather than hardcoded.  The server runs
        ``predict_unit.validate_atoms_data`` on a stub ``Atoms`` and
        returns the mutated ``atoms.info`` dict which is applied locally.

        Reuse validation only for the same dictionary, task and immutable
        metadata values. Charge/spin edits and mutable array values must be
        revalidated; validation can reject values, not just install defaults.
        """
        snapshot = _validation_info_snapshot(atoms.info)
        cached = self._validated_info
        if (
            snapshot is not None
            and cached is not None
            and cached[0] is atoms.info
            and cached[1] == task_name
            and cached[2] == snapshot
        ):
            return
        updated_info = self.server_handle.validate_atoms_data.remote(
            copy.deepcopy(atoms.info), task_name, spec=self._model_spec
        ).result(timeout_s=self._request_timeout_s)
        atoms.info.update(copy.deepcopy(updated_info))
        snapshot = _validation_info_snapshot(atoms.info)
        self._validated_info = (
            (atoms.info, task_name, snapshot) if snapshot is not None else None
        )

    @cached_property
    def dataset_to_tasks(self) -> dict:
        return self.server_handle.get_predict_unit_attribute.remote(
            "dataset_to_tasks", spec=self._model_spec
        ).result(timeout_s=self._request_timeout_s)

    @cached_property
    def atom_refs(self) -> dict | None:
        return self.server_handle.get_predict_unit_attribute.remote(
            "atom_refs", spec=self._model_spec
        ).result(timeout_s=self._request_timeout_s)

    @cached_property
    def inference_settings(self) -> InferenceSettings:
        return self.server_handle.get_predict_unit_attribute.remote(
            "inference_settings", spec=self._model_spec
        ).result(timeout_s=self._request_timeout_s)

    @cached_property
    def form_elem_refs(self) -> dict:
        return self.server_handle.get_predict_unit_attribute.remote(
            "form_elem_refs", spec=self._model_spec
        ).result(timeout_s=self._request_timeout_s)

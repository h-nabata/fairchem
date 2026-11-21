"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
from os import PathLike
from pathlib import Path
from threading import Condition, get_ident
from typing import TYPE_CHECKING
from zipfile import is_zipfile

import hydra
import torch
from omegaconf import DictConfig

from fairchem.core.common._model_state import MODEL_STATE_LOCK
from fairchem.core.common.precision import onednn_matmul_policy
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import load_state_dict, match_state_dict
from fairchem.core.models.uma.compat import apply_uma_compat_fixups

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint
    from fairchem.core.units.mlip_unit.mlip_unit import Task


def get_backbone_class_from_checkpoint(
    checkpoint: MLIPInferenceCheckpoint,
) -> type:
    """Extract the backbone class from a checkpoint's config."""
    backbone_config = checkpoint.model_config.get("backbone", {})
    backbone_model_name = backbone_config.get("model")

    if backbone_model_name is None:
        raise ValueError("Cannot determine backbone class from checkpoint config")

    return registry.get_model_class(backbone_model_name)


def load_inference_checkpoint(path):
    """Map tensor storages on demand for modern zip checkpoints; support legacy files."""
    use_mmap = (
        isinstance(path, (str, PathLike)) and Path(path).is_file() and is_zipfile(path)
    )
    return torch.load(path, map_location="cpu", weights_only=False, mmap=use_mmap)


@torch.inference_mode(False)
def load_inference_model(
    checkpoint_location: str,
    overrides: dict | None = None,
    use_ema: bool = False,
    return_checkpoint: bool = True,
    strict: bool = True,
    preloaded_checkpoint: MLIPInferenceCheckpoint | None = None,
) -> tuple[torch.nn.Module, MLIPInferenceCheckpoint] | torch.nn.Module:
    if preloaded_checkpoint is not None:
        # Compatibility fixups and overrides own their metadata. Parameters
        # remain shared read-only sources, avoiding a second checkpoint-sized
        # tensor allocation while allowing this checkpoint to be loaded again.
        checkpoint = copy(preloaded_checkpoint)
        checkpoint.model_config = deepcopy(preloaded_checkpoint.model_config)
    else:
        checkpoint = load_inference_checkpoint(checkpoint_location)

    apply_uma_compat_fixups(checkpoint, checkpoint_location=checkpoint_location)

    if overrides is not None:
        checkpoint.model_config = update_configs(checkpoint.model_config, overrides)

    # Loading on CPU must also hold for newly initialized/nonpersistent tensors.
    # map_location only controls checkpoint tensors, not the caller's ambient
    # torch default device (which may even be a meta construction context).
    with MODEL_STATE_LOCK, torch.device("cpu"):
        model = hydra.utils.instantiate(checkpoint.model_config)
    if use_ema:
        with MODEL_STATE_LOCK, torch.device("cpu"):
            # The checkpoint model is already private. Only construct the EMA
            # bookkeeping, avoiding a second full copy of its initialized tensors.
            averaged = torch.optim.swa_utils.AveragedModel(torch.nn.Identity())
            averaged.module = model
            model = averaged
        model_dict = model.state_dict()
        # Preserve module-version metadata used by PyTorch's checkpoint migration
        # hooks while removing the EMA counter only from our private mapping.
        ema_state_dict = copy(checkpoint.ema_state_dict)

        n_averaged = ema_state_dict["n_averaged"]
        del model_dict["n_averaged"]
        del ema_state_dict["n_averaged"]

        matched_dict = match_state_dict(model_dict, ema_state_dict)

        matched_dict["n_averaged"] = n_averaged

        load_state_dict(model, matched_dict, strict=strict)
    else:
        load_state_dict(model, checkpoint.model_state_dict, strict=strict)

    return (model, checkpoint) if return_checkpoint is True else model


def load_tasks(checkpoint_location: str) -> list[Task]:
    """
    Load tasks from a checkpoint file.

    Args:
        checkpoint_location (str): Path to the checkpoint file.

    Returns:
        list[Task]: A list of instantiated Task objects from the checkpoint's tasks_config.
    """
    checkpoint: MLIPInferenceCheckpoint = load_inference_checkpoint(checkpoint_location)
    with MODEL_STATE_LOCK, torch.device("cpu"), torch.inference_mode(False):
        return [
            hydra.utils.instantiate(task_config)
            for task_config in checkpoint.tasks_config
        ]


class _PrecisionGuard:
    """Strict XPU scopes can share a policy; CUDA and B03 writers need exclusive access."""

    def __init__(self):
        self.condition = Condition()
        self.readers: dict[int, int] = {}
        self.writer: int | None = None
        self.waiters = 0

    @contextmanager
    def __call__(self, readonly: bool):
        thread = get_ident()
        with self.condition:
            nested_writer = self.writer == thread
            if not nested_writer:
                if readonly:
                    self.condition.wait_for(
                        lambda: (
                            self.writer is None
                            and (not self.waiters or thread in self.readers)
                        )
                    )
                    self.readers[thread] = self.readers.get(thread, 0) + 1
                else:
                    if thread in self.readers:
                        raise RuntimeError(
                            "Cannot change precision inside a strict XPU inference scope"
                        )
                    self.waiters += 1
                    try:
                        self.condition.wait_for(
                            lambda: self.writer is None and not self.readers
                        )
                        self.writer = thread
                    finally:
                        self.waiters -= 1
        try:
            yield
        finally:
            with self.condition:
                if not nested_writer:
                    if readonly:
                        self.readers[thread] -= 1
                        if not self.readers[thread]:
                            del self.readers[thread]
                    else:
                        self.writer = None
                    self.condition.notify_all()


_precision_guard = _PrecisionGuard()


@contextmanager
def tf32_context_manager(tf32: bool, *, device_type: str = "cuda"):
    """
    Temporarily configure and restore TF32-related precision settings.

    Args:
        tf32: Whether to enable TF32 for float32 matmul and cuDNN operations.
    """
    with _precision_guard(readonly=device_type == "xpu" and not tf32):
        if device_type == "xpu":
            modern, legacy = onednn_matmul_policy()
            if not tf32 and (legacy or modern in ("tf32", "bf16")):
                raise ValueError(
                    "Strict FP32 XPU inference requires torch.backends.mkldnn.allow_tf32=False. "
                    "On newer PyTorch, set torch.backends.mkldnn.matmul.fp32_precision to ieee."
                )
            backend = torch.backends.mkldnn
            if modern is not None and (
                modern != "none" or not isinstance(legacy, bool)
            ):
                target, attribute, original, value = (
                    backend.matmul,
                    "fp32_precision",
                    modern,
                    "tf32",
                )
            else:
                target, attribute, original, value = backend, "allow_tf32", legacy, True
            if tf32 and original is None:
                raise RuntimeError("This PyTorch build does not expose XPU oneDNN TF32")
            try:
                if tf32 and value != original:
                    setattr(target, attribute, value)
                yield
            finally:
                if tf32 and value != original:
                    setattr(target, attribute, original)
            return
        if device_type != "cuda":
            raise ValueError("TF32 policy is only defined for CUDA or XPU")
        original_precision = torch.get_float32_matmul_precision()
        original_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
        configured_precision = "high" if tf32 else "highest"
        try:
            torch.backends.cudnn.allow_tf32 = tf32
            if configured_precision != original_precision:
                torch.set_float32_matmul_precision(configured_precision)
            yield
        finally:
            if configured_precision != original_precision:
                torch.set_float32_matmul_precision(original_precision)
            torch.backends.cudnn.allow_tf32 = original_allow_tf32_cudnn


def update_configs(original_config, new_config):
    updated_config = deepcopy(original_config)
    for k, v in new_config.items():
        is_dict_config = (
            k in updated_config
            and (isinstance(v, (dict, DictConfig)))
            and (isinstance(updated_config[k], (dict, DictConfig)))
        )
        if is_dict_config and k in updated_config:
            updated_config[k] = update_configs(updated_config[k], v)
        else:
            updated_config[k] = deepcopy(v)
    return updated_config

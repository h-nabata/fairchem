"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from pathlib import Path

    from torch.nn import Module
    from torch.utils.data import Dataset


_CHECKPOINT_LOAD_LOCK = threading.RLock()


def save_checkpoint(
    state,
    checkpoint_dir: str = "checkpoints/",
    checkpoint_file: str = "checkpoint.pt",
) -> str:
    filename = os.path.join(checkpoint_dir, checkpoint_file)
    torch.save(state, filename)
    return filename


def _load_check_duplicates(config: dict, name: str) -> dict[str, torch.nn.Module]:
    """Attempt to load a single file with normalizers/element references and check config for duplicate targets.

    Args:
        config: configuration dictionary
        name: Name of module to use for logging

    Returns:
        dictionary of normalizer or element reference modules
    """
    modules = {}
    if "file" in config:
        # These checkpoints contain modules written by save_checkpoint. Allow
        # only our statistic containers, without enabling arbitrary pickle code.
        # Local imports avoid the modules' imports of this helper at startup.
        from .element_references import ElementReferences, LinearReferences
        from .normalizer import Normalizer

        # safe_globals adds/removes process-wide entries without reference
        # counting. Retain prior caller registrations and serialize our loads.
        with _CHECKPOINT_LOAD_LOCK:
            existing = torch.serialization.get_safe_globals()
            temporary = [
                cls
                for cls in (Normalizer, LinearReferences, ElementReferences)
                if cls not in existing
            ]
            with torch.serialization.safe_globals(temporary):
                modules = torch.load(
                    config["file"], map_location="cpu", weights_only=True
                )
        logging.info(f"Loaded {name} for the following targets: {list(modules.keys())}")
    # make sure that element-refs are not specified both as fit and file
    fit_targets = config["fit"]["targets"] if "fit" in config else []
    duplicates = list(
        filter(
            lambda x: x in fit_targets,
            list(config) + list(modules.keys()),
        )
    )
    if len(duplicates) > 0:
        logging.warning(
            f"{name} values for the following targets {duplicates} have been specified to be fit and also read"
            f" from a file. The values read from the file will be used instead of fitting."
        )
    duplicates = list(filter(lambda x: x in modules, config))
    if len(duplicates) > 0:
        logging.warning(
            f"Duplicate {name} values for the following targets {duplicates} were specified in the file "
            f"{config['file']} and an explicitly set file. The normalization values read from "
            f"{config['file']} will be used."
        )
    return modules


def _load_from_config(
    config: dict,
    name: str,
    fit_fun: Callable[[list[str], Dataset, Any, ...], dict[str, Module]],
    create_fun: Callable[[str | Path], Module],
    dataset: Dataset,
    checkpoint_dir: str | Path | None = None,
    **fit_kwargs,
) -> dict[str, torch.nn.Module]:
    """Load or fit normalizers or element references from config

    If a fit is done, a fitted key with value true is added to the config to avoid re-fitting
    once a checkpoint has been saved.

    Args:
        config: configuration dictionary
        name: Name of module to use for logging
        fit_fun: Function to fit modules
        create_fun: Function to create a module from file
        checkpoint_dir: directory to save modules. If not given, modules won't be saved.

    Returns:
        dictionary of normalizer or element reference modules

    """
    modules = _load_check_duplicates(config, name)
    for target in config:
        if target == "fit":
            if not config["fit"].get("fitted", False):
                # remove values for output targets that have already been read from files
                targets = [
                    target
                    for target in config["fit"]["targets"]
                    if target not in modules
                ]
                fit_kwargs.update(
                    {
                        k: v
                        for k, v in config["fit"].items()
                        if k not in ("targets", "fitted")
                    }
                )
                modules.update(fit_fun(targets=targets, dataset=dataset, **fit_kwargs))
                config["fit"]["fitted"] = True
            else:
                configured_targets = set(config) - {"file", "fit"}
                missing = set(config["fit"]["targets"]) - (
                    set(modules) | configured_targets
                )
                if missing:
                    raise ValueError(
                        f"{name} are marked fitted but statistics for {sorted(missing)} "
                        "were not loaded. Provide their saved file or explicit values, "
                        "or set fit.fitted=False to fit again."
                    )
        # if a single file for all outputs is not provided,
        # then check if a single file is provided for a specific output
        elif target != "file":
            modules[target] = create_fun(**config[target])
        # save the linear references for possible subsequent use
        if checkpoint_dir is not None:
            path = save_checkpoint(
                modules,
                checkpoint_dir,
                f"{name}.pt",
            )
            logging.info(
                f"{name} checkpoint for targets {list(modules.keys())} have been saved to: {path}"
            )

    return modules

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Runtime exports are supplied by __getattr__; these imports serve type checkers.
# ruff: noqa: TCH004
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fairchem.core.calculate._batch import InferenceBatcher
    from fairchem.core.calculate._ray_inference_cluster import (
        get_local_inference_raycluster,
        get_slurm_inference_raycluster,
    )
    from fairchem.core.calculate.ase_calculator import (
        FAIRChemCalculator,
        FormationEnergyCalculator,
    )
    from fairchem.core.calculate.dftd3_calculator import DFTD3Calculator
    from fairchem.core.components.batch_server import (
        setup_batch_predict_server,
        setup_multiplexed_batch_predict_server,
    )
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    from fairchem.core.units.mlip_unit.api.model_spec import ModelSpec

__all__ = [
    "FAIRChemCalculator",
    "FormationEnergyCalculator",
    "DFTD3Calculator",
    "InferenceBatcher",
    "InferenceSettings",
    "ModelSpec",
    "get_local_inference_raycluster",
    "get_slurm_inference_raycluster",
    "setup_batch_predict_server",
    "setup_multiplexed_batch_predict_server",
]

_LAZY_IMPORTS = {
    "FAIRChemCalculator": (
        "fairchem.core.calculate.ase_calculator",
        "FAIRChemCalculator",
    ),
    "FormationEnergyCalculator": (
        "fairchem.core.calculate.ase_calculator",
        "FormationEnergyCalculator",
    ),
    "DFTD3Calculator": ("fairchem.core.calculate.dftd3_calculator", "DFTD3Calculator"),
    "InferenceBatcher": ("fairchem.core.calculate._batch", "InferenceBatcher"),
    "InferenceSettings": (
        "fairchem.core.units.mlip_unit.api.inference",
        "InferenceSettings",
    ),
    # ModelSpec is also re-exported by batch_server; use its defining module to
    # avoid loading the server deployments to construct or deserialize a spec.
    "ModelSpec": ("fairchem.core.units.mlip_unit.api.model_spec", "ModelSpec"),
    "get_local_inference_raycluster": (
        "fairchem.core.calculate._ray_inference_cluster",
        "get_local_inference_raycluster",
    ),
    "get_slurm_inference_raycluster": (
        "fairchem.core.calculate._ray_inference_cluster",
        "get_slurm_inference_raycluster",
    ),
    "setup_batch_predict_server": (
        "fairchem.core.components.batch_server",
        "setup_batch_predict_server",
    ),
    "setup_multiplexed_batch_predict_server": (
        "fairchem.core.components.batch_server",
        "setup_multiplexed_batch_predict_server",
    ),
}


def __getattr__(name: str):
    """
    Resolve optional calculation and serving APIs on first access.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """
    Keep all public exports discoverable before their first access.
    """
    return sorted(set(globals()) | set(__all__))

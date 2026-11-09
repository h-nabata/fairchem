"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Runtime exports are supplied by __getattr__; these imports serve type checkers.
# ruff: noqa: TCH004
import warnings
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from fairchem.core._config import clear_cache

if TYPE_CHECKING:
    from fairchem.core import calculate as calculate  # noqa: PLC0414
    from fairchem.core.calculate import pretrained_mlip
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.calculate.dftd3_calculator import DFTD3Calculator

# TODO: Remove this warning filter when torchtnt fixes pkg_resources deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=(
        "pkg_resources is deprecated as an API. "
        "See https://setuptools.pypa.io/en/latest/pkg_resources.html."
    ),
    category=UserWarning,
)

try:
    __version__ = version("fairchem.core")
except PackageNotFoundError:
    __version__ = ""

__all__ = [
    "FAIRChemCalculator",
    "DFTD3Calculator",
    "pretrained_mlip",
    "clear_cache",
]

# Importing registry/configuration helpers must not initialize PyTorch or serving
# dependencies. Resolve public classes on first access, then retain the original
# objects so that identity checks, pickling, and subsequent imports are unchanged.
_LAZY_IMPORTS = {
    "FAIRChemCalculator": (
        "fairchem.core.calculate.ase_calculator",
        "FAIRChemCalculator",
    ),
    "DFTD3Calculator": ("fairchem.core.calculate.dftd3_calculator", "DFTD3Calculator"),
    "pretrained_mlip": ("fairchem.core.calculate.pretrained_mlip", None),
    "calculate": ("fairchem.core.calculate", None),
}


def __getattr__(name: str):
    """
    Load a public calculation entry point only when it is requested.
    """
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """
    Include lazy public names without importing their dependencies.
    """
    return sorted(set(globals()) | _LAZY_IMPORTS.keys())

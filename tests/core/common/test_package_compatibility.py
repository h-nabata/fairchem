"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def _project(package):
    root = Path(__file__).resolve().parents[3]
    with (root / "packages" / package / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]


def _requirements(requirements):
    return {Requirement(requirement).name: requirement for requirement in requirements}


def test_numpy126_variant_retains_core_runtime_and_optional_dependencies():
    """
    The alternate wheel loads the same modules and exposes the same extras.

    A missing direct dependency such as backoff prevents even public UMA imports
    in an otherwise valid clean installation. Only the NumPy constraint differs.
    """
    core = _project("fairchem-core")
    variant = _project("fairchem-core-numpy126")
    core_requirements = _requirements(core["dependencies"])
    variant_requirements = _requirements(variant["dependencies"])
    assert core_requirements.pop("numpy") == "numpy>=2.0,<2.5"
    assert variant_requirements.pop("numpy") == "numpy>=1.26.4,<2"
    assert variant_requirements == core_requirements
    assert variant["optional-dependencies"] == core["optional-dependencies"]
    assert "backoff" in variant_requirements
    assert "ray" not in variant_requirements


@pytest.mark.parametrize(
    ("version", "supported"),
    [("3.10", False), ("3.11", True), ("3.12", True), ("3.13", False)],
)
def test_numpy126_python_range_matches_core_and_numpy_support(version, supported):
    project = _project("fairchem-core-numpy126")
    assert (version in SpecifierSet(project["requires-python"])) is supported


@pytest.mark.parametrize("package", ["fairchem-core", "fairchem-core-numpy126"])
def test_existing_supported_xpu_wheel_satisfies_package_requirement(package):
    requirements = _requirements(_project(package)["dependencies"])
    torch_requirement = Requirement(requirements["torch"])
    # Installing the editable fork after the XPU wheel must not force a CUDA
    # replacement merely because PyTorch encodes the backend as a local version.
    assert "2.15.0+xpu" in torch_requirement.specifier
    assert "2.16.0+xpu" not in torch_requirement.specifier

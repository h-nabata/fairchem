"""Compute local versions for dependency-specific builds of fairchem-core."""

from email.parser import Parser
from pathlib import Path

from setuptools_scm import get_version


def variant_version(variant):
    # Hatch loads this module through the project-local path. In a checkout
    # it is a symlink; in an sdist force-include embeds the same source.
    project = Path(__file__).absolute().parent
    metadata = project / "PKG-INFO"
    if metadata.is_file():
        return Parser().parsestr(metadata.read_text(encoding="utf-8"))["Version"]
    base = get_version(
        root=str(project.parent.parent),
        git_describe_command="git describe --tags --match fairchem_core-*",
    )
    return base + ("." if "+" in base else "+") + variant

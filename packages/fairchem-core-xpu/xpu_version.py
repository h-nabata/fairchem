"""Give the Intel build a PEP 440 local version for shared wheelhouses."""

from email.parser import Parser
from pathlib import Path

from setuptools_scm import get_version

project = Path(__file__).resolve().parent
metadata = project / "PKG-INFO"
if metadata.is_file():
    __version__ = Parser().parsestr(metadata.read_text(encoding="utf-8"))["Version"]
else:
    base = get_version(
        root=str(project.parent.parent),
        git_describe_command="git describe --tags --match fairchem_core-*",
    )
    __version__ = base + (".xpu" if "+" in base else "+xpu")

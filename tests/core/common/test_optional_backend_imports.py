"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from fairchem.core.common import utils
from fairchem.core.common.registry import registry


def test_uma_import_does_not_require_cuda_extensions():
    """An installed but broken CUDA extension must not prevent portable UMA use."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        import torch

        # Test fairchem's imports, after PyTorch's own optional CUDA discovery.
        blocked_roots = {"cuda", "cutlass", "fairchem_cpp", "nvalchemiops", "warp"}
        for name in tuple(sys.modules):
            if name.split(".")[0] in blocked_roots:
                del sys.modules[name]

        class UnusableCudaExtensions(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked_roots:
                    raise OSError("CUDA shared library unavailable: " + fullname)

        sys.meta_path.insert(0, UnusableCudaExtensions())
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.models.uma.escn_md import eSCNMDBackbone
        from fairchem.core.common.registry import registry
        from fairchem.core.common.utils import setup_imports

        setup_imports({"skip_experimental_imports": True})
        assert registry.get_model_class("escnmd_backbone") is eSCNMDBackbone
        assert callable(pretrained_mlip.get_predict_unit)
        assert FAIRChemCalculator is not None
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_model_discovery_does_not_import_cuda_kernel_implementations():
    """Model registration must work even with unusable accelerator extensions."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class UnusableCudaKernels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                roots = (
                    "fairchem.core.models.uma.triton",
                    "fairchem.core.models.uma.uma_nitro",
                )
                if any(fullname == root or fullname.startswith(root + ".") for root in roots):
                    raise RuntimeError("CUDA implementation imported: " + fullname)

        sys.meta_path.insert(0, UnusableCudaKernels())
        from fairchem.core.common.registry import registry
        from fairchem.core.common.utils import setup_imports

        setup_imports({"skip_experimental_imports": True})
        assert registry.get_model_class("escnmd_backbone") is not None
        assert registry.get("imports_setup", no_warning=True)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_failed_model_discovery_can_be_retried(monkeypatch):
    """An import failure must not leave the registry marked as initialized."""
    monkeypatch.setitem(registry.mapping, "state", {})
    attempts = 0

    def import_with_first_attempt_failure(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("optional shared library is unavailable")

    monkeypatch.setattr(utils, "_import_local_file", import_with_first_attempt_failure)
    with pytest.raises(OSError, match="optional shared library"):
        utils.setup_imports({"skip_experimental_imports": True})
    assert not registry.get("imports_setup", no_warning=True)

    utils.setup_imports({"skip_experimental_imports": True})
    assert attempts > 1
    assert registry.get("imports_setup", no_warning=True)

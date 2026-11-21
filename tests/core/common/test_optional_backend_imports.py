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


@pytest.mark.parametrize(
    "module_name",
    ["fairchem.core", "fairchem.core.calculate", "fairchem.core.common.registry"],
)
def test_configuration_imports_do_not_initialize_compute_or_serving(module_name):
    """
    Registry/configuration imports and introspection must stay lightweight.
    """
    script = textwrap.dedent(
        f"""
        import importlib
        import importlib.abc
        import sys

        class UnexpectedHeavyImport(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {{"torch", "ray", "wandb"}}:
                    raise AssertionError("unexpected dependency: " + fullname)

        sys.meta_path.insert(0, UnexpectedHeavyImport())
        module = importlib.import_module({module_name!r})
        for name in getattr(module, "__all__", ()):
            assert name in dir(module)
        try:
            module.not_a_fairchem_export
        except AttributeError as exc:
            assert "not_a_fairchem_export" in str(exc)
            assert module.__name__ in str(exc)
        else:
            raise AssertionError("Unknown attributes must raise AttributeError")
        assert not any(name in sys.modules for name in ("torch", "ray", "wandb"))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("leaf_first", [False, True])
def test_lazy_public_exports_preserve_identity_and_pickle(leaf_first):
    """
    Both import orders and star imports must expose the original public objects.
    """
    script = textwrap.dedent(
        """
        import importlib
        import pickle
        import sys
        import types

        exports = {
            "fairchem.core": {
                "FAIRChemCalculator": ("fairchem.core.calculate.ase_calculator", "FAIRChemCalculator"),
                "DFTD3Calculator": ("fairchem.core.calculate.dftd3_calculator", "DFTD3Calculator"),
                "pretrained_mlip": ("fairchem.core.calculate.pretrained_mlip", None),
                "clear_cache": ("fairchem.core._config", "clear_cache"),
            },
            "fairchem.core.calculate": {
                "FAIRChemCalculator": ("fairchem.core.calculate.ase_calculator", "FAIRChemCalculator"),
                "FormationEnergyCalculator": ("fairchem.core.calculate.ase_calculator", "FormationEnergyCalculator"),
                "DFTD3Calculator": ("fairchem.core.calculate.dftd3_calculator", "DFTD3Calculator"),
                "InferenceBatcher": ("fairchem.core.calculate._batch", "InferenceBatcher"),
                "InferenceSettings": ("fairchem.core.units.mlip_unit.api.inference", "InferenceSettings"),
                "ModelSpec": ("fairchem.core.units.mlip_unit.api.model_spec", "ModelSpec"),
                "get_local_inference_raycluster": ("fairchem.core.calculate._ray_inference_cluster", "get_local_inference_raycluster"),
                "get_slurm_inference_raycluster": ("fairchem.core.calculate._ray_inference_cluster", "get_slurm_inference_raycluster"),
                "setup_batch_predict_server": ("fairchem.core.components.batch_server", "setup_batch_predict_server"),
                "setup_multiplexed_batch_predict_server": ("fairchem.core.components.batch_server", "setup_multiplexed_batch_predict_server"),
            },
        }
        leaf_first = sys.argv[1] == "True"
        if leaf_first:
            for names in reversed(list(exports.values())):
                for module_name, _ in reversed(list(names.values())):
                    importlib.import_module(module_name)
        else:
            from fairchem.core.calculate import ModelSpec
            assert "fairchem.core.components.batch_server" not in sys.modules

        for package_name, names in exports.items():
            package = importlib.import_module(package_name)
            assert set(package.__all__) == set(names)
            assert set(names) <= set(dir(package))
            namespace = {}
            exec("from " + package_name + " import *", namespace)
            assert set(namespace) - {"__builtins__"} == set(names)
            for name, (module_name, attribute) in names.items():
                module = importlib.import_module(module_name)
                canonical = module if attribute is None else getattr(module, attribute)
                assert getattr(package, name) is canonical
                assert getattr(package, name) is namespace[name]
                if not isinstance(canonical, types.ModuleType):
                    assert pickle.loads(pickle.dumps(canonical)) is canonical

        import fairchem.core as core
        import fairchem.core.calculate as calculate
        from fairchem.core.components.batch_server import ModelSpec as ServerSpec
        assert core.calculate is calculate
        assert core.pretrained_mlip is calculate.pretrained_mlip
        assert calculate.ModelSpec is ServerSpec
        spec = calculate.ModelSpec("test-model", source="registry", device="cpu")
        restored = pickle.loads(pickle.dumps(spec))
        assert type(restored) is ServerSpec
        assert restored.model_id == spec.model_id
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(leaf_first)],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_uma_import_without_optional_application_packages():
    """
    A base installation must not require serving or scientific-workflow extras.
    """
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys
        import torch

        blocked = {"ray", "pymatgen", "pyarrow", "pandas", "quacc", "torch_sim"}
        for name in tuple(sys.modules):
            if name.split(".")[0] in blocked:
                del sys.modules[name]

        class MissingOptionalPackages(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise ModuleNotFoundError("optional package absent: " + fullname)

        # An absent package's discovery probe returns None; import raises.
        # In particular, wandb probes pandas before deciding to use it.
        original_find_spec = importlib.util.find_spec
        def find_spec(name, *args, **kwargs):
            if name.split(".")[0] in blocked:
                return None
            return original_find_spec(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        sys.meta_path.insert(0, MissingOptionalPackages())
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        from fairchem.core.calculate import InferenceBatcher
        from fairchem.core.common.registry import registry
        from fairchem.core.common.utils import setup_imports

        setup_imports({"skip_experimental_imports": True})
        assert registry.get_model_class("escnmd_backbone") is not None
        assert callable(pretrained_mlip.get_predict_unit)
        assert FAIRChemCalculator is not None
        try:
            InferenceBatcher(None)
        except RuntimeError as exc:
            assert "ray[serve]" in str(exc)
        else:
            raise AssertionError("Missing Ray must fail at the serving entry point")
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

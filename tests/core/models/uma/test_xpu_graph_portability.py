"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import builtins
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from ase import Atoms
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.graph.compute import generate_graph
from fairchem.core.graph.radius_graph_pbc import canonical_pbc, radius_graph_pbc_v2
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.nn import layer_norm, mole


def _single_atom(pbc):
    return AtomicData.from_ase(
        Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=pbc),
        task_name="omat",
    )


@pytest.mark.parametrize(
    ("stored_pbc", "override", "expected_edges"),
    [
        (False, [True, True, True], 6),
        (True, [False, False, False], 0),
        (False, [True, False, False], 2),
    ],
)
def test_graph_v2_respects_explicit_pbc(stored_pbc, override, expected_edges):
    data = _single_atom(stored_pbc)
    stored = data.pbc.clone()
    edges, offsets, neighbors = radius_graph_pbc_v2(
        data, 2.1, 100, pbc=torch.tensor(override)
    )
    assert edges.shape == (2, expected_edges)
    assert offsets.shape == (expected_edges, 3)
    assert neighbors.tolist() == [expected_edges]
    torch.testing.assert_close(data.pbc, stored)


def test_graph_v2_respects_mixed_pbc_override():
    data = atomicdata_list_to_batch([_single_atom(False), _single_atom(True)])
    override = torch.tensor([[True, True, True], [False, False, False]])
    edges, _, neighbors = radius_graph_pbc_v2(data, 2.1, 100, pbc=override)
    assert neighbors.tolist() == [6, 0]
    assert (edges == 0).all()


def test_backbone_always_pbc_uses_system_count():
    # The graph method needs no weights: exercise the actual backbone dispatch
    # with an AtomicData whose len() is its number of fields, not systems.
    backbone = SimpleNamespace(
        otf_graph=True,
        always_use_pbc=True,
        cutoff=2.1,
        max_neighbors=100,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=2,
    )
    data = atomicdata_list_to_batch([_single_atom(False), _single_atom(False)])
    graph = eSCNMDBackbone._generate_graph(backbone, data)
    assert graph["neighbors"].tolist() == [6, 6]
    assert graph["edge_index"].shape == (2, 12)


@pytest.mark.parametrize("device", ["cpu", "xpu:0"])
def test_canonical_pbc_keeps_reduction_on_input_device(monkeypatch, device):
    # FakeTensor verifies placement only, not values or XPU kernel execution.
    with FakeTensorMode(allow_fallback_kernels=False):
        flags = torch.empty((2, 3), dtype=torch.bool, device=device)
        data = SimpleNamespace(pbc=flags)

        def read_flags(tensor):
            assert tensor.device == flags.device
            assert tensor.dtype == torch.bool
            assert tensor.shape == (3,)
            return [True, False, True]

        monkeypatch.setattr(type(flags), "tolist", read_flags)
        assert canonical_pbc(data, None) == [True, False, True]


def test_graph_v3_rejects_xpu_before_optional_import(monkeypatch):
    original_import = builtins.__import__

    def block_nvidia(name, *args, **kwargs):
        if name.endswith("radius_graph_pbc_nvidia"):
            pytest.fail("XPU graph v3 reached the optional NVIDIA import")
        return original_import(name, *args, **kwargs)

    with FakeTensorMode(allow_fallback_kernels=False):
        data = SimpleNamespace(pos=torch.empty((1, 3), device="xpu:0"))
        monkeypatch.setattr(builtins, "__import__", block_nvidia)
        with pytest.raises(ValueError, match="radius_pbc_version=2"):
            generate_graph(data, 3.0, 100, False, 3, None)


@pytest.mark.parametrize(
    "norm_class",
    [
        layer_norm.EquivariantLayerNormArray,
        layer_norm.EquivariantLayerNormArraySphericalHarmonics,
        layer_norm.EquivariantRMSNormArraySphericalHarmonics,
        layer_norm.EquivariantRMSNormArraySphericalHarmonicsV2,
    ],
)
def test_norm_disables_xpu_autocast_and_restores_it(monkeypatch, norm_class):
    kwargs = {"affine": False}
    if norm_class in (
        layer_norm.EquivariantLayerNormArraySphericalHarmonics,
        layer_norm.EquivariantRMSNormArraySphericalHarmonicsV2,
    ):
        kwargs["std_balance_degrees"] = False
    norm = norm_class(2, 4, **kwargs)
    mean = torch.mean
    states = []

    def check_mean(*args, **kwargs):
        states.append(torch.is_autocast_enabled("xpu"))
        return mean(*args, **kwargs)

    monkeypatch.setattr(torch, "mean", check_mean)
    with FakeTensorMode(allow_fallback_kernels=False):
        x = torch.empty((3, 9, 4), device="xpu:0")
        with torch.autocast("xpu", dtype=torch.bfloat16):
            result = norm(x)
            assert torch.is_autocast_enabled("xpu")
        assert result.device == x.device
        assert result.dtype == torch.float32
    assert states
    assert not any(states)


@pytest.mark.parametrize(
    "norm_class",
    [
        layer_norm.EquivariantLayerNormArray,
        layer_norm.EquivariantLayerNormArraySphericalHarmonics,
        layer_norm.EquivariantRMSNormArraySphericalHarmonics,
        layer_norm.EquivariantRMSNormArraySphericalHarmonicsV2,
    ],
)
def test_norm_device_autocast_scope_is_traceable(norm_class):
    # Eager backend tests Dynamo tracing and gradients, not an XPU compiler.
    norm = norm_class(2, 4)
    x = torch.randn(3, 9, 4, requires_grad=True)
    compiled = torch.compile(norm, backend="eager", fullgraph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        reference = norm(x)
        result = compiled(x)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, reference)
    reference_grad = torch.autograd.grad(reference.square().sum(), x)[0]
    result_grad = torch.autograd.grad(result.square().sum(), x)[0]
    torch.testing.assert_close(result_grad, reference_grad)


@pytest.mark.parametrize("error_type", [ImportError, OSError])
def test_dgl_reports_broken_optional_extension(monkeypatch, error_type):
    original_import = builtins.__import__

    def broken_extension(name, *args, **kwargs):
        if name == "fairchem_cpp":
            raise error_type("incompatible binary")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_extension)
    mole._load_fairchem_cpp.cache_clear()
    globals_ = mole.MOLEGlobals(torch.ones(1, 2), torch.tensor([3]))
    layer = mole.MOLEDGL(2, 4, 4, globals_, bias=True)
    with pytest.raises(ImportError, match="mole_layer_type='pytorch'") as exc:
        layer(torch.ones(3, 4))
    assert isinstance(exc.value.__cause__, error_type)
    portable = mole.MOLE(2, 4, 4, globals_, bias=True)
    assert torch.isfinite(portable(torch.ones(3, 4))).all()


def test_portable_uma_import_ignores_broken_cpp_extension():
    script = """
import importlib.abc
import sys

class BrokenExtension(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fairchem_cpp' or fullname.startswith('fairchem_cpp.'):
            raise OSError('CUDA/PyTorch ABI mismatch in optional extension')

sys.meta_path.insert(0, BrokenExtension())
from fairchem.core.models.uma.escn_moe import eSCNMDMoeBackbone
assert 'fairchem_cpp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

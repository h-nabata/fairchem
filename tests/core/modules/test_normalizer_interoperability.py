"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.modules.normalization._load_utils import save_checkpoint
from fairchem.core.modules.normalization.element_references import (
    LinearReferences,
    load_references_from_config,
)
from fairchem.core.modules.normalization.normalizer import (
    Normalizer,
    create_normalizer,
    fit_normalizers,
    load_normalizers_from_config,
)


@pytest.mark.parametrize("value", [0, np.int64(0), torch.tensor(0)])
def test_integer_statistics_use_floating_buffers_and_load_fractional_values(value):
    normalizer = Normalizer(mean=value, rmsd=1)
    assert normalizer.mean.dtype == normalizer.rmsd.dtype == torch.get_default_dtype()
    normalizer.load_state_dict({"mean": torch.tensor(0.25), "rmsd": torch.tensor(1.5)})
    torch.testing.assert_close(normalizer.norm(torch.tensor(3.25)), torch.tensor(2.0))


@pytest.mark.parametrize("value", [True, np.bool_(True), 1j, np.array([1j]), "1"])
def test_nonreal_statistics_are_rejected(value):
    with pytest.raises(TypeError, match="real numbers"):
        Normalizer(mean=value)


def test_statistics_own_tensor_numpy_storage_without_retaining_training_tapes():
    mean = torch.tensor([1.123456789], dtype=torch.float64, requires_grad=True) * 2
    rmsd = np.array([3.123456789], dtype=np.float64)
    normalizer = Normalizer(mean=mean, rmsd=rmsd)
    assert normalizer.mean.dtype == normalizer.rmsd.dtype == torch.float64
    assert normalizer.mean.grad_fn is None
    assert not normalizer.mean.requires_grad
    before = copy.deepcopy(normalizer.state_dict())
    mean.detach().zero_()
    rmsd[:] = 0
    torch.testing.assert_close(normalizer.mean, before["mean"], rtol=0, atol=0)
    torch.testing.assert_close(normalizer.rmsd, before["rmsd"], rtol=0, atol=0)
    copy.deepcopy(normalizer)


@pytest.mark.parametrize("legacy", [False, True])
def test_factory_checkpoint_preserves_vector_shape_precision_and_ownership(legacy):
    scale_name = "std" if legacy else "rmsd"
    state = {
        "mean": torch.tensor([1.123456789, 2.987654321], dtype=torch.float64),
        scale_name: torch.tensor([2.123456789, 3.987654321], dtype=torch.float64),
    }
    normalizer = create_normalizer(state_dict=state)
    assert set(state) == {"mean", scale_name}
    assert normalizer.mean.dtype == normalizer.rmsd.dtype == torch.float64
    torch.testing.assert_close(normalizer.mean, state["mean"], rtol=0, atol=0)
    torch.testing.assert_close(normalizer.rmsd, state[scale_name], rtol=0, atol=0)
    state["mean"].zero_()
    state[scale_name].zero_()
    assert bool((normalizer.mean > 0).all())
    assert bool((normalizer.rmsd > 0).all())


def test_assigned_checkpoint_buffers_are_owned_and_mapping_remains_unchanged():
    state = {"mean": np.float64(0.25), "std": torch.tensor(1.5)}
    normalizer = Normalizer()
    normalizer.load_state_dict(state, assign=True)
    assert set(state) == {"mean", "std"}
    assert normalizer.mean.dtype == torch.float64
    state["std"].zero_()
    assert normalizer.rmsd.item() == 1.5


def test_checkpoint_strict_keys_are_still_validated():
    with pytest.raises(RuntimeError, match="Unexpected key"):
        create_normalizer(state_dict={"mean": 0.0, "std": 1.0, "typo": 3})
    with pytest.raises(RuntimeError, match="Missing key"):
        create_normalizer(state_dict={"mean": 0.0})


def test_checkpoint_accepts_readonly_mapping():
    state = MappingProxyType({"mean": 0.25, "rmsd": 1.5})
    normalizer = Normalizer()
    normalizer.load_state_dict(state)
    assert normalizer.mean.item() == 0.25
    assert normalizer.rmsd.item() == 1.5


@pytest.mark.parametrize("legacy", [False, True])
def test_npz_vector_statistics_and_modern_key_precedence(tmp_path, legacy):
    file = tmp_path / "normalizer.npz"
    values = {"mean": np.array([1.0, 2.0]), "std": np.array([3.0, 4.0])}
    if not legacy:
        values["rmsd"] = np.array([0.0, 5.0])
    np.savez(file, **values)
    normalizer = create_normalizer(file=file)
    expected = values["std"] if legacy else values["rmsd"]
    np.testing.assert_array_equal(normalizer.rmsd.numpy(), expected)
    np.testing.assert_array_equal(normalizer.mean.numpy(), values["mean"])


def test_npz_file_takes_priority_over_supplied_state_dict(tmp_path):
    file = tmp_path / "normalizer.npz"
    np.savez(file, mean=0.25, rmsd=1.5)
    normalizer = create_normalizer(file=file, state_dict={"mean": 20.0, "rmsd": 4.0})
    assert normalizer.mean.item() == 0.25
    assert normalizer.rmsd.item() == 1.5


def test_pt_statistics_load_on_cpu_without_cuda_checkpoint_dependency(
    tmp_path, monkeypatch
):
    file = tmp_path / "normalizer.pt"
    state = {"mean": torch.tensor([1.0, 2.0]), "rmsd": torch.tensor([3.0, 4.0])}
    torch.save(state, file)
    original_load = torch.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs.get("map_location"))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", load)
    normalizer = create_normalizer(file=file)
    assert calls == ["cpu"]
    torch.testing.assert_close(normalizer.rmsd, state["rmsd"])


def _dataset():
    data = []
    for energy, force in ((1.0, -1.0), (3.0, 1.0)):
        item = AtomicData.from_ase(Atoms("H", positions=[[0, 0, 0]]))
        item.energy = torch.tensor([energy])
        item.forces = torch.tensor([[force, 0.0, 0.0]])
        data.append(item)
    return data


def test_fit_accepts_default_override_values():
    normalizers = fit_normalizers(["energy"], _dataset(), batch_size=2, shuffle=False)
    torch.testing.assert_close(normalizers["energy"].mean, torch.tensor(2.0))
    torch.testing.assert_close(normalizers["energy"].rmsd, torch.tensor(2.0).sqrt())


@pytest.mark.parametrize("targets", [["energy", "forces"], ["forces", "energy"]])
def test_fit_correction_is_independent_of_target_order(targets):
    overrides = {"forces": {"mean": 0.0}}
    normalizers = fit_normalizers(
        targets,
        _dataset(),
        batch_size=2,
        shuffle=False,
        override_values=overrides,
    )
    assert overrides == {"forces": {"mean": 0.0}}
    torch.testing.assert_close(normalizers["energy"].rmsd, torch.tensor(2.0).sqrt())
    torch.testing.assert_close(normalizers["forces"].rmsd, torch.tensor(1.0))


def test_fit_explicit_correction_applies_to_every_target():
    normalizers = fit_normalizers(
        ["energy", "forces"],
        _dataset(),
        batch_size=2,
        shuffle=False,
        override_values={"forces": {"mean": 0.0}},
        rmsd_correction=0,
    )
    assert all(normalizer.rmsd.item() == 1.0 for normalizer in normalizers.values())


def test_config_fitted_false_is_control_metadata_and_reuse_fails_clearly():
    config = {"fit": {"targets": ["energy"], "batch_size": 2, "fitted": False}}
    result = load_normalizers_from_config(config, _dataset())
    assert result["energy"].mean.item() == 2.0
    assert config["fit"]["fitted"] is True
    with pytest.raises(ValueError, match="marked fitted.*energy"):
        load_normalizers_from_config(config, _dataset())


def test_fitted_config_can_restore_explicit_statistics():
    config = {
        "fit": {"targets": ["energy"], "fitted": True},
        "energy": {"mean": 2.0, "rmsd": 1.5},
    }
    result = load_normalizers_from_config(config, _dataset())
    assert result["energy"].rmsd.item() == 1.5


@pytest.mark.parametrize("kind", ["normalizer", "references"])
def test_fitted_config_uses_modules_loaded_from_saved_file(tmp_path, monkeypatch, kind):
    values = torch.tensor([1.123456789, 2.987654321], dtype=torch.float64)
    if kind == "normalizer":
        module = Normalizer(mean=values, rmsd=values + 1)
        load = load_normalizers_from_config
    else:
        module = LinearReferences(element_references=values)
        load = load_references_from_config
    file = Path(save_checkpoint({"energy": module}, tmp_path))
    original_bytes = file.read_bytes()
    prior_safe_globals = set(torch.serialization.get_safe_globals())
    original_load = torch.load
    calls = []

    def checked_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", checked_load)
    config = {"file": file, "fit": {"targets": ["energy"], "fitted": True}}
    result = load(config, [])
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert isinstance(result["energy"], type(module))
    assert result["energy"] is not module
    for name, tensor in result["energy"].state_dict().items():
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64
        assert tensor.shape == (2,)
        assert tensor.data_ptr() != module.state_dict()[name].data_ptr()
        torch.testing.assert_close(tensor, module.state_dict()[name], rtol=0, atol=0)
    assert file.read_bytes() == original_bytes
    assert set(torch.serialization.get_safe_globals()) == prior_safe_globals


def test_checkpoint_loading_preserves_existing_user_allowlist(tmp_path):
    file = save_checkpoint({"energy": Normalizer()}, tmp_path)
    with torch.serialization.safe_globals([Normalizer]):
        existing = set(torch.serialization.get_safe_globals())
        loaded = load_normalizers_from_config({"file": file}, [])
        assert isinstance(loaded["energy"], Normalizer)
        assert set(torch.serialization.get_safe_globals()) == existing

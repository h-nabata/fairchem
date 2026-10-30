"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch
from ase.build import molecule

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.base import HydraModel
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu
from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit, _ModelWrapper


def _small_predictor(device="cpu"):
    torch.manual_seed(42)
    model = HydraModel(
        backbone={
            "model": "fairchem.core.models.uma.escn_moe.eSCNMDMoeBackbone",
            "sphere_channels": 8,
            "hidden_channels": 16,
            "edge_channels": 8,
            "num_distance_basis": 8,
            "num_layers": 1,
            "num_experts": 2,
            "lmax": 2,
            "mmax": 2,
            "otf_graph": True,
            "cutoff": 3.0,
            "use_dataset_embedding": True,
            "dataset_mapping": {"omol": "omol", "omat": "omat"},
            "use_composition_embedding": True,
            "always_use_pbc": False,
            "direct_forces": False,
            "regress_forces": True,
            "regress_stress": True,
            "model_version": 1.2,
            "execution_mode": "general",
        },
        heads={
            "efs": {
                "module": "fairchem.core.models.uma.escn_moe.DatasetSpecificMoEWrapper",
                "head_cls": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                "head_kwargs": {"wrap_property": False},
                "dataset_mapping": {"omol": "omol", "omat": "omat"},
            }
        },
        pass_through_head_outputs=True,
        model_id="UMA-S-1.2.1",
    )
    # These tests exercise preparation and the actual UMA head directly, without
    # task normalization or a downloaded checkpoint.
    model._tasks = {}
    model.eval()
    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    super(MLIPPredictUnit, predictor).__init__()
    predictor.model = _ModelWrapper(model)
    predictor.device = device
    predictor._requested_device = device
    predictor.inference_settings = inference_settings_xpu()
    predictor.lazy_model_intialized = False
    predictor._warned_upcast = False
    predictor._nitro_announced = False
    predictor.assert_on_nans = False
    predictor.atom_refs = {}
    return predictor


def _data():
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    return AtomicData.from_ase(atoms, task_name="omol")


@pytest.mark.parametrize(
    "input_device",
    [
        "cpu",
        pytest.param(
            "xpu:0",
            marks=pytest.mark.skipif(
                not torch.xpu.is_available(), reason="Requires a real Intel XPU"
            ),
        ),
    ],
)
def test_lazy_mole_preparation_normalizes_private_batch(input_device):
    reference = _small_predictor()
    candidate = copy.deepcopy(reference)
    candidate.device = input_device
    reference._lazy_init(_data())

    data = _data().to(input_device)
    for key, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.double()
    before = data.clone()
    candidate._lazy_init(data)

    assert candidate.lazy_model_intialized
    assert not any(isinstance(module, MOLE) for module in candidate.model.modules())
    for name, value in candidate.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), reference.model.state_dict()[name])
    for key, value in before:
        if torch.is_tensor(value):
            torch.testing.assert_close(data[key], value)


def test_lazy_mole_preparation_preserves_fp32_under_cpu_autocast():
    reference = _small_predictor()
    candidate = copy.deepcopy(reference)
    reference._lazy_init(_data())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        candidate._lazy_init(_data())
        assert torch.is_autocast_enabled("cpu")
    for name, value in candidate.model.state_dict().items():
        torch.testing.assert_close(
            value, reference.model.state_dict()[name], rtol=0, atol=0
        )


def test_predict_rejects_inference_mode_before_merging_force_model():
    predictor = _small_predictor()
    data = _data()
    with torch.inference_mode(), pytest.raises(
        RuntimeError, match=r"outside torch.inference_mode\(\)"
    ):
        predictor.predict(data)
    assert not predictor.lazy_model_intialized
    assert any(isinstance(module, MOLE) for module in predictor.model.modules())


@pytest.mark.parametrize("loaded_dtype", [torch.float32, torch.float64])
def test_lazy_mole_preparation_with_cpu_fp64_request(loaded_dtype):
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(loaded_dtype)
    try:
        predictor = _small_predictor()
    finally:
        torch.set_default_dtype(original_dtype)
    predictor.inference_settings.base_precision_dtype = torch.float64
    data = _data()
    for key, value in data:
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.double()
    before = data.clone()
    predictor._lazy_init(data)
    assert torch.get_default_dtype() == original_dtype
    actual = predictor.model(data.clone())
    for name in ("energy", "forces", "stress"):
        value = actual[f"omol_{name}"][name]
        assert value.dtype == torch.float64
        assert torch.isfinite(value).all()
    assert all(
        tensor.dtype == torch.float64
        for tensor in predictor.model.state_dict().values()
        if tensor.is_floating_point()
    )
    for key, value in before:
        if torch.is_tensor(value):
            torch.testing.assert_close(data[key], value)

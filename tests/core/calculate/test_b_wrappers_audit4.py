"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor
from tests.core.units.mlip_unit.test_xpu_preparation import _data


@dataclass(frozen=True, slots=True)
class _JobMetadata:
    tensor: object
    child: object = None


@pytest.mark.parametrize("metadata_type", [_JobMetadata, SimpleNamespace])
def test_metadata_clone_and_conversion_preserve_autograd_and_aliases(metadata_type):
    leaf = torch.ones(3, requires_grad=True)
    data = _data()
    data.metadata_values = leaf * 2
    links = []
    data.metadata = metadata_type(tensor=data.metadata_values, child=links)
    links.append(data.metadata)
    cloned = data.clone().to(dtype=torch.float64)
    assert cloned.metadata is not data.metadata
    assert cloned.metadata.child[0] is cloned.metadata
    assert cloned.metadata.tensor is cloned.metadata_values
    assert cloned.metadata.tensor.dtype == torch.float64
    cloned.metadata.tensor.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))
    assert data.metadata.tensor.dtype == torch.float32


@pytest.mark.parametrize("nonleaf", [False, True])
def test_metadata_back_reference_keeps_converted_atomic_data_owner(nonleaf):
    data = _data()
    leaf = data.pos.detach().clone().requires_grad_()
    data.pos = leaf * 2 if nonleaf else leaf
    data.metadata = SimpleNamespace(owner=data, tensor=data.pos)
    converted = data.clone().to(dtype=torch.float64)
    assert converted.metadata.owner is converted
    assert converted.metadata.tensor is converted.pos
    assert converted.pos.dtype == torch.float64
    assert data.pos.dtype == torch.float32
    assert data.metadata.owner is data
    gradient = torch.autograd.grad(converted.pos.sum(), leaf)[0]
    torch.testing.assert_close(gradient, torch.full_like(leaf, 2 if nonleaf else 1))


@pytest.mark.parametrize("problem", ["capacity", "autograd", "device"])
def test_staging_validates_tensors_inside_slotted_job_metadata(problem):
    from fairchem.core.calculate.device_buffers import DeviceBatchStager

    data = _data()
    value = {
        "capacity": lambda: torch.zeros(2000),
        "autograd": lambda: torch.ones(3, requires_grad=True),
        "device": lambda: torch.empty(3, device="meta"),
    }[problem]()
    data.metadata = _JobMetadata(value)
    stager = DeviceBatchStager("cpu", max_batch_bytes=4096)
    message = {
        "capacity": "max_batch_bytes",
        "autograd": "differentiable",
        "device": "must be on CPU",
    }[problem]
    with pytest.raises(ValueError, match=message):
        stager.stage(data)
    stager.close()


def test_staged_slotted_metadata_owns_storage_and_preserves_aliases():
    from fairchem.core.calculate.device_buffers import DeviceBatchStager

    data = _data()
    data.metadata_values = torch.ones(3)
    data.metadata = _JobMetadata(data.metadata_values)
    stager = DeviceBatchStager("cpu")
    with stager.stage(data) as staged:
        assert staged.metadata.tensor is staged.metadata_values
        data.metadata_values.fill_(4)
        torch.testing.assert_close(staged.metadata.tensor, torch.ones(3))
    stager.close()


@pytest.mark.parametrize("use_overrides", [False, True])
def test_torchsim_accepts_series_b_startup_configuration(monkeypatch, use_overrides):
    pytest.importorskip("torch_sim")
    from torch_sim.optimizers.fire import fire_init, fire_step

    from fairchem.core.calculate import torchsim_interface as interface
    from tests.core.calculate.test_torchsim_interop import _state

    predictor = _public_predictor()
    predictor.inference_settings.merge_mole = False
    predictor.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    settings = copy.deepcopy(predictor.inference_settings)
    original_settings = asdict(settings)
    overrides = {"series_b": {"weight_bits": 8}}
    received = []

    def load(name, **kwargs):
        received.append(kwargs)
        assert name == "small-uma.pt"
        assert kwargs["inference_settings"] == ("batch" if use_overrides else settings)
        assert kwargs["overrides"] == (overrides if use_overrides else None)
        assert kwargs["cache_dir"] == "/tmp/uma-models"
        return predictor

    # Substitute checkpoint acquisition only; UMA, autograd, conversion and FIRE
    # all execute with B enabled through the public wrapper configuration.
    monkeypatch.setattr(interface.pretrained_mlip, "get_predict_unit", load)
    model = interface.FairChemModel(
        "small-uma.pt",
        device="cpu",
        task_name="omol",
        model_cache_dir="/tmp/uma-models",
        inference_settings="batch" if use_overrides else settings,
        overrides=overrides if use_overrides else None,
    )
    state = _state()
    before = state.clone()
    output = model(fire_step(fire_init(state.clone(), model), model))
    assert len(received) == 1
    assert predictor.approximation_manifest["enabled"]
    assert asdict(settings) == original_settings
    assert overrides == {"series_b": {"weight_bits": 8}}
    for name in ("energy", "forces"):
        assert output[name].dtype == torch.float32
        assert torch.isfinite(output[name]).all()
    torch.testing.assert_close(state.positions, before.positions, atol=0, rtol=0)


@pytest.mark.parametrize(
    "dtype,series_b",
    [
        (
            torch.float32,
            SeriesBSettings(
                weight_bits=8,
                saved_feature_dtype="bf16",
                saved_feature_min_elements=1,
                compensated_reduction=True,
                radial_table_points=64,
            ),
        ),
        (
            torch.float32,
            SeriesBSettings(
                gemm_precision="bf16_split2",
                low_precision_emulation=True,
                saved_feature_dtype="fp16",
                saved_feature_min_elements=1,
                radial_fast_exp=True,
            ),
        ),
        (
            torch.float32,
            SeriesBSettings(
                low_rank_fraction=0.75,
                saved_feature_dtype="bf16",
                saved_feature_min_elements=1,
            ),
        ),
        (
            torch.float64,
            SeriesBSettings(saved_feature_dtype="fp32", saved_feature_min_elements=1),
        ),
    ],
    ids=[
        "int8-storage-reduction-radial",
        "split2-storage-exp",
        "lowrank-storage",
        "fp64-storage",
    ],
)
def test_combined_ab_ase_and_torchsim_match_direct_predictions(
    monkeypatch, dtype, series_b
):
    ts = pytest.importorskip("torch_sim")
    from ase.build import molecule

    from fairchem.core.calculate import torchsim_interface as interface
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
    from fairchem.core.datasets.atomic_data import AtomicData

    predictor = _public_predictor()
    predictor.inference_settings = replace(
        predictor.inference_settings,
        base_precision_dtype=dtype,
        neighbor_skin=0.4,
        mole_cache_size=2,
        series_b=series_b,
    )
    predictor.model.to(dtype=dtype)
    ase_predictor, sim_predictor = copy.deepcopy(predictor), copy.deepcopy(predictor)
    calc = FAIRChemCalculator(ase_predictor, task_name="omol")
    monkeypatch.setattr(
        interface.pretrained_mlip, "get_predict_unit", lambda *a, **kw: sim_predictor
    )
    model = interface.FairChemModel(
        "small-uma.pt",
        device="cpu",
        task_name="omol",
        inference_settings=sim_predictor.inference_settings,
    )
    atoms = molecule("H2O")
    atoms.set_cell([8, 9, 10])
    atoms.info.update(charge=0, spin=1)
    state = ts.io.atoms_to_state([atoms], device="cpu", dtype=dtype)
    assert model.dtype == dtype
    for _ in range(3):
        before = state.positions.clone()
        atoms.positions[:] = before.double().numpy()
        data = AtomicData.from_ase(
            atoms, task_name="omol", target_dtype=dtype, r_data_keys=["charge", "spin"]
        )
        expected = predictor.predict(data, requested_properties=("energy", "forces"))
        calc.calculate(atoms, properties=["energy", "forces"])
        actual = model(state)
        tolerance = (
            dict(atol=2e-5, rtol=2e-4)
            if dtype == torch.float32
            else dict(atol=1e-11, rtol=1e-9)
        )
        for name in ("energy", "forces"):
            torch.testing.assert_close(
                actual[name], expected[name].to(dtype), **tolerance
            )
            torch.testing.assert_close(
                torch.as_tensor(calc.results[name], dtype=dtype).reshape_as(
                    actual[name]
                ),
                expected[name].to(dtype),
                **tolerance,
            )
            actual[name].fill_(123)
        torch.testing.assert_close(state.positions, before, atol=0, rtol=0)
        state.positions[0, 0] += 0.001

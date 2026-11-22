"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  UMA Cartesian Hessians, their symmetry and all force derivatives.
        The periodic CPU case uses a small randomly initialized model.
Models: The CUDA tests use uma-s-1p1 and uma-s-1p2 checkpoints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from ase.build import molecule
from numpy import testing as npt

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.api.inference import inference_settings_fp64
from tests.conftest import get_predict_unit_for_test

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol

pretrained_uma = pytest.mark.pretrained("uma-s-1p1", "uma-s-1p2")


def _assert_hessian_matches_forces(
    data: AtomicData,
    predict_unit: MLIPPredictUnitProtocol,
    *,
    atol: float,
    rtol: float,
) -> None:
    # Keep the same predictor, system size, cell, dtype and backend on both
    # sides. Only the requested output and one Cartesian coordinate change.
    batch = atomicdata_list_to_batch([data])
    assert batch.pos.dtype == torch.float64
    ndof = batch.pos.numel()
    hessian = predict_unit.predict(batch, requested_properties=("hessian",))["hessian"]
    assert hessian.shape == (1, ndof, ndof)
    assert hessian.dtype == torch.float64

    eps = 1e-4  # Angstrom
    columns = []
    for atom in range(len(data.pos)):
        for axis in range(3):
            forces = []
            for displacement in (eps, -eps):
                displaced = data.clone()
                displaced.pos[atom, axis] += displacement
                result = predict_unit.predict(
                    atomicdata_list_to_batch([displaced]),
                    requested_properties=("forces",),
                )
                forces.append(result["forces"].reshape(ndof))
            columns.append((forces[1] - forces[0]) / (2 * eps))
    numerical = torch.stack(columns, dim=1)
    torch.testing.assert_close(hessian[0], numerical, rtol=rtol, atol=atol)


@pretrained_uma
@pytest.mark.gpu()
@pytest.mark.parametrize("vmap", [True, False])
def test_hessian(vmap, pretrained_checkpoint):
    """Test Hessian calculation using MLIPPredictUnit directly."""
    predict_unit = get_predict_unit_for_test(
        pretrained_checkpoint,
        device="cuda",
        inference_settings=InferenceSettings(
            predict_untrained_hessian={"omol"}, hessian_vmap=vmap
        ),
    )
    assert predict_unit.model.module.backbone.regress_config.hessian_vmap is vmap

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    preds = predict_unit.predict(batch)
    hessian = preds["hessian"].detach().cpu().numpy()

    assert hessian.shape == (1, 9, 9)
    assert np.isfinite(hessian).all()


@pretrained_uma
@pytest.mark.gpu()
def test_hessian_vs_numerical(pretrained_checkpoint):
    settings = inference_settings_fp64()
    settings.predict_untrained_hessian = {"omol"}
    unit = get_predict_unit_for_test(
        pretrained_checkpoint, device="cuda", inference_settings=settings
    )
    atoms = molecule("H2O")
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
        target_dtype=torch.float64,
    )
    _assert_hessian_matches_forces(data, unit, atol=1e-4, rtol=1e-3)


def test_periodic_hessian_vs_numerical_cpu():
    from tests.core.units.mlip_unit.test_predict_interoperability import (
        _public_predictor,
    )

    unit = _public_predictor()
    unit.inference_settings.base_precision_dtype = torch.float64
    unit.model.double().eval()
    atoms = molecule("H2O")
    atoms.set_cell([[2.74, 0, 0], [0.14, 2.81, 0], [0.11, 0.21, 2.89]])
    atoms.pbc = True
    atoms.translate([0.8, 0.61, 1.2])
    atoms.info.update(charge=0, spin=1)
    # Periodic images are inside the model's 3 Å cutoff.
    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        target_dtype=torch.float64,
    )
    _assert_hessian_matches_forces(data, unit, atol=1e-7, rtol=1e-4)


@pretrained_uma
@pytest.mark.gpu()
def test_hessian_symmetry(pretrained_checkpoint):
    """Test that the Hessian matrix is symmetric."""
    predict_unit = get_predict_unit_for_test(
        pretrained_checkpoint,
        device="cuda",
        inference_settings=InferenceSettings(
            predict_untrained_hessian={"omol"}, hessian_vmap=True
        ),
    )

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    preds = predict_unit.predict(batch)
    hessian = preds["hessian"].squeeze()

    npt.assert_allclose(
        hessian.detach().cpu().numpy(),
        hessian.T.detach().cpu().numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Hessian matrix is not symmetric",
    )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.units.mlip_unit.api.series_b import SeriesBSettings
from tests.core.components.test_batch_server_interop import _data, _server
from tests.core.units.mlip_unit.test_predict_interoperability import _public_predictor


@pytest.mark.parametrize("difference", ["fields", "dtype", "shape"])
def test_series_b_server_isolates_incompatible_job_metadata(difference):
    predictor = _public_predictor()
    predictor.inference_settings.merge_mole = False
    predictor.inference_settings.series_b = SeriesBSettings(weight_bits=8)
    oracle = copy.deepcopy(predictor)
    items = [_data(0.7), _data(0.9), _data(1.1)]
    for item in items:
        item["job_values"] = torch.tensor([[1.0, 2.0]])
    if difference == "fields":
        del items[1]["job_values"]
    elif difference == "dtype":
        items[1]["job_values"] = items[1]["job_values"].double()
    else:
        items[1]["job_values"] = torch.tensor([[1.0, 2.0, 3.0]])
    before = [item.clone() for item in items]
    expected = [oracle.predict(item, False) for item in items]
    actual = _server()._run_batched_inference(items, predictor, False)
    for result, reference, source, snapshot in zip(
        actual, expected, items, before, strict=True
    ):
        for name in ("energy", "forces"):
            torch.testing.assert_close(
                result[name], reference[name], atol=2e-5, rtol=2e-4
            )
        assert source.keys() == snapshot.keys()
        if "job_values" in source:
            torch.testing.assert_close(source.job_values, snapshot.job_values)

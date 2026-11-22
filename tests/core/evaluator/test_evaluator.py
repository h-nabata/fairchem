"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fairchem.core.modules.evaluator import (
    Evaluator,
    cosine_similarity,
    magnitude_error,
)


@pytest.fixture(scope="class")
def load_evaluator_s2ef(request) -> None:
    request.cls.evaluator = Evaluator(task="s2ef")
    prediction = {
        "energy": torch.randn(6),
        "forces": torch.randn(1000000, 3),
        "natoms": torch.tensor((100000, 200000, 300000, 200000, 100000, 100000)),
    }
    target = {
        "energy": torch.randn(6),
        "forces": torch.randn(1000000, 3),
        "natoms": torch.tensor((100000, 200000, 300000, 200000, 100000, 100000)),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


@pytest.fixture(scope="class")
def load_evaluator_is2rs(request) -> None:
    request.cls.evaluator = Evaluator(task="is2rs")
    prediction = {
        "positions": torch.randn(50, 3),
        "natoms": torch.tensor((5, 5, 10, 12, 18)),
        "cell": torch.randn(5, 3, 3),
        "pbc": torch.tensor([True, True, True]),
    }
    target = {
        "positions": torch.randn(50, 3),
        "cell": torch.randn(5, 3, 3),
        "natoms": torch.tensor((5, 5, 10, 12, 18)),
        "pbc": torch.tensor([True, True, True]),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


@pytest.fixture(scope="class")
def load_evaluator_is2re(request) -> None:
    request.cls.evaluator = Evaluator(task="is2re")
    prediction = {
        "energy": torch.randn(50),
    }
    target = {
        "energy": torch.randn(50),
    }
    request.cls.metrics = request.cls.evaluator.eval(prediction, target)


class TestMetrics:
    def test_cosine_similarity(self) -> None:
        v1, v2 = torch.randn(1000000, 3), torch.randn(1000000, 3)
        res = cosine_similarity(v1, v2)
        np.testing.assert_almost_equal(res["metric"], 0, decimal=2)
        np.testing.assert_almost_equal(res["total"] / res["numel"], res["metric"])

    def test_magnitude_error(self) -> None:
        v1, v2 = (
            torch.tensor([[0.0, 1], [-1, 0]]),
            torch.tensor([[0.0, 0], [0, 0]]),
        )
        res = magnitude_error(v1, v2)
        np.testing.assert_equal(res["metric"], 1.0)


@pytest.mark.usefixtures("load_evaluator_s2ef")
class TestS2EFEval:
    def test_metrics_exist(self) -> None:
        assert "energy_mae" in self.metrics
        assert "forces_mae" in self.metrics
        assert "forces_cosine_similarity" in self.metrics
        assert "energy_forces_within_threshold" in self.metrics


@pytest.mark.usefixtures("load_evaluator_is2rs")
class TestIS2RSEval:
    def test_metrics_exist(self) -> None:
        assert "positions_average_distance_within_threshold" in self.metrics


@pytest.mark.usefixtures("load_evaluator_is2re")
class TestIS2REEval:
    def test_metrics_exist(self) -> None:
        assert "energy_mae" in self.metrics
        assert "energy_mse" in self.metrics
        assert "energy_within_threshold" in self.metrics


def test_rmse_matches_pooled_components_and_exposes_historical_vector_norm():
    evaluator = Evaluator(eval_metrics={"forces": ["rmse", "mean_euclidean_error"]})
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    label = torch.zeros_like(pred)
    metrics = evaluator.eval({"forces": pred[:1]}, {"forces": label[:1]})
    result = evaluator.eval(
        {"forces": pred[1:]}, {"forces": label[1:]}, prev_metrics=metrics
    )
    assert result["forces_rmse"]["numel"] == 6
    assert result["forces_rmse"]["total"] == 91
    assert result["forces_rmse"]["metric"] == pytest.approx(np.sqrt(91 / 6))
    assert result["forces_mean_euclidean_error"]["metric"] == pytest.approx(
        torch.linalg.vector_norm(pred, dim=1).mean().item()
    )
    previous = {"forces_rmse": {"numel": 2, "total": 5.0, "metric": 2.5}}
    with pytest.raises(ValueError, match="new accumulator"):
        evaluator.eval({"forces": pred}, {"forces": label}, prev_metrics=previous)

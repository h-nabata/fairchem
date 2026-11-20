"""Exercise the Ray Serve control plane and a real CPU predictor across workers."""

from __future__ import annotations

import pytest
import ray
import torch
from ase import Atoms
from ray import serve

from fairchem.core.components import batch_server
from fairchem.core.datasets.atomic_data import AtomicData
from tests.core.components.test_batch_server_interop import MixedTaskPredictor


@pytest.mark.serial()
def test_live_cpu_prediction_update_delete_and_redeploy(monkeypatch):
    monkeypatch.setenv("RAY_USAGE_STATS_ENABLED", "0")
    ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
    name = "cpu-lifecycle"
    data = AtomicData.from_ase(
        Atoms("H2", positions=[[0, 0, 0], [2, 0, 0]], cell=[8, 8, 8]),
        task_name="omol",
    )

    def predictor(revision):
        unit = MixedTaskPredictor()
        unit.device = "cpu"
        unit.revision = revision
        return unit

    def deploy(revision):
        return batch_server.setup_batch_predict_server(
            predictor(revision),
            deployment_name=name,
            route_prefix="/cpu-lifecycle",
            deployment_config={
                "num_replicas": 1,
                "ray_actor_options": {"num_cpus": 1, "num_gpus": 0},
            },
            batch_config={"max_batch_size": 2, "batch_wait_timeout_s": 0.0},
        )

    def verify(handle, revision):
        served = handle.get_predict_unit_attribute.remote("revision").result(
            timeout_s=30
        )
        assert served == revision
        output = handle.predict.remote(data, False).result(timeout_s=30)
        torch.testing.assert_close(output["energy"], torch.tensor([4.0]))
        torch.testing.assert_close(output["forces"], -2 * data.pos)

    try:
        handle = deploy("initial")
        verify(handle, "initial")
        batch_server.update_batch_config(name, {"max_batch_size": 4})
        batch_server.update_served_predict_unit(name, predictor("updated"))
        verify(handle, "updated")
        assert batch_server._DEPLOYED_APPS[name].batch_config.max_batch_size == 4

        batch_server.delete_batch_predict_server(name)
        assert name not in serve.status().applications
        assert name not in batch_server._DEPLOYED_APPS
        replacement = deploy("recreated")
        verify(replacement, "recreated")
        batch_server.delete_batch_predict_server(name)
        assert name not in serve.status().applications
        assert name not in batch_server._DEPLOYED_APPS
    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()

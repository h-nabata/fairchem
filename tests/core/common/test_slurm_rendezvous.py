"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core.common import distutils


def test_missing_scheduler_environment_fails_early(monkeypatch):
    for k in ("SLURM_STEP_NODELIST", "SLURM_JOB_NODELIST"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="SLURM_JOB_NODELIST"):
        distutils.setup({"submit": True})


def test_slurm_rendezvous_uses_assigned_port(monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29933")
    monkeypatch.setattr(
        distutils.subprocess, "check_output", lambda _: b"node1\nnode2\n"
    )
    result = distutils.get_init_method(
        "tcp", world_size=2, rank=1, node_list="node[1-2]"
    )
    assert result.startswith("tcp://node1:29933")

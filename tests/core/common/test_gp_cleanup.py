"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest

from fairchem.core.common import gp_utils


@pytest.fixture()
def groups(monkeypatch):
    data_group, graph_group = object(), object()
    monkeypatch.setattr(gp_utils, "_DATA_PARALLEL_GROUP", data_group)
    monkeypatch.setattr(gp_utils, "_GRAPH_PARALLEL_GROUP", graph_group)
    monkeypatch.setattr(gp_utils, "_GP_CONFIG", gp_utils.GraphParallelConfig())
    return data_group, graph_group


def _assert_reset():
    assert gp_utils._DATA_PARALLEL_GROUP is None
    assert gp_utils._GRAPH_PARALLEL_GROUP is None
    assert gp_utils.get_gp_config() is None
    assert not gp_utils.initialized()


@pytest.mark.parametrize("has_data", [False, True])
@pytest.mark.parametrize("has_graph", [False, True])
def test_cleanup_accepts_partial_setup_and_is_idempotent(
    monkeypatch, groups, has_data, has_graph
):
    data_group, graph_group = groups
    if not has_data:
        gp_utils._DATA_PARALLEL_GROUP = None
    if not has_graph:
        gp_utils._GRAPH_PARALLEL_GROUP = None
    destroyed = []
    monkeypatch.setattr(gp_utils.dist, "destroy_process_group", destroyed.append)
    gp_utils.cleanup_gp()
    assert destroyed == [
        group
        for group, present in ((data_group, has_data), (graph_group, has_graph))
        if present
    ]
    _assert_reset()
    gp_utils.cleanup_gp()
    assert len(destroyed) == int(has_data) + int(has_graph)


@pytest.mark.parametrize(
    "failures",
    [
        (ValueError("stale data"), None),
        (None, ValueError("stale graph")),
        (RuntimeError("data failure"), None),
        (None, RuntimeError("graph failure")),
        (RuntimeError("data failure"), OSError("graph failure")),
        (ValueError("stale data"), OSError("graph failure")),
        (KeyboardInterrupt(), None),
    ],
)
def test_cleanup_attempts_both_groups_and_preserves_failures(
    monkeypatch, groups, failures
):
    destroyed = []

    def destroy(group):
        destroyed.append(group)
        error = failures[groups.index(group)]
        if error is not None:
            raise error

    monkeypatch.setattr(gp_utils.dist, "destroy_process_group", destroy)
    expected = [
        error
        for error in failures
        if error is not None and not isinstance(error, ValueError)
    ]
    if len(expected) == 1:
        with pytest.raises(type(expected[0])) as raised:
            gp_utils.cleanup_gp()
        assert raised.value is expected[0]
    elif expected:
        with pytest.raises(BaseExceptionGroup) as raised:
            gp_utils.cleanup_gp()
        assert raised.value.exceptions == tuple(expected)
    else:
        gp_utils.cleanup_gp()
    assert destroyed == list(groups)
    _assert_reset()
    gp_utils.cleanup_gp()
    assert destroyed == list(groups)


def test_failed_graph_group_creation_can_be_cleaned_up_and_retried(monkeypatch, groups):
    gp_utils._DATA_PARALLEL_GROUP = None
    gp_utils._GRAPH_PARALLEL_GROUP = None
    gp_utils._GP_CONFIG = None
    monkeypatch.setattr(gp_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gp_utils.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(gp_utils.dist, "get_rank", lambda: 0)
    created = []

    def create(*args, **kwargs):
        if len(created) == 1:
            raise RuntimeError("graph group setup failed")
        group = object()
        created.append(group)
        return group

    monkeypatch.setattr(gp_utils.dist, "new_group", create)
    with pytest.raises(RuntimeError, match="graph group setup failed"):
        gp_utils.setup_graph_parallel_groups(1, "gloo")
    assert not gp_utils.initialized()
    assert gp_utils._DATA_PARALLEL_GROUP is created[0]
    destroyed = []
    monkeypatch.setattr(gp_utils.dist, "destroy_process_group", destroyed.append)
    gp_utils.cleanup_gp()
    assert destroyed == created
    _assert_reset()
    monkeypatch.setattr(gp_utils.dist, "new_group", lambda *args, **kwargs: object())
    gp_utils.setup_graph_parallel_groups(1, "gloo")
    assert gp_utils.initialized()
    assert gp_utils.get_gp_config().group_size == 1
    gp_utils.cleanup_gp()
    _assert_reset()

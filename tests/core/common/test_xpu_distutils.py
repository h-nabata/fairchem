"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.common import distutils


@pytest.mark.parametrize("selected, expected", [("xpu", "xpu:0"), ("xpu:1", "xpu:1")])
def test_xpu_device_resolution_preserves_selected_index(
    monkeypatch, selected, expected
):
    monkeypatch.setenv(distutils.CURRENT_DEVICE_TYPE_STR, selected)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 0)

    def cuda_must_not_be_queried():
        raise AssertionError("Explicit XPU selection must not query CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", cuda_must_not_be_queried)
    assert distutils.get_device_for_local_rank() == expected


def test_unavailable_xpu_does_not_fall_back_to_cpu(monkeypatch):
    monkeypatch.setenv(distutils.CURRENT_DEVICE_TYPE_STR, "xpu")
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="XPU was selected"):
        distutils.get_device_for_local_rank()


def test_invalid_xpu_index_does_not_fall_back_to_current_device(monkeypatch):
    monkeypatch.setenv(distutils.CURRENT_DEVICE_TYPE_STR, "xpu:2")
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="index 2 is not available"):
        distutils.get_device_for_local_rank()

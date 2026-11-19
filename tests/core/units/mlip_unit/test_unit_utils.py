"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import pytest
import torch

from fairchem.core.common import distutils, gp_utils
from fairchem.core.units.mlip_unit.utils import tf32_context_manager


@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("initial_precision", ["highest", "high"])
@pytest.mark.parametrize("initial_cudnn_tf32", [False, True])
def test_tf32_context_manager_applies_and_restores_settings(
    tf32, initial_precision, initial_cudnn_tf32
):
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision(initial_precision)
        torch.backends.cudnn.allow_tf32 = initial_cudnn_tf32
        with tf32_context_manager(tf32):
            expected_precision = "high" if tf32 else "highest"
            assert torch.get_float32_matmul_precision() == expected_precision
            assert torch.backends.cuda.matmul.allow_tf32 is tf32
            assert torch.backends.cudnn.allow_tf32 is tf32
        assert torch.get_float32_matmul_precision() == initial_precision
        assert torch.backends.cuda.matmul.allow_tf32 is (initial_precision == "high")
        assert torch.backends.cudnn.allow_tf32 is initial_cudnn_tf32
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


@pytest.mark.parametrize("tf32", [False, True])
def test_tf32_context_manager_restores_after_error(tf32):
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        initial_precision = "high" if not tf32 else "highest"
        torch.set_float32_matmul_precision(initial_precision)
        torch.backends.cudnn.allow_tf32 = not tf32
        with (
            pytest.raises(RuntimeError, match="failure"),
            tf32_context_manager(tf32),
        ):
            raise RuntimeError("failure")
        assert torch.get_float32_matmul_precision() == initial_precision
        assert torch.backends.cuda.matmul.allow_tf32 is (initial_precision == "high")
        assert torch.backends.cudnn.allow_tf32 is not tf32
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


def test_tf32_context_manager_restores_nested_settings():
    original_precision = torch.get_float32_matmul_precision()
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        with tf32_context_manager(True):
            assert torch.get_float32_matmul_precision() == "high"
            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
            with tf32_context_manager(False):
                assert torch.get_float32_matmul_precision() == "highest"
                assert not torch.backends.cuda.matmul.allow_tf32
                assert not torch.backends.cudnn.allow_tf32
            assert torch.get_float32_matmul_precision() == "high"
            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
        assert torch.get_float32_matmul_precision() == "highest"
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
    finally:
        torch.set_float32_matmul_precision(original_precision)
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32


class GradSaveOptimizer(torch.optim.AdamW):
    def __init__(
        self,
        params,
        save_path,
    ):
        super().__init__(params=params, lr=0.0, weight_decay=0.0)
        self.save_path = save_path
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
        self.save_step = 0
        # self.params = params

    def step(self, closure=None):
        if self.save_path:
            gp_size = 0
            gp_rank = 0
            if gp_utils.initialized():
                gp_size = gp_utils.get_gp_world_size()
                gp_rank = gp_utils.get_dp_rank()

            ddp_size = distutils.get_world_size()
            ddp_rank = distutils.get_rank()

            torch.save(
                {
                    "param": list(self.param_groups[0]["params"]),
                    "grad": [
                        param.grad
                        for param in self.param_groups[0]["params"]
                        if param.grad is not None
                    ],
                },
                f"{self.save_path}/ddp{ddp_size}.{ddp_rank}_gp{gp_size}.{gp_rank}_step{self.save_step}.pt",
            )
        self.save_step += 1
        super().step()


def test_strict_xpu_readers_overlap_while_precision_writer_waits(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier, Event
    from types import SimpleNamespace

    backend = SimpleNamespace(allow_tf32=False)
    monkeypatch.setattr(torch.backends, "mkldnn", backend)
    concurrent_readers = Barrier(3)
    finish = Event()
    writer_entered = Event()

    def read():
        with tf32_context_manager(False, device_type="xpu"):
            concurrent_readers.wait(timeout=10)
            assert finish.wait(timeout=10)
            assert not backend.allow_tf32

    def write():
        with tf32_context_manager(True, device_type="xpu"):
            assert backend.allow_tf32
            writer_entered.set()

    with ThreadPoolExecutor(3) as pool:
        readers = [pool.submit(read) for _ in range(2)]
        concurrent_readers.wait(timeout=10)
        writer = pool.submit(write)
        try:
            assert not writer_entered.wait(timeout=0.1)
        finally:
            finish.set()
        for reader in readers:
            reader.result()
        writer.result()
    assert not backend.allow_tf32


def test_modern_onednn_precision_guard_and_key(monkeypatch):
    from types import SimpleNamespace

    from fairchem.core.common.precision import matmul_precision_key

    matmul = SimpleNamespace(fp32_precision="bf16")
    monkeypatch.setattr(torch.backends, "mkldnn", SimpleNamespace(matmul=matmul))

    def incompatible_legacy_getter():
        raise RuntimeError("legacy and modern controls cannot be mixed")

    monkeypatch.setattr(
        torch, "get_float32_matmul_precision", incompatible_legacy_getter
    )
    before = matmul_precision_key("xpu")
    with (
        pytest.raises(ValueError, match="Strict FP32"),
        tf32_context_manager(False, device_type="xpu"),
    ):
        pass
    with tf32_context_manager(True, device_type="xpu"):
        assert matmul.fp32_precision == "tf32"
        assert before != matmul_precision_key("xpu")
    assert matmul.fp32_precision == "bf16"

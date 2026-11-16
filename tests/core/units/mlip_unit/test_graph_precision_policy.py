"""Captured numeric graphs must not survive a changed execution policy."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from fairchem.core.units.mlip_unit.inference_runtime import NumericCoreRuntime


@pytest.mark.parametrize("policy", ["autocast", "matmul", "onednn", "inference"])
def test_graph_key_separates_precision_and_inference_policies(monkeypatch, policy):
    # No capture/kernel is emulated: inspect admission keys with fake XPU input.
    monkeypatch.setattr(torch.xpu, "current_stream", lambda device: "stream")
    runtime = NumericCoreRuntime(lambda x: x.square(), xpu_graph=True)
    with FakeTensorMode():
        value = torch.empty(4, 4, device="xpu:1", dtype=torch.float32)

    def key():
        return runtime._graph_arguments((value,), {})[-1]

    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        monkeypatch.setattr(torch.backends, "mkldnn", SimpleNamespace(allow_tf32=False))
        with torch.no_grad():
            original = key()
            if policy == "autocast":
                with torch.autocast("xpu", dtype=torch.bfloat16):
                    changed = key()
            elif policy == "matmul":
                torch.set_float32_matmul_precision("high")
                changed = key()
                torch.set_float32_matmul_precision("highest")
            elif policy == "onednn":
                torch.backends.mkldnn.allow_tf32 = True
                changed = key()
                torch.backends.mkldnn.allow_tf32 = False
            else:
                with torch.inference_mode():
                    changed = key()
            assert changed != original
            assert key() == original
    finally:
        torch.set_float32_matmul_precision(precision)

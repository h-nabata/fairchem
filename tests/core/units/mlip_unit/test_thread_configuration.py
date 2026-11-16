"""Regression for process-owned PyTorch thread pools shared by predictors."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_inference_threads_can_change_after_interop_pool_initialization():
    # Inter-op configuration cannot be reset in a process. Use a child so this
    # real PyTorch regression does not permanently configure the pytest runner.
    script = textwrap.dedent(
        """
        import torch
        from types import SimpleNamespace
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

        torch.set_num_interop_threads(1)
        torch.ones(32, 32).square().sum()
        for threads in (2, 1, 2):
            MLIPPredictUnit._setup_threads(
                SimpleNamespace(), InferenceSettings(torch_num_threads=threads)
            )
            assert torch.get_num_threads() == threads
            assert torch.get_num_interop_threads() == 1
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

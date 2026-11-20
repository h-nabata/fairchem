"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import platform
import shutil
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

import torch

from fairchem.core.common.distutils import parse_device

_TRITON_SEED_CACHE = "FAIRCHEM_TRITON_SEED_CACHE_DIR"


def execution_environment(device: str) -> dict[str, Any]:
    """
    Describe the execution architecture without node-specific identity.
    """
    parsed = parse_device(device)
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
    except OSError:
        cpuinfo = ""
    cpu_model = next(
        (
            line.partition(":")[2].strip()
            for line in cpuinfo.splitlines()
            if line.partition(":")[0].strip() in {"model name", "Hardware"}
        ),
        platform.processor() or None,
    )
    result: dict[str, Any] = {
        "cpu_model": cpu_model,
        "cpu_count": len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "device_type": parsed.type,
        "cpu_architecture": platform.machine(),
        "torch_version": str(torch.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "accelerator_name": None,
        "compute_capability": None,
    }
    if parsed.type == "cuda":
        index = torch.cuda.current_device() if parsed.index is None else parsed.index
        properties = torch.cuda.get_device_properties(index)
        result.update(
            accelerator_name=properties.name,
            compute_capability=[properties.major, properties.minor],
        )
    elif parsed.type == "xpu":
        properties = torch.xpu.get_device_properties(parsed)
        result.update(
            accelerator_name=properties.name,
            xpu_driver_version=getattr(properties, "driver_version", None),
            xpu_runtime_version=getattr(torch.version, "xpu", None),
        )
    return result


def prepare_compilation_cache() -> None:
    """
    Put Triton and TorchInductor caches on node-local storage.

    Shared home-directory caches can mix architectures and race when many
    array tasks compile concurrently. A compatible prebuilt Triton CUDA helper
    is copied first because some compute images lack Python development
    headers. Explicit user settings are preserved.
    """
    root = Path(os.environ.get("SLURM_TMPDIR", tempfile.gettempdir()))
    if not root.is_dir():
        root = Path(tempfile.gettempdir())
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    # Local tasks share their allocation's job ID, so they use their process.
    name = (
        f"{os.environ.get('SLURM_JOB_ID')}-{task_id}"
        if task_id
        else f"local-{os.getpid()}"
    )
    cache_root = root / "fairchem-regression" / name
    inductor = Path(
        os.environ.get("TORCHINDUCTOR_CACHE_DIR", cache_root / "torchinductor")
    ).expanduser()
    inductor.mkdir(parents=True, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor)

    if "TRITON_CACHE_DIR" in os.environ:
        triton = Path(os.environ["TRITON_CACHE_DIR"]).expanduser()
        triton.mkdir(parents=True, exist_ok=True)
        os.environ["TRITON_CACHE_DIR"] = str(triton)
        return

    triton = cache_root / "triton"
    triton.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton)

    extension = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(extension, str) or not extension:
        return
    seed_root = Path(
        os.environ.get(_TRITON_SEED_CACHE, Path.home() / ".triton" / "cache")
    ).expanduser()
    seed_helpers = sorted(seed_root.glob(f"*/cuda_utils{extension}"))
    for source in seed_helpers:
        destination = triton / source.parent.name / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

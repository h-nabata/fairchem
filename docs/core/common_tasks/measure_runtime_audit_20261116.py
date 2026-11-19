"""Reproduce the CPU component measurements from the repository root.

Run with the fairchem test environment, for example:
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
        docs/core/common_tasks/measure_runtime_audit_20261116.py

Requires the baseline commit in local git history. This does not benchmark a
pretrained model or a GPU; the resident-loop case uses synthetic allocations.
"""

from __future__ import annotations

import ast
import gc
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from fairchem.core.calculate import batched_inference, series_b_curvature
from fairchem.core.models.uma import outputs

BASE = "a536295dd6a3aecd4c7e1209ef6f1be1e0945cbe"
ROOT = Path.cwd()
torch.set_num_threads(2)


def previous(module, name):
    path = Path(module.__file__).relative_to(ROOT)
    text = subprocess.check_output(["git", "show", f"{BASE}:{path}"], text=True)
    tree = ast.parse(text)
    node = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    namespace = dict(vars(module))
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def measure(call):
    for _ in range(3):
        call()
    gc.collect()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as trace:
        result = call()
        del result
    trace.export_chrome_trace("/tmp/fairchem_runtime_memory_trace.json")
    events = [
        e["args"]
        for e in json.loads(
            Path("/tmp/fairchem_runtime_memory_trace.json").read_text()
        )["traceEvents"]
        if e.get("name") == "[memory]" and e["args"].get("Device Type") == 0
    ]
    baseline = events[0]["Total Allocated"] - events[0]["Bytes"] if events else 0
    samples = []
    for _ in range(9):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return {
        "tracked_peak_extra_bytes": max(e["Total Allocated"] for e in events)
        - baseline,
        "positive_allocated_bytes": sum(max(0, e["Bytes"]) for e in events),
        "median_seconds": statistics.median(samples),
        "seconds_samples": samples,
    }


report = {
    "base_commit": BASE,
    "torch": torch.__version__,
    "python": platform.python_version(),
    "device": "cpu",
    "threads": 2,
    "notes": "CPU allocator events only; excludes inputs prepared before profiling, Python/native-library memory and GPU memory. Timings are component measurements on a shared host, not whole UMA throughput.",
    "measurements": [],
}


def measure_dtype(dtype):
    generator = torch.Generator().manual_seed(1611)
    matrix = torch.randn(32768, 32, generator=generator, dtype=dtype)
    basis = tuple(torch.linalg.qr(matrix).Q.unbind(1))
    vector = torch.randn(32768, generator=generator, dtype=dtype)
    old = previous(series_b_curvature, "_orthogonalize")
    new = series_b_curvature._orthogonalize
    torch.testing.assert_close(old(vector, basis), new(vector, basis), rtol=0, atol=0)
    report["measurements"].append(
        {
            "component": "lanczos_reorthogonalization",
            "dtype": str(dtype),
            "dimension": 32768,
            "basis_vectors": 32,
            "passes": 2,
            "before": measure(lambda: old(vector, basis)),
            "after": measure(lambda: new(vector, basis)),
        }
    )
    old_hessian = previous(outputs, "compute_hessian_vmap")

    def hessian(function):
        pos = (
            torch.linspace(0.1, 1.8, 768, dtype=dtype).reshape(256, 3).requires_grad_()
        )
        forces = -4 * pos.pow(3)
        return function(forces.flatten(), pos, False, chunk_size=64)

    torch.testing.assert_close(
        hessian(old_hessian), hessian(outputs.compute_hessian_vmap), rtol=0, atol=0
    )
    report["measurements"].append(
        {
            "component": "analytic_hessian",
            "dtype": str(dtype),
            "coordinates": 768,
            "chunk_size": 64,
            "before": measure(lambda: hessian(old_hessian)),
            "after": measure(lambda: hessian(outputs.compute_hessian_vmap)),
        }
    )

    class Model:
        device = torch.device("cpu")

        def __call__(self, state):
            _temporary = torch.zeros(2 * 1048576, dtype=dtype)
            forces = torch.ones(1048576, dtype=dtype)
            return {"forces": forces}

    old_steps = previous(batched_inference, "run_resident_steps")

    def resident(function):
        return function(
            Model(),
            SimpleNamespace(device=torch.device("cpu")),
            lambda state, predictions: state,
            4,
        )

    report["measurements"].append(
        {
            "component": "synthetic_resident_loop",
            "dtype": str(dtype),
            "output_elements": 1048576,
            "temporary_elements": 2097152,
            "steps": 4,
            "before": measure(lambda: resident(old_steps)),
            "after": measure(lambda: resident(batched_inference.run_resident_steps)),
        }
    )


for dtype in (torch.float32, torch.float64):
    measure_dtype(dtype)

path = ROOT / "docs/core/common_tasks/intel_xpu_runtime_measurements_20261116.json"
path.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        [
            {
                "component": m["component"],
                "dtype": m["dtype"],
                "before": m["before"]["tracked_peak_extra_bytes"],
                "after": m["after"]["tracked_peak_extra_bytes"],
            }
            for m in report["measurements"]
        ],
        indent=2,
    )
)

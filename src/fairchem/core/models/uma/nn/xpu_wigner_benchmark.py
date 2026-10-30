"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure optional strict-FP32 XPU kernels and report actual autotune choices.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from fairchem.core.models.uma.nn.portable_wigner import (
    consumer_reference,
    get_csr_topology,
    producer_projected_reference,
    producer_reference,
)
from fairchem.core.models.uma.nn.unified_radial import RadialProjection


def benchmark(
    device="xpu:0", atoms=128, edges=1024, channels=128, hidden=128, repeats=20
):
    """
    Validate each measured forward against PyTorch before recording warm timings.

    Timings are hardware measurements, never inferred from selected tile sizes.
    Every autotune consumer invocation overwrites its output; no candidate can
    accumulate twice into a residual. Peak memory uses PyTorch allocator counters.
    """
    device = torch.device(device)
    if device.type != "xpu" or not torch.xpu.is_available():
        raise ValueError("benchmark requires an available Intel XPU")
    if min(atoms, edges, channels, hidden, repeats) <= 0:
        raise ValueError("atoms, edges, channels, hidden and repeats must be positive")
    if hidden > 512:
        raise ValueError("projected producer benchmark requires hidden <= 512")
    if getattr(torch.backends.mkldnn, "allow_tf32", False):
        raise ValueError("FP32 reference benchmarking requires oneDNN allow_tf32=False")
    if torch.is_autocast_enabled("xpu"):
        raise ValueError("FP32 benchmarking requires XPU autocast to be disabled")
    from fairchem.core.models.uma.nn.xpu_wigner_kernels import (
        FusedConsumer,
        FusedProducer,
        FusedProjectedProducer,
        _consumer_kernel,
        _device_key,
        _producer_kernel,
    )

    with torch.xpu.device(device), torch.no_grad():
        generator = torch.Generator(device=device).manual_seed(712)
        x = torch.randn(atoms, 9, channels, device=device, generator=generator)
        edge_index = torch.randint(
            atoms, (2, edges), device=device, generator=generator
        )
        wigner = torch.randn(edges, 35, device=device, generator=generator) * 0.1
        radial = torch.randn(edges, 12 * channels, device=device, generator=generator)
        projection = RadialProjection(
            *(
                torch.randn(shape, device=device, generator=generator)
                for shape in (
                    (edges, hidden),
                    (12 * channels, hidden),
                    (12 * channels,),
                )
            )
        )
        blocks = [
            torch.randn(edges, n * channels, device=device, generator=generator)
            for n in (3, 4, 2)
        ]
        residual = torch.randn_like(x)
        target = edge_index[1]
        topology = get_csr_topology(target, atoms)

        def fused_producer():
            return FusedProducer.apply(x, edge_index, wigner, radial, channels)

        def reference_producer():
            return producer_reference(x, edge_index, wigner, radial, channels)

        def fused_projected_producer():
            return FusedProjectedProducer.apply(
                x, edge_index, wigner, *projection, channels
            )

        def reference_projected_producer():
            return producer_projected_reference(
                x, edge_index, wigner, projection, channels
            )

        def fused_consumer():
            return FusedConsumer.apply(
                *blocks,
                wigner,
                target,
                topology.order,
                topology.offsets,
                atoms,
                channels,
                residual,
            )

        def reference_consumer():
            return consumer_reference(
                *blocks, wigner, target, atoms, channels, residual
            )

        report = {
            "device": str(device),
            "device_key": _device_key(device),
            "torch_version": torch.__version__,
            "dtype": "float32",
            "atoms": atoms,
            "edges": edges,
            "channels": channels,
            "hidden": hidden,
            "repeats": repeats,
            "index_bits": 32 if topology.index32 else 64,
        }
        for name, fused, reference, kernel in (
            ("producer", fused_producer, reference_producer, _producer_kernel),
            (
                "projected_producer",
                fused_projected_producer,
                reference_projected_producer,
                _producer_kernel,
            ),
            ("consumer", fused_consumer, reference_consumer, _consumer_kernel),
        ):
            torch.xpu.synchronize(device)
            started = time.perf_counter()
            actual, expected = fused(), reference()
            torch.xpu.synchronize(device)
            cold_seconds = time.perf_counter() - started
            actual_values = actual if isinstance(actual, tuple) else (actual,)
            expected_values = expected if isinstance(expected, tuple) else (expected,)
            maximum = 0.0
            for a, b in zip(actual_values, expected_values):
                torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)
                maximum = max(maximum, float((a - b).abs().max()))
            timings = {}
            for label, call in (("reference", reference), ("fused", fused)):
                for _ in range(3):
                    call()
                torch.xpu.synchronize(device)
                torch.xpu.reset_peak_memory_stats(device)
                baseline = torch.xpu.memory_allocated(device)
                started = time.perf_counter()
                for _ in range(repeats):
                    call()
                torch.xpu.synchronize(device)
                timings[label] = {
                    "seconds_per_call": (time.perf_counter() - started) / repeats,
                    "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
                    "extra_peak_allocated_bytes": torch.xpu.max_memory_allocated(device)
                    - baseline,
                    "peak_reserved_bytes": torch.xpu.max_memory_reserved(device),
                }
            config = kernel.best_config
            report[name] = {
                "cold_validation_seconds": cold_seconds,
                "maximum_absolute_error": maximum,
                "selected_config": {**config.kwargs, "num_warps": config.num_warps},
                **timings,
            }
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--edges", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(benchmark(**vars(args)), indent=2))


if __name__ == "__main__":
    main()

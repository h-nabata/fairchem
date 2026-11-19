"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure optional strict-FP32 XPU kernels and report actual autotune choices.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from fairchem.core.common.precision import onednn_matmul_policy
from fairchem.core.models.uma.nn.portable_wigner import (
    CSRTopologyCache,
    consumer,
    consumer_reference,
    producer,
    producer_projected,
    producer_projected_reference,
    producer_reference,
)
from fairchem.core.models.uma.nn.unified_radial import RadialProjection


def benchmark(
    device="xpu:0",
    atoms=128,
    edges=1024,
    channels=128,
    hidden=128,
    repeats=20,
    fused_first=False,
    degree_distribution="uniform",
    measure_adjoint=False,
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
    if degree_distribution not in {"uniform", "hub", "sorted"}:
        raise ValueError("Unsupported target degree distribution")
    if hidden > 512:
        raise ValueError("projected producer benchmark requires hidden <= 512")
    effective_precision, legacy_tf32 = onednn_matmul_policy()
    if effective_precision in {"tf32", "bf16"} or legacy_tf32:
        raise ValueError(
            "FP32 reference benchmarking requires strict oneDNN FP32 matmul (ieee or legacy allow_tf32=False)"
        )
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

        def normal(*shape):
            return torch.randn(
                *shape, device=device, generator=generator, dtype=torch.float32
            )

        x = normal(atoms, 9, channels)
        edge_index = torch.randint(
            atoms, (2, edges), device=device, generator=generator, dtype=torch.int64
        )
        if degree_distribution == "hub":
            edge_index[1, : edges // 2] = 0
        elif degree_distribution == "sorted":
            edge_index = edge_index[:, edge_index[1].argsort(stable=True)].contiguous()
        wigner = normal(edges, 35) * 0.1
        radial = normal(edges, 12 * channels)
        projection = RadialProjection(
            *(
                normal(shape)
                for shape in (
                    (edges, hidden),
                    (12 * channels, hidden),
                    (12 * channels,),
                )
            )
        )
        blocks = [normal(edges, n * channels) for n in (3, 4, 2)]
        residual = normal(x.shape)
        target = edge_index[1]
        torch.xpu.synchronize(device)
        started = time.perf_counter()
        # A private, disabled cache measures actual topology construction.
        topology = CSRTopologyCache(max_entries=0).get(target, atoms)
        torch.xpu.synchronize(device)
        csr_seconds = time.perf_counter() - started

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
            "csr_index_bits": topology.order.element_size() * 8,
            "input_index_bits": edge_index.element_size() * 8,
            "csr_build_seconds": csr_seconds,
            "target_degree_distribution": degree_distribution,
            "seed": 712,
            "measurement_order": ["fused", "reference"]
            if fused_first
            else ["reference", "fused"],
            "scope": "Synthetic operation, forward-only unless adjoint requested. Direct fused calls bypass public dispatch and reuse an existing CSR.",
            "kernel_environment": {
                name: os.getenv(name)
                for name in (
                    "FAIRCHEM_XPU_FUSED_KERNELS",
                    "FAIRCHEM_XPU_FUSED_WIGNER",
                    "FAIRCHEM_XPU_FUSED_RADIAL",
                    "FAIRCHEM_XPU_WIGNER_CSR",
                    "FAIRCHEM_XPU_WIGNER_CSR_CHUNK_EDGES",
                    "TRITON_INTEL_FAST_MATH",
                )
            },
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
            choices_before = len(getattr(kernel, "cache", {}))
            actual = fused()
            torch.xpu.synchronize(device)
            cold_seconds = time.perf_counter() - started
            choices_after = len(getattr(kernel, "cache", {}))
            expected = reference()
            actual_values = actual if isinstance(actual, tuple) else (actual,)
            expected_values = expected if isinstance(expected, tuple) else (expected,)
            maximum = 0.0
            for a, b in zip(actual_values, expected_values, strict=True):
                torch.testing.assert_close(a, b, rtol=5e-5, atol=5e-5)
                maximum = max(maximum, float((a - b).abs().max()))
            inputs = {
                "producer": (x, wigner, radial),
                "projected_producer": (x, wigner, projection[0], projection[1]),
                "consumer": (*blocks, wigner),
            }[name]
            wide = any(t.numel() > 2**31 - 1 for t in (*inputs, *actual_values))
            del a, b, actual_values, expected_values, actual, expected
            timings = {}
            calls = {"reference": reference, "fused": fused}
            for label in report["measurement_order"]:
                call = calls[label]
                for _ in range(3):
                    call()
                torch.xpu.synchronize(device)
                torch.xpu.reset_peak_memory_stats(device)
                baseline = torch.xpu.memory_allocated(device)
                reserved = torch.xpu.memory_reserved(device)
                started = time.perf_counter()
                for _ in range(repeats):
                    call()
                torch.xpu.synchronize(device)
                timings[label] = {
                    "seconds_per_call": (time.perf_counter() - started) / repeats,
                    "baseline_allocated_bytes": baseline,
                    "baseline_reserved_bytes": reserved,
                    "extra_peak_reserved_bytes": torch.xpu.max_memory_reserved(device)
                    - reserved,
                    "peak_allocated_bytes": torch.xpu.max_memory_allocated(device),
                    "extra_peak_allocated_bytes": torch.xpu.max_memory_allocated(device)
                    - baseline,
                    "peak_reserved_bytes": torch.xpu.max_memory_reserved(device),
                }
            public_calls = {
                "producer": lambda: producer(x, edge_index, wigner, radial, channels),
                "projected_producer": lambda: producer_projected(
                    x, edge_index, wigner, projection, channels
                ),
                "consumer": lambda: consumer(
                    *blocks, wigner, target, atoms, channels, residual
                ),
            }
            public = public_calls[name]
            public()
            torch.xpu.synchronize(device)
            started = time.perf_counter()
            for _ in range(repeats):
                public()
            torch.xpu.synchronize(device)
            public_seconds = (time.perf_counter() - started) / repeats
            config = kernel.best_config
            report[name] = {
                "fused_first_call_seconds": cold_seconds,
                "pointer_arithmetic_bits": 64 if wide else 32,
                "public_dispatch_seconds_per_call": public_seconds,
                "in_process_autotune_cache_entries_before": choices_before,
                "in_process_autotune_cache_entries_after": choices_after,
                "maximum_absolute_error": maximum,
                "selected_config": {**config.kwargs, "num_warps": config.num_warps},
                **timings,
            }
        if measure_adjoint:
            # Public dispatch includes bounds checks, topology lookup, forward
            # and the actual adjoint selected by the configured backend.
            with torch.enable_grad():
                x.requires_grad_()
                gs = [t.detach().requires_grad_() for t in blocks]
                cases = [
                    (
                        "producer",
                        lambda: producer(x, edge_index, wigner, radial, channels),
                        reference_producer,
                        (x,),
                    ),
                    (
                        "projected_producer",
                        lambda: producer_projected(
                            x, edge_index, wigner, projection, channels
                        ),
                        reference_projected_producer,
                        (x,),
                    ),
                    (
                        "consumer",
                        lambda: consumer(
                            *gs, wigner, target, atoms, channels, residual
                        ),
                        lambda: consumer_reference(
                            *gs, wigner, target, atoms, channels, residual
                        ),
                        tuple(gs),
                    ),
                ]
                for name, call, ref, args in cases:

                    def backward(fn, args=args):
                        output = fn()
                        outputs = output if isinstance(output, tuple) else (output,)
                        return torch.autograd.grad(sum(t.sum() for t in outputs), args)

                    for actual, expected in zip(
                        backward(call), backward(ref), strict=True
                    ):
                        torch.testing.assert_close(
                            actual, expected, atol=1e-4, rtol=1e-4
                        )
                    del actual, expected
                    torch.xpu.synchronize(device)
                    start = time.perf_counter()
                    for _ in range(repeats):
                        backward(call)
                    torch.xpu.synchronize(device)
                    report[name]["public_forward_adjoint_seconds_per_call"] = (
                        time.perf_counter() - start
                    ) / repeats
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--atoms", type=int, default=128)
    parser.add_argument("--edges", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--fused-first", action="store_true")
    parser.add_argument(
        "--degree-distribution", choices=("uniform", "hub", "sorted"), default="uniform"
    )
    parser.add_argument("--measure-adjoint", action="store_true")
    parser.add_argument(
        "--edge-sweep",
        nargs="+",
        type=int,
        help="Re-run in this process at each listed edge count to observe autotune cache growth",
    )
    args = vars(parser.parse_args())
    sweep = args.pop("edge_sweep")
    result = (
        [benchmark(**{**args, "edges": n}) for n in sweep]
        if sweep
        else benchmark(**args)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

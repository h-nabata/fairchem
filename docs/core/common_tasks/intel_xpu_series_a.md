# Series A: implementation and validation

2026-10-30 / `intel-xpu-fp32`

Follow-up: [library interoperability audit](intel_xpu_interoperability_audit.md)
records the ASE/TorchSim job checks, ownership fixes, configuration-loading
regressions, and boundaries of CPU versus physical-GPU validation.

This document tracks all 33 accuracy-preserving items in the
[optimization roadmap](intel_xpu_optimization_roadmap.md). Series B is deferred:
this change adds no mixed precision, quantization, reduced cutoff, approximate
potential, or altered integrator. FP32 network arithmetic and the existing FP64
energy/stress reductions remain the contract. Reordered sums and factored linear
maps can produce floating-point rounding differences; bitwise identity and
identical long MD trajectories are not promised.

Implementation and hardware qualification are separate. CPU tests execute real
PyTorch operators, synthetic UMA models, gradients and second derivatives.
Mocks check dispatch and lifecycle rules only. No B70 or CUDA performance result
is claimed. The Intel-specific kernels, XPU compilation, graph capture, transfer
overlap and hardware tuning must still pass the supplied tests on the target GPU.

## Implemented items

Paths below are relative to `src/fairchem/core/`.

| ID | Implementation | Activation and boundaries |
| --- | --- | --- |
| A01 | `graph/neighbor_cache.py`: bounded Verlet candidate cache; uncapped outer graph, current-distance filtering and original neighbor cap on every call | `neighbor_skin > 0`, internal graph version 2; cell/PBC/species/partition/settings changes invalidate; cutoff and near-cap ambiguities rebuild, including non-strict cap thresholds |
| A02 | `graph/portable_search.py`: tiled all-pairs, cell and balanced BVH search; shape heuristic | `graph_search_method`, internal graph version 2; `auto` is a heuristic, not a measured device optimum; strict-cap compatibility retains cell traversal |
| A03 | Target-point versus source-cell bounding-box distance bounds remove impossible candidates | Conservative bounds preserve cutoff-boundary candidates |
| A04 | Post-selection Morton edge ordering and owned, bounded CSR topology cache | `graph_edge_order="spatial"`; original atom IDs remain unchanged; single-device graphs |
| A05 | `graph/compute.py:edge_cell_offsets`: form only the needed E×3 image shifts | Removes E×3×3 cell replication; position/cell higher derivatives retained |
| A06 | `nn/portable_geometry.py`: joint distance, Gaussian, envelope and quaternion-Wigner core; 35 quartic monomials instead of 256 products | Portable l=2 quaternion/Gaussian path; optional compilation controls physical kernel fusion |
| A07 | Compact 35-value l=2 Wigner representation throughout portable message passing | `execution_mode="umas_fast_portable"` requires lmax=mmax=2, merged MoLE, checkpointing disabled and `tf32=False`; incompatible explicit settings raise an error |
| A08 | Gather/rotation/radial/packing producer, including lazy radial projection | Portable PyTorch path plus opt-in Triton-XPU forward; differentiable PyTorch adjoint |
| A09 | Unpack/inverse-rotation/CSR reduction consumer with residual output | Portable PyTorch path plus opt-in Triton-XPU forward; first/second derivatives retained |
| A10 | `nn/unified_radial.py`: separate distance and elemental first-layer projections; invalidate tables on weight/embedding changes | Fast PyTorch and portable backends; gradient-bearing weights use a fresh differentiable path |
| A11 | Recomputing LayerNorm/SiLU adjoint reduces saved activations; optional fused Linear/LayerNorm/SiLU XPU forward | Original LayerNorm eps retained; strict FP32 kernel is opt-in and hardware-unverified |
| A12 | Lazy radial final projection, consumed by coefficient block instead of retaining concatenated radial outputs | Portable backend; kernel/reference paths preserve autograd |
| A13 | Blockwise gate plus residual/reduction output construction | Portable backend; no repeated residual accumulation during tuning |
| A14 | Existing `umas_fast_pytorch` made dtype-correct and cache-safe; end-to-end geometry/force/curvature tests | Requires merged MoLE and checkpointing disabled; fixes FP64 conversion rounding, stale block weights and detached weight gradients; actual XPU cases supplied |
| A15 | Stable CSR segmentation and segmented reduction with higher-derivative support | Portable backend; preserves each operation's original accumulation dtype |
| A16 | `units/mlip_unit/inference_runtime.py`: compile geometry, radial and message-passing numerical callables separately | `compile=True, compile_scope="blocks"`, checkpointing disabled; graph building and property heads stay eager; Hessian/HVP use eager cores |
| A17 | Edge-count bucketing through padding to a configurable shape quantum; finite fake edges with zero envelope | `edge_chunk_size > 1`; atom-count padding is not implemented; composition/norm counts remain physical; single-device only |
| A18 | Bounded XPU graph replay of eligible tensor-only numerical calls, with owned inputs and cloned outputs | `xpu_graph=True`, checkpointing disabled, merged experts; shape/weight/stream guards; higher derivatives bypass capture; unsupported cases fall back, device faults/OOM propagate |
| A19 | `calculate/torchsim_interface.py`: reusable validated static metadata; fresh positions/cells each step | `cache_state_metadata=True`; resident step driver accepts the caller's unchanged update rule |
| A20 | `calculate/device_buffers.py`: owned pinned/device buffers, leases and copy/consumer events | Explicit `DeviceBatchStager`; double buffering is optional; differentiable/resident inputs use normal path |
| A21 | `calculate/batched_inference.py`: independent structures packed under atom/edge/system limits; output order restored | `MicrobatchPredictor`; explicit edge counts/bounds required when internal graphs and an edge budget are used |
| A22 | One retained predictor and serial queue per explicit device ordinal | `IndependentDevicePool`; independent jobs, no distributed model parallelism or VRAM pooling |
| A23 | Exact composition/charge/spin/task/settings/device/dtype prepared-MoLE model LRU, including absolute elemental counts | `mole_cache_size > 0`, `merge_mole=True`, Nitro disabled; optional retained-parameter/buffer byte bound; model/template, graph pools and allocator overhead need separate accounting |
| A24 | Capacity reuse for owned staging buffers; bounded prepared-model/CSR/graph caches; explicit ownership | No reuse of autograd-saved intermediates; no hot-loop allocator flushing; reservations are not a global VRAM bound |
| A25 | Streaming accumulation of edge chunks removes the stack of full-node chunk outputs | Existing chunk/checkpoint controls; checkpointing trades recomputation for memory |
| A26 | Per-call requested energy/force/stress/Hessian/HVP/embedding outputs; ASE and TorchSim propagation | `predict(..., requested_properties=(...))`; energy-only avoids coordinate/cell differentiation |
| A27 | Hessian basis rows evaluated in blocks; no full identity allocation | `hessian_chunk_size`; full Hessian output still requires O(N²) storage; one system per dense Hessian |
| A28 | Reverse-over-reverse automatic-differentiation Hessian-vector product | `hessian_vector_product` plus input `hessian_vector`; no finite differences or full Hessian construction |
| A29 | XPU tile candidates and hardware-keyed autotuning; parity-before-timing benchmark | Experimental kernels; no claim that a candidate is B70-optimal before measurement |
| A30 | Range-checked 32-bit fused-kernel indices with 64-bit fallback | Index narrowing only; ordinary graph indices remain int64; floating arithmetic unchanged |
| A31 | Non-strict graph search uses a local origin instead of absolute coordinate magnitude | Physical positions and derivative inputs remain unchanged; strict positive neighbor caps retain historical grid traversal |
| A32 | Conservative boundary-only periodic source images for non-strict search | Includes triclinic/small-cell multiple-image cases; strict positive caps retain historical image traversal; no cutoff shortening |
| A33 | Immutable integer stencil/image tables keyed by shape/device | At most 32 cached tables; tables exceeding 32,768 integer entries bypass caching; floating cell shifts recomputed; device events and stream lifetime handling |

The table describes implemented mechanisms, not 33 independently additive speed
gains. Several share a backend. Edge padding does not stabilize changing atom
counts. Native BVH traversal and correctness guards can synchronize the host and
be slower for some structures. CSR sorting also has a cost. Benchmark these
options separately before combining them.

## Start with reversible, independently measurable options

The example assumes an installed XPU environment and access to the UMA checkpoint;
see [environment setup](intel_xpu.md). It constructs a neutral singlet water
molecule without an external neighbor graph; the XPU preset generates its graph
internally.

```python
from dataclasses import replace

import torch
from ase.build import molecule

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import inference_settings_xpu

atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)
data = AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["charge", "spin"])
settings = replace(
    inference_settings_xpu(),
    neighbor_skin=0.4,
    graph_search_method="cell",
    hessian_chunk_size=32,
)
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="xpu:0", inference_settings=settings
)
prediction = predictor.predict(data, requested_properties=("energy", "forces"))
print(prediction["energy"], prediction["forces"])
```

The skin is an additional candidate-search margin in the same length units as
the cutoff; the physical cutoff is unchanged. Use `neighbor_skin=0` for a fresh
reference. Compare timings for the actual structure sequence, including rebuilds.

For the compact UMA-S path, independently test:

```python
settings = replace(
    inference_settings_xpu(),
    execution_mode="umas_fast_portable",
    activation_checkpointing=False,
    edge_chunk_size=128,
    graph_edge_order="spatial",
)
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="xpu:0", inference_settings=settings
)
```

Compilation can then be enabled with `compile=True, compile_scope="blocks"`;
keep `activation_checkpointing=False`. Test `xpu_graph=True` separately with
merged MoLE weights. These are opt-in because compilation/capture
costs and memory can exceed their benefit on small or highly variable workloads.
The eager paths remain available for curvature and unsupported capture signatures.
Construct a new predictor after changing settings; changing a local `settings`
variable does not reconfigure an existing predictor. Prepared-MoLE caching also
requires a new predictor after model parameters or buffers are modified.

Triton-XPU kernels additionally require `FAIRCHEM_XPU_FUSED_KERNELS=1` and a
compatible Intel Triton backend. With that environment variable unset, the normal
portable path remains usable without Triton. The radial GEMM requests IEEE FP32
matrix products; the Wigner kernels use explicit FP32 products and sums. Both
decline or reject enabled fast-math where detectable; the predictor rejects ambient oneDNN TF32 permission
instead of changing the application's global precision settings. Their backward
paths use differentiable PyTorch expressions, not an untested custom backward
kernel. End-to-end force performance can therefore differ from forward timings.
The fused Linear/LayerNorm/SiLU kernel accepts contiguous FP32 matrices with
input width 1–2048 and output width 16–512. The fused final radial projection
accepts hidden width 1–512. Unsupported layouts or transformation modes use the
native path. Kernel compilation/execution errors propagate.

```bash
FAIRCHEM_XPU_FUSED_KERNELS=1 python -m \
  fairchem.core.models.uma.nn.xpu_wigner_benchmark \
  --device xpu:0 --atoms 128 --edges 4096 --channels 128 --repeats 30
```

This benchmark measures the Wigner producer, projected producer and consumer;
it validates outputs before reporting actual timings and allocator peaks.
It does not measure the separate Linear/LayerNorm/SiLU kernel or substitute for
the full-model force/stress/Hessian tests.

The checkpoint validation CLI exposes the independent Series A switches.
Compare the general FP32 baseline first, then the portable path:

```bash
python -m fairchem.core.scripts.validate_xpu --device xpu:0 \
  --model uma-s-1p2 --stress --compare-merged --output xpu-reference.json
python -m fairchem.core.scripts.validate_xpu --device xpu:0 \
  --model uma-s-1p2 --stress --compare-merged \
  --execution-mode umas_fast_portable --neighbor-skin 0.4 \
  --graph-edge-order spatial --edge-chunk-size 128 \
  --output xpu-series-a.json
```

Leave the fused-kernel environment variable unset for this first comparison.
Then test compilation, capture and explicit kernels individually. The CLI
requests energy, forces and stress together, avoiding duplicate model evaluations
inside each timing sample. It records the settings and checks against the CPU
reference. Its small built-in structures do not predict production MD throughput;
measure the actual sequence, including neighbor rebuilds and cold preparation.

## Curvature and batches

```python
data.hessian_vector = torch.ones_like(data.pos)  # Example direction, [N, 3]
hv = predictor.predict(data, requested_properties=("hessian_vector_product",))[
    "hessian_vector_product"
]
hessian = predictor.predict(data, requested_properties=("hessian",))["hessian"]
```

Use an energy-derived force model. HVP preserves the requested vector shape at
the numerical helper level; the public predictor returns the atom layout [N,3].
Energy normalizer scale applies to derivatives, while energy offsets and element
references do not. A dense Hessian is one-system-only; neither curvature helper
supports graph parallelism. HVP is not a substitute for a complete vibrational
spectrum. Call derivative predictions outside `torch.inference_mode()`;
`torch.no_grad()` is supported because the conservative model enables autograd
where needed.

See [resident simulation and batching](intel_xpu_workloads.md) for independent
NEB/AFIR/displacement jobs, transfer buffers, device workers and TorchSim usage.
The utilities do not change the scientific algorithm or its convergence rules.

## Validation evidence

The broad regression selection on Python 3.12.14 / PyTorch 2.15.0+cpu completed
with **770 passed, 36 skipped, 0 failed**. The skips comprise 20 CUDA-only cases
and 16 Intel-GPU cases. This selection includes graph and UMA numerical unit
tests, existing XPU portability regressions, prediction/runtime, calculator and
validation-CLI tests; it is not the entire repository test suite. Changed files
also pass all required pre-commit checks and `git diff --check`.
The final changed-test selection was repeated after the last runtime corrections:
**274 passed, 13 skipped, 0 failed**. These selections overlap and their counts
must not be added. The repeated selection also covers mixed-dataset unmerged
MoLE with edge padding and the effective compiler fast-math guards.
The standalone geometry/benchmark suite then passed **10 tests**, including
three additional checks that reject TF32, XPU autocast and unsupported projected
width before the optional kernel compiler is imported.

The regression selection covers graph identity and cap ties, cutoff boundaries,
triclinic/mixed PBC, cell/species/metadata changes, stream/ownership contracts,
empty-edge systems, padding masks, combined cache/order/portable paths, actual
synthetic UMA energy/force/stress/curvature, normalized public outputs, and
autograd gradcheck/gradgradcheck/JVP/vmap where supported.

An existing inconsistency in `compute_energy(reduce="mean")` was also fixed:
the energy part used for force differentiation now has the same normalization
as the returned mean energy. Constant-energy/unused-position derivatives return
mathematical zeros while genuinely detached nonempty force inputs are rejected.

CPU validation does not establish pretrained checkpoint parity, B70 driver
stability, graph replay correctness on hardware, maximum production structure
size or a speedup over CUDA. Run the real-XPU cases and checkpoint validation on
the Ubuntu machine before enabling experimental kernels in production.

## Backend and upstream scope

This remains a PyTorch/XPU fork with optional Triton-XPU kernels, not a complete
SYCL rewrite of UMA. The ordinary path delegates to PyTorch's Intel operators and
math libraries. A CUDA-versus-XPU slowdown percentage cannot be inferred from
source alone. Compare identical checkpoint, structures, FP32 policy and requested
outputs, separating first compilation, warm latency, throughput and peak memory.

Changes are isolated into graph, differentiation, numerical backends, runtime
and workload utilities to support separate upstream review. No upstream merge
or hardware qualification is implied by publishing this branch.

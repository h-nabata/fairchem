# Portable UMA performance and memory audit

Date: 2026-10-30. Baseline: fork commit
`588133792a4c36760f841c205dc16b444b731356`.

This pass targets the general PyTorch implementation used by the experimental
single-device Intel XPU path. It removes identified computations, transfers
and allocations without lowering FP32 model precision or the intentional FP64
energy/stress accumulations. Measurements below use Python 3.12.14 and
PyTorch 2.15.0+cpu; they are not B70 measurements.

## Implementation

| Area | Change | Numerical or ownership contract checked |
| --- | --- | --- |
| Wigner coefficient layout | Replace two dense permutation products with indexed gathers; compose reduced-coefficient selection first | FP32/FP64 values, strided/empty inputs, first/second derivatives, state-dict compatibility, CPU Dynamo tracing |
| Graph v2 | Store occupied cells and expand actual candidate counts; bound the direct lookup used for compact grids | Independent ASE neighbors, exact baseline edge order and cell offsets, mixed/skew PBC, neighbor caps |
| Graph temporaries | Reduce system bounds without padding to the largest system, concatenate PBC indices once, retain cell offsets only for returned edges | Unequal batches, large translations and unchanged cutoff arithmetic |
| Neighbor cap | Sort distance rows only for atoms exceeding the cap; retain the all-over-cap path for homogeneous batches | Exact masks/counts for tied, finite and nonfinite distances, strict/non-strict caps and the existing degeneracy rule |
| MoLE chunks | Mix the active system range only; omit empty boundary GEMMs and single-result concatenation | Independent row-wise reference, parameter/routing/input gradients, empty and strided chunks |
| MoLE merge | Build the Linear shell on meta and materialize only the required mixed weights; preserve each backbone tensor's dtype/device and derived buffer values | Independent parameter storage, FP32 models with FP64 buffers, full synthetic FP64 merged/unmerged energy/force/stress agreement |
| Dataset routing | Reuse system/atom masks on the output device and use `where` instead of CPU membership tests and gather/writeback | Dataset ordering, zeros outside each dataset, exclusion of unselected NaN/Inf, independent output storage, higher derivatives |
| Input/output copies | Combine private copy, device transfer and dtype conversion; bypass single-system output gathers/concatenation | Caller and metadata isolation, non-leaf autograd, same synthetic UMA outputs, no retention of unselected multi-system storage |
| Fallback lifetime | Release bound callbacks, local backbone references and recoverable traceback chains before replacement-model loading | Weak-reference checks through real prediction control flow and partially constructed Nitro failures |

Graph v2 also now accepts empty target partitions and privately sorts unsorted
partitions before neighbor limiting. These previously failed during reductions
or neighbor-count indexing. The caller's partition is not changed.

The merged `xpu` preset does not repeatedly run unmerged MoLE. Active-chunk
savings apply to the unmerged/checkpointed path; Wigner, graph and copy changes
also affect ordinary merged inference.

## Reproduced derivative and precision bugs

- Euler `Safeacos` saved a detached intermediate inside custom autograd's
  forward, dropping the second derivative. Saving the input and recomputing
  the clamped denominator in backward preserves the existing regularized first
  derivative while restoring higher derivatives.
- The loop Hessian freed the graph on its final row even when a differentiable
  Hessian was requested. It now retains the graph when `create_graph=True`.
- The vectorized inference Hessian always retained saved intermediates. It now
  releases them when higher derivatives are not requested, matching the loop
  path. A saved-tensor lifetime test reproduced six surviving holders out of
  seven before the fix and none after it while keeping the returned tensors.
- Multidimensional node reduction used `view`, which failed for noncontiguous
  and empty inputs. Direct multidimensional FP64 `index_add_` preserves the
  calculation without requiring a contiguous copy.
- MoLE merging could round loaded FP64 tensors through new FP32 layers. Both
  layer and backbone replacement now preserve source tensor specifications;
  this resolves the CPU merge-fidelity limitation in the preceding audit.

Coupled position/cell/weight calculations independently checked forces, stress,
Hessians and their training gradients against `torch.autograd.functional.hessian`.
This does not validate every pretrained model's Hessian or Intel kernel.

## CPU allocation evidence

The metrics have different scopes and must not be added together. Wigner and
routing values sum positive `self_cpu_memory_usage` after grouping profiler
events by operator type. This excludes nested-operator double counting, but
frees can offset allocations within each group: it is a net allocation metric,
not total allocation traffic or simultaneously live memory. Graph values are
the largest individual event's self allocation. Neither metric is peak memory.
No row is a GPU-capacity or end-to-end speedup claim.

| Case | Metric | Before (bytes) | After (bytes) |
| --- | --- | ---: | ---: |
| 64-atom cluster plus one distant atom, graph v2 | Largest single operator allocation | 4,160,000 | 195,000 |
| Periodic Cu, 256 atoms, graph v2 | Largest single operator allocation | 1,024,000 | 985,536 |
| Neighbor limiting, 2000 atoms: 1999 with degree 1, one with degree 256; cap 16 | Largest single operator allocation | 6,141,952 | 18,040 |
| MoLE chunk touching 1 of 40 systems; 4 experts, 96x64 weights | Mixed-weight allocation | 983,040 | 24,576 |
| Same MoLE chunk, input 5x2x64 | Output concatenation allocation | 3,840 | 0 |
| Wigner layout, 2048 edges, lmax=4, mmax=2, FP32 | Sum of positive operator-net self CPU allocations | 19,456,000 | 7,782,552 |
| Dataset routing, 32 systems and 131,072 atoms | Sum of positive operator-net self CPU allocations | 21,501,952 | 14,421,824 |
| Copy of 10,000-atom input, FP64 to FP32 | Cumulative tensor-copy payload | 760,143 | 440,071 |

The Wigner operation removes two dense products. The routing example removes
four `isin` and twenty `nonzero` calls. The isolated input-copy example removes
nine scalar reads. These operation counts establish eliminated work without
assuming a particular GPU's throughput.

CPU timing experiments indicated improvements for several isolated operations,
but ran alongside other audit jobs. Dense crystal timings varied across runs;
alternating measurements were approximately unchanged. No general CPU or B70
speedup percentage is assigned from those timings.

## Real-device memory measurement

The [XPU guide](intel_xpu.md) gives the hardware test command. Its JSON now
records allocated/reserved bytes and their peaks for model loading, the first
evaluation, warmup and repeated evaluation. Each window synchronizes and resets
its peak statistics. Queries are outside timing intervals, and the cache is
not flushed between repetitions.

These are process-local PyTorch allocator counters for the selected XPU, not
whole-board usage. Existing allocations and reserved cache at a window's start
contribute to its peak. Initial GPU transfer and preparation are lazy and occur
in the first-evaluation window. CPU results have no fabricated XPU memory data.

The preliminary FP64 probe now also checks noncontiguous multidimensional
`index_add_` and backward, matching the stress-reduction path. Failure cleanup
releases the validation function's own predictor/calculator references and
reports cleanup errors separately from the original computation error.

## Regression verification

| Selection | Passed | Skipped | Setup errors |
| --- | ---: | ---: | ---: |
| Portable graphs, MoLE, Wigner, routing and model integration | 229 | 4 | 0 |
| Predictor, device/import guards and hardware-validation CLI | 164 | 2 | 12 |
| Node reductions, force/stress outputs and Hessians | 46 | 0 | 0 |

These selections contain 439 passing tests without double-counting repeated
runs. The final graph selection passed 111 tests, including 13 neighbor-cap
cases added after the initial model selection; repeated graph cases are counted
once in the table. All changed files passed the repository pre-commit hooks
and `git diff --check`. Five skips require an Intel GPU; one requires an unavailable
UMA 1.0 checkpoint. The 12 setup errors occur in existing fixtures that train
small checkpoints: Gloo `init_process_group` fails with `Operation not permitted`
in this execution environment before the affected test bodies run. Those cases
remain unverified; their fixtures were not replaced with mocks to report a pass.

Independent graph reviews additionally matched every output tensor in 100
randomized structures and 24 large-translation cases. The final neighbor-cap
helper matched the old mask and per-system counts in 3000 randomized cases,
including ties, NaN/Inf and finite/nonfinite degeneracy tolerances.

Run from a test environment at the repository root:

```bash
export CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

.venv/bin/python -m pytest \
  tests/core/graph/test_radius_graph.py tests/core/graph/test_pbc.py \
  tests/core/graph/test_radius_graph_memory.py \
  tests/core/models/uma/test_xpu_graph_portability.py \
  tests/core/models/uma/test_xpu_portable.py \
  tests/core/models/uma/nn/test_mole.py \
  tests/core/models/uma/nn/test_portable_matmul_memory.py \
  tests/core/models/uma/test_portable_wigner_memory.py \
  tests/core/models/uma/test_moe_routing_memory.py \
  tests/core/models/uma/test_escn_mole.py tests/core/models/uma/test_escn_md.py \
  -m 'not gpu and not pretrained' \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra

.venv/bin/python -m pytest \
  tests/core/units/mlip_unit/test_inference_settings.py \
  tests/core/units/mlip_unit/test_unit_utils.py \
  tests/core/units/mlip_unit/test_xpu_predict.py \
  tests/core/units/mlip_unit/test_xpu_preparation.py \
  tests/core/units/mlip_unit/test_predict_memory.py \
  tests/core/units/mlip_unit/test_predict.py \
  tests/core/common/test_xpu_distutils.py \
  tests/core/common/test_optional_backend_imports.py \
  tests/core/calculate/test_xpu_pretrained.py \
  tests/core/calculate/test_xpu_torchsim_dispatch.py \
  tests/core/calculate/test_xpu_unsupported_backends.py \
  tests/core/scripts/test_validate_xpu.py \
  -m 'not gpu and not pretrained' \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra

.venv/bin/python -m pytest tests/core/models/uma/test_outputs_memory.py \
  tests/core/models/uma/test_outputs.py tests/core/models/uma/test_efs_head_hessian.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra
```

## Limits

There is no Intel or NVIDIA hardware validation in this record. PyTorch 2.13
and 2.14, complete training/distributed suites, large pretrained systems,
long dynamics and Intel compilation still require their respective tests.

Graph construction still scales with the actual number of periodic images and
candidate pairs. Its existing absolute-coordinate grid-coarsening heuristic
also remains; changing it requires a separate check of integer range and
neighbor tie ordering. None of the allocation reductions makes arbitrarily
dense or large systems fit in a fixed amount of memory.

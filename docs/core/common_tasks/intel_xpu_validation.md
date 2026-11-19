# Intel XPU source audit and validation record

Date: 2026-10-30. This second audit builds on fork commit
`3b071a757edc8ceb9f27ded0322f384b87d8c781` and upstream source
`1d408a1138ddfb76e0a9d64e2e055d73339fbeba`.

The subsequent [performance and memory audit](intel_xpu_performance.md)
records later changes and validation, including the CPU FP64 merge-fidelity fix.

The test host used Python 3.12.14 and PyTorch 2.15.0+cpu. No Intel or NVIDIA
GPU execution, downloaded UMA checkpoint, or B70 timing is represented below.
See [the XPU guide](intel_xpu.md) for setup, hardware acceptance commands and
the supported single-device FP32 scope.

## Results

These four disjoint selections passed 314 tests in total. Five tests requiring
real Intel hardware were skipped. Repeated reviewer runs are not counted.

| Selection | Passed | Skipped | Deselected |
| --- | ---: | ---: | ---: |
| Neighbor graphs, AtomicData, normalization and portable UMA | 142 | 4 | 21 |
| Inference, imports, device routing, CPU D3 and validation CLI | 138 | 1 | 0 |
| Existing Ray model/device/deployment configuration tests | 32 | 0 | 29 |
| Forces-only Hessian analytic regression | 2 | 0 | 0 |

All changed files passed the repository pre-commit hooks and `git diff --check`.
This is a targeted regression selection, not the complete fairchem suite or
the CPU/CUDA version matrix.

## Reproduction

From the repository root in a CPU test environment with the test dependencies:

```bash
export CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

.venv/bin/python -m pytest \
  tests/core/graph/test_radius_graph.py tests/core/graph/test_pbc.py \
  tests/core/datasets/test_atomic_data.py \
  tests/core/models/uma/nn/test_layer_norm.py \
  tests/core/models/uma/test_xpu_graph_portability.py \
  tests/core/models/uma/test_xpu_portable.py \
  -m 'not gpu and not pretrained' \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra

.venv/bin/python -m pytest \
  tests/core/units/mlip_unit/test_inference_settings.py \
  tests/core/units/mlip_unit/test_unit_utils.py \
  tests/core/units/mlip_unit/test_xpu_predict.py \
  tests/core/units/mlip_unit/test_xpu_preparation.py \
  tests/core/common/test_xpu_distutils.py \
  tests/core/common/test_optional_backend_imports.py \
  tests/core/calculate/test_xpu_pretrained.py \
  tests/core/calculate/test_xpu_torchsim_dispatch.py \
  tests/core/calculate/test_xpu_unsupported_backends.py \
  tests/core/calculate/test_dftd3_properties.py \
  tests/core/scripts/test_validate_xpu.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra

.venv/bin/python -m pytest tests/core/components/test_batch_server.py \
  -k 'model_spec or check_predict_unit_device or resolve_device or prepare_deployment_config' \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra

.venv/bin/python -m pytest tests/core/models/uma/test_efs_head_hessian.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -ra
```

## What the tests establish

- Fresh-process imports succeed when optional CUDA extensions raise import or
  shared-library errors. Nitro/Triton implementation packages are not eagerly
  loaded by portable model registration.
- FakeTensor and mocks exercise XPU placement, ordinal preservation, autocast
  scopes and early rejection. They do not emulate Intel numerical kernels.
- Real CPU computations use synthetic UMA weights, independent ASE neighbor
  references and local synthetic D3 parameters. Analytic quadratic energies
  verify both loop and vectorized Hessians.
- CPU normalization layers trace with Dynamo `backend="eager"`, `fullgraph=True`,
  with matching gradients. This does not test XPU compilation.
- The validation CLI rejects broadcastable wrong shapes and later NaN/Inf
  outputs, and checks all six stress components against strain derivatives.
  An independent CPU ASE EMT check on distorted Cu gave a maximum stress
  difference of 9.46e-11 eV/Angstrom^3 at strain step 1e-5. This validates the
  finite-difference calculation, not UMA or Intel-device accuracy.

An empty Python environment also resolved the editable fairchem package with
`torch==2.15.0+xpu` from `https://download.pytorch.org/whl/xpu` using
`uv pip install --dry-run`: 131 packages resolved successfully. This was
dependency metadata resolution only, not an XPU installation or kernel test.

## Remaining hardware checks

Real B70 execution must still verify pretrained energy/force/stress agreement,
FP64 accumulation and backward kernels, memory use and synchronized timings.
Full FP64 network merging, XPU compilation, distributed serving, all TorchSim
integrators, and long dynamics trajectories are not validated by this record.
Unsupported Ray allocation and XPU D3 requests now fail explicitly; D3 can
run on CPU alongside a local XPU UMA calculator.

### Reproduce a hardware comparison

Each validation report identifies the resolved XPU ordinal, physical device,
driver version when available, PyTorch build, source revision, effective settings
and SHA-256 of the local checkpoint. Use the same checkpoint and frame sequence
on each backend. Precision-specific tolerances and finite-difference widths
can be overridden with the CLI flags.

```bash
python -m fairchem.core.scripts.validate_xpu --device xpu:1 --model "$MODEL" \
  --compare-merged --stress --hvp --fp64-reference --neighbor-skin 0.5 \
  --geometry-workloads subskin rebuild strain --output validation.json
python -m fairchem.core.scripts.benchmark_xpu_batch --checkpoint "$MODEL" \
  --devices xpu:0 xpu:1 --systems 4 --output batch-two-devices.json
```

Use `--structure` and `--task` with the validator to test an ASE trajectory.
The fixed-geometry timing records repeatability. The optional workloads alternate
between the initial geometry and a displaced or strained geometry, report
neighbor-cache counters and compare the changed-geometry outputs with CPU.
Requested graph replay or compilation must show execution evidence. Per-block
call counts include eager bypasses during derivative evaluation.

The batch benchmark runs serial and packed requests in each independent
process, synchronizes the selected cards, and counts total structures over the
shared elapsed interval. Run it again with a single device for the one-card
comparison. The Ray serving benchmark supports CPU and CUDA. Its per-concurrency
speedup uses the same structures in its local serial measurement. Each
successful predictor invocation contributes an identifier to its outputs so the
report can count a single packed call serving multiple clients. The identifier
and its transport contribute to both reported timings.

For fixed-cell NVE, `benchmark_md --sample-stream trajectory.jsonl --aggregate-only
--timing-block 10` retains bounded timing observations while writing the sampled
energy trajectory. The report includes both maximum absolute drift from the
post-warmup energy and signed least-squares slope in eV/atom/ps.
`compare_series_b --all-frames` alternates model order by frame, measures
deviations across repeated calls and compares relative energies within the
same composition, charge and spin. `--candidate-first` reverses the starting
order when repeating a run in a fresh process.

The `Xe hardware validation` workflow runs only on the fork's trusted
`intel-xpu-fp32` branch when `XE_B70_CI=true`. Point `XE_B70_CHECKPOINT` at
a local, readable UMA checkpoint on a Linux runner labeled `xe-b70-dual`
with two visible B70s, an XPU build of PyTorch, Intel Triton, and the project's
test dependencies. It archives the numerical comparisons and kernel test results.

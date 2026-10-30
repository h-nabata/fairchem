# Experimental Intel XPU inference

This branch adds single-device UMA inference through PyTorch's native `xpu`
backend. It targets FP32 model parameters and inputs. It has **not yet been
validated on an Intel Arc Pro B70**: CPU regression tests and simulated device
dispatch cannot establish real XPU kernel correctness or performance.

## Environment

Intel lists Arc Pro B70 with Ubuntu 24.04/26.04 in its
[PyTorch prerequisites](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-15.html).
The documented current combination is PyTorch 2.15 and OMIX 0.4.0. Consult the
[OMIX installation guide](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)
for the exact supported OS revisions and driver packages. Ubuntu 26.10 is not
listed in that matrix. Do not replace a working system driver merely to install
this Python package.

Use an isolated environment. The XPU wheel supplies its Python runtime
dependencies; Intel advises against sourcing `oneapi/setvars.sh` when using
prebuilt wheels because it can introduce version conflicts. IPEX is not
required. From this repository:

```bash
uv venv --python 3.12 .venv-xpu
uv pip install --python .venv-xpu/bin/python 'torch==2.15.0' \
    --index https://download.pytorch.org/whl/xpu
uv pip install --python .venv-xpu/bin/python -e packages/fairchem-core
.venv-xpu/bin/python -c 'import torch; print(torch.__version__); print(torch.xpu.is_available())'
```

Install the XPU wheel first so the editable install reuses it. The existing
`uv sync` defaults select CUDA and should not be used for this environment.
The CPU/CUDA dependency groups keep PyTorch 2.13. This branch widens the
package requirement to `>=2.13,<2.16`; this is experimental compatibility,
not a claim that every training, CUDA or compiled path is tested on all three
minor versions. Upstream's Linux package dependencies still install NVIDIA
libraries for other features. Their presence does not make UMA execute on
CUDA, and this branch does not remove those dependencies globally.

## FP32 calculation

After obtaining access to the [UMA checkpoints](https://huggingface.co/facebook/UMA)
and authenticating with the Hugging Face CLI, run:

```python
from ase.build import molecule

from fairchem.core import FAIRChemCalculator, pretrained_mlip

predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2p1",
    device="xpu:0",
    inference_settings="xpu",
)
atoms = molecule("H2O")
atoms.info.update(charge=0, spin=1)
atoms.calc = FAIRChemCalculator(predictor, task_name="omol")
print(atoms.get_potential_energy())
print(atoms.get_forces())
```

The `xpu` preset uses the general PyTorch backend, internal neighbor graph v2,
FP32 parameters, no TF32, no Nitro and eager execution. It merges the linear
experts for repeated evaluation of a fixed composition/task/charge/spin;
the existing consistency checks restore the unmerged path if those inputs
change. This avoids repeatedly evaluating all experts, but a B70 speedup has
not been measured. `device="xpu:1"` selects a second Intel GPU explicitly.

CUDA Nitro, `umas_fast_gpu`, graph v3, TF32 and multi-worker graph parallelism
are outside this implementation. Unsupported XPU settings raise an error.
If XPU is unavailable, an explicit XPU request raises an error instead of
silently running on the CPU. On a mixed NVIDIA/Intel host, explicitly select
`xpu:0`: automatic device selection retains CUDA preference.
Explicit XPU availability, ordinal and parallel-worker checks run before the
public pretrained loader downloads a checkpoint.

Do not wrap conservative UMA calculations in `torch.inference_mode()`: forces,
stress and Hessians require autograd. The predictor rejects that context before
preparing the model. An outer `torch.no_grad()` context is supported because the
conservative prediction section explicitly enables gradients and then restores
the caller's state.

FP32 here describes the network's base precision. UMA intentionally accumulates
system energies and virial terms in FP64. This branch preserves those sums
and their gradients rather than silently reducing their precision. The
validation script probes FP64 `index_add` and its backward operation as well
as reporting the device's `has_fp64` property when available.

This is not an end-to-end FP64 network implementation. In particular, existing
MoLE merge constructors create new layers at the default dtype; selecting an
FP64 base dtype does not establish full FP64 fidelity during merging. The
FP32 network path and its intentional FP64 accumulations are the target here.

## Hardware acceptance test

From the repository root, use the environment created above:

```bash
.venv-xpu/bin/python -m fairchem.core.scripts.validate_xpu --help
.venv-xpu/bin/python -m fairchem.core.scripts.validate_xpu \
    --device xpu:0 --stress --compare-merged --output b70-xpu-validation.json
```

The report distinguishes CPU reference calculations from actual XPU execution.
It checks tensor placement, finite outputs, energy/force agreement, optional
periodic stress, a force component against a central energy difference, and
optional merged/unmerged agreement. Timings synchronize XPU and force fresh
ASE calculations so that the calculator cache is not mistaken for GPU speed.
Every warmup and timed evaluation checks output shapes and finite values; these
checks are included in the reported timings. With `--stress`, all six Voigt
components are also checked independently against energy strain derivatives on
CPU and each requested XPU mode. `--stress-fd-step` defaults to `0.001` and
`--stress-fd-tolerance` to `5e-4` eV/Å³. These new finite-difference criteria do
not change the stricter CPU/XPU stress comparison tolerance.
Warmup and repeated timings are separate. Tolerances are numerical acceptance
criteria for these small tests, not the model's error relative to DFT.
The JSON records command options and, on an exception, `active_check` identifies
the structure, device, merge setting and failing stage.

The default cases are a molecule and a small distorted periodic crystal.
They do not establish production suitability for large slabs, long molecular
dynamics trajectories, other tasks, Hessians or finite-difference vibrational
frequencies. In particular, validate the displacement and force noise at the
step size used in a scientific workflow.

Compilation can be tested separately with the script's `--compile` option.
PyTorch supports XPU compilation, but UMA's graph construction, dynamic shapes
and derivative paths still require real-device validation. Keep eager mode
as the initial correctness reference. This branch does not claim that XPU
matches NVIDIA Nitro throughput.

## Other interfaces and optional backends

The TorchSim wrapper preserves an explicit `xpu:<index>` and keeps the concrete
ordinal resolved by its predictor. Its automatic selection prefers CUDA, then
XPU, then CPU. The wrapper uses the unmerged `batch` settings for heterogeneous
systems. Dispatch tests do not establish that all TorchSim integrators or
optional kernels work on Intel GPUs.

Ray Serve, `InferenceBatcher`, multiplexed `ModelSpec` serving, and automatic
batch-size probing do not implement XPU resource allocation in this branch.
Explicit XPU requests are rejected rather than scheduled as CPU jobs or given
unmeasured CPU timing defaults. Local independent single-device predictors
remain available.

The nvalchemiops/Warp D3 implementation supports CPU and CUDA, not XPU. To add
D3 to an XPU UMA calculator, select `device="cpu"` on `DFTD3Calculator`; this
does not change the UMA calculator's device. Selecting XPU for D3 raises an
error before optional-backend import or parameter download.

Portable UMA model registration no longer imports Nitro/Triton implementation
packages. The NVIDIA neighbor-list module and `fairchem_cpp` are loaded only
when their respective paths are used. This avoids otherwise unrelated CUDA
binary failures blocking ordinary UMA imports. The CUDA dependencies declared
by the upstream package remain installed; they have not been removed from
features that use them.

## Source-level regression coverage

Additional CPU/simulated-device tests cover:

- Public loader rejection before checkpoint download and XPU ordinal retention
  through TorchSim.
- Importing public UMA APIs and registering models when optional CUDA extensions
  raise a shared-library error; retrying model registration after a failure.
- Explicit periodic-boundary overrides, mixed-periodicity batches and
  `always_use_pbc` batch shapes. Neighbor lists are compared with known cells
  and the existing independent ASE references.
- Device-derived autocast protection in normalization layers and CPU Dynamo
  full-graph tracing of those layers with gradient comparison.
- Nested override precedence without mutating the caller, CPU preparation of
  FP32 MoLE from double-precision inputs, and protection against ambient CPU
  autocast rounding merged weights.
- Conservative prediction under `no_grad`, clear rejection of `inference_mode`,
  and no CUDA TF32 state mutation during XPU inference.
- Forces-only Hessians in evaluation mode, checked against a quadratic energy
  with an exact Hessian for both loop and vectorized differentiation.
- Validation-tool shape checks, nonfinite outputs during timing, six-component
  stress finite differences, and restoration after failed evaluations.

These tests use synthetic weights and CPU kernels where numerical values are
required. FakeTensor checks test device/shape/dtype propagation, not Intel
kernel execution. Real-XPU tests are skipped on CPU-only hosts.
The [source-audit validation record](intel_xpu_validation.md) lists the exact
test selections, results and remaining coverage gaps.

## Upstream scope

The existing [Aurora XPU proposal, PR #2182](https://github.com/facebookresearch/fairchem/pull/2182)
addresses a broader Intel GPU environment. This branch is an independent,
narrow single-device FP32 implementation. Before proposing a merge, compare
with that PR's current state, attach real B70 correctness and timing reports,
and run the applicable CPU/CUDA regression matrix. The dependency-range change
and the validation utility can be reviewed separately from the device-routing
change. Upstream acceptance and the contributor license agreement remain
maintainer/contributor decisions.

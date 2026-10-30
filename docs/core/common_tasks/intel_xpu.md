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

FP32 here describes the network's base precision. UMA intentionally accumulates
system energies and virial terms in FP64. This branch preserves those sums
and their gradients rather than silently reducing their precision. The
validation script probes FP64 `index_add` and its backward operation as well
as reporting the device's `has_fp64` property when available.

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
Warmup and repeated timings are separate. Tolerances are numerical acceptance
criteria for these small tests, not the model's error relative to DFT.

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

## Upstream scope

The existing [Aurora XPU proposal, PR #2182](https://github.com/facebookresearch/fairchem/pull/2182)
addresses a broader Intel GPU environment. This branch is an independent,
narrow single-device FP32 implementation. Before proposing a merge, compare
with that PR's current state, attach real B70 correctness and timing reports,
and run the applicable CPU/CUDA regression matrix. The dependency-range change
and the validation utility can be reviewed separately from the device-routing
change. Upstream acceptance and the contributor license agreement remain
maintainer/contributor decisions.

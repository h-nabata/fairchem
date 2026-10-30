# Resident simulation and independent structure batching

These utilities implement roadmap items A19–A22 without changing model
precision, integration rules, finite-difference displacements, NEB spring forces
or convergence criteria. CPU numerical tests include the actual synthetic UMA
backbone, energy/force/stress heads and prediction path. XPU stream overlap and
multi-B70 throughput still require measurements on the target machine.

## Independent structures on one device

`MicrobatchPredictor` groups compatible single-structure `AtomicData` inputs by
atom, edge and system budgets. It packs larger structures first, then restores
the original input order. Every returned property owns its storage. Different
datasets, tensor dtypes and source devices are kept in different batches.

```python
from ase.io import read

from fairchem.core import pretrained_mlip
from fairchem.core.calculate.batched_inference import (
    InferenceBatchLimits,
    MicrobatchPredictor,
)
from fairchem.core.datasets.atomic_data import AtomicData

frames = read("structures.xyz", index=":")
for atoms in frames:
    atoms.info.setdefault("charge", 0)
    atoms.info.setdefault("spin", 1)
structures = [AtomicData.from_ase(atoms, task_name="omol") for atoms in frames]
predictor = pretrained_mlip.get_predict_unit(
    "uma-s-1p2", device="xpu:0", inference_settings="batch"
)
runner = MicrobatchPredictor(
    predictor,
    InferenceBatchLimits(max_atoms=1024, max_systems=16),
    requested_properties=("energy", "forces"),
)
predictions = runner.predict_many(structures)
```

An input larger than a budget is rejected before prediction. These limits never
truncate model neighbors. `max_edges` can additionally limit graph size:

```python
runner = MicrobatchPredictor(
    predictor,
    InferenceBatchLimits(max_atoms=1024, max_systems=16, max_edges=100_000),
)
# One known count or conservative upper bound per input structure:
predictions = runner.predict_many(structures, edge_counts=edge_counts)
```

For internally generated graphs, an empty input edge array is not an edge-count
estimate. Supplying `max_edges` without counts in this case raises an error.
Estimated counts are planning hints, not a guarantee of model peak memory. Peak
memory also depends on channels, derivatives, checkpointing and actual graphs.

The default output layouts cover energy, force, stress, charges, magnetic moments
and Hessian-vector products. Additional outputs require an explicit mapping,
for example `property_levels={"descriptor": "atom"}`. A whole-batch dense
Hessian must not be represented as independent per-system output without an
appropriate model API.

## Reusable input staging

The optional stager copies CPU inputs into owned pinned storage and then into
reusable device buffers. A copy stream and consumer events protect asynchronous
lifetimes. Two slots allow the next transfer to overlap current computation.

```python
from fairchem.core.calculate.device_buffers import DeviceBatchStager

stager = DeviceBatchStager("xpu:0", slots=2, max_batch_bytes=64 * 1024 * 1024)
runner = MicrobatchPredictor(predictor, stager=stager)
try:
    predictions = runner.predict_many(structures)
finally:
    stager.close()
```

`max_batch_bytes` bounds tensor capacity per slot on each memory side; two slots
can therefore reserve up to four times that amount across host and device.
Staging preserves input dtypes. It rejects differentiable or GPU-resident inputs;
those should use the normal/resident prediction path. The predictor can still
compute conservative forces because it creates private differentiable inputs.
Each worker needs its own stager. The stager is not a concurrent shared queue.

When using `stage()` directly, consume it with a context manager and retain only
outputs that own their storage:

```python
with stager.stage(batch) as device_batch:
    result = predictor.predict(device_batch)
    result = {key: value.clone() for key, value in result.items()}
```

The leased input and its views become reusable after context exit. Slots cannot
be reused while leased, and reuse waits for the recorded consumer event. The
pool never flushes the accelerator allocator in the simulation hot path.
If consuming a lease on several streams, call `lease.wait()` on each stream
before submitting work. Release joins their work and protects allocator
lifetimes; no work may use the leased input after release.

## Independent device workers

Each `IndependentDevicePool` worker creates one predictor lazily and serializes
its jobs. Inputs are distributed by queued atom/edge work. Device ordinals are
explicit and remain associated with their worker. The pool does not use Ray,
NCCL, a distributed process group or model sharding.

```python
from fairchem.core.calculate.batched_inference import IndependentDevicePool


def make_predictor(device):
    return pretrained_mlip.get_predict_unit(
        "uma-s-1p2", device=str(device), inference_settings="batch"
    )


with IndependentDevicePool(
    make_predictor,
    ["xpu:0", "xpu:1"],
    limits=InferenceBatchLimits(max_atoms=1024, max_systems=16),
    requested_properties=("energy", "forces"),
) as pool:
    predictions = pool.predict_many(structures)
```

Outputs remain on the worker's device. Two 32 GB devices do not become one 64 GB
allocation. For asynchronous jobs, use `future = pool.submit(structures)` and
`future.result()`. Input objects must remain unchanged until their future is
complete. Worker completion waits for its GPU event, so returned tensors are
ready even when the caller selects another stream. Different devices still
compute concurrently. Exceptions propagate to the caller; cancelled queued jobs release
their scheduling load. The context manager waits for running work and releases
the retained worker predictors.

## TorchSim state on the GPU

`FairChemModel` now caches validated static composition and batch metadata.
Unchanged metadata avoids rebuilding `bincount`, Python slice lists and repeated
`AtomicData` validation. Positions and cells are refreshed on every call;
distance-dependent graph features and autograd graphs are never cached here.
Normal in-place changes, tensor replacement, device changes and charge/spin
changes invalidate the cache. Strong source references prevent pointer reuse
from accidentally matching another tensor.

For programs mutating metadata through raw pointers or `.data`, construct
`FairChemModel(..., cache_state_metadata=False)`, because those writes bypass
PyTorch's mutation counters. Inference tensors without counters automatically
take the uncached validation path. The wrapper requests only its implemented
properties when the predictor supports output selection.

`run_resident_steps(model, state, update, steps)` provides an optional device-local
loop. The caller supplies `update(state, predictions)` using the same selected
integrator or optimizer; the utility changes no physical rule. It calls the
model, optionally observes the matching pre-update state/predictions, and applies
the update exactly once per step. The update must keep state on the model device.
Only an optional observer decides whether to copy data to CPU or log a trajectory.

## Validation

```bash
CI=true OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest \
  tests/core/calculate/test_batched_inference.py \
  tests/core/calculate/test_torchsim_resident_metadata.py \
  tests/core/calculate/test_xpu_torchsim_dispatch.py \
  -c packages/fairchem-core/pyproject.toml -o addopts='' -q
```

The real-XPU staging test is skipped without an Intel GPU. The two-device ordinal
and serialization test uses dispatch mocks; it establishes routing behavior, not
hardware concurrency or speed. CPU worker tests run genuine tensor operations,
and a separate integration test compares synthetic UMA energy, forces and stress
for single structures, microbatches and the retained CPU worker.

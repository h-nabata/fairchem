"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from ase.build import molecule

from fairchem.core.calculate import torchsim_interface as interface
from fairchem.core.datasets.atomic_data import AtomicData

ts = pytest.importorskip("torch_sim")


def _state(dtype=torch.float32, periodic=False):
    atoms = molecule("H2O")
    atoms.set_cell([8, 9, 10])
    atoms.pbc = periodic
    return ts.io.atoms_to_state([atoms], device="cpu", dtype=dtype)


class HarmonicPredictor:
    """
    Analytic conservative potential for actual TorchSim integrator execution.
    """

    device = "cpu"
    supports_requested_properties = True

    def __init__(self):
        self.dataset_to_tasks = {"omol": []}

    def predict(self, data, requested_properties=None):
        energy = data.pos.new_zeros(len(data.natoms)).index_add(
            0, data.batch, 0.5 * data.pos.square().sum(-1)
        )
        self.latest = {"energy": energy, "forces": -data.pos}
        return self.latest


def _wrapper(monkeypatch, predictor=None, **kwargs):
    predictor = predictor or HarmonicPredictor()
    monkeypatch.setattr(interface, "setup_imports", lambda: None)
    monkeypatch.setattr(interface, "setup_logging", lambda: None)
    monkeypatch.setattr(
        interface.pretrained_mlip, "get_predict_unit", lambda *a, **kw: predictor
    )
    return interface.FairChemModel("unused", device="cpu", **kwargs)


@pytest.mark.parametrize("field", ["atomic_numbers", "charge", "spin", "pbc"])
def test_external_numpy_mutation_invalidates_cached_metadata(field):
    state = _state()
    state.system_extras.update(charge=torch.tensor([0.0]), spin=torch.tensor([1.0]))
    converter = interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    value = getattr(state, field)
    version = value._version
    value.numpy().flat[0] = (
        7 if field == "atomic_numbers" else 1 if field != "spin" else 2
    )
    assert value._version == version
    actual = converter(state, "omol", torch.float32)
    expected = interface._simstate_to_atomicdata_batch(state, "omol", torch.float32)
    for name in ("atomic_numbers", "charge", "spin", "pbc", "pos"):
        torch.testing.assert_close(actual[name], expected[name])


def test_returned_batches_cannot_mutate_state_or_private_cache():
    state = _state()
    before = state.clone()
    converter = interface._SimStateConverter()
    for _ in range(3):
        batch = converter(state, "omol", torch.float32)
        expected = interface._simstate_to_atomicdata_batch(state, "omol", torch.float32)
        for field in ("atomic_numbers", "charge", "spin", "pbc", "pos", "cell"):
            torch.testing.assert_close(batch[field], expected[field])
            batch[field].zero_()
        batch.dataset[0] = "corrupted"
        batch.get_batch_stats()[0]["pos"][1] = 999
        torch.testing.assert_close(state.positions, before.positions)
        torch.testing.assert_close(state.atomic_numbers, before.atomic_numbers)
        torch.testing.assert_close(state.cell, before.cell)
    assert copy.deepcopy(converter)._template is None


def test_shared_converter_keeps_concurrent_system_metadata_separate():
    states = [_state(), _state()]
    states[0].system_extras["charge"] = torch.tensor([0])
    states[1].system_extras["charge"] = torch.tensor([1])
    converter = interface._SimStateConverter()

    def run(index):
        batch = converter(states[index], "omol", torch.float32)
        return int(batch.charge[0])

    indices = [0, 1] * 20
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(run, indices)) == indices


@pytest.mark.parametrize(
    "name,value",
    [
        ("charge", 0.5),
        ("spin", 1.5),
        ("charge", float("nan")),
        ("spin", float("inf")),
        ("charge", 101.0),
        ("spin", -1.0),
    ],
)
def test_invalid_physical_metadata_rejected_before_integer_conversion(name, value):
    state = _state()
    state.system_extras[name] = torch.tensor([value])
    with pytest.raises(ValueError, match=name):
        interface._simstate_to_atomicdata_batch(state, "omol")


def test_charge_and_spin_are_per_system_and_omol_default_matches_ase():
    state = _state()
    data = interface._simstate_to_atomicdata_batch(state, "omol")
    assert data.charge.tolist() == [0]
    assert data.spin.tolist() == [1]
    state.atom_extras["charge"] = torch.zeros(3)
    with pytest.raises(ValueError, match="per-system"):
        interface._simstate_to_atomicdata_batch(state, "omol")


def test_cached_metadata_revalidates_extra_scope():
    state = _state()
    state.system_extras["charge"] = torch.tensor([0])
    converter = interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    state.atom_extras["charge"] = state.system_extras.pop("charge")
    with pytest.raises(ValueError, match="per-system"):
        converter(state, "omol", torch.float32)


def test_reversed_system_order_is_rejected_before_graph_construction():
    state = ts.io.atoms_to_state([molecule("H2"), molecule("H2")], device="cpu")
    # TorchSim permits this equal-size reversal; AtomicData graphs require order.
    state.system_idx = torch.tensor([1, 1, 0, 0])
    with pytest.raises(ValueError, match="ascending contiguous"):
        interface._simstate_to_atomicdata_batch(state, "omol")


def test_custom_task_and_cache_directory_are_forwarded(monkeypatch, tmp_path):
    calls = []
    predictor = HarmonicPredictor()
    predictor.dataset_to_tasks = {"custom_finetune": []}
    monkeypatch.setattr(interface, "setup_imports", lambda: None)
    monkeypatch.setattr(interface, "setup_logging", lambda: None)

    def load(*args, **kwargs):
        calls.append(kwargs)
        return predictor

    monkeypatch.setattr(interface.pretrained_mlip, "get_predict_unit", load)
    path = tmp_path / "new-cache"
    wrapper = interface.FairChemModel("unused", device="cpu", model_cache_dir=path)
    assert wrapper.task_name == "custom_finetune"
    assert calls[0]["cache_dir"] == str(path)
    wrapper(_state())
    with pytest.raises(ValueError, match="Unknown task_name"):
        interface.FairChemModel("unused", device="cpu", task_name="wrong")
    predictor.dataset_to_tasks["second"] = []
    with pytest.raises(ValueError, match="task_name is required"):
        interface.FairChemModel("unused", device="cpu")


@pytest.mark.parametrize("device", ["cuda:1", torch.device("cuda:1")])
def test_explicit_cuda_ordinal_is_not_discarded(monkeypatch, device):
    calls = []
    predictor = HarmonicPredictor()
    predictor.device = "cuda:1"
    monkeypatch.setattr(interface, "setup_imports", lambda: None)
    monkeypatch.setattr(interface, "setup_logging", lambda: None)

    def load(*args, **kwargs):
        calls.append(kwargs)
        return predictor

    monkeypatch.setattr(interface.pretrained_mlip, "get_predict_unit", load)
    wrapper = interface.FairChemModel("unused", device=device)
    assert calls[0]["device"] == "cuda:1"
    assert wrapper.device == torch.device("cuda:1")


def test_model_outputs_own_storage_and_accept_state_dictionary(monkeypatch):
    predictor = HarmonicPredictor()
    wrapper = _wrapper(monkeypatch, predictor)
    result = wrapper(_state().attributes)
    expected = {name: value.clone() for name, value in result.items()}
    for value in predictor.latest.values():
        value.zero_()
    for name, value in result.items():
        torch.testing.assert_close(value, expected[name])
        assert not value.requires_grad


@pytest.mark.parametrize("bad", ["missing", "shape", "integer", "nonfinite"])
def test_missing_or_invalid_outputs_fail_at_wrapper_boundary(monkeypatch, bad):
    predictor = HarmonicPredictor()
    predict = predictor.predict

    def broken(data, **kwargs):
        output = predict(data, **kwargs)
        if bad == "missing":
            output.pop("forces")
        elif bad == "shape":
            output["forces"] = output["forces"].flatten()
        elif bad == "integer":
            output["energy"] = output["energy"].long()
        else:
            output["energy"].fill_(float("nan"))
        return output

    predictor.predict = broken
    wrapper = _wrapper(monkeypatch, predictor)
    with pytest.raises((TypeError, ValueError), match="Predictor"):
        wrapper(_state())


def test_stress_request_cannot_silently_omit_output(monkeypatch):
    wrapper = _wrapper(monkeypatch, compute_stress=True)
    with pytest.raises(ValueError, match="requested property 'stress'"):
        wrapper(_state())


def test_requested_stress_rejects_undefined_volume(monkeypatch):
    wrapper = _wrapper(monkeypatch, compute_stress=True)
    state = _state()
    state.cell.zero_()
    with pytest.raises(ValueError, match="Stress requires a nonsingular cell"):
        wrapper(state)


@pytest.mark.parametrize("compute_stress", [False, True])
def test_actual_uma_passes_torchsim_model_interface_validator(
    monkeypatch, compute_stress
):
    from torch_sim.models.interface import validate_model_outputs

    from tests.core.calculate.test_ase_interop_jobs import _real_calculator
    from tests.core.calculate.test_torchsim_interface import _prepare_validation_state

    predictor = _real_calculator().predictor
    wrapper = _wrapper(monkeypatch, predictor, compute_stress=compute_stress)
    validate_model_outputs(
        wrapper,
        torch.device("cpu"),
        torch.float32,
        state_modifier=_prepare_validation_state,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_wrapper_rejects_unsupported_simulation_dtype(monkeypatch, dtype):
    with pytest.raises(ValueError, match="float32 or float64"):
        _wrapper(monkeypatch, dtype=dtype)


@pytest.mark.parametrize("field", ["atomic_numbers", "charge", "spin"])
def test_transfer_validates_metadata_before_torchsim_dtype_coercion(monkeypatch, field):
    state = _state()
    if field == "atomic_numbers":
        state.atomic_numbers = state.atomic_numbers.double() + 0.5
    else:
        state.system_extras[field] = torch.tensor([1 + 2**-30], dtype=torch.float64)
    wrapper = _wrapper(monkeypatch)
    # A CPU ordinal exercises the transfer branch without requiring a GPU.
    wrapper._device = torch.device("cpu:0")
    with pytest.raises(ValueError, match=field):
        wrapper(state)


def test_transfer_preserves_high_precision_cell(monkeypatch):
    state = _state()
    state.cell = state.cell.double()
    state.cell[0, 0, 0] += 2**-30
    predictor = HarmonicPredictor()
    predict = predictor.predict

    def capture(data, **kwargs):
        torch.testing.assert_close(data.cell, state.row_vector_cell, atol=0, rtol=0)
        return predict(data, **kwargs)

    predictor.predict = capture
    wrapper = _wrapper(monkeypatch, predictor, dtype=torch.float64)
    wrapper._device = torch.device("cpu:0")
    wrapper(state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_torchsim_fire_and_velocity_verlet_jobs(monkeypatch, dtype):
    from torch_sim.integrators.nve import nve_init, nve_step
    from torch_sim.optimizers.fire import fire_init, fire_step

    wrapper = _wrapper(monkeypatch, dtype=dtype)
    state = _state(dtype)
    initial = wrapper(state)["energy"]
    optimized = fire_init(state, wrapper)
    for _ in range(4):
        optimized = fire_step(optimized, wrapper)
    assert torch.all(optimized.energy < initial)
    assert optimized.forces.dtype == dtype
    state = _state(dtype)
    state.rng = 19
    md = nve_init(state, wrapper, kT=0.01)
    for _ in range(4):
        md = nve_step(md, wrapper, dt=0.01)
    expected = wrapper(md)
    torch.testing.assert_close(md.energy, expected["energy"])
    torch.testing.assert_close(md.forces, expected["forces"])
    assert torch.isfinite(md.momenta).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("periodic", [False, True])
def test_actual_uma_torchsim_job_matches_direct_predictor(monkeypatch, dtype, periodic):
    from torch_sim.optimizers.cell_filters import CellFilter
    from torch_sim.optimizers.fire import fire_init, fire_step

    from tests.core.units.mlip_unit.test_predict_memory import (
        _IdentityNormalizer,
        _task,
    )
    from tests.core.units.mlip_unit.test_xpu_preparation import _small_predictor

    predictor = _small_predictor()
    predictor.inference_settings.merge_mole = False
    tasks = []
    properties = ("energy", "forces", "stress") if periodic else ("energy", "forces")
    for prop in properties:
        task = _task(f"omol_{prop}", "atom" if prop == "forces" else "system", prop)
        task.normalizer = _IdentityNormalizer()
        task.element_references = None
        tasks.append(task)
    predictor.model.module._tasks = {task.name: task for task in tasks}
    predictor.model.module._dataset_to_tasks = {"omol": tasks}
    reference = copy.deepcopy(predictor)
    model = _wrapper(monkeypatch, predictor, dtype=dtype, compute_stress=periodic)
    state = _state(dtype, periodic)
    output = model(state)
    assert torch.count_nonzero(output["forces"]) > 0
    optimized = fire_init(
        state, model, cell_filter=CellFilter.frechet if periodic else None
    )
    for _ in range(2):
        optimized = fire_step(optimized, model)
    atoms = optimized.to_atoms()[0]
    atoms.info.update(charge=0, spin=1)
    data = AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["charge", "spin"])
    expected = reference.predict(data, requested_properties=properties)
    actual = model(optimized)
    for name in expected:
        target = expected[name].to(dtype)
        if name == "stress":
            target = target.reshape(1, 3, 3)
        torch.testing.assert_close(actual[name], target, atol=2e-5, rtol=2e-4)


def test_cached_geometry_rejects_nonfinite_update():
    state = _state()
    converter = interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    state.positions[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        converter(state, "omol", torch.float32)


@pytest.mark.parametrize("cached", [False, True])
def test_complex_cell_is_rejected_before_discarding_imaginary_values(cached):
    state = _state()
    converter = interface._SimStateConverter()
    if cached:
        converter(state, "omol", torch.float32)
    state.cell = state.cell.to(torch.complex64)
    state.cell[0, 0, 0] += 1j
    with pytest.raises(ValueError, match="cell must use float32 or float64"):
        converter(state, "omol", torch.float32)


@pytest.mark.skipif(not torch.xpu.is_available(), reason="Requires a real Intel XPU")
def test_metadata_cache_waits_for_producer_and_owns_cross_stream_storage():
    state = _state().to("xpu:0")
    default = torch.xpu.current_stream("xpu:0")
    first, second = torch.xpu.Stream(device="xpu:0"), torch.xpu.Stream(device="xpu:0")
    first.wait_stream(default)
    second.wait_stream(default)
    converter = interface._SimStateConverter()
    with torch.xpu.stream(first):
        initial = converter(state, "omol", torch.float32)
    with torch.xpu.stream(second):
        current = converter(state, "omol", torch.float32)
        torch.testing.assert_close(current.atomic_numbers, state.atomic_numbers.long())
        current.atomic_numbers.zero_()
        converter.clear()
    second.synchronize()
    torch.testing.assert_close(
        initial.atomic_numbers.cpu(), _state().atomic_numbers.long()
    )

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from fairchem.core.calculate import torchsim_interface

ts = pytest.importorskip("torch_sim")


def _state(periodic=False):
    return ts.SimState(
        positions=torch.tensor([[0.1, 0.3, 0.2], [1.0, 0.2, 0.0], [2.0, 0.0, 0.3]]),
        masses=torch.tensor([16.0, 1.0, 1.0]),
        atomic_numbers=torch.tensor([8, 1, 1]),
        system_idx=torch.zeros(3, dtype=torch.long),
        cell=torch.tensor([[[4.0, 0.0, 0.0], [0.2, 3.0, 0.0], [0.1, 0.4, 5.0]]]),
        pbc=torch.tensor([periodic, periodic, periodic]),
        _system_extras={"charge": torch.tensor([0]), "spin": torch.tensor([1])},
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_cached_metadata_refreshes_geometry_without_revalidation(periodic, monkeypatch):
    state = _state(periodic)
    converter = torchsim_interface._SimStateConverter()
    original = torchsim_interface._simstate_to_atomicdata_batch
    first = converter(state, "omol", torch.float32)
    first_position = first.pos.clone()
    state.positions = state.positions + 0.4
    state.cell = state.cell * 1.1
    expected = original(state, "omol", torch.float32)

    def forbidden(*args, **kwargs):
        raise AssertionError("Static metadata was unnecessarily reconstructed")

    with monkeypatch.context() as conversion_patch:
        conversion_patch.setattr(
            torchsim_interface, "_simstate_to_atomicdata_batch", forbidden
        )
        conversion_patch.setattr(torch, "bincount", forbidden)
        second = converter(state, "omol", torch.float32)
    torch.testing.assert_close(second.pos, expected.pos)
    torch.testing.assert_close(second.cell, expected.cell)
    torch.testing.assert_close(first.pos, first_position)
    assert second.get_batch_stats() == expected.get_batch_stats()
    second.validate()


@pytest.mark.parametrize(
    "field", ["atomic_numbers", "system_idx", "pbc", "charge", "spin"]
)
def test_metadata_inplace_mutation_invalidates_template(field, monkeypatch):
    state = _state()
    converter = torchsim_interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    original = torchsim_interface._simstate_to_atomicdata_batch
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(torchsim_interface, "_simstate_to_atomicdata_batch", counted)
    if field == "atomic_numbers":
        state.atomic_numbers[0] = 7
    elif field == "pbc":
        state.pbc[0] = True
    elif field == "system_idx":
        # Same values still increment the version; a same-storage edit must be checked.
        state.system_idx.add_(0)
    else:
        getattr(state, field).add_(1)
    actual = converter(state, "omol", torch.float32)
    expected = original(state, "omol", torch.float32)
    assert len(calls) == 1
    for key in ("pos", "cell", "atomic_numbers", "pbc", "charge", "spin"):
        torch.testing.assert_close(actual[key], expected[key])


def test_metadata_cache_does_not_retain_old_geometry_graph():
    state = _state()
    state.positions = state.positions.clone().requires_grad_()
    state.cell = state.cell.clone().requires_grad_()
    position_ref = weakref.ref(state.positions)
    cell_ref = weakref.ref(state.cell)
    converter = torchsim_interface._SimStateConverter()
    converter(state, "omol", torch.float32)
    state.positions = state.positions.detach().clone()
    state.cell = state.cell.detach().clone()
    gc.collect()
    assert position_ref() is None
    assert cell_ref() is None


def test_cached_conversion_keeps_position_and_cell_derivatives():
    state = _state(periodic=True)
    converter = torchsim_interface._SimStateConverter()
    converter(state, "omol", torch.float64)
    state.positions = state.positions.double().requires_grad_()
    state.cell = state.cell.double().requires_grad_()
    cached = converter(state, "omol", torch.float64)
    reference = torchsim_interface._simstate_to_atomicdata_batch(
        state, "omol", torch.float64
    )
    inputs = (state.positions, state.cell)
    outputs = []
    for data in (cached, reference):
        energy = data.pos.square().sum() + data.cell.square().sum()
        gradients = torch.autograd.grad(energy, inputs, create_graph=True)
        second = torch.autograd.grad(sum(g.square().sum() for g in gradients), inputs)
        outputs.append((gradients, second))
    for actual, expected in zip(outputs[0], outputs[1]):
        for left, right in zip(actual, expected):
            torch.testing.assert_close(left, right)


def test_wrapper_requests_only_implemented_properties(monkeypatch):
    class Predictor:
        device = "cpu"
        supports_requested_properties = True

        def predict(self, data, *, requested_properties):
            assert requested_properties == ("energy", "forces")
            return {"energy": data.pos.square().sum().view(1), "forces": -2 * data.pos}

    monkeypatch.setattr(torchsim_interface, "setup_imports", lambda: None)
    monkeypatch.setattr(torchsim_interface, "setup_logging", lambda: None)
    monkeypatch.setattr(
        torchsim_interface.pretrained_mlip,
        "get_predict_unit",
        lambda *args, **kw: Predictor(),
    )
    model = torchsim_interface.FairChemModel("unused", device="cpu")
    state = _state()
    first = model(state)
    state.positions = state.positions + 0.25
    second = model(state)
    torch.testing.assert_close(second["forces"], -2 * state.positions)
    assert not torch.equal(first["forces"], second["forces"])

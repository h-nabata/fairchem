"""Completed inference temporaries must not overlap the next calculation."""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.calculate.batched_inference import run_resident_steps
from fairchem.core.calculate.series_b_curvature import _orthogonalize
from fairchem.core.models.uma.outputs import compute_hessian


@pytest.mark.parametrize("retain", [False, True])
def test_resident_steps_release_predictions_unless_observer_retains_them(retain):
    seen, saved = [], []

    class Model:
        device = torch.device("cpu")

        def __call__(self, state):
            assert all((ref() is not None) == retain for ref in seen)
            forces = torch.ones(8192)
            seen.append(weakref.ref(forces))
            return {"forces": forces}

    def observer(step, state, prediction):
        if retain:
            saved.append(prediction)

    state = SimpleNamespace(device=torch.device("cpu"))
    assert (
        run_resident_steps(
            Model(), state, lambda state, prediction: state, 4, observer=observer
        )
        is state
    )
    assert all((ref() is not None) == retain for ref in seen)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hessian_releases_completed_blocks_before_next_backward(monkeypatch, dtype):
    refs = []
    native = torch.vmap

    def tracked(function, *args, **kwargs):
        mapped = native(function, *args, **kwargs)

        def run(basis):
            assert all(ref() is None for ref in refs)
            block = mapped(basis)
            refs.extend((weakref.ref(basis), weakref.ref(block)))
            return block

        return run

    monkeypatch.setattr(torch, "vmap", tracked)
    pos = torch.linspace(0.1, 1.8, 18, dtype=dtype).reshape(6, 3).requires_grad_()
    force = -4 * pos.pow(3)
    actual = compute_hessian(force, pos, chunk_size=4)
    torch.testing.assert_close(actual[0], torch.diag(12 * pos.flatten().square()))
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reorthogonalization_preserves_basis_and_operator_products(dtype):
    generator = torch.Generator().manual_seed(176)
    matrix = torch.randn(64, 12, dtype=dtype, generator=generator)
    basis = tuple(torch.linalg.qr(matrix).Q.unbind(1))
    value = torch.randn(64, dtype=dtype, generator=generator)
    original = value.clone()
    saved = [q.clone() for q in basis]
    reference = value
    for _ in range(2):
        for q in basis:
            reference = reference - torch.dot(q, reference) * q
    actual = _orthogonalize(value, basis)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(value, original, rtol=0, atol=0)
    for q, old in zip(basis, saved):
        torch.testing.assert_close(q, old, rtol=0, atol=0)

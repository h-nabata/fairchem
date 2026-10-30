"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from fairchem.core.models.uma import outputs as output_module
from fairchem.core.models.uma.outputs import (
    compute_energy,
    compute_forces,
    compute_forces_and_stress,
    compute_hessian,
    compute_hessian_vector_product,
    resolve_requested_properties,
)


def _energy(pos, cell, weight):
    transformed = pos @ cell
    return (weight * transformed.sin().square()).sum() + pos.sum().pow(4) / 7


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 64, None])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chunked_hessian_matches_independent_energy_derivative(chunk_size, dtype):
    # Noncontiguous positions exercise view/copy handling in the derivative API.
    pos = torch.linspace(-0.3, 0.4, 6, dtype=dtype).view(3, 2).T.requires_grad_()
    assert not pos.is_contiguous()
    cell = torch.tensor(
        [[1.2, 0.2, 0.1], [0.1, 0.9, 0.2], [0.3, 0.1, 1.1]], dtype=dtype
    ).requires_grad_()
    weight = torch.tensor(0.7, dtype=dtype, requires_grad=True)
    expected = torch.autograd.functional.hessian(
        lambda p: _energy(p, cell, weight), pos, create_graph=True
    ).reshape(6, 6)
    forces = compute_forces(_energy(pos, cell, weight), pos, training=True)
    result = compute_hessian(forces, pos, training=True, chunk_size=chunk_size)[0]
    torch.testing.assert_close(result, expected)
    # Mixed cell/position/weight derivatives must survive every chunk boundary.
    actual_gradients = torch.autograd.grad(result.square().sum(), (pos, cell, weight))
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (pos, cell, weight)
    )
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference)


class _BasisAllocationRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.zero_shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.eye.default, torch.ops.aten.eye.m):
            raise AssertionError("Dense identity allocated during chunked Hessian")
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.new_zeros.default:
            self.zero_shapes.append(tuple(result.shape))
        return result


def test_chunked_inference_bounds_basis_allocation():
    pos = torch.linspace(-0.2, 0.4, 30, dtype=torch.float64).view(10, 3)
    pos.requires_grad_()
    forces = compute_forces(pos.sin().sum(), pos, training=True)
    recorder = _BasisAllocationRecorder()
    with recorder:
        result = compute_hessian(forces, pos, chunk_size=4)
    assert result.shape == (1, 30, 30)
    assert recorder.zero_shapes
    assert all(shape[0] <= 4 for shape in recorder.zero_shapes)
    assert not result.requires_grad


@pytest.mark.parametrize("vmap", [False, True])
@pytest.mark.parametrize("kind", ["constant", "linear", "quadratic"])
def test_zero_or_constant_curvature_has_correct_higher_derivative(vmap, kind):
    pos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    if kind == "constant":
        energy = torch.tensor(2.0, dtype=pos.dtype)
        expected = pos.new_zeros((3, 3))
    elif kind == "linear":
        energy = 2 * pos.sum()
        expected = pos.new_zeros((3, 3))
    else:
        energy = pos.square().sum()
        expected = 2 * torch.eye(3, dtype=pos.dtype)
    forces = compute_forces(energy, pos, training=True)
    hessian = compute_hessian(forces, pos, vmap=vmap, training=True, chunk_size=1)[0]
    torch.testing.assert_close(hessian, expected)
    torch.testing.assert_close(
        torch.autograd.grad(hessian.sum(), pos)[0], torch.zeros_like(pos)
    )


@pytest.mark.parametrize("flat", [False, True])
def test_hvp_matches_dense_hessian_in_coupled_system(flat):
    pos = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64).view(2, 3).requires_grad_()
    cell = torch.eye(3, dtype=pos.dtype).requires_grad_()
    weight = torch.tensor(0.7, dtype=pos.dtype, requires_grad=True)
    vector = torch.tensor([[0.5, -1, 0.2], [0.7, 0.1, -0.4]], dtype=pos.dtype)
    if flat:
        vector = vector.flatten()
    expected_hessian = torch.autograd.functional.hessian(
        lambda p: _energy(p, cell, weight), pos
    ).reshape(6, 6)
    forces = compute_forces(_energy(pos, cell, weight), pos, training=True)
    actual = compute_hessian_vector_product(forces, pos, vector, retain_graph=True)
    torch.testing.assert_close(
        actual, (expected_hessian @ vector.flatten()).reshape(vector.shape)
    )
    # Explicit retention permits a subsequent full Hessian on the same graph.
    actual_hessian = compute_hessian(forces, pos, chunk_size=2)[0]
    torch.testing.assert_close(actual_hessian, expected_hessian)


def test_hvp_position_cell_weight_and_vector_gradgradcheck():
    pos = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float64, requires_grad=True)
    cell = (torch.eye(3, dtype=pos.dtype) * 1.1).requires_grad_()
    weight = torch.tensor(0.8, dtype=pos.dtype, requires_grad=True)
    vector = torch.tensor([[0.3, -0.2, 0.4]], dtype=pos.dtype, requires_grad=True)

    def product(p, c, w, v):
        forces = compute_forces(_energy(p, c, w), p, training=True)
        return compute_hessian_vector_product(forces, p, v, training=True)

    assert torch.autograd.gradcheck(product, (pos, cell, weight, vector))
    assert torch.autograd.gradgradcheck(product, (pos, cell, weight, vector))


def test_hvp_keeps_independent_system_blocks_separate():
    pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    energies = torch.stack((pos[0].pow(4).sum(), 2 * pos[1].pow(4).sum()))
    forces = compute_forces(energies, pos, training=True)
    vector = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    product = compute_hessian_vector_product(forces, pos, vector)
    expected = torch.stack((12 * pos[0].square() * vector[0], torch.zeros(3)))
    torch.testing.assert_close(product, expected)


def test_unused_position_and_cell_produce_zero_force_stress_and_curvature():
    pos = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    cell = torch.eye(3, dtype=pos.dtype).unsqueeze(0).requires_grad_()
    forces, stress = compute_forces_and_stress(
        torch.tensor([2.0], dtype=pos.dtype),
        pos,
        cell,
        torch.zeros(2, dtype=torch.long),
        training=True,
    )
    torch.testing.assert_close(forces, torch.zeros_like(pos))
    torch.testing.assert_close(stress, torch.zeros(1, 9, dtype=pos.dtype))
    product = compute_hessian_vector_product(forces, pos, torch.ones_like(pos))
    torch.testing.assert_close(product, torch.zeros_like(pos))


def test_mean_energy_derivatives_use_global_per_system_atom_counts():
    pos = torch.linspace(0.1, 0.9, 9, dtype=torch.float64).reshape(3, 3)
    pos.requires_grad_()
    cell = torch.eye(3, dtype=pos.dtype).repeat(2, 1, 1).requires_grad_()
    batch = torch.tensor([0, 0, 1])

    def node_energy(p, c):
        transformed = torch.bmm(p.unsqueeze(1), c[batch]).squeeze(1)
        return transformed.pow(4).sum(-1)

    def reference(p, c):
        values = node_energy(p, c)
        return values[:2].mean() + values[2]

    nodes = node_energy(pos, cell)
    energy, energy_part = compute_energy(
        {"node_embedding": nodes.reshape(3, 1, 1)},
        torch.nn.Identity(),
        batch,
        num_systems=2,
        natoms=torch.tensor([2, 1]),
        reduce="mean",
    )
    torch.testing.assert_close(energy_part, energy)
    force, stress = compute_forces_and_stress(
        energy_part, pos, cell, batch, training=True
    )
    expected_force = -torch.autograd.grad(reference(pos, cell), pos)[0]
    torch.testing.assert_close(force, expected_force)
    strain = torch.zeros(2, 3, 3, dtype=pos.dtype, requires_grad=True)
    symmetric_strain = (strain + strain.mT) / 2
    strained_pos = pos + torch.bmm(pos.unsqueeze(1), symmetric_strain[batch]).squeeze(1)
    strained_cell = cell + cell @ symmetric_strain
    virial = torch.autograd.grad(reference(strained_pos, strained_cell), strain)[0]
    torch.testing.assert_close(
        stress.reshape(2, 3, 3), virial / cell.det()[:, None, None]
    )
    vector = torch.linspace(-0.3, 0.5, 9, dtype=pos.dtype).reshape_as(pos)
    expected_hessian = torch.autograd.functional.hessian(
        lambda p: reference(p, cell), pos
    ).reshape(9, 9)
    product = compute_hessian_vector_product(force, pos, vector)
    torch.testing.assert_close(product.flatten(), expected_hessian @ vector.flatten())


def test_mean_energy_gp_uses_global_counts_for_unreduced_energy(monkeypatch):
    monkeypatch.setattr(output_module.gp_utils, "initialized", lambda: True)
    monkeypatch.setattr(
        output_module.gp_utils,
        "reduce_from_model_parallel_region",
        lambda value: 2 * value,
    )
    pos = torch.tensor([[0.1, 0.2, 0.3], [0.3, 0.4, 0.5]], requires_grad=True)
    nodes = pos.square().sum(-1)
    # Two local atoms represent half a four-atom system in this reduction test.
    energy, local_energy = compute_energy(
        {"node_embedding": nodes.reshape(2, 1, 1)},
        torch.nn.Identity(),
        torch.zeros(2, dtype=torch.long),
        num_systems=1,
        natoms=torch.tensor([4]),
        reduce="mean",
    )
    torch.testing.assert_close(local_energy, nodes.double().sum().reshape(1) / 4)
    torch.testing.assert_close(energy, 2 * local_energy)
    force = compute_forces(local_energy, pos)
    torch.testing.assert_close(force, -pos)


@pytest.mark.parametrize("vmap", [False, True])
def test_parameter_dependent_constant_force_has_zero_curvature(vmap):
    pos = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    weight = torch.randn_like(pos, requires_grad=True)
    forces = compute_forces((weight * pos).sum(), pos, training=True)
    hessian = compute_hessian(forces, pos, vmap=vmap, training=True, chunk_size=2)
    torch.testing.assert_close(hessian, pos.new_zeros((1, 3, 3)))
    for result in torch.autograd.grad(hessian.sum(), (pos, weight)):
        torch.testing.assert_close(result, torch.zeros_like(result))


@pytest.mark.parametrize("vmap", [False, True])
def test_empty_curvature(vmap):
    pos = torch.empty((0, 3), dtype=torch.float64, requires_grad=True)
    forces = compute_forces(pos.sum(), pos, training=True)
    result = compute_hessian(forces, pos, vmap=vmap, training=True, chunk_size=2)
    assert result.shape == (1, 0, 0)
    torch.testing.assert_close(torch.autograd.grad(result.sum(), pos)[0], pos)
    product = compute_hessian_vector_product(forces, pos, pos.new_empty(0))
    assert product.shape == (0,)


class _Holder:
    def __init__(self, value):
        self.value = value


@pytest.mark.parametrize("hvp", [False, True])
def test_curvature_releases_saved_intermediates(hvp):
    references = []

    def pack(value):
        holder = _Holder(value)
        references.append(weakref.ref(holder))
        return holder

    pos = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda holder: holder.value):
        forces = compute_forces(pos.square().sin().sum(), pos, training=True)
        if hvp:
            result = compute_hessian_vector_product(forces, pos, torch.ones_like(pos))
        else:
            result = compute_hessian(forces, pos, chunk_size=2)
    assert torch.isfinite(result).all()
    gc.collect()
    assert references
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_reject_invalid_chunk_size(chunk_size):
    pos = torch.randn(1, 3, requires_grad=True)
    forces = compute_forces(pos.square().sum(), pos, training=True)
    with pytest.raises(ValueError, match="chunk_size"):
        compute_hessian(forces, pos, chunk_size=chunk_size)


def test_reject_detached_force_graph_and_incompatible_vector():
    pos = torch.randn(1, 3, dtype=torch.float64, requires_grad=True)
    with pytest.raises(ValueError, match="differentiable forces"):
        compute_hessian_vector_product(-2 * pos.detach(), pos, torch.ones_like(pos))
    forces = compute_forces(pos.square().sum(), pos, training=True)
    with pytest.raises(ValueError, match="position shape"):
        compute_hessian_vector_product(forces, pos, torch.ones(2, dtype=pos.dtype))
    with pytest.raises(ValueError, match="dtype"):
        compute_hessian_vector_product(forces, pos, torch.ones_like(pos).float())


def test_requested_properties_preserve_defaults_and_do_not_emit_dependencies():
    config = SimpleNamespace(
        forces=True, stress=True, direct_stress=False, hessian=False
    )
    assert resolve_requested_properties(config) == {
        "energy",
        "forces",
        "stress",
        "embeddings",
    }
    assert resolve_requested_properties(config, ("hessian",)) == {"hessian"}
    assert resolve_requested_properties(config, ("energy",)) == {"energy"}
    with pytest.raises(ValueError, match="Unknown"):
        resolve_requested_properties(config, ("energies",))
    with pytest.raises(TypeError, match="iterable"):
        resolve_requested_properties(config, "energy")

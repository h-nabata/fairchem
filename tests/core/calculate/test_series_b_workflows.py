"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import weakref
from dataclasses import FrozenInstanceError
from typing import ClassVar

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from fairchem.core.calculate.series_b_workflows import (
    CoarseRefineOptions,
    DyNEBOptions,
    RESPAOptions,
    run_coarse_refine,
    run_dyneb,
    run_respa,
)


class Harmonic(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def __init__(self, stiffness=1.0, center=0.0, fail_at=None):
        super().__init__()
        self.stiffness = stiffness
        self.center = np.array([center, 0.0, 0.0])
        self.calls = 0
        self.fail_at = fail_at

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.calls += 1
        displacement = atoms.positions - self.center
        forces = -self.stiffness * displacement
        if self.calls == self.fail_at:
            forces[:] = np.nan
        self.results = {
            "energy": 0.5 * self.stiffness * np.square(displacement).sum(),
            "forces": forces,
        }


def _atom():
    atom = Atoms("H", positions=[[1.0, 0.1, 0.0]], masses=[1.0], cell=[8, 8, 8])
    atom.set_momenta([[0.0, 0.1, 0.2]])
    atom.info["metadata"] = {"array": np.array([1, 2, 3])}
    return atom


def _factory(created, cls=Harmonic, **kwargs):
    def make():
        calculator = cls(**kwargs)
        created.append(calculator)
        return calculator

    return make


def test_respa_inner_step_one_matches_teacher_velocity_verlet():
    initial = _atom()
    before = initial.copy()
    reference = initial.copy()
    reference.calc = Harmonic(stiffness=1.2)
    VelocityVerlet(reference, timestep=0.02, logfile=None).run(100)
    teachers, students = [], []
    result = run_respa(
        initial,
        teacher_factory=_factory(teachers, stiffness=1.2),
        student_factory=_factory(students, stiffness=0.5),
        options=RESPAOptions(inner_timestep=0.02, inner_steps=1),
        outer_steps=100,
    )
    np.testing.assert_allclose(result.atoms.positions, reference.positions, atol=1e-13)
    np.testing.assert_allclose(
        result.atoms.get_momenta(), reference.get_momenta(), atol=1e-13
    )
    np.testing.assert_array_equal(initial.positions, before.positions)
    np.testing.assert_array_equal(initial.get_momenta(), before.get_momenta())
    result.atoms.info["metadata"]["array"][0] = 9
    assert initial.info["metadata"]["array"][0] == 1
    teachers[0].results["forces"][:] = 100
    assert np.max(np.abs(result.reference_forces)) < 2


def test_respa_reduces_teacher_calls_with_bounded_analytic_nve_drift():
    initial = _atom()
    reference = initial.copy()
    reference.calc = Harmonic()
    VelocityVerlet(reference, timestep=0.02, logfile=None).run(500)
    teachers, students = [], []
    result = run_respa(
        initial,
        teacher_factory=_factory(teachers),
        student_factory=_factory(students, stiffness=0.8),
        options=RESPAOptions(inner_timestep=0.02, inner_steps=5),
        outer_steps=100,
    )
    assert result.teacher_force_evaluations == teachers[0].calls == 101
    assert result.student_force_evaluations == students[0].calls == 501
    assert reference.calc.calls == 501
    assert result.max_energy_drift < 5e-4
    np.testing.assert_allclose(result.atoms.positions, reference.positions, atol=2e-3)
    np.testing.assert_allclose(
        result.atoms.get_momenta(), reference.get_momenta(), atol=2e-3
    )
    assert len(result.total_energies) == 101


@pytest.mark.parametrize("fail", [False, True])
def test_respa_inplace_is_transactional_and_preserves_existing_calculator(fail):
    initial = _atom()
    initial.calc = Harmonic(stiffness=2.0)
    initial.get_forces()
    original = initial.copy()
    results = copy.deepcopy(initial.calc.results)
    options = dict(
        teacher_factory=lambda: Harmonic(fail_at=2 if fail else None),
        student_factory=lambda: Harmonic(stiffness=0.8),
        options=RESPAOptions(0.02, 3),
        outer_steps=2,
        inplace=True,
    )
    if fail:
        with pytest.raises(ValueError, match="finite real"):
            run_respa(initial, **options)
        np.testing.assert_array_equal(initial.positions, original.positions)
        np.testing.assert_array_equal(initial.get_momenta(), original.get_momenta())
    else:
        result = run_respa(initial, **options)
        assert result.atoms is initial
        assert not np.array_equal(initial.positions, original.positions)
    assert initial.calc.calls == 1
    np.testing.assert_array_equal(initial.calc.results["forces"], results["forces"])


def test_respa_rejects_constraints_and_nonfinite_or_zero_masses():
    for invalid in ("constraints", "mass", "nan"):
        atoms = _atom()
        if invalid == "constraints":
            atoms.set_constraint(FixAtoms(indices=[0]))
        else:
            atoms.set_masses([0 if invalid == "mass" else np.nan])
        with pytest.raises(ValueError):
            run_respa(
                atoms,
                teacher_factory=Harmonic,
                student_factory=Harmonic,
                options=RESPAOptions(0.02),
                outer_steps=1,
            )


def test_workflows_reject_factory_reuse_of_caller_or_between_stages():
    initial = _atom()
    initial.calc = Harmonic()
    for teacher, student in (
        (lambda: initial.calc, Harmonic),
        (Harmonic, lambda: initial.calc),
    ):
        with pytest.raises(ValueError, match="privately owned"):
            run_respa(
                initial,
                teacher_factory=teacher,
                student_factory=student,
                options=RESPAOptions(0.02),
                outer_steps=1,
            )
    calculator = Harmonic()
    with pytest.raises(ValueError, match="privately owned"):
        run_coarse_refine(
            initial,
            coarse_factory=lambda: calculator,
            reference_factory=lambda: calculator,
            options=CoarseRefineOptions(),
        )


def test_coarse_refinement_reports_reference_residual_and_reduces_reference_calls():
    atoms = _atom()
    atoms.positions = [[2, 0, 0]]
    before = atoms.copy()
    baseline = atoms.copy()
    baseline.calc = Harmonic()
    BFGS(baseline, logfile=None).run(fmax=1e-4, steps=100)
    coarse, reference = [], []
    result = run_coarse_refine(
        atoms,
        coarse_factory=_factory(coarse, center=-0.3),
        reference_factory=_factory(reference),
        options=CoarseRefineOptions(coarse_fmax=0.01, reference_fmax=1e-4),
    )
    assert result.coarse_converged
    assert result.converged
    assert result.reference_force_max < 1e-4
    assert reference[0].calls < baseline.calc.calls
    np.testing.assert_allclose(result.atoms.positions, 0, atol=1e-4)
    np.testing.assert_allclose(result.reference_forces, -result.atoms.positions)
    assert result.reference_energy == pytest.approx(
        0.5 * np.square(result.atoms.positions).sum()
    )
    np.testing.assert_array_equal(atoms.positions, before.positions)
    assert atoms.calc is None


def test_reference_recheck_does_not_reuse_coarse_convergence_or_results():
    atoms = _atom()
    result = run_coarse_refine(
        atoms,
        coarse_factory=lambda: Harmonic(center=0.7),
        reference_factory=Harmonic,
        options=CoarseRefineOptions(
            coarse_fmax=0.01, reference_fmax=1e-4, reference_steps=0
        ),
    )
    assert result.coarse_converged
    assert not result.converged
    assert result.reference_force_max > 0.6
    assert result.reference_energy > result.coarse_energy


def test_reference_residual_respects_constraints_but_reports_raw_forces():
    atoms = _atom()
    atoms.set_constraint(FixAtoms(indices=[0]))
    result = run_coarse_refine(
        atoms,
        coarse_factory=Harmonic,
        reference_factory=Harmonic,
        options=CoarseRefineOptions(),
    )
    assert result.converged
    assert result.reference_force_max == 0
    assert np.linalg.norm(result.reference_forces) > 1
    np.testing.assert_array_equal(result.atoms.positions, atoms.positions)


def test_coarse_model_is_released_before_loading_reference_model():
    previous = []

    def coarse_factory():
        calculator = Harmonic()
        previous.append(weakref.ref(calculator))
        return calculator

    def reference_factory():
        assert previous[0]() is None
        return Harmonic()

    run_coarse_refine(
        _atom(),
        coarse_factory=coarse_factory,
        reference_factory=reference_factory,
        options=CoarseRefineOptions(coarse_steps=0, reference_steps=0),
    )


class DoubleWell(Calculator):
    implemented_properties: ClassVar[list[str]] = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x, y, z = atoms.positions.T
        self.results = {
            "energy": ((x**2 - 1) ** 2 + 0.5 * (y**2 + z**2)).sum(),
            "forces": np.array([-4 * x * (x**2 - 1), -y, -z]).T,
        }


def _band():
    return [
        Atoms("H", positions=[[x, y, 0]], cell=[8, 8, 8])
        for x, y in zip(np.linspace(-1, 1, 5), [0, 0.2, 0, 0.2, 0], strict=True)
    ]


@pytest.mark.parametrize("refine", [False, True])
def test_scaled_dyneb_always_checks_unmasked_reference_forces(refine):
    images = _band()
    original_positions = [image.positions.copy() for image in images]
    created = []
    references = []
    result = run_dyneb(
        images,
        calculator_factory=_factory(created, cls=DoubleWell),
        reference_calculator_factory=_factory(references, cls=DoubleWell),
        options=DyNEBOptions(
            fmax=0.05,
            scale_fmax=100,
            max_steps=0,
            final_refine=refine,
            reference_fmax=0.001,
            reference_steps=300,
        ),
    )
    assert result.dynamic_converged
    assert result.refined is refine
    assert result.converged is refine
    assert len(created) == len(references) == len(images)
    assert len({id(image.calc) for image in result.images}) == len(images)
    if refine:
        assert result.reference_force_max < 0.001
        assert max(result.reference_energies) == pytest.approx(1, abs=1e-5)
    else:
        assert result.reference_force_max > 0.1
    np.testing.assert_array_equal(result.images[0].positions, original_positions[0])
    np.testing.assert_array_equal(result.images[-1].positions, original_positions[-1])
    for image, before in zip(images, original_positions, strict=True):
        np.testing.assert_array_equal(image.positions, before)
        assert image.calc is None


def test_dyneb_rejects_shared_calculators_and_too_few_images():
    calculator = DoubleWell()
    with pytest.raises(ValueError, match="privately owned"):
        run_dyneb(
            _band(), calculator_factory=lambda: calculator, options=DyNEBOptions()
        )
    with pytest.raises(ValueError, match="three images"):
        run_dyneb(_band()[:2], calculator_factory=DoubleWell, options=DyNEBOptions())


def test_dyneb_validates_incompatible_images_before_loading_models():
    images = _band()
    images[1].numbers[0] = 2
    created = []
    with pytest.raises(ValueError):
        run_dyneb(
            images,
            calculator_factory=_factory(created, cls=DoubleWell),
            options=DyNEBOptions(),
        )
    assert not created


def test_dyneb_releases_dynamic_models_before_reference_replacement():
    previous = []
    live_counts = []

    def dynamic_factory():
        calculator = DoubleWell()
        previous.append(weakref.ref(calculator))
        return calculator

    def reference_factory():
        live_counts.append(sum(reference() is not None for reference in previous))
        return DoubleWell()

    run_dyneb(
        _band(),
        calculator_factory=dynamic_factory,
        reference_calculator_factory=reference_factory,
        options=DyNEBOptions(max_steps=0, final_refine=False),
    )
    assert live_counts == [0, 0, 0, 0, 0]


@pytest.mark.parametrize(
    "factory,options",
    [
        (RESPAOptions, {"inner_timestep": np.nan}),
        (RESPAOptions, {"inner_timestep": -1}),
        (RESPAOptions, {"inner_timestep": 0.1, "inner_steps": True}),
        (RESPAOptions, {"inner_timestep": 0.1, "inner_steps": 1.5}),
        (CoarseRefineOptions, {"reference_fmax": 0.2}),
        (CoarseRefineOptions, {"coarse_steps": -1}),
        (DyNEBOptions, {"parallel": True}),
        (DyNEBOptions, {"reference_fmax": 0.2}),
        (DyNEBOptions, {"scale_fmax": np.inf}),
    ],
)
def test_workflow_options_validate_unsafe_or_lossy_configuration(factory, options):
    with pytest.raises(ValueError):
        factory(**options)


def test_workflow_options_are_frozen_and_normalize_numpy_scalars():
    options = RESPAOptions(np.float64(0.02), np.int64(3))
    assert type(options.inner_steps) is int
    assert type(options.inner_timestep) is float
    with pytest.raises(FrozenInstanceError):
        options.inner_steps = 4

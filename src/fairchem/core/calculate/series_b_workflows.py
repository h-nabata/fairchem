"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Explicit, optional workflows that can change trajectories or convergence paths.
None is enabled by constructing an ordinary FAIRChem calculator. Calculator
factories must return fresh, privately owned ASE calculators; their settings
are never edited. GPU models need not be deep-copied by these helpers.
"""

from __future__ import annotations

import copy
import gc
import weakref
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.mep import NEB, DyNEB
from ase.optimize import BFGS, FIRE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def _positive(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        or (not zero and value == 0)
    ):
        raise ValueError(
            f"{name} must be a finite {'nonnegative' if zero else 'positive'} number"
        )
    return float(value)


def _count(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or value < int(not zero)
    ):
        raise ValueError(
            f"{name} must be a {'nonnegative' if zero else 'positive'} integer"
        )
    return int(value)


@dataclass(frozen=True)
class RESPAOptions:
    """Two-level NVE integration; ``inner_timestep`` uses ASE time units.

    For example, use ``0.2 * ase.units.fs`` for a 0.2 fs inner step. A larger
    outer step can introduce resonance and energy drift; validate against a
    teacher-only small-step trajectory for the intended material and temperature.
    """

    inner_timestep: float
    inner_steps: int = 2

    def __post_init__(self):
        object.__setattr__(
            self, "inner_timestep", _positive(self.inner_timestep, "inner_timestep")
        )
        object.__setattr__(self, "inner_steps", _count(self.inner_steps, "inner_steps"))
        _positive(self.inner_timestep * self.inner_steps, "outer_timestep")


@dataclass(frozen=True)
class CoarseRefineOptions:
    """Approximate relaxation followed by an independently configured reference.

    Final convergence is assessed using reference forces. The coarse stage can
    lead to a different local minimum than reference-only optimization.
    """

    coarse_fmax: float = 0.1
    reference_fmax: float = 0.01
    coarse_steps: int = 100
    reference_steps: int = 200

    def __post_init__(self):
        for name in ("coarse_fmax", "reference_fmax"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        for name in ("coarse_steps", "reference_steps"):
            object.__setattr__(self, name, _count(getattr(self, name), name, zero=True))
        if self.reference_fmax > self.coarse_fmax:
            raise ValueError("reference_fmax must not exceed coarse_fmax")


@dataclass(frozen=True)
class DyNEBOptions:
    """Serial scaled DyNEB, with an optional final unscaled NEB relaxation.

    Even without final refinement, the returned convergence result uses an
    unmasked NEB force check. ``reference_fmax=None`` uses ``fmax``.
    """

    fmax: float = 0.05
    scale_fmax: float = 1.0
    max_steps: int = 200
    final_refine: bool = True
    reference_fmax: float | None = None
    reference_steps: int = 200
    spring_constant: float = 0.1
    climb: bool = False
    parallel: bool = False

    def __post_init__(self):
        if self.reference_fmax is None:
            object.__setattr__(self, "reference_fmax", self.fmax)
        for name in ("fmax", "reference_fmax", "spring_constant"):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self, "scale_fmax", _positive(self.scale_fmax, "scale_fmax", zero=True)
        )
        for name in ("max_steps", "reference_steps"):
            object.__setattr__(self, name, _count(getattr(self, name), name, zero=True))
        for name in ("final_refine", "climb", "parallel"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.parallel:
            raise ValueError("Scaled DyNEB supports serial execution only")
        if self.reference_fmax > self.fmax:
            raise ValueError("reference_fmax must not exceed fmax")


@dataclass(frozen=True)
class RESPAResult:
    atoms: Atoms
    reference_energy: float
    reference_forces: np.ndarray
    total_energies: tuple[float, ...]
    max_energy_drift: float
    teacher_force_evaluations: int
    student_force_evaluations: int


@dataclass(frozen=True)
class CoarseRefineResult:
    atoms: Atoms
    coarse_energy: float
    coarse_converged: bool
    reference_energy: float
    reference_forces: np.ndarray
    reference_force_max: float
    converged: bool


@dataclass(frozen=True)
class DyNEBResult:
    images: tuple[Atoms, ...]
    dynamic_converged: bool
    reference_energies: tuple[float, ...]
    reference_forces: tuple[np.ndarray, ...]
    neb_forces: np.ndarray
    reference_force_max: float
    refined: bool
    converged: bool


def _owned_atoms(atoms):
    if not isinstance(atoms, Atoms):
        raise TypeError("Workflows require ASE Atoms with a fixed simulation cell")
    if not len(atoms):
        raise ValueError("Empty structures are not supported")
    result = atoms.copy()
    result.info = copy.deepcopy(atoms.info)
    _finite(result.positions, "positions")
    _finite(result.cell.array, "cell")
    return result


def _finite(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in "iuf" or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite real numbers")
    return np.array(result, dtype=float, copy=True)


def _forces(atoms, *, constrained=False):
    forces = _finite(atoms.get_forces(apply_constraint=constrained), "forces")
    if forces.shape != (len(atoms), 3):
        raise ValueError("Forces must have shape (number of atoms, 3)")
    return forces


def _energy(atoms):
    energy = _finite(atoms.get_potential_energy(), "energy")
    if energy.size != 1:
        raise ValueError("Energy must contain exactly one value")
    return float(energy.item())


def _new_calculator(factory, forbidden):
    if not callable(factory):
        raise TypeError("Provide a factory returning a fresh ASE calculator")
    calculator = factory()
    if not isinstance(calculator, BaseCalculator):
        raise TypeError("Calculator factories must return an ASE calculator")
    if any(calculator is other for other in forbidden):
        raise ValueError(
            "Calculator factories must return distinct, privately owned calculators"
        )
    return calculator


def _force_max(forces):
    return float(np.linalg.norm(forces, axis=-1).max())


def run_respa(
    atoms: Atoms,
    *,
    teacher_factory: Callable[[], BaseCalculator],
    student_factory: Callable[[], BaseCalculator],
    options: RESPAOptions,
    outer_steps: int,
    inplace: bool = False,
) -> RESPAResult:
    """Integrate the teacher potential using a cheap student force decomposition.

    Slow forces are ``teacher - student`` at each outer boundary. Each outer
    step applies slow half-kicks around ``inner_steps`` student velocity-Verlet
    steps. Both calculators must provide conservative forces for the same
    atoms, units and energy convention. No thermostat, barostat, cell evolution,
    stress integration or constraints are supported. This is an approximate
    trajectory, even if the teacher is full FP32.

    The input and its calculator are untouched by default. With ``inplace=True``
    only positions and momenta are updated, after the entire run succeeds; the
    original calculator is retained. Returned reference arrays own their storage.
    Factory-created calculators are used only on private copies of the structure.
    """
    if not isinstance(options, RESPAOptions):
        raise TypeError("options must be RESPAOptions")
    outer_steps = _count(outer_steps, "outer_steps", zero=True)
    if not isinstance(inplace, bool):
        raise TypeError("inplace must be boolean")
    working = _owned_atoms(atoms)
    if working.constraints:
        raise ValueError("RESPA supports fixed-cell, unconstrained NVE only")
    masses = _finite(working.get_masses(), "masses")[:, None]
    if np.any(masses <= 0):
        raise ValueError("Masses must be positive")
    momentum = _finite(working.get_momenta(), "momenta")
    teacher = _owned_atoms(working)
    teacher.calc = _new_calculator(teacher_factory, [atoms.calc])
    student = _owned_atoms(working)
    student.calc = _new_calculator(student_factory, [atoms.calc, teacher.calc])
    teacher_force = _forces(teacher)
    reference_energy = _energy(teacher)
    student_force = _forces(student)
    energies = [reference_energy + float((momentum**2 / (2 * masses)).sum())]
    _finite(energies, "total energy")
    inner_dt = options.inner_timestep
    outer_dt = inner_dt * options.inner_steps
    for _ in range(outer_steps):
        momentum += 0.5 * outer_dt * (teacher_force - student_force)
        for _ in range(options.inner_steps):
            momentum += 0.5 * inner_dt * student_force
            positions = _finite(
                working.positions + inner_dt * momentum / masses, "positions"
            )
            working.set_positions(positions)
            student.set_positions(positions)
            student_force = _forces(student)
            momentum += 0.5 * inner_dt * student_force
            _finite(momentum, "momenta")
        teacher.set_positions(working.positions)
        teacher_force = _forces(teacher)
        reference_energy = _energy(teacher)
        momentum += 0.5 * outer_dt * (teacher_force - student_force)
        momentum = _finite(momentum, "momenta")
        total = reference_energy + float((momentum**2 / (2 * masses)).sum())
        _finite(total, "total energy")
        energies.append(total)
    working.set_momenta(momentum)
    if inplace:
        atoms.set_positions(working.positions)
        atoms.set_momenta(momentum)
        working = atoms
    return RESPAResult(
        atoms=working,
        reference_energy=reference_energy,
        reference_forces=teacher_force.copy(),
        total_energies=tuple(energies),
        max_energy_drift=float(np.max(np.abs(np.asarray(energies) - energies[0]))),
        teacher_force_evaluations=outer_steps + 1,
        student_force_evaluations=outer_steps * options.inner_steps + 1,
    )


def run_coarse_refine(
    atoms: Atoms,
    *,
    coarse_factory: Callable[[], BaseCalculator],
    reference_factory: Callable[[], BaseCalculator],
    options: CoarseRefineOptions,
    optimizer_cls=BFGS,
) -> CoarseRefineResult:
    """Relax a private structure with coarse then reference calculators.

    Constraints on ASE Atoms are preserved; cell-filter and stress optimization
    are outside this helper. The final force array is unprojected, while the
    reported convergence residual includes ASE constraints, like the optimizer.
    A zero reference step budget performs a reference recheck without refinement.
    The result never asserts that the final basin matches reference-only search.
    """
    if not isinstance(options, CoarseRefineOptions):
        raise TypeError("options must be CoarseRefineOptions")
    working = _owned_atoms(atoms)
    coarse = _new_calculator(coarse_factory, [atoms.calc])
    working.calc = coarse
    with optimizer_cls(working, logfile=None) as optimizer:
        coarse_converged = bool(
            optimizer.run(fmax=options.coarse_fmax, steps=options.coarse_steps)
        )
    _forces(working)
    coarse_energy = _energy(working)
    # Release the private coarse model before loading the reference model.
    # A weak identity guard still rejects a factory that reuses a live model.
    previous = weakref.ref(coarse)
    working.calc = None
    del coarse
    # ASE calculators can contain bound-method cycles; refcounting alone does
    # not release their model storage before the next model is allocated.
    gc.collect()
    reference = _new_calculator(reference_factory, [atoms.calc, previous()])
    working.calc = reference
    with optimizer_cls(working, logfile=None) as optimizer:
        optimizer.run(fmax=options.reference_fmax, steps=options.reference_steps)
    forces = _forces(working)
    residual = _force_max(_forces(working, constrained=True))
    return CoarseRefineResult(
        atoms=working,
        coarse_energy=coarse_energy,
        coarse_converged=coarse_converged,
        reference_energy=_energy(working),
        reference_forces=forces,
        reference_force_max=residual,
        converged=residual < options.reference_fmax,
    )


def run_dyneb(
    images: Sequence[Atoms],
    *,
    calculator_factory: Callable[[], BaseCalculator],
    options: DyNEBOptions,
    reference_calculator_factory: Callable[[], BaseCalculator] | None = None,
    optimizer_cls=FIRE,
) -> DyNEBResult:
    """Run serial scaled DyNEB on owned images and recheck unmasked NEB forces.

    Each image receives a fresh calculator. If provided, the reference factory
    replaces all calculators after dynamic optimization, before either final
    refinement or rechecking. Otherwise the same private calculators are reused.
    Reference calculators should be configured for the user's strict accuracy.
    No calculator precision or inference setting is changed by this function.
    Endpoint geometries remain fixed. Returned images and result arrays are owned.
    """
    from ase.parallel import world

    if not isinstance(options, DyNEBOptions):
        raise TypeError("options must be DyNEBOptions")
    if world.size != 1:
        raise ValueError("Scaled DyNEB supports serial execution only")
    images = tuple(images)
    if len(images) < 3:
        raise ValueError("NEB requires at least three images")
    working = tuple(_owned_atoms(image) for image in images)
    neb_kwargs = {
        "k": options.spring_constant,
        "climb": options.climb,
        "parallel": False,
        "method": "improvedtangent",
    }
    # Do not pass dynamic_relaxation: newer ASE removed this obsolete keyword;
    # DyNEB already enables dynamic relaxation in both old and current APIs.
    dynamic = DyNEB(
        working, fmax=options.fmax, scale_fmax=options.scale_fmax, **neb_kwargs
    )
    # Let ASE validate atom counts, species and cells before allocating models.
    calculators = [
        weakref.ref(image.calc) for image in images if image.calc is not None
    ]
    for image in working:
        image.calc = _new_calculator(
            calculator_factory, [reference() for reference in calculators]
        )
        calculators.append(weakref.ref(image.calc))
    with optimizer_cls(dynamic, logfile=None) as optimizer:
        dynamic_converged = bool(
            optimizer.run(fmax=options.fmax, steps=options.max_steps)
        )
    if reference_calculator_factory is not None:
        for image in working:
            image.calc = None
        gc.collect()
        for image in working:
            image.calc = _new_calculator(
                reference_calculator_factory, [reference() for reference in calculators]
            )
            calculators.append(weakref.ref(image.calc))
    reference = NEB(working, **neb_kwargs)
    if options.final_refine:
        with optimizer_cls(reference, logfile=None) as optimizer:
            optimizer.run(fmax=options.reference_fmax, steps=options.reference_steps)
    neb_forces = _finite(reference.get_forces(), "unmasked NEB forces")
    residual = _force_max(neb_forces)
    return DyNEBResult(
        images=working,
        dynamic_converged=dynamic_converged,
        reference_energies=tuple(_energy(image) for image in working),
        reference_forces=tuple(_forces(image) for image in working),
        neb_forces=neb_forces,
        reference_force_max=residual,
        refined=options.final_refine,
        converged=residual < options.reference_fmax,
    )

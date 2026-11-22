"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ase.constraints import FixAtoms, FixCom, FixSubsetCom
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

if TYPE_CHECKING:
    from ase import Atoms


def initialize_momenta(
    atoms: Atoms,
    temperature_K: float,
    seed: int,
    remove_center_of_mass_momentum: bool = True,
) -> None:
    """
    Initialize Maxwell-Boltzmann momenta reproducibly.

    Args:
        atoms: Structure whose momenta will be replaced.
        temperature_K: Target initialization temperature.
        seed: NumPy random seed recorded with the run.
        remove_center_of_mass_momentum: Remove net linear momentum when true.
    """

    if not np.isfinite(temperature_K) or temperature_K < 0:
        raise ValueError("Initialization temperature must be finite and nonnegative")

    if remove_center_of_mass_momentum:
        unsupported_constraints = [
            type(constraint).__name__
            for constraint in atoms.constraints
            if not isinstance(constraint, (FixAtoms, FixCom))
        ]
        if unsupported_constraints:
            raise ValueError(
                "Center-of-mass momentum removal supports only FixAtoms and "
                f"FixCom constraints, got {unsupported_constraints}. Set "
                "remove_center_of_mass_momentum=False and initialize the "
                "constrained system explicitly."
            )

        mobile = np.ones(len(atoms), dtype=bool)
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                mobile[constraint.get_indices()] = False
        if mobile.sum() < 2:
            raise ValueError(
                "Center-of-mass momentum removal requires at least two mobile "
                "atoms. Set remove_center_of_mass_momentum=False for this system."
            )

    rng = np.random.RandomState(seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)
    if remove_center_of_mass_momentum:
        momenta = atoms.get_momenta()
        masses = atoms.get_masses()
        momenta[~mobile] = 0.0
        initial_kinetic_energy = np.sum(
            momenta[mobile] ** 2 / (2 * masses[mobile, np.newaxis])
        )
        subsets = [c for c in atoms.constraints if isinstance(c, FixSubsetCom)]
        if subsets:
            # Project onto all requested zero-momentum constraints together.
            # Sequential corrections can reintroduce drift in overlapping sets.
            groups = np.zeros((1 + len(subsets), len(atoms)))
            groups[0] = mobile
            for row, constraint in zip(groups[1:], subsets, strict=True):
                row[constraint.get_indices()] = 1
            groups[:, ~mobile] = 0
            weighted = groups * masses
            correction, _, rank, _ = np.linalg.lstsq(
                weighted @ groups.T, groups @ momenta, rcond=None
            )
            if rank == mobile.sum():
                momenta.fill(0)
            else:
                momenta -= weighted.T @ correction
        else:
            center_of_mass_velocity = momenta[mobile].sum(axis=0) / masses[mobile].sum()
            momenta[mobile] -= masses[mobile, np.newaxis] * center_of_mass_velocity
        corrected_kinetic_energy = np.sum(
            momenta[mobile] ** 2 / (2 * masses[mobile, np.newaxis])
        )
        if corrected_kinetic_energy <= 0 and initial_kinetic_energy > 0:
            raise ValueError(
                "Center-of-mass momentum removal produced zero kinetic energy. "
                "Set remove_center_of_mass_momentum=False for this system."
            )
        if initial_kinetic_energy > 0:
            momenta[mobile] *= np.sqrt(
                initial_kinetic_energy / corrected_kinetic_energy
            )
        atoms.set_momenta(momenta, apply_constraint=False)
    atoms.info["velocity_seed"] = seed
    atoms.info["initial_temperature_K"] = temperature_K

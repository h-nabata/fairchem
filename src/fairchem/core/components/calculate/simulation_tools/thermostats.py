"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixCom, FixSubsetCom
from ase.md.bussi import Bussi
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.verlet import VelocityVerlet
from monty.json import jsanitize

if TYPE_CHECKING:
    from ase import Atoms
    from ase.md.md import MolecularDynamics


class Thermostat(ABC):
    """
    Abstract base class defining the interface for MD thermostats.

    All thermostats must implement three methods:
        - build: construct an ASE MolecularDynamics integrator
        - save_state: serialize thermostat-specific state for checkpointing
        - restore_state: restore thermostat state from a checkpoint
    """

    @abstractmethod
    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        """
        Build and return an ASE MolecularDynamics integrator.

        Args:
            atoms: The atomic system to simulate.
            timestep_fs: Integration timestep in femtoseconds.

        Returns:
            An ASE MolecularDynamics object.
        """

    def build_for_restart(
        self, atoms: Atoms, timestep_fs: float, state: dict
    ) -> MolecularDynamics:
        dyn = self.build(atoms, timestep_fs)
        self.restore_state(dyn, state)
        return dyn

    @abstractmethod
    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        """
        Save thermostat-specific state for checkpointing.

        Args:
            dyn: The active MolecularDynamics integrator.

        Returns:
            A JSON-serializable dict of thermostat state.
        """

    @abstractmethod
    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        """
        Restore thermostat state from a checkpoint.

        Args:
            dyn: The active MolecularDynamics integrator.
            state: The state dict previously returned by save_state.
        """


@dataclass
class VelocityVerletThermostat(Thermostat):
    """
    NVE dynamics (no thermostat).
    """

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return VelocityVerlet(atoms=atoms, timestep=timestep_fs * units.fs)

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {"class_name": "VelocityVerletThermostat"}

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        pass


@dataclass
class NoseHooverNVT(Thermostat):
    """
    Nose-Hoover chain NVT thermostat.
    """

    temperature_K: float  # Kelvin
    tdamp_fs: float  # femtoseconds

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return NoseHooverChainNVT(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            tdamp=self.tdamp_fs * units.fs,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        thermostat = dyn._thermostat
        return jsanitize(
            {
                "class_name": "NoseHooverNVT",
                "eta": thermostat._eta,
                "p_eta": thermostat._p_eta,
            }
        )

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        thermostat = dyn._thermostat
        thermostat._eta = np.array(state["eta"])
        thermostat._p_eta = np.array(state["p_eta"])


def _save_rng(rng: np.random.RandomState) -> dict:
    algorithm, keys, pos, has_gauss, cached_gaussian = rng.get_state()
    return {
        "version": 1,
        "algorithm": algorithm,
        "keys": keys.tolist(),
        "pos": pos,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _restore_rng(rng: np.random.RandomState, state: dict) -> None:
    # The pre-versioned format stored the same five MT19937 fields.
    if state.get("version", 1) != 1:
        raise ValueError("Unsupported thermostat RNG state version")
    rng.set_state(
        (
            state["algorithm"],
            np.array(state["keys"], dtype=np.uint32),
            state["pos"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )


@dataclass
class BussiThermostat(Thermostat):
    """
    Bussi stochastic velocity rescaling thermostat.
    """

    temperature_K: float  # Kelvin
    taut_fs: float  # femtoseconds
    thermostat_seed: int | None = None

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return Bussi(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            taut=self.taut_fs * units.fs,
            rng=np.random.RandomState(self.thermostat_seed),
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {
            "class_name": "BussiThermostat",
            "rng_state": _save_rng(dyn.rng),
            "transferred_energy": float(dyn.transferred_energy),
        }

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        _restore_rng(dyn.rng, state["rng_state"])
        if "transferred_energy" in state:
            dyn.transferred_energy = state["transferred_energy"]


@dataclass
class LangevinThermostat(Thermostat):
    """
    Langevin stochastic dynamics thermostat.
    """

    temperature_K: float  # Kelvin
    friction_per_fs: float  # 1/femtoseconds
    # None selects FixCom for new runs and the original behavior for legacy restarts.
    use_fix_com_constraint: bool | None = None
    thermostat_seed: int | None = None

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        use_constraint = self.use_fix_com_constraint is not False
        if use_constraint:
            mobile = np.ones(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    mobile[constraint.get_indices()] = False
            if mobile.sum() < 2:
                raise ValueError(
                    "FixCom requires at least two mobile atoms in Langevin dynamics."
                )
            com_constraints = [c for c in atoms.constraints if isinstance(c, FixCom)]
            if not com_constraints:
                fix_com = (
                    FixCom()
                    if mobile.all()
                    else FixSubsetCom(indices=np.flatnonzero(mobile))
                )
                atoms.set_constraint([*atoms.constraints, fix_com])
            elif not mobile.all():
                constraints = []
                for constraint in atoms.constraints:
                    selected = constraint
                    if isinstance(constraint, FixSubsetCom):
                        indices = constraint.get_indices()
                        if len(indices) < 2 or not mobile[indices].all():
                            raise ValueError(
                                "FixSubsetCom requires at least two mobile atoms and no fixed atoms"
                            )
                    elif isinstance(constraint, FixCom):
                        selected = FixSubsetCom(indices=np.flatnonzero(mobile))
                    constraints.append(selected)
                atoms.set_constraint(constraints)
            if atoms.get_number_of_degrees_of_freedom() <= 0:
                raise ValueError(
                    "No unconstrained degrees of freedom remain for Langevin dynamics"
                )

        return Langevin(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            friction=self.friction_per_fs / units.fs,
            fixcm=not use_constraint,
            rng=np.random.RandomState(self.thermostat_seed),
        )

    def build_for_restart(self, atoms, timestep_fs, state):
        mode = state.get("use_fix_com_constraint", bool(self.use_fix_com_constraint))
        dyn = replace(self, use_fix_com_constraint=mode).build(atoms, timestep_fs)
        self.restore_state(dyn, state)
        return dyn

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {
            "class_name": "LangevinThermostat",
            "rng_state": _save_rng(dyn.rng),
            "use_fix_com_constraint": not dyn.fix_com,
        }

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        _restore_rng(dyn.rng, state["rng_state"])


@dataclass
class BerendsenNPT(Thermostat):
    """
    Berendsen NPT thermostat/barostat for constant pressure simulations.
    """

    temperature_K: float
    pressure_bar: float = 1.0
    taut_fs: float = 5.0
    taup_fs: float = 500.0
    compressibility_bar: float = 5e-7

    def build(self, atoms: Atoms, timestep_fs: float) -> MolecularDynamics:
        return NPTBerendsen(
            atoms=atoms,
            timestep=timestep_fs * units.fs,
            temperature_K=self.temperature_K,
            pressure_au=self.pressure_bar * units.bar,
            taut=self.taut_fs * units.fs,
            taup=self.taup_fs * units.fs,
            compressibility_au=self.compressibility_bar / units.bar,
        )

    def save_state(self, dyn: MolecularDynamics) -> dict[str, Any]:
        return {"class_name": "BerendsenNPT"}

    def restore_state(self, dyn: MolecularDynamics, state: dict[str, Any]) -> None:
        pass

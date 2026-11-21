"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ase.io
import numpy as np
from ase import Atoms
from ase.data import atomic_masses

from fairchem.core.regression_suite.spec.hashing import json_value, stable_hash
from fairchem.core.regression_suite.spec.schema import (
    CaseSpec,
    RegressionSchemaError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# Existing plans implicitly use this ASE standard atomic-mass table. Retain
# their hashes only when the active table and all effective masses agree.
_LEGACY_MASS_TABLE = "7d0812980eef95181692381ed0b25eb7d30372a5dc77507efe7cd34266963895"


def atoms_hash(atoms: Atoms, *, mass_dependent: bool = False) -> str:
    """
    Return a stable hash of model-relevant ASE structure data.
    """
    payload = {
        "numbers": np.asarray(atoms.numbers, dtype=np.int64).tolist(),
        "positions": np.asarray(atoms.positions, dtype=np.float64).tolist(),
        "cell": np.asarray(atoms.cell, dtype=np.float64).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "tags": np.asarray(atoms.get_tags(), dtype=np.int64).tolist(),
        "constraints": [
            json_value(constraint.todict()) for constraint in atoms.constraints
        ],
        "charge": json_value(atoms.info.get("charge")),
        "spin": json_value(atoms.info.get("spin")),
    }
    if mass_dependent:
        masses = atoms.get_masses()
        if stable_hash(
            atomic_masses.tolist()
        ) != _LEGACY_MASS_TABLE or not np.array_equal(
            masses, atomic_masses[atoms.numbers]
        ):
            payload["mass_fingerprint_v1"] = masses.tolist()
    return stable_hash(payload)


def frame_identities(
    source: str | Path | Iterable[Atoms], *, mass_dependent: bool = False
) -> list[tuple[str, str]]:
    """
    Return the ``structure_id`` and plan hash of each frame of a structure file
    or sequence of generated structures.
    """
    frames = (
        ase.io.read(source, index=":") if isinstance(source, (str, Path)) else source
    )
    return [
        (
            str(atoms.info["structure_id"]),
            atoms_hash(atoms, mass_dependent=mass_dependent),
        )
        for atoms in frames
    ]


def load_source_atoms(
    case: CaseSpec, suite_directory: str | Path, *, mass_dependent: bool = False
) -> Atoms:
    """
    Load and hash-check the source structure for one plan case.
    """
    relative_path = Path(case.structure)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RegressionSchemaError("structure path must stay within its suite")
    path = Path(suite_directory) / relative_path
    atoms = ase.io.read(path, index=case.structure_index)
    if not isinstance(atoms, Atoms):
        raise RegressionSchemaError(f"structure did not resolve to Atoms: {path}")
    return check_source_atoms(atoms, case, mass_dependent=mass_dependent)


def check_source_atoms(
    atoms: Atoms, case: CaseSpec, *, mass_dependent: bool = False
) -> Atoms:
    """
    Check a loaded or generated source structure against its plan hash.
    """
    if atoms_hash(atoms, mass_dependent=mass_dependent) != case.structure_hash:
        raise RegressionSchemaError(f"structure hash differs for {case.case_id!r}")
    atoms.calc = None
    return atoms

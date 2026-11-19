"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from ase.data import atomic_masses, chemical_symbols
from ase.geometry import wrap_positions

from fairchem.core.datasets.atomic_data import AtomicData
from lammps import lammps

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol
    from fairchem.core.units.mlip_unit.utils import DictConfig

FIX_EXT_ID = "ext_fc"
FIX_EXTERNAL_CMD = f"fix {FIX_EXT_ID} all external pf/callback 1 1"

FORCE_COMMANDS = ["pair_style", "bond_style", "angle_style", "dihedral_style"]


def check_input_script(input_script: str):
    for cmd in FORCE_COMMANDS:
        if cmd in input_script:
            logging.warning(
                f"Input script contains force field command '{cmd}'. These forces will be incorrectly added to the MLIP forces, please remove them unless you know what you are doing."
            )


def check_atom_id_match_masses(types_arr, masses):
    for atom_id in types_arr:
        assert np.allclose(masses[atom_id], atomic_masses[atom_id], atol=1e-1), (
            f"Atom {chemical_symbols[atom_id]} (type {atom_id}) has mass {masses[atom_id]} but is expected to have mass {atomic_masses[atom_id]}."
        )


def atomic_data_from_lammps_data(
    x: np.ndarray | torch.Tensor,
    atomic_numbers,
    nlocal,
    cell,
    periodicity,
    task_name,
    charge: int = 0,
    spin: int = 0,
    dtype: torch.dtype = torch.float32,
):
    pos = torch.as_tensor(x, dtype=dtype, device="cpu")
    pbc = torch.tensor(periodicity, dtype=torch.bool, device="cpu").unsqueeze(0)
    edge_index = torch.empty((2, 0), dtype=torch.long, device="cpu")
    cell_offsets = torch.empty((0, 3), dtype=dtype, device="cpu")
    nedges = torch.tensor([0], dtype=torch.long, device="cpu")
    tags = torch.zeros(nlocal, dtype=torch.long, device="cpu")
    fixed = torch.zeros(nlocal, dtype=torch.long, device="cpu")
    batch = torch.zeros(nlocal, dtype=torch.long, device="cpu")
    atomic_data = AtomicData(
        pos=pos,
        atomic_numbers=torch.as_tensor(atomic_numbers, dtype=torch.long, device="cpu"),
        cell=torch.as_tensor(cell, dtype=dtype, device="cpu"),
        pbc=pbc,
        natoms=torch.tensor([nlocal], dtype=torch.long, device="cpu"),
        edge_index=edge_index,
        cell_offsets=cell_offsets,
        nedges=nedges,
        charge=torch.tensor([charge], dtype=torch.long, device="cpu"),
        spin=torch.tensor([spin], dtype=torch.long, device="cpu"),
        fixed=fixed,
        tags=tags,
        batch=batch,
        dataset=[task_name],
    )
    return atomic_data


def lookup_atomic_number_by_mass(mass_arr: np.ndarray | float) -> np.ndarray | int:
    """
    Lookup atomic numbers by closest atomic masses.

    Args:
        mass_arr (float or np.ndarray): Target atomic mass(es)

    Returns:
        int or np.ndarray: Atomic number(s)

    Raises:
        ValueError: If any mass doesn't match within 0.1 tolerance
    """
    # Convert input to numpy array
    mass_arr = np.asarray(mass_arr)
    scalar_input = mass_arr.ndim == 0

    # Ensure we work with at least 1D array
    if scalar_input:
        mass_arr = mass_arr.reshape(1)

    # Convert ASE atomic masses to numpy array (skip index 0 which is empty)
    reference_masses = np.array(atomic_masses[1:])

    # Calculate absolute differences for all input masses vs all reference masses
    # Shape: (len(mass_arr), len(reference_masses))
    diffs = np.abs(mass_arr[:, np.newaxis] - reference_masses[np.newaxis, :])

    # Find minimum difference and corresponding index for each input mass
    min_indices = np.argmin(diffs, axis=1)
    min_diffs = np.min(diffs, axis=1)

    # Check if any mass is outside tolerance
    bad_matches = ~np.isfinite(min_diffs) | (min_diffs > 0.1)
    if np.any(bad_matches):
        bad_masses = mass_arr[bad_matches]
        raise ValueError(f"No atomic mass found within 0.1 of {bad_masses}")

    # Add 1 because we skipped index 0
    atomic_numbers = min_indices + 1

    # Return scalar if input was scalar
    if scalar_input:
        return atomic_numbers[0]

    return atomic_numbers


def separate_run_commands(input_script: str) -> tuple[list[str], list[str]]:
    lines = input_script.splitlines()
    for index, line in enumerate(lines):
        command = line.split("#", 1)[0].split(maxsplit=1)
        if command and command[0] in {"run", "minimize", "rerun"}:
            return lines[:index], lines[index:]
    return lines, []


def restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz, dtype=torch.float32):
    lx = boxhi[0] - boxlo[0]
    ly = boxhi[1] - boxlo[1]
    lz = boxhi[2] - boxlo[2]

    # Lammps uses a restricted triclinic box defined by 6 parameters: lx, ly, lz, xy, xz, yz. See https://docs.lammps.org/Howto_triclinic.html for details.
    # ASE uses a 3x3 cell matrix where the rows are the lattice vectors.
    unit_cell_matrix = torch.tensor(
        [
            [lx, 0, 0],  # a vector
            [xy, ly, 0],  # b vector
            [xz, yz, lz],  # c vector
        ],
        dtype=dtype,
        device="cpu",
    )
    return unit_cell_matrix.unsqueeze(0)


class FixExternalCallback:
    def __init__(self, charge: int = 0, spin: int = 0, type_to_atomic_number=None):
        self.charge = charge
        self.spin = spin
        self.type_to_atomic_number = dict(type_to_atomic_number or {})
        for type_id, number in self.type_to_atomic_number.items():
            if (
                not isinstance(type_id, int)
                or type_id < 1
                or not isinstance(number, int)
                or not 1 <= number < len(atomic_masses)
            ):
                raise ValueError(
                    "Expected positive integer LAMMPS types mapped to atomic numbers 1..118"
                )
        self._masses = None
        self._numbers = None

    def _atomic_numbers(self, atom_types, masses):
        # LAMMPS exposes a mutable type table. Copy masses, so a later mass
        # command invalidates just those entries even if LAMMPS reuses its buffer.
        if self._masses is None or self._masses.shape != masses.shape:
            self._masses = np.full(masses.shape, np.nan)
            self._numbers = np.zeros(masses.shape, dtype=np.int64)
        if np.any(atom_types <= 0) or np.any(atom_types >= len(masses)):
            raise ValueError("LAMMPS returned an atom type outside the mass table")
        self._numbers[self._masses != masses] = 0
        self._masses[:] = masses
        numbers = self._numbers[atom_types]
        if np.any(numbers == 0):
            for type_id in np.unique(atom_types[numbers == 0]):
                number = self.type_to_atomic_number.get(type_id)
                if number is None:
                    number = lookup_atomic_number_by_mass(masses[type_id])
                self._numbers[type_id] = number
            numbers = self._numbers[atom_types]
        return numbers

    def __call__(self, lmp, ntimestep, nlocal, tag, x, f):
        atom_types = lmp.numpy.extract_atom("type")[:nlocal]
        atomic_numbers = self._atomic_numbers(
            atom_types, lmp.numpy.extract_atom("mass")
        )
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        predictor = lmp._predictor
        settings = getattr(predictor, "inference_settings", None)
        dtype = getattr(settings, "base_precision_dtype", torch.float32)
        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz, dtype=dtype)

        x_wrapped = wrap_positions(
            x, cell=cell.squeeze().numpy(), pbc=periodicity, eps=0
        )

        atomic_data = atomic_data_from_lammps_data(
            x_wrapped,
            atomic_numbers,
            nlocal,
            cell,
            periodicity,
            lmp._task_name,
            charge=self.charge,
            spin=self.spin,
            dtype=dtype,
        )
        if getattr(predictor, "supports_requested_properties", False):
            properties = (
                ("energy", "forces", "stress") if box_change else ("energy", "forces")
            )
            results = predictor.predict(atomic_data, requested_properties=properties)
        else:
            results = predictor.predict(atomic_data)
        assert "forces" in results, "forces must be in results"
        f[:] = results["forces"].detach().cpu().numpy()[:]
        lmp.fix_external_set_energy_global(FIX_EXT_ID, results["energy"].item())

        # during NPT for example, box_change should be set to 1 by lammps to allow the cell to change
        if box_change:
            # stress is defined as -virial/volume in lammps
            assert "stress" in results, (
                f"stress must be in results to compute virial. Predictios can be enabled by setting `predict_untrained_stress=set('{lmp._task_name}')` "
            )
            volume = torch.det(cell).abs().item()
            v = (
                (-results["stress"].detach().cpu() * volume)
                .reshape(3, 3)
                .flatten()
                .tolist()
            )
            # virials need to be in this order: xx, yy, zz, xy, xz, yz. https://docs.lammps.org/Library_utility.html#_CPPv437lammps_fix_external_set_virial_globalPvPKcPd
            virial_arr = [v[0], v[4], v[8], v[1], v[2], v[5]]
            lmp.fix_external_set_virial_global(FIX_EXT_ID, virial_arr)


def run_lammps_with_fairchem(
    predictor: MLIPPredictUnitProtocol,
    lammps_input_path: str,
    task_name: str,
    charge: int = 0,
    spin: int = 0,
    type_to_atomic_number=None,
):
    machine = None
    if "LAMMPS_MACHINE_NAME" in os.environ:
        machine = os.environ["LAMMPS_MACHINE_NAME"]
    lmp = lammps(name=machine, cmdargs=["-nocite", "-log", "none", "-echo", "screen"])
    lmp._predictor = predictor
    lmp._task_name = task_name
    # run_cmds = []
    with open(lammps_input_path) as f:
        input_script = f.read()
        check_input_script(input_script)
        script, run_cmds = separate_run_commands(input_script)
        logging.info(f"Running input script: {input_script}")
        lmp.commands_list(script)
        lmp.command(FIX_EXTERNAL_CMD)
        fix_external_call_back = FixExternalCallback(
            charge=charge, spin=spin, type_to_atomic_number=type_to_atomic_number
        )
        lmp.set_fix_external_callback(FIX_EXT_ID, fix_external_call_back, lmp)
        lmp.commands_list(run_cmds)
    return lmp


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="lammps_fc_config",
)
def main(cfg: DictConfig):
    predict_unit = hydra.utils.instantiate(cfg.predict_unit)
    lmp = run_lammps_with_fairchem(
        predict_unit,
        cfg.lmp_in,
        cfg.task_name,
        cfg.charge,
        cfg.spin,
        **(
            {"type_to_atomic_number": cfg.type_to_atomic_number}
            if cfg.get("type_to_atomic_number")
            else {}
        ),
    )
    # this is required to cleanup the predictor
    del lmp._predictor


if __name__ == "__main__":
    main()

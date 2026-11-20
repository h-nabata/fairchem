"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
import shlex
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols
from ase.geometry import wrap_positions

from fairchem.core.calculate.ase_calculator import (
    _calculator_predictions,
    _predictor_graph_options,
)
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import validate_uma_atoms_data
from lammps import lammps

if TYPE_CHECKING:
    from collections.abc import Callable

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
        assert np.allclose(
            masses[atom_id], atomic_masses[atom_id], atol=1e-1
        ), f"Atom {chemical_symbols[atom_id]} (type {atom_id}) has mass {masses[atom_id]} but is expected to have mass {atomic_masses[atom_id]}."


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
    def __init__(
        self,
        charge=None,
        spin=None,
        type_to_atomic_number=None,
        *,
        graph_options=None,
        compute_stress=True,
        comm=None,
    ):
        self.graph_options = graph_options
        self.compute_stress = compute_stress
        self.comm = comm
        self.error = None
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

    def _predict(self, lmp, x, atom_types):
        atomic_numbers = self._atomic_numbers(
            atom_types, lmp.numpy.extract_atom("mass")
        )
        boxlo, boxhi, xy, yz, xz, periodicity, box_change = lmp.extract_box()
        if box_change and not self.compute_stress:
            raise ValueError("Cell-changing simulations require compute_stress=True")
        if self.graph_options is None:
            # Preserve direct use as a LAMMPS callback as well as the script API.
            self.graph_options = _predictor_graph_options(lmp._predictor)
            metadata = _charge_spin(lmp._task_name, self.charge, self.spin)
            self.charge, self.spin = metadata["charge"], metadata["spin"]
        dtype = self.graph_options["target_dtype"]
        cell = restricted_cell_from_lammps_box(boxlo, boxhi, xy, yz, xz, dtype=dtype)
        if self.graph_options["r_edges"]:
            atoms = Atoms(
                numbers=atomic_numbers,
                positions=x,
                cell=cell[0].numpy(),
                pbc=periodicity,
                info={"charge": self.charge, "spin": self.spin},
            )
            data = AtomicData.from_ase(
                atoms,
                task_name=lmp._task_name,
                r_data_keys=["charge", "spin"],
                **self.graph_options,
            )
        else:
            wrapped = wrap_positions(x, cell=cell[0].numpy(), pbc=periodicity, eps=0)
            data = atomic_data_from_lammps_data(
                wrapped,
                atomic_numbers,
                len(x),
                cell,
                periodicity,
                lmp._task_name,
                self.charge,
                self.spin,
                dtype,
            )
        properties = (
            ("energy", "forces", "stress")
            if self.compute_stress
            else ("energy", "forces")
        )
        predictor = lmp._predictor
        if getattr(predictor, "supports_requested_properties", False):
            raw = predictor.predict(data, requested_properties=properties)
        else:
            raw = predictor.predict(data)
        missing = set(properties).difference(raw)
        if missing:
            raise ValueError(
                f"LAMMPS predictor did not return required properties: {sorted(missing)}"
            )
        result = _calculator_predictions(raw, properties, len(x))
        virial = None
        if self.compute_stress:
            # ASE order is xx, yy, zz, yz, xz, xy. LAMMPS requires xx, yy, zz, xy, xz, yz.
            volume = np.prod(np.subtract(boxhi, boxlo), dtype=np.float64)
            virial = (-volume * result["stress"][[0, 1, 2, 5, 4, 3]]).tolist()
        return result["energy"], virial, result["forces"]

    def _distributed_prediction(self, lmp, nlocal, tag, x):
        # LAMMPS may reorder or migrate atoms every step. Sort global inputs by
        # tag, then scatter forces in each rank's original callback order.
        comm = self.comm
        local_error = None
        try:
            if nlocal:
                types = lmp.numpy.extract_atom("type")[:nlocal]
                local = (tag.reshape(nlocal), x.reshape(nlocal, 3), types)
            else:
                local = (np.empty(0, np.int64), np.empty((0, 3)), np.empty(0, np.int64))
        except BaseException as exc:
            local_error = exc
            local = None
        blocks = comm.gather(
            (str(local_error) if local_error is not None else None, local), root=0
        )
        root_error = None
        force_blocks = None
        status = None
        if comm.rank == 0:
            try:
                for rank, (message, _) in enumerate(blocks):
                    if message is not None:
                        raise RuntimeError(
                            f"LAMMPS rank {rank} input failed: {message}"
                        )
                ids, positions, types = zip(*(block for _, block in blocks))
                ids_flat = np.concatenate(ids)
                order = np.argsort(ids_flat)
                if np.any(np.diff(ids_flat[order]) <= 0):
                    raise ValueError(
                        "Distributed LAMMPS inference requires unique atom IDs"
                    )
                energy, virial, forces = self._predict(
                    lmp, np.concatenate(positions)[order], np.concatenate(types)[order]
                )
                rank_forces = np.empty_like(forces)
                rank_forces[order] = forces
                force_blocks = np.split(
                    rank_forces, np.cumsum([len(v) for v in ids[:-1]])
                )
                status = (None, energy, virial)
            except BaseException as exc:
                root_error = exc
                status = (f"{type(exc).__name__}: {exc}", None, None)
        message, energy, virial = comm.bcast(status, root=0)
        if message is not None:
            if local_error is not None:
                raise local_error
            if root_error is not None:
                raise root_error
            raise RuntimeError(f"LAMMPS global prediction failed: {message}")
        return energy, virial, comm.scatter(force_blocks, root=0)

    def __call__(self, lmp, ntimestep, nlocal, tag, x, f):
        if self.error is not None:
            return
        try:
            if self.comm is None:
                types = lmp.numpy.extract_atom("type")[:nlocal]
                energy, virial, forces = self._predict(lmp, x, types)
            else:
                energy, virial, forces = self._distributed_prediction(
                    lmp, nlocal, tag, x
                )
            lmp.fix_external_set_energy_global(FIX_EXT_ID, energy)
            if virial is not None:
                lmp.fix_external_set_virial_global(FIX_EXT_ID, virial)
            if nlocal:
                f[:] = forces
        except BaseException as exc:
            # ctypes cannot propagate Python exceptions across the C callback.
            # Stop the active run and mark a failed thermodynamic sample as invalid.
            self.error = exc
            lmp.force_timeout()
            lmp.fix_external_set_energy_global(FIX_EXT_ID, float("nan"))
            lmp.fix_external_set_virial_global(FIX_EXT_ID, [float("nan")] * 6)
            if nlocal:
                f[:] = np.nan


class _FairChemLammps(lammps):
    """Execute through a Python error boundary for every simulation command."""

    def file(self, path):
        path = Path(path).resolve()
        if path in self._input_stack:
            raise ValueError(f"Recursive LAMMPS include: {path}")
        comm = self._fc_callback.comm
        error, source = None, None
        if comm is None or comm.rank == 0:
            try:
                source = path.read_text()
            except (OSError, UnicodeError) as exc:
                error = exc
        if comm is not None:
            # Like LAMMPS, only rank zero needs access to its input files.
            error, source = comm.bcast((error, source), root=0)
        if error is not None:
            raise error
        self._input_stack.append(path)
        try:
            self.commands_string(source)
        finally:
            self._input_stack.pop()

    def commands_string(self, source):
        check_input_script(source)
        self.commands_list(source.splitlines())

    def commands_list(self, lines):
        pending = ""
        for raw_line in lines:
            line = raw_line.rstrip()
            if line.endswith("&") and not line.lstrip().startswith("#"):
                pending += line[:-1]
                continue
            self.command(pending + line)
            pending = ""
        if pending:
            raise ValueError("Unterminated LAMMPS command continuation")

    def command(self, line):
        if not line:
            return
        callback = self._fc_callback
        if callback.error is not None:
            raise callback.error
        words = shlex.split(line, comments=True, posix=True)
        if not words:
            return
        command = words[0]
        if command == "include":
            args = shlex.split(self.expand(line), comments=True, posix=True)
            if len(args) != 2:
                raise ValueError(f"Expected a single LAMMPS include filename: {line}")
            self.file(args[1])
            return
        if (
            command in {"jump", "if", "python", "partition"}
            or "$" in command
            or (command == "run" and "every" in words[1:])
        ):
            raise ValueError(
                f"Unsupported nested LAMMPS command in FAIRChem script: {line}"
            )
        if (
            command in {"run", "minimize", "rerun"}
            and not callback.compute_stress
            and self.extract_setting("box_exist")
            and self.extract_box()[-1]
        ):
            raise ValueError("Cell-changing simulations require compute_stress=True")
        if (
            command in {"fix", "unfix", "fix_modify"}
            and len(words) > 1
            and self.expand(words[1]) == FIX_EXT_ID
        ):
            raise ValueError(f"LAMMPS fix ID {FIX_EXT_ID!r} is reserved by FAIRChem")
        try:
            super().command(line)
        finally:
            if callback.error is not None:
                raise callback.error
        # Install as soon as the box exists. Other commands can also evaluate
        # forces (e.g. accelerated dynamics); the first evaluation must see the fix.
        if self.extract_setting("box_exist") and not self.has_id("fix", FIX_EXT_ID):
            units = self.extract_global("units")
            if units != "metal":
                raise ValueError(f"FAIRChem requires LAMMPS units metal, got {units!r}")
            super().command(FIX_EXTERNAL_CMD)
            self.fix_external_set_energy_global(FIX_EXT_ID, 0.0)
            self.fix_external_set_virial_global(FIX_EXT_ID, [0.0] * 6)
            self.set_fix_external_callback(FIX_EXT_ID, callback, self)


def _initialize_predictor(predictor, comm):
    # Only the inference rank needs model weights or accelerator allocations.
    graph_options, error, message = None, None, None
    if comm is None or comm.rank == 0:
        try:
            if not hasattr(predictor, "predict"):
                predictor = predictor()
            graph_options = _predictor_graph_options(predictor)
        except BaseException as exc:
            error = exc
            message = f"{type(exc).__name__}: {exc}"
    else:
        predictor = None
    if comm is not None:
        message = comm.bcast(message, root=0)
    if error is not None:
        raise error
    if message is not None:
        raise RuntimeError(
            f"LAMMPS predictor initialization failed on rank zero: {message}"
        )
    return predictor, graph_options


def _charge_spin(task_name, charge, spin):
    atoms = Atoms(
        info={k: v for k, v in (("charge", charge), ("spin", spin)) if v is not None}
    )
    validate_uma_atoms_data(atoms, task_name)
    return atoms.info


def run_lammps_with_fairchem(
    predictor: MLIPPredictUnitProtocol | Callable[[], MLIPPredictUnitProtocol],
    lammps_input_path: str,
    task_name: str,
    charge: int | None = None,
    spin: int | None = None,
    type_to_atomic_number=None,
    *,
    compute_stress: bool = True,
):
    """Run an input file with a predictor or a factory evaluated on rank zero."""
    metadata = _charge_spin(task_name, charge, spin)
    lmp = _FairChemLammps(
        name=os.environ.get("LAMMPS_MACHINE_NAME"),
        cmdargs=["-nocite", "-log", "none", "-echo", "screen"],
    )
    try:
        comm = None
        if lmp.extract_setting("world_size") > 1:
            comm = lmp.get_mpi_comm()
            if comm is None:
                raise RuntimeError(
                    "Multi-rank LAMMPS requires mpi4py built for its MPI library"
                )
        predictor, graph_options = _initialize_predictor(predictor, comm)
        callback = FixExternalCallback(
            **metadata,
            type_to_atomic_number=type_to_atomic_number,
            graph_options=graph_options,
            compute_stress=compute_stress,
            comm=comm,
        )
        lmp._predictor = predictor
        lmp._task_name = task_name
        lmp._fc_callback = callback
        lmp._input_stack = []
        lmp.file(lammps_input_path)
        return lmp
    except BaseException:
        lmp.close()
        raise


@hydra.main(
    version_base=None,
    config_path=".",
    config_name="lammps_fc_config",
)
def main(cfg: DictConfig):
    predict_unit = partial(hydra.utils.instantiate, cfg.predict_unit)
    lmp = run_lammps_with_fairchem(
        predict_unit,
        cfg.lmp_in,
        cfg.task_name,
        cfg.charge,
        cfg.spin,
        compute_stress=cfg.compute_stress,
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

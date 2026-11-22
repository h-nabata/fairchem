"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import ase.units
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fairchem.core.components.runner import _sync_directory

if TYPE_CHECKING:
    from ase import Atoms

from ase.calculators.calculator import PropertyNotImplementedError
from ase.constraints import FixAtoms


@dataclass
class TrajectoryFrame:
    """
    Single frame of simulation trajectory data.

    Note:
        Currently only step, atomic_numbers, positions, cell, and pbc are
        required. Additional fields may be made optional in the future as
        new simulation types are supported.
    """

    step: int
    atomic_numbers: np.ndarray  # (N,)
    positions: np.ndarray  # (N, 3)
    cell: np.ndarray  # (3, 3)
    pbc: np.ndarray  # (3,) bool
    time: float | None = None  # femtoseconds
    velocities: np.ndarray | None = None  # (N, 3)
    energy: float | None = None
    forces: np.ndarray | None = None  # (N, 3)
    stress: np.ndarray | None = None  # (6,) Voigt notation
    temperature: float | None = None  # Kelvin
    kinetic_energy: float | None = None  # eV
    pressure: float | None = None  # bar
    sid: str | int | None = None
    tags: np.ndarray | None = None  # (N,) int
    fixed: np.ndarray | None = None  # (N,) bool
    charge: int | None = None
    spin: int | None = None

    def to_dict(self) -> dict:
        """
        Convert to dictionary for Parquet serialization.
        """
        return self._snapshot(as_lists=True)

    def _snapshot(self, *, as_lists: bool) -> dict:
        row = {"natoms": len(self.positions)}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                value = value.tolist() if as_lists else value.copy()
            if value is not None or field.name == "sid":
                row[field.name] = value
        return row

    @classmethod
    def from_atoms(
        cls, atoms: Atoms, step: int, time: float | None = None
    ) -> TrajectoryFrame:
        """
        Create a TrajectoryFrame from an ASE Atoms object.

        Args:
            atoms: ASE Atoms object with calculator attached
            step: Current MD step number
            time: Current simulation time

        Returns:
            TrajectoryFrame populated with atoms data
        """

        def property_value(name, getter):
            if atoms.calc is None:
                return None
            try:
                return getter()
            except PropertyNotImplementedError:
                return None
            except Exception as exc:
                exc.add_note(
                    f"Reading {name} at MD step {step}, sid={atoms.info.get('sid')!r}"
                )
                raise

        # A molecule or partial cell can have energy and forces without a
        # physical 3-D volume in which to define stress.
        stress = (
            None
            if atoms.cell.rank < 3 or atoms.cell.volume == 0
            else property_value("stress", atoms.get_stress)
        )
        energy = property_value("energy", atoms.get_potential_energy)
        forces = property_value("forces", atoms.get_forces)
        velocities = atoms.get_velocities()
        kinetic_energy = atoms.get_kinetic_energy()
        temperature = (
            atoms.get_temperature()
            if atoms.get_number_of_degrees_of_freedom() > 0
            else None
        )

        # Pressure from stress: P = -trace(stress)/3, converted to bar
        if stress is not None:
            pressure = -stress[:3].mean() / ase.units.bar
        else:
            pressure = None

        # Tags: only store if any are nonzero (all zeros is ASE default)
        raw_tags = atoms.get_tags()
        tags = raw_tags if np.any(raw_tags) else None

        # Fixed atoms: build bool mask from FixAtoms constraints
        fixed = None
        if atoms.constraints:
            mask = np.zeros(len(atoms), dtype=bool)
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    mask[constraint.index] = True
            if np.any(mask):
                fixed = mask

        charge = atoms.info.get("charge", None)
        spin = atoms.info.get("spin", None)

        sid = atoms.info.get("sid")
        if isinstance(sid, (list, tuple)) and len(sid) == 1:
            # AtomicData.to_ase_single retains the singleton batch dimension.
            sid = sid[0]

        return cls(
            step=step,
            time=time,
            atomic_numbers=atoms.get_atomic_numbers(),
            positions=atoms.get_positions(),
            cell=atoms.get_cell().array,
            pbc=atoms.get_pbc(),
            velocities=velocities,
            energy=energy,
            forces=forces,
            stress=stress,
            temperature=temperature,
            kinetic_energy=kinetic_energy,
            pressure=pressure,
            sid=sid,
            tags=tags,
            fixed=fixed,
            charge=charge,
            spin=spin,
        )


_FLOATS = pa.list_(pa.float64())
_VECTORS = pa.list_(_FLOATS)
_INTS = pa.list_(pa.int64())
_TRAJECTORY_SCHEMA = pa.schema(
    [
        ("step", pa.int64()),
        ("natoms", pa.int64()),
        ("atomic_numbers", _INTS),
        ("positions", _VECTORS),
        ("cell", _VECTORS),
        ("pbc", pa.list_(pa.bool_())),
        ("sid", pa.string()),
        ("sid_integer", pa.int64()),
        ("time", pa.float64()),
        ("velocities", _VECTORS),
        ("energy", pa.float64()),
        ("forces", _VECTORS),
        ("stress", _FLOATS),
        ("temperature", pa.float64()),
        ("kinetic_energy", pa.float64()),
        ("pressure", pa.float64()),
        ("tags", _INTS),
        ("fixed", pa.list_(pa.bool_())),
        ("charge", pa.int64()),
        ("spin", pa.int64()),
    ],
    metadata={b"fairchem.trajectory_version": b"1"},
)


def _trajectory_table(rows):
    for row in rows:
        sid = row.get("sid")
        if isinstance(sid, (int, np.integer)):
            row["sid_integer"] = int(sid)
            row["sid"] = str(sid)
    columns = []
    for field in _TRAJECTORY_SCHEMA:
        values = [row.get(field.name) for row in rows]
        if field.type != _VECTORS:
            columns.append(pa.array(values, type=field.type))
            continue
        # Pack each (N, 3) column in native buffers. Nested tolist() would
        # create a Python object for every coordinate of every buffered frame.
        present = [np.asarray(value) for value in values if value is not None]
        if not present:
            columns.append(pa.nulls(len(rows), type=_VECTORS))
            continue
        coordinates = np.concatenate([value.reshape(-1, 3) for value in present])
        vectors = pa.ListArray.from_arrays(
            np.arange(0, coordinates.size + 1, 3),
            pa.array(coordinates.ravel(), type=pa.float64()),
        )
        lengths = [0 if value is None else len(value) for value in values]
        columns.append(
            pa.ListArray.from_arrays(
                np.cumsum([0, *lengths]),
                vectors,
                mask=pa.array([value is None for value in values]),
            )
        )
    return pa.Table.from_arrays(columns, schema=_TRAJECTORY_SCHEMA)


class ParquetTrajectoryWriter:
    """Buffer row groups and seal an immutable Parquet part at each checkpoint.

    The public output is a single ordinary Parquet file. A checkpoint identifies
    exactly which parts belong to its history, even after a later restart.
    """

    def __init__(self, path: Path | str, flush_interval: int = 1000):
        self.path = Path(path)
        self.flush_interval = flush_interval
        self.total_frames = 0
        self.buffer: list[dict] = []
        self._directory = self.path.with_name(f".{self.path.name}.parts")
        self._id = uuid4().hex
        self._parts: list[dict] = []
        self._writer = None
        self._part = None
        self._closed = False

    def resume(self, step: int, state: dict | None) -> None:
        """Select the committed prefix, or import a pre-versioned trajectory."""
        if state is not None:
            if state["version"] != 1:
                raise ValueError("Unsupported trajectory checkpoint version")
            parts = state["parts"]
            if any(
                Path(part["name"]).name != part["name"] or part["last_step"] > step
                for part in parts
            ):
                raise ValueError("Invalid trajectory checkpoint parts")
            same_output = str(self.path.resolve()) == state["output"]
            if same_output or (parts and (self._directory / parts[0]["name"]).exists()):
                previous_last = None
                for part in parts:
                    path = self._directory / part["name"]
                    with pq.ParquetFile(path) as parquet:
                        metadata = parquet.metadata
                        first = metadata.row_group(0).column(0).statistics.min
                        last = (
                            metadata.row_group(metadata.num_row_groups - 1)
                            .column(0)
                            .statistics.max
                        )
                    if (
                        part["rows"] != metadata.num_rows
                        or part["first_step"] != first
                        or part["last_step"] != last
                        or (previous_last is not None and first <= previous_last)
                    ):
                        raise ValueError(
                            f"Trajectory part {path} does not match its checkpoint index"
                        )
                    previous_last = last
                self._parts = parts.copy()
                self._id = state["id"]
                self.total_frames = sum(part["rows"] for part in parts)
                return
            if self.path.exists():
                raise ValueError(
                    f"Existing trajectory {self.path} belongs to a different run"
                )
        elif self.path.exists():
            with pq.ParquetFile(self.path) as previous:
                for batch in previous.iter_batches(batch_size=128):
                    rows = [row for row in batch.to_pylist() if row["step"] <= step]
                    if rows:
                        self._write_rows(rows)
            self._seal_part()

    def append(self, frame: TrajectoryFrame) -> None:
        if self._closed:
            raise ValueError("Cannot append to a closed trajectory")
        self.buffer.append(frame._snapshot(as_lists=False))
        if len(self.buffer) >= self.flush_interval:
            self.flush()

    def _write_rows(self, rows):
        first, last = rows[0]["step"], rows[-1]["step"]
        previous = self._part or (self._parts[-1] if self._parts else None)
        if previous and first <= previous["last_step"]:
            raise ValueError(f"Trajectory step {first} follows {previous['last_step']}")
        if any(a["step"] >= b["step"] for a, b in pairwise(rows)):
            raise ValueError("Trajectory steps must increase")
        table = _trajectory_table(rows)
        if self._writer is None:
            self._directory.mkdir(exist_ok=True)
            name = f"{self._id}-{uuid4().hex}.parquet"
            self._writer = pq.ParquetWriter(
                self._directory / name, _TRAJECTORY_SCHEMA, compression="zstd"
            )
            self._part = {"name": name, "first_step": first, "rows": 0}
        self._writer.write_table(table)
        self._part["last_step"] = last
        self._part["rows"] += len(rows)
        self.total_frames += len(rows)

    def flush(self) -> None:
        if self.buffer:
            self._write_rows(self.buffer)
            self.buffer.clear()

    def _seal_part(self) -> None:
        self.flush()
        if self._writer is None:
            return
        self._writer.close()
        with (self._directory / self._part["name"]).open("rb") as part:
            os.fsync(part.fileno())
        _sync_directory(self._directory)
        _sync_directory(self.path.parent)
        self._parts.append(self._part)
        self._writer = self._part = None

    def checkpoint(self) -> dict:
        self._seal_part()
        return {
            "version": 1,
            "id": self._id,
            "output": str(self.path.resolve()),
            "parts": self._parts.copy(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._seal_part()
        if self._parts:
            temporary = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
            try:
                if len(self._parts) == 1:
                    source = self._directory / self._parts[0]["name"]
                    try:
                        os.link(source, temporary)
                    except OSError:
                        shutil.copyfile(source, temporary)
                else:
                    with pq.ParquetWriter(
                        temporary, _TRAJECTORY_SCHEMA, compression="zstd"
                    ) as output:
                        for part in self._parts:
                            with pq.ParquetFile(
                                self._directory / part["name"]
                            ) as parquet:
                                for batch in parquet.iter_batches(batch_size=128):
                                    output.write_batch(batch)
                # Keep a complete index even when a fresh run reuses this path.
                manifest = self._directory / f"{self._id}-{uuid4().hex}.json"
                with manifest.open("x", encoding="utf-8") as file:
                    json.dump(self.checkpoint(), file)
                    file.flush()
                    os.fsync(file.fileno())
                _sync_directory(self._directory)
                with temporary.open("rb") as file:
                    os.fsync(file.fileno())
                os.replace(temporary, self.path)
                _sync_directory(self.path.parent)
            finally:
                temporary.unlink(missing_ok=True)
        self._closed = True

    def abort(self) -> None:
        """Discard the uncommitted tail and keep all published output readable."""
        self.buffer.clear()
        if self._writer is not None:
            self._writer.close()
            (self._directory / self._part["name"]).unlink(missing_ok=True)
            self._writer = self._part = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

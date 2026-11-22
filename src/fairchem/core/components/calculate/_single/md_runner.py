"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import ase.io
import ase.units
from ase import Atoms
from ase.constraints import (
    FixBondLengths,
    FixInternals,
    FixLinearTriatomic,
    dict2constraint,
)
from ase.io.jsonio import decode, encode
from ase.md import MDLogger
from monty.json import jsanitize

from fairchem.core.components.calculate._calculate_runner import CalculateRunner
from fairchem.core.components.calculate.simulation_tools.initialization import (
    initialize_momenta,
)
from fairchem.core.components.calculate.simulation_tools.trajectory import (
    ParquetTrajectoryWriter,
    TrajectoryFrame,
)
from fairchem.core.components.runner import (
    PreemptableMixin,
    StopfairDetected,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ase.calculators.calculator import Calculator
    from ase.md.md import MolecularDynamics
    from omegaconf import DictConfig

    from fairchem.core.components.calculate.simulation_tools.thermostats import (
        Thermostat,
    )


# ASE builds these arrays from the original bond lengths and masses on the
# first step. Reinitializing from rounded, constrained positions changes them.
_TRIATOMIC_ARRAYS = (
    "bondlengths",
    "bondlengths_nm",
    "mass_n",
    "mass_m",
    "mass_o",
    "C1",
    "C2",
    "C3",
    "C4",
)


def _write_extxyz(atoms: Atoms, path: Path, write_results: bool) -> None:
    # ExtXYZ supports one- or two-dimensional arrays of a limited set of
    # scalar types. The checkpoint JSON retains the complete atom state.
    view = copy(atoms)
    view.arrays = {
        key: array
        for key, array in atoms.arrays.items()
        if key != "move_mask" and array.ndim <= 2 and array.dtype.kind in "fiOSUb"
    }
    ase.io.write(path, view, format="extxyz", write_results=write_results)


def _save_atoms(atoms: Atoms, path: Path) -> None:
    state = atoms.todict()
    constraints = []
    for constraint in atoms.constraints:
        try:
            descriptor = constraint.todict()
            if isinstance(constraint, FixBondLengths):
                # ASE omits the original target lengths, which can differ from
                # the last constrained positions by the solver tolerance.
                descriptor["kwargs"]["bondlengths"] = constraint.bondlengths
            elif (
                isinstance(constraint, FixLinearTriatomic)
                and constraint.bondlengths is not None
            ):
                descriptor["arrays"] = {
                    name: getattr(constraint, name) for name in _TRIATOMIC_ARRAYS
                }
            elif isinstance(constraint, FixInternals) and constraint.initialized:
                targets = iter(sub.targetvalue for sub in constraint.constraints)
                for key in ("bonds", "angles_deg", "dihedrals_deg", "bondcombos"):
                    descriptor["kwargs"][key] = [
                        [next(targets), indices]
                        for _, indices in descriptor["kwargs"][key]
                    ]
            # Some ASE constructors (e.g. FixSymmetry) modify their arguments.
            restored = dict2constraint(decode(encode(descriptor), always_array=False))
            if type(restored) is not type(constraint):
                raise ValueError("constraint type changed during serialization")
        except Exception as exc:
            raise ValueError(
                f"Cannot checkpoint constraint {type(constraint).__name__}"
            ) from exc
        constraints.append(descriptor)
    state["constraints"] = constraints
    path.write_text(encode(state), encoding="utf-8")


class MDRunner(PreemptableMixin, CalculateRunner):
    """
    General-purpose molecular dynamics runner for single structures.

    This class provides a flexible framework for running MD simulations with any ASE
    dynamics integrator and any trajectory writer.
    """

    result_glob_pattern: ClassVar[str] = "trajectory*.*"

    def __init__(
        self,
        calculator: Calculator,
        thermostat: Thermostat,
        atoms: Atoms | None = None,
        timestep_fs: float = 1.0,
        steps: int = 1000,
        trajectory_interval: int = 1,
        log_interval: int = 10,
        checkpoint_interval: int | None = None,
        heartbeat_interval: int | None = None,
        trajectory_writer: Callable[
            ..., ParquetTrajectoryWriter
        ] = ParquetTrajectoryWriter,
        velocity_seed: int | None = None,
        initialization_temperature_K: float | None = None,
        remove_center_of_mass_momentum: bool = True,
    ):
        """
        Initialize the MDRunner for single-structure MD.

        Args:
            calculator: ASE calculator for energy/force calculations
            thermostat: Thermostat dataclass that knows how to build its ASE
                dynamics object, save state, and restore state. One of
                VelocityVerletThermostat, NoseHooverNVT, BussiThermostat,
                or LangevinThermostat.
            atoms: Single Atoms object to run MD on
            timestep_fs: MD timestep in femtoseconds.
            steps: Total number of MD steps to run
            trajectory_interval: Interval for writing trajectory frames
            log_interval: Interval for writing thermodynamic data to log
            checkpoint_interval: Interval (in steps) for saving a rolling
                checkpoint of the simulation state. If None, no periodic
                checkpointing is performed.
            heartbeat_interval: Interval (in steps) for checking for a
                STOPFAIR file in run_dir. If a STOPFAIR file is found, the
                simulation saves state and stops gracefully. If None, no
                STOPFAIR checking is performed.
            trajectory_writer: Factory or partial for trajectory writer.
                Called as ``trajectory_writer_fn(path)`` to create the writer.
                Defaults to ParquetTrajectoryWriter. Use Hydra
                ``_partial_: true`` in config to bind extra kwargs (e.g.
                flush_interval) while leaving the path argument for runtime.
            velocity_seed: Seed for deterministic initial momentum generation.
            initialization_temperature_K: Temperature used to initialize momenta.
                Both this value and velocity_seed are required to initialize.
            remove_center_of_mass_momentum: Remove net momentum after initialization.
        """
        if trajectory_interval <= 0 or log_interval <= 0:
            raise ValueError("trajectory_interval and log_interval must be positive")
        self._atoms = atoms
        self.thermostat = thermostat
        self.timestep_fs = timestep_fs
        self.steps = steps
        self.trajectory_interval = trajectory_interval
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        self.heartbeat_interval = heartbeat_interval
        self._trajectory_writer_fn = trajectory_writer
        self.velocity_seed = velocity_seed
        self.initialization_temperature_K = initialization_temperature_K
        self.remove_center_of_mass_momentum = remove_center_of_mass_momentum

        # State tracking
        self._dyn: MolecularDynamics | None = None
        self._trajectory_writer: ParquetTrajectoryWriter | None = None
        self._start_step = 0
        self._resuming = False
        self._trajectory_state: dict | None = None
        self._log_state: dict | None = None
        self._log_file: Path | None = None
        self._thermostat_state_to_restore: dict | None = None
        self._elapsed_wall_time: float = 0.0
        self._wall_t0: float | None = None

        super().__init__(calculator=calculator, input_data=[atoms])

    def calculate(self, job_num: int = 0, num_jobs: int = 1) -> dict[str, Any]:
        """
        Run MD simulation on a single structure.

        Args:
            job_num: Current job number (used for file naming)
            num_jobs: Total number of jobs (used for file naming)

        Returns:
            Dictionary containing MD results and metadata
        """
        assert self._atoms is not None, (
            "No atoms provided. Pass atoms= to MDRunner or set runner_state_path "
            "in the config to resume from a checkpoint."
        )

        results_dir = Path(self.job_config.metadata.results_dir)
        sid = self._atoms.info.get("sid", f"{job_num}_{num_jobs}")

        if (
            not self._resuming
            and self._start_step == 0
            and (
                self.velocity_seed is not None
                or self.initialization_temperature_K is not None
            )
        ):
            if self.velocity_seed is None or self.initialization_temperature_K is None:
                raise ValueError(
                    "velocity_seed and initialization_temperature_K must be set "
                    "together."
                )
            initialize_momenta(
                self._atoms,
                self.initialization_temperature_K,
                self.velocity_seed,
                remove_center_of_mass_momentum=self.remove_center_of_mass_momentum,
            )

        # Save the initial atoms before MD begins.
        init_atoms_file = results_dir / "init_atoms.extxyz"
        if not init_atoms_file.exists():
            _write_extxyz(self._atoms, init_atoms_file, write_results=True)

        trajectory_file = results_dir / "trajectory.parquet"
        log_file = results_dir / "thermo.log"

        self._atoms.calc = self.calculator

        if self._thermostat_state_to_restore is None:
            self._dyn = self.thermostat.build(self._atoms, self.timestep_fs)
        elif hasattr(self.thermostat, "build_for_restart"):
            self._dyn = self.thermostat.build_for_restart(
                self._atoms, self.timestep_fs, self._thermostat_state_to_restore
            )
        else:
            self._dyn = self.thermostat.build(self._atoms, self.timestep_fs)
            self.thermostat.restore_state(self._dyn, self._thermostat_state_to_restore)

        self._dyn.nsteps = self._start_step

        self._trajectory_writer = self._trajectory_writer_fn(trajectory_file)
        if self._resuming and hasattr(self._trajectory_writer, "resume"):
            self._trajectory_writer.resume(self._start_step, self._trajectory_state)
        elif self._resuming and trajectory_file.exists():
            raise ValueError(
                "Resuming into existing output requires a restart-aware trajectory writer"
            )

        # ASE schedules observers against the restored global nsteps.
        def collect_frame():
            step = self._dyn.get_number_of_steps()
            if self._resuming and step == self._start_step:
                return
            frame = TrajectoryFrame.from_atoms(
                self._atoms, step=step, time=self._dyn.get_time() / ase.units.fs
            )
            self._trajectory_writer.append(frame)

        self._dyn.attach(collect_frame, interval=self.trajectory_interval)
        self._wall_t0 = time.monotonic()

        # Discard rows from an uncommitted tail before appending a continuation.
        append_log = self._resuming and log_file.exists()
        if (
            append_log
            and self._log_state is not None
            and self._log_state["output"] == str(log_file.resolve())
        ):
            offset = self._log_state["size"]
            with log_file.open("r+b") as file:
                if not 0 <= offset <= os.fstat(file.fileno()).st_size:
                    raise ValueError("Thermodynamic log is shorter than its checkpoint")
                file.truncate(offset)
        logger = MDLogger(
            dyn=self._dyn,
            atoms=self._atoms,
            logfile=str(log_file),
            header=not append_log or log_file.stat().st_size == 0,
            mode="a" if append_log else "w",
        )
        logger.logfile.flush()
        self._log_file = log_file

        def log_step():
            if not self._resuming or self._dyn.nsteps != self._start_step:
                logger()

        self._dyn.attach(log_step, interval=self.log_interval)

        # Attach STOPFAIR checker if heartbeat_interval is configured
        if self.heartbeat_interval is not None and self.heartbeat_interval > 0:

            def check_stopfair_callback():
                if self._dyn.get_number_of_steps() == self._start_step:
                    return
                if self.check_stopfair():
                    self.handle_stopfair()

            self._dyn.attach(check_stopfair_callback, interval=self.heartbeat_interval)

        # Attach periodic checkpoint saving if checkpoint_interval is configured
        if self.checkpoint_interval is not None and self.checkpoint_interval > 0:
            checkpoint_save_path = self.job_config.metadata.preemption_checkpoint_dir

            def save_periodic_checkpoint():
                if self._dyn.get_number_of_steps() == self._start_step:
                    return
                current_step = self._dyn.get_number_of_steps()
                logging.info(
                    f"Saving periodic checkpoint at step {current_step} "
                    f"to {checkpoint_save_path}"
                )
                if not self.save_state(checkpoint_save_path, is_preemption=False):
                    raise OSError(
                        f"Failed to save periodic checkpoint at step {current_step}"
                    )

            self._dyn.attach(
                save_periodic_checkpoint, interval=self.checkpoint_interval
            )

        remaining_steps = self.steps - self._start_step
        stopped_by_stopfair = False

        try:
            self._dyn.run(remaining_steps)
        except StopfairDetected:
            stopped_by_stopfair = True
        except BaseException:
            if hasattr(self._trajectory_writer, "abort"):
                self._trajectory_writer.abort()
            else:
                self._trajectory_writer.close()
            raise
        finally:
            self._elapsed_wall_time += time.monotonic() - self._wall_t0
            self._wall_t0 = None
            logger.close()
        if not stopped_by_stopfair:
            self._trajectory_writer.close()

        return {
            "trajectory_file": trajectory_file,
            "log_file": log_file,
            "total_steps": self._dyn.get_number_of_steps(),
            "target_steps": self.steps,
            "start_step": self._start_step,
            "structure_id": sid,
            "stopped_by_stopfair": stopped_by_stopfair,
            "elapsed_time_s": self._elapsed_wall_time,
        }

    def write_results(
        self,
        results: dict[str, Any],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """
        Write results metadata after simulation completes.

        Trajectory data is already written during simulation.
        This method writes summary metadata as JSON.

        Args:
            results: Dictionary containing results from calculate()
            results_dir: Directory path where results are saved
            job_num: Index of the current job
            num_jobs: Total number of jobs
        """
        trajectory_file = Path(results["trajectory_file"])
        log_file = Path(results["log_file"])

        if not trajectory_file.exists():
            return
        if not log_file.exists():
            return

        metadata = jsanitize(
            {
                "trajectory_file": trajectory_file,
                "log_file": log_file,
                "total_steps": results["total_steps"],
                "target_steps": results["target_steps"],
                "stopped_by_stopfair": results["stopped_by_stopfair"],
                "trajectory_interval": self.trajectory_interval,
                "log_interval": self.log_interval,
                "thermostat_class": type(self.thermostat).__name__,
                "structure_id": results["structure_id"],
                "elapsed_time_s": results["elapsed_time_s"],
            }
        )

        metadata_file = Path(results_dir) / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _modify_resume_config(self, cfg: DictConfig) -> DictConfig:
        if "atoms" in cfg.get("runner", {}):
            del cfg.runner.atoms
        return cfg

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        saved = super().save_state(checkpoint_location, is_preemption)
        if saved and is_preemption and self._trajectory_writer:
            try:
                self._trajectory_writer.close()
            except Exception:
                logging.exception("Failed to finalize the preemption trajectory")
                return False
        return saved

    def save_simulation_state(self, checkpoint_dir: Path, is_preemption: bool) -> None:
        """
        Save MD-specific state files into checkpoint_dir.

        Saves:
        - Lossless ASE atom state and an ExtXYZ compatibility view
        - Thermostat/barostat state in JSON format
        - MD metadata (step count, etc.) in JSON format

        Args:
            checkpoint_dir: Directory to write state files into.
            is_preemption: Whether this save is due to preemption.
        """
        if self._dyn is None or self._atoms is None:
            raise RuntimeError("Cannot checkpoint before MD dynamics are initialized")

        # Snapshot bookkeeping must not change calculator inputs on the live atoms.
        checkpoint_atoms = copy(self._atoms)
        checkpoint_atoms.info = {**self._atoms.info, "md_step": self._dyn.nsteps}
        _save_atoms(checkpoint_atoms, checkpoint_dir / "atoms.json")
        _write_extxyz(
            checkpoint_atoms, checkpoint_dir / "checkpoint.extxyz", write_results=False
        )

        thermostat_state = self.thermostat.save_state(self._dyn)
        thermostat_path = checkpoint_dir / "thermostat_state.json"
        with open(thermostat_path, "w") as f:
            json.dump(thermostat_state, f)

        trajectory_state = None
        if self._trajectory_writer:
            if hasattr(self._trajectory_writer, "checkpoint"):
                trajectory_state = self._trajectory_writer.checkpoint()
            elif hasattr(self._trajectory_writer, "flush"):
                self._trajectory_writer.flush()

        log_state = None
        if self._log_file is not None:
            with self._log_file.open("rb") as file:
                os.fsync(file.fileno())
                log_state = {
                    "output": str(self._log_file.resolve()),
                    "size": os.fstat(file.fileno()).st_size,
                }

        md_state = {
            "version": 1,
            "timestep_fs": self.timestep_fs,
            "thermostat_parameters": {
                key: getattr(self.thermostat, key)
                for key in (
                    "temperature_K",
                    "friction_per_fs",
                    "tdamp_fs",
                    "taut_fs",
                    "taup_fs",
                    "pressure_bar",
                    "compressibility_bar",
                )
                if hasattr(self.thermostat, key)
            },
            "trajectory": trajectory_state,
            "thermo_log": log_state,
            "current_step": self._dyn.get_number_of_steps(),
            "total_steps": self._dyn.get_number_of_steps(),
            "target_steps": self.steps,
            "elapsed_wall_time": (
                self._elapsed_wall_time
                + (time.monotonic() - self._wall_t0 if self._wall_t0 is not None else 0)
            ),
        }
        state_path = checkpoint_dir / "md_state.json"
        with open(state_path, "w") as f:
            json.dump(md_state, f)

    def load_state(self, checkpoint_location: str | None) -> None:
        """
        Load MD state from checkpoint.

        Delegates to PreemptableRunner.load_state which calls
        load_simulation_state.

        Args:
            checkpoint_location: Directory containing checkpoint files, or None.
        """
        PreemptableMixin.load_state(self, checkpoint_location)

    def load_simulation_state(self, checkpoint_dir: Path) -> None:
        """
        Load MD-specific state from checkpoint_dir.

        Restores:
        - Atoms arrays and constraints from the versioned state or legacy ExtXYZ
        - Starting step count from metadata
        - Thermostat/barostat state (applied after dynamics creation)

        Args:
            checkpoint_dir: Directory containing checkpoint files.
        """
        atoms_path = checkpoint_dir / "checkpoint.extxyz"
        state_path = checkpoint_dir / "md_state.json"

        if not state_path.exists():
            if atoms_path.exists():
                raise FileNotFoundError(state_path)
            return
        with open(state_path) as f:
            md_state = json.load(f)

        version = md_state.get("version", 0)
        if version == 1:
            state = decode(
                (checkpoint_dir / "atoms.json").read_text(encoding="utf-8"),
                always_array=False,
            )
            self._atoms = Atoms.fromdict(state)
            for constraint, descriptor in zip(
                self._atoms.constraints, state["constraints"], strict=True
            ):
                if (
                    isinstance(constraint, FixLinearTriatomic)
                    and "arrays" in descriptor
                ):
                    for name in _TRIATOMIC_ARRAYS:
                        setattr(constraint, name, descriptor["arrays"][name])
            # ASE's constructor normalizes the dtype of these two arrays.
            self._atoms.arrays.update(
                {key: state[key] for key in ("numbers", "positions")}
            )
            if self.timestep_fs != md_state["timestep_fs"]:
                raise ValueError("MD timestep differs from saved checkpoint")
            parameters = md_state.get("thermostat_parameters") or {}
            for key, value in parameters.items():
                if getattr(self.thermostat, key, value) != value:
                    raise ValueError(
                        f"Thermostat parameter {key} differs from saved checkpoint"
                    )
        elif version == 0:
            self._atoms = ase.io.read(str(atoms_path), format="extxyz")
        else:
            raise ValueError(f"Unsupported MD checkpoint version {version}")
        self._trajectory_state = md_state.get("trajectory")
        self._log_state = md_state.get("thermo_log")
        self._resuming = True
        self._start_step = md_state["current_step"]
        self._elapsed_wall_time = md_state.get("elapsed_wall_time", 0.0)

        # A final periodic checkpoint can precede publication of the trajectory.
        # Reopen its immutable parts and finish publication without advancing MD.
        if self._start_step > self.steps or (
            self._start_step == self.steps and not self._trajectory_state
        ):
            logging.warning(
                f"Checkpoint step ({self._start_step}) is already at or beyond "
                f"configured total steps ({self.steps}). Stopping immediately."
            )
            self._already_calculated = True
            return

        thermostat_path = checkpoint_dir / "thermostat_state.json"
        if thermostat_path.exists():
            with open(thermostat_path) as f:
                self._thermostat_state_to_restore = json.load(f)

            checkpoint_thermostat = self._thermostat_state_to_restore.get("class_name")
            current_thermostat = type(self.thermostat).__name__
            if checkpoint_thermostat and checkpoint_thermostat != current_thermostat:
                raise ValueError(
                    f"Thermostat mismatch: checkpoint was saved with "
                    f"{checkpoint_thermostat} but current config uses "
                    f"{current_thermostat}."
                )
        elif version == 1:
            raise FileNotFoundError(thermostat_path)
        else:
            self._thermostat_state_to_restore = None

        logging.info(
            f"Loaded MD checkpoint from {checkpoint_dir}, resuming from step {self._start_step}"
        )

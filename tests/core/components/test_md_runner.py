"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import ase.io
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import (
    FixAtoms,
    FixBondLengths,
    FixCom,
    FixInternals,
    FixLinearTriatomic,
    FixSubsetCom,
)
from ase.io import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from fairchem.core.components import runner as runner_module
from fairchem.core.components.calculate import (
    BerendsenNPT,
    BussiThermostat,
    LangevinThermostat,
    MDRunner,
    NoseHooverNVT,
    ParquetTrajectoryWriter,
    TrajectoryFrame,
    VelocityVerletThermostat,
)
from fairchem.core.components.calculate.simulation_tools.initialization import (
    initialize_momenta,
)
from fairchem.core.datasets.common_structures import get_slab_adsorbate


@dataclass
class MockMetadata:
    results_dir: str
    checkpoint_dir: str = ""
    preemption_checkpoint_dir: str = ""
    config_path: str = ""
    array_job_num: int = 0


@dataclass
class MockScheduler:
    num_array_jobs: int = 1


@dataclass
class MockJobConfig:
    metadata: MockMetadata
    scheduler: MockScheduler


def _create_mock_job_config(
    results_dir: str,
    checkpoint_dir: str = "",
    preemption_checkpoint_dir: str = "",
) -> MockJobConfig:
    return MockJobConfig(
        metadata=MockMetadata(
            results_dir=results_dir,
            checkpoint_dir=checkpoint_dir,
            preemption_checkpoint_dir=preemption_checkpoint_dir or checkpoint_dir,
        ),
        scheduler=MockScheduler(num_array_jobs=1),
    )


@pytest.fixture()
def cu_atoms():
    atoms = bulk("Cu", cubic=True) * (2, 2, 2)
    atoms.info["sid"] = 123
    np.random.seed(42)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    return atoms


@pytest.fixture()
def results_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMDRunner:
    def test_initialize_momenta_is_reproducible(self):
        """
        Initial momenta use the requested seed and have no center-of-mass drift.
        """
        atoms1 = bulk("Cu", cubic=True) * (2, 2, 2)
        atoms2 = atoms1.copy()

        initialize_momenta(atoms1, temperature_K=300.0, seed=17)
        initialize_momenta(atoms2, temperature_K=300.0, seed=17)

        npt.assert_array_equal(atoms1.get_momenta(), atoms2.get_momenta())
        npt.assert_allclose(atoms1.get_momenta().sum(axis=0), 0.0, atol=1e-12)
        assert atoms1.info["velocity_seed"] == 17
        assert atoms1.info["initial_temperature_K"] == 300.0

    def test_initialize_momenta_rejects_single_mobile_atom(self):
        atoms = Atoms("Cu")

        with pytest.raises(ValueError, match="at least two mobile atoms"):
            initialize_momenta(atoms, temperature_K=300.0, seed=17)

        initialize_momenta(
            atoms,
            temperature_K=300.0,
            seed=17,
            remove_center_of_mass_momentum=False,
        )
        assert np.isfinite(atoms.get_momenta()).all()
        assert not np.allclose(atoms.get_momenta(), 0.0)

    def test_initialize_momenta_removes_mobile_subset_momentum(self):
        atoms = bulk("Cu", cubic=True)
        fixed = np.array([0, 1])
        mobile = np.array([2, 3])
        atoms.set_constraint(FixAtoms(indices=fixed))

        initialize_momenta(atoms, temperature_K=300.0, seed=17)

        momenta = atoms.get_momenta()
        npt.assert_array_equal(momenta[fixed], 0.0)
        npt.assert_allclose(momenta[mobile].sum(axis=0), 0.0, atol=1e-12)
        assert np.isfinite(momenta).all()

    @pytest.mark.parametrize(
        ("velocity_seed", "initialization_temperature_K"),
        [(17, None), (None, 300.0)],
    )
    def test_initialization_requires_seed_and_temperature(
        self,
        cu_atoms,
        results_dir,
        velocity_seed,
        initialization_temperature_K,
    ):
        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            thermostat=VelocityVerletThermostat(),
            steps=0,
            velocity_seed=velocity_seed,
            initialization_temperature_K=initialization_temperature_K,
        )
        runner._job_config = _create_mock_job_config(str(results_dir))

        with pytest.raises(
            ValueError,
            match="velocity_seed and initialization_temperature_K must be set together",
        ):
            runner.calculate()

    def test_initialization_is_not_repeated_after_resume(self, results_dir):
        atoms = bulk("Cu", cubic=True) * (2, 2, 2)
        atoms.set_momenta(np.zeros((len(atoms), 3)))
        runner = MDRunner(
            calculator=EMT(),
            atoms=atoms,
            thermostat=VelocityVerletThermostat(),
            steps=1,
            velocity_seed=17,
            initialization_temperature_K=300.0,
        )
        runner._start_step = 1
        runner._job_config = _create_mock_job_config(str(results_dir))

        runner.calculate()

        npt.assert_array_equal(runner._atoms.get_momenta(), np.zeros((len(atoms), 3)))

    def test_md_correctness_vs_ase(self, cu_atoms, results_dir):
        """
        Verify MDRunner produces identical trajectories to plain ASE.
        """
        atoms_mdrunner = cu_atoms.copy()
        atoms_ase = cu_atoms.copy()

        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_mdrunner, temperature_K=300)
        np.random.seed(42)
        MaxwellBoltzmannDistribution(atoms_ase, temperature_K=300)

        steps, interval = 20, 5

        mdrunner_dir = results_dir / "mdrunner"
        mdrunner_dir.mkdir()
        runner = MDRunner(
            calculator=EMT(),
            atoms=atoms_mdrunner,
            thermostat=VelocityVerletThermostat(),
            timestep_fs=1.0,
            steps=steps,
            trajectory_interval=interval,
            log_interval=10,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=100),
        )
        runner._job_config = _create_mock_job_config(str(mdrunner_dir))
        results = runner.calculate(job_num=0, num_jobs=1)

        # Reference ASE run
        ase_traj_file = results_dir / "ase_traj.traj"
        atoms_ase.calc = EMT()
        dyn_ase = VelocityVerlet(atoms_ase, timestep=1.0 * units.fs)
        traj_ase = Trajectory(str(ase_traj_file), "w", atoms_ase)
        dyn_ase.attach(traj_ase.write, interval=interval)
        dyn_ase.run(steps)
        traj_ase.close()

        traj_df = pd.read_parquet(results["trajectory_file"])
        ase_frames = Trajectory(str(ase_traj_file), "r")
        assert len(traj_df) == len(ase_frames)
        assert (mdrunner_dir / "init_atoms.extxyz").exists()

        for i, ase_atoms in enumerate(ase_frames):
            row = traj_df.iloc[i]
            npt.assert_allclose(
                np.vstack(row["positions"]), ase_atoms.get_positions(), atol=1e-10
            )
            npt.assert_allclose(
                np.vstack(row["velocities"]), ase_atoms.get_velocities(), atol=1e-10
            )
            npt.assert_allclose(
                row["energy"], ase_atoms.get_potential_energy(), atol=1e-10
            )

    def test_langevin_fix_com_uses_mobile_subset(self):
        atoms = bulk("Cu", cubic=True)
        atoms.set_constraint(FixAtoms(indices=[0, 1]))
        thermostat = LangevinThermostat(
            temperature_K=300.0,
            friction_per_fs=0.01,
            use_fix_com_constraint=True,
        )

        dynamics = thermostat.build(atoms, timestep_fs=1.0)

        assert dynamics.fix_com is False
        constraints = [
            constraint
            for constraint in atoms.constraints
            if isinstance(constraint, FixCom)
        ]
        assert len(constraints) == 1
        assert isinstance(constraints[0], FixSubsetCom)
        npt.assert_array_equal(constraints[0].get_indices(), [2, 3])

    @pytest.mark.parametrize(
        "thermostat",
        [
            VelocityVerletThermostat(),
            NoseHooverNVT(temperature_K=300.0, tdamp_fs=25.0),
            BussiThermostat(temperature_K=300.0, taut_fs=25.0),
            LangevinThermostat(temperature_K=300.0, friction_per_fs=0.01),
            BerendsenNPT(
                temperature_K=300.0,
                pressure_bar=1.0,
                taut_fs=500.0,
                taup_fs=1000.0,
                compressibility_bar=1.0 / 140e9,
            ),
        ],
        ids=["VelocityVerlet", "NoseHoover", "Bussi", "Langevin", "BerendsenNPT"],
    )
    @pytest.mark.parametrize(
        "interrupt_at_step",
        [36, 40],
        ids=["non-aligned", "aligned"],
    )
    def test_checkpoint_resume(
        self, cu_atoms, results_dir, thermostat, interrupt_at_step
    ):
        """
        Checkpoint at both a non-aligned and an interval-aligned step, resume,
        and verify the concatenated trajectory stays on a uniform grid with no
        duplicated frames across the restart boundary (regardless of where
        preemption lands relative to trajectory_interval).
        """
        results_dir1 = results_dir / "results1"
        results_dir2 = results_dir / "results2"
        checkpoint_dir = results_dir / "checkpoint"
        results_dir1.mkdir()
        results_dir2.mkdir()

        trajectory_interval = 10
        total_steps = 100

        runner1 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            thermostat=thermostat,
            timestep_fs=1.0,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            log_interval=10,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=1000),
        )
        runner1._job_config = _create_mock_job_config(str(results_dir1))

        class SimulatedInterrupt(Exception):
            pass

        def interrupt_callback():
            if runner1._dyn.get_number_of_steps() >= interrupt_at_step:
                raise SimulatedInterrupt

        # Run 1: manually drive dynamics so we can attach an interrupt
        try:
            runner1._atoms.calc = runner1.calculator
            runner1._dyn = thermostat.build(runner1._atoms, timestep_fs=1.0)
            parquet_file1 = results_dir1 / "trajectory.parquet"
            runner1._trajectory_writer = ParquetTrajectoryWriter(
                parquet_file1, flush_interval=1000
            )

            def collect_frame():
                step = runner1._dyn.get_number_of_steps()
                if step % trajectory_interval == 0:
                    frame = TrajectoryFrame.from_atoms(
                        runner1._atoms, step=step, time=runner1._dyn.get_time()
                    )
                    runner1._trajectory_writer.append(frame)

            runner1._dyn.attach(collect_frame, interval=1)
            runner1._dyn.attach(interrupt_callback, interval=1)
            runner1._dyn.run(total_steps)
        except SimulatedInterrupt:
            final_positions = runner1._atoms.get_positions().copy()
            final_velocities = runner1._atoms.get_velocities().copy()
            runner1.save_state(str(checkpoint_dir), is_preemption=True)

        df1 = pd.read_parquet(parquet_file1)
        steps1 = list(df1["step"])
        # Run 1 writes every interval-aligned step up to (and including) the
        # interrupt step if that step is itself aligned.
        expected_steps1 = list(range(0, interrupt_at_step + 1, trajectory_interval))
        assert steps1 == expected_steps1

        # Verify checkpoint files
        assert (checkpoint_dir / "checkpoint.extxyz").exists()
        assert (checkpoint_dir / "thermostat_state.json").exists()
        checkpoint_atoms = ase.io.read(str(checkpoint_dir / "checkpoint.extxyz"))
        assert checkpoint_atoms.info["md_step"] == interrupt_at_step

        # Run 2: resume from checkpoint
        runner2 = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            thermostat=thermostat,
            timestep_fs=1.0,
            steps=total_steps,
            trajectory_interval=trajectory_interval,
            log_interval=10,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=1000),
        )
        runner2._job_config = _create_mock_job_config(str(results_dir2))
        runner2.load_state(str(checkpoint_dir))

        assert runner2._start_step == interrupt_at_step
        npt.assert_allclose(runner2._atoms.get_positions(), final_positions, atol=1e-8)
        npt.assert_allclose(
            runner2._atoms.get_velocities(), final_velocities, atol=1e-8
        )

        results2 = runner2.calculate(job_num=0, num_jobs=1)
        runner2.write_results(results2, str(results_dir2), job_num=0, num_jobs=1)
        df2 = pd.read_parquet(results2["trajectory_file"])
        steps2 = list(df2["step"])

        # No frame is written at the resume step, so the concatenated
        # trajectory stays on a uniform grid with no duplicates, whether or not
        # the checkpoint step was interval-aligned.
        assert len(steps2) == len(set(steps2))
        all_steps = sorted(steps1 + steps2)
        assert len(all_steps) == len(
            set(all_steps)
        ), f"Duplicate frame(s) across restart boundary: {all_steps}"
        expected = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        assert all_steps == expected, f"Expected {expected}, got {all_steps}"

        # Verify write_results produced valid metadata JSON
        metadata_file = results_dir2 / "metadata.json"
        assert metadata_file.exists()
        with open(metadata_file) as f:
            metadata = json.load(f)
        assert metadata["structure_id"] == cu_atoms.info["sid"]
        assert metadata["total_steps"] == total_steps

    def test_load_state_beyond_total_steps(self, cu_atoms, results_dir):
        """
        When checkpoint step >= total steps, load_state should warn
        and the simulation should end immediately without error.
        """
        checkpoint_dir = results_dir / "checkpoint"
        checkpoint_dir.mkdir()
        md_results_dir = results_dir / "results"
        md_results_dir.mkdir()

        # Create a fake checkpoint with current_step beyond what we'll configure
        atoms = cu_atoms.copy()
        atoms.info["md_step"] = 50
        ase.io.write(str(checkpoint_dir / "checkpoint.extxyz"), atoms, format="extxyz")

        with open(checkpoint_dir / "md_state.json", "w") as f:
            json.dump({"current_step": 50, "total_steps": 100}, f)

        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            thermostat=VelocityVerletThermostat(),
            timestep_fs=1.0,
            steps=30,  # less than checkpoint step of 50
            trajectory_interval=10,
            log_interval=10,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=1000),
        )
        runner._job_config = _create_mock_job_config(str(md_results_dir))

        # load_state should not raise, just warn
        runner.load_state(str(checkpoint_dir))
        assert runner._start_step == 50
        assert runner._already_calculated

    def test_stopfair_graceful_stop(self, cu_atoms, results_dir):
        """
        Verify STOPFAIR triggers graceful stop, saves state, and deletes
        the sentinel file.
        """
        md_results_dir = results_dir / "results"
        checkpoint_dir = results_dir / "checkpoints"
        md_results_dir.mkdir()
        checkpoint_dir.mkdir()

        runner = MDRunner(
            calculator=EMT(),
            atoms=cu_atoms.copy(),
            thermostat=NoseHooverNVT(temperature_K=300.0, tdamp_fs=25.0),
            timestep_fs=1.0,
            steps=100,
            trajectory_interval=10,
            heartbeat_interval=20,
            log_interval=10,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=1000),
        )
        runner._job_config = _create_mock_job_config(
            str(md_results_dir), checkpoint_dir=str(checkpoint_dir)
        )

        stopfair_path = checkpoint_dir.parent / "STOPFAIR"
        stopfair_path.write_text("")

        results = runner.calculate(job_num=0, num_jobs=1)

        assert results["stopped_by_stopfair"] is True
        assert results["total_steps"] == 20
        assert results["target_steps"] == 100
        assert (checkpoint_dir / "checkpoint.extxyz").exists()
        assert (checkpoint_dir / "md_state.json").exists()
        assert not stopfair_path.exists()

        with open(checkpoint_dir / "md_state.json") as f:
            md_state = json.load(f)
        assert md_state["current_step"] == 20

        traj_df = pd.read_parquet(results["trajectory_file"])
        assert list(traj_df["step"]) == [0, 10, 20]

    def test_npt_cell_changes(self, cu_atoms, results_dir):
        """
        Verify that NPT simulation changes the cell volume.
        """
        md_results_dir = results_dir / "results"
        md_results_dir.mkdir()

        atoms = cu_atoms.copy()
        initial_volume = atoms.get_volume()

        # Use a large pressure to drive a noticeable volume change
        thermostat = BerendsenNPT(
            temperature_K=300.0,
            pressure_bar=1e5,
            taut_fs=100.0,
            taup_fs=100.0,
            compressibility_bar=1.0 / 140e9,
        )

        runner = MDRunner(
            calculator=EMT(),
            atoms=atoms,
            thermostat=thermostat,
            timestep_fs=1.0,
            steps=200,
            trajectory_interval=50,
            log_interval=50,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=1000),
        )
        runner._job_config = _create_mock_job_config(str(md_results_dir))
        runner.calculate(job_num=0, num_jobs=1)

        final_volume = atoms.get_volume()
        assert initial_volume != pytest.approx(final_volume, rel=1e-6)


class TestTrajectoryFrame:
    def test_round_trip_oc20_slab_adsorbate(self, results_dir):
        """
        Run MD on an OC20-style slab+adsorbate and verify the parquet
        trajectory preserves tags, FixAtoms constraints, and charge/spin.
        """
        atoms = get_slab_adsorbate()

        assert np.any(atoms.get_tags() != 0)
        assert len(atoms.constraints) > 0
        assert "charge" in atoms.info
        assert "spin" in atoms.info

        original_tags = atoms.get_tags().copy()
        original_fixed_indices = sorted(atoms.constraints[0].index)

        atoms.calc = EMT()

        md_results_dir = results_dir / "results"
        md_results_dir.mkdir()

        runner = MDRunner(
            calculator=atoms.calc,
            atoms=atoms,
            thermostat=VelocityVerletThermostat(),
            timestep_fs=1.0,
            steps=10,
            trajectory_interval=5,
            log_interval=5,
            trajectory_writer=partial(ParquetTrajectoryWriter, flush_interval=100),
        )
        runner._job_config = _create_mock_job_config(str(md_results_dir))
        results = runner.calculate(job_num=0, num_jobs=1)

        traj_df = pd.read_parquet(results["trajectory_file"])
        assert len(traj_df) > 0

        for _, row in traj_df.iterrows():
            d = row.to_dict()

            # Reconstruct atoms directly from parquet row
            def _to_array(val, dtype=float):
                if isinstance(val, np.ndarray) and val.dtype == object:
                    return np.stack(val).astype(dtype)
                return np.asarray(val, dtype=dtype)

            reconstructed = Atoms(
                numbers=_to_array(d["atomic_numbers"], dtype=int),
                positions=_to_array(d["positions"]),
                cell=_to_array(d["cell"]),
                pbc=_to_array(d["pbc"], dtype=bool),
            )
            if d.get("tags") is not None:
                reconstructed.set_tags(_to_array(d["tags"], dtype=int))
            if d.get("velocities") is not None:
                reconstructed.set_velocities(_to_array(d["velocities"]))
            if d.get("fixed") is not None:
                reconstructed.constraints = [
                    FixAtoms(indices=np.where(_to_array(d["fixed"], dtype=bool))[0])
                ]
            if d.get("charge") is not None:
                reconstructed.info["charge"] = d["charge"]
            if d.get("spin") is not None:
                reconstructed.info["spin"] = d["spin"]

            npt.assert_array_equal(reconstructed.get_tags(), original_tags)

            assert len(reconstructed.constraints) == 1
            assert isinstance(reconstructed.constraints[0], FixAtoms)
            npt.assert_array_equal(
                sorted(reconstructed.constraints[0].index), original_fixed_indices
            )

            assert reconstructed.info["charge"] == atoms.info["charge"]
            assert reconstructed.info["spin"] == atoms.info["spin"]


@pytest.mark.parametrize("interval", [1, 3])
def test_nullable_trajectory_schema(tmp_path, interval):
    atoms = Atoms("H")
    path = tmp_path / "trajectory.parquet"
    with ParquetTrajectoryWriter(path, interval) as writer:
        for step, sid, energy in [(0, None, None), (1, 12, -3.25), (2, "12", None)]:
            frame = TrajectoryFrame.from_atoms(atoms, step)
            frame.sid, frame.energy = sid, energy
            writer.append(frame)
    rows = pq.read_table(path).to_pylist()
    assert [row["energy"] for row in rows] == [None, -3.25, None]
    assert [row["sid"] for row in rows] == [None, "12", "12"]
    assert [row["sid_integer"] for row in rows] == [None, 12, None]
    assert all(row["stress"] is None for row in rows)


class _StrictCellStressEMT(EMT):
    """Model the FAIRChemCalculator stress contract without loading a checkpoint."""

    def get_stress(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        volume = atoms.cell.volume
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Stress requires a cell with finite nonzero volume")
        return super().get_stress(atoms)


@pytest.mark.parametrize(
    ("cell", "pbc"),
    [
        ([0, 0, 0], False),
        ([5, 6, 0], [True, True, False]),
        ([5, 0, 0], [True, False, False]),
        ([[5, 0, 0], [0, 6, 0], [5, 6, 0]], [True, True, False]),
    ],
)
def test_frame_omits_stress_without_physical_volume(cell, pbc):
    class EnergyForceOnly(SinglePointCalculator):
        def get_stress(self, atoms=None):
            raise AssertionError("Must not ask for stress without physical volume")

    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0.75, 0, 0]],
        cell=cell,
        pbc=pbc,
    )
    atoms.calc = EnergyForceOnly(atoms, energy=-1.0, forces=np.zeros((2, 3)))
    frame = TrajectoryFrame.from_atoms(atoms, step=7)
    assert frame.energy is not None
    assert frame.forces.shape == (2, 3)
    assert frame.stress is None
    assert frame.pressure is None
    npt.assert_array_equal(frame.cell, atoms.cell.array)
    npt.assert_array_equal(frame.pbc, atoms.pbc)


@pytest.mark.parametrize("pbc", [False, [True, True, False], True])
def test_frame_keeps_stress_when_physical_volume_exists(pbc):
    class KnownStressEMT(EMT):
        def get_stress(self, atoms=None):
            return np.arange(6, dtype=float)

    atoms = Atoms("Cu", cell=[5, 6, 7], pbc=pbc, calculator=KnownStressEMT())
    frame = TrajectoryFrame.from_atoms(atoms, step=7)
    npt.assert_array_equal(frame.stress, np.arange(6))
    assert frame.pressure == pytest.approx(-1 / units.bar)


def test_frame_propagates_stress_failure_for_valid_cell():
    class BrokenStressEMT(EMT):
        def get_stress(self, atoms=None):
            raise ValueError("stress kernel failed")

    atoms = Atoms("Cu", cell=[5, 6, 7], pbc=True, calculator=BrokenStressEMT())
    with pytest.raises(ValueError, match="stress kernel failed") as error:
        TrajectoryFrame.from_atoms(atoms, step=7)
    assert "stress at MD step 7" in error.value.__notes__[0]


def test_frame_reports_calculation_failure():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.75, 0, 0]], calculator=EMT())
    atoms.info["sid"] = "molecule"
    frame = TrajectoryFrame.from_atoms(atoms, step=7)
    assert frame.stress is None
    assert frame.energy is not None

    class BrokenEMT(EMT):
        def get_forces(self, atoms=None):
            raise RuntimeError("force kernel failed")

    atoms.calc = BrokenEMT()
    with pytest.raises(RuntimeError, match="force kernel failed") as error:
        TrajectoryFrame.from_atoms(atoms, step=7)
    assert "forces at MD step 7, sid='molecule'" in error.value.__notes__[0]


@pytest.mark.parametrize(
    "thermostat",
    [
        BussiThermostat(300, 25, thermostat_seed=17),
        LangevinThermostat(300, 0.01, thermostat_seed=17),
    ],
)
def test_thermostat_rng_isolation(cu_atoms, thermostat):
    first = thermostat.build(cu_atoms.copy(), 1.0)
    other = thermostat.build(cu_atoms.copy(), 1.0)
    state = thermostat.save_state(first)
    assert first.rng is not other.rng
    expected = first.rng.normal(size=4)
    npt.assert_array_equal(other.rng.normal(size=4), expected)
    expected_next = first.rng.normal(size=4)
    thermostat.restore_state(first, state)
    npt.assert_array_equal(first.rng.normal(size=4), expected)
    npt.assert_array_equal(other.rng.normal(size=4), expected_next)


def test_langevin_new_and_legacy_com_modes(cu_atoms):
    thermostat = LangevinThermostat(300, 0.01, thermostat_seed=3)
    dyn = thermostat.build(cu_atoms.copy(), 1.0)
    assert not dyn.fix_com
    assert dyn.atoms.get_number_of_degrees_of_freedom() == 3 * len(cu_atoms) - 3
    state = thermostat.save_state(dyn)
    state.pop("use_fix_com_constraint")
    state["rng_state"].pop("version")
    assert thermostat.build_for_restart(cu_atoms.copy(), 1.0, state).fix_com
    with pytest.raises(ValueError, match="at least two mobile atoms"):
        thermostat.build(Atoms("Cu", constraint=FixCom()), 1.0)


def test_same_directory_restarts_match_constrained_nve(tmp_path):
    atoms = Atoms("H2", positions=[[0.11532198765, 0, 0], [0.879652345678, 0, 0]])
    atoms.set_momenta([[0.012, 0.031, 0], [-0.012, -0.026, 0]])
    atoms.set_constraint(FixBondLengths([[0, 1]]))
    atoms.new_array("labels_int32", np.array([1, 2], dtype=np.int32))
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()

    def run(directory, steps, load=False, interval=None):
        checkpoint = tmp_path / "checkpoint"
        runner = MDRunner(
            _StrictCellStressEMT(),
            VelocityVerletThermostat(),
            atoms.copy(),
            steps=steps,
            timestep_fs=0.5,
            trajectory_interval=1,
            checkpoint_interval=interval,
        )
        runner._job_config = _create_mock_job_config(str(directory), str(checkpoint))
        if load:
            runner.load_state(str(checkpoint))
            assert isinstance(runner._atoms.constraints[0], FixBondLengths)
            assert runner._atoms.arrays["labels_int32"].dtype == np.int32
        return runner, runner.calculate()

    reference, _ = run(reference_dir, 9)
    run(tmp_path, 5, interval=3)  # Published output extends beyond the checkpoint.
    run(tmp_path, 8, load=True, interval=4)
    resumed, result = run(tmp_path, 9, load=True)
    assert result["total_steps"] == result["target_steps"] == 9
    reference_frames = pd.read_parquet(reference_dir / "trajectory.parquet")
    frames = pd.read_parquet(tmp_path / "trajectory.parquet")
    assert list(frames.step) == list(range(10))
    assert frames["stress"].isna().all()
    assert frames["pressure"].isna().all()
    for name in ("energy", "kinetic_energy"):
        npt.assert_array_equal(frames[name], reference_frames[name])
    for name in ("positions", "forces"):
        for actual, expected in zip(frames[name], reference_frames[name], strict=True):
            npt.assert_array_equal(np.vstack(actual), np.vstack(expected))
    npt.assert_array_equal(reference._atoms.get_momenta(), resumed._atoms.get_momenta())
    npt.assert_array_equal(
        reference._atoms.constraints[0].bondlengths,
        resumed._atoms.constraints[0].bondlengths,
    )


def test_stopfair_failed_publication_keeps_old_state(tmp_path, cu_atoms, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    runner = MDRunner(
        EMT(),
        VelocityVerletThermostat(),
        cu_atoms,
        steps=3,
        heartbeat_interval=1,
        checkpoint_interval=1,
    )
    runner._job_config = _create_mock_job_config(str(tmp_path), str(checkpoint))
    runner.calculate()
    previous = (checkpoint / "md_state.json").read_bytes()
    trajectory = (tmp_path / "trajectory.parquet").read_bytes()
    sentinel = tmp_path / "STOPFAIR"
    sentinel.touch()
    real_replace = runner_module.os.replace

    def fail_publish(src, dst):
        if Path(dst).name == ".current":
            raise OSError("injected failure")
        real_replace(src, dst)

    monkeypatch.setattr(runner_module.os, "replace", fail_publish)
    with pytest.raises(OSError, match="Failed to save STOPFAIR"):
        runner.calculate()
    assert sentinel.is_file()
    assert (checkpoint / "md_state.json").read_bytes() == previous
    assert (tmp_path / "trajectory.parquet").read_bytes() == trajectory


@pytest.mark.parametrize(
    "boundary", ["writing", "before_publish", "after_publish", "cleanup"]
)
def test_checkpoint_survives_process_exit(tmp_path, boundary):
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("Fault injection requires POSIX fork")

    class State(runner_module.PreemptableMixin):
        def __init__(self, value):
            self.value = value

        def save_simulation_state(self, directory, is_preemption):
            (directory / "a").write_text(self.value)
            if self.value == "interrupted" and boundary == "writing":
                os._exit(7)
            (directory / "b").write_text(self.value)

        def load_simulation_state(self, directory):
            self.loaded = [(directory / key).read_text() for key in ("a", "b")]

    checkpoint = tmp_path / "cp"
    assert State("first").save_state(str(checkpoint))
    assert State("good").save_state(str(checkpoint))

    def interrupt():
        replace, remove = os.replace, runner_module.shutil.rmtree

        def publish(src, dst):
            if Path(dst).name == ".current" and boundary == "before_publish":
                os._exit(7)
            replace(src, dst)
            if Path(dst).name == ".current" and boundary == "after_publish":
                os._exit(7)

        def cleanup(*args, **kwargs):
            if boundary == "cleanup":
                os._exit(7)
            remove(*args, **kwargs)

        os.replace, runner_module.shutil.rmtree = publish, cleanup
        State("interrupted").save_state(str(checkpoint))

    client = multiprocessing.get_context("fork").Process(target=interrupt)
    client.start()
    client.join(timeout=15)
    assert client.exitcode == 7
    reader = State("reader")
    reader.load_state(str(checkpoint))
    expected = "interrupted" if boundary in {"after_publish", "cleanup"} else "good"
    assert reader.loaded == [expected, expected]
    assert State("recovered").save_state(str(checkpoint))
    assert len(list((checkpoint / ".generations").glob("state-*"))) == 2


def test_resume_legacy_parquet_preserves_prefix_and_optional_columns(tmp_path):
    path = tmp_path / "trajectory.parquet"
    # Legacy schema depends on which properties appeared in the first frame.
    pq.write_table(
        pa.table({"step": [0, 2, 4], "sid": [2, 2, 2], "energy": [None, -2.0, 14.0]}),
        path,
    )
    frame = TrajectoryFrame.from_atoms(Atoms("H"), 4)
    frame.sid = "new"
    with ParquetTrajectoryWriter(path) as writer:
        writer.resume(step=2, state=None)
        writer.append(frame)
    result = pq.read_table(path).to_pylist()
    assert [row["step"] for row in result] == [0, 2, 4]
    assert [row["energy"] for row in result] == [None, -2.0, None]
    assert [row["sid"] for row in result] == ["2", "2", "new"]
    assert [row["sid_integer"] for row in result] == [2, 2, None]


@pytest.mark.parametrize(
    "thermostat",
    [
        BussiThermostat(300, 25, thermostat_seed=47),
        LangevinThermostat(300, 0.01, thermostat_seed=47),
    ],
)
def test_interleaved_and_resumed_stochastic_paths(cu_atoms, thermostat):
    def new_atoms():
        atoms = cu_atoms.copy()
        atoms.calc = EMT()
        return atoms

    reference = thermostat.build(new_atoms(), 0.5)
    reference.run(6)
    a, b = (thermostat.build(new_atoms(), 0.5) for _ in range(2))
    for step in range(6):
        a.run(1)
        b.run(1)
        if step == 2:
            state = json.loads(json.dumps(thermostat.save_state(a)))
            restored = a.atoms.copy()
            restored.calc = EMT()
            a = thermostat.build_for_restart(restored, 0.5, state)
    npt.assert_array_equal(b.atoms.positions, reference.atoms.positions)
    npt.assert_array_equal(b.atoms.get_momenta(), reference.atoms.get_momenta())
    # A fresh EMT neighbor list can accumulate the same forces in a different order.
    npt.assert_allclose(
        a.atoms.positions, reference.atoms.positions, rtol=0, atol=1e-13
    )
    npt.assert_allclose(
        a.atoms.get_momenta(), reference.atoms.get_momenta(), rtol=0, atol=1e-13
    )


@pytest.mark.parametrize(
    "constraint",
    [
        FixInternals(bonds=[[None, [0, 1]]]),
        FixLinearTriatomic([[0, 1, 2]]),
    ],
)
def test_restart_retains_original_constraint_targets(tmp_path, constraint):
    atoms = Atoms("OCO", positions=[[0, 0, 0], [1.15, 0, 0], [2.29, 0, 0]])
    atoms.set_constraint(constraint)
    constraint.initialize(atoms)
    # The restart geometry lies slightly away from the original target.
    atoms.positions[0, 0] = 1e-10
    runner = MDRunner(EMT(), VelocityVerletThermostat(), atoms, steps=1)
    runner._job_config = _create_mock_job_config(str(tmp_path))
    runner._dyn = runner.thermostat.build(atoms, 1.0)
    assert runner.save_state(str(tmp_path / "checkpoint"))
    runner.load_state(str(tmp_path / "checkpoint"))
    restored = runner._atoms.constraints[0]
    npt.assert_array_equal(atoms.positions, runner._atoms.positions)
    if isinstance(constraint, FixInternals):
        restored.initialize(runner._atoms)
        assert (
            restored.constraints[0].targetvalue == constraint.constraints[0].targetvalue
        )
    else:
        npt.assert_array_equal(restored.bondlengths, constraint.bondlengths)
        npt.assert_array_equal(restored.C1, constraint.C1)


def test_unsupported_constraint_is_reported(tmp_path, caplog):
    class CustomConstraint(FixAtoms):
        pass

    atoms = Atoms("Cu", constraint=CustomConstraint(indices=[0]))
    runner = MDRunner(EMT(), VelocityVerletThermostat(), atoms, steps=1)
    runner._job_config = _create_mock_job_config(str(tmp_path))
    runner._dyn = runner.thermostat.build(atoms, 1.0)
    assert not runner.save_state(str(tmp_path / "checkpoint"))
    assert "Cannot checkpoint constraint CustomConstraint" in caplog.text


def test_parallel_checkpoint_publication_preserves_shards(tmp_path):
    class Job(runner_module.PreemptableMixin):
        checkpoint_shard_pattern = "shard-*.json"

        def __init__(self, index):
            self.index = index

        def save_simulation_state(self, directory, is_preemption):
            (directory / f"shard-{self.index}.json").write_text(str(self.index))
            (directory / "step").write_text(str(self.index))

        def load_simulation_state(self, directory):
            step = int((directory / "step").read_text())
            assert int((directory / f"shard-{step}.json").read_text()) == step

    checkpoint = str(tmp_path / "cp")

    def save_then_read(i):
        job = Job(i)
        assert job.save_state(checkpoint)
        job.load_state(checkpoint)

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(save_then_read, range(8)))
    assert sorted(
        int(path.read_text()) for path in Path(checkpoint).glob("shard-*.json")
    ) == list(range(8))

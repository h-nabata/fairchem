"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from ase import build, db
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import Trajectory, write

from fairchem.core.datasets import (
    AseDBDataset,
    AseReadDataset,
    AseReadMultiStructureDataset,
)


@pytest.fixture(
    params=[
        "db_dataset",
        "db_dataset_folder",
        "db_dataset_list",
        "db_dataset_path_list",
        "lmdb_dataset",
        "aselmdb_dataset",
    ],
)
def ase_dataset(request, structures, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dataset")
    mult = 1
    a2g_args = {
        "r_energy": True,
        "r_forces": True,
        "r_stress": True,
        "r_data_keys": ["extensive_property", "tensor_property"],
    }
    if request.param == "db_dataset":
        with db.connect(tmp_path / "asedb.db") as database:
            for _i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)
        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.db"), "a2g_args": a2g_args}
        )
    elif request.param == "db_dataset_folder" or request.param == "db_dataset_list":
        for db_name in ("asedb1.db", "asedb2.db"):
            with db.connect(tmp_path / db_name) as database:
                for _i, atoms in enumerate(structures):
                    database.write(atoms, data=atoms.info)
        mult = 2
        src = (
            str(tmp_path)
            if request.param == "db_dataset_folder"
            else [str(tmp_path / "asedb1.db"), str(tmp_path / "asedb2.db")]
        )
        dataset = AseDBDataset(config={"src": src, "a2g_args": a2g_args})
    elif request.param == "db_dataset_path_list":
        os.mkdir(tmp_path / "dir1")
        os.mkdir(tmp_path / "dir2")

        for dir_name in ("dir1", "dir2"):
            for db_name in ("asedb1.db", "asedb2.db"):
                with db.connect(tmp_path / dir_name / db_name) as database:
                    for _i, atoms in enumerate(structures):
                        database.write(atoms, data=atoms.info)
        mult = 4
        dataset = AseDBDataset(
            config={
                "src": [str(tmp_path / "dir1"), str(tmp_path / "dir2")],
                "a2g_args": a2g_args,
            }
        )
    else:  # "aselmbd_dataset" with .aselmdb file extension
        with connect(str(tmp_path / "asedb.aselmdb")) as database:
            for _i, atoms in enumerate(structures):
                database.write(atoms, data=atoms.info)

        dataset = AseDBDataset(
            config={"src": str(tmp_path / "asedb.aselmdb"), "a2g_args": a2g_args}
        )

    return dataset, mult


def test_ase_dataset(ase_dataset, structures):
    dataset, mult = ase_dataset
    assert len(dataset) == mult * len(structures)
    for i, data in enumerate(dataset):
        assert data.forces.shape == (data.natoms, 3)
        assert data.stress.shape == (1, 3, 3)
        assert "sid" in dataset.get_atoms(i).info


def test_ase_read_dataset(tmp_path, structures):
    # unfortunately there is currently no clean (already implemented) way to save atoms.info when saving
    # individual structures - so test separately
    for i, structure in enumerate(structures):
        write(tmp_path / f"{i}.cif", structure)

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.cif",
        }
    )

    assert len(dataset) == len(structures)
    data = dataset[0]
    del data

    # Make sure get_atoms does not raise
    atoms = dataset.get_atoms(0)
    assert "sid" in atoms.info


def test_ase_get_metadata(ase_dataset):
    assert ase_dataset[0].get_metadata("natoms", [0])[0] == 3


def test_db_add_delete(tmp_path, structures):
    database = db.connect(tmp_path / "asedb.db")
    for _i, atoms in enumerate(structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == len(structures)
    orig_len = len(dataset)

    database.delete([1])

    new_structures = [
        build.molecule("CH3COOH", vacuum=4),
        build.bulk("Al"),
    ]

    for _i, atoms in enumerate(new_structures):
        database.write(atoms, data=atoms.info)

    dataset = AseDBDataset(config={"src": str(tmp_path / "asedb.db")})
    assert len(dataset) == orig_len + len(new_structures) - 1

    # Make sure get_atoms does not raise
    dataset.get_atoms(0)


def test_ase_multiread_dataset(tmp_path):
    atoms_objects = [build.bulk("Cu", a=a) for a in np.linspace(3.5, 3.7, 10)]

    energies = np.linspace(1, 0, len(atoms_objects))

    traj = Trajectory(tmp_path / "test.traj", mode="w")

    for atoms, energy in zip(atoms_objects, energies):
        calc = SinglePointCalculator(atoms, energy=energy, forces=atoms.positions)
        atoms.calc = calc
        traj.write(atoms)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "keep_in_memory": True,
            "atoms_transform_args": {
                "skip_always": True,
            },
        }
    )

    assert len(dataset) == len(atoms_objects)

    with open(tmp_path / "test_index_file", "w") as f:
        f.write(f"{tmp_path / 'test.traj'} {len(atoms_objects)}")

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "index_file": str(tmp_path / "test_index_file"),
        },
    )

    assert len(dataset) == len(atoms_objects)

    dataset = AseReadMultiStructureDataset(
        config={
            "src": str(tmp_path),
            "index_file": str(tmp_path / "test_index_file"),
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert len(dataset) == len(atoms_objects)

    assert hasattr(dataset[0], "energy_relaxed")
    assert dataset[0].energy_relaxed != dataset[0].energy
    assert dataset[-1].energy_relaxed == dataset[-1].energy

    dataset = AseReadDataset(
        config={
            "src": str(tmp_path),
            "pattern": "*.traj",
            "ase_read_args": {
                "index": "0",
            },
            "a2g_args": {
                "r_energy": True,
                "r_forces": True,
            },
            "include_relaxed_energy": True,
        }
    )

    assert hasattr(dataset[0], "energy_relaxed")
    assert dataset[0].energy_relaxed != dataset[0].energy

    # Make sure get_atoms does not raise
    atoms = dataset.get_atoms(0)
    assert "sid" in atoms.info


def test_empty_dataset(tmp_path):
    # raises error on empty dataset
    with pytest.raises(ValueError):
        AseReadMultiStructureDataset(config={"src": str(tmp_path)})

    with pytest.raises(ValueError):
        AseDBDataset(config={"src": str(tmp_path)})


@pytest.mark.parametrize("use_index", [True, False])
def test_trajectory_frame_selection_cache_and_path_spaces(
    tmp_path, monkeypatch, use_index
):
    import ase.io

    paths = [tmp_path / "run one.traj", tmp_path / "run two.traj"]
    for path, count in zip(paths, (3, 4)):
        with Trajectory(path, mode="w") as writer:
            for i in range(count):
                atoms = build.bulk("Cu", a=3.5 + 0.1 * i)
                atoms.calc = SinglePointCalculator(atoms, energy=float(i))
                writer.write(atoms)
    cfg = dict(
        src=str(tmp_path),
        pattern="*.traj",
        keep_in_memory=True,
        include_relaxed_energy=True,
        ase_read_args={"index": "1::2"},
    )
    if use_index:
        index = tmp_path / "index"
        index.write_text(f"{paths[0]} 3\n\n{paths[1]} 4\n")
        cfg["index_file"] = str(index)
    transforms = []

    def transform(atoms):
        transforms.append(len(atoms))
        # A rigid translation preserves this synthetic scalar label. Reattach
        # it explicitly to the transformed state, as a label-aware transform.
        energy = atoms.get_potential_energy()
        atoms.positions += 0.5
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        return atoms

    dataset = AseReadMultiStructureDataset(cfg, atoms_transform=transform)
    assert cfg["ase_read_args"] == {"index": "1::2"}
    assert sorted(dataset.ids) == sorted(
        [f"{paths[0]} 1", f"{paths[1]} 1", f"{paths[1]} 3"]
    )
    if use_index:
        assert dataset.ids[1] == f"{paths[1]} 1"
    reads = []
    original = ase.io.read

    def read(filename, *, index, **kwargs):
        reads.append(index)
        return original(filename, index=index, **kwargs)

    monkeypatch.setattr(ase.io, "read", read)
    i = dataset.ids.index(f"{paths[1]} 1")
    first = dataset[i]
    first.pos.add_(10)
    second = dataset[i - len(dataset)]
    assert reads == [1, -1]
    assert len(transforms) == 2
    assert second.energy_relaxed == 3
    assert np.allclose(second.pos, 0.5)
    dataset[dataset.ids.index(f"{paths[1]} 3")]
    assert reads == [1, -1, 3]


def test_db_ids_distinguish_sources_and_preserve_explicit_identifiers(tmp_path):
    from ase import Atoms
    from ase.db import connect

    for name in ("first", "second"):
        db = connect(tmp_path / f"{name}.db")
        db.write(Atoms("H"))
        db.write(Atoms("He"), sid=17)
    dataset = AseDBDataset({"src": str(tmp_path)})
    assert dataset[0].sid[0].endswith("first.db#1")
    assert dataset[2].sid[0].endswith("second.db#1")
    assert dataset[1].sid == ["17"]

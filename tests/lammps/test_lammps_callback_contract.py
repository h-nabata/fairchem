"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from ase import Atoms

lammps = pytest.importorskip("lammps")
from fairchem.lammps.lammps_fc import run_lammps_with_fairchem  # noqa: E402

from fairchem.core.calculate.ase_calculator import FAIRChemCalculator  # noqa: E402


class BondPredictor:
    supports_requested_properties = True
    dataset_to_tasks: ClassVar = {
        "omat": [SimpleNamespace(property=p) for p in ("energy", "forces", "stress")]
    }

    def __init__(self, *, external=False, failure=None):
        self.inference_settings = SimpleNamespace(
            base_precision_dtype=torch.float64, external_graph_gen=external
        )
        self.graph_settings = {
            "cutoff": 1.0,
            "max_neighbors": 3,
            "otf_graph": not external,
        }
        self.inputs = []
        self.failure = failure

    def validate_atoms_data(self, atoms, task):
        pass

    def predict(self, data, requested_properties):
        self.inputs.append(data)
        if self.failure == "exception":
            raise ArithmeticError("analytic predictor failed")
        x = data.pos
        delta = x[1] - x[0]
        stretch = delta.norm() - 1
        force = stretch * delta / delta.norm()
        result = {
            "energy": stretch.square() / 2,
            "forces": torch.stack([force, -force]),
            "stress": x.new_tensor([0.2, 0.1, -0.3, 0.05, 0.06, 0.07]),
        }
        if self.failure == "shape":
            result["forces"] = force
        elif self.failure == "nan":
            result["energy"] = x.new_tensor(float("nan"))
        elif self.failure == "stress":
            result["stress"] = x.new_zeros(8)
        return {k: result[k] for k in requested_properties}


def write_script(tmp_path, commands, units="metal"):
    # Use two atoms in different x subdomains. Run the same test under
    # mpiexec -n 2 (or -n 3, including empty ranks) to exercise force scattering.
    units_line = f"units {units}" if units else ""
    path = tmp_path / "simulation.in"
    path.write_text(f"""{units_line}
atom_style atomic
boundary p p p
processors * 1 1
region sim block 0 5 0 5 0 5
create_box 1 sim
create_atoms 1 single 2.25 1 1
create_atoms 1 single 2.75 1 1
mass 1 12.011
pair_style zero 2
pair_coeff * *
thermo_style custom step pe pxx pyy pzz pxy pxz pyz
{commands}
""")
    return path


@pytest.mark.parametrize(
    "integrator", ["nve", "nvt temp 300 300 0.1", "npt temp 300 300 .1 iso 0 0 1"]
)
@pytest.mark.parametrize("included", [False, True])
def test_whole_system_forces_and_virial(tmp_path, integrator, included):
    command = f"fix integrate all {integrator}\nrun &\n 0"
    if included:
        child = tmp_path / "force step.in"
        child.write_text(command)
        command = f'include "{child}"'
    predictor = BondPredictor(external=True)
    lmp = run_lammps_with_fairchem(predictor, write_script(tmp_path, command), "omat")
    try:
        assert lmp.get_thermo("pe") == pytest.approx(0.125)
        thermo = [lmp.get_thermo(k) for k in ("pxx", "pyy", "pzz", "pxy", "pxz", "pyz")]
        np.testing.assert_allclose(
            thermo,
            -lmp.extract_global("nktv2p")
            * np.array([0.2, 0.1, -0.3, 0.07, 0.06, 0.05]),
        )
        nlocal = lmp.extract_setting("nlocal")
        ids = lmp.numpy.extract_atom("id")[:nlocal]
        np.testing.assert_allclose(
            lmp.numpy.extract_atom("f")[:nlocal, 0], np.where(ids == 1, -0.5, 0.5)
        )
        if lmp.extract_setting("world_rank") == 0:
            assert len(predictor.inputs) == 1
            data = predictor.inputs[0]
            assert data.spin.item() == 0
            assert data.nedges.item() == 2
            ase_atoms = Atoms(
                "C2", positions=[[2.25, 1, 1], [2.75, 1, 1]], cell=[5] * 3, pbc=True
            )
            ase_atoms.calc = FAIRChemCalculator(predictor, "omat")
            assert ase_atoms.get_potential_energy() == pytest.approx(
                lmp.get_thermo("pe")
            )
            np.testing.assert_array_equal(
                data.edge_index, predictor.inputs[-1].edge_index
            )
            np.testing.assert_array_equal(
                data.cell_offsets, predictor.inputs[-1].cell_offsets
            )
        else:
            assert not predictor.inputs
    finally:
        lmp.close()


@pytest.mark.parametrize(
    "failure,match",
    [
        ("exception", "analytic predictor failed"),
        ("shape", "Forces must have shape"),
        ("nan", "finite real numbers"),
        ("stress", "Stress must be"),
    ],
)
def test_callback_failure_aborts_rest_of_script(tmp_path, failure, match):
    output = tmp_path / "after-failure"
    child = tmp_path / "failure.in"
    child.write_text(f'run 1\nprint "invalid" file "{output}"')
    script = write_script(tmp_path, f'include "{child}"\nrun 0')
    with pytest.raises((ArithmeticError, ValueError, RuntimeError), match=match):
        run_lammps_with_fairchem(BondPredictor(failure=failure), script, "omat")
    assert not output.exists()


@pytest.mark.parametrize("units", ["real", ""])
def test_requires_effective_metal_units(tmp_path, units):
    predictor = BondPredictor()
    child = write_script(tmp_path, "run 0", units)
    script = tmp_path / "parent.in"
    script.write_text(f'include "{child}"')
    with pytest.raises(ValueError, match="requires LAMMPS units metal"):
        run_lammps_with_fairchem(predictor, script, "omat")
    assert not predictor.inputs


@pytest.mark.parametrize("spin,expected", [(None, 1), (3, 3)])
def test_omol_spin_and_force_only_mode(tmp_path, spin, expected):
    predictor = BondPredictor()
    lmp = run_lammps_with_fairchem(
        predictor,
        write_script(tmp_path, "run 0"),
        "omol",
        spin=spin,
        compute_stress=False,
    )
    try:
        if lmp.extract_setting("world_rank") == 0:
            assert predictor.inputs[0].spin.item() == expected
    finally:
        lmp.close()


def test_setup_only_then_live_run_propagates_error(tmp_path):
    predictor = BondPredictor()
    calls = []

    def factory():
        calls.append(True)
        return predictor

    lmp = run_lammps_with_fairchem(factory, write_script(tmp_path, ""), "omat")
    assert lmp.has_id("fix", "ext_fc")
    assert len(calls) == int(lmp.extract_setting("world_rank") == 0)
    try:
        lmp.command("run 0")
        assert lmp.get_thermo("pe") == pytest.approx(0.125)
        predictor.failure = "exception"
        output = tmp_path / "later"
        with pytest.raises(
            (ArithmeticError, RuntimeError), match="analytic predictor failed"
        ):
            lmp.commands_string(f'run 0\nprint "invalid" file "{output}"')
        assert not output.exists()
    finally:
        lmp.close()


def test_barostat_requires_stress(tmp_path):
    script = write_script(tmp_path, "fix int all npt temp 300 300 .1 iso 0 0 1\nrun 0")
    with pytest.raises((RuntimeError, ValueError), match="compute_stress"):
        run_lammps_with_fairchem(BondPredictor(), script, "omat", compute_stress=False)


def test_multiline_print_does_not_execute_its_content(tmp_path):
    output = tmp_path / "multiline.txt"
    command = (
        f'print """first line\nrun 1 # literal content\nlast line""" file "{output}"\n'
        "run 0"
    )
    predictor = BondPredictor()
    lmp = run_lammps_with_fairchem(predictor, write_script(tmp_path, command), "omat")
    try:
        if lmp.extract_setting("world_rank") == 0:
            assert "run 1 # literal content" in output.read_text()
            assert len(predictor.inputs) == 1
    finally:
        lmp.close()


@pytest.mark.skipif(os.name == "nt", reason="backslash is a Windows path separator")
def test_include_preserves_literal_backslash(tmp_path):
    child = tmp_path / r"child\file.in"
    child.write_text("run 0\n")
    # Unlike shell syntax, LAMMPS leaves an unquoted backslash literal.
    command = f"include {child}"
    lmp = run_lammps_with_fairchem(
        BondPredictor(), write_script(tmp_path, command), "omat"
    )
    try:
        assert lmp.get_thermo("pe") == pytest.approx(0.125)
    finally:
        lmp.close()


def test_nonperiodic_callback_preserves_fp64_and_owns_positions(tmp_path):
    script = write_script(
        tmp_path,
        "run 0",
    )
    script.write_text(
        script.read_text()
        .replace("boundary p p p", "boundary f f f")
        .replace("2.25 1 1", "2.2500000000123 1 1")
    )
    predictor = BondPredictor()
    lmp = run_lammps_with_fairchem(predictor, script, "omat", compute_stress=False)
    try:
        if lmp.extract_setting("world_rank") == 0:
            positions = predictor.inputs[0].pos
            assert positions.dtype == torch.float64
            assert positions[0, 0].item() == 2.2500000000123
        lmp.command("displace_atoms all move 0.01 0 0 units box")
        if lmp.extract_setting("world_rank") == 0:
            assert positions[0, 0].item() == 2.2500000000123
    finally:
        lmp.close()


@pytest.mark.parametrize(
    "commands, expected",
    [
        ("# comment &\nprint before file OUTPUT\nprint after append OUTPUT", "after\n"),
        ("print abc'd#e'f file OUTPUT", "abc'd#e'f\n"),
        ('print """beg  \nmid &\nend""" file OUTPUT', "beg\nmid end\n"),
    ],
)
def test_script_lexical_edges_match_native_lammps(tmp_path, commands, expected):
    output = tmp_path / "parser.txt"
    path = tmp_path / "parser.in"
    path.write_text(commands.replace("OUTPUT", f'"{output}"') + "\n")
    native = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
    try:
        native.file(str(path))
        is_root = native.extract_setting("world_rank") == 0
        if is_root:
            assert output.read_text() == expected
            output.unlink()
    finally:
        native.close()
    wrapped = run_lammps_with_fairchem(BondPredictor(), path, "omat")
    try:
        if is_root:
            assert output.read_text() == expected
    finally:
        wrapped.close()

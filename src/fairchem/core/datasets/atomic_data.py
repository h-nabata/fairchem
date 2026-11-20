"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

modified from torch_geometric Data class
"""

from __future__ import annotations

import copy
import logging
import types
import warnings
from collections import deque
from collections.abc import Sequence
from numbers import Integral
from typing import Union

import ase
import ase.db.sqlite
import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.constraints import FixAtoms
from ase.data import atomic_masses
from ase.geometry import wrap_positions
from ase.stress import full_3x3_to_voigt_6_stress, voigt_6_to_full_3x3_stress
from monty.dev import requires

from fairchem.core.common.utils import StrEnum
from fairchem.core.graph.compute import canonicalize_external_edges
from fairchem.core.graph.radius_graph_pbc import get_max_neighbors_mask

try:
    from pymatgen.io.ase import AseAtomsAdaptor

    pmg_installed = True
except ImportError:
    AseAtomsAdaptor = None
    pmg_installed = False

IndexType = Union[slice, torch.Tensor, np.ndarray, Sequence]


class ExternalGraphMethod(StrEnum):
    """Enum for external graph generation methods."""

    PYMATGEN = "pymatgen"
    NVIDIA = "nvidia"


# these are all currently certainly output by the current a2g
# except for tags, all fields are required for network inference.
_REQUIRED_KEYS = [
    "pos",
    "atomic_numbers",
    "cell",
    "pbc",
    "natoms",
    "edge_index",
    "cell_offsets",
    "nedges",
    "charge",
    "spin",
    "fixed",
    "tags",
]

_OPTIONAL_KEYS = ["energy", "forces", "stress", "dataset"]


def _tensors(value, path=(), seen=None, *, strict=False):
    """Yield each distinct tensor in Python metadata, including keys and slots.

    Paths identify reusable buffers without retaining caller-owned objects.
    Strict consumers require inspectable metadata to account for tensor storage.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))
    if torch.is_tensor(value):
        yield path, value
        return
    if value is None or isinstance(
        value, (str, bytes, int, float, complex, range, torch.dtype, torch.device)
    ):
        return
    if isinstance(value, (np.ndarray, np.generic)):
        if strict and value.dtype.hasobject:
            raise TypeError(
                "Object arrays are not supported in tensor staging metadata"
            )
        return

    inspected = False
    if isinstance(value, dict):
        inspected = True
        for i, (key, child) in enumerate(value.items()):
            yield from _tensors(key, (*path, ("key", i)), seen, strict=strict)
            yield from _tensors(child, (*path, ("value", i)), seen, strict=strict)
    elif isinstance(value, (tuple, list, set, frozenset, deque)):
        inspected = True
        for i, child in enumerate(value):
            yield from _tensors(child, (*path, ("item", i)), seen, strict=strict)
    if type(value) in (list, tuple, set, frozenset, dict, deque):
        return
    if not isinstance(
        value, (type, types.ModuleType, types.FunctionType, types.MethodType)
    ):
        if hasattr(value, "__dict__"):
            inspected = True
            yield from _tensors(vars(value), (*path, "attributes"), seen, strict=strict)
        # Read the actual descriptor, including inherited/private slots. An
        # instance can declare both a __dict__ and slots, or leave slots unset.
        for depth, cls in enumerate(type(value).__mro__):
            if "__slots__" not in vars(cls):
                continue
            inspected = True
            for name, descriptor in vars(cls).items():
                if isinstance(descriptor, types.MemberDescriptorType):
                    try:
                        child = descriptor.__get__(value)
                    except AttributeError:
                        continue
                    yield from _tensors(
                        child, (*path, ("slot", depth, name)), seen, strict=strict
                    )
    if strict and not inspected:
        raise TypeError(f"Unsupported tensor staging metadata: {type(value).__name__}")


def _clone_tensor_tree(value, tensor_transform=None, memo=None):
    """Deep-copy nested data with differentiable clones of all referenced tensors."""
    memo = {} if memo is None else memo
    for _, tensor in _tensors(value, seen=set(memo)):
        memo[id(tensor)] = (
            tensor.clone() if tensor_transform is None else tensor_transform(tensor)
        )
    return copy.deepcopy(value, memo)


# TODO: potential future keys
# ["virials", "atom_attr", "edge_attr"]


def warn_if_upcasting(source_dtype: torch.dtype, target_dtype: torch.dtype) -> bool:
    """
    Log a warning if target_dtype has more precision than source_dtype.

    Returns True if a warning was issued, False otherwise.
    """
    if torch.finfo(target_dtype).bits > torch.finfo(source_dtype).bits:
        logging.warning(
            "Upcasting atomic coordinates from %s to %s. "
            "Accuracy may be limited by the precision of the "
            "input coordinates.",
            source_dtype,
            target_dtype,
        )
        return True
    return False


def size_repr(key: str, item: torch.Tensor, indent=0) -> str:
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, (list, tuple)):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


@requires(pmg_installed, message="Requires `pymatgen` to be installed")
def get_neighbors_pymatgen(
    atoms: ase.Atoms,
    cutoff,
    max_neigh,
    enforce_max_neighbors_strictly=None,
    *,
    target_dtype=torch.float64,
):
    """Performs nearest neighbor search and returns edge index, distances,
    and cell offsets"""
    struct = AseAtomsAdaptor.get_structure(atoms)

    # Pymatgen searches in double precision. Obtain a superset and apply the
    # same squared-distance predicate and shell cap as graph v2 in model dtype.
    pos = torch.as_tensor(atoms.get_positions(), dtype=target_dtype)
    cell = torch.as_tensor(np.asarray(atoms.cell), dtype=target_dtype)
    eps = torch.finfo(target_dtype).eps
    scale = cutoff + (float(pos.abs().max()) if atoms.pbc.any() and len(pos) else 0)
    _c_index, _n_index, _offsets, _ = struct.get_neighbor_list(
        r=cutoff + 16 * eps * (1 + scale), numerical_tol=1e-8, exclude_self=False
    )
    order = np.argsort(_c_index, kind="stable")
    _c_index, _n_index, _offsets = _c_index[order], _n_index[order], _offsets[order]
    shifts = torch.as_tensor(_offsets, dtype=target_dtype) @ cell
    squared = (pos[_c_index] - (pos[_n_index] + shifts)).square().sum(-1)
    selected = torch.where((squared > 0) & (squared <= cutoff * cutoff))[0]
    centers = torch.as_tensor(_c_index)[selected].long()
    squared = squared[selected]
    if enforce_max_neighbors_strictly is None:
        # The original standalone converter uses Python slicing, including for
        # nonpositive caps, and exposes distance-sorted edges for each center.
        by_center = selected.numpy()
        starts = np.cumsum(np.bincount(centers.numpy(), minlength=len(atoms)))[:-1]
        d2 = squared.numpy()
        ranked = [
            row[np.argsort(d2[row])][:max_neigh]
            for row in np.split(np.arange(len(d2)), starts)
        ]
        kept = np.concatenate(ranked)
        indices = by_center[kept]
        n_distance = np.sqrt(d2[kept])
    else:
        keep, _ = get_max_neighbors_mask(
            torch.tensor([len(atoms)]),
            centers,
            squared,
            max_neigh,
            enforce_max_strictly=enforce_max_neighbors_strictly,
        )
        # Public converter output has always been distance-sorted within centers.
        kept = torch.where(keep)[0]
        kept = kept[squared[kept].argsort(stable=True)]
        kept = kept[centers[kept].argsort(stable=True)]
        indices = selected[kept].numpy()
        n_distance = squared[kept].sqrt().numpy()
    _c_index, _n_index, _offsets = (
        _c_index[indices],
        _n_index[indices],
        _offsets[indices],
    )
    return _c_index, _n_index, n_distance, _offsets


def reshape_features(
    c_index: np.ndarray,
    n_index: np.ndarray,
    n_distance: np.ndarray,
    offsets: np.ndarray,
    target_dtype: torch.dtype = torch.float32,
):
    """Stack center and neighbor index and reshapes distances,
    takes in np.arrays and returns torch tensors"""
    edge_index = torch.LongTensor(np.vstack((n_index, c_index)))
    edge_distances = torch.tensor(n_distance, dtype=target_dtype, device="cpu")
    cell_offsets = torch.tensor(offsets, dtype=target_dtype, device="cpu")

    # Neighbor search and model geometry exclude exact zero-distance edges.
    nonzero = torch.where(edge_distances > 0)[0]
    edge_index = edge_index[:, nonzero]
    cell_offsets = cell_offsets[nonzero]

    return edge_index, cell_offsets


class AtomicData:
    def __init__(
        self,
        pos: torch.Tensor,  # (num_node, 3)
        atomic_numbers: torch.Tensor,  # (num_node,)
        cell: torch.Tensor,  # (num_graph, 3, 3)
        pbc: torch.Tensor,  # (num_graph, 3)
        natoms: torch.Tensor,  # (1,)
        edge_index: torch.Tensor,  # (2, num_edge)
        cell_offsets: torch.Tensor,  # (num_edge, 3)
        nedges: torch.Tensor,  # (1,)
        charge: torch.Tensor,  # (num_graph,)
        spin: torch.Tensor,  # (num_graph,)
        fixed: torch.Tensor,  # (num_node,)
        tags: torch.Tensor,  # (num_node,)
        energy: torch.Tensor | None = None,  # (num_graph,)
        forces: torch.Tensor | None = None,  # (num_node, 3)
        stress: torch.Tensor | None = None,  # (num_graph, 3, 3)
        batch: torch.Tensor | None = None,  # (num_node,)
        sid: list[str] | None = None,
        dataset: list[str] | str | None = None,
    ):
        self.__keys__ = set(_REQUIRED_KEYS)

        # this conversion must have been done somewhere in
        # pytorch geometric data
        self.pos = pos
        self.atomic_numbers = atomic_numbers
        self.cell = cell.to(self.pos.dtype)
        self.pbc = pbc
        self.natoms = natoms
        self.edge_index = edge_index
        self.cell_offsets = cell_offsets
        self.nedges = nedges
        self.charge = charge
        self.spin = spin
        self.fixed = fixed
        self.tags = tags

        # Always normalize dataset to a list for consistent batching
        if dataset is not None:
            if isinstance(dataset, str):
                self.dataset = [dataset]
            elif isinstance(dataset, list):
                self.dataset = dataset
            else:
                raise ValueError(
                    f"dataset must be a string or list of strings, got {type(dataset)}"
                )

        # targets
        if energy is not None:
            self.energy = energy
        if forces is not None:
            self.forces = forces
        if stress is not None:
            self.stress = stress

        # batch related
        if batch is not None:
            self.batch = batch
        else:
            self.batch = torch.zeros_like(self.atomic_numbers)

        # id
        if sid is None:
            self.sid = [""] * self.num_graphs
        elif isinstance(sid, (str, Integral)):
            self.sid = [str(sid)]
        elif isinstance(sid, list) and all(
            isinstance(value, (str, Integral)) for value in sid
        ):
            self.sid = [str(value) for value in sid]
        else:
            raise TypeError("sid must be an integer, string or list of identifiers")

        self.__slices__ = None
        self.__cumsum__ = None
        self.__cat_dims__ = None
        self.__natoms_list__ = None

        # self.custom_fields = {}

        self.validate()

    @property
    def task_name(self):
        return getattr(self, "dataset", None)

    @task_name.setter
    def task_name(self, value):
        self.dataset = value

    @property
    def masses(self) -> torch.Tensor:
        options = {"dtype": self.pos.dtype, "device": self.pos.device}
        explicit = self.get("atomic_masses")
        if torch.is_tensor(explicit):  # serialized tensor overrides
            return explicit.to(**options)
        if explicit is not None and None not in explicit:
            return torch.tensor(explicit, **options)
        defaults = torch.as_tensor(atomic_masses, **options)[self.atomic_numbers]
        if explicit is None:
            return defaults
        mask = torch.tensor(
            [mass is not None for mass in explicit], device=self.pos.device
        )
        values = torch.tensor(
            [0 if mass is None else mass for mass in explicit], **options
        )
        return torch.where(mask, values, defaults)

    def assign_batch_stats(self, slices, cumsum, cat_dims, natoms_list):
        self.__slices__ = slices
        self.__cumsum__ = cumsum
        self.__cat_dims__ = cat_dims
        self.__natoms_list__ = natoms_list

    def get_batch_stats(self):
        return self.__slices__, self.__cumsum__, self.__cat_dims__, self.__natoms_list__

    def validate(self):
        # shape checks
        if self.natoms.numel() == 0 or bool((self.natoms <= 0).any()):
            raise ValueError("AtomicData requires at least one atom per system")
        if not torch.isfinite(self.pos).all() or not torch.isfinite(self.cell).all():
            raise ValueError("Atomic positions and cell must contain finite values")
        assert (
            self.pos.shape[0]
            == self.atomic_numbers.shape[0]
            == self.num_nodes
            == self.natoms.sum().item()
        )
        assert self.pos.dim() == 2
        assert self.pos.shape[1] == 3
        assert self.atomic_numbers.dim() == 1
        assert self.cell.shape[0] == self.pbc.shape[0] == self.num_graphs
        assert self.cell.dim() == 3
        assert self.cell.shape[1:] == (3, 3)
        assert self.pbc.dim() == 2
        assert self.pbc.shape[1] == 3
        assert self.edge_index.shape[0] == 2
        assert self.cell_offsets.shape[0] == self.edge_index.shape[1]
        assert self.fixed.shape[0] == self.pos.shape[0]
        assert self.tags.shape[0] == self.pos.shape[0]
        assert self.batch.shape == self.atomic_numbers.shape
        assert len(self.sid) == self.num_graphs

        if "dataset" in self.__keys__:
            assert isinstance(self.dataset, list), "dataset must always be a list"
            assert len(self.dataset) == self.num_graphs

        # dtype checks
        assert self.cell.dtype == self.pos.dtype
        assert self.cell_offsets.dtype in (
            self.pos.dtype,
            torch.long,
        ), "Cell offsets must use the position dtype or exact int64 image IDs"
        assert self.pos.dtype in (
            torch.float32,
            torch.float64,
        ), "Positions, cell, cell_offsets are all expected to be f32/f64"
        assert self.atomic_numbers.dtype == torch.long
        assert self.edge_index.dtype == torch.long
        assert self.pbc.dtype == torch.bool
        assert self.fixed.dtype == self.tags.dtype == torch.long
        assert self.batch.dtype == torch.long
        for name in _REQUIRED_KEYS:
            if getattr(self, name).device != self.pos.device:
                raise ValueError(
                    f"AtomicData field '{name}' must be on the positions device"
                )
        for name in ("natoms", "nedges", "charge", "spin"):
            value = getattr(self, name)
            assert value.dtype == torch.long, f"{name} must be an int64 tensor"
            assert value.shape == (
                self.num_graphs,
            ), f"{name} must have one entry per system"
        if self.num_graphs == 1:
            grouped = not bool(self.batch.any())
        elif self.batch.device.type == "cpu":
            ids = np.repeat(np.arange(self.num_graphs), self.natoms.numpy())
            grouped = np.array_equal(self.batch.numpy(), ids)
        else:
            ids = torch.arange(self.num_graphs, device=self.batch.device)
            grouped = torch.equal(
                self.batch,
                ids.repeat_interleave(self.natoms, output_size=self.num_nodes),
            )
        if not grouped:
            raise ValueError(
                "Batch atoms must be grouped contiguously in system order according to natoms"
            )
        self.edge_index, self.cell_offsets = canonicalize_external_edges(
            self.batch, self.edge_index, self.cell_offsets, self.nedges
        )
        assert isinstance(self.num_nodes, int)
        assert isinstance(self.num_graphs, int)

        # NOTE maybe do re matching instead of exact key matching for energy/forces/stress
        if hasattr(self, "energy"):
            assert self.energy.dim() == 1
            assert self.energy.shape[0] == self.num_graphs
            assert self.energy.dtype == self.pos.dtype
        if hasattr(self, "forces"):
            assert self.forces.shape[0] == self.pos.shape[0]
            assert self.forces.shape[1] == 3
            assert self.forces.dtype == self.pos.dtype
        if hasattr(self, "stress"):
            # NOTE: usually decomposed. for EFS prediction right now we reshape to (9,). need to discuss, perhaps use (1,3,3)
            assert (self.stress.dim() == 3 and self.stress.shape[1:] == (3, 3)) or (
                self.stress.dim() == 2 and self.stress.shape[1:] == (9,)
            )
            assert self.stress.shape[0] == self.num_graphs
            assert self.stress.dtype == self.pos.dtype

        if self.sid is not None:
            assert isinstance(self.sid, list)
            assert all(isinstance(s, str) for s in self.sid)
            assert len(self.sid) == self.num_graphs

        # for k, v in self.custom_fields.items():
        #     assert self[k].shape == v['shape']
        #     assert self[k].dtype == v['dtype']

    @classmethod
    def from_ase(
        cls,
        input_atoms: ase.Atoms,
        r_edges: bool = False,
        radius: float = 6.0,
        max_neigh: int | None = None,
        sid: str | None = None,
        molecule_cell_size: float | None = None,
        r_energy: bool = True,  # deprecated
        r_forces: bool = True,
        r_stress: bool = True,
        r_data_keys: list[str] | None = None,  # Explicitly select charge/spin metadata.
        task_name: str | None = None,
        target_dtype: torch.dtype = torch.float32,
        external_graph_method: ExternalGraphMethod | str = ExternalGraphMethod.PYMATGEN,
        enforce_max_neighbors_strictly: bool | None = None,
    ) -> AtomicData:
        atoms = input_atoms.copy()
        calc = input_atoms.calc
        if np.all(np.asarray(atoms.pbc)) and atoms.cell.volume == 0:
            raise ValueError("A fully periodic structure requires a nonsingular cell")
        # TODO: maybe compute a safe cell size if not provided.
        if molecule_cell_size is not None:
            assert (
                atoms.cell.volume == 0.0
            ), "atoms must not have a unit cell to begin with to create a molecule cell"
            # create a molecule box with the molecule centered on it if specified
            atoms.center(vacuum=(molecule_cell_size))
            atoms.pbc = np.array([True, True, True])
        atomic_numbers = np.array(atoms.get_atomic_numbers(), copy=True)
        pos = np.array(atoms.get_positions(), copy=True)
        pbc = np.array(atoms.pbc, copy=True)
        physical_cell = atoms.cell.copy()
        if pbc.any() and np.linalg.matrix_rank(physical_cell.array[pbc]) < pbc.sum():
            raise ValueError(
                "Periodic lattice vectors must span a nonsingular subspace"
            )
        search_cell = physical_cell.copy()
        if pbc.any() and physical_cell.volume == 0:
            search_cell[~pbc] = 0
        search_cell = search_cell.complete()
        pos = wrap_positions(pos, search_cell, pbc=pbc, eps=0)
        # Keep the supplied physical volume for slabs/chains. A cell-less
        # nonperiodic molecule retains its historical completed model cell;
        # that completion has no physical molecular volume.
        cell = np.array(physical_cell if pbc.any() else search_cell, copy=True)

        atomic_numbers = torch.from_numpy(atomic_numbers).long()
        pos = torch.from_numpy(pos)
        warn_if_upcasting(pos.dtype, target_dtype)
        pos = pos.to(target_dtype)
        pbc = torch.from_numpy(pbc).bool().view(1, 3)
        cell = torch.from_numpy(cell).to(target_dtype).view(1, 3, 3)
        natoms = torch.tensor([pos.shape[0]], dtype=torch.long, device="cpu")

        # graph construction
        if r_edges:
            assert (
                radius is not None
            ), "cutoff must be specified for cpu graph construction."
            assert (
                max_neigh is not None
            ), "max_neigh must be specified for cpu graph construction."

            # Promote model-rounded coordinates back to ASE's double container.
            # Constraints must not restore the original unwrapped geometry.
            atoms.set_positions(pos.numpy(), apply_constraint=False)
            atoms.set_cell(
                torch.as_tensor(np.asarray(search_cell)).to(target_dtype).numpy(),
                apply_constraint=False,
            )
            if external_graph_method == ExternalGraphMethod.PYMATGEN:
                split_idx_dist = get_neighbors_pymatgen(
                    atoms,
                    radius,
                    max_neigh,
                    enforce_max_neighbors_strictly,
                    target_dtype=target_dtype,
                )
            elif external_graph_method == ExternalGraphMethod.NVIDIA:
                from fairchem.core.graph.radius_graph_pbc_nvidia import (
                    get_neighbors_nvidia_atoms,
                )

                split_idx_dist = get_neighbors_nvidia_atoms(
                    atoms,
                    radius,
                    max_neigh,
                    target_dtype=target_dtype,
                    enforce_max_neighbors_strictly=enforce_max_neighbors_strictly,
                )
            else:
                raise ValueError(
                    f"external_graph_method must be 'pymatgen' or 'nvidia', got {external_graph_method}"
                )

            edge_index, cell_offsets = reshape_features(
                *split_idx_dist, target_dtype=target_dtype
            )
            nedges = torch.tensor([edge_index.shape[1]], dtype=torch.long, device="cpu")
        else:
            # empty graph
            edge_index = torch.empty((2, 0), dtype=torch.long, device="cpu")
            cell_offsets = torch.empty((0, 3), dtype=target_dtype, device="cpu")
            nedges = torch.tensor([0], dtype=torch.long, device="cpu")

        # initialized to torch.zeros(natoms) if tags missing.
        # https://docs.ase-lib.org/ase/atoms.html#ase.Atoms.get_tags
        tags = torch.as_tensor(atoms.get_tags(), dtype=torch.long, device="cpu").clone()
        fixed = torch.zeros(natoms, dtype=torch.long, device="cpu")
        if hasattr(atoms, "constraints"):
            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    fixed[constraint.index] = 1

        # Explicit info labels take precedence. Validate only calculator labels
        # actually read, against the original ASE geometry (before wrapping).
        results = {}
        if isinstance(calc, (SinglePointCalculator, SinglePointDFTCalculator)):
            requested = {"energy"}
            if r_forces:
                requested.add("forces")
            if r_stress:
                requested.add("stress")
            keys = (requested & calc.results.keys()) - atoms.info.keys()
            if keys and (changes := calc.check_state(input_atoms)):
                raise ValueError(
                    f"Stale single-point labels after changes to {', '.join(changes)}"
                )
            results.update({key: calc.results[key] for key in keys})
        for key in ("energy", "forces", "stress"):
            if key in atoms.info:
                results[key] = atoms.info[key]
        energy = (
            torch.tensor(results["energy"], dtype=target_dtype, device="cpu").reshape(1)
            if results.get("energy") is not None
            else None
        )
        forces = (
            torch.tensor(results["forces"], dtype=target_dtype, device="cpu").reshape(
                -1, 3
            )
            if r_forces and results.get("forces") is not None
            else None
        )
        stress = None
        if r_stress and results.get("stress") is not None:
            values = np.asarray(results["stress"])
            if values.shape == (6,):
                values = voigt_6_to_full_3x3_stress(values)
            elif values.shape not in ((3, 3), (9,)):
                raise ValueError(f"Unknown stress shape, {values.shape}")
            stress = torch.tensor(values, dtype=target_dtype, device="cpu").reshape(
                1, 3, 3
            )

        # TODO another way to specify this is to specify a key. maybe total_charge
        def integer_metadata(name):
            if (
                name in atoms.info
                and (r_data_keys is None or name not in r_data_keys)
                and not np.array_equal(atoms.info[name], 0)
            ):
                warnings.warn(
                    f"Discarding nonzero ASE {name}; include {name!r} in r_data_keys to retain it",
                    UserWarning,
                    stacklevel=3,
                )
            value = (
                atoms.info.get(name, 0)
                if r_data_keys is not None and name in r_data_keys
                else 0
            )
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"atoms.info['{name}'] must be an integer")
            return torch.tensor([int(value)], dtype=torch.long, device="cpu")

        charge, spin = integer_metadata("charge"), integer_metadata("spin")

        # NOTE: code assumes these are ints.. not tensors
        # charge = atoms.info.get("charge", 0)
        # spin = atoms.info.get("spin", 0)
        data = cls(
            pos=pos,
            atomic_numbers=atomic_numbers,
            cell=cell,
            pbc=pbc,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            nedges=nedges,
            charge=tensor_or_int_to_tensor(charge, torch.long),
            spin=tensor_or_int_to_tensor(spin, torch.long),
            fixed=fixed,
            tags=tags,
            energy=energy,
            forces=forces,
            stress=stress,
            sid=atoms.info.get("sid") if sid is None else sid,
            dataset=task_name,
        )

        if atoms.has("masses"):
            # Keep ASE's double-precision isotope values as host metadata even
            # when all model tensors are converted to FP32 or moved to a GPU.
            data["atomic_masses"] = tuple(atoms.get_masses().tolist())
        return data

    def to_ase_single(self) -> ase.Atoms:
        assert self.num_graphs == 1, "Data object must contain a single graph."

        masses = self.get("atomic_masses")
        if torch.is_tensor(masses):
            masses = masses.detach().cpu().tolist()
        if masses is not None and all(value is None for value in masses):
            masses = None  # preserve the original absence of an ASE masses array
        atoms = ase.Atoms(
            numbers=self.atomic_numbers.cpu().numpy(),
            positions=self.pos.detach().cpu().numpy(),
            cell=self.cell.detach().squeeze().cpu().numpy(),
            pbc=self.pbc.squeeze().tolist(),
            constraint=FixAtoms(mask=self.fixed.bool().tolist()),
            tags=self.tags.cpu().numpy(),
            masses=masses,
        )

        if self.__keys__.intersection(["energy", "forces", "stress"]):
            fields = {}
            if hasattr(self, "energy") and self.energy is not None:
                fields["energy"] = self.energy.detach().item()
            if hasattr(self, "forces") and self.forces is not None:
                fields["forces"] = self.forces.detach().cpu().numpy()
            if hasattr(self, "stress") and self.stress is not None:
                stress = self.stress.detach().cpu().numpy()
                if stress.size == 9:
                    fields["stress"] = full_3x3_to_voigt_6_stress(stress.reshape(3, 3))
                elif stress.size == 6:
                    fields["stress"] = stress.reshape(6)
                else:
                    raise ValueError("A single-system stress must have 6 or 9 values")
            atoms.calc = SinglePointCalculator(atoms=atoms, **fields)

        atoms.info = {
            "charge": self.charge.item(),
            "spin": self.spin.item(),
        }

        if self.sid is not None:
            atoms.info["sid"] = copy.deepcopy(self.sid)

        return atoms

    def to_ase(self) -> list[ase.Atoms]:
        return [self.get_example(i).to_ase_single() for i in range(self.num_graphs)]

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        assert set(_REQUIRED_KEYS).issubset(
            dictionary.keys()
        ), f"Missing required keys: {set(_REQUIRED_KEYS) - set(dictionary.keys())}"

        data = cls(
            pos=dictionary["pos"],
            atomic_numbers=dictionary["atomic_numbers"],
            cell=dictionary["cell"],
            pbc=dictionary["pbc"],
            natoms=dictionary["natoms"],
            edge_index=dictionary["edge_index"],
            cell_offsets=dictionary["cell_offsets"],
            nedges=dictionary["nedges"],
            charge=dictionary["charge"],
            spin=dictionary["spin"],
            fixed=dictionary.get("fixed", None),
            tags=dictionary.get("tags", None),
            energy=dictionary.get("energy", None),
            forces=dictionary.get("forces", None),
            stress=dictionary.get("stress", None),
            batch=dictionary.get("batch", None),
            sid=dictionary.get("sid", None),
            dataset=dictionary.get("dataset", None),
        )

        # TODO: may require validation for them in the future
        # assign extra keys in dict...
        for key in dictionary.keys() - set(
            _REQUIRED_KEYS + _OPTIONAL_KEYS + ["sid", "batch"]
        ):
            data[key] = dictionary[key]

        return data

    # TODO clean
    def to_dict(self):
        return dict(self)

    # TODO clean
    def values(self):
        return [item for _, item in self]

    ###############################
    #       basic operations      #
    ###############################

    @property
    def num_nodes(self) -> int:
        r"""Returns or sets the number of nodes in the graph."""
        return self.pos.size(0)

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the graph."""
        return self["edge_index"].size(1)

    @property
    def num_graphs(self) -> int:
        """Returns the number of graphs in the batch."""
        return self.natoms.numel()

    def __len__(self):
        return self.num_graphs

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def __getitem__(self, idx):
        if isinstance(idx, str):
            return getattr(self, idx)
        elif isinstance(idx, (int, np.integer)):
            return self.get_example(idx)
        else:
            return self.index_select(idx)

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __setitem__(self, key: str, value: torch.Tensor):
        # TODO call add custom field? if validation checks are given?
        """Sets the attribute :obj:`key` to :obj:`value`."""
        # renaming keys is not supported if this function validate keys.
        # assert key in self.__keys__ or key in set(_REQUIRED_KEYS + _OPTIONAL_KEYS + ['batch']), f"Invalid key: {key}"
        setattr(self, key, value)
        if key not in self.__keys__:
            self.__keys__.add(key)
        # self.validate()

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __setattr__(self, key: str, value: torch.Tensor):
        super().__setattr__(key, value)
        if not key.startswith("__") and key not in self.__keys__:
            self.__keys__.add(key)

    # TODO: maybe should not allow this. this is currently used a lot though.
    def __delitem__(self, key: str):
        """Deletes the attribute :obj:`key`."""
        assert key in self.__keys__, f"Key: <{key}> not found."
        delattr(self, key)
        self.__keys__.remove(key)

    def keys(self):
        return self.__keys__

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
        data."""
        return key in self.__keys__

    def __iter__(self):
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.__keys__):
            yield key, self[key]

    def __call__(self, *keys):
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in keys if keys else sorted(self.__keys__):
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value) -> int:
        r"""Returns the dimension for which :obj:`value` of attribute
        :obj:`key` will get concatenated when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        if key == "edge_index":
            return -1
        return 0

    def __inc__(self, key, value) -> int:
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only known atom indices are shifted; job IDs can also contain "index".
        assert self.num_graphs == 1
        return self.num_nodes if key == "edge_index" else 0

    def __apply__(self, item, func, memo=None):
        if memo is None:
            memo = {}
        if id(item) in memo:
            return memo[id(item)]
        if torch.is_tensor(item):
            result = func(item)
            memo[id(item)] = result
            return result
        if isinstance(item, (str, bytes, int, float, complex)) or item is None:
            return item
        return _clone_tensor_tree(item, tensor_transform=func, memo=memo)

    def apply(self, func):
        r"""Applies the function :obj:`func` to all tensor attributes"""
        # Conversion mutates this owner in place. Metadata back-references must
        # keep pointing here instead of creating a second AtomicData instance.
        memo = {id(self): self}
        # The memo is indexed by Python ids. Keep original field roots alive
        # until all conversions finish, so replacing e.g. sid cannot recycle
        # its id for a later dataclass's temporary deepcopy state list.
        items = [(key, self[key]) for key in self.__keys__]
        for key, item in items:
            self[key] = self.__apply__(item, func, memo)

        return self

    def contiguous(self):
        r"""Ensures a contiguous memory layout for all tensor attributes"""
        return self.apply(lambda x: x.contiguous())

    def to(self, device=None, **kwargs):
        """
        Move tensors and convert floating fields while preserving index types.

        This mutates the object, as the existing device methods do. Clone first
        when another job must retain the original data.
        """
        dtype = kwargs.pop("dtype", None)
        if isinstance(device, torch.dtype):
            if dtype is not None:
                raise TypeError("dtype was specified twice")
            dtype, device = device, None
        elif isinstance(device, torch.Tensor):
            if dtype is not None:
                raise TypeError("dtype cannot accompany a reference tensor")
            dtype, device = device.dtype, device.device
        if dtype is not None and dtype not in (torch.float32, torch.float64):
            raise ValueError("AtomicData floating fields require float32 or float64")
        return self.apply(
            lambda x: x.to(
                device=device, dtype=dtype if x.is_floating_point() else None, **kwargs
            )
        )

    def cpu(self):
        r"""Copies all tensor attributes to CPU memory."""
        return self.apply(lambda x: x.cpu())

    def cuda(self, device=None, non_blocking=False):
        r"""Copies all tensor attributes to GPU memory."""
        return self.apply(lambda x: x.cuda(device=device, non_blocking=non_blocking))

    def clone(self):
        """
        Own all data while preserving aliases and non-leaf autograd links.
        """
        return _clone_tensor_tree(self)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any(isinstance(item, dict) for _, item in self)

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))

    ###############################
    # operations related to batch #
    ###############################

    def get_example(self, idx: int) -> AtomicData:
        r"""
        Reconstructs the :class:`AtomicData` object at index
        :obj:`idx` from a batched AtomicData object.
        """
        if not isinstance(idx, Integral) or isinstance(idx, bool):
            raise TypeError("AtomicData example index must be an integer")
        idx = self.num_graphs + idx if idx < 0 else idx
        if not 0 <= idx < self.num_graphs:
            raise IndexError("AtomicData example index out of range")
        if self.num_graphs == 1:
            return self.clone()
        if self.__slices__ is None:
            raise ValueError("Unbatching requires stored batch slice metadata")

        data_dict = {}
        for key in self.__slices__:
            item = self[key]
            if self.__cat_dims__[key] is None:
                # The item was concatenated along a new batch dimension,
                # so just index in that dimension:
                item = item[idx]
            else:
                # Narrow the item based on the values in `__slices__`.
                if isinstance(item, torch.Tensor):
                    dim = self.__cat_dims__[key]
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    item = item.narrow(dim, start, end - start)
                else:
                    start = self.__slices__[key][idx]
                    end = self.__slices__[key][idx + 1]
                    if self.__cat_dims__[key] in ("list", "tuple"):
                        item = item[start:end]
                        if self.__cat_dims__[key] == "tuple":
                            item = tuple(item)
                    else:
                        item = item[start:end]
                        item = item[0] if len(item) == 1 else item
                    item = _clone_tensor_tree(item)

            # Decrease its value by `cumsum` value:
            cum = self.__cumsum__[key][idx]
            if isinstance(item, torch.Tensor):
                if not isinstance(cum, int) or cum != 0:
                    item = item - cum
            elif isinstance(item, (int, float)) and cum != 0:
                item = item - cum

            data_dict[key] = item.clone() if torch.is_tensor(item) else item

        data_dict["batch"] = torch.zeros_like(data_dict["atomic_numbers"])
        data_dict["sid"] = [self.sid[idx]]

        if hasattr(self, "dataset"):
            data_dict["dataset"] = self.dataset[idx]

        return AtomicData.from_dict(data_dict)

    def index_select(self, idx: IndexType) -> list[AtomicData]:
        if isinstance(idx, slice):
            idx = list(range(self.num_graphs)[idx])

        elif isinstance(idx, torch.Tensor) and idx.dtype in (torch.int32, torch.int64):
            idx = idx.flatten().tolist()

        elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            if idx.numel() != self.num_graphs:
                raise IndexError("Boolean selection must have one entry per system")
            idx = idx.flatten().nonzero(as_tuple=False).flatten().tolist()

        elif isinstance(idx, np.ndarray) and idx.dtype.kind in "iub":
            if idx.dtype.kind == "b":
                if idx.size != self.num_graphs:
                    raise IndexError("Boolean selection must have one entry per system")
                idx = np.flatnonzero(idx).tolist()
            else:
                idx = idx.flatten().tolist()

        elif isinstance(idx, Sequence) and not isinstance(idx, str):
            pass

        else:
            raise IndexError(
                f"Only integers, slices (':'), list, tuples, torch.tensor"
                f"are valid indices (got "
                f"'{type(idx).__name__}')"
            )

        return [self.get_example(i) for i in idx]

    def batch_to_atomicdata_list(self) -> list[AtomicData]:
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able to reconstruct the initial objects."""
        return [self.get_example(i) for i in range(self.num_graphs)]

    def update_batch_edges(
        self, edge_index: torch.Tensor, cell_offsets: torch.Tensor, nedges: torch.Tensor
    ) -> AtomicData:
        r"""Update the connectivity of each batched AtomicData sample.

        Args:
            edge_index (torch.Tensor): New batch edge_index (shape [2, total_edges]).
            cell_offsets (torch.Tensor): Cell offsets per edge (shape [total_edges, 3]).
            nedges (torch.Tensor): Number of edges per system (shape [num_systems]).

        Returns:
            AtomicData: The updated batch object.
        """
        if nedges.shape != self.natoms.shape or cell_offsets.dtype not in (
            self.pos.dtype,
            torch.long,
        ):
            raise ValueError("Invalid edge count shape or cell offset dtype")
        edge_index, cell_offsets = canonicalize_external_edges(
            self.batch, edge_index, cell_offsets, nedges
        )
        if self.__slices__ is not None:
            edge_slices = [0, *nedges.cumsum(0).tolist()]
            self.__slices__ = {
                **self.__slices__,
                "edge_index": edge_slices,
                "cell_offsets": edge_slices,
            }
        self.edge_index, self.cell_offsets, self.nedges = (
            edge_index,
            cell_offsets,
            nedges,
        )
        return self


def atomicdata_list_to_batch(
    data_list: list[AtomicData], exclude_keys: list | None = None
) -> AtomicData:
    """
    all data points must be single graphs and have the same set of keys.
    TODO: exclude keys?
    """
    # this does not include the sid/fids
    # arbitrary set of keys handled at set_item?
    if not data_list:
        raise ValueError("Cannot construct an AtomicData batch from an empty list")
    if isinstance(exclude_keys, str):
        raise TypeError("exclude_keys must be a sequence of field names")
    excluded = set(exclude_keys or ())
    if excluded.intersection(_REQUIRED_KEYS + ["batch", "sid"]):
        raise ValueError("Cannot exclude required AtomicData fields")
    if "atomic_masses" not in excluded and any(
        "atomic_masses" in item for item in data_list
    ):
        normalized = []
        for source in data_list:
            item = source
            values = source.get("atomic_masses")
            if not isinstance(values, tuple):
                item = copy.copy(source)
                item.__keys__ = set(item.__keys__)
                if torch.is_tensor(values):
                    values = values.detach().cpu().tolist()
                item["atomic_masses"] = (
                    (None,) * item.num_nodes if values is None else tuple(values)
                )
            normalized.append(item)
        data_list = normalized
    if any(item.cell_offsets.dtype == torch.long for item in data_list):
        # Preserve large integer image IDs when collating them with ordinary
        # floating-point offsets. Keep each caller-owned sample unchanged.
        normalized = []
        for source in data_list:
            item = source
            if item.cell_offsets.is_floating_point():
                images = item.cell_offsets.long()
                if not torch.equal(images.to(item.cell_offsets), item.cell_offsets):
                    raise ValueError(
                        "Cannot batch fractional cell offsets with exact integer images"
                    )
                item = copy.copy(source)
                item.cell_offsets = images
            normalized.append(item)
        data_list = normalized
    keys = sorted(set(data_list[0].keys()) - excluded)
    for index, data in enumerate(data_list):
        if set(data.keys()) - excluded != set(keys):
            raise ValueError(
                f"AtomicData item {index} has different fields; use exclude_keys explicitly"
            )

    batched_data_dict = {k: [] for k in keys}
    device = data_list[0].pos.device
    slices = {key: [0] for key in keys}
    cumsum = {key: [0] for key in keys}
    cat_dims = {}
    natoms_list = []

    for data in data_list:
        assert (
            data.num_graphs == 1
        ), "data list must only contain single-graph AtomicData objects."

        for key in keys:
            item = data[key]
            reference = data_list[0][key]
            if torch.is_tensor(reference):
                if (
                    not torch.is_tensor(item)
                    or item.dtype != reference.dtype
                    or item.device != reference.device
                    or item.ndim != reference.ndim
                ):
                    raise ValueError(
                        f"Incompatible dtype, device or rank for batched field '{key}'"
                    )
                if item.device != device:
                    raise ValueError(
                        f"Batched tensor field '{key}' must be on the positions device"
                    )
            elif type(item) is not type(reference):
                raise TypeError(
                    f"Incompatible metadata types for batched field '{key}'"
                )

            # Increase values by `cumsum` value.
            cum = cumsum[key][-1]
            if isinstance(item, torch.Tensor) and item.dtype != torch.bool:
                if not isinstance(cum, int) or cum != 0:
                    item = item + cum
            elif isinstance(item, (int, float)) and cum != 0:
                item = item + cum

            # Gather the size of the `cat` dimension.
            size = 1
            cat_dim = data.__cat_dim__(key, data[key])
            if not torch.is_tensor(item):
                cat_dim = (
                    "list"
                    if isinstance(item, list)
                    else "tuple"
                    if type(item) is tuple
                    else "object"
                )
                size = len(item) if cat_dim in ("list", "tuple") else 1
            # 0-dimensional torch.Tensors have no dimension along which to
            # concatenate, so we set `cat_dim` to `None`.
            if isinstance(item, torch.Tensor) and item.dim() == 0:
                cat_dim = None
            cat_dims[key] = cat_dim

            # Add a batch dimension to items whose `cat_dim` is `None`:
            if isinstance(item, torch.Tensor) and cat_dim is None:
                cat_dim = 0  # Concatenate along this new batch dimension.
                item = item.unsqueeze(0)
            elif isinstance(item, torch.Tensor):
                size = item.size(cat_dim)

            batched_data_dict[key].append(item)  # Append item to the attribute list.

            slices[key].append(size + slices[key][-1])
            inc = data.__inc__(key, item)
            if isinstance(inc, (tuple, list)):
                inc = torch.tensor(inc)
            cumsum[key].append(inc + cumsum[key][-1])

        natoms_list.append(data.num_nodes)

    for key in keys:
        if key == "batch":
            continue
        items = batched_data_dict[key]
        item = items[0]
        cat_dim = cat_dims[key]
        cat_dim = 0 if cat_dim is None else cat_dim
        if torch.is_tensor(item):
            batched_data_dict[key] = torch.cat(items, cat_dim)

        # TODO: this allows non-tensor fields to be batched.
        # we might want to remove support for that.
        else:
            # For list attributes, flatten nested lists to maintain consistency
            # This handles cases where already-batched data is re-batched
            if cat_dim in ("list", "tuple"):
                flattened = []
                for item in items:
                    flattened.extend(item)
                batched_data_dict[key] = _clone_tensor_tree(flattened)
            else:
                batched_data_dict[key] = _clone_tensor_tree(items)

    if device.type == "cpu":
        # NumPy handles these small CPU metadata arrays without a parallel
        # PyTorch launch (which can dominate a two-structure collation).
        batched_data_dict["batch"] = torch.from_numpy(
            np.repeat(np.arange(len(data_list), dtype=np.int64), natoms_list)
        )
    else:
        batched_data_dict["batch"] = torch.repeat_interleave(
            torch.arange(len(data_list), dtype=torch.long, device=device),
            torch.tensor(natoms_list, dtype=torch.long, device=device),
            output_size=sum(natoms_list),
        )
    atomic_data_batch = AtomicData.from_dict(batched_data_dict)
    atomic_data_batch.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)

    return atomic_data_batch.contiguous()


def tensor_or_int_to_tensor(x, dtype=torch.int):
    if isinstance(x, Integral) and not isinstance(x, (bool, np.bool_)):
        return torch.tensor([int(x)], dtype=dtype)
    elif isinstance(x, torch.Tensor):
        assert x.dtype == dtype, "Tensor is not of right dtype"
        return x
    raise ValueError(f"type({x}) is not an int or tensor")

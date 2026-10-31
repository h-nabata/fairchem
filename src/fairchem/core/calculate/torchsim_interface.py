"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import threading
import typing
from pathlib import Path
from typing import Any

import torch

from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import UMATask
from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    CHARGE_RANGE,
    DEFAULT_SPIN,
    DEFAULT_SPIN_OMOL,
    SPIN_RANGE,
    InferenceSettings,
)

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.transforms import pbc_wrap_batched
except ImportError:
    ts = None  # type: ignore[assignment]
    ModelInterface = None  # type: ignore[assignment]
    pbc_wrap_batched = None  # type: ignore[assignment]


if typing.TYPE_CHECKING:
    from torch_sim import SimState

# Use object as fallback base class if ModelInterface is not available
# The __init__ method will raise ImportError if torch-sim is not installed
_TSModelInterface = ModelInterface if ModelInterface is not None else object


class _SimStateConverter:
    """
    Cache validated, immutable batch metadata while refreshing geometry each call.

    Cache keys retain the current tensors and check PyTorch's mutation versions,
    so allocator address reuse cannot produce a false hit. Owned value snapshots
    also detect writes through NumPy/DLPack or ``.data`` that bypass mutation
    counters. Returned batches never expose the private cached template.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._owners = None
        self._versions = None
        self._signature = None
        self._template = None
        self._snapshots = None
        self._periodic = False
        self._ready_event = None
        self._cache_device = None

    def clear(self):
        with self._lock:
            self._owners = None
            self._versions = None
            self._signature = None
            self._template = None
            self._snapshots = None
            self._ready_event = None
            self._cache_device = None

    def __call__(self, state, task_name, target_dtype):
        with self._lock:
            return self._convert(state, task_name, target_dtype)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    def _convert(self, state, task_name, target_dtype):
        if state.positions.dtype not in (torch.float32, torch.float64):
            raise ValueError("TorchSim positions must use float32 or float64")
        if state.cell is not None and state.cell.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise ValueError("TorchSim cell must use float32 or float64")
        if (
            self._ready_event is not None
            and self._cache_device == state.positions.device
        ):
            backend = getattr(torch, self._cache_device.type)
            stream = backend.current_stream(self._cache_device)
            stream.wait_event(self._ready_event)
            for value in (
                *self._owners,
                *self._snapshots,
                *vars(self._template).values(),
            ):
                if torch.is_tensor(value):
                    value.record_stream(stream)
        owners = (
            state.atomic_numbers,
            state.system_idx,
            state.pbc,
            state.charge if state.has_extras("charge") else None,
            state.spin if state.has_extras("spin") else None,
        )
        if any(value is not None and not torch.is_tensor(value) for value in owners):
            self.clear()
            return _simstate_to_atomicdata_batch(state, task_name, target_dtype)
        signature = (
            state.positions.shape,
            state.positions.device,
            target_dtype,
            task_name,
            tuple(
                (state.has_extras(name), name in state.system_extras)
                for name in ("charge", "spin")
            ),
            None if state.cell is None else (state.cell.shape, state.cell.device),
        )
        try:
            versions = tuple(t._version if t is not None else None for t in owners)
        except RuntimeError:
            # Inference tensors do not expose mutation counters. Revalidate rather
            # than treating an unobservable mutation as immutable metadata.
            self.clear()
            return _simstate_to_atomicdata_batch(state, task_name, target_dtype)
        same_owners = self._owners is not None and all(
            old is new for old, new in zip(self._owners, owners)
        )
        hit = (
            self._template is not None
            and self._signature == signature
            and (not same_owners or self._versions == versions)
            and all(
                (old is None and new is None)
                or (
                    old is not None
                    and new is not None
                    and old.dtype == new.dtype
                    and old.device == new.device
                    and torch.equal(old, new)
                )
                for old, new in zip(self._snapshots, owners)
            )
        )
        if not hit:
            batch = _simstate_to_atomicdata_batch(state, task_name, target_dtype)
            # Keep owned static values only. Never retain an old coordinate/cell
            # tensor or its autograd graph in the metadata cache.
            memo = {}
            for key, value in vars(batch).items():
                if isinstance(value, torch.Tensor):
                    memo[id(value)] = (
                        torch.empty(0, device=value.device, dtype=value.dtype)
                        if key in {"pos", "cell"}
                        else value.detach().clone()
                    )
            self._template = copy.deepcopy(batch, memo)
            self._snapshots = tuple(
                value.detach().clone() if value is not None else None
                for value in owners
            )
            self._owners = owners
            self._versions = versions
            self._signature = signature
            self._periodic = bool(batch.pbc.any())
            self._cache_device = state.positions.device
            self._ready_event = None
            if self._cache_device.type in {"xpu", "cuda"}:
                backend = getattr(torch, self._cache_device.type)
                self._ready_event = backend.Event()
                self._ready_event.record(backend.current_stream(self._cache_device))
            return batch

        # Integrators may clone a state while retaining the same composition.
        # Exact snapshot comparison permits reuse without rebuilding metadata;
        # replace owners so the cache does not pin an obsolete state's storage.
        self._owners = owners
        self._versions = versions
        positions = state.positions.to(target_dtype)
        n_systems = self._template.natoms.numel()
        cell = (
            state.row_vector_cell.to(target_dtype)
            if state.cell is not None
            else torch.eye(3, device=positions.device, dtype=target_dtype)
            .unsqueeze(0)
            .expand(n_systems, -1, -1)
        )
        _validate_geometry(positions, cell, n_systems, self._periodic)
        if self._periodic:
            positions = pbc_wrap_batched(
                positions,
                cell.transpose(-2, -1),
                self._template.batch,
                self._template.pbc[0],
            )
        batch = copy.deepcopy(self._template)
        batch.pos = positions.clone(memory_format=torch.contiguous_format)
        batch.cell = cell.clone(memory_format=torch.contiguous_format)
        return batch


def _integer_metadata(value, name, size, device, minimum=None, maximum=None):
    """
    Validate integral values before casting, without truncating physical metadata.
    """
    if not torch.is_tensor(value) or value.shape != (size,):
        raise ValueError(f"{name} must be a tensor with shape ({size},)")
    if value.device != device:
        raise ValueError(f"{name} must be on the same device as positions")
    if value.dtype == torch.bool or value.is_complex():
        raise TypeError(f"{name} must contain integer-valued numbers")
    if value.is_floating_point() and (
        not torch.isfinite(value).all() or not torch.equal(value, value.round())
    ):
        raise ValueError(f"{name} must contain finite integer-valued numbers")
    if minimum is not None and torch.any(value < minimum):
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and torch.any(value > maximum):
        raise ValueError(f"{name} must be at most {maximum}")
    return value.to(dtype=torch.long, copy=True)


def _system_integer_extra(state, name, n_systems, default, bounds):
    """
    Charge and spin are per-system values, never per-atom extras.
    """
    if not state.has_extras(name):
        return torch.full(
            (n_systems,), default, dtype=torch.long, device=state.positions.device
        )
    if name not in state.system_extras:
        raise ValueError(f"{name} must be a per-system extra")
    return _integer_metadata(
        getattr(state, name), name, n_systems, state.positions.device, *bounds
    )


def _validate_geometry(positions, cell, n_systems, periodic):
    """
    Reject malformed geometry before graph construction or periodic wrapping.
    """
    if not torch.isfinite(positions).all():
        raise ValueError("positions must contain finite values after dtype conversion")
    if cell.shape != (n_systems, 3, 3) or cell.device != positions.device:
        raise ValueError(
            "cell must have shape (n_systems, 3, 3) on the positions device"
        )
    if not torch.isfinite(cell).all():
        raise ValueError("cell must contain finite values after dtype conversion")
    if periodic and torch.any(torch.linalg.det(cell) == 0):
        raise ValueError("Periodic wrapping requires a nonsingular cell")


def _simstate_to_atomicdata_batch(
    sim_state: ts.SimState,
    task_name: UMATask | str | None,
    target_dtype: torch.dtype = torch.float32,
) -> AtomicData:
    """Convert a batched SimState directly to a batched AtomicData in a single vectorized operation.

    Args:
        sim_state: A SimState containing one or more systems
        task_name: Task name for UMA models
        target_dtype: Target dtype for tensors

    Returns:
        Batched AtomicData object
    """
    if target_dtype not in (torch.float32, torch.float64):
        raise ValueError("TorchSim conversion requires float32 or float64")
    if sim_state.positions.dtype not in (torch.float32, torch.float64):
        raise ValueError("TorchSim positions must use float32 or float64")
    if sim_state.cell is not None and sim_state.cell.dtype not in (
        torch.float32,
        torch.float64,
    ):
        raise ValueError("TorchSim cell must use float32 or float64")
    positions = sim_state.positions.to(dtype=target_dtype, copy=True)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("positions must have nonempty shape (n_atoms, 3)")
    if not torch.isfinite(positions).all():
        raise ValueError("positions must contain finite values after dtype conversion")
    atomic_numbers = _integer_metadata(
        sim_state.atomic_numbers,
        "atomic_numbers",
        len(positions),
        positions.device,
        1,
        118,
    )

    n_systems = sim_state.n_systems
    batch_indices = _integer_metadata(
        sim_state.system_idx,
        "system_idx",
        len(positions),
        positions.device,
        0,
        n_systems - 1,
    )
    n_atoms_per_system = torch.bincount(batch_indices, minlength=n_systems)
    expected_indices = torch.repeat_interleave(
        torch.arange(n_systems, device=positions.device), n_atoms_per_system
    )
    if not torch.equal(batch_indices, expected_indices) or torch.any(
        n_atoms_per_system == 0
    ):
        raise ValueError(
            "system_idx must group atoms in ascending contiguous system order"
        )

    if sim_state.cell is not None:
        cell = sim_state.row_vector_cell.to(target_dtype)  # (n_systems, 3, 3)
    else:
        # Create identity cells for molecules
        cell = (
            torch.eye(3, dtype=target_dtype, device=positions.device)
            .unsqueeze(0)
            .expand(n_systems, -1, -1)
        )

    if sim_state.pbc.device != positions.device:
        raise ValueError("pbc must be on the same device as positions")
    if sim_state.pbc.shape != (3,) or torch.any(
        (sim_state.pbc != 0) & (sim_state.pbc != 1)
    ):
        raise ValueError("TorchSim pbc must contain three boolean values")
    pbc = sim_state.pbc.bool().unsqueeze(0).expand(n_systems, 3).clone()
    _validate_geometry(positions, cell, n_systems, bool(pbc.any()))

    if torch.any(pbc):
        cell_col = cell.transpose(-2, -1)  # (n_systems, 3, 3) column vectors
        pbc_single = pbc[0] if pbc.ndim > 0 else pbc  # (3,) or bool
        positions = pbc_wrap_batched(
            positions, cell_col, sim_state.system_idx, pbc_single
        )

    natoms = n_atoms_per_system  # (n_systems,)

    edge_index = torch.empty((2, 0), dtype=torch.long, device=positions.device)
    cell_offsets = torch.empty((0, 3), dtype=target_dtype, device=positions.device)
    nedges = torch.zeros(n_systems, dtype=torch.long, device=positions.device)

    charge = _system_integer_extra(sim_state, "charge", n_systems, 0, CHARGE_RANGE)
    spin = _system_integer_extra(
        sim_state,
        "spin",
        n_systems,
        DEFAULT_SPIN_OMOL if task_name == UMATask.OMOL else DEFAULT_SPIN,
        SPIN_RANGE,
    )

    fixed = torch.zeros_like(atomic_numbers, dtype=torch.long)
    tags = torch.zeros_like(atomic_numbers, dtype=torch.long)

    batched_dict = {
        "pos": positions,
        "atomic_numbers": atomic_numbers,
        "cell": cell.clone(),
        "pbc": pbc,
        "natoms": natoms,
        "edge_index": edge_index,
        "cell_offsets": cell_offsets,
        "nedges": nedges,
        "charge": charge,
        "spin": spin,
        "fixed": fixed,
        "tags": tags,
        "batch": batch_indices,
        "sid": [""] * n_systems,  # Empty string IDs for each system
    }

    if task_name is not None:
        task_name_str = task_name if isinstance(task_name, str) else task_name.value
        batched_dict["dataset"] = [task_name_str] * n_systems

    atomic_data_batch = AtomicData.from_dict(batched_dict)

    natoms_list = n_atoms_per_system.tolist()
    n_atoms_cumsum = torch.cumsum(n_atoms_per_system, dim=0).tolist()

    slices = {}
    cumsum = {}
    cat_dims = {}

    for key, value in batched_dict.items():
        if key == "batch" or not isinstance(value, torch.Tensor):
            continue

        cat_dims[key] = atomic_data_batch.__cat_dim__(key, value) or 0

        is_edge_key = "index" in key or key == "cell_offsets"
        if is_edge_key:
            slices[key] = [0] * (n_systems + 1)
            cumsum[key] = [0] + n_atoms_cumsum
        elif key in ("cell", "pbc", "charge", "spin", "natoms", "nedges"):
            slices[key] = list(range(n_systems + 1))
            cumsum[key] = [0] * (n_systems + 1)
        else:
            slices[key] = [0] + n_atoms_cumsum
            cumsum[key] = [0] * (n_systems + 1)

    atomic_data_batch.assign_batch_stats(slices, cumsum, cat_dims, natoms_list)

    return atomic_data_batch.contiguous()


class FairChemModel(_TSModelInterface):
    """FairChem model wrapper for computing atomistic properties.

    Wraps FairChem models to compute energies, forces, and stresses. Can be
    initialized with a model checkpoint path or pretrained model name.

    Uses the fairchem-core-2.2.0+ predictor API for batch inference.

    Attributes:
        predictor: The FairChem predictor for batch inference
        task_name (UMATask): Task type for the model
        _device (torch.device): Device where computation is performed
        _dtype (torch.dtype): Data type used for computation
        _compute_stress (bool): Whether to compute stress tensor
        implemented_properties (list): Model outputs the model can compute

    Examples:
        >>> model = FairChemModel(model="path/to/checkpoint.pt", compute_stress=True)
        >>> results = model(state)
    """

    def __init__(
        self,
        model: str | Path,
        *,  # force remaining arguments to be keyword-only
        model_cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        compute_stress: bool = False,
        task_name: UMATask | str | None = None,
        cache_state_metadata: bool = True,
        inference_settings: InferenceSettings | str = "batch",
        overrides: dict | None = None,
    ) -> None:
        """Initialize the FairChem model.

        Args:
            model (str | Path): Either a pretrained model name or path to model
                checkpoint file. The function will first check if the input matches
                a known pretrained model name, then check if it's a valid file path.
            neighbor_list_fn (Callable | None): Function to compute neighbor lists
                (not currently supported)
            model_cache_dir (str | Path | None): Path where to save the model
            device (str | torch.device | None): Device to use for computation. If
                None, prefers CUDA, then XPU, then CPU. Specify ``xpu:<index>``
                explicitly on hosts with both NVIDIA and Intel GPUs.
            dtype (torch.dtype | None): Simulation input/output dtype (float32 or
                float64). Network precision is selected by inference_settings.
            compute_stress (bool): Whether to compute stress tensor
            task_name (UMATask | str | None): Task type for UMA models (optional,
                only needed for UMA models)
            cache_state_metadata: Reuse validated composition and batch metadata
                across steps, checking owned snapshots for external mutations.
            inference_settings: Predictor startup configuration, including optional
                series_b settings. Defaults to the heterogeneous "batch" preset.
            overrides: Optional overrides applied to the predictor startup settings.

        Raises:
            ImportError: If torch-sim is not installed
            NotImplementedError: If custom neighbor list function is provided
            ValueError: If model is not a known model name or valid file path
        """
        if ts is None or ModelInterface is None:
            raise ImportError(
                "torch-sim is required to use FairChemModel. "
                + "Install it with: pip install fairchem-core[torchsim]"
            )

        setup_imports()
        setup_logging()
        super().__init__()

        self._dtype = dtype or torch.float32
        if self._dtype not in (torch.float32, torch.float64):
            raise ValueError("FairChemModel dtype must be float32 or float64")
        self._compute_stress = compute_stress
        self._compute_forces = True
        self._state_converter = _SimStateConverter() if cache_state_metadata else None
        self._memory_scales_with = "n_atoms"  # TODO: this does vary with model type

        # Convert Path to string for consistency
        model = str(model) if isinstance(model, Path) else model
        model_cache_dir = (
            str(model_cache_dir)
            if isinstance(model_cache_dir, Path)
            else model_cache_dir
        )

        # Keep arbitrary fine-tuning dataset names as strings.
        if isinstance(task_name, str) and task_name in set(UMATask):
            task_name = UMATask(task_name)

        # Use the efficient predictor API for optimal performance
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        self._device = torch.device(device)
        device_str = str(self._device)
        self.task_name = task_name

        # Create efficient batch predictor for fast inference. TorchSim sends
        # batches of systems, so use the batch preset rather than the
        # single-system default (Nitro).
        loader_kwargs = (
            {} if model_cache_dir is None else {"cache_dir": model_cache_dir}
        )
        self.predictor = pretrained_mlip.get_predict_unit(
            model,
            inference_settings=inference_settings,
            overrides=overrides,
            device=device_str,
            **loader_kwargs,
        )

        if self._device.type in {"xpu", "cuda"}:
            # The predictor resolves a bare accelerator to a concrete ordinal. Keep
            # state transfers on that GPU even if the current device changes.
            self._device = torch.device(self.predictor.device)

        dataset_to_tasks = getattr(self.predictor, "dataset_to_tasks", None)
        if dataset_to_tasks is not None:
            if self.task_name is None:
                if len(dataset_to_tasks) != 1:
                    raise ValueError(
                        "task_name is required when the predictor supports multiple datasets"
                    )
                self.task_name = next(iter(dataset_to_tasks))
            elif self.task_name not in dataset_to_tasks:
                raise ValueError(
                    f"Unknown task_name {self.task_name!r}; supported datasets: "
                    f"{list(dataset_to_tasks)}"
                )

        self.implemented_properties = ["energy", "forces"]
        if compute_stress:
            self.implemented_properties.append("stress")

    @property
    def dtype(self) -> torch.dtype:
        """Return the data type used by the model."""
        return self._dtype

    @property
    def device(self) -> torch.device:
        """Return the device where the model is located."""
        return self._device

    def forward(self, state: SimState, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and other properties.

        Args:
            state (SimState | StateDict): State object containing positions, cells,
                atomic numbers, and other system information. If a dictionary is provided,
                it will be converted to a SimState.

        Returns:
            dict: Dictionary of model predictions, which may include:
                - energy (torch.Tensor): Energy with shape [batch_size]
                - forces (torch.Tensor): Forces with shape [n_atoms, 3]
                - stress (torch.Tensor): Stress tensor with shape [batch_size, 3, 3]
        """
        if isinstance(state, dict):
            state = ts.SimState(**state)
        if not isinstance(state, ts.SimState):
            raise TypeError(
                "state must be a torch_sim.SimState or its field dictionary"
            )
        # Convert SimState to batched AtomicData in a single vectorized operation
        converter = self._state_converter or _simstate_to_atomicdata_batch
        # Validate before transfer: SimState.to() coerces atomic numbers to int
        # and floating extras/cells to the positions dtype. That can silently
        # truncate invalid metadata or discard cell precision before validation.
        batch = converter(state, self.task_name, self._dtype)
        if batch.pos.device != self._device:
            batch = batch.to(self._device)
        if self._compute_stress and torch.any(torch.linalg.det(batch.cell) == 0):
            raise ValueError("Stress requires a nonsingular cell for every system")

        # Run efficient batch prediction
        predict_kwargs = (
            {"requested_properties": tuple(self.implemented_properties)}
            if getattr(self.predictor, "supports_requested_properties", False)
            else {}
        )
        predictions = self.predictor.predict(batch, **predict_kwargs)

        # Convert predictions to torch-sim format
        results: dict[str, torch.Tensor] = {}
        expected_shapes = {
            "energy": (len(batch.natoms),),
            "forces": (len(batch.pos), 3),
            "stress": (len(batch.natoms), 3, 3),
        }
        for name in self.implemented_properties:
            if name not in predictions:
                raise ValueError(
                    f"Predictor did not return requested property {name!r}"
                )
            value = predictions[name]
            if not torch.is_tensor(value) or not value.is_floating_point():
                raise TypeError(f"Predictor {name!r} must be a floating-point tensor")
            if name == "stress" and value.shape == (len(batch.natoms), 9):
                value = value.reshape(expected_shapes[name])
            if value.shape != expected_shapes[name]:
                raise ValueError(
                    f"Predictor {name!r} has shape {tuple(value.shape)}, "
                    f"expected {expected_shapes[name]}"
                )
            value = value.detach().to(device=self._device, dtype=self._dtype, copy=True)
            if not torch.isfinite(value).all():
                raise ValueError(f"Predictor {name!r} contains nonfinite values")
            results[name] = value
        return results

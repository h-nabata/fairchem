"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import os
import typing
from pathlib import Path
from typing import Any

import torch

from fairchem.core import pretrained_mlip
from fairchem.core.calculate.ase_calculator import UMATask
from fairchem.core.common.utils import setup_imports, setup_logging
from fairchem.core.datasets.atomic_data import AtomicData

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

    Cache keys retain the original tensors and check PyTorch's mutation versions,
    so allocator address reuse cannot produce a false hit. Normal in-place tensor
    updates, replacement, device/dtype changes and charge/spin changes invalidate
    the template. Tensor mutation through ``.data`` or foreign raw pointers is not
    supported; disable metadata caching for such callers.
    """

    def __init__(self):
        self._owners = None
        self._versions = None
        self._signature = None
        self._template = None
        self._periodic = False

    def clear(self):
        self._owners = None
        self._versions = None
        self._signature = None
        self._template = None

    def __call__(self, state, task_name, target_dtype):
        owners = (
            state.atomic_numbers,
            state.system_idx,
            state.pbc,
            state.charge if state.has_extras("charge") else None,
            state.spin if state.has_extras("spin") else None,
        )
        signature = (
            state.positions.shape,
            state.positions.device,
            target_dtype,
            task_name,
            None if state.cell is None else (state.cell.shape, state.cell.device),
        )
        try:
            versions = tuple(t._version if t is not None else None for t in owners)
        except RuntimeError:
            # Inference tensors do not expose mutation counters. Revalidate rather
            # than treating an unobservable mutation as immutable metadata.
            self.clear()
            return _simstate_to_atomicdata_batch(state, task_name, target_dtype)
        hit = (
            self._template is not None
            and self._signature == signature
            and self._versions == versions
            and all(old is new for old, new in zip(self._owners, owners))
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
            self._owners = owners
            self._versions = versions
            self._signature = signature
            self._periodic = bool(batch.pbc.any())
            return batch

        positions = state.positions.to(target_dtype)
        n_systems = self._template.natoms.numel()
        cell = (
            state.row_vector_cell.to(target_dtype)
            if state.cell is not None
            else torch.eye(3, device=positions.device, dtype=target_dtype)
            .unsqueeze(0)
            .expand(n_systems, -1, -1)
        )
        if self._periodic:
            positions = pbc_wrap_batched(
                positions,
                cell.transpose(-2, -1),
                self._template.batch,
                self._template.pbc[0],
            )
        batch = copy.copy(self._template)
        batch.__dict__ = self._template.__dict__.copy()
        batch.__keys__ = self._template.__keys__.copy()
        batch.pos = positions.contiguous()
        batch.cell = cell.contiguous()
        return batch


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
    positions = sim_state.positions.to(target_dtype)
    atomic_numbers = sim_state.atomic_numbers.long()

    n_atoms_per_system = torch.bincount(sim_state.system_idx)
    n_systems = sim_state.n_systems

    if sim_state.cell is not None:
        cell = sim_state.row_vector_cell.to(target_dtype)  # (n_systems, 3, 3)
    else:
        # Create identity cells for molecules
        cell = (
            torch.eye(3, dtype=target_dtype, device=positions.device)
            .unsqueeze(0)
            .expand(n_systems, -1, -1)
        )

    pbc = sim_state.pbc.bool()  # Already a tensor, shape (3,)
    pbc = pbc.unsqueeze(0).expand(n_systems, 3)  # (n_systems, 3)

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

    charge = (
        sim_state.charge.long()
        if sim_state.has_extras("charge")
        else torch.zeros(n_systems, dtype=torch.long, device=positions.device)
    )
    spin = (
        sim_state.spin.long()
        if sim_state.has_extras("spin")
        else torch.zeros(n_systems, dtype=torch.long, device=positions.device)
    )

    fixed = torch.zeros_like(atomic_numbers, dtype=torch.long)
    tags = torch.zeros_like(atomic_numbers, dtype=torch.long)

    batch_indices = sim_state.system_idx.long()

    batched_dict = {
        "pos": positions,
        "atomic_numbers": atomic_numbers,
        "cell": cell,
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
            dtype (torch.dtype | None): Data type to use for computation
            compute_stress (bool): Whether to compute stress tensor
            task_name (UMATask | str | None): Task type for UMA models (optional,
                only needed for UMA models)
            cache_state_metadata: Reuse validated composition and batch metadata
                across steps. Disable if metadata is changed via raw pointers or
                tensor .data operations that bypass mutation counters.

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

        # Convert task_name to UMATask if it's a string (only for UMA models)
        if isinstance(task_name, str):
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
        # XPU ordinals are part of the predictor's device contract. Dropping
        # the index here sends a state on xpu:1 to the current/default GPU.
        # Keep the existing local-rank CUDA resolution in the predictor.
        device_str = (
            str(self._device) if self._device.type == "xpu" else self._device.type
        )
        self.task_name = task_name

        # Create efficient batch predictor for fast inference. TorchSim sends
        # batches of systems, so use the batch preset rather than the
        # single-system default (Nitro).
        if model_cache_dir and os.path.exists(model_cache_dir):
            self.predictor = pretrained_mlip.get_predict_unit(
                model,
                inference_settings="batch",
                device=device_str,
                cache_dir=model_cache_dir,
            )
        else:
            self.predictor = pretrained_mlip.get_predict_unit(
                model, inference_settings="batch", device=device_str
            )

        if self._device.type == "xpu":
            # The predictor resolves a bare "xpu" to a concrete ordinal. Keep
            # state transfers on that GPU even if the current device changes.
            self._device = torch.device(self.predictor.device)

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
        if state.device != self._device:
            state = state.to(self._device)

        # Convert SimState to batched AtomicData in a single vectorized operation
        converter = self._state_converter or _simstate_to_atomicdata_batch
        # state is already resident on the selected device. The converter updates
        # differentiable positions/cells but can reuse immutable batch metadata.
        batch = converter(state, self.task_name, self._dtype)

        # Run efficient batch prediction
        predict_kwargs = (
            {"requested_properties": tuple(self.implemented_properties)}
            if getattr(self.predictor, "supports_requested_properties", False)
            else {}
        )
        predictions = self.predictor.predict(batch, **predict_kwargs)

        # Convert predictions to torch-sim format
        results: dict[str, torch.Tensor] = {}
        results["energy"] = predictions["energy"].to(dtype=self._dtype)
        results["forces"] = predictions["forces"].to(dtype=self._dtype)

        # Handle stress if requested and available
        if self._compute_stress and "stress" in predictions:
            stress = predictions["stress"].to(dtype=self._dtype)
            # Ensure stress has correct shape [batch_size, 3, 3]
            if stress.dim() == 2 and stress.shape[0] == batch.num_graphs:
                stress = stress.view(-1, 3, 3)
            results["stress"] = stress

        return {k: v.detach() for k, v in results.items()}

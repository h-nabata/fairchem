"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
import logging
from collections import Counter
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from ase.calculators.calculator import (
    Calculator,
    PropertyNotImplementedError,
    all_changes,
    equal,
)
from ase.constraints import FixAtoms
from ase.stress import full_3x3_to_voigt_6_stress

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    UMATask,
)

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.units.mlip_unit import MLIPPredictUnit


def _metadata_equal(first, second, seen=None):
    """
    Compare ASE metadata without ndarray truth-value coercion or cycle recursion.
    """
    if seen is None:
        seen = set()
    pair = (id(first), id(second))
    if pair in seen:
        return True
    if isinstance(first, dict) and isinstance(second, dict):
        seen.add(pair)
        return first.keys() == second.keys() and all(
            _metadata_equal(first[key], second[key], seen) for key in first
        )
    if isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
        seen.add(pair)
        return len(first) == len(second) and all(
            _metadata_equal(a, b, seen) for a, b in zip(first, second)
        )
    try:
        return equal(first, second)
    except (TypeError, ValueError, RuntimeError):
        return False


def _owned_prediction_array(value, name):
    """
    Copy prediction storage across the Tensor/NumPy calculator boundary.
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.array(value, copy=True)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError(f"Prediction {name!r} must contain finite real numbers")
    return array


def _calculator_predictions(pred, properties, natoms):
    """
    Normalize ASE output shapes without exposing predictor buffers.
    """
    results = {}
    for name in properties:
        if name not in pred:
            continue
        value = _owned_prediction_array(pred[name], name)
        if name == "energy":
            if value.size != 1:
                raise ValueError(
                    f"Energy must contain one value, got shape {value.shape}"
                )
            results["energy"] = results["free_energy"] = float(value.item())
        elif name == "forces":
            if value.shape != (natoms, 3):
                raise ValueError(
                    f"Forces must have shape ({natoms}, 3), got {value.shape}"
                )
            results[name] = value
        elif name == "stress":
            if value.shape in ((6,), (1, 6)):
                results[name] = value.reshape(6)
            elif value.shape in ((9,), (1, 9), (3, 3), (1, 3, 3)):
                results[name] = full_3x3_to_voigt_6_stress(value.reshape(3, 3))
            else:
                raise ValueError(
                    f"Stress must be a 3x3 tensor or six Voigt components, got {value.shape}"
                )
        elif name == "hessian":
            expected = (3 * natoms, 3 * natoms)
            if value.shape == (1, *expected):
                value = value[0]
            if value.shape != expected:
                raise ValueError(
                    f"Hessian must have shape {expected}, got {value.shape}"
                )
            results[name] = value
        else:
            results[name] = value.item() if value.ndim == 0 else value
    return results


def _predictor_graph_options(predict_unit):
    """Resolve the same model graph and precision contract for ASE and LAMMPS."""
    settings = getattr(predict_unit, "inference_settings", None)
    graph = getattr(predict_unit, "graph_settings", None)
    if not isinstance(graph, dict):
        backbone = getattr(
            getattr(getattr(predict_unit, "model", None), "module", None),
            "backbone",
            None,
        )
        graph = {
            name: getattr(backbone, name)
            for name in (
                "cutoff",
                "max_neighbors",
                "otf_graph",
                "enforce_max_neighbors_strictly",
            )
            if hasattr(backbone, name)
            and isinstance(getattr(backbone, name), (float, int, bool))
        }
    otf_graph = graph.get("otf_graph")
    external = getattr(settings, "external_graph_gen", False) is True
    r_edges = external or otf_graph is False
    if external and otf_graph is True:
        raise ValueError(
            "external_graph_gen=True conflicts with backbone otf_graph=True"
        )
    radius = graph.get("cutoff", 6.0) if r_edges else 6.0
    max_neigh = graph.get("max_neighbors", 300) if r_edges else None

    return {
        "r_edges": r_edges,
        "max_neigh": max_neigh,
        "radius": radius,
        "enforce_max_neighbors_strictly": graph.get(
            "enforce_max_neighbors_strictly", True
        ),
        "target_dtype": getattr(settings, "base_precision_dtype", torch.float32),
    }


def _fixed_mask(atoms):
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[constraint.index] = True
    return mask


def _model_input_changes(previous, atoms):
    if not _metadata_equal(previous.info, atoms.info):
        return ["info"]
    if ("tags" in atoms.arrays or "tags" in previous.arrays) and not np.array_equal(
        previous.get_tags(), atoms.get_tags()
    ):
        return ["tags"]
    if (previous.constraints or atoms.constraints) and not np.array_equal(
        _fixed_mask(previous), _fixed_mask(atoms)
    ):
        return ["constraints"]
    return []


def _child_cache_snapshot(calculator):
    # ASE replaces its stored Atoms on calculate/reset. Keep the identity so
    # an independent evaluation of a shared child also invalidates the parent.
    return calculator.atoms, copy.deepcopy(calculator.parameters)


def _child_cache_changes(calculator, snapshot, atoms, tol):
    changes = calculator.check_state(atoms, tol=tol)
    if changes:
        return changes
    if snapshot is None:
        return ["parameters"]
    previous_atoms, parameters = snapshot
    if calculator.atoms is not previous_atoms or not _metadata_equal(
        parameters, calculator.parameters
    ):
        return ["parameters"]
    return []


class FAIRChemCalculator(Calculator):
    def __init__(
        self,
        predict_unit: MLIPPredictUnit,
        task_name: UMATask | str | None = None,
        seed: int | None = None,  # deprecated
    ):
        """
        Initialize the FAIRChemCalculator from a model MLIPPredictUnit

        Args:
            predict_unit (MLIPPredictUnit): A pretrained MLIPPredictUnit.
            task_name (UMATask or str, optional): Name of the task to use if using a UMA checkpoint.
                Determines default key names for energy, forces, and stress.
                Can be one of 'omol', 'omat', 'oc20', 'odac', or 'omc'.
        Notes:
            - For models that require total charge and spin multiplicity (currently UMA models on omol mode), `charge`
              and `spin` (corresponding to `spin_multiplicity`) are pulled from `atoms.info` during calculations.
                - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
                - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
                - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to
                charge=`0` and spin=`1`.
        """

        super().__init__()

        if seed is not None:
            logging.warning(
                "The 'seed' argument is deprecated and will be removed in future versions. "
                "Please set the seed in the MLIPPredictUnit configuration instead."
            )

        if isinstance(task_name, UMATask):
            task_name = task_name.value

        valid_datasets = list(predict_unit.dataset_to_tasks.keys())
        if task_name is not None:
            if task_name not in valid_datasets:
                raise ValueError(
                    f"Invalid task_name '{task_name}'. Valid options are {valid_datasets}"
                )
            self._task_name = task_name
        elif len(valid_datasets) == 1:
            self._task_name = valid_datasets[0]
        else:
            raise RuntimeError(
                f"A task name must be provided. Valid options are {valid_datasets}"
            )

        self.implemented_properties = [
            task.property for task in predict_unit.dataset_to_tasks[self.task_name]
        ]
        if "energy" in self.implemented_properties:
            self.implemented_properties.append(
                "free_energy"
            )  # Free energy is a copy of energy, see docstring above

        self.predictor = predict_unit

        self.a2g = partial(
            AtomicData.from_ase,
            task_name=self.task_name,
            r_data_keys=["spin", "charge"],
            **_predictor_graph_options(predict_unit),
        )

    @property
    def task_name(self) -> str:
        return self._task_name

    @classmethod
    def from_model_checkpoint(
        cls,
        name_or_path: str,
        task_name: UMATask | None = None,
        inference_settings: InferenceSettings | str = "default",
        overrides: dict | None = None,
        device: str | None = None,
        seed: int = 41,
        workers: int = 1,
    ) -> FAIRChemCalculator:
        """Instantiate a FAIRChemCalculator from a checkpoint file.

        Args:
            cls: The class reference
            name_or_path: A model name from fairchem.core.pretrained.available_models or a path to the checkpoint
                file
            task_name: Task name
            inference_settings: Settings for inference. "default" (the same as "nitro")
                runs UMA Nitro for uma-s-1p2p1 and its fine-tunes on Nitro's
                checkpoint allowlist on CUDA capability 8.x-12.x and "multigpu"
                otherwise. "nitro" and "multigpu" are reduced-precision modes;
                "multigpu" uses merge_mole + compile with TF32 matrix multiplies on
                CUDA. "high-precision" uses merge_mole + compile in full FP32.
                "multigpu" and "high-precision" fall back automatically if their
                fixed-input contract is broken. "batch" keeps MOLE unmerged for
                heterogeneous inputs. More advanced use cases can use a custom
                InferenceSettings object.
            overrides: Optional dictionary of settings to override default inference settings.
            device: Optional device ("cpu", "cuda", "xpu", or "xpu:<index>").
            seed: Random seed for reproducibility.
            workers: Number of parallel workers for prediction unit. Default is 1.
        """

        predict_unit = pretrained_mlip.get_predict_unit(
            name_or_path,
            inference_settings=inference_settings,
            overrides=overrides,
            device=device,
            workers=workers,
            seed=seed,
        )
        return cls(predict_unit=predict_unit, task_name=task_name)

    def check_state(self, atoms: Atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """
        state = super().check_state(atoms, tol=tol)
        if not state:
            state = _model_input_changes(self.atoms, atoms)
        return state

    def get_property(self, name, atoms=None, allow_calculation=True):
        try:
            result = super().get_property(
                name, atoms=atoms, allow_calculation=allow_calculation
            )
        except PropertyNotImplementedError as exc:
            msg = str(exc)
            if name in ("forces", "stress", "hessian"):
                msg += (
                    f"\n {name} prediction can be enabled by setting `predict_untrained_{name}={{{self.task_name!r}}}` "
                    f"in the InferenceSettings."
                )
            raise PropertyNotImplementedError(msg) from exc

        return result

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        """
        Perform the calculation for the given atomic structure.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.

        Notes:
            - `charge` must be an integer representing the total charge on the system and can range from -100 to 100.
            - `spin` must be an integer representing the spin multiplicity and can range from 0 to 100.
            - If `task_name="omol"`, and `charge` or `spin` are not set in `atoms.info`, they will default to `0`.
            - `charge` and `spin` are currently only used for the `omol` head.
            - The `free_energy` is simply a copy of the `energy` and is not the actual electronic free energy.
              It is only set for ASE routines/optimizers that are hard-coded to use this rather than the `energy` key.
        """

        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("No atoms supplied to the calculator")
        try:
            self._calculate(atoms, properties, system_changes)
        except BaseException:
            # Direct calculate_properties() bypasses get_property()'s normal
            # invalidation. Never leave old results on a new Atoms snapshot.
            self.reset()
            raise

    def _calculate(self, atoms, properties, system_changes):
        # Our calculators won't work if natoms=0
        if len(atoms) == 0:
            raise ValueError("Atoms object has no atoms inside.")

        # Check if the atoms object has periodic boundary conditions (PBC) set correctly
        if np.all(np.asarray(atoms.pbc)) and np.allclose(atoms.cell, 0):
            raise AllZeroUnitCellError

        # AtomicData completes missing cell vectors for nonperiodic molecules.
        # Only the original physical volume can normalize a stress.
        output_properties = self.implemented_properties
        if "stress" in properties or "stress" in output_properties:
            volume = atoms.cell.volume
            if not np.isfinite(volume) or volume <= 0:
                if "stress" in properties:
                    raise ValueError(
                        "Stress requires a cell with finite nonzero volume"
                    )
                output_properties = [
                    name for name in output_properties if name != "stress"
                ]

        self.predictor.validate_atoms_data(atoms, self.task_name)

        # Standard call to check system_changes etc
        Calculator.calculate(self, atoms, properties, system_changes)
        # ASE's Atoms.copy() shallow-copies info; own nested metadata too.
        self.atoms.info = copy.deepcopy(atoms.info)

        # Convert using the current a2g object
        data = self.a2g(atoms)

        # Compute requested derivatives only. Energy is cheap once forces have
        # been evaluated and retaining it avoids a second optimizer evaluation.
        if getattr(self.predictor, "supports_requested_properties", False):
            requested = {
                "energy" if name == "free_energy" else name for name in properties
            }
            if "energy" in self.implemented_properties:
                requested.add("energy")
            if (
                getattr(self.predictor.inference_settings, "nitro", False)
                and requested <= {"energy", "forces", "stress"}
                and not torch.is_inference_mode_enabled()
            ):
                # Nitro computes E/F/stress together. Retain its cheap extra
                # results so ASE integrators do not replay the same geometry
                # when they ask for stress or forces after energy. Generic
                # backends still compute only the requested derivatives.
                requested.update(
                    {"energy", "forces"}.intersection(self.implemented_properties)
                )
                if "stress" in output_properties:
                    requested.add("stress")
            pred = self.predictor.predict(
                data, requested_properties=tuple(sorted(requested))
            )
        else:
            pred = self.predictor.predict(data)

        results = {} if system_changes else dict(self.results)
        results.update(_calculator_predictions(pred, output_properties, len(atoms)))
        self.results = results

    def _check_atoms_pbc(self, atoms) -> None:
        """
        Check for invalid PBC conditions

        Args:
            atoms (ase.Atoms): The atomic structure to check.
        """


class FormationEnergyCalculator(Calculator):
    def __init__(
        self,
        calculator: Calculator,
        element_references: dict | None = None,
        apply_corrections: bool | None = None,
        correction_type: Literal["MP2020", "OMat24"] = "OMat24",
    ):
        """
        A calculator wrapper that computes formation energies. Assumes task naming matches UMA.

        Args:
            calculator (Calculator): The base calculator to wrap.
            element_references (dict, optional): Dictionary of formation reference energies for each element.
                If None and calculator is FAIRChemCalculator, uses default references from the predictor.
            apply_corrections (bool, optional): Whether to apply MP style corrections to formation energies.
                Only relevant for OMat task. Defaults to True for OMat task if calculator is FAIRChemCalculator.
            correction_type (Literal["MP2020", "OMat24"], optional): Type of corrections to apply. Defaults to "OMat24".
        """
        super().__init__()
        self.calculator = calculator

        if element_references is None:
            if isinstance(calculator, FAIRChemCalculator):
                element_references = calculator.predictor.form_elem_refs[
                    calculator.task_name
                ]
            else:
                raise ValueError("element_references must be provided")
        self.element_references = copy.deepcopy(element_references)
        self._reference_snapshot = None
        self._child_snapshot = None

        if apply_corrections is True:
            if isinstance(calculator, FAIRChemCalculator):
                if calculator.task_name != UMATask.OMAT.value:
                    raise ValueError(
                        "MP style corrections can only be applied for the OMat task."
                    )
            else:
                logging.warning(
                    "apply_corrections=True specified for non-FAIRChemCalculator. "
                    "Corrections will be attempted."
                )

        if (
            apply_corrections is None
            and isinstance(calculator, FAIRChemCalculator)
            and calculator.task_name == UMATask.OMAT.value
        ):
            apply_corrections = True

        self.apply_corrections = (
            apply_corrections if apply_corrections is not None else False
        )
        self._correction_type = correction_type

        if hasattr(calculator, "implemented_properties"):
            self.implemented_properties = list(calculator.implemented_properties)

    def reset(self):
        super().reset()
        self._child_snapshot = self._reference_snapshot = None

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if not state:
            state = _model_input_changes(self.atoms, atoms)
        configuration = (
            self.element_references,
            self.apply_corrections,
            self._correction_type,
        )
        if not state and not _metadata_equal(self._reference_snapshot, configuration):
            state.append("parameters")
        return state or _child_cache_changes(
            self.calculator, self._child_snapshot, atoms, tol
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = all_changes,
    ) -> None:
        """
        Calculate formation energy by wrapping the base calculator.

        Args:
            atoms (Atoms): The atomic structure to calculate properties for.
            properties (list[str]): The list of properties to calculate.
            system_changes (list[str]): The list of changes in the system.
        """
        atoms = self.atoms if atoms is None else atoms
        if atoms is None:
            raise ValueError("No atoms supplied to the calculator")
        try:
            self._calculate(atoms, properties, system_changes)
        except BaseException:
            self.reset()
            raise

    def _calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.atoms.info = copy.deepcopy(atoms.info)
        self.calculator.calculate(atoms, properties, system_changes)

        results = copy.deepcopy(self.calculator.results)

        energy_names = [name for name in ("energy", "free_energy") if name in results]
        if energy_names:
            element_counts = Counter(atoms.get_chemical_symbols())
            missing_elements = element_counts.keys() - self.element_references.keys()
            if missing_elements:
                raise ValueError(
                    f"Missing reference energies for elements: {missing_elements}"
                )
            reference = sum(
                self.element_references[element] * count
                for element, count in element_counts.items()
            )
            if self.apply_corrections:
                try:
                    from fairchem.data.omat.entries.compatibility import (
                        apply_mp_style_corrections,
                    )
                except ImportError as err:
                    raise ImportError(
                        "fairchem.data.omat is required to apply MP style corrections. Please install it."
                    ) from err
                # MP corrections depend on composition and structure and are additive.
                correction = apply_mp_style_corrections(
                    0.0, atoms, correction_type=self._correction_type
                )
            else:
                correction = 0.0
            for name in energy_names:
                results[name] = (results[name] + correction) - reference
        self.results = results
        self._child_snapshot = _child_cache_snapshot(self.calculator)
        self._reference_snapshot = copy.deepcopy(
            (self.element_references, self.apply_corrections, self._correction_type)
        )


class AllZeroUnitCellError(ValueError):
    def __init__(
        self,
        message="Atoms object claims to have PBC set, but the unit cell is identically 0. Please ensure that the atoms"
        "object has a non-zero unit cell.",
    ):
        self.message = message
        super().__init__(self.message)

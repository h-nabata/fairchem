"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Iterator

import numba as nb
import numpy as np
from torch.utils.data import Sampler

from fairchem.core.common import gp_utils
from fairchem.core.datasets.base_dataset import natoms_for_sampling
from fairchem.core.datasets.samplers.distributed_indices import distributed_indices

if TYPE_CHECKING:
    from fairchem.core.datasets.base_dataset import BaseDataset


@nb.njit
def get_batches(
    natoms_list: np.array,  # List of number of atoms in each sample
    indices: np.array,  # Indices of the samples
    max_atoms: int,  # Maximum number of atoms allowed in a batch
    min_atoms: int,  # Minimum number of atoms allowed in a batch
) -> tuple[list[list[int]], list[int], int]:
    """
    Greedily creates batches from a list of samples with varying numbers of atoms.

    Args:
    natoms_list: Array of number of atoms in each sample.
    indices: Array of indices of the samples.
    max_atoms: Maximum number of atoms allowed in a batch.

    Returns:
    tuple[list[list[int]], list[int], int]:
        A tuple containing a list of batches, a list of the total number of atoms in each batch,
        and the number of samples that were filtered out because they exceeded the maximum number of atoms.
    """

    # Ensure the inputs are valid
    assert max_atoms > 0
    assert 0 <= min_atoms <= max_atoms
    assert len(natoms_list) == len(indices)

    # Initialize variables to keep track of the batches and the total number of atoms in each batch
    batches = []  # List of batches, where each batch is a list of indices
    run_sum = 0  # Running total of the number of atoms in the current batch
    cur_batch = nb.typed.List.empty_list(nb.int64)  # Current batch being constructed
    atom_counts = nb.typed.List.empty_list(
        nb.int64
    )  # List of total number of atoms in each batch
    samples_filtered = 0  # Number of samples filtered out because they exceeded the maximum number of atoms

    # Iterate over the samples
    for idx, atoms in zip(indices, natoms_list):
        # If the sample has too many atoms, filter it out
        if atoms > max_atoms:
            samples_filtered += 1
            continue

        # If adding the sample to the current batch would not exceed the maximum number of atoms, add it
        if atoms <= max_atoms - run_sum:
            cur_batch.append(idx)
            run_sum += atoms
        # Otherwise, start a new batch
        else:
            if run_sum >= min_atoms:
                batches.append(cur_batch)
                atom_counts.append(run_sum)
            cur_batch = nb.typed.List([idx])
            run_sum = atoms

    # Add the last batch
    if len(cur_batch) and run_sum >= min_atoms:
        batches.append(cur_batch)
        atom_counts.append(run_sum)

    # Return the batches, atom counts, and the number of samples filtered out
    return [list(x) for x in batches], list(atom_counts), samples_filtered


class MaxAtomDistributedBatchSampler(Sampler[list[int]]):
    """
    A custom batch sampler that distributes batches across multiple GPUs to ensure efficient training.

    Args:
    dataset (BaseDataset): The dataset to sample from.
    max_atoms (int): The maximum number of atoms allowed in a batch.
    num_replicas (int): The number of GPUs to distribute the batches across.
    rank (int): The rank of the current GPU.
    seed (int): The seed for shuffling the dataset.
    shuffle (bool): Whether to shuffle the dataset. Defaults to True.
    drop_last (bool): Whether to drop the last batch if its size is less than the maximum allowed size. Defaults to False.

    This batch sampler is designed to work with the BaseDataset class and is optimized for distributed training.
    It takes into account the number of atoms in each sample and ensures that the batches are distributed evenly across GPUs.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        max_atoms: int,
        num_replicas: int,
        rank: int,
        seed: int,
        shuffle: bool = True,
        drop_last: bool = False,
        min_atoms: int = 0,
    ) -> None:
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.min_atoms = min_atoms
        self.num_replicas = num_replicas
        self.rank = rank
        if not 0 <= rank < num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")
        if max_atoms <= 0 or not 0 <= min_atoms <= max_atoms:
            raise ValueError("Expected max_atoms > 0 and 0 <= min_atoms <= max_atoms")

        if gp_utils.initialized():
            assert (
                min_atoms >= gp_utils.get_gp_world_size()
            ), "Min atoms needs to be at least gp world size!"

        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.epoch = 0
        self.start_iter = 0

        # we pre-create the batches here and can only do this once, otherwise every time we get this iterator we might get a different number of batches
        self.all_batches = self._prepare_batches()
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.all_batches) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.all_batches) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.all_batches) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def _prepare_batches(self) -> list[list[int]]:
        # shuffle is not optional here since the metadata tend to be sorted by atom count and the resulting batches will be highly uneven
        rng = np.random.default_rng(self.seed)
        original_indices = rng.permutation(len(self.dataset))
        t0 = time.time()
        natoms_list = self.dataset.get_metadata("natoms", original_indices.tolist())
        natoms_list = natoms_for_sampling(natoms_list, len(original_indices))
        t1 = time.time()
        indices, atoms_count, samples_filtered = get_batches(
            natoms_list, original_indices, self.max_atoms, self.min_atoms
        )
        if not indices:
            raise ValueError("No batch satisfies the configured atom count bounds")
        t2 = time.time()
        logging.info(
            f"Sampler batch generation times: get natoms: {t1 - t0}, total: {t2 - t0}"
        )
        logging.info(
            f"MaxAtomDistributedSampler generated {len(indices)} batches with total atoms {np.sum(natoms_list)}, max: {max(atoms_count)}, min: {min(atoms_count)}, mean: {np.mean(atoms_count)}, std: {np.std(atoms_count)}"
        )
        logging.info(
            f"{samples_filtered} samples were removed because they exceed {self.max_atoms} atoms"
        )
        return indices

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[list[int]]:
        indices = distributed_indices(
            len(self.all_batches),
            self.total_size,
            self.rank,
            self.num_replicas,
            self.shuffle,
            self.seed + self.epoch,
            self.start_iter,
        )
        return (self.all_batches[i] for i in indices)

    def set_epoch_and_start_iteration(self, epoch: int, start_iter: int) -> None:
        if not 0 <= start_iter <= self.num_samples:
            raise ValueError(f"start_iter must be in [0, {self.num_samples}]")
        self.epoch = epoch
        self.start_iter = start_iter

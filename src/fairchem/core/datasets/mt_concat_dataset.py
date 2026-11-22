"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import bisect
from operator import index as integer_index
from typing import TypeVar

import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from fairchem.core.datasets.base_dataset import create_dataset

# This draws highly from fairseq's ConcatDataset implementation
# https://github.com/facebookresearch/fairseq/blob/nllb/fairseq/data/concat_dataset.py

T_co = TypeVar("T_co", covariant=True)


class ConcatDataset(Dataset[T_co]):
    @staticmethod
    def cumsum(sequence, sample_ratios):
        r, s = [], 0
        for e, ratio in zip(sequence, sample_ratios):
            curr_len = int(ratio * len(e))
            r.append(curr_len + s)
            s += curr_len
        return r

    def __init__(self, datasets, sampling: dict):
        super().__init__()
        if len(datasets) == 0:
            raise RuntimeError(
                "datasets should not be an empty iterable. at least one dataset must have 'key_mappings'"
            )

        # convert the dictionary into two same order lists
        self.datasets = []
        self.dataset_names = []
        for k, v in datasets.items():
            self.datasets.append(v)
            self.dataset_names.append(k)

        self.sample_ratios = self._dataset_sampling(
            [len(d) for d in self.datasets], self.dataset_names, sampling
        )
        if len(self.sample_ratios) != len(self.datasets) or any(
            not np.isfinite(ratio) or ratio < 0 for ratio in self.sample_ratios
        ):
            raise ValueError(
                "Expected one finite, nonnegative sampling ratio per dataset"
            )
        self.cumulative_sizes = self.cumsum(self.datasets, self.sample_ratios)
        self.real_sizes = [len(d) for d in self.datasets]

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        data_object = self.datasets[dataset_idx][sample_idx]

        # Add additional attributes and cast to appropriate types
        # TODO setting data_object.dataset to dataset.split breaks the collater due to the task map
        #  we should clean that up by using a .dataset and .split keys for all data objects
        data_object.dataset_name = self.dataset_names[dataset_idx]
        # TODO this should be ensured in the datasets themselves, rather than casting/redefining here?
        data_object.fixed = data_object.fixed.long()
        data_object.atomic_numbers = data_object.atomic_numbers.long()

        return data_object

    def _get_dataset_and_sample_index_list(self, sample_idxs: list):
        indices = np.asarray(sample_idxs)
        if indices.size == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(
                "Dataset indices must be a one-dimensional array of integers"
            )
        if (indices >= len(self)).any() or (indices < -len(self)).any():
            raise IndexError("Dataset index out of range")
        indices = indices.astype(np.intp, copy=False)
        indices = np.where(indices < 0, indices + len(self), indices)
        owners = np.searchsorted(self.cumulative_sizes, indices, side="right")
        offsets = np.asarray([0, *self.cumulative_sizes])
        return owners, (indices - offsets[owners]) % np.asarray(self.real_sizes)[owners]

    # @functools.cache
    # B019 Use of `functools.lru_cache` or `functools.cache` on methods can lead to memory leaks
    def _get_dataset_and_sample_index(self, idx: int):
        idx = range(len(self))[integer_index(idx)]
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]
        return dataset_idx, sample_idx

    @property
    def updated_dataset_sizes(self):
        new_sizes = [
            (
                self.cumulative_sizes[i]
                if i == 0
                else self.cumulative_sizes[i] - self.cumulative_sizes[i - 1]
            )
            for i, _ in enumerate(self.cumulative_sizes)
        ]
        return (self.real_sizes, new_sizes, self.sample_ratios)

    def metadata_hasattr(self, attr) -> bool:
        for dataset in self.datasets:
            check = getattr(dataset, "metadata_hasattr", None)
            if not callable(check) or not check(attr):
                return False
        return True

    def get_metadata(self, attr, idx):
        if np.ndim(idx) == 0:
            owner, local = self._get_dataset_and_sample_index(idx)
            return self.datasets[owner].get_metadata(attr, local)
        owners, local = self._get_dataset_and_sample_index_list(idx)
        if not len(owners):
            return self.datasets[0].get_metadata(attr, [])
        # Group by source once, retaining duplicate indices and original order.
        order = np.argsort(owners, kind="stable")
        boundaries = np.flatnonzero(np.diff(owners[order])) + 1
        parts = []
        for positions in np.split(order, boundaries):
            values = self.datasets[owners[positions[0]]].get_metadata(
                attr, local[positions].tolist()
            )
            if values is None:
                return None
            values = np.asarray(values)
            if values.ndim == 0 or len(values) != len(positions):
                raise ValueError(f"Expected one {attr!r} metadata entry per sample")
            parts.append(values)
        if len(parts) == 1:
            return parts[0]
        grouped = np.concatenate(parts)
        result = np.empty_like(grouped)
        result[order] = grouped
        return result

    @staticmethod
    def _dataset_sampling(
        dataset_sizes: list[int], dataset_names: list[str], sampling: dict
    ) -> list[float]:
        """
        Return expansion ratios for each dataset based on sampling strategy
        """
        if sampling["type"] == "explicit":
            for dataset_name in dataset_names:
                if dataset_name not in sampling["ratios"]:
                    raise ValueError(
                        f"Missing ratio for dataset with name: {dataset_name}"
                    )
            return [sampling["ratios"][dataset_name] for dataset_name in dataset_names]
        if sampling["type"] == "balanced":
            indv_target_size = max(dataset_sizes)
            return [indv_target_size / size if size else 0.0 for size in dataset_sizes]
        elif sampling["type"] == "temperature":
            assert (
                sampling["temperature"] >= 1.0
            ), "Temperature must be >= 1.0, for custom weights use weighted sampling."
            # total size before expansion
            total_size = sum(dataset_sizes)
            if total_size == 0:
                return [0.0] * len(dataset_sizes)
            # calc the temperature sampling probabilities for each dataset
            # p_i = (D_i / D_total)^(1/T)
            temp_prob = [
                (size / total_size) ** (1.0 / sampling["temperature"])
                for size in dataset_sizes
            ]
            # normalize temp probabilities i.e. sum to 1
            temp_prob = [r / sum(temp_prob) for r in temp_prob]
            # find the target dataset size with temp sampling
            # this assume largest dataset ratio is 1
            max_idx = np.argmax(dataset_sizes)
            target_size = dataset_sizes[max_idx] / temp_prob[max_idx]
            # return expansion ratios
            return [
                target_size * prob / size if size else 0.0
                for prob, size in zip(temp_prob, dataset_sizes)
            ]
        elif sampling["type"] == "weighted":
            return sampling["ratios"]
        else:
            raise NotImplementedError(f"{sampling} not implemented.")


# This version is used for the Hydra configs
def create_concat_dataset(
    dataset_configs: DictConfig, combined_dataset_config: dict
) -> ConcatDataset:
    """Make a concat dataset with all the splits for each dataset. Keys will be {dataset}.{split}"""
    datasets: dict[str, Dataset] = {}
    # we need to convert omegaConf DictConfig objects to a straight up dict
    dataset_configs = OmegaConf.to_object(dataset_configs)
    for dataset_name in sorted(dataset_configs.keys()):
        dataset_config = dataset_configs[dataset_name]
        assert (
            dataset_config.get("lin_ref", None) is None
        ), "lin_refs in the dataset config are deprecated, please move them to the task config"
        for split in dataset_config["splits"]:
            try:
                datasets[f"{dataset_name}.{split}"] = create_dataset(
                    dataset_config, split=split
                )
            except ValueError as e:
                raise ValueError(f"Error creating dataset '{dataset_name}' {e}") from e

    return ConcatDataset(datasets, sampling=combined_dataset_config["sampling"])

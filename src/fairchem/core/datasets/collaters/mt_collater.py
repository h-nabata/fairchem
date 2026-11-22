"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch


class MTCollater:
    def __init__(
        self,
        task_config,
        exclude_keys,
        otf_graph: bool = False,  # Deprecated, do not use!
    ) -> None:
        self.exclude_keys = exclude_keys
        self.task_config = task_config
        self.dataset_task_map = self._create_dataset_task_map(task_config)

    def __call__(self, data_list: list[AtomicData]) -> AtomicData:
        batch = self.data_list_collater(
            data_list,
            dataset_task_map=self.dataset_task_map,
            exclude_keys=self.exclude_keys,
        )
        return batch

    def data_list_collater(
        self,
        data_list: list[AtomicData],
        dataset_task_map: dict,
        exclude_keys: list,
    ) -> AtomicData:
        data_list = self._add_missing_attr(data_list, dataset_task_map)

        # Legacy training configs treated PBC as optional dataset metadata.
        # AtomicData requires the real per-axis boundary flags for each system.
        if exclude_keys is not None and "pbc" in exclude_keys:
            exclude_keys = [key for key in exclude_keys if key != "pbc"]
        return atomicdata_list_to_batch(data_list, exclude_keys=exclude_keys)

    # takes in the task config
    def _create_dataset_task_map(self, config):
        datasets_task_map = {}
        for task, task_dict in config.items():
            for dataset in task_dict["datasets"]:
                if dataset not in datasets_task_map:
                    datasets_task_map[dataset] = {}
                datasets_task_map[dataset][task] = {"level": task_dict["level"]}
        return datasets_task_map

    def _add_missing_attr(self, data_list, dataset_task_map):
        """Fill missing task labels on temporary owners before collation."""
        prepared = []
        for source in data_list:
            labels = (
                [source.dataset] if isinstance(source.dataset, str) else source.dataset
            )
            if (
                source.num_graphs != 1
                or not isinstance(labels, (list, tuple))
                or len(labels) != 1
                or not isinstance(labels[0], str)
            ):
                raise ValueError(
                    "Each training sample must have one system and one dataset name"
                )
            name = labels[0]
            provided = dataset_task_map[name]
            # Collation copies tensors and metadata. Only the attribute table
            # needs separate ownership while adding fields to this sample.
            data = copy.copy(source)
            data.__keys__ = source.__keys__.copy()
            data.dataset = [name]
            for task, config in self.task_config.items():
                if config["level"] not in ("atom", "system"):
                    raise ValueError(f"Unknown task level {config['level']!r}")
                if task in provided:
                    if task not in data:
                        raise ValueError(f"Dataset {name!r} is missing task {task!r}")
                    continue
                dim = config["out_spec"]["dim"]
                if config["level"] == "atom":
                    dim = [data.num_nodes, *dim]
                dtype = getattr(torch, config["out_spec"]["dtype"])
                data[task] = torch.full(
                    dim, torch.inf, dtype=dtype, device=data.pos.device
                )
            prepared.append(data)
        return prepared

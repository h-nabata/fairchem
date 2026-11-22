"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from torch.utils.data import Dataset, DistributedSampler

from fairchem.core.common.data_parallel import (
    BalancedBatchSampler,
    StatefulDistributedSampler,
)
from fairchem.core.datasets.base_dataset import (
    BaseDataset,
    Subset,
    UnsupportedDatasetError,
)
from fairchem.core.datasets.mt_concat_dataset import ConcatDataset
from fairchem.core.datasets.samplers.max_atom_distributed_sampler import (
    MaxAtomDistributedBatchSampler,
)


def make_dataset(counts):
    data = BaseDataset({})
    data.num_samples = len(counts)
    data._metadata = {
        "natoms": np.asarray(counts, dtype=np.int64),
        "pair": np.array([(v, -v) for v in counts], dtype=float).reshape(-1, 2),
    }
    return data


def test_mixed_subset_shards_and_metadata_shape():
    a = make_dataset([3, 9, 7])
    b = Subset(make_dataset([4, 8, 5]), [2, 0], metadata={})
    data = ConcatDataset(
        {"a": a, "b": b}, {"type": "explicit", "ratios": {"a": 1, "b": 2}}
    )
    np.testing.assert_array_equal(b.get_metadata("natoms", [0, 1]), [5, 4])
    for indices in ([1, 0], [6, 2, 2, -3, 1, -1], np.array([4, 3])):
        for attr in ("natoms", "pair"):
            np.testing.assert_array_equal(
                data.get_metadata(attr, indices),
                [data.get_metadata(attr, idx) for idx in indices],
            )
    assert data.get_metadata("pair", []).shape == (0, 2)
    assert data.get_metadata("natoms", -1) == 4
    for invalid in ([len(data)], [-len(data) - 1], len(data), -len(data) - 1):
        with pytest.raises(IndexError):
            data.get_metadata("natoms", invalid)
    sampler = MaxAtomDistributedBatchSampler(data, 10, 1, 0, 11)
    batches = list(sampler)
    assert sorted(i for group in batches for i in group) == list(range(len(data)))
    assert all(sum(data.get_metadata("natoms", group)) <= 10 for group in batches)


def test_shared_metadata_archive_and_path_input(tmp_path):
    files = [tmp_path / "a.rdb", tmp_path / "b.rdb"]
    for f in files:
        f.touch()
    np.savez(tmp_path / "metadata.npz", natoms=np.array([2, 3]))
    for src in (tmp_path, files):
        data = BaseDataset({"src": src})
        data.num_samples = 2
        np.testing.assert_array_equal(data.get_metadata("natoms", [1, 0]), [3, 2])


@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("drop_last", [True, False])
def test_rank_local_resume_matches_pytorch(shuffle, drop_last):
    for size, ranks in [(0, 3), (2, 5), (17, 4), (16, 4)]:
        for rank in range(ranks):
            args = dict(
                rank=rank,
                num_replicas=ranks,
                seed=384,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            sampler = StatefulDistributedSampler(range(size), batch_size=2, **args)
            reference = DistributedSampler(range(size), **args)
            reference.set_epoch(4)
            expected = list(reference)
            for start in range(len(expected) // 2 + 2):
                sampler.set_epoch_and_start_iteration(4, start)
                assert list(sampler) == expected[start * 2 :]


def test_max_atoms_finish_and_small_world_padding():
    data = make_dataset([5, 7])
    for rank in range(4):
        sampler = MaxAtomDistributedBatchSampler(data, 10, 4, rank, 3)
        assert len(list(sampler)) == 1
        sampler.set_epoch_and_start_iteration(1, 1)
        assert list(sampler) == []
        with pytest.raises(ValueError, match="start_iter"):
            sampler.set_epoch_and_start_iteration(1, -1)
    with pytest.raises(ValueError, match="No batch"):
        MaxAtomDistributedBatchSampler(data, 4, 1, 0, 1)


def test_missing_metadata_obeys_balancing_policy():
    class Plain(Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, i):
            return SimpleNamespace(num_nodes=i + 1)

    def sampler(mode, ranks, policy="raise"):
        return BalancedBatchSampler(
            Plain(),
            batch_size=2,
            num_replicas=ranks,
            rank=0,
            device="cpu",
            seed=0,
            mode=mode,
            on_error=policy,
            shuffle=False,
        )

    assert list(sampler(False, 2)) == [[0, 2], [4]]
    assert list(sampler(True, 1)) == [[0, 1], [2, 3], [4]]
    assert sampler(True, 2, "warn_and_no_balance").disabled
    np.testing.assert_array_equal(
        sampler(True, 2, "warn_and_balance")._get_natoms([2, 0]), [3, 1]
    )
    with pytest.raises(UnsupportedDatasetError):
        sampler(True, 2)


def test_python_sequences_unsigned_counts_and_foreign_dataset():
    data = make_dataset([2, 3])
    data._metadata["natoms"] = [2, 3]
    assert data.get_metadata("natoms", [1, 0]) == [3, 2]
    for dtype in (np.uint32, np.uint64):
        data._metadata["natoms"] = np.array([2, 3], dtype=dtype)
        sampler = BalancedBatchSampler(
            data, batch_size=1, rank=0, num_replicas=2, device="cpu", seed=0
        )
        assert sampler._get_natoms([0, 1]).dtype == np.int64
        assert len(list(MaxAtomDistributedBatchSampler(data, 5, 1, 0, 0))) == 1
    data._metadata["natoms"] = np.array([2, 2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="int64"):
        MaxAtomDistributedBatchSampler(data, 5, 1, 0, 0)

    class Plain(Dataset):
        def __len__(self):
            return 2

    mixed = ConcatDataset({"data": data, "plain": Plain()}, {"type": "balanced"})
    sampler = BalancedBatchSampler(
        mixed,
        batch_size=1,
        rank=0,
        num_replicas=2,
        device="cpu",
        seed=0,
        on_error="warn_and_no_balance",
    )
    assert sampler.disabled

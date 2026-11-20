"""Exercise nonzero global ranks, including one-way empty peer messages."""

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from fairchem.core.common import distutils
from fairchem.core.common.parallelism.graph_parallel_a2a import _safe_all_to_all
from fairchem.core.modules.loss import DDPMTLoss, MSELoss


def _subgroup_worker(rank, rendezvous):
    timeout = timedelta(seconds=35)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timeout
    )
    try:
        group = dist.new_group([2, 3], backend="gloo", timeout=timeout)
        if rank in (2, 3):
            assert distutils.all_reduce(float(rank), group=group, average=True) == 2.5
            assert [
                int(t) for t in distutils.all_gather(torch.tensor(rank), group=group)
            ] == [2, 3]
            objects = distutils.gather_objects(rank, group=group)
            assert objects == ([2, 3] if rank == 2 else None)
            chunks = (
                [torch.tensor([20]), torch.tensor([21])]
                if rank == 2
                else [torch.empty(0, dtype=torch.long), torch.tensor([31])]
            )
            received = [
                torch.empty(n, dtype=torch.long)
                for n in ((1, 0) if rank == 2 else (1, 1))
            ]
            _safe_all_to_all(received, chunks, group)
            assert [t.tolist() for t in received] == (
                [[20], []] if rank == 2 else [[21], [31]]
            )
        for mode in ("sum", "mean", "per_valid_structure"):
            pred = torch.tensor([float("nan") if rank == 1 else 1.0])
            with pytest.raises(RuntimeError, match="Nonfinite"):
                DDPMTLoss(MSELoss(), reduction=mode)(
                    pred,
                    torch.zeros_like(pred),
                    torch.tensor([True]),
                    torch.tensor([1]),
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_nonzero_subgroup_reductions_and_one_way_exchange(tmp_path):
    mp.start_processes(
        _subgroup_worker,
        ((tmp_path / "store").as_uri(),),
        nprocs=4,
        start_method="fork",
        join=True,
    )

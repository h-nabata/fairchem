"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from fairchem.core.calculate import pretrained_mlip


@pytest.mark.parametrize("available,index", [(False, 0), (True, 2)])
def test_unavailable_xpu_prevents_checkpoint_download(available, index, monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: available)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    resolve = Mock(side_effect=AssertionError("must not download checkpoint"))
    monkeypatch.setattr(pretrained_mlip, "resolve_checkpoint", resolve)
    with pytest.raises(
        (RuntimeError, ValueError), match="XPU was requested|Requested xpu"
    ):
        pretrained_mlip.get_predict_unit("uma-s-1p2p1", device=f"xpu:{index}")
    resolve.assert_not_called()


@pytest.mark.parametrize("workers,gp_config", [(2, None), (1, object())])
def test_xpu_parallel_request_rejected_before_download(workers, gp_config, monkeypatch):
    resolve = Mock(side_effect=AssertionError("must not download checkpoint"))
    monkeypatch.setattr(pretrained_mlip, "resolve_checkpoint", resolve)
    with pytest.raises(ValueError, match="workers=1"):
        pretrained_mlip.get_predict_unit(
            "uma-s-1p2p1", device="xpu:0", workers=workers, gp_config=gp_config
        )
    resolve.assert_not_called()


def test_pretrained_xpu_keeps_ordinal_resolved_before_io(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)
    monkeypatch.setattr(
        pretrained_mlip,
        "resolve_checkpoint",
        lambda *args: SimpleNamespace(path="model.pt", atom_refs={}, form_elem_refs={}),
    )
    loader = Mock()
    monkeypatch.setattr(pretrained_mlip, "load_predict_unit", loader)
    pretrained_mlip.get_predict_unit("uma-s-1p2p1", device="xpu")
    assert loader.call_args.args[3] == "xpu:1"


def test_checkpoint_and_reference_energies_share_custom_cache(monkeypatch, tmp_path):
    import json

    from fairchem.core.calculate import pretrained_mlip as pretrained

    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"refs": {}}))
        return str(path)

    monkeypatch.setattr(pretrained, "hf_hub_download", download)
    pretrained.resolve_checkpoint("uma-s-1p2p1", cache_dir=str(tmp_path))
    assert len(calls) == 3
    assert all(call["cache_dir"] == str(tmp_path) for call in calls)


def test_references_use_snapshot_of_downloaded_weights(monkeypatch, tmp_path):
    revision = "b" * 40
    requested = []

    def download(*, filename, subfolder, **kwargs):
        requested.append(kwargs["revision"])
        path = (
            tmp_path
            / "models--facebook--UMA"
            / "snapshots"
            / revision
            / subfolder
            / filename
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"refs": {}}')
        return str(path)

    monkeypatch.setattr(pretrained_mlip, "hf_hub_download", download)
    pretrained_mlip.resolve_checkpoint("uma-s-1p1", str(tmp_path))
    assert requested == [None, revision, revision]


def test_auto_selected_xpu_preflight_before_model_io(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    download = Mock(side_effect=AssertionError("unexpected checkpoint download"))
    monkeypatch.setattr(pretrained_mlip, "resolve_checkpoint", download)
    with pytest.raises(ValueError, match="workers=1"):
        pretrained_mlip.get_predict_unit("uma-s-1p1", workers=2)
    download.assert_not_called()

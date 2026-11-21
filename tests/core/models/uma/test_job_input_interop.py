"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from tests.core.models.uma.test_series_a_integration import _data, _model


def test_reusing_direct_backbone_input_refreshes_scatter_targets():
    backbone, head = _model()
    data = _data(False)
    data._requested_properties = ("energy", "forces")
    first = head(data, backbone(data))
    assert first["forces"].abs().max() > 0
    old_count = data.scatter_target.numel()
    with torch.no_grad():
        data.pos.mul_(20)
    second = head(data, backbone(data))
    assert data.scatter_target.numel() < old_count
    torch.testing.assert_close(second["forces"], torch.zeros_like(second["forces"]))

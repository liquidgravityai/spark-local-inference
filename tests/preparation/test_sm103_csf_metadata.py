# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 expert declarations distinguish compact weights from shared scratch."""

import torch
from b12x._lib.quant.csf_blocks import CsfBlockPayload, Nvfp4CompactBlockPlane
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder

from ignition.components import weights as _weight_containers  # Registers weight-bearing fields.
from ignition.components.moe_sm103 import TrtllmNvfp4Experts
from ignition.preparation.rebuild_manifest import declare_weights, declared_views


def _tensor(dtype=torch.float32):
    return torch.zeros(4, dtype=dtype)


def _experts(compact):
    w13, w2 = _tensor(torch.uint8), _tensor(torch.uint8)
    expanded13, expanded2 = _tensor(torch.uint8), _tensor(torch.uint8)
    calibration = tuple(_tensor() for _ in range(7))
    decoder = None
    if compact:
        planes = tuple(Nvfp4CompactBlockPlane(CsfBlockPayload(
            _tensor(torch.uint8), 128, 4, 1, "nvfp4", native_codes=True,
            group_record_bytes=3)) for _ in range(2))
        decoder = Nvfp4CsfDecoder(*planes, (), (), _tensor(torch.int32))
    expert = TrtllmNvfp4Experts(w13, expanded13, w2, expanded2, *calibration,
                                nvfp4_csf=decoder)
    return expert, expanded13, expanded2, calibration


def _views(expert):
    declaration = declare_weights({"weights": expert}, components=("weights",), layers=())
    return {name: tensor for name, _owner, tensor, _group, _storage in declared_views(declaration)}


def test_compact_expert_declares_payloads_and_calibration_not_scratch():
    expert, out13, out2, calibration = _experts(True)
    views = _views(expert)
    assert views["weights.w13"] is expert.w13
    assert views["weights.w2"] is expert.w2
    assert views["weights.stored_scales[0]"] is expert.nvfp4_csf.first.fixed
    assert views["weights.stored_scales[1]"] is expert.nvfp4_csf.second.fixed
    for name, tensor in zip(("input_global_scale", "output1_scale", "output1_gate_scale",
                             "output2_scale", "gate_alpha", "gate_beta", "clamp_limit"),
                            calibration, strict=True):
        assert views[f"weights.{name}"] is tensor
    declared = {id(value) for value in views.values()}
    assert id(out13) not in declared and id(out2) not in declared
    assert id(expert.nvfp4_csf.active) not in declared
    assert len(views) == 11


def test_resident_expert_declares_expanded_scale_weights():
    expert, out13, out2, _ = _experts(False)
    views = _views(expert)
    assert views["weights.stored_scales[0]"] is out13
    assert views["weights.stored_scales[1]"] is out2

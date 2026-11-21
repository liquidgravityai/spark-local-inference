# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for SM103 dynamic expert mover configuration and predictor validation."""

import json
from types import SimpleNamespace

import pytest

from ignition.components.expert_placement import ExpertPlacement, PrefillExpertStaging
from ignition.components.sm103_expert_mover import PredictorConfig
from ignition.recipes.deepseek41.sm103 import DynamicExpertMoverConfig, SM103Config


def _sample_placements(layers=2, experts=256, hbm_count=128):
    return tuple(
        (f"layers.{i}.ffn", ExpertPlacement(experts, tuple(range(hbm_count, experts)), tuple(range(hbm_count))))
        for i in range(layers)
    )


def test_dynamic_mover_config_json_roundtrip_and_budget(tmp_path):
    cfg = SM103Config(
        placements=_sample_placements(layers=2, experts=256, hbm_count=128),
        host_lane_clusters=0,
        dynamic_mover=DynamicExpertMoverConfig(worker_sm_count=8, spare_slots=2, max_enqueues=2),
    )
    path = tmp_path / "sm103_dynamic.json"
    path.write_text(json.dumps(cfg.to_dict()))
    loaded = SM103Config.load(path)
    assert loaded == cfg
    assert loaded.dynamic_mover.predictor_config() == PredictorConfig(max_enqueues=2)

    fake_model = SimpleNamespace(
        layers=2,
        text={"n_routed_experts": 256, "hidden_size": 6144, "moe_intermediate_size": 2048},
        draft=SimpleNamespace(layers=0, experts=0),
    )
    bytes_map = loaded.expert_bytes(fake_model, dspark=False)
    per_expert = (3 * 6144 * 2048) // 2 + (3 * 6144 * 2048) // 32
    assert bytes_map["host"] == 2 * 256 * per_expert
    assert bytes_map["hbm"] == 2 * (128 + 2) * per_expert


def test_dynamic_mover_config_validation_rejects_invalid():
    with pytest.raises(ValueError, match="worker_sm_count"):
        DynamicExpertMoverConfig(worker_sm_count=0)
    with pytest.raises(ValueError, match="spare"):
        DynamicExpertMoverConfig(spare_slots=0)
    with pytest.raises(ValueError, match="power of two"):
        DynamicExpertMoverConfig(queue_capacity=100)
    with pytest.raises(ValueError, match="max_enqueues"):
        DynamicExpertMoverConfig(max_enqueues=5)

    # Cannot combine dynamic_mover with static prefill_staging
    with pytest.raises(ValueError, match="cannot be combined"):
        SM103Config(
            placements=_sample_placements(layers=1, experts=64, hbm_count=32),
            prefill_staging=PrefillExpertStaging(min_rows=1024, min_cold_experts=8),
            dynamic_mover=DynamicExpertMoverConfig(),
        )

    # HBM active + spare_slots cannot exceed total canonical experts
    with pytest.raises(ValueError, match="exceeds total canonical experts"):
        SM103Config(
            placements=(("layers.0.ffn", ExpertPlacement(64, (63,), tuple(range(63)))),),
            dynamic_mover=DynamicExpertMoverConfig(spare_slots=2),
        )

    # Must have at least one tiered target layer with host experts
    with pytest.raises(ValueError, match="at least one tiered target layer"):
        SM103Config(
            placements=(("layers.0.ffn", ExpertPlacement(64, (), tuple(range(64)))),),
            dynamic_mover=DynamicExpertMoverConfig(spare_slots=2),
        )

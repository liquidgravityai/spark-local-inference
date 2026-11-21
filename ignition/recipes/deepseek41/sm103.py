# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Explicit TP1 expert placement and reference attention preparation on SM103."""

from dataclasses import dataclass, replace
import json
from math import lcm
from pathlib import Path

from ignition.components.expert_placement import ExpertPlacement, PrefillExpertStaging


def _reference_cache_costs(model, *, verification_width, dspark, page_tokens=128):
    def align(size, step=256):
        return (size + step - 1) // step * step

    window_bytes = align((model.text["sliding_window"] + verification_width - 1) * 528)
    layers = model.layers + (model.draft.layers if dspark else 0)
    row = layers * window_bytes
    pages = 0
    for source in model.sources:
        records = source.records(page_tokens)
        pages += records * (288 + 68)
        row += align(records * 288) + align(records * 68)
        if source.ratio == 2:
            row += align((source.ratio + verification_width - 1) * 2 * model.text["head_dim"] * 4)
    row += align(model.text["hidden_size"] * 2) + align(8) + align(24)
    return align(row, lcm(256, 528)), pages


@dataclass(frozen=True)
class DynamicExpertMoverConfig:
    """Configuration for the SM103 GPU-side persistent expert mover and inline predictor."""

    worker_sm_count: int = 8
    spare_slots: int = 2
    queue_capacity: int = 1024
    decay_shift: int = 4
    promote_threshold: int = 384
    hysteresis: int = 64
    cooldown_steps: int = 2
    max_enqueues: int = 2
    max_predict_rows: int = 64

    def __post_init__(self):
        if any(type(v) is not int for v in vars(self).values()):
            raise ValueError("dynamic_mover fields must be integers")
        if not 1 <= self.worker_sm_count < 64:
            raise ValueError("dynamic_mover worker_sm_count must be in [1, 63]")
        if self.spare_slots < 1:
            raise ValueError("dynamic_mover requires at least one spare HBM slot")
        if self.queue_capacity < 16 or (self.queue_capacity & (self.queue_capacity - 1)) != 0:
            raise ValueError("dynamic_mover queue_capacity must be a power of two >= 16")
        if not 0 <= self.decay_shift < 31:
            raise ValueError("dynamic_mover decay_shift must be in [0, 30]")
        if self.promote_threshold <= 0:
            raise ValueError("dynamic_mover promote_threshold must be positive")
        if self.hysteresis < 0:
            raise ValueError("dynamic_mover hysteresis must be nonnegative")
        if self.cooldown_steps < 1:
            raise ValueError("dynamic_mover cooldown_steps must be at least 1")
        if not 0 <= self.max_enqueues <= 4:
            raise ValueError("dynamic_mover max_enqueues must be in [0, 4]")
        if self.max_predict_rows < 1:
            raise ValueError("dynamic_mover max_predict_rows must be positive")

    def predictor_config(self):
        from ignition.components.sm103_expert_mover import PredictorConfig

        return PredictorConfig(
            decay_shift=self.decay_shift,
            promote_threshold=self.promote_threshold,
            hysteresis=self.hysteresis,
            cooldown_steps=self.cooldown_steps,
            max_enqueues=self.max_enqueues,
            max_predict_rows=self.max_predict_rows,
        )

    def to_dict(self):
        return {
            "worker_sm_count": self.worker_sm_count,
            "spare_slots": self.spare_slots,
            "queue_capacity": self.queue_capacity,
            "decay_shift": self.decay_shift,
            "promote_threshold": self.promote_threshold,
            "hysteresis": self.hysteresis,
            "cooldown_steps": self.cooldown_steps,
            "max_enqueues": self.max_enqueues,
            "max_predict_rows": self.max_predict_rows,
        }


@dataclass(frozen=True)
class SM103Config:
    placements: tuple[tuple[str, ExpertPlacement], ...]
    host_lane_clusters: int = 4
    prefill_staging: PrefillExpertStaging | None = None
    execution: str = "reference"
    context: int | None = None
    dynamic_mover: DynamicExpertMoverConfig | None = None

    @classmethod
    def load(cls, path):
        path = Path(path).resolve(strict=True)
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate SM103 preparation field: {key}")
                result[key] = value
            return result
        value = json.loads(path.read_text(), object_pairs_hook=unique)
        if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("SM103 preparation requires version 1")
        execution = value.get("execution", "reference")
        if execution != "reference":
            raise ValueError("SM103 execution must be 'reference'")
        required = {"version", "placements", "host_lane_clusters"}
        optional = {"execution", "context", "prefill_staging", "dynamic_mover"}
        if not required <= set(value) or set(value) - required - optional or not isinstance(value["placements"], dict):
            raise ValueError("SM103 preparation fields do not match reference execution")
        placements = []
        for name, rows in value["placements"].items():
            if (not isinstance(rows, dict) or set(rows) != {"experts", "host", "hbm"}
                    or not isinstance(rows["host"], list) or not isinstance(rows["hbm"], list)):
                raise ValueError(f"{name}: placement requires experts, host and hbm arrays")
            placements.append((name, ExpertPlacement(rows["experts"], tuple(rows["host"]), tuple(rows["hbm"]))))
        staging = value.get("prefill_staging")
        if staging is not None:
            if not isinstance(staging, dict) or set(staging) != {"min_rows", "min_cold_experts"}:
                raise ValueError("prefill_staging requires min_rows and min_cold_experts")
            staging = PrefillExpertStaging(**staging)
        mover = value.get("dynamic_mover")
        if mover is not None:
            allowed_mover = set(DynamicExpertMoverConfig.__dataclass_fields__)
            if not isinstance(mover, dict) or set(mover) - allowed_mover:
                raise ValueError("dynamic_mover contains unsupported fields")
            mover = DynamicExpertMoverConfig(**mover)
        return cls(tuple(placements), value["host_lane_clusters"], staging,
                   execution=execution, context=value.get("context"), dynamic_mover=mover)

    def __post_init__(self):
        if (not isinstance(self.placements, tuple) or not self.placements
                or any(not isinstance(item, tuple) or len(item) != 2
                       or not isinstance(item[0], str) or not isinstance(item[1], ExpertPlacement)
                       for item in self.placements)
                or len(dict(self.placements)) != len(self.placements)):
            raise ValueError("SM103 preparation requires distinct named expert placements")
        if self.execution != "reference":
            raise ValueError("SM103 execution must be 'reference'")
        if self.context is not None and (
                type(self.context) is not int or not 128 <= self.context <= 1048576 or self.context % 128):
            raise ValueError("SM103 context must be 128-aligned in [128, 1048576]")
        if type(self.host_lane_clusters) is not int or not 0 <= self.host_lane_clusters < 76:
            raise ValueError("SM103 host lane clusters must be in [0, 75]")
        if self.prefill_staging is not None and not isinstance(self.prefill_staging, PrefillExpertStaging):
            raise ValueError("SM103 prefill staging requires explicit row and cold-demand thresholds")
        if self.dynamic_mover is not None:
            if not isinstance(self.dynamic_mover, DynamicExpertMoverConfig):
                raise ValueError("SM103 dynamic_mover must be a DynamicExpertMoverConfig")
            if self.prefill_staging is not None:
                raise ValueError("SM103 dynamic_mover cannot be combined with static prefill_staging")
            tiered_target = False
            for name, placement in self.placements:
                if name.startswith("layers.") and placement.host:
                    tiered_target = True
                    if len(placement.hbm) + self.dynamic_mover.spare_slots > placement.experts:
                        raise ValueError(
                            f"{name}: HBM active slots plus spare_slots exceeds total canonical experts"
                        )
            if not tiered_target:
                raise ValueError("SM103 dynamic_mover requires at least one tiered target layer with host experts")

    def validate_model(self, model, *, dspark):
        expected = {f"layers.{i}.ffn": model.text["n_routed_experts"] for i in range(model.layers)}
        if dspark:
            expected.update({f"mtp.{i}.ffn": model.draft.experts for i in range(model.draft.layers)})
        if {name: placement.experts for name, placement in self.placements} != expected:
            raise ValueError("SM103 placements must match every enabled target and draft layer's canonical experts")

    def to_dict(self):
        result = dict(version=1, execution="reference",
                      placements={name: placement.to_dict() for name, placement in self.placements},
                      host_lane_clusters=self.host_lane_clusters)
        if self.prefill_staging is not None:
            result["prefill_staging"] = self.prefill_staging.to_dict()
        if self.context is not None:
            result["context"] = self.context
        if self.dynamic_mover is not None:
            result["dynamic_mover"] = self.dynamic_mover.to_dict()
        return result

    def staging_bytes(self, model):
        """Conservative three-plane-set allowance for two-consumer lookahead."""
        if self.prefill_staging is None:
            return 0
        per_expert = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"] * 17 // 32
        rows = [len(p.host) for name, p in self.placements if name.startswith("layers.")
                and len(p.host) >= self.prefill_staging.min_cold_experts]
        if not rows:
            raise ValueError("prefill staging requires a target host tier that can meet its cold-demand threshold")
        return 3 * max(rows) * per_expert

    def expert_bytes(self, model, *, dspark):
        """Native packed MXFP4 payload and expanded UE8M0 scales, including copies and dynamic spare slots."""
        self.validate_model(model, dspark=dspark)
        elements = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"]
        per_expert = elements // 2 + elements // 32
        hbm_rows = 0
        host_rows = 0
        for name, placement in self.placements:
            if self.dynamic_mover is not None and name.startswith("layers.") and placement.host:
                host_rows += placement.experts
                hbm_rows += len(placement.hbm) + self.dynamic_mover.spare_slots
            else:
                host_rows += len(placement.host)
                hbm_rows += len(placement.hbm)
        return {"hbm": hbm_rows * per_expert, "host": host_rows * per_expert}

    def planning_inputs(self, recipe, checkpoint, snapshot, *, dspark, draft_tokens=None):
        """Estimate native tensor extents from headers, independently of CSF file sizes."""
        from ignition.preparation.checkpoint import inspect_indexed_checkpoint
        from ignition.recipes.declaration import RankCosts
        from .config import ModelConfig

        if recipe.name != "deepseek41":
            raise ValueError("SM103 reference preparation is integrated with DeepSeek V4.1")
        model = ModelConfig.load(snapshot)
        has_draft = dspark and (draft_tokens is None or draft_tokens > 0)
        tier = self.expert_bytes(model, dspark=has_draft)
        manifest = inspect_indexed_checkpoint(snapshot)
        other = draft = 0
        for name, tensor in manifest.tensors.items():
            if ".ffn.experts." in name or ".engram.embed." in name:
                continue
            is_draft = name.startswith("mtp.")
            if is_draft and not has_draft:
                continue
            # Native 1x32 scale planes replace block scales for eligible FP8
            # projections. Retaining the source scale bytes makes this an upper
            # estimate, including projections that retain their block format.
            size = tensor.nbytes + (tensor.nbytes // 32 if tensor.dtype == "F8_E4M3" else 0)
            size = (size + 511) // 512 * 512
            other += size
            if is_draft:
                draft += size
        if has_draft:
            # Draft context uses another grouped copy of each layer's KV weight.
            context = sum(t.nbytes + (t.nbytes // 32 if t.dtype == "F8_E4M3" else 0)
                for name, t in manifest.tensors.items() if name.startswith("mtp.") and ".attn.wkv." in name)
            other += context
            draft += context
            per_expert = 3 * model.text["hidden_size"] * model.text["moe_intermediate_size"] * 17 // 32
            draft += sum(len(p.hbm) for name, p in self.placements if name.startswith("mtp.")) * per_expert
        # Physical row/page costs are checked against ReferenceContinuationPool in tests.
        # Graph storage is an estimate; measured capture accounting supersedes it.
        graph_bytes = (8 << 30) + self.staging_bytes(model)
        width = ((recipe.max_draft_tokens if draft_tokens is None else draft_tokens) + 1
                 if has_draft else 1)
        spec_row, ref_page = _reference_cache_costs(model, verification_width=width, dspark=has_draft)
        plain_row, _ = _reference_cache_costs(model, verification_width=1, dspark=False)
        ranks = {1: RankCosts(spec_row, ref_page, graph_bytes)}
        plain = {1: RankCosts(plain_row, ref_page, graph_bytes)}
        recipe_updates = dict(ranks=ranks, plain_ranks=plain, verification_rows=0,
                              expert_scale_compression=None)
        if self.context is not None:
            recipe_updates["context"] = self.context
        return (replace(recipe, **recipe_updates),
                replace(checkpoint, device_bytes=tier["hbm"] + other, draft_bytes=draft,
                    mapped_host_bytes=tier["host"], expansion_bytes=0,
                    scale_compressed_device_bytes={}, ranks=ranks, plain_ranks=plain))

# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compose reference target, draft and vision computation for Ignition capture."""

from dataclasses import dataclass

import torch
from torch import nn

from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace
from ignition.components.megamoe_tiers import MegaMoeLayerLayout
from ignition.components.weights import DirectWeightLoader
from ignition.preparation.rebuild_manifest import declare_weights
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.reference_experts import ReferenceExperts


def _record_model_numeric_tensors(root):
    """Register learned, scale and derived numeric tensors with the core exporter."""
    def record(value):
        if isinstance(value, torch.Tensor) and value.is_cuda and value.layout == torch.strided:
            record_numeric_tensor(value)
            sf = getattr(value, "sf", None)
            if isinstance(sf, torch.Tensor):
                record(sf)
            for extra in getattr(value, "ue8m0_sf", ()):
                record(extra)

    for module in root.modules():
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is not None:
                record(tensor)
        for key, value in vars(module).items():
            if key in ("_parameters", "_buffers", "_modules", "operations", "delivery", "engram_runtime"):
                continue
            if isinstance(value, torch.Tensor):
                record(value)
            elif isinstance(value, tuple):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        record(item)


@dataclass(frozen=True)
class ReferenceDeclaredWeights:
    """Semantic weight container pairing per-layer MegaMoE storage with reference modules."""

    layers: tuple
    draft_experts: tuple
    draft_heads: tuple
    model: object
    scales: tuple


def declare_reference_weights(model):
    """Build the prepared-weight declaration matching target expert row bindings."""
    target_count = len(model.model.layers)
    draft_count = 0 if model.draft is None else len(model.draft.layers)
    scales, seen_scales = [], set()
    for parameter in model.parameters():
        for extra in (getattr(parameter, "sf", None), *getattr(parameter, "ue8m0_sf", ())):
            if isinstance(extra, torch.Tensor) and extra.data_ptr() not in seen_scales:
                seen_scales.add(extra.data_ptr())
                scales.append(extra)
    derived_heads = tuple(h for h in (getattr(model, "draft_head", None), getattr(model, "markov_head", None))
                          if h is not None)
    roots = ReferenceDeclaredWeights(
        layers=tuple(model.experts.weights[f"layers.{i}.ffn"] for i in range(target_count)),
        draft_experts=tuple(model.experts.weights[f"mtp.{i}.ffn"] for i in range(draft_count)),
        draft_heads=derived_heads,
        model=model,
        scales=tuple(scales),
    )
    components = ["model", "scales"]
    if draft_count:
        components.insert(0, "draft_experts")
    if derived_heads:
        components.insert(0, "draft_heads")
    aliases = ()
    if draft_count:
        aliases = (
            ("weights.model.draft.embed_tokens.weight", "weights.model.model.embed_tokens.weight"),
            ("weights.model.draft.lm_head.weight", "weights.model.lm_head.weight"),
        )
    return declare_weights(roots, components=tuple(components), layers=("layers",), aliases=aliases)


@dataclass(frozen=True)
class ReferenceExpertPreparation:
    modules: dict
    weights: dict
    mover: object | None = None

    @classmethod
    def load(cls, checkpoint, placements, *, target_rows, draft_rows=(), storage,
             kernel_cache, session, host_lane_clusters=4, dspark=True,
             dynamic_mover=None, compute_sm_count=0):
        """Load all target/draft experts into explicit tiers using bounded batches.

        The caller owns the native storage context and retains this preparation
        until graph execution drains. Scale expansion preserves checkpoint bits;
        the prepared MegaMoE planes use native scales in both physical tiers.
        """
        from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier, MegaMoeLayerWeights
        from ignition.components.megamoe_weights import (
            allocate_mxfp4_weights, load_mxfp4_weights_into,
            load_shared_fp8_planes, pack_shared_fp8_planes,
        )
        from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime
        from ignition.preparation.persistent import persistent_copy

        model = ModelConfig.load(checkpoint.source.metadata_path)
        placements = dict(placements)
        required = [f"layers.{i}.ffn" for i in range(model.layers)]
        if dspark:
            required.extend(f"mtp.{i}.ffn" for i in range(model.draft.layers))
        if set(placements) != set(required):
            raise ValueError("reference preparation requires placements for every enabled target and draft layer")
        if not target_rows or (dspark and not draft_rows):
            raise ValueError("reference preparation requires row capacities for every enabled expert group")
        device = torch.device("cuda", storage.device)
        workspaces, modules, owners = {}, {}, {}
        dynamic_layers = []
        for name in required:
            draft = name.startswith("mtp.")
            rows = tuple(draft_rows if draft else target_rows)
            placement = placements[name]
            experts = model.draft.experts if draft else model.text["n_routed_experts"]
            if placement.experts != experts:
                raise ValueError(f"reference placement differs from checkpoint expert count: {name}")
            use_dynamic = dynamic_mover is not None and not draft and bool(placement.host)
            physical_rows = experts if use_dynamic else placement.physical_rows
            geometry = MegaMoeGeometry(physical_rows, max(rows),
                model.draft.top_k if draft else model.text["num_experts_per_tok"],
                model.text["hidden_size"], model.text["moe_intermediate_size"], model.text["n_shared_experts"])
            if geometry not in workspaces:
                workspaces[geometry] = MegaMoeWorkspace(geometry, device=device, kernel_cache=kernel_cache)
            ws = workspaces[geometry]
            with DirectWeightLoader(checkpoint.source, device, prefixes=(name + ".",)) as loader:
                if use_dynamic:
                    hbm_slots = len(placement.hbm) + dynamic_mover.spare_slots
                    tier = MegaMoeDynamicHostTier(
                        ws.geometry, hbm_slots=hbm_slots, initial_hbm_experts=placement.hbm,
                        storage=storage, name=name,
                    )
                    prefixes = [f"{name}.experts.{i}" for i in range(experts)]
                    load_mxfp4_weights_into(
                        loader, prefixes, tier.cpu_weights,
                        hidden_size=ws.geometry.hidden, intermediate_size=ws.geometry.intermediate,
                        projection_names=("w1", "w3", "w2"), scale_name="scale",
                    )
                    routed = allocate_mxfp4_weights(hbm_slots, ws.geometry.hidden, ws.geometry.intermediate, device=device)
                    for dst, src_cpu in zip((*routed.gate_up, *routed.down), tier.host_views):
                        for slot, expert_id in enumerate(placement.hbm):
                            dst[slot].copy_(src_cpu[expert_id].to(device))
                        dst[len(placement.hbm):].zero_()
                    shared = pack_shared_fp8_planes(load_shared_fp8_planes(
                        loader, name + ".shared_experts",
                        hidden_size=ws.geometry.hidden, intermediate_size=ws.geometry.intermediate,
                        block_size=tuple(model.quantization["weight_block_size"]),
                        projection_names=("w1", "w3", "w2"), scale_name="scale",
                    ))
                    expert_map = persistent_copy(torch.arange(experts, dtype=torch.int32, device=device))
                    record_numeric_tensor(expert_map)
                    weights = MegaMoeLayerWeights(
                        routed, shared, ws, tier, expert_map,
                        tuple(range(experts)), placement.hbm,
                    )
                    dynamic_layers.append((tier, routed))
                else:
                    layout = MegaMoeLayerLayout(ws, placement.host, placement.hbm,
                                                storage, host_lane_clusters)
                    weights = layout.load(loader, (f"{name}.experts.{i}" for i in range(experts)),
                        shared_prefix=name + ".shared_experts", block_size=tuple(model.quantization["weight_block_size"]),
                        projection_names=("w1", "w3", "w2"), name=name)
                weight = loader.load_tensor(name + ".gate.weight")
                bias = loader.load_tensor(name + ".gate.bias")
                image_bias = loader.load_tensor(name + ".gate.bias_vl") if model.vision is not None else None
            owners[name] = weights
            modules[name] = ReferenceExperts.prepare(weights, rows, weight=weight, bias=bias, image_bias=image_bias,
                scaling=model.text["routed_scaling_factor"], scoring_func=model.text["scoring_func"],
                clamp=model.text["swiglu_limit"], session=session, name=name,
                compute_sm_count=compute_sm_count)
            if modules[name].placement.data_ptr() != weights.expert_map.data_ptr():
                raise RuntimeError(f"reference router placement lost its tier mapping alias: {name}")
        mover = None
        if dynamic_layers:
            mover = SM103ExpertMoverRuntime(
                dynamic_layers,
                storage=storage,
                kernel_cache=kernel_cache,
                worker_sm_count=dynamic_mover.worker_sm_count,
                queue_capacity=dynamic_mover.queue_capacity,
            )
            mover.attach_predictors(dynamic_mover.predictor_config())
        return cls(modules, owners, mover=mover)

    def close_host_regions(self):
        """Release mapped host weights after every reader and graph has drained."""
        if self.mover is not None:
            self.mover.close()
        for weights in self.weights.values():
            if weights.host_tier is not None:
                for region in weights.host_tier.regions:
                    region.close()


REFERENCE_YOCO_MIN_KEEP_TOKENS = 144


def reference_yoco_keep_tokens(n_prefill_layer, num_layers, window_size, verification_width, prefill_tokens):
    """Return the 16-aligned YOCO consumer suffix width when CED truncation applies."""
    if not 0 < n_prefill_layer < num_layers:
        return None
    keep = max(REFERENCE_YOCO_MIN_KEEP_TOKENS, (window_size + verification_width - 1 + 15) // 16 * 16)
    return keep if prefill_tokens > keep else None


class ReferenceModel(nn.Module):
    """Own the original computation with Flare expert and Engram storage bindings.

    This is a preparation/capture component. Sampling, accepted-state commit,
    vision input packing and native graph publication belong to the enclosing
    recipe. Construction consumes reference parameter storage; it cannot be
    retried with the same parameter tree after a partial failure.
    """

    def __init__(self, parameters, experts, delivery, *, execution, context,
                 request_capacity, prefill_tokens, verification_tokens, yoco_keep_tokens=None):
        from deepinfer import AttrDict
        from deepinfer.engine.engine_type import EngineType
        from deepinfer.hugging_face import build_mm_processor, load_processor_config
        from deepinfer.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from deepinfer.model_executor.models.deepseek import DeepseekModel
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM
        from deepinfer.model_executor.models.mtp.dflash import DFlash
        from deepinfer.model_executor.models.v4_modules.cross_layer import build_shared_layer_topology
        from deepinfer.model_executor.models.vision_encoder.vl_aligner import VisionLanguageInstructTuner
        from deepinfer.timer import cpu_timer, cuda_graph_timer, cuda_timer

        super().__init__()
        cpu_timer.active = cuda_graph_timer.active = cuda_timer.active = False
        if not parameters.loaded or parameters.assembled:
            raise ValueError("reference model requires loaded, unconsumed parameters")
        if (any(type(value) is not int or value < 1 for value in
                (context, request_capacity, prefill_tokens, verification_tokens))
                or prefill_tokens > context or execution.draft_tokens < 0):
            raise ValueError("reference assembly requires positive capacities within the context limit")
        if yoco_keep_tokens is None:
            if execution.yoco_prefill_truncation != "off":
                raise ValueError("reference YOCO compaction requires an explicit suffix token capacity")
        elif (type(yoco_keep_tokens) is not int or execution.yoco_prefill_truncation != "compact"
                or yoco_keep_tokens < parameters.config.gpt.window_size[0] + execution.draft_tokens
                or yoco_keep_tokens >= prefill_tokens):
            raise ValueError("reference YOCO suffix capacity must cover the sliding-window ring within prefill chunks")
        self.yoco_keep_tokens = yoco_keep_tokens
        self.expert_stagers = {}
        self.stage_min_rows = 1024
        self.draft_head = None
        self.markov_head = None
        cfg = AttrDict(parameters.config.gpt, runtime_max_model_len=context,
            max_num_seqs=request_capacity, max_num_batched_tokens=max(prefill_tokens, verification_tokens),
            max_num_decoding_tokens=verification_tokens, max_num_prefilling_tokens=prefill_tokens,
            engine_type=EngineType.NORMAL)
        target_keys = [f"layers.{i}.ffn" for i in range(cfg.num_hidden_layers)]
        draft_keys = ([f"mtp.{i}.ffn" for i in range(cfg.num_nextn_predict_layers)]
                      if execution.draft_tokens else [])
        if set(experts.modules) != set(target_keys + draft_keys):
            raise ValueError("reference model requires every prepared target and draft expert module")
        cfg.vision = parameters.config.vl_aligner.vision.params
        cfg.aligner = parameters.config.vl_aligner.aligner.params
        processor = build_mm_processor(load_processor_config(parameters.checkpoint.source.metadata_path))
        DeepseekVLForCausalLM.derive_image_transform_cfg(processor.image_processor, cfg)
        self.cfg, self.experts, self.delivery = cfg, experts, delivery
        parameters.assembled = True
        sources = parameters.model.module.module
        total_layers = cfg.num_hidden_layers + (cfg.num_nextn_predict_layers if execution.draft_tokens else 0)
        topology = build_shared_layer_topology(cfg, total_layers)
        self.model = DeepseekModel(sources[:cfg.num_hidden_layers + 1], sources[cfg.num_hidden_layers + 1],
            cfg, execution.draft_tokens, topology,
            prepared_mlps=[experts.modules[key] for key in target_keys], execution=execution)
        self.lm_head = ParallelLMHead(cfg.vocab_size, cfg.hidden_size,
            weight=sources[cfg.num_hidden_layers + 1].shared_head.head.weight)
        self.draft = (DFlash(parameters.model, cfg, self.model.embed_tokens, self.lm_head,
            prepared_mlps=[experts.modules[key] for key in draft_keys], execution=execution)
            if execution.draft_tokens else None)
        layers = [self.model.layers[index].engram for index in cfg.engram.engram_layer_ids]
        delivery.bind(layers)
        self.model.engram_runtime = delivery
        with torch.device(self.lm_head.weight.device):
            self.vision_language_tuner = VisionLanguageInstructTuner(cfg, sources[0],
                vision_source=parameters.model.vision_pipeline.aligner)
        _record_model_numeric_tensors(self)

    def _special_image_token_mask(self, input_ids):
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM

        return DeepseekVLForCausalLM._special_image_token_mask(self, input_ids)

    def bind_draft_heads(self, *, draft_head=None, markov_head=None):
        """Attach quantized draft vocabulary or Markov heads derived from checkpoint weights."""
        if (draft_head is not None or markov_head is not None) and self.draft is None:
            raise ValueError("derived draft heads require an enabled draft model")
        if markov_head is not None and not self.draft.enable_markov:
            raise ValueError("derived Markov head requires an enabled Markov head")
        self.draft_head = draft_head
        self.markov_head = markov_head
        for obj in (draft_head, markov_head):
            if obj is None:
                continue
            for value in vars(obj).values():
                if isinstance(value, torch.Tensor):
                    record_numeric_tensor(value)

    def bind_prefill_staging(self, expert_stagers, *, min_rows=1024):
        """Attach prepared double-buffered cold-expert stagers for prompt execution."""
        if type(min_rows) is not int or min_rows < 1:
            raise ValueError("prefill staging requires a positive row threshold")
        self.expert_stagers = dict(expert_stagers)
        self.stage_min_rows = min_rows

    def bind_low_latency_fp8_projections(self, *, target_rows, draft_rows=(), session):
        """Wrap eligible FP8 attention projections with SM103 low-latency GEMMs for M <= 8."""
        from deepinfer.model_executor.layers.linear import ReplicatedLinear
        from ignition.components.sm103_projection import (
            LOW_LATENCY_FP8_MAX_ROWS, LowLatencyFP8Linear, is_sm103,
        )

        if not is_sm103(self.lm_head.weight.device):
            return
        small_target = tuple(sorted({r for r in target_rows if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}))
        small_draft = tuple(sorted({r for r in draft_rows if 1 <= r <= LOW_LATENCY_FP8_MAX_ROWS}))

        def wrap_attention(attn, rows, prefix):
            if not rows:
                return
            for attr in ("q_a_kv_swa_proj", "q_a_kv_proj", "kv_swa_proj", "q_b_proj", "out_proj"):
                mod = getattr(attn, attr, None)
                if (isinstance(mod, ReplicatedLinear)
                        and mod.weight.dtype == torch.float8_e4m3fn
                        and mod.bias is None and not mod.use_fp32_output
                        and mod.input_size % 128 == 0 and mod.output_size % 128 == 0):
                    setattr(attn, attr, LowLatencyFP8Linear(mod, rows, session=session, name=f"{prefix}.{attr}"))

        for index, layer in enumerate(self.model.layers):
            wrap_attention(layer.self_attn, small_target, f"layers.{index}.attn")
        if self.draft is not None and small_draft:
            for index, layer in enumerate(self.draft.layers):
                wrap_attention(layer.self_attn, small_draft, f"mtp.{index}.attn")

    def forward(self, caches, metadata, hidden_states=None, *, speculation_enabled=True):
        from deepinfer.model_executor.models.deepseek_vl import DeepseekVLForCausalLM

        if hidden_states is None and metadata.is_prompt:
            hidden_states = DeepseekVLForCausalLM.get_input_embeddings(self, metadata)
        dflash = self.draft if speculation_enabled else None
        return self.model(caches, metadata, hidden_states=hidden_states, dflash=dflash,
                          expert_stagers=self.expert_stagers,
                          stage_min_rows=self.stage_min_rows)

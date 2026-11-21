# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare DeepSeek text weights, file-backed Engram tables, and native graph catalogs."""

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import torch

from ignition.components.completion import CompletionOutputs, VECTOR_TOKENS_FIELD
from ignition.components.prompt_logprobs import (
    PreparedScoredPrefill, PromptScoring, prefill_graph_count, prefill_modes,
)

from ignition.components.capacity import (
    speculative_token_capacity, batch_capacities, prefill_capacities,
    request_capacity, serving_modes,
)
from ignition.components.parallel import TensorParallelShard
from ignition.components.sampling import AdmissionBuffers, DeviceRequestState
from ignition.components.state_transfer import StateCommands
from ignition.preparation.capture import capture_forwards
from ignition.preparation.rebuild_manifest import declare_weights
from .contract import prepare as prepare_contract, PATCH_CAPACITIES, INPUT_VOCABULARY
from ignition.components.deepseek_vision import (PreparedVision, VisionInputs, VisionScratch,
    PrefillVisionInputs, PreparedVisionGeneration, VisionState)
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.collectives import (CollectivePreparation, create_all_gather_runtime,
    create_all_reduce_runtime, create_dma_all_reduce_runtime, create_result_broadcast_runtime, create_roce_runtime)
from ignition.preparation.disposable import storage_ranges
from ignition.preparation.execution import GraphSignature
from ignition.preparation.host_memory import virtual_device_memory, release_unused
from ignition.preparation.frontend import PromptEncoder, prepare_frontend
from ignition.preparation.model import PreparedModel
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession, with_autotune_effort
from ignition.preparation.tuning_replicas import TuningReplica
from ignition.progress import phase

from ignition.components.paged_batch import BatchInputs, Metadata, PreparedBatch
from .recipe import RECIPE
from .attention import INDEXER_QUERY_ROWS
from .decoder import PreparedDecoder, Scratch, DraftScratch
from ignition.components.dspark import PreparedContext, PreparedMarkov
from .dspark import PreparedDraft
from .engram import Tables
from ignition.components.paged_generation import PreparedGeneration
from ignition.components.paged_maintenance import PreparedMaintenance
from .pool import ContinuationPool, PAGE_TOKENS, continuation_layout, page_bytes
from ignition.components.paged_resume import PreparedResume
from .weights import ModelWeights
from .sm103 import SM103Config
from ignition.preparation.bootstrap import BasePreparationConfig


_CONTEXT = partial(PreparedContext, compact_prefill=True)


def _prepare_draft(*args, draft_vocabulary_runtime, vocabulary_runtime=None, **kwargs):
    # Fixed-width PCIe channels distinguish FP32 target words from BF16 proposals.
    runtime = vocabulary_runtime if draft_vocabulary_runtime is None else draft_vocabulary_runtime
    return PreparedDraft(*args, vocabulary_runtime=runtime, **kwargs)


def _forward_factories(mla_compute_mode, *, vision,
                       draft_vocabulary_runtime=None, examples=None):
    """Bind attention arithmetic and the proposal head's BF16 gather channel."""
    return (partial(PreparedDecoder, vision=vision, mla_compute_mode=mla_compute_mode, examples=examples),
            partial(_prepare_draft, draft_vocabulary_runtime=draft_vocabulary_runtime,
                    mla_compute_mode=mla_compute_mode, examples=examples))


def _can_race_contextual_plans(config, signatures, device):
    if not config.joint_tuning:
        return False
    if config.autotune_effort == "off":
        raise ValueError("--joint-tuning requires --autotune-effort default or audit")
    from b12x.quantization import mla_norm_mxfp8

    if not mla_norm_mxfp8.is_supported(device):
        raise ValueError("--joint-tuning requires fused MXFP8 query normalization on the selected device")
    if not any(signature.kind in ("prefill", "scored_prefill", "decode", "speculative")
               for signature in signatures.values()):
        raise ValueError("--joint-tuning requires prepared text graphs")
    return True


@dataclass(frozen=True)
class PreparationConfig(BasePreparationConfig):
    prompt_logprobs: bool = False
    image_concurrency: int | None = None
    joint_tuning: bool = False
    devices: tuple[int, ...] = (0, 1, 2, 3)
    # Device ordinals are host-local when each rank runs on its own Spark.
    hosts: tuple[str, ...] | None = None
    roce_ring: bool = False
    roce_chain: bool = False
    context: int = 1048576
    max_active: int = 32
    minimum_pages: int = 8
    # An explicit cache capacity makes memory comparisons independent of automatic sizing.
    kv_pages: int | None = None
    dspark: bool = True
    draft_head_quantization: str | None = None
    markov_head_quantization: str | None = None
    vocabulary_parallel_sampling: bool = False
    speculative_token_capacity: int | None = 128
    speculative_graph_budget: int = 64
    # Overlap shared and routed experts within each eligible text graph.
    stream_overlap: bool = True
    expert_scale_compression: str | None = None
    mla_compute_mode: str = "auto"
    packed_verification: bool | None = None
    draft_prefix_selection: str | None = None
    speculation_reference_graphs: bool = False
    sm103: SM103Config | None = None

    def __post_init__(self):
        if self.mtp_only and (not self.dspark
                or max(self.speculative_widths, default=1) <= 1 or self.speculative_fixed_width == 1):
            raise ValueError("--mtp-only requires enabled learned speculation with positive draft depth")
        if self.speculative_widths and max(self.speculative_widths) > 8:
            raise ValueError("DeepSeek V4.1 draft model supports at most seven draft tokens")
        from ignition.components.capacity import image_capacity
        from ignition.components.compressed_mla import validate_mla_compute_mode

        validate_mla_compute_mode(self.mla_compute_mode)
        from ignition.recipes.declaration import Speculation
        Speculation(self.draft_tokens, draft_head_quantization=self.draft_head_quantization,
                    markov_head_quantization=self.markov_head_quantization,
                    vocabulary_parallel_sampling=self.vocabulary_parallel_sampling)
        self.speculative_shapes
        self.packed_speculative_shapes
        drafting = self.dspark and self.draft_tokens > 0
        for name in ("packed_verification",):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, drafting)
            elif type(value) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.draft_prefix_selection is None:
            object.__setattr__(self, "draft_prefix_selection", "gpu" if drafting else "cpu")
        if self.draft_prefix_selection not in ("cpu", "gpu"):
            raise ValueError("draft prefix selection must be cpu or gpu")
        if drafting:
            if not self.packed_verification or self.draft_prefix_selection != "gpu":
                raise ValueError("DeepSeek V4.1 DSpark requires packed GPU draft prefix selection")
            if self.speculative_fixed_width is not None or self.speculation_reference_graphs:
                raise ValueError("DeepSeek V4.1 DSpark requires adaptive packed GPU rounds")
        elif self.packed_verification or self.draft_prefix_selection != "cpu" or self.speculation_reference_graphs:
            raise ValueError("packed GPU speculation requires positive DSpark draft depth")
        object.__setattr__(self, "image_concurrency", image_capacity(self.max_active, self.image_concurrency))
        if self.expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        if type(self.stream_overlap) is not bool:
            raise TypeError("stream_overlap must be a boolean")
        placements = tuple(zip(self.hosts or ("",) * len(self.devices), self.devices))
        if self.sm103 is not None and (not isinstance(self.sm103, SM103Config)
                or self.hosts is not None or self.kernel_cache is None):
            raise ValueError("SM103 preparation requires explicit placement and kernel cache on one local GPU")
        if self.sm103 is not None and self.expert_scale_compression is not None:
            raise ValueError("SM103 MegaMoE requires expanded native expert scales; CSF checkpoint loading remains lossless")
        if self.sm103 is not None:
            if self.prompt_logprobs:
                raise ValueError("reference SM103 preparation does not support --prompt-logprobs")
            if self.joint_tuning:
                raise ValueError("reference SM103 preparation does not support --joint-tuning")
            if self.vocabulary_parallel_sampling:
                raise ValueError("reference SM103 preparation does not support vocabulary-parallel sampling on one GPU")
            if self.draft_head_quantization not in (None, "nvfp4", "mxfp8"):
                raise ValueError("reference SM103 draft head quantization must be None, 'nvfp4', or 'mxfp8'")
            if self.markov_head_quantization not in (None, "fp8"):
                raise ValueError("reference SM103 Markov head quantization must be None or 'fp8'")
        worlds = (1,) if self.sm103 is not None else (3, 4, 6)
        if (not isinstance(self.devices, tuple) or len(self.devices) not in worlds
                or len(set(placements)) != len(self.devices)
                or self.hosts is not None and (not isinstance(self.hosts, tuple) or len(self.hosts) != len(self.devices))
                or any(type(d) is not int or d < 0 for d in self.devices)):
            raise ValueError("DeepSeek V4.1 Flash preparation requires three, four, or six distinct GPUs, or explicit SM103 TP1 preparation")
        from ignition.planning import host_shape_error

        error = host_shape_error(self.roce_ring, self.roce_chain, None if self.hosts is None else len(self.hosts))
        if error:
            raise ValueError(error)
        if not PAGE_TOKENS <= self.context <= 1048576 or self.context % PAGE_TOKENS:
            raise ValueError("DeepSeek context must be 128-aligned in [128,1048576]")
        if not 64 <= self.prefill_chunk_size <= self.context or self.prefill_chunk_size % 64:
            raise ValueError("prefill chunk size must be 64-aligned in [64,context]")
        if (self.sm103 is not None and self.sm103.prefill_staging is not None
                and self.sm103.prefill_staging.min_rows > self.prefill_chunk_size):
            raise ValueError("prefill staging row threshold exceeds the prepared prefill capacity")
        request_capacity(self.max_active)
        self.speculative_shapes
        if self.speculative_fixed_width is not None and self.speculative_fixed_width not in self.speculative_widths:
            raise ValueError("fixed verification width must belong to the prepared widths")
        if self.minimum_pages <= 0 or self.snapshot_budget_bytes <= 0 or self.headroom_bytes < 0:
            raise ValueError("page and snapshot budgets must be positive and headroom nonnegative")
        if self.kv_pages is not None and (type(self.kv_pages) is not int or self.kv_pages < self.minimum_pages):
            raise ValueError("kv_pages must cover the minimum page capacity")

    @property
    def request_capacity(self):
        return request_capacity(self.max_active)

    @property
    def prefill_capacities(self):
        return prefill_capacities(self.prefill_chunk_size)

    @property
    def batch_capacities(self):
        return tuple(reversed(batch_capacities(self.max_active)))

    @property
    def speculative_shapes(self):
        from ignition.components.capacity import speculative_shapes

        return tuple((batch, width) for batch, width in
            speculative_shapes(self.max_active, self.speculative_widths, self.speculative_token_capacity)
            if width == 1) if self.dspark else ()

    @property
    def packed_speculative_shapes(self):
        return tuple(sorted({(batch, tokens) for batch, _, tokens in self.packed_speculative_depth_shapes}))

    @property
    def packed_speculative_depth_shapes(self):
        from ignition.components.capacity import packed_speculative_depth_shapes

        return packed_speculative_depth_shapes(self.max_active, self.speculative_widths,
            self.speculative_token_capacity, self.speculative_graph_budget) if self.draft_tokens else ()

    @property
    def decode_tokens(self):
        return max(self.request_capacity, max((tokens for _, tokens in self.packed_speculative_shapes), default=0))

    @property
    def draft_tokens(self):
        return max(self.speculative_widths) - 1 if self.dspark else 0

    @property
    def all_reduce_tokens(self):
        return max(self.decode_tokens, self.request_capacity * self.draft_tokens,
                   *(capacity for capacity in self.prefill_capacities if capacity < 128))

    @property
    def completion_token_capacity(self):
        return max(self.speculative_widths) if self.dspark else 1


def _prepare_speculation_catalog(config, speculation, *, add, buffers,
                                 forward_inputs, host_allocate, device):
    if speculation is None:
        return None
    from ignition.components.parallel_draft import PreparedParallelDraftRound

    confidence = None
    if config.draft_tokens:
        from ignition.components.packed_speculation import PreparedPackedSpeculativeRound

        slots = config.request_capacity
        confidence = host_allocate("proposal_confidence", (2, slots, config.draft_tokens), torch.float32)
        buffers["proposal_confidence"] = confidence.device_view
        prefix_limits = torch.zeros(slots, dtype=torch.int32, device=device)
        buffers["speculative_prefix_limits"] = prefix_limits
        packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
        packed_inputs += ("speculative_prefix_limits",)
        proposal_shapes = {(batch, width - 1) for batch, width, _ in config.packed_speculative_depth_shapes}
        proposals = {}
        for batch, depth in sorted(proposal_shapes, reverse=True):
            name = f"proposal{batch}x{depth + 1}"
            source = PreparedParallelDraftRound(speculation, batch, depth + 1, name,
                proposal_only=True)
            source.prepare()
            proposals[batch, depth] = source
    for batch, _ in reversed(config.speculative_shapes):
        name = f"speculative{batch}x1"
        forward = PreparedParallelDraftRound(speculation, batch, 1, name)
        forward.prepare()
        add(name, forward, "speculative", batch, 1, forward_inputs, enabled=True)
    for batch, logical_width, packed_tokens in sorted(config.packed_speculative_depth_shapes,
            key=lambda shape: (shape[2], shape[0], shape[1]), reverse=True):
        depth = logical_width - 1
        target_name = f"verification{batch}x{logical_width}-tokens{packed_tokens}"
        target = PreparedParallelDraftRound(speculation, batch, logical_width, target_name,
            generate_proposals=False, target_token_capacity=packed_tokens)
        target.prepare()
        name = f"packed-speculative{batch}x{logical_width}-drafts{depth}-tokens{packed_tokens}"
        forward = PreparedPackedSpeculativeRound(proposals[batch, depth], target, prefix_limits, confidence.device_view,
            name=name)
        add(name, forward, "speculative", batch, logical_width, packed_inputs,
            enabled=True, packed_tokens=packed_tokens, draft_tokens=depth)
    return confidence


def _collective_runtimes(config, group, device, *, coordinator, core_storage, verification_samples):
    """Share one RoCE channel across ordered graph collectives on separate hosts."""
    if len(config.devices) == 1:
        return None, None, None, None, None
    tokens = max(config.prefill_chunk_size, config.decode_tokens, config.request_capacity * config.draft_tokens)
    sample_rows = max(verification_samples, config.request_capacity * config.draft_tokens)
    layouts = ()
    if config.dspark:
        from ignition.components.draft_sampling import PreparedDraftBlockBroadcast
        from ignition.components.speculation import SpeculativeAcceptance

        batch = max(config.batch_capacities)
        layouts = (SpeculativeAcceptance.broadcast_layout(batch, max(config.speculative_widths)),)
        if config.draft_tokens:
            layouts += (PreparedDraftBlockBroadcast.broadcast_layout(batch, config.draft_tokens),)
    if coordinator.spans_hosts:
        from ignition.components.collectives import packed_broadcast_layout

        # Vocabulary partitions and packed sampling results use the same serial gather channel.
        result_bytes = max((packed_broadcast_layout(layout.values())[1] for layout in layouts), default=0)
        _, local_vocab = TensorParallelShard(0, len(config.devices)).padded_partition(129280, alignment=64)
        gather_bytes = max(sample_rows * local_vocab * 4,
                           (result_bytes + 15) // 16 * 16)
        decode_tokens = config.all_reduce_tokens
        runtime = create_roce_runtime(group, device, max_tokens=decode_tokens, pipe_tokens=tokens,
            hidden_size=5120, gather_bytes=gather_bytes, ring=config.roce_ring, ring_open=config.roce_chain, core_storage=core_storage)
        return runtime, runtime, runtime, runtime if layouts else None, runtime if config.dspark else None
    ipc = core_storage.peer_storage
    collective = create_all_reduce_runtime(group, device, collective_tuning=config.collective_tuning, max_tokens=config.all_reduce_tokens, hidden_size=5120, ipc=ipc)
    prefill = create_dma_all_reduce_runtime(group, device,
        max_tokens=config.prefill_chunk_size, hidden_size=5120, ipc=ipc)
    _, local_vocab = TensorParallelShard(0, len(config.devices)).padded_partition(129280, alignment=64)
    vocabulary = create_all_gather_runtime(group, device, max_rows=sample_rows,
        columns=local_vocab * len(config.devices) * 2, ipc=ipc)
    draft_vocabulary = (create_all_gather_runtime(group, device, max_rows=sample_rows,
        columns=local_vocab * len(config.devices), ipc=ipc) if config.dspark else None)
    result = create_result_broadcast_runtime(group, device, layouts=layouts, ipc=ipc) if layouts else None
    return collective, prefill, vocabulary, result, draft_vocabulary


def _bind_reference_draft_heads(model, config):
    if model.draft is None or (config.draft_head_quantization is None and config.markov_head_quantization is None):
        return
    if model.draft_head is not None or model.markov_head is not None:
        return
    from ignition.components.markov_sampling import fp8_markov_head
    from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head

    draft_head = None
    if config.draft_head_quantization == "nvfp4":
        draft_head = nvfp4_draft_head(model.lm_head.weight)
    elif config.draft_head_quantization == "mxfp8":
        draft_head = mxfp8_draft_head(model.lm_head.weight)
    markov_head = (fp8_markov_head(model.draft.markov_w2.weight)
                   if config.markov_head_quantization == "fp8" else None)
    model.bind_draft_heads(draft_head=draft_head, markov_head=markov_head)


def _prepare_reference_rank(snapshot, config, *, rank, group, progress=None, core_storage=None):
    import gc
    from deepinfer.model_executor.execution_settings import ExecutionSettings
    from deepinfer.model_executor.parallel_utils.parallel_state import LocalProcessGroup, PG
    from ignition.preparation.persistent import allocated_tensor_bytes
    from .config import ModelConfig
    from .reference_checkpoint import ReferenceCheckpoint
    from .reference_engram import ReferenceEngramDelivery
    from .reference_model import (
        ReferenceExpertPreparation, ReferenceModel, reference_yoco_keep_tokens,
    )
    from .reference_parameters import ReferenceParameters

    if core_storage is None:
        raise ValueError("DeepSeek preparation requires core storage")
    if rank != 0 or group is not None or len(config.devices) != 1:
        raise ValueError("reference SM103 preparation requires one local GPU rank")
    snapshot = Path(snapshot).resolve()
    device = torch.device("cuda", config.devices[0])
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device) != (10, 3):
        raise ValueError("reference SM103 preparation requires an SM103 GPU")
    if torch.distributed.is_initialized():
        raise ValueError("reference SM103 preparation requires single-device process geometry")
    if PG.world is None:
        PG.initialize_local()
    if not isinstance(PG.world, LocalProcessGroup):
        raise ValueError("reference SM103 preparation requires LocalProcessGroup")

    with phase(progress, "Inspect checkpoint"):
        checkpoint = ReferenceCheckpoint(snapshot)
        metadata_path = checkpoint.source.metadata_path
        model_config = ModelConfig.load(metadata_path)
    dspark_loaded = config.dspark and config.draft_tokens > 0
    config.sm103.validate_model(model_config, dspark=dspark_loaded)
    modes = serving_modes(config.dspark, config.mtp_only)
    verification_width = max(config.speculative_widths) if dspark_loaded else 1
    yoco_keep_tokens = reference_yoco_keep_tokens(
        model_config.decoder_start if model_config.decoder_start is not None else model_config.layers,
        model_config.layers, model_config.text["sliding_window"], verification_width,
        config.prefill_chunk_size)
    target_row_set = set(config.prefill_capacities)
    if yoco_keep_tokens is not None:
        target_row_set.add(yoco_keep_tokens)
    if False in modes:
        target_row_set.update(config.batch_capacities)
    if True in modes:
        target_row_set.update(batch for batch, _ in config.speculative_shapes)
        target_row_set.update(tokens for _, _, tokens in config.packed_speculative_depth_shapes)
    target_rows = tuple(sorted(target_row_set))
    draft_rows = tuple(sorted({batch * (width - 1) for batch, width, _ in config.packed_speculative_depth_shapes}))

    identity = hashlib.sha256(json.dumps({"config": json.loads((metadata_path / "config.json").read_text()),
        "headers": [s.header_sha256 for s in checkpoint.source.shards], "world_size": 1,
        "contract": "deepseek41.reference-sm103.v2",
        "sm103": config.sm103.to_dict()}, sort_keys=True).encode()).hexdigest()
    session = RecordedPreparationSession(
        compile_workers=config.compile_workers, collective_tuning=config.collective_tuning,
        collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort,
        tuning_replica=config.tuning_replica, native_dense_activations=config.native_dense_activations,
        lock_gpu_clocks=config.lock_gpu_clocks, device=device, checkpoint=identity,
        workload_schema="deepseek41-reference-sm103", numerical_policy="reference-checkpoint",
        coordinator=None, headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
    session.__enter__()

    with core_storage.transient():
        before_weights = allocated_tensor_bytes(device)
        with phase(progress, "Load reference model and tiered experts"):
            compute_sm_count = 0
            if config.sm103.dynamic_mover is not None:
                import deep_gemm
                from ignition.components.sm103_expert_mover import query_green_context_compute_sms

                _, compute_sm_count = query_green_context_compute_sms(
                    device.index, config.sm103.dynamic_mover.worker_sm_count
                )
                deep_gemm.set_num_sms(compute_sm_count)
                try:
                    import tile_kernels
                    if hasattr(tile_kernels, "set_num_sms"):
                        tile_kernels.set_num_sms(compute_sm_count)
                except ImportError:
                    pass
            parameters = ReferenceParameters(checkpoint)
            parameters.load_ordinary(device)
            experts = ReferenceExpertPreparation.load(checkpoint, dict(config.sm103.placements),
                target_rows=target_rows, draft_rows=draft_rows, storage=core_storage,
                kernel_cache=config.kernel_cache, session=session,
                host_lane_clusters=config.sm103.host_lane_clusters, dspark=dspark_loaded,
                dynamic_mover=config.sm103.dynamic_mover, compute_sm_count=compute_sm_count)
            sequence = torch.zeros(1, dtype=torch.int64, device=device)
            frame = torch.zeros(1, dtype=torch.int32, device=device)
            delivery = ReferenceEngramDelivery.from_checkpoint(checkpoint, parameters.config.gpt,
                capacity=max(config.prefill_chunk_size, config.decode_tokens),
                sequence=sequence, frame=frame, host_allocator=core_storage.host_allocate)
            verification_samples = (config.request_capacity * max(config.speculative_widths)
                                    if config.packed_verification else config.decode_tokens)
            execution = ExecutionSettings(draft_tokens=config.draft_tokens, batch_invariant=False,
                yoco_prefill_truncation="compact" if yoco_keep_tokens is not None else "off",
                index_memory_limit=max(256 << 20, config.context * 1024),
                sampler_topk=0, confidence_temperature=1.)
            model = ReferenceModel(parameters, experts, delivery, execution=execution,
                context=config.context, request_capacity=config.request_capacity,
                prefill_tokens=config.prefill_chunk_size, verification_tokens=verification_samples,
                yoco_keep_tokens=yoco_keep_tokens)
            _bind_reference_draft_heads(model, config)
            del parameters
            gc.collect()
            release_unused()
        weights_bytes = max(0, allocated_tensor_bytes(device) - before_weights)

    if progress is not None:
        progress(f"Resident reference tensors: {weights_bytes / (1 << 30):.3f} GiB")
    tier_bytes = config.sm103.expert_bytes(model_config, dspark=dspark_loaded)
    return _assemble_reference_prepared_model(model, config=config, metadata_path=metadata_path,
        snapshot_name=snapshot.name, identity=identity, session=session, core_storage=core_storage,
        sequence=sequence, frame=frame, weights_bytes=weights_bytes, tier_bytes=tier_bytes,
        has_vision=model_config.vision is not None, progress=progress)


def _assemble_reference_prepared_model(model, *, config, metadata_path, snapshot_name, identity,
                                       session, core_storage, sequence, frame, weights_bytes,
                                       tier_bytes, has_vision=True, progress=None):
    from ignition.components.expert_row_storage import megamoe_row_storage
    from ignition.components.speculation import SpeculativeRngState
    from ignition.preparation.tensor_metadata import record_numeric_tensor
    from .reference_cache import ReferenceCacheLayout, ReferenceCacheStorage, ReferenceCacheTransaction
    from .reference_engram import ReferenceEngramHistory
    from .reference_generation import ReferenceGeneration
    from .reference_model import declare_reference_weights
    from .reference_resume import ReferenceResume
    from .reference_speculation import ReferenceSpeculation
    from .reference_vision import ReferenceVision

    device = torch.device("cuda", config.devices[0])
    shard = TensorParallelShard(0, 1)
    experts, delivery = model.experts, model.delivery
    dspark_loaded = config.dspark and config.draft_tokens > 0
    modes = serving_modes(config.dspark, config.mtp_only)

    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    _bind_reference_draft_heads(model, config)
    target_small_rows = sorted({
        *(r for r in config.batch_capacities if r <= 8),
        *(b for b, _ in config.speculative_shapes if b <= 8),
        *(t for _, _, t in config.packed_speculative_depth_shapes if t <= 8),
    })
    draft_small_rows = sorted({
        b * (w - 1) for b, w, _ in config.packed_speculative_depth_shapes if 1 <= b * (w - 1) <= 8
    })
    model.bind_low_latency_fp8_projections(
        target_rows=target_small_rows, draft_rows=draft_small_rows, session=session)
    with phase(progress, "Allocate reference model state"):
        verification_width = max(config.speculative_widths) if dspark_loaded else 1
        geometry = ReferenceCacheLayout.from_config(model.cfg, verification_width=verification_width,
            page_tokens=PAGE_TOKENS, dspark=dspark_loaded)
        snapshots = geometry.continuation.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete DeepSeek continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots
        state_rows = snapshots + execution_rows
        pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        if config.kv_pages is not None:
            pages = config.kv_pages
        bytes_per_page = sum(shape[0] for _, shape in geometry.pages)
        total_memory = virtual_device_memory(device, None, tuning_replica=config.tuning_replica)
        pool = ReferenceCacheStorage.allocate(geometry, state_rows + 1, pages, device,
            storage=core_storage, maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // geometry.continuation.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots))
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        history = ReferenceEngramHistory.from_config(pool.reserved_region("engram_history"), model.cfg)
        requests = DeviceRequestState.allocate(device, vocab_size=model.cfg.vocab_size, capacity=slots)
        inputs = BatchInputs.allocate(max(config.prefill_chunk_size, slots), blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=model.cfg.vocab_size, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        completion = CompletionOutputs.allocate(
            (2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)

        vision_state = vision_inputs = vision_prefill = feature_pool = None
        if has_vision and config.image_concurrency:
            tuner = model.vision_language_tuner
            vision_weights = {name: getattr(tuner.params, name)
                              for name in ("image_start", "image_end", "image_newline")}
            vision_inputs = VisionInputs.allocate(device, patches=max(PATCH_CAPACITIES))
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            feature_pool = core_storage.scratch_pool()
            with feature_pool:
                feature_arena = torch.zeros((config.image_concurrency, 4096,
                    vision_weights["image_start"].numel()), dtype=torch.bfloat16, device=device)
            vision_state = VisionState(vision_weights, inputs, vision_prefill, requests, features=4096,
                token_base=model.cfg.vocab_size,
                sentinel_names=("image_start", "image_start", "image_newline", "image_end"),
                feature_type=1, image_slots=config.image_concurrency, arena=feature_arena)

        buffers = dict(inputs.buffers(), **delivery.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        if experts.mover is not None:
            expert_rows, row_buffers = None, {}
        else:
            expert_rows, row_buffers = megamoe_row_storage(
                (index, experts.weights[f"layers.{index}.ffn"], f"weights.layers[{index}]")
                for index in range(len(model.model.layers))
                if experts.weights[f"layers.{index}.ffn"].host_tier is not None)
        buffers.update(row_buffers)
        if vision_state is not None:
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())

        expert_stagers = {}
        staging = None if config.sm103 is None else config.sm103.prefill_staging
        if staging is not None:
            from ignition.components.megamoe_staging import PreparedExpertStage

            streams = ([torch.cuda.Stream(device=device) for _ in range(2)] if config.stream_overlap
                       else [None, None])
            placements = dict(config.sm103.placements)
            stage_limit = (model.model.yoco_boundary_layer_idx
                           if model.yoco_keep_tokens is not None and model.yoco_keep_tokens < staging.min_rows
                           else len(model.model.layers))
            for index in range(stage_limit):
                expert = experts.weights[f"layers.{index}.ffn"]
                host_experts = placements[f"layers.{index}.ffn"].host
                if expert.host_tier is not None and expert.host_tier.rows >= staging.min_cold_experts:
                    expert_stagers[index] = PreparedExpertStage(
                        expert.host_tier, streams[len(expert_stagers) % 2],
                        expert.expert_map, host_experts, session=session, workspace=arena,
                        name=f"prefill-stage-{index}", min_cold=staging.min_cold_experts)
            if not expert_stagers:
                raise ValueError("prefill staging requires eligible target host experts")
        model.bind_prefill_staging(expert_stagers, min_rows=1024 if staging is None else staging.min_rows)

    forwards, signatures = {}, {}
    count = (len(modes) * (len(config.prefill_capacities) + 4 * len(config.batch_capacities))
             + len(config.speculative_shapes) + len(config.packed_speculative_depth_shapes)
             + (len(config.batch_capacities) if False in modes else 0)
             + (len(PATCH_CAPACITIES) if vision_state is not None else 0))
    catalog_bytes = {}
    catalog_previous = torch.cuda.memory_allocated(device)
    confidence = rng = proposal_generation = proposal_offset = None
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            nonlocal catalog_previous
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields),
                speculation_enabled=enabled, packed_token_capacity=packed_tokens,
                draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0)
            allocated = torch.cuda.memory_allocated(device)
            catalog_bytes[kind] = catalog_bytes.get(kind, 0) + allocated - catalog_previous
            catalog_previous = allocated
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, None):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    batch_op = PreparedBatch(inputs, requests, Metadata.allocate(capacity, rows, blocks, device),
                        mode=mode, shard=shard, pool=pool,
                        input_vocabulary=INPUT_VOCABULARY if vision_state is not None else None,
                        validate_prefill=vision_state.validate_prefill
                            if vision_state is not None and mode == "prefill" else None)
                    forward = ReferenceGeneration(model, batch_op, history, completion, frame, sequence,
                        session=session, vision=vision_state if mode == "prefill" else None,
                        speculation_enabled=enabled)
                    fields = ((*forward_inputs, "vision_token_indices", "vision_feature_slot")
                              if mode == "prefill" and vision_state is not None else forward_inputs)
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled=enabled)

        if config.dspark:
            rng = SpeculativeRngState.allocate(device, capacity=slots)
            proposal_generation = torch.zeros(slots, dtype=torch.int64, device=device)
            proposal_offset = torch.zeros(slots, dtype=torch.int64, device=device)
            for tensor in (rng.generation, rng.target_offset, proposal_generation, proposal_offset):
                record_numeric_tensor(tensor)
            shared_transaction = ReferenceCacheTransaction(pool) if config.draft_tokens else None
            if config.draft_tokens:
                confidence = core_storage.host_allocate(
                    "proposal_confidence", (2, slots, config.draft_tokens), torch.float32)
                confidence.host_view.zero_()
                buffers["proposal_confidence"] = confidence.device_view
                prefix_limits = torch.zeros(slots, dtype=torch.int32, device=device)
                buffers["speculative_prefix_limits"] = prefix_limits
                packed_inputs = tuple(field for field in forward_inputs if field != "query_lengths")
                packed_inputs += ("speculative_prefix_limits",)
            for batch, _ in reversed(config.speculative_shapes):
                name = f"speculative{batch}x1"
                batch_op = PreparedBatch(inputs, requests, Metadata.allocate(batch, batch, blocks, device),
                    mode="decode", shard=shard, pool=pool, verification_width=1)
                forward = ReferenceSpeculation(model, batch_op, history, completion, frame, sequence,
                    rng=rng, proposal_generation=proposal_generation, proposal_offset=proposal_offset,
                    session=session, workspace=arena)
                add(name, forward, "speculative", batch, 1, forward_inputs, enabled=True)
            for batch, logical_width, packed_tokens in sorted(config.packed_speculative_depth_shapes,
                    key=lambda shape: (shape[2], shape[0], shape[1]), reverse=True):
                depth = logical_width - 1
                name = f"packed-speculative{batch}x{logical_width}-drafts{depth}-tokens{packed_tokens}"
                batch_op = PreparedBatch(inputs, requests, Metadata.allocate(batch, batch, blocks, device),
                    mode="decode", shard=shard, pool=pool, verification_width=logical_width)
                forward = ReferenceSpeculation(model, batch_op, history, completion, frame, sequence,
                    rng=rng, proposal_generation=proposal_generation, proposal_offset=proposal_offset,
                    session=session, prefix_limits=prefix_limits, confidence_output=confidence.device_view,
                    target_token_capacity=packed_tokens, transaction=shared_transaction, workspace=arena)
                add(name, forward, "speculative", batch, logical_width, packed_inputs,
                    enabled=True, packed_tokens=packed_tokens, draft_tokens=depth)

        if vision_state is not None:
            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                encoder = ReferenceVision(model.vision_language_tuner, vision_inputs, capacity=capacity)
                add(name, PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence),
                    "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))

        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        draft_excluded = (tuple(f"layer{model.cfg.num_hidden_layers + i}.window"
                                for i in range(len(model.draft.layers)))
                          if model.draft is not None else ())
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = ReferenceResume(model, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, session=session, speculation_enabled=enabled)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=None, coordinator=None, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled,
                        excluded_regions=() if enabled else draft_excluded)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)

    transient.close()
    release_unused()
    if progress is not None:
        progress(f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
                 f"{torch.cuda.memory_reserved(device) >> 20} MiB reserved, "
                 f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free")
        progress(f"Graph operand allocation by operation: {catalog_bytes}")

    def materialize():
        if feature_pool is not None:
            feature_pool.materialize()
        pool.clear_for_preparation()

    if feature_pool is not None:
        feature_pool.release_backing()
    try:
        with phase(progress, "Prepare and tune kernels") as stage, core_storage.transient():
            tuning = session.tune(show_progress=progress is not None,
                                  console=stage.console, contextual=None)
        stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
        arena.release_preparation()
        release_unused()
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=None, coordinator=None, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=core_storage,
            compile_kinds=("prefill", "decode", "resume", "speculative", "vision"))
        persistent_worker = None
        if experts.mover is not None:
            persistent_worker = experts.mover.capture_worker_graph(
                name="sm103_expert_mover", retain_graphs=config.retain_graphs, stream=stream
            )
            buffers[experts.mover.control_buffer_name] = experts.mover.control_region.device_view
    finally:
        if feature_pool is not None:
            feature_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    for stager in expert_stagers.values():
        stager.decision.zero_()
    if experts.mover is not None:
        experts.mover.reset_runtime_state()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    free, total = torch.cuda.mem_get_info(device)
    assets = {role: {"path": str(path)}
        for role, path in (("tokenizer", metadata_path / "tokenizer.json"),)}
    buffers["request_pending_token"] = requests.pending_token
    server_config = {"identity": identity, "model_id": snapshot_name,
        "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(metadata_path, eos_token_ids=[1], prompt_encoder=PromptEncoder.DEEPSEEK_V41,
            image_profile="deepseek_row_major" if vision_state is not None else None),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
            page_tokens=PAGE_TOKENS, draft_lag=0 if config.dspark else None, rows=1,
            embedding_tables=range(len(delivery.sources)))}
    staging_estimate = 0
    if config.sm103 is not None and config.sm103.prefill_staging is not None:
        from .config import ModelConfig
        staging_estimate = config.sm103.staging_bytes(ModelConfig.load(metadata_path))
    workspace_bytes = sum(workspace.buffer.nbytes for workspace in {
        id(w.workspace): w.workspace for w in experts.weights.values()}.values())
    expert_host_bytes = sum(region.extent for w in experts.weights.values()
                            if w.host_tier is not None for region in w.host_tier.regions)
    offloaded_bytes = sum(weight_src.nbytes + scale_src.nbytes
                          for weight_src, scale_src in delivery.sources.values())
    memory = {"rank": 0, "world_size": 1, "device": config.devices[0],
        "catalog_operand_bytes": catalog_bytes,
        "weights_bytes": weights_bytes, "mapped_host_bytes": expert_host_bytes,
        "expert_tier_bytes": tier_bytes,
        "expert_staging_scratch_estimate": staging_estimate,
        "megamoe_workspace_bytes": workspace_bytes,
        "expert_scale_compression": None,
        "sm103": config.sm103.to_dict(),
        "mla_compute_mode": config.mla_compute_mode,
        "draft_head_quantization": config.draft_head_quantization,
        "markov_head_quantization": config.markov_head_quantization,
        "vocabulary_parallel_sampling": False,
        "target_logits_dtype": "float32",
        "draft_logits_dtype": ("bfloat16" if config.draft_head_quantization is not None else "float32") if config.dspark else None,
        "offloaded_bytes": offloaded_bytes,
        "continuation_bytes": pool.backing.nbytes, "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * geometry.continuation.row_bytes,
        "pages": pages, "page_bytes": bytes_per_page, "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page, "completion_output_bytes": completion.nbytes,
        "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder,
        "draft_prefix_selection": config.draft_prefix_selection}
    if config.dspark:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(
                config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes,
            packed_speculative_shapes=config.packed_speculative_shapes,
            packed_speculative_depth_shapes=config.packed_speculative_depth_shapes,
            speculative_graph_budget=config.speculative_graph_budget,
            packed_speculative_graphs=len(config.packed_speculative_depth_shapes))
    disposable = [delivery.buffers()]
    if vision_state is not None:
        disposable.append(vision_state.arena)
    prepared = PreparedModel(rebuild_weights=declare_reference_weights(model), expert_rows=expert_rows,
        persistent_worker=persistent_worker,
        device=config.devices[0], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*storage_ranges(disposable), *completion.zero_ranges()),
        export_events=delivery.export_events(),
        embedding_catalog=delivery.resident_catalog(),
        core_data=delivery,
        resources=(model, experts, experts.mover, delivery, history, pool, requests, inputs, admission, commands,
            forwards, compiled, stream, graph_pool, session, rng, proposal_generation, proposal_offset,
            confidence, vision_state, vision_inputs, vision_prefill, feature_pool),
        close_resources=())
    if confidence is not None:
        prepared.proposal_confidence = confidence.host_view
    return prepared


@torch.no_grad()
@with_autotune_effort
def prepare_rank(snapshot, config, *, rank, group, progress=None, core_storage=None):
    if config.sm103 is not None:
        return _prepare_reference_rank(snapshot, config, rank=rank, group=group,
                                       progress=progress, core_storage=core_storage)
    if core_storage is None:
        raise ValueError("DeepSeek preparation requires core storage")
    snapshot = Path(snapshot).resolve()
    shard = TensorParallelShard(rank, len(config.devices))
    device = torch.device("cuda", config.devices[rank])
    torch.cuda.set_device(device)
    with phase(progress, "Inspect checkpoint"):
        manifest = inspect_indexed_checkpoint(snapshot)
    metadata_path = manifest.metadata_path
    with core_storage.transient():
        weights = ModelWeights.load(manifest, device, shard=shard, progress=progress, dspark=config.dspark,
                                    confidence=config.draft_tokens > 0,
                                    expert_scale_compression=config.expert_scale_compression,
                                    draft_head_quantization=config.draft_head_quantization,
                                    markov_head_quantization=config.markov_head_quantization,
                                    vocabulary_parallel_sampling=config.vocabulary_parallel_sampling)
        release_unused()
    if progress is not None:
        progress("Target head: BF16 checkpoint weights with FP32 accumulation and logits")
    transient = contextlib.ExitStack()
    transient.enter_context(core_storage.transient())
    if progress is not None:
        progress(f"Resident model tensors: {weights.allocated_bytes / (1 << 30):.3f} GiB per rank")
    sequence = torch.zeros(1, dtype=torch.int64, device=device)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    tables = Tables.prepare(manifest, weights.config, tokens=max(config.prefill_chunk_size, config.decode_tokens),
        device=device, sequence=sequence, frame=frame, host_allocator=core_storage.host_allocate, progress=progress, shard=shard)
    with phase(progress, "Allocate model state and collectives"):
        layout = continuation_layout(weights.config, dspark=config.dspark)
        snapshots = layout.snapshot_capacity(config.snapshot_budget_bytes)
        if snapshots < 1:
            raise ValueError("snapshot budget must hold a complete DeepSeek continuation")
        slots, blocks = config.request_capacity, config.context // PAGE_TOKENS
        execution_rows = slots * (max(config.speculative_widths) if config.dspark else 1)
        state_rows = snapshots + execution_rows
        pages = (config.prefill_chunk_size + PAGE_TOKENS - 1) // PAGE_TOKENS + 2 * slots
        if config.kv_pages is not None:
            pages = config.kv_pages
        bytes_per_page = page_bytes(weights.config)
        total_memory = virtual_device_memory(device, group, tuning_replica=config.tuning_replica)
        pool = ContinuationPool.allocate(weights.config, state_rows + 1, pages, device,
            storage=core_storage, maximum_pages=total_memory // bytes_per_page,
            maximum_rows=total_memory // layout.row_bytes,
            backed_slots=min(state_rows + 1, 1 + execution_rows + slots), dspark=config.dspark)
        requests = DeviceRequestState.allocate(device, vocab_size=129280, capacity=slots)
        inputs = BatchInputs.allocate(config.prefill_chunk_size, blocks, device, capacity=slots)
        admission = AdmissionBuffers.allocate(device, vocab_size=129280, capacity=slots)
        commands = StateCommands.allocate(device, capacity=slots)
        completion = CompletionOutputs.allocate((2, slots, VECTOR_TOKENS_FIELD + config.completion_token_capacity), device=device)
        coordinator = CollectivePreparation(group)
        verification_samples = (slots * max(config.speculative_widths)
                                if config.packed_verification else config.decode_tokens)
        collective, prefill_collective, vocabulary_collective, result_collective, draft_vocabulary_collective = _collective_runtimes(
            config, group, device, coordinator=coordinator, core_storage=core_storage,
            verification_samples=verification_samples)
        sampling_collective = logprob_collective = None
        if config.vocabulary_parallel_sampling and shard.world_size > 1:
            sampling_collective = (collective if coordinator.spans_hosts else create_all_gather_runtime(
                group, device, max_rows=verification_samples, columns=64 * shard.world_size,
                ipc=core_storage.peer_storage))
            logprob_collective = (collective if coordinator.spans_hosts else create_all_gather_runtime(
                group, device, max_rows=verification_samples, columns=512 * shard.world_size,
                ipc=core_storage.peer_storage))
        tables = replace(tables, readiness_runtime=collective)
        identity = hashlib.sha256(json.dumps({"config": json.loads((metadata_path / "config.json").read_text()),
            "headers": [s.header_sha256 for s in manifest.shards], "world_size": shard.world_size,
            "contract": "deepseek41.graph-inputs.v1"}, sort_keys=True).encode()).hexdigest()
        session = RecordedPreparationSession(
            compile_workers=config.compile_workers, collective_tuning=config.collective_tuning, collective_ladder=config.collective_ladder, autotune_effort=config.autotune_effort, tuning_replica=config.tuning_replica,
            native_dense_activations=config.native_dense_activations, lock_gpu_clocks=config.lock_gpu_clocks,
            device=device, checkpoint=identity,
            workload_schema="deepseek41-text", numerical_policy="default", coordinator=coordinator,
            headroom_bytes=config.headroom_bytes, benchmark_storage=core_storage)
        session.__enter__()
        session.register_benchmark_resource("continuations", pool, restore=pool.clear_for_preparation)
        arena = ScratchArena(device, overlap=config.stream_overlap)
        session.register_scratch(arena)
        examples = {}
        candidate_tokens = config.prefill_chunk_size
        if (weights.config.decoder_start is not None
                and weights.config.text["candidate_source_layer_id"] >= weights.config.decoder_start):
            candidate_tokens = min(candidate_tokens, 128)
        prefill_storage = Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device,
            candidate_tokens=candidate_tokens)
        scored_storage = (Scratch.allocate(config.prefill_chunk_size, 1, shard.world_size, device,
            candidate_tokens=config.prefill_chunk_size)
            if config.prompt_logprobs else None)
        decoder_storage = Scratch.allocate(config.decode_tokens, verification_samples, shard.world_size, device)
        vision_state = vision_inputs = vision_prefill = vision_storage = feature_pool = None
        if weights.vision is not None:
            vision_inputs = VisionInputs.allocate(device, patches=max(PATCH_CAPACITIES))
            vision_prefill = PrefillVisionInputs.allocate(config.prefill_chunk_size, device)
            feature_pool = core_storage.scratch_pool()
            with feature_pool:
                feature_arena = torch.zeros((config.image_concurrency, 4096,
                    weights.vision["image_start"].numel()), dtype=torch.bfloat16, device=device)
            vision_state = VisionState(weights.vision, inputs, vision_prefill, requests, features=4096,
                token_base=129280, sentinel_names=("image_start", "image_start", "image_newline", "image_end"),
                feature_type=1, image_slots=config.image_concurrency, arena=feature_arena)
            vision_storage = {}
        decoder_factory, draft_factory = _forward_factories(config.mla_compute_mode,
            vision=vision_state,
            draft_vocabulary_runtime=draft_vocabulary_collective, examples=examples)
        buffers = dict(inputs.buffers(), **tables.buffers(), **commands.buffers(), **{
            name: getattr(admission, name) for name in admission.__dataclass_fields__},
            state_pool=pool.backing, memory_capacity=pool.memory_capacity,
            sampling_grammar_mask=requests.grammar_mask, frame_index=frame,
            submission_sequence=sequence, **completion.buffers())
        if vision_state is not None:
            buffers.update(vision_inputs.buffers(), **vision_prefill.buffers())
        forwards, signatures, metadata, scratch = {}, {}, {}, {}
        speculation = None
        if config.dspark:
            from ignition.components.dspark import DSparkForwards, SpeculativePreparation

            speculation = SpeculativePreparation(weights, tables, pool, requests, inputs, decoder_storage,
                draft_tokens=config.draft_tokens, completion=completion, frame=frame, sequence=sequence,
                session=session, workspace=arena,
                runtime=collective, vocabulary_runtime=vocabulary_collective,
                markov_runtime=None, coordinator=coordinator, result_runtime=result_collective,
                block_proposal_broadcast=True,
                vocabulary_parallel_sampling=config.vocabulary_parallel_sampling,
                sampling_runtime=sampling_collective, logprob_runtime=logprob_collective,
                forwards=DSparkForwards(decoder_factory, DraftScratch, _CONTEXT,
                    draft_factory, PreparedMarkov))

    scoring = PromptScoring.allocate(config, device, buffers, frame=frame, group=group,
        ipc=core_storage.peer_storage, runtime=collective, coordinator=coordinator)
    del manifest
    modes = serving_modes(config.dspark, config.mtp_only)
    count = (len(modes) * (prefill_graph_count(config.prefill_capacities, scoring)
                           + 4 * len(config.batch_capacities))
             + len(config.speculative_shapes) + len(config.packed_speculative_depth_shapes)
             + (len(config.batch_capacities) if False in modes else 0))
    count += len(PATCH_CAPACITIES) if vision_state is not None else 0
    catalog_bytes = {}
    catalog_previous = torch.cuda.memory_allocated(device)
    with phase(progress, "Construct graph catalog", total=count) as stage:
        def add(name, forward, kind, rows, tokens, fields, *, enabled=None, packed_tokens=0, draft_tokens=None):
            nonlocal catalog_previous
            if kind == "scored_prefill":
                forward = PreparedScoredPrefill(forward, requests, inputs)
                fields = (*fields, "prompt_target_tokens")
            forwards[name] = forward
            signatures[name] = GraphSignature(kind, rows, tokens, tuple(fields), speculation_enabled=enabled,
                                              packed_token_capacity=packed_tokens, draft_tokens=draft_tokens,
                confidence_width=draft_tokens if "speculative_prefix_limits" in fields else 0)
            allocated = torch.cuda.memory_allocated(device)
            catalog_bytes[kind] = catalog_bytes.get(kind, 0) + allocated - catalog_previous
            catalog_previous = allocated
            stage.update(advance=1)

        forward_inputs = (*inputs.buffers(), "sampling_grammar_mask")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for kind, mode, capacities in prefill_modes(
                    config.prefill_capacities, () if enabled else config.batch_capacities, scoring):
                for capacity in capacities:
                    rows = 1 if mode == "prefill" else capacity
                    name = f"{kind}{capacity}" + (suffix if mode == "prefill" else "")
                    metadata[name] = Metadata.allocate(capacity, rows, blocks, device)
                    storage = prefill_storage if mode == "prefill" else decoder_storage
                    if kind == "scored_prefill":
                        storage = scored_storage
                    scratch[name] = storage.rows(capacity, rows)
                    decoder = decoder_factory(mode_weights, tables, scratch[name], metadata[name], pool,
                        mode=mode,
                        scoring=scoring if kind == "scored_prefill" else None,
                        runtime=prefill_collective if mode == "prefill" and capacity >= 128 else collective,
                        vocabulary_runtime=vocabulary_collective,
                        coordinator=coordinator, session=session, name=name, workspace=arena)
                    batch = PreparedBatch(inputs, requests, metadata[name], mode=mode, shard=shard, pool=pool,
                        input_vocabulary=INPUT_VOCABULARY if vision_state is not None else None,
                        validate_prefill=vision_state.validate_prefill if vision_state is not None and mode == "prefill" else None,
                        )
                    forward = PreparedGeneration(decoder, batch, completion, frame, sequence, runtime=collective,
                        coordinator=coordinator, session=session, name=name + ".generation", workspace=arena,
                        context_factory=PreparedContext if kind == "scored_prefill" else _CONTEXT)
                    fields = (*forward_inputs, "vision_token_indices", "vision_feature_slot") if mode == "prefill" and vision_state is not None else forward_inputs
                    add(name, forward, kind, rows, capacity if mode == "prefill" else 1, fields, enabled=enabled)
        confidence = _prepare_speculation_catalog(config, speculation, add=add, buffers=buffers,
            forward_inputs=forward_inputs,
            host_allocate=core_storage.host_allocate, device=device)
        if vision_state is not None:
            for capacity in reversed(PATCH_CAPACITIES):
                name = f"vision{capacity}"
                vision_storage[capacity] = VisionScratch.allocate(capacity, device, hidden=5120)
                encoder = PreparedVision(weights.vision, vision_storage[capacity], vision_inputs,
                    session=session, workspace=arena, name=name)
                add(name, PreparedVisionGeneration(encoder, vision_state, completion, frame, sequence),
                    "vision", 1, capacity, (*vision_inputs.buffers(), "vision_feature_slot"))
        row_inputs = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows")
        for enabled in modes:
            suffix = "-no-mtp" if config.dspark and not enabled else ""
            mode_weights = weights if enabled else replace(weights, draft=None)
            for capacity in config.batch_capacities:
                name = f"resume{capacity}" + suffix
                forward = PreparedResume(mode_weights, requests, pool, inputs, completion, frame, sequence,
                    rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective,
                    coordinator=coordinator, session=session, name=name, logits_dtype=torch.float32,
                    workspace=arena)
                add(name, forward, "resume", capacity, 1, (*row_inputs, "sampling_grammar_mask"), enabled=enabled)
                for mode in ("init", "restore", "snapshot"):
                    name = f"{mode}{capacity}" + suffix
                    forward = PreparedMaintenance(mode, requests, pool, commands, inputs, completion, frame, sequence,
                        runtime=collective, coordinator=coordinator, session=session, name=name,
                        capacity=capacity, admission_buffers=admission, draft_state=enabled)
                    fields = row_inputs if mode == "snapshot" else ("request_state_rows", *admission.__dataclass_fields__)
                    add(name, forward, mode, capacity, 1, (*fields, "page_tables", *commands.buffers()), enabled=enabled)
    transient.close()
    release_unused()
    if progress is not None:
        progress(f"before tuning: {torch.cuda.memory_allocated(device) >> 20} MiB allocated, "
                 f"{torch.cuda.memory_reserved(device) >> 20} MiB reserved, "
                 f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB free")
        progress(f"Graph operand allocation by operation: {catalog_bytes}")
    def materialize():
        if feature_pool is not None:
            feature_pool.materialize()
        pool.clear_for_preparation()

    if feature_pool is not None:
        feature_pool.release_backing()
    try:
        contextual = None
        if _can_race_contextual_plans(config, signatures, device):
            from .contextual_selection import CatalogContextualSelection

            contextual = CatalogContextualSelection(forwards, signatures,
                coordinator=coordinator, collective=collective, device=device,
                graph_pool=core_storage.graph_pool.id, cache_root=config.kernel_cache,
                compile_decoder=config.compile_decoder, warmup_storage=core_storage,
                progress=progress)
        with phase(progress, "Prepare and tune kernels") as stage, core_storage.transient():
            tuning = session.tune(show_progress=progress is not None and rank == 0,
                                  console=stage.console, contextual=contextual)
        if contextual is not None and progress is not None:
            from .contextual_selection import contextual_summary

            summary = contextual_summary(tuning["contextual_tuning"])
            if hasattr(progress, "console"):
                progress.console.print(summary, markup=False)
            else:
                progress(summary)
        stream, graph_pool = torch.cuda.Stream(device=device), core_storage.graph_pool.id
        if speculation is not None:
            speculation.release_examples()
        metadata.clear()
        scratch.clear()
        examples.clear()
        vision_storage = None
        prefill_storage = decoder_storage = scored_storage = storage = decoder = batch = None
        arena.release_preparation()
        release_unused()
        graphs, compiled = capture_forwards(forwards, signatures, session=session, device=device, stream=stream,
            pool=graph_pool, collective=collective, coordinator=coordinator, compile_decoder=config.compile_decoder,
            progress=progress, cache_root=config.kernel_cache, retain_graphs=config.retain_graphs,
            materialize=materialize, warmup_storage=core_storage,
            compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "vision"))
    finally:
        if feature_pool is not None:
            feature_pool.materialize()
    session.freeze()
    pool.clear_for_preparation()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    release_unused()
    if group is not None:
        torch.distributed.barrier(group=group)
    free, total = torch.cuda.mem_get_info(device)
    assets = {role: {"path": str(path)}
        for role, path in (("tokenizer", metadata_path / "tokenizer.json"),)}
    buffers["request_pending_token"] = requests.pending_token
    server_config = {"identity": identity, "model_id": snapshot.name,
        "max_sequence_tokens": config.context, "prefill_chunk_size": config.prefill_chunk_size,
        "pages": pages, "state_rows": state_rows, "snapshots": snapshots, "page_tokens": PAGE_TOKENS,
        "frontend": prepare_frontend(metadata_path, eos_token_ids=[1], prompt_encoder=PromptEncoder.DEEPSEEK_V41,
            image_profile="deepseek_row_major" if vision_state is not None else None),
        "sampling_defaults": RECIPE.sampling_defaults.as_dict(), "assets": assets,
        "model_contract": prepare_contract(signatures, buffers, pool, image_slots=config.image_concurrency,
            page_tokens=PAGE_TOKENS, draft_lag=0 if config.dspark else None, rows=None,
            embedding_tables=range(len(tables.sources)))}
    memory = {"rank": rank, "world_size": shard.world_size, "device": config.devices[rank],
        "catalog_operand_bytes": catalog_bytes,
        "weights_bytes": weights.allocated_bytes, "mapped_host_bytes": 0,
        "expert_scale_compression": config.expert_scale_compression,
        "mla_compute_mode": config.mla_compute_mode,
        "draft_head_quantization": config.draft_head_quantization,
        "markov_head_quantization": config.markov_head_quantization,
        "vocabulary_parallel_sampling": config.vocabulary_parallel_sampling,
        "target_logits_dtype": "float32",
        "draft_logits_dtype": "bfloat16" if config.dspark else None,
        "offloaded_bytes": sum(tables.partition(ordinal, shard.rank).rows * 264
                               for ordinal in range(len(tables.sources))),
        "continuation_bytes": pool.backing.nbytes, "state_rows": state_rows,
        "preparation_state_rows": int(pool.memory_capacity[0].item()),
        "preparation_state_bytes": int(pool.memory_capacity[0].item()) * layout.row_bytes,
        "pages": pages, "page_bytes": bytes_per_page, "page_tokens": PAGE_TOKENS,
        "kv_bytes": pages * bytes_per_page, "completion_output_bytes": completion.nbytes, "operator_scratch_bytes": arena.nbytes,
        "graphs": len(graphs), "prefill_capacities": config.prefill_capacities,
        "indexer_query_rows": INDEXER_QUERY_ROWS,
        "free_device_bytes": free, "total_device_bytes": total, "headroom_bytes": config.headroom_bytes,
        "allocated_bytes": torch.cuda.memory_allocated(device), "reserved_bytes": torch.cuda.memory_reserved(device),
        "tuning": tuning, "compiled_decoder": config.compile_decoder}
    if scoring is not None:
        memory.update(scoring.memory_report())
    memory["draft_prefix_selection"] = config.draft_prefix_selection
    if config.dspark:
        server_config.update(speculative_widths=config.speculative_widths,
            speculative_token_capacity=speculative_token_capacity(config.max_active, max(config.speculative_widths), config.speculative_token_capacity),
            speculative_fixed_width=config.speculative_fixed_width)
        memory.update(speculative_shapes=config.speculative_shapes,
            packed_speculative_shapes=config.packed_speculative_shapes,
            packed_speculative_depth_shapes=config.packed_speculative_depth_shapes,
            speculative_graph_budget=config.speculative_graph_budget,
            packed_speculative_graphs=len(config.packed_speculative_depth_shapes))
    disposable = [tables.buffers()]
    if vision_state is not None:
        disposable.append(vision_state.arena)
    for forward in forwards.values():
        if isinstance(forward, PreparedScoredPrefill):
            forward = forward.forward
        if hasattr(forward, "completion") and hasattr(forward.completion, "scratch_tensors"):
            disposable.append(forward.completion.scratch_tensors)
    model = PreparedModel(rebuild_weights=declare_weights(weights, components=('embedding', 'norm', 'head', 'vision', 'draft')), device=config.devices[rank], graphs=graphs, buffers=buffers, signatures=signatures,
        max_active=config.max_active, session=session, server_config=server_config, memory_report=memory,
        export_zero_ranges=(*(() if scoring is None else scoring.zero_ranges()),
                           *storage_ranges(disposable), *completion.zero_ranges()),
        export_events=tables.events(),
        embedding_catalog=tables.resident_catalog(),
        core_data=tables if rank == 0 else None,
        resources=(scoring, weights, tables, pool, requests, inputs, admission, commands, forwards,
            compiled, stream, graph_pool, session,
            collective, prefill_collective, vocabulary_collective, draft_vocabulary_collective, result_collective, sampling_collective, logprob_collective, coordinator, speculation, confidence,
            vision_state, vision_inputs, vision_prefill, feature_pool),
        close_resources=tuple(dict.fromkeys(c for c in
            (scoring, sampling_collective, logprob_collective, result_collective, prefill_collective, vocabulary_collective, draft_vocabulary_collective, collective) if c is not None)))
    if confidence is not None:
        model.proposal_confidence = confidence.host_view
    return model

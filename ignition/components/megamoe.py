# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared TP1 MXFP8/MXFP4 MegaMoE with optional fused FP8 shared experts."""

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import triton
from b12x._lib.compile_plan import _NATIVE_JITS, load_programs
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.components.megamoe_kernel import initialize
from ignition.components.megamoe_kernel.inputs import _prepare_megamoe_inputs_kernel
from ignition.components.native_operator import native_contract, native_plan
from ignition.preparation.persistent import persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_finalizer


@dataclass(frozen=True)
class MegaMoeGeometry:
    experts: int
    capacity: int
    top_k: int
    hidden: int
    intermediate: int
    shared_experts: int

    def __post_init__(self):
        if (any(type(value) is not int for value in vars(self).values())
                or min(self.experts, self.capacity, self.top_k, self.hidden, self.intermediate) <= 0
                or self.top_k > self.experts or self.hidden % 128 or self.intermediate % 128
                or self.shared_experts not in (0, 1)):
            raise ValueError("MegaMoE requires positive capacities, 128-aligned widths and zero or one shared expert")


class MegaMoeWorkspace:
    """Model-owned protocol storage shared only by serialized, identical geometries.

    Kernel semaphores and ring storage share one native allocation. Initialize it
    once, retain it across graph variants and layers, and expose every launch as
    a mutation. Do not give this buffer to another operator's scratch allocator.
    """

    def __init__(self, geometry, *, device, kernel_cache: Path):
        device = torch.device(device)
        with torch.cuda.device(device):
            if torch.cuda.get_device_capability() != (10, 3):
                raise ValueError("the reference MegaMoE path requires SM103")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("MegaMoE workspace must be prepared before capture")
            self.kernel_cache = kernel_cache
            self.library = initialize(kernel_cache)
            alignment = self.library._C.get_token_alignment_for_mega_moe()
            self.geometry = MegaMoeGeometry(geometry.experts,
                triton.cdiv(geometry.capacity, alignment) * alignment, geometry.top_k,
                geometry.hidden, geometry.intermediate, geometry.shared_experts)
            g = self.geometry
            nbytes, self.slice_inputs = self.library._C.get_symm_buffer_size_for_mega_moe(
                1, g.experts, g.capacity, g.top_k, g.hidden, g.intermediate,
                "fp8xfp4", "swiglu", g.shared_experts)
            self.buffer = persistent_zeros((nbytes,), dtype=torch.int8, device=device)
            self.views = self.slice_inputs(self.buffer)


class _LocalRank:
    """Rank metadata for DeepGEMM's single-device launch wrapper."""

    @staticmethod
    def rank():
        return 0


@dataclass(frozen=True)
class MegaMoeQuery:
    rows: int
    experts: int
    capacity: int
    top_k: int
    hidden: int
    intermediate: int
    shared_experts: int
    x_sf_stride: tuple
    shared_sf_stride: tuple
    shared_block_m: int
    clamp: float
    host_rows: int = 0
    host_lane_clusters: int = 0
    host_shared_first: bool = False
    indirection: bool = False
    hbm_slots: int = 0
    compute_sm_count: int = 0

    def __post_init__(self):
        object.__setattr__(self, "x_sf_stride", tuple(self.x_sf_stride))
        object.__setattr__(self, "shared_sf_stride", tuple(self.shared_sf_stride))


def _validate_inputs(source, ids, probabilities, padding, geometry, device, rows):
    if (source.dtype != torch.bfloat16 or source.shape != (rows, geometry.hidden)
            or not 0 < rows <= geometry.capacity or not source.is_contiguous()
            or ids.shape != (rows, geometry.top_k) or ids.dtype != torch.int32 or not ids.is_contiguous()
            or probabilities.shape != ids.shape or probabilities.dtype != torch.float32
            or not probabilities.is_contiguous() or padding.shape != (rows,)
            or padding.dtype != torch.bool or not padding.is_contiguous()):
        raise ValueError("MegaMoE requires the prepared contiguous BF16 source, int32 routes, FP32 weights and boolean padding")
    if any(t.device != device for t in (source, ids, probabilities, padding)):
        raise ValueError("MegaMoE operands and workspace must share one CUDA device")


def _validate_weights(weights, geometry, *, shared, count=None):
    g = geometry
    prefix = () if shared else (g.experts if count is None else count,)
    values_dtype = torch.float8_e4m3fn if shared else torch.int8
    divisor = 1 if shared else 2
    for (values, scales), rows, columns in (
        (weights.gate_up, 2 * g.intermediate, g.hidden),
        (weights.down, g.hidden, g.intermediate),
    ):
        sf_columns = columns // 128
        scale_stride = (1, rows) if shared else (rows * sf_columns, 1, rows)
        if (values.shape != (*prefix, rows, columns // divisor)
                or values.dtype != values_dtype or not values.is_contiguous()
                or scales.shape != (*prefix, rows, sf_columns) or scales.dtype != torch.int32
                or scales.stride() != scale_stride):
            raise ValueError("MegaMoE weights require matching packed values and MN-major UTCCP scales")


def _stage_constants(q):
    return (q.hidden, 1, q.hidden, 1, *q.x_sf_stride, *q.shared_sf_stride,
            q.top_k, 1, q.top_k, 1, 1, q.top_k, 1, q.top_k, 1,
            q.rows, q.hidden, q.top_k, 8 if q.rows >= 64 else 1,
            128, 32, triton.next_power_of_2(q.top_k), q.shared_block_m, True)


_NATIVE_JITS.add(_prepare_megamoe_inputs_kernel)


@program_cache(scope="preparation")
def compile_inputs(payload, ordinal):
    q = MegaMoeQuery(**dict(payload))
    with torch.cuda.device(ordinal):
        return _prepare_megamoe_inputs_kernel.warmup(
            torch.bfloat16, torch.float8_e4m3fn, torch.int32,
            torch.int32 if q.shared_experts else None, torch.int32, torch.float32,
            torch.bool, torch.int64, torch.float32, *_stage_constants(q),
            grid=(1, 1, 1), num_warps=4)


CONTRACT = native_contract(component_id="flare.megamoe.tp1_mxfp8_mxfp4",
                           query_type=MegaMoeQuery, tactics=lambda query: (None,))


class _State:
    def __init__(self, query, workspace):
        self.query, self.workspace = query, workspace
        self.host_binding = None
        self.mover_binding = None
        if query.host_rows:
            from ignition.components.megamoe_kernel.host import prepare
            self.host_binding = prepare(workspace.kernel_cache)
            if query.compute_sm_count > 0:
                self.host_binding.set_num_sms(query.compute_sm_count)
        if query.indirection:
            from ignition.components.sm103_mover_kernel import prepare as prepare_mover
            self.mover_binding = prepare_mover(workspace.kernel_cache)
        self.stager = load_programs(compile_inputs(CONTRACT.encode_query(query), workspace.buffer.device.index))

    def run(self, source, ids, probabilities, padding, matrices, output, buffer, control=None, descriptors=None,
            live_slot_map=None, layer_epoch=None, predictor_binding=None):
        q = self.query
        x, sf, out_ids, out_weights, _, shared_sf, *_ = self.workspace.slice_inputs(buffer)
        self.stager[(triton.cdiv(q.rows, 8 if q.rows >= 64 else 1), q.hidden // 128, 1)](
            source, x, sf, shared_sf if q.shared_experts else None,
            ids, probabilities, padding, out_ids, out_weights, *_stage_constants(q))
        self.run_encoded(matrices, output, buffer, control, descriptors, live_slot_map, layer_epoch, predictor_binding)

    def run_encoded(self, matrices, output, buffer, control=None, descriptors=None,
                    live_slot_map=None, layer_epoch=None, predictor_binding=None):
        q = self.query
        shared_gate = tuple(matrices[4:6]) if q.shared_experts else None
        shared_down = tuple(matrices[6:8]) if q.shared_experts else None
        if self.host_binding is not None:
            views = self.workspace.slice_inputs(buffer)
            if q.indirection:
                if live_slot_map is None or layer_epoch is None or control is None:
                    raise ValueError("indirection MegaMoE requires live_slot_map, layer_epoch and control buffers")
                if predictor_binding is not None:
                    self.mover_binding.launch_layer_enter_predict(
                        live_slot_map, control, layer_epoch, views[2], q.rows, q.top_k, q.experts,
                        predictor_binding.state, predictor_binding.queue_header, predictor_binding.queue_ring,
                        predictor_binding.metrics_ptr, predictor_binding.metrics_count,
                    )
                else:
                    self.mover_binding.launch_layer_enter(live_slot_map, control, layer_epoch, q.experts)
            self.host_binding.launch(output, list(views[4:]), matrices[:4],
                matrices[4:8] if q.shared_experts else [], control, buffer,
                q.capacity, q.experts, q.host_rows, q.top_k, q.hidden, q.intermediate, q.clamp,
                q.host_shared_first, q.hbm_slots if q.indirection else -1)
            return
        binding = SimpleNamespace(buffer=buffer, handle=SimpleNamespace(buffer_ptrs=[buffer.data_ptr()]),
            group=_LocalRank(), num_max_tokens_per_rank=q.capacity, num_experts=q.experts, num_topk=q.top_k)
        self.workspace.library.fp8_fp4_mega_moe(
            output, tuple(matrices[:2]), tuple(matrices[2:4]), binding,
            shared_l1_weights=shared_gate, shared_l2_weights=shared_down,
            recipe=(1, 1, 32), activation="swiglu", activation_clamp=q.clamp, fast_math=True)


_DYNAMIC_TIERS = {}


@torch.library.custom_op("flare::megamoe", mutates_args=("output", "workspace", "control", "descriptors", "layer_epoch"))
def _run(source: torch.Tensor, ids: torch.Tensor, probabilities: torch.Tensor,
         padding: torch.Tensor, matrices: list[torch.Tensor], output: torch.Tensor,
         workspace: torch.Tensor, control: torch.Tensor | None, descriptors: torch.Tensor | None,
         live_slot_map: torch.Tensor | None, layer_epoch: torch.Tensor | None, handle: int) -> None:
    plan = plan_from_handle(handle)
    host_tier = _DYNAMIC_TIERS.get(handle)
    require_prepared(plan, plan.component_id, source.device).run(
        source, ids, probabilities, padding, matrices, output, workspace, control, descriptors,
        live_slot_map, layer_epoch, getattr(host_tier, "predictor_binding", None))


@_run.register_fake
def _run_fake(source, ids, probabilities, padding, matrices, output, workspace, control, descriptors,
              live_slot_map, layer_epoch, handle):
    pass


@torch.library.custom_op("flare::megamoe_encoded", mutates_args=("output", "workspace", "control", "descriptors", "layer_epoch"))
def _run_encoded(source: torch.Tensor, encoded: list[torch.Tensor], matrices: list[torch.Tensor], output: torch.Tensor,
                 workspace: torch.Tensor, control: torch.Tensor | None,
                 descriptors: torch.Tensor | None, live_slot_map: torch.Tensor | None,
                 layer_epoch: torch.Tensor | None, handle: int) -> None:
    plan = plan_from_handle(handle)
    state = require_prepared(plan, plan.component_id, source.device)
    views = state.workspace.slice_inputs(workspace)
    rows = plan.query.rows
    if len(encoded) != 4 or any(
            actual.data_ptr() != expected.data_ptr() or actual.shape != expected[:rows].shape
            or actual.stride() != expected.stride() or actual.dtype != expected.dtype
            for actual, expected in zip(encoded, views[:4])):
        raise ValueError("encoded MegaMoE inputs must alias the prepared activation, scale and route views")
    host_tier = _DYNAMIC_TIERS.get(handle)
    state.run_encoded(matrices, output, workspace, control, descriptors, live_slot_map, layer_epoch,
                      getattr(host_tier, "predictor_binding", None))


@_run_encoded.register_fake
def _run_encoded_fake(source, encoded, matrices, output, workspace, control, descriptors,
                      live_slot_map, layer_epoch, handle):
    pass


class PreparedMegaMoe:
    """Declare fused expert execution with compiler-visible weights and protocol storage."""

    def __init__(self, source, ids, probabilities, padding, routed, shared, *,
                 workspace: MegaMoeWorkspace, clamp: float, session, name, host_tier=None,
                 host_shared_first: bool = False, compute_sm_count: int = 0):
        g = workspace.geometry
        rows = source.shape[0]
        _validate_inputs(source, ids, probabilities, padding, g, workspace.buffer.device, rows)
        if not math.isfinite(clamp) or clamp <= 0:
            raise ValueError("MegaMoE SwiGLU clamp must be finite and positive")
        if (shared is not None) != bool(g.shared_experts):
            raise ValueError("MegaMoE shared weights must match the workspace geometry")
        host_rows = 0 if host_tier is None else host_tier.rows
        indirection = bool(getattr(host_tier, "indirection", False))
        hbm_slots = host_tier.hbm_slots if indirection else (g.experts - host_rows)
        if host_tier is not None and host_tier.geometry != g:
            raise ValueError("MegaMoE host tier must match the protocol workspace geometry")
        if type(compute_sm_count) is not int or compute_sm_count < 0:
            raise ValueError("MegaMoE compute_sm_count must be a nonnegative integer")
        _validate_weights(routed, g, shared=False, count=hbm_slots)
        if shared is not None:
            _validate_weights(shared, g, shared=True)
        self.matrices = list((*routed.gate_up, *routed.down))
        if shared is not None:
            self.matrices.extend((*shared.gate_up, *shared.down))
        self.host_tier = host_tier
        self.control = None if host_tier is None else host_tier.control
        self.descriptors = None if host_tier is None else host_tier.descriptors
        self.live_slot_map = host_tier.live_slot_map if indirection else None
        self.layer_epoch = host_tier.layer_epoch if indirection else None
        if host_tier is not None:
            _validate_weights(host_tier.weights, g, shared=False, count=host_rows)
            self.matrices.extend(host_tier.device_views)
        if any(t.device != workspace.buffer.device for t in
               (source, ids, probabilities, padding, *self.matrices)):
            raise ValueError("MegaMoE operands and workspace must share one CUDA device")
        for tensor in self.matrices:
            record_numeric_tensor(tensor)
        block_m = workspace.library.get_block_m_for_mega_moe(
            1, g.experts, g.capacity, rows, g.top_k, "fp8xfp4") if shared is not None else 1
        q = MegaMoeQuery(rows, g.experts, g.capacity, g.top_k, g.hidden, g.intermediate,
            g.shared_experts, tuple(workspace.views[1].stride()),
            tuple(workspace.views[5].stride()) if shared is not None else (0, 0), block_m, clamp,
            host_rows, 0 if host_tier is None else host_tier.host_lane_clusters, host_shared_first,
            indirection=indirection, hbm_slots=hbm_slots if indirection else 0,
            compute_sm_count=compute_sm_count)
        self.workspace = workspace
        self.plan = native_plan(contract=CONTRACT, query=q, device=source.device,
            scratch_bytes=lambda tactic: 0,
            compile_jobs=lambda tactic, device: (CompileJob.create(
                "ignition.components.megamoe:compile_inputs", CONTRACT.encode_query(q), device.ordinal),),
            materialize=lambda tactic: _State(q, workspace))
        if indirection:
            _DYNAMIC_TIERS[self.plan.handle] = host_tier

        examples = [source, ids, probabilities, padding]

        def prepare(state):
            source, ids, probabilities, padding = examples
            output = torch.empty_like(source)
            return PreparedCall(run=lambda: state.run(source, ids, probabilities, padding,
                self.matrices, output, workspace.buffer, self.control, self.descriptors,
                self.live_slot_map, self.layer_epoch,
                getattr(self.host_tier, "predictor_binding", None) if indirection else None),
                output=output, owners=(workspace, self.matrices, self.host_tier))

        declare(session, (self.plan.request(name=name, prepare_call=prepare),))
        register_finalizer(session, examples.clear)

    def __call__(self, source, ids, probabilities, padding, *, staged=None):
        _validate_inputs(source, ids, probabilities, padding, self.workspace.geometry,
                         self.workspace.buffer.device, self.plan.query.rows)
        output = torch.empty_like(source)
        matrices, control = self.matrices, self.control
        if staged is not None:
            if self.host_tier is None:
                raise ValueError("expert staging requires a prepared host tier")
            stager, pending = staged
            if stager.tier is not self.host_tier:
                raise ValueError("expert staging must use the consumer's immutable host rows")
            planes, control = stager.finish(pending, source)
            matrices = [*self.matrices[:-4], *planes]
        _run(source, ids, probabilities, padding, matrices, output,
             self.workspace.buffer, control, self.descriptors,
             self.live_slot_map, self.layer_epoch, self.plan.handle)
        if staged is not None:
            stager.observe(control)
        return output

    def encoded(self, source, inputs=None, *, staged=None):
        """Consume activations, scales and routes written into this workspace.

        The producer must write every active input row and shared-expert scale
        row in the layout returned by ``encoded_inputs``. It must mask inactive
        routes before this call. The workspace cannot serve another invocation
        until the fused expert kernel completes.
        """
        q = self.plan.query
        if (source.shape != (q.rows, q.hidden) or source.dtype != torch.bfloat16
                or source.device != self.workspace.buffer.device or not source.is_contiguous()):
            raise ValueError("encoded MegaMoE requires its prepared BF16 producer shape and device")
        output = torch.empty_like(source)
        if inputs is None:
            inputs = self.encoded_inputs()[:2]
        # The native view slicer creates separate Tensor storages over one CUDA
        # allocation. Explicit reads order producers even when the compiler
        # cannot infer that those storages alias the protocol buffer.
        routed = self.encoded_inputs()[2:4]
        matrices, control = self.matrices, self.control
        if staged is not None:
            stager, pending = staged
            if stager.tier is not self.host_tier:
                raise ValueError("expert staging must use the consumer's immutable host rows")
            planes, control = stager.finish(pending, source)
            matrices = [*self.matrices[:-4], *planes]
        _run_encoded(source, [*inputs, *routed], matrices, output, self.workspace.buffer,
                     control, self.descriptors, self.live_slot_map, self.layer_epoch, self.plan.handle)
        if staged is not None:
            stager.observe(control)
        return output

    def encoded_inputs(self):
        """Return producer views and the selected shared-scale row grouping."""
        q = self.plan.query
        x, scales, ids, probabilities, _, shared_scales, *_ = self.workspace.views
        return (x[:q.rows], scales[:q.rows], ids[:q.rows], probabilities[:q.rows],
                shared_scales[:q.rows] if q.shared_experts else None, q.shared_block_m)

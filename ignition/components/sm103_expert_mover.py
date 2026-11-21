# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation owner, inline on-GPU predictor state, and persistent graph builder for the SM103 GPU expert mover."""

from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
import struct

import torch

from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
from ignition.components.megamoe_weights import MegaMoeWeights
from ignition.components.sm103_mover_kernel import prepare as prepare_mover
from ignition.preparation.execution import PersistentWorker
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.runtime_core import capture_graph, record_export_graph
from ignition.preparation.tensor_metadata import record_numeric_tensor


@dataclass(frozen=True)
class PredictorConfig:
    """Parameters for the inline on-GPU decayed-frequency + LFU/LRU routing predictor."""

    decay_shift: int = 4
    promote_threshold: int = 384
    hysteresis: int = 64
    cooldown_steps: int = 2
    max_enqueues: int = 2
    max_predict_rows: int = 64

    def __post_init__(self):
        if any(type(v) is not int for v in vars(self).values()):
            raise ValueError("PredictorConfig fields must be integers")
        if not 0 <= self.decay_shift < 31:
            raise ValueError("decay_shift must be in [0, 30]")
        if self.promote_threshold <= 0:
            raise ValueError("promote_threshold must be positive")
        if self.hysteresis < 0:
            raise ValueError("hysteresis must be nonnegative")
        if self.cooldown_steps < 1:
            raise ValueError("cooldown_steps must be at least 1")
        if not 0 <= self.max_enqueues <= 4:
            raise ValueError("max_enqueues must be in [0, 4]")
        if self.max_predict_rows < 1:
            raise ValueError("max_predict_rows must be positive")


@dataclass(frozen=True)
class PredictorLayerBinding:
    """Per-layer persistent HBM predictor state bound to a MegaMoeDynamicHostTier."""

    layer_idx: int
    state: torch.Tensor
    scores: torch.Tensor
    last_seen: torch.Tensor
    cooldown: torch.Tensor
    queue_header: torch.Tensor
    queue_ring: torch.Tensor
    metrics_ptr: int
    metrics_count: int


@dataclass(frozen=True)
class MoverMetrics:
    swaps_completed: int
    bytes_copied: int
    rcu_waits: int
    skipped_already_resident: int
    last_layer: int
    last_expert: int
    predict_hits: int = 0
    predict_misses: int = 0
    predict_enqueues: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.predict_hits + self.predict_misses
        return 0.0 if total == 0 else self.predict_hits / total


def query_green_context_compute_sms(device_ordinal: int, worker_sm_count: int) -> tuple[int, int]:
    """Query CUDA driver SM green-context split so preparation compiles compute kernels for exact compute SMs."""
    if type(worker_sm_count) is not int or worker_sm_count <= 0:
        raise ValueError("worker_sm_count must be a positive integer")
    torch.cuda.init()
    lib = C.CDLL("libcuda.so.1")

    class _DevResource(C.Structure):
        _fields_ = [
            ("type", C.c_int),
            ("_internal_padding", C.c_ubyte * 92),
            ("sm_count", C.c_uint32),
            ("min_sm_partition_size", C.c_uint32),
            ("sm_coscheduled_alignment", C.c_uint32),
            ("flags", C.c_uint32),
            ("_tail", C.c_ubyte * 32),
        ]

    def call(name, *args):
        fn = getattr(lib, name)
        fn.restype = C.c_int
        rc = fn(*args)
        if rc != 0:
            msg = C.c_char_p()
            lib.cuGetErrorString(C.c_int(rc), C.byref(msg))
            raise RuntimeError(f"{name} failed with CUDA {rc}: {msg.value!r}")

    sm_res = _DevResource()
    w_res = _DevResource()
    c_res = _DevResource()
    nb = C.c_uint(1)
    call("cuDeviceGetDevResource", C.c_int(device_ordinal), C.byref(sm_res), C.c_int(1))
    call(
        "cuDevSmResourceSplitByCount",
        C.byref(w_res), C.byref(nb), C.byref(sm_res), C.byref(c_res),
        C.c_uint(0), C.c_uint(worker_sm_count),
    )
    w_sms, c_sms = int(w_res.sm_count), int(c_res.sm_count)
    if w_sms == 0 or c_sms == 0:
        raise ValueError(f"invalid green context SM split for worker_sm_count={worker_sm_count}: ({w_sms}, {c_sms})")
    return w_sms, c_sms


class SM103ExpertMoverRuntime:
    """Own GPU descriptor tables, lock-free prefetch ring, predictor state, and mapped-host controls.

    Prepares the persistent 1D-TMA mover kernel for capture as a dedicated
    ``Execution.persistent_worker`` graph running on a spatially partitioned
    CUDA green context alongside compute graphs.
    """

    METRICS_COUNT = 8
    CONTROL_BYTES = 128

    def __init__(
        self,
        layers: list[tuple[MegaMoeDynamicHostTier, MegaMoeWeights]] | tuple[tuple[MegaMoeDynamicHostTier, MegaMoeWeights], ...],
        *,
        storage,
        kernel_cache: Path,
        worker_sm_count: int = 8,
        queue_capacity: int = 1024,
        control_buffer_name: str = "sm103_mover_control",
    ):
        layers = tuple(layers)
        if not layers:
            raise ValueError("SM103ExpertMoverRuntime requires at least one dynamic layer")
        if type(worker_sm_count) is not int or worker_sm_count <= 0:
            raise ValueError("worker_sm_count must be a positive integer")
        if type(queue_capacity) is not int or queue_capacity < 16 or (queue_capacity & (queue_capacity - 1)) != 0:
            raise ValueError("queue_capacity must be a power of two >= 16")
        if not isinstance(control_buffer_name, str) or not control_buffer_name.strip():
            raise ValueError("control_buffer_name must be a nonempty string")

        self.layers = layers
        self.storage = storage
        self.device = torch.device("cuda", storage.device)
        self.worker_sm_count = worker_sm_count
        self.queue_capacity = queue_capacity
        self.control_buffer_name = control_buffer_name
        self.binding = prepare_mover(kernel_cache)
        with torch.cuda.device(self.device):
            self.binding.configure_persistent_mover()

        desc_bytes = self.binding.layer_desc_bytes()
        hdr_bytes = self.binding.queue_header_bytes()
        req_bytes = self.binding.queue_request_bytes()
        pred_bytes = self.binding.predictor_state_bytes()
        if desc_bytes != 192 or hdr_bytes != 128 or req_bytes != 16 or pred_bytes != 64:
            raise RuntimeError(
                f"unexpected mover struct sizes: desc={desc_bytes}, hdr={hdr_bytes}, req={req_bytes}, pred={pred_bytes}"
            )

        # MappedHost control block: [0..4) stop_flag (uint32), [8..8+8*8) metrics (uint64[8]).
        self.control_region = storage.host_allocate(control_buffer_name, (self.CONTROL_BYTES,), torch.uint8)
        self.control_region.host_view.zero_()
        self.stop_flag_ptr = int(self.control_region.device_view.data_ptr())
        self.metrics_ptr = self.stop_flag_ptr + 8

        # Pack immutable per-layer descriptors into persistent HBM storage.
        packed = bytearray()
        for tier, hbm_weights in layers:
            if not isinstance(tier, MegaMoeDynamicHostTier):
                raise TypeError("SM103ExpertMoverRuntime requires MegaMoeDynamicHostTier layers")
            hbm_planes = (*hbm_weights.gate_up, *hbm_weights.down)
            host_planes = tuple(tier.device_views)
            if len(hbm_planes) != 4 or len(host_planes) != 4:
                raise ValueError("each dynamic MegaMoE layer requires four weight/scale planes")
            for p_idx, (hp, dp, row_b) in enumerate(zip(host_planes, hbm_planes, tier.plane_row_bytes)):
                if dp.shape[0] != tier.hbm_slots or hp.shape[0] != tier.rows:
                    raise ValueError(f"plane {p_idx} row counts do not match tier geometry")
                if dp.stride(0) * dp.element_size() != row_b:
                    raise ValueError(f"plane {p_idx} HBM row stride differs from HOST_NUMA row stride")
            entry = struct.pack(
                "<4Q4Q4Q3Q4I",
                *(int(t.data_ptr()) for t in host_planes),
                *(int(t.data_ptr()) for t in hbm_planes),
                *(int(b) for b in tier.plane_row_bytes),
                int(tier.live_slot_map.data_ptr()),
                int(tier.slot_owner.data_ptr()),
                int(tier.layer_epoch.data_ptr()),
                int(tier.rows),
                int(tier.hbm_slots),
                0,  # victim_cursor
                0,  # _pad
            )
            assert len(entry) == 136, len(entry)
            packed.extend(entry)
            packed.extend(b"\x00" * (desc_bytes - len(entry)))

        self.layer_descs = persistent_copy(torch.tensor(list(packed), dtype=torch.uint8, device=self.device))
        header_init = bytearray(hdr_bytes)
        struct.pack_into("<III", header_init, 0, 0, 0, queue_capacity - 1)
        self.queue_header = persistent_copy(torch.tensor(list(header_init), dtype=torch.uint8, device=self.device))
        self.queue_ring = persistent_zeros((queue_capacity * req_bytes,), dtype=torch.uint8, device=self.device)
        self.predictor_bindings: list[PredictorLayerBinding] = []
        for tensor in (self.layer_descs, self.queue_header, self.queue_ring, self.control_region.device_view):
            record_numeric_tensor(tensor)

    def attach_predictors(self, config: PredictorConfig | None = None) -> tuple[PredictorLayerBinding, ...]:
        """Allocate per-layer GPU predictor tables and bind them to each MegaMoeDynamicHostTier."""
        if self.predictor_bindings:
            raise RuntimeError("predictors have already been attached to this mover runtime")
        cfg = PredictorConfig() if config is None else config
        pred_bytes = self.binding.predictor_state_bytes()
        bindings = []
        init_clock = max(1, cfg.cooldown_steps)
        for layer_idx, (tier, _) in enumerate(self.layers):
            if tier.rows > 256:
                raise ValueError(f"inline GPU predictor supports up to 256 experts, got {tier.rows}")
            scores = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            last_seen = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            cooldown = persistent_zeros((tier.rows,), dtype=torch.int32, device=self.device)
            raw = struct.pack(
                "<3Q8I",
                int(scores.data_ptr()),
                int(last_seen.data_ptr()),
                int(cooldown.data_ptr()),
                init_clock,
                layer_idx,
                cfg.decay_shift,
                cfg.promote_threshold,
                cfg.hysteresis,
                cfg.cooldown_steps,
                cfg.max_enqueues,
                cfg.max_predict_rows,
            )
            assert len(raw) == 56, len(raw)
            state = persistent_copy(
                torch.tensor(list(raw) + [0] * (pred_bytes - len(raw)), dtype=torch.uint8, device=self.device)
            )
            for tensor in (scores, last_seen, cooldown, state):
                record_numeric_tensor(tensor)
            binding = PredictorLayerBinding(
                layer_idx=layer_idx,
                state=state,
                scores=scores,
                last_seen=last_seen,
                cooldown=cooldown,
                queue_header=self.queue_header,
                queue_ring=self.queue_ring,
                metrics_ptr=self.metrics_ptr,
                metrics_count=self.METRICS_COUNT,
            )
            tier.predictor_binding = binding
            bindings.append(binding)
        self.predictor_bindings = bindings
        return tuple(bindings)

    def enqueue_swap(self, layer_idx: int, expert_in: int, victim_expert: int = -1) -> None:
        self.binding.launch_enqueue_swap(
            self.queue_header, self.queue_ring, int(layer_idx), int(expert_in), int(victim_expert)
        )

    def enqueue_swap_batch(
        self, layers: torch.Tensor, experts_in: torch.Tensor, victims: torch.Tensor
    ) -> None:
        self.binding.launch_enqueue_swap_batch(
            self.queue_header, self.queue_ring, layers, experts_in, victims
        )

    def launch_worker(self, *, stream_handle: int = 0) -> None:
        self.binding.launch_persistent_mover(
            self.layer_descs,
            len(self.layers),
            self.queue_header,
            self.queue_ring,
            self.stop_flag_ptr,
            self.metrics_ptr,
            self.METRICS_COUNT,
            self.worker_sm_count,
            int(stream_handle),
        )

    def capture_worker_graph(
        self, *, name: str = "sm103_expert_mover", retain_graphs: bool = True, stream=None
    ) -> PersistentWorker:
        """Capture the persistent mover kernel into a dedicated CUDA graph without executing it."""
        graph = capture_graph(retain_graphs=retain_graphs)
        capture_stream = torch.cuda.Stream(device=self.device) if stream is None else stream
        with torch.cuda.device(self.device), torch.cuda.stream(capture_stream):
            with torch.cuda.graph(graph, pool=self.storage.graph_pool, stream=capture_stream):
                self.launch_worker(stream_handle=int(capture_stream.cuda_stream))
        if retain_graphs:
            record_export_graph(graph, self.storage, owners=(self,))
        return PersistentWorker(
            name=name,
            graph=graph,
            worker_sm_count=self.worker_sm_count,
            stop_flag_buffer=self.control_buffer_name,
            stop_flag_offset=0,
            metrics_buffer=self.control_buffer_name,
            metrics_offset=8,
            metrics_count=self.METRICS_COUNT,
        )

    def read_metrics(self) -> MoverMetrics:
        raw = bytes(self.control_region.host_view[8:8 + self.METRICS_COUNT * 8].tolist())
        swaps, copied, rcu_waits, skipped, last_packed, p_hits, p_misses, p_enq = struct.unpack("<8Q", raw)
        return MoverMetrics(
            swaps_completed=swaps,
            bytes_copied=copied,
            rcu_waits=rcu_waits,
            skipped_already_resident=skipped,
            last_layer=(last_packed >> 32) & 0xFFFFFFFF,
            last_expert=last_packed & 0xFFFFFFFF,
            predict_hits=p_hits,
            predict_misses=p_misses,
            predict_enqueues=p_enq,
        )

    def reset_metrics(self) -> None:
        self.control_region.host_view[8:8 + self.METRICS_COUNT * 8].zero_()

    def reset_runtime_state(self) -> None:
        """Reset mutable queue, epoch, predictor, and metric state after preparation warmup."""
        header_init = bytearray(self.binding.queue_header_bytes())
        struct.pack_into("<III", header_init, 0, 0, 0, self.queue_capacity - 1)
        self.queue_header.copy_(torch.tensor(list(header_init), dtype=torch.uint8, device=self.device))
        self.queue_ring.zero_()
        self.control_region.host_view.zero_()
        for tier, _ in self.layers:
            tier.layer_epoch.zero_()
        for binding in self.predictor_bindings:
            binding.scores.zero_()
            binding.last_seen.zero_()
            binding.cooldown.zero_()
            # Restore step_clock = max(1, cooldown_steps) at offset 24 (after 3 x uint64 pointers).
            cooldown_steps = struct.unpack_from("<I", bytes(binding.state[44:48].cpu().tolist()), 0)[0]
            init_clock = max(1, cooldown_steps)
            binding.state[24:28].copy_(
                torch.tensor(list(struct.pack("<I", init_clock)), dtype=torch.uint8, device=self.device)
            )

    def reset_stop_flag(self) -> None:
        self.binding.store_stop_flag_release(self.stop_flag_ptr, 0)

    def signal_stop(self) -> None:
        self.binding.store_stop_flag_release(self.stop_flag_ptr, 1)

    def close(self) -> None:
        if not self.control_region.closed:
            self.control_region.close()

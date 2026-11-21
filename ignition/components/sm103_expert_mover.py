# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preparation owner and persistent CUDA graph builder for the SM103 GPU expert mover."""

from __future__ import annotations

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
class MoverMetrics:
    swaps_completed: int
    bytes_copied: int
    rcu_waits: int
    skipped_already_resident: int
    last_layer: int
    last_expert: int


class SM103ExpertMoverRuntime:
    """Own GPU descriptor tables, lock-free prefetch ring, and mapped-host controls.

    Prepares the persistent 1D-TMA mover kernel for capture as a dedicated
    ``Execution.persistent_worker`` graph running on a spatially partitioned
    CUDA green context alongside compute graphs.
    """

    METRICS_COUNT = 5
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
        if desc_bytes != 192 or hdr_bytes != 64 or req_bytes != 16:
            raise RuntimeError(f"unexpected mover struct sizes: desc={desc_bytes}, hdr={hdr_bytes}, req={req_bytes}")

        # MappedHost control block: [0..4) stop_flag (uint32), [8..8+5*8) metrics (uint64[5]).
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
        for tensor in (self.layer_descs, self.queue_header, self.queue_ring, self.control_region.device_view):
            record_numeric_tensor(tensor)

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
        swaps, copied, rcu_waits, skipped, last_packed = struct.unpack("<5Q", raw)
        return MoverMetrics(
            swaps_completed=swaps,
            bytes_copied=copied,
            rcu_waits=rcu_waits,
            skipped_already_resident=skipped,
            last_layer=(last_packed >> 32) & 0xFFFFFFFF,
            last_expert=last_packed & 0xFFFFFFFF,
        )

    def reset_stop_flag(self) -> None:
        self.control_region.host_view[:4].view(torch.int32)[0] = 0

    def signal_stop(self) -> None:
        self.control_region.host_view[:4].view(torch.int32)[0] = 1

    def close(self) -> None:
        if not self.control_region.closed:
            self.control_region.close()

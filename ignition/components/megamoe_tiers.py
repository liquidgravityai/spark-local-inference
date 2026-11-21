# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Mapped host weight storage and per-layer control for the MegaMoE Grace tier."""

import math
from dataclasses import dataclass

import torch

from ignition.components.megamoe_weights import (
    MegaMoeWeights, allocate_mxfp4_weights, load_mxfp4_weights_into,
    load_shared_fp8_planes, mxfp4_weight_layouts, pack_shared_fp8_planes,
)
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor


@dataclass(frozen=True)
class MegaMoeLayerWeights:
    """Fused expert operands, canonical row mapping and retained protocol owners."""

    routed: MegaMoeWeights
    shared: MegaMoeWeights | None
    workspace: object
    host_tier: object | None
    expert_map: torch.Tensor
    host_experts: tuple[int, ...] = ()
    hbm_experts: tuple[int, ...] = ()


@dataclass(frozen=True)
class MegaMoeLayerLayout:
    """Explicit physical rows for one layer; repeated cross-tier IDs are copies.

    HBM rows override host rows in the initial canonical mapping. The caller
    owns placement selection and any later publication after readers drain.
    Workspaces may be shared only across serialized calls of equal geometry.
    """

    workspace: object
    host_experts: tuple[int, ...]
    hbm_experts: tuple[int, ...]
    storage: object
    host_lane_clusters: int = 4

    def load(self, loader, prefixes, *, shared_prefix, block_size,
             projection_names=("w1", "w3", "w2"), scale_name="scale", name):
        prefixes = tuple(prefixes)
        g = self.workspace.geometry
        physical = self.host_experts + self.hbm_experts
        if (not prefixes or len(prefixes) != len(set(prefixes))
                or not self.hbm_experts or len(physical) != g.experts
                or any(type(i) is not int or not 0 <= i < len(prefixes) for i in physical)
                or any(len(set(rows)) != len(rows) for rows in (self.host_experts, self.hbm_experts))
                or set(physical) != set(range(len(prefixes)))
                or (shared_prefix is not None) != bool(g.shared_experts)
                or loader.device != self.workspace.buffer.device):
            raise ValueError("MegaMoE placement must cover canonical experts with unique rows per tier and matching workspace")
        routed = allocate_mxfp4_weights(len(self.hbm_experts), g.hidden, g.intermediate, device=loader.device)
        tier = (MegaMoeHostTier(g, rows=len(self.host_experts), storage=self.storage,
                               name=name, host_lane_clusters=self.host_lane_clusters)
                if self.host_experts else None)
        destinations = [(self.hbm_experts, routed)]
        if tier is not None:
            destinations.append((self.host_experts, tier.cpu_weights))
        for ids, target in destinations:
            load_mxfp4_weights_into(loader, [prefixes[i] for i in ids], target,
                hidden_size=g.hidden, intermediate_size=g.intermediate,
                projection_names=projection_names, scale_name=scale_name)
        shared = (pack_shared_fp8_planes(load_shared_fp8_planes(loader, shared_prefix,
            hidden_size=g.hidden, intermediate_size=g.intermediate, block_size=block_size,
            projection_names=projection_names, scale_name=scale_name)) if shared_prefix is not None else None)
        mapping = [0] * len(prefixes)
        for row, canonical in enumerate(physical):
            mapping[canonical] = row
        expert_map = persistent_copy(torch.tensor(mapping, dtype=torch.int32, device=loader.device))
        record_numeric_tensor(expert_map)
        return MegaMoeLayerWeights(routed, shared, self.workspace, tier, expert_map,
                                   self.host_experts, self.hbm_experts)


class MegaMoeHostTier:
    """Own physical host rows; routing supplies host-first logical row identities.

    Rows below ``rows`` read registered host memory. Subsequent rows address the
    HBM matrices supplied to PreparedMegaMoe. Each layer owns its control block
    and tensor-map clone table, including when layers share protocol workspace.
    This storage does not choose residency or promote experts.
    """

    def __init__(self, geometry, *, rows, storage, name, host_lane_clusters=4):
        device = torch.device("cuda", storage.device)
        if (type(rows) is not int or not 0 < rows < geometry.experts
                or type(host_lane_clusters) is not int or host_lane_clusters < 0):
            raise ValueError("MegaMoE host storage requires nonempty host and HBM tiers and a nonnegative lane size")
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) != (10, 3):
            raise ValueError("the MegaMoE Grace tier requires SM103")
        if host_lane_clusters >= properties.multi_processor_count // 2:
            raise ValueError("MegaMoE host lane must leave at least one cluster for HBM work")
        self.geometry, self.rows = geometry, rows
        self.host_lane_clusters = host_lane_clusters
        self.regions, self.host_views, self.device_views = [], [], []
        labels = ("gate_up", "gate_up_scales", "down", "down_scales")
        layouts = mxfp4_weight_layouts(rows, geometry.hidden, geometry.intermediate)
        for label, (shape, dtype, stride) in zip(labels, layouts):
            region = storage.host_allocate(f"{name}.{label}", (math.prod(shape),), dtype)
            self.regions.append(region)
            self.host_views.append(region.host_view.as_strided(shape, stride))
            self.device_views.append(region.device_view.as_strided(shape, stride))
        self.weights = MegaMoeWeights(tuple(self.device_views[:2]), tuple(self.device_views[2:]))
        self.cpu_weights = MegaMoeWeights(tuple(self.host_views[:2]), tuple(self.host_views[2:]))
        for value in self.device_views:
            record_numeric_tensor(value)
        # The pinned kernel's table contains four 128-byte maps per CTA, at most 160 CTAs.
        self.descriptors = persistent_zeros((160 * 4 * 128,), dtype=torch.uint8, device=device)
        header = torch.zeros(16 + geometry.experts, dtype=torch.int32)
        header[0], header[1], header[10] = 0x53505244, rows, host_lane_clusters
        header[:16].view(torch.int64)[1:5] = torch.tensor([t.data_ptr() for t in self.device_views])
        header[:16].view(torch.int64)[7] = self.descriptors.data_ptr()
        self.control = persistent_copy(header.to(device))

    def copy_weights(self, weights, *, start=0):
        """Fill host rows during loading from a bounded batch of transformed experts."""
        source = (*weights.gate_up, *weights.down)
        count = source[0].shape[0]
        if type(start) is not int or count <= 0 or start < 0 or start + count > self.rows:
            raise ValueError("MegaMoE host weight copy exceeds the allocated expert rows")
        for src, dst in zip(source, self.host_views):
            if src.shape != (count, *dst.shape[1:]) or src.dtype != dst.dtype or src.stride() != dst.stride():
                raise ValueError("MegaMoE host weight copy requires the prepared physical layout")
        for src, dst in zip(source, self.host_views):
            dst[start:start + count].copy_(src)


class MegaMoeDynamicHostTier:
    """Canonical HOST_NUMA backing for all E experts plus dynamic HBM slot-map indirection.

    Every canonical expert ``0..E-1`` lives permanently at its canonical row in mapped host
    memory. HBM holds ``hbm_slots = active_slots + spare_slots`` (with ``spare_slots >= 1``)
    so a concurrent GPU mover can copy a cold expert into an unmapped spare slot, publish the
    swap into ``live_slot_map``, and wait out the Dekker/epoch-RCU grace period via
    ``layer_epoch`` before reclaiming the evicted slot.
    """

    MAGIC_INDIR = 0x49505244  # "DRPI"

    def __init__(self, geometry, *, hbm_slots, initial_hbm_experts, storage, name):
        device = torch.device("cuda", storage.device)
        initial_hbm_experts = tuple(initial_hbm_experts)
        if (type(hbm_slots) is not int or not 2 <= hbm_slots <= geometry.experts):
            raise ValueError("dynamic MegaMoE tier requires hbm_slots in [2, experts]")
        if (not initial_hbm_experts or len(initial_hbm_experts) >= hbm_slots
                or len(set(initial_hbm_experts)) != len(initial_hbm_experts)
                or any(type(e) is not int or not 0 <= e < geometry.experts for e in initial_hbm_experts)):
            raise ValueError("dynamic MegaMoE tier requires unique initial HBM experts leaving at least one spare slot")
        properties = torch.cuda.get_device_properties(device)
        if (properties.major, properties.minor) != (10, 3):
            raise ValueError("the dynamic MegaMoE Grace tier requires SM103")
        self.geometry = geometry
        self.rows = geometry.experts
        self.hbm_slots = hbm_slots
        self.host_lane_clusters = 0
        self.indirection = True
        self.regions, self.host_views, self.device_views = [], [], []
        self.plane_row_bytes = []
        labels = ("gate_up", "gate_up_scales", "down", "down_scales")
        layouts = mxfp4_weight_layouts(geometry.experts, geometry.hidden, geometry.intermediate)
        for label, (shape, dtype, stride) in zip(labels, layouts):
            region = storage.host_allocate(f"{name}.{label}", (math.prod(shape),), dtype)
            self.regions.append(region)
            self.host_views.append(region.host_view.as_strided(shape, stride))
            self.device_views.append(region.device_view.as_strided(shape, stride))
            row_bytes = stride[0] * dtype.itemsize
            if row_bytes % 16 != 0:
                raise ValueError(f"dynamic MegaMoE plane {label} row size {row_bytes} must be 16-byte aligned for 1D TMA")
            self.plane_row_bytes.append(row_bytes)
        self.weights = MegaMoeWeights(tuple(self.device_views[:2]), tuple(self.device_views[2:]))
        self.cpu_weights = MegaMoeWeights(tuple(self.host_views[:2]), tuple(self.host_views[2:]))
        for value in self.device_views:
            record_numeric_tensor(value)

        slot_map = [-1] * geometry.experts
        slot_owner = [-1] * hbm_slots
        for slot, expert in enumerate(initial_hbm_experts):
            slot_map[expert] = slot
            slot_owner[slot] = expert
        self.live_slot_map = persistent_copy(torch.tensor(slot_map, dtype=torch.int32, device=device))
        self.slot_owner = persistent_copy(torch.tensor(slot_owner, dtype=torch.int32, device=device))
        self.layer_epoch = persistent_zeros((1,), dtype=torch.int32, device=device)
        self.descriptors = persistent_zeros((160 * 4 * 128,), dtype=torch.uint8, device=device)
        header = torch.zeros(16 + geometry.experts, dtype=torch.int32)
        header[0] = self.MAGIC_INDIR
        header[1] = hbm_slots
        header[:16].view(torch.int64)[1:5] = torch.tensor([t.data_ptr() for t in self.device_views], dtype=torch.int64)
        header[:16].view(torch.int64)[6] = self.layer_epoch.data_ptr()
        header[:16].view(torch.int64)[7] = self.descriptors.data_ptr()
        header[16:] = torch.tensor(slot_map, dtype=torch.int32)
        self.control = persistent_copy(header.to(device))
        for t in (self.live_slot_map, self.slot_owner, self.layer_epoch, self.descriptors, self.control):
            record_numeric_tensor(t)

    def copy_weights(self, weights, *, start=0):
        """Fill canonical host rows during loading from a bounded batch of transformed experts."""
        source = (*weights.gate_up, *weights.down)
        count = source[0].shape[0]
        if type(start) is not int or count <= 0 or start < 0 or start + count > self.rows:
            raise ValueError("MegaMoE host weight copy exceeds the allocated expert rows")
        for src, dst in zip(source, self.host_views):
            if src.shape != (count, *dst.shape[1:]) or src.dtype != dst.dtype or src.stride() != dst.stride():
                raise ValueError("MegaMoE host weight copy requires the prepared physical layout")
        for src, dst in zip(source, self.host_views):
            dst[start:start + count].copy_(src)

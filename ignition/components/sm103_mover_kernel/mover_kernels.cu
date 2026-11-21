// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0

// SM103 GPU-side persistent expert mover (1D TMA cp.async.bulk over NVLink-C2C
// HOST_NUMA VMM), lock-free MPSC prefetch ring buffer, and Dekker/Epoch-RCU
// layer-enter snapshot kernel.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdint>

namespace flare::sm103_mover {

static constexpr uint32_t kTmaChunkBytes = 32768u; // 32 KiB x 2 stages = 64 KiB dynamic SMEM

struct alignas(64) LayerMoverDesc {
    uint64_t host_planes[4];     // HOST_NUMA base pointers: gate_up, gate_up_sf, down, down_sf
    uint64_t hbm_planes[4];      // HBM pool base pointers: gate_up, gate_up_sf, down, down_sf
    uint64_t plane_row_bytes[4]; // 16-byte-aligned byte stride per expert row
    uint64_t live_slot_map;      // int32_t[num_experts] in HBM (-1 = cold, >= 0 = HBM slot)
    uint64_t slot_owner;         // int32_t[num_hbm_slots] in HBM (-1 = spare/free)
    uint64_t layer_epoch;        // uint32_t* in HBM (even = idle, odd = active)
    uint32_t num_experts;
    uint32_t num_hbm_slots;
    uint32_t victim_cursor;      // clock-hand cursor in [0, num_hbm_slots)
    uint32_t _pad;
};

static_assert(sizeof(LayerMoverDesc) == 192, "LayerMoverDesc layout mismatch");

struct alignas(16) MoverRequest {
    uint32_t layer_idx;
    int32_t  expert_in;
    int32_t  victim_expert; // -1 = choose resident victim via clock hand
    uint32_t seq;           // MPSC publication tag: queue index + 1
};

static_assert(sizeof(MoverRequest) == 16, "MoverRequest layout mismatch");

struct alignas(64) MoverQueueHeader {
    uint32_t head;
    uint32_t tail;
    uint32_t mask;
    uint32_t barrier_count;
    uint32_t barrier_sense;
    uint32_t active_stop;
    uint32_t active_layer;
    int32_t  active_expert_in;
    int32_t  active_spare_slot;
    int32_t  active_victim_expert;
    int32_t  active_victim_slot;
    uint32_t _pad[5];
};

static_assert(sizeof(MoverQueueHeader) == 64, "MoverQueueHeader layout mismatch");

__device__ __forceinline__ void worker_grid_barrier(MoverQueueHeader* hdr, uint32_t& local_sense) {
    __syncthreads();
    if (threadIdx.x == 0) {
        const uint32_t target_sense = local_sense ^ 1u;
        uint32_t arrived = 0;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(arrived) : "l"(&hdr->barrier_count) : "memory");
        if (arrived + 1u == gridDim.x) {
            asm volatile("st.relaxed.gpu.global.u32 [%0], 0;"
                         :: "l"(&hdr->barrier_count) : "memory");
            asm volatile("st.release.gpu.global.u32 [%0], %1;"
                         :: "l"(&hdr->barrier_sense), "r"(target_sense) : "memory");
        } else {
            uint32_t cur = 0;
            do {
                asm volatile("nanosleep.u32 32;");
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(cur) : "l"(&hdr->barrier_sense) : "memory");
            } while (cur != target_sense);
        }
        local_sense = target_sense;
    }
    __syncthreads();
}

__device__ __forceinline__ void store_metric_sys(uint64_t* metrics, uint32_t metrics_count, uint32_t idx, uint64_t value) {
    if (metrics != nullptr && idx < metrics_count) {
        asm volatile("st.release.sys.global.u64 [%0], %1;"
                     :: "l"(metrics + idx), "l"(value) : "memory");
    }
}

__global__ void sm103_layer_enter_kernel(
    const int32_t* __restrict__ live_slot_map,
    int32_t* __restrict__ active_slot_map,
    uint32_t* __restrict__ layer_epoch,
    uint32_t num_experts
) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) {
        uint32_t old_ep;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(old_ep) : "l"(layer_epoch) : "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    __syncthreads();
    for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
        int32_t slot = -1;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                     : "=r"(slot) : "l"(live_slot_map + e) : "memory");
        active_slot_map[e] = slot;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
        asm volatile("fence.acq_rel.gpu;" ::: "memory");
    }
}

__global__ void sm103_enqueue_swap_kernel(
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    uint32_t layer_idx,
    int32_t expert_in,
    int32_t victim_expert
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t head = 0;
    uint32_t tail = 0;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(head) : "l"(&hdr->head) : "memory");
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(tail) : "l"(&hdr->tail) : "memory");
    const uint32_t capacity = hdr->mask + 1u;
    if (tail - head >= capacity) {
        // Queue full: drop best-effort prefetch request rather than stalling compute.
        return;
    }
    uint32_t idx = 0;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                 : "=r"(idx) : "l"(&hdr->tail) : "memory");
    MoverRequest* slot = ring + (idx & hdr->mask);
    slot->layer_idx = layer_idx;
    slot->expert_in = expert_in;
    slot->victim_expert = victim_expert;
    asm volatile("st.release.gpu.global.u32 [%0], %1;"
                 :: "l"(&slot->seq), "r"(idx + 1u) : "memory");
}

__global__ void sm103_enqueue_swap_batch_kernel(
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    const uint32_t* __restrict__ layers,
    const int32_t* __restrict__ experts_in,
    const int32_t* __restrict__ victims,
    uint32_t count
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint32_t capacity = hdr->mask + 1u;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t head = 0;
        uint32_t tail = 0;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(head) : "l"(&hdr->head) : "memory");
        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(tail) : "l"(&hdr->tail) : "memory");
        if (tail - head >= capacity) {
            break;
        }
        uint32_t idx = 0;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(idx) : "l"(&hdr->tail) : "memory");
        MoverRequest* slot = ring + (idx & hdr->mask);
        slot->layer_idx = layers[i];
        slot->expert_in = experts_in[i];
        slot->victim_expert = victims[i];
        asm volatile("st.release.gpu.global.u32 [%0], %1;"
                     :: "l"(&slot->seq), "r"(idx + 1u) : "memory");
    }
}

__global__ void sm103_persistent_mover_kernel(
    LayerMoverDesc* __restrict__ layers,
    uint32_t num_layers,
    MoverQueueHeader* __restrict__ hdr,
    MoverRequest* __restrict__ ring,
    volatile uint32_t* __restrict__ stop_flag,
    uint64_t* __restrict__ metrics,
    uint32_t metrics_count
) {
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(8) uint64_t mbar[2];
    uint8_t* buf[2] = {smem, smem + kTmaChunkBytes};

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[0]))));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[1]))));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    uint32_t phase[2] = {0u, 0u};
    int stage = 0;
    uint32_t local_sense = 0u;
    uint64_t swaps_completed = 0ull;
    uint64_t bytes_copied = 0ull;
    uint64_t rcu_waits = 0ull;
    uint64_t skipped_resident = 0ull;

    while (true) {
        // Step 1: CTA 0 polls stop_flag and lock-free MPSC queue, selects spare & victim.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            uint32_t should_stop = 0u;
            uint32_t chosen_layer = 0u;
            int32_t chosen_expert_in = -1;
            int32_t chosen_spare_slot = -1;
            int32_t chosen_victim_expert = -1;
            int32_t chosen_victim_slot = -1;

            while (true) {
                uint32_t stop = 0u;
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                             : "=r"(stop) : "l"(stop_flag) : "memory");
                if (stop != 0u) {
                    should_stop = 1u;
                    break;
                }

                uint32_t tail = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(tail) : "l"(&hdr->tail) : "memory");
                const uint32_t head = hdr->head;
                if (tail == head) {
                    asm volatile("nanosleep.u32 64;");
                    continue;
                }

                MoverRequest* slot = ring + (head & hdr->mask);
                uint32_t seq = 0u;
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                             : "=r"(seq) : "l"(&slot->seq) : "memory");
                if (seq != head + 1u) {
                    // Producer reserved tail and is still writing payload fields.
                    asm volatile("nanosleep.u32 32;");
                    continue;
                }

                const uint32_t layer_idx = slot->layer_idx;
                const int32_t expert_in = slot->expert_in;
                const int32_t req_victim = slot->victim_expert;
                asm volatile("st.release.gpu.global.u32 [%0], %1;"
                             :: "l"(&hdr->head), "r"(head + 1u) : "memory");

                if (layer_idx >= num_layers) {
                    continue;
                }
                LayerMoverDesc& L = layers[layer_idx];
                if (expert_in < 0 || static_cast<uint32_t>(expert_in) >= L.num_experts || L.num_hbm_slots < 2u) {
                    continue;
                }
                int32_t* live_map = reinterpret_cast<int32_t*>(L.live_slot_map);
                int32_t* slot_owner = reinterpret_cast<int32_t*>(L.slot_owner);

                if (live_map[expert_in] >= 0) {
                    skipped_resident += 1ull;
                    store_metric_sys(metrics, metrics_count, 3u, skipped_resident);
                    continue;
                }

                // Locate an unmapped spare HBM slot.
                int32_t spare_slot = -1;
                for (uint32_t s = 0; s < L.num_hbm_slots; ++s) {
                    if (slot_owner[s] < 0) {
                        spare_slot = static_cast<int32_t>(s);
                        break;
                    }
                }
                if (spare_slot < 0) {
                    continue;
                }

                // Choose resident victim: explicit request if currently resident, else clock hand.
                int32_t victim_expert = -1;
                int32_t victim_slot = -1;
                if (req_victim >= 0 && static_cast<uint32_t>(req_victim) < L.num_experts) {
                    int32_t s = live_map[req_victim];
                    if (s >= 0 && s != spare_slot) {
                        victim_expert = req_victim;
                        victim_slot = s;
                    }
                }
                if (victim_slot < 0) {
                    uint32_t cursor = L.victim_cursor % L.num_hbm_slots;
                    for (uint32_t step = 0; step < L.num_hbm_slots; ++step) {
                        const uint32_t cand = (cursor + step) % L.num_hbm_slots;
                        const int32_t owner = slot_owner[cand];
                        if (owner >= 0 && static_cast<int32_t>(cand) != spare_slot) {
                            victim_expert = owner;
                            victim_slot = static_cast<int32_t>(cand);
                            L.victim_cursor = (cand + 1u) % L.num_hbm_slots;
                            break;
                        }
                    }
                }
                if (victim_slot < 0) {
                    continue;
                }

                chosen_layer = layer_idx;
                chosen_expert_in = expert_in;
                chosen_spare_slot = spare_slot;
                chosen_victim_expert = victim_expert;
                chosen_victim_slot = victim_slot;
                break;
            }

            hdr->active_stop = should_stop;
            hdr->active_layer = chosen_layer;
            hdr->active_expert_in = chosen_expert_in;
            hdr->active_spare_slot = chosen_spare_slot;
            hdr->active_victim_expert = chosen_victim_expert;
            hdr->active_victim_slot = chosen_victim_slot;
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }

        // Step 2: Inter-CTA Barrier #1 broadcasts command (or stop) to all worker CTAs.
        worker_grid_barrier(hdr, local_sense);
        if (hdr->active_stop != 0u) {
            break;
        }

        const uint32_t layer_idx = hdr->active_layer;
        const int32_t expert_in = hdr->active_expert_in;
        const int32_t spare_slot = hdr->active_spare_slot;
        const LayerMoverDesc& L = layers[layer_idx];

        // Step 3: Parallel 8-SM double-buffered 1D TMA copy across all 4 MXFP4 planes.
        if (threadIdx.x == 0) {
            for (int p = 0; p < 4; ++p) {
                const uint64_t row_bytes = L.plane_row_bytes[p];
                if (row_bytes == 0ull) continue;
                const uint32_t num_chunks = static_cast<uint32_t>((row_bytes + kTmaChunkBytes - 1ull) / kTmaChunkBytes);
                const uint8_t* src_base = reinterpret_cast<const uint8_t*>(L.host_planes[p]) +
                                          static_cast<uint64_t>(expert_in) * row_bytes;
                uint8_t* dst_base = reinterpret_cast<uint8_t*>(L.hbm_planes[p]) +
                                    static_cast<uint64_t>(spare_slot) * row_bytes;
                for (uint32_t ch = blockIdx.x; ch < num_chunks; ch += gridDim.x) {
                    const uint64_t off = static_cast<uint64_t>(ch) * kTmaChunkBytes;
                    const uint64_t rem = row_bytes - off;
                    const uint32_t chunk_bytes = static_cast<uint32_t>(rem < kTmaChunkBytes ? rem : kTmaChunkBytes);
                    const uint32_t s_addr = static_cast<uint32_t>(__cvta_generic_to_shared(buf[stage]));
                    const uint32_t b_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&mbar[stage]));

                    asm volatile("cp.async.bulk.wait_group.read 1;");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%2], [%3], %1, [%0];\n"
                        :: "r"(b_addr), "r"(chunk_bytes), "r"(s_addr), "l"(src_base + off) : "memory"
                    );
                    uint32_t done = 0u;
                    do {
                        asm volatile(
                            "{ .reg .pred p; "
                            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; "
                            "selp.b32 %0, 1, 0, p; }"
                            : "=r"(done) : "r"(b_addr), "r"(phase[stage]) : "memory"
                        );
                    } while (!done);
                    phase[stage] ^= 1u;
                    asm volatile("fence.proxy.async.shared::cta;");
                    asm volatile(
                        "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
                        "cp.async.bulk.commit_group;\n"
                        :: "l"(dst_base + off), "r"(s_addr), "r"(chunk_bytes) : "memory"
                    );
                    stage ^= 1;
                }
            }
            asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
            asm volatile("fence.proxy.async.global;" ::: "memory");
            asm volatile("fence.acq_rel.gpu;" ::: "memory");
        }

        // Step 4: Inter-CTA Barrier #2 ensures all 4 planes are committed in HBM.
        worker_grid_barrier(hdr, local_sense);

        // Step 5: CTA 0 atomically publishes the swap and drains Dekker/Epoch-RCU before reclaiming victim slot.
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            const int32_t victim_expert = hdr->active_victim_expert;
            const int32_t victim_slot = hdr->active_victim_slot;
            int32_t* live_map = reinterpret_cast<int32_t*>(L.live_slot_map);
            int32_t* slot_owner = reinterpret_cast<int32_t*>(L.slot_owner);
            uint32_t* epoch_ptr = reinterpret_cast<uint32_t*>(L.layer_epoch);

            asm volatile("fence.acq_rel.gpu;" ::: "memory");
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(live_map + victim_expert), "r"(-1) : "memory");
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(live_map + expert_in), "r"(spare_slot) : "memory");
            // Dekker half 2: order unmap of victim_expert before reading layer_epoch.
            asm volatile("fence.sc.gpu;" ::: "memory");

            uint32_t ep = 0u;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                         : "=r"(ep) : "l"(epoch_ptr) : "memory");
            if (ep & 1u) {
                rcu_waits += 1ull;
                uint32_t cur = ep;
                while (cur == ep) {
                    uint32_t stop = 0u;
                    asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                                 : "=r"(stop) : "l"(stop_flag) : "memory");
                    if (stop != 0u) break;
                    asm volatile("nanosleep.u32 32;");
                    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                 : "=r"(cur) : "l"(epoch_ptr) : "memory");
                }
            }

            // Victim slot is now guaranteed unused by any active MegaMoE invocation.
            slot_owner[spare_slot] = expert_in;
            slot_owner[victim_slot] = -1;

            swaps_completed += 1ull;
            bytes_copied += L.plane_row_bytes[0] + L.plane_row_bytes[1] + L.plane_row_bytes[2] + L.plane_row_bytes[3];
            store_metric_sys(metrics, metrics_count, 0u, swaps_completed);
            store_metric_sys(metrics, metrics_count, 1u, bytes_copied);
            store_metric_sys(metrics, metrics_count, 2u, rcu_waits);
            store_metric_sys(metrics, metrics_count, 4u,
                             (static_cast<uint64_t>(layer_idx) << 32) | static_cast<uint32_t>(expert_in));
        }
    }
}

void launch_layer_enter(
    torch::Tensor live_slot_map,
    torch::Tensor control,
    torch::Tensor layer_epoch,
    int num_experts
) {
    TORCH_CHECK(live_slot_map.is_cuda() && live_slot_map.scalar_type() == torch::kInt && live_slot_map.is_contiguous());
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == torch::kInt && control.is_contiguous());
    TORCH_CHECK(layer_epoch.is_cuda() && layer_epoch.is_contiguous() && layer_epoch.numel() >= 1);
    TORCH_CHECK(num_experts > 0 && live_slot_map.numel() >= num_experts && control.numel() >= 16 + num_experts);
    const c10::cuda::CUDAGuard device_guard(live_slot_map.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_layer_enter_kernel<<<1, 256, 0, stream>>>(
        live_slot_map.data_ptr<int32_t>(),
        control.data_ptr<int32_t>() + 16,
        reinterpret_cast<uint32_t*>(layer_epoch.data_ptr()),
        static_cast<uint32_t>(num_experts)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_enqueue_swap(
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    int layer_idx,
    int expert_in,
    int victim_expert
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    const c10::cuda::CUDAGuard device_guard(queue_header.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_enqueue_swap_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        static_cast<uint32_t>(layer_idx),
        static_cast<int32_t>(expert_in),
        static_cast<int32_t>(victim_expert)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_enqueue_swap_batch(
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    torch::Tensor layers,
    torch::Tensor experts_in,
    torch::Tensor victims
) {
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(layers.is_cuda() && layers.scalar_type() == torch::kInt && layers.is_contiguous());
    TORCH_CHECK(experts_in.is_cuda() && experts_in.scalar_type() == torch::kInt && experts_in.is_contiguous());
    TORCH_CHECK(victims.is_cuda() && victims.scalar_type() == torch::kInt && victims.is_contiguous());
    TORCH_CHECK(layers.numel() == experts_in.numel() && layers.numel() == victims.numel());
    const c10::cuda::CUDAGuard device_guard(queue_header.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_enqueue_swap_batch_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<const uint32_t*>(layers.data_ptr()),
        experts_in.data_ptr<int32_t>(),
        victims.data_ptr<int32_t>(),
        static_cast<uint32_t>(layers.numel())
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void configure_persistent_mover() {
    const size_t smem_bytes = 2u * kTmaChunkBytes;
    cudaError_t err = cudaFuncSetAttribute(
        sm103_persistent_mover_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes)
    );
    TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute failed: ", cudaGetErrorString(err));
}

void launch_persistent_mover(
    torch::Tensor layer_descs,
    int num_layers,
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    uint64_t stop_flag_ptr,
    uint64_t metrics_ptr,
    int metrics_count,
    int worker_sm_count,
    uint64_t stream_handle
) {
    TORCH_CHECK(layer_descs.is_cuda() && layer_descs.is_contiguous());
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(num_layers > 0 && worker_sm_count > 0 && stop_flag_ptr != 0);
    TORCH_CHECK(layer_descs.numel() * layer_descs.element_size() >= static_cast<int64_t>(num_layers) * static_cast<int64_t>(sizeof(LayerMoverDesc)));
    TORCH_CHECK(queue_header.numel() * queue_header.element_size() >= static_cast<int64_t>(sizeof(MoverQueueHeader)));
    const c10::cuda::CUDAGuard device_guard(layer_descs.device());
    cudaStream_t stream = stream_handle != 0
        ? reinterpret_cast<cudaStream_t>(stream_handle)
        : c10::cuda::getCurrentCUDAStream().stream();
    const size_t smem_bytes = 2u * kTmaChunkBytes;
    sm103_persistent_mover_kernel<<<static_cast<uint32_t>(worker_sm_count), 32, smem_bytes, stream>>>(
        reinterpret_cast<LayerMoverDesc*>(layer_descs.data_ptr()),
        static_cast<uint32_t>(num_layers),
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<volatile uint32_t*>(stop_flag_ptr),
        reinterpret_cast<uint64_t*>(metrics_ptr),
        static_cast<uint32_t>(metrics_count)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int layer_desc_bytes() { return static_cast<int>(sizeof(LayerMoverDesc)); }
int queue_header_bytes() { return static_cast<int>(sizeof(MoverQueueHeader)); }
int queue_request_bytes() { return static_cast<int>(sizeof(MoverRequest)); }

} // namespace flare::sm103_mover

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("configure_persistent_mover", &flare::sm103_mover::configure_persistent_mover);
    m.def("launch_layer_enter", &flare::sm103_mover::launch_layer_enter);
    m.def("launch_enqueue_swap", &flare::sm103_mover::launch_enqueue_swap);
    m.def("launch_enqueue_swap_batch", &flare::sm103_mover::launch_enqueue_swap_batch);
    m.def("launch_persistent_mover", &flare::sm103_mover::launch_persistent_mover);
    m.def("layer_desc_bytes", &flare::sm103_mover::layer_desc_bytes);
    m.def("queue_header_bytes", &flare::sm103_mover::queue_header_bytes);
    m.def("queue_request_bytes", &flare::sm103_mover::queue_request_bytes);
}

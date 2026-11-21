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
    uint32_t _pad0;
    uint64_t predict_hits;
    uint64_t predict_misses;
    uint64_t predict_enqueues;
    uint64_t _pad1;
};

static_assert(sizeof(MoverQueueHeader) == 80 || sizeof(MoverQueueHeader) == 128, "MoverQueueHeader layout mismatch");

struct alignas(64) PredictorLayerState {
    uint64_t scores;            // uint32_t[num_experts] fixed-point decayed frequency (1 hit = 256)
    uint64_t last_seen;         // uint32_t[num_experts] monotonic step clock of last active hit
    uint64_t cooldown;          // uint32_t[num_experts] monotonic step clock of last enqueue
    uint32_t step_clock;        // initialized >= cooldown_steps; incremented only on active steps
    uint32_t layer_idx;
    uint32_t decay_shift;       // e.g. 4 -> score -= score >> 4 (0.9375x decay per active step)
    uint32_t promote_threshold; // fixed-point threshold (256 = 1 hit, 384 = 1.5 hits)
    uint32_t hysteresis;        // minimum score margin above victim to enqueue swap
    uint32_t cooldown_steps;    // minimum steps before re-enqueuing the same cold expert
    uint32_t max_enqueues;      // cap on swap requests enqueued per layer invocation (<= 4)
    uint32_t max_predict_rows;  // skip enqueueing when num_rows > max_predict_rows (scores still update)
};

static_assert(sizeof(PredictorLayerState) == 64, "PredictorLayerState layout mismatch");

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

__device__ __forceinline__ void store_metric_sys_relaxed(uint64_t* metrics, uint32_t metrics_count, uint32_t idx, uint64_t value) {
    if (metrics != nullptr && idx < metrics_count) {
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
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

__global__ void sm103_layer_enter_predict_kernel(
    const int32_t* __restrict__ live_slot_map,
    int32_t* __restrict__ active_slot_map,
    uint32_t* __restrict__ layer_epoch,
    const int64_t* __restrict__ topk_ids,
    uint32_t num_rows,
    uint32_t num_topk,
    uint32_t num_experts,
    PredictorLayerState* __restrict__ pred,
    MoverQueueHeader* __restrict__ queue_hdr,
    MoverRequest* __restrict__ queue_ring,
    uint64_t* __restrict__ metrics,
    uint32_t metrics_count
) {
    if (blockIdx.x != 0) return;

    // 1. Dekker half 1: mark layer ACTIVE (even -> odd) and fence SC before reading live_slot_map.
    if (threadIdx.x == 0) {
        uint32_t old_ep;
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], 1;"
                     : "=r"(old_ep) : "l"(layer_epoch) : "memory");
        asm volatile("fence.sc.gpu;" ::: "memory");
    }
    __syncthreads();

    __shared__ int32_t  s_slot[256];
    __shared__ uint32_t s_hits[256];
    __shared__ uint32_t s_active_routes;
    __shared__ uint32_t s_hbm_hits;
    __shared__ uint32_t s_host_misses;

    if (threadIdx.x == 0) {
        s_active_routes = 0u;
        s_hbm_hits = 0u;
        s_host_misses = 0u;
    }
    if (threadIdx.x < 256u) {
        s_slot[threadIdx.x] = -1;
        s_hits[threadIdx.x] = 0u;
    }

    // 2. Snapshot live_slot_map[0..E-1] -> active_slot_map[0..E-1] and shared cache.
    for (uint32_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
        int32_t slot = -1;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];"
                     : "=r"(slot) : "l"(live_slot_map + e) : "memory");
        active_slot_map[e] = slot;
        if (e < 256u) {
            s_slot[e] = slot;
        }
    }
    __syncthreads();

    // 3. Tally active routed token-expert pairs (inactive/padded rows carry -1 and never mutate state).
    if (pred != nullptr && topk_ids != nullptr && num_experts <= 256u) {
        const uint32_t total_routes = num_rows * num_topk;
        const uint32_t lane_id = threadIdx.x & 31u;
        uint32_t local_hbm_hits = 0u;
        uint32_t local_host_misses = 0u;
        for (uint32_t base = 0u; base < total_routes; base += blockDim.x) {
            const uint32_t i = base + threadIdx.x;
            bool is_hit = false;
            bool is_miss = false;
            if (i < total_routes) {
                const int64_t e = topk_ids[i];
                if (e >= 0 && static_cast<uint64_t>(e) < static_cast<uint64_t>(num_experts)) {
                    const uint32_t ue = static_cast<uint32_t>(e);
                    atomicAdd(&s_hits[ue], 1u);
                    if (s_slot[ue] >= 0) {
                        is_hit = true;
                    } else {
                        is_miss = true;
                    }
                }
            }
            const uint32_t warp_hits = __popc(__ballot_sync(0xFFFFFFFFu, is_hit));
            const uint32_t warp_misses = __popc(__ballot_sync(0xFFFFFFFFu, is_miss));
            if (lane_id == 0u) {
                local_hbm_hits += warp_hits;
                local_host_misses += warp_misses;
            }
        }
        if (lane_id == 0u && (local_hbm_hits + local_host_misses) > 0u) {
            atomicAdd(&s_hbm_hits, local_hbm_hits);
            atomicAdd(&s_host_misses, local_host_misses);
            atomicAdd(&s_active_routes, local_hbm_hits + local_host_misses);
        }
        __syncthreads();

        // 4. Update decayed frequency & recency ONLY when at least one active route occurred.
        if (s_active_routes > 0u) {
            const uint32_t step_now = pred->step_clock + 1u;
            uint32_t* scores = reinterpret_cast<uint32_t*>(pred->scores);
            uint32_t* last_seen = reinterpret_cast<uint32_t*>(pred->last_seen);
            uint32_t* cooldown = reinterpret_cast<uint32_t*>(pred->cooldown);

            // Parallel score/recency update and candidate key evaluation across all 256 threads.
            // Cold key (maximize): ((uint64_t)sc << 32) | (uint64_t)(e + 1), 0 = ineligible.
            // Victim key (minimize): ((uint64_t)sc << 32) | ((uint64_t)(ls & 0xFFFFFFu) << 8) | e, ~0ull = ineligible.
            uint64_t my_cold_key = 0ull;
            uint64_t my_vic_key = ~0ull;

            if (threadIdx.x < num_experts) {
                const uint32_t e = threadIdx.x;
                uint32_t sc = scores[e];
                if (pred->decay_shift > 0u && pred->decay_shift < 31u) {
                    sc = sc - (sc >> pred->decay_shift);
                }
                const uint32_t hits = s_hits[e];
                if (hits > 0u) {
                    const uint32_t capped = hits < 16u ? hits : 16u;
                    const uint32_t boost = capped << 8;
                    sc = (sc + boost < 0x7FFFFFFFu) ? (sc + boost) : 0x7FFFFFFFu;
                    last_seen[e] = step_now;
                }
                scores[e] = sc;

                if (s_host_misses > 0u && num_rows <= pred->max_predict_rows && pred->max_enqueues > 0u) {
                    if (s_slot[e] < 0) {
                        if (hits > 0u && sc >= pred->promote_threshold &&
                            (step_now - cooldown[e]) >= pred->cooldown_steps) {
                            my_cold_key = (static_cast<uint64_t>(sc) << 32) | static_cast<uint64_t>(e + 1u);
                        }
                    } else if (hits == 0u) {
                        const uint32_t ls = last_seen[e];
                        my_vic_key = (static_cast<uint64_t>(sc) << 32) |
                                     (static_cast<uint64_t>(ls & 0x00FFFFFFu) << 8) |
                                     static_cast<uint64_t>(e);
                    }
                }
            }

            // 8-warp parallel reduction: each of the 8 warps produces its top cold candidate & best victim in 5 shuffles.
            __shared__ uint64_t s_warp_cold[8];
            __shared__ uint64_t s_warp_vic[8];
            const uint32_t lane = threadIdx.x & 31u;
            const uint32_t warp = threadIdx.x >> 5;
            if (s_host_misses > 0u && num_rows <= pred->max_predict_rows && pred->max_enqueues > 0u) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const uint64_t other_c = __shfl_down_sync(0xFFFFFFFFu, my_cold_key, offset);
                    if (other_c > my_cold_key) my_cold_key = other_c;
                    const uint64_t other_v = __shfl_down_sync(0xFFFFFFFFu, my_vic_key, offset);
                    if (other_v < my_vic_key) my_vic_key = other_v;
                }
                if (lane == 0u && warp < 8u) {
                    s_warp_cold[warp] = my_cold_key;
                    s_warp_vic[warp] = my_vic_key;
                }
            }
            __syncthreads();

            // 5. Thread 0 inspects the 8 warp winners and updates HBM counters without host-bus traffic.
            if (threadIdx.x == 0) {
                pred->step_clock = step_now;
                if (queue_hdr != nullptr) {
                    const uint64_t tot_hits = queue_hdr->predict_hits + static_cast<uint64_t>(s_hbm_hits);
                    const uint64_t tot_misses = queue_hdr->predict_misses + static_cast<uint64_t>(s_host_misses);
                    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                 :: "l"(&queue_hdr->predict_hits), "l"(tot_hits) : "memory");
                    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                 :: "l"(&queue_hdr->predict_misses), "l"(tot_misses) : "memory");
                }

                if (s_host_misses > 0u && num_rows <= pred->max_predict_rows &&
                    pred->max_enqueues > 0u && queue_hdr != nullptr && queue_ring != nullptr) {
                    const uint32_t k_max = pred->max_enqueues < 4u ? pred->max_enqueues : 4u;
                    uint64_t top_cold[4] = {0ull, 0ull, 0ull, 0ull};
                    uint64_t top_vic[4] = {~0ull, ~0ull, ~0ull, ~0ull};
                    uint32_t n_cold = 0u;
                    uint32_t n_vic = 0u;

                    #pragma unroll
                    for (uint32_t w = 0u; w < 8u; ++w) {
                        const uint64_t ck = s_warp_cold[w];
                        if (ck != 0ull) {
                            uint32_t pos = n_cold;
                            if (pos < k_max) {
                                n_cold += 1u;
                            } else if (ck <= top_cold[k_max - 1u]) {
                                pos = 4u;
                            } else {
                                pos = k_max - 1u;
                            }
                            if (pos < 4u) {
                                while (pos > 0u && ck > top_cold[pos - 1u]) {
                                    top_cold[pos] = top_cold[pos - 1u];
                                    pos -= 1u;
                                }
                                top_cold[pos] = ck;
                            }
                        }
                        const uint64_t vk = s_warp_vic[w];
                        if (vk != ~0ull) {
                            uint32_t pos = n_vic;
                            if (pos < k_max) {
                                n_vic += 1u;
                            } else if (vk >= top_vic[k_max - 1u]) {
                                pos = 4u;
                            } else {
                                pos = k_max - 1u;
                            }
                            if (pos < 4u) {
                                while (pos > 0u && vk < top_vic[pos - 1u]) {
                                    top_vic[pos] = top_vic[pos - 1u];
                                    pos -= 1u;
                                }
                                top_vic[pos] = vk;
                            }
                        }
                    }

                    const uint32_t pairs = n_cold < n_vic ? n_cold : n_vic;
                    int32_t enq_cold[4];
                    int32_t enq_vic[4];
                    uint32_t n_valid = 0u;
                    for (uint32_t p = 0u; p < pairs; ++p) {
                        const uint32_t cold_score = static_cast<uint32_t>(top_cold[p] >> 32);
                        const uint32_t vic_score = static_cast<uint32_t>(top_vic[p] >> 32);
                        if (cold_score < vic_score + pred->hysteresis) {
                            break;
                        }
                        enq_cold[n_valid] = static_cast<int32_t>((top_cold[p] & 0xFFFFFFFFull) - 1ull);
                        enq_vic[n_valid] = static_cast<int32_t>(top_vic[p] & 0xFFull);
                        n_valid += 1u;
                    }

                    if (n_valid > 0u) {
                        uint32_t head = 0u, tail = 0u;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
                                     : "=r"(head) : "l"(&queue_hdr->head) : "memory");
                        asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];"
                                     : "=r"(tail) : "l"(&queue_hdr->tail) : "memory");
                        const uint32_t cap = queue_hdr->mask + 1u;
                        const uint32_t avail = (tail - head < cap) ? (cap - (tail - head)) : 0u;
                        if (n_valid > avail) {
                            n_valid = avail;
                        }
                        if (n_valid > 0u) {
                            uint32_t base_idx = 0u;
                            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;"
                                         : "=r"(base_idx) : "l"(&queue_hdr->tail), "r"(n_valid) : "memory");
                            const uint32_t mask = queue_hdr->mask;
                            for (uint32_t p = 0u; p < n_valid; ++p) {
                                MoverRequest* rslot = queue_ring + ((base_idx + p) & mask);
                                rslot->layer_idx = pred->layer_idx;
                                rslot->expert_in = enq_cold[p];
                                rslot->victim_expert = enq_vic[p];
                                cooldown[enq_cold[p]] = step_now;
                            }
                            asm volatile("fence.acq_rel.gpu;" ::: "memory");
                            for (uint32_t p = 0u; p < n_valid; ++p) {
                                MoverRequest* rslot = queue_ring + ((base_idx + p) & mask);
                                asm volatile("st.relaxed.gpu.global.u32 [%0], %1;"
                                             :: "l"(&rslot->seq), "r"(base_idx + p + 1u) : "memory");
                            }
                            const uint64_t tot_enq = queue_hdr->predict_enqueues + static_cast<uint64_t>(n_valid);
                            asm volatile("st.relaxed.gpu.global.u64 [%0], %1;"
                                         :: "l"(&queue_hdr->predict_enqueues), "l"(tot_enq) : "memory");
                        }
                    }
                }
            }
        }
    }

    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.global;" ::: "memory");
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
    uint64_t mirrored_hits = 0ull;
    uint64_t mirrored_misses = 0ull;
    uint64_t mirrored_enqueues = 0ull;

    auto mirror_predictor_metrics = [&]() {
        if (metrics == nullptr || metrics_count < 8u) return;
        uint64_t h = 0ull, m = 0ull, e = 0ull;
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(h) : "l"(&hdr->predict_hits) : "memory");
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(m) : "l"(&hdr->predict_misses) : "memory");
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(e) : "l"(&hdr->predict_enqueues) : "memory");
        if (h != mirrored_hits) {
            mirrored_hits = h;
            store_metric_sys(metrics, metrics_count, 5u, h);
        }
        if (m != mirrored_misses) {
            mirrored_misses = m;
            store_metric_sys(metrics, metrics_count, 6u, m);
        }
        if (e != mirrored_enqueues) {
            mirrored_enqueues = e;
            store_metric_sys(metrics, metrics_count, 7u, e);
        }
    };

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
                mirror_predictor_metrics();
                uint32_t stop = 0u;
                asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
                             : "=r"(stop) : "l"(stop_flag) : "memory");
                if (stop != 0u) {
                    mirror_predictor_metrics();
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
        uint32_t stop_cmd = 0u;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(stop_cmd) : "l"(&hdr->active_stop) : "memory");
        if (stop_cmd != 0u) {
            break;
        }

        uint32_t layer_idx = 0u;
        int32_t expert_in = 0;
        int32_t spare_slot = 0;
        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(layer_idx) : "l"(&hdr->active_layer) : "memory");
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(expert_in) : "l"(&hdr->active_expert_in) : "memory");
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(spare_slot) : "l"(&hdr->active_spare_slot) : "memory");
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

void launch_layer_enter_predict(
    torch::Tensor live_slot_map,
    torch::Tensor control,
    torch::Tensor layer_epoch,
    torch::Tensor topk_ids,
    int num_rows,
    int num_topk,
    int num_experts,
    torch::Tensor predictor_state,
    torch::Tensor queue_header,
    torch::Tensor queue_ring,
    uint64_t metrics_ptr,
    int metrics_count
) {
    TORCH_CHECK(live_slot_map.is_cuda() && live_slot_map.scalar_type() == torch::kInt && live_slot_map.is_contiguous());
    TORCH_CHECK(control.is_cuda() && control.scalar_type() == torch::kInt && control.is_contiguous());
    TORCH_CHECK(layer_epoch.is_cuda() && layer_epoch.is_contiguous() && layer_epoch.numel() >= 1);
    TORCH_CHECK(topk_ids.is_cuda() && topk_ids.scalar_type() == torch::kLong && topk_ids.is_contiguous());
    TORCH_CHECK(predictor_state.is_cuda() && predictor_state.is_contiguous());
    TORCH_CHECK(queue_header.is_cuda() && queue_header.is_contiguous());
    TORCH_CHECK(queue_ring.is_cuda() && queue_ring.is_contiguous());
    TORCH_CHECK(num_rows > 0 && num_topk > 0 && num_experts > 0);
    TORCH_CHECK(live_slot_map.numel() >= num_experts && control.numel() >= 16 + num_experts);
    TORCH_CHECK(predictor_state.numel() * predictor_state.element_size() >= static_cast<int64_t>(sizeof(PredictorLayerState)));
    const c10::cuda::CUDAGuard device_guard(live_slot_map.device());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sm103_layer_enter_predict_kernel<<<1, 256, 0, stream>>>(
        live_slot_map.data_ptr<int32_t>(),
        control.data_ptr<int32_t>() + 16,
        reinterpret_cast<uint32_t*>(layer_epoch.data_ptr()),
        topk_ids.data_ptr<int64_t>(),
        static_cast<uint32_t>(num_rows),
        static_cast<uint32_t>(num_topk),
        static_cast<uint32_t>(num_experts),
        reinterpret_cast<PredictorLayerState*>(predictor_state.data_ptr()),
        reinterpret_cast<MoverQueueHeader*>(queue_header.data_ptr()),
        reinterpret_cast<MoverRequest*>(queue_ring.data_ptr()),
        reinterpret_cast<uint64_t*>(metrics_ptr),
        static_cast<uint32_t>(metrics_count)
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
int predictor_state_bytes() { return static_cast<int>(sizeof(PredictorLayerState)); }

void store_stop_flag_release(uint64_t stop_flag_ptr, uint32_t value) {
    TORCH_CHECK(stop_flag_ptr != 0);
    __atomic_store_n(reinterpret_cast<uint32_t*>(stop_flag_ptr), value, __ATOMIC_RELEASE);
}

} // namespace flare::sm103_mover

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("configure_persistent_mover", &flare::sm103_mover::configure_persistent_mover);
    m.def("launch_layer_enter", &flare::sm103_mover::launch_layer_enter);
    m.def("launch_layer_enter_predict", &flare::sm103_mover::launch_layer_enter_predict);
    m.def("launch_enqueue_swap", &flare::sm103_mover::launch_enqueue_swap);
    m.def("launch_enqueue_swap_batch", &flare::sm103_mover::launch_enqueue_swap_batch);
    m.def("launch_persistent_mover", &flare::sm103_mover::launch_persistent_mover);
    m.def("layer_desc_bytes", &flare::sm103_mover::layer_desc_bytes);
    m.def("queue_header_bytes", &flare::sm103_mover::queue_header_bytes);
    m.def("queue_request_bytes", &flare::sm103_mover::queue_request_bytes);
    m.def("predictor_state_bytes", &flare::sm103_mover::predictor_state_bytes);
    m.def("store_stop_flag_release", &flare::sm103_mover::store_stop_flag_release);
}

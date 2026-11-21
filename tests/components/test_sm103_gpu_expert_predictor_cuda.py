# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware qualification for the inline on-GPU routing predictor + 8-SM 1D-TMA expert mover."""

import ctypes as C
import os
from pathlib import Path
import time

import pytest
import torch

from tests.components.test_sm103_gpu_expert_mover_cuda import (
    _GreenCtxPair,
    _make_random_mxfp4_weights,
    _make_random_shared_fp8,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


@pytest.fixture(scope="module")
def sm103_pred_storage(megamoe_core_storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")
    return megamoe_core_storage


def test_sm103_inline_gpu_predictor_adapts_across_regime_shifts_bit_exact(sm103_pred_storage):
    """Verify inline on-GPU predictor promotes cold working sets across topic shifts with bit-exact MegaMoE output."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import (
        PredictorConfig,
        SM103ExpertMoverRuntime,
        query_green_context_compute_sms,
    )

    device = torch.device("cuda", 0)
    worker_sms, compute_sms = query_green_context_compute_sms(0, 8)
    gc = _GreenCtxPair(0, worker_sms)
    assert gc.worker_sms == worker_sms and gc.compute_sms == compute_sms

    E = 64
    S_ACTIVE = 16
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE  # 18 slots
    HIDDEN = 1024
    INTER = 512
    TOP_K = 4
    CAPACITY = 128
    ROWS = 16
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_pred_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        gen = torch.Generator(device="cpu").manual_seed(20261121)
        geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
        ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
        ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

        full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
        shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

        # Start with experts 0..15 resident in HBM; 16..63 are cold in HOST_NUMA.
        tier = MegaMoeDynamicHostTier(
            ws_dyn.geometry,
            hbm_slots=HBM_SLOTS,
            initial_hbm_experts=tuple(range(S_ACTIVE)),
            storage=storage,
            name="pred_layer0",
        )
        tier.copy_weights(
            type(full_weights)(
                (full_weights.gate_up[0].cpu(), full_weights.gate_up[1].cpu()),
                (full_weights.down[0].cpu(), full_weights.down[1].cpu()),
            )
        )
        hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
        for dst, src in zip((*hbm_pool.gate_up, *hbm_pool.down), (*full_weights.gate_up, *full_weights.down)):
            dst[:S_ACTIVE].copy_(src[:S_ACTIVE])
            dst[S_ACTIVE:].zero_()

        mover = SM103ExpertMoverRuntime(
            [(tier, hbm_pool)],
            storage=storage,
            kernel_cache=kernel_cache,
            worker_sm_count=worker_sms,
            queue_capacity=1024,
        )
        # Fast adaptation settings: 256 = 1-hit promote threshold, 4 enqueues/step, decay_shift=2 (0.75x) for rapid regime shift.
        mover.attach_predictors(
            PredictorConfig(
                decay_shift=2,
                promote_threshold=256,
                hysteresis=64,
                cooldown_steps=1,
                max_enqueues=4,
                max_predict_rows=64,
            )
        )

        with PreparationSession(device=device, autotune=False) as session:
            ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
            probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
            pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            ref_op = PreparedMegaMoe(
                ex, ids, probs, pad, full_weights, shared_weights,
                workspace=ws_ref, clamp=CLAMP, session=session, name="pred_ref",
                host_shared_first=True,
            )
            dyn_op = PreparedMegaMoe(
                ex, ids, probs, pad, hbm_pool, shared_weights,
                workspace=ws_dyn, clamp=CLAMP, session=session, name="pred_dyn",
                host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
            )

        def make_regime_batches(expert_start: int, count: int):
            out = []
            for _ in range(count):
                x = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16)
                perm = torch.stack([torch.randperm(S_ACTIVE, generator=gen)[:TOP_K] + expert_start for _ in range(ROWS)])
                ids_cpu = perm.to(torch.int32).contiguous()
                probs_cpu = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32)
                pad_cpu = torch.zeros((ROWS,), dtype=torch.bool)
                gold = ref_op(x.to(device), ids_cpu.to(device), probs_cpu.to(device), pad_cpu.to(device)).cpu()
                out.append((x, ids_cpu, probs_cpu, pad_cpu, gold))
            torch.cuda.synchronize()
            return out

        regime_a = make_regime_batches(32, 24)  # Cold experts 32..47 (0% initial HBM hit rate)
        regime_b = make_regime_batches(48, 24)  # Abrupt topic shift to cold experts 48..63.

        # Capture compute graph (with inline predictor in sm103_layer_enter_predict_kernel) and mover graph.
        x_buf = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
        probs_buf = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
        out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
        out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        torch.cuda.synchronize()

        cap_stream = torch.cuda.Stream(device=device)
        comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_stream):
            with torch.cuda.graph(comp_graph, pool=storage.graph_pool, stream=cap_stream):
                out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
        comp_graph.instantiate()

        worker_decl = mover.capture_worker_graph(name="sm103_pred_mover", retain_graphs=True, stream=cap_stream)
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)

        def run_step(x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu=None):
            gc.set_compute()
            for dst, src in ((x_buf, x_cpu), (ids_buf, ids_cpu), (probs_buf, probs_cpu), (pad_buf, pad_cpu)):
                gc._call(
                    "cuMemcpyHtoDAsync_v2",
                    C.c_uint64(dst.data_ptr()),
                    C.c_void_p(src.data_ptr()),
                    C.c_size_t(src.numel() * src.element_size()),
                    gc.c_stream,
                )
            gc.launch_graph(int(comp_graph.raw_cuda_graph_exec()), worker=False)
            gc._call(
                "cuMemcpyDtoHAsync_v2",
                C.c_void_p(host_out.data_ptr()),
                C.c_uint64(out_buf.data_ptr()),
                C.c_size_t(host_out.numel() * host_out.element_size()),
                gc.c_stream,
            )
            gc.sync_compute()
            if gold_cpu is not None:
                assert torch.equal(host_out, gold_cpu), (
                    f"inline predictor live swap mismatch: max_diff={(host_out.float() - gold_cpu.float()).abs().max().item()}"
                )

        gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
        try:
            # --- Regime A: Cold experts 32..47 (starts at 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_a):
                run_step(*batch)
                # Give background 8-SM mover realistic inter-step decode window (~2ms) during warmup steps
                if step_i < 10:
                    time.sleep(0.002)
            m_before_tail_a = mover.read_metrics()
            for batch in regime_a[-6:]:
                run_step(*batch)
            m_after_a = mover.read_metrics()
            tail_hits_a = m_after_a.predict_hits - m_before_tail_a.predict_hits
            tail_misses_a = m_after_a.predict_misses - m_before_tail_a.predict_misses
            tail_rate_a = tail_hits_a / max(1, tail_hits_a + tail_misses_a)
            print(
                f"\n[Predictor Regime A (32..47)] tail_hit_rate={tail_rate_a*100:.1f}% "
                f"({tail_hits_a}/{tail_hits_a+tail_misses_a}), swaps={m_after_a.swaps_completed}, "
                f"enqueues={m_after_a.predict_enqueues}"
            )
            assert tail_rate_a >= 0.95, f"expected Regime A tail hit rate >= 95%, got {tail_rate_a:.3f} ({m_after_a})"

            # --- Regime B: Abrupt shift to cold experts 48..63 (drops back to 0% HBM hit rate) ---
            for step_i, batch in enumerate(regime_b):
                run_step(*batch)
                if step_i < 12:
                    time.sleep(0.002)
            m_before_tail_b = mover.read_metrics()
            for batch in regime_b[-6:]:
                run_step(*batch)
            m_after_b = mover.read_metrics()
            tail_hits_b = m_after_b.predict_hits - m_before_tail_b.predict_hits
            tail_misses_b = m_after_b.predict_misses - m_before_tail_b.predict_misses
            tail_rate_b = tail_hits_b / max(1, tail_hits_b + tail_misses_b)
            print(
                f"[Predictor Regime B (48..63)] tail_hit_rate={tail_rate_b*100:.1f}% "
                f"({tail_hits_b}/{tail_hits_b+tail_misses_b}), swaps={m_after_b.swaps_completed}, "
                f"enqueues={m_after_b.predict_enqueues}"
            )
            assert tail_rate_b >= 0.95, f"expected Regime B tail hit rate >= 95%, got {tail_rate_b:.3f} ({m_after_b})"

            # --- Principle 7: Inactive/padded rows perform ZERO predictor state mutations ---
            m_before_pad = mover.read_metrics()
            all_pad_x = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16)
            all_pad_ids = torch.zeros((ROWS, TOP_K), dtype=torch.int32)
            all_pad_probs = torch.zeros((ROWS, TOP_K), dtype=torch.float32)
            all_pad_mask = torch.ones((ROWS,), dtype=torch.bool)
            for _ in range(5):
                run_step(all_pad_x, all_pad_ids, all_pad_probs, all_pad_mask)
            m_after_pad = mover.read_metrics()
            assert m_after_pad.predict_hits == m_before_pad.predict_hits
            assert m_after_pad.predict_misses == m_before_pad.predict_misses
            assert m_after_pad.predict_enqueues == m_before_pad.predict_enqueues
        finally:
            gc.sync_compute()
            mover.signal_stop()
            gc.sync_worker()
            gc.set_primary()
            worker_decl.graph.reset()
            comp_graph.reset()

        assert int(tier.layer_epoch.item()) % 2 == 0
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_inline_predictor_kernel_latency_deepseek41_shape(sm103_pred_storage):
    """Measure sm103_layer_enter_predict_kernel latency on exact DeepSeek V4.1 dimensions (E=256, rows=16, topk=8)."""
    from ignition.components.megamoe import MegaMoeGeometry
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import PredictorConfig, SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    E, HBM_SLOTS, ROWS, TOP_K = 256, 130, 16, 8
    kernel_cache = Path("/tmp/flare-test-kcache")
    geom = MegaMoeGeometry(E, 128, TOP_K, 128, 128, shared_experts=1)  # Minimal weights; predictor depends only on E/rows/top_k
    tier = MegaMoeDynamicHostTier(
        geom, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(128)),
        storage=sm103_pred_storage, name="pred_lat_layer0",
    )
    hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, 128, 128, device=device)
    mover = SM103ExpertMoverRuntime(
        [(tier, hbm_pool)], storage=sm103_pred_storage, kernel_cache=kernel_cache, worker_sm_count=8
    )
    try:
        (binding,) = mover.attach_predictors(PredictorConfig(max_enqueues=2))
        gen = torch.Generator(device="cpu").manual_seed(99)
        topk_hit = torch.stack([torch.randperm(128, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int64).to(device).contiguous()
        topk_miss = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int64).to(device).contiguous()
        # Warmup
        for _ in range(20):
            mover.binding.launch_layer_enter_predict(
                tier.live_slot_map, tier.control, tier.layer_epoch, topk_miss,
                ROWS, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                binding.metrics_ptr, binding.metrics_count,
            )
        torch.cuda.synchronize()
        mover.reset_runtime_state()
        torch.cuda.synchronize()

        # Measure bare layer_enter (Dekker/RCU snapshot only), steady-state HBM-hit enter+predict,
        # and active cold-miss+enqueue enter+predict inside CUDA graphs.
        cap_s = torch.cuda.Stream(device=device)
        iters = 200
        g_base = torch.cuda.CUDAGraph()
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_base, stream=cap_s):
                for _ in range(iters):
                    mover.binding.launch_layer_enter(tier.live_slot_map, tier.control, tier.layer_epoch, E)
        g_hit = torch.cuda.CUDAGraph()
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_hit, stream=cap_s):
                for _ in range(iters):
                    mover.binding.launch_layer_enter_predict(
                        tier.live_slot_map, tier.control, tier.layer_epoch, topk_hit,
                        ROWS, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                        binding.metrics_ptr, binding.metrics_count,
                    )
        g_miss = torch.cuda.CUDAGraph()
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_miss, stream=cap_s):
                for _ in range(iters):
                    mover.binding.launch_layer_enter_predict(
                        tier.live_slot_map, tier.control, tier.layer_epoch, topk_miss,
                        ROWS, TOP_K, E, binding.state, binding.queue_header, binding.queue_ring,
                        binding.metrics_ptr, binding.metrics_count,
                    )
        torch.cuda.synchronize()
        g_base.replay()
        g_hit.replay()
        g_miss.replay()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        g_base.replay()
        end.record()
        torch.cuda.synchronize()
        base_us = (start.elapsed_time(end) * 1000.0) / iters

        start.record()
        g_hit.replay()
        end.record()
        torch.cuda.synchronize()
        hit_us = (start.elapsed_time(end) * 1000.0) / iters

        start.record()
        g_miss.replay()
        end.record()
        torch.cuda.synchronize()
        miss_us = (start.elapsed_time(end) * 1000.0) / iters

        g_base.reset()
        g_hit.reset()
        g_miss.reset()
        hit_delta_us = hit_us - base_us
        miss_delta_us = miss_us - base_us
        print(
            f"\n[Predictor Latency (Graph) E=256 rows=16 topk=8] "
            f"base_enter={base_us:.2f} us/layer, "
            f"steady_hit={hit_us:.2f} us/layer ({hit_delta_us:+.2f} us, {hit_us * 40:.1f} us/40L), "
            f"active_miss={miss_us:.2f} us/layer ({miss_delta_us:+.2f} us, {miss_us * 40:.1f} us/40L)"
        )
        assert hit_us < 8.0, f"steady-state inline predictor too slow: {hit_us:.2f} us/layer"
        assert miss_us < 12.0, f"active-miss inline predictor too slow: {miss_us:.2f} us/layer"
    finally:
        mover.close()
        for r in tier.regions:
            r.close()

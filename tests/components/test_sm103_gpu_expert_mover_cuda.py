# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""SM103 hardware qualification for indirection-aware MegaMoE + 8-SM 1D-TMA persistent mover."""

import ctypes as C
import os
from pathlib import Path
import time

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned SM103 CUDA device (FLARE_RUN_CUDA_TESTS=1)",
)


class _GreenCtxPair:
    """Minimal CUDA driver helper splitting device SMs into worker + compute green contexts."""

    CU_DEV_RESOURCE_TYPE_SM = 1
    CU_GREEN_CTX_DEFAULT_STREAM = 1
    CU_STREAM_NON_BLOCKING = 1

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

    def __init__(self, device_ordinal: int, worker_sm_count: int):
        self.lib = C.CDLL("libcuda.so.1")
        self.device = C.c_int(device_ordinal)
        self.primary_ctx = C.c_void_p()
        self._call("cuDevicePrimaryCtxRetain", C.byref(self.primary_ctx), self.device)
        self._call("cuCtxSetCurrent", self.primary_ctx)

        sm_res = self._DevResource()
        self._call("cuDeviceGetDevResource", self.device, C.byref(sm_res), C.c_int(self.CU_DEV_RESOURCE_TYPE_SM))
        w_res = self._DevResource()
        c_res = self._DevResource()
        nb = C.c_uint(1)
        self._call(
            "cuDevSmResourceSplitByCount",
            C.byref(w_res), C.byref(nb), C.byref(sm_res), C.byref(c_res),
            C.c_uint(0), C.c_uint(worker_sm_count),
        )
        self.worker_sms = int(w_res.sm_count)
        self.compute_sms = int(c_res.sm_count)

        w_desc, c_desc = C.c_void_p(), C.c_void_p()
        self._call("cuDevResourceGenerateDesc", C.byref(w_desc), C.byref(w_res), C.c_uint(1))
        self._call("cuDevResourceGenerateDesc", C.byref(c_desc), C.byref(c_res), C.c_uint(1))

        self.w_gctx, self.c_gctx = C.c_void_p(), C.c_void_p()
        self._call("cuGreenCtxCreate", C.byref(self.w_gctx), w_desc, self.device, C.c_uint(self.CU_GREEN_CTX_DEFAULT_STREAM))
        self._call("cuGreenCtxCreate", C.byref(self.c_gctx), c_desc, self.device, C.c_uint(self.CU_GREEN_CTX_DEFAULT_STREAM))

        self.w_ctx, self.c_ctx = C.c_void_p(), C.c_void_p()
        self._call("cuCtxFromGreenCtx", C.byref(self.w_ctx), self.w_gctx)
        self.w_stream, self.c_stream = C.c_void_p(), C.c_void_p()
        self._call("cuCtxFromGreenCtx", C.byref(self.c_ctx), self.c_gctx)
        self._call("cuGreenCtxStreamCreate", C.byref(self.w_stream), self.w_gctx, C.c_uint(self.CU_STREAM_NON_BLOCKING), C.c_int(0))
        self._call("cuGreenCtxStreamCreate", C.byref(self.c_stream), self.c_gctx, C.c_uint(self.CU_STREAM_NON_BLOCKING), C.c_int(0))

    def _call(self, name, *args):
        fn = getattr(self.lib, name)
        fn.restype = C.c_int
        rc = fn(*args)
        if rc != 0:
            msg = C.c_char_p()
            self.lib.cuGetErrorString(C.c_int(rc), C.byref(msg))
            raise RuntimeError(f"{name} failed with CUDA {rc}: {msg.value!r}")

    def set_worker(self):
        self._call("cuCtxSetCurrent", self.w_ctx)

    def set_compute(self):
        self._call("cuCtxSetCurrent", self.c_ctx)

    def set_primary(self):
        self._call("cuCtxSetCurrent", self.primary_ctx)

    def launch_graph(self, graph_exec: int, *, worker: bool):
        self._call("cuCtxSetCurrent", self.w_ctx if worker else self.c_ctx)
        stream = self.w_stream if worker else self.c_stream
        self._call("cuGraphLaunch", C.c_void_p(graph_exec), stream)

    def sync_worker(self):
        self._call("cuCtxSetCurrent", self.w_ctx)
        self._call("cuStreamSynchronize", self.w_stream)

    def sync_compute(self):
        self._call("cuCtxSetCurrent", self.c_ctx)
        self._call("cuStreamSynchronize", self.c_stream)

    def close(self):
        self._call("cuCtxSetCurrent", self.primary_ctx)
        self._call("cuStreamDestroy_v2", self.c_stream)
        self._call("cuStreamDestroy_v2", self.w_stream)
        self._call("cuGreenCtxDestroy", self.c_gctx)
        self._call("cuGreenCtxDestroy", self.w_gctx)
        self._call("cuDevicePrimaryCtxRelease_v2", self.device)


def _make_random_mxfp4_weights(experts, hidden, intermediate, device, gen):
    from ignition.components.megamoe_weights import allocate_mxfp4_weights

    weights = allocate_mxfp4_weights(experts, hidden, intermediate, device=device)
    # Use deterministic non-zero FP4 nibbles and moderate UE8M0 exponents (124..127 -> 0.125..1.0)
    weights.gate_up[0].copy_(torch.randint(-120, 120, weights.gate_up[0].shape, dtype=torch.int8, generator=gen).to(device))
    weights.down[0].copy_(torch.randint(-120, 120, weights.down[0].shape, dtype=torch.int8, generator=gen).to(device))
    for scale in (weights.gate_up[1], weights.down[1]):
        raw = torch.randint(123, 127, (*scale.shape, 4), dtype=torch.uint8, generator=gen)
        packed = raw.view(torch.int32).squeeze(-1).to(device)
        scale.copy_(packed)
    return weights


def _make_random_shared_fp8(hidden, intermediate, device, gen):
    from ignition.components.megamoe_weights import MegaMoeWeights

    def make_pair(rows, cols):
        sf_cols = cols // 128
        vals = (torch.randn((rows, cols), generator=gen) * 0.25).to(torch.float8_e4m3fn).to(device)
        raw = torch.randint(125, 128, (rows, sf_cols, 4), dtype=torch.uint8, generator=gen)
        sf_cpu = raw.view(torch.int32).squeeze(-1)
        sf = torch.empty_strided((rows, sf_cols), (1, rows), dtype=torch.int32, device=device)
        sf.copy_(sf_cpu.to(device))
        return (vals, sf)

    return MegaMoeWeights(make_pair(2 * intermediate, hidden), make_pair(hidden, intermediate))


@pytest.fixture(scope="module")
def sm103_storage(megamoe_core_storage):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")
    return megamoe_core_storage


def test_sm103_indirection_megamoe_bit_exact_under_concurrent_8sm_tma_swaps(sm103_storage):
    """Verify bit-exact parity of indirection MegaMoE vs all-HBM reference under live concurrent swaps."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    worker_sms = 8
    # Match CUDA green-context SM split rounding (always even for 2-CTA MegaMoE cluster).
    gc = _GreenCtxPair(0, worker_sms)
    compute_sms = gc.compute_sms
    assert gc.worker_sms == worker_sms and compute_sms > 0 and compute_sms % 2 == 0
    print(f"\n[SM103] total_sms={props.multi_processor_count} worker_sms={gc.worker_sms} compute_sms={compute_sms}")

    E = 32
    S_ACTIVE = 14
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE  # 16 slots (< 32 canonical experts)
    HIDDEN = 1024
    INTER = 512
    TOP_K = 4
    CAPACITY = 128
    ROW_SHAPES = (1, 16, 128)
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        if True:
            gen = torch.Generator(device="cpu").manual_seed(20261120)
            geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
            ws_ref = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)
            ws_dyn = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

            full_weights = _make_random_mxfp4_weights(E, HIDDEN, INTER, device, gen)
            shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)

            # Prepare dynamic tier: ALL E canonical experts in HOST_NUMA, initial 0..S_ACTIVE-1 in HBM, K_SPARE free.
            initial_hbm = tuple(range(S_ACTIVE))
            tier = MegaMoeDynamicHostTier(
                ws_dyn.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=initial_hbm, storage=storage, name="layer0"
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
                worker_sm_count=gc.worker_sms,
                queue_capacity=1024,
            )

            # Prepare static all-HBM reference and dynamic indirection MegaMoE for all row shapes.
            ref_ops = {}
            dyn_ops = {}
            with PreparationSession(device=device, autotune=False) as session:
                for rows in ROW_SHAPES:
                    ex = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                    ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
                    probs = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                    pad = torch.zeros((rows,), dtype=torch.bool, device=device)
                    ref_ops[rows] = PreparedMegaMoe(
                        ex, ids, probs, pad, full_weights, shared_weights,
                        workspace=ws_ref, clamp=CLAMP, session=session, name=f"ref.{rows}",
                        host_shared_first=True,
                    )
                    dyn_ops[rows] = PreparedMegaMoe(
                        ex, ids, probs, pad, hbm_pool, shared_weights,
                        workspace=ws_dyn, clamp=CLAMP, session=session, name=f"dyn.{rows}",
                        host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
                    )

            # Build deterministic test batches across all row shapes (mixing hot & cold experts).
            batches = []
            for step in range(40):
                rows = ROW_SHAPES[step % len(ROW_SHAPES)]
                x = (torch.randn((rows, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
                # Unique top-k expert IDs per token across all 0..E-1.
                perm = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(rows)])
                ids = perm.to(torch.int32).to(device).contiguous()
                probs = torch.softmax(torch.randn((rows, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
                pad = torch.zeros((rows,), dtype=torch.bool, device=device)
                if rows >= 8 and step % 3 == 0:
                    pad[-2:] = True  # Exercise masked inactive rows
                batches.append((rows, x, ids, probs, pad))

            # 1. Compute golden all-HBM reference outputs on primary context.
            print("[SM103] Session prepared! Computing golden all-HBM reference...", flush=True)
            golden = []
            for rows, x, ids, probs, pad in batches:
                out = ref_ops[rows](x, ids, probs, pad)
                golden.append(out.clone())
            torch.cuda.synchronize()
            print(f"[SM103] Golden reference ready ({len(golden)} batches). Checking static indirection...", flush=True)

            # 2. Verify static indirection parity BEFORE starting mover (mix of HBM + HOST_NUMA experts).
            for idx, (rows, x, ids, probs, pad) in enumerate(batches[:6]):
                out = dyn_ops[rows](x, ids, probs, pad)
                torch.cuda.synchronize()
                max_diff = (out.float() - golden[idx].float()).abs().max().item()
                assert max_diff == 0.0, f"static indirection mismatch at batch {idx} (rows={rows}): max_diff={max_diff}"

            # 3. Pre-capture compute graphs (zero GPU allocation while worker is running) and worker graph.
            print("[SM103] Static indirection bit-exact! Pre-capturing compute + worker graphs...", flush=True)
            # Capture one compute graph per row shape using fixed persistent I/O buffers.
            graph_io = {}
            compute_graphs = {}
            cap_stream = torch.cuda.Stream(device=device)
            for rows in ROW_SHAPES:
                x_buf = torch.zeros((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                ids_buf = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
                probs_buf = torch.full((rows, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                pad_buf = torch.zeros((rows,), dtype=torch.bool, device=device)
                out_buf = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device=device)
                # Warmup once outside capture
                out_buf.copy_(dyn_ops[rows](x_buf, ids_buf, probs_buf, pad_buf))
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.stream(cap_stream):
                    with torch.cuda.graph(g, pool=storage.graph_pool, stream=cap_stream):
                        out_buf.copy_(dyn_ops[rows](x_buf, ids_buf, probs_buf, pad_buf))
                g.instantiate()
                graph_io[rows] = (x_buf, ids_buf, probs_buf, pad_buf, out_buf)
                compute_graphs[rows] = g

            # Capture a batch-swap enqueue graph (rotates through all E experts) and the persistent mover graph.
            swap_layers = torch.zeros((E,), dtype=torch.int32, device=device)
            swap_experts = torch.arange(E, dtype=torch.int32, device=device)
            swap_victims = torch.full((E,), -1, dtype=torch.int32, device=device)
            enqueue_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_stream):
                with torch.cuda.graph(enqueue_graph, pool=storage.graph_pool, stream=cap_stream):
                    mover.enqueue_swap_batch(swap_layers, swap_experts, swap_victims)
            enqueue_graph.instantiate()

            worker_decl = mover.capture_worker_graph(name="sm103_mover_test", retain_graphs=True, stream=cap_stream)
            worker_exec = int(worker_decl.graph.raw_cuda_graph_exec())
            enqueue_exec = int(enqueue_graph.raw_cuda_graph_exec())

            # Pre-allocate pinned host staging for bit-exact output comparison without touching GPU allocator.
            cpu_batches = [
                (rows, x.cpu(), ids.cpu(), probs.cpu(), pad.cpu(), g_out.cpu())
                for (rows, x, ids, probs, pad), g_out in zip(batches, golden)
            ]
            host_out = {rows: torch.empty((rows, HIDDEN), dtype=torch.bfloat16, pin_memory=False) for rows in ROW_SHAPES}
            mover.reset_stop_flag()
            torch.cuda.synchronize()

            print("[SM103] Launching persistent mover graph on 8-SM worker green context...", flush=True)
            gc.launch_graph(worker_exec, worker=True)

            try:
                # Step 3a: Verify mover executes a batch of swaps cleanly before compute interleaving.
                gc.launch_graph(enqueue_exec, worker=False)
                gc.sync_compute()
                deadline = time.monotonic() + 5.0
                while mover.read_metrics().swaps_completed < 10 and time.monotonic() < deadline:
                    time.sleep(0.005)
                m0 = mover.read_metrics()
                print(f"[SM103] Initial mover burst OK: {m0}", flush=True)
                assert m0.swaps_completed >= 10, f"mover did not progress on worker green context: {m0}"

                # Step 3b: Hammer 144-SM compute graphs concurrently with live 8-SM swaps!
                for round_idx in range(4):
                    shift = (round_idx * 7) % E
                    rotated = torch.roll(torch.arange(E, dtype=torch.int32), shifts=shift)
                    gc.set_compute()
                    gc._call(
                        "cuMemcpyHtoDAsync_v2",
                        C.c_uint64(swap_experts.data_ptr()),
                        C.c_void_p(rotated.data_ptr()),
                        C.c_size_t(rotated.numel() * 4),
                        gc.c_stream,
                    )
                    gc.launch_graph(enqueue_exec, worker=False)

                    for idx, (rows, x_cpu, ids_cpu, probs_cpu, pad_cpu, gold_cpu) in enumerate(cpu_batches):
                        x_buf, ids_buf, probs_buf, pad_buf, out_buf = graph_io[rows]
                        gc.set_compute()
                        for dst, src in ((x_buf, x_cpu), (ids_buf, ids_cpu), (probs_buf, probs_cpu), (pad_buf, pad_cpu)):
                            gc._call(
                                "cuMemcpyHtoDAsync_v2",
                                C.c_uint64(dst.data_ptr()),
                                C.c_void_p(src.data_ptr()),
                                C.c_size_t(src.numel() * src.element_size()),
                                gc.c_stream,
                            )
                        gc.launch_graph(int(compute_graphs[rows].raw_cuda_graph_exec()), worker=False)
                        dst_cpu = host_out[rows]
                        gc._call(
                            "cuMemcpyDtoHAsync_v2",
                            C.c_void_p(dst_cpu.data_ptr()),
                            C.c_uint64(out_buf.data_ptr()),
                            C.c_size_t(dst_cpu.numel() * dst_cpu.element_size()),
                            gc.c_stream,
                        )
                        gc.sync_compute()
                        if not torch.equal(dst_cpu, gold_cpu):
                            max_diff = (dst_cpu.float() - gold_cpu.float()).abs().max().item()
                            raise AssertionError(
                                f"LIVE SWAP CORRUPTION at round={round_idx} batch={idx} (rows={rows}): "
                                f"max_diff={max_diff}"
                            )
                    print(f"[SM103] Round {round_idx} bit-exact! metrics={mover.read_metrics()}", flush=True)
            finally:
                # Drain compute BEFORE signaling mover stop!
                gc.sync_compute()
                mover.signal_stop()
                gc.sync_worker()
                gc.set_primary()
                worker_decl.graph.reset()
                enqueue_graph.reset()
                for g in compute_graphs.values():
                    g.reset()

            metrics = mover.read_metrics()
            print(
                f"[SM103] LIVE SWAP PARITY PASSED! rounds=4 batches={4 * len(batches)} "
                f"swaps_completed={metrics.swaps_completed} bytes_copied={metrics.bytes_copied} "
                f"rcu_waits={metrics.rcu_waits} skipped_resident={metrics.skipped_already_resident}"
            )
            assert metrics.swaps_completed >= 20, f"expected active live swaps, got {metrics}"
            assert metrics.bytes_copied == metrics.swaps_completed * sum(tier.plane_row_bytes)
            assert int(tier.layer_epoch.item()) % 2 == 0, "layer_epoch must finish even after drain"

            # Verify slot_map <-> slot_owner invariant after hundreds of live swaps!
            final_map = tier.live_slot_map.cpu().tolist()
            final_owner = tier.slot_owner.cpu().tolist()
            resident_from_map = {e: s for e, s in enumerate(final_map) if s >= 0}
            resident_from_owner = {e: s for s, e in enumerate(final_owner) if e >= 0}
            assert resident_from_map == resident_from_owner, (
                f"slot_map / slot_owner diverged: {resident_from_map} vs {resident_from_owner}"
            )
            assert len(resident_from_map) == S_ACTIVE
            assert final_owner.count(-1) == K_SPARE
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_deepseek41_expert_geometry_tma_bandwidth_and_parity(sm103_storage):
    """Verify full DeepSeek V4.1 expert geometry (E=256, H=6144, I=2048, 20.05 MiB/expert) parity and 8-SM TMA BW."""
    import deep_gemm
    from b12x.preparation import PreparationSession
    from ignition.components.megamoe import MegaMoeGeometry, MegaMoeWorkspace, PreparedMegaMoe
    from ignition.components.megamoe_tiers import MegaMoeDynamicHostTier
    from ignition.components.megamoe_weights import allocate_mxfp4_weights
    from ignition.components.sm103_expert_mover import SM103ExpertMoverRuntime

    device = torch.device("cuda", 0)
    worker_sms = 8
    gc = _GreenCtxPair(0, worker_sms)
    compute_sms = gc.compute_sms
    E = 256
    S_ACTIVE = 32
    K_SPARE = 2
    HBM_SLOTS = S_ACTIVE + K_SPARE
    HIDDEN = 6144
    INTER = 2048
    TOP_K = 8
    CAPACITY = 128
    ROWS = 16
    CLAMP = 7.0

    kernel_cache = Path("/tmp/flare-test-kcache")
    kernel_cache.mkdir(parents=True, exist_ok=True)
    storage = sm103_storage
    mover = None
    tier = None
    try:
        gc.set_primary()
        deep_gemm.set_num_sms(compute_sms)
        if True:
            gen = torch.Generator(device="cpu").manual_seed(4102026)
            geom = MegaMoeGeometry(E, CAPACITY, TOP_K, HIDDEN, INTER, shared_experts=1)
            ws = MegaMoeWorkspace(geom, device=device, kernel_cache=kernel_cache)

            # Fill 256 canonical experts in HOST_NUMA (~5.13 GiB) and 32 initial active + 2 spare in HBM (~682 MiB).
            tier = MegaMoeDynamicHostTier(
                ws.geometry, hbm_slots=HBM_SLOTS, initial_hbm_experts=tuple(range(S_ACTIVE)),
                storage=storage, name="dsv41_layer0",
            )
            # Populate all 256 host experts deterministically (distinct per-expert byte patterns) without OOMing GPU.
            for p_idx, dst_cpu in enumerate(tier.host_views):
                if dst_cpu.dtype == torch.int8:
                    vals = torch.randint(-120, 120, (E, 1, dst_cpu.shape[2]), dtype=torch.int8, generator=gen)
                    dst_cpu.copy_(vals.expand_as(dst_cpu))
                else:
                    raw = torch.randint(123, 127, (E, 1, dst_cpu.shape[2], 4), dtype=torch.uint8, generator=gen)
                    packed = raw.view(torch.int32).squeeze(-1)
                    dst_cpu.copy_(packed.expand_as(dst_cpu))

            hbm_pool = allocate_mxfp4_weights(HBM_SLOTS, HIDDEN, INTER, device=device)
            for dst, src_cpu in zip((*hbm_pool.gate_up, *hbm_pool.down), tier.host_views):
                dst[:S_ACTIVE].copy_(src_cpu[:S_ACTIVE].to(device))
                dst[S_ACTIVE:].zero_()

            shared_weights = _make_random_shared_fp8(HIDDEN, INTER, device, gen)
            mover = SM103ExpertMoverRuntime(
                [(tier, hbm_pool)], storage=storage, kernel_cache=kernel_cache,
                worker_sm_count=gc.worker_sms, queue_capacity=1024,
            )

            with PreparationSession(device=device, autotune=False) as session:
                ex = torch.zeros((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
                ids = torch.arange(TOP_K, dtype=torch.int32, device=device).expand(ROWS, -1).contiguous()
                probs = torch.full((ROWS, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
                pad = torch.zeros((ROWS,), dtype=torch.bool, device=device)
                dyn_op = PreparedMegaMoe(
                    ex, ids, probs, pad, hbm_pool, shared_weights,
                    workspace=ws, clamp=CLAMP, session=session, name="dsv41_dyn",
                    host_tier=tier, host_shared_first=True, compute_sm_count=compute_sms,
                )

            # Capture static-output golden BEFORE any swaps on cold experts 128..191 (read from HOST_NUMA)
            x_buf = (torch.randn((ROWS, HIDDEN), generator=gen) * 0.5).to(torch.bfloat16).to(device)
            ids_buf = torch.stack([torch.randperm(E, generator=gen)[:TOP_K] for _ in range(ROWS)]).to(torch.int32).to(device).contiguous()
            probs_buf = torch.softmax(torch.randn((ROWS, TOP_K), generator=gen), dim=-1).to(torch.float32).to(device)
            pad_buf = torch.zeros((ROWS,), dtype=torch.bool, device=device)
            out_buf = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16, device=device)
            golden = dyn_op(x_buf, ids_buf, probs_buf, pad_buf).clone()
            torch.cuda.synchronize()

            cap_stream = torch.cuda.Stream(device=device)
            comp_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_stream):
                with torch.cuda.graph(comp_graph, pool=storage.graph_pool, stream=cap_stream):
                    out_buf.copy_(dyn_op(x_buf, ids_buf, probs_buf, pad_buf))
            comp_graph.instantiate()

            # Enqueue 32 full 20.05 MiB DeepSeek V4.1 expert swaps (641.7 MiB total) to measure sustained 8-SM 1D TMA BW.
            n_swaps = 32
            swap_layers = torch.zeros((n_swaps,), dtype=torch.int32, device=device)
            swap_experts = torch.arange(128, 128 + n_swaps, dtype=torch.int32, device=device)
            swap_victims = torch.full((n_swaps,), -1, dtype=torch.int32, device=device)
            enq_graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.stream(cap_stream):
                with torch.cuda.graph(enq_graph, pool=storage.graph_pool, stream=cap_stream):
                    mover.enqueue_swap_batch(swap_layers, swap_experts, swap_victims)
            enq_graph.instantiate()

            worker_decl = mover.capture_worker_graph(name="dsv41_mover", retain_graphs=True, stream=cap_stream)
            mover.reset_stop_flag()
            torch.cuda.synchronize()

            gc.launch_graph(int(worker_decl.graph.raw_cuda_graph_exec()), worker=True)
            try:
                t0 = time.perf_counter()
                gc.launch_graph(int(enq_graph.raw_cuda_graph_exec()), worker=False)
                gc.sync_compute()
                # Also run 16 concurrent MegaMoE replays while 32 full 20 MiB experts stream over NVLink-C2C!
                host_out = torch.empty((ROWS, HIDDEN), dtype=torch.bfloat16)
                gold_cpu = golden.cpu()
                for _ in range(16):
                    gc.launch_graph(int(comp_graph.raw_cuda_graph_exec()), worker=False)
                    gc.set_compute()
                    gc._call(
                        "cuMemcpyDtoHAsync_v2",
                        C.c_void_p(host_out.data_ptr()),
                        C.c_uint64(out_buf.data_ptr()),
                        C.c_size_t(host_out.numel() * host_out.element_size()),
                        gc.c_stream,
                    )
                    gc.sync_compute()
                    assert torch.equal(host_out, gold_cpu), "DS V4.1 full-geometry live swap parity mismatch!"
                deadline = time.monotonic() + 5.0
                while mover.read_metrics().swaps_completed < n_swaps and time.monotonic() < deadline:
                    time.sleep(0.0005)
                elapsed = time.perf_counter() - t0
            finally:
                gc.sync_compute()
                mover.signal_stop()
                gc.sync_worker()
                gc.set_primary()
                worker_decl.graph.reset()
                enq_graph.reset()
                comp_graph.reset()

            m = mover.read_metrics()
            expert_mib = sum(tier.plane_row_bytes) / (1024 * 1024)
            bw_gbs = (m.bytes_copied / 1e9) / elapsed
            print(
                f"\n[DS V4.1 Full Geometry] E={E} H={HIDDEN} I={INTER} ({expert_mib:.3f} MiB/expert): "
                f"swaps={m.swaps_completed}/{n_swaps} bytes={m.bytes_copied / 1e6:.1f} MB "
                f"elapsed={elapsed * 1e3:.2f} ms effective_bw={bw_gbs:.1f} GB/s rcu_waits={m.rcu_waits} -> BIT-EXACT!"
            )
            assert m.swaps_completed == n_swaps
            assert m.bytes_copied == n_swaps * sum(tier.plane_row_bytes)
            assert int(tier.layer_epoch.item()) % 2 == 0
    finally:
        if mover is not None:
            mover.close()
        if tier is not None:
            for r in tier.regions:
                r.close()
        gc.close()


def test_sm103_8sm_1d_tma_mover_host_numa_vmm_bandwidth():
    """Verify sustained 8-SM 1D-TMA mover throughput over CU_MEM_LOCATION_TYPE_HOST_NUMA VMM on SM103."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (10, 3):
        pytest.skip("SM103 GB300 required")

    import struct
    from ignition.components.sm103_mover_kernel import prepare as prepare_mover

    dev = torch.device("cuda", 0)
    gc = _GreenCtxPair(0, 8)
    binding = prepare_mover(Path("/tmp/flare-test-kcache"))
    binding.configure_persistent_mover()

    class _Loc(C.Structure):
        _fields_ = [("type", C.c_int), ("id", C.c_int)]

    class _Prop(C.Structure):
        _fields_ = [
            ("type", C.c_int),
            ("requestedHandleTypes", C.c_int),
            ("location", _Loc),
            ("win32HandleMetaData", C.c_void_p),
            ("allocFlags", C.c_ubyte * 8),
        ]

    class _Access(C.Structure):
        _fields_ = [("location", _Loc), ("flags", C.c_int)]

    numa_id = C.c_int(0)
    gc._call("cuDeviceGetAttribute", C.byref(numa_id), C.c_int(134), C.c_int(0))
    host_numa = max(numa_id.value, 0)
    vmm_allocs = []

    def alloc_host_numa(nbytes: int):
        prop = _Prop(type=1, requestedHandleTypes=0, location=_Loc(type=3, id=host_numa))
        gran = C.c_size_t(0)
        gc._call("cuMemGetAllocationGranularity", C.byref(gran), C.byref(prop), C.c_int(0))
        padded = ((nbytes + gran.value - 1) // gran.value) * gran.value
        handle = C.c_uint64(0)
        ptr = C.c_uint64(0)
        gc._call("cuMemCreate", C.byref(handle), C.c_size_t(padded), C.byref(prop), C.c_uint64(0))
        gc._call("cuMemAddressReserve", C.byref(ptr), C.c_size_t(padded), C.c_size_t(gran.value), C.c_uint64(0), C.c_uint64(0))
        gc._call("cuMemMap", ptr, C.c_size_t(padded), C.c_size_t(0), handle, C.c_uint64(0))
        acc = (_Access * 2)(_Access(_Loc(1, 0), 3), _Access(_Loc(3, host_numa), 3))
        gc._call("cuMemSetAccess", ptr, C.c_size_t(padded), acc, C.c_size_t(2))
        vmm_allocs.append((ptr.value, handle, padded))
        return ptr.value

    E, S = 64, 16
    row_bytes = [12582912, 786432, 6291456, 393216]  # Exact DeepSeek V4.1 MXFP4 + UE8M0 plane strides (19.125 MiB)
    g_work = None
    g_enq = None
    try:
        host_ptrs = [alloc_host_numa(E * rb) for rb in row_bytes]
        ctrl_ptr = alloc_host_numa(4096)
        C.memset(ctrl_ptr, 0, 128)

        hbm_planes = [torch.empty((S * rb,), dtype=torch.uint8, device=dev) for rb in row_bytes]
        live_map = torch.tensor([i if i < S - 2 else -1 for i in range(E)], dtype=torch.int32, device=dev)
        slot_owner = torch.tensor([i if i < S - 2 else -1 for i in range(S)], dtype=torch.int32, device=dev)
        layer_epoch = torch.zeros((1,), dtype=torch.int32, device=dev)

        entry = struct.pack(
            "<4Q4Q4Q3Q4I",
            *host_ptrs,
            *(int(t.data_ptr()) for t in hbm_planes),
            *row_bytes,
            int(live_map.data_ptr()),
            int(slot_owner.data_ptr()),
            int(layer_epoch.data_ptr()),
            E, S, 0, 0,
        )
        descs = torch.tensor(list(entry) + [0] * (192 - len(entry)), dtype=torch.uint8, device=dev)
        q_hdr = torch.zeros((binding.queue_header_bytes(),), dtype=torch.uint8, device=dev)
        q_hdr[:12].copy_(torch.tensor(list(struct.pack("<III", 0, 0, 1023)), dtype=torch.uint8))
        q_ring = torch.zeros((1024 * 16,), dtype=torch.uint8, device=dev)

        cap_s = torch.cuda.Stream(device=dev)
        g_work = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_work, stream=cap_s):
                binding.launch_persistent_mover(descs, 1, q_hdr, q_ring, ctrl_ptr, ctrl_ptr + 8, 5, 8, int(cap_s.cuda_stream))
        g_work.instantiate()

        n_swaps = 48
        layers = torch.zeros((n_swaps,), dtype=torch.int32, device=dev)
        experts = torch.arange(S - 2, S - 2 + n_swaps, dtype=torch.int32, device=dev)
        victims = torch.full((n_swaps,), -1, dtype=torch.int32, device=dev)
        g_enq = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.stream(cap_s):
            with torch.cuda.graph(g_enq, stream=cap_s):
                binding.launch_enqueue_swap_batch(q_hdr, q_ring, layers, experts, victims)
        g_enq.instantiate()
        torch.cuda.synchronize()

        gc.launch_graph(int(g_work.raw_cuda_graph_exec()), worker=True)
        try:
            t0 = time.perf_counter()
            gc.launch_graph(int(g_enq.raw_cuda_graph_exec()), worker=False)
            gc.sync_compute()
            deadline = time.monotonic() + 5.0
            while C.c_uint64.from_address(ctrl_ptr + 8).value < n_swaps and time.monotonic() < deadline:
                pass
            elapsed = time.perf_counter() - t0
        finally:
            C.c_uint32.from_address(ctrl_ptr).value = 1
            gc.sync_worker()
            gc.set_primary()

        swaps = C.c_uint64.from_address(ctrl_ptr + 8).value
        copied = C.c_uint64.from_address(ctrl_ptr + 16).value
        bw_gbs = (copied / 1e9) / elapsed
        print(
            f"\n[HOST_NUMA VMM 8-SM 1D TMA] swaps={swaps}/{n_swaps} bytes={copied / 1e6:.1f} MB "
            f"elapsed={elapsed * 1e3:.2f} ms -> {bw_gbs:.1f} GB/s ({elapsed * 1e6 / swaps:.1f} us/expert)"
        )
        assert swaps == n_swaps
        assert copied == n_swaps * sum(row_bytes)
        assert bw_gbs >= 50.0, f"expected >= 50 GB/s over HOST_NUMA VMM on SM103, got {bw_gbs:.1f} GB/s"
    finally:
        if g_work is not None:
            g_work.reset()
        if g_enq is not None:
            g_enq.reset()
        gc.set_primary()
        for ptr, h, sz in vmm_allocs:
            gc._call("cuMemUnmap", C.c_uint64(ptr), C.c_size_t(sz))
            gc._call("cuMemRelease", h)
            gc._call("cuMemAddressFree", C.c_uint64(ptr), C.c_size_t(sz))
        gc.close()

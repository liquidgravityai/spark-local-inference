from dataclasses import replace
from time import monotonic, sleep

import pytest

from scripts.parallax.contracts import Demand, Expert, Hardware, Model, Observation
from scripts.parallax.live import LiveBakeOff, RoutingSource
from scripts.parallax.policy import Economic, NoMovement
from scripts.parallax.predictors import Recent
from ignition import planning
from ignition.recipes.glm53.recipe import RECIPE
from ignition.errors import UsageError

A, B = Expert(0, 0), Expert(0, 1)
MODEL = Model("fixture-package-fingerprint", "fixture", (A, B), (100, 100))


def response(sequence=0, frames=()):
    return dict(version=1, enabled=True, model_id="fixture", geometry=[1, 2],
                clock="native_process_monotonic_ns", demand_timing="host_dispatch_lower_bound",
                session="fixture-session", native_now_ns=10100, latest_sequence=sequence,
                dropped_total=0, invalid_total=0, frames=list(frames))


def frame(sequence=1, demand=11000, available=12000):
    return dict(sequence=sequence, phase="decode", batch_size=1, concurrency=2, graph=0,
        sample=dict(submission=sequence + 2, demand_lower_bound_ns=demand, available_ns=available,
            layers=[dict(layer=0, tokens=2, top_k=1, counts=[0, 2])]))


def test_source_uses_receipt_time_and_conservative_clock_bound():
    source = RoutingSource(MODEL, "fixture", response(), 100, 300)
    added = source.ingest(response(1, [frame()]), 3000, max_frames=10, max_demand_entries=100)
    assert added == (Observation(0, 1000, 3000, (Demand(B, 2),), "decode", 1, 2),)
    assert source.clock_uncertainty_ns == 200
    # Repeated HTTP reads cannot duplicate observations.
    assert source.ingest(response(1, [frame()]), 4000, max_frames=10, max_demand_entries=100) == ()


def test_metadata_only_attach_keeps_live_cursor_without_historical_frames():
    source = RoutingSource(MODEL, "fixture", response(7), 100, 300)
    assert source.cursor == 7 and source.observations == []
    added = source.ingest(response(8, [frame(8)]), 3000, max_frames=10, max_demand_entries=100)
    assert len(added) == 1 and added[0].seq == 0 and source.cursor == 8


def test_fixed_worker_catalog_and_future_target_survive_transport():
    from scripts.parallax.contracts import Context
    from scripts.parallax.shadow import ShadowPool

    pool = ShadowPool(model=MODEL)
    try:
        pool.register("recent", Recent(), budget_ns=5_000_000_000)
        now = pool.now_ns()
        context = Context(MODEL, (Observation(0, now - 1, now, (Demand(B, 2),)),), now, 1000, 4)
        # Identity text alone cannot admit an incompatible representation.
        with pytest.raises(ValueError, match="catalog"):
            pool.submit(replace(context, model=replace(MODEL, expert_bytes=(200, 200))))
        pool.submit(context)
        deadline, records = monotonic() + 5, []
        while not records and monotonic() < deadline:
            sleep(.01)
            records = pool.poll()
        assert len(records) == 1
        prediction = records[0][1]
        assert prediction.forecast.target == 4 and prediction.forecast.cutoff == 0
        assert prediction.forecast.scores == ((B, 2.),)
        assert prediction.available_ns == now
        assert prediction.completed_ns >= prediction.started_ns + prediction.compute_ns
    finally:
        pool.close()


class SlowEvaluationPolicy:
    def decide(self, *args):
        sleep(60)


def test_evaluation_budget_terminates_policy_and_keeps_snapshot_replayable():
    from scripts.parallax.contracts import Context, PredictionRecord
    from scripts.parallax.live import SnapshotEvaluator

    observation = Observation(0, 0, 10, (Demand(B, 2),))
    forecast = Recent().predict(Context(MODEL, (observation,), 10, 1000))
    snapshot = dict(model=MODEL, hardware=Hardware("test", 100, 200, 1, 0, 0, 100),
        initial=(A,), workload="synthetic-timeout", observations=(observation, Observation(1, 1000, 1010, (Demand(B, 2),))),
        records={"recent": (PredictionRecord(forecast, 10, 10, 20, 10, "modeled"),)},
        policies={"recent": {"static": NoMovement()}}, coalesced_points=0, workers={})
    evaluator = SnapshotEvaluator()
    try:
        evaluator.submit(snapshot, 5_000_000_000)
        expected = evaluator.result()
        slow = dict(snapshot, policies={"recent": {"slow": SlowEvaluationPolicy()}})
        evaluator.submit(slow, 200_000_000)
        with pytest.raises(TimeoutError, match="budget"):
            evaluator.result()
        assert evaluator.failed and evaluator.pool is None
        # A timeout never mutates the retained evidence or affects a fresh evaluator.
        replacement = SnapshotEvaluator()
        try:
            replacement.submit(snapshot, 5_000_000_000)
            assert replacement.result()["contenders"] == expected["contenders"]
        finally:
            replacement.close()
    finally:
        evaluator.close()


@pytest.mark.parametrize("change", [
    {"session": "restarted"}, {"invalid_total": 1}, {"dropped_total": 1}, {"enabled": False},
    {"model_id": "different-model"}, {"version": 999},
])
def test_source_rejects_session_loss_and_compatibility_changes(change):
    source = RoutingSource(MODEL, "fixture", response(), 100, 300)
    with pytest.raises(ValueError):
        source.ingest(response(1, [frame()]) | change, 3000, max_frames=10, max_demand_entries=100)


def test_retention_gap_is_not_hidden_by_renumbering_observations():
    source = RoutingSource(MODEL, "fixture", response(), 100, 300)
    with pytest.raises(ValueError, match="gap"):
        source.ingest(response(2, [frame(2)]), 3000, max_frames=10, max_demand_entries=100)
    assert not source.observations


def test_capture_omits_pre_attach_inflight_work_and_bounds_retention():
    source = RoutingSource(MODEL, "fixture", response(), 100, 300)
    rows = [frame(1, demand=10000), frame(2), frame(3, demand=12000, available=13000)]
    added = source.ingest(response(3, rows), 4000, max_frames=1, max_demand_entries=100)
    assert len(added) == 1 and source.cursor == 2
    assert source.skipped_inflight_at_attach == 1
    assert source.demand_entries == 1


def test_source_rejects_clock_reversal_before_mutating_retained_trace():
    source = RoutingSource(MODEL, "fixture", response(), 100, 300)
    source.ingest(response(1, [frame()]), 3000, max_frames=10, max_demand_entries=100)
    with pytest.raises(ValueError, match="clock/order"):
        source.ingest(response(2, [frame(2, demand=10500)]), 4000, max_frames=10, max_demand_entries=100)
    assert source.cursor == 1 and len(source.observations) == 1


def test_live_workers_feed_independent_policy_evaluations():
    hardware = Hardware("synthetic-test", 100, 200, 1, 10, 5, 1000)
    bakeoff = LiveBakeOff(MODEL, hardware, horizon_ns=1_000_000_000, initial=(A,), workload="SYNTHETIC-test")
    try:
        bakeoff.register("recent", Recent(), {"stay": NoMovement(), "move": Economic()}, budget_ns=5_000_000_000)
        now = bakeoff.pool.now_ns()
        first = Observation(0, now - 1, now, (Demand(B, 2),))
        history = [first]
        bakeoff.observe(history, (first,))
        deadline = monotonic() + 5
        while not bakeoff.records["recent"] and monotonic() < deadline:
            sleep(.01)
            bakeoff.poll()
        assert len(bakeoff.records["recent"]) == 1
        sleep(.001)
        now = bakeoff.pool.now_ns()
        history.append(Observation(1, now - 1, now, (Demand(B, 2),)))
        report = bakeoff.comparison(history)
        assert report["contenders"]["recent/stay"]["hits"] == 0
        assert report["contenders"]["recent/move"]["hits"] == 2
        assert report["contenders"]["recent/move"]["timing_sources"] == ["measured"]
        # A queued report cannot observe later collection or registration changes.
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        from scripts.parallax.live import compare_snapshot

        snapshot = bakeoff.comparison_snapshot(history)
        history.append(Observation(2, now + 1, now + 2, (Demand(A, 2),)))
        bakeoff.records["recent"].clear()
        bakeoff.policies["recent"].clear()
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as worker:
            asynchronous = worker.submit(compare_snapshot, snapshot).result(timeout=10)
        assert asynchronous["routing_frames"] == 2
        assert asynchronous["contenders"] == report["contenders"]
    finally:
        bakeoff.close()


def test_preparation_plan_records_capture_and_rejects_unsupported_modes():
    gpu = planning.Gpu(0, "fixture-GB300", "GPU-fixture", 288 << 30, 280 << 30, (10, 3))
    plain = planning.plan(RECIPE, RECIPE.checkpoints[0], (gpu,), selected=(0,), concurrency=1,
        reserve_bytes=0, speculation=None, parallax_routing=True, language_model_only=True)
    assert plain.parallax_routing and plain.to_dict()["parallax_routing"] is True
    assert replace(plain, speculation=RECIPE.speculation).speculation is RECIPE.speculation
    with pytest.raises(UsageError, match="one GPU"):
        replace(plain, gpus=(gpu, replace(gpu, index=1)))
    with pytest.raises(UsageError, match="GLM-Flash"):
        replace(plain, recipe=replace(RECIPE, name="different-recipe"))


def test_lil_preparation_retains_mtp_and_native_scales_for_runtime_selection(monkeypatch):
    from ignition.recipes import resolve
    from ignition.recipes.glm53 import bootstrap

    model = resolve("lil-glm-5.3-flash")
    gpu = planning.Gpu(0, "fixture-GB300", "GPU-fixture", 288 << 30, 280 << 30, (10, 3))
    plan = planning.fit(model.recipe, model.checkpoint, (gpu,), concurrency=32,
        reserve_bytes=0, prefill_chunk=8192, language_model_only=True, parallax_routing=True,
        speculation=replace(model.recipe.speculation, draft_tokens=5, token_capacity=192))
    assert plan.recipe.name == "lil_glm53"
    assert plan.expert_scale_compression is None
    assert plan.speculative_token_capacity == 192
    monkeypatch.setattr(bootstrap, "prepare_rank", lambda snapshot, config, **kwargs: config)
    config = model.recipe.prepare(None, plan, compile_forwards=False, kernel_cache=None, progress=None)
    assert config.parallax_routing and config.mtp and config.expand_expert_scales
    assert config.max_active == 32 and config.prefill_chunk_size == 8192
    assert config.language_model_only and config.speculative_token_capacity == 192

from dataclasses import FrozenInstanceError, replace
import json
import subprocess
import sys
from time import monotonic, sleep
from random import Random

import pytest

from scripts.parallax.__main__ import synthetic_trace
from scripts.parallax.contracts import Context, Demand, Expert, Hardware, Model, MovementPlan, Observation, PredictionRecord
from scripts.parallax.oracle import constrained_heuristic, idealized
from scripts.parallax.policy import Economic, NoMovement
from scripts.parallax.predictors import EWMA, Recent, Static, forecast
from scripts.parallax.replay import Contender, evaluate, generate, replay
from scripts.parallax.shadow import ShadowPool
from scripts.parallax.simulator import VirtualResidency
from scripts.parallax.trace import Trace, read_predictions, read_trace, write_predictions, write_trace

A, B, C = Expert(0, 0), Expert(0, 1), Expert(1, 0)
MODEL = Model("test-revision-format", "test-moe", (A, B, C), (100, 100, 200))
HARDWARE = Hardware("synthetic-test", 100, 400, 1, 10, 5, 1000)


def obs(seq, expert=A, *, demand_ns=None, delay=10, count=2):
    now = seq * 1000 if demand_ns is None else demand_ns
    return Observation(seq, now, now + delay, (Demand(expert, count),))


def trace(*observations):
    return Trace(MODEL, "synthetic-test", True, tuple(observations))


def record(observation, predictor=None, *, compute_ns=10, horizon_ns=900):
    context = Context(MODEL, (observation,), observation.available_ns, horizon_ns)
    f = (predictor or Recent()).predict(context)
    return PredictionRecord(f, context.available_ns, context.available_ns,
                            context.available_ns + compute_ns, compute_ns, "modeled")


def test_trace_roundtrip_and_version_rejection(tmp_path):
    path = tmp_path / "routing.jsonl"
    original = trace(obs(0), obs(1, B))
    write_trace(path, original)
    assert read_trace(path) == original
    path.write_text(path.read_text().replace('"version":1', '"version":99'))
    with pytest.raises(ValueError, match="version"):
        read_trace(path)


@pytest.mark.parametrize("observations", [
    (obs(0), obs(2)),
    (obs(0, delay=2000), obs(1)),
    (obs(0, demand_ns=2000), obs(1)),
    (obs(0, Expert(99, 0)),),
])
def test_trace_rejects_gaps_ordering_and_unknown_experts(observations):
    with pytest.raises(ValueError):
        trace(*observations)


def test_schema_rejects_noncausal_invalid_and_nonfinite_inputs():
    with pytest.raises(ValueError):
        Observation(0, 10, 9, ())
    with pytest.raises(ValueError):
        Demand(A, -1)
    with pytest.raises(ValueError):
        replace(HARDWARE, bandwidth_bytes_per_ns=float("nan"))
    context = Context(MODEL, (obs(0),), 10, 100)
    f = Recent().predict(context)
    for bad in (replace(f, cutoff=1), replace(f, model_identity="other"),
                replace(f, scores=((A, float("nan")),)), replace(f, kind="ambiguous"),
                replace(f, kind="probability", scores=((A, 2.),))):
        with pytest.raises(ValueError):
            bad.validate(context)


class Audit:
    def __init__(self):
        self.snapshots = []

    def predict(self, context):
        self.snapshots.append(context)
        assert all(o.available_ns <= context.available_ns for o in context.history)
        assert all(o.seq <= context.cutoff for o in context.history)
        return Recent().predict(context)


def test_predictors_get_identical_immutable_eligible_histories():
    t = trace(obs(0), obs(1, B), obs(2, A))
    a, b = Audit(), Audit()
    generate(t, Contender("a", a, NoMovement()), 900, history_limit=2)
    generate(t, Contender("b", b, NoMovement()), 900, history_limit=2)
    assert a.snapshots == b.snapshots
    assert [len(c.history) for c in a.snapshots] == [1, 2, 2]
    with pytest.raises(FrozenInstanceError):
        a.snapshots[0].history[0].seq = 99


def test_future_frame_horizons_preserve_cutoffs_and_legacy_forecasts(tmp_path):
    traffic = trace(*(obs(i, B) for i in range(6)))
    provider = Audit()
    records, _ = generate(traffic, Contender("recent", provider, NoMovement()), 2900, target_offset=3)
    assert all(r.forecast.target == r.forecast.cutoff + 3 for r in records)
    assert all(context.history[-1].seq == i for i, context in enumerate(provider.snapshots))
    assert [r.forecast.scores for r in records] == [((B, 2.),)] * 6
    summary, _ = evaluate(traffic, records, NoMovement(), HARDWARE)
    assert summary["scored_frames"] == 3
    assert summary["usable_lead_p50_ns"] == 2990
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, records)
    assert read_predictions(path) == records
    context = Context(MODEL, (obs(0),), 10, 900)
    legacy = replace(Recent().predict(context), version=1)
    legacy.validate(context)
    evaluate(traffic, (replace(record(obs(0)), forecast=legacy),), NoMovement(), HARDWARE)
    for invalid in (replace(legacy, target=3), replace(records[0].forecast, version=99),
                    replace(records[0].forecast, target=0), replace(records[0].forecast, target=True)):
        with pytest.raises(ValueError):
            invalid.validate(provider.snapshots[0])
    for offset in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="target offset"):
            Context(MODEL, (obs(0),), 10, 900, offset)


class Stateful:
    def __init__(self):
        self.calls = 0

    def predict(self, context):
        self.calls += 1
        return forecast(context, "stateful-test", {A: float(self.calls)})


def test_shared_provider_object_cannot_leak_state_between_contenders_or_replays():
    provider = Stateful()
    t = trace(obs(0), obs(1))
    contenders = (Contender("a", provider, Economic()), Contender("b", provider, NoMovement()))
    rows = replay(t, contenders, HARDWARE, horizon_ns=900)
    assert rows["a"]["records"] == rows["b"]["records"]
    assert provider.calls == 0


def test_mutable_history_and_scores_cannot_enter_shared_snapshots():
    with pytest.raises(ValueError, match="immutable"):
        replace(MODEL, experts=list(MODEL.experts))
    with pytest.raises(ValueError, match="immutable"):
        replace(obs(0), demands=list(obs(0).demands))
    with pytest.raises(ValueError, match="immutable"):
        Context(MODEL, [obs(0)], 10, 100)
    with pytest.raises(ValueError, match="immutable"):
        replace(record(obs(0)).forecast, scores=[(A, 1.)])


def test_independent_residency_and_policies_on_identical_predictions():
    t = trace(obs(0, B), obs(1, B), obs(2, B))
    contenders = (Contender("move", Recent(), Economic()), Contender("stay", Recent(), NoMovement()))
    rows = replay(t, contenders, HARDWARE, horizon_ns=900, initial=(A,))
    assert rows["move"]["records"] == rows["stay"]["records"]
    assert rows["move"]["summary"]["hits"] == 4
    assert rows["stay"]["summary"]["hits"] == 0
    reverse = replay(t, tuple(reversed(contenders)), HARDWARE, horizon_ns=900, initial=(A,))
    assert {k: v["summary"] for k, v in rows.items()} == {k: v["summary"] for k, v in reverse.items()}


class Broken:
    def predict(self, context):
        raise RuntimeError("fixture")


def test_failure_and_slow_predictor_do_not_change_other_contenders():
    t = trace(obs(0), obs(1), obs(2))
    rows = replay(t, (Contender("broken", Broken(), Economic()),
                     Contender("slow", Recent(), Economic(), compute_ns=2000),
                     Contender("fast", Recent(), Economic())), HARDWARE, horizon_ns=900)
    assert rows["broken"]["execution"]["failures"] == 3
    assert rows["slow"]["execution"]["dropped_observations"] == 1
    assert rows["fast"]["summary"]["prediction_records"] == 3
    assert rows["broken"]["summary"]["misses"] == 6


def test_late_accuracy_never_becomes_a_prefetch_hit():
    first, second = obs(0, B), obs(1, B)
    late = record(first, compute_ns=1500)
    summary, virtual = evaluate(trace(first, second), (late,), Economic(), HARDWARE)
    assert summary["late_predictions"] == 1
    assert summary["recall"] == 0
    assert summary["forecast_precision"] == summary["forecast_recall"] == 1
    assert summary["forecast_demand_coverage"] == 1 and summary["forecast_target_frames"] == 1
    assert summary["usable_lead_p50_ns"] == -510
    assert summary["hits"] == summary["movement_bytes"] == 0
    assert not virtual.resident


def test_equal_timestamp_forecast_cannot_evict_before_its_own_observation():
    first, second = obs(0, A, delay=0), obs(1, B)
    r = record(first, Static(((B, 3.),)), compute_ns=0)
    summary, _ = evaluate(trace(first, second), (r,), Economic(), HARDWARE, initial=(A,))
    assert summary["hits"] == 4
    assert summary["misses"] == 0


def test_expired_forecast_cannot_earn_coverage():
    first, second = obs(0, B), obs(1, B, demand_ns=3000)
    summary, _ = evaluate(trace(first, second), (record(first),), NoMovement(), HARDWARE)
    assert summary["expired_predictions"] == 1
    assert summary["demand_coverage"] == 0


def test_no_move_is_correct_for_insufficient_lead_or_uneconomic_demand():
    virtual = VirtualResidency(MODEL, HARDWARE)
    for r in (record(obs(0, B), horizon_ns=100),
              record(obs(0, B), Static(((B, .01),)))):
        assert not Economic().decide(r, virtual.snapshot(), MODEL, HARDWARE).promote
    ranking = replace(record(obs(0, B)), forecast=replace(record(obs(0, B)).forecast, kind="ranking"))
    assert not Economic().decide(ranking, virtual.snapshot(), MODEL, HARDWARE).promote


def test_uneconomic_candidates_do_not_scan_resident_victims():
    from scripts.parallax.contracts import Residency

    class ResidentSet(frozenset):
        def __iter__(self):
            raise AssertionError("an unaffordable transfer cannot justify inspecting evictions")

    state = Residency(0, ResidentSet((A,)), frozenset(), 100)
    # Ample lookahead reaches economics; even evicting a zero-value expert
    # cannot cover the movement cost. Large resident catalogs need no sort.
    r = record(obs(0, B), Static(((B, .01), (C, .01))), horizon_ns=1_000_000)
    assert Economic().decide(r, state, MODEL, HARDWARE) == NoMovement().decide(r, state, MODEL, HARDWARE)


def test_simulator_reserves_before_completion_and_serves_cold_without_dropping():
    virtual = VirtualResidency(MODEL, HARDWARE)
    virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000, (A,)))
    assert virtual.snapshot().reserved_bytes == 100
    virtual.demand(obs(0, demand_ns=114, delay=0))
    assert virtual.counters["misses"] == 2
    assert virtual.counters["late_prefetch_demands"] == 2
    virtual.demand(obs(1, demand_ns=115, delay=0))
    assert virtual.counters["hits"] == 2
    assert virtual.counters["successful_prefetches"] == 1


def test_reservation_accounting_survives_exchange_completion_and_failed_plan():
    hardware = replace(HARDWARE, hbm_bytes=200)
    virtual = VirtualResidency(MODEL, hardware, (A, B))
    virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000, (C,), (A, B)))
    assert virtual.snapshot().reserved_bytes == 200
    assert not virtual.resident
    before = virtual.snapshot()
    with pytest.raises(ValueError, match="HBM"):
        virtual.apply(MovementPlan(MODEL.identity, before.generation, 0, 1000, (A,)))
    assert virtual.snapshot() == before
    virtual.advance(215)
    assert virtual.snapshot().reserved_bytes == 200 and virtual.resident == {C}
    virtual.apply(MovementPlan(MODEL.identity, virtual.generation, 215, 1000))
    assert virtual.snapshot().reserved_bytes == 200


def test_plan_rejection_is_transactional_and_pending_storage_counts():
    virtual = VirtualResidency(MODEL, HARDWARE, (A,))
    initial = virtual.snapshot()
    with pytest.raises(ValueError, match="HBM"):
        virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000, (C,), (A,)))
    assert virtual.snapshot() == initial
    assert virtual.history == []
    virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000, (B,), (A,)))
    after = virtual.snapshot()
    with pytest.raises(ValueError, match="HBM"):
        virtual.apply(MovementPlan(MODEL.identity, after.generation, 0, 1000, (A,)))
    assert virtual.snapshot() == after
    with pytest.raises(ValueError, match="stale"):
        virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000))


def test_copy_engine_serialization_and_plan_deadlines():
    hw = replace(HARDWARE, hbm_bytes=200)
    virtual = VirtualResidency(MODEL, hw)
    with pytest.raises(ValueError, match="expiration"):
        virtual.apply(MovementPlan(MODEL.identity, 0, 0, 200, (A, B)))
    assert virtual.snapshot().pending == frozenset()
    virtual.apply(MovementPlan(MODEL.identity, 0, 0, 230, (A, B)))
    assert [(t.start_ns, t.complete_ns) for t in virtual.pending] == [(0, 115), (115, 230)]


def test_evicted_unused_prefetch_is_wasted_but_end_of_trace_is_censored():
    virtual = VirtualResidency(MODEL, HARDWARE)
    virtual.apply(MovementPlan(MODEL.identity, 0, 0, 1000, (A,)))
    virtual.advance(200)
    assert virtual.summary()["unused_at_end"] == 1
    assert virtual.summary()["wasted_prefetches"] == 0
    virtual.apply(MovementPlan(MODEL.identity, virtual.generation, 200, 1000, (B,), (A,)))
    virtual.demand(obs(0, A, demand_ns=201))
    assert virtual.counters["wasted_prefetches"] == 1
    assert virtual.counters["eviction_reuses"] == 1


def test_backing_capacity_required_and_initial_state_validated():
    with pytest.raises(ValueError, match="backing"):
        VirtualResidency(MODEL, replace(HARDWARE, backing_bytes=399))
    with pytest.raises(ValueError):
        VirtualResidency(MODEL, HARDWARE, (A, B))


def test_oracle_exact_idealized_bound_and_constrained_heuristic_labels():
    t = trace(obs(0, B), obs(1, B), obs(2, B))
    ideal = idealized(t, HARDWARE, (A,))
    constrained = constrained_heuristic(t, HARDWARE, (A,))
    assert ideal["cold_cost_ns"] == 0
    assert ideal["opportunity_ns"] == 6000
    assert 0 < constrained["summary"]["estimated_net_route_cost_saved_ns"] <= ideal["opportunity_ns"]
    assert "NOT_bound" in constrained["kind"]


def test_oracle_variable_sizes_does_not_assume_equal_slots():
    t = trace(Observation(0, 0, 1, (Demand(A, 2), Demand(B, 2), Demand(C, 3))))
    result = idealized(t, replace(HARDWARE, hbm_bytes=200))
    assert result["cold_cost_ns"] == 3000  # A+B beats the larger expert C.


def test_deterministic_replay_separates_wall_timings_and_cost_estimates():
    t = synthetic_trace()
    hw = Hardware("synthetic", 2000, 4000, 1, 100, 50, 2000)
    contender = Contender("ewma", EWMA(), Economic(), compute_ns=100)
    a = replay(t, (contender,), hw, horizon_ns=10000)["ewma"]
    b = replay(t, (contender,), hw, horizon_ns=10000)["ewma"]
    assert a["summary"] == b["summary"]
    assert a["records"] == b["records"]
    assert a["execution"]["measured_predict_and_validate_ns"]
    assert a["summary"]["confidence_interval"] is None


def test_record_cutoff_validation_prevents_future_information_claims():
    t = trace(obs(0), obs(1))
    r = record(t.observations[0])
    with pytest.raises(ValueError, match="cutoff"):
        evaluate(t, (replace(r, available_ns=0, started_ns=0),), NoMovement(), HARDWARE)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate(t, (r, r), NoMovement(), HARDWARE)


def test_prediction_capture_roundtrip_reproduces_evaluation(tmp_path):
    t = trace(obs(0), obs(1))
    records, _ = generate(t, Contender("a", Recent(), Economic()), 900)
    path = tmp_path / "predictions.jsonl"
    write_predictions(path, records)
    loaded = read_predictions(path)
    assert loaded == records
    before, _ = evaluate(t, records, Economic(), HARDWARE)
    after, _ = evaluate(t, loaded, Economic(), HARDWARE)
    assert before == after


def test_estimated_savings_never_exceed_matching_idealized_bound_on_small_traces():
    rng = Random(715)
    for _ in range(40):
        observations = tuple(obs(i, rng.choice(MODEL.experts), count=rng.randrange(1, 8)) for i in range(10))
        t = trace(*observations)
        hw = replace(HARDWARE, hbm_bytes=rng.choice((0, 100, 200, 400)))
        initial = (A,) if hw.hbm_bytes else ()
        rows = replay(t, (Contender("r", Recent(), Economic(), compute_ns=10),
                         Contender("e", EWMA(), Economic(), compute_ns=10)), hw, horizon_ns=900, initial=initial)
        bound = idealized(t, hw, initial)["opportunity_ns"]
        for row in rows.values():
            assert row["summary"]["estimated_net_route_cost_saved_ns"] <= bound
            assert row["summary"]["hits"] + row["summary"]["misses"] == sum(d.count for o in observations for d in o.demands)


def test_demand_weighting_and_layer_drilldown():
    first = obs(0, A)
    second = Observation(1, 1000, 1010, (Demand(A, 3), Demand(C, 1)), "verify", 4, 2)
    summary, _ = evaluate(trace(first, second), (record(first),), NoMovement(), HARDWARE, top_k=1)
    assert summary["weighted_correct"] == 3
    assert summary["demand_coverage"] == .5  # Includes uncovered first-frame demand.
    layer = next(row for row in summary["strata"] if row["layer"] == 1)
    assert layer["phase"] == "verify" and layer["covered_demand"] == 0


class Slow:
    def predict(self, context):
        sleep(10)
        return Recent().predict(context)


def test_shadow_workers_isolate_hang_failure_and_hot_swap():
    with ShadowPool(max_workers=3) as pool:
        pool.register("fast", Recent(), budget_ns=5_000_000_000)
        pool.register("slow", Slow(), budget_ns=300_000_000)
        pool.register("broken", Broken(), budget_ns=5_000_000_000)
        context = Context(MODEL, (obs(0, delay=0),), 0, 10000000)
        pool.submit(context)
        pool.submit(context)
        received = []
        deadline = monotonic() + 6
        while monotonic() < deadline:
            received.extend(pool.poll())
            if (received and pool.workers["slow"].counters["timeouts"]
                    and pool.workers["broken"].counters["failures"]):
                break
            sleep(.01)
        assert [name for name, _ in received] == ["fast"]
        r = received[0][1]
        assert r.timing_source == "measured"
        assert r.available_ns <= r.started_ns <= r.completed_ns
        assert pool.workers["slow"].counters["timeouts"] == 1
        assert pool.workers["fast"].counters["dropped_busy"] == 1
        pool.remove("slow")
        pool.register("slow", EWMA())
        assert not pool.workers["slow"].disabled


def test_shadow_resource_limits_and_future_snapshot_rejection():
    with ShadowPool(max_workers=1, history_limit=1) as pool:
        pool.register("one", Recent())
        with pytest.raises(ValueError):
            pool.register("two", Recent())
        with pytest.raises(ValueError):
            pool.submit(Context(MODEL, (obs(0), obs(1)), 1010, 100))
        with pytest.raises(ValueError):
            pool.submit(Context(MODEL, (obs(0),), pool.now_ns() + 10**12, 100))


def test_cli_produces_reproducible_labeled_artifact(tmp_path):
    output, fixture = tmp_path / "results.json", tmp_path / "trace.jsonl"
    result = subprocess.run([sys.executable, "-m", "scripts.parallax", "--synthetic",
                             "--output", str(output), "--write-synthetic", str(fixture)],
                            capture_output=True, text=True, check=True)
    report = json.loads(output.read_text())
    assert report["synthetic"] is True
    assert "NOT measured serving speedups" in result.stdout
    assert report["contenders"]["static-residency"]["summary"]["estimated_net_route_cost_saved_ns"] == 0
    assert read_trace(fixture) == synthetic_trace()

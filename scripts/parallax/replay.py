"""Causal forecast generation and fixed-traffic evaluation owned by the observatory."""

from dataclasses import dataclass
from copy import deepcopy
from math import ceil
from time import perf_counter_ns

from .contracts import Context, PredictionRecord
from .policy import transfer_ns
from .simulator import VirtualResidency


@dataclass(frozen=True)
class Contender:
    name: str
    predictor: object
    policy: object
    compute_ns: int = 0
    queue_ns: int = 0

    def __post_init__(self):
        if not self.name or any(type(v) is not int or v < 0 for v in (self.compute_ns, self.queue_ns)):
            raise ValueError("named contender and nonnegative modeled timing required")


def generate(trace, contender, horizon_ns, history_limit=64, target_offset=1):
    """Deterministic modeled release times; real CPU timings are separate evidence.

    One outstanding prediction per contender, with busy observations dropped.
    History still advances through the common eligible observation stream.
    """
    if history_limit < 1:
        raise ValueError("history limit must be positive")
    records, timings, failures, dropped = [], [], 0, 0
    busy_until = 0
    for i, obs in enumerate(trace.observations):
        if obs.available_ns < busy_until:
            dropped += 1
            continue
        context = Context(trace.model, trace.observations[max(0, i + 1 - history_limit):i + 1],
                          obs.available_ns, horizon_ns, target_offset)
        start = perf_counter_ns()
        try:
            f = contender.predictor.predict(context)
            f.validate(context)
            begun = obs.available_ns + contender.queue_ns
            busy_until = begun + contender.compute_ns
            records.append(PredictionRecord(f, obs.available_ns, begun, busy_until,
                                            contender.compute_ns, "modeled"))
        except Exception:
            failures += 1
        timings.append(perf_counter_ns() - start)
    return tuple(records), dict(failures=failures, dropped_observations=dropped,
                               measured_predict_and_validate_ns=timings)


def percentile(values, q):
    return sorted(values)[max(0, ceil(q * len(values)) - 1)] if values else None


def evaluate(trace, records, policy, hardware, *, initial=(), top_k=8):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    policy = deepcopy(policy)
    # Recorded predictions are untrusted inputs, including those from live workers.
    by_seq = {o.seq: o for o in trace.observations}
    seen_targets = set()
    for record in records:
        f = record.forecast
        cutoff = by_seq.get(f.cutoff)
        if cutoff is None or cutoff.available_ns > record.available_ns or f.target in seen_targets:
            raise ValueError("invalid cutoff or duplicate target record")
        f.validate(Context(trace.model, (cutoff,), record.available_ns,
                           f.expected_demand_ns - record.available_ns, f.target - f.cutoff))
        seen_targets.add(f.target)
    virtual = VirtualResidency(trace.model, hardware, initial)
    baseline = VirtualResidency(trace.model, hardware, initial)
    forecasts = {r.forecast.target: r for r in records}
    counts = dict(predicted=0, selected=0, correct=0, weighted_correct=0, demand=0,
                  scored_frames=0, uncovered_frames=0, late_predictions=0,
                  expired_predictions=0, actionable_predictions=0, rejected_plans=0,
                  policy_failures=0)
    quality = dict(target_frames=0, predicted=0, selected=0, correct=0, demand=0, covered_demand=0)
    strata, leads = {}, []
    # At equal timestamps, the cutoff's demand precedes the forecast produced
    # from it; that forecast may precede a later sequence's demand. A generic
    # "predictions first" tie break would let new evictions rewrite past hits.
    timeline = [(o.demand_ns, 2 * o.seq, 1, i, o) for i, o in enumerate(trace.observations)]
    timeline += [(r.completed_ns, 2 * r.forecast.cutoff + 1, 0, i, r) for i, r in enumerate(records)
                 if trace.observations and r.completed_ns <= trace.observations[-1].demand_ns]
    consumed = set()
    for now, _, kind, _, value in sorted(timeline, key=lambda row: row[:4]):
        virtual.advance(now)
        if kind == 0:
            if value.forecast.target in consumed or now >= value.forecast.expires_ns:
                continue
            try:
                plan = policy.decide(value, virtual.snapshot(), trace.model, hardware)
            except Exception:
                counts["policy_failures"] += 1
                continue
            try:
                virtual.apply(plan)
            except (ValueError, TypeError, AttributeError):
                counts["rejected_plans"] += 1
            continue
        observation = value
        consumed.add(observation.seq)
        virtual.demand(observation)
        baseline.demand(observation)
        actual = {d.expert: d.count for d in observation.demands}
        counts["selected"] += len(actual)
        counts["demand"] += sum(actual.values())
        record = forecasts.get(observation.seq)
        predicted = set()
        horizon = None
        if record is not None:
            f = record.forecast
            ranked = {e for e, score in sorted(f.scores, key=lambda p: (-p[1], p[0]))[:top_k] if score > 0}
            matched = ranked & actual.keys()
            quality["target_frames"] += 1
            quality["predicted"] += len(ranked)
            quality["selected"] += len(actual)
            quality["correct"] += len(matched)
            quality["demand"] += sum(actual.values())
            quality["covered_demand"] += sum(actual[e] for e in matched)
            horizon = f.expected_demand_ns - record.available_ns
            lead = now - record.completed_ns
            leads.append(lead)
            late, expired = lead < 0, now >= f.expires_ns
            counts["late_predictions"] += late
            counts["expired_predictions"] += expired
            if not late and not expired:
                predicted = ranked
                counts["scored_frames"] += 1
                # Timing feasibility only; actual queue/capacity feasibility is in residency results.
                counts["actionable_predictions"] += bool(predicted) and any(
                    transfer_ns(trace.model, hardware, e) + hardware.synchronization_ns <= lead
                    for e in predicted & actual.keys())
            else:
                counts["uncovered_frames"] += 1
        else:
            counts["uncovered_frames"] += 1
        correct = predicted & actual.keys()
        counts["predicted"] += len(predicted)
        counts["correct"] += len(correct)
        counts["weighted_correct"] += sum(actual[e] for e in correct)
        for layer in {e.layer for e in actual}:
            key = (layer, observation.phase, observation.batch_size, observation.concurrency, horizon)
            row = strata.setdefault(key, dict(frames=0, demand=0, covered_demand=0))
            row["frames"] += 1
            row["demand"] += sum(n for e, n in actual.items() if e.layer == layer)
            row["covered_demand"] += sum(actual[e] for e in correct if e.layer == layer)
    summary = virtual.summary()
    modeled_cost = summary["cold_cost_ns"] + summary["transfer_ns"] + summary["synchronization_ns"]
    # Conservatively charge all producer CPU/transport delay within this trace.
    overhead = sum(r.completed_ns - r.available_ns for r in records)
    summary.update(counts)
    # Keep provider computation separate from delivery and observation delay.
    # Host-dispatch demand timestamps make observation age an upper bound.
    stages = dict(
        prediction_queue=[r.started_ns - r.available_ns for r in records],
        prediction_publication=[r.completed_ns - r.started_ns - r.compute_ns for r in records],
        prediction_end_to_end=[r.completed_ns - r.available_ns for r in records],
        observation_age=[r.available_ns - by_seq[r.forecast.cutoff].demand_ns for r in records])
    summary.update({f"{stage}_p{p}_ns": percentile(values, p / 100)
                    for stage, values in stages.items() for p in (50, 95, 99)})
    summary.update(
        forecast_target_frames=quality["target_frames"],
        forecast_precision=quality["correct"] / quality["predicted"] if quality["predicted"] else None,
        forecast_recall=quality["correct"] / quality["selected"] if quality["selected"] else None,
        forecast_demand_coverage=quality["covered_demand"] / quality["demand"] if quality["demand"] else None,
        precision=counts["correct"] / counts["predicted"] if counts["predicted"] else None,
        recall=counts["correct"] / counts["selected"] if counts["selected"] else None,
        demand_coverage=counts["weighted_correct"] / counts["demand"] if counts["demand"] else None,
        hbm_hit_rate=summary["hits"] / counts["demand"] if counts["demand"] else None,
        usable_lead_p50_ns=percentile(leads, .5), usable_lead_p95_ns=percentile(leads, .95),
        actionable_rate=counts["actionable_predictions"] / len(trace.observations) if trace.observations else None,
        prediction_compute_p50_ns=percentile([r.compute_ns for r in records], .5),
        prediction_compute_p95_ns=percentile([r.compute_ns for r in records], .95),
        prediction_compute_p99_ns=percentile([r.compute_ns for r in records], .99),
        prediction_overhead_proxy_ns=overhead,
        estimated_net_route_cost_saved_ns=baseline.counters["cold_cost_ns"] - modeled_cost - overhead,
        prediction_records=len(records), routing_frames=len(trace.observations),
        confidence_interval=None, confidence_note="No independent workload repeats; frames are correlated.",
        timing_sources=sorted({r.timing_source for r in records}),
        strata=[dict(layer=k[0], phase=k[1], batch_size=k[2], concurrency=k[3], horizon_ns=k[4], **v)
                for k, v in strata.items()],
    )
    return summary, virtual


def replay(trace, contenders, hardware, *, horizon_ns, initial=(), top_k=8, history_limit=64, target_offset=1):
    if len({c.name for c in contenders}) != len(contenders):
        raise ValueError("contender names must be unique")
    result = {}
    for contender in contenders:
        contender = deepcopy(contender)
        records, execution = generate(trace, contender, horizon_ns, history_limit, target_offset)
        summary, _ = evaluate(trace, records, contender.policy, hardware, initial=initial, top_k=top_k)
        result[contender.name] = dict(summary=summary, execution=execution, records=records)
    return result

"""Offline-only clairvoyant controls, never registered as live predictors."""

from .contracts import Context, PredictionRecord
from .policy import Economic, NoMovement
from .predictors import forecast
from .replay import evaluate


def idealized(trace, hardware, initial=()):
    """Exact small-catalog upper bound for the additive route-cost objective.

    Each routing frame may choose any HBM subset at zero transfer/sync cost.
    This is not a serving-throughput bound or a cost-constrained optimizer.
    """
    model = trace.model
    if len(model.experts) > 18:
        raise ValueError("exact idealized enumeration is limited to 18 experts")
    feasible = []
    for mask in range(1 << len(model.experts)):
        selected = frozenset(e for i, e in enumerate(model.experts) if mask & (1 << i))
        if sum(model.size(e) for e in selected) <= hardware.hbm_bytes:
            feasible.append(selected)
    cost, baseline = 0, 0
    for obs in trace.observations:
        cost += min(sum(d.count for d in obs.demands if d.expert not in subset) for subset in feasible)
        baseline += sum(d.count for d in obs.demands if d.expert not in initial)
    return dict(kind="exact_idealized_free_movement_route_cost_bound", cold_cost_ns=cost * hardware.cold_cost_ns,
                opportunity_ns=(baseline - cost) * hardware.cold_cost_ns)


def constrained_heuristic(trace, hardware, initial=(), top_k=8):
    """Perfect next-frame demand, real modeled deadlines/capacity; NOT an upper bound.

    Uses the same conservative policy and simulator as deployable contenders.
    A larger lookahead or exact constrained solver may make better decisions.
    """
    records = []
    for current, target in zip(trace.observations, trace.observations[1:]):
        if target.demand_ns <= current.available_ns:
            continue
        context = Context(trace.model, (current,), current.available_ns, target.demand_ns - current.available_ns)
        f = forecast(context, "OFFLINE_ORACLE", {d.expert: float(d.count) for d in target.demands})
        records.append(PredictionRecord(f, current.available_ns, current.available_ns,
                                        current.available_ns, 0, "modeled"))
    summary, _ = evaluate(trace, records, Economic(), hardware, initial=initial, top_k=top_k)
    static, _ = evaluate(trace, (), NoMovement(), hardware, initial=initial, top_k=top_k)
    # Clairvoyance can always choose the no-movement control for the full trace.
    best = max((summary, static), key=lambda row: row["estimated_net_route_cost_saved_ns"])
    return dict(kind="cost_constrained_next_frame_clairvoyant_heuristic_NOT_bound", summary=best)

"""A conservative expected-count policy and a static/no-movement control."""

from dataclasses import dataclass
from math import ceil

from .contracts import MovementPlan


def transfer_ns(model, hardware, expert):
    return ceil(model.size(expert) / hardware.bandwidth_bytes_per_ns) + hardware.transfer_latency_ns


@dataclass(frozen=True)
class NoMovement:
    def decide(self, record, state, model, hardware):
        return MovementPlan(model.identity, state.generation, record.completed_ns,
                            record.forecast.expires_ns)


@dataclass(frozen=True)
class Economic:
    """One exchange per forecast; rank/probability forecasts need another policy.

    Charges full transfer occupancy conservatively, including an eviction's
    predicted demand. Pending work suppresses replacements to avoid churn.
    """

    margin_ns: int = 0

    def __post_init__(self):
        if type(self.margin_ns) is not int or self.margin_ns < 0:
            raise ValueError("margin must be nonnegative integer nanoseconds")

    def decide(self, record, state, model, hardware):
        empty = NoMovement().decide(record, state, model, hardware)
        f = record.forecast
        if f.kind != "expected_count" or state.pending or record.completed_ns >= f.expires_ns:
            return empty
        scores = dict(f.scores)
        candidates = sorted((e for e in scores if e not in state.resident),
                            key=lambda e: (-scores[e], e))
        victim_order = None
        for expert in candidates:
            duration = transfer_ns(model, hardware, expert) + hardware.synchronization_ns
            if record.completed_ns + duration > min(f.expected_demand_ns, f.expires_ns):
                continue
            # Eviction can only reduce benefit. Reject an unaffordable transfer
            # before inspecting resident victims, even with ample lookahead.
            if scores[expert] * hardware.cold_cost_ns <= duration + self.margin_ns:
                continue
            victims, free = [], hardware.hbm_bytes - state.reserved_bytes
            if free < model.size(expert) and victim_order is None:
                victim_order = sorted(state.resident, key=lambda e: (scores.get(e, 0), e))
            for victim in victim_order or ():
                if free >= model.size(expert):
                    break
                victims.append(victim)
                free += model.size(victim)
            benefit = (scores[expert] - sum(scores.get(e, 0) for e in victims)) * hardware.cold_cost_ns
            if free >= model.size(expert) and benefit > duration + self.margin_ns:
                return MovementPlan(model.identity, state.generation, record.completed_ns,
                                    f.expires_ns, (expert,), tuple(victims))
        return empty

"""Stateless-over-window baselines; each invocation sees only eligible history."""

from dataclasses import dataclass

from .contracts import Forecast


def forecast(context, provider, scores, *, kind="expected_count", config=()):
    return Forecast(context.model.identity, provider, "1", config, context.cutoff,
                    context.cutoff + context.target_offset, context.available_ns + context.horizon_ns,
                    context.available_ns + 2 * context.horizon_ns, kind,
                    tuple(sorted(scores.items())))


@dataclass(frozen=True)
class Static:
    """Profile supplied separately, never fitted on the evaluation trace."""

    profile: tuple
    profile_identity: str = "explicit-profile"

    def predict(self, context):
        return forecast(context, "static", dict(self.profile),
                        config=(("profile", self.profile_identity),))


@dataclass(frozen=True)
class Recent:
    def predict(self, context):
        return forecast(context, "recent", {d.expert: float(d.count) for d in context.history[-1].demands})


@dataclass(frozen=True)
class EWMA:
    alpha: float = 0.5

    def __post_init__(self):
        if not 0 < self.alpha <= 1:
            raise ValueError("EWMA alpha must be in (0, 1]")

    def predict(self, context):
        scores = {}
        for observation in context.history:
            scores = {e: v * (1 - self.alpha) for e, v in scores.items()}
            for demand in observation.demands:
                scores[demand.expert] = scores.get(demand.expert, 0) + self.alpha * demand.count
        return forecast(context, "ewma", scores, config=(("alpha", str(self.alpha)),))

"""Versioned immutable wire values and independent research interfaces.

Times are integer nanoseconds in one trace-relative monotonic clock domain.
Sequence numbers identify executed routing frames, not committed output tokens.
"""

from dataclasses import dataclass
from bisect import bisect_left
from math import isfinite
from typing import Protocol

VERSION = 1
FORECAST_VERSION = 2


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True, order=True)
class Expert:
    layer: int
    expert: int

    def __post_init__(self):
        integer(self.layer, "layer")
        integer(self.expert, "expert")


@dataclass(frozen=True)
class Model:
    identity: str  # Checkpoint revision + quantization/representation fingerprint.
    architecture: str
    experts: tuple[Expert, ...]
    expert_bytes: tuple[int, ...]

    def __post_init__(self):
        if (type(self.experts) is not tuple or type(self.expert_bytes) is not tuple
                or any(not isinstance(e, Expert) for e in self.experts)):
            raise ValueError("model catalog must contain immutable expert and size tuples")
        if (not isinstance(self.identity, str) or not self.identity
                or not isinstance(self.architecture, str) or not self.architecture or not self.experts):
            raise ValueError("model identity, architecture and expert catalog are required")
        if len(set(self.experts)) != len(self.experts) or len(self.experts) != len(self.expert_bytes):
            raise ValueError("expert catalog must be unique and sizes must match")
        if self.experts != tuple(sorted(self.experts)):
            raise ValueError("expert catalog must be in canonical layer/expert order")
        for size in self.expert_bytes:
            integer(size, "expert bytes", 1)

    def size(self, expert):
        index = bisect_left(self.experts, expert)
        if index == len(self.experts) or self.experts[index] != expert:
            raise ValueError("unknown canonical expert")
        return self.expert_bytes[index]


@dataclass(frozen=True)
class Demand:
    expert: Expert
    count: int

    def __post_init__(self):
        if not isinstance(self.expert, Expert):
            raise ValueError("canonical expert identity required")
        integer(self.count, "demand count", 1)


@dataclass(frozen=True)
class Observation:
    seq: int
    demand_ns: int
    available_ns: int
    demands: tuple[Demand, ...]
    phase: str = "decode"
    batch_size: int = 1
    concurrency: int = 1

    def __post_init__(self):
        if type(self.demands) is not tuple or any(not isinstance(d, Demand) for d in self.demands):
            raise ValueError("demands must be an immutable tuple")
        integer(self.seq, "sequence")
        integer(self.demand_ns, "demand time")
        integer(self.available_ns, "availability time", self.demand_ns)
        integer(self.batch_size, "batch size", 1)
        integer(self.concurrency, "concurrency", 1)
        if self.phase not in ("decode", "prefill", "draft", "verify", "mixed"):
            raise ValueError("unknown execution phase")
        if len({d.expert for d in self.demands}) != len(self.demands):
            raise ValueError("aggregate duplicate expert demands within each frame")


@dataclass(frozen=True)
class Context:
    model: Model
    history: tuple[Observation, ...]
    available_ns: int
    horizon_ns: int
    target_offset: int = 1  # Future routing frames, independently of expected wall time.

    def __post_init__(self):
        if type(self.history) is not tuple or any(not isinstance(o, Observation) for o in self.history):
            raise ValueError("history must be an immutable observation tuple")
        integer(self.available_ns, "prediction point")
        integer(self.horizon_ns, "horizon", 1)
        integer(self.target_offset, "target offset", 1)
        if not self.history or any(o.available_ns > self.available_ns for o in self.history):
            raise ValueError("history must be nonempty and available at the cutoff")
        if any(a.seq >= b.seq or a.available_ns > b.available_ns for a, b in zip(self.history, self.history[1:])):
            raise ValueError("history must preserve causal sequence and availability order")

    @property
    def cutoff(self):
        return self.history[-1].seq


@dataclass(frozen=True)
class Forecast:
    model_identity: str
    provider: str
    provider_version: str
    config: tuple[tuple[str, str], ...]
    cutoff: int
    target: int
    expected_demand_ns: int
    expires_ns: int
    kind: str  # expected_count, probability, or ranking (larger score is better).
    scores: tuple[tuple[Expert, float], ...]
    confidence: float | None = None
    version: int = FORECAST_VERSION

    def __post_init__(self):
        if (type(self.scores) is not tuple or type(self.config) is not tuple
                or any(type(pair) is not tuple or len(pair) != 2 for pair in self.scores + self.config)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in self.config)):
            raise ValueError("forecast scores and configuration must be immutable pairs")

    def validate(self, context):
        if type(self.version) is not int or self.version not in (1, FORECAST_VERSION) or self.model_identity != context.model.identity:
            raise ValueError("incompatible forecast schema or model")
        if (not isinstance(self.provider, str) or not self.provider
                or not isinstance(self.provider_version, str) or not self.provider_version):
            raise ValueError("provider identity and version required")
        integer(self.cutoff, "cutoff")
        integer(self.target, "target")
        if self.version == 1 and self.target != self.cutoff + 1:
            raise ValueError("v1 forecasts target the next routing frame at this cutoff")
        if self.cutoff != context.cutoff or self.target != context.cutoff + context.target_offset:
            raise ValueError("forecast target must match the declared future frame offset")
        if not context.available_ns < self.expected_demand_ns <= self.expires_ns:
            raise ValueError("invalid expected demand or expiry")
        integer(self.expected_demand_ns, "expected demand")
        integer(self.expires_ns, "expiry")
        if self.kind not in ("expected_count", "probability", "ranking"):
            raise ValueError("unknown forecast meaning")
        if len(dict(self.scores)) != len(self.scores):
            raise ValueError("duplicate forecast expert")
        catalog = frozenset(context.model.experts)
        for expert, score in self.scores:
            if expert not in catalog or not isfinite(score) or score < 0:
                raise ValueError("invalid forecast expert or score")
            if self.kind == "probability" and score > 1:
                raise ValueError("probabilities must be <= 1")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("invalid confidence")


@dataclass(frozen=True)
class PredictionRecord:
    forecast: Forecast
    available_ns: int
    started_ns: int
    completed_ns: int  # Consumer receipt/publication, including transport.
    compute_ns: int
    timing_source: str  # measured or modeled

    def __post_init__(self):
        for name in ("available_ns", "started_ns", "completed_ns", "compute_ns"):
            integer(getattr(self, name), name)
        if not self.available_ns <= self.started_ns <= self.completed_ns:
            raise ValueError("prediction times must be causal")
        if self.compute_ns > self.completed_ns - self.started_ns:
            raise ValueError("compute duration exceeds publication interval")
        if self.timing_source not in ("measured", "modeled"):
            raise ValueError("unknown timing source")


@dataclass(frozen=True)
class Hardware:
    identity: str
    hbm_bytes: int  # Expert-only reservation after KV, graphs, and workspace.
    backing_bytes: int  # Immutable complete backing catalog in v1.
    bandwidth_bytes_per_ns: float
    transfer_latency_ns: int
    synchronization_ns: int
    cold_cost_ns: int  # Additive route-cost proxy, NEVER end-to-end latency.

    def __post_init__(self):
        for name in ("hbm_bytes", "backing_bytes", "transfer_latency_ns", "synchronization_ns", "cold_cost_ns"):
            integer(getattr(self, name), name)
        if not self.identity or not isfinite(self.bandwidth_bytes_per_ns) or self.bandwidth_bytes_per_ns <= 0:
            raise ValueError("hardware identity and positive finite bandwidth required")


@dataclass(frozen=True)
class Residency:
    generation: int
    resident: frozenset[Expert]
    pending: frozenset[Expert]
    reserved_bytes: int


@dataclass(frozen=True)
class MovementPlan:
    model_identity: str
    generation: int
    created_ns: int
    expires_ns: int
    promote: tuple[Expert, ...] = ()
    evict: tuple[Expert, ...] = ()
    version: int = VERSION

    def __post_init__(self):
        for name in ("generation", "created_ns", "expires_ns", "version"):
            integer(getattr(self, name), name)
        if (type(self.promote) is not tuple or type(self.evict) is not tuple
                or any(not isinstance(e, Expert) for e in self.promote + self.evict)):
            raise ValueError("movement identities must be immutable expert tuples")


class Predictor(Protocol):
    def predict(self, context: Context) -> Forecast: ...


class PlacementPolicy(Protocol):
    def decide(self, record: PredictionRecord, state: Residency,
               model: Model, hardware: Hardware) -> MovementPlan: ...


@dataclass(frozen=True)
class BackendCapabilities:
    """Storage capabilities, not kernel dispatch or model-specific policy flags."""

    cold_execution: bool
    graph_stable_mapping: bool
    asynchronous_staging: bool
    requires_quiescence: bool
    staging_bytes: int


class ResidencyBackend(Protocol):
    """Design boundary only. No physical implementation is shipped here.

    prepare validates/reserves resources without publishing; commit requires a
    scheduler-owned reader fence and a completed transfer transaction. Abort
    retains the old mapping. Stable addresses alone do not satisfy this contract.
    """

    capabilities: BackendCapabilities

    def snapshot(self) -> Residency: ...
    def prepare(self, plan: MovementPlan) -> object: ...
    def commit(self, transaction: object, reader_fence: object) -> Residency: ...
    def abort(self, transaction: object) -> None: ...

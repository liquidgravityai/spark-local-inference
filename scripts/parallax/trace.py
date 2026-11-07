"""Compact JSONL trace transport; reject unsupported versions and malformed order."""

from dataclasses import asdict, dataclass
import json

from .contracts import Demand, Expert, Forecast, Model, Observation, PredictionRecord, VERSION


@dataclass(frozen=True)
class Trace:
    model: Model
    workload: str
    synthetic: bool
    observations: tuple[Observation, ...]

    def __post_init__(self):
        if type(self.observations) is not tuple:
            raise ValueError("trace observations must be immutable")
        if not self.workload or type(self.synthetic) is not bool:
            raise ValueError("workload identity and explicit synthetic marker required")
        previous = None
        catalog = frozenset(self.model.experts)
        for obs in self.observations:
            if any(d.expert not in catalog for d in obs.demands):
                raise ValueError("unknown canonical expert")
            if previous is not None and (obs.seq != previous.seq + 1
                    or obs.demand_ns < previous.demand_ns or obs.available_ns < previous.available_ns):
                raise ValueError("v1 requires contiguous execution and availability order; gaps need separate traces")
            previous = obs


def write_trace(path, trace):
    with open(path, "w") as stream:
        header = dict(type="header", version=VERSION, clock="relative_monotonic_ns",
                      model=asdict(trace.model), workload=trace.workload, synthetic=trace.synthetic)
        stream.write(json.dumps(header, separators=(",", ":"), allow_nan=False) + "\n")
        for observation in trace.observations:
            stream.write(json.dumps(dict(type="routing", **asdict(observation)), separators=(",", ":")) + "\n")


def read_trace(path):
    with open(path) as stream:
        header = json.loads(next(stream))
        if set(header) != {"type", "version", "clock", "model", "workload", "synthetic"}:
            raise ValueError("unknown or missing trace header field")
        if (header["type"] != "header" or type(header["version"]) is not int
                or header["version"] != VERSION or header["clock"] != "relative_monotonic_ns"):
            raise ValueError("unsupported trace version or clock")
        model = dict(header["model"])
        model["experts"] = tuple(Expert(**e) for e in model["experts"])
        model["expert_bytes"] = tuple(model["expert_bytes"])
        model = Model(**model)
        observations = []
        for line in stream:
            row = json.loads(line)
            if row.pop("type") != "routing":
                raise ValueError("unknown trace event")
            row["demands"] = tuple(Demand(Expert(**d["expert"]), d["count"]) for d in row["demands"])
            observations.append(Observation(**row))
    return Trace(model, header["workload"], header["synthetic"], tuple(observations))


def write_predictions(path, records):
    """One contender's immutable records; pair with a routing trace for evaluation."""
    with open(path, "w") as stream:
        stream.write(json.dumps(dict(type="predictions", version=VERSION)) + "\n")
        for record in records:
            stream.write(json.dumps(asdict(record), separators=(",", ":"), allow_nan=False) + "\n")


def read_predictions(path):
    with open(path) as stream:
        header = json.loads(next(stream))
        if (header != dict(type="predictions", version=VERSION)
                or type(header["version"]) is not int):
            raise ValueError("unsupported prediction stream schema")
        records = []
        for line in stream:
            row = json.loads(line)
            f = dict(row["forecast"])
            f["scores"] = tuple((Expert(**e), score) for e, score in f["scores"])
            f["config"] = tuple(tuple(pair) for pair in f["config"])
            row["forecast"] = Forecast(**f)
            records.append(PredictionRecord(**row))
    # Model, cutoff and target compatibility are checked by evaluate against ground truth.
    return tuple(records)

"""Replay Parallax routing traces; no GPU work or serving changes."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .contracts import Demand, Expert, Hardware, Model, Observation, VERSION
from .oracle import constrained_heuristic, idealized
from .policy import Economic, NoMovement
from .predictors import EWMA, Recent, Static
from .replay import Contender, replay
from .trace import Trace, read_trace, write_trace


def synthetic_trace():
    experts = tuple(Expert(0, i) for i in range(4))
    model = Model("SYNTHETIC-not-a-real-model", "synthetic-moe", experts, (1000,) * 4)
    observations = tuple(Observation(i, i * 10_000, i * 10_000 + 100,
                                    (Demand(experts[(i // 8) % 4], 4),)) for i in range(64))
    return Trace(model, "SYNTHETIC-block-locality-test-only", True, observations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", help="JSONL routing trace; omit only with --synthetic")
    parser.add_argument("--synthetic", action="store_true", help="run a labeled infrastructure fixture")
    parser.add_argument("--write-synthetic", help="write the fixture trace to this path")
    parser.add_argument("--hardware", help="JSON Hardware fields; required for real traces")
    parser.add_argument("--horizon-ns", type=int, default=10_000)
    parser.add_argument("--target-offset", type=int, default=1, help="future routing frames from the observation cutoff")
    parser.add_argument("--compute-ns", type=int, default=100, help="modeled predictor cost; not a measurement")
    parser.add_argument("--history", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--output", required=True, help="JSON comparison including immutable forecasts")
    args = parser.parse_args()
    if bool(args.trace) == args.synthetic:
        parser.error("choose one routing trace or --synthetic")
    if not args.synthetic and not args.hardware:
        parser.error("real traces require an explicit hardware cost model")
    trace = synthetic_trace() if args.synthetic else read_trace(args.trace)
    if args.write_synthetic:
        if not args.synthetic:
            parser.error("--write-synthetic requires --synthetic")
        write_trace(args.write_synthetic, trace)
    if args.hardware:
        with open(args.hardware) as stream:
            hardware = Hardware(**json.load(stream))
    else:
        hardware = Hardware("SYNTHETIC-UNCALIBRATED", 2000, 4000, 1, 100, 50, 2000)
    # Deterministic catalog-order placement: deliberately not fit on evaluation data.
    initial, used = [], 0
    for expert in trace.model.experts:
        if used + trace.model.size(expert) <= hardware.hbm_bytes:
            initial.append(expert)
            used += trace.model.size(expert)
    profile = Static(tuple((e, 1.) for e in initial), "catalog-order-control")
    contenders = (Contender("static-residency", profile, NoMovement()),
                  Contender("recent-economic", Recent(), Economic(), args.compute_ns),
                  Contender("ewma-economic", EWMA(), Economic(), args.compute_ns),
                  Contender("recent-no-movement", Recent(), NoMovement(), args.compute_ns))
    rows = replay(trace, contenders, hardware, horizon_ns=args.horizon_ns,
                  initial=tuple(initial), top_k=args.top_k, history_limit=args.history, target_offset=args.target_offset)
    # Replay state is deterministic; measured CPU timings remain deliberately separate.
    canonical = json.dumps(asdict(trace), sort_keys=True, separators=(",", ":"))
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode() + b"\0" + path.read_bytes())
    output = dict(version=VERSION, synthetic=trace.synthetic, workload=trace.workload,
                  trace_sha256=hashlib.sha256(canonical.encode()).hexdigest(),
                  implementation_sha256=source.hexdigest(),
                  model=asdict(trace.model), hardware=asdict(hardware),
                  config=dict(horizon_ns=args.horizon_ns, target_offset=args.target_offset, compute_ns=args.compute_ns,
                              history_limit=args.history, top_k=args.top_k, initial=[asdict(e) for e in initial]),
                  result_kind="MODELED additive route-cost estimates; NOT measured serving speedups",
                  contenders={name: dict(summary=row["summary"], execution=row["execution"],
                                         policy=type(next(c.policy for c in contenders if c.name == name)).__name__,
                                         policy_config=asdict(next(c.policy for c in contenders if c.name == name)),
                                         records=[asdict(r) for r in row["records"]]) for name, row in rows.items()},
                  constrained_oracle=constrained_heuristic(trace, hardware, tuple(initial), args.top_k))
    output["idealized_oracle"] = (idealized(trace, hardware, tuple(initial)) if len(trace.model.experts) <= 18
                                   else dict(kind="not_run", reason="exact enumeration limited to 18 experts"))
    with open(args.output, "w") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(output["result_kind"])
    print(f"workload={trace.workload} synthetic={trace.synthetic} hardware={hardware.identity}")
    for name, row in sorted(rows.items(), key=lambda p: -p[1]["summary"]["estimated_net_route_cost_saved_ns"]):
        s = row["summary"]
        print(f"{name:24} frames={s['routing_frames']} recall={s['recall']} "
              f"HBM-hit={s['hbm_hit_rate']} estimated-net-cost-saved={s['estimated_net_route_cost_saved_ns']}ns")


if __name__ == "__main__":
    main()

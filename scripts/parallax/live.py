"""Bounded same-host Flare routing collector and live shadow comparison CLI.

No inference callbacks, model loads, weight movement, or automatic active-policy
selection. Receipt time, including HTTP polling, is the predictor availability
boundary. Demand uses a conservative host-dispatch lower bound.
"""

import argparse
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
import resource
from time import monotonic_ns, process_time_ns, sleep
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .contracts import Context, Demand, Expert, Hardware, Model, Observation
from .policy import Economic, NoMovement
from .predictors import EWMA, Recent
from .replay import evaluate
from .shadow import ShadowPool
from .trace import Trace, write_predictions, write_trace


def compare_snapshot(snapshot):
    """Evaluate an immutable traffic prefix outside the collection process."""
    started = process_time_ns()
    trace = Trace(snapshot["model"], snapshot["workload"], False, snapshot["observations"])
    summaries = {}
    for provider, policies in snapshot["policies"].items():
        for name, policy in policies.items():
            summary, _ = evaluate(trace, snapshot["records"][provider], policy,
                                  snapshot["hardware"], initial=snapshot["initial"])
            summaries[f"{provider}/{name}"] = summary
    return dict(result_kind="SHADOW modeled route-cost estimates; NOT measured serving improvement",
                routing_frames=len(trace.observations),
                coalesced_prediction_points=snapshot["coalesced_points"],
                workers=snapshot["workers"], contenders=summaries,
                evaluation_cpu_ns=process_time_ns() - started)


class SnapshotEvaluator:
    """One bounded evaluation job; a failed policy cannot block shutdown."""

    def __init__(self):
        self.pool = multiprocessing.get_context("spawn").Pool(1)
        self.pending = None
        self.deadline_ns = 0
        self.failed = False

    def submit(self, snapshot, budget_ns):
        if self.failed or self.pending is not None or budget_ns <= 0:
            raise ValueError("evaluation requires an idle worker and positive budget")
        self.deadline_ns = monotonic_ns() + budget_ns
        self.pending = self.pool.apply_async(compare_snapshot, (snapshot,))

    def ready(self):
        if self.pending is None:
            return False
        if not self.pending.ready() and monotonic_ns() >= self.deadline_ns:
            self.failed = True
            self.close()
            raise TimeoutError("leaderboard evaluation exceeded its budget")
        return self.pending.ready()

    def result(self):
        try:
            return self.pending.get(timeout=max(0, self.deadline_ns - monotonic_ns()) / 1e9)
        except multiprocessing.TimeoutError as exc:
            self.failed = True
            self.close()
            raise TimeoutError("leaderboard evaluation exceeded its budget") from exc
        except Exception:
            self.failed = True
            self.close()
            raise
        finally:
            self.pending = None

    def close(self):
        if self.pool is not None:
            self.pool.terminate()
            self.pool.join()
            self.pool = None


class RoutingSource:
    """Replaceable source adapter; epoch/gaps are explicit evaluation boundaries."""

    def __init__(self, model, model_id, first, send_ns, receipt_ns):
        self.model, self.model_id = model, model_id
        self.catalog = frozenset(model.experts)
        self._validate(first)
        if not 0 <= send_ns <= receipt_ns:
            raise ValueError("invalid clock calibration interval")
        self.session = first["session"]
        # Native clock sample occurred between local request send and receipt.
        # Subtracting the upper offset gives a conservative local demand bound.
        self.offset_upper_ns = first["native_now_ns"] - send_ns
        self.clock_uncertainty_ns = receipt_ns - send_ns
        self.attach_ns = first["native_now_ns"]
        self.cursor = first["latest_sequence"]
        self.initial_invalid = first["invalid_total"]
        self.initial_dropped = first["dropped_total"]
        self.observations = []
        self.demand_entries = 0
        self.skipped_inflight_at_attach = 0

    def _validate(self, response):
        if (response.get("version") != 1 or response.get("enabled") is not True
                or response.get("clock") != "native_process_monotonic_ns"
                or response.get("demand_timing") != "host_dispatch_lower_bound"
                or response.get("model_id") != self.model_id):
            raise ValueError("incompatible or disabled routing source; check instrumented package and model identity")
        layers, experts = response["geometry"]
        if any(e.layer >= layers or e.expert >= experts for e in self.model.experts):
            raise ValueError("model catalog exceeds native routing geometry")

    def ingest(self, response, receipt_ns, *, max_frames, max_demand_entries):
        self._validate(response)
        if response["session"] != self.session:
            raise ValueError("serving session changed; start a new evaluation window")
        if (response["invalid_total"] != self.initial_invalid or response["dropped_total"] != self.initial_dropped):
            raise ValueError("native routing loss/invalid data; start a new evaluation window")
        added = []
        for row in response["frames"]:
            sequence = row["sequence"]
            if sequence <= self.cursor:
                continue
            if sequence != self.cursor + 1:
                raise ValueError("routing retention gap; cannot join disjoint histories")
            sample = row["sample"]
            if sample["demand_lower_bound_ns"] < self.attach_ns:
                self.skipped_inflight_at_attach += 1
                self.cursor = sequence
                continue
            demands = tuple(Demand(Expert(layer["layer"], e), count)
                for layer in sample["layers"] for e, count in enumerate(layer["counts"]) if count)
            if any(d.expert not in self.catalog for d in demands):
                raise ValueError("routing uses an expert missing from the model catalog")
            if len(self.observations) >= max_frames or self.demand_entries + len(demands) > max_demand_entries:
                break
            demand_ns = sample["demand_lower_bound_ns"] - self.offset_upper_ns
            if sample["available_ns"] < sample["demand_lower_bound_ns"] or demand_ns > receipt_ns:
                raise ValueError("noncausal routing timestamps")
            observation = Observation(len(self.observations), demand_ns, receipt_ns, demands,
                row["phase"], row["batch_size"], max(1, row["concurrency"]))
            if self.observations and (demand_ns < self.observations[-1].demand_ns
                    or receipt_ns < self.observations[-1].available_ns):
                raise ValueError("routing clock/order changed; start a new evaluation window")
            self.observations.append(observation)
            self.demand_entries += len(demands)
            self.cursor = sequence
            added.append(observation)
        return tuple(added)


class LiveBakeOff:
    """External coordinator with replaceable providers, source and policy pairings.

    Metrics replay the retained bounded prefix, so delayed routing observations
    cannot retroactively make a forecast available before consumer receipt.
    """

    def __init__(self, model, hardware, *, horizon_ns, history_limit=8, initial=(), workload="live-serving", target_offset=1):
        self.model, self.hardware = model, hardware
        self.horizon_ns, self.history_limit = horizon_ns, history_limit
        self.target_offset = target_offset
        self.initial, self.workload = initial, workload
        self.pool = ShadowPool(history_limit=history_limit, max_frame_experts=len(model.experts), model=model)
        self.records, self.policies = {}, {}
        self.coalesced_points = 0

    def register(self, name, provider, policies, *, budget_ns=1_000_000_000):
        self.pool.register(name, provider, budget_ns=budget_ns)
        self.records[name] = []
        self.policies[name] = dict(policies)

    def poll(self):
        for name, record in self.pool.poll():
            self.records[name].append(record)

    def observe(self, history, added):
        self.poll()
        if not added:
            return
        self.coalesced_points += len(added) - 1
        # One HTTP response made all these observations available together.
        context = Context(self.model, tuple(history[-self.history_limit:]),
                          added[-1].available_ns, self.horizon_ns, self.target_offset)
        self.pool.submit(context)

    def comparison(self, observations):
        return compare_snapshot(self.comparison_snapshot(observations))

    def comparison_snapshot(self, observations):
        return dict(model=self.model, hardware=self.hardware, initial=tuple(self.initial),
                    workload=self.workload, observations=tuple(observations),
                    records={name: tuple(records) for name, records in self.records.items()},
                    policies={name: dict(policies) for name, policies in self.policies.items()},
                    coalesced_points=self.coalesced_points,
                    workers={name: dict(worker.counters) for name, worker in self.pool.workers.items()})

    def close(self):
        self.pool.close()


def fetch(url, after, now, token=None):
    headers = {} if token is None else {"Authorization": "Bearer " + token}
    sent = now()
    request = Request(url.rstrip("/") + f"/v1/parallax/routing?after={after}", headers=headers)
    with urlopen(request, timeout=2) as response:
        payload = response.read(32 * 1024 * 1024 + 1)
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError("routing response exceeds collector memory limit")
    received = now()
    return json.loads(payload), sent, received, len(payload)


def collector_memory():
    # Linux rusage high-water RSS can survive fork/exec from a large launcher.
    # /proc describes this executable's address space, excluding that history.
    try:
        status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
        return dict(collector_rss_kib=int(status["VmRSS"].split()[0]),
                    collector_peak_rss_kib=int(status["VmHWM"].split()[0]))
    except (OSError, KeyError, ValueError):
        return dict(collector_rss_kib=None, collector_peak_rss_kib=None)


def publish_report(directory, report, format):
    """Publish a complete snapshot for terminal viewers without partial JSON reads."""
    text = json.dumps(report, allow_nan=False)
    temporary = directory / "leaderboard.json.tmp"
    temporary.write_text(text + "\n")
    temporary.replace(directory / "leaderboard.json")
    if format == "table":
        from .presentation import render
        text = render(report)
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="same-host Flare URL")
    parser.add_argument("--model-id", required=True, help="exact native model_id; do not silently mix packages")
    parser.add_argument("--catalog", required=True, type=Path, help="JSON Model fields; identity must fingerprint this package")
    parser.add_argument("--hardware", required=True, type=Path, help="explicit Hardware cost model; identify calibration status")
    parser.add_argument("--output", required=True, type=Path, help="new capture directory")
    parser.add_argument("--token-file", type=Path, help="optional local bearer-token file; never recorded")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--poll-ms", type=float, default=10)
    parser.add_argument("--max-frames", type=int, default=1024)
    parser.add_argument("--max-demand-entries", type=int, default=250000)
    parser.add_argument("--horizon-ns", type=int, default=10_000_000)
    parser.add_argument("--target-offset", type=int, default=1, help="future routing frames from the observation cutoff")
    parser.add_argument("--predictor-budget-ms", type=float, default=1000)
    parser.add_argument("--evaluation-budget-ms", type=float, default=15000)
    parser.add_argument("--format", choices=("json", "table"), default="json")
    parser.add_argument("--stop-file", type=Path, help="finish after a final poll when this file appears")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "::1", "localhost") or parsed.username or parsed.password:
        parser.error("v1 requires same-host loopback HTTP for a shared monotonic clock rate; no tunnels")
    if any(v <= 0 for v in (args.seconds, args.poll_ms, args.max_frames, args.max_demand_entries,
                            args.horizon_ns, args.target_offset, args.predictor_budget_ms, args.evaluation_budget_ms)):
        parser.error("capture limits and timing values must be positive")
    values = json.loads(args.catalog.read_text())
    values["experts"] = tuple(Expert(**e) for e in values["experts"])
    values["expert_bytes"] = tuple(values["expert_bytes"])
    model = Model(**values)
    hardware = Hardware(**json.loads(args.hardware.read_text()))
    initial, used = [], 0
    for expert in model.experts:
        if used + model.size(expert) <= hardware.hbm_bytes:
            initial.append(expert)
            used += model.size(expert)
    token = args.token_file.read_text().strip() if args.token_file else None
    args.output.mkdir(parents=True, exist_ok=False)
    bakeoff = LiveBakeOff(model, hardware, horizon_ns=args.horizon_ns, initial=tuple(initial), workload=args.workload,
                         target_offset=args.target_offset)
    policies = {"static": NoMovement(), "economic": Economic()}
    source, error, transferred = None, None, 0
    started_cpu = process_time_ns()
    started_wall = monotonic_ns()
    evaluator = SnapshotEvaluator()
    evaluation_budget_ns = int(args.evaluation_budget_ms * 1e6)
    evaluation_cpu_ns = skipped_reports = 0
    try:
        # Metadata-only clock sample: retained histograms inflate the clock
        # uncertainty and cannot be used as observations before attachment.
        first, sent, received, count = fetch(args.url, (1 << 64) - 1, bakeoff.pool.now_ns, token)
        source = RoutingSource(model, args.model_id, first, sent, received)
        transferred += count
        for name, provider in (("recent", Recent()), ("ewma", EWMA())):
            bakeoff.register(name, provider, policies, budget_ns=int(args.predictor_budget_ms * 1e6))
        deadline = bakeoff.pool.now_ns() + int(args.seconds * 1e9)
        report_at = 0
        with (args.output / "native-routing.jsonl").open("w") as raw:
            raw.write(json.dumps(dict(response=first, local_send_ns=sent, local_receipt_ns=received)) + "\n")
            while bakeoff.pool.now_ns() < deadline and len(source.observations) < args.max_frames:
                response, sent, received, count = fetch(args.url, source.cursor, bakeoff.pool.now_ns, token)
                transferred += count
                raw.write(json.dumps(dict(response=response, local_send_ns=sent, local_receipt_ns=received)) + "\n")
                raw.flush()
                added = source.ingest(response, received, max_frames=args.max_frames,
                                      max_demand_entries=args.max_demand_entries)
                bakeoff.observe(source.observations, added)
                if args.stop_file is not None and args.stop_file.exists():
                    break
                if response["frames"] and not added and source.cursor < response["latest_sequence"]:
                    break  # Bounded demand retention exhausted; preserve the complete prefix.
                if evaluator.ready():
                    report = evaluator.result()
                    evaluation_cpu_ns += report["evaluation_cpu_ns"]
                    report.update(hardware=asdict(hardware), workload=args.workload)
                    publish_report(args.output, report, args.format)
                if received >= report_at:
                    if evaluator.pending is None:
                        evaluator.submit(bakeoff.comparison_snapshot(source.observations), evaluation_budget_ns)
                    else:
                        skipped_reports += 1
                    report_at = received + 1_000_000_000
                sleep(args.poll_ms / 1000)
    except KeyboardInterrupt:
        error = "interrupted"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        collection_cpu_ns = process_time_ns() - started_cpu
        collection_wall_ns = monotonic_ns() - started_wall
        try:
            bakeoff.poll()
            observations = () if source is None else tuple(source.observations)
            write_trace(args.output / "routing.jsonl", Trace(model, args.workload, False, observations))
            for name, records in bakeoff.records.items():
                write_predictions(args.output / f"{name}.predictions.jsonl", records)
            snapshot = bakeoff.comparison_snapshot(observations)
            bakeoff.close()
            # Durable observations and predictions precede optional evaluation.
            # Never rerun a failed policy synchronously in the collector.
            report = dict(result_kind="SHADOW evaluation incomplete; no ranking available",
                          routing_frames=len(observations), workers=snapshot["workers"], contenders={})
            try:
                if not evaluator.failed:
                    if evaluator.pending is not None:
                        evaluation_cpu_ns += evaluator.result()["evaluation_cpu_ns"]
                    evaluator.submit(snapshot, evaluation_budget_ns)
                    report = evaluator.result()
                    evaluation_cpu_ns += report["evaluation_cpu_ns"]
            except Exception as exc:
                error = error or f"leaderboard: {type(exc).__name__}: {exc}"
            finally:
                evaluator.close()
            report.update(error=error, model=asdict(model), hardware=asdict(hardware),
                config=dict(horizon_ns=args.horizon_ns, target_offset=args.target_offset, history_limit=bakeoff.history_limit,
                            initial=[asdict(e) for e in initial], poll_ms=args.poll_ms,
                            predictor_budget_ms=args.predictor_budget_ms, max_frames=args.max_frames,
                            evaluation_budget_ms=args.evaluation_budget_ms,
                            max_demand_entries=args.max_demand_entries),
                session=None if source is None else source.session,
                clock_uncertainty_ns=None if source is None else source.clock_uncertainty_ns,
                demand_timing="conservative host-dispatch lower bound",
                collector_cpu_ns=process_time_ns() - started_cpu, response_bytes=transferred,
                collection_cpu_ns=collection_cpu_ns, collection_wall_ns=collection_wall_ns,
                asynchronous_evaluation_cpu_ns=evaluation_cpu_ns,
                skipped_leaderboard_updates=skipped_reports)
            bakeoff.close()
            children = resource.getrusage(resource.RUSAGE_CHILDREN)
            report.update(children_cpu_ns=int((children.ru_utime + children.ru_stime) * 1e9),
                          **collector_memory(),
                          collector_rusage_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                          largest_child_peak_rss_kib=children.ru_maxrss)
            (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            publish_report(args.output, report, args.format)
        finally:
            bakeoff.close()
            evaluator.close()
    if error and error != "interrupted":
        raise SystemExit(error)


if __name__ == "__main__":
    main()

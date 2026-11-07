"""Demo capture integrity and the collector's cooperative completion protocol."""

from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Thread
from time import monotonic_ns

import pytest

from scripts.parallax.contracts import Demand, Expert, Hardware, Model, Observation
from scripts.parallax.demo import load_configuration, replay_capture
from scripts.parallax.live import publish_report
from scripts.parallax.policy import Economic, NoMovement
from scripts.parallax.predictors import Recent, EWMA
from scripts.parallax.presentation import render
from scripts.parallax.replay import Contender, evaluate, generate
from scripts.parallax.trace import Trace, write_predictions, write_trace

A, B = Expert(0, 0), Expert(0, 1)
MODEL = Model("synthetic;package=test", "fixture", (A, B), (100, 100))
HARDWARE = Hardware("SYNTHETIC demo test costs", 100, 200, 1, 10, 5, 1000)


@pytest.fixture
def capture(tmp_path):
    trace = Trace(MODEL, "SYNTHETIC demo regression", True, tuple(
        Observation(i, i * 10000, i * 10000 + 1, (Demand(B, 2),)) for i in range(5)))
    write_trace(tmp_path / "routing.jsonl", trace)
    report = dict(model=asdict(MODEL), hardware=asdict(HARDWARE), routing_frames=5,
                  error=None, config=dict(initial=[asdict(A)]), workers={}, contenders={})
    for name, provider in (("recent", Recent()), ("ewma", EWMA())):
        records, _ = generate(trace, Contender(name, provider, NoMovement(), compute_ns=10), 10000)
        write_predictions(tmp_path / f"{name}.predictions.jsonl", records)
        report["workers"][name] = dict(completed=len(records), dropped_busy=0, failures=0, timeouts=0)
        for label, policy in (("static", NoMovement()), ("economic", Economic())):
            report["contenders"][f"{name}/{label}"], _ = evaluate(trace, records, policy, HARDWARE, initial=(A,))
    (tmp_path / "comparison.json").write_text(json.dumps(report))
    return tmp_path


def test_demo_replay_preserves_independent_placements_and_saved_scores(capture, capsys):
    before = (capture / "comparison.json").read_bytes()
    report = replay_capture(capture)
    assert report["replay"] == "exact saved-summary match"
    assert report["contenders"]["recent/static"]["hits"] == 0
    assert report["contenders"]["recent/economic"]["hits"] > 0
    publish_report(capture, report, "table")
    printed = capsys.readouterr().out
    assert "modeled estimates" in printed and "NOT serving speedups" in printed
    assert "SYNTHETIC" in printed and "recent/static" in printed and "0.00%" in printed
    assert json.loads((capture / "leaderboard.json").read_text()) == report
    assert not (capture / "leaderboard.json.tmp").exists()
    assert (capture / "comparison.json").read_bytes() == before


@pytest.mark.parametrize("mutation, message", [
    (lambda r: r.update(error="routing gap"), "incomplete"),
    (lambda r: r["model"].update(identity="other-package"), "catalog differ"),
    (lambda r: r["contenders"]["recent/static"].update(hits=999), "does not reproduce"),
])
def test_replay_rejects_incomplete_or_inconsistent_results(capture, mutation, message):
    path = capture / "comparison.json"
    report = json.loads(path.read_text())
    mutation(report)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        replay_capture(capture)


def test_counterfactual_costs_are_labeled_and_do_not_rewrite_capture(capture):
    before = (capture / "comparison.json").read_bytes()
    profile = capture / "alternative.json"
    profile.write_text(json.dumps(asdict(HARDWARE) | dict(cold_cost_ns=0)))
    report = replay_capture(capture, profile)
    assert report["replay"] == "counterfactual hardware assumptions"
    assert report["contenders"]["recent/economic"]["promotion_count"] == 0
    assert (capture / "comparison.json").read_bytes() == before
    empty = dict(routing_frames=0, error="timeout", contenders={}, workers={})
    assert "INCOMPLETE: timeout" in render(empty)


def test_configuration_rejects_an_unbound_package_before_gpu_work(tmp_path, monkeypatch):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps(asdict(MODEL)))
    hardware = tmp_path / "hardware.json"
    hardware.write_text(json.dumps(asdict(HARDWARE)))
    config = tmp_path / "config.json"
    values = dict(binary=sys.executable, package=str(tmp_path), catalog=str(catalog),
                  hardware=str(hardware), gpu="GPU-test", package_id="test")
    config.write_text(json.dumps(values))
    inspection = dict(package=dict(signatures_verified=True, package_id="test", tensor_parallel=1),
                      model=dict(model_id="fixture"))
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: json.dumps(inspection))
    assert load_configuration(config)["model_id"] == "fixture"
    catalog.write_text(json.dumps(asdict(MODEL) | dict(identity="unbound")))
    with pytest.raises(ValueError, match="not bound"):
        load_configuration(config)
    inspection["package"]["signatures_verified"] = False
    with pytest.raises(ValueError, match="signature"):
        load_configuration(config)


def test_stop_file_collects_final_poll_and_produces_replayable_capture(tmp_path):
    stop = tmp_path / "stop"
    catalog, hardware = tmp_path / "catalog.json", tmp_path / "hardware.json"
    catalog.write_text(json.dumps(asdict(MODEL)))
    hardware.write_text(json.dumps(asdict(HARDWARE)))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            now = monotonic_ns()
            rows = []
            if not self.path.endswith(str((1 << 64) - 1)):
                rows = [dict(sequence=1, phase="decode", batch_size=1, concurrency=1, graph=0,
                    sample=dict(submission=1, demand_lower_bound_ns=now - 1, available_ns=now,
                                layers=[dict(layer=0, tokens=1, top_k=1, counts=[0, 1])]))]
                stop.touch()
            body = json.dumps(dict(version=1, enabled=True, model_id="fixture", geometry=[1, 2],
                clock="native_process_monotonic_ns", demand_timing="host_dispatch_lower_bound",
                session="synthetic-http", native_now_ns=now, latest_sequence=len(rows),
                dropped_total=0, invalid_total=0, frames=rows)).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "capture"
    try:
        result = subprocess.run([sys.executable, "-m", "scripts.parallax.live", "--url",
            f"http://127.0.0.1:{server.server_port}", "--model-id", "fixture", "--catalog", str(catalog),
            "--hardware", str(hardware), "--output", str(output), "--workload", "SYNTHETIC HTTP test",
            "--seconds", "60", "--stop-file", str(stop), "--format", "table"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
        report = replay_capture(output)
        assert report["routing_frames"] == 1 and report["error"] is None
        assert "INCOMPLETE" not in result.stdout
        assert json.loads((output / "leaderboard.json").read_text())["routing_frames"] == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

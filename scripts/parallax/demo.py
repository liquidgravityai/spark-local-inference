"""Run or replay a bounded Parallax shadow demonstration on an owned Flare server."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from .contracts import Expert, Hardware, Model
from .policy import Economic, NoMovement
from .presentation import render
from .replay import evaluate
from .trace import read_predictions, read_trace

GAUGES = ("flare_active_requests", "flare_pending_requests", "flare_retiring_request_slots",
          "flare_http_inflight_requests", "flare_frontend_active_workers")
PROMPTS = (
    "A useful way to explain GPU memory and expert routing to a software engineer is",
    "When planning a weekend walk through a city, I start by considering",
    "To debug a program that behaves differently under concurrency, first",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def request(url, path, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    with urlopen(Request(url + path, data=data, headers={"Content-Type": "application/json"}), timeout=timeout) as response:
        return json.load(response)


def load_configuration(path):
    values = json.loads(path.read_text())
    for name in ("binary", "package", "catalog", "hardware"):
        values[name] = str(Path(values[name]).expanduser().resolve(strict=True))
    if not values["gpu"].startswith("GPU-") or not 1024 <= values.get("port", 30153) <= 65535:
        raise ValueError("demo requires an explicit GPU UUID and unprivileged local port")
    inspection = json.loads(subprocess.check_output([values["binary"], "inspect", values["package"], "--json"], text=True))
    if not inspection["package"]["signatures_verified"] or inspection["package"]["package_id"] != values["package_id"]:
        raise ValueError("package identity or signature differs from the demo configuration")
    if inspection["package"]["tensor_parallel"] != 1:
        raise ValueError("the owned-server demo requires a single-rank package")
    catalog = json.loads(Path(values["catalog"]).read_text())
    if not catalog["identity"].endswith(";package=" + values["package_id"]):
        raise ValueError("expert catalog is not bound to the configured package")
    Model(catalog["identity"], catalog["architecture"], tuple(Expert(**e) for e in catalog["experts"]), tuple(catalog["expert_bytes"]))
    Hardware(**json.loads(Path(values["hardware"]).read_text()))
    values["model_id"] = inspection["model"]["model_id"]
    return values


def gpu_guard(uuid):
    deadline, idle = time.monotonic() + 30, 0
    while idle < 2:
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if any(line.split(",")[0].strip() == uuid for line in processes.splitlines()):
            raise RuntimeError("assigned GPU has a compute process; demo will not stop it")
        fields = subprocess.check_output(["nvidia-smi", "--id=" + uuid,
            "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
        memory, utilization = map(int, fields.strip().split(","))
        idle = idle + 1 if memory < 512 and utilization == 0 else 0
        if time.monotonic() >= deadline:
            raise RuntimeError("assigned GPU did not become idle")
        time.sleep(1)


def drain(url, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with urlopen(url + "/metrics", timeout=5) as response:
            text = response.read().decode()
        values = dict(line.split() for line in text.splitlines()
                      if line and not line.startswith("#") and line.split()[0] in GAUGES)
        if set(values) == set(GAUGES) and all(float(v) == 0 for v in values.values()):
            return
        time.sleep(.1)
    raise TimeoutError("owned server did not drain; it has been left running")


@contextmanager
def serving(config, output):
    port = config.get("port", 30153)
    url = f"http://127.0.0.1:{port}"
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    gpu_guard(config["gpu"])
    key = output / "empty-api-key"
    key.write_text("")
    argv = [config["binary"], "run", config["package"], "--gpus", "0", "--host", "127.0.0.1",
            "--port", str(port), "--served-model-name", config["model_id"], "--api-key-file", str(key),
            "--verify-weights", "--embedding-table-storage", "ram", "--scheduler-policy", "balanced",
            "--max-concurrency", "32", "--mtp-draft-tokens", "0", "--no-mtp-adaptive-verification",
            "--prefix-cache-memory", "0", "--prefix-cache-disk-limit", "0", "--prefix-cache-drain-on-shutdown=off",
            "--parallax-routing"]
    launch = dict(argv=argv, package_id=config["package_id"], complete=False)
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=config["gpu"]), start_new_session=True)
        launch["pid"] = process.pid
        save(output / "launch.json", launch)
        ready = False
        try:
            print(f"Loading verified package; startup log: {output / 'server.log'}", flush=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"native server exited {process.returncode}; see server.log")
                try:
                    request(url, "/health", timeout=2)
                    ready = True
                    break
                except (OSError, ValueError):
                    time.sleep(1)
            if not ready:
                raise TimeoutError("native server did not become ready")
            listener = subprocess.check_output(["ss", "-ltnp", "sport", "=", f":{port}"], text=True)
            if f"pid={process.pid}," not in listener:
                raise RuntimeError("ready endpoint is not owned by this demo")
            metadata = request(url, "/v1/parallax/routing?after=18446744073709551615")
            if metadata["model_id"] != config["model_id"] or not metadata["enabled"]:
                raise ValueError("native model identity or tracing selection mismatch")
            yield url
            launch["complete"] = True
        finally:
            try:
                if process.poll() is None:
                    if ready:
                        drain(url)
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=120)
            finally:
                launch["server_returncode"] = process.poll()
                launch["complete"] = launch["complete"] and process.returncode == 0
                save(output / "launch.json", launch)
                print(f"Owned server exit: {process.poll()}", flush=True)
            if process.returncode != 0:
                raise RuntimeError("native server did not exit cleanly; inspect launch.json")


def replay_capture(path, hardware_path=None):
    directory = path / "capture" if (path / "capture").is_dir() else path
    original = json.loads((directory / "comparison.json").read_text())
    if original.get("error"):
        raise ValueError("capture was incomplete; do not present it as a completed demo")
    trace = read_trace(directory / "routing.jsonl")
    if json.loads(json.dumps(asdict(trace.model))) != original["model"]:
        raise ValueError("saved model and trace catalog differ")
    hardware = Hardware(**(json.loads(hardware_path.read_text()) if hardware_path else original["hardware"]))
    initial = tuple(Expert(**e) for e in original["config"]["initial"])
    summaries = {}
    for name in original["workers"]:
        records = read_predictions(directory / f"{name}.predictions.jsonl")
        for label, policy in (("static", NoMovement()), ("economic", Economic())):
            summary, _ = evaluate(trace, records, policy, hardware, initial=initial)
            key = f"{name}/{label}"
            if hardware_path is None and summary != original["contenders"][key]:
                raise ValueError(f"replay does not reproduce saved summary: {key}")
            summaries[key] = summary
    return dict(original, hardware=asdict(hardware), contenders=summaries,
                replay="counterfactual hardware assumptions" if hardware_path else "exact saved-summary match")


def run_live(config, output, requests, prompt):
    with serving(config, output) as url:
        capture, stop = output / "capture", output / "stop-collection"
        command = [sys.executable, "-m", "scripts.parallax.live", "--url", url, "--model-id", config["model_id"],
            "--catalog", config["catalog"], "--hardware", config["hardware"], "--output", str(capture),
            "--workload", "Parallax demonstration; artificial prompts, real inference routing",
            "--seconds", "60", "--max-frames", "2048", "--max-demand-entries", "2000000",
            "--target-offset", "8", "--horizon-ns", "60000000", "--poll-ms", "1",
            "--predictor-budget-ms", "3000", "--evaluation-budget-ms", "30000", "--stop-file", str(stop)]
        responses = []
        with (output / "collector.log").open("w") as log:
            collector = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 20
                while not (capture / "native-routing.jsonl").exists():
                    if collector.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("collector did not attach; see collector.log")
                    time.sleep(.1)
                print("Serving real requests. All placement decisions remain virtual.\n", flush=True)
                displayed = None
                with ThreadPoolExecutor(max_workers=1) as traffic:
                    for index in range(requests):
                        text = prompt or PROMPTS[index % len(PROMPTS)]
                        body = dict(model=config["model_id"], prompt=f"Demonstration {index + 1}. {text}",
                                    max_tokens=128, ignore_eos=True, temperature=1., top_p=.95, top_k=-1, seed=1234)
                        future = traffic.submit(request, url, "/v1/completions", body)
                        while not future.done():
                            if collector.poll() is not None:
                                raise RuntimeError("collector stopped while inference was active")
                            board = capture / "leaderboard.json"
                            if board.exists():
                                encoded = board.read_text()
                                if encoded != displayed:
                                    print(render(json.loads(encoded)) + "\n", flush=True)
                                    displayed = encoded
                            time.sleep(.2)
                        result = future.result()
                        if result["usage"]["completion_tokens"] != 128 or len(result["choices"]) != 1:
                            raise ValueError("demo response has incomplete token counts")
                        responses.append(result)
                        save(output / "responses.json", responses)
                        print(f"Response {index + 1}: {result['choices'][0]['text']}\n", flush=True)
                drain(url)
                stop.touch()
                while collector.poll() is None:
                    board = capture / "leaderboard.json"
                    if board.exists():
                        encoded = board.read_text()
                        if encoded != displayed:
                            print(render(json.loads(encoded)) + "\n", flush=True)
                            displayed = encoded
                    if time.monotonic() > deadline + 180:
                        raise TimeoutError("collector did not finish")
                    time.sleep(.2)
                if collector.returncode:
                    report = json.loads((capture / "comparison.json").read_text())
                    raise RuntimeError(f"collector rejected the window: {report.get('error')}; evidence: {capture}")
            finally:
                stop.touch()
                if collector.poll() is None:
                    collector.send_signal(signal.SIGINT)
                    try:
                        collector.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        # The collector and its CPU workers belong to this demo.
                        os.killpg(collector.pid, signal.SIGKILL)
                        collector.wait()
                        raise RuntimeError("collector exceeded its shutdown budget; capture is incomplete")
        report = replay_capture(capture)
        if report["routing_frames"] == 0 or set(report["workers"]) != {"recent", "ewma"} or any(v["completed"] == 0 for v in report["workers"].values()):
            raise ValueError("demo did not observe routing and predictions from every provider")
        print(render(report), flush=True)
        print("Offline replay exactly reproduces all four contender summaries.", flush=True)
    save(output / "demo.json", dict(complete=True, mode="shadow", package_id=config["package_id"],
                                  routing_frames=report["routing_frames"], responses=len(responses), replay_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="local package/runtime/GPU configuration; needed only for live mode")
    modes = parser.add_subparsers(dest="mode", required=True)
    live = modes.add_parser("live", help="start an owned native server, generate demo traffic, compare predictors, then drain")
    live.add_argument("--output", type=Path, help="new capture directory; defaults to a fresh temporary directory")
    live.add_argument("--requests", type=int, choices=range(1, 7), default=3)
    live.add_argument("--prompt", help="optional request prefix; generated responses are saved locally")
    replay = modes.add_parser("replay", help="verify and display saved predictions without a GPU")
    replay.add_argument("capture", type=Path)
    replay.add_argument("--hardware", type=Path, help="evaluate alternative explicit cost assumptions without changing saved evidence")
    args = parser.parse_args()
    if args.mode == "replay":
        report = replay_capture(args.capture, args.hardware)
        print(render(report))
        print("Replay:", report["replay"])
        return
    if args.config is None:
        parser.error("live requires --config")
    config = load_configuration(args.config)
    output = args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="parallax-demo-"))
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    print("Demo artifacts:", output, flush=True)
    try:
        run_live(config, output, args.requests, args.prompt)
    except BaseException as exc:
        save(output / "demo.json", dict(complete=False, error=f"{type(exc).__name__}: {exc}"))
        raise


if __name__ == "__main__":
    main()

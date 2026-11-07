"""External CPU shadow-worker prototype, with no Flare or CUDA integration.

    submit(context) -> poll() -> immutable PredictionRecord values

Run in a collector process, never the serving thread. Workers are trusted code;
process isolation controls hangs/failures, not malicious code or GPU allocation.
"""

from dataclasses import dataclass, field
import multiprocessing as mp
from queue import Empty, Full
from time import monotonic_ns

from .contracts import Context, PredictionRecord


def _worker(provider, requests, replies, origin, model):
    while True:
        context = requests.get()
        if model is not None:
            context = Context(model, *context)
        start = monotonic_ns() - origin
        try:
            prediction = provider.predict(context)
            prediction.validate(context)
            error = None
        except Exception as exc:
            prediction, error = None, type(exc).__name__
        end = monotonic_ns() - origin
        replies.put((prediction, start, end - start, error))


@dataclass
class Worker:
    process: object
    requests: object
    replies: object
    budget_ns: int
    pending: object = None
    submitted_ns: int = 0
    disabled: bool = False
    counters: dict = field(default_factory=lambda: dict(submitted=0, dropped_busy=0, failures=0,
                                                        timeouts=0, completed=0, dropped_disabled=0))


class ShadowPool:
    def __init__(self, *, max_workers=4, history_limit=64, max_frame_experts=4096, max_catalog_experts=65536, model=None):
        if any(type(x) is not int or x < 1 for x in (max_workers, history_limit, max_frame_experts, max_catalog_experts)):
            raise ValueError("positive resource limits required")
        self.max_workers, self.history_limit, self.max_frame_experts = max_workers, history_limit, max_frame_experts
        self.max_catalog_experts = max_catalog_experts
        if model is not None and len(model.experts) > max_catalog_experts:
            raise ValueError("model exceeds catalog limit")
        self.model = model
        self.origin_ns = monotonic_ns()
        self.workers = {}
        self.context = mp.get_context("spawn")

    def now_ns(self):
        return monotonic_ns() - self.origin_ns

    def register(self, name, provider, *, budget_ns=1_000_000_000):
        if not name or name in self.workers or len(self.workers) >= self.max_workers:
            raise ValueError("invalid/duplicate name or worker limit exceeded")
        if type(budget_ns) is not int or budget_ns <= 0:
            raise ValueError("positive worker budget required")
        requests, replies = self.context.Queue(maxsize=1), self.context.Queue(maxsize=1)
        process = self.context.Process(target=_worker, args=(provider, requests, replies, self.origin_ns, self.model), daemon=True)
        process.start()
        self.workers[name] = Worker(process, requests, replies, budget_ns)

    def submit(self, context):
        """One frozen eligible snapshot fans out to every contender; never wait."""
        if self.model is not None and context.model != self.model:
            raise ValueError("snapshot model differs from the worker catalog")
        if (len(context.history) > self.history_limit
                or any(len(o.demands) > self.max_frame_experts for o in context.history)
                or len(context.model.experts) > self.max_catalog_experts
                or context.available_ns > self.now_ns()):
            raise ValueError("snapshot exceeds limits or is not yet available")
        for worker in self.workers.values():
            if worker.disabled:
                worker.counters["dropped_disabled"] += 1
            elif worker.pending is not None:
                worker.counters["dropped_busy"] += 1
            else:
                try:
                    # A fixed immutable catalog is installed at worker startup;
                    # only eligible observations and their target cross per job.
                    request = (context.history, context.available_ns, context.horizon_ns, context.target_offset)
                    worker.requests.put_nowait(context if self.model is None else request)
                except Full:
                    worker.counters["dropped_busy"] += 1
                    continue
                worker.pending, worker.submitted_ns = context, self.now_ns()
                worker.counters["submitted"] += 1

    def poll(self):
        results = []
        for name, worker in self.workers.items():
            if worker.disabled or worker.pending is None:
                continue
            now = self.now_ns()
            # Budget includes scheduling, IPC, startup on first use, and receipt.
            if now - worker.submitted_ns > worker.budget_ns:
                worker.process.terminate()
                worker.disabled = True
                worker.pending = None
                worker.counters["timeouts"] += 1
                continue
            try:
                prediction, start, compute, error = worker.replies.get_nowait()
            except Empty:
                if not worker.process.is_alive():
                    worker.disabled = True
                    worker.pending = None
                    worker.counters["failures"] += 1
                continue
            context, worker.pending = worker.pending, None
            try:
                if error:
                    raise ValueError(error)
                prediction.validate(context)
                record = PredictionRecord(prediction, context.available_ns, start, self.now_ns(), compute, "measured")
            except (ValueError, TypeError, AttributeError):
                worker.counters["failures"] += 1
                continue
            worker.counters["completed"] += 1
            results.append((name, record))
        return results

    def remove(self, name):
        worker = self.workers.pop(name)
        worker.process.terminate()
        worker.process.join(timeout=1)
        if worker.process.is_alive():
            worker.process.kill()
            worker.process.join(timeout=1)
        for queue in (worker.requests, worker.replies):
            queue.cancel_join_thread()
            queue.close()
        worker.process.close()
        return dict(worker.counters)

    def close(self):
        for name in list(self.workers):
            self.remove(name)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

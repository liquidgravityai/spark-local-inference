"""Independent virtual residency with one serialized copy engine and cold reads.

All costs are additive route-cost proxies on a fixed demand timeline. No elapsed
inference time, critical path, bandwidth/compute interference, or throughput is
inferred from these counters. Full immutable backing copies make evictions free.
"""

from dataclasses import dataclass

from .contracts import Residency, VERSION
from .policy import transfer_ns


@dataclass(frozen=True)
class Transfer:
    expert: object
    start_ns: int
    complete_ns: int


class VirtualResidency:
    def __init__(self, model, hardware, initial=()):
        self.model, self.hardware = model, hardware
        catalog = frozenset(model.experts)
        if len(set(initial)) != len(initial) or any(e not in catalog for e in initial):
            raise ValueError("invalid initial residency")
        if sum(model.expert_bytes) > hardware.backing_bytes:
            raise ValueError("v1 requires backing capacity for the complete immutable catalog")
        self.resident, self.pending = set(initial), []
        self.reserved_bytes = sum(model.size(e) for e in self.resident)
        self.generation, self.now_ns = 0, 0
        self.unused, self.evicted = set(), set()
        self.history = []
        self.counters = dict(hits=0, misses=0, promotion_count=0, eviction_count=0,
                             movement_bytes=0, transfer_ns=0, synchronization_ns=0,
                             successful_prefetches=0, late_prefetch_demands=0,
                             wasted_prefetches=0, eviction_reuses=0, cold_cost_ns=0)
        if self.reserved_bytes > hardware.hbm_bytes:
            raise ValueError("initial residency exceeds HBM reservation")

    def snapshot(self):
        pending = frozenset(t.expert for t in self.pending)
        return Residency(self.generation, frozenset(self.resident), pending,
                         self.reserved_bytes)

    def advance(self, now_ns):
        if now_ns < self.now_ns:
            raise ValueError("simulation clock cannot run backwards")
        self.now_ns = now_ns
        remaining = []
        for transfer in self.pending:
            if transfer.complete_ns <= now_ns:
                self.resident.add(transfer.expert)
                self.unused.add(transfer.expert)
                self.generation += 1
                self.history.append((transfer.complete_ns, "publish", transfer.expert))
            else:
                remaining.append(transfer)
        self.pending = remaining

    def apply(self, plan):
        """Validate the entire plan before changing state; stale plans fail closed."""
        if (plan.version != VERSION or plan.model_identity != self.model.identity
                or plan.generation != self.generation or plan.created_ns != self.now_ns
                or plan.expires_ns <= self.now_ns):
            raise ValueError("incompatible, stale or expired movement plan")
        if not plan.promote and not plan.evict:
            return
        pending = {t.expert for t in self.pending}
        promotes, evicts = set(plan.promote), set(plan.evict)
        if (len(promotes) != len(plan.promote) or len(evicts) != len(plan.evict)
                or promotes & evicts or not evicts <= self.resident
                or not promotes <= set(self.model.experts)
                or promotes & (self.resident | pending)):
            raise ValueError("invalid movement identities or dependencies")
        reserved = self.reserved_bytes + sum(self.model.size(e) for e in promotes) - sum(self.model.size(e) for e in evicts)
        if reserved > self.hardware.hbm_bytes:
            raise ValueError("movement exceeds HBM including pending reservations")
        start = max([self.now_ns] + [t.complete_ns for t in self.pending])
        transfers = []
        for expert in plan.promote:
            end = start + transfer_ns(self.model, self.hardware, expert) + self.hardware.synchronization_ns
            if end > plan.expires_ns:
                raise ValueError("serialized transfer cannot meet plan expiration")
            transfers.append(Transfer(expert, start, end))
            start = end
        for expert in plan.evict:
            self.resident.remove(expert)
            self.evicted.add(expert)
            self.counters["eviction_count"] += 1
            if expert in self.unused:
                self.counters["wasted_prefetches"] += 1
                self.unused.remove(expert)
            self.history.append((self.now_ns, "evict", expert))
        for transfer in transfers:
            self.counters["movement_bytes"] += self.model.size(transfer.expert)
            self.counters["promotion_count"] += 1
            self.counters["transfer_ns"] += transfer_ns(self.model, self.hardware, transfer.expert)
            self.counters["synchronization_ns"] += self.hardware.synchronization_ns
            self.history.append((transfer.start_ns, "copy", transfer.expert))
        self.pending.extend(transfers)
        # Pending copies already reserve their destination; publication changes
        # ownership and generation, but never the occupied byte count.
        self.reserved_bytes = reserved
        if promotes or evicts:
            self.generation += 1

    def demand(self, observation):
        self.advance(observation.demand_ns)
        pending = {t.expert for t in self.pending}
        for demand in observation.demands:
            expert, count = demand.expert, demand.count
            hit = expert in self.resident
            self.counters["hits" if hit else "misses"] += count
            if not hit:
                self.counters["cold_cost_ns"] += count * self.hardware.cold_cost_ns
            if expert in pending:
                self.counters["late_prefetch_demands"] += count
            if hit and expert in self.unused:
                self.counters["successful_prefetches"] += 1
                self.unused.remove(expert)
            if expert in self.evicted:
                self.counters["eviction_reuses"] += 1
                self.evicted.remove(expert)

    def summary(self):
        return dict(self.counters, pending_at_end=len(self.pending), unused_at_end=len(self.unused))

# Parallax: predictive expert residency

This document specifies the Parallax prototype developed against Flare revision
`a161b17f3ef3ac5c0cd3e219719f7d1d36a01175`. The prototype source, tests,
preparation flag and HTTP endpoint are separate from this documentation-only
branch. Commands below require a build containing those components.

Parallax evaluates whether forecasting Mixture-of-Experts (MoE) demand can make
expert weight placement cheaper than static residency or reactive caching. A
useful forecast must identify demand early enough to complete a transfer, and the
execution savings must exceed prediction, movement and synchronization costs.

The implementation provides deterministic offline replay, statistical predictors,
independent residency simulations and a native routing feed for shadow evaluation.
Shadow evaluation runs predictors against serving observations while leaving
physical expert placement unchanged. GLM-5.3-Flash preparation includes optional
routing instrumentation for single-GPU ordinary decode and prefill. End-to-end
qualification of an instrumented GLM serving package remains outstanding.

Physical expert movement is a proposed extension. Its hardware target is GB300
HBM and coherent Grace memory. GB300-to-RTX transfers and remote expert execution
are outside the implemented path. Hardware cost calibration and measured serving
improvement are also outstanding.

## Component boundaries

Prediction, placement, execution and measurement have separate owners. The
immutable values and protocols are defined in
`scripts/parallax/contracts.py`.

| Component | Contract | Responsibility |
| --- | --- | --- |
| Prediction provider | Eligible observation history → demand forecast | Predict utilization; identify the provider, configuration and compatible model |
| Placement policy | Forecast, residency snapshot and hardware costs → movement plan | Decide whether an exchange is economical, including choosing no movement |
| Residency orchestration | Movement plan → admitted transaction and publication outcome | Validate capacity, generations, deadlines and reader safety; recover from failure |
| Residency backend | Admitted transaction and scheduler fence → physical movement | Own storage, copies and mapping publication under explicit hardware capabilities |
| Observatory | Observations, forecasts and residency outcomes → common measurements | Score predictions and compare independent simulations |

The predictor and policy protocols have working implementations in
`scripts/parallax/predictors.py` and
`scripts/parallax/policy.py`. The physical backend protocol is an
interface proposal. Its capability record describes cold execution, graph-stable
mapping, asynchronous staging, quiescence requirements and staging capacity.
Neither the protocol nor the simulator implements GPU weight movement.

Model computation stays with Flare's prepared FlashInfer/b12x kernels. Residency
backends describe storage mechanisms; they do not select interchangeable GEMM
implementations. Helios owns shared scheduling and native execution. Python
providers run in an external collector and receive no inference callbacks.

Selene owns prefix and KV reuse. Parallax's simulated HBM budget reserves space
for experts after KV, graphs and workspace. Physical admission must account for
all of those allocations and any staging capacity.

## Observation and prediction contracts

An expert has a canonical `(layer, expert)` identity independent of its physical
slot. The model catalog records architecture, a checkpoint/representation
fingerprint, sorted canonical identities and corresponding byte sizes. Sizes
include every weight plane and scale that must move with an expert.

The routing trace format in `scripts/parallax/trace.py` is
versioned JSONL. Its header contains model metadata, clock and workload identities,
and an explicit `synthetic` flag. Each observation contains an execution sequence,
demand time, consumer availability time, expert demand counts, execution phase,
batch size and concurrency. Times are integer nanoseconds in one trace-relative
monotonic domain. Availability cannot precede demand.

Trace schema version 1 requires contiguous execution sequences and ordered demand
and availability times. A gap ends an evaluation window; disjoint windows require
separate evaluator state. Counts describe executed routes and exclude padding and
inactive rows. The schema can label speculative work, but the native collector
supports ordinary execution only. Prompt text, token IDs, hidden states, routing
weights and request identifiers are absent.

A provider receives an immutable bounded history whose observations were
available at the prediction point. Its forecast identifies the model, provider
version and configuration, observation cutoff, target frame, anticipated demand
time, expiration, score meaning and optional confidence. Forecasts target the
routing frame immediately following the cutoff. Anticipated demand time comes
from a configured horizon, without consulting that frame's recorded demand.

Scores may represent expected counts, probabilities or rankings. Rankings are
ordered scores, with no implied probability calibration. Arbitrary target
windows, feature availability manifests and desired-placement recommendations
require contract extensions.

A prediction record adds observation availability, execution start, consumer
receipt, computation duration and a measured-or-modeled timing label. Queue delay
is the interval from observation availability to execution start. Usable lead is
demand time minus consumer receipt; a negative value means the forecast arrived
after demand. Receipt includes IPC and collector delays.

At equal timestamps, demand at the observation cutoff precedes a forecast derived
from it; that forecast may precede demand at the following frame. Collectors must
preserve this causal order or use conservative availability bounds. Device
execution timestamps require host-clock calibration and an error bound.

The evaluator rejects incompatible versions or model identities, invalid cutoffs
and duplicate targets. Providers are trusted research code: these checks validate
records, not a provider's internal access to information. Live providers receive
neither the full replay trace nor offline oracle data.

## Placement and virtual residency

Each predictor-policy pairing has its own resident set, pending transfers,
capacity reservations, generation and movement history. All pairings consume the
same recorded demand under one hardware cost model. Policies can be compared
against identical immutable forecasts by passing the records to
`evaluate` (`scripts/parallax/replay.py`) separately for each policy.

The supplied prediction baselines are a fixed external profile, recent-frame
frequency and finite-window exponentially weighted moving average (EWMA).
EWMA includes zero demand for experts absent from a frame. Static frequency
profiles require separate calibration data. The CLI's static control selects
experts in catalog order; it does not derive a profile from evaluation traffic.
There is no trained predictor or training pipeline.

The static policy preserves placement. The economic policy accepts expected-count
forecasts and proposes at most one promotion per forecast. It subtracts predicted
victim demand, charges transfer and synchronization costs, checks usable lead and
applies a configurable benefit margin. Pending transfers suppress additional
exchanges. Unsupported score types, expired forecasts and unprofitable exchanges
produce an empty plan.

Movement plans carry model identity, residency generation, creation and expiration
times, and explicit promotion and eviction lists. The
virtual residency engine (`scripts/parallax/simulator.py`) validates the whole
plan before mutation. Rejected plans have no partial effect. Desired-placement
reconciliation is not implemented.

The simulation assumes:

- An expert-only HBM capacity and immutable backing storage for the complete catalog.
- One serialized copy engine with configured bandwidth, startup latency and
  synchronization cost.
- Immediate HBM reservation for promotions, publication after transfer completion,
  and cold execution until publication.
- Immediate slot release on eviction, without dirty writeback or in-flight readers.
- A fixed demand timeline on which transfers can overlap computation. Every
  selected expert executes, including misses.

The simulator can admit multiple serialized transfers even though the economic
policy proposes one promotion at a time. Kernel contention, scheduler pauses,
shared-memory pressure, replacement cooldowns and separate temporary staging are
not modeled. Physical admission and performance evaluation require those effects
to be measured and incorporated where material.

## Measurement and oracle controls

The observatory separates prediction quality, modeled residency benefit and
measured serving performance. Its evaluator reports:

| Measurement | Definition and limits |
| --- | --- |
| Precision, recall and coverage | Expert-set accuracy and demand-weighted top-K coverage; K applies across a frame's catalog |
| Missing, late and expired forecasts | Zero operational coverage with their demand retained in the denominator |
| Predictor timing | Compute p50/p95/p99, queue p95 and usable-lead p50/p95 |
| Actionable rate | Fraction of routing frames with enough lead to transfer at least one correctly predicted expert; queue and capacity feasibility are evaluated separately by residency simulation |
| Residency effectiveness | HBM hits, cold demand, useful completed prefetches, late pending demand, promotions, evictions, bytes moved, transfer/synchronization cost and reuse after eviction |
| Unused transfers | Unused promotions become wasted on eviction; unused and pending promotions at trace end are censored |
| Workload strata | Layer, execution phase, batch size, concurrency and anticipated horizon |
| Sample support | Routing and prediction counts; confidence intervals are unavailable without independent workload repeats |

Reuse-after-eviction counts have no near-term cutoff. Busy workers skip prediction
points independently; skipped targets remain in operational coverage. Forecast
stability, uncertainty calibration, resource utilization and block-bootstrap
confidence intervals are not implemented.

The net-value estimate subtracts modeled cold-route, transfer and synchronization
costs from the matching static-residency cost, then subtracts prediction release
delay. It charges full copy occupancy even when copies overlap computation. This
additive route-cost objective does not model elapsed serving time or throughput.
Policy execution, instrumentation, memory consumption and resource contention are
not calibrated into the estimate. Measured offline predictor durations are stored
separately from the configured computation and queue delays used for deterministic
replay.

The offline controls in `scripts/parallax/oracle.py` use future
routing knowledge:

1. **Idealized bound.** For catalogs of at most 18 experts, exact subset enumeration
   chooses the best HBM set for each frame with free instantaneous movement. The
   result bounds savings under the additive cold-route-cost objective.
2. **Cost-constrained heuristic.** Perfect demand knowledge for the following frame
   feeds the economic policy and the same capacity, transfer and deadline model
   used by contenders. The evaluator selects the better result of this strategy
   and static residency. It is not an optimal constrained solution or an upper bound.

An exact cost-constrained optimizer is not implemented. The idealized bound can
measure the route-cost opportunity recovered by realistic predictors on small
traces; a serving-throughput opportunity requires a different calibrated model.

## Native capture and live comparison

Preparation captures a read-only histogram operation after expert routing in
[`ignition/recipes/glm53/moe.py`](../ignition/recipes/glm53/moe.py). The reusable
routing counter (`ignition/components/routing_trace.py`) reads canonical expert
IDs and masks inactive packed rows. It writes a separate output without changing
IDs, routing weights, hidden states or cache state.

The preparation flag `--parallax-routing` requires GLM-5.3-Flash, one GPU,
`--no-mtp` and an ordinary graph catalog. Decode, prefill and scored prefill are
instrumented; speculative and mixed graphs are rejected. Instrumentation is off
by default. An instrumented package executes counters and native readback even
when no collector is attached. Changing instrumentation requires re-export.

The prototype extends the [execution manifest](../helios/manifest.md) with the
optional output `parallax_routes_v1`. It uses two persistent histogram
frames indexed by the executor's frame index and submission sequence. Native
[transport](../helios/src/execution/transport.rs) retains each readback until the
frame transfer fence completes. Reusable router scratch is not exposed to readers.
Unknown output versions are rejected. Stale or malformed histogram contents count
as invalid telemetry; CUDA transfer failures follow executor error handling.

The native routing telemetry (`helios/src/server/parallax.rs`) retains 32 records.
Its producer uses `try_lock` and records contention drops without waiting for an
HTTP reader. Retention eviction is visible as a sequence gap. The authenticated
`GET /v1/parallax/routing?after=N`
endpoint returns model/session identity, geometry, routing and submission
sequences, execution phase, batch size, histograms, timestamps and loss counters.
Concurrency is sampled from active requests at completion; the collector clamps
it to at least one. It is a workload-stratification proxy rather than dispatch-time
scheduler history.

Demand uses host dispatch before graph launch as a conservative lower bound.
Native readback availability is recorded separately. The same-host
collector (`scripts/parallax/live.py`) calibrates native-to-local clock offset
using an HTTP round trip and treats actual HTTP receipt as observation availability.
Polling, serialization and IPC therefore consume prediction lead time. The feed
provides no per-layer execution timestamps or within-graph lookahead.

The collector excludes work dispatched before attachment and rejects session
changes, clock reversals, sequence gaps and changes in loss counters. It saves raw
responses, normalized routing, prediction records and JSON comparisons. When one
response contains multiple observations, it retains all demand but schedules a
prediction point at the most recent observation and counts the coalesced points.

Trusted CPU providers run in separate spawned processes managed by
`ShadowPool` (`scripts/parallax/shadow.py`). The pool bounds contender count,
history and catalog sizes, and permits one outstanding job per contender.
Submission does not wait for providers. Polling publishes results, enforces
budgets and terminates timed-out workers; startup counts toward the first job's
budget. Failed workers do not stop other contenders. Explicit removal and
registration replace a provider without restarting the pool.

Process isolation does not impose hard CPU, RAM or GPU quotas. Collection and
prediction can still contend with inference. Artifact admission and operating
system resource limits are required for untrusted or resource-intensive providers.

The live CLI compares recent-frequency and EWMA providers, each paired with static
and economic policies. Each refresh reconstructs independent virtual residency
from the retained trace prefix. Frame and demand-entry limits bound retention;
the evaluator is not an incremental, indefinitely running simulator. HTTP polling
and the 32-record ring can lose observations at high replay rates. A gap ends the
comparison window rather than joining incomplete histories.

## Running replay and shadow evaluation

Run the CPU-only evaluator from the repository root with Python 3.12:

```sh
python -m scripts.parallax --synthetic \
  --write-synthetic /tmp/parallax-routing.jsonl \
  --output /tmp/parallax-results.json
python -m pytest -q tests/test_parallax.py tests/test_parallax_live.py
```

The synthetic fixture supplies artificial locality and uncalibrated costs for
infrastructure validation. JSON results contain trace and source digests,
workload/model identity, hardware profile, initial placement, evaluation settings,
forecasts, predictor timings, metrics and oracle controls. Forecasts and modeled
results reproduce; measured CPU durations vary.

To replay a real trace, pass its path as the positional argument and supply
`--hardware profile.json`. The options `--horizon-ns`, `--compute-ns`, `--history`
and `--top-k` configure prediction and evaluation. `--compute-ns` supplies modeled
prediction cost. Programmatic evaluation also accepts measured worker records,
custom profiles, initial placements and explicit queue delays.

Native shadow evaluation requires the repository's
[preparation environment and native build](model-preparation.md#environment-setup),
an available GPU, an instrumented package and request traffic. Export and load
that package using the [preparation and serving commands](cli-design.md):

```sh
ignition export glm-5.3-flash --gpus 0 --no-mtp --parallax-routing
flare run PACKAGE
python -m scripts.parallax.live --url http://127.0.0.1:8080 \
  --model-id EXACT_SERVED_MODEL_ID --catalog catalog.json \
  --hardware profile.json --workload held-out-workload-name \
  --output /tmp/parallax-live --seconds 60
```

Replace GPU index `0` with the assigned device, `PACKAGE` with the exported path,
port `8080` with the server's port, and `EXACT_SERVED_MODEL_ID` with its configured
model ID. The collector must run on the serving host through loopback HTTP and
requires a fresh output directory. Supply `--token-file` for API authentication.
Generate requests separately; the collector does not create inference traffic.

The catalog JSON contains the `Model` fields from `contracts.py`: `identity`,
`architecture`, sorted `experts` records and corresponding `expert_bytes`.
Measure complete expert representation sizes, including scales. The collector
checks the served model name and geometry, but the caller must associate the
catalog fingerprint with the loaded package; the endpoint does not attest it.

The hardware JSON contains the `Hardware` fields: `identity`, `hbm_bytes`,
`backing_bytes`, `bandwidth_bytes_per_ns`, `transfer_latency_ns`,
`synchronization_ns` and `cold_cost_ns`. Identify the calibration status in the
profile. Reducing its expert HBM budget simulates forced offload while actual
residency remains unchanged.

## Qualification and physical execution

The tests in `tests/test_parallax.py` cover causal
boundaries, independent providers and policies, deterministic replay, atomic plan
rejection, byte capacity, pending reservations, transfer deadlines, late and
expired forecasts, censored transfers, oracle controls and worker failure.
`tests/test_parallax_live.py` covers collector
clock bounds, retention gaps, session compatibility and policy independence.

The captured-counter tests in
`tests/test_parallax_capture.py` passed on an
RTX PRO 6000 Blackwell Max-Q with direct and whole-forward-compiled capture. They
exercise alternating frames, inactive rows, invalid IDs and unchanged routing
IDs. These component tests do not establish full-model RTX compatibility or GB300
serving qualification. Native tests cover output validation, histogram decoding
and bounded publication.

An instrumented GLM package still requires real-request qualification, numerical
and state equivalence checks, and measurement of capture and predictor overhead.
Framework tests can run independently of model correctness work. Model errors
must be resolved before comparative serving-performance evaluation.

For GB300, use a GLM package that fits entirely in HBM as the matched reference.
Verify capacity including KV, graph storage and workspace. Vary simulated expert
HBM capacity while holding checkpoint precision, kernel representation, request
settings, KV budget, concurrency and graph capacities fixed. The all-HBM result
measures that implementation; it does not establish maximum hardware performance.
If offload requires a different weight representation, measure an all-HBM control
for that representation too.

Measure uninstrumented serving, instrumented serving without an HTTP collector,
collection alone, and collection with predictors. Compare outputs and complete
state at fixed inputs. Record decode throughput, inter-token and tail latency,
TTFT, prefill performance and concurrency scaling. Physical-movement experiments
also require transfer bytes and duration, scheduler stalls and a matched static
baseline. Calibration and held-out workloads must be separate, with related
request continuations kept in the same split.

Physical orchestration requires immutable backing ownership, spare staging
reservations, canonical-to-slot indirection supported by the kernel, reader
fences, transfer completion, atomic mapping publication and rollback. The proposed
backend transaction prepares and reserves without publishing; commit requires
completed transfers and a scheduler-owned reader fence; abort retains the prior
mapping. Stable addresses alone do not protect in-flight readers.

Temporary staging and long-lived promotion need separate operations and cost
measurements. Compare quiescent rotation with overlapped staging. Preserve every
selected expert's computation and checkpoint precision, and provide static
residency as a fallback. Runtime compilation, graph recapture and steady-state GPU
allocation remain prohibited. Additional captured operation families require the
approval specified in [repository guidance](../AGENTS.md).

Active residency and champion/challenger operation are unimplemented. Admission
requires hardware calibration, correctness validation and reproducible evidence
of positive benefit after overhead; there is no fixed speedup threshold.
Leaderboard changes do not authorize automatic promotion to active control.

## Research basis

The expert-residency work in
[b12x PR #426](https://github.com/local-inference-lab/b12x/pull/426) reports GB300
graph replay, bitwise all-HBM parity, mapping updates, quiescent exchanges and
journaled rollback. The PR is unmerged. Adaptive exchange performance was not
qualified, and its HBM path was slower than the comparison FlashInfer kernel.
Its ownership and synchronization mechanisms inform backend design; Flare's
kernel and movement costs require independent measurement.

[ProMoE](https://arxiv.org/abs/2410.22134) studies intermediate-feature prediction
and proactive fetching. [MoE-Infinity](https://arxiv.org/abs/2401.14361) studies
activation-aware caching in single-user settings.
[SpecMD](https://proceedings.mlr.press/v306/hoang26d.html) studies policy interactions
and limitations of temporal-locality assumptions. Applying these methods requires
matching feature availability, hardware costs and workload conditions.

Per-layer temporal locality and batch-level demand are measurable with aggregate
routing traces. Cross-layer prediction within a graph, speculative lookahead and
learned providers require richer inputs with explicit availability times. Compare
those providers against statistical baselines on held-out traffic, including
workload shifts, and charge feature extraction and execution overhead. The
recoverable opportunity between static residency and an oracle determines whether
more complex prediction and movement mechanisms are justified.

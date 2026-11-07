# Parallax: predictive expert residency

Parallax evaluates whether forecasting Mixture-of-Experts (MoE) demand can make
expert weight placement cheaper than static residency or reactive caching. A
useful forecast must identify demand early enough to complete a transfer, and the
execution savings must exceed prediction, movement and synchronization costs.

The implementation provides deterministic offline replay, statistical predictors,
independent residency simulations and a native routing feed for shadow evaluation.
Shadow evaluation runs predictors against serving observations while leaving
physical expert placement unchanged. GLM-5.3-Flash preparation includes optional
routing instrumentation for single-GPU decode, prefill and MTP verification. Instrumented
LIL GLM serving has been exercised on GB300 with routing capture and exact offline
replay. Full-model numerical/state qualification and sustained high-concurrency
overhead qualification remain outstanding. The [native serving guide](parallax-demo.md)
describes in-process providers, the serving dashboard and CPU-only replay. The
[research guide](parallax-research.md) records measured findings, the native
predictor contract, a canary implementation and workload examples.

An experimental storage backend implements GB300 Grace/HBM allocation, staging
and quiescent virtual-address remapping in Helios. Explicit residency configuration
connects it to package loading and native graph execution. An optional economic
or rank-based policy submits real exchanges from one selected predictor; other contenders keep
independent shadow state. GB300-to-RTX transfers and remote expert execution are
outside the implemented path. Startup calibration measures idle pinned-host
copies in both directions. An observational graph-time model estimates cold-weight
costs during serving; it does not isolate all contention costs. A bounded fixed-MTP3
forced-offload comparison observed 18.8% higher request throughput with EWMA-driven
movement. General serving gains and full correctness qualification remain
unestablished; the [findings and limitations](parallax-research.md#what-the-gb300-experiments-establish)
define that result's scope.

## Component boundaries

Prediction, placement, execution and measurement have separate owners. The
offline schemas live in
[`scripts/parallax/contracts.py`](../scripts/parallax/contracts.py). Native
[prediction contracts](../helios/src/server/parallax/prediction.rs) and
[residency contracts](../helios/src/parallax/residency/mod.rs) keep provider output
separate from physical execution.

| Component | Contract | Responsibility |
| --- | --- | --- |
| Prediction provider | Eligible observation history → demand forecast | Predict utilization; identify the provider, configuration and compatible model |
| Placement policy | Forecast, residency snapshot and hardware costs → movement plan | Decide whether an exchange is economical, including choosing no movement |
| Residency orchestration | Movement plan → admitted transaction and publication outcome | Validate capacity, generations, deadlines and reader safety; recover from failure |
| Residency backend | Admitted transaction and scheduler fence → physical movement | Own storage, copies and mapping publication under explicit hardware capabilities |
| Observatory | Observations, forecasts and residency outcomes → common measurements | Score predictions and compare independent simulations |

The native live provider registry, runtime settings, independent policies and
evaluation workers are implemented in `helios/src/server/parallax/`. Runtime
selection uses `flare run --parallax-routing --parallax-profile PATH`, and the
serving dashboard includes a Parallax pane. Predictors and independent virtual
residency run for the server lifetime. Bounded trace retention and rolling
prediction statistics do not reset placement state or stop providers. Startup
copy calibration supplies measured transfer costs; cold execution costs remain
profile inputs. Evaluation version 2 provides per-layer ranking coverage, a greedy
batch placement policy and a ledger separating cold-execution savings, eviction
losses and full copy costs. Predictor runtime is reported separately from modeled
route savings. See the [native workflow](parallax-demo.md).

The offline predictor and policy protocols have working implementations in
[`predictors.py`](../scripts/parallax/predictors.py) and
[`policy.py`](../scripts/parallax/policy.py). Native storage contracts and the
transaction controller live in
[`helios/src/parallax/residency/`](../helios/src/parallax/residency/mod.rs), below
the server-specific prediction and evaluation workers. The experimental GB300
implementation requires the `runtime-package` feature and uses existing CUDA
driver bindings. It adds no dependency or separate process. The simulator never
calls a physical backend.

Model computation stays with Flare's prepared FlashInfer/b12x kernels. Residency
backends describe storage mechanisms; they do not select interchangeable GEMM
implementations. Helios owns shared scheduling and native execution. Python
providers used by the offline research tools remain outside serving. Live statistical
providers are compiled Rust plugins on bounded background threads inside Flare;
they receive no inference callbacks and add no Python serving dependency.

Selene owns prefix and KV reuse. Parallax's simulated HBM budget reserves space
for experts after KV, graphs and workspace. Physical admission must account for
all of those allocations and any staging capacity.

## Observation and prediction contracts

An expert has a canonical `(layer, expert)` identity independent of its physical
slot. The model catalog records architecture, a checkpoint/representation
fingerprint, sorted canonical identities and corresponding byte sizes. Sizes
include every weight plane and scale that must move with an expert.

The routing trace format in [`trace.py`](../scripts/parallax/trace.py) is
versioned JSONL. Its header contains model metadata, clock and workload identities,
and an explicit `synthetic` flag. Each observation contains an execution sequence,
demand time, consumer availability time, expert demand counts, execution phase,
batch size and concurrency. Times are integer nanoseconds in one trace-relative
monotonic domain. Availability cannot precede demand.

Trace schema version 1 requires contiguous execution sequences and ordered demand
and availability times. A gap ends an evaluation window; disjoint windows require
separate evaluator state. Counts describe executed routes and exclude padding and
inactive rows. Native speculative observations use phase `verify` and include
executed target routes for rejected candidates. Draft-layer experts are excluded
from the target catalog and remain resident. Prompt text, token IDs, hidden states, routing
weights and request identifiers are absent.

A provider receives an immutable bounded history whose observations were
available at the prediction point. Its forecast identifies the model, provider
version and configuration, observation cutoff, target frame, anticipated demand
time, expiration, score meaning and optional confidence. Forecast schema 2 targets
one future routing frame at a configured positive offset from the cutoff. Schema 1
records remain readable and must target the immediately following frame.
Anticipated demand time comes from a separate configured time horizon, without
consulting the target frame's recorded demand. The trace transport remains version 1.

Scores may represent expected counts, probabilities or rankings. Rankings are
ordered scores, with no implied probability calibration. Arbitrary target
windows, feature availability manifests and desired-placement recommendations
require contract extensions.

A prediction record adds observation availability, execution start, consumer
receipt, computation duration and a measured-or-modeled timing label. Queue delay
is the interval from observation availability to execution start. Usable lead is
demand time minus consumer receipt; a negative value means the forecast arrived
after demand. Receipt includes IPC and collector delays. `forecast_precision`, `forecast_recall`
and `forecast_demand_coverage` score every matched target, including late or expired
forecasts. Their sample count is `forecast_target_frames`. The existing `precision`,
`recall` and `demand_coverage` fields measure timely, unexpired coverage over the
evaluation traffic. Reports separate observation age,
queueing, computation, post-computation publication and total prediction latency.
A fixed model catalog is installed once per live worker; each job transmits only
eligible observations and target settings. Attachment calibrates clocks with a
metadata-only HTTP response, excluding retained histogram transfer time.
Resource counters distinguish collection CPU/wall time from final evaluation,
report child-process CPU time and resident memory, and do not sum child peak
RSS values into a concurrent memory estimate. Linux collector RSS and its peak
come from `/proc/self/status`; raw `rusage` high-water memory is reported separately
because it can include the launching process's pre-exec address space.

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
[`evaluate`](../scripts/parallax/replay.py) separately for each policy.

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
[`virtual residency engine`](../scripts/parallax/simulator.py) validates the whole
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
| Precision, recall and coverage | Raw accuracy on matched targets and separate timely coverage; K applies across a frame's catalog |
| Missing, late and expired forecasts | Zero operational coverage with their demand retained in the denominator |
| Predictor timing | Compute, queue, publication and total latency p50/p95/p99; usable-lead p50/p95 |
| Actionable rate | Fraction of routing frames with enough lead to transfer at least one correctly predicted expert; queue and capacity feasibility are evaluated separately by residency simulation |
| Residency effectiveness | HBM hits, cold demand, useful completed prefetches, late pending demand, promotions, evictions, bytes moved, transfer/synchronization cost and reuse after eviction |
| Unused transfers | Unused promotions become wasted on eviction; unused and pending promotions at trace end are censored |
| Workload strata | Layer, execution phase, batch size, concurrency and anticipated horizon |
| Sample support | Routing and prediction counts; confidence intervals are unavailable without independent workload repeats |

Reuse-after-eviction counts have no near-term cutoff. Busy workers skip prediction
points independently; skipped targets remain in operational coverage. Forecast
stability, uncertainty calibration, GPU contention measurement and block-bootstrap
confidence intervals are not implemented. CPU accounting and process RSS are
reported separately from serving throughput.

The net-value estimate subtracts modeled cold-route, transfer and synchronization
costs from the matching static-residency cost, then subtracts prediction release
delay. It charges full copy occupancy even when copies overlap computation. This
additive route-cost objective does not model elapsed serving time or throughput.
Policy execution, instrumentation, memory consumption and resource contention are
not calibrated into the estimate. Measured offline predictor durations are stored
separately from the configured computation and queue delays used for deterministic
replay.

The offline controls in [`oracle.py`](../scripts/parallax/oracle.py) use future
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
[`routing counter`](../ignition/components/routing_trace.py) reads canonical expert
IDs and masks inactive packed rows. It writes a separate output without changing
IDs, routing weights, hidden states or cache state.

Preparation with `--parallax-routing` supports the `glm53` and `lil_glm53`
recipes on one GPU. It retains counter-free graphs and adds traced variants of
decode, prefill, scored prefill and complete speculative rounds. The native serving
option `--parallax-routing` requires complete tracing coverage for the selected
execution mode. Packages exported with only ordinary traced variants still require
`--mtp-draft-tokens 0`; enabling MTP tracing requires a package with traced MTP
graphs. Mixed tracing fails admission.

Telemetry is disabled by default. Native startup selects counter-free graphs and
omits routing readback storage from the execution manifest; no routing records
are published. Enabling tracing selects the matching prepared variants and their
readback output without re-export, compilation or recapture. Selection rejects
missing counterparts or differences in graph geometry and inputs. The package
may retain counter buffers and additional graph resources when tracing is off;
these have a separate memory cost. GLM traced variants share their ordinary
counterparts' prepared operators, metadata, sampling and scratch. Separate decoder
and MoE wrappers bind counters without modifying ordinary forwards. Serial graph
execution permits that storage reuse; it does not permit concurrent replay.

CPU tests cover catalog selection, compatibility defaults, metadata and admission.
GB300 captured-counter and native two-frame tests passed, as did bounded C1/C32
serving checks with tracing off/on. Repeated synthetic-input C1 and C32 burst
measurements cover instrumentation overhead; C32 admission/timing variation is
unresolved. Full-model tensor/state equivalence and steady C32 overhead remain
unqualified. Switching within a running process is not implemented; it requires
a drained execution boundary and a fresh evaluation window.

The optional output `parallax_routes_v1` is specified in the
[execution manifest](../helios/manifest.md). It uses two persistent histogram
frames indexed by the executor's frame index and submission sequence. Native
[transport](../helios/src/execution/transport.rs) retains each readback until the
frame transfer fence completes. Reusable router scratch is not exposed to readers.
Unknown output versions are rejected. Stale or malformed histogram contents count
as invalid telemetry; CUDA transfer failures follow executor error handling.

The native [routing telemetry](../helios/src/server/parallax.rs) retains 32 records.
Its producer uses `try_lock` and records contention drops without waiting for an
HTTP reader. Retention eviction is visible as a sequence gap. The authenticated
[`GET /v1/parallax/routing?after=N`](api.md#experimental-routing-observations)
endpoint returns model/session identity, geometry, routing and submission
sequences, execution phase, batch size, histograms, timestamps and loss counters.
Concurrency is sampled from active requests at completion; the collector clamps
it to at least one. It is a workload-stratification proxy rather than dispatch-time
scheduler history.

Compiled native providers receive routing records through a separate bounded
512-record queue before HTTP-ring publication. They do not poll this endpoint and
do not depend on its 32-record retention. The native observatory records queue loss
or invalid routing as an incomplete window and suppresses comparative scoring.
Prediction and evaluation run on background threads inside Flare. The serving
TUI and `/v1/parallax/predictions` read the resulting snapshots.

### Python research collector

The following collector is an optional research tool, not Flare's live subsystem.
Its process workers and dependencies are not required by native serving. Use the
[native runtime profile](parallax-demo.md) for in-process prediction plugins.

Demand uses host dispatch before graph launch as a conservative lower bound.
Native readback availability is recorded separately. The same-host
[collector](../scripts/parallax/live.py) calibrates native-to-local clock offset
using an HTTP round trip and treats actual HTTP receipt as observation availability.
Polling, serialization and IPC therefore consume prediction lead time. The feed
provides no per-layer execution timestamps or within-graph lookahead.

The collector excludes work dispatched before attachment and rejects session
changes, clock reversals, sequence gaps and changes in loss counters. It saves raw
responses, normalized routing, prediction records and JSON comparisons. When one
response contains multiple observations, it retains all demand but schedules a
prediction point at the most recent observation and counts the coalesced points.

The Python collector runs trusted CPU providers in separate spawned processes managed by
[`ShadowPool`](../scripts/parallax/shadow.py). The pool bounds contender count,
history and catalog sizes, and permits one outstanding job per contender.
Submission does not wait for providers. Polling publishes results, enforces
budgets and terminates timed-out workers; startup counts toward the first job's
budget. Failed workers do not stop other contenders. Explicit removal and
registration replace a provider without restarting the pool.

Process isolation does not impose hard CPU, RAM or GPU quotas. Collection and
prediction can still contend with inference. Artifact admission and operating
system resource limits are required for untrusted or resource-intensive providers.

The Python research CLI compares recent-frequency and EWMA providers, each paired with static
and economic policies. One separate evaluation process reconstructs independent
virtual residency from immutable trace-prefix snapshots. At most one report is
outstanding; a busy evaluator skips leaderboard refreshes without stopping route
polling or predictor scheduling. Reports include the evaluated frame count,
evaluation CPU time and skipped refresh count. Frame and demand-entry limits bound retention;
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
`--hardware profile.json`. The options `--horizon-ns`, `--target-offset`, `--compute-ns`, `--history`
and `--top-k` configure prediction and evaluation. The time horizon anticipates
when demand will occur; the target offset selects which future frame is scored.
For example, `--target-offset 8 --horizon-ns 60000000` predicts frame cutoff + 8
with an expected demand time 60 ms after observation availability. Neither value
is inferred from future observations. `--compute-ns` supplies modeled
prediction cost. Programmatic evaluation also accepts measured worker records,
custom profiles, initial placements and explicit queue delays.

Native shadow evaluation requires the repository's
[preparation environment and native build](model-preparation.md#environment-setup),
an available GPU, an instrumented package and request traffic. Export and load
that package using the [preparation and serving commands](cli-design.md):

```sh
ignition export lil-glm-5.3-flash --gpus 0 --max-concurrency 32 \
  --prefill-chunk-size 8192 --mtp-draft-tokens 5 --speculative-token-capacity 192 \
  --language-model-only --expert-scale-compression native --parallax-routing
flare run PACKAGE --mtp-draft-tokens 3 --parallax-routing
python -m scripts.parallax.live --url http://127.0.0.1:8080 \
  --model-id EXACT_SERVED_MODEL_ID --catalog catalog.json \
  --hardware profile.json --workload held-out-workload-name \
  --output /tmp/parallax-live --seconds 60
```

To serve with telemetry disabled, omit `--parallax-routing` from `flare run`.
The same package uses counter-free MTP graphs when telemetry is disabled. Leaderboard
jobs have a separate `--evaluation-budget-ms` limit (15 seconds by default), with
one job in flight. A failed or over-budget evaluation ends the bounded collection
window, retains observations and predictions, and reports no ranking. Final
scoring uses the same budget; it never runs an unbounded fallback in the collector.

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

The tests in [`tests/test_parallax.py`](../tests/test_parallax.py) cover causal
boundaries, independent providers and policies, deterministic replay, atomic plan
rejection, byte capacity, pending reservations, transfer deadlines, late and
expired forecasts, censored transfers, oracle controls and worker failure.
[`tests/test_parallax_live.py`](../tests/test_parallax_live.py) covers collector
clock bounds, retention gaps, session compatibility and policy independence.

The captured-counter tests in
[`tests/test_parallax_capture.py`](../tests/test_parallax_capture.py) passed on an
RTX PRO 6000 Blackwell Max-Q and GB300 with direct and whole-forward-compiled capture. They
exercise alternating frames, inactive rows, invalid IDs and unchanged routing
IDs. These component tests do not establish full-model RTX compatibility or model
accuracy. Native tests cover output validation, histogram decoding
and bounded publication.

An instrumented GLM package requires real-request instrumentation checks, paired
output/state comparisons and measurement of capture and predictor overhead.
Model qualification remains separate. Preserve known baseline failures in reports;
Parallax acceptance tests whether instrumentation introduces additional changes
under identical inputs, starting state and graph schedules.

The native readback test in
[`tests/test_parallax_native.py`](../tests/test_parallax_native.py) uses synthetic
state and routing IDs to compare counter-free and traced execution. It exercises
two outstanding native frames, canonical layers, retained readbacks and invalid
telemetry without depending on model answers. Build Heliopause with its
`qualification` feature to expose the readback in test completions. Run these
tests only on an explicitly assigned idle GPU:

```sh
CUDA_VISIBLE_DEVICES=GPU_UUID FLARE_RUN_CUDA_TESTS=1 python -m pytest -q \
  tests/test_parallax_capture.py tests/test_parallax_native.py
```

Replace `GPU_UUID` with the assigned device's UUID. The compiled-counter tests
include the 32-row decode and 8192-row prefill geometries, with padded invalid IDs
that must be excluded from demand counts.

For GB300, use a GLM package that fits entirely in HBM as the matched reference.
Verify capacity including KV, graph storage and workspace. Vary simulated expert
HBM capacity while holding checkpoint precision, kernel representation, request
settings, KV budget, concurrency and graph capacities fixed. The all-HBM result
measures that implementation; it does not establish maximum hardware performance.
If offload requires a different weight representation, measure an all-HBM control
for that representation too.

Measure uninstrumented serving, telemetry-capable serving with telemetry disabled,
explicitly enabled instrumentation without an HTTP collector,
collection alone, and collection with predictors. Compare outputs and complete
state at fixed inputs. Verify disabled execution has no telemetry kernels, routing
readback or publication; report retained memory and measured residual overhead.
Record decode throughput, inter-token and tail latency,
TTFT, prefill performance and concurrency scaling. Physical-movement experiments
also require transfer bytes and duration, scheduler stalls and a matched static
baseline. Calibration and held-out workloads must be separate, with related
request continuations kept in the same split.

### Native residency transactions

The [`residency controller`](../helios/src/parallax/residency/controller.rs) owns
canonical expert locations, an authoritative generation, staging reservations and
reader leases. A versioned plan names the package/representation catalog, expected
generation, expiration time, promotions and evictions. Admission rejects foreign
catalogs, stale generations, expired or conflicting plans, unknown experts and
insufficient spare capacity before submitting copies. An empty plan does nothing.

Each expert consists of one or more immutable storage regions. The controller
stages all promoted regions into unused HBM slots; an eviction in the same plan
does not make its occupied slot available for staging. Publication requires
completed copies, zero outstanding reader leases and an external submission gate
that prevents concurrent graph launches. Completing a reader lease is unsafe:
the execution owner must first observe completion of every covered GPU read.
Dropping an unfinished lease retains the reader, preventing replacement and
storage release. A failed transaction restores all attempted mappings, including
the mapping whose replacement failed. Uncertain rollback or cleanup poisons the
controller, blocks readers and retains storage. Recovery requires the context
owner to establish safe teardown; continuing inference is prohibited.

The [`GB300 backend`](../helios/src/parallax/residency/gb300.rs) allocates immutable
Grace backing and a fixed pool of spare HBM slots at construction. Canonical
virtual addresses retain their prepared offsets. Private aliases support copying
into spare slots on a low-priority stream while existing mappings remain readable.
After a reader drain, the backend remaps canonical addresses to the selected
physical backing and reapplies access permissions. It also synchronizes the CUDA
context before publication; this does not replace the external submission gate.
CUDA virtual-memory access and mapping requirements are defined by the
[NVIDIA driver API](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-driver-api/group__CUDA__VA.html).

Regions must occupy distinct allocation-granularity pages. Admission rejects
overlapping rounded ranges and charges physical padding to backing, HBM and
transfer accounting. HBM slots are distributed evenly among the distinct rounded
region sizes; each mapping requires a slot of exactly that size. This avoids
subrange aliases of larger physical handles, which the qualified GB300 driver
rejected for unequal weight planes. Admission accounts for the actual slot sizes
and requires a compatible spare for every promoted region. No GPU allocation
occurs during a transaction. The backend
begins with all regions in Grace; an owner must establish initial HBM placement
before admitting serving readers.

CPU tests exercise spare-capacity admission, multi-region exchanges, transfer and
reader waits, expiration, cancellation, foreign catalog rejection, partial mapping
failure and failed rollback. They test the ownership protocol and shared rollback
algorithm, not CUDA behavior. An opt-in GPU test instantiates one memcpy graph
and checks exact synthetic bytes through repeated Grace/HBM exchanges without
re-instantiation. On GB300 SM103, the debug test passed with four 4,093-byte
regions, two parts per synthetic expert, and 2 MiB allocation granularity. All
payload bytes and canonical addresses remained unchanged across 22 commits:
initial promotion, 20 exchanges and final eviction. The test also verified that
an outstanding reader prevents publication. Each tier reserved 8 MiB of physical
storage. This is a correctness result, not a movement-performance measurement.
Run it only after acquiring an idle GB300, replacing `GPU_UUID` with its assigned UUID:

```sh
CUDA_VISIBLE_DEVICES='' cargo test --locked --offline \
  --manifest-path helios/Cargo.toml --features server,runtime-package \
  --target-dir target parallax --lib

CUDA_VISIBLE_DEVICES=GPU_UUID FLARE_RUN_CUDA_TESTS=1 cargo test --release \
  --locked --offline --manifest-path helios/Cargo.toml --features runtime-package \
  --target-dir target stable_graph_replay_across_grace_hbm_exchanges \
  --lib -- --ignored --nocapture
```

The storage tests also cover 8 MiB and 4 MiB weight planes with matching physical
slots, checking every byte after promotion and eviction. Their result establishes
storage and graph memcpy behavior only.

The separate
[`captured MoE test`](../tests/components/test_parallax_moe_residency_cuda.py)
uses target layer 3 of
`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD`, revision
`dec48abd33efa73c3bb7c95b74eee10cad34f9be`, with all 288 experts and top-8
routing. The two packed weight planes move through native storage while expanded
scales and calibration vectors remain in HBM. On GB300, the whole-forward-compiled
trtllm NVFP4 graph produced bitwise-identical outputs at 1, 32, 257 and 8,192 rows
in all-HBM, all-Grace and mixed placement, including three eight-expert exchanges
and restoration to HBM. The test checks all-HBM repeatability before comparing
tiers and checks that graph replay makes no PyTorch device allocations. It does
not recapture between placements.

These are isolated component results, not complete-model serving qualification.
The generic backend still advertises `cold_moe_execution_qualified: false`; byte
storage cannot qualify a caller's kernel catalog. The qualification-only
`ExpertResidencyTestStorage` Heliopause binding is unavailable without the
`qualification` build feature and is not a serving control interface.

The test harness observed roughly 4.7–5.4 ms wall time for an eight-expert exchange.
That includes staging, polling and mapping publication; it is not a serving
stall measurement. Modeling only transfer bandwidth omits this cost. Serving
integration must measure staging and publication separately, charge scheduler
pauses, and compare measured serving results against a matched static placement.

### Connecting storage to serving

The [`storage binding resolver`](../helios/src/parallax/residency/package.rs)
matches a versioned configuration to an authenticated package's prepared-weight
catalog. Configuration names weight views and maps their first tensor axis to
canonical expert IDs; it cannot supply addresses or replacement weight data.
Resolution checks package identity, single-rank geometry, contiguous extents,
complete page ownership, immutable-weight initializers and disjoint regions.
Pointer-bearing, dynamically sized and peer allocations are rejected. Scales and
calibration views omitted from the movement configuration retain ordinary HBM
ownership.

The MTP-instrumented LIL GLM package
`73afdd3add534c3473addaabe68c9fa88c19089b117623c6da4c26cb02c26172`
has a signed catalog that resolves 24,192 packed-weight
regions for 12,096 target experts, totaling 152,202,903,552 movable bytes, at 2 MiB
granularity. No export or kernel layout change was needed for this metadata
preflight. The native loader additionally verifies that these regions cover 84
entire immutable allocations before substituting Grace mappings.

To reproduce the metadata preflight, set `FLARE_PARALLAX_PACKAGE` to the
instrumented GLM package and `FLARE_PUBLISHER_PROFILES` to its trusted public-key
profile JSON. `FLARE_PARALLAX_STORAGE_CONFIG` optionally names the output file
for the resolved configuration. This opt-in test uses no GPU:

```sh
CUDA_VISIBLE_DEVICES='' cargo test --locked --offline \
  --manifest-path helios/Cargo.toml --features server,runtime-package \
  --target-dir target glm_package_storage_bindings_resolve_without_gpu \
  --lib -- --ignored --nocapture
```

`NativeResources` restores the selected weight allocations directly into Grace at
their captured addresses. The package owns virtual reservations; Parallax owns
physical mappings and a fixed HBM promotion pool. Normal authenticated payload
loading initializes writable Grace mappings, then seals them read-only before
initial placement and graph instantiation. Unselected allocations, including
scales and calibration vectors, retain ordinary ownership. Memory accounting
subtracts the moved allocations and includes the actual HBM pool when sizing KV.
Startup validates the physical size classes, initial placement and overall memory
limit before allocating Grace backing or the HBM pool.

The execution manifest carries an optional residency binding. Exactly one native
executor may claim it. Each submitted frame retains a reader lease until its
completion event. Copies can overlap readers; when copies finish, the executor
stops new submissions until existing readers complete and mappings publish.
Recoverable copy or publication failures disable movement and preserve static
storage. Unconfirmed mappings or readers stop execution and retain owners.
There is no steady-state GPU allocation or graph recapture.

`flare run --parallax-residency PATH` selects a version-1 JSON configuration:
`storage` contains the resolved storage binding configuration; `catalog_id` and
`canonical` must exactly match the predictor catalog; `hbm_slots`, `hbm_bytes`
and `backing_bytes` bound physical allocations; `initial_hbm` selects canonical
experts promoted before graph execution. Slots are individual weight parts, not
complete experts. Reserve spare slots of every required size for exchanges.
The implementation requires local TP1 SM103, `--parallax-routing`, a predictor
profile and complete tracing coverage for the selected execution mode. Reader
leases cover the entire speculative round, including proposal and accepted-state
commit. Publication occurs only after those readers drain.

Without an `active` entry in the predictor profile, serving retains the configured
initial placement while all contenders run in shadow. An optional `active` object
contains `predictor`, `max_promotions`, `interval_ns`, `hold_ns` and
`publication_ns`. The economic policy swaps equal-sized expert representations,
uses only spare staging slots, charges predicted eviction losses and copying,
and requires a positive aggregate benefit after the assumed transaction-wide
publication cost. It rejects movements that cannot meet forecast demand time.
An explicit `policy: "predictor_rank"` instead permits bounded ranking-driven
exploration without a cold-execution price. It requires a strictly higher-ranked
promotion and retains capacity, deadline and replacement limits. The default
`economic_exchange` behavior is unchanged. Both modes expose an independent
[actual residency ledger and empirical cost model](parallax-demo.md#experimental-gb300-residency).
The executor also cancels staged work once the target routing event has occurred, even if the
forecast's estimated time deadline has not expired.
`interval_ns` limits decisions to one per 0.1–60 seconds; `hold_ns` prevents rapid
replacement. Missing copy calibration, invalid observations or predictor failure
suppress proposals. The residency controller independently validates every plan.

The prediction endpoint and dashboard distinguish actual HBM/Grace counts and
mapping generations from shadow contender ledgers. Physical copy and publication
counters include startup placement. The endpoint and saved live comparison retain
the last 256 serving movement outcomes, their target events and timing, plus the
lifetime event count. Expired or cancelled copies are charged even when placement
never changes. Forecast gains and cold-execution penalties remain estimates; enabling active movement does not establish a serving speedup.

The GLM expert byte catalog is representation accounting, not a page-layout
manifest. Packed weights, expanded scales and per-expert calibration values need
separate verified bindings. Small calibration arrays may share pages across
experts; they must remain resident or use explicitly declared movement groups.
The storage backend rejects overlapping expert pages rather than remapping bytes
belonging to another expert. Any layout change must preserve the prepared kernel
contract and receive a matching all-HBM comparison.

Temporary staging and long-lived promotion need separate operations and cost
measurements. Compare quiescent rotation with overlapped staging. Preserve every
selected expert's computation and checkpoint precision, and provide static
residency as a fallback. Runtime compilation, graph recapture and steady-state GPU
allocation remain prohibited. Additional captured operation families require the
approval specified in [repository guidance](../AGENTS.md).

Active residency is experimental. A GB300 native serving check with the ordinary
tracing package `8597b2ad0229e33f9e0fb6c100a2e0b1b45cb8ce2a3523d8199bfc7cde2342e0`
used TP1, runtime MTP disabled, concurrency up to four, 8,064 HBM
experts and 4,032 Grace experts. Seven requests completed while 41 serving
transactions promoted and evicted 328 experts each, copying 4,127,195,136 bytes.
There were no movement faults, cancellations or rollbacks. Three sequential
requests matched the all-HBM reference exactly in output, token log probabilities,
finish reason and usage. This bounded check does not establish full-model state
equivalence or correctness for untested capacities.

The MTP-instrumented package identified above was tested on GB300 with fixed
three-draft execution, TP1, concurrency up to four and the same 8,064/4,032
HBM/Grace placement. Traced all-HBM, static split and active split runs each
matched the untraced all-HBM reference on three sequential requests: output,
token log probabilities, finish reason and usage were exact. A separate
8,192-token prefill with 16 output tokens also matched. Native readback tests
exercised verification widths 1, 4 and 6 across both execution frames, including
invalid telemetry, exact synthetic state and allocation stability. These checks
do not establish full-model state equivalence or long-context correctness.

With a two-second forecast horizon, the fixed-MTP3 static and active split runs
averaged 66.6 and 74.3 output tokens/s respectively across three four-request
coding cohorts of 256 output tokens per request. Untraced and traced all-HBM
runs averaged 464.5 and 500.7 tokens/s. Timing includes prefill and HTTP. Each
configuration ran once; scheduling and speculative acceptance varied, and no
confidence interval was established. The active split run completed 84 serving
exchanges, promoting and evicting 672 experts each, without movement faults or
rollbacks. These measurements demonstrate working movement during MTP serving;
they do not establish a tracing speedup or a reproducible policy advantage.

Adaptive five-draft execution exercised verification widths 1 through 6 and
completed 96 active exchanges without routing or movement faults. Its three
sequential responses differed from the untraced all-HBM reference. A fresh
untraced all-HBM repeat and a traced all-HBM control each differed on two of
those three requests. This
baseline variability prevents attributing the adaptive differences to Parallax;
it does not establish instrumentation equivalence. Adaptive MTP exact-response
qualification remains unresolved. Use fixed three-draft execution for the
bounded residency demonstration described above.

The exchange check assumed a synthetic cold penalty of 1 ms per use to exercise
physical movement. It establishes neither predictive value nor a serving speedup.

A separate EWMA `predictor_rank` comparison uses zero cold-cost input and measures
actual movement alongside an observational execution-cost ledger. The
[research findings](parallax-research.md#what-the-gb300-experiments-establish)
record its package, workload, static/active throughput and qualification limits.
The [ledger explanation](parallax-research.md#what-the-cost-ledger-means)
distinguishes estimated resource savings from measured serving time.

Production qualification requires measured cold-execution and publication costs,
broader correctness coverage and comparable static-placement serving benchmarks.
Controlled switching between active providers is unimplemented. Leaderboard
changes do not authorize automatic promotion to active control.

## Research basis

The live ledger's execution-cost fit is ordinary least squares, evaluated on
subsequent observations before updating its coefficients. It does not provide
confidence intervals or a causal estimate of serving improvement.
[River's online Bayesian linear regression](https://riverml.xyz/dev/api/linear-model/BayesianLinearRegression/)
is an external reference for adding predictive uncertainty and forgetting;
River is not a serving dependency. Uncertainty estimates still require validation
against held-out traffic and correlated execution observations.

Controlled serving comparisons must account for persistent residency and cache
state. [Switchback experiment research](https://proceedings.mlr.press/v267/wen25d.html)
analyzes carryover and temporal correlation when alternating policies.
[Vowpal Wabbit's off-policy evaluation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/9.11.2/tutorials/off_policy_evaluation.html)
provides references for randomized action logging and propensity-based estimates.
The deterministic ranking policy does not collect the action probabilities or
exploration coverage needed for those estimators. Neither method is implemented
by the live ledger.

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

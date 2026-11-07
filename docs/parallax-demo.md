# Parallax in Flare

Parallax runs compiled prediction plugins, shadow evaluation and optional expert
residency control inside the Flare serving process. A runtime profile selects
providers and their settings.
Recent-observation, EWMA and frozen-frequency predictors run on background threads with bounded queues;
model execution never waits for them. Ordinary serving starts no prediction workers.
The live subsystem needs no Python environment, dynamically loaded plugins or
separate prediction service.

## Observe serving traffic

Prepare a single-rank telemetry-capable package as described in
[Parallax](parallax.md), then create a runtime profile:

```json
{
  "version": 1,
  "catalog": "/absolute/path/to/package-bound-expert-catalog.json",
  "hardware": "/absolute/path/to/hardware-costs.json",
  "output": "/absolute/path/to/new-evaluation-directory",
  "predictors": [
    {"name": "recent", "config": {}},
    {"name": "ewma", "config": {"alpha": 0.5}},
    {"name": "frequency", "config": {"warmup_frames": 128}}
  ],
  "target_offset": 8,
  "horizon_ns": 60000000,
  "budget_ns": 1000000000,
  "max_frames": 2048,
  "evaluation": {"version": 2, "top_k_per_layer": 32, "max_promotions_per_forecast": 128}
}
```

The expert catalog contains `identity`, `architecture`, canonical `experts`, and
corresponding `expert_bytes`. Its identity must end with `;package=<package_id>`,
where the package ID comes from authenticated `flare inspect --json` output.
Hardware fields are `identity`, `hbm_bytes`, `backing_bytes`,
`bandwidth_bytes_per_ns`, `transfer_latency_ns`, `synchronization_ns`, and
`cold_cost_ns`. Label uncalibrated inputs explicitly. The demo's input profile
uses synthetic costs; startup calibration replaces its copy estimates.

Launch the normal native server with the profile and explicit telemetry selection:

```sh
flare run /path/to/package --mtp-draft-tokens 0 \
  --parallax-routing --parallax-profile /path/to/profile.json
```

Use Tab or the arrow keys to select **Parallax** in Flare's serving dashboard.
Predictions are evaluated against actual client traffic. The authenticated
`GET /v1/parallax/predictions` endpoint exposes the same structured comparison
when serving without a terminal. Reading it never activates telemetry.

Predictors run until serving stops. Prediction-quality statistics use a rolling
window bounded by `max_frames` (at most 2,048) and 2,000,000 demand/forecast entries.
All contenders share the same retained observation window. Each contender's
virtual hot/cold residency, pending transfers, movement counts and hit/miss totals
persist for the server lifetime. Reaching a retention limit does not stop a
predictor or reset its residency. The dashboard shows both measurement scopes.

A separate trace retains the initial complete prefix within the same bounds.
Admission reserves space for pending predictor replies. On graceful shutdown,
`capture.json` and `comparison.json` describe that replayable prefix;
`live-comparison.json` describes the continuous evaluation. Lifetime results
beyond the retained prefix cannot be reproduced from that prefix alone. Worker
counters in both reports cover the server lifetime. Queue
loss or invalid observations invalidate continuous comparison; they do not
invalidate an already complete trace prefix. Live provider reconfiguration,
continuous archival capture and production overhead qualification remain
outstanding. Contender ledgers always retain independent virtual state. An
explicitly selected active policy also submits real movement plans, as described
below.

With a Parallax profile, Flare measures pinned-host/HBM copies before opening the
server. It allocates 16 MiB each of host and GPU scratch space, warms both copy
directions, and takes at most 18 samples at 64 KiB, 1 MiB and 16 MiB. It stops
issuing samples after one second and accepts calibration only within a three-second
wall-time budget. An in-flight CUDA call must finish before its memory is freed;
this budget cannot interrupt a hung driver. Ordinary serving without a Parallax
profile performs no calibration.

The fitted host-to-device latency and bandwidth replace the profile's transfer
costs. Synchronous timings already include API and synchronization overhead.
Both directional fits and raw samples are recorded in `hardware.transfer_calibration`.
Calibration failure suppresses placement recommendations while observation
continues. Costs for direct coherent Grace execution, cold expert computation,
and contention during inference are not measured by this benchmark. The profile's
cold-execution penalty remains an explicit assumption. Read-only expert eviction
discards an HBM copy; it does not require a device-to-host writeback.

`target_offset` selects the future routing frame; `horizon_ns` separately estimates
when it will be needed. Providers receive only eligible observation history.
Prediction availability includes scheduling and delivery delay. A timed-out or
failed provider is disabled for the session. In-process plugins are trusted code:
a hung thread cannot be forcibly terminated safely, so shutdown does not join it.
No replacement thread is spawned for a disabled provider.

## Run the bounded demonstration

`flare parallax` starts an owned loopback Flare server, submits three ordinary
128-token requests, displays a TUI, drains the server and verifies saved predictions
with native offline replay. It uses the same in-process subsystem as `flare run`.
It does not execute a second model or invoke Python.

Create an experiment JSON containing absolute `binary`, `package`, `catalog` and
`hardware` paths, the authenticated `package_id`, the assigned `gpu` UUID and an
unprivileged `port`. The binary must identify the running Flare executable.
Reserve the GPU with other users before launching; idle guards check occupancy
but do not grant ownership.

```sh
flare parallax --experiment /path/to/experiment.json \
  --output /path/to/new-demo-directory
```

Omit `--output` for a fresh temporary directory. `--requests 1` through
`--requests 6` changes the workload size; `--plain` prints status without a TUI.
Tab and arrow keys switch between comparison, timing and generated responses.
Press `q` to stop requests and drain. The completed result remains visible until
`q` or Enter closes the display. Errors appear inside the TUI and retain their
underlying cause and evidence path.

Demonstration prompts are artificial; routing observations come from real
inference. Generated text is saved in `responses.json`. The `launch.json` record
contains the owned PID and exit status. `demo.json` is complete only after
nonempty routing, predictions from both providers, clean server shutdown and
exact replay agreement. Failure to drain leaves the owned server running for
inspection and reports its PID. No unrelated process is stopped.

## Interpret and replay

Evaluation version 2 scores positive forecast rankings per layer. The profile's
`top_k_per_layer` selects the displayed coverage; the structured report also
includes precision, recall and timely demand coverage at 8, 16, 32 and 64 ranks
per layer, plus a ranking constrained by the HBM byte budget. Coverage measures
prediction arrival and expiry. It does not claim transfer completion or profitable
placement. Increasing K trades precision for recall; coverage alone is not a winner
criterion.

The economic policy consumes the entire forecast and proposes a greedy batch of
positive-value promotions, up to `max_promotions_per_forecast`. Every proposal
reserves HBM space, charges the selected victim's forecast demand, and advances
the shared serialized transfer queue. A transfer must fit the declared forecast
deadline. A promotion can use free space or evict one sufficiently large resident
expert. The planner does not combine several smaller victims to accommodate one
larger expert, and it is not an optimal placement solver.

The serving table exposes a lifetime modeled ledger relative to static residency:

| Column | Meaning |
| --- | --- |
| Saved ms | Demand served by completed promotions that would have been cold under static residency, multiplied by the assumed cold penalty |
| Evict ms | Demand made cold by evicting an initially resident expert, multiplied by that penalty |
| Copy ms | Full modeled transfer and synchronization cost of proposed promotions |
| Balance | Saved − eviction losses − copy costs; positive values favor the simulated policy under these assumptions |
| Moves | Promotions / evictions |
| HBM hit% | Fraction of demand served from virtual HBM |

Copy time is charged in full, even when transfers could overlap computation.
Predictor runtime is reported separately; asynchronous queueing and delivery delay
reduce lookahead but are not charged as inference stalls. CPU/GPU contention and
end-to-end overlap remain uncalibrated. The balance is an illustrative route-cost
ledger, not measured wall time saved or a throughput prediction. Cold-execution
penalties remain explicit assumptions despite measured startup copy costs.

Prediction coverage, median usable lead time, median predictor duration and worker
drops appear below the ledger. They cover the recent statistics window; residency
counters cover the full serving lifetime. The `frequency` provider averages counts
from its first `warmup_frames` distinct eligible observations and then freezes.
Its early forecasts include the fitting period; assess the frozen contender after
that period has left the statistics window. Static residency remains a placement
control, separate from this frequency predictor.

Profiles without `evaluation` retain version 1 scoring (eight global ranks), its
single-promotion policy and legacy cost arithmetic for exact historical replay.
They do not silently adopt version 2 semantics. New experiments should explicitly
select version 2.

After graceful serving shutdown, the output directory contains a versioned native
`capture.json` and `comparison.json`. Replay needs only the Flare binary:

```sh
flare parallax --replay /path/to/evaluation-directory/capture.json
```

Replay rejects incomplete captures and verifies the saved comparison. Native
capture version 1 contains the profile, observations, forecasts and worker counts.
The Python JSONL trace tools remain available for offline research; they are not
a runtime dependency and use a separate trace schema.

## Build-time extension

Implement the Rust `Predictor` trait and register its factory in
`Registry::builtins` under `helios/src/server/parallax/prediction.rs`. Configuration
is validated before worker startup. The observatory owns scoring; the provider
returns demand scores and cannot choose its own evaluation method. Placement
policies and virtual residency live separately in `evaluation.rs`.

The [predictor research guide](parallax-research.md#native-predictor-contract)
specifies the timing and score contract and walks through `uniform_canary`, a
compiled provider that assigns equal scores to every expert. It also supplies
[authored workload commands](parallax-research.md#authored-workloads) for coding,
recorded tool histories, prose, chat, long-context retrieval and workload shifts.

A future neural-provider client can implement the same trait while contacting a
separate GPU inference process. No such backend is compiled into this prototype.
Transport latency must count against usable lead, and service failure must never
block model execution. The experimental GB300 residency path described below
executes physical movement inside the native serving process.

## Validation

CPU regressions cover continuous prediction beyond trace limits, preservation of
virtual residency across rolling statistics windows, independent policies, exact
prefix replay, provider timeouts, default-off telemetry, directional cost fitting
and observation-only behavior after failed calibration:

```sh
CUDA_VISIBLE_DEVICES='' cargo test --locked --offline \
  --manifest-path helios/Cargo.toml --features server,runtime-package \
  --target-dir target parallax --lib
```

The ignored `gpu_transfer_calibration_finishes_within_startup_budget` test requires
an explicitly reserved GPU selected through `CUDA_VISIBLE_DEVICES`. It allocates
only scratch buffers and checks copy verification, both directional fits, the
sample-count bound and wall time. It does not load model weights.

An evaluation-version-1 GB300 release serving check with the instrumented LIL GLM package, TP1 and MTP
disabled measured 8.5 ms for startup calibration. Ten artificial prompts produced
1,291 real routing frames. Both providers completed all 1,291 forecasts without
busy drops, failures or timeouts. Initial trace retention filled at 805 frames;
continuous prediction reached the end with an 832-frame statistics window and
unchanged lifetime residency accounting. The retained prefix replayed exactly.
These checks validate instrumentation and lifecycle behavior, not model accuracy,
predictive generalization, production overhead or physical residency performance.


To compare expanded scoring and placement against the same recorded forecasts,
write an evaluation JSON such as
`{"version":2,"top_k_per_layer":32,"max_promotions_per_forecast":128}` and run:

```sh
flare parallax --replay /path/to/capture.json \
  --evaluation-config /path/to/evaluation.json --plain
```

This mode validates the original information boundaries and preserves every
forecast, timestamp and hardware cost. It does not overwrite the capture or claim
agreement with its historical comparison. Adding a predictor requires collecting
its forecasts separately; reevaluation cannot invent a frequency contender for an
existing trace.

## Experimental GB300 residency

A signed package with page-isolated expert weight allocations can use
`--parallax-residency /absolute/path/to/residency.json` alongside the profile,
with ordinary or MTP execution selected through the usual serving flags. MTP
tracing requires a package containing paired traced MTP prefill and speculative
graphs; ordinary-only tracing packages reject that selection. The
[storage configuration contract](parallax.md#connecting-storage-to-serving)
defines package binding, capacity and initial placement. This changes physical
storage: declared packed weights have immutable Grace backing and selected
experts have HBM copies. Scales omitted from that configuration remain in HBM.

Launch residency through the normal serving command, using the Flare binary built
with Parallax and the package-bound configuration files:

```sh
flare run /path/to/package --gpus 0 --mtp-draft-tokens 3 \
  --no-mtp-adaptive-verification \
  --parallax-routing \
  --parallax-profile /path/to/profile.json \
  --parallax-residency /path/to/residency.json
```

Select **Parallax** in the serving dashboard. This command owns prediction,
evaluation, physical residency and HTTP serving in one process. No demonstration
launcher is required. The profile's output directory must not already exist;
select a fresh capture path for each run. GPU selection and memory limits follow
the normal `flare run` options.

MTP observations count target verification work, including candidates later
rejected by sampling. They use phase `verify`. Draft-layer experts remain resident
and are excluded from the target catalog. The executor retains residency reader
leases through each complete speculative graph, including its state commit;
weight mappings cannot change while a round is in flight.

The bounded GLM residency checks used fixed three-draft execution. Adaptive
five-draft serving exercised tracing and exchanges, but exact-response
qualification remains unresolved because untraced repeats also varied. See the
[serving measurements](parallax.md#connecting-storage-to-serving) for the tested
package, comparison scope and limitations.

The dashboard's **Actual weights** line and `physical_residency` response describe
real storage. Predictor comparison rows still describe independent simulations.
`commits_including_startup` counts published transactions, not individual experts;
`staged_copy_bytes_including_startup` and `publication_host_ns_including_startup` include
initial placement. A changed generation establishes successful mapping publication,
not a throughput improvement. `cancelled_transactions` and bounded
`movement_events` distinguish expired staging from successful publication; the
lifetime `movement_event_total` exposes history truncation. These records are
also saved in `live-comparison.json`.

To let one registered provider propose exchanges, add an `active` object to its
profile, for example:

```json
"active": {
  "predictor": "ewma",
  "policy": "predictor_rank",
  "max_promotions": 8,
  "interval_ns": 500000000,
  "hold_ns": 2000000000,
  "publication_ns": 6000000
}
```

`predictor_rank` is bounded experimental exploration: it exchanges a lower-ranked
resident expert for a higher-ranked cold expert without assigning a price to a
cold selection. Capacity, spare slots, deadlines, hold time and interval limits
still apply. This mode can make serving slower. The example publication allowance
is an assumption used for deadline admission; measured costs appear separately.
Omitting `policy` retains `economic_exchange`, which requires positive expected
savings under the profile's cold-execution penalty.

With physical storage enabled, the dashboard displays the actual residency
ledger. GPU-timed copies and host mapping publication exclude startup placement.
The submission-gate interval includes useful work by draining readers; it is not
a measurement of idle GPU time. Counts compare the placement used by each graph
against the initial static placement. A resident promotion still awaiting use is
unresolved, not yet wasted.

The basic execution-cost model fits GPU graph duration from total and cold
distinct expert-weight extents, separately for each graph, batch size and valid
target-row count. It retains at most 256 uncontended samples per shape, with 512
shapes maximum. Movement-overlapping frames do not train the fit. Estimates use
only earlier observations; fewer than 32 samples or insufficient independent
variation produces `learning`, not a fabricated saving. These associations do
not establish causal savings: routing composition and memory interference remain
possible confounders.

The estimated accounting balance credits only frames with an available model
and charges all completed GPU copy time plus host publication time. Reader drain
is reported separately because it can contain useful inference. This deliberately
charges overlapped copying in full; it is resource accounting, not an end-to-end
performance bound. Predictor overhead and copy-induced compute slowdown are not
attributed. `actual_ledger` in the prediction endpoint and saved comparison
contains model coverage, prediction error on subsequent samples and 512 recent
frame measurements. Routing or movement-history gaps invalidate the ledger.

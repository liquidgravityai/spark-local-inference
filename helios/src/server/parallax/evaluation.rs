//! Common scoring and independent virtual residency; all cost values are estimates.

use super::prediction::{Expert, Forecast, Observation, Profile};
mod placement;
mod coverage;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct WorkerCounts {
    pub submitted: u64,
    pub completed: u64,
    pub dropped_busy: u64,
    pub failures: u64,
    pub timeouts: u64,
    pub disabled: bool,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureLimit {
    Frames,
    DemandScores,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Capture {
    pub native_capture_version: u32,
    pub profile: Profile,
    pub observations: Vec<Observation>,
    pub forecasts: BTreeMap<String, Vec<Forecast>>,
    pub workers: BTreeMap<String, WorkerCounts>,
    pub error: Option<String>,
    #[serde(default)]
    pub limit_reason: Option<CaptureLimit>,
}
/// Policies decide independently of predictor internals and physical execution.
pub trait PlacementPolicy {
    fn decide(
        &self,
        forecast: &Forecast,
        state: &VirtualResidency,
        profile: &Profile,
    ) -> Option<Movement>;
    fn plan(&self, forecast: &Forecast, state: &VirtualResidency, profile: &Profile) -> Vec<Movement> {
        self.decide(forecast, state, profile).into_iter().collect()
    }
}
pub struct Movement {
    pub promote: Expert,
    pub evict: Option<Expert>,
}
pub struct Static;
impl PlacementPolicy for Static {
    fn decide(&self, _: &Forecast, _: &VirtualResidency, _: &Profile) -> Option<Movement> {
        None
    }
}
pub struct Economic;
fn transfer(profile: &Profile, expert: Expert) -> u64 {
    let size = profile.model.expert_bytes[profile.model.experts.binary_search(&expert).unwrap()];
    (size as f64 / profile.hardware.bandwidth_bytes_per_ns).ceil() as u64
        + profile.hardware.transfer_latency_ns
        + profile.hardware.synchronization_ns
}
impl PlacementPolicy for Economic {
    fn plan(&self, f: &Forecast, state: &VirtualResidency, p: &Profile) -> Vec<Movement> {
        if let Some(config) = &p.settings.evaluation {
            placement::plan(f, state, p, config.max_promotions_per_forecast)
        } else { self.decide(f, state, p).into_iter().collect() }
    }
    fn decide(&self, f: &Forecast, state: &VirtualResidency, p: &Profile) -> Option<Movement> {
        if p.hardware.transfer_calibration.as_ref().is_some_and(|c| !c.usable()) {
            return None;
        }
        let scores: BTreeMap<_, _> = f.scores.iter().copied().collect();
        let candidate = f
            .scores
            .iter()
            .filter(|(e, s)| {
                !state.resident.contains(e)
                    && !state.pending.iter().any(|(_, x)| x == e)
                    && *s * p.hardware.cold_cost_ns as f64 > transfer(p, *e) as f64
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        let e = candidate.0;
        let copy = transfer(p, e);
        if state.copy_until.max(f.completed_ns).saturating_add(copy) > f.expected_demand_ns {
            return None;
        }
        let size = p.model.expert_bytes[p.model.experts.binary_search(&e).unwrap()];
        if state.reserved + size <= p.hardware.hbm_bytes {
            return Some(Movement {
                promote: e,
                evict: None,
            });
        }
        let victim = state
            .resident
            .iter()
            .filter(|v| state.sizes[v] >= size)
            .min_by(|a, b| {
                scores
                    .get(a)
                    .unwrap_or(&0.)
                    .total_cmp(scores.get(b).unwrap_or(&0.))
            })?;
        if (candidate.1 - scores.get(victim).unwrap_or(&0.)) * p.hardware.cold_cost_ns as f64
            <= copy as f64
        {
            return None;
        }
        Some(Movement {
            promote: e,
            evict: Some(*victim),
        })
    }
}
pub struct VirtualResidency {
    resident: BTreeSet<Expert>,
    initial: BTreeSet<Expert>,
    cold_avoided: u64,
    eviction_misses: u64,
    pending: Vec<(u64, Expert)>,
    sizes: BTreeMap<Expert, u64>,
    reserved: u64,
    copy_until: u64,
    hits: u64,
    misses: u64,
    promotions: u64,
    evictions: u64,
    bytes: u64,
    cost: u64,
}
impl VirtualResidency {
    fn new(p: &Profile) -> Self {
        let sizes: BTreeMap<_, _> = p
            .model
            .experts
            .iter()
            .copied()
            .zip(p.model.expert_bytes.iter().copied())
            .collect();
        let mut reserved = 0;
        let resident: BTreeSet<_> = sizes
            .iter()
            .filter_map(|(e, s)| {
                if reserved + *s <= p.hardware.hbm_bytes {
                    reserved += *s;
                    Some(*e)
                } else {
                    None
                }
            })
            .collect();
        Self {
            initial: resident.clone(), cold_avoided: 0, eviction_misses: 0,
            resident,
            pending: vec![],
            sizes,
            reserved,
            copy_until: 0,
            hits: 0,
            misses: 0,
            promotions: 0,
            evictions: 0,
            bytes: 0,
            cost: 0,
        }
    }
    fn advance(&mut self, time: u64) {
        let mut ready = vec![];
        self.pending.retain(|(at, e)| {
            if *at <= time {
                ready.push(*e);
                false
            } else {
                true
            }
        });
        self.resident.extend(ready);
    }
    fn apply(&mut self, m: Movement, now: u64, p: &Profile) {
        let Some(&size) = self.sizes.get(&m.promote) else { return; };
        let freed = match m.evict {
            Some(v) if self.resident.contains(&v) && v != m.promote => self.sizes[&v],
            Some(_) => return,
            None => 0,
        };
        if self.resident.contains(&m.promote)
            || self.pending.iter().any(|(_, e)| *e == m.promote)
            || self.reserved - freed + size > p.hardware.hbm_bytes { return; }
        if let Some(v) = m.evict {
            self.resident.remove(&v);
            self.reserved -= freed;
            self.evictions += 1;
        }
        let cost = transfer(p, m.promote);
        self.copy_until = self.copy_until.max(now).saturating_add(cost);
        self.pending.push((self.copy_until, m.promote));
        self.reserved += size;
        self.promotions += 1;
        self.bytes += size;
        self.cost += cost;
    }
    fn demand(&mut self, o: &Observation) {
        self.advance(o.demand_ns);
        for (e, n) in &o.demands {
            if self.resident.contains(e) {
                self.hits += n;
                if !self.initial.contains(e) { self.cold_avoided += n; }
            } else {
                self.misses += n;
                if self.initial.contains(e) { self.eviction_misses += n; }
            }
        }
    }
}
fn percentile(values: &[u64], q: f64) -> Option<u64> {
    if values.is_empty() {
        None
    } else {
        let mut v = values.to_vec();
        v.sort_unstable();
        Some(v[((v.len() as f64 * q).ceil() as usize).saturating_sub(1)])
    }
}
fn lead_median(values: &[i128]) -> Option<i128> {
    if values.is_empty() {
        None
    } else {
        let mut v = values.to_vec();
        v.sort_unstable();
        Some(v[(v.len() - 1) / 2])
    }
}
fn ratio(n: u64, d: u64) -> Option<f64> {
    (d > 0).then(|| n as f64 / d as f64)
}
fn score(c: &Capture, name: &str, policy: &dyn PlacementPolicy, simulate: bool) -> Value {
    let records = &c.forecasts[name];
    let mut state = VirtualResidency::new(&c.profile);
    let mut baseline = VirtualResidency::new(&c.profile);
    let by_target: BTreeMap<_, _> = records.iter().map(|r| (r.target, r)).collect();
    let mut timeline = vec![];
    for (i, o) in c.observations.iter().enumerate() {
        timeline.push((o.demand_ns, o.seq * 2, false, i));
    }
    let last = c.observations.last().map(|o| o.demand_ns).unwrap_or(0);
    for (i, f) in records.iter().enumerate() {
        if f.completed_ns <= last {
            timeline.push((f.completed_ns, f.cutoff * 2 + 1, true, i));
        }
    }
    timeline.sort_unstable();
    let (
        mut predicted,
        mut correct,
        mut selected,
        mut demand,
        mut covered,
        mut late,
        mut expired,
        mut actionable,
        mut matched,
    ) = (0, 0, 0, 0, 0, 0, 0, 0, 0);
    let mut leads = vec![];
    let mut coverage = c.profile.settings.evaluation.as_ref().map(|e| coverage::Coverage::new(e.top_k_per_layer));
    for (now, _, prediction, index) in timeline {
        if simulate { state.advance(now); }
        if prediction {
            let f = &records[index];
            if simulate && now < f.expires_ns {
                for m in policy.plan(f, &state, &c.profile) {
                    state.apply(m, now, &c.profile);
                }
            }
            continue;
        }
        let o = &c.observations[index];
        if simulate { state.demand(o); baseline.demand(o); }
        let actual: BTreeMap<_, _> = o.demands.iter().copied().collect();
        demand += actual.values().sum::<u64>();
        if let Some(f) = by_target.get(&o.seq) {
            if let Some(coverage) = &mut coverage { coverage.observe(f, o, &c.profile); }
            matched += 1;
            selected += actual.len() as u64;
            leads.push(o.demand_ns as i128 - f.completed_ns as i128);
            let mut scores = f.scores.clone();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            scores.truncate(8);
            predicted += scores.len() as u64;
            correct += scores
                .iter()
                .filter(|(e, _)| actual.contains_key(e))
                .count() as u64;
            if f.completed_ns > o.demand_ns {
                late += 1;
            } else if f.expires_ns <= o.demand_ns {
                expired += 1;
            } else {
                covered += scores
                    .iter()
                    .filter_map(|(e, _)| actual.get(e))
                    .sum::<u64>();
                if scores.iter().any(|(e, _)| {
                    f.completed_ns.saturating_add(transfer(&c.profile, *e)) <= o.demand_ns
                }) {
                    actionable += 1;
                }
            }
        }
    }
    let compute: Vec<_> = records.iter().map(|r| r.compute_ns).collect();
    let queue: Vec<_> = records
        .iter()
        .map(|r| r.started_ns - r.available_ns)
        .collect();
    let overhead: u64 = records
        .iter()
        .map(|r| r.completed_ns - r.available_ns)
        .sum();
    let net = (baseline.misses as i128 - state.misses as i128)
        * c.profile.hardware.cold_cost_ns as i128
        - state.cost as i128
        - overhead as i128;
    let mut report = json!({"prediction_records":records.len(),"forecast_target_frames":matched,"forecast_precision":ratio(correct,predicted),
        "forecast_recall":ratio(correct,selected),"demand_coverage":ratio(covered,demand),"usable_lead_p50_ns":lead_median(&leads),
        "prediction_compute_p50_ns":percentile(&compute,0.5),"prediction_compute_p95_ns":percentile(&compute,0.95),"prediction_compute_p99_ns":percentile(&compute,0.99),
        "prediction_queue_p50_ns":percentile(&queue,0.5),"prediction_publication_p50_ns":percentile(&records.iter().map(|r|r.completed_ns-r.started_ns-r.compute_ns).collect::<Vec<_>>(),0.5),
        "hits":state.hits,"misses":state.misses,"hbm_hit_rate":ratio(state.hits,demand),"promotion_count":state.promotions,"eviction_count":state.evictions,
        "movement_bytes":state.bytes,"modeled_movement_cost_ns":state.cost,"estimated_net_route_cost_saved_ns":net as f64,
        "actionable_rate":ratio(actionable,c.observations.len() as u64),"late_predictions":late,"expired_predictions":expired,
        "confidence_note":"Correlated frames; no independent workload repeats or confidence interval."});
    if let Some(config) = &c.profile.settings.evaluation {
        coverage.unwrap().apply(&mut report, config.top_k_per_layer, demand);
        apply_ledger(&mut report, &state, &baseline, compute.iter().sum(), &c.profile);
    }
    report
}
pub fn evaluate(c: &Capture) -> Value {
    let mut contenders = serde_json::Map::new();
    if c.error.is_none() {
        for name in c.forecasts.keys() {
            contenders.insert(format!("{name}/static"), score(c, name, &Static, true));
            contenders.insert(format!("{name}/economic"), score(c, name, &Economic, true));
        }
    }
    let mut report = json!({"result_kind":"SHADOW: modeled route costs, NOT measured serving improvement","routing_frames":c.observations.len(),"error":c.error,
        "capture_limit_reached":c.limit_reason.is_some() || c.observations.len() >= c.profile.settings.max_frames,
        "hardware":c.profile.hardware,"workers":c.workers,"contenders":contenders,"mode":"in_process","native_capture_version":1});
    if let Some(config) = &c.profile.settings.evaluation { report["evaluation"] = json!(config); }
    if let Some(reason) = c.limit_reason {
        report["capture_limit_reason"] = json!(reason);
    }
    report
}

/// Lifetime residency simulation with a bounded window for prediction-quality statistics.
/// Trace retention never resets resident experts or pending transfers.
pub(super) struct Continuous {
    pub window: Capture,
    pub frames: u64,
    cells: usize,
    limit: usize,
    pending: Vec<Forecast>,
    pending_cells: usize,
    states: BTreeMap<String, VirtualResidency>,
    baseline: VirtualResidency,
    overhead: BTreeMap<String, u64>,
    compute: BTreeMap<String, u64>,
}
impl Continuous {
    pub fn new(capture: &Capture, limit: usize) -> Self {
        let states = capture.forecasts.keys().flat_map(|name| {
            ["static", "economic"].map(|policy| (
                format!("{name}/{policy}"), VirtualResidency::new(&capture.profile)
            ))
        }).collect();
        Self {
            window: capture.clone(), frames: 0, cells: 0, limit,
            pending: vec![], pending_cells: 0, states,
            baseline: VirtualResidency::new(&capture.profile), overhead: BTreeMap::new(), compute: BTreeMap::new(),
        }
    }
    pub fn forecast(&mut self, f: Forecast) {
        *self.overhead.entry(f.provider.clone()).or_default() += f.completed_ns - f.available_ns;
        *self.compute.entry(f.provider.clone()).or_default() += f.compute_ns;
        if self.pending_cells.saturating_add(f.scores.len()) > self.limit {
            self.window.error.get_or_insert("placement evaluation backlog exceeded its bounded capacity".into());
            return;
        }
        self.pending_cells += f.scores.len();
        self.pending.push(f.clone());
        if self.window.observations.first().is_some_and(|o| f.cutoff >= o.seq) {
            self.cells += f.scores.len();
            self.window.forecasts.get_mut(&f.provider).unwrap().push(f);
            self.trim();
        }
    }
    pub fn observe(&mut self, o: &Observation) {
        self.pending.sort_by_key(|f| (f.completed_ns, f.cutoff));
        let ready = self.pending.partition_point(|f| f.completed_ns <= o.demand_ns);
        for f in self.pending.drain(..ready) {
            self.pending_cells -= f.scores.len();
            let state = self.states.get_mut(&format!("{}/economic", f.provider)).unwrap();
            state.advance(f.completed_ns);
            if f.completed_ns < f.expires_ns {
                for movement in Economic.plan(&f, state, &self.window.profile) {
                    state.apply(movement, f.completed_ns, &self.window.profile);
                }
            }
        }
        for state in self.states.values_mut() { state.demand(o); }
        self.baseline.demand(o);
        self.frames += 1;
        self.cells += o.demands.len();
        self.window.observations.push(o.clone());
        self.trim();
    }
    fn trim(&mut self) {
        while self.window.observations.len() > self.window.profile.settings.max_frames || self.cells > self.limit {
            if self.window.observations.is_empty() { break; }
            let old = self.window.observations.remove(0);
            self.cells -= old.demands.len();
            for records in self.window.forecasts.values_mut() {
                records.retain(|f| {
                    if f.cutoff <= old.seq { self.cells -= f.scores.len(); false } else { true }
                });
            }
        }
    }
    pub fn snapshot(&self, workers: &BTreeMap<String, WorkerCounts>) -> (Capture, Value) {
        let mut window = self.window.clone();
        window.workers = workers.clone();
        let mut residency = serde_json::Map::new();
        let demand = self.baseline.hits + self.baseline.misses;
        for (name, state) in &self.states {
            let provider = name.split('/').next().unwrap();
            let overhead = self.overhead.get(provider).copied().unwrap_or(0);
            let net = (self.baseline.misses as i128 - state.misses as i128)
                * self.window.profile.hardware.cold_cost_ns as i128 - state.cost as i128 - overhead as i128;
            let mut row = json!({
                "hits":state.hits,"misses":state.misses,"hbm_hit_rate":ratio(state.hits,demand),
                "promotion_count":state.promotions,"eviction_count":state.evictions,
                "movement_bytes":state.bytes,"modeled_movement_cost_ns":state.cost,
                "estimated_net_route_cost_saved_ns":net as f64,
                "resident_experts":state.resident.len(),"cold_experts":self.window.profile.model.experts.len()-state.resident.len(),"pending_transfers":state.pending.len(),
                "reserved_hbm_bytes":state.reserved,
            });
            if self.window.profile.settings.evaluation.is_some() {
                apply_ledger(&mut row, state, &self.baseline, self.compute.get(provider).copied().unwrap_or(0), &self.window.profile);
            }
            residency.insert(name.clone(), row);
        }
        (window, json!({"routing_frames":self.frames,"residency":residency}))
    }
}

pub(super) fn evaluate_continuous(window: &Capture, totals: Value) -> Value {
    let mut quality = serde_json::Map::new();
    if window.error.is_none() {
        for name in window.forecasts.keys() {
            let row = score(window, name, &Static, false);
            quality.insert(format!("{name}/static"), row.clone());
            quality.insert(format!("{name}/economic"), row);
        }
    }
    let mut report = json!({"result_kind":"SHADOW: modeled route costs, NOT measured serving improvement",
        "error":window.error,"hardware":window.profile.hardware,"workers":window.workers,
        "contenders":quality,"mode":"in_process","native_capture_version":1});
    report.as_object_mut().unwrap().remove("capture_limit_reached");
    report.as_object_mut().unwrap().remove("capture_limit_reason");
    report["evaluation_window_frames"] = json!(window.observations.len());
    report["evaluation_window_start"] = json!(window.observations.first().map(|o| o.seq));
    report["routing_frames"] = totals["routing_frames"].clone();
    if let Some(config) = &window.profile.settings.evaluation { report["evaluation"] = json!(config); }
    report["continuous"] = json!(true);
    report["trace_complete"] = totals["trace_complete"].clone();
    if totals["active_policy"].is_object() { report["active_policy"] = totals["active_policy"].clone(); }
    if totals["physical_residency"].is_object() { report["physical_residency"] = totals["physical_residency"].clone(); }
    if totals["actual_ledger"].is_object() { report["actual_ledger"] = totals["actual_ledger"].clone(); }
    report["metrics_scope"] = json!("Prediction quality: bounded recent window. Residency and worker counts: server lifetime. Costs are modeled, not throughput gains.");
    if report["error"].is_null() {
        for (name, row) in report["contenders"].as_object_mut().unwrap() {
            for (key, value) in totals["residency"][name].as_object().unwrap() {
                row[key] = value.clone();
            }
        }
    }
    report
}

fn apply_ledger(report: &mut Value, state: &VirtualResidency, baseline: &VirtualResidency, compute: u64, p: &Profile) {
    let saved = state.cold_avoided as i128 * p.hardware.cold_cost_ns as i128;
    let evicted = state.eviction_misses as i128 * p.hardware.cold_cost_ns as i128;
    report.as_object_mut().unwrap().remove("estimated_net_route_cost_saved_ns");
    report["cold_demand_avoided"] = json!(state.cold_avoided);
    report["eviction_induced_misses"] = json!(state.eviction_misses);
    report["cold_execution_saved_ns"] = json!(saved as f64);
    report["eviction_execution_loss_ns"] = json!(evicted as f64);
    report["modeled_balance_ns"] = json!((saved - evicted - state.cost as i128) as f64);
    report["predictor_runtime_total_ns"] = json!(compute);
    report["cold_miss_reduction"] = if baseline.misses > 0 { json!((state.cold_avoided as f64 - state.eviction_misses as f64) / baseline.misses as f64) } else { Value::Null };
    report["balance_scope"] = json!("Illustrative route-cost balance: avoided cold execution minus eviction-induced cold execution minus full transfer cost. Cold penalty is a profile assumption. Predictor runtime is reported separately; asynchronous wait is not charged as inference stall. Serving contention and overlap are not calibrated; this is not measured wall time saved.");
}

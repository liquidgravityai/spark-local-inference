//! Bounded in-process provider workers. Neither execution nor publication waits on a plugin.

use super::{
    Record,
    evaluation::{Capture, CaptureLimit, Continuous, WorkerCounts, evaluate, evaluate_continuous},
    prediction::{Context, Forecast, Profile, Registry},
};
use crate::parallax::now_ns;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{self, SyncSender, TryRecvError},
    },
    thread::{self, JoinHandle},
    time::Duration,
};

pub struct Observer {
    input: SyncSender<Arc<Record>>,
    stop: Arc<AtomicBool>,
    dropped: Arc<AtomicU64>,
    report: Arc<Mutex<Value>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}
struct Reply {
    context: Context,
    started: u64,
    compute: u64,
    result: Result<Vec<(super::prediction::Expert, f64)>, String>,
}
struct Worker {
    input: SyncSender<Context>,
    reply: mpsc::Receiver<Reply>,
    pending: Option<u64>,
    version: String,
}
fn save(path: &Path, value: &impl serde::Serialize) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(value).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    fs::rename(tmp, path).map_err(|e| e.to_string())
}
impl Observer {
    pub fn start(profile: Profile, registry: Registry) -> Result<Self, String> {
        Self::start_with_limit(profile, registry, 2_000_000)
    }
    pub(super) fn start_with_limit(profile: Profile, registry: Registry, cell_limit: usize) -> Result<Self, String> {
        Self::start_bound(profile, registry, cell_limit, None)
    }
    pub fn start_with_residency(profile: Profile, registry: Registry,
        binding: Option<crate::parallax::residency::execution::ExecutionBinding>) -> Result<Self, String> {
        Self::start_bound(profile, registry, 2_000_000, binding)
    }
    fn start_bound(profile: Profile, registry: Registry, cell_limit: usize,
        binding: Option<crate::parallax::residency::execution::ExecutionBinding>) -> Result<Self, String> {
        profile.validate()?;
        let mut active = super::active::Active::new(&profile, binding.clone())?;
        let mut ledger = binding.as_ref().map(|b| {
            Ok::<_, String>(super::ledger::Ledger::new(b.snapshot().map_err(|e|e.to_string())?,
                b.expert_storage().map_err(|e|e.to_string())?))
        }).transpose()?;
        fs::create_dir(&profile.settings.output).map_err(|e| e.to_string())?;
        let (input, rx) = mpsc::sync_channel::<Arc<Record>>(512);
        let stop = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicU64::new(0));
        let report = Arc::new(Mutex::new(
            json!({"mode":"in_process","routing_frames":0,"contenders":{},"workers":{},"hardware":profile.hardware}),
        ));
        let mut workers = BTreeMap::new();
        for selection in &profile.settings.predictors {
            let mut provider = registry.create(selection)?;
            let version = provider.version().to_owned();
            let (tx, requests) = mpsc::sync_channel::<Context>(1);
            let (replies, reply) = mpsc::sync_channel(1);
            thread::Builder::new()
                .name(format!("parallax-{}", selection.name))
                .spawn(move || {
                    while let Ok(context) = requests.recv() {
                        let started = now_ns();
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            provider.predict(&context)
                        }))
                        .unwrap_or_else(|_| Err("predictor panicked".into()));
                        let compute = now_ns() - started;
                        if replies
                            .send(Reply {
                                context,
                                started,
                                compute,
                                result,
                            })
                            .is_err()
                        {
                            break;
                        }
                    }
                })
                .map_err(|e| e.to_string())?;
            workers.insert(
                selection.name.clone(),
                Worker {
                    input: tx,
                    reply,
                    pending: None,
                    version,
                },
            );
        }
        let (control, lost, display) = (stop.clone(), dropped.clone(), report.clone());
        let worker = thread::Builder::new()
            .name("parallax-observatory".into())
            .spawn(move || {
                let mut capture = Capture {
                    native_capture_version: 1,
                    profile,
                    observations: vec![],
                    forecasts: workers.keys().map(|n| (n.clone(), vec![])).collect(),
                    workers: workers
                        .keys()
                        .map(|n| (n.clone(), WorkerCounts::default()))
                        .collect(),
                    error: None,
                    limit_reason: None,
                };
                let mut live = Continuous::new(&capture, cell_limit);
                let (evaluation_tx, evaluation_rx) = mpsc::sync_channel::<(Capture, Value, Option<Capture>)>(1);
                let reports = display.clone();
                let evaluation = thread::Builder::new()
                    .name("parallax-evaluation".into())
                    .spawn(move || {
                        while let Ok((window, totals, trace)) = evaluation_rx.recv() {
                            let mut report = evaluate_continuous(&window, totals);
                            let result = save(&window.profile.settings.output.join("live-comparison.json"), &report)
                                .and_then(|_| {
                                    if let Some(trace) = trace {
                                        save(&trace.profile.settings.output.join("comparison.json"), &evaluate(&trace))?;
                                        save(&trace.profile.settings.output.join("capture.json"), &trace)
                                    } else { Ok(()) }
                                });
                            if let Err(error) = result {
                                report["error"] = json!(format!("cannot save capture: {error}"));
                                report["contenders"] = json!({});
                            }
                            *reports.lock().unwrap() = report;
                        }
                    });
                let Ok(evaluation) = evaluation else {
                    *display.lock().unwrap() =
                        json!({"error":"cannot start evaluation thread","contenders":{}});
                    return;
                };
                let mut last_sequence = None;
                let mut first_sequence = None;
                let mut report_at = 0;
                let mut history = vec![];
                let model=Arc::new(capture.profile.model.clone());
                let mut retained_cells=0usize;
                loop {
                    if lost.load(Ordering::Relaxed) > 0 {
                        if let Some(ledger) = &mut ledger { ledger.invalidate("routing delivery failure; ledger incomplete"); }
                        live.window.error.get_or_insert("native routing delivery failure or observatory queue overflow; incomplete stream".into());
                        if capture.limit_reason.is_none() { capture.error.get_or_insert(
                            "native routing delivery failure or observatory queue overflow; incomplete window".into(),
                        ); }
                    }
                    if let Some(policy) = &mut active {
                        let config = capture.profile.settings.active.as_ref().unwrap();
                        if lost.load(Ordering::Relaxed) != 0 || live.window.error.is_some() || capture.workers[&config.predictor].disabled {
                            policy.disable("routing observations were lost or the active predictor failed");
                        }
                    }
                    for (name, worker) in &mut workers {
                        let counts = capture.workers.get_mut(name).unwrap();
                        if counts.disabled {
                            continue;
                        }
                        if worker.pending.is_some_and(|start| {
                            now_ns() - start > capture.profile.settings.budget_ns
                        }) {
                            counts.timeouts += 1;
                            counts.disabled = true;
                            worker.pending = None;
                            continue;
                        }
                        match worker.reply.try_recv() {
                            Ok(reply) => {
                                worker.pending = None;
                                let cutoff = reply.context.history.last().unwrap().seq;
                                match reply.result {
                                    Ok(scores) => {
                                        let f = Forecast {
                                            provider: name.clone(),
                                            provider_version: worker.version.clone(),
                                            cutoff,
                                            target: cutoff + capture.profile.settings.target_offset,
                                            available_ns: reply.context.available_ns,
                                            started_ns: reply.started,
                                            completed_ns: now_ns(),
                                            compute_ns: reply.compute,
                                            expected_demand_ns: reply.context.available_ns
                                                + capture.profile.settings.horizon_ns,
                                            expires_ns: reply.context.available_ns
                                                + 2 * capture.profile.settings.horizon_ns,
                                            scores,
                                        };
                                        if f.validate(&capture.profile).is_ok() {
                                            // Admission reserves a full catalog for every pending reply.
                                            if let Some(policy) = &mut active {
                                                if let Some(target) = first_sequence.and_then(|base: u64| base.checked_add(f.target)) {
                                                    policy.forecast(&f, &capture.profile, now_ns(), target);
                                                }
                                            }
                                            live.forecast(f.clone());
                                            if f.cutoff < capture.observations.len() as u64 {
                                                retained_cells += f.scores.len();
                                                capture.forecasts.get_mut(name).unwrap().push(f);
                                            }
                                            counts.completed += 1;
                                        } else {
                                            counts.failures += 1;
                                            counts.disabled = true;
                                        }
                                    }
                                    Err(_) => {
                                        counts.failures += 1;
                                        counts.disabled = true;
                                    }
                                }
                            }
                            Err(TryRecvError::Disconnected) => {
                                worker.pending = None;
                                counts.failures += 1;
                                counts.disabled = true;
                            }
                            Err(TryRecvError::Empty) => {}
                        }
                    }
                    match rx.recv_timeout(Duration::from_millis(1)) {
                        Ok(record) => {
                            if let (Some(ledger), Some(binding)) = (&mut ledger, &binding) {
                                let active_since = ledger.sync(binding);
                                ledger.observe(&record, active_since);
                            }
                            if last_sequence.is_some_and(|last| record.sequence != last + 1) {
                                live.window.error = Some("native routing sequence gap; incomplete stream".into());
                                if capture.limit_reason.is_none() { capture.error = live.window.error.clone(); }
                                continue;
                            }
                            first_sequence.get_or_insert(record.sequence);
                            last_sequence = Some(record.sequence);
                            let mut demands = vec![];
                            for layer in &record.sample.layers {
                                for (expert, count) in layer.counts.iter().enumerate() {
                                    if *count != 0 {
                                        let key = super::prediction::Expert {
                                            layer: layer.layer,
                                            expert,
                                        };
                                        if capture
                                            .profile
                                            .model
                                            .experts
                                            .binary_search(&key)
                                            .is_err()
                                        {
                                            live.window.error = Some("routing expert missing from bound catalog".into());
                                            if capture.limit_reason.is_none() { capture.error = live.window.error.clone(); }
                                        }
                                        demands.push((key, u64::from(*count)));
                                    }
                                }
                            }
                            // Reserve space before admitting a shared observation, independently of
                            // provider completion order. Busy providers already own their reservation.
                            let enabled = capture.workers.values().filter(|w| !w.disabled).count();
                            let reserved = enabled * model.experts.len();
                            if capture.limit_reason.is_none() && retained_cells.saturating_add(demands.len()).saturating_add(reserved) > cell_limit {
                                capture.limit_reason = Some(CaptureLimit::DemandScores);
                            }

                            let observation = Arc::new(super::prediction::Observation {
                                seq: live.frames,
                                demand_ns: record.sample.demand_lower_bound_ns,
                                available_ns: now_ns(),
                                phase: record.phase.into(),
                                batch_size: record.batch_size,
                                concurrency: record.concurrency,
                                demands,
                            });
                            if capture.limit_reason.is_none() && capture.error.is_none() {
                                retained_cells += observation.demands.len();
                                capture.observations.push((*observation).clone());
                                if capture.observations.len() == capture.profile.settings.max_frames {
                                    capture.limit_reason = Some(CaptureLimit::Frames);
                                }
                            }
                            live.observe(&observation);
                            history.push(observation);
                            if history.len() > 8 {
                                history.remove(0);
                            }
                            if live.window.error.is_none() {
                                for (name, worker) in &mut workers {
                                    let counts = capture.workers.get_mut(name).unwrap();
                                    if counts.disabled {
                                        continue;
                                    }
                                    if worker.pending.is_some() {
                                        counts.dropped_busy += 1;
                                        continue;
                                    }
                                    let context = Context {
                                    model:model.clone(),target_offset:capture.profile.settings.target_offset,horizon_ns:capture.profile.settings.horizon_ns,
                                        history: history.clone(),
                                        available_ns: history.last().unwrap().available_ns,
                                    };
                                    if worker.input.try_send(context).is_ok() {
                                        worker.pending = Some(now_ns());
                                        counts.submitted += 1;
                                    } else {
                                        counts.dropped_busy += 1;
                                    }
                                }
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        Err(mpsc::RecvTimeoutError::Timeout) if control.load(Ordering::Relaxed) && workers.values().all(|w| w.pending.is_none()) => {
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                    }
                    if now_ns() >= report_at {
                        let (window, mut totals) = live.snapshot(&capture.workers);
                        totals["trace_complete"] = json!(capture.limit_reason.is_some());
                        if let (Some(ledger), Some(binding)) = (&mut ledger, &binding) {
                            ledger.sync(binding);
                            totals["actual_ledger"] = ledger.report();
                        }
                        if let Some(policy) = &active { totals["active_policy"] = policy.report(); }
                        if let Some(binding) = &binding { totals["physical_residency"] = super::physical_report(binding,active.is_some()); }
                        let _ = evaluation_tx.try_send((window, totals, None));
                        report_at = now_ns() + 1_000_000_000;
                    }
                }
                control.store(true, Ordering::Relaxed);
                if capture.limit_reason.is_none() && lost.load(Ordering::Relaxed)>0 {capture.error.get_or_insert("native routing delivery failure or observatory queue overflow; incomplete window".into());}
                if lost.load(Ordering::Relaxed)>0 { live.window.error.get_or_insert("native routing delivery failure or observatory queue overflow; incomplete stream".into()); }
                let (window, mut totals) = live.snapshot(&capture.workers);
                totals["trace_complete"] = json!(capture.limit_reason.is_some());
                        if let (Some(ledger), Some(binding)) = (&mut ledger, &binding) {
                            ledger.sync(binding);
                            totals["actual_ledger"] = ledger.report();
                        }
                        if let Some(policy) = &active { totals["active_policy"] = policy.report(); }
                        if let Some(binding) = &binding { totals["physical_residency"] = super::physical_report(binding,active.is_some()); }
                let _ = evaluation_tx.send((window, totals, Some(capture)));
                drop(evaluation_tx);
                let _ = evaluation.join();
                // Dropping the channels retires cooperative providers. A hung trusted plugin
                // is detached; no join may hold serving shutdown hostage.
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            input,
            stop,
            dropped,
            report,
            worker: Mutex::new(Some(worker)),
        })
    }
    pub(super) fn publish(&self, record: Arc<Record>) {
        if !self.stop.load(Ordering::Relaxed) && self.input.try_send(record).is_err() {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn invalidate(&self) {
        if !self.stop.load(Ordering::Relaxed) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
    pub fn snapshot(&self) -> Value {
        self.report.lock().unwrap().clone()
    }
    pub fn close(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.lock().unwrap().take() {
            let _ = worker.join();
        }
    }
}
impl Drop for Observer {
    fn drop(&mut self) {
        self.close();
    }
}
fn load_capture(path: &Path) -> Result<Capture, String> {
    let capture: Capture = serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if capture.native_capture_version != 1 {
        return Err("unsupported native capture version".into());
    }
    capture.profile.validate()?;
    if let Some(error) = &capture.error {
        return Err(format!("incomplete native capture: {error}"));
    }
    if capture.observations.len() > capture.profile.settings.max_frames {
        return Err("capture exceeds frame limit".into());
    }
    for (index, o) in capture.observations.iter().enumerate() {
        if o.seq != index as u64
            || o.available_ns < o.demand_ns
            || (index > 0
                && (o.demand_ns < capture.observations[index - 1].demand_ns
                    || o.available_ns < capture.observations[index - 1].available_ns))
        {
            return Err("invalid observation order or availability".into());
        }
        let mut experts = std::collections::BTreeSet::new();
        for (e, n) in &o.demands {
            if *n == 0
                || !experts.insert(*e)
                || capture.profile.model.experts.binary_search(e).is_err()
            {
                return Err("invalid observed expert demand".into());
            }
        }
    }
    for (name, records) in &capture.forecasts {
        let mut targets = std::collections::BTreeSet::new();
        if !capture
            .profile
            .settings
            .predictors
            .iter()
            .any(|p| &p.name == name)
            || !capture.workers.contains_key(name)
        {
            return Err("unconfigured predictor in capture".into());
        }
        for f in records {
            f.validate(&capture.profile)?;
            let cutoff = capture
                .observations
                .get(f.cutoff as usize)
                .ok_or("forecast cutoff is absent")?;
            if &f.provider != name
                || f.available_ns < cutoff.available_ns
                || !targets.insert(f.target)
                || Some(f.expected_demand_ns)
                    != f.available_ns
                        .checked_add(capture.profile.settings.horizon_ns)
                || Some(f.expires_ns)
                    != f.available_ns
                        .checked_add(2 * capture.profile.settings.horizon_ns)
            {
                return Err("invalid forecast information boundary".into());
            }
        }
    }
    Ok(capture)
}

pub fn replay(path: &Path) -> Result<Value, String> {
    Ok(evaluate(&load_capture(path)?))
}
pub fn reevaluate(path: &Path, config: super::prediction::EvaluationConfig) -> Result<Value, String> {
    let mut capture = load_capture(path)?;
    capture.profile.settings.evaluation = Some(config);
    capture.profile.validate()?;
    let mut report = evaluate(&capture);
    report["comparison_kind"] = json!("Recorded forecasts with alternate scoring and placement; original timing and hardware costs retained.");
    Ok(report)
}

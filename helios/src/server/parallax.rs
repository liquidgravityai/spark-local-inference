//! Optional routing capture and an in-process shadow observatory.

pub mod calibration;
pub mod active;
mod evaluation;
mod ledger;
mod canary;
mod observer;
pub mod prediction;
pub use observer::{replay, reevaluate};
pub use prediction::{Profile, Registry};

use crate::parallax::{RoutingReadout, RoutingSample};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::VecDeque;
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
};
use std::time::{SystemTime, UNIX_EPOCH};

const CAPACITY: usize = 32;

#[derive(Serialize)]
struct Record {
    sequence: u64,
    graph: usize,
    phase: &'static str,
    batch_size: usize,
    concurrency: usize,
    sample: RoutingSample,
}

pub(super) struct Telemetry {
    geometry: Option<(usize, usize)>,
    residency: Option<crate::parallax::residency::execution::ExecutionBinding>,
    session: String,
    sequence: AtomicU64,
    dropped: AtomicU64,
    invalid: AtomicU64,
    history: Mutex<VecDeque<Arc<Record>>>,
    observer: OnceLock<observer::Observer>,
}

impl Default for Telemetry {
    fn default() -> Self {
        Self {
            geometry: None,
            residency: None,
            session: format!(
                "{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            sequence: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            invalid: AtomicU64::new(0),
            history: Mutex::new(VecDeque::with_capacity(CAPACITY)),
            observer: OnceLock::new(),
        }
    }
}

impl Telemetry {
    pub fn configure(&mut self, manifest: &crate::execution::ExecutionManifest) {
        self.residency = manifest.residency.clone();
        self.geometry = manifest
            .buffers
            .iter()
            .find(|b| b.name == crate::parallax::BUFFER)
            .map(|b| (b.shape[1], b.shape[2] - crate::parallax::HEADER));
    }

    pub fn publish(
        &self,
        readout: RoutingReadout,
        graph: usize,
        kind: crate::execution::GraphKind,
        batch_size: usize,
        concurrency: usize,
    ) {
        let sequence = self.sequence.fetch_add(1, Ordering::Relaxed) + 1;
        if let Some(binding) = &self.residency { binding.record_demand(sequence); }
        let RoutingReadout::Valid(sample) = readout else {
            self.invalid.fetch_add(1, Ordering::Relaxed);
            if let Some(observer) = self.observer.get() {
                observer.invalidate();
            }
            return;
        };
        let record = Arc::new(Record {
            sequence,
            graph,
            phase: if kind.is_prefill() {
                "prefill"
            } else if kind == crate::execution::GraphKind::Speculative {
                "verify"
            } else {
                "decode"
            },
            batch_size,
            concurrency,
            sample,
        });
        if let Some(observer) = self.observer.get() {
            observer.publish(record.clone());
        }
        let Ok(mut history) = self.history.try_lock() else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if history.len() == CAPACITY {
            history.pop_front();
        }
        history.push_back(record);
    }

    pub fn start_predictions(&self, profile: Profile) -> Result<(), String> {
        let (layers, experts) = self
            .geometry
            .ok_or("Parallax requires explicitly enabled routing telemetry")?;
        if profile
            .model
            .experts
            .iter()
            .any(|e| e.layer >= layers || e.expert >= experts)
        {
            return Err("catalog exceeds native routing geometry".into());
        }
        let observer = observer::Observer::start_with_residency(profile, Registry::builtins(), self.residency.clone())?;
        self.observer
            .set(observer)
            .map_err(|_| "prediction providers already started".into())
    }
    pub fn predictions(&self) -> Value {
        let mut report = self.observer.get().map(|o| o.snapshot()).unwrap_or_else(
            || json!({"enabled":false,"contenders":{},"routing_frames":0,"workers":{}}),
        );
        if let Some(binding) = &self.residency {
            report["physical_residency"] = physical_report(binding, report["active_policy"].is_object());
        }
        report
    }
    pub fn close_predictions(&self) {
        if let Some(observer) = self.observer.get() {
            observer.close();
        }
    }

    pub fn snapshot(&self, after: u64, model: &str) -> Value {
        // Clone only bounded Arc handles under the lock; JSON encoding happens outside it.
        let records: Vec<_> = self
            .history
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|row| row.sequence > after)
            .cloned()
            .collect();
        let rows: Vec<_> = records
            .iter()
            .map(|r| serde_json::to_value(r.as_ref()).expect("routing values serialize"))
            .collect();
        json!({"version":1,"session":self.session,"model_id":model,
            "enabled":self.geometry.is_some(),"geometry":self.geometry,
            "clock":"native_process_monotonic_ns","native_now_ns":crate::parallax::now_ns(),
            "demand_timing":"host_dispatch_lower_bound",
            "demand_scope":"target_expert_execution_including_rejected_candidates; draft experts excluded",
            "capacity":CAPACITY,"latest_sequence":self.sequence.load(Ordering::Relaxed),
            "dropped_total":self.dropped.load(Ordering::Relaxed),
            "invalid_total":self.invalid.load(Ordering::Relaxed),"frames":rows})
    }
}

fn physical_report(binding: &crate::parallax::residency::execution::ExecutionBinding, active: bool) -> Value {
    match binding.report() {
        Ok(report) => {
            let state=report.snapshot;
            let hbm=state.experts.values().filter(|resident| **resident).count();
            json!({"mode":if active {"active_policy"} else {"configured_initial_placement"},"catalog_id":state.catalog_id,
                "generation":state.generation,"hbm_experts":hbm,"grace_experts":state.experts.len()-hbm,
                "hbm_pool_bytes":state.hbm_allocated_bytes,"grace_backing_bytes":state.backing_allocated_bytes,
                "free_slots":state.free_slots,"pending":state.pending,"readers":state.readers,"poisoned":state.poisoned,
                "commits_including_startup":state.statistics.commits,
                "staged_copy_bytes_including_startup":state.statistics.staged_bytes,
                "committed_copy_bytes_including_startup":state.statistics.committed_copy_bytes,
                "promoted_experts_including_startup":state.statistics.promoted_experts,
                "evicted_experts_including_startup":state.statistics.evicted_experts,
                "publication_host_ns_including_startup":state.statistics.publication_host_ns,
                "copy_gpu_ns_including_startup":state.statistics.copy_gpu_ns,
                "timed_copy_transactions_including_startup":state.statistics.timed_copies,
                "rollbacks":state.statistics.rollbacks,"rejected_plans":state.statistics.rejected,
                "cancelled_transactions":state.statistics.cancellations,"movement_error":report.disabled,
                "movement_history_capacity":256,"movement_event_total":report.event_count,"movement_events":report.events})
        },
        Err(error) => json!({"error":error.to_string()}),
    }
}

#[cfg(test)]
#[path = "parallax/tests.rs"]
mod prediction_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::GraphKind;
    fn sample() -> RoutingReadout {
        RoutingReadout::Valid(RoutingSample {
            residency_generation: None,
            gpu_ns: None,
            submission: 1,
            demand_lower_bound_ns: 10,
            available_ns: 20,
            layers: Vec::new(),
        })
    }

    #[test]
    fn retention_and_cursor_are_bounded_and_loss_is_visible() {
        let telemetry = Telemetry::default();
        for _ in 0..40 {
            telemetry.publish(sample(), 0, GraphKind::Decode, 1, 2);
        }
        let value = telemetry.snapshot(0, "fixture");
        assert_eq!(value["frames"].as_array().unwrap().len(), CAPACITY);
        assert_eq!(value["frames"][0]["sequence"], 9);
        assert_eq!(
            telemetry.snapshot(39, "fixture")["frames"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn telemetry_content_failure_and_busy_consumer_never_block_producer() {
        let telemetry = Telemetry::default();
        telemetry.publish(RoutingReadout::Invalid, 0, GraphKind::Decode, 1, 1);
        let lock = telemetry.history.lock().unwrap();
        telemetry.publish(sample(), 0, GraphKind::Decode, 1, 1);
        drop(lock);
        let value = telemetry.snapshot(0, "fixture");
        assert_eq!(value["invalid_total"], 1);
        assert_eq!(value["dropped_total"], 1);
        assert_eq!(value["latest_sequence"], 2);
        assert!(value["frames"].as_array().unwrap().is_empty());
    }
}

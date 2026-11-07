//! Actual placement accounting and a bounded, observational execution-cost fit.
use super::Record;
use crate::parallax::{ExpertId, residency::{execution::{ExecutionBinding, MovementEvent}, Snapshot}};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

const WINDOW: usize = 256;
const MAX_SHAPES: usize = 512;

#[derive(Clone, Copy)]
struct Point { total_mib: f64, cold_mib: f64, gpu_ms: f64 }
#[derive(Clone, Copy)]
struct Fit { intercept: f64, total: f64, cold: f64 }
impl Fit {
    fn predict(self, p: Point) -> f64 { self.intercept + self.total * p.total_mib + self.cold * p.cold_mib }
}
#[derive(Default)]
struct Shape {
    points: VecDeque<Point>,
    evaluated: u64,
    absolute_error_ms: f64,
}
impl Shape {
    fn fit(&self) -> Option<Fit> {
        if self.points.len() < 32 { return None; }
        let n = self.points.len() as f64;
        let mean = Point {
            total_mib: self.points.iter().map(|p| p.total_mib).sum::<f64>() / n,
            cold_mib: self.points.iter().map(|p| p.cold_mib).sum::<f64>() / n,
            gpu_ms: self.points.iter().map(|p| p.gpu_ms).sum::<f64>() / n,
        };
        let (mut xx, mut zz, mut xz, mut xy, mut zy) = (0., 0., 0., 0., 0.);
        for p in &self.points {
            let (x,z,y) = (p.total_mib-mean.total_mib, p.cold_mib-mean.cold_mib, p.gpu_ms-mean.gpu_ms);
            xx += x*x; zz += z*z; xz += x*z; xy += x*y; zy += z*y;
        }
        if zz / n < 1. { return None; }
        let (total, cold) = if xx / n < 1e-12 {
            (0., zy / zz)
        } else {
            let determinant = xx*zz-xz*xz;
            if determinant <= xx*zz*1e-6 { return None; }
            ((xy*zz-zy*xz)/determinant, (zy*xx-xy*xz)/determinant)
        };
        let intercept = mean.gpu_ms-total*mean.total_mib-cold*mean.cold_mib;
        [intercept,total,cold].iter().all(|v| v.is_finite()).then_some(Fit {intercept,total,cold})
    }
    fn observe(&mut self, p: Point) {
        if let Some(fit) = self.fit() {
            self.absolute_error_ms += (fit.predict(p)-p.gpu_ms).abs();
            self.evaluated += 1;
        }
        if self.points.len() == WINDOW { self.points.pop_front(); }
        self.points.push_back(p);
    }
}

pub(super) struct Ledger {
    baseline: BTreeSet<ExpertId>,
    resident: BTreeSet<ExpertId>,
    bytes: BTreeMap<ExpertId,u64>,
    generation: u64,
    events: VecDeque<MovementEvent>,
    last_event: u64,
    last_record: Option<u64>,
    error: Option<String>,
    frames: u64,
    hot_routes: u64,
    cold_routes: u64,
    avoided_routes: u64,
    eviction_routes: u64,
    avoided_touches: u64,
    eviction_touches: u64,
    promoted_unused: BTreeSet<ExpertId>,
    used_promotions: u64,
    unused_before_eviction: u64,
    commits: u64,
    staged_bytes: u64,
    copy_gpu_ns: u64,
    timed_copies: u64,
    untimed_copies: u64,
    publication_ns: u64,
    gate_ns: u64,
    charged_ns: u64,
    estimated_saved_ns: f64,
    estimated_frames: u64,
    overlap_frames: u64,
    shapes: BTreeMap<(usize,usize,u32),Shape>,
    rows: VecDeque<Value>,
}
impl Ledger {
    pub fn new(state: Snapshot, storage: Vec<(ExpertId,u64,usize)>) -> Self {
        let resident: BTreeSet<_> = state.experts.iter().filter_map(|(e,hot)| hot.then_some(*e)).collect();
        Self { baseline: resident.clone(), resident, bytes: storage.into_iter().map(|(e,b,_)|(e,b)).collect(),
            generation: state.generation, events: VecDeque::new(), last_event: 0, last_record: None, error: None,
            frames:0, hot_routes:0, cold_routes:0, avoided_routes:0, eviction_routes:0, avoided_touches:0,
            eviction_touches:0, promoted_unused:BTreeSet::new(), used_promotions:0, unused_before_eviction:0,
            commits:0, staged_bytes:0, copy_gpu_ns:0, timed_copies:0, untimed_copies:0, publication_ns:0, gate_ns:0, charged_ns:0,
            estimated_saved_ns:0., estimated_frames:0, overlap_frames:0, shapes:BTreeMap::new(), rows:VecDeque::new() }
    }
    pub fn invalidate(&mut self, message: &str) { self.error.get_or_insert(message.into()); }
    pub fn sync(&mut self, binding: &ExecutionBinding) -> Option<u64> {
        match binding.events_after(self.last_event) {
            Ok((events, active)) => { self.ingest(events); active }
            Err(e) => { self.invalidate(&e.to_string()); None }
        }
    }
    fn ingest(&mut self, events: Vec<MovementEvent>) {
        for event in events {
            if event.sequence != self.last_event + 1 { self.invalidate("movement history gap; ledger incomplete"); }
            self.last_event = event.sequence;
            self.staged_bytes += event.staged_bytes;
            self.publication_ns += event.publication_host_ns;
            self.gate_ns += event.submission_gate_closed_ns;
            if let Some(ns) = event.copy_gpu_ns { self.copy_gpu_ns += ns; self.timed_copies += 1; }
            else if event.staged_bytes > 0 { self.untimed_copies += 1; }
            // Charge measured copy/publication resource time. Reader drain may be useful inference.
            self.charged_ns += event.copy_gpu_ns.unwrap_or(0)
                + event.publication_host_ns;
            if self.events.len() == WINDOW { self.events.pop_front(); }
            self.events.push_back(event);
        }
    }
    fn advance(&mut self, generation: u64) -> bool {
        if generation < self.generation { self.invalidate("routing generations arrived out of order"); return false; }
        while self.generation < generation {
            let Some(e) = self.events.iter().find(|e| e.generation == self.generation
                && e.outcome == format!("committed:{}",self.generation+1)) else {
                self.invalidate("routing generation has no retained publication event"); return false;
            };
            for expert in &e.evict {
                self.resident.remove(expert);
                if self.promoted_unused.remove(expert) { self.unused_before_eviction += 1; }
            }
            for expert in &e.promote { self.resident.insert(*expert); self.promoted_unused.insert(*expert); }
            self.generation += 1;
            self.commits += 1;
        }
        true
    }
    pub fn observe(&mut self, record: &Record, active_since: Option<u64>) {
        if self.last_record.is_some_and(|seq| record.sequence != seq+1) { self.invalidate("routing observation gap; ledger incomplete"); }
        self.last_record = Some(record.sequence);
        let Some(generation) = record.sample.residency_generation else { self.invalidate("missing execution residency generation"); return; };
        if self.error.is_some() || !self.advance(generation) { return; }
        let (mut total_bytes, mut cold_bytes, mut baseline_cold_bytes) = (0u64,0u64,0u64);
        for layer in &record.sample.layers {
            for (id,count) in layer.counts.iter().enumerate().filter(|(_,n)| **n > 0) {
                let e = ExpertId {layer:layer.layer, expert:id};
                let Some(bytes) = self.bytes.get(&e) else { self.invalidate("routing expert absent from physical catalog"); return; };
                let hot = self.resident.contains(&e);
                let originally_hot = self.baseline.contains(&e);
                total_bytes += bytes;
                if hot { self.hot_routes += *count as u64; } else { self.cold_routes += *count as u64; cold_bytes += bytes; }
                if !originally_hot { baseline_cold_bytes += bytes; }
                if hot && !originally_hot { self.avoided_routes += *count as u64; self.avoided_touches += 1; }
                if !hot && originally_hot { self.eviction_routes += *count as u64; self.eviction_touches += 1; }
                if hot && self.promoted_unused.remove(&e) { self.used_promotions += 1; }
            }
        }
        self.frames += 1;
        let start = record.sample.demand_lower_bound_ns;
        let end = record.sample.available_ns;
        let overlap = active_since.is_some_and(|at| at <= end)
            || self.events.iter().any(|e| e.started_ns <= end && e.finished_ns >= start);
        if overlap { self.overlap_frames += 1; }
        let rows = record.sample.layers.iter().map(|l| l.tokens).max().unwrap_or(0);
        let key = (record.graph,record.batch_size,rows);
        let mut saved = None;
        if let Some(gpu_ns) = record.sample.gpu_ns.filter(|ns| *ns > 0) {
            let point = Point { total_mib: total_bytes as f64/1048576., cold_mib:cold_bytes as f64/1048576., gpu_ms:gpu_ns as f64/1e6 };
            if !self.shapes.contains_key(&key) && self.shapes.len() < MAX_SHAPES { self.shapes.insert(key,Shape::default()); }
            if let Some(shape) = self.shapes.get_mut(&key) {
                if let Some(fit) = shape.fit() {
                    // Predict before incorporating this frame's timing; no hindsight scoring.
                    let estimate = fit.cold * (baseline_cold_bytes as f64-cold_bytes as f64)/1048576.*1e6;
                    if estimate.is_finite() { saved=Some(estimate); self.estimated_saved_ns+=estimate; self.estimated_frames+=1; }
                }
                if !overlap { shape.observe(point); }
            }
        }
        if self.rows.len() == 512 { self.rows.pop_front(); }
        self.rows.push_back(json!({"sequence":record.sequence,"generation":generation,"graph":record.graph,
            "batch":record.batch_size,"target_rows":rows,"gpu_ns":record.sample.gpu_ns,
            "distinct_weight_bytes":total_bytes,"cold_distinct_weight_bytes":cold_bytes,
            "static_cold_distinct_weight_bytes":baseline_cold_bytes,"movement_overlap":overlap,
            "estimated_execution_saved_ns":saved}));
    }
    pub fn report(&self) -> Value {
        let shapes:Vec<_> = self.shapes.iter().map(|((graph,batch,rows),shape)| {
            let fit=shape.fit();
            json!({"graph":graph,"batch":batch,"target_rows":rows,"samples":shape.points.len(),
                "cold_ms_per_mib":fit.map(|f|f.cold),"total_ms_per_mib":fit.map(|f|f.total),
                "prequential_samples":shape.evaluated,"prequential_mae_ms":(shape.evaluated>0).then(||shape.absolute_error_ms/shape.evaluated as f64)})
        }).collect();
        json!({"version":1,"error":self.error,"observed_frames":self.frames,"accounted_generation":self.generation,
            "hbm_routes":self.hot_routes,"cold_routes":self.cold_routes,
            "cold_routes_avoided_vs_static":self.avoided_routes,"eviction_routes_vs_static":self.eviction_routes,
            "cold_expert_touches_avoided_vs_static":self.avoided_touches,"eviction_expert_touches_vs_static":self.eviction_touches,
            "promotions_used":self.used_promotions,"promotions_unused_before_eviction":self.unused_before_eviction,
            "promotions_awaiting_use":self.promoted_unused.len(),"accounted_commits":self.commits,
            "movement_events":self.last_event,"staged_bytes":self.staged_bytes,"copy_gpu_ns":self.copy_gpu_ns,
            "timed_copy_transactions":self.timed_copies,"publication_host_ns":self.publication_ns,
            "untimed_copy_transactions":self.untimed_copies,
            "submission_gate_closed_ns":self.gate_ns,"full_movement_charge_ns":self.charged_ns,
            "modeled_frames":self.estimated_frames,"movement_overlap_frames":self.overlap_frames,
            "estimated_execution_saved_ns":(self.estimated_frames>0 && self.error.is_none()).then_some(self.estimated_saved_ns),
            "estimated_balance_ns":(self.estimated_frames>0 && self.error.is_none() && self.untimed_copies==0).then_some(self.estimated_saved_ns-self.charged_ns as f64),
            "models":shapes,"recent_frames":self.rows,
            "scope":"Actual executed target routes versus unchanged initial placement; draft experts excluded. Startup movement excluded. Distinct weight extents are features, not measured memory traffic. Cost fits use prior uncontended frames with matching graph, batch and target rows; observational, not causal. Balance covers modeled frames and charges all completed GPU copy time plus CPU publication time; it is resource accounting, not serving time saved. Reader drain is reported separately because it can be useful inference. Copy interference and predictor overhead are not attributed. Unused resident promotions are censored, not wasted."})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallax::{LayerDemand,RoutingSample};
    fn expert(expert:usize)->ExpertId { ExpertId {layer:3,expert} }
    fn ledger()->Ledger {
        Ledger::new(Snapshot { catalog_id:"fixture".into(), generation:1,
            experts:(0..4).map(|i|(expert(i),i==0)).collect(),free_slots:1,hbm_allocated_bytes:100,
            backing_allocated_bytes:400,readers:0,pending:false,poisoned:false,statistics:Default::default() },
            (0..4).map(|i|(expert(i),100,1)).collect())
    }
    fn event()->MovementEvent {
        MovementEvent {sequence:1,generation:1,started_ns:100,finished_ns:200,deadline_ns:300,target_demand:None,
            promote:vec![expert(1)],evict:vec![expert(0)],outcome:"committed:2".into(),
            staged_bytes:100,copy_gpu_ns:Some(10),publication_host_ns:5,submission_gate_closed_ns:20}
    }
    fn record(sequence:u64,generation:u64,start:u64)->Record {
        Record {sequence,graph:1,phase:"verify",batch_size:1,concurrency:1,
            sample:RoutingSample {submission:sequence,residency_generation:Some(generation),gpu_ns:Some(50),
                demand_lower_bound_ns:start,available_ns:start+10,
                layers:vec![LayerDemand {layer:3,tokens:5,top_k:1,counts:vec![2,3,0,0]}]}}
    }
    #[test]
    fn late_observation_uses_its_execution_generation_and_cancelled_copies_are_charged() {
        let mut l=ledger();l.ingest(vec![event()]);
        l.observe(&record(1,1,20),None);
        assert_eq!(l.report()["cold_routes_avoided_vs_static"],0);
        l.observe(&record(2,2,201),None);
        let r=l.report();
        assert_eq!(r["cold_routes_avoided_vs_static"],3);
        assert_eq!(r["eviction_routes_vs_static"],2);
        assert_eq!(r["cold_expert_touches_avoided_vs_static"],1);
        assert_eq!(r["promotions_used"],1);
        assert_eq!(r["full_movement_charge_ns"],15); // Reader drain is not a measured execution loss.
        let mut cancelled=event();cancelled.sequence=2;cancelled.generation=2;
        cancelled.outcome="Cancelled".into();cancelled.publication_host_ns=0;cancelled.submission_gate_closed_ns=0;
        l.ingest(vec![cancelled]);l.observe(&record(3,2,220),None);
        assert_eq!(l.report()["full_movement_charge_ns"],25);
        assert_eq!(l.report()["accounted_commits"],1);
        assert!(l.report()["estimated_balance_ns"].is_null());
        l.observe(&record(4,1,250),None);
        assert!(l.report()["error"].is_string());
    }
    #[test]
    fn incomplete_history_is_not_scored_and_overlapping_work_does_not_train() {
        let mut l=ledger();
        l.observe(&record(1,1,20),Some(10));
        assert_eq!(l.report()["models"][0]["samples"],0);
        let mut e=event();e.sequence=2;l.ingest(vec![e]);
        l.observe(&record(2,2,210),None);
        assert_eq!(l.report()["observed_frames"],1);
        assert!(l.report()["error"].is_string());
    }
    #[test]
    fn regression_controls_total_expert_work_and_scores_before_learning() {
        let mut shape=Shape::default();
        for i in 0..64 {
            let total=100.+(i%7) as f64*10.;let cold=20.+(i%11) as f64*2.;
            shape.observe(Point {total_mib:total,cold_mib:cold,gpu_ms:3.+0.2*total+0.7*cold});
        }
        let fit=shape.fit().unwrap();assert!((fit.cold-0.7).abs()<1e-8);assert!((fit.total-0.2).abs()<1e-8);
        assert_eq!(shape.evaluated,32);assert!(shape.absolute_error_ms<1e-8);
        shape.observe(Point {total_mib:100.,cold_mib:20.,gpu_ms:1000.});
        assert!(shape.absolute_error_ms>900.);
        for _ in 0..512 {shape.observe(Point {total_mib:100.,cold_mib:20.,gpu_ms:30.});}
        assert_eq!(shape.points.len(),WINDOW);assert!(shape.fit().is_none());
    }
}

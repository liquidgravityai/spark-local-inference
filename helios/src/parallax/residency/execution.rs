//! One executor owns publication; policy threads can only enqueue a bounded plan.
use super::*;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering},
    },
};

trait Storage: Send {
    fn expert_storage(&self) -> Vec<(ExpertId, u64, usize)>;
    fn snapshot(&self) -> Snapshot;
    fn statistics(&self) -> Statistics;
    fn reader(&mut self) -> Result<ReaderLease>;
    fn begin(&mut self, plan: Plan, now: u64) -> Result<Progress>;
    unsafe fn progress(&mut self, now: u64) -> Result<Progress>;
    fn cancel(&mut self) -> Result<Progress>;
    fn closeable(&mut self) -> Result<()>;
}
impl<B: Backend + Send> Storage for Controller<B> {
    fn expert_storage(&self) -> Vec<(ExpertId, u64, usize)> {
        self.expert_storage()
    }
    fn snapshot(&self) -> Snapshot {
        self.snapshot()
    }
    fn statistics(&self) -> Statistics { self.statistics() }
    fn reader(&mut self) -> Result<ReaderLease> {
        self.acquire_reader()
    }
    fn begin(&mut self, plan: Plan, now: u64) -> Result<Progress> {
        self.begin(plan, now)
    }
    unsafe fn progress(&mut self, now: u64) -> Result<Progress> {
        unsafe { self.progress(now) }
    }
    fn cancel(&mut self) -> Result<Progress> {
        self.cancel()
    }
    fn closeable(&mut self) -> Result<()> {
        let snapshot = self.snapshot();
        if snapshot.poisoned || snapshot.readers != 0 {
            return Err(Error::Poisoned(
                "expert storage has unconfirmed readers or mappings".into(),
            ));
        }
        if matches!(self.cancel()?, Progress::WaitingForTransfer) {
            return Err(Error::Busy);
        }
        Ok(())
    }
}
#[derive(Clone, Debug)]
#[cfg_attr(
    any(feature = "server", feature = "runtime-package"),
    derive(serde::Serialize)
)]
pub struct MovementEvent {
    pub sequence: u64,
    pub generation: u64,
    pub started_ns: u64,
    pub finished_ns: u64,
    pub deadline_ns: u64,
    pub target_demand: Option<u64>,
    pub promote: Vec<ExpertId>,
    pub evict: Vec<ExpertId>,
    pub outcome: String,
    pub staged_bytes: u64,
    pub copy_gpu_ns: Option<u64>,
    pub publication_host_ns: u64,
    /// Includes reader drain and publication; not necessarily GPU idle time.
    pub submission_gate_closed_ns: u64,
}
pub struct ExecutionReport {
    pub snapshot: Snapshot,
    pub events: Vec<MovementEvent>,
    pub event_count: u64,
    pub disabled: Option<String>,
}
struct ActiveMovement {
    plan: Plan,
    started: u64,
    target: Option<u64>,
    before: Statistics,
    gate_closed_at: Option<u64>,
}
struct State {
    storage: Option<Box<dyn Storage>>,
    queued: Option<(Plan, Option<u64>)>,
    claimed: bool,
    disabled: Option<String>,
    active: Option<ActiveMovement>,
    events: VecDeque<MovementEvent>,
    event_count: u64,
}
impl State {
    fn record(
        &mut self,
        plan: Plan,
        started_ns: u64,
        finished_ns: u64,
        outcome: String,
        target_demand: Option<u64>,
        before: Statistics,
        gate_closed_at: Option<u64>,
    ) {
        let after = self.storage.as_ref().unwrap().statistics();
        self.event_count += 1;
        if self.events.len() == 256 {
            self.events.pop_front();
        }
        self.events.push_back(MovementEvent {
            sequence: self.event_count,
            generation: plan.generation,
            started_ns,
            finished_ns,
            deadline_ns: plan.valid_until_ns,
            target_demand,
            promote: plan.promote,
            evict: plan.evict,
            outcome,
            staged_bytes: after.staged_bytes.saturating_sub(before.staged_bytes),
            copy_gpu_ns: (after.timed_copies > before.timed_copies)
                .then(|| after.copy_gpu_ns.saturating_sub(before.copy_gpu_ns)),
            publication_host_ns: after.publication_host_ns.saturating_sub(before.publication_host_ns),
            submission_gate_closed_ns: gate_closed_at.map_or(0, |at| finished_ns.saturating_sub(at)),
        });
    }
}

struct Shared {
    state: Mutex<State>,
    latest_demand: AtomicU64,
}
#[derive(Clone)]
pub struct ExecutionBinding(Arc<Shared>);
impl std::fmt::Debug for ExecutionBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionBinding").finish_non_exhaustive()
    }
}
impl ExecutionBinding {
    pub fn new<B: Backend + Send + 'static>(controller: Controller<B>) -> Self {
        Self(Arc::new(Shared {
            latest_demand: AtomicU64::new(0),
            state: Mutex::new(State {
                storage: Some(Box::new(controller)),
                queued: None,
                claimed: false,
                disabled: None,
                active: None,
                events: VecDeque::with_capacity(256),
                event_count: 0,
            }),
        }))
    }
    fn state(&self) -> Result<MutexGuard<'_, State>> {
        self.0
            .state
            .lock()
            .map_err(|_| Error::Poisoned("residency owner lock failed".into()))
    }
    pub fn record_demand(&self, sequence: u64) {
        self.0.latest_demand.fetch_max(sequence, Ordering::Release);
    }
    pub fn latest_demand(&self) -> u64 {
        self.0.latest_demand.load(Ordering::Acquire)
    }
    pub fn expert_storage(&self) -> Result<Vec<(ExpertId, u64, usize)>> {
        self.state()?
            .storage
            .as_ref()
            .map(|s| s.expert_storage())
            .ok_or_else(|| Error::Invalid("expert storage is closed".into()))
    }
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.state()?
            .storage
            .as_ref()
            .map(|s| s.snapshot())
            .ok_or_else(|| Error::Invalid("expert storage is closed".into()))
    }
    pub fn report(&self) -> Result<ExecutionReport> {
        let state = self.state()?;
        Ok(ExecutionReport {
            snapshot: state
                .storage
                .as_ref()
                .ok_or_else(|| Error::Invalid("expert storage is closed".into()))?
                .snapshot(),
            events: state.events.iter().cloned().collect(),
            event_count: state.event_count,
            disabled: state.disabled.clone(),
        })
    }
    pub fn events_after(&self, sequence: u64) -> Result<(Vec<MovementEvent>, Option<u64>)> {
        let state = self.state()?;
        Ok((state.events.iter().filter(|e| e.sequence > sequence).cloned().collect(),
            state.active.as_ref().map(|a| a.started)))
    }
    pub fn submit(&self, plan: Plan) -> Result<()> {
        self.enqueue(plan, None)
    }
    pub fn submit_for_demand(&self, plan: Plan, sequence: u64) -> Result<()> {
        self.enqueue(plan, Some(sequence))
    }
    fn enqueue(&self, plan: Plan, target: Option<u64>) -> Result<()> {
        if target.is_some_and(|s| self.latest_demand() >= s) {
            return Err(Error::Expired);
        }
        if plan.promote.len() > 64 || plan.evict.len() > 64 {
            return Err(Error::Invalid(
                "serving plans are limited to 64 promotions and 64 evictions".into(),
            ));
        }
        let mut state = self.state()?;
        if state.storage.is_none() || !state.claimed || state.disabled.is_some() {
            return Err(Error::Invalid("expert movement is unavailable".into()));
        }
        if state.queued.is_some() {
            return Err(Error::Busy);
        }
        state.queued = Some((plan, target));
        Ok(())
    }
    /// # Safety
    /// No graph may run or claim storage until initialization returns.
    pub(crate) unsafe fn initialize(&self, plan: Plan) -> Result<()> {
        let mut state = self.state()?;
        if state.claimed {
            return Err(Error::Busy);
        }
        let storage = state
            .storage
            .as_mut()
            .ok_or_else(|| Error::Invalid("expert storage is closed".into()))?;
        storage.begin(plan, 0)?;
        loop {
            match unsafe { storage.progress(0) }? {
                Progress::WaitingForTransfer => std::thread::yield_now(),
                Progress::Noop | Progress::Committed(_) => return Ok(()),
                _ => {
                    return Err(Error::Invalid(
                        "initial placement has an unexpected reader".into(),
                    ));
                }
            }
        }
    }
    /// # Safety
    /// This is the only executor for every graph reading this storage. It keeps
    /// a reader lease through each completion and stops submissions at publication.
    pub unsafe fn claim(&self) -> Result<ExecutionGate> {
        let mut state = self.state()?;
        if state.claimed || state.storage.is_none() {
            return Err(Error::Invalid(
                "expert storage requires exactly one executor".into(),
            ));
        }
        state.claimed = true;
        Ok(ExecutionGate {
            binding: self.clone(),
            paused: false,
        })
    }
    /// # Safety
    /// The context is drained, no executor can submit, and address reservations
    /// remain owned until this returns. Failure requires retaining that context.
    pub unsafe fn close(&self) -> Result<()> {
        let mut state = self.state()?;
        if let Some(storage) = state.storage.as_mut() {
            storage.closeable()?;
        }
        state.queued = None;
        drop(state.storage.take());
        Ok(())
    }
}

pub struct ExecutionGate {
    binding: ExecutionBinding,
    paused: bool,
}
impl ExecutionGate {
    pub fn paused(&self) -> bool {
        self.paused
    }
    pub fn reader(&mut self) -> Result<ReaderLease> {
        if self.paused {
            return Err(Error::Busy);
        }
        self.binding
            .state()?
            .storage
            .as_mut()
            .ok_or_else(|| Error::Invalid("expert storage is closed".into()))?
            .reader()
    }
    /// # Safety
    /// No other thread may launch a graph using this storage during this call.
    pub unsafe fn tick(&mut self, now: u64) -> Result<()> {
        let mut state = self.binding.state()?;
        if state.disabled.is_some() {
            return Ok(());
        }
        if state.storage.is_none() {
            return Err(Error::Invalid("expert storage is closed".into()));
        }
        if let Some((plan, target)) = state.queued.take() {
            let before = state.storage.as_ref().unwrap().statistics();
            let begin = if target.is_some_and(|s| self.binding.latest_demand() >= s) {
                Err(Error::Expired)
            } else {
                state.storage.as_mut().unwrap().begin(plan.clone(), now)
            };
            match begin {
                Ok(Progress::Staged) => state.active = Some(ActiveMovement { plan, started: now, target, before, gate_closed_at: None }),
                Ok(_) => state.record(plan, now, now, "no_movement".into(), target, before, None),
                Err(Error::Busy) => state.queued = Some((plan, target)),
                Err(error) => {
                    state.record(
                        plan,
                        now,
                        crate::parallax::now_ns().max(now),
                        format!("rejected: {error}"),
                        target,
                        before,
                        None,
                    );
                    match error {
                        Error::Poisoned(e) => return Err(Error::Poisoned(e)),
                        Error::Invalid(_) | Error::Stale | Error::Expired | Error::Capacity => {}
                        error => {
                            state.disabled = Some(error.to_string());
                            self.paused = false;
                            return Ok(());
                        }
                    }
                }
            }
        }
        let missed = state
            .active
            .as_ref()
            .and_then(|active| active.target)
            .is_some_and(|s| self.binding.latest_demand() >= s);
        let result = if missed {
            state.storage.as_mut().unwrap().cancel()
        } else {
            unsafe { state.storage.as_mut().unwrap().progress(now) }
        };
        if matches!(result, Ok(Progress::WaitingForReaders)) {
            if let Some(active) = &mut state.active { active.gate_closed_at.get_or_insert(now); }
        }
        if !matches!(
            result,
            Ok(Progress::WaitingForReaders | Progress::WaitingForTransfer)
        ) {
            if let Some(active) = state.active.take() {
                let outcome = match &result {
                    Ok(Progress::Committed(generation)) => format!("committed:{generation}"),
                    Ok(other) => format!("{other:?}"),
                    Err(error) => format!("cancelled_or_failed: {error}"),
                };
                state.record(
                    active.plan,
                    active.started,
                    crate::parallax::now_ns().max(now),
                    outcome,
                    active.target,
                    active.before,
                    active.gate_closed_at,
                );
            }
        }
        match result {
            Ok(Progress::WaitingForReaders) => self.paused = true,
            Ok(_) | Err(Error::Expired) => self.paused = false,
            Err(Error::Poisoned(e)) => return Err(Error::Poisoned(e)),
            Err(e) => {
                state.disabled = Some(e.to_string());
                self.paused = false;
            }
        }
        Ok(())
    }
}

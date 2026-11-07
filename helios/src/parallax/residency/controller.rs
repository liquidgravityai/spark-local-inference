use super::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Clone, Debug, Default)]
pub struct Statistics {
    pub transfers: u64,
    pub staged_bytes: u64,
    pub commits: u64,
    pub publication_host_ns: u64,
    pub copy_gpu_ns: u64,
    pub timed_copies: u64,
    pub promoted_experts: u64,
    pub evicted_experts: u64,
    pub committed_copy_bytes: u64,
    pub rollbacks: u64,
    pub cancellations: u64,
    pub rejected: u64,
}
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub catalog_id: String,
    pub generation: u64,
    pub experts: BTreeMap<ExpertId, bool>,
    pub free_slots: usize,
    pub hbm_allocated_bytes: usize,
    pub backing_allocated_bytes: usize,
    pub readers: usize,
    pub pending: bool,
    pub poisoned: bool,
    pub statistics: Statistics,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Progress {
    Noop,
    Staged,
    WaitingForTransfer,
    WaitingForReaders,
    Committed(u64),
    Cancelled,
}

/// Dropping an unfinished lease deliberately leaves its reader registered. A lost
/// fence must prevent reuse, not authorize replacement of potentially live bytes.
pub struct ReaderLease {
    readers: Arc<AtomicUsize>,
    pub generation: u64,
}
impl ReaderLease {
    /// # Safety
    /// Every GPU operation issued under this lease has completed. This includes
    /// queued graph launches, side streams and asynchronous reads of expert data.
    pub unsafe fn complete(self) {
        self.readers.fetch_sub(1, Ordering::AcqRel);
    }
}
struct Pending {
    plan: Plan,
    bindings: Vec<Binding>,
    slots: Vec<usize>,
    bytes: u64,
    timing_recorded: bool,
}

pub struct Controller<B: Backend> {
    catalog_id: String,
    backend: Option<B>,
    experts: BTreeMap<ExpertId, Vec<usize>>,
    locations: Vec<Location>,
    generation: u64,
    readers: Arc<AtomicUsize>,
    pending: Option<Pending>,
    poisoned: Option<String>,
    statistics: Statistics,
}
impl<B: Backend> Controller<B> {
    pub fn statistics(&self) -> Statistics { self.statistics.clone() }
    fn record_copy_timing(&mut self) {
        if let Some(pending) = &mut self.pending {
            if !pending.timing_recorded {
                if let Ok(Some(ns)) = self.backend.as_ref().unwrap().copy_elapsed_ns() {
                    self.statistics.copy_gpu_ns = self.statistics.copy_gpu_ns.saturating_add(ns);
                    self.statistics.timed_copies += 1;
                }
                pending.timing_recorded = true;
            }
        }
    }
    pub fn new(catalog_id: String, backend: B) -> Result<Self> {
        if catalog_id.trim().is_empty() {
            return Err(Error::Invalid(
                "storage requires a package and representation identity".into(),
            ));
        }
        let mut experts = BTreeMap::<ExpertId, Vec<usize>>::new();
        let mut identities = BTreeSet::new();
        for (i, r) in backend.regions().iter().enumerate() {
            if r.bytes == 0
                || r.bytes > backend.slot_bytes()
                || !identities.insert((r.expert, r.part))
            {
                return Err(Error::Invalid("invalid or duplicate expert region".into()));
            }
            experts.entry(r.expert).or_default().push(i);
        }
        if experts.is_empty()
            || backend.slots() == 0
            || (0..backend.slots()).any(|slot| backend.slot_capacity(slot) == 0)
            || (0..backend.slots())
                .try_fold(0usize, |n, slot| n.checked_add(backend.slot_capacity(slot)))
                .is_none()
        {
            return Err(Error::Invalid(
                "empty catalog or invalid staging capacity".into(),
            ));
        }
        let locations = vec![Location::Backing; backend.regions().len()];
        Ok(Self {
            catalog_id,
            backend: Some(backend),
            experts,
            locations,
            generation: 0,
            readers: Arc::new(AtomicUsize::new(0)),
            pending: None,
            poisoned: None,
            statistics: Statistics::default(),
        })
    }
    fn healthy(&self) -> Result<()> {
        match &self.poisoned {
            Some(e) => Err(Error::Poisoned(e.clone())),
            None => Ok(()),
        }
    }
    fn failure<T>(&mut self, error: Error) -> Result<T> {
        if let Error::Poisoned(message) = &error {
            self.poisoned = Some(message.clone());
        }
        Err(error)
    }
    fn failed_pending<T>(&mut self, error: Error) -> Result<T> {
        if matches!(error, Error::Poisoned(_)) {
            return self.failure(error);
        }
        if let Err(cleanup) = self.backend.as_mut().unwrap().discard() {
            return self.failure(Error::Poisoned(format!(
                "transaction failed ({error}); staging cleanup failed ({cleanup})"
            )));
        }
        self.pending = None;
        self.failure(error)
    }
    pub fn acquire_reader(&mut self) -> Result<ReaderLease> {
        self.healthy()?;
        let count = self.readers.load(Ordering::Acquire);
        count
            .checked_add(1)
            .ok_or_else(|| Error::Invalid("reader count overflow".into()))?;
        self.readers.fetch_add(1, Ordering::AcqRel);
        Ok(ReaderLease {
            readers: self.readers.clone(),
            generation: self.generation,
        })
    }
    pub fn address(&self, expert: ExpertId, part: usize) -> Result<u64> {
        self.healthy()?;
        let backend = self.backend.as_ref().unwrap();
        self.experts
            .get(&expert)
            .and_then(|parts| parts.iter().find(|i| backend.regions()[**i].part == part))
            .map(|i| backend.address(*i))
            .ok_or_else(|| Error::Invalid("unknown expert part".into()))
    }
    pub fn expert_storage(&self) -> Vec<(ExpertId, u64, usize)> {
        let backend = self.backend.as_ref().unwrap();
        self.experts
            .iter()
            .map(|(expert, parts)| {
                (
                    *expert,
                    parts
                        .iter()
                        .map(|i| backend.transfer_bytes(*i) as u64)
                        .sum(),
                    parts.len(),
                )
            })
            .collect()
    }
    pub fn snapshot(&self) -> Snapshot {
        let backend = self.backend.as_ref().unwrap();
        let mut used: BTreeSet<usize> = self
            .locations
            .iter()
            .filter_map(|l| {
                if let Location::Hbm(slot) = l {
                    Some(*slot)
                } else {
                    None
                }
            })
            .collect();
        if let Some(p) = &self.pending {
            used.extend(&p.slots);
        }
        Snapshot {
            catalog_id: self.catalog_id.clone(),
            generation: self.generation,
            experts: self
                .experts
                .iter()
                .map(|(e, parts)| {
                    (
                        *e,
                        parts
                            .iter()
                            .all(|i| matches!(self.locations[*i], Location::Hbm(_))),
                    )
                })
                .collect(),
            free_slots: backend.slots() - used.len(),
            hbm_allocated_bytes: (0..backend.slots())
                .map(|slot| backend.slot_capacity(slot))
                .sum(),
            backing_allocated_bytes: backend.backing_bytes(),
            readers: self.readers.load(Ordering::Acquire),
            pending: self.pending.is_some(),
            poisoned: self.poisoned.is_some(),
            statistics: self.statistics.clone(),
        }
    }
    pub fn begin(&mut self, plan: Plan, now_ns: u64) -> Result<Progress> {
        let result = self.begin_checked(plan, now_ns);
        if result.is_err() {
            self.statistics.rejected += 1;
        }
        result
    }
    fn begin_checked(&mut self, plan: Plan, now_ns: u64) -> Result<Progress> {
        self.healthy()?;
        if self.pending.is_some() {
            return Err(Error::Busy);
        }
        if plan.version != 1 {
            return Err(Error::Invalid("unsupported movement plan version".into()));
        }
        if plan.catalog_id != self.catalog_id {
            return Err(Error::Invalid(
                "movement plan names a different package or representation".into(),
            ));
        }
        if plan.generation != self.generation || self.generation == u64::MAX {
            return Err(Error::Stale);
        }
        if now_ns > plan.valid_until_ns {
            return Err(Error::Expired);
        }
        let mut unique = BTreeSet::new();
        for (promotion, ids) in [(true, &plan.promote), (false, &plan.evict)] {
            for id in ids {
                if !unique.insert(*id) {
                    return Err(Error::Invalid(
                        "duplicate or conflicting expert movement".into(),
                    ));
                }
                let parts = self
                    .experts
                    .get(id)
                    .ok_or_else(|| Error::Invalid("unknown canonical expert".into()))?;
                if parts
                    .iter()
                    .any(|i| matches!(self.locations[*i], Location::Hbm(_)) == promotion)
                {
                    return Err(Error::Invalid(
                        "movement disagrees with authoritative residency".into(),
                    ));
                }
            }
        }
        if unique.is_empty() {
            return Ok(Progress::Noop);
        }
        let backend = self.backend.as_mut().unwrap();
        let used: BTreeSet<_> = self
            .locations
            .iter()
            .filter_map(|l| {
                if let Location::Hbm(s) = l {
                    Some(*s)
                } else {
                    None
                }
            })
            .collect();
        let mut reserved = used;
        let mut copies = vec![];
        let mut bindings = vec![];
        let mut slots = vec![];
        let mut bytes = 0u64;
        for expert in &plan.promote {
            for &region in &self.experts[expert] {
                let slot = (0..backend.slots())
                    .find(|slot| !reserved.contains(slot) && backend.slot_accepts(region, *slot))
                    .ok_or(Error::Capacity)?;
                reserved.insert(slot);
                bytes = bytes
                    .checked_add(backend.transfer_bytes(region) as u64)
                    .ok_or_else(|| Error::Invalid("movement byte count overflow".into()))?;
                copies.push((region, slot));
                slots.push(slot);
                bindings.push(Binding {
                    region,
                    from: Location::Backing,
                    to: Location::Hbm(slot),
                });
            }
        }
        for expert in &plan.evict {
            for &region in &self.experts[expert] {
                bindings.push(Binding {
                    region,
                    from: self.locations[region],
                    to: Location::Backing,
                });
            }
        }
        if let Err(error) = backend.stage(&copies) {
            return self.failure(error);
        }
        self.statistics.transfers += copies.len() as u64;
        self.statistics.staged_bytes += bytes;
        self.pending = Some(Pending {
            plan,
            bindings,
            slots,
            bytes,
            timing_recorded: false,
        });
        Ok(Progress::Staged)
    }
    /// # Safety
    /// All context users participate in this controller's reader protocol, and
    /// the caller excludes new graph submissions until this call returns.
    pub unsafe fn progress(&mut self, now_ns: u64) -> Result<Progress> {
        self.healthy()?;
        let Some(_) = &self.pending else {
            return Ok(Progress::Noop);
        };
        let ready = match self.backend.as_mut().unwrap().ready() {
            Ok(v) => v,
            Err(e) => return self.failed_pending(e),
        };
        if !ready {
            return Ok(Progress::WaitingForTransfer);
        }
        self.record_copy_timing();
        let pending = self.pending.as_ref().unwrap();
        if now_ns > pending.plan.valid_until_ns {
            if let Err(e) = self.backend.as_mut().unwrap().discard() {
                return self.failure(e);
            }
            self.pending = None;
            self.statistics.cancellations += 1;
            return Err(Error::Expired);
        }
        if self.readers.load(Ordering::Acquire) != 0 {
            return Ok(Progress::WaitingForReaders);
        }
        let started = std::time::Instant::now();
        let result = unsafe { self.backend.as_mut().unwrap().publish(&pending.bindings) };
        self.statistics.publication_host_ns = self
            .statistics
            .publication_host_ns
            .saturating_add(started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
        if let Err(error) = result {
            self.statistics.rollbacks += 1;
            return self.failed_pending(error);
        }
        let pending = self.pending.take().unwrap();
        for binding in pending.bindings {
            self.locations[binding.region] = binding.to;
        }
        self.statistics.promoted_experts += pending.plan.promote.len() as u64;
        self.statistics.evicted_experts += pending.plan.evict.len() as u64;
        self.statistics.committed_copy_bytes += pending.bytes;
        self.generation += 1;
        self.statistics.commits += 1;
        Ok(Progress::Committed(self.generation))
    }
    pub fn cancel(&mut self) -> Result<Progress> {
        self.healthy()?;
        if self.pending.is_none() {
            return Ok(Progress::Noop);
        }
        match self.backend.as_mut().unwrap().ready() {
            Ok(false) => Ok(Progress::WaitingForTransfer),
            Err(e) => self.failed_pending(e),
            Ok(true) => {
                self.record_copy_timing();
                if let Err(e) = self.backend.as_mut().unwrap().discard() {
                    return self.failure(e);
                }
                self.pending = None;
                self.statistics.cancellations += 1;
                Ok(Progress::Cancelled)
            }
        }
    }
}
impl<B: Backend> Drop for Controller<B> {
    fn drop(&mut self) {
        if self.readers.load(Ordering::Acquire) != 0 || self.poisoned.is_some() {
            // A missing completion fence must retain storage, even during error cleanup.
            if let Some(backend) = self.backend.take() {
                std::mem::forget(backend);
            }
        }
    }
}

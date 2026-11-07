use super::*;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
struct State {
    ready: bool,
    stage_error: Option<Error>,
    fail_mapping: Option<usize>,
    fail_rollback: bool,
    mapping_calls: usize,
    copies: usize,
    published: Vec<Option<Location>>,
}
struct Memory {
    regions: Vec<Region>,
    slots: usize,
    state: Arc<Mutex<State>>,
    dropped: Arc<AtomicUsize>,
}
impl Drop for Memory {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

// The fake backend exercises transaction ownership and fault recovery, not CUDA.
unsafe impl Backend for Memory {
    fn regions(&self) -> &[Region] {
        &self.regions
    }
    fn slots(&self) -> usize {
        self.slots
    }
    fn slot_bytes(&self) -> usize {
        64
    }
    fn backing_bytes(&self) -> usize {
        self.regions.len() * 64
    }
    fn transfer_bytes(&self, _: usize) -> usize {
        64
    }
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            stable_addresses: true,
            asynchronous_staging: true,
            publication_requires_quiescence: true,
            cold_moe_execution_qualified: false,
        }
    }
    fn address(&self, region: usize) -> u64 {
        4096 + self.regions[region].offset as u64
    }
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        if let Some(e) = state.stage_error.take() {
            return Err(e);
        }
        state.copies += copies.len();
        Ok(())
    }
    fn ready(&mut self) -> Result<bool> {
        Ok(self.state.lock().unwrap().ready)
    }
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> { Ok(Some(123)) }
    fn discard(&mut self) -> Result<()> {
        if self.ready()? {
            Ok(())
        } else {
            Err(Error::Busy)
        }
    }
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        transaction::publish(bindings, |region, location| {
            state.mapping_calls += 1;
            // Reproduce a failure after the old mapping was already removed.
            state.published[region] = None;
            if state.fail_mapping == Some(state.mapping_calls)
                || (state.fail_rollback
                    && state
                        .fail_mapping
                        .is_some_and(|call| state.mapping_calls > call))
            {
                return Err(Error::Backend("injected map failure".into()));
            }
            state.published[region] = Some(location);
            Ok(())
        })
    }
}
fn expert(expert: usize) -> ExpertId {
    ExpertId { layer: 3, expert }
}
fn setup(slots: usize) -> (Controller<Memory>, Arc<Mutex<State>>, Arc<AtomicUsize>) {
    let state = Arc::new(Mutex::new(State {
        ready: true,
        published: vec![Some(Location::Backing); 4],
        ..State::default()
    }));
    let dropped = Arc::new(AtomicUsize::new(0));
    let regions = (0..4)
        .map(|i| Region {
            expert: expert(i / 2),
            part: i % 2,
            offset: i * 64,
            bytes: 37,
        })
        .collect();
    let controller = Controller::new(
        "synthetic-native".into(),
        Memory {
            regions,
            slots,
            state: state.clone(),
            dropped: dropped.clone(),
        },
    )
    .unwrap();
    (controller, state, dropped)
}
fn plan(generation: u64, promote: &[usize], evict: &[usize]) -> Plan {
    Plan {
        version: 1,
        catalog_id: "synthetic-native".into(),
        generation,
        valid_until_ns: 100,
        promote: promote.iter().copied().map(expert).collect(),
        evict: evict.iter().copied().map(expert).collect(),
    }
}

#[test]
fn publication_waits_for_both_transfer_and_readers() {
    let (mut c, state, _) = setup(4);
    let addresses: Vec<_> = (0..2)
        .map(|part| c.address(expert(0), part).unwrap())
        .collect();
    let reader = c.acquire_reader().unwrap();
    state.lock().unwrap().ready = false;
    assert_eq!(c.begin(plan(0, &[0], &[]), 0).unwrap(), Progress::Staged);
    assert_eq!(
        unsafe { c.progress(1) }.unwrap(),
        Progress::WaitingForTransfer
    );
    state.lock().unwrap().ready = true;
    assert_eq!(
        unsafe { c.progress(2) }.unwrap(),
        Progress::WaitingForReaders
    );
    assert_eq!(c.snapshot().generation, 0);
    assert!(
        state
            .lock()
            .unwrap()
            .published
            .iter()
            .all(|v| *v == Some(Location::Backing))
    );
    unsafe {
        reader.complete();
    }
    assert_eq!(unsafe { c.progress(3) }.unwrap(), Progress::Committed(1));
    assert!(c.snapshot().experts[&expert(0)]);
    assert!(!c.snapshot().experts[&expert(1)]);
    assert_eq!(c.snapshot().statistics.staged_bytes, 128); // physical padding is charged
    assert_eq!(
        (0..2)
            .map(|part| c.address(expert(0), part).unwrap())
            .collect::<Vec<_>>(),
        addresses
    );
}

#[test]
fn exchanges_need_spare_capacity_even_when_victims_will_be_evicted() {
    let (mut c, state, _) = setup(2);
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    unsafe { c.progress(1) }.unwrap();
    assert_eq!(c.begin(plan(1, &[1], &[0]), 2), Err(Error::Capacity));
    assert_eq!(state.lock().unwrap().copies, 2);
    assert!(c.snapshot().experts[&expert(0)]);
    c.begin(plan(1, &[], &[0]), 3).unwrap();
    unsafe { c.progress(4) }.unwrap();
    c.begin(plan(2, &[1], &[]), 5).unwrap();
    unsafe { c.progress(6) }.unwrap();
    assert!(c.snapshot().experts[&expert(1)]);
}

#[test]
fn cancellation_and_expiration_retain_slots_until_copy_completion() {
    let (mut c, state, _) = setup(4);
    state.lock().unwrap().ready = false;
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert_eq!(c.cancel().unwrap(), Progress::WaitingForTransfer);
    assert_eq!(c.snapshot().free_slots, 2);
    assert_eq!(c.begin(plan(0, &[1], &[]), 1), Err(Error::Busy));
    assert_eq!(
        unsafe { c.progress(101) }.unwrap(),
        Progress::WaitingForTransfer
    );
    state.lock().unwrap().ready = true;
    assert_eq!(unsafe { c.progress(102) }, Err(Error::Expired));
    assert_eq!(c.snapshot().free_slots, 4);
    assert_eq!(c.snapshot().generation, 0);
    c.begin(plan(0, &[1], &[]), 1).unwrap();
    assert_eq!(c.cancel().unwrap(), Progress::Cancelled);
    assert_eq!(c.snapshot().free_slots, 4);
}

#[test]
fn invalid_and_noop_plans_do_not_submit_copies() {
    let (mut c, state, _) = setup(4);
    assert_eq!(c.begin(plan(1, &[0], &[]), 0), Err(Error::Stale));
    assert_eq!(c.begin(plan(0, &[0], &[]), 101), Err(Error::Expired));
    for bad in [
        plan(0, &[0, 0], &[]),
        plan(0, &[0], &[0]),
        plan(0, &[99], &[]),
        plan(0, &[], &[0]),
    ] {
        assert!(matches!(c.begin(bad, 0), Err(Error::Invalid(_))));
    }
    let mut unsupported = plan(0, &[0], &[]);
    unsupported.version = 2;
    assert!(matches!(c.begin(unsupported, 0), Err(Error::Invalid(_))));
    let mut foreign = plan(0, &[0], &[]);
    foreign.catalog_id = "different-package".into();
    assert!(matches!(c.begin(foreign, 0), Err(Error::Invalid(_))));
    assert_eq!(c.begin(plan(0, &[], &[]), 0).unwrap(), Progress::Noop);
    assert_eq!(state.lock().unwrap().copies, 0);
    assert!(!c.snapshot().pending);
}

#[test]
fn recoverable_copy_failure_preserves_capacity_and_allows_retry() {
    let (mut c, state, _) = setup(4);
    state.lock().unwrap().stage_error = Some(Error::Backend("copy failed after drain".into()));
    assert!(matches!(
        c.begin(plan(0, &[0], &[]), 0),
        Err(Error::Backend(_))
    ));
    assert_eq!(c.snapshot().free_slots, 4);
    assert!(!c.snapshot().pending);
    c.begin(plan(0, &[0], &[]), 1).unwrap();
    assert_eq!(unsafe { c.progress(2) }.unwrap(), Progress::Committed(1));
}

#[test]
fn every_partial_exchange_failure_restores_all_previous_mappings() {
    for failure in 1..=4 {
        let (mut c, state, _) = setup(4);
        c.begin(plan(0, &[0], &[]), 0).unwrap();
        unsafe { c.progress(1) }.unwrap();
        let before = state.lock().unwrap().published.clone();
        {
            let mut s = state.lock().unwrap();
            s.mapping_calls = 0;
            s.fail_mapping = Some(failure);
        }
        c.begin(plan(1, &[1], &[0]), 2).unwrap();
        assert!(matches!(unsafe { c.progress(3) }, Err(Error::Backend(_))));
        assert_eq!(state.lock().unwrap().published, before);
        assert_eq!(c.snapshot().generation, 1);
        assert_eq!(c.snapshot().free_slots, 2);
        assert!(!c.snapshot().pending);
        state.lock().unwrap().fail_mapping = None;
        c.begin(plan(1, &[1], &[0]), 4).unwrap();
        assert_eq!(unsafe { c.progress(5) }.unwrap(), Progress::Committed(2));
        assert!(!c.snapshot().experts[&expert(0)]);
        assert!(c.snapshot().experts[&expert(1)]);
    }
}

#[test]
fn rollback_failure_poisoning_blocks_readers_and_retains_storage() {
    let (mut c, state, dropped) = setup(4);
    {
        let mut s = state.lock().unwrap();
        s.fail_mapping = Some(2);
        s.fail_rollback = true;
    }
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert!(matches!(unsafe { c.progress(1) }, Err(Error::Poisoned(_))));
    assert!(c.snapshot().poisoned);
    assert!(matches!(c.acquire_reader(), Err(Error::Poisoned(_))));
    assert!(matches!(c.address(expert(0), 0), Err(Error::Poisoned(_))));
    assert!(matches!(
        c.begin(plan(0, &[], &[]), 2),
        Err(Error::Poisoned(_))
    ));
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
}

#[test]
fn abandoned_reader_never_authorizes_replacement_or_free() {
    let (mut c, _, dropped) = setup(4);
    drop(c.acquire_reader().unwrap());
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert_eq!(
        unsafe { c.progress(1) }.unwrap(),
        Progress::WaitingForReaders
    );
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 0);
    let (c, _, dropped) = setup(4);
    drop(c);
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
}

#[test]
fn execution_gate_stages_during_readers_then_blocks_submission_for_publication() {
    use execution::ExecutionBinding;
    let (controller, state, dropped) = setup(4);
    let binding = ExecutionBinding::new(controller);
    assert!(binding.submit(plan(0, &[0], &[])).is_err());
    let mut gate = unsafe { binding.claim() }.unwrap();
    assert!(unsafe { binding.claim() }.is_err());
    let reader = gate.reader().unwrap();
    state.lock().unwrap().ready = false;
    binding.submit(plan(0, &[0], &[])).unwrap();
    assert_eq!(binding.submit(plan(0, &[1], &[])), Err(Error::Busy));
    unsafe { gate.tick(1) }.unwrap();
    assert!(!gate.paused());
    assert_eq!(state.lock().unwrap().copies, 2);
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    state.lock().unwrap().ready = true;
    unsafe { gate.tick(2) }.unwrap();
    assert!(gate.paused());
    assert!(matches!(gate.reader(), Err(Error::Busy)));
    assert!(unsafe { binding.close() }.is_err());
    unsafe { reader.complete(); gate.tick(3).unwrap(); }
    assert!(!gate.paused());
    assert_eq!(binding.snapshot().unwrap().generation, 1);
    let report = binding.report().unwrap();
    assert_eq!(report.events.len(),1);
    assert_eq!(report.events[0].outcome,"committed:1");
    assert_eq!(report.events[0].promote,vec![expert(0)]);
    assert_eq!(report.snapshot.statistics.promoted_experts,1);
    assert_eq!(report.snapshot.statistics.committed_copy_bytes,128);
    assert_eq!(state.lock().unwrap().mapping_calls, 2);
    unsafe { binding.close() }.unwrap();
    assert_eq!(dropped.load(Ordering::Relaxed), 1);
    assert!(binding.snapshot().is_err());
    assert!(gate.reader().is_err());
}

#[test]
fn execution_gate_rejects_stale_plans_and_retains_static_storage_after_copy_failure() {
    let (controller, state, _) = setup(4);
    let binding = execution::ExecutionBinding::new(controller);
    let mut gate = unsafe { binding.claim() }.unwrap();
    binding.submit(plan(1, &[0], &[])).unwrap();
    unsafe { gate.tick(1) }.unwrap();
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    state.lock().unwrap().stage_error = Some(Error::Backend("copy failure".into()));
    binding.submit(plan(0, &[0], &[])).unwrap();
    unsafe { gate.tick(2) }.unwrap();
    let reader = gate.reader().unwrap();
    unsafe { reader.complete() };
    assert!(binding.submit(plan(0, &[1], &[])).is_err());
    assert_eq!(binding.snapshot().unwrap().generation, 0);
    unsafe { binding.close() }.unwrap();
}

#[test]
fn movement_timing_is_once_per_copy_and_excludes_initial_placement() {
    let (controller, state, _) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    unsafe { binding.initialize(plan(0,&[0],&[])) }.unwrap();
    let mut gate=unsafe {binding.claim()}.unwrap();
    let reader=gate.reader().unwrap();
    state.lock().unwrap().ready=false;
    binding.submit(plan(1,&[1],&[0])).unwrap();
    unsafe {gate.tick(10)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().statistics.copy_gpu_ns,123);
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(20)}.unwrap();
    unsafe {gate.tick(30)}.unwrap();
    assert!(gate.paused());
    assert_eq!(binding.snapshot().unwrap().statistics.copy_gpu_ns,246);
    unsafe {reader.complete();gate.tick(40)}.unwrap();
    let report=binding.report().unwrap();
    assert_eq!(report.events.len(),1);
    assert_eq!(report.events[0].copy_gpu_ns,Some(123));
    assert_eq!(report.events[0].staged_bytes,128); // Two 64-byte weight parts.
    assert!(report.events[0].submission_gate_closed_ns>=20);
    assert_eq!(report.snapshot.statistics.timed_copies,2);
    unsafe {binding.close()}.unwrap();
}

#[test]
fn execution_gate_stops_on_uncertain_publication_and_initial_placement_is_strict() {
    let (controller, state, _) = setup(4);
    let binding = execution::ExecutionBinding::new(controller);
    assert_eq!(unsafe { binding.initialize(plan(1, &[0], &[])) }, Err(Error::Stale));
    unsafe { binding.initialize(plan(0, &[0], &[])) }.unwrap();
    let mut gate = unsafe { binding.claim() }.unwrap();
    assert_eq!(unsafe { binding.initialize(plan(1, &[1], &[])) }, Err(Error::Busy));
    let mut state = state.lock().unwrap();
    state.fail_mapping = Some(state.mapping_calls + 1);
    state.fail_rollback = true;
    drop(state);
    binding.submit(plan(1, &[1], &[0])).unwrap();
    assert!(matches!(unsafe { gate.tick(2) }, Err(Error::Poisoned(_))));
    assert!(gate.reader().is_err());
    assert!(unsafe { binding.close() }.is_err());
}

#[test]
fn movement_history_is_bounded_and_expiry_retains_the_original_mapping() {
    let (controller,state,_) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    let mut gate=unsafe {binding.claim()}.unwrap();
    for _ in 0..260 {
        binding.submit(plan(7,&[0],&[])).unwrap();
        unsafe {gate.tick(0)}.unwrap();
    }
    let history=binding.report().unwrap();
    assert_eq!(history.event_count,260);
    assert_eq!(history.events.len(),256);
    assert_eq!(history.events[0].sequence,5);
    state.lock().unwrap().ready=false;
    binding.submit(plan(0,&[0],&[])).unwrap();
    unsafe {gate.tick(1)}.unwrap();
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(101)}.unwrap();
    let history=binding.report().unwrap();
    assert_eq!(history.snapshot.statistics.cancellations,1);
    assert_eq!(history.snapshot.statistics.committed_copy_bytes,0);
    assert_eq!(history.snapshot.statistics.staged_bytes,128);
    assert_eq!(history.snapshot.generation,0);
    assert!(history.events.last().unwrap().outcome.contains("Expired"));
    unsafe {binding.close()}.unwrap();
}

#[test]
fn observed_target_cancels_a_copy_even_before_its_estimated_time_deadline() {
    let (controller,state,_) = setup(4);
    let binding=execution::ExecutionBinding::new(controller);
    let mut gate=unsafe {binding.claim()}.unwrap();
    binding.record_demand(1);
    assert_eq!(binding.submit_for_demand(plan(0,&[0],&[]),1),Err(Error::Expired));
    state.lock().unwrap().ready=false;
    binding.submit_for_demand(plan(0,&[0],&[]),3).unwrap();
    unsafe {gate.tick(1)}.unwrap();
    binding.record_demand(3);
    unsafe {gate.tick(2)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().generation,0);
    state.lock().unwrap().ready=true;
    unsafe {gate.tick(3)}.unwrap();
    let report=binding.report().unwrap();
    assert_eq!(report.snapshot.statistics.cancellations,1);
    assert_eq!(report.snapshot.generation,0);
    assert_eq!(report.events[0].outcome,"Cancelled");
    assert_eq!(report.events[0].target_demand,Some(3));
    binding.submit_for_demand(plan(0,&[0],&[]),4).unwrap();
    binding.record_demand(4);
    unsafe {gate.tick(4)}.unwrap();
    assert_eq!(binding.snapshot().unwrap().statistics.staged_bytes,128);
    assert!(binding.report().unwrap().events[1].outcome.contains("Expired"));
    unsafe {binding.close()}.unwrap();
}

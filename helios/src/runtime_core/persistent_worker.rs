/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Generic persistent-worker CUDA Green Context partition and lifecycle controller.

use std::ffi::c_uint;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::driver::{
    CU_DEV_RESOURCE_TYPE_SM, CU_GREEN_CTX_DEFAULT_STREAM, CU_STREAM_NON_BLOCKING, DevResource,
    Driver, Handle,
};
use super::format::invalid;
use crate::{Error, ExecutionError, Result};

/// Partitioned CUDA Green Contexts and non-blocking streams for a core with a persistent worker.
pub(super) struct GreenContextPartition {
    pub(super) worker_green_ctx: Handle,
    pub(super) compute_green_ctx: Handle,
    pub(super) worker_context: Handle,
    pub(super) compute_context: Handle,
    pub(super) worker_stream: Handle,
    pub(super) compute_stream: Handle,
    pub(super) worker_sm_count: u32,
    pub(super) compute_sm_count: u32,
}

// SAFETY: CUDA green contexts and streams may be transferred across threads while owned by
// `NativeResources`, which serializes creation and teardown.
unsafe impl Send for GreenContextPartition {}

impl GreenContextPartition {
    /// Split the device's SMs into a worker partition of at least `requested_worker_sms` and a
    /// non-empty compute remainder, then create green contexts and non-blocking streams for both.
    pub(super) unsafe fn create(
        driver: &Driver,
        device: i32,
        requested_worker_sms: u32,
    ) -> Result<Self> {
        let gc = driver
            .green_ctx
            .as_ref()
            .ok_or_else(|| invalid("persistent worker requires CUDA driver Green Context APIs"))?;
        let mut partition = Self {
            worker_green_ctx: null_mut(),
            compute_green_ctx: null_mut(),
            worker_context: null_mut(),
            compute_context: null_mut(),
            worker_stream: null_mut(),
            compute_stream: null_mut(),
            worker_sm_count: 0,
            compute_sm_count: 0,
        };
        let setup = (|| -> Result<()> {
            // SAFETY: Output structs and pointers satisfy cuda.h contracts; any handles created
            // before an error are released by `cleanup` below.
            unsafe {
                let mut device_sm = DevResource::zeroed();
                driver.check(
                    "cuDeviceGetDevResource(SM)",
                    (gc.device_get_dev_resource)(device, &mut device_sm, CU_DEV_RESOURCE_TYPE_SM),
                )?;
                let total_sms = device_sm
                    .sm()
                    .map(|sm| sm.sm_count)
                    .filter(|&count| count > 0)
                    .ok_or_else(|| invalid("device reported zero streaming multiprocessors"))?;
                if requested_worker_sms == 0 || requested_worker_sms >= total_sms {
                    return Err(invalid(format!(
                        "persistent worker requested {requested_worker_sms} SMs out of {total_sms}, leaving no compute SMs"
                    )));
                }
                let mut worker_res = DevResource::zeroed();
                let mut compute_res = DevResource::zeroed();
                let mut nb_groups: c_uint = 1;
                driver.check(
                    "cuDevSmResourceSplitByCount",
                    (gc.dev_sm_resource_split_by_count)(
                        &mut worker_res,
                        &mut nb_groups,
                        &device_sm,
                        &mut compute_res,
                        0,
                        requested_worker_sms,
                    ),
                )?;
                let worker_split = worker_res.sm().map(|sm| sm.sm_count).unwrap_or(0);
                let compute_split = compute_res.sm().map(|sm| sm.sm_count).unwrap_or(0);
                if nb_groups != 1 || worker_split == 0 || compute_split == 0 {
                    return Err(invalid(format!(
                        "SM split for persistent worker ({requested_worker_sms} requested out of {total_sms}) left no compute SMs"
                    )));
                }
                let mut worker_desc = null_mut();
                driver.check(
                    "cuDevResourceGenerateDesc(worker)",
                    (gc.dev_resource_generate_desc)(&mut worker_desc, &mut worker_res, 1),
                )?;
                let mut compute_desc = null_mut();
                driver.check(
                    "cuDevResourceGenerateDesc(compute)",
                    (gc.dev_resource_generate_desc)(&mut compute_desc, &mut compute_res, 1),
                )?;
                driver.check(
                    "cuGreenCtxCreate(worker)",
                    (gc.green_ctx_create)(
                        &mut partition.worker_green_ctx,
                        worker_desc,
                        device,
                        CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxCreate(compute)",
                    (gc.green_ctx_create)(
                        &mut partition.compute_green_ctx,
                        compute_desc,
                        device,
                        CU_GREEN_CTX_DEFAULT_STREAM,
                    ),
                )?;
                let mut actual_worker = DevResource::zeroed();
                driver.check(
                    "cuGreenCtxGetDevResource(worker)",
                    (gc.green_ctx_get_dev_resource)(
                        partition.worker_green_ctx,
                        &mut actual_worker,
                        CU_DEV_RESOURCE_TYPE_SM,
                    ),
                )?;
                let mut actual_compute = DevResource::zeroed();
                driver.check(
                    "cuGreenCtxGetDevResource(compute)",
                    (gc.green_ctx_get_dev_resource)(
                        partition.compute_green_ctx,
                        &mut actual_compute,
                        CU_DEV_RESOURCE_TYPE_SM,
                    ),
                )?;
                partition.worker_sm_count = actual_worker.sm().map(|sm| sm.sm_count).unwrap_or(0);
                partition.compute_sm_count = actual_compute.sm().map(|sm| sm.sm_count).unwrap_or(0);
                if partition.worker_sm_count == 0 || partition.compute_sm_count == 0 {
                    return Err(invalid("green context partition has zero SMs"));
                }
                driver.check(
                    "cuCtxFromGreenCtx(worker)",
                    (gc.ctx_from_green_ctx)(
                        &mut partition.worker_context,
                        partition.worker_green_ctx,
                    ),
                )?;
                driver.check(
                    "cuCtxFromGreenCtx(compute)",
                    (gc.ctx_from_green_ctx)(
                        &mut partition.compute_context,
                        partition.compute_green_ctx,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxStreamCreate(worker)",
                    (gc.green_ctx_stream_create)(
                        &mut partition.worker_stream,
                        partition.worker_green_ctx,
                        CU_STREAM_NON_BLOCKING,
                        0,
                    ),
                )?;
                driver.check(
                    "cuGreenCtxStreamCreate(compute)",
                    (gc.green_ctx_stream_create)(
                        &mut partition.compute_stream,
                        partition.compute_green_ctx,
                        CU_STREAM_NON_BLOCKING,
                        0,
                    ),
                )?;
                Ok(())
            }
        })();
        if let Err(error) = setup {
            // SAFETY: No work was launched on these newly created streams or contexts.
            unsafe {
                partition.destroy_streams(driver);
                partition.destroy_green_contexts(driver);
            }
            return Err(error);
        }
        Ok(partition)
    }

    /// Destroy the worker and compute streams after all work on them has drained.
    pub(super) unsafe fn destroy_streams(&mut self, driver: &Driver) {
        unsafe {
            if !self.worker_stream.is_null() {
                (driver.stream_destroy)(self.worker_stream);
                self.worker_stream = null_mut();
            }
            if !self.compute_stream.is_null() {
                (driver.stream_destroy)(self.compute_stream);
                self.compute_stream = null_mut();
            }
        }
    }

    /// Destroy both green contexts after streams, graphs, modules, and allocations are released.
    pub(super) unsafe fn destroy_green_contexts(&mut self, driver: &Driver) {
        self.worker_context = null_mut();
        self.compute_context = null_mut();
        let Some(gc) = driver.green_ctx.as_ref() else {
            self.worker_green_ctx = null_mut();
            self.compute_green_ctx = null_mut();
            return;
        };
        unsafe {
            if !self.worker_green_ctx.is_null() {
                (gc.green_ctx_destroy)(self.worker_green_ctx);
                self.worker_green_ctx = null_mut();
            }
            if !self.compute_green_ctx.is_null() {
                (gc.green_ctx_destroy)(self.compute_green_ctx);
                self.compute_green_ctx = null_mut();
            }
        }
    }
}

/// Lifecycle state of a persistent worker graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker resources are bound and ready; the worker graph has not been launched.
    Idle,
    /// Worker graph was launched once on `worker_stream` and is active.
    Running,
    /// Worker either never launched or observed `stop_flag == 1` and drained cleanly.
    Stopped,
    /// `worker_stream` failed to drain; GPU resources must remain retained until process exit.
    FailedDrain,
}

/// Deterministic state machine governing start, stop, atomic flag updates, and metrics reads.
struct PersistentWorkerMachine {
    state: WorkerState,
    stop_flag_address: usize,
    metrics_address: usize,
    metrics_count: usize,
}

impl PersistentWorkerMachine {
    fn new(stop_flag_address: usize, metrics_address: usize, metrics_count: usize) -> Self {
        Self {
            state: WorkerState::Idle,
            stop_flag_address,
            metrics_address,
            metrics_count,
        }
    }

    fn state(&self) -> WorkerState {
        self.state
    }

    /// Zero the mapped-host stop flag and invoke `launch` once.
    ///
    /// Calling `start` while already `Running` is an idempotent no-op.
    /// Calling `start` after `Stopped` or `FailedDrain` fails explicitly.
    ///
    /// # Safety
    /// `stop_flag_address` must point to live, 4-byte-aligned, host-writable memory.
    unsafe fn start(&mut self, launch: impl FnOnce() -> Result<()>) -> Result<()> {
        match self.state {
            WorkerState::Idle => {
                // SAFETY: Validated mapped-host address aligned to 4 bytes and retained through drain.
                unsafe {
                    AtomicU32::from_ptr(self.stop_flag_address as *mut u32)
                        .store(0, Ordering::Release);
                }
                launch()?;
                self.state = WorkerState::Running;
                Ok(())
            }
            WorkerState::Running => Ok(()),
            WorkerState::Stopped => Err(invalid("persistent worker has already been stopped")),
            WorkerState::FailedDrain => Err(Error::execution(
                ExecutionError::Drain,
                "persistent worker failed to drain",
            )),
        }
    }

    /// Store `1u32` (Release) to the mapped-host stop flag and invoke `sync` to drain the worker.
    ///
    /// Idempotent when `Idle` or `Stopped`. Transitions to `FailedDrain` if `sync` fails.
    ///
    /// # Safety
    /// `stop_flag_address` must point to live, 4-byte-aligned, host-writable memory.
    unsafe fn stop(&mut self, sync: impl FnOnce() -> Result<()>) -> Result<()> {
        match self.state {
            WorkerState::Idle => {
                self.state = WorkerState::Stopped;
                Ok(())
            }
            WorkerState::Running => {
                // SAFETY: Validated mapped-host address aligned to 4 bytes and retained through drain.
                unsafe {
                    AtomicU32::from_ptr(self.stop_flag_address as *mut u32)
                        .store(1, Ordering::Release);
                }
                match sync() {
                    Ok(()) => {
                        self.state = WorkerState::Stopped;
                        Ok(())
                    }
                    Err(error) => {
                        self.state = WorkerState::FailedDrain;
                        Err(error)
                    }
                }
            }
            WorkerState::Stopped => Ok(()),
            WorkerState::FailedDrain => Err(Error::execution(
                ExecutionError::Drain,
                "persistent worker failed to drain",
            )),
        }
    }

    /// Read the optional `u64` counter array from mapped-host memory without a D2H copy.
    ///
    /// # Safety
    /// When `metrics_count > 0`, `metrics_address` must point to `metrics_count` live,
    /// 8-byte-aligned host-readable `u64` words.
    unsafe fn metrics(&self) -> Option<Vec<u64>> {
        if self.metrics_count == 0 || self.metrics_address == 0 {
            return None;
        }
        let mut values = Vec::with_capacity(self.metrics_count);
        for index in 0..self.metrics_count {
            let ptr = (self.metrics_address + index * std::mem::size_of::<u64>()) as *mut u64;
            // SAFETY: Validated mapped-host range aligned to 8 bytes and retained through drain.
            values.push(unsafe { AtomicU64::from_ptr(ptr).load(Ordering::Acquire) });
        }
        Some(values)
    }
}

/// Thread-safe runtime controller shared between `NativeResources` and `ExecutionManifest`.
pub(super) struct PersistentWorkerRuntime {
    driver: Arc<Driver>,
    context: Handle,
    worker_stream: Handle,
    worker_graph: Handle,
    machine: Mutex<PersistentWorkerMachine>,
}

// SAFETY: All state transitions and CUDA driver calls are serialized by `self.machine`,
// and every driver operation saves and restores the calling thread's CUDA context.
unsafe impl Send for PersistentWorkerRuntime {}
unsafe impl Sync for PersistentWorkerRuntime {}

impl PersistentWorkerRuntime {
    pub(super) fn new(
        driver: Arc<Driver>,
        context: Handle,
        worker_stream: Handle,
        worker_graph: Handle,
        stop_flag_address: usize,
        metrics_address: usize,
        metrics_count: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            driver,
            context,
            worker_stream,
            worker_graph,
            machine: Mutex::new(PersistentWorkerMachine::new(
                stop_flag_address,
                metrics_address,
                metrics_count,
            )),
        })
    }

    fn with_context(&self, operation: impl FnOnce() -> Result<()>) -> Result<()> {
        // SAFETY: `self.context` remains live while the worker is active or draining,
        // and the previous context is restored before returning.
        unsafe {
            let mut previous = null_mut();
            let had_previous = (self.driver.context_get)(&mut previous) == 0;
            self.driver
                .check("cuCtxSetCurrent", (self.driver.context_set)(self.context))?;
            let result = operation();
            if had_previous && previous != self.context {
                let _ = (self.driver.context_set)(previous);
            }
            result
        }
    }

    pub(super) fn start(&self) -> Result<()> {
        let mut machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: Mapped-host stop flag remains owned by `NativeResources` through teardown.
        unsafe {
            machine.start(|| {
                self.with_context(|| {
                    self.driver.check(
                        "cuGraphLaunch(persistent_worker)",
                        (self.driver.graph_launch)(self.worker_graph, self.worker_stream),
                    )
                })
            })
        }
    }

    pub(super) fn stop(&self) -> Result<()> {
        let mut machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: Mapped-host stop flag remains owned by `NativeResources` through teardown.
        unsafe {
            machine.stop(|| {
                self.with_context(|| {
                    self.driver.check(
                        "cuStreamSynchronize(persistent_worker)",
                        (self.driver.stream_sync)(self.worker_stream),
                    )
                })
            })
        }
    }

    pub(super) fn state(&self) -> WorkerState {
        let machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        machine.state()
    }

    pub(super) fn metrics(&self) -> Option<Vec<u64>> {
        let machine = self
            .machine
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // SAFETY: Mapped-host metrics range remains owned by `NativeResources` through teardown.
        unsafe { machine.metrics() }
    }

    pub(super) fn control(self: &Arc<Self>) -> crate::execution::PersistentWorkerControl {
        let start_handle = Arc::clone(self);
        let stop_handle = Arc::clone(self);
        let metrics_handle = Arc::clone(self);
        crate::execution::PersistentWorkerControl::new(
            Arc::new(move || start_handle.start()),
            Arc::new(move || stop_handle.stop()),
            Arc::new(move || metrics_handle.metrics()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_start_is_idempotent_and_stop_signals_flag_once() {
        let flag = AtomicU32::new(7);
        let metrics = [AtomicU64::new(11), AtomicU64::new(29)];
        let mut machine = PersistentWorkerMachine::new(
            (&flag as *const AtomicU32) as usize,
            metrics.as_ptr() as usize,
            metrics.len(),
        );
        assert_eq!(machine.state(), WorkerState::Idle);
        let mut launches = 0usize;
        let mut syncs = 0usize;
        unsafe {
            machine
                .start(|| {
                    assert_eq!(flag.load(Ordering::Acquire), 0);
                    launches += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(machine.state(), WorkerState::Running);
            // Second start is idempotent and does not launch again.
            machine
                .start(|| {
                    launches += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(launches, 1);
            assert_eq!(machine.metrics().unwrap(), vec![11, 29]);

            // Stop stores 1u32 before syncing and transitions to Stopped.
            machine
                .stop(|| {
                    assert_eq!(flag.load(Ordering::Acquire), 1);
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(machine.state(), WorkerState::Stopped);
            assert_eq!(syncs, 1);

            // Double stop is idempotent and does not sync again.
            machine
                .stop(|| {
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(syncs, 1);

            // Restart after stop is rejected.
            assert!(machine.start(|| Ok(())).is_err());
        }
    }

    #[test]
    fn stop_before_start_transitions_without_cuda_sync() {
        let flag = AtomicU32::new(0);
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize, 0, 0);
        let mut syncs = 0usize;
        unsafe {
            machine
                .stop(|| {
                    syncs += 1;
                    Ok(())
                })
                .unwrap();
            assert_eq!(syncs, 0);
            assert_eq!(machine.state(), WorkerState::Stopped);
            assert!(machine.metrics().is_none());
        }
    }

    #[test]
    fn launch_failure_leaves_worker_idle_and_drain_failure_is_sticky() {
        let flag = AtomicU32::new(9);
        let mut machine = PersistentWorkerMachine::new((&flag as *const AtomicU32) as usize, 0, 0);
        unsafe {
            assert!(
                machine
                    .start(|| Err(invalid("injected launch error")))
                    .is_err()
            );
            assert_eq!(machine.state(), WorkerState::Idle);
            assert_eq!(flag.load(Ordering::Acquire), 0);

            machine.start(|| Ok(())).unwrap();
            assert_eq!(machine.state(), WorkerState::Running);

            assert!(
                machine
                    .stop(|| Err(Error::execution(ExecutionError::Drain, "stuck stream")))
                    .is_err()
            );
            assert_eq!(machine.state(), WorkerState::FailedDrain);
            assert_eq!(flag.load(Ordering::Acquire), 1);

            // Subsequent stop and start both surface FailedDrain without re-invoking callbacks.
            assert!(machine.stop(|| Ok(())).is_err());
            assert!(machine.start(|| Ok(())).is_err());
        }
    }

    #[test]
    #[ignore = "requires an assigned CUDA_VISIBLE_DEVICES GPU supporting Green Contexts"]
    fn cuda_green_context_persistent_worker_runs_concurrently_with_cooperative_compute_and_drains_cleanly()
     {
        use super::super::NativeResources;
        use super::super::archive::PayloadFile;
        use super::super::driver::CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM;
        use super::super::format::{
            Allocation, Context, Core, Function, Graph, Memory, Module, Reservation,
        };
        use super::super::payload::{Files, RecordReader};
        use super::super::protobuf::wire;
        use prost::Message;
        use std::collections::HashMap;
        use std::time::{Duration, Instant};

        struct TestFiles(HashMap<String, Vec<u8>>);

        impl Files for TestFiles {
            fn span<'a>(&self, _core: &'a Core, _key: &str) -> Result<&'a wire::Span> {
                Err(invalid("unused test span"))
            }
            fn source(&self, _span: &wire::Span) -> Result<(&PayloadFile, u64)> {
                Err(invalid("unused test source"))
            }
            fn read(&self, _core: &Core, key: &str) -> Result<Vec<u8>> {
                self.0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test resource {key}")))
            }
            fn records<'a>(&'a self, _core: &Core, key: &str) -> Result<RecordReader<'a>> {
                let bytes = self
                    .0
                    .get(key)
                    .cloned()
                    .ok_or_else(|| invalid(format!("missing test record resource {key}")))?;
                Ok(RecordReader::Slice { bytes, position: 0 })
            }
            fn path(&self) -> &std::path::Path {
                std::path::Path::new("")
            }
            fn source_index(&self, _span: &wire::Span) -> Result<usize> {
                Err(invalid("unused test source_index"))
            }
            fn asset_path(&self, _span: &wire::Span) -> Result<std::path::PathBuf> {
                Err(invalid("unused test asset_path"))
            }
            fn release_page_cache(&self) {}
            fn retained_files(&self) -> Result<Vec<std::sync::Arc<std::fs::File>>> {
                Ok(Vec::new())
            }
            fn direct_files(
                &self,
            ) -> Box<dyn Iterator<Item = Result<std::sync::Arc<std::fs::File>>> + '_> {
                Box::new(std::iter::empty())
            }
            fn validate(&self, _core: &Core) -> Result<()> {
                Ok(())
            }
            fn resource_reader<'a>(
                &'a self,
                _core: &Core,
                _key: &str,
            ) -> Result<Box<dyn std::io::Read + 'a>> {
                Err(invalid("unused test resource_reader"))
            }
        }

        let driver = Driver::load().unwrap();
        let (major, minor, total_sms, host_numa_vmm, max_smem_per_sm, max_smem_optin) = unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            let mut device = 0;
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            let attr = |code: i32| {
                let mut val = 0;
                driver
                    .check(
                        "cuDeviceGetAttribute",
                        (driver.device_attribute)(&mut val, code, device),
                    )
                    .unwrap();
                val
            };
            (
                attr(75),
                attr(76),
                attr(16),
                attr(CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM),
                attr(81),
                attr(97),
            )
        };
        let device_va = 0x800_0000_0000u64;
        let host_va = 0xc00_0000_0000u64;
        assert!(
            driver.green_ctx.is_some(),
            "GPU driver must support Green Contexts"
        );
        assert!(total_sms >= 16, "test requires >= 16 SMs, got {total_sms}");
        let one_block_smem = ((max_smem_per_sm as u32) / 2 + 4096).min(max_smem_optin as u32);

        let cu_source = r#"
extern "C" __global__ void worker_spin(const unsigned int* stop_flag, unsigned long long* metrics) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long iters = 0;
        while (true) {
            unsigned int stop = 0;
            asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(stop) : "l"(stop_flag) : "memory");
            if (stop != 0u) {
                break;
            }
            ++iters;
            asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics), "l"(iters) : "memory");
            asm volatile("nanosleep.u32 500;");
        }
        asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(metrics + 1), "l"(1ULL) : "memory");
    }
}

extern "C" __global__ void coop_all_sms(int* state, int step) {
    extern __shared__ unsigned int dyn_smem[];
    if (threadIdx.x == 0) {
        const unsigned int n = gridDim.x;
        const unsigned int b = blockIdx.x;
        dyn_smem[0] = b;
        state[4 + b] = step * 1000 + static_cast<int>(dyn_smem[0]);
        __threadfence();
        unsigned int prev = 0;
        asm volatile("atom.add.acq_rel.gpu.u32 %0, [%1], 1;" : "=r"(prev) : "l"(state) : "memory");
        (void)prev;
        while (true) {
            unsigned int count = 0;
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(count) : "l"(state) : "memory");
            if (count >= n) {
                break;
            }
            asm volatile("nanosleep.u32 50;");
        }
        const int neighbor = state[4 + (n - 1u - b)];
        state[256 + b] = neighbor + 7;
        if (b == 0u) {
            state[2] = static_cast<int>(n);
        }
    }
}
"#;
        let unique = format!(
            "flare-worker-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        )
        .replace(|c: char| !c.is_ascii_alphanumeric() && c != '-', "_");
        let cu_path = std::env::temp_dir().join(format!("{unique}.cu"));
        let cubin_path = std::env::temp_dir().join(format!("{unique}.cubin"));
        std::fs::write(&cu_path, cu_source).unwrap();

        let nvcc = if std::path::Path::new("/usr/local/cuda-13.2/bin/nvcc").exists() {
            "/usr/local/cuda-13.2/bin/nvcc"
        } else {
            "nvcc"
        };
        let arch = if (major, minor) == (10, 3) {
            format!("-arch=sm_{major}{minor}a")
        } else {
            format!("-arch=sm_{major}{minor}")
        };
        let compiled = std::process::Command::new(nvcc)
            .args([
                "-cubin",
                "-O3",
                &arch,
                cu_path.to_str().unwrap(),
                "-o",
                cubin_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to execute nvcc for CUDA test kernel");
        assert!(
            compiled.status.success(),
            "nvcc failed ({}): {}",
            compiled.status,
            String::from_utf8_lossy(&compiled.stderr)
        );
        let cubin_bytes = std::fs::read(&cubin_path).unwrap();
        let _ = std::fs::remove_file(&cu_path);
        let _ = std::fs::remove_file(&cubin_path);

        let encode_record = |msg_bytes: Vec<u8>, out: &mut Vec<u8>| {
            out.extend_from_slice(&(msg_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(&msg_bytes);
        };
        let ptr_blob = |allocation: &str, offset: u64| wire::Blob {
            data: vec![0u8; 8],
            relocations: vec![wire::Relocation {
                offset: 0,
                target: Some(wire::Reference {
                    allocation: allocation.into(),
                    offset,
                }),
                kind: wire::RelocationKind::Pointer as i32,
            }],
        };

        let mut worker_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 0,
                    grid: vec![1, 1, 1],
                    block: vec![32, 1, 1],
                    shared: 0,
                    arguments: vec![ptr_blob("worker_ctrl", 0), ptr_blob("worker_ctrl", 64)],
                    attributes: vec![],
                })),
            }
            .encode_to_vec(),
            &mut worker_records,
        );

        let step_val: i32 = 17;
        let mut coop_records = Vec::new();
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Memset(wire::Memset {
                    destination: Some(wire::Reference {
                        allocation: "compute_state".into(),
                        offset: 0,
                    }),
                    pitch: 4096,
                    value: 0,
                    element_size: 4,
                    width: 1024,
                    height: 1,
                })),
            }
            .encode_to_vec(),
            &mut coop_records,
        );
        encode_record(
            wire::Node {
                operation: Some(wire::node::Operation::Kernel(wire::Kernel {
                    function: 1,
                    // Intentionally request full-device SMs so prepare_graphs proves clamping
                    // down to the complementary compute green context's SM count.
                    grid: vec![total_sms as u32, 1, 1],
                    block: vec![256, 1, 1],
                    shared: one_block_smem,
                    arguments: vec![
                        ptr_blob("compute_state", 0),
                        wire::Blob {
                            data: step_val.to_le_bytes().to_vec(),
                            relocations: vec![],
                        },
                    ],
                    attributes: vec![wire::Attribute {
                        id: 2,
                        value: Some(wire::Blob {
                            data: {
                                let mut bytes = vec![0u8; 64];
                                bytes[0] = 1;
                                bytes
                            },
                            relocations: vec![],
                        }),
                    }],
                })),
            }
            .encode_to_vec(),
            &mut coop_records,
        );
        encode_record(
            wire::Edge {
                source: 0,
                destination: 1,
                data: vec![0; 8],
            }
            .encode_to_vec(),
            &mut coop_records,
        );

        let mut blobs = HashMap::new();
        blobs.insert("worker_test.cubin".to_string(), cubin_bytes);
        blobs.insert("graph_worker".to_string(), worker_records);
        blobs.insert("graph_coop".to_string(), coop_records);
        let files = TestFiles(blobs);

        let build_core = || Core {
            resources: HashMap::new(),
            architecture: [major, minor],
            allocations: vec![
                Allocation {
                    name: "worker_ctrl".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(host_va),
                    memory: Memory::MappedHost,
                    sizing: None,
                },
                Allocation {
                    name: "compute_state".into(),
                    nbytes: 2 << 20,
                    alignment: 2 << 20,
                    file: None,
                    data_spans: vec![],
                    sources: vec![],
                    relocations: vec![],
                    address: Some(device_va),
                    memory: Memory::Device,
                    sizing: None,
                },
            ],
            modules: vec![Module {
                name: "worker_mod".into(),
                file: "worker_test.cubin".into(),
                globals: vec![],
            }],
            functions: vec![
                Function {
                    id: 0,
                    module: "worker_mod".into(),
                    symbol: "worker_spin".into(),
                    parameter_sizes: vec![8, 8],
                    attributes: vec![],
                },
                Function {
                    id: 1,
                    module: "worker_mod".into(),
                    symbol: "coop_all_sms".into(),
                    parameter_sizes: vec![8, 4],
                    attributes: vec![],
                },
            ],
            graphs: vec![
                Graph {
                    name: "worker_spin".into(),
                    flags: 0,
                    resource: "graph_worker".into(),
                    node_count: 1,
                    edge_count: 0,
                    functions: vec![0],
                },
                Graph {
                    name: "coop_step".into(),
                    flags: 0,
                    resource: "graph_coop".into(),
                    node_count: 2,
                    edge_count: 1,
                    functions: vec![1],
                },
            ],
            reservations: vec![Reservation {
                address: device_va,
                nbytes: 64 << 20,
            }],
            execution: Some(wire::Execution {
                abi_version: 5,
                persistent_worker: Some(wire::PersistentWorker {
                    worker_sm_count: 8,
                    graph: "worker_spin".into(),
                    stop_flag_buffer: "worker_ctrl".into(),
                    stop_flag_offset: 0,
                    metrics_buffer: "worker_ctrl".into(),
                    metrics_offset: 64,
                    metrics_count: 2,
                }),
                ..Default::default()
            }),
            context: Context::default(),
            events: vec![],
        };

        for round in 0..2 {
            let mut owner =
                unsafe { NativeResources::load_prepared(&files, build_core(), 0).unwrap() };
            assert!(
                owner.allocations["worker_ctrl"].host_vmm || host_numa_vmm == 0,
                "expected HOST_NUMA VMM backing when CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM={host_numa_vmm}"
            );
            let partition = owner.green_partition.as_ref().unwrap();
            let worker_sms = partition.worker_sm_count;
            let compute_sms = partition.compute_sm_count;
            assert!(worker_sms >= 8, "worker SM count {worker_sms}");
            assert_eq!(
                compute_sms,
                total_sms as u32 - worker_sms,
                "partition SMs must sum to total {total_sms}"
            );
            eprintln!(
                "round={round} host_vmm={} total_sms={total_sms} worker_sms={worker_sms} compute_sms={compute_sms}",
                owner.allocations["worker_ctrl"].host_vmm
            );

            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Idle));
            owner.start_persistent_worker().unwrap();
            owner.start_persistent_worker().unwrap();
            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Running));

            let deadline = Instant::now() + Duration::from_secs(2);
            let metrics_before = loop {
                let metrics = owner.persistent_worker_metrics().unwrap();
                if metrics[0] > 0 {
                    break metrics[0];
                }
                assert!(
                    Instant::now() < deadline,
                    "persistent worker did not increment mapped-host counter in time"
                );
                std::thread::yield_now();
            };

            assert!(
                owner.download("compute_state").is_err(),
                "primary context download must refuse while persistent worker is running"
            );

            let coop_idx = owner
                .names
                .iter()
                .position(|name| name == "coop_step")
                .unwrap();
            let coop_graph = owner.executables[coop_idx] as usize;
            let compute_addr = owner.allocation("compute_state").unwrap().address;
            // Retain Stream and PinnedHost until after stop_persistent_worker(): cuMemFreeHost
            // synchronizes device work and must run only after the persistent worker drains,
            // matching GraphExecutor::close().
            let stream = crate::cuda::Stream::open(owner.execution_context() as usize, 0).unwrap();
            let mut host_buf = stream.host_buffer(4096).unwrap();
            for iter in 0..20 {
                host_buf.bytes_mut().fill(0);
                stream.launch(coop_graph).unwrap();
                unsafe {
                    stream.download(&host_buf, compute_addr, 4096).unwrap();
                }
                stream.synchronize().unwrap();
                let words: Vec<i32> = host_buf
                    .bytes()
                    .chunks_exact(4)
                    .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();
                assert_eq!(
                    words[2], compute_sms as i32,
                    "round={round} iter={iter}: expected clamped cooperative grid {compute_sms}, got {}",
                    words[2]
                );
                for b in 0..compute_sms as usize {
                    let expected_mirror =
                        step_val * 1000 + (compute_sms as usize - 1 - b) as i32 + 7;
                    assert_eq!(
                        words[256 + b],
                        expected_mirror,
                        "round={round} iter={iter} cta={b}: barrier/mirror mismatch"
                    );
                }
            }

            let metrics_after_compute = owner.persistent_worker_metrics().unwrap();
            assert!(
                metrics_after_compute[0] > metrics_before,
                "worker counter did not advance during concurrent cooperative launches ({} <= {metrics_before})",
                metrics_after_compute[0]
            );

            owner.stop_persistent_worker().unwrap();
            owner.stop_persistent_worker().unwrap();
            assert_eq!(owner.persistent_worker_state(), Some(WorkerState::Stopped));
            let final_metrics = owner.persistent_worker_metrics().unwrap();
            assert_eq!(final_metrics[1], 1, "worker exit sentinel not set");

            drop(host_buf);
            drop(stream);

            let downloaded = owner.download("compute_state").unwrap();
            let observed_grid = i32::from_le_bytes(downloaded[8..12].try_into().unwrap());
            assert_eq!(observed_grid, compute_sms as i32);

            let _ = &mut owner;
            drop(owner);
        }
    }
}

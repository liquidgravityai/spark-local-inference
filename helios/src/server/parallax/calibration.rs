//! Startup-only measurement of synchronous copies between pinned host memory and HBM.
//! These measurements do not estimate direct execution from coherent host memory.
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Sample {
    pub bytes: u64,
    pub elapsed_ns: u64,
    pub host_to_device: bool,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct CopyCost {
    pub bandwidth_bytes_per_ns: f64,
    pub latency_ns: u64,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Calibration {
    pub version: u32,
    pub memory_path: String,
    pub gpu_uuid: String,
    pub elapsed_ns: u64,
    pub host_to_device: Option<CopyCost>,
    pub device_to_host: Option<CopyCost>,
    pub samples: Vec<Sample>,
    pub failure: Option<String>,
}
impl Calibration {
    pub fn usable(&self) -> bool {
        self.failure.is_none() && self.host_to_device.is_some() && self.device_to_host.is_some()
    }
}
fn fit(samples: &[Sample], upload: bool) -> Option<CopyCost> {
    let mut sizes = std::collections::BTreeMap::<u64, Vec<u64>>::new();
    for s in samples.iter().filter(|s| s.host_to_device == upload) {
        if s.bytes == 0 || s.elapsed_ns == 0 { return None; }
        sizes.entry(s.bytes).or_default().push(s.elapsed_ns);
    }
    if sizes.len() < 3 { return None; }
    let points: Vec<_> = sizes.into_iter().map(|(size, mut times)| {
        times.sort_unstable();
        (size as f64, times[times.len() / 2] as f64)
    }).collect();
    let n = points.len() as f64;
    let x = points.iter().map(|p| p.0).sum::<f64>() / n;
    let y = points.iter().map(|p| p.1).sum::<f64>() / n;
    let slope = points.iter().map(|p| (p.0-x)*(p.1-y)).sum::<f64>()
        / points.iter().map(|p| (p.0-x).powi(2)).sum::<f64>();
    if !slope.is_finite() || slope <= 0. { return None; }
    // Cover every size's median rather than predicting a cheaper copy than measured.
    let latency = points.iter().map(|p| p.1 - slope*p.0).fold(0., f64::max).ceil();
    let bandwidth = 1. / slope;
    if !bandwidth.is_finite() || bandwidth < 0.000001 { return None; }
    Some(CopyCost { bandwidth_bytes_per_ns: bandwidth, latency_ns: latency as u64 })
}

/// The caller owns a live, idle CUDA context and retains it for the entire call.
/// No model buffer or captured graph is modified. A driver call already in flight
/// must finish before its scratch allocation can be freed; the sampling budget
/// is not a watchdog for a hung CUDA driver.
#[cfg(feature = "runtime-package")]
pub unsafe fn measure(context: usize, device: i32) -> Calibration {
    use crate::runtime_package::driver::{Driver, Handle};
    use std::time::{Duration, Instant};
    const BYTES: usize = 16 * 1024 * 1024;
    let started = Instant::now();
    let mut result = Calibration {
        version: 1, memory_path: "cuda_pinned_host_synchronous_copy; idle; includes API/synchronization overhead; not coherent cold execution".into(),
        gpu_uuid: String::new(), elapsed_ns: 0, host_to_device: None, device_to_host: None,
        samples: vec![], failure: None,
    };
    let run = (|| -> Result<(), String> {
        let driver = Driver::load().map_err(|e| e.to_string())?;
        struct Scratch<'a> { driver: &'a Driver, host: Handle, gpu: u64, previous: Handle }
        impl Drop for Scratch<'_> {
            fn drop(&mut self) {
                // Copies use synchronous driver entrypoints; no transfer borrows these buffers.
                unsafe {
                    if self.gpu != 0 { (self.driver.free)(self.gpu); }
                    if !self.host.is_null() { (self.driver.host_free)(self.host); }
                    (self.driver.context_set)(self.previous);
                }
            }
        }
        let mut scratch = Scratch { driver: &driver, host: std::ptr::null_mut(), gpu: 0, previous: std::ptr::null_mut() };
        driver.check("cuCtxGetCurrent", unsafe { (driver.context_get)(&mut scratch.previous) }).map_err(|e|e.to_string())?;
        driver.check("cuCtxSetCurrent", unsafe { (driver.context_set)(context as Handle) }).map_err(|e|e.to_string())?;
        let (_, actual) = crate::cuda::current_context().map_err(|e|e.to_string())?;
        if actual != device { return Err("calibration context/device mismatch".into()); }
        let uuid = crate::cuda::device_uuid(device).map_err(|e|e.to_string())?;
        result.gpu_uuid = uuid.iter().map(|b|format!("{b:02x}")).collect();
        driver.check("cuMemHostAlloc", unsafe { (driver.host_allocate)(&mut scratch.host, BYTES, 0) }).map_err(|e|e.to_string())?;
        driver.check("cuMemAlloc", unsafe { (driver.allocate)(&mut scratch.gpu, BYTES) }).map_err(|e|e.to_string())?;
        unsafe { std::ptr::write_bytes(scratch.host.cast::<u8>(), 0x5a, BYTES); }
        let copy = |bytes, upload| -> Result<(), String> {
            if upload {
                driver.check("cuMemcpyHtoD", unsafe { (driver.upload)(scratch.gpu, scratch.host, bytes) })
            } else {
                driver.check("cuMemcpyDtoH", unsafe { (driver.download)(scratch.host, scratch.gpu, bytes) })
            }.map_err(|e|e.to_string())
        };
        copy(BYTES, true)?;
        unsafe { std::ptr::write_bytes(scratch.host.cast::<u8>(), 0, BYTES); }
        copy(BYTES, false)?;
        // At most 18 measured copies. Stop issuing work after one second,
        // leaving time for the final synchronous copy and allocation cleanup.
        'sampling: for _ in 0..3 {
            for bytes in [64 * 1024, 1024 * 1024, BYTES] {
                for upload in [true, false] {
                    if started.elapsed() >= Duration::from_secs(1) { break 'sampling; }
                    let at = Instant::now();
                    copy(bytes, upload)?;
                    result.samples.push(Sample { bytes: bytes as u64, elapsed_ns: at.elapsed().as_nanos() as u64, host_to_device: upload });
                }
            }
        }
        let bytes = unsafe { std::slice::from_raw_parts(scratch.host.cast::<u8>(), BYTES) };
        if bytes.iter().any(|b| *b != 0x5a) { return Err("calibration copy verification failed".into()); }
        Ok(())
    })();
    result.elapsed_ns = started.elapsed().as_nanos() as u64;
    result.failure = run.err();
    if result.failure.is_none() {
        result.host_to_device = fit(&result.samples, true);
        result.device_to_host = fit(&result.samples, false);
        if result.host_to_device.is_none() || result.device_to_host.is_none() || result.elapsed_ns > 3_000_000_000 {
            result.failure = Some("insufficient stable transfer samples within the startup budget".into());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fits_directional_costs_and_rejects_insufficient_or_flat_samples() {
        let mut samples = vec![];
        for bytes in [1024, 4096, 16384] {
            samples.push(Sample { bytes, elapsed_ns: 100 + bytes / 2, host_to_device: true });
            samples.push(Sample { bytes, elapsed_ns: 200 + bytes / 4, host_to_device: false });
        }
        let up = fit(&samples, true).unwrap();
        let down = fit(&samples, false).unwrap();
        assert_eq!(up.bandwidth_bytes_per_ns, 2.);
        assert_eq!(up.latency_ns, 100);
        assert_eq!(down.bandwidth_bytes_per_ns, 4.);
        assert_eq!(down.latency_ns, 200);
        assert!(fit(&samples[..2], true).is_none());
        for sample in &mut samples { sample.elapsed_ns = 10; }
        assert!(fit(&samples, true).is_none());
    }
}

#[cfg(all(test, feature = "runtime-package"))]
#[test]
#[ignore = "requires an explicitly reserved GPU via CUDA_VISIBLE_DEVICES"]
fn gpu_transfer_calibration_finishes_within_startup_budget() {
    use crate::runtime_package::driver::Driver;
    let driver = Driver::load().unwrap();
    unsafe {
        driver.check("cuInit", (driver.init)(0)).unwrap();
        let mut device = 0;
        driver.check("cuDeviceGet", (driver.device)(&mut device, 0)).unwrap();
        let mut context = std::ptr::null_mut();
        driver.check("cuCtxCreate", (driver.context_create)(&mut context, 0, device)).unwrap();
        let result = measure(context as usize, device);
        driver.check("cuCtxDestroy", (driver.context_destroy)(context)).unwrap();
        println!("{}", serde_json::to_string(&result).unwrap());
        assert!(result.usable(), "{:?}", result.failure);
        assert!(result.elapsed_ns < 3_000_000_000);
        assert!(result.samples.len() <= 18);
    }
}

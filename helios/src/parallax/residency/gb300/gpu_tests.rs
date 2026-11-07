//! Storage qualification only: replay one instantiated graph across tier changes.
use super::*;
use crate::runtime_package::driver::{CopyEndpoint, CopyParams};
use std::{
    ptr,
    time::{Duration, Instant},
};

struct Harness {
    driver: Driver,
    context: Handle,
    graph: Handle,
    executable: Handle,
    output: u64,
}
impl Harness {
    fn new() -> Self {
        let driver = Driver::load().unwrap();
        let mut device = -1;
        let mut context = ptr::null_mut();
        unsafe {
            driver.check("cuInit", (driver.init)(0)).unwrap();
            driver
                .check("cuDeviceGet", (driver.device)(&mut device, 0))
                .unwrap();
            driver
                .check(
                    "cuCtxCreate",
                    (driver.context_create)(&mut context, 0, device),
                )
                .unwrap();
        }
        Self {
            driver,
            context,
            graph: ptr::null_mut(),
            executable: ptr::null_mut(),
            output: 0,
        }
    }
}
impl Drop for Harness {
    fn drop(&mut self) {
        unsafe {
            if (self.driver.context_set)(self.context) != 0 || (self.driver.context_sync)() != 0 {
                // Let process teardown handle a CUDA context that cannot drain.
                return;
            }
            if !self.executable.is_null() {
                (self.driver.executable_destroy)(self.executable);
            }
            if !self.graph.is_null() {
                (self.driver.graph_destroy)(self.graph);
            }
            if self.output != 0 {
                (self.driver.free)(self.output);
            }
            (self.driver.context_destroy)(self.context);
        }
    }
}
fn expert(index: usize) -> ExpertId {
    ExpertId {
        layer: 3,
        expert: index,
    }
}
fn plan(generation: u64, promote: &[usize], evict: &[usize]) -> Plan {
    Plan {
        version: 1,
        catalog_id: "synthetic-native".into(),
        generation,
        valid_until_ns: u64::MAX,
        promote: promote.iter().copied().map(expert).collect(),
        evict: evict.iter().copied().map(expert).collect(),
    }
}
fn settle(c: &mut Controller<Gb300>) -> Progress {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        // This test owns the entire context; no concurrent submitter exists.
        let result = unsafe { c.progress(0) }.unwrap();
        if result != Progress::WaitingForTransfer {
            return result;
        }
        assert!(
            Instant::now() < deadline,
            "staging did not complete within five seconds"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "requires an explicitly assigned idle GB300 and FLARE_RUN_CUDA_TESTS=1"]
fn stable_graph_replay_across_grace_hbm_exchanges() {
    assert_eq!(std::env::var("FLARE_RUN_CUDA_TESTS").as_deref(), Ok("1"));
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").expect("select the assigned GPU UUID");
    assert!(
        visible.starts_with("GPU-") && !visible.contains(','),
        "select exactly one GPU UUID"
    );
    let mut h = Harness::new();
    let d = &h.driver;
    let mut device = -1;
    let mut node = -1;
    unsafe {
        d.check("cuDeviceGet", (d.device)(&mut device, 0)).unwrap();
        d.check(
            "cuDeviceGetAttribute",
            (d.device_attribute)(&mut node, 134, device),
        )
        .unwrap();
    }
    let mut alignment = 1;
    for p in [properties(1, device), properties(3, node)] {
        let mut bytes = 0;
        d.check("cuMemGetAllocationGranularity", unsafe {
            (d.granularity)(&mut bytes, &p, 0)
        })
        .unwrap();
        alignment = alignment.max(bytes);
    }
    let payload = 4093;
    assert!(alignment >= payload && alignment <= 16 * 1024 * 1024);
    let regions: Vec<_> = (0..4)
        .map(|i| Region {
            expert: expert(i / 2),
            part: i % 2,
            offset: i * alignment,
            bytes: payload,
        })
        .collect();
    let backend = unsafe {
        Gb300::new(
            h.context as usize,
            device,
            regions,
            Limits {
                hbm_slots: 4,
                hbm_bytes: 4 * alignment,
                backing_bytes: 4 * alignment,
                virtual_base: None,
            },
            |r, bytes| {
                for (i, byte) in bytes.iter_mut().enumerate() {
                    *byte = (i + 31 * r.expert.expert + 17 * r.part) as u8;
                }
                Ok(())
            },
        )
    }
    .unwrap();
    assert!(!backend.capabilities().cold_moe_execution_qualified);
    let mut c = Controller::new("synthetic-native".into(), backend).unwrap();
    let addresses: Vec<_> = (0..4)
        .map(|i| c.address(expert(i / 2), i % 2).unwrap())
        .collect();
    unsafe {
        d.check("cuMemAlloc", (d.allocate)(&mut h.output, payload * 4))
            .unwrap();
        d.check("cuGraphCreate", (d.graph_create)(&mut h.graph, 0))
            .unwrap();
        for (i, &address) in addresses.iter().enumerate() {
            let params = CopyParams {
                source: CopyEndpoint {
                    memory_type: 2,
                    device: address,
                    pitch: payload,
                    height: 1,
                    ..Default::default()
                },
                destination: CopyEndpoint {
                    memory_type: 2,
                    device: h.output + (i * payload) as u64,
                    pitch: payload,
                    height: 1,
                    ..Default::default()
                },
                width: payload,
                height: 1,
                depth: 1,
            };
            let mut node = ptr::null_mut();
            d.check(
                "cuGraphAddMemcpyNode",
                (d.graph_copy)(&mut node, h.graph, ptr::null(), 0, &params, h.context),
            )
            .unwrap();
        }
        d.check(
            "cuGraphInstantiate",
            (d.graph_instantiate)(&mut h.executable, h.graph, 0),
        )
        .unwrap();
    }
    let stream = cuda::Stream::open(h.context as usize, device).unwrap();
    let verify = |c: &mut Controller<Gb300>| {
        let reader = c.acquire_reader().unwrap();
        stream.launch(h.executable as usize).unwrap();
        stream.synchronize().unwrap();
        unsafe {
            reader.complete();
        }
        let mut result = vec![0; payload * 4];
        d.check("cuMemcpyDtoH", unsafe {
            (d.download)(result.as_mut_ptr().cast(), h.output, result.len())
        })
        .unwrap();
        for (i, &address) in addresses.iter().enumerate() {
            assert_eq!(c.address(expert(i / 2), i % 2).unwrap(), address);
            for j in 0..payload {
                assert_eq!(
                    result[i * payload + j],
                    (j + 31 * (i / 2) + 17 * (i % 2)) as u8
                );
            }
        }
    };
    verify(&mut c);
    let reader = c.acquire_reader().unwrap();
    c.begin(plan(0, &[0], &[]), 0).unwrap();
    assert_eq!(settle(&mut c), Progress::WaitingForReaders);
    unsafe {
        reader.complete();
    }
    assert_eq!(settle(&mut c), Progress::Committed(1));
    verify(&mut c);
    for generation in 1..=20 {
        let promote = generation as usize % 2;
        c.begin(plan(generation, &[promote], &[1 - promote]), 0)
            .unwrap();
        assert_eq!(settle(&mut c), Progress::Committed(generation + 1));
        verify(&mut c);
    }
    c.begin(plan(21, &[], &[0]), 0).unwrap();
    assert_eq!(settle(&mut c), Progress::Committed(22));
    verify(&mut c);
    assert_eq!(c.snapshot().free_slots, 4);
    assert_eq!(
        c.snapshot().statistics.staged_bytes,
        (42 * alignment) as u64
    );
    eprintln!(
        "verified Grace/HBM bytes and unchanged graph across 22 commits; allocation granularity {alignment} bytes"
    );
    drop(stream);
    drop(c);
}

#[test]
#[ignore = "requires an explicitly assigned idle GB300 and FLARE_RUN_CUDA_TESTS=1"]
fn unequal_weight_planes_use_matching_physical_slots() {
    assert_eq!(std::env::var("FLARE_RUN_CUDA_TESTS").as_deref(), Ok("1"));
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap();
    assert!(visible.starts_with("GPU-") && !visible.contains(','));
    let h = Harness::new();
    let mut device = -1;
    h.driver
        .check("cuDeviceGet", unsafe { (h.driver.device)(&mut device, 0) })
        .unwrap();
    let mib = 1024 * 1024;
    let sizes = [8 * mib, 4 * mib];
    let regions = vec![
        Region {
            expert: expert(0),
            part: 0,
            offset: 0,
            bytes: sizes[0],
        },
        Region {
            expert: expert(0),
            part: 1,
            offset: sizes[0],
            bytes: sizes[1],
        },
    ];
    let backend = unsafe {
        Gb300::new(
            h.context as usize,
            device,
            regions,
            Limits {
                hbm_slots: 4,
                hbm_bytes: 24 * mib,
                backing_bytes: 12 * mib,
                virtual_base: None,
            },
            |r, bytes| {
                bytes.fill((r.part + 71) as u8);
                Ok(())
            },
        )
    }
    .unwrap();
    let mut c = Controller::new("synthetic-native".into(), backend).unwrap();
    assert_eq!(c.snapshot().hbm_allocated_bytes, 24 * mib);
    for generation in 0..4 {
        let movement = if generation % 2 == 0 {
            plan(generation, &[0], &[])
        } else {
            plan(generation, &[], &[0])
        };
        c.begin(movement, 0).unwrap();
        assert_eq!(settle(&mut c), Progress::Committed(generation + 1));
        for (part, bytes) in sizes.into_iter().enumerate() {
            let mut readback = vec![0u8; bytes];
            h.driver
                .check("cuMemcpyDtoH", unsafe {
                    (h.driver.download)(
                        readback.as_mut_ptr().cast(),
                        c.address(expert(0), part).unwrap(),
                        bytes,
                    )
                })
                .unwrap();
            assert!(readback.iter().all(|byte| *byte == (part + 71) as u8));
        }
    }
}

#[test]
#[ignore = "requires an explicitly assigned idle GB300 and FLARE_RUN_CUDA_TESTS=1"]
fn package_loading_borrows_reservations_and_seals_initialized_weights() {
    assert_eq!(std::env::var("FLARE_RUN_CUDA_TESTS").as_deref(), Ok("1"));
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").unwrap();
    assert!(visible.starts_with("GPU-") && !visible.contains(','));
    let h = Harness::new();
    let d = &h.driver;
    let bytes = 2 << 20;
    let mut base = 0;
    d.check("cuMemAddressReserve", unsafe {
        (d.reserve)(&mut base, bytes * 2, bytes, 0, 0)
    })
    .unwrap();
    let regions = (0..2)
        .map(|i| Region {
            expert: expert(i),
            part: 0,
            offset: i * bytes,
            bytes,
        })
        .collect();
    let loading = unsafe {
        Gb300::prepare_package(
            h.context as usize,
            0,
            regions,
            Limits {
                hbm_slots: 2,
                hbm_bytes: bytes * 2,
                backing_bytes: bytes * 2,
                virtual_base: Some(base),
            },
        )
    }
    .unwrap();
    let payload: Vec<u8> = (0..bytes * 2)
        .map(|i| (i.wrapping_mul(31) + i / bytes) as u8)
        .collect();
    // The ordinary package loader writes through the canonical GPU address.
    d.check("cuMemcpyHtoD", unsafe {
        (d.upload)(base, payload.as_ptr().cast(), payload.len())
    })
    .unwrap();
    let backend = unsafe { loading.finish() }.unwrap();
    let mut controller = Controller::new("synthetic-native".into(), backend).unwrap();
    for (generation, promote, evict) in [
        (0, vec![0], vec![]),
        (1, vec![1], vec![0]),
        (2, vec![], vec![1]),
    ] {
        controller
            .begin(plan(generation, &promote, &evict), 0)
            .unwrap();
        assert!(matches!(settle(&mut controller), Progress::Committed(_)));
        let mut actual = vec![0u8; payload.len()];
        d.check("cuMemcpyDtoH", unsafe {
            (d.download)(actual.as_mut_ptr().cast(), base, actual.len())
        })
        .unwrap();
        assert_eq!(actual, payload);
    }
    drop(controller);
    // Closing storage must leave the package's virtual address reservation owned.
    let mut handle = 0;
    d.check("cuMemCreate", unsafe {
        (d.physical_create)(&mut handle, bytes * 2, &properties(1, 0), 0)
    })
    .unwrap();
    d.check("cuMemMap", unsafe {
        (d.map)(base, bytes * 2, 0, handle, 0)
    })
    .unwrap();
    d.check("cuMemUnmap", unsafe { (d.unmap)(base, bytes * 2) })
        .unwrap();
    d.check("cuMemRelease", unsafe { (d.physical_release)(handle) })
        .unwrap();
    d.check("cuMemAddressFree", unsafe {
        (d.release_address)(base, bytes * 2)
    })
    .unwrap();
}

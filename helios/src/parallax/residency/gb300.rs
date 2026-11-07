//! GB300 storage prototype: immutable Grace backing, spare HBM slots and stable VAs.
//! Publication is quiescent. No model package or MoE kernel is admitted by this module.
use super::*;
use crate::{
    cuda,
    runtime_package::driver::{
        Access, AllocationProperties, Driver, Handle, Location as MemoryLocation,
    },
};
use std::{
    collections::{BTreeMap, BTreeSet},
    mem::ManuallyDrop,
    sync::Arc,
};

#[derive(Clone, Debug)]
pub struct Limits {
    pub hbm_slots: usize,
    pub hbm_bytes: usize,
    pub backing_bytes: usize,
    pub virtual_base: Option<u64>,
}
struct Resources {
    driver: Arc<Driver>,
    context: usize,
    device: i32,
    stream: ManuallyDrop<cuda::Stream>,
    done: ManuallyDrop<cuda::Event>,
    copy_started: ManuallyDrop<cuda::Event>,
    reservations: Vec<(u64, usize)>,
    handles: Vec<u64>,
    mapped: BTreeMap<u64, usize>,
    poisoned: bool,
}
impl Resources {
    fn bind(&self) -> Result<()> {
        self.driver
            .check("cuCtxSetCurrent", unsafe {
                (self.driver.context_set)(self.context as Handle)
            })
            .map_err(|e| Error::Backend(e.to_string()))
    }
    fn reserve(&mut self, bytes: usize, alignment: usize, wanted: Option<u64>) -> Result<u64> {
        let mut address = 0;
        self.driver
            .check("cuMemAddressReserve", unsafe {
                (self.driver.reserve)(&mut address, bytes, alignment, wanted.unwrap_or(0), 0)
            })
            .map_err(|e| Error::Backend(e.to_string()))?;
        self.reservations.push((address, bytes));
        if wanted.is_some_and(|base| base != address) {
            return Err(Error::Invalid(
                "prepared virtual address could not be reserved exactly".into(),
            ));
        }
        Ok(address)
    }
    fn allocate(&mut self, bytes: usize, properties: &AllocationProperties) -> Result<u64> {
        let mut handle = 0;
        self.driver
            .check("cuMemCreate", unsafe {
                (self.driver.physical_create)(&mut handle, bytes, properties, 0)
            })
            .map_err(|e| Error::Backend(e.to_string()))?;
        self.handles.push(handle);
        Ok(handle)
    }
    fn map(
        &mut self,
        address: u64,
        bytes: usize,
        handle: u64,
        host_node: Option<i32>,
        writable: bool,
    ) -> Result<()> {
        self.driver
            .check("cuMemMap", unsafe {
                (self.driver.map)(address, bytes, 0, handle, 0)
            })
            .map_err(|e| Error::Backend(e.to_string()))?;
        self.mapped.insert(address, bytes);
        let mut access = vec![Access {
            location: MemoryLocation {
                kind: 1,
                device: self.device,
            },
            flags: if writable { 3 } else { 1 },
        }];
        if let Some(node) = host_node {
            access.push(Access {
                location: MemoryLocation {
                    kind: 3,
                    device: node,
                },
                flags: 3,
            });
        }
        self.driver
            .check("cuMemSetAccess", unsafe {
                (self.driver.access)(address, bytes, access.as_ptr(), access.len())
            })
            .map_err(|e| Error::Backend(e.to_string()))
    }
    fn unmap(&mut self, address: u64) -> Result<()> {
        if let Some(&bytes) = self.mapped.get(&address) {
            self.driver
                .check("cuMemUnmap", unsafe { (self.driver.unmap)(address, bytes) })
                .map_err(|e| Error::Backend(e.to_string()))?;
            self.mapped.remove(&address);
        }
        Ok(())
    }
}
impl Drop for Resources {
    fn drop(&mut self) {
        // Unknown asynchronous work must retain its allocations and driver owner.
        if self.poisoned || self.bind().is_err() || self.stream.synchronize().is_err() {
            std::mem::forget(self.driver.clone());
            return;
        }
        unsafe {
            ManuallyDrop::drop(&mut self.done);
            ManuallyDrop::drop(&mut self.copy_started);
            ManuallyDrop::drop(&mut self.stream);
            for (&address, &bytes) in &self.mapped {
                (self.driver.unmap)(address, bytes);
            }
            for &handle in &self.handles {
                (self.driver.physical_release)(handle);
            }
            for &(address, bytes) in self.reservations.iter().rev() {
                (self.driver.release_address)(address, bytes);
            }
        }
    }
}

/// Writable canonical mappings retained only while authenticated weights load.
/// The package owns their address reservations and must outlive this owner.
pub struct Loading(Gb300);
impl Loading {
    pub fn hbm_bytes(&self) -> usize {
        self.0.hbm_sizes.iter().sum()
    }
    pub fn backing_bytes(&self) -> usize {
        self.0.backing_bytes
    }

    /// # Safety
    /// Every weight initializer has finished and no writer will use these mappings again.
    pub unsafe fn finish(mut self) -> Result<Gb300> {
        let resources = &mut self.0.resources;
        resources.bind()?;
        if let Err(error) = resources.driver.check("cuCtxSynchronize", unsafe {
            (resources.driver.context_sync)()
        }) {
            resources.poisoned = true;
            return Err(Error::Poisoned(error.to_string()));
        }
        let access = Access {
            location: MemoryLocation {
                kind: 1,
                device: resources.device,
            },
            flags: 1,
        };
        for (region, &bytes) in self.0.regions.iter().zip(&self.0.rounded) {
            resources
                .driver
                .check("cuMemSetAccess", unsafe {
                    (resources.driver.access)(self.0.base + region.offset as u64, bytes, &access, 1)
                })
                .map_err(|e| Error::Backend(e.to_string()))?;
        }
        Ok(self.0)
    }
}

pub struct Gb300 {
    resources: Resources,
    regions: Vec<Region>,
    rounded: Vec<usize>,
    host_handles: Vec<u64>,
    host_addresses: Vec<u64>,
    hbm_handles: Vec<u64>,
    hbm_addresses: Vec<u64>,
    hbm_sizes: Vec<usize>,
    locations: Vec<Location>,
    base: u64,
    slot_bytes: usize,
    backing_bytes: usize,
    active: bool,
    staged: BTreeMap<usize, usize>,
}
// SAFETY: Ownership serializes every operation; each operation binds its CUDA
// context. The context owner must drain and close storage before destruction.
unsafe impl Send for Gb300 {}

fn properties(kind: i32, id: i32) -> AllocationProperties {
    AllocationProperties {
        kind: 1,
        handles: 0,
        location: MemoryLocation { kind, device: id },
        windows: std::ptr::null_mut(),
        flags: [0; 8],
    }
}
struct Layout {
    rounded: Vec<usize>,
    total: usize,
    slot_bytes: usize,
    backing_bytes: usize,
    hbm_bytes: usize,
    hbm_sizes: Vec<usize>,
}
fn layout(regions: &[Region], alignment: usize, limits: &Limits) -> Result<Layout> {
    if regions.is_empty() || limits.hbm_slots == 0 || !alignment.is_power_of_two() {
        return Err(Error::Invalid(
            "invalid mapping geometry or empty staging pool".into(),
        ));
    }
    let mut rounded = Vec::with_capacity(regions.len());
    let mut intervals = vec![];
    let mut backing_bytes = 0usize;
    let mut ids = BTreeSet::new();
    for r in regions {
        let bytes = r
            .bytes
            .checked_next_multiple_of(alignment)
            .filter(|n| *n > 0 && *n <= isize::MAX as usize)
            .ok_or_else(|| Error::Invalid("region size overflow".into()))?;
        let end = r
            .offset
            .checked_add(bytes)
            .ok_or_else(|| Error::Invalid("region offset overflow".into()))?;
        if r.offset % alignment != 0 || !ids.insert((r.expert, r.part)) {
            return Err(Error::Invalid(
                "expert regions require unique identities and page-aligned offsets".into(),
            ));
        }
        backing_bytes = backing_bytes.checked_add(bytes).ok_or(Error::Capacity)?;
        rounded.push(bytes);
        intervals.push((r.offset, end));
    }
    intervals.sort_unstable();
    if intervals.windows(2).any(|w| w[0].1 > w[1].0) {
        return Err(Error::Invalid("expert regions share mapped pages".into()));
    }
    let total = intervals.last().unwrap().1;
    let slot_bytes = *rounded.iter().max().unwrap();
    // Match physical allocation sizes to canonical mappings. Subrange aliases
    // of larger handles are rejected by the GB300 driver used for qualification.
    let classes: Vec<_> = rounded
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if limits
        .hbm_slots
        .checked_mul(classes[0])
        .is_none_or(|bytes| bytes > limits.hbm_bytes)
    {
        return Err(Error::Capacity);
    }
    let hbm_sizes: Vec<_> = (0..limits.hbm_slots)
        .map(|slot| classes[slot % classes.len()])
        .collect();
    let hbm_bytes = hbm_sizes
        .iter()
        .try_fold(0usize, |n, bytes| n.checked_add(*bytes))
        .ok_or(Error::Capacity)?;
    if hbm_bytes > limits.hbm_bytes || backing_bytes > limits.backing_bytes {
        return Err(Error::Capacity);
    }
    Ok(Layout {
        rounded,
        total,
        slot_bytes,
        backing_bytes,
        hbm_bytes,
        hbm_sizes,
    })
}
/// Exact byte charge for admitted package pages, before creating physical storage.
/// The constructor still verifies hardware page alignment before allocation.
pub(crate) fn package_footprint(
    regions: &[Region],
    limits: &Limits,
    initial: &[ExpertId],
) -> Result<(u64, u64)> {
    let planned = layout(regions, 1, limits)?;
    let mut free = BTreeMap::<usize, usize>::new();
    for bytes in planned.hbm_sizes {
        *free.entry(bytes).or_default() += 1;
    }
    let initial: BTreeSet<_> = initial.iter().copied().collect();
    for region in regions.iter().filter(|r| initial.contains(&r.expert)) {
        let count = free.get_mut(&region.bytes).ok_or(Error::Capacity)?;
        *count = count.checked_sub(1).ok_or(Error::Capacity)?;
    }
    Ok((planned.hbm_bytes as u64, planned.backing_bytes as u64))
}

impl Gb300 {
    /// # Safety
    /// `context` is live on `device` and outlives this backend, all graph readers
    /// and every transfer. A poisoned backend requires retaining that context
    /// until its owner can confirm drain; it must not resume inference.
    /// The loader writes only the supplied region and cannot retain mutable aliases.
    pub unsafe fn new(
        context: usize,
        device: i32,
        regions: Vec<Region>,
        limits: Limits,
        load: impl FnMut(&Region, &mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        unsafe { Self::build(context, device, regions, limits, false, load) }
    }

    /// Map package-owned, reserved addresses into Grace before loading payloads.
    ///
    /// # Safety
    /// The context and reservations outlive storage and all readers. Regions are
    /// unmapped, exclusively owned pages. Only the package loader may write them
    /// until `Loading::finish`; no graph may execute before that call succeeds.
    pub unsafe fn prepare_package(
        context: usize,
        device: i32,
        regions: Vec<Region>,
        limits: Limits,
    ) -> Result<Loading> {
        if limits.virtual_base.is_none() {
            return Err(Error::Invalid(
                "package storage requires reserved addresses".into(),
            ));
        }
        unsafe { Self::build(context, device, regions, limits, true, |_, _| Ok(())).map(Loading) }
    }

    unsafe fn build(
        context: usize,
        device: i32,
        regions: Vec<Region>,
        limits: Limits,
        borrowed: bool,
        mut load: impl FnMut(&Region, &mut [u8]) -> Result<()>,
    ) -> Result<Self> {
        if regions.is_empty() || limits.hbm_slots == 0 {
            return Err(Error::Invalid(
                "empty expert storage or staging pool".into(),
            ));
        }
        let driver = Arc::new(Driver::load().map_err(|e| Error::Backend(e.to_string()))?);
        let stream = cuda::Stream::open_lowest_priority(context, device)
            .map_err(|e| Error::Backend(e.to_string()))?;
        let done = stream.event_with_timing(true).map_err(|e| Error::Backend(e.to_string()))?;
        let copy_started = stream.event_with_timing(true).map_err(|e| Error::Backend(e.to_string()))?;
        let mut resources = Resources {
            driver,
            context,
            device,
            stream: ManuallyDrop::new(stream),
            done: ManuallyDrop::new(done),
            copy_started: ManuallyDrop::new(copy_started),
            reservations: vec![],
            handles: vec![],
            mapped: BTreeMap::new(),
            poisoned: false,
        };
        let attribute = |attribute| -> Result<i32> {
            let mut value = 0;
            resources
                .driver
                .check("cuDeviceGetAttribute", unsafe {
                    (resources.driver.device_attribute)(&mut value, attribute, device)
                })
                .map_err(|e| Error::Backend(e.to_string()))?;
            Ok(value)
        };
        if attribute(75)? != 10 || attribute(76)? != 3 {
            return Err(Error::Invalid("GB300 residency requires SM103".into()));
        }
        let node = attribute(134)?;
        if node < 0 {
            return Err(Error::Invalid(
                "GB300 host NUMA placement is unavailable".into(),
            ));
        }
        let host = properties(3, node);
        let gpu = properties(1, device);
        let mut alignment = 1;
        for props in [&host, &gpu] {
            let mut granularity = 0;
            resources
                .driver
                .check("cuMemGetAllocationGranularity", unsafe {
                    (resources.driver.granularity)(&mut granularity, props, 0)
                })
                .map_err(|e| Error::Backend(e.to_string()))?;
            if !granularity.is_power_of_two() {
                return Err(Error::Invalid(
                    "unsupported memory mapping granularity".into(),
                ));
            }
            alignment = alignment.max(granularity);
        }
        if borrowed && regions.iter().any(|r| r.bytes % alignment != 0) {
            return Err(Error::Invalid(
                "package expert regions must own full mapping pages".into(),
            ));
        }
        let Layout {
            rounded,
            total,
            slot_bytes,
            backing_bytes,
            hbm_bytes,
            hbm_sizes,
        } = layout(&regions, alignment, &limits)?;
        let base = if borrowed {
            limits.virtual_base.unwrap()
        } else {
            resources.reserve(total, alignment, limits.virtual_base)?
        };
        if base % alignment as u64 != 0 || base.checked_add(total as u64).is_none() {
            return Err(Error::Invalid("invalid canonical address extent".into()));
        }
        let backing_base = resources.reserve(total, alignment, None)?;
        let staging_base = resources.reserve(hbm_bytes, alignment, None)?;
        let mut host_handles = vec![];
        let mut host_addresses = vec![];
        for (r, &bytes) in regions.iter().zip(&rounded) {
            let handle = resources.allocate(bytes, &host)?;
            let address = backing_base + r.offset as u64;
            resources.map(address, bytes, handle, Some(node), true)?;
            // Private backing is initialized once; no mutable slice escapes startup.
            if !borrowed {
                let storage = unsafe { std::slice::from_raw_parts_mut(address as *mut u8, bytes) };
                storage.fill(0);
                load(r, &mut storage[..r.bytes])?;
            }
            resources.map(base + r.offset as u64, bytes, handle, None, borrowed)?;
            host_handles.push(handle);
            host_addresses.push(address);
        }
        let mut hbm_handles = vec![];
        let mut hbm_addresses = vec![];
        let mut address = staging_base;
        for &bytes in &hbm_sizes {
            let handle = resources.allocate(bytes, &gpu)?;
            resources.map(address, bytes, handle, None, true)?;
            hbm_handles.push(handle);
            hbm_addresses.push(address);
            address += bytes as u64;
        }
        let locations = vec![Location::Backing; regions.len()];
        Ok(Self {
            resources,
            regions,
            rounded,
            host_handles,
            host_addresses,
            hbm_handles,
            hbm_addresses,
            hbm_sizes,
            locations,
            base,
            slot_bytes,
            backing_bytes,
            active: false,
            staged: BTreeMap::new(),
        })
    }
    fn healthy(&self) -> Result<()> {
        if self.resources.poisoned {
            Err(Error::Poisoned(
                "GB300 storage has an unconfirmed mapping or transfer".into(),
            ))
        } else {
            Ok(())
        }
    }
    fn transfer_error(&mut self, error: impl std::fmt::Display) -> Error {
        if let Err(drain) = self.resources.stream.synchronize() {
            self.resources.poisoned = true;
            Error::Poisoned(format!("transfer failed ({error}); drain failed ({drain})"))
        } else {
            self.active = false;
            self.staged.clear();
            Error::Backend(error.to_string())
        }
    }
}

// SAFETY: Staging uses distinct preallocated slots, publication restores all
// attempted mappings on failure, and uncertain work retains its resource owner.
unsafe impl Backend for Gb300 {
    fn regions(&self) -> &[Region] {
        &self.regions
    }
    fn slots(&self) -> usize {
        self.hbm_handles.len()
    }
    fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
    fn slot_capacity(&self, slot: usize) -> usize {
        self.hbm_sizes[slot]
    }
    fn slot_accepts(&self, region: usize, slot: usize) -> bool {
        self.rounded[region] == self.hbm_sizes[slot]
    }
    fn backing_bytes(&self) -> usize {
        self.backing_bytes
    }
    fn transfer_bytes(&self, region: usize) -> usize {
        self.rounded[region]
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
        self.base + self.regions[region].offset as u64
    }
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()> {
        self.healthy()?;
        self.resources.bind()?;
        if self.active {
            return Err(Error::Busy);
        }
        let mut slots = BTreeSet::new();
        let mut regions = BTreeSet::new();
        for &(region, slot) in copies {
            if self.locations.get(region) != Some(&Location::Backing)
                || slot >= self.slots()
                || !self.slot_accepts(region, slot)
                || self.locations.contains(&Location::Hbm(slot))
                || !slots.insert(slot)
                || !regions.insert(region)
            {
                return Err(Error::Invalid(
                    "staging would overwrite live or duplicate storage".into(),
                ));
            }
        }
        if let Err(error) = self.resources.stream.record(&self.resources.copy_started) {
            return Err(self.transfer_error(error));
        }
        for &(region, slot) in copies {
            let mut offset = 0;
            while offset < self.rounded[region] {
                let bytes = (self.rounded[region] - offset).min(cuda::BATCH_BYTES);
                let copied = unsafe {
                    self.resources.stream.copy_batch(
                        &[self.hbm_addresses[slot] + offset as u64],
                        &[self.host_addresses[region] + offset as u64],
                        &[bytes],
                    )
                };
                if let Err(error) = copied {
                    return Err(self.transfer_error(error));
                }
                offset += bytes;
            }
        }
        if let Err(error) = self.resources.stream.record(&self.resources.done) {
            return Err(self.transfer_error(error));
        }
        self.staged = copies.iter().copied().collect();
        self.active = true;
        Ok(())
    }
    fn ready(&mut self) -> Result<bool> {
        self.healthy()?;
        self.resources.bind()?;
        if !self.active {
            return Ok(true);
        }
        match self.resources.done.completed() {
            Ok(v) => Ok(v),
            Err(e) => Err(self.transfer_error(e)),
        }
    }
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> {
        self.resources.done.elapsed_ns(&self.resources.copy_started)
            .map(Some).map_err(|e| Error::Backend(e.to_string()))
    }
    fn discard(&mut self) -> Result<()> {
        if !self.ready()? {
            return Err(Error::Busy);
        }
        self.staged.clear();
        self.active = false;
        Ok(())
    }
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()> {
        if !self.ready()? {
            return Err(Error::Busy);
        }
        let mut final_locations = self.locations.clone();
        let mut changed = BTreeSet::new();
        for b in bindings {
            if self.locations.get(b.region) != Some(&b.from) || !changed.insert(b.region) {
                self.resources.poisoned = true;
                return Err(Error::Poisoned(
                    "mapping state disagrees with the transaction".into(),
                ));
            }
            if let Location::Hbm(slot) = b.to {
                if self.staged.get(&b.region) != Some(&slot) {
                    return Err(Error::Invalid(
                        "promotion has no completed staged copy".into(),
                    ));
                }
            }
            final_locations[b.region] = b.to;
        }
        let mut slots = BTreeSet::new();
        if final_locations
            .iter()
            .any(|l| matches!(l,Location::Hbm(slot) if !slots.insert(*slot)))
        {
            return Err(Error::Invalid(
                "two experts cannot publish the same HBM slot".into(),
            ));
        }
        // A submission gate is still required: synchronization alone cannot stop
        // another host thread launching a reader during the following remaps.
        if let Err(e) = self.resources.driver.check("cuCtxSynchronize", unsafe {
            (self.resources.driver.context_sync)()
        }) {
            self.resources.poisoned = true;
            return Err(Error::Poisoned(e.to_string()));
        }
        let result = super::transaction::publish(bindings, |region, location| {
            let address = self.base + self.regions[region].offset as u64;
            self.resources.unmap(address)?;
            let handle = match location {
                Location::Backing => self.host_handles[region],
                Location::Hbm(slot) => self.hbm_handles[slot],
            };
            self.resources
                .map(address, self.rounded[region], handle, None, false)
        });
        match result {
            Ok(()) => {
                self.locations = final_locations;
                self.staged.clear();
                self.active = false;
                Ok(())
            }
            Err(Error::Poisoned(e)) => {
                self.resources.poisoned = true;
                Err(Error::Poisoned(e))
            }
            Err(e) => {
                self.staged.clear();
                self.active = false;
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod gpu_tests;

#[cfg(test)]
mod layout_tests {
    use super::*;

    #[test]
    fn package_budget_preflight_counts_size_classes_and_initial_placement() {
        let regions = vec![
            Region {
                expert: ExpertId {
                    layer: 3,
                    expert: 0,
                },
                part: 0,
                offset: 0,
                bytes: 128,
            },
            Region {
                expert: ExpertId {
                    layer: 3,
                    expert: 0,
                },
                part: 1,
                offset: 128,
                bytes: 64,
            },
        ];
        let mut limits = Limits {
            hbm_slots: 2,
            hbm_bytes: 256,
            backing_bytes: 192,
            virtual_base: Some(4096),
        };
        assert_eq!(
            package_footprint(&regions, &limits, &[regions[0].expert]).unwrap(),
            (192, 192)
        );
        limits.hbm_slots = 1;
        assert!(matches!(
            package_footprint(&regions, &limits, &[regions[0].expert]),
            Err(Error::Capacity)
        ));
        limits.hbm_slots = 2;
        limits.hbm_bytes = 191;
        assert!(matches!(
            package_footprint(&regions, &limits, &[]),
            Err(Error::Capacity)
        ));
    }

    #[test]
    fn mapping_admission_charges_padding_and_rejects_shared_pages() {
        let mut regions = vec![
            Region {
                expert: ExpertId {
                    layer: 3,
                    expert: 0,
                },
                part: 0,
                offset: 0,
                bytes: 65,
            },
            Region {
                expert: ExpertId {
                    layer: 3,
                    expert: 1,
                },
                part: 0,
                offset: 128,
                bytes: 5,
            },
        ];
        let mut limits = Limits {
            hbm_slots: 2,
            hbm_bytes: 256,
            backing_bytes: 192,
            virtual_base: None,
        };
        let admitted = layout(&regions, 64, &limits).unwrap();
        assert_eq!(admitted.rounded, vec![128, 64]);
        assert_eq!(
            (
                admitted.slot_bytes,
                admitted.backing_bytes,
                admitted.hbm_bytes
            ),
            (128, 192, 192)
        );
        assert_eq!(admitted.hbm_sizes, vec![64, 128]);
        limits.hbm_bytes = 191;
        assert!(matches!(
            layout(&regions, 64, &limits),
            Err(Error::Capacity)
        ));
        limits.hbm_bytes = 192;
        limits.backing_bytes = 191;
        assert!(matches!(
            layout(&regions, 64, &limits),
            Err(Error::Capacity)
        ));
        limits.backing_bytes = 192;
        for offset in [65, 64] {
            regions[1].offset = offset;
            assert!(matches!(
                layout(&regions, 64, &limits),
                Err(Error::Invalid(_))
            ));
        }
        regions[1].offset = 128;
        regions[1].bytes = usize::MAX;
        assert!(matches!(
            layout(&regions, 64, &limits),
            Err(Error::Invalid(_))
        ));
    }
}

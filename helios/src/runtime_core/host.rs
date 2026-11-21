/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-backed CUDA VMM preserves recorded mapped-host addresses on Windows and Linux
//! when the device supports `CU_MEM_LOCATION_TYPE_HOST_NUMA` virtual memory management.

use super::{Result, invalid};

fn ranges(
    allocations: impl Iterator<Item = (u64, usize)>,
    granularity: usize,
) -> Result<Vec<(u64, usize)>> {
    let granularity = granularity as u64;
    if !granularity.is_power_of_two() {
        return Err(invalid("invalid host allocation granularity"));
    }
    let mut ranges = allocations
        .map(|(address, bytes)| {
            let end = address
                .checked_add(bytes as u64)
                .and_then(|end| end.checked_add(granularity - 1))
                .ok_or_else(|| invalid("host allocation exceeds address space"))?;
            Ok((address & !(granularity - 1), end & !(granularity - 1)))
        })
        .collect::<Result<Vec<_>>>()?;
    ranges.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| start <= last.1) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    Ok(merged
        .into_iter()
        .map(|(start, end)| (start, (end - start) as usize))
        .collect())
}

fn range_bytes(ranges: &[(u64, usize)]) -> Result<u64> {
    ranges.iter().try_fold(0u64, |total, &(_, bytes)| {
        total
            .checked_add(bytes as u64)
            .ok_or_else(|| invalid("host VMM footprint overflows address space"))
    })
}

#[cfg(any(windows, target_os = "linux"))]
impl super::NativeResources {
    /// Allocate and map all `Memory::MappedHost` allocations through host-NUMA CUDA VMM
    /// when the device supports `CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM`.
    ///
    /// Returns `Ok(true)` when host-NUMA VMM backed the mapped-host allocations, or
    /// `Ok(false)` when no mapped-host allocations exist or when Linux falls back to
    /// `mmap` + `cuMemHostRegister_v2`.
    pub(super) unsafe fn allocate_host(&mut self, core: &super::Core) -> Result<bool> {
        use super::driver::{
            Access, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM,
        };
        use super::{AllocationProperties, Location, Memory};

        if !core
            .allocations
            .iter()
            .any(|a| matches!(a.memory, Memory::MappedHost))
        {
            return Ok(false);
        }
        let driver = &self.driver;
        unsafe {
            driver.check("cuCtxSetCurrent", (driver.context_set)(self.context))?;
            let mut supported = 0;
            driver.check(
                "cuDeviceGetAttribute(HOST_NUMA_VMM)",
                (driver.device_attribute)(
                    &mut supported,
                    CU_DEVICE_ATTRIBUTE_HOST_NUMA_VMM,
                    self.device,
                ),
            )?;
            if supported == 0 {
                #[cfg(windows)]
                return Err(invalid(
                    "mapped core storage requires host-backed CUDA VMM on Windows",
                ));
                #[cfg(target_os = "linux")]
                return Ok(false);
            }
            let mut numa = 0;
            driver.check(
                "cuDeviceGetAttribute(HOST_NUMA_ID)",
                (driver.device_attribute)(&mut numa, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, self.device),
            )?;
            let properties = AllocationProperties {
                kind: 1,
                handles: 0,
                location: Location {
                    kind: 3,
                    device: numa.max(0),
                },
                windows: std::ptr::null_mut(),
                flags: [0; 8],
            };
            let mut granularity = 0;
            driver.check(
                "cuMemGetAllocationGranularity(host)",
                (driver.granularity)(&mut granularity, &properties, 0),
            )?;
            let physical_ranges = ranges(
                core.allocations
                    .iter()
                    .filter(|a| matches!(a.memory, Memory::MappedHost))
                    .map(|a| (a.address.unwrap(), a.nbytes)),
                granularity,
            )?;
            let physical_bytes = range_bytes(&physical_ranges)?;
            #[cfg(target_os = "linux")]
            let va_granularity = granularity.max(64 << 20);
            #[cfg(not(target_os = "linux"))]
            let va_granularity = granularity;
            let va_ranges = ranges(physical_ranges.iter().copied(), va_granularity)?;
            #[cfg(target_os = "linux")]
            {
                let overlaps = va_ranges.iter().any(|&(va_start, va_bytes)| {
                    let va_end = va_start + va_bytes as u64;
                    core.reservations
                        .iter()
                        .any(|r| va_start < r.address + r.nbytes as u64 && r.address < va_end)
                        || core.allocations.iter().any(|a| {
                            !matches!(a.memory, Memory::MappedHost)
                                && a.address.is_some_and(|addr| {
                                    let end = addr.saturating_add(a.extent() as u64);
                                    va_start < end && addr < va_end
                                })
                        })
                });
                if overlaps {
                    return Ok(false);
                }
            }
            let available_host = if self.integrated {
                self.available_memory()? as u64
            } else {
                super::memory::available_host_memory()?
            };
            super::memory::check_host_memory_fit(physical_bytes, available_host)?;
            let access = [
                Access {
                    location: Location {
                        kind: 3,
                        device: numa.max(0),
                    },
                    flags: 3,
                },
                Access {
                    location: Location {
                        kind: 1,
                        device: self.device,
                    },
                    flags: 3,
                },
            ];
            let bits = super::address_space::virtual_address_bits(self.device);
            for &(va_address, va_bytes) in &va_ranges {
                super::address_space::check_fits(
                    "host reservation",
                    va_address,
                    va_bytes as u64,
                    bits,
                )
                .map_err(|error| invalid(format!("{error}; re-export the core for this GPU")))?;
                let mut pointer = 0;
                driver.check(
                    "cuMemAddressReserve(host)",
                    (driver.reserve)(&mut pointer, va_bytes, va_granularity, va_address, 0),
                )?;
                self.reservations.push((pointer, va_bytes));
                if pointer != va_address {
                    return Err(invalid(format!(
                        "host CUDA address collision at {va_address:#x}"
                    )));
                }
            }
            for (address, bytes) in physical_ranges {
                let mut physical = 0;
                driver.check(
                    "cuMemCreate(host)",
                    (driver.physical_create)(&mut physical, bytes, &properties, 0),
                )?;
                let status = (driver.map)(address, bytes, 0, physical, 0);
                (driver.physical_release)(physical);
                driver.check("cuMemMap(host)", status)?;
                self.host_mappings.push((address, bytes));
                driver.check(
                    "cuMemSetAccess(host)",
                    (driver.access)(address, bytes, access.as_ptr(), access.len()),
                )?;
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_host_pages_share_physical_backing() {
        let base = 0xc00_0000_0000;
        let result = ranges(
            [
                (base + 4096, 4096),
                (base + 8192, 4096),
                (base + (4 << 20), 4096),
            ]
            .into_iter(),
            2 << 20,
        )
        .unwrap();
        assert_eq!(result, [(base, 2 << 20), (base + (4 << 20), 2 << 20)]);
        assert_eq!(range_bytes(&result).unwrap(), 4 << 20);
    }

    #[test]
    fn host_range_rounding_rejects_overflow() {
        assert!(ranges([(u64::MAX - 4095, 4096)].into_iter(), 2 << 20).is_err());
        assert!(ranges([(4096, 4096)].into_iter(), 0).is_err());
        assert!(range_bytes(&[(0, usize::MAX), (0, 1)]).is_err());
    }

    #[test]
    fn va_chunk_rounding_merges_sparse_physical_pages_without_inflating_physical_bytes() {
        let base = 0xc00_0000_0000;
        let physical_ranges = ranges(
            [(base + 4096, 4096), (base + (4 << 20), 4096)].into_iter(),
            2 << 20,
        )
        .unwrap();
        assert_eq!(
            physical_ranges,
            [(base, 2 << 20), (base + (4 << 20), 2 << 20)]
        );
        assert_eq!(range_bytes(&physical_ranges).unwrap(), 4 << 20);
        let va_ranges = ranges(physical_ranges.iter().copied(), 64 << 20).unwrap();
        assert_eq!(va_ranges, [(base, 64 << 20)]);
    }
}

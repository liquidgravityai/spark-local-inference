//! Expert storage, package binding and quiescent residency transactions.
//! The executor must retain reader leases through confirmed GPU completion.

mod controller;
pub mod execution;
#[cfg(feature = "runtime-package")]
pub mod gb300;
#[cfg(feature = "runtime-package")]
pub mod package;
mod transaction;
use super::ExpertId;
pub use controller::{Controller, Progress, ReaderLease, Snapshot, Statistics};

pub type Result<T> = std::result::Result<T, Error>;
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Invalid(String),
    Busy,
    Stale,
    Expired,
    Capacity,
    Backend(String),
    Poisoned(String),
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for Error {}

/// A page-isolated part of one expert's immutable representation. Offsets preserve
/// prepared tensor layout; the backend must reject overlapping rounded mappings.
#[derive(Clone, Debug)]
pub struct Region {
    pub expert: ExpertId,
    pub part: usize,
    pub offset: usize,
    pub bytes: usize,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Backing,
    Hbm(usize),
}
#[derive(Clone, Copy, Debug)]
pub struct Binding {
    pub region: usize,
    pub from: Location,
    pub to: Location,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub version: u32,
    /// Package and representation identity bound by the storage owner.
    pub catalog_id: String,
    pub generation: u64,
    /// Deadline in the owner's monotonic clock domain.
    pub valid_until_ns: u64,
    pub promote: Vec<ExpertId>,
    pub evict: Vec<ExpertId>,
}
#[derive(Clone, Copy, Debug)]
pub struct Capabilities {
    pub stable_addresses: bool,
    pub asynchronous_staging: bool,
    pub publication_requires_quiescence: bool,
    /// Byte transfers and graph-copy tests do not qualify a model's MoE kernels.
    pub cold_moe_execution_qualified: bool,
}

/// # Safety
/// Implementations keep all addresses valid while readers or copies can use them.
/// `stage` touches only free staging slots. A recoverable publication error means
/// every old binding is restored; uncertain rollback/drain must return Poisoned.
/// A recoverable staging error has drained all submitted copies. `discard` may
/// succeed only after all copies finish, and is idempotent after publication.
/// New backends begin with every region in immutable backing storage.
/// No device allocation is allowed after construction.
pub unsafe trait Backend {
    fn regions(&self) -> &[Region];
    fn slots(&self) -> usize;
    fn slot_bytes(&self) -> usize;
    fn slot_capacity(&self, _slot: usize) -> usize {
        self.slot_bytes()
    }
    fn slot_accepts(&self, region: usize, slot: usize) -> bool {
        self.transfer_bytes(region) <= self.slot_capacity(slot)
    }
    fn backing_bytes(&self) -> usize;
    fn transfer_bytes(&self, region: usize) -> usize {
        self.regions()[region].bytes
    }
    fn capabilities(&self) -> Capabilities;
    fn address(&self, region: usize) -> u64;
    fn stage(&mut self, copies: &[(usize, usize)]) -> Result<()>;
    fn ready(&mut self) -> Result<bool>;
    /// GPU duration of the completed staging operation, when measured by the backend.
    fn copy_elapsed_ns(&self) -> Result<Option<u64>> { Ok(None) }
    fn discard(&mut self) -> Result<()>;
    /// The caller has closed submission and drained every reader of this context.
    unsafe fn publish(&mut self, bindings: &[Binding]) -> Result<()>;
}

#[cfg(test)]
mod tests;

# Kuiper: Mixed-Architecture Single-Host Ranks

Kuiper is Flare's single-host asymmetric multi-rank architecture for pairing GPUs
of different NVIDIA compute capabilities in one serving process—for example, an
SM103 primary GPU paired with an SM120 or SM121 auxiliary GPU over PCIe. Rank
zero owns request admission, KV cache, continuation state, Engram tables, and
token completion; non-zero auxiliary ranks execute lockstep CUDA graphs over
portable mapped-host exchange buffers without device-to-device PCIe P2P memory
mappings, KV storage, or sampling sidecars.

Helios remains model-agnostic. Every asymmetry in loading, cache sizing, upload
filtering, and completion handling is derived from each rank's declared metadata
and bound buffers rather than model identity or capability flags. Model-specific
layer partitioning, cold-expert placement, and exchange kernels live outside
this repository in model recipes (such as `local-inference-lab/hyperflare-recipes`).

See [tensor-parallel ranks in one process](same-process-tp.md) for the shared
same-process threading model, [runtime core design](runtime-core-design.md) for
serialization and loading, and [Flare core design](core-design.md) for the
preparation/runtime boundary.

## 1. Implementation and Qualification Status

| Capability | Status | Scope and Location |
|---|---|---|
| Per-rank compute capability metadata (`RankBody.architecture`) and validation | Implemented | [core.proto](../helios/proto/core.proto), [schema.rs](../helios/src/core/schema.rs), [core.rs](../heliopause/src/export/core.rs), [protection.rs](../helios/src/runtime_core/protection.rs), [describe.rs](../helios/src/runtime_core/describe.rs), [inspect.rs](../flare-cli/src/inspect.rs) |
| Portable `MappedHost` peer export, import, validation, and registration | Implemented (Linux/POSIX) | [core_storage.py](../ignition/preparation/core_storage.py), [peer_storage.py](../ignition/preparation/peer_storage.py), [capture.rs](../heliopause/src/export/capture.rs), [mod.rs](../helios/src/runtime_core/mod.rs), [model.rs](../helios/src/runtime_core/model.rs) |
| Cacheless, Engram-free, and completion-free auxiliary peer ranks | Implemented | [memory.rs](../helios/src/runtime_core/memory.rs), [cache_memory.rs](../helios/src/runtime_core/cache_memory.rs), [model.rs](../helios/src/runtime_core/model.rs), [manifest.rs](../helios/src/execution/manifest.rs), [transport.rs](../helios/src/execution/transport.rs), [threaded.rs](../helios/src/execution/threaded.rs), [serving.rs](../helios/src/runtime_core/serving.rs), [execution.py](../ignition/preparation/execution.py) |
| Mixed-architecture rank compilation and primary-rank memory planning | Implemented | [rank_group.py](../ignition/preparation/rank_group.py), [aot.py](../ignition/preparation/aot.py), [planning.py](../ignition/planning.py), [model.py](../ignition/preparation/model.py) |
| Split-MoE exchange kernels and model-recipe cold-expert placement | External recipe scope | Implemented in external recipes (`hyperflare-recipes`); not part of core Ignition or Helios |
| Three-tier Kuiper + Parallax dynamic residency | Designed | Not implemented in this repository (Section 6.2) |

Qualification status on this branch:
- **Verified:** Rust unit and integration tests for `helios` (`--features runtime-core`),
  `heliopause`, `flare`, and `flare-cli`, plus CPU Python tests for `ignition`
  planning and preparation contracts.
- **Not qualified on this branch:** GPU-gated CUDA/TP tests
  (`FLARE_RUN_CUDA_TESTS=1`), end-to-end mixed-architecture core export and
  `flare run` serving with an external recipe, more than two ranks, and Windows
  shared host peer regions.

## 2. Per-Rank Compute Capability and Core Format

Mixed-architecture cores record each rank's SM compute capability independently
without breaking signed cores exported by earlier versions.

- **Wire schema:** `RankBody` in [helios/proto/core.proto](../helios/proto/core.proto)
  adds optional `repeated int32 architecture = 18` (`[major, minor]`). When
  empty, the loader and describer inherit `DeploymentBody.architecture`, so
  existing single-architecture signed cores load without re-export or
  re-signing.
- **Validation:** `validate_rank` in [helios/src/core/schema.rs](../helios/src/core/schema.rs)
  accepts an empty `architecture` vector or an exact two-element `[major, minor]`
  pair with `major > 0` and `minor >= 0`, rejecting any other shape or value.
- **Single-host vs. multi-host export:** `validate_ranks` in
  [heliopause/src/export/core.rs](../heliopause/src/export/core.rs) permits
  multiple hardware identities in a single-host export when every rank supplies
  a valid two-element compute capability and hardware identities map
  one-to-one with distinct compute capabilities. Multi-host cluster exports
  continue to reject mixed hardware.
- **Loading and inspection:** [protection.rs](../helios/src/runtime_core/protection.rs)
  propagates each rank's resolved `[major, minor]` into `Core::architecture` and
  `RankNeeds`. [describe.rs](../helios/src/runtime_core/describe.rs) emits each
  rank's effective architecture in inspection JSON, and
  [inspect.rs](../flare-cli/src/inspect.rs) renders all distinct rank
  architectures (for example, `sm_103 / sm_120`).

## 3. Portable Mapped-Host Peer Memory

Symmetric single-host tensor parallelism shares device memory via CUDA VMM
physical handles and `cuMemSetAccess` peer grants ([same-process-tp.md](same-process-tp.md)).
Mixed-architecture PCIe pairs cannot rely on device-to-device P2P DMA. Kuiper
replaces device-memory peer mappings with portable pinned host memory at a
fixed, identical virtual address across preparation and serving.

### 3.1 Preparation Lifecycle

1. **Owner allocation:** `CoreStorage.host_allocate(name, shape, dtype, shared=True)`
   in [core_storage.py](../ignition/preparation/core_storage.py) aligns the host
   cursor and region extent to `max(mmap.PAGESIZE, self.granularity)` so the
   recorded peer extent satisfies CUDA VMM alignment checks. On POSIX hosts,
   `HostRegion(..., shared=True)` creates an anonymous `memfd`
   (`os.memfd_create`), maps it with `MAP_SHARED | MAP_FIXED_NOREPLACE` at the
   reserved host virtual address, locks pages when `lock=True`, and registers
   the region with `cuMemHostRegister_v2` using
   `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP` (`flags = 3`).
   Windows rejects `shared=True` explicitly because POSIX `memfd` descriptor
   passing is unavailable there.
2. **Cross-process export and import:** `PeerStorage.export_host_region` in
   [peer_storage.py](../ignition/preparation/peer_storage.py) records the
   region's address and extent in `owned` and serializes duplicated file
   descriptors (`multiprocessing.reduction.DupFd`) for the requested reader
   ranks. Each peer calls `PeerStorage.import_host_region`, which constructs an
   `ImportedHostRegion`, maps the descriptor at the exact owner address,
   registers it with `cuMemHostRegister_v2`, verifies that
   `cuMemHostGetDevicePointer_v2` returns the recorded address, and records the
   mapping as a `Memory::Peer` import.
3. **Teardown:** After rank graphs drain during preparation shutdown,
   [rank_group.py](../ignition/preparation/rank_group.py) calls
   `PeerStorage.close_host_imports()` before closing owner `HostRegion`s,
   synchronizing each context before `cuMemHostUnregister` and `munmap`.

### 3.2 Export Validation and Native Loading

- **Capture validation:** `prepare_allocations` in
  [heliopause/src/export/capture.rs](../heliopause/src/export/capture.rs)
  accepts peer mappings backed by either `"device"` or `"mapped_host"` recorded
  allocations with matching address and byte extent.
- **Core validation:** `validate_peer_memory` in
  [helios/src/runtime_core/model.rs](../helios/src/runtime_core/model.rs)
  accepts `PeerMemory` entries whose owner holds a matching `Memory::Device` or
  `Memory::MappedHost` allocation, while continuing to reject address
  mismatches, extent mismatches, duplicate exports, or unlisted reader ranks.
- **Native serving registration:** During single-process loading in
  [helios/src/runtime_core/mod.rs](../helios/src/runtime_core/mod.rs),
  `NativeResources` registers a `Memory::MappedHost` allocation with
  `CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP` (`flags = 3`)
  whenever `self.peer_readers` contains that allocation's address (and keeps
  `flags = 2` for rank-private mapped host regions). Because every rank shares
  one process address space, peer graphs read and write the registered address
  directly without re-mapping or IPC handles.

## 4. Asymmetric Cache, Engram, and Completion Ownership

Auxiliary ranks that compute only offloaded sub-layers (such as cold MoE
experts) do not store KV pages, continuation state, Engram tables, or sampled
completions. Helios supports these ranks through declared storage and buffer
presence rather than rank-role flags.

### 4.1 Cacheless Peer Ranks

- **Detection:** `rank_owns_cache(core)` in
  [helios/src/runtime_core/memory.rs](../helios/src/runtime_core/memory.rs)
  returns true when a rank declares at least one runtime-sized allocation
  (`allocation.sizing.is_some()`). Rank zero is always required to own
  continuation-row and KV-page allocations; a core where rank zero lacks them
  is rejected.
- **Validation:** In [helios/src/runtime_core/model.rs](../helios/src/runtime_core/model.rs),
  a non-zero rank without sized allocations cannot declare `state_pool` or
  `memory_capacity` buffers.
- **Memory and offload sizing:** `capacities`, prefix-cache offload headroom,
  Selene pool registration (`minimum`, `budget.devices`, `gpu_uuids`, and
  `charges`), GPU KV codec cubin loading (`gpu_codec_cubins`), and
  `prepare_offload` operate only on cache-owning ranks (`cache_rank_indices`),
  storing a per-rank mapping in `NativeModel::offload_ranks`. A smaller-VRAM
  auxiliary peer therefore does not shrink rank zero's KV budget.
- **VMM handle budgeting:** `compact_cache_layouts` in
  [helios/src/runtime_core/cache_memory.rs](../helios/src/runtime_core/cache_memory.rs)
  receives both the full rank slice and `cache_rank_indices`: every rank's
  fixed allocations still count toward the process-wide Linux VMM handle limit,
  while cache layout compaction adjusts only cache-owning ranks.

### 4.2 Engram-Free Peer Ranks

`core_binds_engram` in [helios/src/runtime_core/model.rs](../helios/src/runtime_core/model.rs)
checks whether a rank's execution buffers reference any table buffer from the
core's Engram catalog (`weight_buffer`, `scale_buffer`, `hash_buffer`, or
`request_buffer`). Non-zero ranks with no Engram bindings skip Engram decoding
and store `None` in their `NativeModel::engram` slot.

### 4.3 Completion-Free Peer Ranks

- **Manifest contract:** `ExecutionManifest::validate_with_completion(false)` in
  [helios/src/execution/manifest.rs](../helios/src/execution/manifest.rs) (and
  `PreparedExecution(..., require_completion=False)` in
  [ignition/preparation/execution.py](../ignition/preparation/execution.py))
  permits a non-zero peer rank to omit `completion_records`. Rank zero always
  validates with `require_completion = true`.
- **Forbidden sidecars:** A completion-free peer must not declare
  `completion_records`, `completion_logprobs`, `logprob_requested`,
  `logprob_token_ids`, `logprob_token_id_counts`, `proposal_confidence`,
  `prompt_logprobs`, `prompt_target_tokens`, or
  `completion_speculative_survival`.
- **Layout matching:** `same_layout` in
  [helios/src/execution/threaded.rs](../helios/src/execution/threaded.rs)
  requires a completion-free peer to have no cache, offload, or Engram
  bindings, to match rank zero's `completion_token_capacity`, `max_active`, and
  graph list (name, kind, sampling mode, batch/token/draft/confidence/packed/
  prefill capacities), to match buffer geometry for every shared buffer (except
  rank-local startup tuning buffers), and to bind a subset of each graph's
  input buffers with identical byte sizes.
- **Step execution and timing:**
  - `filter_peer_uploads` sends a completion-free peer only the host uploads
    corresponding to buffers bound by the selected graph (such as `frame_index`
    and `submission_sequence`).
  - `GraphExecutor::new_unstarted_peer` in
    [helios/src/execution/transport.rs](../helios/src/execution/transport.rs)
    allocates `output: None` and skips completion/logprob/confidence/survival
    downloads and decoding, returning an empty `CompletionRecords` vector.
  - `ThreadedExecutor` skips completion-agreement checks for completion-free
    peers, rejects any unexpected completion records from them, and allows
    `ExecutionTiming::merge_rank` to merge rank zero's readiness timestamps with
    a peer that does not record readiness timestamps.
  - Startup tuning iterates up to the maximum number of tuning graphs across
    ranks so primary and auxiliary ranks may declare different startup tuning
    graph counts.

## 5. Mixed-Architecture Preparation

Preparing two different SM architectures in one local rank group requires four
preparation-layer behaviors:

1. **Per-rank CuTe DSL target:** `_configure_rank_cute_arch(device)` in
   [ignition/preparation/rank_group.py](../ignition/preparation/rank_group.py)
   sets `CUTE_DSL_ARCH` (`sm_<major><minor>a` for SM90+) and clears cached
   CuTe/b12x compile options in each rank process before spawning offline
   compiler workers. Without this, CuTe DSL probes CUDA device zero in the rank
   process and falls back to its default architecture in device-hidden compiler
   workers, producing target and cache-key mismatches on mixed-architecture
   hosts.
2. **Architecture-grouped AOT compilation:** `ForwardCache` in
   [ignition/preparation/aot.py](../ignition/preparation/aot.py) gathers each
   rank's CUDA compute capability and partitions forward compilation balancing
   by `(host, capability)` rather than host alone, so ranks with distinct SM
   targets never share compiled forward artifacts.
3. **Primary-rank memory budgeting:** `Plan.asymmetric_primary` in
   [ignition/planning.py](../ignition/planning.py) bases `Plan.free_bytes` on
   `self.gpus[0].free_bytes` instead of `min(gpu.free_bytes)` when a recipe
   budgets primary-rank KV/weights separately from an auxiliary peer's VRAM.
4. **Peer contract guard:** [ignition/preparation/model.py](../ignition/preparation/model.py)
   applies `tuned_bindings` updates to `server_config["model_contract"]` only
   when `model_contract` is present.

## 6. External Recipe Architecture (Designed / External Scope)

The components in this section are not implemented in core Flare/Helios; they
specify how recipes in `hyperflare-recipes` use the core interfaces above.

### 6.1 Lockstep Split-MoE Exchange over Portable `MappedHost`

A recipe splits each selected MoE layer's routed experts into a primary hot set
(on rank zero) and an auxiliary cold set (on rank one):

- Both ranks share one double-buffered `MappedHost` exchange slab sized to the
  active token capacity of split layers, indexed by `frame_index & 1` and
  synchronized per split layer with `submission_sequence` using system-scope
  memory ordering (`fence.sc.sys`, `st.release.sys`, `ld.acquire.sys`).
- On rank zero, after routing, a compacting kernel masks cold expert IDs out of
  the local MoE input, copies only tokens with at least one cold route (using
  already-quantized activations and scales) into the exchange slab, and
  publishes `(sequence, cold_row_count)`. Rank zero then executes its local hot
  and shared experts while rank one concurrently waits on the layer header,
  dequantizes only the `cold_row_count` active rows into local VRAM, runs its
  prepared `b12x` MoE kernels, writes back only active rows, and releases the
  completion header. Rank zero waits on the completion header and scatter-adds
  the cold outputs into the layer residual.
- For YOCO models where prefill layers (`0..N_prefill`) run over full prompt
  chunks while decoder layers (`N_prefill..N_layers`) run over compacted suffix
  tokens, keeping all prefill-layer experts resident on rank zero eliminates
  peer round-trips during prefill and shrinks the peer exchange slab and peer
  graph token capacities to the compacted suffix width.

### 6.2 Three-Tier Kuiper + Parallax Residency (Designed, Not Implemented)

Static split-MoE placement leaves a trade-off between prefill throughput and
decode latency when hot-set demand shifts across prompts or batches. A designed
three-tier extension combines Kuiper with dynamic HBM residency (Parallax):

- **Tier 1 (Primary HBM, pinned + dynamic slots):** Prefill layers stay pinned
  in primary HBM. Decoder layers reserve a pinned core of always-hot experts
  plus a dynamic HBM slot pool managed by Parallax residency updates.
- **Tier 2 (Auxiliary GPU GDDR7, read-only cold execution):** Every movable
  decoder expert is stored once in the auxiliary GPU's VRAM in its native `b12x`
  format. Uncached routes execute on the auxiliary GPU over the `MappedHost`
  exchange slab without moving expert weights across PCIe.
- **Tier 3 (Host RAM, promotion source):** Movable decoder experts are also
  backed in primary-accessible host RAM (for example, Grace RAM over NVLink-C2C)
  in the primary GPU's native format so Parallax promotes newly hot experts
  into Tier 1 at host-interconnect bandwidth while Tier 2 absorbs transient cold
  misses at GDDR7 bandwidth.

Implementing this design requires multi-rank compatibility for persistent
residency workers (currently rejected by `validate_persistent_worker_topology`
in [helios/src/runtime_core/model.rs](../helios/src/runtime_core/model.rs)) as
well as recipe-side routing and placement integration in `hyperflare-recipes`.

## 7. Invariants and Limitations

- **Single-host only:** Multi-host clusters reject mixed hardware during export
  and require homogeneous deployment ranks.
- **Rank zero remains authoritative:** Rank zero must own sized KV/state
  allocations and `completion_records`. Auxiliary optimizations apply only to
  non-zero ranks.
- **Persistent workers:** `validate_persistent_worker_topology` restricts
  persistent workers to single-rank cores; combining multi-rank Kuiper cores
  with persistent residency workers is not yet supported.
- **POSIX shared host memory:** `CoreStorage.host_allocate(..., shared=True)`
  requires Linux/POSIX `memfd_create` and `DupFd`; Windows preparation rejects
  it explicitly.
- **Backward compatibility:** Cores without `RankBody.architecture` or with
  symmetric cache/Engram/completion ranks preserve their existing wire
  encoding, validation, and execution paths without re-export.

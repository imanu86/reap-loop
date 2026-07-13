# Patch 0051: dynamic pinned expert arena

**Status:** implementation-grade design, 2026-07-13. No code is implemented by
this document. The design is anchored to the live read-only WSL worktree
`/root/ds4-v2-work` at DS4 base
`da0b3f63d7cc87c1f11c3c876fb57de3e0caca50`, with WIP changes in `ds4.c`,
`ds4_cuda.cu`, and `ds4_gpu.h` and last successful build `build_0050i.log`.

Patch 0051 replaces patch 0050's static `cudaHostRegister` ranges with a
reassignable `cudaHostAlloc` slot arena. It preserves the central 0050 result:
expert GEMMs continue to read VRAM, and pinned host memory is only a source for
bulk H2D DMA.

## 1. Decision summary

0051 has four distinct states of an expert. They must not be collapsed:

1. **Selected:** the router may choose `(layer, expert)`.
2. **Pageable-backed:** the immutable model mmap is authoritative and its pages
   may be resident in the Linux page cache.
3. **Pinned-ready:** a complete gate/up/down triplet is in a reusable pinned
   arena slot and can source H2D DMA without `pread` staging.
4. **VRAM-resident:** the existing streaming expert cache has a complete copy
   used by the GEMM.

The selection window must remain wide enough for quality. For Flash, the
current mechanism fixture keeps 154 of 256 experts in each of the 40 maskable
layers, or 6,160 expert triplets and 40.6055 GiB. That width produced one L2
render at `n=1`; it is not quality-validated until the pending `n>=3` graded
matrix passes. A 24 or 28 GiB pinned arena is therefore a hot tier, not the
selection policy and not the whole RAM window. Selected experts outside the
arena remain valid through the pageable mmap and the existing
`pread -> pinned stage -> H2D` fallback.

The final policy MUST NOT read a static domain mask. `mask60_self.txt` and
`DS4_CUDA_STREAM_FROM_RAM_MASKED` are permitted only as deterministic mechanism
fixtures while bringing up 0051. Production target sets are built from routing
mass observed in the current interaction, with prior live state used only as a
churn-reduction tie-break.

## 2. Hard invariants

These are release gates, not aspirations.

1. **VRAM compute only.** No arena pointer is inserted in `g_model_ranges`,
   returned by `cuda_model_range_ptr()`, or passed to a GEMM kernel.
2. **Immutable backing.** The model mmap/file is authoritative. Arena slots are
   disposable copies and are never written back to the model.
3. **Complete expert unit.** Gate, up, and down for one `(layer, expert)` move
   through the slot state machine together. A partial triplet is never READY.
4. **Generation validation.** A lookup is a hit only when the table entry, slot
   owner, slot content generation, model-map identity, geometry, and READY state
   all match. Any mismatch is a normal fallback miss, never a usable pointer.
5. **No overwrite under DMA.** A slot cannot enter LOADING until a CUDA event
   proves that all earlier H2D reads from the slot have completed.
6. **No early selection.** A target selection mask is not published until the
   target slot map is staged and the pageable portion of the target has passed
   its WRAP batch. Publication happens only at an inference-quiescent boundary.
7. **One observable snapshot.** Inference observes either the old
   `(selection mask, slot map)` snapshot or the new snapshot, never a partially
   applied mix. This is a quiescent transaction, not a claim of a lock-free
   cross-CPU/GPU atomic instruction.
8. **Safe abort.** An aborted transaction keeps the old selection mask. Slots
   already repurposed are removed from a sanitized slot-map snapshot, so those
   old selections fall back to pageable/pread serving rather than reading stale
   bytes.
9. **Bounded physical footprint.** Once a pinned copy is validated, the
   corresponding file-backed source pages are advised `DONTNEED`. The runtime
   must not retain both a 24/28 GiB pinned copy and the same bytes in page cache.
10. **Off means old behavior.** With `DS4_CUDA_DYNAMIC_ARENA` unset, 0051 adds no
    allocation, policy mutation, or hot-path lookup.

## 3. Measured geometry and budget

The loaded Flash GGUF has a uniform routed-expert size class:

| Component | Bytes per expert |
|---|---:|
| gate, IQ2_XXS | 2,162,688 |
| up, IQ2_XXS | 2,162,688 |
| down, Q2_K | 2,752,512 |
| **slot payload** | **7,077,888 (6.75 MiB)** |

All three component sizes and the full slot are 4 KiB aligned. The arena needs
no padding for this GGUF.

| Requested arena | Whole slots | Used pinned bytes |
|---:|---:|---:|
| 24 GiB | 3,640 | 23.9941 GiB |
| 28 GiB | 4,247 | 27.9954 GiB |

The 154-wide maskable selection window contains 6,160 experts. If every arena
slot were assigned only to those 40 layers, the pageable tail would be 2,520
experts / 16.6113 GiB at 24 GiB, or 1,913 experts / 12.6101 GiB at 28 GiB.
In practice, the pin allocator also reserves demand-ranked slots for the three
unmaskable hash layers, so telemetry must report the exact maskable pinned
coverage rather than assuming these ideal maxima.

### 3.1 Allocation order

Patch 0050 registers 24 GiB in `ds4_engine_open()` before session allocations.
0051 must not do that.

1. `ds4_engine_open()` binds the model mmap/fd and records routed-expert
   geometry, but does not allocate the large arena.
2. `ds4_session_create()` completes `metal_graph_alloc_raw_cap()`. This creates
   context tensors and the SPEX async readback, including the existing
   `ds4_gpu_async_read_alloc()` allocation.
3. A new CUDA prepare call creates the selected-upload stream, its four staging
   buffers/events, and any other mandatory small pinned allocations.
4. Only then does it call one contiguous `cudaHostAlloc` for the arena.

Use `cudaHostAllocDefault`. Do not request mapped access and never call
`cudaHostGetDevicePointer` for this allocation. Page locking is required for
DMA; a compute-visible UVA pointer is specifically not part of the contract.

Default request is 24 GiB. `DS4_CUDA_DYNAMIC_ARENA_GB=28` is the experimental
arm. `28 -> 24 -> disabled` is the startup fallback order when 28 is requested.
The default must remain 24 until the A/B gate in section 13 promotes 28.

The arena is process/model-map lifetime, not prompt lifetime. 0051 v1 supports
the current single active CUDA engine/session topology. A second concurrent CUDA
session must either reuse already-accounted mandatory allocations or reject
dynamic-arena mode explicitly; it must not discover the constraint as a later
opaque pinned-allocation failure.

### 3.2 Pageable backing is the existing mmap

Do not allocate a second anonymous 40 GiB RAM store. The pageable tier is
`ds4_model.map` plus the Linux page cache. WRAP page-touches selected experts
that are not pinned. After copying an expert into a pinned slot, reuse
`cuda_model_discard_source_pages()` and `cuda_model_drop_file_pages()` to make
the duplicate source pages reclaimable.

At a turn switch, old-only pageable ranges may be dropped after inference is
quiesced and before the new WRAP batch starts. This limits the transition peak
to approximately pinned bytes plus retained/new pageable tail. If the
transaction later aborts, those old pages can fault back from the immutable
file; correctness is unchanged and the abort telemetry records the cold
rollback.

## 4. Arena data model

The authoritative arena state lives in `ds4_cuda.cu`. Use fixed-size vectors
sized from runtime `n_layer * n_expert` and `slot_count`; do not use an
`unordered_map` on the expert-load hot path.

```c
typedef enum ds4_gpu_arena_slot_state {
    DS4_GPU_ARENA_FREE = 0,
    DS4_GPU_ARENA_RETIRING,
    DS4_GPU_ARENA_LOADING,
    DS4_GPU_ARENA_STAGED,
    DS4_GPU_ARENA_READY,
    DS4_GPU_ARENA_POISONED,
} ds4_gpu_arena_slot_state;

typedef struct ds4_gpu_arena_binding {
    uint32_t slot;             /* UINT32_MAX means no pinned binding */
    uint32_t state;            /* copied for diagnostics; slot is authoritative */
    uint64_t slot_generation;
    uint64_t snapshot_generation;
} ds4_gpu_arena_binding;
```

Internal slot metadata additionally contains:

```c
struct cuda_dynamic_arena_slot {
    char *host_ptr;
    uint32_t layer;
    uint32_t expert;
    uint64_t content_generation;
    uint64_t checksum;
    uint64_t last_dma_sequence;
    ds4_gpu_arena_slot_state state;
};
```

The arena owns two dense binding tables, active and staging. A snapshot switch
exchanges their roles. A table lookup at index `layer * n_expert + expert` is a
hit only if all of the following hold:

- binding slot is in range;
- binding snapshot generation is the active generation;
- slot state is READY;
- slot `(layer, expert)` equals the query;
- binding slot generation equals slot content generation;
- model map, model size, gate/up/down offsets, and expert sizes equal the arena
  geometry captured at bind time.

Use 64-bit generations. Increment a slot generation before each repurpose, not
after loading, so a torn or failed load can never validate against its previous
owner. Generation zero is invalid.

### 4.1 State transitions

```text
FREE ------ reserve ------> LOADING ---- all 3 copies/hash OK ----> STAGED
  ^                            |                                  |
  |                            +---- failure ----> POISONED       |
  |                                                               |
  +---- transaction cleanup <---- RETIRING <---- READY <--- publish
                                      |
                                      +---- retire event fails ---> POISONED
```

`STAGED` means bytes are complete but the active snapshot cannot see them.
`READY` means the active binding table may reference the slot. A retained READY
slot does not change generation and is never copied.

## 5. Public CUDA API contract

Add CUDA-only declarations to `ds4_gpu.h`, under the same platform guard used
by patch 0050. Names may change during implementation, but responsibilities and
failure semantics may not.

```c
typedef struct ds4_gpu_dynamic_arena_txn ds4_gpu_dynamic_arena_txn;

typedef struct ds4_gpu_dynamic_arena_load {
    uint32_t layer;
    uint32_t expert;
    uint32_t slot;
    uint64_t slot_generation;
    void    *host_ptr;
    uint64_t host_bytes;
} ds4_gpu_dynamic_arena_load;

int ds4_gpu_dynamic_arena_bind(
        const void *model_map, uint64_t model_size,
        uint32_t n_layer, uint32_t n_expert,
        uint64_t gate_bytes, uint64_t down_bytes);

int ds4_gpu_dynamic_arena_prepare(
        uint64_t requested_bytes,
        uint64_t *allocated_bytes,
        uint32_t *slot_count);

int ds4_gpu_dynamic_arena_begin(
        const ds4_gpu_stream_expert_table *layer_tables,
        const uint8_t *target_pinned,
        const uint8_t *target_pruned,
        uint32_t entry_count,
        ds4_gpu_dynamic_arena_txn **txn,
        ds4_gpu_dynamic_arena_load **loads,
        uint32_t *load_count);

int ds4_gpu_dynamic_arena_finish_load(
        ds4_gpu_dynamic_arena_txn *txn,
        uint32_t slot, uint64_t slot_generation,
        uint64_t checksum, int copy_ok);

int ds4_gpu_dynamic_arena_prepare_mask(
        ds4_gpu_dynamic_arena_txn *txn,
        const void *model_map,
        const uint64_t *bias_offsets,
        const float *bias_rows,
        uint32_t row_count, uint32_t row_width);

int ds4_gpu_dynamic_arena_publish(
        ds4_gpu_dynamic_arena_txn *txn,
        uint64_t *snapshot_generation);

void ds4_gpu_dynamic_arena_abort(ds4_gpu_dynamic_arena_txn *txn);
void ds4_gpu_dynamic_arena_release(void);
```

`begin()` computes retain/evict/load against the active binding table, reserves
slots, records and waits for the retire event, and returns writable load
descriptors only after victims are DMA-safe. There is at most one transaction.

`finish_load()` is called once per descriptor after the WRAP workers join. It
validates the lease `(slot, generation)`. A failed lease becomes POISONED and
cannot be silently reused in the same transaction.

`prepare_mask()` allocates/copies an inactive contiguous device bias block and
prepares replacement `g_model_ranges` entries without exposing them. It must
not mutate the live bias entries on allocation or copy failure.

`publish()` requires every target binding to be retained READY or newly STAGED.
It promotes STAGED to READY, switches the active binding table and prepared bias
view while inference is quiescent, and returns the new generation. Old device
bias storage is released only after the transaction's CUDA fence proves it is
unused.

`abort()` publishes a sanitized slot map containing only still-valid retained
slots and leaves the old selection bias active. It may reduce pinned coverage,
but it cannot change which experts are selectable.

## 6. Dynamic policy, not a domain file

### 6.1 Interaction lifecycle

The controller lives in `ds4.c` and owns the target mask. It uses these phases:

1. **Begin interaction:** at `ds4_pace_reset_for_prefill()`, clear current-turn
   rating accumulators and publish K0/unmasked selection for the new prompt
   suffix. Keep the old arena snapshot as a performance hint only.
2. **Rating-only prefill:** `metal_graph_spex_note_selected()` records selected
   expert frequency and router weight for every prefill layer. Since prefill is
   unmasked, these observations are not censored by the previous domain.
3. **Build target:** at `ds4_pace_apply_prefill_mask()`, rank experts by current
   interaction mass. In each maskable layer keep exactly
   `ceil(0.60 * DS4_N_EXPERT)` (154 on Flash). Hash layers remain unmasked.
4. **Build pin target:** rank the selected set plus observed hash-layer experts
   for the finite arena. Allocate a fair per-routed-layer quota first, then
   distribute remainder by normalized within-layer rank. Prefer an already
   READY expert only to break equal/near-equal scores; residency must not
   override materially higher current-turn mass.
5. **Promote and publish:** execute the transaction in sections 7-9 before
   decode starts.
6. **During decode:** continue the existing unbiased full-router knock
   observation. LIVEMASK runs in rating-only mode: it may propose a bounded
   delta, but it may not write `g_reap_mask_pruned` directly. SPEX proposals use
   the same transaction path and are never selected before residency/backing is
   ready.

The current `DS4_PACE_LIVEMASK` boolean needs a mode split. A suitable interface
is `DS4_PACE_LIVEMASK_MODE=off|rating|actuate`; 0051 production uses `rating`.
`actuate` retains the old experimental behavior only for regression comparison.

### 6.2 Sparse or short prompts

A short prompt may leave many experts with zero mass. Selection still needs a
wide quality envelope. Rank ties in this order:

1. positive current-turn mass;
2. expert selected at least once in the current turn;
3. retained expert from the immediately previous live snapshot;
4. expert id for deterministic replay.

This is not a static domain policy. The previous snapshot is a one-turn
hysteresis input and cannot displace a positive current-turn candidate.
Zero-score experts do not need pinned slots; unused arena capacity may remain
FREE until decode supplies evidence.

### 6.3 Policy exclusions

- `DS4_REAP_MASK_FILE` cannot be a concurrent writer in dynamic mode.
- `DS4_CUDA_STREAM_FROM_RAM_MASKED` cannot coexist with the arena because both
  consume the same WDDM pinned-memory budget.
- The narrow K8/K23/K32 masks are not fallback policy. If dynamic target
  construction fails, stay K0 or retain the last successfully published wide
  target.
- A static family/domain mask can be replayed in mechanism tests, but a run that
  needs it to choose the production target has not completed 0051.

## 7. Retain, evict, load, warm, and drop delta

Given active snapshot `A`, target selection `S`, and target pinned set `P`:

```text
retain_pin = ready(A) intersect P
evict_pin  = ready(A) minus P
load_pin   = P minus ready(A)
warm_ram   = S minus P
drop_ram   = selected(A) minus S
```

There is one subtle case: an expert moving from pinned to pageable remains in
`S` but its source mmap pages may have been dropped when it was pinned. Include
it in `warm_ram` before reusing its slot.

Transaction order:

1. Acquire `g_dynamic_window_mutex`; reject or coalesce a second proposal.
2. Assert the selected async-load worker has no outstanding job and reach a
   token/turn boundary after `ds4_gpu_end_commands()`.
3. `begin()` records a transaction-owned retire event on
   `g_stream_selected_upload_stream`. Wait for it before returning victim
   pointers to CPU workers.
4. Advise `drop_ram` old-only ranges `DONTNEED`. This is safe even on abort.
5. WRAP page-in `warm_ram`, including pinned-to-pageable demotions.
6. WRAP-copy `load_pin` into reserved slot descriptors.
7. Validate every slot lease and checksum; mark successful loads STAGED.
8. Advise source mmap/file pages for successful pinned loads `DONTNEED`.
9. Prepare the inactive device bias rows.
10. Publish slot map plus bias snapshot, then release-store the matching host
    selection snapshot.
11. Release the mutex and emit one transaction telemetry record.

Retained slots are never copied. The cost of coding -> Roman history is thus
the measured delta, not 24/28 GiB unless the two live windows are disjoint.

### 7.1 WRAP implementation

Keep the existing page-touch machinery for `warm_ram`. Add a sibling
`ds4_dynamic_arena_wrap_batch_main()` near
`ds4_reap_prefetch_batch_main()` for slot copies.

- Sort load descriptors by their first source offset to favor sequential file
  access.
- Assign one whole expert to one worker. That worker copies gate, up, and down
  into fixed offsets inside the leased slot.
- Use `DS4_REAP_PREFETCH_THREADS`, default 8 and max 16, for the first patch.
- Join every worker. The transaction path is not detached/fire-and-forget.
- Compute a 64-bit FNV-1a checksum over the destination triplet using the
  existing `hash_bytes()` convention. Full source-vs-destination verification
  is enabled in test/diagnostic mode; production stores the destination hash
  and relies on immutable-source `memcpy` plus end-to-end exactness tests.
- A worker error marks only its lease failed, but any failed target load aborts
  publication of the target selection.

The current detached `ds4_reap_prefetch_batch()` may remain as a non-binding
hint when dynamic mode is off. It is not a valid readiness signal for 0051.

## 8. CUDA H2D path and event lifetime

### 8.1 Read order

Add `cuda_dynamic_arena_copy_expert_to_device()` and call it before the cold
sidecar/legacy path in both existing expert loaders:

1. `cuda_stream_expert_cache_load_slot()` for the global VRAM expert cache.
2. The direct-load branch of
   `cuda_stream_selected_cache_begin_compact_load()`.

On an arena hit, enqueue exactly three H2D copies from the slot to the VRAM
gate/up/down destinations on `g_stream_selected_upload_stream`. Record the
existing selected-upload completion dependency after the triplet, not once per
component. The GEMM continues to wait through
`cuda_stream_selected_upload_wait_if_recorded()`.

`cuda_model_copy_to_device_streamed()` remains the fallback implementation. Its
0050 static-range lookup is removed or compiled only in explicit 0050
compatibility mode; it must not discover dynamic slots by raw offset alone.

### 8.2 Event ownership

The existing single `g_stream_selected_upload_done_event` expresses the
compute dependency for the latest upload batch. It does not prove when an
individual host slot may be overwritten. Add transaction-owned retirement:

- Increment a global `arena_dma_sequence` after each expert triplet enqueue and
  store it in `slot.last_dma_sequence`.
- At transaction begin, prohibit new expert-load submissions, create a fresh
  `cudaEventDisableTiming` event, and record it after all earlier upload-stream
  work.
- The event object belongs to the transaction until record and wait have both
  completed. It is never overwritten by another transaction.
- A single event covers all victims because one upload stream provides total
  order. Per-slot CUDA events are unnecessary for 3,640/4,247 slots.
- Destroy the event only in publish/abort cleanup after its last query/wait.

If event creation or record fails before any H2D is enqueued, synchronize the
upload stream and continue conservatively if synchronization succeeds. If an
arena H2D enqueue succeeds but recording its compute-dependency event fails,
preserve the 0050 hardening rule: synchronize and accept the completed copy.
Never start an overlapping pread fallback into the same VRAM destination.

If synchronization fails after an enqueue, invalidate the destination VRAM
slot, poison the source arena slot, fail the current expert load, and disable
arena hits until a clean arena reset.

## 9. Atomic publication

The current code mutates `g_reap_mask_pruned` in place and performs one
`ds4_gpu_model_range_update()` per layer. That is not a transaction.

Introduce a double-buffered host `ds4_dynamic_window_snapshot` containing:

- `uint64_t generation`;
- the full pruned/eligible bitmap;
- the target-pinned bitmap;
- the CUDA arena snapshot generation/handle;
- per-layer selected and pinned counts for diagnostics.

Replace direct policy writes with copy-on-write target construction. CPU router
code loads the active snapshot once per layer through an accessor instead of
reading a mutable global array during publication.

Publication occurs only while the inference/control mutex is held and no GPU
commands or selected-load worker are active:

1. CUDA prepares all device bias rows in inactive storage.
2. CUDA switches the active arena binding table and bias-range pointers in
   `ds4_gpu_dynamic_arena_publish()`.
3. `ds4.c` release-stores the matching host snapshot pointer/generation. The
   final host step cannot fail.
4. Inference is allowed to resume.

This ordering gives inference atomic semantics because there is no reader in
the interval. Do not describe the pair as a hardware-atomic CPU/GPU store.

SPEX consume cannot clear an eligibility bit immediately while an upload or
GEMM may still depend on it. It submits a bounded next-snapshot delta. Physical
slots are retired only by the same transaction/event path.

## 10. Fallback and recovery matrix

| Failure | Required behavior |
|---|---|
| 28 GiB `cudaHostAlloc` fails | Clear CUDA error, try 24 GiB once. |
| 24 GiB allocation fails | Disable 0051; continue existing pageable/pread serving. |
| Model map/geometry changes | Quiesce, drain upload stream, release arena, then bind the new map. |
| Arena lookup miss or stale generation | Count reason and use existing fallback. |
| WRAP page-in/copy/checksum failure | Abort target publication; keep old mask or K0 and sanitize slot map. |
| Mask-device prepare failure | Abort before host mask changes. |
| Event create/record failure before enqueue | Stream synchronize; continue only on success. |
| Event failure after enqueue | No overlapping fallback; synchronize, then accept or fail/poison. |
| Transaction cancelled | Join workers, sanitize bindings, keep old selection, release leases/event. |
| All slots busy/pinned | Do not evict a protected core ad hoc. Defer the proposal or serve it pageable. |
| Telemetry/invariant violation | One-shot error, disable arena hits, preserve normal serving path. |

No arena failure is allowed to narrow selection as a memory-saving response.
Selection and residency remain separate concerns.

## 11. Telemetry contract

Keep hot-path counters in memory and emit summaries at transaction completion,
periodically under the existing diagnostic gate, and at cleanup. Do not log one
line per expert in production.

### 11.1 Startup

- requested/attempted/allocated GiB;
- slot payload, stride, count, and unused tail bytes;
- allocation time and fallback `28 -> 24 -> off` reason;
- mandatory pinned allocations completed before arena allocation;
- model-map identity and geometry hash.

### 11.2 Per transaction JSONL

- transaction id, old/new generation, reason (`first_turn`, `turn_switch`,
  `livemask_delta`, `spex_delta`);
- selected count/bytes and pinned count/bytes;
- retain/evict/load/warm/drop experts and bytes;
- old/new intersection, Jaccard, churn, and zero-score tie count;
- retire-event wait, WRAP page-in, WRAP copy, checksum, device-mask prepare,
  commit, and total milliseconds;
- committed/aborted plus exact abort stage/reason;
- pinned coverage of selected demand mass, not just expert count.

### 11.3 H2D tier counters

- arena lookup queries and hits;
- miss by `unbound`, `loading`, `stale_generation`, `owner_mismatch`,
  `geometry_mismatch`, and `poisoned`;
- arena DMA calls/bytes/failures;
- pageable/pread fallback calls/bytes;
- VRAM resident hits/misses, retaining existing SPEX counters;
- event create/record/wait failures, sync fallbacks, and retire wait time;
- maximum concurrent LOADING/RETIRING slots and poisoned-slot count.

### 11.4 Policy/quality safety

- selected width min/mean/max per maskable layer, expected 154 on Flash;
- union-of-selection and union-of-pinned experts over the run;
- selection and pin churn per turn/token;
- current-turn mass retained by selection and by pinned tier;
- count of selected experts lacking READY pin (expected and served pageable);
- count of selected experts lacking any valid backing (must remain zero).

## 12. Exact live code changes

Only three existing DS4 source files need production changes for 0051 v1.
`ds4_ssd.c`, `ds4_metal.m`, `ds4_rocm.cu`, and `Makefile` are not required if
all new APIs and call sites retain the existing CUDA platform guards.

### 12.1 `ds4_gpu.h`

- Replace the patch-0050-only `ds4_gpu_register_masked_ranges()` contract with
  the dynamic arena types and lifecycle/transaction APIs in section 5.
- Keep the API CUDA-only so Metal/ROCm do not need false implementations in this
  patch.

### 12.2 `ds4_cuda.cu`

Replace or retire these 0050 structures/functions:

- `cuda_masked_pin_range` and `g_masked_pin_ranges`;
- `cuda_masked_pin_register()`;
- `cuda_masked_pin_finalize()`;
- `cuda_masked_pin_covers()`;
- `cuda_masked_pin_release_all()`;
- exported `ds4_gpu_register_masked_ranges()`.

Add the arena state, dense binding tables, transaction object, diagnostics, and
the exported APIs near those definitions. Reuse the existing source-page drop
helpers.

Modify these exact existing functions:

- `cuda_stream_selected_stage_pool_alloc()`: split small-resource preparation
  from lazy fallback use so staging/events exist before the large arena.
- `cuda_stream_selected_stage_release()`: drain before destroy and leave arena
  release ordering explicit.
- `cuda_stream_selected_upload_done_event_ensure()` and
  `cuda_stream_selected_upload_wait_if_recorded()`: keep compute dependency;
  do not reuse this event as a slot-retirement fence.
- `cuda_stream_expert_cache_load_slot()`: try generation-validated arena triplet
  DMA before cold/pageable loading; preserve invalid-before-write semantics.
- `cuda_stream_selected_cache_begin_compact_load()`: add the same arena triplet
  path to the direct-load branch around the current three
  `cuda_model_copy_to_device_streamed()` calls.
- `cuda_model_copy_to_device_streamed()`: remove the dynamic policy's dependence
  on static offset coverage; retain the hardened legacy fallback.
- `cuda_model_set_host_map()`: a true map change must block new transactions,
  drain uploads, release the arena, then release ordinary CUDA ranges. Same-map
  cache rebuilds retain the arena.
- `ds4_gpu_model_range_update()`: leave for non-transactional legacy users, but
  add a separate inactive bulk-bias prepare/swap path for dynamic publication.
- `ds4_gpu_cleanup()`: synchronize, abort/join any transaction, release arena
  while mmap is valid, then destroy upload/stage resources.
- `ds4_gpu_init()`: initialize arena diagnostics/lifecycle only; do not allocate
  the 24/28 GiB block here.

Do not change the existing `cuda_stream_expert_cache_slot` VRAM cache into the
host arena. The two caches have different ownership and event lifetimes.

### 12.3 `ds4.c`

Remove production dependence on:

- `g_stream_masked_pruned`;
- `ds4_stream_from_ram_masked_load()`;
- `ds4_stream_from_ram_masked_register()`;
- the `ds4_engine_open()` call to that registration function.

Add the dynamic window controller and WRAP-copy worker next to
`ds4_reap_prefetch_batch_main()` so it can reuse range geometry and worker-count
configuration.

Reuse `graph_stream_expert_table_make()` to produce the per-layer source
geometry passed to the CUDA transaction API; do not derive a second offset
scheme for the arena.

Modify these exact functions/groups:

- `ds4_session_create()`: after `metal_graph_alloc_raw_cap()` and graph
  configuration succeed, bind geometry, prepare small CUDA staging, then
  allocate the arena. Arena failure is non-fatal.
- `ds4_engine_close()`: retain current `ds4_gpu_cleanup()` before
  `model_close()`/`munmap`; update the comment to include arena ownership.
- `ds4_reap_mask_effectively_pruned()`, `ds4_reap_mask_apply()`,
  `ds4_reap_mask_apply_layer()`, and
  `ds4_reap_mask_apply_layer_direct()`: read from an immutable active snapshot;
  dynamic publication uses bulk inactive bias rows instead of in-place partial
  mutation.
- `ds4_pace_init()`, `ds4_pace_livemask_wants_router()`,
  `ds4_pace_livemask_note_probs()`, and
  `metal_graph_spex_note_selected()`: implement rating-only observation across
  prefill and decode without writing selection.
- `ds4_pace_livemask_seed()`, `ds4_pace_livemask_adapt_k()`, and
  `ds4_pace_livemask_scan()`: build target bitmaps copy-on-write and submit
  bounded transaction proposals. Remove direct writes to
  `g_reap_mask_pruned` in dynamic mode.
- `ds4_pace_livemask_publish_pin_mass()`: read eligibility from the active
  snapshot and continue to feed VRAM pin priority; it is rating input, not an
  independent selection-mask publisher.
- `ds4_reap_prefetch_batch_sync()`: keep as a legacy page-touch helper only.
  Dynamic SPEX readiness must come from a completed arena transaction, not its
  current boolean return after worker joins.
- `ds4_spex_mask_update_layer()` and `ds4_spex_mask_consume_layer()`: replace
  page-touch-then-in-place-mask behavior with transaction submit/lease release.
- `ds4_pace_reset_for_prefill()`: start a new interaction rating epoch and
  unmask selection without discarding valid pinned hints.
- `ds4_pace_apply_prefill_mask()`: build the 154-wide runtime target, calculate
  pin quotas/delta, run WRAP synchronously, and publish before decode.
- `ds4_session_sync()`: preserve all four existing full/resumed and
  chunked/short-suffix exits, ensuring each successful suffix calls the dynamic
  end-prefill commit exactly once and each cancel/failure aborts a pending
  transaction.

All 42 current direct references to `g_reap_mask_pruned` must be audited. Reads
move to an active-snapshot accessor; writers either build a staging bitmap or
remain in explicitly legacy/off-mode code. A mixed mutable/snapshot model is not
acceptable.

## 13. Incremental patch and test sequence

Each step is independently buildable and has an off-gate. Do not land the whole
state machine as one unbisectable patch.

### Step 0: close the 0050 baseline

- Export the current 0050 patch and record binary/model hashes.
- Repeat coffee `temp=0`, 60-token ON/OFF exactness on `build_0050i` or newer.
- Preserve the measured 5 GiB and 24 GiB diagnostic artifacts.
- No performance claim is required to start 0051.

### Step 1: allocator and lifecycle, no reads

- Add bind/prepare/release APIs and allocate the arena after session resources.
- Verify exact Flash counts: 3,640 slots at 24 GiB; 4,247 at 28 GiB.
- Exercise `28 -> 24` with a test-only allocation fault injection.
- Open/create/free/close repeatedly, with and without MTP map switching.
- Gate: env unset is byte-identical and has no arena allocation/log line.

### Step 2: table and transaction self-test, no inference use

- Add a diagnostic self-test for FREE/LOADING/STAGED/READY/RETIRING/POISONED,
  generation mismatch, duplicate owner, abort sanitization, and retain delta.
- Fill a small test arena using a MiB/slot-count override; compare each copied
  triplet against its mmap source and verify stored checksum.
- Inject failure after gate and after up; neither slot may become READY.

### Step 3: arena H2D read path

- Wire both global-cache and direct selected-load paths.
- Run with a small mixed window so diagnostics prove arena hits and normal
  fallback misses in the same request.
- Exactness gate: identical selection target, arena ON versus arena disabled,
  coffee `temp=0`, 60 tokens, identical output SHA256.
- Counter gate: arena DMA bytes are nonzero; compute never receives a host/UVA
  pointer; no 0.10 t/s fine-grained PCIe signature.

### Step 4: event and reuse stress

- Alternate two disjoint small pin sets for hundreds of transactions while
  selected-upload events are enabled.
- Inject event-create and event-record failure before and after enqueue.
- Add a test delay between enqueue and retire to maximize the overwrite race.
- Verify destination checksums/output, zero stale-generation hits, and no
  overlapping fallback after a successful enqueue.

### Step 5: atomic mask plus slot-map publication

- Add inactive bulk bias storage and host snapshot double buffering.
- Inject failures after N slot loads and during mask preparation.
- Gate: target mask generation never becomes visible on failure; the old mask
  remains active; repurposed old bindings miss and fall back rather than hit.
- Audit every `g_reap_mask_pruned` reference and run both CPU-router and GPU
  router paths used by the current build.

### Step 6: pageable-tail WRAP and memory accounting

- Warm target pageable tail, drop old-only ranges, and drop source pages after
  pin copy.
- Measure WSL `MemAvailable` and Windows available memory throughout first fill
  and a disjoint switch. Do not infer residency from requested bytes.
- Gate: physical use does not behave like `arena + full 40.6 GiB duplicate`;
  cleanup restores memory; no unregister/munmap errors.

### Step 7: final runtime policy

- Enable rating-only prefill and build top-154 per maskable layer from the live
  prompt; no static mask env is set.
- Run two-turn same-session transitions in both directions: coding -> Roman
  history and Roman history -> coding. Include a short resumed suffix and a
  chunked prefill.
- Gate: target logs show current-interaction mass, nonzero retain where domains
  overlap, delta-sized load, and publication before first decode token.
- SPEX bounded deltas use the same transaction; no stale candidate is selected.

### Step 8: controlled 24 versus 28 GiB A/B

- Same binary, model, prompts, context, dynamic policy, and target generations.
- Use balanced back-to-back order such as 24/28/28/24 and record page-cache and
  thermal state; at least three measured runs per arm after warmup.
- Record TTFT transition cost, decode t/s, arena hit bytes, pread fallback
  bytes, event waits, WSL/Windows memory minima, and all allocation failures.
- Keep 24 as default unless 28 completes repeated 4k-context startup/cleanup
  without pinned-allocation failures, preserves at least 7 GiB WSL
  `MemAvailable`, and produces a repeatable end-to-end benefit.

### Step 9: quality and failure protocol

- Mechanism exactness is arena ON/OFF with the same dynamic target.
- Quality is a separate n=3 campaign for the runtime-generated wide target,
  following fail-fast and long-output protocol. A micro-smoke may reject but
  cannot promote the policy.
- Apply the existing speed abort rule: an initially slow ramp is allowed; an
  out-of-scale value around 0.1 t/s or no ramp after about 100 tokens is aborted
  and attributed with tier counters.
- Every run stores env, CLI, binary/model hashes, patch chain, prompt, output,
  transaction telemetry, memory trace, and stop reason. Never use `pkill`.

## 14. Acceptance criteria

0051 is complete only when all are true:

- a contiguous 24 GiB arena works after mandatory allocations, with 28 GiB an
  evidence-based optional/default arm;
- `(layer, expert) -> (slot, generation, state)` validation prevents stale hits;
- slot reuse is fenced by a transaction-owned CUDA event;
- turn switch cost and telemetry are proportional to retain/evict/load delta;
- selected but unpinned experts use the pageable/pread fallback correctly;
- slot map and selection mask publish as one quiescent inference snapshot;
- aborts preserve correctness and never expose partial triplets;
- source-page reclamation prevents pinned-plus-page-cache duplication;
- arena ON/OFF is bit-exact for an identical selection target;
- dynamic two-turn prompts work without `DS4_REAP_MASK_FILE` or
  `DS4_CUDA_STREAM_FROM_RAM_MASKED`;
- no final verdict relies on a static domain mask.

## 15. Deliberate non-goals for 0051 v1

- Multiple simultaneous model maps or concurrent CUDA inference sessions.
- A second pinned size class for mixed-precision expert layers. If encountered,
  those layers remain pageable/fallback and are reported explicitly.
- GPU kernels reading pinned host weights directly.
- Lock-free publication while decode is running.
- Predicting a named domain from prompt text. Routing mass is the policy input;
  labels such as `coding` or `history` are evaluation descriptions only.
- Removing the existing VRAM expert cache, pin-by-mass, tier, or in-place epoch
  machinery. 0051 feeds those mechanisms with a faster host source; it does not
  replace them.

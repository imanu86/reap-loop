# Patch 0052: dynamic quantized cold expert tier after 0051

**Status:** implementation-grade experimental design, 2026-07-13. This
document implements no code and makes no speed, memory, or quality benefit
claim. Patch 0052 may start only after patch 0051 satisfies its own acceptance
criteria. In the current repository, 0051 is itself a design document and its
24/28 GiB runtime A/B, dynamic-window exactness, and quality gates remain
uncompleted.

0052 adds an optional lossy, pageable cold representation below 0051's native
pinned expert arena. It does not replace the immutable GGUF, the exact mmap
fallback, the existing VRAM expert cache, or the selection policy. Its purpose
is to test whether batched, ahead-of-use cold compression and promotion can be
useful on the RTX 3060 plus 64 GiB host. Existing evidence says that doing CQ1
materialization or repacking synchronously on a selected miss is unusably slow.

## 1. Evidence labels and decision boundary

Every factual statement in this design belongs to one of these classes:

- **MEASURED-ARTIFACT:** recorded in a checked-in target-hardware run or probe
  artifact that can be inspected in this repository.
- **LEDGER-REPORTED:** recorded as measured in a checked-in ledger/design, but
  its complete raw runtime artifact is not present here or is cited at an
  external `/root` path. It is context, not independently reverified benchmark
  evidence.
- **TESTED-MECHANISM:** established by unit tests or a smoke, but not by a
  controlled end-to-end experiment.
- **PROJECTED:** produced by trace replay, arithmetic, or an offline latency
  model. It is not a runtime result.
- **PROPOSED:** new 0052 behavior or an experiment still required.

The decision is:

1. **PROPOSED:** the GGUF's original routed-expert bytes are the only
   authoritative representation. Gate/up are IQ2_XXS and down is Q2_K. The
   immutable file/mmap remains available for exact fallback and regeneration.
2. **PROPOSED:** a CQ record is a disposable, lossy derivative in bounded
   pageable RAM. It is never authoritative, never written into the GGUF, and
   never the only route to an expert.
3. **PROPOSED:** residency and representation are orthogonal. An expert can be
   selected without a CQ copy, can have a CQ copy without being selected, and
   can be exact or approximate in the pinned/VRAM tiers.
4. **PROPOSED:** no compression, CQ decode, or native-layout repack may execute
   on the synchronous selected-miss path. A miss uses 0051's exact
   mmap/pread/staging fallback. CQ is usable only after a WRAP batch prepares a
   complete arena slot ahead of inference.
5. **PROPOSED:** 0052 is off by default. A lossy representation may become a
   candidate only after separate mechanism, memory, latency, and long-output
   quality gates. Passing a codec unit test or smoke cannot promote it.
6. **PROPOSED:** the first implementation supports the existing CQ1 family
   only because that path has prior mechanical coverage. CQ1g32, CQ1g64, and
   CQ1g256 are experiment arms, not approved production formats.

## 2. What the repository already establishes

### 2.1 Target geometry and source bytes

**LEDGER-REPORTED (ledger J35, geometry repeated in 0051):** the tested model
has 43 routed layers, 256 experts per layer, and 11,008 routed experts. Each
expert has:

| Component | Native format | Bytes | MiB |
|---|---|---:|---:|
| gate | IQ2_XXS | 2,162,688 | 2.0625 |
| up | IQ2_XXS | 2,162,688 | 2.0625 |
| down | Q2_K | 2,752,512 | 2.6250 |
| **triplet** | mixed native | **7,077,888** | **6.7500** |

The complete routed-expert payload is 72.5625 GiB. Generic compression of
sampled native bytes was ineffective: zlib1 averaged ratio 0.9931, zlib6
0.9923, lzma0 0.9966, and bz2 expanded the data. Entropy was about 7.95
bits/byte. A lossless compressed copy is therefore not a memory tier candidate
on current evidence.

**TESTED-MECHANISM:** `tests/test_gguf_inspect_ds4.py` checks native block-byte
geometry for IQ2_XXS and Q2_K. It does not check every tensor in the production
GGUF and does not establish runtime correctness.

### 2.2 CQ1 lab and runtime evidence

**TESTED-MECHANISM:** `tests/test_ds4_cold_codec_lab.py` checks CQ1 payload
sizes, the group-mean-absolute scale rule, and only the ordering
`all-cold < down-only < native` for one group size. It does not test a complete
expert, a CUDA kernel, output quality, or performance.

**LEDGER-REPORTED (ledger J35):** an offline sample of 768 native blocks reported
aggregate dot nMAE around 0.029 across IQ2_XXS and Q2_K and weight nRMSE around
0.58. Estimated all-expert CQ1 footprints were 34.27 GiB for CQ1g256,
40.31 GiB for CQ1g64, and 48.38 GiB for CQ1g32. These are codec-error and size
observations, not model-quality results.

**LEDGER-REPORTED mechanism result (ledger J34):** an older lossless RAM sidecar smoke returned
`OK`, reported no copy/verify failures, materialized 1,836 entries, performed
4,056 copies, and allocated 12.393 GiB of native RAM blobs. Prompt time was
158.896 s. This established addressing/checksum/fallback plumbing on that
older path, not a useful memory or latency policy and not integration with
0051.

**LEDGER-REPORTED mechanism result with negative output (ledger J36):** the older CQ1 prototype
could materialize, repack, verify, and copy without reported mechanical errors,
but broad CQ1 prefill produced `????` and was very slow. This rejects broad
prefill admission; it does not isolate which codec, component, or quality
failure caused the output.

**MEASURED-ARTIFACT negative (ledger J38 and its checked-in local run):** with CQ1g32 admitted
after 50 native decode tokens, the first 50 tokens ran at 1.52 t/s and the last
14 fell to 0.06 t/s. The run recorded 1,073 entries, 3,612 CQ copies,
4,828.50 MiB of CQ bytes, and 24,381.00 MiB repacked, with no reported
mechanical failures. The output degraded/truncated. This directly rejects
synchronous CQ materialization/repack on selected or cache misses.

**MEASURED-ARTIFACT negative (ledger J39 and checked-in exchange runs):** one-expert-per-token exchange plus WRAP ran
a large WRAP on every micro-step and was extremely slow. The exchange signal
worked mechanically, but per-token WRAP is not an acceptable actuator.

### 2.3 Residency traces and projections

**LEDGER-REPORTED (ledger J31):** in the local HTML160 pair, 6,922 of 6,923 observe
rows used `selected_direct`; only one was resident. Trace-replayed LRU hit rates
were about 0.34 at cap258, 0.61 at cap512, 0.76 at cap1024, and 0.81 at cap2048.
The trace rates do not prove the runtime would attain them.

**PROJECTED (ledger J32):** prompt-preload trace replay projected hot-hit
0.8489 for a cap1024 policy and 0.6811 for cap512. No runtime tensors were
changed in that simulation.

**PROJECTED (`runs/ds4/20260710_elat_tier_latency/REPORT.md`):** the offline
latency model projected possible tiering outcomes and an asynchronous CQ
scenario. It explicitly lists unmeasured CQ isolation, contention, prefill,
and resident-hit work. Its 3.7-4.1 t/s tiering range and 8.95 t/s theoretical
ceiling are not measured 0052 benefits and must not be used as acceptance
numbers.

### 2.4 0050/0051 and host-memory evidence

**MEASURED-ARTIFACT:** the corrected contiguous `cudaHostAlloc` probe passed at 24, 28,
30, and 31 GiB and failed to reach 32 GiB in the block probe. Windows reports
63.8905 GiB physical RAM, whose WDDM half-memory ceiling is 31.9453 GiB. A
31 GiB probe is not a safe DS4 runtime budget because staging, context, and
other pinned allocations still need space.

**LEDGER-REPORTED and not a closed 0050 gate:** a pre-P1 0050 coffee pair was
bit-exact for 60 tokens. Its timing was uncontrolled and cannot support a
speed conclusion. The post-P1 controlled campaign recorded only its start and
first OFF request in the checked-in status; it did not complete the A/B.

**PROPOSED by 0051, not yet established:** a 24 GiB arena has 3,640 whole
native slots and a 28 GiB arena has 4,247. 0051 requires exact ON/OFF behavior,
transactional publication, source-page reclamation, and controlled 24/28 GiB
testing before 0052 can rely on those mechanisms.

### 2.5 Checked-in evidence index

This design was produced from repository files only. It did not inspect or run
the `/root` worktrees and models cited by historical ledger rows.

| Evidence | Repository source | Use in this design |
|---|---|---|
| 0051 contract and pending gates | `docs/DESIGN_0051_DYNAMIC_ARENA.md` | integration boundary, native geometry, slot counts, invariants, experiments |
| 0050 fast-path/probe report | `docs/PUNTO_V2_ZEROCOPY_DYNAMIC_ARENA_20260713.md` | pre-P1 exactness caveat, 24/28/30/31 GiB probe results |
| WDDM ceiling analysis | `docs/WSL_WDDM_PINNED_LIMIT_20260713.md` | 63.8905 GiB host and 31.9453 GiB graphics-memory ceiling |
| Cold/tiering history | `docs/EXPERIMENTS_LEDGER.md`, rows J31-J39 | ledger-reported mechanics, sizes, traces, negatives, and caveats |
| Generated ledger warning | `docs/DS4_EXPERIMENT_LEDGER_20260710.md` | legacy versus runner-measured evidence boundary; no current speed claim |
| Codec unit coverage | `tests/test_gguf_inspect_ds4.py`, `tests/test_ds4_cold_codec_lab.py` | block/payload formulas and narrow helper behavior only |
| Direct CQ1 native50 run | `runs/ds4/20260709_cq1_parallel/local_3060_cq1_native50/` | direct J38 timing, counters, configuration, and output prefix |
| Direct exchange runs | `runs/ds4/20260709_cq1_parallel/local_3060_exchange_observe*` | direct J39 batching negative/context |
| Offline latency model | `runs/ds4/20260710_elat_tier_latency/REPORT.md` | projections and explicit missing measurements only |
| Incomplete controlled 0050 A/B | `runs/ds4/20260712_v2_zerocopy/controlled_ab_0050/20260713_0050_ab01/` | proves the checked-in campaign did not close |

Where a summary and direct artifact differ, the direct artifact governs. A
ledger-only result may motivate an experiment but cannot satisfy a 0052
acceptance gate.

## 3. Representation ladder and physical placement

CQ1 stores one sign bit per value plus a two-byte fp16 scale per group. For 256
weights its block payload is 34 bytes at group 256, 40 at group 64, and 48 at
group 32. The larger CQ1g32 record is the least size-aggressive CQ1 arm; the
smaller CQ1g256 record is the most aggressive.

The following sizes are arithmetic from the tested geometry and codec formulas.
They are **PROJECTED storage sizes**, not allocation or quality measurements:

| Cold recipe | MiB/expert | All 11,008 GiB | Native bytes retained in record | CQ components |
|---|---:|---:|---|---|
| native exact | 6.7500 | 72.5625 | gate + up + down | none |
| down-only:CQ1g32 | 5.6250 | 60.4688 | gate + up | down |
| down-only:CQ1g64 | 5.3750 | 57.7813 | gate + up | down |
| down-only:CQ1g256 | 5.1875 | 55.7656 | gate + up | down |
| gate-up-only:CQ1g32 | 5.6250 | 60.4688 | down | gate + up |
| gate-up-only:CQ1g64 | 5.1250 | 55.0938 | down | gate + up |
| gate-up-only:CQ1g256 | 4.7500 | 51.0625 | down | gate + up |
| all:CQ1g32 | 4.5000 | 48.3750 | none | gate + up + down |
| all:CQ1g64 | 3.7500 | 40.3125 | none | gate + up + down |
| all:CQ1g256 | 3.1875 | 34.2656 | none | gate + up + down |

The mixed-component rows are required experiment points because the existing
test only establishes their size ordering. There is no evidence that gate/up
or down is safer to quantize. Footprint ordering must not be treated as
quality ordering.

0052 defines these physical copies:

| Copy | Representation | Residence | Lifetime | Authority |
|---|---|---|---|---|
| `SOURCE_EXACT` | original IQ2_XXS gate/up and Q2_K down | immutable GGUF on NVMe, mmap virtual address; pages may enter page cache | model-map lifetime | **only authoritative copy** |
| `COLD_RECORD` | one explicit CQ/mixed recipe | bounded anonymous pageable WSL RAM | evictable policy-cache lifetime | derivative only |
| `ARENA_EXACT` | byte-identical native triplet | 0051 `cudaHostAlloc` slot | until transactional reuse | validated copy of source |
| `ARENA_REPACKED` | native kernel layout reconstructed from CQ components, with exact components copied from source/validated record | 0051 `cudaHostAlloc` slot | until transactional reuse | approximate derivative |
| `VRAM_EXACT` | native exact layout | existing expert-cache/direct destination | existing cache lifetime | compute copy of exact arena/source |
| `VRAM_REPACKED` | native layout containing CQ-reconstructed values | existing expert-cache/direct destination | existing cache lifetime | approximate compute copy |
| `WRAP_SCRATCH` | one block's decode/repack scratch plus descriptors | ordinary pageable worker memory | one job | none |

There is no persistent fp16/fp32 expert, no second pinned cold pool, no CQ
device cache, and no GPU kernel reading host memory. `ARENA_REPACKED` occupies
the same 6.75 MiB native-layout slot as `ARENA_EXACT`; 0052 therefore does not
increase VRAM slot capacity or reduce H2D bytes. Any such benefit would require
a different design and new measurements.

### 3.1 Exact and approximate promotion are different operations

An approximate CQ record cannot reconstruct the original IQ2_XXS/Q2_K bytes.
The word "native" must describe layout or source explicitly:

- **Exact promotion:** `SOURCE_EXACT -> ARENA_EXACT -> VRAM_EXACT`.
- **Approximate promotion:** `COLD_RECORD -> ARENA_REPACKED -> VRAM_REPACKED`.
- **Exact fallback:** `SOURCE_EXACT -> existing 0051 fallback -> VRAM_EXACT`.

Repacking CQ values into IQ2_XXS/Q2_K-shaped buffers does not make them exact.
Telemetry, slot metadata, and output attribution must preserve that fact.

### 3.2 No cascading quantization

CQ1g32, CQ1g64, and CQ1g256 are independent derivatives, not refinement
layers. A g64 record may not be made from g32, a g256 record may not be made
from g64, and an `ARENA_REPACKED` slot may not be compressed into another
recipe. Every new lossy record is produced from `SOURCE_EXACT` or from an
`ARENA_EXACT` slot whose full source checksum is already validated. This avoids
untracked generational quality loss.

Only one active `COLD_RECORD` recipe exists per `(layer, expert)` in v1. A
replacement is built in inactive storage and atomically swaps after validation.

## 4. State and provenance model

0052 extends, but does not weaken, 0051's slot state machine. The cold pool has
its own states:

```text
ABSENT -- reserve --> BUILDING -- payload/hash OK --> COLD_READY
   ^                     |                              |
   |                     +-- failure --> POISONED      +-- evict --> RETIRING
   +---------------- transaction cleanup <-------------------------+

COLD_READY -- WRAP lease --> REPACKING -- triplet/hash OK --> ARENA_STAGED
                                  |
                                  +-- failure --> arena POISONED;
                                                    cold record retained or poisoned
```

Each cold record contains at least:

```c
typedef enum ds4_cold_recipe {
    DS4_COLD_NATIVE_EXACT = 0,
    DS4_COLD_DOWN_CQ1_G32,
    DS4_COLD_DOWN_CQ1_G64,
    DS4_COLD_DOWN_CQ1_G256,
    DS4_COLD_GATE_UP_CQ1_G32,
    DS4_COLD_GATE_UP_CQ1_G64,
    DS4_COLD_GATE_UP_CQ1_G256,
    DS4_COLD_ALL_CQ1_G32,
    DS4_COLD_ALL_CQ1_G64,
    DS4_COLD_ALL_CQ1_G256,
} ds4_cold_recipe;

typedef struct ds4_cold_record_meta {
    uint32_t layer;
    uint32_t expert;
    uint32_t state;
    uint32_t recipe;
    uint64_t record_generation;
    uint64_t source_model_identity;
    uint64_t source_geometry_hash;
    uint64_t source_triplet_checksum;
    uint64_t payload_checksum;
    uint64_t payload_bytes;
} ds4_cold_record_meta;
```

0051 arena and VRAM-cache metadata gain:

- `representation = EXACT | REPACKED`;
- the component recipe;
- source model identity and source checksum;
- cold record generation for repacked slots;
- reconstructed triplet checksum;
- the selection/quality snapshot generation that admitted approximation.

The checksum fields have distinct meanings. `source_triplet_checksum` hashes
the original GGUF bytes. `payload_checksum` hashes the CQ/mixed record.
`reconstructed_triplet_checksum` hashes the native-layout approximate bytes.
Only an exact slot may require reconstructed/source equality.

## 5. Hard invariants

These are release gates.

1. **One exact authority.** The immutable bound GGUF/mmap is the only exact
   source. Deleting or modifying it while the engine is active is unsupported.
2. **Selection is independent.** Quantization may never clear an eligibility
   bit, lower the 0051 selected width, or substitute a narrow mask after a
   memory or codec failure.
3. **Representation is explicit.** Every arena and VRAM hit validates exact or
   repacked provenance. Unknown representation is a miss, never an exact hit.
4. **Complete expert unit.** Gate, up, and down plus one complete recipe move
   together. Mixed old/new components are never READY.
5. **No cascade.** Lossy input is never the source of a new lossy format.
6. **Generation validation.** Record generation, slot generation, owner,
   source identity, geometry, recipe, checksums, and snapshot generation must
   agree before an approximate slot is usable.
7. **Exact means byte-identical.** An exact-arm counter or log may increment
   only after source-byte equality. A native-layout CQ reconstruction is never
   counted as exact.
8. **No CQ on a selected miss.** The token path may look up already READY
   arena/VRAM copies. It may not materialize, compress, decode, or repack CQ.
   Its fallback is the exact 0051 mmap/pread path.
9. **VRAM compute only.** As in 0051, GEMMs receive only VRAM pointers. No cold
   pool or arena pointer enters a compute kernel.
10. **No overwrite under DMA.** 0051's transaction-owned retire event protects
    exact and repacked arena slots equally.
11. **Atomic quality publication.** Inference observes one matching selection
    mask, arena map, representation recipe, and quality generation. It never
    observes a new lossy recipe under an old attribution snapshot.
12. **Safe abort.** Failure retains the old selection and representation
    snapshot. Repurposed bindings are sanitized. Exact fallback remains valid.
13. **Bounded duplicate pages.** After a cold record or exact arena copy is
    validated, its source pages are advised reclaimable. Requested allocation
    bytes are not accepted as proof that physical duplication was removed.
14. **Memory floors.** Cold admission stops before either WSL or Windows
    available memory falls below 8 GiB. Crossing 7 GiB is a hard run abort.
15. **Off is 0051.** With 0052 unset, there is no cold-pool allocation, codec
    work, representation change, or extra hot-path branch beyond one static
    disabled check.
16. **No invisible quality fallback.** If an experiment asks for approximate
    serving but falls back exact, both bytes and expert counts are reported.
    A mostly-exact run cannot be attributed to its requested CQ arm.

## 6. Policy sets and snapshot semantics

For a proposed 0051 selection snapshot define:

```text
S = selected experts; quality/policy width, unchanged by 0052
H = experts required exact in the 0051 arena/VRAM hot path
A = experts allowed to use a named approximate recipe in this experiment
C = experts desired as COLD_RECORD cache entries
R = all remaining experts, served exactly from SOURCE_EXACT on demand
```

The sets obey:

```text
H subset S
A subset S
H intersect A = empty
C need not be selected
S subset (H union A union R)
```

`A` is empty in exact-control mode. In a lossy experiment, being in `A` only
permits use of an already prepared approximate copy; it does not require a
synchronous reconstruction. If an `A` expert lacks a valid prepared copy, it
uses exact fallback and records `approx_not_ready_exact_fallback`.

0052 does not create a second domain classifier. It consumes 0051's current
interaction mass and target generation:

1. current-turn positive mass ranks exact hot demand;
2. 0051's fair per-layer quota prevents a few layers from consuming the arena;
3. retained exact slots may break near-ties but cannot displace materially
   higher current-turn mass;
4. cold records are a bounded cache for next-ranked or recently displaced
   experts, not a selection source;
5. zero-score/old-turn entries are first cold-pool eviction candidates.

No production policy is chosen by this design. Exact hot count, approximate
band width, recipe, and cold-pool byte cap are experiment parameters until the
quality and A/B gates close.

## 7. WRAP compression, promotion, and demotion

### 7.1 Delta classes

0051's retain/evict/load/warm/drop plan is extended with representation-aware
jobs:

```text
retain_exact  = active ARENA_EXACT intersect target H
retain_approx = active ARENA_REPACKED with matching recipe intersect target A
load_exact    = H minus retain_exact
promote_cq    = A with matching COLD_READY minus retain_approx
build_cold    = target C minus matching COLD_READY
evict_cold    = active cold records minus target C, as required by byte budget
exact_warm    = selected exact-pageable targets chosen for WRAP page-in
```

An old approximate slot becoming exact is always `load_exact` from
`SOURCE_EXACT`; it is not blessed exact in place. A recipe change always builds
a new cold record from exact source.

### 7.2 Transaction order

At an interaction boundary or a deliberately batched exchange boundary:

1. Acquire 0051's dynamic-window mutex and freeze new selected-upload work.
2. Build the immutable target selection and representation snapshot without
   changing active state.
3. Reserve cold-record bytes and arena leases. Refuse the lossy delta, not the
   selection, if either memory floor would be threatened.
4. Record/wait 0051's upload-stream retirement event before reading or reusing
   any victim arena slot.
5. Run one bounded WRAP batch containing whole-expert jobs:
   - `LOAD_EXACT`: source mmap -> arena exact slot;
   - `BUILD_COLD`: exact source or checksum-validated exact arena slot ->
     inactive pageable cold record;
   - `PROMOTE_CQ`: ready cold record -> native-layout repacked arena slot;
   - `WARM_EXACT`: page-touch exact fallback ranges selected by the 0051 plan.
6. Join all workers. Validate leases, source identity, payload lengths, and
   checksums. A target triplet is STAGED only when all three components pass.
7. Advise source pages reclaimable for successful exact copies/cold builds.
   Advise evicted cold pages reclaimable only after no worker or snapshot can
   reference them.
8. Prepare 0051's inactive device bias and the inactive representation table.
9. Publish arena bindings, quality recipe, and selection as one quiescent
   snapshot. Then release old cold records whose readers are drained.
10. Emit transaction, memory, codec, and fallback telemetry.

Any target job failure aborts representation publication. The old selection
stays active. If 0051 permits publishing the same selection with fewer pinned
copies, the sanitized map may do so only with exact fallback and an explicit
abort record; it may not silently enable approximation.

### 7.3 WRAP worker rules

- Reuse `DS4_REAP_PREFETCH_THREADS`, default 8 and maximum 16, for the first
  experiment so 0051 and 0052 do not create competing worker pools.
- Assign one complete expert to one worker. A triplet is the scheduling and
  failure unit.
- Sort exact-source jobs by first GGUF offset. Group CQ-promotion jobs by cold
  pool extent. Do not interleave random per-component work across experts.
- Codec conversion is block-streamed. Do not allocate a full fp16/fp32 expert.
- Hash the exact source, CQ payload, and reconstructed triplet according to the
  provenance rules in section 4. Diagnostic mode compares every byte expected
  to be exact and re-runs deterministic reconstruction.
- Source pages for completed jobs are dropped incrementally rather than after
  the entire batch, while recognizing that `DONTNEED` is advisory.
- A worker error poisons its destination lease. Any missing target load aborts
  the target representation snapshot.
- No detached worker may outlive its transaction, model map, arena, or cold
  pool.

### 7.4 Background preparation is prepare-only

J39 motivates batching work away from the token path, but no existing artifact
measures CQ contention during decode. Therefore v1 behavior is:

- default: run codec jobs only in a quiescent WRAP batch;
- experimental background mode: build inactive `COLD_RECORD`s only from the
  immutable mmap, under a byte/time throttle;
- do not reuse active arena victims in background mode;
- collect completed records at the next quiescent boundary;
- if work misses its deadline, defer the representation change or use exact
  fallback; never stall a token waiting for CQ.

Background decompression into free, leased arena slots is a later arm after the
contention experiment. Publication remains quiescent in every arm.

### 7.5 Selected-miss path

The order at an expert request is:

1. valid VRAM resident hit, exact or attributed repacked;
2. valid 0051 arena READY hit followed by native-size H2D;
3. exact 0051 mmap/pread/staging fallback;
4. load failure.

There is intentionally no `COLD_RECORD -> repack -> H2D` step here. A counter
named `sync_cq_miss_attempts` exists and must remain zero. This converts the J38
negative into an executable invariant.

## 8. RTX 3060 plus 64 GiB memory accounting

### 8.1 Measured fixed limits

| Item | Value | Evidence/meaning |
|---|---:|---|
| Windows physical RAM | 63.8905 GiB | measured host value |
| WDDM maximum system memory for graphics | 31.9453 GiB | half physical RAM; matches probe ceiling |
| RTX 3060 VRAM | 12 GiB class | existing target; 0052 adds no persistent VRAM tier |
| 0051 default arena | 24 GiB / 3,640 slots | proposed 0051 default |
| 0051 experimental arena | 28 GiB / 4,247 slots | not default without A/B |
| maximum contiguous probe that passed | 31 GiB | probe only, not a runtime allocation target |
| 0051 minimum WSL floor | 7 GiB | prior design acceptance floor |
| 0052 admission / hard floor | 8 GiB / 7 GiB | proposed conservative admission and abort thresholds |

The prior `.wslconfig` value of 62 GB left too little Windows headroom and was
identified as a host stability risk. 0052 cannot assume all physical RAM is
available to WSL. Every run records the actual VM limit and both WSL and
Windows available memory.

### 8.2 Whole-catalog combinations do not fit this design

The following arithmetic excludes DS4 fixed tensors, context, staging, process
RSS, kernel memory, page cache, Windows, and the 7/8 GiB floors:

| 24 GiB arena plus | Bytes before all other runtime memory |
|---|---:|
| all native experts | 96.5625 GiB |
| all:CQ1g32 | 72.3750 GiB |
| all:CQ1g64 | 64.3125 GiB |
| all:CQ1g256 | 58.2656 GiB |

Even the smallest listed whole-catalog CQ arm is not admissible with a 24 GiB
arena on the recommended roughly 56-57 GiB WSL envelope, before any runtime
headroom. 0052 must therefore use a byte-capped partial cold cache. "All
experts compressed in RAM" is an offline scenario, not the 64 GiB runtime
plan.

### 8.3 Active maskable-tail arithmetic

0051's 40 maskable layers at keep-154 contain 6,160 experts and 40.6055 GiB of
native bytes. In the idealized case where every arena slot serves only those
layers, the native tail is 2,520 experts at 24 GiB or 1,913 at 28 GiB.

| Cold format for ideal tail | 24 GiB: cold GiB | 24 GiB total | 28 GiB: cold GiB | 28 GiB total |
|---|---:|---:|---:|---:|
| all:CQ1g32 | 11.0742 | 35.0742 | 8.4067 | 36.4067 |
| all:CQ1g64 | 9.2285 | 33.2285 | 7.0056 | 35.0056 |
| all:CQ1g256 | 7.8442 | 31.8442 | 5.9548 | 33.9548 |
| gate-up-only:CQ1g32 | 13.8428 | 37.8428 | 10.5084 | 38.5084 |
| down-only:CQ1g32 | 13.8428 | 37.8428 | 10.5084 | 38.5084 |

These are lower-bound planning examples, not promised RSS. 0051 also needs
demand-ranked slots for three unmaskable hash layers. Those allocations reduce
maskable pinned coverage and increase the actual tail. Cold-pool metadata,
allocator fragmentation, exact source pages, and transition overlap also add
physical use. Runtime telemetry must calculate the real expert sets and bytes;
it may not infer them from the ideal table.

### 8.4 Runtime budget formula

Allocate mandatory context, selected-upload staging, the 0051 arena, and cold
pool metadata in that order. Then compute cold admission from measured state:

```text
cold_admission_bytes = min(
    configured_cold_cap,
    wsl_memavailable - 8 GiB - transition_reserve,
    windows_available - 8 GiB - host_monitor_reserve)
```

If either available term is non-positive, cold admission is disabled and 0051
continues. `transition_reserve` includes, at minimum, all in-flight source
pages, destination records not yet published, per-worker scratch, and allocator
rounding for the configured batch. Its initial value is measured in the memory
ramp experiment; it is not assumed from requested bytes.

The allocator is byte-capped, not record-count-capped. It reports requested,
committed, resident, live-payload, fragmentation, and peak transition bytes.
Before reserving a record it evicts inactive zero/low-score cold entries. It
does not evict an active arena slot or alter selection to satisfy the cap.

### 8.5 Physical-page accounting

For every memory experiment record concurrently:

- WSL `MemAvailable`, `MemFree`, cached/file pages, anonymous RSS, swap use,
  major faults, and process RSS/PSS where available;
- Windows available physical memory and commit;
- WDDM/pinned allocations and every CUDA allocation failure;
- arena live bytes, cold live payload, cold allocator overhead, in-flight
  source bytes, and pages advised `DONTNEED`;
- VRAM used and existing expert-cache slot count.

A virtual mmap range is not counted as resident bytes. A successful
`DONTNEED` call is not counted as reclaimed bytes until the OS counters show
it. Cleanup must restore memory to a predeclared tolerance measured by an
allocation-only control; this design does not invent that tolerance.

## 9. Failure and recovery matrix

| Failure | Required behavior |
|---|---|
| Source model identity/geometry changes | Stop new work, join workers, drain uploads, discard all cold/arena derivatives, then bind the new source. |
| Cold pool allocation would cross admission floor | Do not allocate; evict inactive cold records or disable 0052 for the transaction. Preserve 0051 selection. |
| Exact source read/checksum failure | Abort transaction and fail loudly; never substitute CQ as authority. |
| CQ build failure | Poison inactive record, keep prior record if valid, use exact fallback. |
| CQ payload checksum/recipe mismatch | Poison record and use exact fallback. |
| Repack failure or partial triplet | Poison destination arena slot; abort representation publication. |
| Stale record/slot/snapshot generation | Count miss and use exact fallback. |
| Approximate copy appears in exact arm | Correctness failure; stop the run immediately. |
| CQ work requested on token miss | Invariant failure; disable 0052 and stop the experiment. |
| WRAP misses publication deadline | Defer representation update or publish the same selection with exact fallback; never wait inside a token. |
| Worker cancellation | Join all workers, release inactive records/leases, retain old snapshot. |
| CUDA retire-event failure | Follow 0051 synchronize/poison rules; no slot reuse without proven completion. |
| Device mask prepare failure | Abort before host selection/quality snapshot changes. |
| Telemetry attribution incomplete | Run is invalid for benefit/quality claims. Serving may continue exact. |

## 10. Telemetry contract

### 10.1 Startup

- 0051 generation and acceptance/build identity;
- model identity, full routed geometry, and source checksum policy;
- arena requested/allocated bytes and slots;
- cold recipe, configured/admitted cap, allocator alignment, and per-recipe
  expected record bytes;
- WSL limit and available memory, Windows physical/available memory, WDDM cap;
- background mode and worker/time/byte throttle;
- exact-control versus lossy-quality mode.

### 10.2 Per transaction

- old/new selection and quality generations;
- `S/H/A/C/R` experts and bytes, per layer and total;
- retain/load/promote/build/evict/warm counts and bytes;
- source exact bytes read, CQ bytes written/read, native bytes reconstructed,
  source pages dropped, and actual memory deltas;
- queue wait, retire wait, exact load, compress, repack, checksum, WRAP total,
  mask prepare, publish, and total milliseconds;
- committed/aborted, exact stage/reason, and fallback representation;
- WSL/Windows minima and cold-pool peak/fragmentation.

### 10.3 Serving and quality attribution

- VRAM exact hits, VRAM repacked hits, arena exact hits, arena repacked hits;
- exact mmap/pread fallbacks split by `not_cached`, `not_ready`, `stale`,
  `poisoned`, `budget`, and `deadline`;
- exact and approximate H2D calls/bytes;
- selected calls and router mass served by each recipe;
- `sync_cq_miss_attempts`, required to be zero;
- tokens containing any approximate expert call and first such token;
- output run id, quality arm, recipe, approximate call/mass fraction, and all
  fallback fractions.

Production diagnostics are summarized, not logged once per expert request.
Diagnostic tests may emit per-expert provenance to make failures reproducible.

## 11. Exact implementation and experiment sequence

Each step is independently buildable, off-gated, and must preserve 0051 when
disabled.

### Step 0: prerequisite closure

Do not implement the 0052 serving path until all of these exist:

1. post-P1 0050 exactness evidence or an explicit decision that 0051 supersedes
   it;
2. implemented 0051 arena lifecycle, generation table, retirement event, and
   atomic selection/binding publication;
3. 0051 exact arena ON/OFF bit-exactness with the same selection target;
4. completed 24 GiB memory/lifecycle gate, with 28 GiB still experimental
   unless its A/B has passed;
5. selected-pageable exact fallback and abort sanitization tests;
6. the runtime-generated wide-target quality gate required by 0051.

Failure of any prerequisite blocks 0052. It is not repaired by choosing a more
aggressive cold codec.

### Step 1: deterministic CPU codec matrix

Using existing model bytes but no GPU:

- add golden complete-expert fixtures for all nine CQ/component recipes;
- verify exact payload sizes in section 3 for every component geometry;
- run exact source -> CQ -> native-layout reconstruction twice and require
  identical payload/reconstructed checksums;
- compare the new codec byte-for-byte with the older prototype on shared
  golden blocks;
- inject truncated payload, wrong recipe, wrong source identity, bit flip, and
  generation mismatch;
- measure error separately for gate, up, and down, each layer band, and each
  recipe; report distributions rather than one aggregate average;
- verify no full-expert fp16/fp32 allocation and bounded worker scratch.

This step can reject formats. It cannot approve model quality.

### Step 2: cold-pool and transaction self-test

With a MiB-sized pool and no inference:

- exercise ABSENT/BUILDING/COLD_READY/RETIRING/POISONED;
- replace recipes and prove inactive-build/atomic-swap behavior;
- force duplicate owner, stale record, stale slot, source-map change, partial
  component, and every abort point;
- alternate exact and repacked arena slots for hundreds of transactions;
- prove no cascade by rejecting an approximate source descriptor;
- prove selection bits never change when every cold operation fails.

### Step 3: integrated lossless control

Before a lossy arm, run a small `native exact cold record` mechanism control:

- same binary, selection target, arena size, prompts, and cache;
- cold-control OFF versus exact-record ON;
- coffee, temp 0, 60 tokens: identical output SHA256;
- at least one exact cold build, WRAP load, arena hit, VRAM load, eviction, and
  exact fallback must be evidenced by counters;
- inject a failed record and require exact fallback with identical output;
- require `VRAM_REPACKED=0`, `ARENA_REPACKED=0`, and
  `sync_cq_miss_attempts=0`.

This isolates 0052 state integration from lossy quality.

### Step 4: isolated codec and copy microbench

On the target 3060 host, outside generation, measure each complete recipe with
1, 2, 4, 8, and 16 workers:

- exact-source warm-page compression;
- exact-source cold-page compression;
- CQ decode plus native-layout repack without H2D;
- H2D from a prepared exact slot and prepared repacked slot;
- combined WRAP promotion into free slots;
- p50/p95/p99 per expert and GiB/s, CPU utilization, WSL/Windows memory
  bandwidth, source faults, and memory minima.

J38 conflates materialization, repack, copy, and live decode. This experiment
must separate them. No latency threshold is presumed; the measurements choose
which recipes, if any, advance.

### Step 5: WRAP batch and deadline experiment

Use fixed disjoint expert deltas of 43, 256, 512, and 1,024 experts. For each
recipe and worker count:

- prepare at one quiescent boundary, publish at the next, and perform no codec
  work inside tokens;
- run exact-source and CQ-promotion arms from both warm and cold page states;
- measure batch time, throughput, deadline completion, exact fallbacks, memory
  peak, and source-page reclamation;
- compare one batch with the rejected one-expert-per-token shape only as a
  negative control; do not run repeated full WRAP after every token;
- inject a worker delay/failure and prove publication defers without narrowing
  selection.

### Step 6: mixed-path provenance and correctness

Use a small forced `A` band while holding `S` identical:

- exact control has `A=empty`;
- each lossy arm forces known experts through a prepared CQ recipe;
- verify every H2D and VRAM hit inherits the expected provenance;
- verify an unprepared approximate expert takes exact fallback and never
  triggers synchronous CQ;
- alternate two recipes and exact mode, including source-map reset and cancel;
- run both existing global-cache and direct selected-load paths;
- confirm that 0052 OFF remains bit-exact to accepted 0051.

Lossy arms are not expected to be bit-exact. Any claim that they are is an
attribution bug unless approximate calls remained zero.

### Step 7: quality ladder

Quality is tested before performance promotion. Use the same accepted
runtime-generated wide selection target and the same hot-set policy in all
arms. Evaluate in this order:

1. exact 0051 control;
2. down-only:CQ1g32 and gate-up-only:CQ1g32 as paired least-size-aggressive
   component arms;
3. all:CQ1g32;
4. g64 component/all arms only if their g32 parent passes;
5. g256 component/all arms only if their g64 parent passes.

Use the repository's coffee, JSON, Python/code, cyberpunk/frontpage, and Roman
history families, including coding -> Roman history and reverse two-turn
switches. For every arm:

- prompt/prefill and the first 50 decode tokens remain exact for the first
  campaign, matching the only existing phase guard;
- run the established long-output, fail-fast L0-L3 protocol;
- run 1 may reject an arm but cannot promote it;
- promotion requires at least three valid measured runs and the exact control
  in the same campaign;
- record approximate expert-call and router-mass fractions so a low-exposure
  arm is not misread as codec safety;
- reject on any additional L0/L1, loop, tag-salad, truncation attributable to
  the arm, or any paired L0-L3 grade loss versus exact control;
- report perplexity/logit or task metrics only where the existing evaluator is
  valid; do not substitute dot nMAE for model quality.

If the exact control fails its own quality floor, the campaign is invalid, not
a pass for the lossy arm.

### Step 8: 64 GiB memory-envelope ramp

With the accepted 24 GiB 0051 arena and exact serving first, ramp the cold cap
through 2, 4, 6, 8, 10, and 12 GiB only while admission floors permit. For
each cap:

- first fill, same-window reuse, disjoint window switch, cancellation, model
  close/reopen, and cleanup;
- record exact cold payload, allocator overhead, actual RSS/PSS, page cache,
  WSL/Windows minima, swap/faults, and cleanup recovery;
- prove source pages do not remain physically duplicated with both arena and
  cold records;
- stop the ramp at the first floor/allocation/cleanup failure;
- do not test a mathematically impossible whole-catalog allocation as a
  serving arm.

Repeat only the accepted cap with the 28 GiB arena after 0051's 28 GiB arm has
independently passed. Arena size and cold cap are separate factors.

### Step 9: controlled end-to-end A/B

Only recipes passing steps 1-8 enter timing:

- one binary and model; identical selection generations, hot-set policy,
  arena, cache, context, prompts, and output budget;
- exact-source cold control versus prepared-CQ cold arm;
- balanced back-to-back order such as exact/CQ/CQ/exact, with at least three
  measured runs per arm after warmup;
- same-domain turn, disjoint coding/history switch in both directions, short
  resumed suffix, and chunked prefill;
- record TTFT, transition WRAP time, first-token latency, decode t/s, p95 token
  latency, exact fallback fraction, H2D bytes, codec bytes, memory minima,
  worker CPU, and thermal/page state;
- separately test background preparation OFF/ON. An ON result must include
  decode contention, not only faster transition time.

A benefit may be stated only if the paired primary metric is repeatable, its
predeclared 95% confidence interval excludes no change in the beneficial
direction, quality does not regress, memory floors hold, and attribution shows
material approximate exposure. Otherwise the result is neutral or negative.

### Step 10: long lifecycle and failure campaign

- repeated 4k-context startup/cleanup and at least 20 alternating turns;
- hundreds of record/slot generations with delayed upload events;
- client cancellation during build, repack, mask prepare, and publish;
- allocation, checksum, worker, event, and deadline fault injection;
- exact fallback under cold-pool exhaustion;
- no leaked workers, records, pinned bytes, device slots, or stale hits;
- OFF-mode regression through the existing 0051 suite.

## 12. Abort and no-promotion criteria

Stop the current run immediately on any of:

- WSL or Windows available memory below 7 GiB, unexpected swap storm, WSL
  restart, CUDA allocation corruption, or inability to identify the server PID;
- source checksum/identity failure, partial triplet READY, stale-generation hit,
  overwrite-before-retire, or an approximate hit in an exact arm;
- `sync_cq_miss_attempts > 0`;
- selection width/mask changes caused by cold-pool pressure or codec failure;
- unreported exact fallback that invalidates arm attribution;
- output degeneration under the repository fail-fast protocol;
- the existing speed rule: an initially slow ramp is allowed, but an
  out-of-scale value around 0.1 t/s or no ramp after about 100 tokens is
  stopped and attributed with tier/codec counters;
- cleanup failure, a worker surviving model teardown, or memory not returning
  within the separately measured allocation-control tolerance.

Do not promote a recipe or policy when:

- only unit tests, a micro-smoke, trace replay, or the offline latency model is
  positive;
- fewer than three valid quality or timing runs exist;
- approximate exposure is too small to distinguish the arm from exact fallback;
- a timing gain is explained by page-cache order, thermals, different selection,
  different arena/cache size, or failed work silently falling back exact;
- mean improves but p95 token latency, TTFT, quality, or memory-floor behavior
  regresses outside the predeclared gate;
- g32 fails and a more aggressive g64/g256 child has not independently overcome
  that failure in the full quality protocol.

Every aborted run preserves request, environment, source/binary/model hashes,
selection and quality generations, codec counters, memory trace, output, logs,
PID identity, and exact stop reason. Process-wide name kills are prohibited.

## 13. Required code ownership after 0051

When implementation begins, keep responsibilities aligned with 0051:

- `ds4_cuda.cu` owns arena/VRAM representation provenance, generation checks,
  H2D attribution, retirement events, and exact fallback ordering.
- `ds4.c` owns interaction mass, `S/H/A/C/R` planning, pageable cold-pool
  allocation, WRAP codec workers, memory admission, and quiescent transaction
  scheduling.
- `ds4_gpu.h` exposes transaction descriptors and representation metadata, not
  raw cold pointers to compute.
- codec code is a separately testable CPU module or tightly scoped helper; it
  does not duplicate GGUF geometry derivation.

0052 must reuse 0051's source geometry, transaction mutex, model identity,
slot leases, snapshot generations, inactive bias publication, and cleanup
ordering. A parallel mask publisher, second arena owner, or raw-offset-only
cold lookup is not acceptable.

## 14. Acceptance criteria

0052 is complete only when all are true:

- 0051 prerequisites in step 0 are closed;
- source authority and exact fallback remain available for every selected
  expert;
- one explicit recipe/provenance follows each approximate byte through cold
  record, arena, H2D, VRAM, token, and output telemetry;
- no codec conversion occurs on the selected-miss path;
- cold work is whole-expert, batched, bounded, joined, and atomically published;
- generation, checksum, model identity, recipe, and complete-triplet checks
  reject stale or mixed copies;
- cold-pool pressure never narrows selection;
- 24 GiB arena plus the measured cold cap respects both host-memory floors and
  cleanup requirements on the 64 GiB machine;
- exact-control mode is bit-exact to accepted 0051 for an identical target;
- at least one lossy recipe passes the full n>=3 quality ladder with material
  exposure, or the experiment is honestly closed negative;
- any speed/memory benefit is supported by the controlled A/B and not by
  projection, requested allocation size, or a smoke;
- 0052 OFF is accepted 0051 behavior.

It is a valid outcome for every CQ1 arm to fail. In that case 0052 documents a
negative result and retains 0051's exact native/pageable design.

## 15. Deliberate non-goals

- Replacing the authoritative GGUF with a CQ-only model.
- Losslessly recovering IQ2_XXS/Q2_K bytes from CQ1.
- Compressing the existing VRAM cache or increasing its slot count.
- Reducing H2D bytes with a CQ-aware GPU kernel.
- Reading pageable or pinned host weights directly in GEMMs.
- Running codec work synchronously on a token miss.
- Keeping all 11,008 CQ records in RAM on the 64 GiB host.
- Treating routing selection, residence, and numerical precision as one state.
- Approving quality from block error, a one-token smoke, or one greedy run.
- Claiming the offline latency model's projections as measured outcomes.
- Supporting concurrent model maps or concurrent CUDA inference sessions in
  v1 beyond 0051's stated topology.

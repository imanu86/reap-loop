# DS4 P5 — long-position KV/indexer residency and reproducible decode capsule

Date: 2026-07-26 (Europe/Rome)  
Scope: read-only design audit; no source edit, DS4 launch, build, or long benchmark  
Source worktree: `C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\wt-hot-reserve`  
Operational plan: `C:\Users\imanu\Documents\Codex\2026-07-25\legg\DS4_OPERATIONAL_PLAN.md`  
Experiment ledger: `C:\Users\imanu\source\repos\reap-loop\docs\EXPERIMENTS_LEDGER.md`

## Decision

The implementation should keep the following primary KV tensors in VRAM when
the CUDA free-memory watermark permits it:

- all 43 raw SWA rings;
- all 20 ratio-128 compressed-attention caches;
- all 21 ratio-4 indexer-compressed caches.

The 21 ratio-4 compressed-attention caches should remain in the exact staged
path. For each ratio-4 layer, only the indexer-selected 512 attention rows are
packed in their original top-k order and transferred to the device. This is the
only proposed host-resident hot cache in the normal P5 policy.

At allocated context 150,000 this retains exactly **462,539,776 bytes**
(441.112305 MiB) of primary KV in VRAM and stages exactly
**1,612,886,016 bytes** (1,538.167969 MiB). At the proposed benchmark capacity
163,840 it retains exactly **504,206,336 bytes** (480.848633 MiB) and stages
exactly **1,761,693,696 bytes** (1,680.082031 MiB).

This policy cannot be implemented by changing only the staged registry
selection. The current attention dispatch enters the staged path only when the
raw cache is host-resident. P5 therefore needs a small mixed-backing path for:

```text
raw ring=device
ratio-4 attention compressed cache=host staged
ratio-4 indexer compressed cache=device
```

It also needs a packed host buffer for the selected 512 rows. The present code
issues one `cudaMemcpyAsync` for every contiguous run in top-k order; since
top-k is score-ordered rather than row-ordered, the inherited estimate of 189
H2D submissions per token is a best-case logical-batch lower bound, not a
guaranteed CUDA submission count.

`READY_FOR_IMPLEMENTATION_SLOT: YES`

Implementation remains gated by the P0 source owner. Physical DS4 runs remain
gated by explicit authorization.

## Provenance and audit boundary

The worktree was already dirty and is owned by P0. It was inspected without
staging, checkout, formatting, build, or write. The observed Git base was:

```text
f64cc89baf5ba890c11317b8a0283e25ace85eae
ds4_cuda: wrap the CUDA allocator to attribute VRAM (diagnostic)
```

Observed source hashes at 2026-07-26T15:38:21+02:00:

```text
ds4.c        54F5E14378B71CC039D1467B5B9D1D7F7512FDC460B1ADC49F24B963C5EC8466
ds4_cuda.cu  A29EBA3B015DB01452B96FCDA3C8342A633995E981B3D92F16838ED582597F82
ds4_gpu.h    625F477594EF2C835047A8FDD2512B8715ADA82C72C71DE1B4466EFBE76D436C
ds4_server.c CD0A13F839775D1FCB51B8FFE41C4B84673BBE04C0E9892C3862BF00D099273E
README.md    DA936B0A1F0DBB4D4C963F0883ABFB4F63A19308340FC228D2F14957199AC184
```

P0 concurrently changed `ds4_cuda.cu` while this read-only audit was being
closed. At 2026-07-26T15:49:10+02:00 its SHA-256 was:

```text
313A8CCC8BCA6B8EC08E3205621B18D8BA7E58624EF20D568063EC86784B90AD
```

The other four hashes above were unchanged. A closeout symbol check found the
P5 KV registry, migration, staged-indexer, staged-attention and decode-ledger
call sites still at the lines recorded below. The concurrent CUDA delta adds
route-mailbox sequence/generation state; the capsule design below already
requires that state to be captured at a quiescent frontier and normalized on
restore. Implementation and physical runs must use a newly frozen post-P0
source hash rather than either audit-time hash.

The prior position-6,602 run is not evidence for position 150,000. Capacity
and live state are separate:

```text
ctx_capacity = graph allocation bound
live_tokens  = ds4_session_pos() = checkpoint token count
next_pos     = live_tokens, the absolute position passed to the next eval
```

For a valid long-position arm, the harness must assert:

```text
ctx_capacity = 163840
live_tokens_before_recovery = 150000
next_decode_position = 150000
```

Merely starting with `--ctx-size 150000` or 163,840 is not a long-position
test.

## Fixed geometry and exact row counts

The code fixes:

```text
layers                43
raw SWA rows/layer    128
raw/attention row     512 float32 = 2,048 bytes
indexer row           128 float32 = 512 bytes
indexer top-k         512 rows
ratio-0 layers        0,1                         = 2 layers
ratio-4 layers        2,4,...,42                  = 21 layers
ratio-128 layers      3,5,...,41                  = 20 layers
```

Allocated compressed capacity is:

```text
cap(layer) = floor(ctx_capacity / ratio) + 2
```

The live completed-row count after `P` committed tokens is:

```text
n_comp(layer) = floor(P / ratio)
raw_live(layer) = min(P, 128)
```

At `P=150000`:

```text
ratio-4 live rows   = 37,500
ratio-128 live rows = 1,171
raw live rows       = 128
```

The `+2` capacity slack is allocated but is not live KV. Compressor frontier
tensors hold the incomplete next group.

## Byte-exact inventory at allocated context 150,000

All byte counts below are logical allocation requests from `ds4.c`; CUDA/WDDM
allocator granularity and managed-allocation padding are intentionally not
folded into them. Actual headroom must come from `cudaMemGetInfo`.

### Per-layer persistent KV tensors

| Layer family | Tensor | Shape in float32 | Bytes/layer | Live bytes at 150,000 |
|---|---|---:|---:|---:|
| all 43 | `layer_raw_cache` | `128 x 512` | 262,144 | 262,144 |
| ratio-4, 21 | `layer_attn_comp_cache` | `37,502 x 512` | 76,804,096 | 76,800,000 |
| ratio-4, 21 | `layer_attn_state_kv` | `8 x 1,024` | 32,768 | 32,768 |
| ratio-4, 21 | `layer_attn_state_score` | `8 x 1,024` | 32,768 | 32,768 |
| ratio-4, 21 | `layer_index_comp_cache` | `37,502 x 128` | 19,201,024 | 19,200,000 |
| ratio-4, 21 | `layer_index_state_kv` | `8 x 256` | 8,192 | 8,192 |
| ratio-4, 21 | `layer_index_state_score` | `8 x 256` | 8,192 | 8,192 |
| ratio-128, 20 | `layer_attn_comp_cache` | `1,173 x 512` | 2,402,304 | 2,398,208 |
| ratio-128, 20 | `layer_attn_state_kv` | `128 x 512` | 262,144 | 262,144 |
| ratio-128, 20 | `layer_attn_state_score` | `128 x 512` | 262,144 | 262,144 |

Per-layer totals, including the ordinary compressor frontiers:

```text
ratio-0 layer       262,144 bytes
ratio-4 layer    96,349,184 bytes
ratio-128 layer   3,188,736 bytes
```

### Aggregate persistent KV

| Category | Tensors | Allocated bytes | MiB |
|---|---:|---:|---:|
| raw rings | 43 | 11,272,192 | 10.750000 |
| ratio-4 attention compressed | 21 | 1,612,886,016 | 1,538.167969 |
| ratio-4 indexer compressed | 21 | 403,221,504 | 384.541992 |
| ratio-128 attention compressed | 20 | 48,046,080 | 45.820312 |
| ordinary attention/index frontiers | 124 | 12,206,080 | 11.640625 |
| primary row-cache total, excluding frontiers | 105 | 2,075,425,792 | 1,979.280273 |
| ordinary persistent KV total | 229 | 2,087,631,872 | 1,990.920898 |

The desired resident primary set is:

```text
raw + ratio-128 attention + ratio-4 index
= 11,272,192 + 48,046,080 + 403,221,504
= 462,539,776 bytes
```

All compressor frontiers remain device-resident in both A and B; they are not
part of the staged row-cache registry.

### MTP-dependent persistent state

When the graph is allocated with MTP enabled, it also creates two extra copies
of every ordinary compressor frontier: speculative and prefix-1.

| Layer family | Extra speculative frontier bytes/layer | Aggregate |
|---|---:|---:|
| ratio-4 | 163,840 | 3,440,640 |
| ratio-128 | 1,048,576 | 20,971,520 |
| global `mtp_raw_cache` | — | 262,144 |

Total MTP KV/frontier overhead is 24,674,304 bytes. The proposed P5 harness
uses `--mtp-draft 1` and must assert `mtp_draft_valid=false`; no active MTP
frontier is semantically required for that protocol. A future capsule that
allows draft count greater than one must serialize `mtp_n_raw`,
`mtp_raw_cache`, the alternating MTP hidden state, draft logits/token, and
speculative frontier state.

### Live payload size at token 150,000

The existing v1 session payload writes live rows rather than capacity. At
exactly 150,000 committed tokens:

```text
live primary row bytes       2,075,236,352
ordinary frontier bytes         12,206,080
tokens (150000 x u32)               600,000
next logits (129280 x f32)           517,120
header + row counters                    396
------------------------------------------------
existing payload bytes       2,088,559,948
```

This is 1,991.805981 MiB. The proposed capsule adds metadata and hashes but
does not embed the multi-gigabyte expert provenance arena.

## Byte-exact inventory at proposed capacity 163,840

The future benchmark should use `ctx_capacity=163840`, which is strictly above
160,000 and leaves 13,568 token positions after the 16 recovery plus 256
measured tokens.

```text
ratio-4 allocated cap   floor(163840/4)   + 2 = 40,962
ratio-128 allocated cap floor(163840/128) + 2 = 1,282
```

| Category | Bytes/layer | Layers | Aggregate bytes |
|---|---:|---:|---:|
| raw | 262,144 | 43 | 11,272,192 |
| ratio-4 attention compressed | 83,890,176 | 21 | 1,761,693,696 |
| ratio-4 indexer compressed | 20,972,544 | 21 | 440,423,424 |
| ratio-128 attention compressed | 2,625,536 | 20 | 52,510,720 |
| ordinary frontiers | varies | 41 | 12,206,080 |

Totals:

```text
primary row caches                2,265,900,032 bytes
ordinary KV including frontiers  2,278,106,112 bytes
desired resident primary set       504,206,336 bytes
desired resident + frontiers        516,412,416 bytes
desired resident + all MTP state    541,086,720 bytes
ratio-4 attention staged          1,761,693,696 bytes
```

The capsule size remains based on live rows and is therefore the same
2,088,559,948-byte v1 core at live token 150,000, apart from its header
recording the larger allocation capacity.

## Non-layer indexer workspaces and staged-ring overhead

With `prefill_cap=250`, these graph tensors are allocated once and live until
`metal_graph_free`; the current decode-start hook does not release them:

| Tensor | Bytes at ctx 150,000 | Bytes at ctx 163,840 |
|---|---:|---:|
| `indexer_scores` | 37,502,000 | 40,962,000 |
| `comp_mask` | 37,502,000 | 40,962,000 |
| `comp_selected` | 512,000 | 512,000 |
| `batch_indexer_q` | 8,192,000 | 8,192,000 |
| `batch_indexer_weights` | 64,000 | 64,000 |
| decode `indexer_q` | 32,768 | 32,768 |
| decode `indexer_weights` | 256 | 256 |
| total | 83,805,024 | 90,725,024 |

The current staged ring allocates:

```text
2 device slots x 4,194,304       8,388,608 bytes
running_max[64] + running_sum[64]      512 bytes
topk_host[512] pinned                  2,048 bytes
```

The exact device request is 8,389,120 bytes, excluding opaque CUDA
stream/event objects. P5 should add one pinned selected-row pack buffer:

```text
512 rows x 2,048 bytes = 1,048,576 bytes
```

This buffer makes one H2D per ratio-4 layer possible while preserving top-k
order exactly.

Current lifetime behavior is noteworthy: at request end the cache is copied
back to device, but the pinned host arena is retained for reuse unless the
path goes sticky-off. With staged-all at context 150,000 that retained host
arena is 2,075,425,792 bytes. With P5 at context 163,840 it is
1,761,693,696 bytes plus the 1,048,576-byte selected pack buffer.

## Current call sites and lifetimes

Key source locations in the observed snapshot:

| Concern | File and observed line | Finding |
|---|---|---|
| compression schedule | `ds4.c:452` | layers 0–1 ratio 0; even layers ratio 4; odd layers ratio 128 |
| graph/session allocation | `ds4.c:10665` | capacity is derived from `ctx_size`, not live position |
| raw allocation | `ds4.c:10741` | all 43 raw rings use the KV allocator |
| attention compressed allocation | `ds4.c:10747` | 41 compressed caches use the KV allocator |
| indexer compressed allocation | `ds4.c:10768` | 21 index caches use the same untyped KV allocator |
| graph free | `ds4.c:10299` | all caches and frontiers live for the graph/session lifetime |
| session structure | `ds4.c:18005` | token checkpoint and logits are engine state; RNG is absent |
| v1 payload save/load | `ds4.c:18421`, `ds4.c:18616` | exact KV rows/frontiers and logits, but no RNG/MoE/provenance |
| server RNG ownership | `ds4_server.c:7410` | RNG is a local variable in the request generation loop |
| decode migration trigger | `ds4_server.c:7415` | called after prefill and before first sample/eval loop |
| request-end restore | `ds4_server.c:7729` | staged caches are restored to device |
| tensor backing | `ds4_cuda.cu:508` | one `ptr`, one `host_ptr`, no cache-kind metadata |
| staged registration | `ds4_cuda.cu:7291` | untyped vector of all KV tensors |
| KV allocator | `ds4_cuda.cu:7388` | all row caches are registered together |
| staged migration | `ds4_cuda.cu:19741` | one pinned arena; all registered device pointers are freed |
| staged indexer | `ds4_cuda.cu:20216` | scans the complete host index cache through 4 MiB slots |
| staged attention | `ds4_cuda.cu:20165` | requires host raw and, when present, host compressed KV |
| indexed attention dispatch | `ds4_cuda.cu:22054` | staged branch is selected only if raw is host-resident |
| decode-start ledger | `ds4_cuda.cu:26922` | migration currently happens before `g_cuda_tmp` release |
| dynamic arena bindings | `ds4_cuda.cu:1607`, `ds4_cuda.cu:1743` | generation, slot, binding and checksum metadata exist |
| arena request barrier | `ds4_cuda.cu:14270` | drains shared G133 route work at request boundary |
| tiering state | `ds4_cuda.cu:26042` | policy decisions depend on per-expert mass/frequency/epochs |
| VRAM expert cache | `ds4_cuda.cu:26207` | slot order, ages, mapping and route generations affect future behavior |

## Transfer accounting at live token 150,000

The inherited staged-all byte estimate is confirmed exactly for the attention
and index reads:

```text
ratio-4 index full scan
  21 x 37,500 x 512                     = 403,200,000 bytes

ratio-4 selected attention
  21 x 512 x 2,048                      =  22,020,096 bytes

all raw attention reads
  43 x 128 x 2,048                      =  11,272,192 bytes

ratio-128 compressed attention
  20 x 1,171 x 2,048                    =  47,964,160 bytes
---------------------------------------------------------------
staged-all total                           484,456,448 bytes/token
```

That is 462.013672 MiB/token.

The often-cited 189 H2D count is obtained only if every logical batch is one
copy:

```text
ratio-4 index:      21 x ceil(37,500 / 8,192) = 105
ratio-4 attention:  21 x (raw + selected)      =  42
ratio-128 attention 20 x (raw + compressed)    =  40
ratio-0 raw:         2 x raw                   =   2
------------------------------------------------------
best-case logical batches                         189
```

Actual `cudaMemcpyAsync` submissions can be higher:

- a wrapped raw ring uses two spans;
- selected ratio-4 rows are copied once per contiguous run in score-ordered
  top-k output;
- a ratio-4 layer can therefore issue between 1 and 512 H2D calls for its
  selected rows with the current gather loop.

The P5 target with packed top-512 and device raw/index/ratio-128 is:

```text
H2D bytes/token        21 x 512 x 2,048 = 22,020,096
H2D submissions/token  21
top-k D2H/token        21 x 512 x 4     =     43,008 bytes
raw ring D2D/token     21 x 128 x 2,048 =  5,505,024 bytes
```

The raw D2D can use the same online-attention ring so reduction order stays:

```text
logical raw rows first, selected compressed rows second
```

No row sorting is allowed. Sorting selected indices could reduce copies but
would change floating-point accumulation order and violate the exact gate.

## Watermark policy

### Watermarks

Use two explicit byte thresholds:

```text
hard_low  = 512 MiB =   536,870,912 bytes
target_high = 1 GiB = 1,073,741,824 bytes
```

Suggested opt-in names:

```text
DS4_CUDA_KV_TIERED_RESIDENCY=1
DS4_CUDA_VRAM_WATERMARK_LOW_MIB=512
DS4_CUDA_VRAM_WATERMARK_HIGH_MIB=1024
```

Invalid values, `high < low`, arithmetic overflow, or `cudaMemGetInfo` failure
must fail closed to exact staged-all and disable optional VRAM growth. Unset
keeps the current behavior.

### Decode-boundary sequence

The safe order is:

1. Block new route/promotion submissions.
2. Drain route worker, upload stream, default stream and staged copy stream.
3. Release `g_cuda_tmp` when configured.
4. Query `cudaMemGetInfo`.
5. Stage all 21 ratio-4 attention compressed caches.
6. Query `cudaMemGetInfo` again.
7. If free VRAM is below `hard_low`, stage additional whole tensors at the
   same quiescent boundary until free VRAM reaches `target_high` or all
   candidates are staged.
8. Publish one immutable per-tensor backing map and unblock decode.

The normal target keeps 84 primary cache tensors on device and stages 21:

```text
43 raw + 20 ratio-128 attention + 21 ratio-4 index = 84 resident
21 ratio-4 attention = 21 staged
```

### Deterministic fallback order

If more VRAM must be freed:

1. Stage ratio-128 attention caches one complete layer at a time.
2. Stage raw caches one complete layer at a time.
3. Stage ratio-4 indexer caches only as the last exact fallback.
4. If the high watermark still cannot be reached, continue with staged-all,
   disable optional expert/model-cache growth, and record
   `watermark_degraded=1`.

Within a tier, use a fixed layer order recorded in the manifest. The policy
must never depend on unordered container iteration.

Ratio-4 indexer is last because at live token 150,000 it saves roughly the
same bytes it retransfers every token: 403,200,000 live bytes, in five 4 MiB
batches per layer. Ratio-128 caches are the least damaging first fallback:
they free about 2.4–2.6 MiB per layer for one compressed batch per token.

### Shared budget with elastic consumers

Residency alone does not guarantee the watermark. The streamed model cache and
expert cache also size from `cudaMemGetInfo`:

- model stream allocator at `ds4_cuda.cu:6836`;
- expert cache sizing at `ds4_cuda.cu:33143`;
- optional Q1 VRAM LRU at `ds4_cuda.cu:32195`;
- prefill wave budgeting at `ds4_cuda.cu:36145`.

All decode-time optional allocators must use:

```text
effective_reserve = max(existing_purpose_reserve, hard_low)
admit(bytes) iff free_bytes >= bytes + effective_reserve
```

The normal model-stream reserve is currently 1,024 MiB in the 47-variable
baseline, but the hot reserve is 256 MiB and the expert-cache reserve is
0.125 GiB. Without the shared floor, either can consume the headroom P5 just
created.

At capacity 163,840 P5 retains 480.848633 MiB more VRAM than staged-all. With
the observed 6.75 MiB/expert geometry, that is enough for 71 complete expert
slots. This is a real eviction/performance tradeoff, not free memory. The
runner must record the actual `per_expert_bytes`, expert-cache capacity,
resident count, stream-cache bytes and eviction deltas for every arm.

No KV migration is allowed in the middle of a token. If external WDDM pressure
pushes free VRAM below the hard floor during decode:

- reject optional promotion/allocation immediately;
- finish the current token through the exact existing path;
- reclassify KV only at the next fully quiescent token boundary;
- mark the arm invalid for performance if headroom ever fell below 512 MiB.

## Exact fallback and aliasing risks

### Mixed-backing dispatch

Current staged attention assumes both raw and compressed caches have
`host_ptr`. P5 must explicitly handle all four combinations:

| Raw | Compressed | Required action |
|---|---|---|
| device | device | existing direct device kernel |
| host | host | existing staged path |
| device | host | new P5 mixed path |
| host | device | exact fallback path or explicit support; never dereference a null pointer |

The target uses `device/host` only for ratio-4 attention.

### Tensor views

`ds4_gpu_tensor_view` copies `ptr` and `host_ptr` at view creation. A view made
before migration becomes stale if the owning tensor is then moved and its
device pointer freed. P5 must either:

- prove and assert that no cache view survives a migration boundary; or
- make views reference the owning backing descriptor rather than copied
  pointers.

The small patch should use the first option and add a debug live-view count.
All cache views must be ephemeral inside a quiescent phase.

### Partial registry state

The current registry has one global `host_resident` bit and restore assumes
every registered tensor has `host_ptr`. Partial residency requires per-entry:

```text
tensor pointer
cache kind
layer
ratio
backing state
host arena offset
requested bytes
```

Migration publication must be atomic. A failed copy, free, or allocation must
restore the old complete backing map before decode resumes. Never publish half
the registry.

### Request-end restore

Restoring 1.68 GiB of ratio-4 attention at request end can itself violate the
watermark. P5 should keep the staged backing across requests until a
multi-token prefill actually requires device caches. Before such a prefill,
quiesce, reclaim optional streamed/expert VRAM, restore the staged tensors, and
then run prefill. A short decode-only continuation can keep the P5 map.

### Pinned-memory pressure

The proposed context-163,840 staged arena is 1,761,693,696 bytes. Pinned RAM is
not VRAM, but it is scarce OS memory. Failure to allocate it must not silently
switch to a pageable asynchronous path. Acceptable outcomes are:

- reuse an existing adequately sized pinned arena;
- use the already verified device path with optional caches disabled;
- fail the benchmark arm before measurement.

### Concurrent MoE activity

Rotator, promotion, route-worker and arena transactions can change expert
state asynchronously. Capsule capture and backing-map publication must require:

```text
route worker drained
upload stream drained
mailboxes FREE
no transient I/O
no arena transaction
all slot reader/writer refs zero
staged append_pending = 0
default stream synchronized
```

## Reproducible decode-state capsule

The existing KVC/session payload is a useful KV checkpoint but is not a
reproducible benchmark capsule.

It currently includes:

- exact checkpoint token IDs;
- next-token logits;
- per-layer compressed/index row counts;
- logical raw SWA rows;
- live compressed attention/index rows;
- attention and indexer compressor frontiers.

It does not include:

- the server-local RNG state;
- sampling/generation parser state;
- active MTP state (it is explicitly invalidated on load);
- dynamic arena generation/bindings;
- expert tiering and G133 decision state;
- VRAM expert-cache slot order/ages;
- source, binary, model, sidecar, environment and arena provenance hashes.

### Container

Use a new benchmark-only container, not KVC v1. Suggested magic and version:

```text
magic   "D4CP"
version 2
endianness little
```

The fixed header contains context capacity, live token count, section count,
total bytes, flags, capsule UUID, manifest SHA-256 and Merkle/root SHA-256.
Each section-table entry contains:

```text
type
flags
file offset
stored bytes
logical bytes
SHA-256
```

Required sections:

1. `MANIFEST`: canonical UTF-8 JSON provenance.
2. `TOKENS_LOGITS`: checkpoint tokens and next-token logits.
3. `RNG_GENERATION`: RNG and full request-generation state.
4. `KV_META`: caps, ratios, row counts and raw logical/physical mapping.
5. `KV_RAW`: logical raw rows.
6. `KV_ATTN_COMP`: live compressed attention rows.
7. `KV_INDEX_COMP`: live compressed index rows.
8. `KV_FRONTIERS`: attention/index state KV and score tensors.
9. `MTP_STATE`: explicit inactive assertion for draft-1, or complete state.
10. `MOE_STATE`: arena, tiering, expert-cache and policy metadata.
11. `CHECKSUM_TREE`: per-layer/per-tensor hashes and the root.

Write to `.partial`, flush file contents and metadata, and atomically rename.
Load performs a streaming validation pass before mutating the session, then a
second pass to restore. A mid-restore failure invalidates the session and
terminates the arm; it never continues from mixed old/new state.

### Manifest provenance

The manifest must bind:

- SHA-256 and byte size for `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`,
  `ds4_server.c`, executable and runner;
- model SHA-256 and exact byte size;
- every sidecar/bake/catalog SHA-256 and size;
- canonical 47/47 base environment variables;
- arm overlay variables and their unset/set distinction;
- backend, GPU UUID, driver, CUDA runtime, Windows build;
- context capacity, prefill chunk, raw cap, live position;
- cache-kind map and watermark decision receipt;
- arena geometry and source offsets;
- sampling parameters represented by raw IEEE/integer bits.

Any provenance mismatch is a hard load refusal. Cross-quant KVC reuse is not
acceptable for the P5 capsule.

### RNG and generation state

`rng` currently lives at `ds4_server.c:7410`, so it must be moved into an
explicit generation-state object or passed to the harness capsule API. Store:

- 64-bit RNG state after the last committed sample;
- temperature, top-k, top-p and min-p bit patterns;
- emitted token count and exact emitted token IDs/bytes;
- last/pending token semantics;
- finish/stop scanner positions;
- thinking state;
- DSML/tool decode tracker;
- any deterministic request sequence used by policy epochs.

The capture boundary is **after one token has been committed/evaluated and
before the next call to `ds4_session_sample`**. At that boundary, checkpoint
tokens and KV rows describe the same prefix and logits describe the next
sample.

### MoE and provenance arena state

Do not embed the entire 30 GiB-class host arena. Store a reconstruction
manifest whose entries contain:

```text
layer, expert, slot
backing source kind
source file SHA-256
source offset and byte length
slot content generation
snapshot generation
slot checksum
slot state/pageable bit
active binding generation
```

Also store all policy fields that influence future routing/residency:

- `g_dynamic_arena.snapshot_generation`, `next_generation`, active bindings;
- dynamic slot metadata and checksums;
- per-expert tier state, frequency, mass, last call, promotion age and G133
  advisory words;
- policy epoch/budget/call tick and request/position epoch;
- router mask/open/closed state;
- VRAM expert-cache slot keys, layer/expert ownership, age/tick, capacity and
  host/device mapping;
- route generation counters in a quiescent equal state.

On restore:

1. Rebuild host-arena slots from the immutable model/sidecars using the saved
   offsets.
2. Verify every slot checksum.
3. Rebuild VRAM expert-cache contents in the same slot order.
4. Restore only behavior-affecting counters.
5. Reset diagnostic counters separately.
6. Set all transport mailboxes to a quiescent state with submitted, consumed
   and completed generations equal to the captured frontier.
7. Verify the complete exported MoE state hash before recovery token 1.

This makes the capsule reproducible without copying tens of GiB into the
capsule while still refusing altered provenance.

## Capsule verification against an uninterrupted run

The first authorized long run is a qualification run, not a performance arm.

### Qualification sequence

1. Start from the frozen binary and exact 47/47 manifest.
2. Allocate `ctx_capacity=163840`.
3. Prefill/advance deterministically to exactly 150,000 committed tokens.
4. Assert all per-layer row counts and compute the pre-capture KV/MoE root.
5. Quiesce and write the capsule.
6. Branch U: resume the same uninterrupted process for 16 recovery plus 256
   comparison tokens.
7. Branch R: start a fresh process, load the capsule, rebuild the arena/expert
   cache, and run the same 16 plus 256 tokens.
8. Compare U and R exactly.

Required exact comparisons for all 272 tokens:

- sampled token ID sequence;
- decoded output bytes and SHA-256;
- full next-logits SHA-256 per token;
- selected expert IDs in order for every routed layer;
- MoE policy transition/event digest;
- final KV root over logical raw rows, live compressed rows and all frontiers;
- final arena/expert-cache state digest.

Full KV hashing is outside measured time. To localize a mismatch without
hashing two GiB per token, maintain an exact rolling append/update ledger and
compute the complete tensor root at the end.

Qualification passes only if all comparisons are bit-exact. A tolerance gate
is not sufficient.

## Future benchmark protocol

### Arms

Use one binary with a strict opt-in:

```text
A = current DS4_CUDA_KV_STAGED_RING=1, P5 opt-in unset
B = DS4_CUDA_KV_STAGED_RING=1 + DS4_CUDA_KV_TIERED_RESIDENCY=1
```

Both arms use:

```text
ctx_capacity=163840
capsule live position=150000
recovery tokens=16, not measured
measured tokens=256
mtp_draft=1
same seed and sampling state
same 47/47 base manifest
```

Each arm starts from a fresh process and the same qualified capsule. Capsule
load/rebuild time and the 16 recovery tokens are excluded from performance
statistics but included in exactness checks.

### Order and repetitions

Use the six-run schedule:

```text
A B A | B A B
```

This gives `n=3` for each arm and balances first/last thermal and cache order.
No arm reuses the prior arm's live process.

### Measurements

For the 256 measured tokens record:

- wall time from QPC around the complete token;
- graph/default-stream CUDA-event time;
- aggregate wall t/s and graph t/s;
- wall and graph p50/p95/p99 token latency;
- H2D, D2H and D2D submissions and bytes;
- index-score, top-k, staged-attention and expert-route kernel counts;
- SSD expert reads, RAM hits, VRAM hits and evictions;
- expert-cache capacity/count and stream-cache bytes;
- `cudaMemGetInfo` free bytes at decode start and every token boundary;
- minimum and p50 free-memory headroom;
- exact correctness digests.

Report both:

```text
throughput = 256 / sum(measured token wall seconds)
latency-equivalent t/s = 1000 / percentile_latency_ms
```

Do not substitute one for the other.

### Hard gates

An arm is invalid if any of these fail:

- source/binary/model/sidecar/runner provenance;
- 47/47 base manifest or declared overlay;
- live position exactly 150,000 before recovery;
- context capacity strictly above 160,000;
- 16 recovery and 256 measured token counts;
- output/token/logits/expert/KV exactness;
- minimum VRAM headroom 512 MiB;
- expected P5 backing map for arm B;
- clean native shutdown, zero DS4/Nsight processes and free port after run.

Arm B is not the target result if the watermark forced ratio-4 indexer or the
entire ratio-128/raw set to staged. Such a run is a valid correctness fallback
receipt, but its performance is reported as `B_FALLBACK`, not B.

## Small patch sequence

No patch is applied in this audit. The recommended implementation order is:

### P5a — typed cache inventory

Files:

```text
ds4_gpu.h
ds4.c
ds4_cuda.cu
```

Add a narrow cache-kind enum and tagged allocator:

```text
RAW
ATTN_RATIO4
ATTN_RATIO128
INDEX_RATIO4
```

Tag the three allocation call sites at observed `ds4.c:10741`, `10747` and
`10768`. Replace the registry vector of bare pointers with typed entries.
Add a byte-exact startup receipt by kind/layer/capacity.

### P5b — partial migration and watermark

Refactor migration/restore to publish an immutable per-entry backing map.
Release backend prefill scratch before the first policy query. Implement
low/high watermarks and connect the hard-low floor to optional decode-time
model/expert allocators. Preserve unset behavior exactly.

### P5c — exact mixed ratio-4 attention

Add:

- raw-device to ring D2D in logical row order;
- pinned 1 MiB selected-row pack in unchanged top-k order;
- one H2D of the packed 512 rows per ratio-4 layer;
- the existing online attention reduction over raw first, selected compressed
  second;
- explicit support/fail-closed checks for every backing combination.

Static and synthetic tests must prove that no staged tensor is dereferenced
through null `ptr` and no resident tensor through null `host_ptr`.

### P5d — capsule core

Files:

```text
ds4.h
ds4.c
ds4_gpu.h
ds4_cuda.cu
new benchmark harness under tools/tests
```

Add D4CP v2 sectioned serialization, RNG/generation state, streaming hashes,
quiescent CUDA capture barrier, and exact KV export/import. For the first slot,
support only `mtp_draft=1` and fail if an MTP draft is active.

### P5e — MoE/arena export-import

Export/import dynamic-arena bindings, checksums, tiering entries, expert-cache
slot order and route-generation frontier. Reconstruct expert bytes from frozen
model/sidecar provenance rather than embedding the arena.

### P5f — runner and qualification

Add a PowerShell runner that:

- clears inherited `DS4_*`;
- enforces the exact 47/47 base manifest and one declared arm overlay;
- validates hashes and process/port/GPU preflight;
- qualifies the capsule U-versus-R;
- runs `A B A | B A B`;
- emits per-run JSON/CSV, correctness roots and a final receipt;
- performs authenticated native shutdown and postflight.

## Implementation-slot acceptance checklist

Before authorizing the long physical run:

- [ ] P0 source owner grants the implementation slot.
- [ ] Source hash is frozen after P5a–P5f.
- [ ] Static tests cover cache classification, byte totals and fallback order.
- [ ] Mixed-backing tests cover raw device + ratio-4 compressed host.
- [ ] Packed top-512 preserves original index order exactly.
- [ ] `cudaMemGetInfo` failure is fail-closed.
- [ ] No mid-token migration exists.
- [ ] Capsule capture proves a fully quiescent MoE/arena frontier.
- [ ] Capsule parser validates all hashes before mutation.
- [ ] Fresh-process capsule qualification is bit-exact.
- [ ] Runner proves live position, not only context capacity.
- [ ] GPU headroom gate is 512 MiB hard / 1 GiB target.
- [ ] No source edit or physical run overlaps the P0 worktree owner.

## Final status

```text
READ_ONLY_AUDIT=COMPLETE
SOURCE_MODIFIED=NO
DS4_STARTED=NO
LONG_BENCHMARK_RUN=NO
CTX_ALLOCATED_VS_LIVE_POSITION=DISTINGUISHED
BYTE_EXACT_INVENTORY=COMPLETE
WATERMARK_POLICY=DESIGNED
EXACT_STAGED_FALLBACK=DESIGNED
CAPSULE_PROTOCOL=DESIGNED
ABA_BAB_PROTOCOL=DESIGNED
READY_FOR_IMPLEMENTATION_SLOT=YES
BLOCKED_ONLY_BY=P0_SOURCE_SLOT_AND_PHYSICAL_RUN_AUTHORIZATION
```

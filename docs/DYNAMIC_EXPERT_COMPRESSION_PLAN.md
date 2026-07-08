# Dynamic Expert Compression Plan

Status: Step 0 observe-only is implemented in DS4 commit `94e9a7d`
(`cuda: add expert tiering observe mode`). Step 0.1 ID-bearing observe traces
are implemented in DS4 commit `4de3131` (`tiering: optionally log selected
expert ids`) behind `DS4_EXPERT_TIERING_LOG_IDS=1`. Compression, sidecars,
demotion, and lossy cold formats are still design/open work. Do not treat any
speed, RAM, or quality benefit below as a claim until the local/pod tests in
this document pass.

## Goal

Build dynamic expert compression for DS4/reap-loop without a static domain mask:
hot experts stay resident in their fast form, warm experts stay cheap to reload
from RAM/page cache, cold experts are stored in a more aggressive compressed
form, and frozen/coldest experts may be served from SSD. REAP/PACE remains the
controller for live expert relevance; compression is a tiering layer under it.

Target hardware: RTX 3060 12GB under WSL with about 62GB RAM. This is a
capacity- and I/O-bound target, so the design optimizes for controlled fallback
and measurement first, not for assumed throughput gains.

Updated local target after J17: dynamic compression should be measured as
**effective resident-capacity expansion**, not just as smaller cold files. On a
160-token local HTML trace, simulated global LRU hit-rate over compact expert
IDs was 0.3396 at cap 258, 0.5927 at cap 512, and 0.7438 at cap 1024. The
practical goal is therefore to make the 3060 behave closer to cap 512+ without
allocating uncompressed 512+ resident expert slots.

## Current DS4 Facts To Preserve

From the DS4 source inspected on `/root/ds4`:

- CUDA streaming has a global resident expert cache:
  `cuda_stream_expert_cache_slot` and `cuda_stream_expert_cache` keep
  `(model_map, layer, expert, offsets, bytes, age)` plus gate/up/down device
  slabs (`ds4_cuda.cu:249`, `ds4_cuda.cu:264`).
- The resident cache budget is driven by
  `DS4_CUDA_STREAMING_EXPERT_CACHE_N`, `--ssd-streaming-cache-experts`, and
  `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB`; on a 12GB GPU the reserve can
  disable or cap the cache (`ds4_cuda.cu:1551`, `ds4_cuda.cu:1591`,
  `ds4_cuda.cu:1607`).
- A selected expert batch first looks in the global resident cache, then falls
  back to direct streamed copy from the mapped GGUF into the selected compact
  buffers (`ds4_cuda.cu:3094`, `ds4_cuda.cu:3188`, `ds4_cuda.cu:3250`).
- Eviction is currently LRU by `age`; no cold-format payload is attached to a
  slot (`ds4_cuda.cu:2118`, `ds4_cuda.cu:2162`).
- Hotlist/prefill seeding can populate the resident cache before demand
  (`ds4_cuda.cu:3458`, `ds4.c:21469`). This is a useful insertion point for
  promoted/decompressed experts.
- REAP/PACE changes the router bias in memory and preserves expert IDs; it does
  not shrink or recompress expert tensors (`ds4.c:7386`, `ds4.c:7478`).
- WRAP/fattorino touches the kept working set into host page cache after mask
  changes; it is host-side and can contend with SSD I/O if overused
  (`ds4.c:12473`, `ds4.c:12642`).
- MTP and `--ssd-streaming` are not currently compatible (`ds4.c:27471`).

From `CLAIMS_CURRENT.md` and `PACE_DESIGN.md`:

- Current validated value is live/session calibration and adaptive coverage,
  not static domain masking.
- Dynamic staircase/PACE variants that churn the cache are retracted; any new
  tiering must avoid frequent global churn.
- SPEX-style prefetch only helps measured deeply SSD-bound regimes; it can slow
  the practical 3060 regime. Dynamic compression must therefore be proven with
  timed A/B tests.

## Tier Model

Each routed expert has a runtime descriptor:

```
(layer, expert_id, tier, format, location, generation, last_used, score)
```

`expert_id` is always the original DS4 ID. No physical pruning or remapping is
required for the MVP.

| Tier | Location | Format | Intended state | Promotion path |
|---|---|---|---|---|
| hot | VRAM resident cache | native DS4 routed format, e.g. IQ2_XXS/Q2_K or Q4 variant | selected recently, inside PACE keep set, or prefetched for next token | direct cache hit |
| warm | RAM/page cache | native DS4 slabs, optionally pinned/touched by WRAP | likely soon, not worth recompressing | copy to VRAM selected cache |
| cold | RAM, compressed sidecar | aggressive low-bpw copy plus small metadata | evicted from hot/warm but plausible future recall | decompress or repack to native selected/cache slot |
| frozen | SSD sidecar or original GGUF | coldest/low score; no RAM budget | rare recall, safety fallback | read from SSD, then decompress/repack |

Tier movement is event-driven:

- promote to hot on selection, PACE prefill application, or explicit prefetch;
- demote hot to warm/cold only when it leaves the dynamic keep set for a
  grace window or resident cache pressure forces eviction;
- demote cold to frozen only under RAM pressure or long inactivity;
- never destroy the original GGUF path, so every compressed miss has a correct
  fallback.

## Candidate Formats

The safest first target is not "better quality"; it is "less RAM/page-cache
footprint for experts unlikely to be used soon, with exact fallback available."

| Format | Use | Notes |
|---|---|---|
| native pack | hot/warm baseline | Current DS4 slabs copied unchanged. This is the correctness fallback. |
| Q4 pack | warm/cold for models or layers where native is higher bpw | Useful if testing Q4/fuller variants. For current Flash routed experts already near 2-bit, Q4 is not a compression win. |
| Q2 pack | cold baseline for higher-bpw source, or native Flash mirror | Flash already uses IQ2_XXS/Q2_K routed experts, so "Q2 cold" may mostly mean sidecar repack/indexing, not smaller tensors. |
| Q1/1-bit pack | experimental cold/frozen | Only behind flags. Requires quality tests because router-selected expert outputs may degrade sharply. |
| sparse/delta pack | cold/frozen research | Possible later: store residual from native/Q2 or per-channel scale deltas. Not MVP. |

For the current Flash IQ2/Q2 model, the MVP should assume the original GGUF
remains the authoritative representation. If a new cold format cannot beat the
native routed bytes after metadata and decompression cost, the experiment should
record that negative result and stop.

## Timing With REAP, PACE, And SPEX

PACE owns relevance; compression owns representation:

1. During prefill/warmup, PACE observes routing with no compression side effects.
2. When PACE applies the first live mask/coverage decision, the tier manager
   marks experts in the keep set as hot or warm candidates.
3. WRAP may page-in the kept set only if enabled and measured useful. Dynamic
   compression should not automatically enable WRAP on 3060.
4. On each selected expert load:
   - if hot, use the existing resident cache;
   - if warm, copy native slab from RAM/page cache;
   - if cold, decompress/repack into a native selected slot or resident slot;
   - if frozen, read the sidecar/original GGUF, then promote through cold/warm.
5. When PACE breathes or widens coverage, do not synchronously decompress the
   entire widened set. Promote only on actual selection or bounded prefetch.
6. When PACE tightens coverage, do not immediately compress every excluded
   expert. Queue demotion after a grace period and cancel it if the expert is
   selected again.
7. SPEX prefetch may request promotion, but it must be budgeted separately from
   PACE's selected experts. A wrong prefetch is a latency/capacity cost, not a
   correctness change.

The key invariant: router decisions remain DS4/PACE decisions. Compression must
never alter top-k selection semantics.

## Proposed Interfaces

Use env flags so the stock behavior remains the default.

| Env / option | Default | Meaning |
|---|---|---|
| `DS4_EXPERT_TIERING` | `0` | Master enable. Only `observe` is implemented today; off means current DS4 behavior. |
| `DS4_EXPERT_TIERING_LOG` | unset | Implemented optional JSONL event log for current selected-load/cache observations. |
| `DS4_EXPERT_TIERING_LOG_IDS` | `0` | Implemented optional ID trace: appends `selected` and `compact_ids` arrays to observe JSONL rows for offline policy simulation. |
| `DS4_EXPERT_TIERING_SUMMARY_EVERY` | `0` | Implemented optional stderr summary interval by observed batches. |
| `DS4_EXPERT_TIER_POLICY` | `observe` | Planned future policy: `observe`, `warm-only`, `cold-sidecar`, `frozen-ssd`. |
| `DS4_EXPERT_COLD_FORMAT` | `native` | `native`, `q4`, `q2`, `q1`, `pack`. MVP starts with `native`/metadata only. |
| `DS4_EXPERT_COLD_RAM_MB` | `0` | RAM budget for cold sidecar cache; 0 disables RAM cold cache. |
| `DS4_EXPERT_FROZEN_DIR` | unset | Directory for SSD sidecar packs. Unset disables frozen tier. |
| `DS4_EXPERT_DEMOTE_GRACE_TOKENS` | `256` | Tokens after leaving hot/keep before demotion is eligible. |
| `DS4_EXPERT_PROMOTE_MAX_PER_TOKEN` | `6` | Bound decompressions/promotions per token to avoid stalls. |
| `DS4_EXPERT_DECOMPRESS_THREADS` | `1` | Background CPU workers for cold -> warm/native repack. |
| `DS4_EXPERT_VERIFY_NATIVE_FALLBACK` | `1` | On failure or checksum mismatch, reload native GGUF slab. |
| `DS4_EXPERT_TIER_STATS_EVERY` | `0` | Optional stderr interval; keep off during timed runs. |

Possible internal APIs:

```
ds4_expert_tier_note_selected(layer, ids, n, token)
ds4_expert_tier_note_pace_keep(layer, keep_bitmap, generation)
ds4_expert_tier_promote_async(table, expert_id, reason)
ds4_expert_tier_get_native_or_decompress(table, expert_id, dst_gate, dst_up, dst_down)
ds4_expert_tier_demote_queue(layer, expert_id, target_tier)
ds4_expert_tier_stats_snapshot(out)
```

Integration points:

- update or wrap the resident cache miss path around
  `cuda_stream_expert_cache_load_slot`;
- add metadata to or alongside `cuda_stream_expert_cache_slot`;
- feed PACE keep/mask generation after `ds4_reap_mask_apply`;
- optionally let hotlist/prefill seeding call `promote_async` for warm/cold
  candidates.

## MVP: Non-Destructive First

The MVP should not change model files, kernels, router logic, or quant kernels.

### Step 0: observe only

Add only counters/logging around current cache hits, misses, direct loads, slot
evictions, and PACE keep membership. Compute the hypothetical tier for each
expert but always load native DS4 slabs. Gate:

- output byte-identical or token-identical to baseline for deterministic runs;
- no measurable slowdown with logging disabled;
- JSONL contains enough data to replay tier decisions offline.

Implementation status as of 2026-07-08:

- DS4 commit `94e9a7d` adds CUDA observe-only behind
  `DS4_EXPERT_TIERING=observe`.
- DS4 commit `4de3131` adds optional ID-bearing JSONL rows behind
  `DS4_EXPERT_TIERING_LOG_IDS=1`. Default remains off because the arrays make
  logs heavier.
- JSONL events include `time_unix`, `path` (`resident` or
  `selected_direct`), layer, slot/compact counts, cache capacity, before/after
  resident count, hits, misses, direct loads, evictions, and byte estimates.
  With `DS4_EXPERT_TIERING_LOG_IDS=1`, rows also include selected expert IDs
  and the compact working-set IDs used by that batch.
- The local launcher currently defaults to
  `DS4_EXPERT_TIERING=observe` and writes
  `/root/ds4_tiering_observe.jsonl`.
- First smoke on RTX 3060 produced: 473 events; 12 resident-cache events at
  cap 222; 461 selected-direct events; direct loads 2766; hits 0; misses 72;
  evictions 0. This is diagnostic only, not a throughput claim.
- `scripts/analyze_tiering_observe.py` summarizes one or more observe JSONL
  files into path shares, cache capacities, hit/miss/direct totals, byte totals,
  and worst layers. With ID traces, `--simulate-cap` replays a global LRU over
  `(layer, expert)` keys.
- As of J22 the analyzer also accepts `--slot-mib` and `--capacity-scale`, so
  each simulated cap reports the native and compressed memory cost. This is the
  offline sizing bridge for "effective cap512/cap1024" dynamic compression.
- As of J25, `--target-hit-rate` reports the first simulated cap that reaches a
  desired hit target. On the first HTML160 ID trace, target 0.60 needed cap1024
  among the tested caps, while target 0.75 needed cap2048.
- As of J29, the analyzer also accepts historical `DS4_SPEX_TRACE_ROUTING` CSV
  files and `.tgz` archives containing routing CSVs. This means the large
  existing session corpus can be mined for hot/cold policy replay without new
  pod time. These legacy traces cannot report resident-cache versus
  selected-direct runtime paths, but they are valid for `(layer, expert)` reuse,
  LRU, warm-grace, cold-recall, and frozen-recall simulations.
- First historical replay checks were consistent across three old corpora:
  K91 coding trace cap258/cap512/cap1024 LRU hit-rate
  `0.4003/0.5167/0.6611`; product trace `0.4065/0.5250/0.6676`;
  domain trace `0.3927/0.5075/0.6500`. With `warm_grace=64` and
  `freeze_after=512`, served hot+warm rates were roughly
  `0.48-0.50`, `0.57-0.59`, and `0.68-0.69` for the same caps. This confirms
  that old sessions are enough to estimate the hot/cold shape; new observe-ID
  runs are needed only for runtime cache-path/timing validation.
- First local runtime observe-ID smoke (J30) used a tiny `Rispondi solo OK`
  prompt. Each run produced 129 observe events with 86 resident and 43
  selected-direct batches; resident hit-rate was only `0.2248`, direct loads
  were 2351, and direct bytes were 15.87 GiB. Because the prompt is tiny and
  K0/prefill-heavy, the compact-ID LRU rates were not representative
  (cap258/cap512/cap1024: `0.0405/0.0586/0.0907`). Treat this as a runtime path
  smoke, not a policy-quality trace.
- First representative ID trace (J17, HTML160) produced 6923 events and 5653
  unique compact `(layer, expert)` pairs. LRU sim: cap64 0.0000, cap128 0.0000,
  cap258 0.3396, cap512 0.5927, cap1024 0.7438. This is not a speed claim; it
  is a capacity-pressure estimate for the next runtime policy.
- Important implication: the next engineering step is not lossy compression
  yet. First split/measure the selected-direct path versus resident-cache path
  on longer prompts, then decide where cold compression can actually remove
  bytes from the bottleneck.

Local launcher follow-up (J11/J21): hidden-readback SPEX was disabled again
because it made TTFT unusable. GPU-side hidden score/topK now exists behind
`DS4_SPEX_HIDDEN_GPU_SCORE=1`, but it still does not feed residency/prefetch.
The practical 3060 launcher uses `DS4_PACE_PREFILL_APPLY=1` and
`DS4_PACE_PREFILL_WAIT_WRAP=0`. This cuts visible first-token latency versus
waiting for WRAP, but observe logs still show selected-direct traffic and zero
resident hit-rate on short repeated requests. Do not start lossy cold-format work
until this direct/resident split is understood on a real prompt batch.

K0 warmup follow-up (J12): applying the prompt-learned mask before token 1 is
faster on short tests, but it violates the quality hypothesis that the first
generated tokens should be unpruned. `DS4_PACE_PREFILL_APPLY=0` preserves K0
through token 50 and only cuts at token 51, but was slower in the local HTML
sample. The desired runtime split is a new mode: prompt-derived async prefetch
without prompt-derived mask application.

### Step 1: warm/frozen metadata simulation

Still no compressed payload. Treat cold/frozen as labels and measure how often
an expert would require decompression or SSD recall. Gate:

- predicted cold recalls per 1k tokens is low enough to justify real work;
- demotion grace reduces churn compared with immediate LRU demotion;
- PACE breath/tighten does not create mass demotion spikes.

### Step 2: native sidecar round-trip

Write/read a sidecar pack that stores exact native gate/up/down slabs plus
checksum. This proves addressing, metadata, fallback, and threading without
lossy quantization. Gate:

- checksums match native source;
- native sidecar reload equals GGUF reload output;
- fallback to GGUF works when sidecar is missing/corrupt.

### Step 3: one experimental cold format

Add one lossy cold format behind `DS4_EXPERT_COLD_FORMAT`, preferably for a
small layer/expert subset first. Gate:

- functional eval remains at the same L-grade on selected tasks;
- per-token decompression latency is bounded and reported;
- RAM saved is measured after metadata/alignment, not theoretical.

### Step 4: frozen SSD tier

Only after Step 3 proves useful: evict cold packs to `DS4_EXPERT_FROZEN_DIR`,
with bounded background read and native fallback. Gate:

- no hot-path unbounded SSD scan;
- recall latency distribution is reported;
- frozen tier can be disabled at runtime without changing model semantics.

## Metrics

Report all metrics with config, commit, model quant, prompt set, and whether the
run was cold or warm. Do not summarize as a win unless the confidence interval
supports it.

Mechanistic metrics:

- resident cache hit/miss/direct-load counts;
- hot/warm/cold/frozen hit counts;
- promotions/demotions per 1k tokens;
- decompression ms per expert and per token;
- SSD read ms and bytes per frozen recall;
- cache churn: evicted then recalled within N tokens;
- VRAM resident slots and GiB;
- RAM RSS, page-cache touched GiB, cold sidecar GiB;
- PACE state: keep-K/coverage, breaths, mask generations.

Outcome metrics:

- t/s segmented: TTFT, tokens 1-64, 65-256, 257+;
- completion wall time on the graded tasks;
- functional L0-L3 grade, plus exact output checks where deterministic;
- repeat/ngram degeneration and any PACE quality sensors already used;
- A/B versus current DS4 streaming with the same cache budget and flags.

## Local 3060 Test Plan

Use WSL local first because this is the target bottleneck.

Baseline matrix:

1. `--ssd-streaming`, explicit cache budget, reserve set low enough that the
   expert cache is enabled.
2. Current PACE/session-learning profile with tiering off.
3. Tier `observe` with logging disabled during timed sections.
4. Tier `observe` with JSONL on for one untimed diagnostic run.
5. Step 2 native sidecar round-trip on a short deterministic prompt.

Suggested gates:

- deterministic output unchanged for Step 0/1/2;
- no extra timed-run logging on SSD hot path;
- report p50/p95 token latency, not only average t/s;
- run warm first, discard cold-cache startup when measuring warm behavior;
- separately run a cold-cache stress test if testing frozen SSD.

## Pod Test Plan

Use the pod for expensive sweeps and quality gates:

- multiple seeds/prompts for JSON, Python, and frontpage-style tasks;
- cold-start SSD-bound case where prefetch/tiering is most likely to matter;
- lossy-format subset tests before all-layer enablement;
- corruption/fallback tests by deleting or modifying sidecar entries;
- sweep demotion grace, cold RAM budget, and promotion limit;
- compare against random tiering admission as a control.

The pod should produce artifacts comparable to the current ledgers: config,
commit, run logs, JSONL tier events, stats summary, and graded outputs.

## Risks

- Current Flash routed experts are already IQ2_XXS/Q2_K. Further compression may
  save little or damage quality; measure before broad implementation.
- Decompression can move the bottleneck from SSD to CPU, PCIe, or sync points.
- Frequent PACE breaths/tightens can cause tier churn if demotion is immediate.
- WRAP, SPEX prefetch, and cold decompression can all compete for the same SSD
  and RAM bandwidth.
- The CUDA path has blocking sync constraints; async promotion needs explicit
  event ordering before it can touch buffers consumed by kernels.
- WSL page cache and `mlock` behavior are noisy; RSS is not enough to prove RAM
  savings.
- Sidecar formats need checksums and generation IDs, or stale/corrupt packs will
  create hard-to-debug quality failures.
- Hotlist/static frequency admission can fight live PACE decisions. PACE/live
  relevance must win.
- Frozen SSD recall can create p95/p99 latency cliffs even if average t/s looks
  fine.

## Code Steps After This Doc

1. ~~Add observe-only tier stats in DS4 behind `DS4_EXPERT_TIERING=observe`.~~
   Done in `/root/ds4` commit `94e9a7d`.
2. ~~Add a replay script that consumes `DS4_EXPERT_TIERING_LOG` and estimates
   compression opportunities/churn offline.~~ `scripts/analyze_tiering_observe.py`
   now accepts tiering JSONL, routing CSV, and routing `.tgz`; it reports LRU
   capacity, target hit-rate, and a metadata-only hot/warm/cold/frozen policy
   replay. Runtime path timing still requires real observe JSONL.
3. Add exact native sidecar pack/unpack with checksum and GGUF fallback.
4. Wire cold miss promotion into the existing cache miss path.
5. Add one lossy cold format for a tiny opt-in subset, then run pod quality
   gates before wider enablement.

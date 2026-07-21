# REAP revisited: exact resident base plus CPU-GEMV tail

Date: 2026-07-21  
Machine/scope: the measured Windows host; CPU analysis only; no GPU or server was run for this study.

## Executive verdict

**GO for a bounded prototype, with two gates before full integration.** Use **K = 5,600** as the first target, store the resident arena in pageable RAM, retain only a six-expert pinned transfer slab, and do not create a duplicate 320-expert host mirror. First restore/recapture the route trace and then prove one-layer asynchronous CPU/GPU join behavior. Those two gates are important because the requested K-specific coverage curve cannot be measured from the files currently present, and per-token arithmetic alone overstates overlap across the 43 sequential layer barriers.

Conditional on the proposed tail being **5–15% of 258 routes/token**, K=5,600 projects to:

| Scheduling bound | Current plumbing | With F1/F2 |
|---|---:|---:|
| Serial CPU tail, 5–15% fallback | **6.05–5.05 t/s** | **10.89–8.03 t/s** |
| Fully hidden CPU tail (optimistic ceiling) | **6.72 t/s** | **13.25 t/s** |

These are sensitivity bounds, **not replay-derived K=5,600 predictions**, because the route rows are missing. The design still merits a prototype: it restores exact weights, K=5,600 fits the stated 38–42 GiB package budget, and even a 15% fully serial tail remains above the broken-Q1 4.09 t/s baseline. The likely make-or-break issue is not GEMV speed but whether activation D2H, CPU work, output H2D, and the mandatory per-layer join can actually overlap without starving the CUDA submission/route-worker threads.

## 1. Evidence and route-trace integrity

The requested input was `C:\Users\imanu\g130i\u1_$Tag.server.stderr.log`, expected to contain 16,512 `[q1-0-mixed-route]` rows. At analysis time:

- `u1_$Tag.server.stderr.log` contained **0** route rows and only failed-startup/zero-call summaries. It was 1,853 bytes at the final scan and had been rewritten during the analysis window.
- `u1_attrib.server.stderr.log` also contained **0** route rows (`trace_rows=0`). It does corroborate a 64-token decode: `calls=2752`, `slots=16512`, because 64 × 43 layers × 6 routes = 16,512.
- The attribution summary reports the old policy's aggregate route counts: 3,271 exact-IQ2 routes and 13,241 Q1 routes (19.810%/80.190%). This is **not** a mass-ranked keep-K curve: its resident population came from prompt snapshot, VRAM seed, and dynamic policy, so it cannot substitute for the absent expert IDs and weights.

Accordingly, [coverage_curve.csv](coverage_curve.csv) and [coverage_curve.json](coverage_curve.json) contain explicit `missing_route_rows` records and null coverage/latency fields instead of invented numbers.

### Coverage calculation once the trace is restored

For each `(layer, expert)`, aggregate:

- count score: number of selected-route occurrences;
- mass score: sum of logged `weight` values.

Rank by mass descending, with `(layer, expert)` as a deterministic tie-break. For the floor variants, reserve the top 16, 32, or 64 mass-ranked experts in every layer, then fill the remaining global K budget by mass among all unreserved experts. Coverage is:

`count_coverage = resident route rows / all route rows`

`mass_coverage = resident gate-weight sum / all gate-weight sum`

Fallback count and mass are their complements. Ratios remain valid even if the logged router weights have a constant scale.

The floor choices in the machine-readable artifacts are 16, 32, and 64 experts/layer. At K=3,000 the fixed reservations are respectively 688, 1,376, and 2,752 entries; all four requested K values are feasible for all three floors.

### Sampling risk

One 64-token cyberpunk continuation has only 64 independent token clusters. The 16,512 rows are correlated within tokens, layers, and route sets, and ranking 11,008 possible layer/expert pairs from one topic will overfit both the prompt and the early decode state. The danger is largest near the K cutoff and for layer floors: a route that is absent in 64 tokens is not globally cold.

A reasonable minimum is **20 diverse prompts × 256 decoded tokens = 5,120 tokens = 1,320,960 route rows**. A firmer sizing run is **50 prompts × 512 tokens = 25,600 tokens = 6,604,800 rows**, stratified across code, prose, dialogue, multilingual, factual, and long-context continuations. Rank on a training subset, report coverage on held-out prompts, and bootstrap by whole prompt (not individual route row) for 95% intervals. Do not freeze the production manifest until the K=5,000/5,600 fallback-count and fallback-mass intervals are narrow enough to make the latency decision (roughly ±1 percentage point is a useful target).

## 2. Latency model

### Baseline accounting

The supplied spans are:

| Span | ms/token |
|---|---:|
| mixed_q1_call | 95.8 |
| selection_d2h | 65.9 |
| h2d | 32.5 |
| sync | 11.1 |
| hot | 10.7 |
| kernel | 4.2 |
| other | ~24.0 |
| Span sum | 244.2 |
| Measured baseline | 244.7 (4.09 t/s) |

The model is anchored to the measured 244.7 ms/token; the 0.5 ms difference is rounding/residual. Retiring Q1 removes the 95.8 ms Q1-serving term but conservatively keeps selection, H2D, synchronization, hot-IQ2 compute, kernel, and other costs:

- current-plumbing fixed base = `244.7 - 95.8 = 148.9 ms/token`;
- F1/F2 base = `148.9 - (65.9 - 10) - (32.5 - 15) = 75.5 ms/token`.

For replay fallback count `F` across 64 tokens:

- fallback experts/token = `F / 64`;
- serial CPU tail = `(F / 64) × 1.267 ms`;
- serial latency = `fixed_base + CPU tail`;
- fully overlapped lower bound = `max(fixed_base, CPU tail)`.

The fully overlapped formula is only a ceiling on throughput. A routed result is needed before the layer's HC post and the next layer, so CPU work cannot simply float underneath the entire token's GPU time.

### Sensitivity while K coverage is unavailable

There are 43 × 6 = 258 routed experts/token.

| Fallback share | Fallback experts/token | CPU tail, serial ms | Current serial t/s | Current overlap ceiling t/s | F1/F2 serial t/s | F1/F2 overlap ceiling t/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.0 | 0.0000 | 6.7159 | 6.7159 | 13.2450 | 13.2450 |
| 5% | 12.9 | 16.3443 | 6.0516 | 6.7159 | 10.8880 | 13.2450 |
| 10% | 25.8 | 32.6886 | 5.5070 | 6.7159 | 9.2431 | 13.2450 |
| 15% | 38.7 | 49.0329 | 5.0522 | 6.7159 | 8.0300 | 13.2450 |

At each requested K (3,000/4,000/5,000/5,600), the CSV/JSON applies these formulas once fallback count is known. No K-specific t/s value is defensible from the currently available logs.

The 1.267 ms/expert figure is a serial upper-cost model, not necessarily the best multi-expert implementation. The existing CPU primitive parallelizes rows across the shared 8-thread pool, so processing several fallback experts as one layer batch may be cheaper than `N × 1.267`; conversely, CPU contention and transfer/join overhead may erase overlap.

## 3. Exact memory map

### Expert size, measured from the model layout

The exact routed-expert tensor total is 72.5625 GiB for 43 × 256 = 11,008 experts:

`77,913,391,104 bytes / 11,008 = 7,077,888 bytes/expert`

This agrees with the tensor geometry: gate 2,162,688 + up 2,162,688 + down 2,752,512 = 7,077,888 bytes. It is therefore safe to use 7,077,888, not a rounded MiB estimate.

| K | Resident arena bytes | GiB |
|---:|---:|---:|
| 3,000 | 21,233,664,000 | 19.775390625 |
| 4,000 | 28,311,552,000 | 26.367187500 |
| 5,000 | 35,389,440,000 | 32.958984375 |
| 5,600 | 39,636,172,800 | 36.914062500 |

### Fixed host-side allocations

- Six-expert pinned staging slab: **42,467,328 bytes (0.03955078125 GiB)**.
- 320-expert device cache: **2,264,924,160 bytes (2.109375 GiB) of VRAM**, not RAM.
- Persistent 320-expert host mirror in current code: **0 bytes**. `cuda_moe_expert_cache_prepare()` allocates the cache planes with `cudaMalloc`; host allocation is only six route entries. See `ds4_cuda.cu:26725-26834`.
- Conservative hypothetical duplicate 320-entry host mirror: **2,264,924,160 bytes**. This is unnecessary if every seeded cache entry is a pointer/copy sourced from the K-entry arena; the design should enforce that invariant.

The machine's measured physical total from memory-preflight data is **68,601,917,440 bytes (63.890514 GiB)**. The table below protects a deliberately conservative **24 GiB = 25,769,803,776 bytes** for Windows, the engine, driver/kernel pools, mapped non-expert working set, and operating headroom.

| K | Arena + 6-stage, bytes | Plus hypothetical 320 mirror, bytes | GiB with mirror | Headroom after 24-GiB floor |
|---:|---:|---:|---:|---:|
| 3,000 | 21,276,131,328 | 23,541,055,488 | 21.924316406 | 19,291,058,176 bytes (17.966 GiB) |
| 4,000 | 28,354,019,328 | 30,618,943,488 | 28.516113281 | 12,213,170,176 bytes (11.374 GiB) |
| 5,000 | 35,431,907,328 | 37,696,831,488 | 35.107910156 | 5,135,282,176 bytes (4.783 GiB) |
| 5,600 | 39,678,640,128 | 41,943,564,288 | 39.062988281 | 888,549,376 bytes (0.828 GiB) |

With that 24-GiB protected floor:

- max K with the existing deduplicated design is **6,045**;
- max K if a duplicate 320-entry RAM mirror is unnecessarily added is **5,725**.

Those are arithmetic maxima, not good operating points. **K=5,600 is the sustainable recommendation**: it fits the stated 38–42 GiB package budget even under duplicate-mirror accounting. In the preferred deduplicated layout it uses 39,678,640,128 bytes (36.954 GiB). Concretely, the remaining physical headroom is **3,153,473,536 bytes (2.937 GiB)** after the 24-GiB floor.

Do not attempt to pin the full 36.9-GiB arena. The current dynamic arena already supports pinned/pageable slots (`ds4_cuda.cu:954-1014`) and the Q1 bootstrap had to split 26,304,970,752 pinned + 12,651,724,800 pageable bytes. The keep-K arena should be committed/touched pageable RAM; copy selected resident experts into the 42,467,328-byte pinned slab before H2D. Large pinned allocations are a stability risk on a 64-GiB Windows host.

## 4. Quality

Weight representation is exact on both branches: resident routes upload the original IQ2/Q2_K bytes and fallback routes execute those same bytes on CPU. There is no Q1 approximation and no router mask, so the Q1 cosine failure is removed by construction.

The remaining issue is numerical, not quantization quality. A per-expert CPU/GPU cosine of 0.9998 with comparable norms corresponds to about `sqrt(2 × (1 - 0.9998)) = 0.02`, or 2%, relative difference for that expert output. If CPU routes are 5–15% of routed contributions, a reasonable aggregate perturbation envelope is roughly 0.1–0.3% when errors scale with the norm fraction, or 0.45–0.77% under a more conservative energy-fraction model. Both are tiny compared with Q1's 0.55–0.60 activation cosine, but autoregressive decoding can still flip near-tied logits.

Acceptance should therefore require more than the single-expert cosine:

- mixed CPU/GPU routed-output cosine and max-absolute error at 5%, 10%, and 15% forced tails;
- end-to-end logit cosine/KL and top-1/top-5 agreement against all-GPU exact IQ2;
- greedy token parity until first divergence plus distributional generation checks over the diverse replay set;
- deterministic accumulation order where practical, or an explicit tolerance contract if not.

## 5. Implementation delta

### Existing seams

- CPU exact kernels already exist in `ds4.c`: expert pointer extraction at lines 4586–4606, paired IQ2 gate/up GEMV at 4629–4659, batched selected-expert mid construction at 4699–4745, Q2_K accumulated down projection at 4816–4857, and the persistent-scratch six-expert decode path at 6558–6605.
- The CUDA decode bridge computes routes and invokes the Q1 mixed wrapper at `ds4.c:11750-11876`.
- Exact resident cache allocation, six-route host staging, mapped request, and route worker are in `ds4_cuda.cu:26591-26970`.
- Mixed classification is in `ds4_cuda.cu:33726-33784` and `34008-34179`; the hot launch, cold Q1 launch, and GPU join are at `34226-34545`.
- Dynamic arena slots already carry `(layer, expert)`, state, generation, pinned/pageable status, and active bindings at `ds4_cuda.cu:929-1014`.

### Required changes and effort

| Work item | Concrete change | Estimate |
|---|---|---:|
| Offline manifest and validation | Parse replay rows, aggregate mass, apply per-layer floor then global fill, emit deterministic `(layer, expert, score)` manifest with trace hash/model hash/K/floor. Reject duplicates, wrong geometry, and stale model IDs. | 1.0–1.5 d |
| Bootstrap keep-K arena | Generalize the primary dynamic arena bootstrap to load exactly the manifest entries, freeze them for decode, use pageable resident storage plus the six-entry pinned stage, and publish a direct 43×256 resident bitmap/binding table. | 1.5–2.5 d |
| Dispatch split | Replace `Q1_RESIDENT` with `CPU_EXACT`; classify selected routes after the existing route D2H, launch the resident subset through the current exact cache/H2D path, and submit the nonresident subset to a bounded CPU job. | 1.0–1.5 d |
| CPU subset API and join | Refactor the lower-level CPU kernels into an exported arbitrary-subset primitive accepting activation, selected IDs, and weights. Add pinned activation/output mailboxes, activation D2H, worker sequencing, output H2D, failure propagation, and a CUDA-stream event join/add. | 3.0–5.0 d |
| Retire Q1 tiers | Remove Q1 sidecar installation from this mode, Q1 arena/snapshot/probation/promotion branches, Q1 cold launch, and Q1-specific invariants/telemetry. Keep a separate legacy mode until parity is demonstrated. | 1.0–2.0 d |
| Tests and measurement | Static contracts; manifest/floor tests; forced hit/miss mixes; numeric parity; queue cancellation; shutdown; memory-pressure failure; per-layer and end-to-end overlap attribution; 64/256-token A/B. | 2.0–3.0 d |
| **Total** | Prototype-quality integrated path | **9.5–15.5 dev-days** |

The fastest de-risking sequence is a 1–2 day one-layer spike before modifying all Q1 tiering: force 1–3 CPU experts, use the existing exact CPU kernels, transfer a real `ffn_norm` activation and partial output, and measure the critical path with the resident GPU subset active.

### Riskiest unknown

The critical unknown is **layer-local overlap**, not correctness. The supplied 1.267 ms is a warm, eight-thread, single-expert CPU measurement. Production needs a GPU-produced 4,096-float activation on the host, an arbitrary subset GEMV, a 4,096-float result back on device, and a join before the same layer can finish. There are 43 such dependency barriers per token. Tail routes may cluster in a few layers, the eight-thread GEMV may contend with route/H2D workers, and a global `fallback_experts/token × 1.267` model does not reveal the maximum per-layer critical path. Measure a histogram of fallback count and gate mass **per layer per token**, along with activation D2H, queue wait, CPU compute, output H2D, and join wait.

## Decision gates

Proceed from spike to full prototype only if:

1. a restored, held-out replay set shows K=5,600 fallback at or below about 15% by count and no pathological high-mass misses;
2. the per-layer spike sustains at least the serial-model floor and demonstrates useful overlap without starving CUDA submission;
3. K=5,600 starts reliably with a pageable arena and at least 2 GiB measured available-RAM margin under the real server workload (drop K if the conservative duplicate mirror cannot be eliminated);
4. forced-tail end-to-end quality meets the all-exact tolerance contract.

If any of (1)–(3) fails, the architecture is **NO-GO at K=5,600 on 64 GiB**, though a smaller resident set plus a more strongly batched CPU tail may still be worth revisiting.

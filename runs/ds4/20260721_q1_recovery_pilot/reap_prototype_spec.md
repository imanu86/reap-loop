# REAP revisited: exact resident base plus CPU-GEMV tail

Date: 2026-07-21  
Machine/scope: the measured Windows host; CPU analysis only; no GPU or server was run for this study.

## Executive verdict

**GO for a bounded prototype, with two gates before full integration.** Use **K = 5,600** as the first target, store the resident arena in pageable RAM, retain only a six-expert pinned transfer slab, and do not create a duplicate 320-expert host mirror. First restore/recapture the route trace and then prove one-layer asynchronous CPU/GPU join behavior. Those two gates are important because the requested K-specific coverage curve cannot be measured from the files currently present, and per-token arithmetic alone overstates overlap across the 43 sequential layer barriers.

Conditional on the proposed tail being **5â€“15% of 258 routes/token**, K=5,600 projects to:

| Scheduling bound | Current plumbing | With F1/F2 |
|---|---:|---:|
| Serial CPU tail, 5â€“15% fallback | **6.05â€“5.05 t/s** | **10.89â€“8.03 t/s** |
| Fully hidden CPU tail (optimistic ceiling) | **6.72 t/s** | **13.25 t/s** |

These are sensitivity bounds, **not replay-derived K=5,600 predictions**, because the route rows are missing. The design still merits a prototype: it restores exact weights, K=5,600 fits the stated 38â€“42 GiB package budget, and even a 15% fully serial tail remains above the broken-Q1 4.09 t/s baseline. The likely make-or-break issue is not GEMV speed but whether activation D2H, CPU work, output H2D, and the mandatory per-layer join can actually overlap without starving the CUDA submission/route-worker threads.

## 1. Evidence and route-trace integrity

The requested input was `C:\Users\imanu\g130i\u1_$Tag.server.stderr.log`, expected to contain 16,512 `[q1-0-mixed-route]` rows. At analysis time:

- `u1_$Tag.server.stderr.log` contained **0** route rows and only failed-startup/zero-call summaries. It was 1,853 bytes at the final scan and had been rewritten during the analysis window.
- `u1_attrib.server.stderr.log` also contained **0** route rows (`trace_rows=0`). It does corroborate a 64-token decode: `calls=2752`, `slots=16512`, because 64 Ã— 43 layers Ã— 6 routes = 16,512.
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

A reasonable minimum is **20 diverse prompts Ã— 256 decoded tokens = 5,120 tokens = 1,320,960 route rows**. A firmer sizing run is **50 prompts Ã— 512 tokens = 25,600 tokens = 6,604,800 rows**, stratified across code, prose, dialogue, multilingual, factual, and long-context continuations. Rank on a training subset, report coverage on held-out prompts, and bootstrap by whole prompt (not individual route row) for 95% intervals. Do not freeze the production manifest until the K=5,000/5,600 fallback-count and fallback-mass intervals are narrow enough to make the latency decision (roughly Â±1 percentage point is a useful target).

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
- serial CPU tail = `(F / 64) Ã— 1.267 ms`;
- serial latency = `fixed_base + CPU tail`;
- fully overlapped lower bound = `max(fixed_base, CPU tail)`.

The fully overlapped formula is only a ceiling on throughput. A routed result is needed before the layer's HC post and the next layer, so CPU work cannot simply float underneath the entire token's GPU time.

### Sensitivity while K coverage is unavailable

There are 43 Ã— 6 = 258 routed experts/token.

| Fallback share | Fallback experts/token | CPU tail, serial ms | Current serial t/s | Current overlap ceiling t/s | F1/F2 serial t/s | F1/F2 overlap ceiling t/s |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 0.0 | 0.0000 | 6.7159 | 6.7159 | 13.2450 | 13.2450 |
| 5% | 12.9 | 16.3443 | 6.0516 | 6.7159 | 10.8880 | 13.2450 |
| 10% | 25.8 | 32.6886 | 5.5070 | 6.7159 | 9.2431 | 13.2450 |
| 15% | 38.7 | 49.0329 | 5.0522 | 6.7159 | 8.0300 | 13.2450 |

At each requested K (3,000/4,000/5,000/5,600), the CSV/JSON applies these formulas once fallback count is known. No K-specific t/s value is defensible from the currently available logs.

The 1.267 ms/expert figure is a serial upper-cost model, not necessarily the best multi-expert implementation. The existing CPU primitive parallelizes rows across the shared 8-thread pool, so processing several fallback experts as one layer batch may be cheaper than `N Ã— 1.267`; conversely, CPU contention and transfer/join overhead may erase overlap.

## 3. Exact memory map

### Expert size, measured from the model layout

The exact routed-expert tensor total is 72.5625 GiB for 43 Ã— 256 = 11,008 experts:

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

Those are arithmetic maxima, not good operating points. **K=5,600 is the sustainable recommendation**: it fits the stated 38â€“42 GiB package budget even under duplicate-mirror accounting. In the preferred deduplicated layout it uses 39,678,640,128 bytes (36.954 GiB). Concretely, the remaining physical headroom is **3,153,473,536 bytes (2.937 GiB)** after the 24-GiB floor.

Do not attempt to pin the full 36.9-GiB arena. The current dynamic arena already supports pinned/pageable slots (`ds4_cuda.cu:954-1014`) and the Q1 bootstrap had to split 26,304,970,752 pinned + 12,651,724,800 pageable bytes. The keep-K arena should be committed/touched pageable RAM; copy selected resident experts into the 42,467,328-byte pinned slab before H2D. Large pinned allocations are a stability risk on a 64-GiB Windows host.

## 4. Quality

Weight representation is exact on both branches: resident routes upload the original IQ2/Q2_K bytes and fallback routes execute those same bytes on CPU. There is no Q1 approximation and no router mask, so the Q1 cosine failure is removed by construction.

The remaining issue is numerical, not quantization quality. A per-expert CPU/GPU cosine of 0.9998 with comparable norms corresponds to about `sqrt(2 Ã— (1 - 0.9998)) = 0.02`, or 2%, relative difference for that expert output. If CPU routes are 5â€“15% of routed contributions, a reasonable aggregate perturbation envelope is roughly 0.1â€“0.3% when errors scale with the norm fraction, or 0.45â€“0.77% under a more conservative energy-fraction model. Both are tiny compared with Q1's 0.55â€“0.60 activation cosine, but autoregressive decoding can still flip near-tied logits.

Acceptance should therefore require more than the single-expert cosine:

- mixed CPU/GPU routed-output cosine and max-absolute error at 5%, 10%, and 15% forced tails;
- end-to-end logit cosine/KL and top-1/top-5 agreement against all-GPU exact IQ2;
- greedy token parity until first divergence plus distributional generation checks over the diverse replay set;
- deterministic accumulation order where practical, or an explicit tolerance contract if not.

## 5. Implementation delta

### Existing seams

- CPU exact kernels already exist in `ds4.c`: expert pointer extraction at lines 4586â€“4606, paired IQ2 gate/up GEMV at 4629â€“4659, batched selected-expert mid construction at 4699â€“4745, Q2_K accumulated down projection at 4816â€“4857, and the persistent-scratch six-expert decode path at 6558â€“6605.
- The CUDA decode bridge computes routes and invokes the Q1 mixed wrapper at `ds4.c:11750-11876`.
- Exact resident cache allocation, six-route host staging, mapped request, and route worker are in `ds4_cuda.cu:26591-26970`.
- Mixed classification is in `ds4_cuda.cu:33726-33784` and `34008-34179`; the hot launch, cold Q1 launch, and GPU join are at `34226-34545`.
- Dynamic arena slots already carry `(layer, expert)`, state, generation, pinned/pageable status, and active bindings at `ds4_cuda.cu:929-1014`.

### Required changes and effort

| Work item | Concrete change | Estimate |
|---|---|---:|
| Offline manifest and validation | Parse replay rows, aggregate mass, apply per-layer floor then global fill, emit deterministic `(layer, expert, score)` manifest with trace hash/model hash/K/floor. Reject duplicates, wrong geometry, and stale model IDs. | 1.0â€“1.5 d |
| Bootstrap keep-K arena | Generalize the primary dynamic arena bootstrap to load exactly the manifest entries, freeze them for decode, use pageable resident storage plus the six-entry pinned stage, and publish a direct 43Ã—256 resident bitmap/binding table. | 1.5â€“2.5 d |
| Dispatch split | Replace `Q1_RESIDENT` with `CPU_EXACT`; classify selected routes after the existing route D2H, launch the resident subset through the current exact cache/H2D path, and submit the nonresident subset to a bounded CPU job. | 1.0â€“1.5 d |
| CPU subset API and join | Refactor the lower-level CPU kernels into an exported arbitrary-subset primitive accepting activation, selected IDs, and weights. Add pinned activation/output mailboxes, activation D2H, worker sequencing, output H2D, failure propagation, and a CUDA-stream event join/add. | 3.0â€“5.0 d |
| Retire Q1 tiers | Remove Q1 sidecar installation from this mode, Q1 arena/snapshot/probation/promotion branches, Q1 cold launch, and Q1-specific invariants/telemetry. Keep a separate legacy mode until parity is demonstrated. | 1.0â€“2.0 d |
| Tests and measurement | Static contracts; manifest/floor tests; forced hit/miss mixes; numeric parity; queue cancellation; shutdown; memory-pressure failure; per-layer and end-to-end overlap attribution; 64/256-token A/B. | 2.0â€“3.0 d |
| **Total** | Prototype-quality integrated path | **9.5â€“15.5 dev-days** |

The fastest de-risking sequence is a 1â€“2 day one-layer spike before modifying all Q1 tiering: force 1â€“3 CPU experts, use the existing exact CPU kernels, transfer a real `ffn_norm` activation and partial output, and measure the critical path with the resident GPU subset active.

### Riskiest unknown

The critical unknown is **layer-local overlap**, not correctness. The supplied 1.267 ms is a warm, eight-thread, single-expert CPU measurement. Production needs a GPU-produced 4,096-float activation on the host, an arbitrary subset GEMV, a 4,096-float result back on device, and a join before the same layer can finish. There are 43 such dependency barriers per token. Tail routes may cluster in a few layers, the eight-thread GEMV may contend with route/H2D workers, and a global `fallback_experts/token Ã— 1.267` model does not reveal the maximum per-layer critical path. Measure a histogram of fallback count and gate mass **per layer per token**, along with activation D2H, queue wait, CPU compute, output H2D, and join wait.

## Decision gates

Proceed from spike to full prototype only if:

1. a restored, held-out replay set shows K=5,600 fallback at or below about 15% by count and no pathological high-mass misses;
2. the per-layer spike sustains at least the serial-model floor and demonstrates useful overlap without starving CUDA submission;
3. K=5,600 starts reliably with a pageable arena and at least 2 GiB measured available-RAM margin under the real server workload (drop K if the conservative duplicate mirror cannot be eliminated);
4. forced-tail end-to-end quality meets the all-exact tolerance contract.

If any of (1)â€“(3) fails, the architecture is **NO-GO at K=5,600 on 64 GiB**, though a smaller resident set plus a more strongly batched CPU tail may still be worth revisiting.

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Content -LiteralPath 'C:\\Users\\imanu\\g130i\\coverage\\coverage_curve_real.json'; Get-Content -LiteralPath 'C:\\Users\\imanu\\g130i\\overlap_spike\\overlap_spike_fixed.json'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 505ms:
{
  "files": [
    {
      "path": "C:\\Users\\imanu\\g130i\\u1_replay1.server.stderr.log",
      "route_rows": 16512
    },
    {
      "path": "C:\\Users\\imanu\\g130i\\u1_replay2.server.stderr.log",
      "route_rows": 16512
    },
    {
      "path": "C:\\Users\\imanu\\g130i\\u1_replay3.server.stderr.log",
      "route_rows": 16512
    }
  ],
  "input_patterns": [
    "C:/Users/imanu/g130i/u1_replay*.server.stderr.log"
  ],
  "layers_touched": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42
  ],
  "mass_concentration": {
    "top_1000_gate_mass_pct": 63.029734873967556,
    "top_2000_gate_mass_pct": 79.10807325843595,
    "top_4000_gate_mass_pct": 94.08944054740812
  },
  "replays_merged": 3,
  "route_rows_merged": 49536,
  "strategies": {
    "floor_32": {
      "3000": {
        "fallback_gate_mass": 1444.40624567307,
        "fallback_gate_mass_pct": 11.66348713390223,
        "fallback_route_count": 8411,
        "fallback_route_pct": 16.979570413436694,
        "floor": 32,
        "floor_candidate_count": 1376,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 117,
          "layer": 2,
          "mass": 0.890106053,
          "routes": 3
        },
        "resident_count": 3000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 67.52614504339998,
          "fallback_gate_mass_pct": 23.446578168522844,
          "fallback_route_count": 267,
          "fallback_route_pct": 23.177083333333332,
          "layer": 2
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 63.531690590270024,
          "fallback_gate_mass_pct": 22.059614797244578,
          "fallback_route_count": 328,
          "fallback_route_pct": 28.47222222222222,
          "layer": 0
        }
      },
      "4000": {
        "fallback_gate_mass": 731.96368210757,
        "fallback_gate_mass_pct": 5.910559452591624,
        "fallback_route_count": 4681,
        "fallback_route_pct": 9.449693152454781,
        "floor": 32,
        "floor_candidate_count": 1376,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 250,
          "layer": 17,
          "mass": 0.554378822,
          "routes": 3
        },
        "resident_count": 4000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        }
      },
      "5000": {
        "fallback_gate_mass": 309.5559829933,
        "fallback_gate_mass_pct": 2.4996445672265675,
        "fallback_route_count": 2240,
        "fallback_route_pct": 4.521963824289406,
        "floor": 32,
        "floor_candidate_count": 1376,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 236,
          "layer": 8,
          "mass": 0.303848624,
          "routes": 1
        },
        "resident_count": 5000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 9.9625351276,
          "fallback_gate_mass_pct": 3.459213601997509,
          "fallback_route_count": 87,
          "fallback_route_pct": 7.552083333333333,
          "layer": 33
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 9.558013782900003,
          "fallback_gate_mass_pct": 3.318754781072543,
          "fallback_route_count": 95,
          "fallback_route_pct": 8.246527777777779,
          "layer": 32
        }
      },
      "5600": {
        "fallback_gate_mass": 159.342394502,
        "fallback_gate_mass_pct": 1.2866795430486557,
        "fallback_route_count": 1279,
        "fallback_route_pct": 2.5819605943152455,
        "floor": 32,
        "floor_candidate_count": 1376,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 216,
          "layer": 23,
          "mass": 0.20232175689999998,
          "routes": 3
        },
        "resident_count": 5600,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 5.6856262051,
          "fallback_gate_mass_pct": 1.9741757639419166,
          "fallback_route_count": 51,
          "fallback_route_pct": 4.427083333333333,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 5.5760508798,
          "fallback_gate_mass_pct": 1.9361287866777181,
          "fallback_route_count": 52,
          "fallback_route_pct": 4.513888888888889,
          "layer": 33
        }
      },
      "6045": {
        "fallback_gate_mass": 80.2392541492,
        "fallback_gate_mass_pct": 0.6479267942842528,
        "fallback_route_count": 748,
        "fallback_route_pct": 1.5100129198966408,
        "floor": 32,
        "floor_candidate_count": 1376,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 110,
          "layer": 10,
          "mass": 0.154018581,
          "routes": 1
        },
        "resident_count": 6045,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 3.0254450028,
          "fallback_gate_mass_pct": 1.0505017361692517,
          "fallback_route_count": 31,
          "fallback_route_pct": 2.6909722222222223,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 2.7242204637,
          "fallback_gate_mass_pct": 0.9459098819019376,
          "fallback_route_count": 33,
          "fallback_route_pct": 2.8645833333333335,
          "layer": 32
        }
      }
    },
    "floor_64": {
      "3000": {
        "fallback_gate_mass": 1466.04457065157,
        "fallback_gate_mass_pct": 11.83821521039869,
        "fallback_route_count": 8381,
        "fallback_route_pct": 16.919008397932817,
        "floor": 64,
        "floor_candidate_count": 2752,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 126,
          "layer": 0,
          "mass": 0.9812762896,
          "routes": 6
        },
        "resident_count": 3000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 81.59235677160001,
          "fallback_gate_mass_pct": 28.330679468963808,
          "fallback_route_count": 330,
          "fallback_route_pct": 28.645833333333332,
          "layer": 2
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 77.69387037047,
          "fallback_gate_mass_pct": 26.977038334032745,
          "fallback_route_count": 387,
          "fallback_route_pct": 33.59375,
          "layer": 0
        }
      },
      "4000": {
        "fallback_gate_mass": 732.80357029877,
        "fallback_gate_mass_pct": 5.917341495483866,
        "fallback_route_count": 4687,
        "fallback_route_pct": 9.461805555555555,
        "floor": 64,
        "floor_candidate_count": 2752,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 190,
          "layer": 19,
          "mass": 0.556425512,
          "routes": 2
        },
        "resident_count": 4000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        }
      },
      "5000": {
        "fallback_gate_mass": 309.5559829933,
        "fallback_gate_mass_pct": 2.4996445672265675,
        "fallback_route_count": 2240,
        "fallback_route_pct": 4.521963824289406,
        "floor": 64,
        "floor_candidate_count": 2752,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 236,
          "layer": 8,
          "mass": 0.303848624,
          "routes": 1
        },
        "resident_count": 5000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 9.9625351276,
          "fallback_gate_mass_pct": 3.459213601997509,
          "fallback_route_count": 87,
          "fallback_route_pct": 7.552083333333333,
          "layer": 33
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 9.558013782900003,
          "fallback_gate_mass_pct": 3.318754781072543,
          "fallback_route_count": 95,
          "fallback_route_pct": 8.246527777777779,
          "layer": 32
        }
      },
      "5600": {
        "fallback_gate_mass": 159.342394502,
        "fallback_gate_mass_pct": 1.2866795430486557,
        "fallback_route_count": 1279,
        "fallback_route_pct": 2.5819605943152455,
        "floor": 64,
        "floor_candidate_count": 2752,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 216,
          "layer": 23,
          "mass": 0.20232175689999998,
          "routes": 3
        },
        "resident_count": 5600,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 5.6856262051,
          "fallback_gate_mass_pct": 1.9741757639419166,
          "fallback_route_count": 51,
          "fallback_route_pct": 4.427083333333333,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 5.5760508798,
          "fallback_gate_mass_pct": 1.9361287866777181,
          "fallback_route_count": 52,
          "fallback_route_pct": 4.513888888888889,
          "layer": 33
        }
      },
      "6045": {
        "fallback_gate_mass": 80.2392541492,
        "fallback_gate_mass_pct": 0.6479267942842528,
        "fallback_route_count": 748,
        "fallback_route_pct": 1.5100129198966408,
        "floor": 64,
        "floor_candidate_count": 2752,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 110,
          "layer": 10,
          "mass": 0.154018581,
          "routes": 1
        },
        "resident_count": 6045,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 3.0254450028,
          "fallback_gate_mass_pct": 1.0505017361692517,
          "fallback_route_count": 31,
          "fallback_route_pct": 2.6909722222222223,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 2.7242204637,
          "fallback_gate_mass_pct": 0.9459098819019376,
          "fallback_route_count": 33,
          "fallback_route_pct": 2.8645833333333335,
          "layer": 32
        }
      }
    },
    "floor_96": {
      "3000": {
        "fallback_gate_mass": 1460.54860911577,
        "fallback_gate_mass_pct": 11.793835676002981,
        "fallback_route_count": 8362,
        "fallback_route_pct": 16.88065245478036,
        "floor": 96,
        "floor_candidate_count": 4128,
        "floor_satisfied": false,
        "highest_mass_missed_expert": {
          "expert": 1,
          "layer": 2,
          "mass": 1.195188702,
          "routes": 5
        },
        "resident_count": 3000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 105.58654600740002,
          "fallback_gate_mass_pct": 36.66199518594921,
          "fallback_route_count": 430,
          "fallback_route_pct": 37.326388888888886,
          "layer": 2
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 94.48379006386998,
          "fallback_gate_mass_pct": 32.8068715632747,
          "fallback_route_count": 461,
          "fallback_route_pct": 40.017361111111114,
          "layer": 0
        }
      },
      "4000": {
        "fallback_gate_mass": 847.56862988037,
        "fallback_gate_mass_pct": 6.844061938476528,
        "fallback_route_count": 4949,
        "fallback_route_pct": 9.990713824289406,
        "floor": 96,
        "floor_candidate_count": 4128,
        "floor_satisfied": false,
        "highest_mass_missed_expert": {
          "expert": 1,
          "layer": 2,
          "mass": 1.195188702,
          "routes": 5
        },
        "resident_count": 4000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 105.58654600740002,
          "fallback_gate_mass_pct": 36.66199518594921,
          "fallback_route_count": 430,
          "fallback_route_pct": 37.326388888888886,
          "layer": 2
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 94.48379006386998,
          "fallback_gate_mass_pct": 32.8068715632747,
          "fallback_route_count": 461,
          "fallback_route_pct": 40.017361111111114,
          "layer": 0
        }
      },
      "5000": {
        "fallback_gate_mass": 314.682940781,
        "fallback_gate_mass_pct": 2.541044420191779,
        "fallback_route_count": 2254,
        "fallback_route_pct": 4.550226098191215,
        "floor": 96,
        "floor_candidate_count": 4128,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 28,
          "layer": 22,
          "mass": 0.319532126,
          "routes": 1
        },
        "resident_count": 5000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 10.385479811099998,
          "fallback_gate_mass_pct": 3.606069375716388,
          "fallback_route_count": 84,
          "fallback_route_pct": 7.291666666666667,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 10.385479811099998,
          "fallback_gate_mass_pct": 3.606069375716388,
          "fallback_route_count": 84,
          "fallback_route_pct": 7.291666666666667,
          "layer": 35
        }
      },
      "5600": {
        "fallback_gate_mass": 160.2924378077,
        "fallback_gate_mass_pct": 1.2943510813751316,
        "fallback_route_count": 1281,
        "fallback_route_pct": 2.585998062015504,
        "floor": 96,
        "floor_candidate_count": 4128,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 45,
          "layer": 23,
          "mass": 0.2048179437,
          "routes": 2
        },
        "resident_count": 5600,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 5.7787257758,
          "fallback_gate_mass_pct": 2.0065020147815096,
          "fallback_route_count": 53,
          "fallback_route_pct": 4.600694444444445,
          "layer": 33
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 4.9653061756,
          "fallback_gate_mass_pct": 1.7240646418864791,
          "fallback_route_count": 55,
          "fallback_route_pct": 4.774305555555555,
          "layer": 32
        }
      },
      "6045": {
        "fallback_gate_mass": 80.4676154428,
        "fallback_gate_mass_pct": 0.6497707969793144,
        "fallback_route_count": 750,
        "fallback_route_pct": 1.5140503875968991,
        "floor": 96,
        "floor_candidate_count": 4128,
        "floor_satisfied": true,
        "highest_mass_missed_expert": {
          "expert": 187,
          "layer": 30,
          "mass": 0.1547265947,
          "routes": 2
        },
        "resident_count": 6045,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 3.1796364078,
          "fallback_gate_mass_pct": 1.1040404184143322,
          "fallback_route_count": 32,
          "fallback_route_pct": 2.7777777777777777,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 2.7242204637,
          "fallback_gate_mass_pct": 0.9459098819019376,
          "fallback_route_count": 33,
          "fallback_route_pct": 2.8645833333333335,
          "layer": 32
        }
      }
    },
    "global": {
      "3000": {
        "fallback_gate_mass": 1444.40624567307,
        "fallback_gate_mass_pct": 11.66348713390223,
        "fallback_route_count": 8411,
        "fallback_route_pct": 16.979570413436694,
        "floor": null,
        "floor_candidate_count": null,
        "floor_satisfied": null,
        "highest_mass_missed_expert": {
          "expert": 117,
          "layer": 2,
          "mass": 0.890106053,
          "routes": 3
        },
        "resident_count": 3000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 67.52614504339998,
          "fallback_gate_mass_pct": 23.446578168522844,
          "fallback_route_count": 267,
          "fallback_route_pct": 23.177083333333332,
          "layer": 2
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 63.531690590270024,
          "fallback_gate_mass_pct": 22.059614797244578,
          "fallback_route_count": 328,
          "fallback_route_pct": 28.47222222222222,
          "layer": 0
        }
      },
      "4000": {
        "fallback_gate_mass": 731.96368210757,
        "fallback_gate_mass_pct": 5.910559452591624,
        "fallback_route_count": 4681,
        "fallback_route_pct": 9.449693152454781,
        "floor": null,
        "floor_candidate_count": null,
        "floor_satisfied": null,
        "highest_mass_missed_expert": {
          "expert": 250,
          "layer": 17,
          "mass": 0.554378822,
          "routes": 3
        },
        "resident_count": 4000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 22.598274254500012,
          "fallback_gate_mass_pct": 7.846623039268341,
          "fallback_route_count": 161,
          "fallback_route_pct": 13.975694444444445,
          "layer": 31
        }
      },
      "5000": {
        "fallback_gate_mass": 309.5559829933,
        "fallback_gate_mass_pct": 2.4996445672265675,
        "fallback_route_count": 2240,
        "fallback_route_pct": 4.521963824289406,
        "floor": null,
        "floor_candidate_count": null,
        "floor_satisfied": null,
        "highest_mass_missed_expert": {
          "expert": 236,
          "layer": 8,
          "mass": 0.303848624,
          "routes": 1
        },
        "resident_count": 5000,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 9.9625351276,
          "fallback_gate_mass_pct": 3.459213601997509,
          "fallback_route_count": 87,
          "fallback_route_pct": 7.552083333333333,
          "layer": 33
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 9.558013782900003,
          "fallback_gate_mass_pct": 3.318754781072543,
          "fallback_route_count": 95,
          "fallback_route_pct": 8.246527777777779,
          "layer": 32
        }
      },
      "5600": {
        "fallback_gate_mass": 159.342394502,
        "fallback_gate_mass_pct": 1.2866795430486557,
        "fallback_route_count": 1279,
        "fallback_route_pct": 2.5819605943152455,
        "floor": null,
        "floor_candidate_count": null,
        "floor_satisfied": null,
        "highest_mass_missed_expert": {
          "expert": 216,
          "layer": 23,
          "mass": 0.20232175689999998,
          "routes": 3
        },
        "resident_count": 5600,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 5.6856262051,
          "fallback_gate_mass_pct": 1.9741757639419166,
          "fallback_route_count": 51,
          "fallback_route_pct": 4.427083333333333,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 5.5760508798,
          "fallback_gate_mass_pct": 1.9361287866777181,
          "fallback_route_count": 52,
          "fallback_route_pct": 4.513888888888889,
          "layer": 33
        }
      },
      "6045": {
        "fallback_gate_mass": 80.2392541492,
        "fallback_gate_mass_pct": 0.6479267942842528,
        "fallback_route_count": 748,
        "fallback_route_pct": 1.5100129198966408,
        "floor": null,
        "floor_candidate_count": null,
        "floor_satisfied": null,
        "highest_mass_missed_expert": {
          "expert": 110,
          "layer": 10,
          "mass": 0.154018581,
          "routes": 1
        },
        "resident_count": 6045,
        "worst_layer_by_mass_pct": {
          "fallback_gate_mass": 3.0254450028,
          "fallback_gate_mass_pct": 1.0505017361692517,
          "fallback_route_count": 31,
          "fallback_route_pct": 2.6909722222222223,
          "layer": 35
        },
        "worst_layer_by_route_pct": {
          "fallback_gate_mass": 2.7242204637,
          "fallback_gate_mass_pct": 0.9459098819019376,
          "fallback_route_count": 33,
          "fallback_route_pct": 2.8645833333333335,
          "layer": 32
        }
      }
    }
  },
  "total_gate_mass": 12383.99999151727,
  "unique_experts_touched": 6766,
  "variance_caveat": null
}
{
  "schema": "gate2_overlap_spike_v1",
  "question": "can a CPU-GEMV tail hide under one MoE layer GPU compute without starving CUDA submission?",
  "pass_criterion": {"overlap_efficiency_min": 0.70, "submission_stall_ms_max": 2.0},
  "config": {"model": "C:\\ds4-models\\ds4-2bit.gguf", "layer": 3, "hot_m": 8, "gpu_passes": 384, "evict_mb": 512, "eviction_scope": "once_before_tail_outside_timing", "expert_bytes": 7077888, "hot_bytes": 56623104},
  "results": [
    {"mode": "tail_6_of_8", "tail_experts": 4, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 93.976100, "serial_gpu_wall_ms": 74.954600, "serial_gpu_event_ms": 73.064445, "serial_gpu_enqueue_ms": 4.148200, "cpu_tail_ms": 3.467100, "overlapped_wall_ms": 72.403300, "overlapped_gpu_event_ms": 72.218620, "overlapped_enqueue_ms": 0.100100, "overlap_efficiency": 6.222145, "pass": true},
    {"mode": "tail_6_of_8", "tail_experts": 8, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 103.053300, "serial_gpu_wall_ms": 73.273000, "serial_gpu_event_ms": 73.224800, "serial_gpu_enqueue_ms": 0.071300, "cpu_tail_ms": 13.959200, "overlapped_wall_ms": 72.769100, "overlapped_gpu_event_ms": 72.523773, "overlapped_enqueue_ms": 0.132300, "overlap_efficiency": 2.169480, "pass": true},
    {"mode": "tail_6_of_8", "tail_experts": 16, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 101.971300, "serial_gpu_wall_ms": 72.220200, "serial_gpu_event_ms": 72.157181, "serial_gpu_enqueue_ms": 0.074000, "cpu_tail_ms": 14.472500, "overlapped_wall_ms": 72.396400, "overlapped_gpu_event_ms": 72.196098, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 2.043524, "pass": true},
    {"mode": "tail_6_of_8", "tail_experts": 24, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 109.552100, "serial_gpu_wall_ms": 72.225500, "serial_gpu_event_ms": 72.178497, "serial_gpu_enqueue_ms": 0.077700, "cpu_tail_ms": 21.759800, "overlapped_wall_ms": 79.089300, "overlapped_gpu_event_ms": 72.183807, "overlapped_enqueue_ms": 0.105700, "overlap_efficiency": 1.399958, "pass": true},
    {"mode": "tail_8_of_8", "tail_experts": 4, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 90.972200, "serial_gpu_wall_ms": 72.199700, "serial_gpu_event_ms": 72.139618, "serial_gpu_enqueue_ms": 0.094400, "cpu_tail_ms": 2.687100, "overlapped_wall_ms": 73.273900, "overlapped_gpu_event_ms": 73.141052, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 6.586394, "pass": true},
    {"mode": "tail_8_of_8", "tail_experts": 8, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 94.499400, "serial_gpu_wall_ms": 72.194300, "serial_gpu_event_ms": 72.159233, "serial_gpu_enqueue_ms": 0.081600, "cpu_tail_ms": 6.887300, "overlapped_wall_ms": 72.383800, "overlapped_gpu_event_ms": 72.214272, "overlapped_enqueue_ms": 0.082800, "overlap_efficiency": 3.211070, "pass": true},
    {"mode": "tail_8_of_8", "tail_experts": 16, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 99.722800, "serial_gpu_wall_ms": 72.224500, "serial_gpu_event_ms": 72.183807, "serial_gpu_enqueue_ms": 0.088500, "cpu_tail_ms": 11.581400, "overlapped_wall_ms": 72.333800, "overlapped_gpu_event_ms": 72.174591, "overlapped_enqueue_ms": 0.117300, "overlap_efficiency": 2.364913, "pass": true},
    {"mode": "tail_8_of_8", "tail_experts": 24, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 106.233600, "serial_gpu_wall_ms": 73.608800, "serial_gpu_event_ms": 73.571198, "serial_gpu_enqueue_ms": 0.076600, "cpu_tail_ms": 16.771800, "overlapped_wall_ms": 72.448300, "overlapped_gpu_event_ms": 72.186882, "overlapped_enqueue_ms": 0.105500, "overlap_efficiency": 2.014411, "pass": true}
  ]
}

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4.c'; Show-Lines "'$p 2430 2585; Show-Lines $p 3394 3495; Show-Lines $p 19430 19690; Show-Lines $p 20120 20295' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 836ms:
  2430: 
  2431: typedef struct {
  2432:     const ds4_model *model;
  2433:     const ds4_model *primary_model;
  2434:     ds4_tensor *gate[DS4_N_LAYER];
  2435:     ds4_tensor *up[DS4_N_LAYER];
  2436:     ds4_tensor *down[DS4_N_LAYER];
  2437:     uint64_t gate_row_bytes[DS4_N_LAYER];
  2438:     uint64_t gate_expert_bytes[DS4_N_LAYER];
  2439:     uint64_t down_row_bytes[DS4_N_LAYER];
  2440:     uint64_t down_expert_bytes[DS4_N_LAYER];
  2441:     uint32_t first_layer;
  2442:     uint32_t last_layer;
  2443:     bool ready;
  2444: } ds4_q1_0_sidecar;
  2445: 
  2446: typedef struct {
  2447:     const ds4_model *model;
  2448:     const ds4_tensor *gate;
  2449:     const ds4_tensor *up;
  2450:     const ds4_tensor *down;
  2451:     bool sidecar;
  2452: } ds4_routed_expert_source;
  2453: 
  2454: static ds4_iq1_s_sidecar g_iq1_s_sidecar;
  2455: static ds4_q1_0_sidecar g_q1_0_sidecar;
  2456: 
  2457: static int q1_0_resident_arena_requested(void) {
  2458:     const char *value = getenv("DS4_Q1_0_RESIDENT_ARENA");
  2459:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2460:     if (strcmp(value, "1") == 0) return 1;
  2461:     fprintf(stderr,
  2462:             "ds4: invalid DS4_Q1_0_RESIDENT_ARENA=%s; expected 0 or 1\n",
  2463:             value);
  2464:     return -1;
  2465: }
  2466: 
  2467: static int q1_0_dual_arena_requested(void) {
  2468:     const char *value = getenv("DS4_Q1_0_DUAL_ARENA");
  2469:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2470:     if (strcmp(value, "1") == 0) return 1;
  2471:     fprintf(stderr,
  2472:             "ds4: invalid DS4_Q1_0_DUAL_ARENA=%s; expected 0 or 1\n",
  2473:             value);
  2474:     return -1;
  2475: }
  2476: 
  2477: static int q1_0_dual_sparse_companion_requested(void) {
  2478:     const char *value = getenv("DS4_Q1_0_DUAL_SPARSE_COMPANION");
  2479:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2480:     if (strcmp(value, "1") == 0) return 1;
  2481:     fprintf(stderr,
  2482:             "ds4: invalid DS4_Q1_0_DUAL_SPARSE_COMPANION=%s; "
  2483:             "expected 0 or 1\n", value);
  2484:     return -1;
  2485: }
  2486: 
  2487: static int q1_0_mixed_cold_one_requested(void) {
  2488:     const char *value = getenv("DS4_Q1_0_MIXED_COLD_ONE");
  2489:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2490:     if (strcmp(value, "1") == 0) return 1;
  2491:     fprintf(stderr,
  2492:             "ds4: invalid DS4_Q1_0_MIXED_COLD_ONE=%s; expected 0 or 1\n",
  2493:             value);
  2494:     return -1;
  2495: }
  2496: 
  2497: static int q1_0_snapshot_backing_requested(void) {
  2498:     const char *value = getenv("DS4_Q1_0_SNAPSHOT_BACKING");
  2499:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2500:     if (strcmp(value, "1") == 0) return 1;
  2501:     fprintf(stderr,
  2502:             "ds4: invalid DS4_Q1_0_SNAPSHOT_BACKING=%s; expected 0 or 1\n",
  2503:             value);
  2504:     return -1;
  2505: }
  2506: 
  2507: static int q1_0_pageable_overflow_requested(void) {
  2508:     const char *value = getenv("DS4_Q1_0_PAGEABLE_OVERFLOW");
  2509:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2510:     if (strcmp(value, "1") == 0) return 1;
  2511:     fprintf(stderr,
  2512:             "ds4: invalid DS4_Q1_0_PAGEABLE_OVERFLOW=%s; expected 0 or 1\n",
  2513:             value);
  2514:     return -1;
  2515: }
  2516: 
  2517: static int q1_0_dynamic_promotion_requested(void) {
  2518:     const char *value = getenv("DS4_Q1_0_DYNAMIC_PROMOTION");
  2519:     if (!value || !value[0] || strcmp(value, "0") == 0) return 0;
  2520:     if (strcmp(value, "1") == 0) return 1;
  2521:     fprintf(stderr,
  2522:             "ds4: invalid DS4_Q1_0_DYNAMIC_PROMOTION=%s; expected 0 or 1\n",
  2523:             value);
  2524:     return -1;
  2525: }
  2526: 
  2527: static bool iq1_s_mixed_cold_one_requested(void) {
  2528:     const char *value = getenv("DS4_IQ1_S_MIXED_COLD_K");
  2529:     return value && strcmp(value, "1") == 0;
  2530: }
  2531: 
  2532: static bool iq1_s_sidecar_layer_active(
  2533:         const ds4_model *model,
  2534:         uint32_t layer_index) {
  2535:     return g_iq1_s_sidecar.ready &&
  2536:            model == g_iq1_s_sidecar.primary_model &&
  2537:            layer_index >= g_iq1_s_sidecar.first_layer &&
  2538:            layer_index <= g_iq1_s_sidecar.last_layer &&
  2539:            layer_index < DS4_N_LAYER;
  2540: }
  2541: 
  2542: static bool q1_0_sidecar_layer_active(
  2543:         const ds4_model *model,
  2544:         uint32_t layer_index) {
  2545:     return g_q1_0_sidecar.ready &&
  2546:            model == g_q1_0_sidecar.primary_model &&
  2547:            layer_index >= g_q1_0_sidecar.first_layer &&
  2548:            layer_index <= g_q1_0_sidecar.last_layer &&
  2549:            layer_index < DS4_N_LAYER;
  2550: }
  2551: 
  2552: static ds4_routed_expert_source routed_expert_source(
  2553:         const ds4_model *model,
  2554:         const ds4_layer_weights *layer,
  2555:         uint32_t layer_index) {
  2556:     ds4_routed_expert_source source = {
  2557:         model,
  2558:         layer ? layer->ffn_gate_exps : NULL,
  2559:         layer ? layer->ffn_up_exps : NULL,
  2560:         layer ? layer->ffn_down_exps : NULL,
  2561:         false,
  2562:     };
  2563:     if (q1_0_sidecar_layer_active(model, layer_index) &&
  2564:         q1_0_dual_arena_requested() <= 0 &&
  2565:         q1_0_snapshot_backing_requested() <= 0) {
  2566:         source.model = g_q1_0_sidecar.model;
  2567:         source.gate = g_q1_0_sidecar.gate[layer_index];
  2568:         source.up = g_q1_0_sidecar.up[layer_index];
  2569:         source.down = g_q1_0_sidecar.down[layer_index];
  2570:         source.sidecar = true;
  2571:         return source;
  2572:     }
  2573:     if (!iq1_s_mixed_cold_one_requested() &&
  2574:         iq1_s_sidecar_layer_active(model, layer_index)) {
  2575:         source.model = g_iq1_s_sidecar.model;
  2576:         source.gate = g_iq1_s_sidecar.gate[layer_index];
  2577:         source.up = g_iq1_s_sidecar.up[layer_index];
  2578:         source.down = g_iq1_s_sidecar.down[layer_index];
  2579:         source.sidecar = true;
  2580:     }
  2581:     return source;
  2582: }
  2583: 
  2584: typedef struct {
  2585:     ds4_tensor *e_proj;
  3394: static void q1_0_sidecar_bind(
  3395:         ds4_model *model,
  3396:         const ds4_model *primary_model) {
  3397:     if (!model || !primary_model) ds4_die("missing Q1_0 sidecar model");
  3398:     memset(&g_q1_0_sidecar, 0, sizeof(g_q1_0_sidecar));
  3399:     config_validate_model(model);
  3400:     /* Hash-routed layers 0..2 are part of the closed Q1 snapshot too. */
  3401:     const uint32_t routed_first_layer = 0u;
  3402:     const uint32_t routed_last_layer = DS4_N_LAYER - 1u;
  3403:     uint32_t first_layer = routed_first_layer;
  3404:     uint32_t last_layer = routed_last_layer;
  3405:     const char *first_env = getenv("DS4_Q1_0_LAYER_FIRST");
  3406:     const char *last_env = getenv("DS4_Q1_0_LAYER_LAST");
  3407:     if ((first_env && first_env[0]) || (last_env && last_env[0])) {
  3408:         char *first_end = NULL;
  3409:         char *last_end = NULL;
  3410:         errno = 0;
  3411:         const unsigned long first = first_env && first_env[0]
  3412:             ? strtoul(first_env, &first_end, 10)
  3413:             : (unsigned long)routed_first_layer;
  3414:         const int first_errno = errno;
  3415:         errno = 0;
  3416:         const unsigned long last = last_env && last_env[0]
  3417:             ? strtoul(last_env, &last_end, 10)
  3418:             : (unsigned long)routed_last_layer;
  3419:         const int last_errno = errno;
  3420:         if (first_errno != 0 || last_errno != 0 ||
  3421:             (first_env && first_env[0] &&
  3422:              (first_end == first_env || *first_end != '\0')) ||
  3423:             (last_env && last_env[0] &&
  3424:              (last_end == last_env || *last_end != '\0')) ||
  3425:             first < routed_first_layer || first > routed_last_layer ||
  3426:             last < routed_first_layer || last > routed_last_layer ||
  3427:             first > last) {
  3428:             ds4_die("invalid Q1_0 sidecar layer range");
  3429:         }
  3430:         first_layer = (uint32_t)first;
  3431:         last_layer = (uint32_t)last;
  3432:     }
  3433: 
  3434:     iq1_s_sidecar_validate_checkpoint_identity(model, primary_model);
  3435:     q1_0_sidecar_validate_router_identity(
  3436:         model, primary_model, first_layer, last_layer);
  3437: 
  3438:     uint64_t routed_bytes = 0;
  3439:     for (uint32_t il = first_layer; il <= last_layer; il++) {
  3440:         ds4_tensor *gate = required_tensorf(
  3441:             model, "blk.%u.ffn_gate_exps.weight", il);
  3442:         ds4_tensor *up = required_tensorf(
  3443:             model, "blk.%u.ffn_up_exps.weight", il);
  3444:         ds4_tensor *down = required_tensorf(
  3445:             model, "blk.%u.ffn_down_exps.weight", il);
  3446:         tensor_expect_layout(gate, DS4_TENSOR_Q1_0, 3,
  3447:                              DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
  3448:         tensor_expect_layout(up, DS4_TENSOR_Q1_0, 3,
  3449:                              DS4_N_EMBD, DS4_N_FF_EXP, DS4_N_EXPERT);
  3450:         tensor_expect_layout(down, DS4_TENSOR_Q1_0, 3,
  3451:                              DS4_N_FF_EXP, DS4_N_EMBD, DS4_N_EXPERT);
  3452: 
  3453:         const uint64_t gate_row_bytes = routed_expert_row_bytes(gate);
  3454:         const uint64_t gate_expert_bytes =
  3455:             (uint64_t)DS4_N_FF_EXP * gate_row_bytes;
  3456:         const uint64_t down_row_bytes = routed_expert_row_bytes(down);
  3457:         const uint64_t down_expert_bytes =
  3458:             (uint64_t)DS4_N_EMBD * down_row_bytes;
  3459:         if (gate->abs_offset > model->size ||
  3460:             gate_expert_bytes > (model->size - gate->abs_offset) / DS4_N_EXPERT ||
  3461:             up->abs_offset > model->size ||
  3462:             gate_expert_bytes > (model->size - up->abs_offset) / DS4_N_EXPERT ||
  3463:             down->abs_offset > model->size ||
  3464:             down_expert_bytes > (model->size - down->abs_offset) / DS4_N_EXPERT) {
  3465:             ds4_die("Q1_0 sidecar routed expert tensor extends past file end");
  3466:         }
  3467: 
  3468:         g_q1_0_sidecar.gate[il] = gate;
  3469:         g_q1_0_sidecar.up[il] = up;
  3470:         g_q1_0_sidecar.down[il] = down;
  3471:         g_q1_0_sidecar.gate_row_bytes[il] = gate_row_bytes;
  3472:         g_q1_0_sidecar.gate_expert_bytes[il] = gate_expert_bytes;
  3473:         g_q1_0_sidecar.down_row_bytes[il] = down_row_bytes;
  3474:         g_q1_0_sidecar.down_expert_bytes[il] = down_expert_bytes;
  3475:         routed_bytes += 2u * DS4_N_EXPERT * gate_expert_bytes;
  3476:         routed_bytes += DS4_N_EXPERT * down_expert_bytes;
  3477:     }
  3478: 
  3479:     g_q1_0_sidecar.model = model;
  3480:     g_q1_0_sidecar.primary_model = primary_model;
  3481:     g_q1_0_sidecar.first_layer = first_layer;
  3482:     g_q1_0_sidecar.last_layer = last_layer;
  3483:     g_q1_0_sidecar.ready = true;
  3484:     fprintf(stderr,
  3485:             "ds4: Q1_0 routed-expert sidecar validated: "
  3486:             "layers=%u active=%u..%u gate_up_down=q1_0 routed-only block=128/18B "
  3487:             "routed_bytes=%llu dispatch=fail-closed\n",
  3488:             DS4_N_LAYER,
  3489:             first_layer,
  3490:             last_layer,
  3491:             (unsigned long long)routed_bytes);
  3492: }
  3493: 
  3494: static void mtp_weights_bind(ds4_mtp_weights *w, const ds4_model *m) {
  3495:     memset(w, 0, sizeof(*w));
 19430:     }
 19431:     vocab_load(&e->vocab, &e->model);
 19432:     config_validate_model(&e->model);
 19433:     weights_bind(&e->weights, &e->model);
 19434:     ds4_reap_mask_install_bake(&e->model, &e->weights);
 19435:     const char *nested_residual_path =
 19436:         getenv("DS4_NESTED_RESIDUAL_SIDECAR");
 19437:     if (nested_residual_path && nested_residual_path[0]) {
 19438:         const char *exact = getenv("DS4_NESTED_RESIDUAL_EXACT");
 19439:         const char *mask = getenv("DS4_REAP_MASK_FILE");
 19440:         const char *tier_router = getenv("DS4_CUDA_PREFILL_TIER_ROUTER");
 19441:         const char *verify_reconstruction =
 19442:             getenv("DS4_NESTED_RESIDUAL_VERIFY_RECONSTRUCTION");
 19443:         const char *benchmark_unverified =
 19444:             getenv("DS4_NESTED_RESIDUAL_BENCHMARK_UNVERIFIED");
 19445:         const char *iq1_path = getenv("DS4_IQ1_S_EXPERT_SIDECAR");
 19446:         const char *q1_path = getenv("DS4_Q1_0_EXPERT_SIDECAR");
 19447:         const bool nested_residual_verification_allowed =
 19448:             (verify_reconstruction && strcmp(verify_reconstruction, "1") == 0) ||
 19449:             (benchmark_unverified && strcmp(benchmark_unverified, "1") == 0);
 19450:         if (e->backend != DS4_BACKEND_CUDA || e->model.bake_embedded ||
 19451:             !exact || strcmp(exact, "1") != 0 ||
 19452:             !nested_residual_verification_allowed ||
 19453:             (mask && mask[0]) ||
 19454:             !tier_router || strcmp(tier_router, "open") != 0 ||
 19455:             (iq1_path && iq1_path[0]) || (q1_path && q1_path[0]) ||
 19456:             os_file_open_read(
 19457:                 &e->nested_residual_file, nested_residual_path) != 0) {
 19458:             fprintf(stderr,
 19459:                     "ds4: nested residual exact mode requires CUDA, a "
 19460:                     "non-baked primary, open router, verification or an "
 19461:                     "explicit unverified benchmark gate, no quant sidecar, "
 19462:                     "and a readable sidecar\n");
 19463:             ds4_engine_close(e);
 19464:             *out = NULL;
 19465:             return 1;
 19466:         }
 19467:         e->nested_residual_ready = true;
 19468:         fprintf(stderr,
 19469:                 "ds4: nested residual exact sidecar source: %s "
 19470:                 "router=open representation=authoritative-iq2\n",
 19471:                 nested_residual_path);
 19472:     }
 19473:     const char *q1_0_sidecar_path = getenv("DS4_Q1_0_EXPERT_SIDECAR");
 19474:     if (q1_0_sidecar_path && q1_0_sidecar_path[0]) {
 19475:         if (e->backend != DS4_BACKEND_CUDA || e->model.bake_embedded) {
 19476:             fprintf(stderr,
 19477:                     "ds4: Q1_0 expert sidecar requires CUDA and a non-baked primary model\n");
 19478:             ds4_engine_close(e);
 19479:             *out = NULL;
 19480:             return 1;
 19481:         }
 19482:         model_open(&e->q1_0_sidecar_model,
 19483:                    q1_0_sidecar_path,
 19484:                    graph_backend,
 19485:                    false);
 19486:         q1_0_sidecar_bind(&e->q1_0_sidecar_model, &e->model);
 19487:         e->q1_0_sidecar_ready = true;
 19488:         fprintf(stderr, "ds4: Q1_0 expert sidecar source: %s\n",
 19489:                 q1_0_sidecar_path);
 19490:     }
 19491:     const int q1_0_resident_arena = q1_0_resident_arena_requested();
 19492:     const int q1_0_dual_arena = q1_0_dual_arena_requested();
 19493:     const int q1_0_dual_sparse_companion =
 19494:         q1_0_dual_sparse_companion_requested();
 19495:     const int q1_0_mixed_cold_one = q1_0_mixed_cold_one_requested();
 19496:     const int q1_0_snapshot_backing = q1_0_snapshot_backing_requested();
 19497:     const int q1_0_pageable_overflow =
 19498:         q1_0_pageable_overflow_requested();
 19499:     const int q1_0_dynamic_promotion =
 19500:         q1_0_dynamic_promotion_requested();
 19501:     const char *q1_0_selected_load = getenv("DS4_Q1_0_SELECTED_LOAD");
 19502:     if (q1_0_resident_arena < 0 || q1_0_dual_arena < 0 ||
 19503:         q1_0_dual_sparse_companion < 0 || q1_0_mixed_cold_one < 0 ||
 19504:         q1_0_snapshot_backing < 0 || q1_0_pageable_overflow < 0 ||
 19505:         q1_0_dynamic_promotion < 0 ||
 19506:         (q1_0_dual_arena > 0 && q1_0_resident_arena <= 0) ||
 19507:         (q1_0_dual_sparse_companion > 0 &&
 19508:          (q1_0_dual_arena <= 0 || q1_0_resident_arena <= 0 ||
 19509:           q1_0_mixed_cold_one <= 0)) ||
 19510:         (q1_0_mixed_cold_one > 0 && q1_0_dual_sparse_companion <= 0) ||
 19511:         (q1_0_pageable_overflow > 0 &&
 19512:          q1_0_snapshot_backing <= 0 &&
 19513:          !(q1_0_resident_arena > 0 && q1_0_dual_arena > 0)) ||
 19514:         (q1_0_snapshot_backing > 0 &&
 19515:          (q1_0_resident_arena > 0 || q1_0_dual_arena > 0)) ||
 19516:         (q1_0_dynamic_promotion > 0 &&
 19517:          (q1_0_resident_arena <= 0 || q1_0_dual_arena <= 0 ||
 19518:           q1_0_dual_sparse_companion > 0 || q1_0_mixed_cold_one > 0 ||
 19519:           q1_0_snapshot_backing > 0)) ||
 19520:         ((q1_0_resident_arena > 0 || q1_0_snapshot_backing > 0) &&
 19521:          (!e->q1_0_sidecar_ready || !q1_0_selected_load ||
 19522:           strcmp(q1_0_selected_load, "1") != 0))) {
 19523:         fprintf(stderr,
 19524:                 "ds4: invalid Q1_0 arena configuration; dual sparse 5+1 "
 19525:                 "requires resident+dual+cold-one, pageable overflow requires "
 19526:                 "exclusive snapshot backing or resident dual-arena mode, and "
 19527:                 "dynamic promotion requires full resident dual-arena mode, "
 19528:                 "and every Q1_0 arena requires "
 19529:                 "a valid sidecar plus DS4_Q1_0_SELECTED_LOAD=1\n");
 19530:         ds4_engine_close(e);
 19531:         *out = NULL;
 19532:         return 1;
 19533:     }
 19534:     const char *iq1_s_sidecar_path = getenv("DS4_IQ1_S_EXPERT_SIDECAR");
 19535:     if (iq1_s_sidecar_path && iq1_s_sidecar_path[0]) {
 19536:         if (e->backend != DS4_BACKEND_CUDA || e->model.bake_embedded) {
 19537:             fprintf(stderr,
 19538:                     "ds4: IQ1_S expert sidecar requires CUDA and a non-baked primary model\n");
 19539:             ds4_engine_close(e);
 19540:             *out = NULL;
 19541:             return 1;
 19542:         }
 19543:         model_open(&e->iq1_s_sidecar_model,
 19544:                    iq1_s_sidecar_path,
 19545:                    graph_backend,
 19546:                    false);
 19547:         iq1_s_sidecar_bind(&e->iq1_s_sidecar_model, &e->model);
 19548:         e->iq1_s_sidecar_ready = true;
 19549:         fprintf(stderr, "ds4: IQ1_S expert sidecar source: %s\n",
 19550:                 iq1_s_sidecar_path);
 19551:     }
 19552:     if (e->q1_0_sidecar_ready && iq1_s_mixed_cold_one_requested()) {
 19553:         fprintf(stderr,
 19554:                 "ds4: Q1_0 expert sidecar is not compatible with DS4_IQ1_S_MIXED_COLD_K=1 in this smoke\n");
 19555:         ds4_engine_close(e);
 19556:         *out = NULL;
 19557:         return 1;
 19558:     }
 19559:     if (iq1_s_mixed_cold_one_requested() && !e->iq1_s_sidecar_ready) {
 19560:         fprintf(stderr,
 19561:                 "ds4: DS4_IQ1_S_MIXED_COLD_K=1 requires an IQ1_S expert sidecar\n");
 19562:         ds4_engine_close(e);
 19563:         *out = NULL;
 19564:         return 1;
 19565:     }
 19566:     if (iq1_s_mixed_cold_one_requested()) {
 19567:         fprintf(stderr,
 19568:                 "ds4: IQ1_S mixed decode fixture enabled: hot_main=5 cold_iq1=1 prefill=main\n");
 19569:     }
 19570:     if (e->backend == DS4_BACKEND_CPU && !cpu_load_directional_steering(e)) {
 19571:         ds4_engine_close(e);
 19572:         *out = NULL;
 19573:         return 1;
 19574:     }
 19575:     if (opt->mtp_path && opt->mtp_path[0]) {
 19576:         if (e->model.bake_embedded) {
 19577:             fprintf(stderr,
 19578:                     "ds4: sparse bake primary models are not compatible with MTP yet\n");
 19579:             ds4_engine_close(e);
 19580:             *out = NULL;
 19581:             return 1;
 19582:         }
 19583:         model_open(&e->mtp_model, opt->mtp_path, graph_backend, true);
 19584:         if (e->mtp_model.bake_embedded) {
 19585:             fprintf(stderr,
 19586:                     "ds4: sparse bake containers are not valid MTP support models\n");
 19587:             ds4_engine_close(e);
 19588:             *out = NULL;
 19589:             return 1;
 19590:         }
 19591:         mtp_weights_bind(&e->mtp_weights, &e->mtp_model);
 19592:         e->mtp_ready = true;
 19593:         fprintf(stderr, "ds4: MTP support model loaded: %s (draft=%d)\n",
 19594:                 opt->mtp_path,
 19595:                 e->mtp_draft_tokens);
 19596:     }
 19597: 
 19598: #ifndef DS4_NO_GPU
 19599:     if (e->backend == DS4_BACKEND_CUDA) {
 19600: #ifdef __APPLE__
 19601:         fprintf(stderr, "ds4: CUDA backend requested but this build is linked with Metal, not CUDA\n");
 19602:         ds4_engine_close(e);
 19603:         *out = NULL;
 19604:         return 1;
 19605: #endif
 19606:     }
 19607:     if (e->backend == DS4_BACKEND_METAL) {
 19608: #ifndef __APPLE__
 19609:         fprintf(stderr, "ds4: Metal backend requested but this build is linked with CUDA, not Metal\n");
 19610:         ds4_engine_close(e);
 19611:         *out = NULL;
 19612:         return 1;
 19613: #endif
 19614:     }
 19615:     if (graph_backend) {
 19616:         e->metal_ready = ds4_gpu_init() != 0;
 19617:         if (!e->metal_ready) {
 19618:             fprintf(stderr, "ds4: %s backend unavailable; aborting startup\n",
 19619:                     ds4_backend_name(e->backend));
 19620:             ds4_engine_close(e);
 19621:             *out = NULL;
 19622:             return 1;
 19623:         }
 19624:         ds4_gpu_set_quality(e->quality);
 19625:         if (e->model.bake_embedded) {
 19626: #ifdef __APPLE__
 19627:             fprintf(stderr,
 19628:                     "ds4: sparse bake runtime is currently CUDA-only\n");
 19629:             ds4_engine_close(e);
 19630:             *out = NULL;
 19631:             return 1;
 19632: #else
 19633:             if (e->backend != DS4_BACKEND_CUDA ||
 19634:                 !ds4_gpu_sparse_bake_set_retained_mask(
 19635:                     e->model.bake.retained_mask,
 19636:                     DS4_N_LAYER,
 19637:                     DS4_N_EXPERT,
 19638:                     DS4_BAKE_MASK_LEN))
 19639:             {
 19640:                 fprintf(stderr,
 19641:                         "ds4: failed to install sparse bake CUDA guards\n");
 19642:                 ds4_engine_close(e);
 19643:                 *out = NULL;
 19644:                 return 1;
 19645:             }
 19646: #endif
 19647:         }
 19648:         (void)ds4_gpu_set_model_file(&e->model.mmap.file);
 19649:         if (e->nested_residual_ready &&
 19650:             !ds4_gpu_set_nested_residual_sidecar(
 19651:                 &e->nested_residual_file,
 19652:                 e->model.map,
 19653:                 e->model.size)) {
 19654:             fprintf(stderr,
 19655:                     "ds4: failed to install nested residual exact sidecar in CUDA\n");
 19656:             ds4_engine_close(e);
 19657:             *out = NULL;
 19658:             return 1;
 19659:         }
 19660:         if (e->iq1_s_sidecar_ready &&
 19661:             !ds4_gpu_set_iq1_s_sidecar(
 19662:                 &e->iq1_s_sidecar_model.mmap.file,
 19663:                 e->iq1_s_sidecar_model.map,
 19664:                 e->iq1_s_sidecar_model.size)) {
 19665:             fprintf(stderr,
 19666:                     "ds4: failed to install IQ1_S expert sidecar in CUDA\n");
 19667:             ds4_engine_close(e);
 19668:             *out = NULL;
 19669:             return 1;
 19670:         }
 19671:         if (e->q1_0_sidecar_ready &&
 19672:             !ds4_gpu_set_q1_0_sidecar(
 19673:                 &e->q1_0_sidecar_model.mmap.file,
 19674:                 e->q1_0_sidecar_model.map,
 19675:                 e->q1_0_sidecar_model.size)) {
 19676:             fprintf(stderr,
 19677:                     "ds4: failed to install Q1_0 expert sidecar in CUDA\n");
 19678:             ds4_engine_close(e);
 19679:             *out = NULL;
 19680:             return 1;
 19681:         }
 19682:         if (!ds4_gpu_set_model_map_range(e->model.map,
 19683:                                            e->model.size,
 19684:                                            e->model.tensor_data_pos,
 19685:                                            e->model.size - e->model.tensor_data_pos))
 19686:         {
 19687:             fprintf(stderr,
 19688:                     "ds4: %s failed to map model views; aborting startup. "
 19689:                     "This is commonly caused by insufficient memory or accelerator VM budget.\n",
 19690:                     ds4_backend_name(e->backend));
 20120:                                                e->directional_steering_file,
 20121:                                                e->directional_steering_attn_scale,
 20122:                                                e->directional_steering_ffn_scale)) {
 20123:         metal_graph_free(&s->graph);
 20124:         free(s);
 20125:         return 1;
 20126:     }
 20127:     if (e->backend == DS4_BACKEND_CUDA) {
 20128:         const int q1_0_resident_arena = q1_0_resident_arena_requested();
 20129:         const int q1_0_dual_arena = q1_0_dual_arena_requested();
 20130:         const int q1_0_dual_sparse_companion =
 20131:             q1_0_dual_sparse_companion_requested();
 20132:         const int q1_0_mixed_cold_one =
 20133:             q1_0_mixed_cold_one_requested();
 20134:         const int q1_0_snapshot_backing =
 20135:             q1_0_snapshot_backing_requested();
 20136:         const int q1_0_pageable_overflow =
 20137:             q1_0_pageable_overflow_requested();
 20138:         const int q1_0_dynamic_promotion =
 20139:             q1_0_dynamic_promotion_requested();
 20140:         const char *arena_gb_env = getenv("DS4_CUDA_DYNAMIC_ARENA_GB");
 20141:         const double arena_gb = arena_gb_env && arena_gb_env[0]
 20142:             ? strtod(arena_gb_env, NULL) : 0.0;
 20143:         const char *q1_0_arena_gb_env =
 20144:             getenv("DS4_Q1_0_DYNAMIC_ARENA_GB");
 20145:         const double q1_0_arena_gb =
 20146:             q1_0_arena_gb_env && q1_0_arena_gb_env[0]
 20147:                 ? strtod(q1_0_arena_gb_env, NULL) : arena_gb;
 20148:         if (q1_0_resident_arena < 0 || q1_0_dual_arena < 0 ||
 20149:             q1_0_dual_sparse_companion < 0 ||
 20150:             q1_0_mixed_cold_one < 0 ||
 20151:             q1_0_snapshot_backing < 0 || q1_0_pageable_overflow < 0 ||
 20152:             q1_0_dynamic_promotion < 0 ||
 20153:             (q1_0_dual_arena > 0 && q1_0_resident_arena <= 0) ||
 20154:             (q1_0_dual_sparse_companion > 0 &&
 20155:              (q1_0_dual_arena <= 0 || q1_0_resident_arena <= 0 ||
 20156:               q1_0_mixed_cold_one <= 0)) ||
 20157:             (q1_0_mixed_cold_one > 0 &&
 20158:              q1_0_dual_sparse_companion <= 0) ||
 20159:             (q1_0_pageable_overflow > 0 &&
 20160:              q1_0_snapshot_backing <= 0 &&
 20161:              !(q1_0_resident_arena > 0 && q1_0_dual_arena > 0)) ||
 20162:             (q1_0_snapshot_backing > 0 &&
 20163:              (q1_0_resident_arena > 0 || q1_0_dual_arena > 0)) ||
 20164:             (q1_0_dynamic_promotion > 0 &&
 20165:              (q1_0_resident_arena <= 0 || q1_0_dual_arena <= 0 ||
 20166:               q1_0_dual_sparse_companion > 0 ||
 20167:               q1_0_mixed_cold_one > 0 ||
 20168:               q1_0_snapshot_backing > 0))) {
 20169:             metal_graph_free(&s->graph);
 20170:             free(s);
 20171:             return 1;
 20172:         }
 20173:         if (q1_0_resident_arena > 0 && q1_0_arena_gb <= 0.0) {
 20174:             fprintf(stderr,
 20175:                     "ds4: Q1_0 resident arena requires "
 20176:                     "DS4_Q1_0_DYNAMIC_ARENA_GB>0 or "
 20177:                     "DS4_CUDA_DYNAMIC_ARENA_GB>0\n");
 20178:             metal_graph_free(&s->graph);
 20179:             free(s);
 20180:             return 1;
 20181:         }
 20182:         if (q1_0_snapshot_backing > 0 && arena_gb <= 0.0) {
 20183:             fprintf(stderr,
 20184:                     "ds4: Q1_0 snapshot backing requires DS4_CUDA_DYNAMIC_ARENA_GB>0\n");
 20185:             metal_graph_free(&s->graph);
 20186:             free(s);
 20187:             return 1;
 20188:         }
 20189:         if (q1_0_dual_arena > 0 && arena_gb <= 0.0) {
 20190:             fprintf(stderr,
 20191:                     "ds4: Q1_0 dual arena requires "
 20192:                     "DS4_CUDA_DYNAMIC_ARENA_GB>0 for exact IQ2 storage\n");
 20193:             metal_graph_free(&s->graph);
 20194:             free(s);
 20195:             return 1;
 20196:         }
 20197:         if (arena_gb > 0.0 || q1_0_arena_gb > 0.0) {
 20198:             ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER];
 20199:             const uint64_t requested = arena_gb >= (double)UINT64_MAX / 1073741824.0
 20200:                 ? UINT64_MAX : (uint64_t)(arena_gb * 1073741824.0);
 20201:             const uint64_t q1_0_requested =
 20202:                 q1_0_arena_gb >=
 20203:                         (double)UINT64_MAX / 1073741824.0
 20204:                     ? UINT64_MAX
 20205:                     : (uint64_t)(q1_0_arena_gb * 1073741824.0);
 20206:             if (q1_0_snapshot_backing > 0) {
 20207:                 uint64_t allocated = 0;
 20208:                 uint32_t slots = 0;
 20209:                 const bool primary_geometry_ready =
 20210:                     dynamic_arena_build_layers(
 20211:                         layers, &e->model, &e->weights);
 20212:                 const int primary_geometry_bound =
 20213:                     primary_geometry_ready &&
 20214:                     ds4_gpu_set_primary_moe_geometry(
 20215:                         e->model.map, e->model.size, layers,
 20216:                         DS4_N_LAYER, DS4_N_EXPERT);
 20217:                 const bool q1_geometry_ready = primary_geometry_bound &&
 20218:                     dynamic_arena_build_q1_0_layers(
 20219:                         layers, &e->q1_0_sidecar_model);
 20220:                 const int q1_bound = q1_geometry_ready &&
 20221:                     ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
 20222:                         e->q1_0_sidecar_model.map,
 20223:                         e->q1_0_sidecar_model.size,
 20224:                         layers, DS4_N_LAYER, DS4_N_EXPERT,
 20225:                         g_q1_0_sidecar.first_layer,
 20226:                         g_q1_0_sidecar.last_layer);
 20227:                 if (!q1_bound || !ds4_gpu_dynamic_arena_prepare(
 20228:                         requested, &allocated, &slots)) {
 20229:                     fprintf(stderr,
 20230:                             "ds4: Q1_0 sparse snapshot arena unavailable; failed closed\n");
 20231:                     ds4_gpu_dynamic_arena_release();
 20232:                     metal_graph_free(&s->graph);
 20233:                     free(s);
 20234:                     return 1;
 20235:                 }
 20236:                 fprintf(stderr,
 20237:                         "ds4: Q1_0 sparse snapshot arena ready slots=%u allocated=%.3f GiB router=unchanged publication=prefill-mass-wrap\n",
 20238:                         slots, (double)allocated / 1073741824.0);
 20239:             } else if (q1_0_resident_arena <= 0 || q1_0_dual_arena > 0) {
 20240:                 uint64_t allocated = 0;
 20241:                 uint32_t slots = 0;
 20242:                 const bool primary_ready =
 20243:                     dynamic_arena_build_layers(layers, &e->model, &e->weights);
 20244:                 const int primary_bound = primary_ready &&
 20245:                     ds4_gpu_dynamic_arena_bind(
 20246:                         e->model.map, e->model.size, layers,
 20247:                         DS4_N_LAYER, DS4_N_EXPERT);
 20248:                 if (!primary_bound || !ds4_gpu_dynamic_arena_prepare(
 20249:                         requested, &allocated, &slots)) {
 20250:                     fprintf(stderr,
 20251:                             "ds4: CUDA primary dynamic arena requested but unavailable%s\n",
 20252:                             q1_0_dual_arena > 0
 20253:                                 ? "; Q1_0 dual mode failed closed"
 20254:                                 : "; continuing with pageable fallback");
 20255:                     if (q1_0_dual_arena > 0) {
 20256:                         ds4_gpu_dynamic_arena_release();
 20257:                         metal_graph_free(&s->graph);
 20258:                         free(s);
 20259:                         return 1;
 20260:                     }
 20261:                 } else {
 20262:                     const uint32_t keep = dynamic_arena_test_keep();
 20263:                     if (keep != 0) {
 20264:                         const bool published = dynamic_arena_wrap_fixture(
 20265:                             &e->model, layers, keep);
 20266:                         if (published &&
 20267:                             getenv("DS4_CUDA_DYNAMIC_ARENA_TEST_ABORT") != NULL) {
 20268:                             (void)dynamic_arena_abort_fixture(
 20269:                                 &e->model, layers, keep);
 20270:                         }
 20271:                     }
 20272:                 }
 20273:             }
 20274:             if (q1_0_resident_arena > 0 && q1_0_snapshot_backing <= 0) {
 20275:                 uint64_t q1_allocated = 0;
 20276:                 uint32_t q1_slots = 0;
 20277:                 uint64_t q1_generation = 0;
 20278:                 const bool q1_geometry_ready =
 20279:                     dynamic_arena_build_q1_0_layers(
 20280:                         layers, &e->q1_0_sidecar_model);
 20281:                 const int q1_bound = q1_geometry_ready &&
 20282:                     ds4_gpu_dynamic_arena_bind_q1_0(
 20283:                         e->q1_0_sidecar_model.map,
 20284:                         e->q1_0_sidecar_model.size,
 20285:                         layers, DS4_N_LAYER, DS4_N_EXPERT,
 20286:                         g_q1_0_sidecar.first_layer,
 20287:                         g_q1_0_sidecar.last_layer);
 20288:                 const int q1_prepared = q1_bound &&
 20289:                     (q1_0_dual_sparse_companion > 0 ||
 20290:                      ds4_gpu_dynamic_arena_prepare_q1_0(
 20291:                         q1_0_requested, &q1_allocated, &q1_slots,
 20292:                         &q1_generation));
 20293:                 if (!q1_prepared) {
 20294:                     fprintf(stderr,
 20295:                             "ds4: Q1_0 resident arena unavailable; failed closed\n");

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 20274 20410" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 683ms:
 20274:             if (q1_0_resident_arena > 0 && q1_0_snapshot_backing <= 0) {
 20275:                 uint64_t q1_allocated = 0;
 20276:                 uint32_t q1_slots = 0;
 20277:                 uint64_t q1_generation = 0;
 20278:                 const bool q1_geometry_ready =
 20279:                     dynamic_arena_build_q1_0_layers(
 20280:                         layers, &e->q1_0_sidecar_model);
 20281:                 const int q1_bound = q1_geometry_ready &&
 20282:                     ds4_gpu_dynamic_arena_bind_q1_0(
 20283:                         e->q1_0_sidecar_model.map,
 20284:                         e->q1_0_sidecar_model.size,
 20285:                         layers, DS4_N_LAYER, DS4_N_EXPERT,
 20286:                         g_q1_0_sidecar.first_layer,
 20287:                         g_q1_0_sidecar.last_layer);
 20288:                 const int q1_prepared = q1_bound &&
 20289:                     (q1_0_dual_sparse_companion > 0 ||
 20290:                      ds4_gpu_dynamic_arena_prepare_q1_0(
 20291:                         q1_0_requested, &q1_allocated, &q1_slots,
 20292:                         &q1_generation));
 20293:                 if (!q1_prepared) {
 20294:                     fprintf(stderr,
 20295:                             "ds4: Q1_0 resident arena unavailable; failed closed\n");
 20296:                     ds4_gpu_dynamic_arena_release();
 20297:                     metal_graph_free(&s->graph);
 20298:                     free(s);
 20299:                     return 1;
 20300:                 }
 20301:                 const uint32_t required_slots =
 20302:                     (g_q1_0_sidecar.last_layer -
 20303:                      g_q1_0_sidecar.first_layer + 1u) * DS4_N_EXPERT;
 20304:                 if (q1_0_dual_sparse_companion > 0) {
 20305:                     fprintf(stderr,
 20306:                             "ds4: Q1_0 dual sparse companion deferred until "
 20307:                             "prefill candidate publication storage=pageable "
 20308:                             "router=unchanged policy=lowest-weight-one\n");
 20309:                 } else if (q1_slots != required_slots ||
 20310:                            q1_generation == 0) {
 20311:                     fprintf(stderr,
 20312:                             "ds4: Q1_0 resident arena published invalid snapshot slots=%u required=%u generation=%" PRIu64 "; failed closed\n",
 20313:                             q1_slots, required_slots, q1_generation);
 20314:                     ds4_gpu_dynamic_arena_release();
 20315:                     metal_graph_free(&s->graph);
 20316:                     free(s);
 20317:                     return 1;
 20318:                 }
 20319:             }
 20320:         }
 20321:     }
 20322:     s->logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->logits[0]));
 20323:     if (e->mtp_ready) {
 20324:         s->mtp_logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(s->mtp_logits[0]));
 20325:         s->mtp_draft_token = -1;
 20326:     }
 20327:     *out = s;
 20328:     return 0;
 20329: #endif
 20330: }
 20331: 
 20332: void ds4_session_free(ds4_session *s) {
 20333:     if (!s) return;
 20334:     if (ds4_session_is_cpu(s)) {
 20335:         kv_cache_free(&s->cpu_cache);
 20336:         cpu_decode_scratch_free(&s->cpu_scratch);
 20337:     }
 20338: #ifndef DS4_NO_GPU
 20339:     else {
 20340:         metal_graph_free(&s->graph);
 20341:     }
 20342: #endif
 20343:     token_vec_free(&s->checkpoint);
 20344:     free(s->logits);
 20345:     free(s->mtp_logits);
 20346:     free(s);
 20347: }
 20348: 
 20349: void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud) {
 20350:     if (!s) return;
 20351:     s->progress = fn;
 20352:     s->progress_ud = ud;
 20353: }
 20354: 
 20355: #ifndef DS4_NO_GPU
 20356: typedef struct {
 20357:     ds4_session *session;
 20358:     const ds4_tokens *prompt;
 20359:     ds4_session_progress_fn user;
 20360:     void *user_ud;
 20361: } ds4_sync_progress;
 20362: 
 20363: static void ds4_session_note_prefill_progress(void *ud, const char *event, int current, int total) {
 20364:     ds4_sync_progress *p = ud;
 20365:     if (!p || !p->session || !p->prompt) return;
 20366:     if (!strcmp(event, "prefill_chunk") && current > 0 && current <= p->prompt->len) {
 20367:         p->session->checkpoint.len = 0;
 20368:         for (int i = 0; i < current; i++) token_vec_push(&p->session->checkpoint, p->prompt->v[i]);
 20369:         p->session->checkpoint_valid = true;
 20370:         p->session->mtp_draft_valid = false;
 20371:     }
 20372:     if (p->user) p->user(p->user_ud, event, current, total);
 20373: }
 20374: #endif
 20375: 
 20376: /* Bring the live backend state to exactly the supplied token prefix.
 20377:  *
 20378:  * ds4-server and the REPL are stateless at the text/API layer but stateful here:
 20379:  * they resend or rebuild the full transcript, and this function decides whether
 20380:  * the live checkpoint is a prefix.  A matching prefix is extended in one of two
 20381:  * ways:
 20382:  *
 20383:  *   - long suffix: batched layer-major prefill, aligned to absolute chunk
 20384:  *     boundaries so compressor/indexer rows finalize in the same order as a
 20385:  *     cold prompt;
 20386:  *   - short suffix: ordinary one-token decode, which is faster below the
 20387:  *     measured crossover and preserves exact autoregressive semantics.
 20388:  *
 20389:  * A non-matching prompt discards the checkpoint and prefills from token zero.
 20390:  */
 20391: int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen) {
 20392:     if (!s || !prompt || prompt->len <= 0 || prompt->len >= s->ctx_size) {
 20393:         snprintf(err, errlen, "prompt exceeds context");
 20394:         return 1;
 20395:     }
 20396:     if (ds4_session_is_cpu(s)) {
 20397:         ds4_engine *e = s->engine;
 20398:         ds4_reap_mask_poll(&e->model, &e->weights, false);
 20399:         if (s->checkpoint_valid &&
 20400:             prompt->len >= s->checkpoint.len &&
 20401:             ds4_tokens_starts_with(prompt, &s->checkpoint))
 20402:         {
 20403:             s->mtp_draft_valid = false;
 20404:             for (int i = s->checkpoint.len; i < prompt->len; i++) {
 20405:                 ds4_reap_mask_poll(&e->model, &e->weights, false);
 20406:                 forward_token_raw_swa_cpu_decode_scratch(s->logits,
 20407:                                                          &e->model,
 20408:                                                          &e->weights,
 20409:                                                          &s->cpu_cache,
 20410:                                                          prompt->v[i],

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "''^(static|extern '"\\\"C\\\") .*q1_0|"'^static .*dynamic_arena|''^static .*prefill|''^static .*bootstrap" '"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Where-Object { "'$n=[int](($_ -split '"':')[1]); "'$n -ge 6000 -and $n -le 8100 }' in C:\Users\imanu\g130i\prototype_spec
2026-07-21T09:50:38.341046Z ERROR codex_core::tools::router: error=Exit code: 1
Wall time: 0.3 seconds
Output:
rg: regex parse error:
    (?:^(static|extern " C\))
    ^
error: unclosed group

 exited 1 in 292ms:
rg: regex parse error:
    (?:^(static|extern " C\))
    ^
error: unclosed group

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$rows = rg -n '"'cuda_q1_0_[A-Za-z0-9_]+\\(' 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; foreach ("'$row in $rows) { if ($row -match '"':(\\d+):' -and [int]"'$Matches[1] -ge 6000 -and [int]$Matches[1] -le 8100) { $row } }' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 300ms:
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n 'q1_0_[A-Za-z0-9_]+\\(' 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Select-Object -First 300" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 503ms:
568:static void cuda_q1_0_source_unlock_begin(
570:static void cuda_q1_0_source_unlock_bootstrap_layer(
575:static void cuda_q1_0_source_unlock_finish(
579:static void cuda_q1_0_profile_source_working_set_sample(
583:static void cuda_q1_0_ssd_wrap_working_set_sample(const char *phase);
584:static void cuda_q1_0_ssd_wrap_release(int report);
585:static int cuda_q1_0_ssd_wrap_init(void);
586:static void cuda_q1_0_ssd_wrap_flush(void);
1053:static int cuda_q1_0_resident_arena_requested(void) {
1058:static int cuda_q1_0_dual_arena_requested(void) {
1063:static int cuda_q1_0_dual_sparse_companion_requested(void) {
1068:static int cuda_q1_0_mixed_cold_one_requested(void) {
1073:static int cuda_q1_0_snapshot_backing_requested(void) {
1078:static int cuda_q1_0_pageable_overflow_requested(void) {
1083:static int cuda_q1_0_dynamic_promotion_requested(void) {
1096:static int cuda_q1_0_ssd_wrap_requested(void) {
1106:static int cuda_q1_0_ssd_wrap_double_env(
1130:static int cuda_q1_0_ssd_wrap_u32_env(
1153:static int cuda_q1_0_ssd_wrap_arena_partition(
1159:    const int requested = cuda_q1_0_ssd_wrap_requested();
1161:    if (!cuda_q1_0_dynamic_promotion_requested()) {
1169:        !cuda_q1_0_ssd_wrap_double_env(
1964:static int cuda_q1_0_mixed_exact_iq2_resolver_requested(void) {
1965:    return cuda_q1_0_snapshot_backing_requested() ||
1966:        (cuda_q1_0_resident_arena_requested() &&
1967:         cuda_q1_0_dual_arena_requested() &&
1968:         !cuda_q1_0_dual_sparse_companion_requested());
1971:static int cuda_q1_0_resident_transport_requested(void) {
1972:    return cuda_q1_0_resident_arena_requested() ||
1973:        cuda_q1_0_snapshot_backing_requested();
1976:static int cuda_q1_0_exclusive_arena_active(void) {
1979:        !cuda_q1_0_dual_arena_requested();
1994:static int cuda_q1_0_dual_arena_layer_active(uint32_t layer) {
1996:        !cuda_q1_0_dynamic_promotion_requested() ||
2018:    return cuda_q1_0_resident_arena_requested() &&
2019:        cuda_q1_0_dual_arena_requested() &&
2043:static cuda_dynamic_arena *cuda_q1_0_route_arena(uint32_t layer) {
2044:    if (cuda_q1_0_snapshot_backing_requested() &&
2056:    if (cuda_q1_0_dual_arena_layer_active(layer)) {
2194:static const char *cuda_q1_0_mixed_router_mode_name(void);
3093:static int cuda_q1_0_profile_requested(void) {
3102:static void cuda_q1_0_mixed_profile_reset(int new_lifecycle) {
3106:    if (!cuda_q1_0_profile_requested()) return;
3112:static void cuda_q1_0_mixed_profile_record_entered(uint32_t routes) {
3118:static void cuda_q1_0_mixed_profile_record_timer_failure(void) {
3123:static void cuda_q1_0_mixed_profile_record_completed(
3173:static int cuda_q1_0_mixed_profile_timing_values_ok(
3206:static void cuda_q1_0_mixed_profile_finalize(
3208:    if (!cuda_q1_0_profile_requested()) return;
3264:        cuda_q1_0_mixed_profile_timing_values_ok(stats, bucket_seconds);
3683:static void cuda_q1_0_sidecar_clear(void) {
3684:    cuda_q1_0_mixed_profile_reset(0);
3709:            cuda_q1_0_dynamic_promotion_requested() ? "on" : "off";
3736:                cuda_q1_0_mixed_router_mode_name(),
3751:                cuda_q1_0_resident_transport_requested(),
3758:                cuda_q1_0_resident_transport_requested()
3759:                    ? (cuda_q1_0_snapshot_backing_requested()
3761:                        : (cuda_q1_0_dual_arena_requested()
3764:                cuda_q1_0_resident_transport_requested()
3765:                    ? (cuda_q1_0_snapshot_backing_requested()
3767:                        : (cuda_q1_0_dual_arena_requested()
5744:    cuda_q1_0_mixed_profile_reset(1);
5767:            cuda_q1_0_profile_requested()) {
5774:    cuda_q1_0_mixed_profile_finalize(cleanup_sync_error);
5855:    cuda_q1_0_sidecar_clear();
6031:        cuda_q1_0_profile_requested();
6095:    cuda_q1_0_ssd_wrap_release(0);
6136:static void cuda_q1_0_dynamic_arena_release(int report) {
6138:    if (report && arena.host_base && cuda_q1_0_profile_requested()) {
6199:    cuda_q1_0_mixed_profile_reset(0);
6204:    cuda_q1_0_ssd_wrap_release(report);
6257:    cuda_q1_0_dynamic_arena_release(1);
6406:    if (!cuda_q1_0_resident_arena_requested() ||
6466:    cuda_q1_0_dynamic_arena_release(0);
6485:            cuda_q1_0_dual_arena_requested() ? "enabled" : "disabled",
6486:            cuda_q1_0_dual_arena_requested() ? "dual-arena" : "disabled");
6490:extern "C" int ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
6495:    if (!cuda_q1_0_snapshot_backing_requested() ||
6538:        cuda_q1_0_pageable_overflow_requested() &&
6539:        cuda_q1_0_dual_arena_requested();
6684:    cuda_q1_0_mixed_profile_reset(1);
6685:    cuda_q1_0_profile_source_working_set_sample(
6688:    cuda_q1_0_source_unlock_begin(&source_unlock);
6724:        cuda_q1_0_source_unlock_bootstrap_layer(
6727:    cuda_q1_0_profile_source_working_set_sample(
6729:    cuda_q1_0_source_unlock_finish(
6732:    if (cuda_q1_0_profile_requested()) Sleep(1);
6734:    cuda_q1_0_profile_source_working_set_sample(
6776:            cuda_q1_0_dual_arena_requested() ? "enabled" : "disabled",
6777:            cuda_q1_0_dual_arena_requested() ? "dual-arena" : "disabled",
6845:    const int ssd_wrap_requested = cuda_q1_0_ssd_wrap_requested();
6852:        cuda_q1_0_snapshot_backing_requested() &&
6853:        cuda_q1_0_pageable_overflow_requested();
6878:        if (!cuda_q1_0_ssd_wrap_arena_partition(
7373:            cuda_q1_0_snapshot_backing_requested();
7625:static int cuda_q1_0_dual_sparse_copy_load(
7672:static void *cuda_q1_0_dual_sparse_copy_worker(void *arg) {
7679:        context->success[cursor] = cuda_q1_0_dual_sparse_copy_load(
7685:static void cuda_q1_0_dual_sparse_abort(const char *reason) {
7700:static int cuda_q1_0_dual_sparse_stage(
7702:    if (!cuda_q1_0_dual_sparse_companion_requested()) return 1;
7706:        !cuda_q1_0_resident_arena_requested() ||
7707:        !cuda_q1_0_dual_arena_requested() ||
7708:        !cuda_q1_0_mixed_cold_one_requested() ||
7718:        cuda_q1_0_dual_sparse_abort("contract");
7728:            cuda_q1_0_dual_sparse_abort("target-layer");
7736:        cuda_q1_0_dual_sparse_abort("capacity");
7742:        cuda_q1_0_dual_sparse_abort("size");
7748:            cuda_q1_0_dual_sparse_abort("upload-sync");
7770:            cuda_q1_0_dual_sparse_abort("pageable-allocation");
7791:        cuda_q1_0_dual_sparse_abort("metadata-allocation");
7840:    (void)cuda_q1_0_dual_sparse_copy_worker(&context);
7846:            cuda_q1_0_dual_sparse_abort("copy");
7884:static int cuda_q1_0_dual_sparse_commit(
7887:    if (!cuda_q1_0_dual_sparse_companion_requested()) return 1;
7899:        cuda_q1_0_dual_sparse_abort("commit-contract");
7911:            cuda_q1_0_dual_sparse_abort("primary-parity");
7917:            cuda_q1_0_dual_sparse_abort("q1-stage-parity");
7924:            cuda_q1_0_dual_sparse_abort("q1-slot-parity");
7935:        cuda_q1_0_dual_sparse_abort("entry-count");
9034:static void cuda_q1_0_profile_source_working_set_sample(
9038:    if (!cuda_q1_0_profile_requested()) return;
9148:static void cuda_q1_0_source_unlock_snapshot_store(
9172:static void cuda_q1_0_source_unlock_begin(
9182:    cuda_q1_0_source_unlock_snapshot_store(stats, 0);
9186:static void cuda_q1_0_source_unlock_bootstrap_range(
9241:static void cuda_q1_0_source_unlock_bootstrap_layer(
9249:    cuda_q1_0_source_unlock_bootstrap_range(
9251:    cuda_q1_0_source_unlock_bootstrap_range(
9253:    cuda_q1_0_source_unlock_bootstrap_range(
9264:static void cuda_q1_0_source_unlock_finish(
9270:    cuda_q1_0_source_unlock_snapshot_store(stats, 1);
9555:    if ((cuda_q1_0_exclusive_arena_active() ||
11404:        !cuda_q1_0_snapshot_backing_requested()) return;
11414:    if (cuda_q1_0_dual_sparse_companion_requested()) {
11415:        cuda_q1_0_dual_sparse_abort("prefill-compose");
11422:    if (cuda_q1_0_exclusive_arena_active()) {
11512:    if (cuda_q1_0_exclusive_arena_active()) {
11612:        if (cuda_q1_0_dual_sparse_companion_requested()) {
11613:            if (!cuda_q1_0_dual_sparse_stage(
11626:                !cuda_q1_0_dual_sparse_commit(
11645:                cuda_q1_0_dual_sparse_abort("primary-publish");
11743:        cuda_q1_0_snapshot_backing_requested();
12078:             cuda_q1_0_snapshot_backing_requested());
12249:    if (cuda_q1_0_exclusive_arena_active()) {
12286:    cuda_q1_0_ssd_wrap_flush();
12293:    if (cuda_q1_0_exclusive_arena_active()) {
12323:    if (cuda_q1_0_exclusive_arena_active()) {
13441:extern "C" int ds4_gpu_set_q1_0_sidecar(
13445:    cuda_q1_0_sidecar_clear();
13463:        cuda_q1_0_sidecar_clear();
13469:        cuda_q1_0_sidecar_clear();
13475:    cuda_q1_0_mixed_profile_reset(1);
19613:__device__ static float dev_dequant_q1_0_value(
19620:__device__ static float dev_dot_q1_0_q8_K_half128(
19651:__device__ static float dev_dot_q1_0_q8_K_pair(
19655:    return dev_dot_q1_0_q8_K_half128(x0, y, 0u) +
19656:           dev_dot_q1_0_q8_K_half128(x1, y, 1u);
20135:__global__ static void moe_gate_up_mid_q1_0_qwarp32_kernel(
20181:        gate += dev_dot_q1_0_q8_K_pair(gr + 2u * b, gr + 2u * b + 1u, xqb + b);
20182:        up += dev_dot_q1_0_q8_K_pair(ur + 2u * b, ur + 2u * b + 1u, xqb + b);
20199:__global__ static void moe_down_q1_0_qwarp32_kernel(
20233:        acc += dev_dot_q1_0_q8_K_pair(wr + 2u * b, wr + 2u * b + 1u, xq + b);
22668:static const char *cuda_q1_0_mixed_router_mode_name(void) {
22674:static void cuda_q1_0_mixed_capture_router_mode(void) {
22676:        !cuda_q1_0_snapshot_backing_requested() &&
22732:static int cuda_q1_0_ssd_wrap_slot_reserved(uint32_t slot) {
23049:static int cuda_q1_0_promotion_probation_slots_requested(void) {
23061:    const int q1_0_slots = cuda_q1_0_promotion_probation_slots_requested();
23062:    const int q1_0_requested = cuda_q1_0_dynamic_promotion_requested();
23237:    cuda_q1_0_ssd_wrap_flush();
23493:        cuda_q1_0_dynamic_promotion_requested();
23499:    if (cuda_q1_0_exclusive_arena_active() &&
23545:         !cuda_q1_0_dual_arena_requested() ||
23546:         cuda_q1_0_snapshot_backing_requested())) {
23900:            if (cuda_q1_0_ssd_wrap_slot_reserved(slot)) continue;
23925:                cuda_q1_0_ssd_wrap_slot_reserved(entry.ram_slot) ||
24004:static void cuda_q1_0_promotion_record_emit(
24010:static int cuda_q1_0_promotion_checked_expert_range(
24026:static int cuda_q1_0_promotion_checked_triplet(
24038:    if (!cuda_q1_0_promotion_checked_expert_range(
24041:        !cuda_q1_0_promotion_checked_expert_range(
24044:        !cuda_q1_0_promotion_checked_expert_range(
24053:static int cuda_q1_0_promotion_record_context_init(
24098:    return cuda_q1_0_promotion_checked_triplet(
24108:        cuda_q1_0_promotion_checked_triplet(
24220:static const char *cuda_q1_0_ssd_wrap_state_name(
24232:static int cuda_q1_0_ssd_wrap_sha_env_valid(const char *name) {
24245:static uint64_t cuda_q1_0_ssd_wrap_queue_depth_locked(void) {
24253:static void cuda_q1_0_ssd_wrap_working_set_sample(const char *phase) {
24350:static int cuda_q1_0_ssd_wrap_read_job(
24400:static void *cuda_q1_0_ssd_wrap_worker(void *) {
24496:            const int read_ok = cuda_q1_0_ssd_wrap_read_job(
24528:        const uint64_t queue_depth = cuda_q1_0_ssd_wrap_queue_depth_locked();
24552:static int cuda_q1_0_ssd_wrap_init(void) {
24553:    const int requested = cuda_q1_0_ssd_wrap_requested();
24557:    if (!cuda_q1_0_dynamic_promotion_requested() ||
24559:        !cuda_q1_0_ssd_wrap_sha_env_valid("DS4_MODEL_SHA256") ||
24560:        !cuda_q1_0_ssd_wrap_sha_env_valid(
24584:    if (!cuda_q1_0_ssd_wrap_u32_env(
24587:        !cuda_q1_0_ssd_wrap_u32_env(
24590:        !cuda_q1_0_ssd_wrap_u32_env(
24593:        !cuda_q1_0_ssd_wrap_u32_env(
24596:        !cuda_q1_0_ssd_wrap_u32_env(
24599:        !cuda_q1_0_ssd_wrap_u32_env(
24602:        !cuda_q1_0_ssd_wrap_u32_env(
24605:        !cuda_q1_0_ssd_wrap_u32_env(
24608:        !cuda_q1_0_ssd_wrap_double_env(
24611:        !cuda_q1_0_ssd_wrap_double_env(
24649:        cuda_q1_0_ssd_wrap_release(0);
24658:            cuda_q1_0_ssd_wrap_release(0);
24665:        cuda_q1_0_ssd_wrap_release(0);
24694:    cuda_q1_0_ssd_wrap_working_set_sample("init");
24698:static void cuda_q1_0_ssd_wrap_job_release_locked(
24710:static int cuda_q1_0_ssd_wrap_finish_one(
24832:        cuda_q1_0_promotion_record_emit(
24853:        cuda_q1_0_promotion_record_emit(
24859:        cuda_q1_0_ssd_wrap_job_release_locked(&state.jobs[index]);
24866:static int cuda_q1_0_ssd_wrap_poll_internal(int force) {
24885:        if (!cuda_q1_0_ssd_wrap_finish_one(i, force)) ok = 0;
24890:static int cuda_q1_0_ssd_wrap_poll(void) {
24891:    return cuda_q1_0_ssd_wrap_poll_internal(0);
24894:static int cuda_q1_0_ssd_wrap_submit(
24897:    if (!record || !record->active || !cuda_q1_0_ssd_wrap_init() ||
24898:        !cuda_q1_0_ssd_wrap_poll()) {
25003:    const uint64_t depth = cuda_q1_0_ssd_wrap_queue_depth_locked();
25008:    cuda_q1_0_promotion_record_emit(
25016:static void cuda_q1_0_ssd_wrap_flush(void) {
25036:    (void)cuda_q1_0_ssd_wrap_poll_internal(1);
25037:    cuda_q1_0_ssd_wrap_working_set_sample("flush");
25040:static void cuda_q1_0_ssd_wrap_release(int report) {
25042:    if (state.enabled) cuda_q1_0_ssd_wrap_flush();
25062:        cuda_q1_0_ssd_wrap_working_set_sample("release");
25117:static int cuda_q1_0_ssd_wrap_host_is_pageable(
25138:static int cuda_q1_0_ssd_wrap_prepare_h2d_source(
25183:static int cuda_q1_0_ssd_wrap_record_h2d(
25253:    if (!cuda_q1_0_promotion_checked_triplet(
25295:        cuda_q1_0_promotion_record_emit(
25316:            cuda_q1_0_promotion_record_emit(
25330:                cuda_q1_0_promotion_record_emit(
25353:                cuda_q1_0_promotion_record_emit(
25370:                cuda_q1_0_promotion_record_emit(
25466:static int cuda_q1_0_promotion_structural_reject_reason(
25474:static void cuda_q1_0_promotion_record_emit(
25562:static void cuda_q1_0_promotion_record_emit(
25573:    (void)cuda_q1_0_promotion_record_context_init(
25581:    cuda_q1_0_promotion_record_emit(
25605:            cuda_q1_0_promotion_record_emit(
25630:            cuda_q1_0_promotion_record_emit(
25645:            cuda_q1_0_promotion_record_emit(
25717:        !cuda_q1_0_promotion_record_context_init(
25726:        cuda_q1_0_promotion_record_emit(
25752:        const int ssd_wrap_requested = cuda_q1_0_ssd_wrap_requested();
25758:            const int submitted = cuda_q1_0_ssd_wrap_submit(
25786:            cuda_q1_0_promotion_structural_reject_reason(
25790:            cuda_q1_0_promotion_record_emit(
25808:        cuda_q1_0_promotion_record_emit(
26256:        cuda_q1_0_mixed_exact_iq2_resolver_requested();
26298:    if (cuda_q1_0_snapshot_backing_requested() &&
26307:        !cuda_q1_0_snapshot_backing_requested() &&
26422:                cuda_q1_0_snapshot_backing_requested()
26445:                    cuda_q1_0_snapshot_backing_requested()
26519:            cuda_q1_0_snapshot_backing_requested()
26602:        cuda_q1_0_dynamic_promotion_requested();
27517:            const int pageable = cuda_q1_0_ssd_wrap_host_is_pageable(
27519:            if (pageable < 0 || !cuda_q1_0_ssd_wrap_prepare_h2d_source(
27606:            if (!cuda_q1_0_ssd_wrap_record_h2d(
27725:            if (!cuda_q1_0_ssd_wrap_record_h2d(
30144:        cuda_q1_0_snapshot_backing_requested();
31304:    if (cuda_q1_0_resident_transport_requested()) {
31306:            cuda_q1_0_route_arena(layer_index);
31377:        const int q1_profile = cuda_q1_0_profile_requested();
32227:            route_q1_0 && cuda_q1_0_profile_requested();
32239:                            cuda_q1_0_mixed_profile_record_timer_failure();
32421:                                cuda_q1_0_mixed_profile_record_timer_failure();
32434:                        cuda_q1_0_mixed_profile_record_timer_failure();
33650:static const char *cuda_q1_0_mixed_representation_name(
33661:static int cuda_q1_0_mixed_trace_requested(void) {
33666:static int cuda_q1_0_resident_ram_ptrs(
33669:    cuda_dynamic_arena *arena_ptr = cuda_q1_0_route_arena(layer);
33693:static int cuda_q1_0_dual_sparse_pair_contains(
33697:    if (!cuda_q1_0_dual_sparse_companion_requested() ||
33726:static cuda_q1_0_mixed_representation cuda_q1_0_mixed_resolve(
33728:    cuda_q1_0_mixed_capture_router_mode();
33729:    if (cuda_q1_0_snapshot_backing_requested()) {
33740:        return cuda_q1_0_resident_ram_ptrs(
33780:    if (cuda_q1_0_resident_ram_ptrs(
33787:extern "C" int ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
33837:    const int mixed_profile = cuda_q1_0_profile_requested();
33844:        cuda_q1_0_mixed_profile_record_entered(n_expert);
33854:        !cuda_q1_0_route_arena(layer_index) ||
33872:                cuda_q1_0_route_arena(layer_index) != NULL,
33881:    cuda_dynamic_arena *q1_arena = cuda_q1_0_route_arena(layer_index);
33918:    if (!cuda_q1_0_ssd_wrap_poll()) {
33930:        cuda_q1_0_dynamic_promotion_requested();
33932:        cuda_q1_0_mixed_exact_iq2_resolver_requested();
33949:        !cuda_q1_0_snapshot_backing_requested() &&
33950:        !cuda_q1_0_dual_sparse_companion_requested() &&
33967:                cuda_q1_0_route_arena(layer_index) != NULL,
33977:        (cuda_q1_0_snapshot_backing_requested() ||

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' 6320 6795" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 608ms:
  6320:         int same = 1;
  6321:         for (uint32_t il = 0; il < n_layer; il++) {
  6322:             if (memcmp(&g_dynamic_arena.layers[il], &layers[il],
  6323:                        sizeof(layers[il])) != 0) {
  6324:                 same = 0;
  6325:                 break;
  6326:             }
  6327:         }
  6328:         if (same) return 1;
  6329:     }
  6330:     cuda_primary_dynamic_arena_release(1);
  6331:     try {
  6332:         g_dynamic_arena.layers.assign(layers, layers + n_layer);
  6333:     } catch (...) {
  6334:         fprintf(stderr,
  6335:                 "ds4: CUDA dynamic arena layer metadata allocation failed\n");
  6336:         return 0;
  6337:     }
  6338:     g_dynamic_arena.model_map = model_map;
  6339:     g_dynamic_arena.model_size = model_size;
  6340:     g_dynamic_arena.n_layer = n_layer;
  6341:     g_dynamic_arena.n_expert = n_expert;
  6342:     g_dynamic_arena.slot_bytes = slot_bytes;
  6343:     g_dynamic_arena.active_layer_first = active_layer_first;
  6344:     g_dynamic_arena.active_layer_last = active_layer_last;
  6345:     g_dynamic_arena.backing = backing;
  6346:     return 1;
  6347: }
  6348: 
  6349: extern "C" int ds4_gpu_dynamic_arena_bind(
  6350:         const void *model_map, uint64_t model_size,
  6351:         const ds4_gpu_dynamic_arena_layer *layers,
  6352:         uint32_t n_layer, uint32_t n_expert) {
  6353:     return cuda_dynamic_arena_bind(
  6354:         model_map, model_size, layers, n_layer, n_expert,
  6355:         CUDA_DYNAMIC_ARENA_BACKING_PRIMARY, 0u, n_layer ? n_layer - 1u : 0u);
  6356: }
  6357: 
  6358: extern "C" int ds4_gpu_set_primary_moe_geometry(
  6359:         const void *model_map, uint64_t model_size,
  6360:         const ds4_gpu_dynamic_arena_layer *layers,
  6361:         uint32_t n_layer, uint32_t n_expert) {
  6362:     if (!model_map || model_map != g_model_host_base ||
  6363:         model_size == 0 || model_size != g_model_registered_size ||
  6364:         !layers || n_layer == 0 || n_expert == 0) {
  6365:         return 0;
  6366:     }
  6367:     for (uint32_t layer = 0; layer < n_layer; layer++) {
  6368:         const ds4_gpu_dynamic_arena_layer &geometry = layers[layer];
  6369:         if (geometry.gate_expert_bytes == 0 ||
  6370:             geometry.up_expert_bytes == 0 ||
  6371:             geometry.down_expert_bytes == 0 ||
  6372:             geometry.gate_expert_bytes > UINT64_MAX / n_expert ||
  6373:             geometry.up_expert_bytes > UINT64_MAX / n_expert ||
  6374:             geometry.down_expert_bytes > UINT64_MAX / n_expert ||
  6375:             !cuda_dynamic_arena_range_valid(
  6376:                 geometry.gate_offset,
  6377:                 geometry.gate_expert_bytes, n_expert, model_size) ||
  6378:             !cuda_dynamic_arena_range_valid(
  6379:                 geometry.up_offset,
  6380:                 geometry.up_expert_bytes, n_expert, model_size) ||
  6381:             !cuda_dynamic_arena_range_valid(
  6382:                 geometry.down_offset,
  6383:                 geometry.down_expert_bytes, n_expert, model_size)) {
  6384:             return 0;
  6385:         }
  6386:     }
  6387:     try {
  6388:         g_primary_moe_geometry.assign(layers, layers + n_layer);
  6389:     } catch (...) {
  6390:         return 0;
  6391:     }
  6392:     g_primary_moe_geometry_map = model_map;
  6393:     g_primary_moe_geometry_size = model_size;
  6394:     g_primary_moe_geometry_n_expert = n_expert;
  6395:     fprintf(stderr,
  6396:             "ds4: primary IQ2 MoE geometry catalog ready: layers=%u experts=%u storage=metadata-only\n",
  6397:             n_layer, n_expert);
  6398:     return 1;
  6399: }
  6400: 
  6401: extern "C" int ds4_gpu_dynamic_arena_bind_q1_0(
  6402:         const void *model_map, uint64_t model_size,
  6403:         const ds4_gpu_dynamic_arena_layer *layers,
  6404:         uint32_t n_layer, uint32_t n_expert,
  6405:         uint32_t active_layer_first, uint32_t active_layer_last) {
  6406:     if (!cuda_q1_0_resident_arena_requested() ||
  6407:         !g_q1_0_sidecar_file_valid ||
  6408:         model_map != g_q1_0_sidecar_host_base ||
  6409:         model_size != g_q1_0_sidecar_size) {
  6410:         fprintf(stderr,
  6411:                 "ds4: [q1-0-resident-arena] result=failed reason=sidecar-binding\n");
  6412:         return 0;
  6413:     }
  6414:     if (!layers || n_layer == 0 || n_expert == 0 ||
  6415:         active_layer_first >= n_layer ||
  6416:         active_layer_last < active_layer_first ||
  6417:         active_layer_last >= n_layer) {
  6418:         return 0;
  6419:     }
  6420:     uint64_t slot_bytes = 0;
  6421:     for (uint32_t il = active_layer_first; il <= active_layer_last; il++) {
  6422:         const ds4_gpu_dynamic_arena_layer &layer = layers[il];
  6423:         if (!cuda_dynamic_arena_range_valid(layer.gate_offset,
  6424:                                              layer.gate_expert_bytes,
  6425:                                              n_expert, model_size) ||
  6426:             !cuda_dynamic_arena_range_valid(layer.up_offset,
  6427:                                              layer.up_expert_bytes,
  6428:                                              n_expert, model_size) ||
  6429:             !cuda_dynamic_arena_range_valid(layer.down_offset,
  6430:                                              layer.down_expert_bytes,
  6431:                                              n_expert, model_size) ||
  6432:             layer.gate_expert_bytes > UINT64_MAX - layer.up_expert_bytes ||
  6433:             layer.gate_expert_bytes + layer.up_expert_bytes >
  6434:                 UINT64_MAX - layer.down_expert_bytes) {
  6435:             return 0;
  6436:         }
  6437:         const uint64_t layer_slot_bytes = layer.gate_expert_bytes +
  6438:             layer.up_expert_bytes + layer.down_expert_bytes;
  6439:         if (il == active_layer_first) slot_bytes = layer_slot_bytes;
  6440:         if (layer_slot_bytes != slot_bytes) {
  6441:             fprintf(stderr,
  6442:                     "ds4: Q1_0 resident arena requires uniform expert geometry "
  6443:                     "(layer=%u bytes=%llu expected=%llu)\n",
  6444:                     il, (unsigned long long)layer_slot_bytes,
  6445:                     (unsigned long long)slot_bytes);
  6446:             return 0;
  6447:         }
  6448:     }
  6449:     cuda_dynamic_arena &arena = g_q1_0_dynamic_arena;
  6450:     if (arena.model_map == model_map && arena.model_size == model_size &&
  6451:         arena.n_layer == n_layer && arena.n_expert == n_expert &&
  6452:         arena.slot_bytes == slot_bytes &&
  6453:         arena.active_layer_first == active_layer_first &&
  6454:         arena.active_layer_last == active_layer_last &&
  6455:         arena.layers.size() == n_layer) {
  6456:         int same = 1;
  6457:         for (uint32_t il = 0; il < n_layer; il++) {
  6458:             if (memcmp(&arena.layers[il], &layers[il],
  6459:                        sizeof(layers[il])) != 0) {
  6460:                 same = 0;
  6461:                 break;
  6462:             }
  6463:         }
  6464:         if (same) return 1;
  6465:     }
  6466:     cuda_q1_0_dynamic_arena_release(0);
  6467:     try {
  6468:         arena.layers.assign(layers, layers + n_layer);
  6469:     } catch (...) {
  6470:         fprintf(stderr,
  6471:                 "ds4: Q1_0 resident arena layer metadata allocation failed\n");
  6472:         return 0;
  6473:     }
  6474:     arena.model_map = model_map;
  6475:     arena.model_size = model_size;
  6476:     arena.n_layer = n_layer;
  6477:     arena.n_expert = n_expert;
  6478:     arena.slot_bytes = slot_bytes;
  6479:     arena.active_layer_first = active_layer_first;
  6480:     arena.active_layer_last = active_layer_last;
  6481:     arena.backing = CUDA_DYNAMIC_ARENA_BACKING_Q1_0;
  6482:     fprintf(stderr,
  6483:             "ds4: [q1-0-resident-arena] result=bound backing=q1_0 layers=%u..%u iq2_host_arena=%s mixed_host_backing=%s iq2_vram_cache=q1-routes-bypass\n",
  6484:             active_layer_first, active_layer_last,
  6485:             cuda_q1_0_dual_arena_requested() ? "enabled" : "disabled",
  6486:             cuda_q1_0_dual_arena_requested() ? "dual-arena" : "disabled");
  6487:     return 1;
  6488: }
  6489: 
  6490: extern "C" int ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
  6491:         const void *model_map, uint64_t model_size,
  6492:         const ds4_gpu_dynamic_arena_layer *layers,
  6493:         uint32_t n_layer, uint32_t n_expert,
  6494:         uint32_t active_layer_first, uint32_t active_layer_last) {
  6495:     if (!cuda_q1_0_snapshot_backing_requested() ||
  6496:         !g_q1_0_sidecar_file_valid ||
  6497:         model_map != g_q1_0_sidecar_host_base ||
  6498:         model_size != g_q1_0_sidecar_size) {
  6499:         fprintf(stderr,
  6500:                 "ds4: [q1-0-snapshot] result=failed reason=sidecar-binding\n");
  6501:         return 0;
  6502:     }
  6503:     const int bound = cuda_dynamic_arena_bind(
  6504:         model_map, model_size, layers, n_layer, n_expert,
  6505:         CUDA_DYNAMIC_ARENA_BACKING_Q1_0,
  6506:         active_layer_first, active_layer_last);
  6507:     if (bound) {
  6508:         fprintf(stderr,
  6509:                 "ds4: [q1-0-snapshot] result=bound layers=%u..%u policy=sparse-prefill-ranked iq2_host_snapshot=disabled\n",
  6510:                 active_layer_first, active_layer_last);
  6511:     }
  6512:     return bound;
  6513: }
  6514: 
  6515: extern "C" int ds4_gpu_dynamic_arena_prepare_q1_0(
  6516:         uint64_t requested_bytes,
  6517:         uint64_t *allocated_bytes,
  6518:         uint32_t *slot_count,
  6519:         uint64_t *snapshot_generation) {
  6520:     if (allocated_bytes) *allocated_bytes = 0;
  6521:     if (slot_count) *slot_count = 0;
  6522:     if (snapshot_generation) *snapshot_generation = 0;
  6523:     cuda_dynamic_arena &arena = g_q1_0_dynamic_arena;
  6524:     if (!arena.model_map || arena.backing != CUDA_DYNAMIC_ARENA_BACKING_Q1_0 ||
  6525:         arena.slot_bytes == 0 || arena.n_expert == 0 ||
  6526:         arena.active_layer_first > arena.active_layer_last) {
  6527:         return 0;
  6528:     }
  6529:     const uint64_t required_entries =
  6530:         (uint64_t)(arena.active_layer_last - arena.active_layer_first + 1u) *
  6531:         arena.n_expert;
  6532:     if (required_entries == 0 || required_entries > UINT32_MAX ||
  6533:         arena.slot_bytes > UINT64_MAX / required_entries) {
  6534:         return 0;
  6535:     }
  6536:     const uint64_t required_bytes = required_entries * arena.slot_bytes;
  6537:     const int pageable_overflow =
  6538:         cuda_q1_0_pageable_overflow_requested() &&
  6539:         cuda_q1_0_dual_arena_requested();
  6540:     uint64_t pinned_entries = required_entries;
  6541:     if (pageable_overflow) {
  6542:         pinned_entries = requested_bytes / arena.slot_bytes;
  6543:         if (pinned_entries > required_entries) {
  6544:             pinned_entries = required_entries;
  6545:         }
  6546:     }
  6547:     if ((!pageable_overflow && requested_bytes < required_bytes) ||
  6548:         pinned_entries == 0 || required_bytes > SIZE_MAX) {
  6549:         fprintf(stderr,
  6550:                 "ds4: [q1-0-resident-arena] result=failed reason=capacity "
  6551:                 "requested=%llu required=%llu pageable_overflow=%u\n",
  6552:                 (unsigned long long)requested_bytes,
  6553:                 (unsigned long long)required_bytes,
  6554:                 pageable_overflow ? 1u : 0u);
  6555:         return 0;
  6556:     }
  6557:     const uint64_t pageable_entries = required_entries - pinned_entries;
  6558:     const uint64_t pinned_bytes = pinned_entries * arena.slot_bytes;
  6559:     const uint64_t pageable_bytes = pageable_entries * arena.slot_bytes;
  6560:     if (pinned_bytes > SIZE_MAX || pageable_bytes > SIZE_MAX) return 0;
  6561: 
  6562:     const double min_available_gib = cuda_dynamic_arena_min_available_gib();
  6563: #ifdef _WIN32
  6564:     MEMORYSTATUSEX memory_status;
  6565:     memset(&memory_status, 0, sizeof(memory_status));
  6566:     memory_status.dwLength = sizeof(memory_status);
  6567:     const int memory_status_ok = GlobalMemoryStatusEx(&memory_status) != 0;
  6568:     const double available_before_gib = memory_status_ok
  6569:         ? (double)memory_status.ullAvailPhys / 1073741824.0 : -1.0;
  6570: #else
  6571:     const int memory_status_ok = 0;
  6572:     const double available_before_gib = -1.0;
  6573: #endif
  6574:     if (min_available_gib < 0.0 ||
  6575:         (min_available_gib > 0.0 &&
  6576:          (!memory_status_ok ||
  6577:           (long double)required_bytes >
  6578:               (long double)(available_before_gib - min_available_gib) *
  6579:                   1073741824.0L))) {
  6580:         fprintf(stderr,
  6581:                 "ds4: [q1-0-resident-arena] result=failed "
  6582:                 "reason=host-capacity total=%.3f GiB available=%.3f GiB "
  6583:                 "reserve=%.3f GiB\n",
  6584:                 (double)required_bytes / 1073741824.0,
  6585:                 available_before_gib, min_available_gib);
  6586:         return 0;
  6587:     }
  6588: 
  6589:     if (arena.host_base && arena.allocated_bytes == pinned_bytes &&
  6590:         arena.pageable_bytes == pageable_bytes &&
  6591:         arena.snapshot_generation != 0 &&
  6592:         arena.active.size() == (size_t)arena.n_layer * arena.n_expert) {
  6593:         if (allocated_bytes) *allocated_bytes = arena.allocated_bytes;
  6594:         if (slot_count) *slot_count = (uint32_t)arena.slots.size();
  6595:         if (snapshot_generation) {
  6596:             *snapshot_generation = arena.snapshot_generation;
  6597:         }
  6598:         return 1;
  6599:     }
  6600:     if (arena.host_base || arena.pageable_base) {
  6601:         fprintf(stderr,
  6602:                 "ds4: [q1-0-resident-arena] result=failed reason=unexpected-reprepare\n");
  6603:         return 0;
  6604:     }
  6605: 
  6606:     const size_t binding_count = (size_t)arena.n_layer * arena.n_expert;
  6607:     char *host = NULL;
  6608:     std::vector<cuda_dynamic_arena_slot> slots;
  6609:     std::vector<cuda_dynamic_arena_binding> active;
  6610:     std::vector<cuda_dynamic_arena_binding> staging;
  6611:     std::vector<cuda_dynamic_arena_binding> preloaded;
  6612:     std::vector<uint8_t> preloaded_parts;
  6613:     try {
  6614:         slots.resize((size_t)required_entries);
  6615:         const cuda_dynamic_arena_binding empty =
  6616:             cuda_dynamic_arena_empty_binding();
  6617:         active.assign(binding_count, empty);
  6618:         staging.assign(binding_count, empty);
  6619:         preloaded.assign(binding_count, empty);
  6620:         preloaded_parts.assign(binding_count, 0);
  6621:     } catch (...) {
  6622:         return 0;
  6623:     }
  6624:     if (cudaHostAlloc((void **)&host, (size_t)pinned_bytes,
  6625:                       cudaHostAllocDefault) != cudaSuccess) {
  6626:         (void)cudaGetLastError();
  6627:         fprintf(stderr,
  6628:                 "ds4: [q1-0-resident-arena] result=failed reason=pinned-allocation bytes=%llu\n",
  6629:                 (unsigned long long)pinned_bytes);
  6630:         return 0;
  6631:     }
  6632:     char *pageable = NULL;
  6633:     if (pageable_bytes != 0) {
  6634: #ifdef _WIN32
  6635:         pageable = (char *)VirtualAlloc(
  6636:             NULL, (SIZE_T)pageable_bytes,
  6637:             MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  6638: #else
  6639:         pageable = (char *)malloc((size_t)pageable_bytes);
  6640: #endif
  6641:         if (!pageable) {
  6642:             (void)cudaFreeHost(host);
  6643:             fprintf(stderr,
  6644:                     "ds4: [q1-0-resident-arena] result=failed "
  6645:                     "reason=pageable-allocation bytes=%llu\n",
  6646:                     (unsigned long long)pageable_bytes);
  6647:             return 0;
  6648:         }
  6649:     }
  6650: 
  6651:     uint64_t pinned_cursor = 0;
  6652:     uint64_t pageable_cursor = 0;
  6653:     for (uint32_t slot_index = 0;
  6654:          slot_index < (uint32_t)required_entries; slot_index++) {
  6655:         cuda_dynamic_arena_slot &slot = slots[slot_index];
  6656:         const uint64_t pinned_before =
  6657:             (uint64_t)slot_index * pinned_entries / required_entries;
  6658:         const uint64_t pinned_after =
  6659:             (uint64_t)(slot_index + 1u) * pinned_entries /
  6660:             required_entries;
  6661:         const int use_pinned = pinned_after > pinned_before;
  6662:         slot.host_ptr = use_pinned
  6663:             ? host + pinned_cursor++ * arena.slot_bytes
  6664:             : pageable + pageable_cursor++ * arena.slot_bytes;
  6665:         slot.pageable = use_pinned ? 0u : 1u;
  6666:     }
  6667:     if (pinned_cursor != pinned_entries ||
  6668:         pageable_cursor != pageable_entries) {
  6669:         (void)cudaFreeHost(host);
  6670:         if (pageable) {
  6671: #ifdef _WIN32
  6672:             (void)VirtualFree(pageable, 0, MEM_RELEASE);
  6673: #else
  6674:             free(pageable);
  6675: #endif
  6676:         }
  6677:         fprintf(stderr,
  6678:                 "ds4: [q1-0-resident-arena] result=failed "
  6679:                 "reason=slot-partition\n");
  6680:         return 0;
  6681:     }
  6682: 
  6683:     g_q1_0_profile = cuda_q1_0_profile_stats{};
  6684:     cuda_q1_0_mixed_profile_reset(1);
  6685:     cuda_q1_0_profile_source_working_set_sample(
  6686:         "pre-copy", arena.model_map, arena.model_size);
  6687:     cuda_q1_0_source_unlock_stats source_unlock = {};
  6688:     cuda_q1_0_source_unlock_begin(&source_unlock);
  6689:     uint32_t slot_index = 0;
  6690:     uint64_t generation = 1;
  6691:     for (uint32_t layer = arena.active_layer_first;
  6692:          layer <= arena.active_layer_last; layer++) {
  6693:         const ds4_gpu_dynamic_arena_layer &geometry = arena.layers[layer];
  6694:         for (uint32_t expert = 0; expert < arena.n_expert; expert++) {
  6695:             cuda_dynamic_arena_slot &slot = slots[slot_index];
  6696:             char *destination = slot.host_ptr;
  6697:             const char *base = (const char *)arena.model_map;
  6698:             const char *gate = base + geometry.gate_offset +
  6699:                 (uint64_t)expert * geometry.gate_expert_bytes;
  6700:             const char *up = base + geometry.up_offset +
  6701:                 (uint64_t)expert * geometry.up_expert_bytes;
  6702:             const char *down = base + geometry.down_offset +
  6703:                 (uint64_t)expert * geometry.down_expert_bytes;
  6704:             memcpy(destination, gate, (size_t)geometry.gate_expert_bytes);
  6705:             memcpy(destination + geometry.gate_expert_bytes,
  6706:                    up, (size_t)geometry.up_expert_bytes);
  6707:             memcpy(destination + geometry.gate_expert_bytes +
  6708:                        geometry.up_expert_bytes,
  6709:                    down, (size_t)geometry.down_expert_bytes);
  6710:             slot.layer = layer;
  6711:             slot.expert = expert;
  6712:             slot.content_generation = generation++;
  6713:             slot.checksum = cuda_dynamic_arena_fnv1a64(
  6714:                 (const uint8_t *)destination, arena.slot_bytes);
  6715:             slot.last_dma_sequence = 0;
  6716:             slot.state = DS4_GPU_ARENA_READY;
  6717:             const uint32_t entry = layer * arena.n_expert + expert;
  6718:             active[entry] = {
  6719:                 slot_index, DS4_GPU_ARENA_READY,
  6720:                 slot.content_generation, 1
  6721:             };
  6722:             slot_index++;
  6723:         }
  6724:         cuda_q1_0_source_unlock_bootstrap_layer(
  6725:             arena, geometry, layer, &source_unlock);
  6726:     }
  6727:     cuda_q1_0_profile_source_working_set_sample(
  6728:         "post-bootstrap", arena.model_map, arena.model_size);
  6729:     cuda_q1_0_source_unlock_finish(
  6730:         &source_unlock, required_entries, required_bytes);
  6731: #ifdef _WIN32
  6732:     if (cuda_q1_0_profile_requested()) Sleep(1);
  6733: #endif
  6734:     cuda_q1_0_profile_source_working_set_sample(
  6735:         "post-unlock-settle", arena.model_map, arena.model_size);
  6736: 
  6737:     arena.submissions_blocked = 1;
  6738:     arena.hits_disabled = 1;
  6739:     arena.slots.swap(slots);
  6740:     arena.active.swap(active);
  6741:     arena.staging.swap(staging);
  6742:     arena.preloaded.swap(preloaded);
  6743:     arena.preloaded_parts.swap(preloaded_parts);
  6744:     arena.allocated_bytes = pinned_bytes;
  6745:     arena.pageable_base = pageable;
  6746:     arena.pageable_bytes = pageable_bytes;
  6747:     arena.pinned_slot_count = (uint32_t)pinned_entries;
  6748:     arena.pageable_slot_count = (uint32_t)pageable_entries;
  6749:     arena.next_generation = generation;
  6750:     arena.dma_sequence = 0;
  6751:     arena.hits = 0;
  6752:     arena.misses = 0;
  6753:     arena.fatal_errors = 0;
  6754:     arena.bytes_uploaded = 0;
  6755:     arena.pinned_hits = 0;
  6756:     arena.pageable_hits = 0;
  6757:     arena.pinned_bytes_uploaded = 0;
  6758:     arena.pageable_bytes_uploaded = 0;
  6759:     arena.host_base = host;
  6760:     arena.snapshot_generation = 1;
  6761:     arena.hits_disabled = 0;
  6762:     arena.submissions_blocked = 0;
  6763:     g_q1_0_resident_bootstrap_entries = (uint32_t)required_entries;
  6764:     if (allocated_bytes) *allocated_bytes = pinned_bytes;
  6765:     if (slot_count) *slot_count = (uint32_t)required_entries;
  6766:     if (snapshot_generation) *snapshot_generation = 1;
  6767:     fprintf(stderr,
  6768:             "ds4: [q1-0-resident-arena] result=bootstrapped entries=%u "
  6769:             "layers=%u..%u generation=1 source=sidecar-mmap "
  6770:             "route_pread=disabled iq2_host_arena=%s "
  6771:             "mixed_host_backing=%s pinned=%llu pageable=%llu "
  6772:             "pinned_slots=%llu pageable_slots=%llu total_slots=%llu "
  6773:             "total_bytes=%llu\n",
  6774:             (uint32_t)required_entries,
  6775:             arena.active_layer_first, arena.active_layer_last,
  6776:             cuda_q1_0_dual_arena_requested() ? "enabled" : "disabled",
  6777:             cuda_q1_0_dual_arena_requested() ? "dual-arena" : "disabled",
  6778:             (unsigned long long)pinned_bytes,
  6779:             (unsigned long long)pageable_bytes,
  6780:             (unsigned long long)pinned_entries,
  6781:             (unsigned long long)pageable_entries,
  6782:             (unsigned long long)required_entries,
  6783:             (unsigned long long)required_bytes);
  6784:     return 1;
  6785: }
  6786: 
  6787: extern "C" int ds4_gpu_dynamic_arena_prepare(
  6788:         uint64_t requested_bytes,
  6789:         uint64_t *allocated_bytes,
  6790:         uint32_t *slot_count) {
  6791:     if (allocated_bytes) *allocated_bytes = 0;
  6792:     if (slot_count) *slot_count = 0;
  6793:     if (!g_dynamic_arena.model_map || g_dynamic_arena.slot_bytes == 0 ||
  6794:         requested_bytes < g_dynamic_arena.slot_bytes) {
  6795:         return 0;

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' 6787 7470" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 595ms:
  6787: extern "C" int ds4_gpu_dynamic_arena_prepare(
  6788:         uint64_t requested_bytes,
  6789:         uint64_t *allocated_bytes,
  6790:         uint32_t *slot_count) {
  6791:     if (allocated_bytes) *allocated_bytes = 0;
  6792:     if (slot_count) *slot_count = 0;
  6793:     if (!g_dynamic_arena.model_map || g_dynamic_arena.slot_bytes == 0 ||
  6794:         requested_bytes < g_dynamic_arena.slot_bytes) {
  6795:         return 0;
  6796:     }
  6797:     uint64_t slots64 = requested_bytes / g_dynamic_arena.slot_bytes;
  6798:     const uint64_t max_entries =
  6799:         (uint64_t)g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
  6800:     if (slots64 > max_entries) slots64 = max_entries;
  6801:     if (slots64 == 0 || slots64 > UINT32_MAX) return 0;
  6802:     const uint64_t requested_slots64 = slots64;
  6803:     const uint64_t requested_capped_bytes =
  6804:         requested_slots64 * g_dynamic_arena.slot_bytes;
  6805:     const double min_available_gib = cuda_dynamic_arena_min_available_gib();
  6806:     double available_before_gib = -1.0;
  6807:     int memory_status_ok = 0;
  6808:     const char *cap_reason = "ok";
  6809: #ifdef _WIN32
  6810:     MEMORYSTATUSEX ms_before;
  6811:     memset(&ms_before, 0, sizeof(ms_before));
  6812:     ms_before.dwLength = sizeof(ms_before);
  6813:     if (GlobalMemoryStatusEx(&ms_before)) {
  6814:         memory_status_ok = 1;
  6815:         available_before_gib =
  6816:             (double)ms_before.ullAvailPhys / 1073741824.0;
  6817:     }
  6818: #endif
  6819:     if (min_available_gib < 0.0) {
  6820:         slots64 = 0;
  6821:         cap_reason = "invalid-min-available";
  6822:     } else if (min_available_gib > 0.0) {
  6823:         if (!memory_status_ok) {
  6824:             slots64 = 0;
  6825:             cap_reason = "memory-status-unavailable";
  6826:         } else {
  6827:             const double budget_gib = available_before_gib - min_available_gib;
  6828:             if (budget_gib <= 0.0) {
  6829:                 slots64 = 0;
  6830:             } else {
  6831:                 const long double budget_bytes_ld =
  6832:                     (long double)budget_gib * 1073741824.0L;
  6833:                 uint64_t budget_slots = 0;
  6834:                 if (budget_bytes_ld >=
  6835:                     (long double)g_dynamic_arena.slot_bytes) {
  6836:                     budget_slots =
  6837:                         (uint64_t)(budget_bytes_ld /
  6838:                                    (long double)g_dynamic_arena.slot_bytes);
  6839:                 }
  6840:                 if (budget_slots < slots64) slots64 = budget_slots;
  6841:             }
  6842:             if (slots64 == 0) cap_reason = "insufficient-available";
  6843:         }
  6844:     }
  6845:     const int ssd_wrap_requested = cuda_q1_0_ssd_wrap_requested();
  6846:     if (ssd_wrap_requested < 0) return 0;
  6847:     const int primary_ssd_wrap =
  6848:         ssd_wrap_requested == 1 &&
  6849:         g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_PRIMARY;
  6850:     const int q1_pageable_overflow =
  6851:         g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0 &&
  6852:         cuda_q1_0_snapshot_backing_requested() &&
  6853:         cuda_q1_0_pageable_overflow_requested();
  6854:     uint64_t total_slots64 = slots64;
  6855:     if (q1_pageable_overflow) {
  6856:         if (g_dynamic_arena.active_layer_first >
  6857:                 g_dynamic_arena.active_layer_last ||
  6858:             g_dynamic_arena.active_layer_last >=
  6859:                 g_dynamic_arena.n_layer) {
  6860:             return 0;
  6861:         }
  6862:         total_slots64 =
  6863:             (uint64_t)(g_dynamic_arena.active_layer_last -
  6864:                        g_dynamic_arena.active_layer_first + 1u) *
  6865:             g_dynamic_arena.n_expert;
  6866:         if (total_slots64 == 0 || total_slots64 > max_entries ||
  6867:             total_slots64 > UINT32_MAX || slots64 > total_slots64) {
  6868:             return 0;
  6869:         }
  6870:     }
  6871:     uint64_t pinned_allocation_slots64 = slots64;
  6872:     uint64_t resident_pinned_slots64 = slots64;
  6873:     uint64_t pageable_slots64 = total_slots64 - slots64;
  6874:     uint64_t ring_slots64 = 0;
  6875:     if (primary_ssd_wrap) {
  6876:         uint64_t partition_pinned_slots64 = 0;
  6877:         uint64_t partition_pageable_slots64 = 0;
  6878:         if (!cuda_q1_0_ssd_wrap_arena_partition(
  6879:                 slots64 * g_dynamic_arena.slot_bytes,
  6880:                 g_dynamic_arena.slot_bytes,
  6881:                 &partition_pinned_slots64,
  6882:                 &partition_pageable_slots64)) {
  6883:             return 0;
  6884:         }
  6885:         ring_slots64 = CUDA_Q1_0_SSD_WRAP_RING_SLOTS;
  6886:         if (partition_pinned_slots64 <= ring_slots64) return 0;
  6887:         pinned_allocation_slots64 = partition_pinned_slots64;
  6888:         resident_pinned_slots64 = partition_pinned_slots64 - ring_slots64;
  6889:         pageable_slots64 = partition_pageable_slots64;
  6890:         total_slots64 = resident_pinned_slots64 + pageable_slots64;
  6891:         if (total_slots64 == 0 || total_slots64 > UINT32_MAX) return 0;
  6892:     }
  6893:     const uint64_t bytes =
  6894:         pinned_allocation_slots64 * g_dynamic_arena.slot_bytes;
  6895:     if (pageable_slots64 != 0 &&
  6896:         g_dynamic_arena.slot_bytes > UINT64_MAX / pageable_slots64) {
  6897:         return 0;
  6898:     }
  6899:     const uint64_t pageable_bytes =
  6900:         pageable_slots64 * g_dynamic_arena.slot_bytes;
  6901:     if (pageable_bytes > UINT64_MAX - bytes) return 0;
  6902:     const uint64_t total_storage_bytes = bytes + pageable_bytes;
  6903:     if (q1_pageable_overflow && min_available_gib > 0.0) {
  6904:         if (!memory_status_ok ||
  6905:             (long double)total_storage_bytes >
  6906:                 (long double)(available_before_gib - min_available_gib) *
  6907:                     1073741824.0L) {
  6908:             fprintf(stderr,
  6909:                     "ds4: [q1-0-pageable-overflow] result=failed reason=host-capacity total=%.3f GiB available=%.3f GiB reserve=%.3f GiB\n",
  6910:                     (double)total_storage_bytes / 1073741824.0,
  6911:                     available_before_gib, min_available_gib);
  6912:             return 0;
  6913:         }
  6914:     }
  6915:     const int cap_capped = slots64 < requested_slots64 ? 1 : 0;
  6916:     if (primary_ssd_wrap) {
  6917:         fprintf(stderr,
  6918:             "ds4: [arena-cap] requested_gib=%.3f min_available_gib=%.3f available_before_gib=%.3f requested_bytes=%llu requested_slots=%llu chosen_bytes=%llu chosen_slots=%llu pageable_bytes=%llu pageable_slots=%llu total_slots=%llu ring_slots=%llu host_budget_bytes=%llu ssd_wrap=1 capped=%d result=%s reason=%s\n",
  6919:             (double)requested_capped_bytes / 1073741824.0,
  6920:             min_available_gib,
  6921:             available_before_gib,
  6922:             (unsigned long long)requested_capped_bytes,
  6923:             (unsigned long long)requested_slots64,
  6924:             (unsigned long long)bytes,
  6925:             (unsigned long long)resident_pinned_slots64,
  6926:             (unsigned long long)pageable_bytes,
  6927:             (unsigned long long)pageable_slots64,
  6928:             (unsigned long long)total_slots64,
  6929:             (unsigned long long)ring_slots64,
  6930:             (unsigned long long)total_storage_bytes,
  6931:             cap_capped,
  6932:             slots64 == 0 ? "disabled" : "ready",
  6933:             cap_reason);
  6934:     } else {
  6935:         fprintf(stderr,
  6936:             "ds4: [arena-cap] requested_gib=%.3f min_available_gib=%.3f available_before_gib=%.3f requested_bytes=%llu requested_slots=%llu chosen_bytes=%llu chosen_slots=%llu pageable_bytes=%llu pageable_slots=%llu total_slots=%llu capped=%d result=%s reason=%s\n",
  6937:             (double)requested_capped_bytes / 1073741824.0,
  6938:             min_available_gib, available_before_gib,
  6939:             (unsigned long long)requested_capped_bytes,
  6940:             (unsigned long long)requested_slots64,
  6941:             (unsigned long long)bytes,
  6942:             (unsigned long long)resident_pinned_slots64,
  6943:             (unsigned long long)pageable_bytes,
  6944:             (unsigned long long)pageable_slots64,
  6945:             (unsigned long long)total_slots64, cap_capped,
  6946:             slots64 == 0 ? "disabled" : "ready", cap_reason);
  6947:     }
  6948:     if (slots64 == 0) return 0;
  6949:     if (bytes > (uint64_t)SIZE_MAX ||
  6950:         pageable_bytes > (uint64_t)SIZE_MAX) return 0;
  6951:     if (g_dynamic_arena.host_base &&
  6952:         g_dynamic_arena.allocated_bytes == bytes &&
  6953:         g_dynamic_arena.pageable_bytes == pageable_bytes &&
  6954:         g_dynamic_arena.host_budget_bytes == total_storage_bytes &&
  6955:         g_dynamic_arena.pinned_slot_count == resident_pinned_slots64 &&
  6956:         g_dynamic_arena.pageable_slot_count == pageable_slots64 &&
  6957:         g_dynamic_arena.ssd_wrap_ssd_ring_slots ==
  6958:             (primary_ssd_wrap ? CUDA_Q1_0_SSD_WRAP_SSD_RING_SLOTS : 0u) &&
  6959:         g_dynamic_arena.ssd_wrap_h2d_ring_slots ==
  6960:             (primary_ssd_wrap ? CUDA_Q1_0_SSD_WRAP_H2D_RING_SLOTS : 0u) &&
  6961:         g_dynamic_arena.slots.size() == (size_t)total_slots64) {
  6962:         if (allocated_bytes) *allocated_bytes = total_storage_bytes;
  6963:         if (slot_count) *slot_count = (uint32_t)total_slots64;
  6964:         return 1;
  6965:     }
  6966: 
  6967:     cuda_dynamic_arena_storage_release();
  6968:     const uint64_t stage_bytes = cuda_model_copy_chunk_bytes() +
  6969:         (g_model_direct_align > 1 ? g_model_direct_align : 1);
  6970:     if (!cuda_model_stage_pool_alloc(stage_bytes)) {
  6971:         fprintf(stderr,
  6972:                 "ds4: CUDA dynamic arena staging preparation failed; arena disabled\n");
  6973:         return 0;
  6974:     }
  6975:     char *host = NULL;
  6976:     cudaError_t err = cudaHostAlloc((void **)&host, (size_t)bytes, cudaHostAllocDefault);
  6977:     if (err != cudaSuccess || !host) {
  6978:         fprintf(stderr,
  6979:                 "ds4: CUDA dynamic arena allocation %.2f GiB failed: %s; arena disabled\n",
  6980:                 (double)bytes / 1073741824.0, cudaGetErrorString(err));
  6981:         (void)cudaGetLastError();
  6982:         return 0;
  6983:     }
  6984:     char *pageable = NULL;
  6985:     if (pageable_bytes != 0) {
  6986: #ifdef _WIN32
  6987:         pageable = (char *)VirtualAlloc(
  6988:             NULL, (SIZE_T)pageable_bytes,
  6989:             MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  6990: #else
  6991:         pageable = (char *)malloc((size_t)pageable_bytes);
  6992: #endif
  6993:         if (!pageable) {
  6994:             (void)cudaFreeHost(host);
  6995:             fprintf(stderr,
  6996:                     "ds4: Q1_0 pageable overflow allocation %.2f GiB failed; arena disabled\n",
  6997:                     (double)pageable_bytes / 1073741824.0);
  6998:             return 0;
  6999:         }
  7000:     }
  7001: 
  7002:     try {
  7003:         g_dynamic_arena.slots.resize((size_t)total_slots64);
  7004:         const size_t bindings =
  7005:             (size_t)g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
  7006:         const cuda_dynamic_arena_binding empty = cuda_dynamic_arena_empty_binding();
  7007:         g_dynamic_arena.active.assign(bindings, empty);
  7008:         g_dynamic_arena.staging.assign(bindings, empty);
  7009:         g_dynamic_arena.preloaded.assign(bindings, empty);
  7010:         g_dynamic_arena.preloaded_parts.assign(bindings, 0);
  7011:     } catch (...) {
  7012:         (void)cudaFreeHost(host);
  7013:         if (pageable) {
  7014: #ifdef _WIN32
  7015:             (void)VirtualFree(pageable, 0, MEM_RELEASE);
  7016: #else
  7017:             free(pageable);
  7018: #endif
  7019:         }
  7020:         g_dynamic_arena.slots.clear();
  7021:         g_dynamic_arena.active.clear();
  7022:         g_dynamic_arena.staging.clear();
  7023:         g_dynamic_arena.preloaded.clear();
  7024:         g_dynamic_arena.preloaded_parts.clear();
  7025:         fprintf(stderr, "ds4: CUDA dynamic arena metadata allocation failed; arena disabled\n");
  7026:         return 0;
  7027:     }
  7028: 
  7029:     g_dynamic_arena.host_base = host;
  7030:     g_dynamic_arena.allocated_bytes = bytes;
  7031:     g_dynamic_arena.pageable_base = pageable;
  7032:     g_dynamic_arena.pageable_bytes = pageable_bytes;
  7033:     g_dynamic_arena.host_budget_bytes = total_storage_bytes;
  7034:     g_dynamic_arena.pinned_slot_count =
  7035:         (uint32_t)resident_pinned_slots64;
  7036:     g_dynamic_arena.pageable_slot_count = (uint32_t)pageable_slots64;
  7037:     if (primary_ssd_wrap) {
  7038:         g_dynamic_arena.ssd_wrap_ssd_ring_base = host +
  7039:             resident_pinned_slots64 * g_dynamic_arena.slot_bytes;
  7040:         g_dynamic_arena.ssd_wrap_h2d_ring_base =
  7041:             g_dynamic_arena.ssd_wrap_ssd_ring_base +
  7042:             (uint64_t)CUDA_Q1_0_SSD_WRAP_SSD_RING_SLOTS *
  7043:                 g_dynamic_arena.slot_bytes;
  7044:         g_dynamic_arena.ssd_wrap_ssd_ring_slots =
  7045:             CUDA_Q1_0_SSD_WRAP_SSD_RING_SLOTS;
  7046:         g_dynamic_arena.ssd_wrap_h2d_ring_slots =
  7047:             CUDA_Q1_0_SSD_WRAP_H2D_RING_SLOTS;
  7048:     }
  7049:     uint64_t pinned_cursor = 0;
  7050:     uint64_t pageable_cursor = 0;
  7051:     for (uint32_t i = 0; i < (uint32_t)total_slots64; i++) {
  7052:         cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[i];
  7053:         memset(&slot, 0, sizeof(slot));
  7054:         const uint64_t pinned_before =
  7055:             (uint64_t)i * resident_pinned_slots64 / total_slots64;
  7056:         const uint64_t pinned_after =
  7057:             (uint64_t)(i + 1u) * resident_pinned_slots64 / total_slots64;
  7058:         const int use_pinned = pinned_after > pinned_before;
  7059:         if (use_pinned) {
  7060:             slot.host_ptr = host +
  7061:                 pinned_cursor++ * g_dynamic_arena.slot_bytes;
  7062:             slot.pageable = 0;
  7063:         } else {
  7064:             slot.host_ptr = pageable +
  7065:                 pageable_cursor++ * g_dynamic_arena.slot_bytes;
  7066:             slot.pageable = 1;
  7067:         }
  7068:         slot.layer = UINT32_MAX;
  7069:         slot.expert = UINT32_MAX;
  7070:         slot.state = DS4_GPU_ARENA_FREE;
  7071:     }
  7072:     if (pinned_cursor != resident_pinned_slots64 ||
  7073:         pageable_cursor != pageable_slots64) {
  7074:         cuda_dynamic_arena_storage_release();
  7075:         fprintf(stderr,
  7076:                 "ds4: Q1_0 pageable overflow slot partition failed; arena disabled\n");
  7077:         return 0;
  7078:     }
  7079:     if (allocated_bytes) *allocated_bytes = total_storage_bytes;
  7080:     if (slot_count) *slot_count = (uint32_t)total_slots64;
  7081: #ifdef _WIN32
  7082:     MEMORYSTATUSEX ms;
  7083:     memset(&ms, 0, sizeof(ms));
  7084:     ms.dwLength = sizeof(ms);
  7085:     const double avail_gib = GlobalMemoryStatusEx(&ms)
  7086:         ? (double)ms.ullAvailPhys / 1073741824.0 : -1.0;
  7087: #else
  7088:     const double avail_gib = -1.0;
  7089: #endif
  7090:     if (primary_ssd_wrap) {
  7091:         fprintf(stderr,
  7092:             "ds4: CUDA dynamic arena ready pinned=%.2f GiB pageable=%.2f GiB total_slots=%u pinned_slots=%u pageable_slots=%u ring_slots=%u %.2f MiB/slot available_ram=%.2f GiB bytes=%llu host_budget_bytes=%llu slot_bytes=%llu backing=%s iq2_host_arena=%s mixed_host_backing=%s\n",
  7093:             (double)bytes / 1073741824.0,
  7094:             (double)pageable_bytes / 1073741824.0,
  7095:             (uint32_t)total_slots64, (uint32_t)resident_pinned_slots64,
  7096:             (uint32_t)pageable_slots64,
  7097:             (uint32_t)ring_slots64,
  7098:             (double)g_dynamic_arena.slot_bytes / 1048576.0, avail_gib,
  7099:             (unsigned long long)bytes,
  7100:             (unsigned long long)total_storage_bytes,
  7101:             (unsigned long long)g_dynamic_arena.slot_bytes,
  7102:             cuda_dynamic_arena_backing_name(g_dynamic_arena.backing),
  7103:             g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0
  7104:                 ? "disabled" : "enabled",
  7105:             g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0
  7106:                 ? "q1-snapshot-plus-primary-iq2-vram" : "n/a");
  7107:     } else {
  7108:         fprintf(stderr,
  7109:             "ds4: CUDA dynamic arena ready pinned=%.2f GiB pageable=%.2f GiB total_slots=%u pinned_slots=%u pageable_slots=%u %.2f MiB/slot available_ram=%.2f GiB bytes=%llu slot_bytes=%llu backing=%s iq2_host_arena=%s mixed_host_backing=%s\n",
  7110:             (double)bytes / 1073741824.0,
  7111:             (double)pageable_bytes / 1073741824.0,
  7112:             (uint32_t)total_slots64, (uint32_t)resident_pinned_slots64,
  7113:             (uint32_t)pageable_slots64,
  7114:             (double)g_dynamic_arena.slot_bytes / 1048576.0, avail_gib,
  7115:             (unsigned long long)bytes,
  7116:             (unsigned long long)g_dynamic_arena.slot_bytes,
  7117:             cuda_dynamic_arena_backing_name(g_dynamic_arena.backing),
  7118:             g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0
  7119:                 ? "disabled" : "enabled",
  7120:             g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0
  7121:                 ? "q1-snapshot-plus-primary-iq2-vram" : "n/a");
  7122:     }
  7123:     ds4_gpu_dynamic_arena_observer_reset();
  7124:     return 1;
  7125: }
  7126: 
  7127: extern "C" int ds4_gpu_dynamic_arena_begin(
  7128:         const uint8_t *target_resident,
  7129:         uint32_t entry_count,
  7130:         ds4_gpu_dynamic_arena_txn **out_txn,
  7131:         const ds4_gpu_dynamic_arena_load **out_loads,
  7132:         uint32_t *out_load_count) {
  7133:     if (out_txn) *out_txn = NULL;
  7134:     if (out_loads) *out_loads = NULL;
  7135:     if (out_load_count) *out_load_count = 0;
  7136:     const uint64_t expected64 =
  7137:         (uint64_t)g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
  7138:     if (!target_resident || !out_txn || !out_loads || !out_load_count ||
  7139:         !g_dynamic_arena.host_base || g_dynamic_arena.txn ||
  7140:         g_dynamic_arena.tiering_exclusive ||
  7141:         expected64 == 0 || expected64 > UINT32_MAX ||
  7142:         entry_count != (uint32_t)expected64) {
  7143:         return 0;
  7144:     }
  7145: 
  7146:     uint32_t target_count = 0;
  7147:     for (uint32_t i = 0; i < entry_count; i++) {
  7148:         if (target_resident[i] && g_sparse_bake_active) {
  7149:             const uint32_t layer = i / g_dynamic_arena.n_expert;
  7150:             const uint32_t expert = i % g_dynamic_arena.n_expert;
  7151:             if (!cuda_sparse_bake_expert_retained(layer, expert)) {
  7152:                 fprintf(stderr,
  7153:                         "ds4: sparse bake rejected arena target layer=%u expert=%u\n",
  7154:                         layer, expert);
  7155:                 return 0;
  7156:             }
  7157:         }
  7158:         target_count += target_resident[i] != 0;
  7159:     }
  7160:     if (target_count > g_dynamic_arena.slots.size()) return 0;
  7161: 
  7162:     ds4_gpu_dynamic_arena_txn *txn =
  7163:         new (std::nothrow) ds4_gpu_dynamic_arena_txn();
  7164:     if (!txn) return 0;
  7165:     txn->base_generation = g_dynamic_arena.snapshot_generation;
  7166:     txn->target_generation = cuda_dynamic_arena_next_generation();
  7167:     txn->retire_event = NULL;
  7168:     txn->failed = 0;
  7169:     std::vector<uint8_t> active_slots;
  7170:     std::vector<uint32_t> available;
  7171:     try {
  7172:         txn->target.assign(target_resident, target_resident + entry_count);
  7173:         txn->loads.reserve(target_count);
  7174:         txn->reserved_slots.reserve(target_count);
  7175:         active_slots.assign(g_dynamic_arena.slots.size(), 0);
  7176:         available.reserve(g_dynamic_arena.slots.size());
  7177:     } catch (...) {
  7178:         delete txn;
  7179:         return 0;
  7180:     }
  7181: 
  7182:     if (cudaEventCreateWithFlags(&txn->retire_event,
  7183:                                  cudaEventDisableTiming) != cudaSuccess ||
  7184:         cudaEventRecord(txn->retire_event,
  7185:                         g_model_upload_stream ? g_model_upload_stream : 0) != cudaSuccess ||
  7186:         cudaEventSynchronize(txn->retire_event) != cudaSuccess) {
  7187:         (void)cudaGetLastError();
  7188:         if (txn->retire_event) (void)cudaEventDestroy(txn->retire_event);
  7189:         delete txn;
  7190:         return 0;
  7191:     }
  7192: 
  7193:     g_dynamic_arena.txn = txn;
  7194:     g_dynamic_arena.submissions_blocked = 1;
  7195:     g_dynamic_arena.hits_disabled = 1;
  7196:     const cuda_dynamic_arena_binding empty = cuda_dynamic_arena_empty_binding();
  7197:     for (uint32_t i = 0; i < entry_count; i++) {
  7198:         g_dynamic_arena.staging[i] = empty;
  7199:     }
  7200: 
  7201:     for (uint32_t entry = 0; entry < entry_count; entry++) {
  7202:         const uint32_t layer = entry / g_dynamic_arena.n_expert;
  7203:         const uint32_t expert = entry % g_dynamic_arena.n_expert;
  7204:         const cuda_dynamic_arena_binding &active = g_dynamic_arena.active[entry];
  7205:         if (cuda_dynamic_arena_binding_valid(
  7206:                 g_dynamic_arena,
  7207:                 active, layer, expert, txn->base_generation,
  7208:                 DS4_GPU_ARENA_READY)) {
  7209:             active_slots[active.slot] = 1;
  7210:             if (target_resident[entry]) {
  7211:                 cuda_dynamic_arena_binding next = active;
  7212:                 next.snapshot_generation = txn->target_generation;
  7213:                 g_dynamic_arena.staging[entry] = next;
  7214:             }
  7215:         }
  7216:         const cuda_dynamic_arena_binding &preloaded =
  7217:             g_dynamic_arena.preloaded[entry];
  7218:         const int preloaded_loading = cuda_dynamic_arena_binding_valid(
  7219:             g_dynamic_arena,
  7220:             preloaded, layer, expert, 0,
  7221:             DS4_GPU_ARENA_LOADING);
  7222:         const int preloaded_staged = cuda_dynamic_arena_binding_valid(
  7223:             g_dynamic_arena,
  7224:             preloaded, layer, expert, 0,
  7225:             DS4_GPU_ARENA_STAGED);
  7226:         if (preloaded_loading || preloaded_staged) {
  7227:             active_slots[preloaded.slot] = 1;
  7228:             if (preloaded_staged && target_resident[entry] &&
  7229:                 g_dynamic_arena.staging[entry].slot == UINT32_MAX) {
  7230:                 cuda_dynamic_arena_binding next = preloaded;
  7231:                 next.snapshot_generation = txn->target_generation;
  7232:                 g_dynamic_arena.staging[entry] = next;
  7233:             }
  7234:         }
  7235:     }
  7236:     for (uint32_t slot = 0; slot < g_dynamic_arena.slots.size(); slot++) {
  7237:         if (!active_slots[slot]) available.push_back(slot);
  7238:     }
  7239: 
  7240:     uint32_t available_cursor = 0;
  7241:     try {
  7242:         for (uint32_t entry = 0; entry < entry_count; entry++) {
  7243:             if (!target_resident[entry] ||
  7244:                 g_dynamic_arena.staging[entry].slot != UINT32_MAX) {
  7245:                 continue;
  7246:             }
  7247:             if (available_cursor >= available.size()) {
  7248:                 txn->failed = 1;
  7249:                 ds4_gpu_dynamic_arena_abort(txn);
  7250:                 return 0;
  7251:             }
  7252:             const uint32_t layer = entry / g_dynamic_arena.n_expert;
  7253:             const uint32_t expert = entry % g_dynamic_arena.n_expert;
  7254:             const uint32_t slot_index = available[available_cursor++];
  7255:             cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[slot_index];
  7256:             slot.state = DS4_GPU_ARENA_RETIRING;
  7257:             slot.layer = layer;
  7258:             slot.expert = expert;
  7259:             slot.content_generation = cuda_dynamic_arena_next_generation();
  7260:             slot.checksum = 0;
  7261:             slot.last_dma_sequence = 0;
  7262:             slot.state = DS4_GPU_ARENA_LOADING;
  7263:             const cuda_dynamic_arena_binding binding = {
  7264:                 slot_index, DS4_GPU_ARENA_LOADING,
  7265:                 slot.content_generation, txn->target_generation
  7266:             };
  7267:             g_dynamic_arena.staging[entry] = binding;
  7268:             const ds4_gpu_dynamic_arena_load load = {
  7269:                 layer, expert, slot_index, slot.content_generation,
  7270:                 slot.host_ptr, g_dynamic_arena.slot_bytes
  7271:             };
  7272:             txn->loads.push_back(load);
  7273:             txn->reserved_slots.push_back(slot_index);
  7274:         }
  7275:     } catch (...) {
  7276:         txn->failed = 1;
  7277:         ds4_gpu_dynamic_arena_abort(txn);
  7278:         return 0;
  7279:     }
  7280: 
  7281:     *out_txn = txn;
  7282:     *out_loads = txn->loads.empty() ? NULL : txn->loads.data();
  7283:     *out_load_count = (uint32_t)txn->loads.size();
  7284:     fprintf(stderr,
  7285:             "ds4: [arena] begin base=%llu target=%llu resident=%u loads=%u slots=%u\n",
  7286:             (unsigned long long)txn->base_generation,
  7287:             (unsigned long long)txn->target_generation,
  7288:             target_count, (uint32_t)txn->loads.size(),
  7289:             (uint32_t)g_dynamic_arena.slots.size());
  7290:     return 1;
  7291: }
  7292: 
  7293: static int cuda_dynamic_arena_finish_load_impl(
  7294:         ds4_gpu_dynamic_arena_txn *txn,
  7295:         uint32_t load_index,
  7296:         uint64_t checksum,
  7297:         int success,
  7298:         int verify_checksum) {
  7299:     if (!txn || txn != g_dynamic_arena.txn ||
  7300:         load_index >= txn->loads.size()) {
  7301:         return 0;
  7302:     }
  7303:     const ds4_gpu_dynamic_arena_load &load = txn->loads[load_index];
  7304:     if (load.slot >= g_dynamic_arena.slots.size()) {
  7305:         txn->failed = 1;
  7306:         return 0;
  7307:     }
  7308:     cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[load.slot];
  7309:     if (slot.layer != load.layer || slot.expert != load.expert ||
  7310:         slot.content_generation != load.slot_generation ||
  7311:         slot.state != DS4_GPU_ARENA_LOADING) {
  7312:         txn->failed = 1;
  7313:         return 0;
  7314:     }
  7315:     const uint32_t entry = load.layer * g_dynamic_arena.n_expert + load.expert;
  7316:     if (!success) {
  7317:         slot.state = DS4_GPU_ARENA_POISONED;
  7318:         g_dynamic_arena.staging[entry].state = DS4_GPU_ARENA_POISONED;
  7319:         txn->failed = 1;
  7320:         return 0;
  7321:     }
  7322:     const uint64_t actual_checksum = verify_checksum ?
  7323:         cuda_dynamic_arena_fnv1a64(
  7324:             (const uint8_t *)slot.host_ptr, g_dynamic_arena.slot_bytes) :
  7325:         checksum;
  7326:     if (actual_checksum != checksum) {
  7327:         slot.state = DS4_GPU_ARENA_POISONED;
  7328:         g_dynamic_arena.staging[entry].state = DS4_GPU_ARENA_POISONED;
  7329:         txn->failed = 1;
  7330:         return 0;
  7331:     }
  7332:     slot.checksum = actual_checksum;
  7333:     slot.state = DS4_GPU_ARENA_STAGED;
  7334:     g_dynamic_arena.staging[entry].state = DS4_GPU_ARENA_STAGED;
  7335:     return 1;
  7336: }
  7337: 
  7338: extern "C" int ds4_gpu_dynamic_arena_finish_load(
  7339:         ds4_gpu_dynamic_arena_txn *txn,
  7340:         uint32_t load_index,
  7341:         uint64_t checksum,
  7342:         int success) {
  7343:     return cuda_dynamic_arena_finish_load_impl(
  7344:         txn, load_index, checksum, success, 1);
  7345: }
  7346: 
  7347: extern "C" int ds4_gpu_dynamic_arena_publish(
  7348:         ds4_gpu_dynamic_arena_txn *txn,
  7349:         uint64_t *snapshot_generation) {
  7350:     if (snapshot_generation) *snapshot_generation = 0;
  7351:     if (!txn || txn != g_dynamic_arena.txn || txn->failed) return 0;
  7352:     const uint32_t entries =
  7353:         g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
  7354:     uint32_t q1_bootstrap_entries = 0;
  7355:     if (g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0) {
  7356:         const uint64_t expected =
  7357:             (uint64_t)(g_dynamic_arena.active_layer_last -
  7358:                        g_dynamic_arena.active_layer_first + 1u) *
  7359:             g_dynamic_arena.n_expert;
  7360:         int target_out_of_range = txn->target.size() != entries;
  7361:         uint32_t target_entries = 0;
  7362:         if (!target_out_of_range) {
  7363:             for (uint32_t entry = 0; entry < entries; entry++) {
  7364:                 if (!txn->target[entry]) continue;
  7365:                 const uint32_t layer = entry / g_dynamic_arena.n_expert;
  7366:                 target_entries++;
  7367:                 target_out_of_range |=
  7368:                     layer < g_dynamic_arena.active_layer_first ||
  7369:                     layer > g_dynamic_arena.active_layer_last;
  7370:             }
  7371:         }
  7372:         const int sparse_snapshot =
  7373:             cuda_q1_0_snapshot_backing_requested();
  7374:         if (expected > UINT32_MAX || target_out_of_range ||
  7375:             (!sparse_snapshot && target_entries != (uint32_t)expected) ||
  7376:             (sparse_snapshot &&
  7377:              (target_entries == 0u ||
  7378:               target_entries > g_dynamic_arena.slots.size()))) {
  7379:             txn->failed = 1;
  7380:             fprintf(stderr,
  7381:                     "ds4: [q1-0-resident-arena] result=failed reason=%s target_entries=%u required_entries=%llu capacity=%u router=unchanged\n",
  7382:                     sparse_snapshot ? "invalid-sparse-snapshot" :
  7383:                         "full-bootstrap-snapshot-required",
  7384:                     target_entries, (unsigned long long)expected,
  7385:                     (uint32_t)g_dynamic_arena.slots.size());
  7386:             return 0;
  7387:         }
  7388:         q1_bootstrap_entries = target_entries;
  7389:     }
  7390:     std::vector<uint8_t> referenced;
  7391:     try {
  7392:         referenced.assign(g_dynamic_arena.slots.size(), 0);
  7393:     } catch (...) {
  7394:         txn->failed = 1;
  7395:         return 0;
  7396:     }
  7397:     for (uint32_t entry = 0; entry < entries; entry++) {
  7398:         if (!txn->target[entry]) continue;
  7399:         const uint32_t layer = entry / g_dynamic_arena.n_expert;
  7400:         const uint32_t expert = entry % g_dynamic_arena.n_expert;
  7401:         cuda_dynamic_arena_binding &binding = g_dynamic_arena.staging[entry];
  7402:         if (binding.slot >= g_dynamic_arena.slots.size() ||
  7403:             binding.snapshot_generation != txn->target_generation) {
  7404:             txn->failed = 1;
  7405:             return 0;
  7406:         }
  7407:         cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[binding.slot];
  7408:         if ((slot.state != DS4_GPU_ARENA_READY &&
  7409:              slot.state != DS4_GPU_ARENA_STAGED) ||
  7410:             slot.layer != layer || slot.expert != expert ||
  7411:             slot.content_generation != binding.slot_generation) {
  7412:             txn->failed = 1;
  7413:             return 0;
  7414:         }
  7415:         referenced[binding.slot] = 1;
  7416:     }
  7417:     /* Validation above is side-effect free. Only after every target is known
  7418:      * good do staged slots become visible and bindings become READY. */
  7419:     for (uint32_t entry = 0; entry < entries; entry++) {
  7420:         if (!txn->target[entry]) continue;
  7421:         cuda_dynamic_arena_binding &binding = g_dynamic_arena.staging[entry];
  7422:         cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[binding.slot];
  7423:         if (slot.state == DS4_GPU_ARENA_STAGED) {
  7424:             slot.state = DS4_GPU_ARENA_READY;
  7425:         }
  7426:         binding.state = DS4_GPU_ARENA_READY;
  7427:     }
  7428:     for (uint32_t slot_index = 0;
  7429:          slot_index < g_dynamic_arena.slots.size(); slot_index++) {
  7430:         if (referenced[slot_index]) continue;
  7431:         cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[slot_index];
  7432:         slot.state = DS4_GPU_ARENA_FREE;
  7433:         slot.layer = UINT32_MAX;
  7434:         slot.expert = UINT32_MAX;
  7435:         slot.checksum = 0;
  7436:         slot.last_dma_sequence = 0;
  7437:     }
  7438:     g_dynamic_arena.active.swap(g_dynamic_arena.staging);
  7439:     for (uint32_t entry = 0; entry < entries; entry++) {
  7440:         g_dynamic_arena.staging[entry] = cuda_dynamic_arena_empty_binding();
  7441:         g_dynamic_arena.preloaded[entry] = cuda_dynamic_arena_empty_binding();
  7442:         g_dynamic_arena.preloaded_parts[entry] = 0;
  7443:     }
  7444:     g_dynamic_arena.snapshot_generation = txn->target_generation;
  7445:     g_dynamic_arena.submissions_blocked = 0;
  7446:     g_dynamic_arena.hits_disabled = 0;
  7447:     if (g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0) {
  7448:         g_q1_0_resident_bootstrap_entries = q1_bootstrap_entries;
  7449:     }
  7450:     if (snapshot_generation) {
  7451:         *snapshot_generation = g_dynamic_arena.snapshot_generation;
  7452:     }
  7453:     fprintf(stderr,
  7454:             "ds4: [arena] publish generation=%llu loads=%u\n",
  7455:             (unsigned long long)g_dynamic_arena.snapshot_generation,
  7456:             (uint32_t)txn->loads.size());
  7457:     if (txn->retire_event) (void)cudaEventDestroy(txn->retire_event);
  7458:     g_dynamic_arena.txn = NULL;
  7459:     delete txn;
  7460:     return 1;
  7461: }
  7462: 
  7463: extern "C" void ds4_gpu_dynamic_arena_abort(
  7464:         ds4_gpu_dynamic_arena_txn *txn) {
  7465:     if (!txn || txn != g_dynamic_arena.txn) return;
  7466:     for (uint32_t i = 0; i < txn->loads.size(); i++) {
  7467:         const ds4_gpu_dynamic_arena_load &load = txn->loads[i];
  7468:         if (load.slot >= g_dynamic_arena.slots.size()) continue;
  7469:         cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[load.slot];
  7470:         if (slot.content_generation == load.slot_generation &&

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"dynamic_arena_(build|wrap|publish|begin|finish|load)|ds4_gpu_dynamic_arena\" 'C:\\Users\\imanu\\g130i\\ds4.c' 'C:\\Users\\imanu\\g130i\\ds4_gpu.h'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 147ms:
C:\Users\imanu\g130i\ds4_gpu.h:21:typedef struct ds4_gpu_dynamic_arena_txn ds4_gpu_dynamic_arena_txn;
C:\Users\imanu\g130i\ds4_gpu.h:30:} ds4_gpu_dynamic_arena_layer;
C:\Users\imanu\g130i\ds4_gpu.h:39:} ds4_gpu_dynamic_arena_load;
C:\Users\imanu\g130i\ds4_gpu.h:123:int ds4_gpu_dynamic_arena_bind(
C:\Users\imanu\g130i\ds4_gpu.h:125:        const ds4_gpu_dynamic_arena_layer *layers,
C:\Users\imanu\g130i\ds4_gpu.h:130:        const ds4_gpu_dynamic_arena_layer *layers,
C:\Users\imanu\g130i\ds4_gpu.h:132:int ds4_gpu_dynamic_arena_bind_q1_0(
C:\Users\imanu\g130i\ds4_gpu.h:134:        const ds4_gpu_dynamic_arena_layer *layers,
C:\Users\imanu\g130i\ds4_gpu.h:137:int ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
C:\Users\imanu\g130i\ds4_gpu.h:139:        const ds4_gpu_dynamic_arena_layer *layers,
C:\Users\imanu\g130i\ds4_gpu.h:142:int ds4_gpu_dynamic_arena_prepare_q1_0(
C:\Users\imanu\g130i\ds4_gpu.h:147:int ds4_gpu_dynamic_arena_prepare(
C:\Users\imanu\g130i\ds4_gpu.h:151:int ds4_gpu_dynamic_arena_begin(
C:\Users\imanu\g130i\ds4_gpu.h:154:        ds4_gpu_dynamic_arena_txn **out_txn,
C:\Users\imanu\g130i\ds4_gpu.h:155:        const ds4_gpu_dynamic_arena_load **out_loads,
C:\Users\imanu\g130i\ds4_gpu.h:157:int ds4_gpu_dynamic_arena_finish_load(
C:\Users\imanu\g130i\ds4_gpu.h:158:        ds4_gpu_dynamic_arena_txn *txn,
C:\Users\imanu\g130i\ds4_gpu.h:162:int ds4_gpu_dynamic_arena_publish(
C:\Users\imanu\g130i\ds4_gpu.h:163:        ds4_gpu_dynamic_arena_txn *txn,
C:\Users\imanu\g130i\ds4_gpu.h:165:void ds4_gpu_dynamic_arena_abort(ds4_gpu_dynamic_arena_txn *txn);
C:\Users\imanu\g130i\ds4_gpu.h:166:void ds4_gpu_dynamic_arena_request_begin(void);
C:\Users\imanu\g130i\ds4_gpu.h:167:void ds4_gpu_dynamic_arena_observer_reset(void);
C:\Users\imanu\g130i\ds4_gpu.h:168:void ds4_gpu_dynamic_arena_release(void);
C:\Users\imanu\g130i\ds4.c:19823:static bool dynamic_arena_build_layers(
C:\Users\imanu\g130i\ds4.c:19824:        ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER],
C:\Users\imanu\g130i\ds4.c:19830:        ds4_gpu_dynamic_arena_layer *geometry = &layers[il];
C:\Users\imanu\g130i\ds4.c:19852:static bool dynamic_arena_build_q1_0_layers(
C:\Users\imanu\g130i\ds4.c:19853:        ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER],
C:\Users\imanu\g130i\ds4.c:19864:        ds4_gpu_dynamic_arena_layer *geometry = &layers[il];
C:\Users\imanu\g130i\ds4.c:19922:        const ds4_gpu_dynamic_arena_layer  layers[DS4_N_LAYER],
C:\Users\imanu\g130i\ds4.c:19923:        const ds4_gpu_dynamic_arena_load  *load,
C:\Users\imanu\g130i\ds4.c:19929:    const ds4_gpu_dynamic_arena_layer *layer = &layers[load->layer];
C:\Users\imanu\g130i\ds4.c:19978:static bool dynamic_arena_wrap_fixture(
C:\Users\imanu\g130i\ds4.c:19980:        const ds4_gpu_dynamic_arena_layer  layers[DS4_N_LAYER],
C:\Users\imanu\g130i\ds4.c:19990:    ds4_gpu_dynamic_arena_txn *txn = NULL;
C:\Users\imanu\g130i\ds4.c:19991:    const ds4_gpu_dynamic_arena_load *loads = NULL;
C:\Users\imanu\g130i\ds4.c:19993:    if (!ds4_gpu_dynamic_arena_begin(
C:\Users\imanu\g130i\ds4.c:19998:        if (txn) ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20007:        const bool finished = ds4_gpu_dynamic_arena_finish_load(
C:\Users\imanu\g130i\ds4.c:20018:        ds4_gpu_dynamic_arena_publish(txn, &snapshot_generation)) {
C:\Users\imanu\g130i\ds4.c:20026:    ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20035:        const ds4_gpu_dynamic_arena_layer  layers[DS4_N_LAYER],
C:\Users\imanu\g130i\ds4.c:20045:    ds4_gpu_dynamic_arena_txn *txn = NULL;
C:\Users\imanu\g130i\ds4.c:20046:    const ds4_gpu_dynamic_arena_load *loads = NULL;
C:\Users\imanu\g130i\ds4.c:20048:    if (!ds4_gpu_dynamic_arena_begin(
C:\Users\imanu\g130i\ds4.c:20051:        if (txn) ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20058:    const bool finished = ds4_gpu_dynamic_arena_finish_load(
C:\Users\imanu\g130i\ds4.c:20060:    ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20073:    if (!ds4_gpu_dynamic_arena_begin(
C:\Users\imanu\g130i\ds4.c:20075:        if (txn) ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20080:    ds4_gpu_dynamic_arena_abort(txn);
C:\Users\imanu\g130i\ds4.c:20198:            ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER];
C:\Users\imanu\g130i\ds4.c:20210:                    dynamic_arena_build_layers(
C:\Users\imanu\g130i\ds4.c:20218:                    dynamic_arena_build_q1_0_layers(
C:\Users\imanu\g130i\ds4.c:20221:                    ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
C:\Users\imanu\g130i\ds4.c:20227:                if (!q1_bound || !ds4_gpu_dynamic_arena_prepare(
C:\Users\imanu\g130i\ds4.c:20231:                    ds4_gpu_dynamic_arena_release();
C:\Users\imanu\g130i\ds4.c:20243:                    dynamic_arena_build_layers(layers, &e->model, &e->weights);
C:\Users\imanu\g130i\ds4.c:20245:                    ds4_gpu_dynamic_arena_bind(
C:\Users\imanu\g130i\ds4.c:20248:                if (!primary_bound || !ds4_gpu_dynamic_arena_prepare(
C:\Users\imanu\g130i\ds4.c:20256:                        ds4_gpu_dynamic_arena_release();
C:\Users\imanu\g130i\ds4.c:20264:                        const bool published = dynamic_arena_wrap_fixture(
C:\Users\imanu\g130i\ds4.c:20279:                    dynamic_arena_build_q1_0_layers(
C:\Users\imanu\g130i\ds4.c:20282:                    ds4_gpu_dynamic_arena_bind_q1_0(
C:\Users\imanu\g130i\ds4.c:20290:                     ds4_gpu_dynamic_arena_prepare_q1_0(
C:\Users\imanu\g130i\ds4.c:20296:                    ds4_gpu_dynamic_arena_release();
C:\Users\imanu\g130i\ds4.c:20314:                    ds4_gpu_dynamic_arena_release();
C:\Users\imanu\g130i\ds4.c:20448:        ds4_gpu_dynamic_arena_request_begin();
C:\Users\imanu\g130i\ds4.c:20485:            ds4_gpu_dynamic_arena_observer_reset();
C:\Users\imanu\g130i\ds4.c:20502:        ds4_gpu_dynamic_arena_observer_reset();
C:\Users\imanu\g130i\ds4.c:20532:    ds4_gpu_dynamic_arena_observer_reset();

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 19820 20030; Show-Lines 'C:\\Users\\imanu\\g130i\\ds4_gpu.h' 20 175" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 621ms:
 19820:     return true;
 19821: }
 19822: 
 19823: static bool dynamic_arena_build_layers(
 19824:         ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER],
 19825:         const ds4_model            *model,
 19826:         const ds4_weights          *weights) {
 19827:     memset(layers, 0, sizeof(layers[0]) * DS4_N_LAYER);
 19828:     for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
 19829:         const ds4_layer_weights *layer = &weights->layer[il];
 19830:         ds4_gpu_dynamic_arena_layer *geometry = &layers[il];
 19831:         if (!dynamic_arena_tensor_geometry(
 19832:                 model, layer->ffn_gate_exps, il, "gate",
 19833:                 false,
 19834:                 &geometry->gate_offset, &geometry->gate_expert_bytes) ||
 19835:             !dynamic_arena_tensor_geometry(
 19836:                 model, layer->ffn_up_exps, il, "up",
 19837:                 false,
 19838:                 &geometry->up_offset, &geometry->up_expert_bytes) ||
 19839:             !dynamic_arena_tensor_geometry(
 19840:                 model, layer->ffn_down_exps, il, "down",
 19841:                 false,
 19842:                 &geometry->down_offset, &geometry->down_expert_bytes) ||
 19843:             geometry->gate_expert_bytes > UINT64_MAX - geometry->up_expert_bytes ||
 19844:             geometry->gate_expert_bytes + geometry->up_expert_bytes >
 19845:                 UINT64_MAX - geometry->down_expert_bytes) {
 19846:             return false;
 19847:         }
 19848:     }
 19849:     return true;
 19850: }
 19851: 
 19852: static bool dynamic_arena_build_q1_0_layers(
 19853:         ds4_gpu_dynamic_arena_layer layers[DS4_N_LAYER],
 19854:         const ds4_model            *model) {
 19855:     if (!model || !g_q1_0_sidecar.ready ||
 19856:         model != g_q1_0_sidecar.model ||
 19857:         g_q1_0_sidecar.first_layer > g_q1_0_sidecar.last_layer ||
 19858:         g_q1_0_sidecar.last_layer >= DS4_N_LAYER) {
 19859:         return false;
 19860:     }
 19861:     memset(layers, 0, sizeof(layers[0]) * DS4_N_LAYER);
 19862:     for (uint32_t il = g_q1_0_sidecar.first_layer;
 19863:          il <= g_q1_0_sidecar.last_layer; il++) {
 19864:         ds4_gpu_dynamic_arena_layer *geometry = &layers[il];
 19865:         if (!dynamic_arena_tensor_geometry(
 19866:                 model, g_q1_0_sidecar.gate[il], il, "Q1_0 gate",
 19867:                 true,
 19868:                 &geometry->gate_offset, &geometry->gate_expert_bytes) ||
 19869:             !dynamic_arena_tensor_geometry(
 19870:                 model, g_q1_0_sidecar.up[il], il, "Q1_0 up",
 19871:                 true,
 19872:                 &geometry->up_offset, &geometry->up_expert_bytes) ||
 19873:             !dynamic_arena_tensor_geometry(
 19874:                 model, g_q1_0_sidecar.down[il], il, "Q1_0 down",
 19875:                 true,
 19876:                 &geometry->down_offset, &geometry->down_expert_bytes) ||
 19877:             geometry->gate_expert_bytes > UINT64_MAX - geometry->up_expert_bytes ||
 19878:             geometry->gate_expert_bytes + geometry->up_expert_bytes >
 19879:                 UINT64_MAX - geometry->down_expert_bytes) {
 19880:             return false;
 19881:         }
 19882:     }
 19883:     return true;
 19884: }
 19885: 
 19886: static bool dynamic_arena_source(
 19887:         const ds4_model *model,
 19888:         uint64_t         tensor_offset,
 19889:         uint64_t         expert_bytes,
 19890:         uint32_t         expert,
 19891:         const uint8_t  **source) {
 19892:     if (!model || !model->map || !source || expert >= DS4_N_EXPERT ||
 19893:         expert_bytes == 0 ||
 19894:         (expert != 0 && expert_bytes > UINT64_MAX / expert)) {
 19895:         return false;
 19896:     }
 19897:     const uint64_t relative = expert_bytes * expert;
 19898:     if (tensor_offset > model->size || relative > model->size - tensor_offset) {
 19899:         return false;
 19900:     }
 19901:     const uint64_t offset = tensor_offset + relative;
 19902:     if (expert_bytes > model->size - offset ||
 19903:         offset > (uint64_t)SIZE_MAX ||
 19904:         expert_bytes > (uint64_t)SIZE_MAX - offset) {
 19905:         return false;
 19906:     }
 19907:     *source = model->map + (size_t)offset;
 19908:     return true;
 19909: }
 19910: 
 19911: static uint64_t dynamic_arena_fnv1a64(const uint8_t *data, uint64_t bytes) {
 19912:     uint64_t checksum = UINT64_C(14695981039346656037);
 19913:     for (uint64_t i = 0; i < bytes; i++) {
 19914:         checksum ^= data[i];
 19915:         checksum *= UINT64_C(1099511628211);
 19916:     }
 19917:     return checksum;
 19918: }
 19919: 
 19920: static bool dynamic_arena_copy_load(
 19921:         const ds4_model                    *model,
 19922:         const ds4_gpu_dynamic_arena_layer  layers[DS4_N_LAYER],
 19923:         const ds4_gpu_dynamic_arena_load  *load,
 19924:         uint64_t                          *checksum) {
 19925:     if (!load || !load->host_ptr || !checksum ||
 19926:         load->layer >= DS4_N_LAYER || load->expert >= DS4_N_EXPERT) {
 19927:         return false;
 19928:     }
 19929:     const ds4_gpu_dynamic_arena_layer *layer = &layers[load->layer];
 19930:     if (layer->gate_expert_bytes > UINT64_MAX - layer->up_expert_bytes) {
 19931:         return false;
 19932:     }
 19933:     const uint64_t gate_up_bytes =
 19934:         layer->gate_expert_bytes + layer->up_expert_bytes;
 19935:     if (gate_up_bytes > UINT64_MAX - layer->down_expert_bytes) return false;
 19936:     const uint64_t slot_bytes = gate_up_bytes + layer->down_expert_bytes;
 19937:     if (slot_bytes == 0 || load->host_bytes != slot_bytes ||
 19938:         slot_bytes > (uint64_t)SIZE_MAX) {
 19939:         return false;
 19940:     }
 19941: 
 19942:     const uint8_t *gate = NULL;
 19943:     const uint8_t *up = NULL;
 19944:     const uint8_t *down = NULL;
 19945:     if (!dynamic_arena_source(model, layer->gate_offset,
 19946:                               layer->gate_expert_bytes, load->expert, &gate) ||
 19947:         !dynamic_arena_source(model, layer->up_offset,
 19948:                               layer->up_expert_bytes, load->expert, &up) ||
 19949:         !dynamic_arena_source(model, layer->down_offset,
 19950:                               layer->down_expert_bytes, load->expert, &down)) {
 19951:         return false;
 19952:     }
 19953: 
 19954:     uint8_t *destination = load->host_ptr;
 19955:     memcpy(destination, gate, (size_t)layer->gate_expert_bytes);
 19956:     memcpy(destination + (size_t)layer->gate_expert_bytes,
 19957:            up, (size_t)layer->up_expert_bytes);
 19958:     memcpy(destination + (size_t)gate_up_bytes,
 19959:            down, (size_t)layer->down_expert_bytes);
 19960:     *checksum = dynamic_arena_fnv1a64(destination, slot_bytes);
 19961:     return true;
 19962: }
 19963: 
 19964: static uint32_t dynamic_arena_test_keep(void) {
 19965:     const char *env = getenv("DS4_CUDA_DYNAMIC_ARENA_TEST_KEEP");
 19966:     if (!env || !env[0]) return 0;
 19967:     char *end = NULL;
 19968:     errno = 0;
 19969:     const unsigned long value = strtoul(env, &end, 10);
 19970:     if (errno != 0 || end == env || *end != '\0' ||
 19971:         value < 1 || value > DS4_N_EXPERT) {
 19972:         return 0;
 19973:     }
 19974:     return (uint32_t)value;
 19975: }
 19976: 
 19977: /* Disabled-by-default synchronous WRAP fixture; this is not a residency policy. */
 19978: static bool dynamic_arena_wrap_fixture(
 19979:         const ds4_model                    *model,
 19980:         const ds4_gpu_dynamic_arena_layer  layers[DS4_N_LAYER],
 19981:         uint32_t                            keep) {
 19982:     if (keep < 1 || keep > DS4_N_EXPERT) return false;
 19983:     const uint32_t entry_count = DS4_N_LAYER * DS4_N_EXPERT;
 19984:     uint8_t target_resident[DS4_N_LAYER * DS4_N_EXPERT];
 19985:     memset(target_resident, 0, sizeof(target_resident));
 19986:     for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
 19987:         memset(target_resident + il * DS4_N_EXPERT, 1, keep);
 19988:     }
 19989: 
 19990:     ds4_gpu_dynamic_arena_txn *txn = NULL;
 19991:     const ds4_gpu_dynamic_arena_load *loads = NULL;
 19992:     uint32_t load_count = 0;
 19993:     if (!ds4_gpu_dynamic_arena_begin(
 19994:             target_resident, entry_count, &txn, &loads, &load_count) || !txn) {
 19995:         fprintf(stderr,
 19996:                 "ds4: CUDA dynamic arena WRAP fixture begin failed (keep=%u)\n",
 19997:                 keep);
 19998:         if (txn) ds4_gpu_dynamic_arena_abort(txn);
 19999:         return false;
 20000:     }
 20001: 
 20002:     bool all_succeeded = loads != NULL || load_count == 0;
 20003:     for (uint32_t i = 0; i < load_count; i++) {
 20004:         uint64_t checksum = 0;
 20005:         const bool copied = loads &&
 20006:             dynamic_arena_copy_load(model, layers, &loads[i], &checksum);
 20007:         const bool finished = ds4_gpu_dynamic_arena_finish_load(
 20008:             txn, i, checksum, copied ? 1 : 0) != 0;
 20009:         if (!copied || !finished) {
 20010:             all_succeeded = false;
 20011:             fprintf(stderr,
 20012:                     "ds4: CUDA dynamic arena WRAP fixture load %u failed\n", i);
 20013:         }
 20014:     }
 20015: 
 20016:     uint64_t snapshot_generation = 0;
 20017:     if (all_succeeded &&
 20018:         ds4_gpu_dynamic_arena_publish(txn, &snapshot_generation)) {
 20019:         fprintf(stderr,
 20020:                 "ds4: CUDA dynamic arena WRAP fixture published keep=%u "
 20021:                 "loads=%u generation=%" PRIu64 "\n",
 20022:                 keep, load_count, snapshot_generation);
 20023:         return true;
 20024:     }
 20025: 
 20026:     ds4_gpu_dynamic_arena_abort(txn);
 20027:     fprintf(stderr,
 20028:             "ds4: CUDA dynamic arena WRAP fixture aborted (keep=%u loads=%u)\n",
 20029:             keep, load_count);
 20030:     return false;
    20: typedef struct ds4_gpu_async_read ds4_gpu_async_read;
    21: typedef struct ds4_gpu_dynamic_arena_txn ds4_gpu_dynamic_arena_txn;
    22: 
    23: typedef struct {
    24:     uint64_t gate_offset;
    25:     uint64_t up_offset;
    26:     uint64_t down_offset;
    27:     uint64_t gate_expert_bytes;
    28:     uint64_t up_expert_bytes;
    29:     uint64_t down_expert_bytes;
    30: } ds4_gpu_dynamic_arena_layer;
    31: 
    32: typedef struct {
    33:     uint32_t layer;
    34:     uint32_t expert;
    35:     uint32_t slot;
    36:     uint64_t slot_generation;
    37:     void *host_ptr;
    38:     uint64_t host_bytes;
    39: } ds4_gpu_dynamic_arena_load;
    40: 
    41: int ds4_gpu_init(void);
    42: 
    43: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
    44: /* Decode-thread-only host attribution.  The server owns token/request
    45:  * boundaries; CUDA contributes disjoint wall-clock spans without introducing
    46:  * device events, copies, or synchronization. */
    47: void ds4_gpu_g130_attribution_request_begin(double decode_started);
    48: void ds4_gpu_g130_attribution_token_begin(uint64_t token_index,
    49:                                            double decode_started);
    50: void ds4_gpu_g130_attribution_token_end(uint64_t token_index,
    51:                                          uint32_t token_count);
    52: void ds4_gpu_g130_attribution_request_end(double decode_finished);
    53: #endif
    54: void ds4_gpu_cleanup(void);
    55: 
    56: ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
    57: ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
    58: void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
    59: uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
    60: void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
    61: int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
    62: int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
    63: ds4_gpu_async_read *ds4_gpu_async_read_alloc(uint64_t bytes);
    64: ds4_gpu_async_read *ds4_gpu_async_read_ring_alloc(uint64_t bytes, uint32_t slots);
    65: void ds4_gpu_async_read_free(ds4_gpu_async_read *readback);
    66: void *ds4_gpu_async_read_host(ds4_gpu_async_read *readback);
    67: void *ds4_gpu_async_read_host_slot(ds4_gpu_async_read *readback, uint32_t slot);
    68: int ds4_gpu_tensor_read_async(ds4_gpu_async_read *readback,
    69:                               const ds4_gpu_tensor *tensor,
    70:                               uint64_t offset,
    71:                               uint64_t bytes);
    72: int ds4_gpu_tensor_read_async_slot(ds4_gpu_async_read *readback,
    73:                                    uint32_t slot,
    74:                                    const ds4_gpu_tensor *tensor,
    75:                                    uint64_t offset,
    76:                                    uint64_t bytes);
    77: int ds4_gpu_tensor_read_async_ordered_slot(ds4_gpu_async_read *readback,
    78:                                            uint32_t slot,
    79:                                            const ds4_gpu_tensor *tensor,
    80:                                            uint64_t offset,
    81:                                            uint64_t bytes);
    82: int ds4_gpu_async_read_ready(ds4_gpu_async_read *readback);
    83: int ds4_gpu_async_read_ready_slot(ds4_gpu_async_read *readback, uint32_t slot);
    84: int ds4_gpu_async_read_wait(ds4_gpu_async_read *readback);
    85: int ds4_gpu_async_read_wait_slot(ds4_gpu_async_read *readback, uint32_t slot);
    86: ds4_gpu_spex_queue *ds4_gpu_spex_queue_create(
    87:         const os_file_t *model_file,
    88:         uint64_t model_size,
    89:         uint32_t slots,
    90:         uint32_t expert_cap,
    91:         uint64_t gate_expert_bytes,
    92:         uint64_t down_expert_bytes);
    93: void ds4_gpu_spex_queue_destroy(ds4_gpu_spex_queue *queue);
    94: void ds4_gpu_spex_queue_reset(ds4_gpu_spex_queue *queue, uint64_t epoch);
    95: int ds4_gpu_spex_queue_submit(ds4_gpu_spex_queue *queue,
    96:                               const ds4_gpu_spex_job *job);
    97: void ds4_gpu_spex_queue_cancel(ds4_gpu_spex_queue *queue,
    98:                                const ds4_gpu_spex_key *key);
    99: int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
   100:                           const ds4_gpu_tensor *src, uint64_t src_offset,
   101:                           uint64_t bytes);
   102: 
   103: int ds4_gpu_begin_commands(void);
   104: int ds4_gpu_flush_commands(void);
   105: int ds4_gpu_end_commands(void);
   106: int ds4_gpu_synchronize(void);
   107: 
   108: int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
   109: int ds4_gpu_set_model_file(const os_file_t *file);
   110: int ds4_gpu_set_iq1_s_sidecar(const os_file_t *file,
   111:                               const void *model_map,
   112:                               uint64_t model_size);
   113: int ds4_gpu_set_q1_0_sidecar(const os_file_t *file,
   114:                              const void *model_map,
   115:                              uint64_t model_size);
   116: int ds4_gpu_set_nested_residual_sidecar(const os_file_t *file,
   117:                                         const void *source_map,
   118:                                         uint64_t source_size);
   119: int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size);
   120: 
   121: /* Native host-pinned expert arena. The arena is an explicit-copy DMA source;
   122:  * it is deliberately not device-mapped and consumes no proportional VRAM. */
   123: int ds4_gpu_dynamic_arena_bind(
   124:         const void *model_map, uint64_t model_size,
   125:         const ds4_gpu_dynamic_arena_layer *layers,
   126:         uint32_t n_layer, uint32_t n_expert);
   127: /* Metadata-only IQ2 catalog used when the physical host arena is Q1_0. */
   128: int ds4_gpu_set_primary_moe_geometry(
   129:         const void *model_map, uint64_t model_size,
   130:         const ds4_gpu_dynamic_arena_layer *layers,
   131:         uint32_t n_layer, uint32_t n_expert);
   132: int ds4_gpu_dynamic_arena_bind_q1_0(
   133:         const void *model_map, uint64_t model_size,
   134:         const ds4_gpu_dynamic_arena_layer *layers,
   135:         uint32_t n_layer, uint32_t n_expert,
   136:         uint32_t active_layer_first, uint32_t active_layer_last);
   137: int ds4_gpu_dynamic_arena_bind_q1_0_snapshot(
   138:         const void *model_map, uint64_t model_size,
   139:         const ds4_gpu_dynamic_arena_layer *layers,
   140:         uint32_t n_layer, uint32_t n_expert,
   141:         uint32_t active_layer_first, uint32_t active_layer_last);
   142: int ds4_gpu_dynamic_arena_prepare_q1_0(
   143:         uint64_t requested_bytes,
   144:         uint64_t *allocated_bytes,
   145:         uint32_t *slot_count,
   146:         uint64_t *snapshot_generation);
   147: int ds4_gpu_dynamic_arena_prepare(
   148:         uint64_t requested_bytes,
   149:         uint64_t *allocated_bytes,
   150:         uint32_t *slot_count);
   151: int ds4_gpu_dynamic_arena_begin(
   152:         const uint8_t *target_resident,
   153:         uint32_t entry_count,
   154:         ds4_gpu_dynamic_arena_txn **out_txn,
   155:         const ds4_gpu_dynamic_arena_load **out_loads,
   156:         uint32_t *out_load_count);
   157: int ds4_gpu_dynamic_arena_finish_load(
   158:         ds4_gpu_dynamic_arena_txn *txn,
   159:         uint32_t load_index,
   160:         uint64_t checksum,
   161:         int success);
   162: int ds4_gpu_dynamic_arena_publish(
   163:         ds4_gpu_dynamic_arena_txn *txn,
   164:         uint64_t *snapshot_generation);
   165: void ds4_gpu_dynamic_arena_abort(ds4_gpu_dynamic_arena_txn *txn);
   166: void ds4_gpu_dynamic_arena_request_begin(void);
   167: void ds4_gpu_dynamic_arena_observer_reset(void);
   168: void ds4_gpu_dynamic_arena_release(void);
   169: 
   170: int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size, uint64_t offset, uint64_t bytes, const char *label);
   171: int ds4_gpu_model_range_update(const void *model_map, uint64_t offset, const void *data, uint64_t bytes);
   172: void ds4_gpu_reap_router_bias_reset(void);
   173: int ds4_gpu_reap_router_bias_update(uint32_t layer_index, const float *bias, uint32_t count);
   174: int ds4_gpu_sparse_bake_set_retained_mask(
   175:         const uint8_t *mask,

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 31200 31445; Show-Lines $p 31790 32110' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 839ms:
 31200:         return 1;
 31201:     }
 31202:     if (cache->device_route_split_down) {
 31203:         (void)cudaFree(cache->device_route_split_down);
 31204:         cache->device_route_split_down = NULL;
 31205:         cache->device_route_split_down_bytes = 0;
 31206:     }
 31207:     if (cudaMalloc((void **)&cache->device_route_split_down, (size_t)bytes) !=
 31208:         cudaSuccess) {
 31209:         (void)cudaGetLastError();
 31210:         return 0;
 31211:     }
 31212:     cache->device_route_split_down_bytes = bytes;
 31213:     return 1;
 31214: }
 31215: 
 31216: static int cuda_moe_selected_load_q1_0(
 31217:         const void *model_map, uint64_t model_size,
 31218:         uint32_t layer_index,
 31219:         uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
 31220:         uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
 31221:         uint32_t n_total_expert, uint32_t n_expert, uint32_t n_tokens,
 31222:         const ds4_gpu_tensor *selected_arg) {
 31223:     if (!g_q1_0_sidecar_file_valid ||
 31224:         model_map != g_q1_0_sidecar_host_base ||
 31225:         model_size != g_q1_0_sidecar_size ||
 31226:         getenv("DS4_Q1_0_SELECTED_LOAD") == NULL) {
 31227:         return 0;
 31228:     }
 31229:     if (!selected_arg || !selected_arg->ptr ||
 31230:         n_total_expert == 0 || n_expert == 0 || n_tokens == 0 ||
 31231:         gate_expert_bytes == 0 || down_expert_bytes == 0 ||
 31232:         n_tokens > UINT32_MAX / n_expert) {
 31233:         return 0;
 31234:     }
 31235:     const uint32_t slot_count = n_tokens * n_expert;
 31236:     if (selected_arg->bytes < (uint64_t)slot_count * sizeof(int32_t)) return 0;
 31237: 
 31238:     g_moe_gather.h_sel.resize(slot_count);
 31239: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31240:     cuda_g130_attribution_state *attribution_state =
 31241:         cuda_g130_attribution_token_state();
 31242:     cuda_g130_attribution_span attribution_previous = CUDA_G130_ATTRIB_NONE;
 31243:     if (attribution_state) {
 31244:         attribution_previous = cuda_g130_attribution_switch(
 31245:             attribution_state,
 31246:             CUDA_G130_ATTRIB_SELECTION_D2H);
 31247:     }
 31248: #endif
 31249:     const int selected_d2h_ok = cuda_ok(
 31250:         cudaMemcpy(g_moe_gather.h_sel.data(), selected_arg->ptr,
 31251:                    (size_t)slot_count * sizeof(int32_t),
 31252:                    cudaMemcpyDeviceToHost),
 31253:         "Q1_0 selected D2H");
 31254: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31255:     if (attribution_state) {
 31256:         (void)cuda_g130_attribution_switch(
 31257:             attribution_state, attribution_previous);
 31258:     }
 31259: #endif
 31260:     if (!selected_d2h_ok) {
 31261:         return 0;
 31262:     }
 31263:     if (!cuda_sparse_bake_validate_selected(
 31264:             layer_index, g_moe_gather.h_sel.data(), slot_count,
 31265:             "Q1_0 routed selection", 0u)) {
 31266:         return 0;
 31267:     }
 31268: 
 31269:     std::vector<int32_t> &e2s = g_moe_gather.h_expert_to_slot;
 31270:     e2s.assign(n_total_expert, -1);
 31271:     std::vector<int32_t> &compact = g_moe_gather.h_compact_ids;
 31272:     compact.clear();
 31273:     std::vector<int32_t> &slots = g_moe_gather.h_slot_ids;
 31274:     slots.resize(slot_count);
 31275:     for (uint32_t i = 0; i < slot_count; i++) {
 31276:         const int32_t e = g_moe_gather.h_sel[i];
 31277:         if (e < 0 || (uint32_t)e >= n_total_expert) return 0;
 31278:         e2s[(uint32_t)e] = -2;
 31279:     }
 31280:     for (uint32_t e = 0; e < n_total_expert; e++) {
 31281:         if (e2s[e] != -2) continue;
 31282:         e2s[e] = (int32_t)compact.size();
 31283:         compact.push_back((int32_t)e);
 31284:     }
 31285:     for (uint32_t i = 0; i < slot_count; i++) {
 31286:         slots[i] = e2s[(uint32_t)g_moe_gather.h_sel[i]];
 31287:     }
 31288:     const uint32_t compact_count = (uint32_t)compact.size();
 31289:     if (compact_count == 0 || compact_count > n_total_expert) return 0;
 31290: 
 31291:     const uint64_t cgate = (uint64_t)compact_count * gate_expert_bytes;
 31292:     const uint64_t cdown = (uint64_t)compact_count * down_expert_bytes;
 31293:     if (!cuda_moe_gather_ensure(&g_moe_gather.gate, &g_moe_gather.gate_cap,
 31294:                                 cgate, "Q1_0 gather gate") ||
 31295:         !cuda_moe_gather_ensure(&g_moe_gather.up, &g_moe_gather.up_cap,
 31296:                                 cgate, "Q1_0 gather up") ||
 31297:         !cuda_moe_gather_ensure(&g_moe_gather.down, &g_moe_gather.down_cap,
 31298:                                 cdown, "Q1_0 gather down") ||
 31299:         !cuda_moe_gather_ensure_i32(&g_moe_gather.slot, &g_moe_gather.slot_cap,
 31300:                                     slot_count, "Q1_0 gather slots")) {
 31301:         return 0;
 31302:     }
 31303: 
 31304:     if (cuda_q1_0_resident_transport_requested()) {
 31305:         cuda_dynamic_arena *q1_arena =
 31306:             cuda_q1_0_route_arena(layer_index);
 31307:         if (!q1_arena ||
 31308:             q1_arena->backing != CUDA_DYNAMIC_ARENA_BACKING_Q1_0 ||
 31309:             q1_arena->model_map != model_map ||
 31310:             q1_arena->model_size != model_size) {
 31311:             g_q1_0_resident_misses++;
 31312:             return 0;
 31313:         }
 31314:         uint64_t route_h2d_bytes = 0;
 31315:         const uint64_t pinned_h2d_before =
 31316:             q1_arena->pinned_bytes_uploaded;
 31317:         const uint64_t pageable_h2d_before =
 31318:             q1_arena->pageable_bytes_uploaded;
 31319:         for (uint32_t i = 0; i < compact_count; i++) {
 31320:             const uint32_t expert = (uint32_t)compact[i];
 31321:             const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
 31322:             const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
 31323:             const cuda_dynamic_arena_copy_status status =
 31324:                 cuda_dynamic_arena_copy_expert_async(
 31325:                     *q1_arena,
 31326:                     model_map, layer_index, expert,
 31327:                     gate_offset, up_offset, down_offset,
 31328:                     g_moe_gather.gate + gate_dst,
 31329:                     g_moe_gather.up + gate_dst,
 31330:                     g_moe_gather.down + down_dst,
 31331:                     gate_expert_bytes, gate_expert_bytes,
 31332:                     down_expert_bytes);
 31333:             if (status != CUDA_DYNAMIC_ARENA_ENQUEUED) {
 31334:                 g_q1_0_resident_misses++;
 31335: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31336:                 cuda_g130_attribution_span sync_previous =
 31337:                     CUDA_G130_ATTRIB_NONE;
 31338:                 if (attribution_state) {
 31339:                     sync_previous = cuda_g130_attribution_switch(
 31340:                         attribution_state,
 31341:                         CUDA_G130_ATTRIB_EXISTING_STREAM_SYNC_WAIT);
 31342:                 }
 31343: #endif
 31344:                 (void)cudaStreamSynchronize(g_model_upload_stream);
 31345: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31346:                 if (attribution_state) {
 31347:                     (void)cuda_g130_attribution_switch(
 31348:                         attribution_state, sync_previous);
 31349:                 }
 31350: #endif
 31351:                 return 0;
 31352:             }
 31353:             route_h2d_bytes += gate_expert_bytes * 2u + down_expert_bytes;
 31354:         }
 31355: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31356:         cuda_g130_attribution_span h2d_previous = CUDA_G130_ATTRIB_NONE;
 31357:         if (attribution_state) {
 31358:             h2d_previous = cuda_g130_attribution_switch(
 31359:                 attribution_state,
 31360:                 CUDA_G130_ATTRIB_H2D_ENQUEUE);
 31361:         }
 31362: #endif
 31363:         const int slots_h2d_ok = cuda_ok(
 31364:             cudaMemcpyAsync(g_moe_gather.slot, slots.data(),
 31365:                             (size_t)slot_count * sizeof(int32_t),
 31366:                             cudaMemcpyHostToDevice, g_model_upload_stream),
 31367:             "Q1_0 resident slots H2D");
 31368: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31369:         if (attribution_state) {
 31370:             (void)cuda_g130_attribution_switch(
 31371:                 attribution_state, h2d_previous);
 31372:         }
 31373: #endif
 31374:         if (!slots_h2d_ok) {
 31375:             return 0;
 31376:         }
 31377:         const int q1_profile = cuda_q1_0_profile_requested();
 31378:         const double sync_started = q1_profile ? cuda_wall_sec() : 0.0;
 31379: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31380:         cuda_g130_attribution_span sync_previous = CUDA_G130_ATTRIB_NONE;
 31381:         if (attribution_state) {
 31382:             sync_previous = cuda_g130_attribution_switch(
 31383:                 attribution_state,
 31384:                 CUDA_G130_ATTRIB_EXISTING_STREAM_SYNC_WAIT);
 31385:         }
 31386: #endif
 31387:         const cudaError_t sync_error =
 31388:             cudaStreamSynchronize(g_model_upload_stream);
 31389: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31390:         if (attribution_state) {
 31391:             (void)cuda_g130_attribution_switch(
 31392:                 attribution_state, sync_previous);
 31393:         }
 31394: #endif
 31395:         if (q1_profile) {
 31396:             const double sync_seconds = cuda_wall_sec() - sync_started;
 31397:             const uint64_t pinned_h2d =
 31398:                 q1_arena->pinned_bytes_uploaded - pinned_h2d_before;
 31399:             const uint64_t pageable_h2d =
 31400:                 q1_arena->pageable_bytes_uploaded - pageable_h2d_before;
 31401:             const uint64_t attributed_h2d = pinned_h2d + pageable_h2d;
 31402:             g_q1_0_profile.upload_sync_calls++;
 31403:             g_q1_0_profile.upload_sync_seconds += sync_seconds;
 31404:             if (attributed_h2d != 0) {
 31405:                 const double pinned_seconds = sync_seconds *
 31406:                     (double)pinned_h2d / (double)attributed_h2d;
 31407:                 g_q1_0_profile.pinned_upload_sync_seconds +=
 31408:                     pinned_seconds;
 31409:                 g_q1_0_profile.pageable_upload_sync_seconds +=
 31410:                     sync_seconds - pinned_seconds;
 31411:             }
 31412:         }
 31413:         if (!cuda_ok(sync_error, "Q1_0 resident compact upload sync")) {
 31414:             return 0;
 31415:         }
 31416:         g_q1_0_resident_hits += compact_count;
 31417:         g_q1_0_resident_h2d_bytes += route_h2d_bytes;
 31418:     } else {
 31419:         std::vector<uint8_t> host_gate;
 31420:         std::vector<uint8_t> host_up;
 31421:         std::vector<uint8_t> host_down;
 31422:         try {
 31423:             host_gate.resize((size_t)cgate);
 31424:             host_up.resize((size_t)cgate);
 31425:             host_down.resize((size_t)cdown);
 31426:         } catch (...) {
 31427:             return 0;
 31428:         }
 31429: 
 31430:         g_q1_0_direct_pread_fallbacks++;
 31431:         for (uint32_t i = 0; i < compact_count; i++) {
 31432:             const uint32_t expert = (uint32_t)compact[i];
 31433:             const uint64_t gate_src = gate_offset +
 31434:                 (uint64_t)expert * gate_expert_bytes;
 31435:             const uint64_t up_src = up_offset +
 31436:                 (uint64_t)expert * gate_expert_bytes;
 31437:             const uint64_t down_src = down_offset +
 31438:                 (uint64_t)expert * down_expert_bytes;
 31439:             if (gate_src > g_q1_0_sidecar_size ||
 31440:                 gate_expert_bytes > g_q1_0_sidecar_size - gate_src ||
 31441:                 up_src > g_q1_0_sidecar_size ||
 31442:                 gate_expert_bytes > g_q1_0_sidecar_size - up_src ||
 31443:                 down_src > g_q1_0_sidecar_size ||
 31444:                 down_expert_bytes > g_q1_0_sidecar_size - down_src) {
 31445:                 return 0;
 31790:         layer_index, gate_offset, up_offset, down_offset,
 31791:         gate_expert_bytes, down_expert_bytes, expert_in_dim,
 31792:         selected, weights, n_expert, n_tokens, spex_queue, 0u, &sequence);
 31793:     return cuda_moe_gpu_resident_routes_finish(
 31794:         cache, sequence, layer_index, 0, 0u);
 31795: }
 31796: 
 31797: static int routed_moe_launch(
 31798:         ds4_gpu_tensor *out,
 31799:         ds4_gpu_tensor *gate,
 31800:         ds4_gpu_tensor *up,
 31801:         ds4_gpu_tensor *mid,
 31802:         ds4_gpu_tensor *down,
 31803:         const void *model_map,
 31804:         uint64_t model_size,
 31805:         uint32_t layer_index,
 31806:         uint64_t gate_offset,
 31807:         uint64_t up_offset,
 31808:         uint64_t down_offset,
 31809:         uint32_t gate_type,
 31810:         uint32_t down_type,
 31811:         uint64_t gate_expert_bytes,
 31812:         uint64_t gate_row_bytes,
 31813:         uint64_t down_expert_bytes,
 31814:         uint64_t down_row_bytes,
 31815:         uint32_t expert_in_dim,
 31816:         uint32_t expert_mid_dim,
 31817:         uint32_t out_dim,
 31818:         const ds4_gpu_tensor *selected,
 31819:         const ds4_gpu_tensor *weights,
 31820:         const ds4_gpu_tensor *probs,
 31821:         uint32_t n_expert,
 31822:         float clamp,
 31823:         const ds4_gpu_tensor *x,
 31824:         uint32_t n_tokens,
 31825:         ds4_gpu_spex_queue *spex_queue,
 31826:         const ds4_gpu_spex_key *spex_key) {
 31827:     g_moe_last_selected.valid = 0;
 31828:     /* Wave state belongs to this routed-MoE invocation. Decode can take the
 31829:      * resident-route path without calling selected-load, so stale prefill
 31830:      * activation must never survive into the next layer/token. Per-parity
 31831:      * compute ownership is persistent and is cleared only by the reuse fence
 31832:      * in cuda_moe_selected_stage_wave(). */
 31833:     g_moe_gather.wave_active = 0;
 31834:     g_moe_gather.wave_fail_closed = 0;
 31835:     g_moe_gather.wave_double_buffer = 0;
 31836:     for (uint32_t i = 0; i < 2u; i++) {
 31837:         g_moe_gather.wave_active_pairs_db[i] = 0;
 31838:         g_moe_gather.wave_current_experts_db[i] = 0;
 31839:     }
 31840:     if (!out || !gate || !up || !mid || !down || !model_map || !selected || !weights || !x ||
 31841:         n_tokens == 0 || n_expert == 0 ||
 31842:         expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
 31843:         gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
 31844:         x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
 31845:         selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
 31846:         weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
 31847:         gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
 31848:         up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
 31849:         mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
 31850:         down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
 31851:         out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) {
 31852:         return 0;
 31853:     }
 31854:     const uint32_t route_iq2_q2 = gate_type == 16u && down_type == 10u;
 31855:     const uint32_t route_iq1_s =
 31856:         gate_type == 19u && (down_type == 19u || down_type == 10u);
 31857:     const uint32_t route_q1_0 =
 31858:         gate_type == 41u && (down_type == 41u || down_type == 10u);
 31859:     if (!route_iq2_q2 && !route_iq1_s && !route_q1_0) return 0;
 31860: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 31861:     cuda_g130_attribution_state *attribution_state =
 31862:         cuda_g130_attribution_token_state();
 31863: #endif
 31864:     if (route_iq1_s) {
 31865:         g_iq1_s_route_calls++;
 31866:         g_iq1_s_route_slots += (uint64_t)n_tokens * n_expert;
 31867:         if (getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") != NULL) {
 31868:             g_iq1_s_selected_load_failures++;
 31869:             fprintf(stderr,
 31870:                     "ds4: IQ1_S sidecar requires selected expert loading at layer=%u\n",
 31871:                     layer_index);
 31872:             return 0;
 31873:         }
 31874:     }
 31875:     if (route_q1_0) {
 31876:         g_q1_0_route_calls++;
 31877:         g_q1_0_route_slots += (uint64_t)n_tokens * n_expert;
 31878:         if (down_type != 41u) {
 31879:             g_q1_0_selected_load_failures++;
 31880:             fprintf(stderr,
 31881:                     "ds4: Q1_0 sidecar requires Q1_0 down tensor at layer=%u\n",
 31882:                     layer_index);
 31883:             return 0;
 31884:         }
 31885:         if (getenv("DS4_Q1_0_SELECTED_LOAD") == NULL ||
 31886:             getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") != NULL) {
 31887:             g_q1_0_selected_load_failures++;
 31888:             fprintf(stderr,
 31889:                     "ds4: Q1_0 sidecar requires opt-in selected expert loading at layer=%u\n",
 31890:                     layer_index);
 31891:             return 0;
 31892:         }
 31893:     }
 31894:     const uint32_t n_total_expert = 256u;
 31895:     const uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
 31896:     const uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
 31897:     if (gate_bytes > model_size - gate_offset ||
 31898:         gate_bytes > model_size - up_offset ||
 31899:         down_bytes > model_size - down_offset) {
 31900:         return 0;
 31901:     }
 31902:     const int sparse_bake_layer =
 31903:         cuda_sparse_bake_layer_is_sparse(layer_index);
 31904:     if (sparse_bake_layer &&
 31905:         getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") != NULL) {
 31906:         fprintf(stderr,
 31907:                 "ds4: sparse bake requires selected expert loading at layer=%u\n",
 31908:                 layer_index);
 31909:         return 0;
 31910:     }
 31911:     if (sparse_bake_layer) {
 31912:         spex_queue = NULL;
 31913:         spex_key = NULL;
 31914:     }
 31915:     const char *gate_w = NULL;
 31916:     const char *up_w = NULL;
 31917:     const char *down_w = NULL;
 31918:     uint32_t use_mixed_route_ptrs = 0;
 31919:     uint32_t use_split_hit_miss = 0;
 31920:     uint32_t use_split_fused = 0;
 31921:     uint32_t gpu_route_sequence = 0;
 31922:     const uint64_t *mixed_route_ptrs = NULL;
 31923:     const int nested_layer_required =
 31924:         cuda_nested_residual_layer_required(layer_index);
 31925:     const int nested_gpu_cache_requested = nested_layer_required &&
 31926:         cuda_nested_residual_gpu_cache_requested();
 31927:     if (nested_layer_required &&
 31928:         (getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") != NULL ||
 31929:          getenv("DS4_CUDA_PREFILL_WAVES") != NULL ||
 31930:          spex_queue != NULL)) {
 31931:         g_nested_residual.failures++;
 31932:         g_nested_residual.hard_failure = 1;
 31933:         fprintf(stderr,
 31934:                 "ds4: nested residual covered layer requires exclusive "
 31935:                 "selected-load transport layer=%u\n",
 31936:                 layer_index);
 31937:         return 0;
 31938:     }
 31939:     if (g_nested_residual.active && g_nested_residual.hard_failure) {
 31940:         fprintf(stderr,
 31941:                 "ds4: nested residual request refused after hard failure\n");
 31942:         return 0;
 31943:     }
 31944:     /* Selected-expert load: when the whole 256-expert block is beyond the pinned
 31945:      * host window (streaming regime), fetch/gather ONLY the routed experts into a
 31946:      * compact VRAM buffer and rebind `selected` to compact slots -- the unchanged
 31947:      * MoE kernels then touch top-6 of 256 (~42x less streamed per layer). Falls
 31948:      * back to the whole-block path on failure or when fully in-window (zero-copy). */
 31949:     const int whole_in_window =
 31950:         !route_iq1_s && !route_q1_0 && !sparse_bake_layer &&
 31951:         !nested_layer_required &&
 31952:         cuda_model_range_in_window(model_map, gate_offset, gate_bytes) &&
 31953:         cuda_model_range_in_window(model_map, up_offset, gate_bytes) &&
 31954:         cuda_model_range_in_window(model_map, down_offset, down_bytes);
 31955:     cuda_moe_expert_cache *gpu_route_cache = NULL;
 31956:     const int split_hit_miss_requested =
 31957:         (route_iq1_s || route_q1_0) ? 0 : cuda_moe_split_hit_miss_requested();
 31958:     const int split_fused_requested =
 31959:         (route_iq1_s || route_q1_0) ? 0 : cuda_moe_split_fused_requested();
 31960:     if (split_hit_miss_requested && split_fused_requested) {
 31961:         fprintf(stderr,
 31962:                 "ds4: split hit/miss and fused split are mutually exclusive\n");
 31963:         return 0;
 31964:     }
 31965:     if (!route_iq1_s && !route_q1_0 &&
 31966:         (!nested_layer_required || nested_gpu_cache_requested) &&
 31967:         !whole_in_window &&
 31968:         getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") == NULL) {
 31969:         if (!sparse_bake_layer &&
 31970:             (split_hit_miss_requested || split_fused_requested) &&
 31971:             getenv("DS4_CUDA_MOE_PROFILE") == NULL) {
 31972:             gpu_route_cache = cuda_moe_gpu_resident_routes_begin(
 31973:                 layer_index,
 31974:                 gate_offset, up_offset, down_offset,
 31975:                 gate_expert_bytes, down_expert_bytes,
 31976:                 expert_in_dim,
 31977:                 selected, weights, n_expert, n_tokens,
 31978:                 spex_queue, out_dim, &gpu_route_sequence);
 31979:             use_split_fused = gpu_route_cache != NULL && split_fused_requested;
 31980:             use_split_hit_miss = gpu_route_cache != NULL && split_hit_miss_requested;
 31981:         } else {
 31982:             gpu_route_cache = cuda_moe_gpu_resident_routes_submit(
 31983:                 layer_index,
 31984:                 gate_offset, up_offset, down_offset,
 31985:                 gate_expert_bytes, down_expert_bytes,
 31986:                 expert_in_dim,
 31987:                 selected, weights, n_expert, n_tokens,
 31988:                 spex_queue);
 31989:         }
 31990:     }
 31991:     if (!route_iq1_s && !route_q1_0 &&
 31992:         (!nested_layer_required || nested_gpu_cache_requested) &&
 31993:         !gpu_route_cache && !whole_in_window && n_tokens == 1u &&
 31994:         (nested_gpu_cache_requested ||
 31995:          cuda_moe_tiering_mode_requested() != CUDA_MOE_TIER_OFF)) {
 31996:         if (nested_gpu_cache_requested) {
 31997:             g_nested_residual.gpu_cache_failures++;
 31998:             g_nested_residual.hard_failure = 1;
 31999:         }
 32000:         fprintf(stderr,
 32001:                 "ds4: %s decode route unavailable at layer=%u; refusing fallback\n",
 32002:                 nested_gpu_cache_requested ?
 32003:                     "nested residual GPU-cache" : "expert tiering",
 32004:                 layer_index);
 32005:         return 0;
 32006:     }
 32007:     if (gpu_route_cache) {
 32008:         gate_w = gpu_route_cache->gate;
 32009:         up_w = gpu_route_cache->up;
 32010:         down_w = gpu_route_cache->down;
 32011:         use_mixed_route_ptrs = 1;
 32012:         mixed_route_ptrs = gpu_route_cache->device_route_ptrs;
 32013:         if (getenv("DS4_IQ1_MIXED_DEBUG") &&
 32014:             n_expert != CUDA_MOE_ROUTE_COUNT) {
 32015:             uint64_t route_ptrs[CUDA_MOE_ROUTE_COUNT * 3u] = {0};
 32016:             if (cuda_ok(cudaMemcpy(
 32017:                             route_ptrs, mixed_route_ptrs,
 32018:                             (size_t)n_expert * 3u * sizeof(uint64_t),
 32019:                             cudaMemcpyDeviceToHost),
 32020:                         "iq1 mixed route pointers D2H")) {
 32021:                 for (uint32_t route = 0; route < n_expert; route++) {
 32022:                     const uint64_t gate_ptr = route_ptrs[route];
 32023:                     const uint64_t up_ptr = route_ptrs[n_expert + route];
 32024:                     const uint64_t down_ptr = route_ptrs[2u * n_expert + route];
 32025:                     const uint64_t gate_base =
 32026:                         (uint64_t)(uintptr_t)gpu_route_cache->gate;
 32027:                     const uint64_t up_base =
 32028:                         (uint64_t)(uintptr_t)gpu_route_cache->up;
 32029:                     const uint64_t down_base =
 32030:                         (uint64_t)(uintptr_t)gpu_route_cache->down;
 32031:                     fprintf(stderr,
 32032:                             "ds4: [iq1-mixed-debug] phase=route-ptrs route=%u "
 32033:                             "gate_slot=%llu gate_rem=%llu up_slot=%llu up_rem=%llu "
 32034:                             "down_slot=%llu down_rem=%llu\n",
 32035:                             route,
 32036:                             (unsigned long long)((gate_ptr - gate_base) /
 32037:                                 gate_expert_bytes),
 32038:                             (unsigned long long)((gate_ptr - gate_base) %
 32039:                                 gate_expert_bytes),
 32040:                             (unsigned long long)((up_ptr - up_base) /
 32041:                                 gate_expert_bytes),
 32042:                             (unsigned long long)((up_ptr - up_base) %
 32043:                                 gate_expert_bytes),
 32044:                             (unsigned long long)((down_ptr - down_base) /
 32045:                                 down_expert_bytes),
 32046:                             (unsigned long long)((down_ptr - down_base) %
 32047:                                 down_expert_bytes));
 32048:                 }
 32049:             }
 32050:         }
 32051:     } else if (route_q1_0) {
 32052:         cuda_q1_0_mixed_profile_sample *mixed_profile_sample =
 32053:             g_q1_0_mixed_profile_current_sample;
 32054:         if (mixed_profile_sample) {
 32055:             const double selected_started = cuda_wall_sec();
 32056:             mixed_profile_sample->q1_entry_calls++;
 32057:             mixed_profile_sample->q1_entry_seconds +=
 32058:                 selected_started - mixed_profile_sample->q1_call_started;
 32059:             mixed_profile_sample->q1_prepare_started = selected_started;
 32060:         }
 32061: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 32062:         cuda_g130_attribution_span attribution_previous =
 32063:             CUDA_G130_ATTRIB_NONE;
 32064:         if (attribution_state) {
 32065:             attribution_previous = cuda_g130_attribution_switch(
 32066:                 attribution_state,
 32067:                 CUDA_G130_ATTRIB_SELECTED_LOAD);
 32068:         }
 32069: #endif
 32070:         const int selected_loaded = cuda_moe_selected_load_q1_0(
 32071:             model_map, model_size, layer_index,
 32072:             gate_offset, up_offset, down_offset,
 32073:             gate_expert_bytes, down_expert_bytes,
 32074:             n_total_expert, n_expert, n_tokens, selected);
 32075: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 32076:         if (attribution_state) {
 32077:             (void)cuda_g130_attribution_switch(
 32078:                 attribution_state, attribution_previous);
 32079:         }
 32080: #endif
 32081:         if (mixed_profile_sample) {
 32082:             const double selected_finished = cuda_wall_sec();
 32083:             mixed_profile_sample->q1_selected_load_calls++;
 32084:             mixed_profile_sample->q1_selected_load_seconds +=
 32085:                 selected_finished - mixed_profile_sample->q1_prepare_started;
 32086:             mixed_profile_sample->q1_prepare_started = selected_finished;
 32087:         }
 32088:         if (selected_loaded) {
 32089:             g_q1_0_selected_loads++;
 32090:             gate_w = g_moe_gather.gate;
 32091:             up_w = g_moe_gather.up;
 32092:             down_w = g_moe_gather.down;
 32093:             selected = &g_moe_gather.slot_tensor;
 32094:         } else {
 32095:             g_q1_0_selected_load_failures++;
 32096:             ds4_gpu_spex_queue_cancel(spex_queue, spex_key);
 32097:             fprintf(stderr,
 32098:                     "ds4: Q1_0 sidecar selected load failed closed at layer=%u\n",
 32099:                     layer_index);
 32100:             return 0;
 32101:         }
 32102:     } else if (!whole_in_window && getenv("DS4_CUDA_MOE_NO_SELECTED_LOAD") == NULL) {
 32103:         const int selected_loaded =
 32104:             cuda_moe_selected_load(model_map, model_size, layer_index,
 32105:                                    gate_offset, up_offset, down_offset,
 32106:                                    gate_expert_bytes, down_expert_bytes,
 32107:                                     expert_in_dim,
 32108:                                     n_total_expert, n_expert, n_tokens,
 32109:                                     selected, weights, probs,
 32110:                                      spex_queue, spex_key, route_iq1_s);

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 33635 34270; Show-Lines $p 34350 34630' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 882ms:
 33635:             phase ? phase : "unknown", count, nonfinite,
 33636:             finite ? min_value : 0.0f, finite ? max_value : 0.0f,
 33637:             finite ? sum_abs / (double)finite : 0.0,
 33638:             finite ? sqrt(sum_sq / (double)finite) : 0.0);
 33639:     return nonfinite == 0u;
 33640: }
 33641: 
 33642: enum cuda_q1_0_mixed_representation : uint8_t {
 33643:     CUDA_Q1_0_MIXED_UNKNOWN = 0,
 33644:     CUDA_Q1_0_MIXED_IQ2_VRAM = 1,
 33645:     CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM = 2,
 33646:     CUDA_Q1_0_MIXED_IQ2_TIER_RAM = 3,
 33647:     CUDA_Q1_0_MIXED_Q1_RESIDENT = 4,
 33648: };
 33649: 
 33650: static const char *cuda_q1_0_mixed_representation_name(
 33651:         cuda_q1_0_mixed_representation representation) {
 33652:     switch (representation) {
 33653:     case CUDA_Q1_0_MIXED_IQ2_VRAM: return "iq2_vram";
 33654:     case CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM: return "iq2_snapshot_ram";
 33655:     case CUDA_Q1_0_MIXED_IQ2_TIER_RAM: return "iq2_tier_ram";
 33656:     case CUDA_Q1_0_MIXED_Q1_RESIDENT: return "q1_resident";
 33657:     default: return "unknown";
 33658:     }
 33659: }
 33660: 
 33661: static int cuda_q1_0_mixed_trace_requested(void) {
 33662:     const char *value = getenv("DS4_Q1_0_MIXED_TRACE");
 33663:     return value && strcmp(value, "1") == 0;
 33664: }
 33665: 
 33666: static int cuda_q1_0_resident_ram_ptrs(
 33667:         uint32_t layer, uint32_t expert,
 33668:         char **gate, char **up, char **down) {
 33669:     cuda_dynamic_arena *arena_ptr = cuda_q1_0_route_arena(layer);
 33670:     if (!arena_ptr || expert >= 256u ||
 33671:         !gate || !up || !down || layer >= arena_ptr->n_layer ||
 33672:         expert >= arena_ptr->n_expert ||
 33673:         arena_ptr->layers.size() != arena_ptr->n_layer) {
 33674:         return 0;
 33675:     }
 33676:     cuda_dynamic_arena &arena = *arena_ptr;
 33677:     const uint32_t entry = layer * arena.n_expert + expert;
 33678:     if (entry >= arena.active.size()) return 0;
 33679:     const cuda_dynamic_arena_binding &binding = arena.active[entry];
 33680:     if (!cuda_dynamic_arena_binding_valid(
 33681:             arena, binding, layer, expert,
 33682:             arena.snapshot_generation, DS4_GPU_ARENA_READY)) {
 33683:         return 0;
 33684:     }
 33685:     const ds4_gpu_dynamic_arena_layer &geometry = arena.layers[layer];
 33686:     cuda_dynamic_arena_slot &slot = arena.slots[binding.slot];
 33687:     *gate = slot.host_ptr;
 33688:     *up = *gate + geometry.gate_expert_bytes;
 33689:     *down = *up + geometry.up_expert_bytes;
 33690:     return 1;
 33691: }
 33692: 
 33693: static int cuda_q1_0_dual_sparse_pair_contains(
 33694:         uint32_t layer, uint32_t expert) {
 33695:     cuda_dynamic_arena &primary = g_dynamic_arena;
 33696:     cuda_dynamic_arena &q1 = g_q1_0_dynamic_arena;
 33697:     if (!cuda_q1_0_dual_sparse_companion_requested() ||
 33698:         !g_q1_0_dual_sparse_snapshot.ready ||
 33699:         g_q1_0_dual_sparse_snapshot.staged ||
 33700:         primary.backing != CUDA_DYNAMIC_ARENA_BACKING_PRIMARY ||
 33701:         q1.backing != CUDA_DYNAMIC_ARENA_BACKING_Q1_0 ||
 33702:         primary.snapshot_generation == 0 ||
 33703:         q1.snapshot_generation != primary.snapshot_generation ||
 33704:         g_q1_0_dual_sparse_snapshot.primary_generation !=
 33705:             primary.snapshot_generation ||
 33706:         g_q1_0_dual_sparse_snapshot.q1_generation !=
 33707:             q1.snapshot_generation ||
 33708:         layer >= primary.n_layer || layer >= q1.n_layer ||
 33709:         expert >= primary.n_expert || expert >= q1.n_expert ||
 33710:         primary.n_layer != q1.n_layer ||
 33711:         primary.n_expert != q1.n_expert) {
 33712:         return 0;
 33713:     }
 33714:     const uint32_t entry = layer * primary.n_expert + expert;
 33715:     if (entry >= primary.active.size() || entry >= q1.active.size()) {
 33716:         return 0;
 33717:     }
 33718:     return cuda_dynamic_arena_binding_valid(
 33719:                primary, primary.active[entry], layer, expert,
 33720:                primary.snapshot_generation, DS4_GPU_ARENA_READY) &&
 33721:         cuda_dynamic_arena_binding_valid(
 33722:                q1, q1.active[entry], layer, expert,
 33723:                q1.snapshot_generation, DS4_GPU_ARENA_READY);
 33724: }
 33725: 
 33726: static cuda_q1_0_mixed_representation cuda_q1_0_mixed_resolve(
 33727:         uint32_t layer, uint32_t expert) {
 33728:     cuda_q1_0_mixed_capture_router_mode();
 33729:     if (cuda_q1_0_snapshot_backing_requested()) {
 33730:         /* The exclusive snapshot physically contains Q1_0 bytes. The only
 33731:          * exact IQ2 representation in this mode is the protected VRAM seed;
 33732:          * generic snapshot/tier RAM pointers would alias Q1_0 slots and must
 33733:          * never be interpreted as IQ2. */
 33734:         if (cuda_moe_tiering_has_exact_vram(layer, expert)) {
 33735:             return CUDA_Q1_0_MIXED_IQ2_VRAM;
 33736:         }
 33737:         char *gate = NULL;
 33738:         char *up = NULL;
 33739:         char *down = NULL;
 33740:         return cuda_q1_0_resident_ram_ptrs(
 33741:                     layer, expert, &gate, &up, &down)
 33742:             ? CUDA_Q1_0_MIXED_Q1_RESIDENT
 33743:             : CUDA_Q1_0_MIXED_UNKNOWN;
 33744:     }
 33745:     if (g_moe_tiering.mode != CUDA_MOE_TIER_ENFORCE ||
 33746:         !g_moe_tiering.compose_prefill_mass_tiering ||
 33747:         !g_moe_tiering.compose_router_open ||
 33748:         layer >= CUDA_MOE_LAYER_COUNT || expert >= 256u ||
 33749:         g_moe_tiering.entries.size() !=
 33750:             (size_t)CUDA_MOE_LAYER_COUNT * 256u) {
 33751:         return CUDA_Q1_0_MIXED_UNKNOWN;
 33752:     }
 33753:     const cuda_moe_tier_entry &tier = g_moe_tiering.entries[
 33754:         cuda_moe_tiering_entry_index(layer, expert)];
 33755:     if (cuda_moe_tiering_has_exact_vram(layer, expert)) {
 33756:         return tier.state == CUDA_MOE_TIER_VRAM_PROTECTED
 33757:             ? CUDA_Q1_0_MIXED_IQ2_VRAM
 33758:             : CUDA_Q1_0_MIXED_UNKNOWN;
 33759:     }
 33760:     char *gate = NULL;
 33761:     char *up = NULL;
 33762:     char *down = NULL;
 33763:     if (cuda_moe_tiering_snapshot_ram_ptrs(
 33764:             layer, expert, &gate, &up, &down)) {
 33765:         return tier.state == CUDA_MOE_TIER_RAM_PROBATION ||
 33766:                tier.state == CUDA_MOE_TIER_RAM_WARM
 33767:             ? CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM
 33768:             : CUDA_Q1_0_MIXED_UNKNOWN;
 33769:     }
 33770:     if (cuda_moe_tiering_ram_ptrs(
 33771:             layer, expert, &gate, &up, &down)) {
 33772:         return tier.state == CUDA_MOE_TIER_RAM_PROBATION ||
 33773:                tier.state == CUDA_MOE_TIER_RAM_WARM
 33774:             ? CUDA_Q1_0_MIXED_IQ2_TIER_RAM
 33775:             : CUDA_Q1_0_MIXED_UNKNOWN;
 33776:     }
 33777:     if (tier.state != CUDA_MOE_TIER_SSD_COLD || tier.has_2bit_ram) {
 33778:         return CUDA_Q1_0_MIXED_UNKNOWN;
 33779:     }
 33780:     if (cuda_q1_0_resident_ram_ptrs(
 33781:             layer, expert, &gate, &up, &down)) {
 33782:         return CUDA_Q1_0_MIXED_Q1_RESIDENT;
 33783:     }
 33784:     return CUDA_Q1_0_MIXED_UNKNOWN;
 33785: }
 33786: 
 33787: extern "C" int ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
 33788:         ds4_gpu_tensor *out,
 33789:         ds4_gpu_tensor *gate,
 33790:         ds4_gpu_tensor *up,
 33791:         ds4_gpu_tensor *mid,
 33792:         ds4_gpu_tensor *down,
 33793:         const void *main_model_map,
 33794:         uint64_t main_model_size,
 33795:         uint32_t layer_index,
 33796:         uint64_t main_gate_offset,
 33797:         uint64_t main_up_offset,
 33798:         uint64_t main_down_offset,
 33799:         uint32_t main_gate_type,
 33800:         uint32_t main_down_type,
 33801:         uint64_t main_gate_expert_bytes,
 33802:         uint64_t main_gate_row_bytes,
 33803:         uint64_t main_down_expert_bytes,
 33804:         uint64_t main_down_row_bytes,
 33805:         const void *q1_model_map,
 33806:         uint64_t q1_model_size,
 33807:         uint64_t q1_gate_offset,
 33808:         uint64_t q1_up_offset,
 33809:         uint64_t q1_down_offset,
 33810:         uint64_t q1_gate_expert_bytes,
 33811:         uint64_t q1_gate_row_bytes,
 33812:         uint64_t q1_down_expert_bytes,
 33813:         uint64_t q1_down_row_bytes,
 33814:         uint32_t expert_in_dim,
 33815:         uint32_t expert_mid_dim,
 33816:         uint32_t out_dim,
 33817:         const ds4_gpu_tensor *selected,
 33818:         const ds4_gpu_tensor *weights,
 33819:         uint32_t n_expert,
 33820:         float clamp,
 33821:         const ds4_gpu_tensor *x) {
 33822:     g_q1_0_mixed_calls++;
 33823: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 33824:     cuda_g130_attribution_state *attribution_state =
 33825:         cuda_g130_attribution_token_state();
 33826:     const int attribution = attribution_state != NULL;
 33827:     cuda_g130_attribution_span attribution_previous = CUDA_G130_ATTRIB_NONE;
 33828:     if (attribution) {
 33829:         attribution_previous = cuda_g130_attribution_switch(
 33830:             attribution_state,
 33831:             CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 33832:     }
 33833:     cuda_g130_attribution_restore_guard attribution_restore = {
 33834:         attribution_state, attribution_previous
 33835:     };
 33836: #endif
 33837:     const int mixed_profile = cuda_q1_0_profile_requested();
 33838:     cuda_q1_0_mixed_profile_sample mixed_profile_sample;
 33839:     const double mixed_profile_call_started =
 33840:         mixed_profile ? cuda_wall_sec() : 0.0;
 33841:     double mixed_profile_phase_started = mixed_profile_call_started;
 33842:     if (mixed_profile) {
 33843:         memset(&mixed_profile_sample, 0, sizeof(mixed_profile_sample));
 33844:         cuda_q1_0_mixed_profile_record_entered(n_expert);
 33845:     }
 33846:     if (!out || !gate || !up || !mid || !down || !main_model_map ||
 33847:         !q1_model_map || !selected || !weights || !x ||
 33848:         n_expert == 0u || n_expert > CUDA_MOE_ROUTE_COUNT ||
 33849:         main_gate_type != 16u || main_down_type != 10u ||
 33850:         main_model_map != g_model_host_base ||
 33851:         main_model_size != g_model_registered_size ||
 33852:         q1_model_map != g_q1_0_sidecar_host_base ||
 33853:         q1_model_size != g_q1_0_sidecar_size ||
 33854:         !cuda_q1_0_route_arena(layer_index) ||
 33855:         selected->bytes < (uint64_t)n_expert * sizeof(int32_t) ||
 33856:         weights->bytes < (uint64_t)n_expert * sizeof(float) ||
 33857:         out->bytes < (uint64_t)out_dim * sizeof(float)) {
 33858:         fprintf(stderr,
 33859:                 "ds4: [q1-0-mixed] result=failed reason=entry-contract "
 33860:                 "layer=%u routes=%u main_type=%u/%u main_map=%u "
 33861:                 "main_size=%llu registered_size=%llu q1_map=%u "
 33862:                 "q1_size=%llu sidecar_size=%llu q1_arena=%u "
 33863:                 "tier_entries=%llu selected_bytes=%llu weights_bytes=%llu "
 33864:                 "out_bytes=%llu required_out=%llu\n",
 33865:                 layer_index, n_expert, main_gate_type, main_down_type,
 33866:                 main_model_map == g_model_host_base,
 33867:                 (unsigned long long)main_model_size,
 33868:                 (unsigned long long)g_model_registered_size,
 33869:                 q1_model_map == g_q1_0_sidecar_host_base,
 33870:                 (unsigned long long)q1_model_size,
 33871:                 (unsigned long long)g_q1_0_sidecar_size,
 33872:                 cuda_q1_0_route_arena(layer_index) != NULL,
 33873:                 (unsigned long long)g_moe_tiering.entries.size(),
 33874:                 (unsigned long long)(selected ? selected->bytes : 0),
 33875:                 (unsigned long long)(weights ? weights->bytes : 0),
 33876:                 (unsigned long long)(out ? out->bytes : 0),
 33877:                 (unsigned long long)out_dim * sizeof(float));
 33878:         g_q1_0_mixed_failures++;
 33879:         return 0;
 33880:     }
 33881:     cuda_dynamic_arena *q1_arena = cuda_q1_0_route_arena(layer_index);
 33882:     if (!q1_arena) {
 33883:         fprintf(stderr,
 33884:                 "ds4: [q1-0-mixed] result=failed reason=q1-arena-null "
 33885:                 "layer=%u\n", layer_index);
 33886:         g_q1_0_mixed_failures++;
 33887:         return 0;
 33888:     }
 33889:     const ds4_gpu_dynamic_arena_layer &q1_geometry =
 33890:         q1_arena->layers[layer_index];
 33891:     if (q1_geometry.gate_offset != q1_gate_offset ||
 33892:         q1_geometry.up_offset != q1_up_offset ||
 33893:         q1_geometry.down_offset != q1_down_offset ||
 33894:         q1_geometry.gate_expert_bytes != q1_gate_expert_bytes ||
 33895:         q1_geometry.up_expert_bytes != q1_gate_expert_bytes ||
 33896:         q1_geometry.down_expert_bytes != q1_down_expert_bytes) {
 33897:         fprintf(stderr,
 33898:                 "ds4: [q1-0-mixed] result=failed reason=q1-geometry "
 33899:                 "layer=%u gate_offset=%llu/%llu up_offset=%llu/%llu "
 33900:                 "down_offset=%llu/%llu gate_bytes=%llu/%llu "
 33901:                 "up_bytes=%llu/%llu down_bytes=%llu/%llu\n",
 33902:                 layer_index,
 33903:                 (unsigned long long)q1_geometry.gate_offset,
 33904:                 (unsigned long long)q1_gate_offset,
 33905:                 (unsigned long long)q1_geometry.up_offset,
 33906:                 (unsigned long long)q1_up_offset,
 33907:                 (unsigned long long)q1_geometry.down_offset,
 33908:                 (unsigned long long)q1_down_offset,
 33909:                 (unsigned long long)q1_geometry.gate_expert_bytes,
 33910:                 (unsigned long long)q1_gate_expert_bytes,
 33911:                 (unsigned long long)q1_geometry.up_expert_bytes,
 33912:                 (unsigned long long)q1_gate_expert_bytes,
 33913:                 (unsigned long long)q1_geometry.down_expert_bytes,
 33914:                 (unsigned long long)q1_down_expert_bytes);
 33915:         g_q1_0_mixed_failures++;
 33916:         return 0;
 33917:     }
 33918:     if (!cuda_q1_0_ssd_wrap_poll()) {
 33919:         fprintf(stderr,
 33920:                 "ds4: [q1-0-mixed] result=failed "
 33921:                 "reason=ssd-wrap-terminal layer=%u\n", layer_index);
 33922:         g_q1_0_mixed_failures++;
 33923:         return 0;
 33924:     }
 33925: 
 33926:     /* The Q1 fallback path still needs the exact-IQ2 resolver: it decides
 33927:      * whether each full/open route is already resident as IQ2 or must use Q1
 33928:      * for this token. Promotion only controls later RAM-probation mutation. */
 33929:     const int q1_0_dynamic_promotion =
 33930:         cuda_q1_0_dynamic_promotion_requested();
 33931:     const int q1_0_mixed_exact_iq2_resolver =
 33932:         cuda_q1_0_mixed_exact_iq2_resolver_requested();
 33933:     if (q1_0_mixed_exact_iq2_resolver &&
 33934:         cuda_moe_gpu_routes_requested()) {
 33935:         cuda_moe_expert_cache *cache = cuda_moe_expert_cache_prepare(
 33936:             main_gate_expert_bytes, main_down_expert_bytes);
 33937:         if (!cache || !cache->route_thread_started ||
 33938:             !cache->route_worker_ready || cache->route_worker_failed ||
 33939:             cache->route_stop) {
 33940:             fprintf(stderr,
 33941:                     "ds4: [q1-0-mixed] result=failed "
 33942:                     "reason=iq2-vram-cache-not-ready layer=%u\n",
 33943:                     layer_index);
 33944:             g_q1_0_mixed_failures++;
 33945:             return 0;
 33946:         }
 33947:     }
 33948:     if (q1_0_mixed_exact_iq2_resolver &&
 33949:         !cuda_q1_0_snapshot_backing_requested() &&
 33950:         !cuda_q1_0_dual_sparse_companion_requested() &&
 33951:         g_moe_tiering.entries.size() !=
 33952:             (size_t)CUDA_MOE_LAYER_COUNT * 256u) {
 33953:         fprintf(stderr,
 33954:                 "ds4: [q1-0-mixed] result=failed reason=entry-contract "
 33955:                 "layer=%u routes=%u main_type=%u/%u main_map=%u "
 33956:                 "main_size=%llu registered_size=%llu q1_map=%u "
 33957:                 "q1_size=%llu sidecar_size=%llu q1_arena=%u "
 33958:                 "tier_entries=%llu selected_bytes=%llu weights_bytes=%llu "
 33959:                 "out_bytes=%llu required_out=%llu\n",
 33960:                 layer_index, n_expert, main_gate_type, main_down_type,
 33961:                 main_model_map == g_model_host_base,
 33962:                 (unsigned long long)main_model_size,
 33963:                 (unsigned long long)g_model_registered_size,
 33964:                 q1_model_map == g_q1_0_sidecar_host_base,
 33965:                 (unsigned long long)q1_model_size,
 33966:                 (unsigned long long)g_q1_0_sidecar_size,
 33967:                 cuda_q1_0_route_arena(layer_index) != NULL,
 33968:                 (unsigned long long)g_moe_tiering.entries.size(),
 33969:                 (unsigned long long)(selected ? selected->bytes : 0),
 33970:                 (unsigned long long)(weights ? weights->bytes : 0),
 33971:                 (unsigned long long)(out ? out->bytes : 0),
 33972:                 (unsigned long long)out_dim * sizeof(float));
 33973:         g_q1_0_mixed_failures++;
 33974:         return 0;
 33975:     }
 33976:     if (q1_0_dynamic_promotion &&
 33977:         (cuda_q1_0_snapshot_backing_requested() ||
 33978:          !cuda_q1_0_dual_arena_layer_active(layer_index) ||
 33979:          g_moe_tiering.mode != CUDA_MOE_TIER_ENFORCE ||
 33980:          !g_moe_tiering.compose_prefill_mass_tiering ||
 33981:          !g_moe_tiering.compose_router_open ||
 33982:          g_iq1_promotion.requested_slots == 0u)) {
 33983:         fprintf(stderr,
 33984:                 "ds4: [q1-0-mixed] result=failed "
 33985:                 "reason=dynamic-promotion-contract layer=%u dual=%u "
 33986:                 "snapshot=%u tier_mode=%u compose=%u open=%u slots=%u\n",
 33987:                 layer_index,
 33988:                 cuda_q1_0_dual_arena_layer_active(layer_index) ? 1u : 0u,
 33989:                 cuda_q1_0_snapshot_backing_requested() ? 1u : 0u,
 33990:                 (unsigned)g_moe_tiering.mode,
 33991:                 g_moe_tiering.compose_prefill_mass_tiering ? 1u : 0u,
 33992:                 g_moe_tiering.compose_router_open ? 1u : 0u,
 33993:                 g_iq1_promotion.requested_slots);
 33994:         g_q1_0_mixed_failures++;
 33995:         return 0;
 33996:     }
 33997: 
 33998:     const int recovery_trace_layer =
 33999:         g_expert_recovery_trace.enabled &&
 34000:         layer_index == g_expert_recovery_trace.target_layer;
 34001:     const uint64_t recovery_trace_token_index =
 34002:         cuda_expert_recovery_trace_layer_call(layer_index);
 34003:     if (recovery_trace_layer && recovery_trace_token_index == UINT64_MAX) {
 34004:         g_q1_0_mixed_failures++;
 34005:         return 0;
 34006:     }
 34007: 
 34008:     int32_t selected_host[CUDA_MOE_ROUTE_COUNT] = {0};
 34009:     float weights_host[CUDA_MOE_ROUTE_COUNT] = {0};
 34010:     if (mixed_profile) {
 34011:         const double d2h_started = cuda_wall_sec();
 34012:         mixed_profile_sample.entry_contract_calls++;
 34013:         mixed_profile_sample.entry_contract_seconds +=
 34014:             d2h_started - mixed_profile_phase_started;
 34015:         mixed_profile_phase_started = d2h_started;
 34016:     }
 34017: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34018:     if (attribution) {
 34019:         (void)cuda_g130_attribution_switch(
 34020:             attribution_state, CUDA_G130_ATTRIB_SELECTION_D2H);
 34021:     }
 34022: #endif
 34023:     const int mixed_selection_d2h_ok =
 34024:         cuda_ok(cudaMemcpy(selected_host, selected->ptr,
 34025:                            (size_t)n_expert * sizeof(int32_t),
 34026:                            cudaMemcpyDeviceToHost),
 34027:                 "Q1_0 mixed selected D2H") &&
 34028:         cuda_ok(cudaMemcpy(weights_host, weights->ptr,
 34029:                            (size_t)n_expert * sizeof(float),
 34030:                            cudaMemcpyDeviceToHost),
 34031:                 "Q1_0 mixed weights D2H");
 34032: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34033:     if (attribution) {
 34034:         (void)cuda_g130_attribution_switch(
 34035:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34036:     }
 34037: #endif
 34038:     if (mixed_profile) {
 34039:         const double d2h_finished = cuda_wall_sec();
 34040:         mixed_profile_sample.selection_d2h_calls++;
 34041:         mixed_profile_sample.selection_d2h_seconds +=
 34042:             d2h_finished - mixed_profile_phase_started;
 34043:         mixed_profile_phase_started = d2h_finished;
 34044:     }
 34045:     if (!mixed_selection_d2h_ok) {
 34046:         g_q1_0_mixed_failures++;
 34047:         return 0;
 34048:     }
 34049: 
 34050: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34051:     if (attribution) {
 34052:         (void)cuda_g130_attribution_switch(
 34053:             attribution_state, CUDA_G130_ATTRIB_ROUTE_CLASSIFY);
 34054:     }
 34055: #endif
 34056:     int32_t hot_selected[CUDA_MOE_ROUTE_COUNT] = {0};
 34057:     float hot_weights[CUDA_MOE_ROUTE_COUNT] = {0};
 34058:     int32_t cold_selected[CUDA_MOE_ROUTE_COUNT] = {0};
 34059:     float cold_weights[CUDA_MOE_ROUTE_COUNT] = {0};
 34060:     uint32_t hot_count = 0;
 34061:     uint32_t cold_count = 0;
 34062:     uint32_t iq2_vram = 0;
 34063:     uint32_t iq2_snapshot_ram = 0;
 34064:     uint32_t iq2_tier_ram = 0;
 34065:     uint32_t tier_route_entries = 0;
 34066:     uint32_t recovery_trace_rank = UINT32_MAX;
 34067:     float recovery_trace_weight = 0.0f;
 34068:     const char *recovery_trace_representation = NULL;
 34069:     const int cold_one_requested =
 34070:         cuda_q1_0_mixed_cold_one_requested();
 34071:     uint32_t cold_slot = UINT32_MAX;
 34072:     if (cold_one_requested) {
 34073:         if (!cuda_q1_0_dual_sparse_companion_requested() ||
 34074:             n_expert != CUDA_MOE_ROUTE_COUNT ||
 34075:             !g_q1_0_dual_sparse_snapshot.ready) {
 34076:             fprintf(stderr,
 34077:                     "ds4: [q1-0-mixed] result=failed "
 34078:                     "reason=cold-one-contract layer=%u routes=%u\n",
 34079:                     layer_index, n_expert);
 34080:             g_q1_0_mixed_cold_one_invariant_failures++;
 34081:             g_q1_0_mixed_failures++;
 34082:             return 0;
 34083:         }
 34084:         cold_slot = 0u;
 34085:         for (uint32_t route = 1; route < n_expert; route++) {
 34086:             if (weights_host[route] < weights_host[cold_slot]) {
 34087:                 cold_slot = route;
 34088:             }
 34089:         }
 34090:     }
 34091:     for (uint32_t route = 0; route < n_expert; route++) {
 34092:         const int32_t expert_i = selected_host[route];
 34093:         if (expert_i < 0 || expert_i >= 256) {
 34094:             fprintf(stderr,
 34095:                     "ds4: [q1-0-mixed] result=failed reason=expert-range "
 34096:                     "layer=%u route=%u expert=%d\n",
 34097:                     layer_index, route, (int)expert_i);
 34098:             g_q1_0_mixed_failures++;
 34099:             return 0;
 34100:         }
 34101:         cuda_q1_0_mixed_representation representation =
 34102:             CUDA_Q1_0_MIXED_UNKNOWN;
 34103:         if (cold_one_requested) {
 34104:             if (!cuda_q1_0_dual_sparse_pair_contains(
 34105:                     layer_index, (uint32_t)expert_i)) {
 34106:                 fprintf(stderr,
 34107:                         "ds4: [q1-0-mixed] result=failed "
 34108:                         "reason=dual-sparse-parity layer=%u route=%u expert=%d\n",
 34109:                         layer_index, route, (int)expert_i);
 34110:                 g_q1_0_mixed_cold_one_invariant_failures++;
 34111:                 g_q1_0_mixed_failures++;
 34112:                 return 0;
 34113:             }
 34114:             representation = route == cold_slot
 34115:                 ? CUDA_Q1_0_MIXED_Q1_RESIDENT
 34116:                 : (cuda_moe_tiering_has_exact_vram(
 34117:                         layer_index, (uint32_t)expert_i)
 34118:                     ? CUDA_Q1_0_MIXED_IQ2_VRAM
 34119:                     : CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM);
 34120:         } else {
 34121:             representation = cuda_q1_0_mixed_resolve(
 34122:                 layer_index, (uint32_t)expert_i);
 34123:         }
 34124:         const cuda_moe_tier_entry *tier =
 34125:             g_moe_tiering.entries.size() ==
 34126:                     (size_t)CUDA_MOE_LAYER_COUNT * 256u
 34127:                 ? &g_moe_tiering.entries[cuda_moe_tiering_entry_index(
 34128:                     layer_index, (uint32_t)expert_i)]
 34129:                 : NULL;
 34130:         if (tier) tier_route_entries++;
 34131:         if (cuda_expert_recovery_trace_target(
 34132:                 layer_index, (uint32_t)expert_i)) {
 34133:             if (recovery_trace_rank != UINT32_MAX) {
 34134:                 cuda_expert_recovery_trace_fail("duplicate_target_route");
 34135:                 g_q1_0_mixed_failures++;
 34136:                 return 0;
 34137:             }
 34138:             recovery_trace_rank = route;
 34139:             recovery_trace_weight = weights_host[route];
 34140:             recovery_trace_representation =
 34141:                 cuda_q1_0_mixed_representation_name(representation);
 34142:         }
 34143:         if (cuda_q1_0_mixed_trace_requested()) {
 34144:             fprintf(stderr,
 34145:                     "ds4: [q1-0-mixed-route] layer=%u route=%u expert=%d "
 34146:                     "weight=%.9g representation=%s tier=%u has_2bit_ram=%u "
 34147:                     "primary_snapshot=%llu q1_snapshot=%llu\n",
 34148:                     layer_index, route, (int)expert_i, weights_host[route],
 34149:                     cuda_q1_0_mixed_representation_name(representation),
 34150:                     tier ? (unsigned)tier->state : 0u,
 34151:                     tier ? (unsigned)tier->has_2bit_ram : 0u,
 34152:                     (unsigned long long)g_moe_tiering.snapshot_generation,
 34153:                     (unsigned long long)
 34154:                         q1_arena->snapshot_generation);
 34155:             g_q1_0_mixed_trace_rows++;
 34156:         }
 34157:         if (representation == CUDA_Q1_0_MIXED_IQ2_VRAM ||
 34158:             representation == CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM ||
 34159:             representation == CUDA_Q1_0_MIXED_IQ2_TIER_RAM) {
 34160:             hot_selected[hot_count] = expert_i;
 34161:             hot_weights[hot_count] = weights_host[route];
 34162:             hot_count++;
 34163:             iq2_vram += representation == CUDA_Q1_0_MIXED_IQ2_VRAM;
 34164:             iq2_snapshot_ram +=
 34165:                 representation == CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM;
 34166:             iq2_tier_ram +=
 34167:                 representation == CUDA_Q1_0_MIXED_IQ2_TIER_RAM;
 34168:         } else if (representation == CUDA_Q1_0_MIXED_Q1_RESIDENT) {
 34169:             cold_selected[cold_count] = expert_i;
 34170:             cold_weights[cold_count] = weights_host[route];
 34171:             cold_count++;
 34172:         } else {
 34173:             fprintf(stderr,
 34174:                     "ds4: [q1-0-mixed] result=failed reason=unresolved "
 34175:                     "layer=%u route=%u expert=%d router=unchanged\n",
 34176:                     layer_index, route, (int)expert_i);
 34177:             g_q1_0_mixed_failures++;
 34178:             return 0;
 34179:         }
 34180:     }
 34181:     if (q1_0_mixed_exact_iq2_resolver &&
 34182:         !cuda_q1_0_snapshot_backing_requested() &&
 34183:         !cuda_q1_0_dual_sparse_companion_requested() &&
 34184:         tier_route_entries != n_expert) {
 34185:         fprintf(stderr,
 34186:                 "ds4: [q1-0-mixed] result=failed "
 34187:                 "reason=route-entry-contract layer=%u routes=%u "
 34188:                 "tier_route_entries=%u tier_entries=%llu\n",
 34189:                 layer_index, n_expert, tier_route_entries,
 34190:                 (unsigned long long)g_moe_tiering.entries.size());
 34191:         g_q1_0_mixed_failures++;
 34192:         return 0;
 34193:     }
 34194:     g_q1_0_mixed_tier_route_entries += tier_route_entries;
 34195: 
 34196:     if (cold_one_requested) {
 34197:         if (hot_count != CUDA_MOE_ROUTE_COUNT - 1u || cold_count != 1u) {
 34198:             fprintf(stderr,
 34199:                     "ds4: [q1-0-mixed] result=failed "
 34200:                     "reason=cold-one-count layer=%u hot=%u q1=%u\n",
 34201:                     layer_index, hot_count, cold_count);
 34202:             g_q1_0_mixed_cold_one_invariant_failures++;
 34203:             g_q1_0_mixed_failures++;
 34204:             return 0;
 34205:         }
 34206:         g_q1_0_mixed_cold_one_calls++;
 34207:         g_q1_0_mixed_cold_one_hot_routes += hot_count;
 34208:         g_q1_0_mixed_cold_one_q1_routes += cold_count;
 34209:     }
 34210: 
 34211: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34212:     if (attribution) {
 34213:         (void)cuda_g130_attribution_switch(
 34214:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34215:     }
 34216: #endif
 34217: 
 34218:     if (mixed_profile) {
 34219:         const double classify_finished = cuda_wall_sec();
 34220:         mixed_profile_sample.classify_map_calls++;
 34221:         mixed_profile_sample.classify_map_seconds +=
 34222:             classify_finished - mixed_profile_phase_started;
 34223:         mixed_profile_phase_started = classify_finished;
 34224:     }
 34225: 
 34226:     if (cold_count == 0u) {
 34227:         const double hot_branch_started = mixed_profile
 34228:             ? mixed_profile_phase_started : 0.0;
 34229: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34230:         if (attribution) {
 34231:             (void)cuda_g130_attribution_switch(
 34232:                 attribution_state, CUDA_G130_ATTRIB_HOT_ROUTE);
 34233:         }
 34234: #endif
 34235:         const int ok = routed_moe_launch(
 34236:             out, gate, up, mid, down,
 34237:             main_model_map, main_model_size, layer_index,
 34238:             main_gate_offset, main_up_offset, main_down_offset,
 34239:             main_gate_type, main_down_type,
 34240:             main_gate_expert_bytes, main_gate_row_bytes,
 34241:             main_down_expert_bytes, main_down_row_bytes,
 34242:             expert_in_dim, expert_mid_dim, out_dim,
 34243:             selected, weights, NULL, n_expert, clamp, x, 1u, NULL, NULL);
 34244: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34245:         if (attribution) {
 34246:             (void)cuda_g130_attribution_switch(
 34247:                 attribution_state,
 34248:                 CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34249:         }
 34250: #endif
 34251:         if (mixed_profile) {
 34252:             const double hot_branch_finished = cuda_wall_sec();
 34253:             mixed_profile_sample.hot_branch_calls++;
 34254:             mixed_profile_sample.hot_branch_seconds +=
 34255:                 hot_branch_finished - hot_branch_started;
 34256:             mixed_profile_phase_started = hot_branch_finished;
 34257:         }
 34258:         if (!ok) {
 34259:             g_q1_0_mixed_failures++;
 34260:             return 0;
 34261:         }
 34262:         g_q1_0_mixed_all_iq2++;
 34263:         g_q1_0_mixed_iq2_vram += iq2_vram;
 34264:         g_q1_0_mixed_iq2_snapshot_ram += iq2_snapshot_ram;
 34265:         g_q1_0_mixed_iq2_tier_ram += iq2_tier_ram;
 34266:         if (recovery_trace_rank != UINT32_MAX &&
 34267:             !cuda_expert_recovery_trace_capture(
 34268:                 layer_index, g_expert_recovery_trace.target_expert,
 34269:                 recovery_trace_rank, recovery_trace_weight,
 34270:                 recovery_trace_representation, recovery_trace_token_index,
 34350:         !cuda_ok(cudaMemcpy(device_cold_selected, cold_selected,
 34351:                             (size_t)cold_count * sizeof(int32_t),
 34352:                             cudaMemcpyHostToDevice),
 34353:                  "Q1_0 mixed cold selected H2D") ||
 34354:         !cuda_ok(cudaMemcpy(device_cold_weights, cold_weights,
 34355:                             (size_t)cold_count * sizeof(float),
 34356:                             cudaMemcpyHostToDevice),
 34357:                  "Q1_0 mixed cold weights H2D"));
 34358: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34359:     if (attribution) {
 34360:         (void)cuda_g130_attribution_switch(
 34361:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34362:     }
 34363: #endif
 34364:     if (!metadata_h2d_ok) {
 34365:         g_q1_0_mixed_failures++;
 34366:         return 0;
 34367:     }
 34368:     if (mixed_profile) {
 34369:         const double metadata_finished = cuda_wall_sec();
 34370:         mixed_profile_sample.metadata_h2d_calls++;
 34371:         mixed_profile_sample.metadata_h2d_seconds +=
 34372:             metadata_finished - mixed_profile_phase_started;
 34373:         mixed_profile_phase_started = metadata_finished;
 34374:     }
 34375: 
 34376:     int ok = 1;
 34377:     if (hot_count != 0u) {
 34378:         const ds4_gpu_tensor hot_selected_tensor = {
 34379:             device_hot_selected,
 34380:             (uint64_t)hot_count * sizeof(int32_t), 0
 34381:         };
 34382:         const ds4_gpu_tensor hot_weights_tensor = {
 34383:             device_hot_weights,
 34384:             (uint64_t)hot_count * sizeof(float), 0
 34385:         };
 34386: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34387:         if (attribution) {
 34388:             (void)cuda_g130_attribution_switch(
 34389:                 attribution_state, CUDA_G130_ATTRIB_HOT_ROUTE);
 34390:         }
 34391: #endif
 34392:         ok = routed_moe_launch(
 34393:             out, gate, up, mid, down,
 34394:             main_model_map, main_model_size, layer_index,
 34395:             main_gate_offset, main_up_offset, main_down_offset,
 34396:             main_gate_type, main_down_type,
 34397:             main_gate_expert_bytes, main_gate_row_bytes,
 34398:             main_down_expert_bytes, main_down_row_bytes,
 34399:             expert_in_dim, expert_mid_dim, out_dim,
 34400:             &hot_selected_tensor, &hot_weights_tensor, NULL,
 34401:             hot_count, clamp, x, 1u, NULL, NULL);
 34402: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34403:         if (attribution) {
 34404:             (void)cuda_g130_attribution_switch(
 34405:                 attribution_state,
 34406:                 CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34407:         }
 34408: #endif
 34409:     } else {
 34410:         ok = cuda_ok(cudaMemsetAsync(out->ptr, 0, (size_t)out_bytes, 0),
 34411:                      "Q1_0 mixed empty IQ2 output");
 34412:     }
 34413:     if (!ok) {
 34414:         fprintf(stderr,
 34415:                 "ds4: [q1-0-mixed] result=failed reason=iq2-hot-launch "
 34416:                 "layer=%u hot=%u\n", layer_index, hot_count);
 34417:         g_q1_0_mixed_failures++;
 34418:         return 0;
 34419:     }
 34420:     const uint64_t iq2_ssd_bytes_after_hot = g_moe_tiering.ssd_bytes;
 34421:     const uint64_t iq2_cold_to_ram_after_hot = g_moe_tiering.cold_to_ram;
 34422:     if (iq2_ssd_bytes_after_hot != iq2_ssd_bytes_before ||
 34423:         iq2_cold_to_ram_after_hot != iq2_cold_to_ram_before) {
 34424:         const uint64_t delta = iq2_ssd_bytes_after_hot >= iq2_ssd_bytes_before
 34425:             ? iq2_ssd_bytes_after_hot - iq2_ssd_bytes_before : 0;
 34426:         g_q1_0_mixed_iq2_ssd_bytes += delta;
 34427:         g_q1_0_mixed_iq2_ssd_violations++;
 34428:         g_q1_0_mixed_failures++;
 34429:         fprintf(stderr,
 34430:                 "ds4: [q1-0-mixed] result=failed "
 34431:                 "reason=current-token-iq2-ssd layer=%u bytes=%llu "
 34432:                 "cold_to_ram_delta=%llu\n",
 34433:                 layer_index, (unsigned long long)delta,
 34434:                 (unsigned long long)(
 34435:                     iq2_cold_to_ram_after_hot - iq2_cold_to_ram_before));
 34436:         return 0;
 34437:     }
 34438:     if (mixed_profile) {
 34439:         const double hot_branch_finished = cuda_wall_sec();
 34440:         mixed_profile_sample.hot_branch_calls++;
 34441:         mixed_profile_sample.hot_branch_seconds +=
 34442:             hot_branch_finished - mixed_profile_phase_started;
 34443:         mixed_profile_phase_started = hot_branch_finished;
 34444:     }
 34445: 
 34446:     /* The IQ2 route worker already observed the hot subset. Q1 routes bypass
 34447:      * that worker, so account for their router demand on the same logical
 34448:      * clock tick. This changes residency policy only for future tokens; the
 34449:      * current token remains Q1 and performs no IQ2 SSD read. */
 34450:     if (g_moe_tiering.mode != CUDA_MOE_TIER_OFF) {
 34451:         if (hot_count == 0u) {
 34452:             if (cuda_moe_tiering_advance_call_tick(
 34453:                     "q1-0-mixed-cold-only")) {
 34454:                 g_moe_tiering.calls++;
 34455:             }
 34456:         }
 34457:         g_moe_tiering.selected += cold_count;
 34458:         for (uint32_t route = 0; route < cold_count; route++) {
 34459:             (void)cuda_moe_tiering_observe_route(
 34460:                 layer_index, (uint32_t)cold_selected[route],
 34461:                 cold_weights[route]);
 34462:         }
 34463:     }
 34464:     if (mixed_profile) {
 34465:         const double tier_observe_finished = cuda_wall_sec();
 34466:         mixed_profile_sample.tier_observe_calls++;
 34467:         mixed_profile_sample.tier_observe_seconds +=
 34468:             tier_observe_finished - mixed_profile_phase_started;
 34469:         mixed_profile_phase_started = tier_observe_finished;
 34470:     }
 34471: 
 34472:     const ds4_gpu_tensor cold_selected_tensor = {
 34473:         device_cold_selected,
 34474:         (uint64_t)cold_count * sizeof(int32_t), 0
 34475:     };
 34476:     const ds4_gpu_tensor cold_weights_tensor = {
 34477:         device_cold_weights,
 34478:         (uint64_t)cold_count * sizeof(float), 0
 34479:     };
 34480:     ds4_gpu_tensor cold_out_tensor = {
 34481:         device_cold_out, out_bytes, 0
 34482:     };
 34483:     ds4_gpu_tensor cold_down_tensor = {
 34484:         device_cold_down, cold_down_bytes, 0
 34485:     };
 34486:     cuda_q1_0_mixed_profile_sample *previous_profile_sample = NULL;
 34487:     if (mixed_profile) {
 34488:         const double q1_call_started = cuda_wall_sec();
 34489:         mixed_profile_sample.q1_dispatch_prepare_calls++;
 34490:         mixed_profile_sample.q1_dispatch_prepare_seconds +=
 34491:             q1_call_started - mixed_profile_phase_started;
 34492:         mixed_profile_sample.q1_call_started = q1_call_started;
 34493:         previous_profile_sample = g_q1_0_mixed_profile_current_sample;
 34494:         g_q1_0_mixed_profile_current_sample = &mixed_profile_sample;
 34495:     }
 34496:     ok = routed_moe_launch(
 34497:         &cold_out_tensor, gate, up, mid, &cold_down_tensor,
 34498:         q1_model_map, q1_model_size, layer_index,
 34499:         q1_gate_offset, q1_up_offset, q1_down_offset,
 34500:         41u, 41u,
 34501:         q1_gate_expert_bytes, q1_gate_row_bytes,
 34502:         q1_down_expert_bytes, q1_down_row_bytes,
 34503:         expert_in_dim, expert_mid_dim, out_dim,
 34504:         &cold_selected_tensor, &cold_weights_tensor, NULL,
 34505:         cold_count, clamp, x, 1u, NULL, NULL);
 34506:     if (mixed_profile) {
 34507:         g_q1_0_mixed_profile_current_sample = previous_profile_sample;
 34508:         mixed_profile_phase_started = cuda_wall_sec();
 34509:     }
 34510:     if (!ok) {
 34511:         fprintf(stderr,
 34512:                 "ds4: [q1-0-mixed] result=failed reason=q1-cold-launch "
 34513:                 "layer=%u q1=%u\n", layer_index, cold_count);
 34514:         g_q1_0_mixed_failures++;
 34515:         return 0;
 34516:     }
 34517:     cudaEvent_t join_begin = NULL;
 34518:     cudaEvent_t join_end = NULL;
 34519:     if (cuda_q1_0_profile_requested()) {
 34520:         const cudaError_t begin_create = cudaEventCreate(&join_begin);
 34521:         const cudaError_t end_create = begin_create == cudaSuccess ?
 34522:             cudaEventCreate(&join_end) : begin_create;
 34523:         const cudaError_t begin_record = end_create == cudaSuccess ?
 34524:             cudaEventRecord(join_begin, 0) : end_create;
 34525:         if (begin_record != cudaSuccess) {
 34526:             if (join_begin) (void)cudaEventDestroy(join_begin);
 34527:             if (join_end) (void)cudaEventDestroy(join_end);
 34528:             join_begin = NULL;
 34529:             join_end = NULL;
 34530:             g_q1_0_profile.timer_failures++;
 34531:             cuda_q1_0_mixed_profile_record_timer_failure();
 34532:             (void)cudaGetLastError();
 34533:         }
 34534:     }
 34535: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34536:     if (attribution) {
 34537:         (void)cuda_g130_attribution_switch(
 34538:             attribution_state, CUDA_G130_ATTRIB_MIXED_JOIN);
 34539:     }
 34540: #endif
 34541:     add_f32_u64_kernel<<<(out_dim + 255u) / 256u, 256>>>(
 34542:         (float *)out->ptr, (const float *)out->ptr,
 34543:         device_cold_out, out_dim);
 34544:     const int mixed_join_ok =
 34545:         cuda_ok(cudaGetLastError(), "Q1_0 mixed output join");
 34546: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34547:     if (attribution) {
 34548:         (void)cuda_g130_attribution_switch(
 34549:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34550:     }
 34551: #endif
 34552:     if (!mixed_join_ok) {
 34553:         if (join_begin) (void)cudaEventDestroy(join_begin);
 34554:         if (join_end) (void)cudaEventDestroy(join_end);
 34555:         g_q1_0_mixed_failures++;
 34556:         return 0;
 34557:     }
 34558:     if (join_end) {
 34559:         float join_ms = 0.0f;
 34560:         const cudaError_t end_record = cudaEventRecord(join_end, 0);
 34561:         const cudaError_t end_sync = end_record == cudaSuccess ?
 34562:             cudaEventSynchronize(join_end) : end_record;
 34563:         const cudaError_t elapsed = end_sync == cudaSuccess ?
 34564:             cudaEventElapsedTime(&join_ms, join_begin, join_end) : end_sync;
 34565:         if (elapsed == cudaSuccess) {
 34566:             g_q1_0_profile.mixed_join_calls++;
 34567:             g_q1_0_profile.mixed_join_seconds +=
 34568:                 (double)join_ms / 1000.0;
 34569:         } else {
 34570:             g_q1_0_profile.timer_failures++;
 34571:             cuda_q1_0_mixed_profile_record_timer_failure();
 34572:             (void)cudaGetLastError();
 34573:         }
 34574:         (void)cudaEventDestroy(join_begin);
 34575:         (void)cudaEventDestroy(join_end);
 34576:     }
 34577:     if (recovery_trace_rank != UINT32_MAX &&
 34578:         !cuda_expert_recovery_trace_capture(
 34579:             layer_index, g_expert_recovery_trace.target_expert,
 34580:             recovery_trace_rank, recovery_trace_weight,
 34581:             recovery_trace_representation, recovery_trace_token_index,
 34582:             g_moe_tiering.call_tick, x, expert_in_dim)) {
 34583:         g_q1_0_mixed_failures++;
 34584:         return 0;
 34585:     }
 34586:     if (q1_0_dynamic_promotion) {
 34587:         /* The current result is already queued from resident Q1_0. Admission
 34588:          * below is future-token work: exact IQ2 first enters the disjoint RAM
 34589:          * probation arena and cannot become VRAM-eligible until a later call. */
 34590: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34591:         if (attribution) {
 34592:             (void)cuda_g130_attribution_switch(
 34593:                 attribution_state,
 34594:                 CUDA_G130_ATTRIB_PROMOTION_STAGING);
 34595:         }
 34596: #endif
 34597:         for (uint32_t route = 0; route < cold_count; route++) {
 34598:             (void)cuda_moe_tiering_stage_observed_quant_cold_to_2bit_ram(
 34599:                 layer_index, (uint32_t)cold_selected[route],
 34600:                 cold_weights[route],
 34601:                 main_gate_offset, main_up_offset, main_down_offset,
 34602:                 main_gate_expert_bytes, main_down_expert_bytes, 1,
 34603:                 q1_gate_offset, q1_up_offset, q1_down_offset,
 34604:                 q1_gate_expert_bytes, q1_down_expert_bytes);
 34605:         }
 34606: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34607:         if (attribution) {
 34608:             (void)cuda_g130_attribution_switch(
 34609:                 attribution_state,
 34610:                 CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34611:         }
 34612: #endif
 34613:     }
 34614:     if (cuda_q1_0_snapshot_backing_requested()) {
 34615:         g_moe_tiering.snapshot_backing_hits += cold_count;
 34616:     }
 34617:     g_q1_0_mixed_iq2_vram += iq2_vram;
 34618:     g_q1_0_mixed_iq2_snapshot_ram += iq2_snapshot_ram;
 34619:     g_q1_0_mixed_iq2_tier_ram += iq2_tier_ram;
 34620:     g_q1_0_mixed_q1_resident += cold_count;
 34621:     g_q1_0_mixed_joins++;
 34622:     if (mixed_profile) {
 34623:         const double publish_finished = cuda_wall_sec();
 34624:         mixed_profile_sample.join_publish_calls++;
 34625:         mixed_profile_sample.join_publish_seconds +=
 34626:             publish_finished - mixed_profile_phase_started;
 34627:         cuda_q1_0_mixed_profile_record_completed(
 34628:             mixed_profile_sample,
 34629:             publish_finished - mixed_profile_call_started,
 34630:             hot_count, cold_count, 1);

2026-07-21T09:51:27.559933Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 34008 34195; Show-Lines $p 34270 34367' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 870ms:
 34008:     int32_t selected_host[CUDA_MOE_ROUTE_COUNT] = {0};
 34009:     float weights_host[CUDA_MOE_ROUTE_COUNT] = {0};
 34010:     if (mixed_profile) {
 34011:         const double d2h_started = cuda_wall_sec();
 34012:         mixed_profile_sample.entry_contract_calls++;
 34013:         mixed_profile_sample.entry_contract_seconds +=
 34014:             d2h_started - mixed_profile_phase_started;
 34015:         mixed_profile_phase_started = d2h_started;
 34016:     }
 34017: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34018:     if (attribution) {
 34019:         (void)cuda_g130_attribution_switch(
 34020:             attribution_state, CUDA_G130_ATTRIB_SELECTION_D2H);
 34021:     }
 34022: #endif
 34023:     const int mixed_selection_d2h_ok =
 34024:         cuda_ok(cudaMemcpy(selected_host, selected->ptr,
 34025:                            (size_t)n_expert * sizeof(int32_t),
 34026:                            cudaMemcpyDeviceToHost),
 34027:                 "Q1_0 mixed selected D2H") &&
 34028:         cuda_ok(cudaMemcpy(weights_host, weights->ptr,
 34029:                            (size_t)n_expert * sizeof(float),
 34030:                            cudaMemcpyDeviceToHost),
 34031:                 "Q1_0 mixed weights D2H");
 34032: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34033:     if (attribution) {
 34034:         (void)cuda_g130_attribution_switch(
 34035:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34036:     }
 34037: #endif
 34038:     if (mixed_profile) {
 34039:         const double d2h_finished = cuda_wall_sec();
 34040:         mixed_profile_sample.selection_d2h_calls++;
 34041:         mixed_profile_sample.selection_d2h_seconds +=
 34042:             d2h_finished - mixed_profile_phase_started;
 34043:         mixed_profile_phase_started = d2h_finished;
 34044:     }
 34045:     if (!mixed_selection_d2h_ok) {
 34046:         g_q1_0_mixed_failures++;
 34047:         return 0;
 34048:     }
 34049: 
 34050: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34051:     if (attribution) {
 34052:         (void)cuda_g130_attribution_switch(
 34053:             attribution_state, CUDA_G130_ATTRIB_ROUTE_CLASSIFY);
 34054:     }
 34055: #endif
 34056:     int32_t hot_selected[CUDA_MOE_ROUTE_COUNT] = {0};
 34057:     float hot_weights[CUDA_MOE_ROUTE_COUNT] = {0};
 34058:     int32_t cold_selected[CUDA_MOE_ROUTE_COUNT] = {0};
 34059:     float cold_weights[CUDA_MOE_ROUTE_COUNT] = {0};
 34060:     uint32_t hot_count = 0;
 34061:     uint32_t cold_count = 0;
 34062:     uint32_t iq2_vram = 0;
 34063:     uint32_t iq2_snapshot_ram = 0;
 34064:     uint32_t iq2_tier_ram = 0;
 34065:     uint32_t tier_route_entries = 0;
 34066:     uint32_t recovery_trace_rank = UINT32_MAX;
 34067:     float recovery_trace_weight = 0.0f;
 34068:     const char *recovery_trace_representation = NULL;
 34069:     const int cold_one_requested =
 34070:         cuda_q1_0_mixed_cold_one_requested();
 34071:     uint32_t cold_slot = UINT32_MAX;
 34072:     if (cold_one_requested) {
 34073:         if (!cuda_q1_0_dual_sparse_companion_requested() ||
 34074:             n_expert != CUDA_MOE_ROUTE_COUNT ||
 34075:             !g_q1_0_dual_sparse_snapshot.ready) {
 34076:             fprintf(stderr,
 34077:                     "ds4: [q1-0-mixed] result=failed "
 34078:                     "reason=cold-one-contract layer=%u routes=%u\n",
 34079:                     layer_index, n_expert);
 34080:             g_q1_0_mixed_cold_one_invariant_failures++;
 34081:             g_q1_0_mixed_failures++;
 34082:             return 0;
 34083:         }
 34084:         cold_slot = 0u;
 34085:         for (uint32_t route = 1; route < n_expert; route++) {
 34086:             if (weights_host[route] < weights_host[cold_slot]) {
 34087:                 cold_slot = route;
 34088:             }
 34089:         }
 34090:     }
 34091:     for (uint32_t route = 0; route < n_expert; route++) {
 34092:         const int32_t expert_i = selected_host[route];
 34093:         if (expert_i < 0 || expert_i >= 256) {
 34094:             fprintf(stderr,
 34095:                     "ds4: [q1-0-mixed] result=failed reason=expert-range "
 34096:                     "layer=%u route=%u expert=%d\n",
 34097:                     layer_index, route, (int)expert_i);
 34098:             g_q1_0_mixed_failures++;
 34099:             return 0;
 34100:         }
 34101:         cuda_q1_0_mixed_representation representation =
 34102:             CUDA_Q1_0_MIXED_UNKNOWN;
 34103:         if (cold_one_requested) {
 34104:             if (!cuda_q1_0_dual_sparse_pair_contains(
 34105:                     layer_index, (uint32_t)expert_i)) {
 34106:                 fprintf(stderr,
 34107:                         "ds4: [q1-0-mixed] result=failed "
 34108:                         "reason=dual-sparse-parity layer=%u route=%u expert=%d\n",
 34109:                         layer_index, route, (int)expert_i);
 34110:                 g_q1_0_mixed_cold_one_invariant_failures++;
 34111:                 g_q1_0_mixed_failures++;
 34112:                 return 0;
 34113:             }
 34114:             representation = route == cold_slot
 34115:                 ? CUDA_Q1_0_MIXED_Q1_RESIDENT
 34116:                 : (cuda_moe_tiering_has_exact_vram(
 34117:                         layer_index, (uint32_t)expert_i)
 34118:                     ? CUDA_Q1_0_MIXED_IQ2_VRAM
 34119:                     : CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM);
 34120:         } else {
 34121:             representation = cuda_q1_0_mixed_resolve(
 34122:                 layer_index, (uint32_t)expert_i);
 34123:         }
 34124:         const cuda_moe_tier_entry *tier =
 34125:             g_moe_tiering.entries.size() ==
 34126:                     (size_t)CUDA_MOE_LAYER_COUNT * 256u
 34127:                 ? &g_moe_tiering.entries[cuda_moe_tiering_entry_index(
 34128:                     layer_index, (uint32_t)expert_i)]
 34129:                 : NULL;
 34130:         if (tier) tier_route_entries++;
 34131:         if (cuda_expert_recovery_trace_target(
 34132:                 layer_index, (uint32_t)expert_i)) {
 34133:             if (recovery_trace_rank != UINT32_MAX) {
 34134:                 cuda_expert_recovery_trace_fail("duplicate_target_route");
 34135:                 g_q1_0_mixed_failures++;
 34136:                 return 0;
 34137:             }
 34138:             recovery_trace_rank = route;
 34139:             recovery_trace_weight = weights_host[route];
 34140:             recovery_trace_representation =
 34141:                 cuda_q1_0_mixed_representation_name(representation);
 34142:         }
 34143:         if (cuda_q1_0_mixed_trace_requested()) {
 34144:             fprintf(stderr,
 34145:                     "ds4: [q1-0-mixed-route] layer=%u route=%u expert=%d "
 34146:                     "weight=%.9g representation=%s tier=%u has_2bit_ram=%u "
 34147:                     "primary_snapshot=%llu q1_snapshot=%llu\n",
 34148:                     layer_index, route, (int)expert_i, weights_host[route],
 34149:                     cuda_q1_0_mixed_representation_name(representation),
 34150:                     tier ? (unsigned)tier->state : 0u,
 34151:                     tier ? (unsigned)tier->has_2bit_ram : 0u,
 34152:                     (unsigned long long)g_moe_tiering.snapshot_generation,
 34153:                     (unsigned long long)
 34154:                         q1_arena->snapshot_generation);
 34155:             g_q1_0_mixed_trace_rows++;
 34156:         }
 34157:         if (representation == CUDA_Q1_0_MIXED_IQ2_VRAM ||
 34158:             representation == CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM ||
 34159:             representation == CUDA_Q1_0_MIXED_IQ2_TIER_RAM) {
 34160:             hot_selected[hot_count] = expert_i;
 34161:             hot_weights[hot_count] = weights_host[route];
 34162:             hot_count++;
 34163:             iq2_vram += representation == CUDA_Q1_0_MIXED_IQ2_VRAM;
 34164:             iq2_snapshot_ram +=
 34165:                 representation == CUDA_Q1_0_MIXED_IQ2_SNAPSHOT_RAM;
 34166:             iq2_tier_ram +=
 34167:                 representation == CUDA_Q1_0_MIXED_IQ2_TIER_RAM;
 34168:         } else if (representation == CUDA_Q1_0_MIXED_Q1_RESIDENT) {
 34169:             cold_selected[cold_count] = expert_i;
 34170:             cold_weights[cold_count] = weights_host[route];
 34171:             cold_count++;
 34172:         } else {
 34173:             fprintf(stderr,
 34174:                     "ds4: [q1-0-mixed] result=failed reason=unresolved "
 34175:                     "layer=%u route=%u expert=%d router=unchanged\n",
 34176:                     layer_index, route, (int)expert_i);
 34177:             g_q1_0_mixed_failures++;
 34178:             return 0;
 34179:         }
 34180:     }
 34181:     if (q1_0_mixed_exact_iq2_resolver &&
 34182:         !cuda_q1_0_snapshot_backing_requested() &&
 34183:         !cuda_q1_0_dual_sparse_companion_requested() &&
 34184:         tier_route_entries != n_expert) {
 34185:         fprintf(stderr,
 34186:                 "ds4: [q1-0-mixed] result=failed "
 34187:                 "reason=route-entry-contract layer=%u routes=%u "
 34188:                 "tier_route_entries=%u tier_entries=%llu\n",
 34189:                 layer_index, n_expert, tier_route_entries,
 34190:                 (unsigned long long)g_moe_tiering.entries.size());
 34191:         g_q1_0_mixed_failures++;
 34192:         return 0;
 34193:     }
 34194:     g_q1_0_mixed_tier_route_entries += tier_route_entries;
 34195: 
 34270:                 recovery_trace_representation, recovery_trace_token_index,
 34271:                 g_moe_tiering.call_tick, x, expert_in_dim)) {
 34272:             g_q1_0_mixed_failures++;
 34273:             return 0;
 34274:         }
 34275:         if (mixed_profile) {
 34276:             const double publish_finished = cuda_wall_sec();
 34277:             mixed_profile_sample.join_publish_calls++;
 34278:             mixed_profile_sample.join_publish_seconds +=
 34279:                 publish_finished - mixed_profile_phase_started;
 34280:             cuda_q1_0_mixed_profile_record_completed(
 34281:                 mixed_profile_sample,
 34282:                 publish_finished - mixed_profile_call_started,
 34283:                 hot_count, cold_count, 0);
 34284:         }
 34285:         return 1;
 34286:     }
 34287: 
 34288:     const uint64_t iq2_ssd_bytes_before = g_moe_tiering.ssd_bytes;
 34289:     const uint64_t iq2_cold_to_ram_before = g_moe_tiering.cold_to_ram;
 34290: 
 34291:     const uint64_t max_routes = CUDA_MOE_ROUTE_COUNT;
 34292:     const uint64_t out_bytes = (uint64_t)out_dim * sizeof(float);
 34293:     const uint64_t xq_blocks = expert_in_dim / CUDA_QK_K;
 34294:     const uint64_t xq_bytes = xq_blocks * sizeof(cuda_block_q8_K);
 34295:     const uint64_t off_hot_selected = 0;
 34296:     const uint64_t off_hot_weights = cuda_round_up(
 34297:         off_hot_selected + max_routes * sizeof(int32_t), 4u);
 34298:     const uint64_t off_cold_selected = cuda_round_up(
 34299:         off_hot_weights + max_routes * sizeof(float), 4u);
 34300:     const uint64_t off_cold_weights = cuda_round_up(
 34301:         off_cold_selected + max_routes * sizeof(int32_t), 4u);
 34302:     const uint64_t off_cold_out = cuda_round_up(
 34303:         off_cold_weights + max_routes * sizeof(float), 256u);
 34304:     const uint64_t cold_down_bytes = std::max(
 34305:         (uint64_t)cold_count * out_bytes, xq_bytes);
 34306:     const uint64_t off_cold_down = cuda_round_up(
 34307:         off_cold_out + out_bytes, 256u);
 34308:     if (cold_down_bytes > UINT64_MAX - off_cold_down ||
 34309:         !cuda_iq1_mixed_scratch_ensure(off_cold_down + cold_down_bytes)) {
 34310:         fprintf(stderr,
 34311:                 "ds4: [q1-0-mixed] result=failed reason=scratch "
 34312:                 "layer=%u requested=%llu cold_down=%llu\n",
 34313:                 layer_index,
 34314:                 (unsigned long long)(off_cold_down + cold_down_bytes),
 34315:                 (unsigned long long)cold_down_bytes);
 34316:         g_q1_0_mixed_failures++;
 34317:         return 0;
 34318:     }
 34319:     char *scratch = (char *)g_iq1_mixed_scratch;
 34320:     int32_t *device_hot_selected =
 34321:         (int32_t *)(scratch + off_hot_selected);
 34322:     float *device_hot_weights = (float *)(scratch + off_hot_weights);
 34323:     int32_t *device_cold_selected =
 34324:         (int32_t *)(scratch + off_cold_selected);
 34325:     float *device_cold_weights = (float *)(scratch + off_cold_weights);
 34326:     float *device_cold_out = (float *)(scratch + off_cold_out);
 34327:     float *device_cold_down = (float *)(scratch + off_cold_down);
 34328:     if (mixed_profile) {
 34329:         const double scratch_finished = cuda_wall_sec();
 34330:         mixed_profile_sample.scratch_prepare_calls++;
 34331:         mixed_profile_sample.scratch_prepare_seconds +=
 34332:             scratch_finished - mixed_profile_phase_started;
 34333:         mixed_profile_phase_started = scratch_finished;
 34334:     }
 34335: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34336:     if (attribution) {
 34337:         (void)cuda_g130_attribution_switch(
 34338:             attribution_state, CUDA_G130_ATTRIB_H2D_ENQUEUE);
 34339:     }
 34340: #endif
 34341:     const int metadata_h2d_ok = !((hot_count != 0u &&
 34342:          (!cuda_ok(cudaMemcpy(device_hot_selected, hot_selected,
 34343:                               (size_t)hot_count * sizeof(int32_t),
 34344:                               cudaMemcpyHostToDevice),
 34345:                    "Q1_0 mixed hot selected H2D") ||
 34346:           !cuda_ok(cudaMemcpy(device_hot_weights, hot_weights,
 34347:                               (size_t)hot_count * sizeof(float),
 34348:                               cudaMemcpyHostToDevice),
 34349:                    "Q1_0 mixed hot weights H2D"))) ||
 34350:         !cuda_ok(cudaMemcpy(device_cold_selected, cold_selected,
 34351:                             (size_t)cold_count * sizeof(int32_t),
 34352:                             cudaMemcpyHostToDevice),
 34353:                  "Q1_0 mixed cold selected H2D") ||
 34354:         !cuda_ok(cudaMemcpy(device_cold_weights, cold_weights,
 34355:                             (size_t)cold_count * sizeof(float),
 34356:                             cudaMemcpyHostToDevice),
 34357:                  "Q1_0 mixed cold weights H2D"));
 34358: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
 34359:     if (attribution) {
 34360:         (void)cuda_g130_attribution_switch(
 34361:             attribution_state, CUDA_G130_ATTRIB_MIXED_Q1_CALL);
 34362:     }
 34363: #endif
 34364:     if (!metadata_h2d_ok) {
 34365:         g_q1_0_mixed_failures++;
 34366:         return 0;
 34367:     }

2026-07-21T09:51:33.993518Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"cuda_moe_expert_cache_prepare|cuda_moe_gpu_resident_routes_(begin|submit|finish)|cuda_moe_route_worker|route_thread|host_route|route_stage|device_route\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Select-Object -First 200" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 479ms:
22760:    uint64_t *device_route_ptrs;
22761:    uint32_t *device_route_hit_mask;
22762:    float *device_route_split_down;
22763:    uint64_t device_route_split_down_bytes;
22766:    uint32_t *device_route_ready_sequence;
22767:    uint32_t *device_route_failed_sequence;
22779:    os_thread_t route_thread;
22784:    int route_thread_started;
22815:static void *cuda_moe_route_worker(void *arg);
25907:    if (g_moe_expert_cache.device_route_hit_mask) {
25908:        (void)cudaMemset(g_moe_expert_cache.device_route_hit_mask, 0, sizeof(uint32_t));
25917:    if (g_moe_expert_cache.route_thread_started) {
25918:        os_thread_join(g_moe_expert_cache.route_thread);
25919:        g_moe_expert_cache.route_thread_started = 0;
25975:    if (g_moe_expert_cache.device_route_ptrs) {
25976:        (void)cudaFree(g_moe_expert_cache.device_route_ptrs);
25978:    if (g_moe_expert_cache.device_route_hit_mask) {
25979:        (void)cudaFree(g_moe_expert_cache.device_route_hit_mask);
25981:    if (g_moe_expert_cache.device_route_split_down) {
25982:        (void)cudaFree(g_moe_expert_cache.device_route_split_down);
26021:    g_moe_expert_cache.device_route_ptrs = NULL;
26022:    g_moe_expert_cache.device_route_hit_mask = NULL;
26023:    g_moe_expert_cache.device_route_split_down = NULL;
26024:    g_moe_expert_cache.device_route_split_down_bytes = 0;
26031:    g_moe_expert_cache.device_route_ready_sequence = NULL;
26032:    g_moe_expert_cache.device_route_failed_sequence = NULL;
26591:static cuda_moe_expert_cache *cuda_moe_expert_cache_prepare(
26731:        uint64_t *device_routes = NULL;
26776:            err = cudaMalloc((void **)&device_routes,
26902:            g_moe_expert_cache.device_route_ptrs = device_routes;
26903:            g_moe_expert_cache.device_route_hit_mask = device_hit_mask;
26906:            g_moe_expert_cache.device_route_ready_sequence = device_ready;
26907:            g_moe_expert_cache.device_route_failed_sequence = device_failed;
26950:                os_thread_create(&g_moe_expert_cache.route_thread,
26951:                                 cuda_moe_route_worker,
26957:            g_moe_expert_cache.route_thread_started = gpu_routes ? 1 : 0;
27014:        if (device_routes) (void)cudaFree(device_routes);
27081:static int cuda_moe_route_worker_pick_slot(
27213:        const int slot = cuda_moe_route_worker_pick_slot(cache, claimed);
27628:                cache->device_route_ptrs,
27738:                cache->device_route_ptrs,
27812:static void *cuda_moe_route_worker(void *arg) {
28026:            const int cache_slot_i = cuda_moe_route_worker_pick_slot(cache, claimed.data());
28154:                cache->device_route_ptrs,
30631:        cuda_moe_expert_cache_prepare(gate_expert_bytes, down_expert_bytes);
31198:    if (cache->device_route_split_down &&
31199:        cache->device_route_split_down_bytes >= bytes) {
31202:    if (cache->device_route_split_down) {
31203:        (void)cudaFree(cache->device_route_split_down);
31204:        cache->device_route_split_down = NULL;
31205:        cache->device_route_split_down_bytes = 0;
31207:    if (cudaMalloc((void **)&cache->device_route_split_down, (size_t)bytes) !=
31212:    cache->device_route_split_down_bytes = bytes;
31505:static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_begin(
31560:        cuda_moe_expert_cache_prepare(gate_expert_bytes, down_expert_bytes);
31561:    if (!cache || !cache->route_thread_started || !cache->route_worker_ready ||
31564:        !cache->device_slot_by_layer_expert || !cache->device_route_ptrs ||
31565:        !cache->device_route_hit_mask ||
31567:        !cache->device_route_ready_sequence ||
31568:        !cache->device_route_failed_sequence ||
31577:                    cache ? cache->route_thread_started : 0,
31583:                    cache && cache->device_route_ptrs,
31584:                    cache && cache->device_route_hit_mask,
31587:                    cache && cache->device_route_ready_sequence,
31588:                    cache && cache->device_route_failed_sequence,
31611:        cache->device_route_ptrs,
31612:        cache->device_route_hit_mask,
31613:        cache->device_route_ready_sequence,
31649:static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_finish(
31775:static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_submit(
31789:    cuda_moe_expert_cache *cache = cuda_moe_gpu_resident_routes_begin(
31793:    return cuda_moe_gpu_resident_routes_finish(
31972:            gpu_route_cache = cuda_moe_gpu_resident_routes_begin(
31982:            gpu_route_cache = cuda_moe_gpu_resident_routes_submit(
32012:        mixed_route_ptrs = gpu_route_cache->device_route_ptrs;
32970:            const uint32_t *hit_mask = gpu_route_cache->device_route_hit_mask;
32971:            float *split_down = gpu_route_cache->device_route_split_down;
33008:            gpu_route_cache = cuda_moe_gpu_resident_routes_finish(
33935:        cuda_moe_expert_cache *cache = cuda_moe_expert_cache_prepare(
33937:        if (!cache || !cache->route_thread_started ||

2026-07-21T09:51:39.763093Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 26591 26975; Show-Lines $p 31505 31800' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 824ms:
 26591: static cuda_moe_expert_cache *cuda_moe_expert_cache_prepare(
 26592:         uint64_t gate_expert_bytes,
 26593:         uint64_t down_expert_bytes) {
 26594:     const uint32_t requested = cuda_moe_expert_cache_requested();
 26595:     const int tier_mode = cuda_moe_tiering_mode_requested();
 26596:     const int compose_requested = cuda_moe_prefill_tier_compose_requested();
 26597:     const int compose_router_open =
 26598:         cuda_moe_prefill_tier_router_open_requested();
 26599:     const int promotion_probation_slots =
 26600:         cuda_quant_promotion_probation_slots_requested();
 26601:     const int promotion_q1_0 =
 26602:         cuda_q1_0_dynamic_promotion_requested();
 26603:     const int gpu_routes = cuda_moe_gpu_routes_requested();
 26604:     const int packed_copy_requested = cuda_moe_route_packed_copy_requested();
 26605:     const int prefill_vram_seed_per_layer =
 26606:         cuda_moe_prefill_vram_seed_per_layer_requested();
 26607:     const int prefill_vram_seed_total =
 26608:         cuda_moe_prefill_vram_seed_total_requested();
 26609:     const int prefill_vram_seed_floor_per_layer =
 26610:         cuda_moe_prefill_vram_seed_floor_per_layer_requested();
 26611:     uint64_t promotion_min_touches = 1u;
 26612:     double promotion_min_weight = 0.0;
 26613:     double promotion_min_mass = 0.0;
 26614:     uint64_t promotion_request_budget = 0u;
 26615:     uint64_t promotion_window_calls = 0u;
 26616:     uint64_t promotion_window_budget = 0u;
 26617:     if (promotion_probation_slots > 0 &&
 26618:         !cuda_quant_promotion_gate_config(
 26619:             promotion_q1_0,
 26620:             &promotion_min_touches, &promotion_min_weight,
 26621:             &promotion_min_mass, &promotion_request_budget,
 26622:             &promotion_window_calls, &promotion_window_budget)) {
 26623:         return NULL;
 26624:     }
 26625:     if (compose_requested > 0 && g_prefill_mass_observer.enabled &&
 26626:         (!g_prefill_mass_observer.finalized ||
 26627:          !g_prefill_mass_observer.wrap_published)) {
 26628:         return NULL;
 26629:     }
 26630:     if (tier_mode < 0 || compose_requested < 0 || compose_router_open < 0 ||
 26631:         promotion_probation_slots < 0 ||
 26632:         prefill_vram_seed_per_layer < 0 ||
 26633:         prefill_vram_seed_total < 0 ||
 26634:         prefill_vram_seed_floor_per_layer < 0 ||
 26635:         (tier_mode != CUDA_MOE_TIER_OFF && !gpu_routes)) {
 26636:         if (tier_mode != CUDA_MOE_TIER_OFF) {
 26637:             fprintf(stderr,
 26638:                     "ds4: expert tiering requires GPU-resident routes\n");
 26639:         }
 26640:         return NULL;
 26641:     }
 26642:     if ((prefill_vram_seed_per_layer > 0 ||
 26643:          prefill_vram_seed_total > 0) &&
 26644:         (tier_mode != CUDA_MOE_TIER_ENFORCE ||
 26645:          !compose_requested || !gpu_routes)) {
 26646:         fprintf(stderr,
 26647:                 "ds4: prefill VRAM seed requires composed enforce tiering and GPU-resident routes\n");
 26648:         return NULL;
 26649:     }
 26650:     if ((prefill_vram_seed_per_layer > 0 &&
 26651:          prefill_vram_seed_total > 0) ||
 26652:         (prefill_vram_seed_floor_per_layer > 0 &&
 26653:          prefill_vram_seed_total == 0) ||
 26654:         (prefill_vram_seed_total > 0 &&
 26655:          (uint64_t)prefill_vram_seed_floor_per_layer *
 26656:              (CUDA_MOE_LAYER_COUNT - 3u) >
 26657:              (uint64_t)prefill_vram_seed_total)) {
 26658:         fprintf(stderr,
 26659:                 "ds4: prefill VRAM seed mode/budget contract invalid\n");
 26660:         return NULL;
 26661:     }
 26662:     if (requested == 0) {
 26663:         if (g_moe_expert_cache.capacity != 0) cuda_moe_expert_cache_release();
 26664:         return NULL;
 26665:     }
 26666:     if (gate_expert_bytes == 0 || down_expert_bytes == 0 ||
 26667:         gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
 26668:         return NULL;
 26669:     }
 26670:     if (g_moe_expert_cache.capacity != 0 &&
 26671:         g_moe_expert_cache.requested == requested &&
 26672:         g_moe_expert_cache.gate_expert_bytes == gate_expert_bytes &&
 26673:         g_moe_expert_cache.down_expert_bytes == down_expert_bytes &&
 26674:         g_moe_expert_cache.packed_copy_requested ==
 26675:             (packed_copy_requested ? 1u : 0u) &&
 26676:         g_moe_expert_cache.prefill_vram_seed_per_layer ==
 26677:             (uint32_t)prefill_vram_seed_per_layer &&
 26678:         g_moe_expert_cache.prefill_vram_seed_total ==
 26679:             (uint32_t)prefill_vram_seed_total &&
 26680:         g_moe_expert_cache.prefill_vram_seed_floor_per_layer ==
 26681:             (uint32_t)prefill_vram_seed_floor_per_layer &&
 26682:         g_moe_tiering.mode == (cuda_moe_tier_mode)tier_mode &&
 26683:         g_moe_tiering.compose_prefill_mass_tiering == compose_requested &&
 26684:         g_moe_tiering.compose_router_open == compose_router_open &&
 26685:         g_iq1_promotion.requested_slots ==
 26686:             (uint32_t)promotion_probation_slots &&
 26687:         g_iq1_promotion.min_touches == promotion_min_touches &&
 26688:         g_iq1_promotion.min_weight == promotion_min_weight &&
 26689:         g_iq1_promotion.min_mass == promotion_min_mass &&
 26690:         g_iq1_promotion.request_budget == promotion_request_budget &&
 26691:         g_iq1_promotion.window_calls == promotion_window_calls &&
 26692:         g_iq1_promotion.window_budget == promotion_window_budget &&
 26693:         (tier_mode != CUDA_MOE_TIER_ENFORCE ||
 26694:          (g_moe_expert_cache.route_transient_gate &&
 26695:           g_moe_expert_cache.route_transient_up &&
 26696:           g_moe_expert_cache.route_transient_down))) {
 26697:         return &g_moe_expert_cache;
 26698:     }
 26699:     cuda_moe_expert_cache_release();
 26700: 
 26701:     const uint64_t per_expert = gate_expert_bytes * 2ull + down_expert_bytes;
 26702:     size_t free_b = 0;
 26703:     size_t total_b = 0;
 26704:     cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
 26705:     if (err != cudaSuccess) {
 26706:         fprintf(stderr, "ds4: CUDA resident expert cache memory query failed: %s\n",
 26707:                 cudaGetErrorString(err));
 26708:         (void)cudaGetLastError();
 26709:         return NULL;
 26710:     }
 26711:     uint64_t reserve = cuda_moe_expert_cache_reserve_bytes();
 26712:     if (total_b != 0 && reserve > (uint64_t)total_b / 2ull) reserve = (uint64_t)total_b / 2ull;
 26713:     uint32_t cap = 0;
 26714:     if ((uint64_t)free_b > reserve) {
 26715:         const uint64_t max_slots = ((uint64_t)free_b - reserve) / per_expert;
 26716:         cap = max_slots < requested ? (uint32_t)max_slots : requested;
 26717:     }
 26718:     if (cap == 0) {
 26719:         fprintf(stderr,
 26720:                 "ds4: CUDA resident expert cache disabled: free %.2f GiB <= reserve %.2f GiB\n",
 26721:                 (double)free_b / 1073741824.0, (double)reserve / 1073741824.0);
 26722:         return NULL;
 26723:     }
 26724: 
 26725:     while (cap != 0) {
 26726:         void *packed_base = NULL;
 26727:         void *gate = NULL;
 26728:         void *up = NULL;
 26729:         void *down = NULL;
 26730:         int32_t *device_map = NULL;
 26731:         uint64_t *device_routes = NULL;
 26732:         uint32_t *device_hit_mask = NULL;
 26733:         uint32_t *ready_host = NULL;
 26734:         uint32_t *failed_host = NULL;
 26735:         uint32_t *device_ready = NULL;
 26736:         uint32_t *device_failed = NULL;
 26737:         cuda_moe_route_request *request_host = NULL;
 26738:         cuda_moe_route_request *request_device = NULL;
 26739:         char *host_packed_base = NULL;
 26740:         char *host_gate = NULL;
 26741:         char *host_up = NULL;
 26742:         char *host_down = NULL;
 26743:         char *transient_packed_base = NULL;
 26744:         char *transient_gate = NULL;
 26745:         char *transient_up = NULL;
 26746:         char *transient_down = NULL;
 26747:         cudaStream_t route_stream = NULL;
 26748:         if (packed_copy_requested) {
 26749:             const uint64_t gate_plane_bytes =
 26750:                 (uint64_t)cap * gate_expert_bytes;
 26751:             const uint64_t down_plane_bytes =
 26752:                 (uint64_t)cap * down_expert_bytes;
 26753:             if (gate_plane_bytes >
 26754:                     (UINT64_MAX - down_plane_bytes) / 2ull) {
 26755:                 err = cudaErrorMemoryAllocation;
 26756:             } else {
 26757:                 const uint64_t packed_bytes =
 26758:                     gate_plane_bytes * 2ull + down_plane_bytes;
 26759:                 err = cudaMalloc(&packed_base, (size_t)packed_bytes);
 26760:             }
 26761:             if (err == cudaSuccess) {
 26762:                 gate = packed_base;
 26763:                 up = (char *)packed_base + gate_plane_bytes;
 26764:                 down = (char *)packed_base + gate_plane_bytes * 2ull;
 26765:             }
 26766:         } else {
 26767:             err = cudaMalloc(&gate, (size_t)((uint64_t)cap * gate_expert_bytes));
 26768:             if (err == cudaSuccess) err = cudaMalloc(&up, (size_t)((uint64_t)cap * gate_expert_bytes));
 26769:             if (err == cudaSuccess) err = cudaMalloc(&down, (size_t)((uint64_t)cap * down_expert_bytes));
 26770:         }
 26771:         if (err == cudaSuccess && gpu_routes) {
 26772:             err = cudaMalloc((void **)&device_map,
 26773:                 (size_t)CUDA_MOE_LAYER_COUNT * 256u * sizeof(int32_t));
 26774:         }
 26775:         if (err == cudaSuccess && gpu_routes) {
 26776:             err = cudaMalloc((void **)&device_routes,
 26777:                 (size_t)CUDA_MOE_ROUTE_COUNT * 3u * sizeof(uint64_t));
 26778:         }
 26779:         if (err == cudaSuccess && gpu_routes) {
 26780:             err = cudaMalloc((void **)&device_hit_mask, sizeof(uint32_t));
 26781:         }
 26782:         if (err == cudaSuccess && gpu_routes) {
 26783:             err = cudaHostAlloc((void **)&ready_host, sizeof(uint32_t),
 26784:                                 cudaHostAllocMapped);
 26785:         }
 26786:         if (err == cudaSuccess && gpu_routes) {
 26787:             err = cudaHostGetDevicePointer((void **)&device_ready, ready_host, 0);
 26788:         }
 26789:         if (err == cudaSuccess && gpu_routes) {
 26790:             err = cudaHostAlloc((void **)&failed_host, sizeof(uint32_t),
 26791:                                 cudaHostAllocMapped);
 26792:         }
 26793:         if (err == cudaSuccess && gpu_routes) {
 26794:             err = cudaHostGetDevicePointer((void **)&device_failed, failed_host, 0);
 26795:         }
 26796:         if (err == cudaSuccess && gpu_routes) {
 26797:             err = cudaHostAlloc((void **)&request_host, sizeof(*request_host),
 26798:                                 cudaHostAllocMapped);
 26799:         }
 26800:         if (err == cudaSuccess && gpu_routes) {
 26801:             err = cudaHostGetDevicePointer((void **)&request_device, request_host, 0);
 26802:         }
 26803:         if (err == cudaSuccess && gpu_routes && packed_copy_requested) {
 26804:             const uint64_t gate_plane_bytes =
 26805:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes;
 26806:             const uint64_t down_plane_bytes =
 26807:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * down_expert_bytes;
 26808:             if (gate_plane_bytes >
 26809:                     (UINT64_MAX - down_plane_bytes) / 2ull) {
 26810:                 err = cudaErrorMemoryAllocation;
 26811:             } else {
 26812:                 const uint64_t packed_bytes =
 26813:                     gate_plane_bytes * 2ull + down_plane_bytes;
 26814:                 err = cudaHostAlloc((void **)&host_packed_base,
 26815:                     (size_t)packed_bytes, cudaHostAllocDefault);
 26816:             }
 26817:             if (err == cudaSuccess) {
 26818:                 host_gate = host_packed_base;
 26819:                 host_up = host_packed_base + gate_plane_bytes;
 26820:                 host_down = host_packed_base + gate_plane_bytes * 2ull;
 26821:             }
 26822:         }
 26823:         if (err == cudaSuccess && gpu_routes && !packed_copy_requested) {
 26824:             err = cudaHostAlloc((void **)&host_gate,
 26825:                 (size_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes, cudaHostAllocDefault);
 26826:         }
 26827:         if (err == cudaSuccess && gpu_routes && !packed_copy_requested) {
 26828:             err = cudaHostAlloc((void **)&host_up,
 26829:                 (size_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes, cudaHostAllocDefault);
 26830:         }
 26831:         if (err == cudaSuccess && gpu_routes && !packed_copy_requested) {
 26832:             err = cudaHostAlloc((void **)&host_down,
 26833:                 (size_t)CUDA_MOE_ROUTE_COUNT * down_expert_bytes, cudaHostAllocDefault);
 26834:         }
 26835:         if (err == cudaSuccess && tier_mode == CUDA_MOE_TIER_ENFORCE &&
 26836:             packed_copy_requested) {
 26837:             const uint64_t gate_plane_bytes =
 26838:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes;
 26839:             const uint64_t down_plane_bytes =
 26840:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * down_expert_bytes;
 26841:             if (gate_plane_bytes >
 26842:                     (UINT64_MAX - down_plane_bytes) / 2ull) {
 26843:                 err = cudaErrorMemoryAllocation;
 26844:             } else {
 26845:                 const uint64_t packed_bytes =
 26846:                     gate_plane_bytes * 2ull + down_plane_bytes;
 26847:                 err = cudaMalloc((void **)&transient_packed_base,
 26848:                     (size_t)packed_bytes);
 26849:             }
 26850:             if (err == cudaSuccess) {
 26851:                 transient_gate = transient_packed_base;
 26852:                 transient_up = transient_packed_base + gate_plane_bytes;
 26853:                 transient_down = transient_packed_base +
 26854:                     gate_plane_bytes * 2ull;
 26855:             }
 26856:         }
 26857:         if (err == cudaSuccess && tier_mode == CUDA_MOE_TIER_ENFORCE &&
 26858:             !packed_copy_requested) {
 26859:             err = cudaMalloc((void **)&transient_gate,
 26860:                 (size_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes);
 26861:         }
 26862:         if (err == cudaSuccess && tier_mode == CUDA_MOE_TIER_ENFORCE &&
 26863:             !packed_copy_requested) {
 26864:             err = cudaMalloc((void **)&transient_up,
 26865:                 (size_t)CUDA_MOE_ROUTE_COUNT * gate_expert_bytes);
 26866:         }
 26867:         if (err == cudaSuccess && tier_mode == CUDA_MOE_TIER_ENFORCE &&
 26868:             !packed_copy_requested) {
 26869:             err = cudaMalloc((void **)&transient_down,
 26870:                 (size_t)CUDA_MOE_ROUTE_COUNT * down_expert_bytes);
 26871:         }
 26872:         if (err == cudaSuccess && gpu_routes) {
 26873:             err = cudaStreamCreateWithFlags(&route_stream, cudaStreamNonBlocking);
 26874:         }
 26875:         if (err == cudaSuccess) {
 26876:             try {
 26877:                 g_moe_expert_cache.slots.resize(cap);
 26878:                 if (gpu_routes) {
 26879:                     g_moe_expert_cache.host_slot_by_layer_expert.assign(
 26880:                         (size_t)CUDA_MOE_LAYER_COUNT * 256u, -1);
 26881:                 }
 26882:             } catch (...) {
 26883:                 err = cudaErrorMemoryAllocation;
 26884:             }
 26885:         }
 26886:         if (err == cudaSuccess) {
 26887:             g_moe_expert_cache.packed_base = (char *)packed_base;
 26888:             g_moe_expert_cache.gate = (char *)gate;
 26889:             g_moe_expert_cache.up = (char *)up;
 26890:             g_moe_expert_cache.down = (char *)down;
 26891:             g_moe_expert_cache.gate_expert_bytes = gate_expert_bytes;
 26892:             g_moe_expert_cache.down_expert_bytes = down_expert_bytes;
 26893:             g_moe_expert_cache.requested = requested;
 26894:             g_moe_expert_cache.capacity = cap;
 26895:             g_moe_expert_cache.prefill_vram_seed_per_layer =
 26896:                 (uint32_t)prefill_vram_seed_per_layer;
 26897:             g_moe_expert_cache.prefill_vram_seed_total =
 26898:                 (uint32_t)prefill_vram_seed_total;
 26899:             g_moe_expert_cache.prefill_vram_seed_floor_per_layer =
 26900:                 (uint32_t)prefill_vram_seed_floor_per_layer;
 26901:             g_moe_expert_cache.device_slot_by_layer_expert = device_map;
 26902:             g_moe_expert_cache.device_route_ptrs = device_routes;
 26903:             g_moe_expert_cache.device_route_hit_mask = device_hit_mask;
 26904:             g_moe_expert_cache.route_ready_host = ready_host;
 26905:             g_moe_expert_cache.route_failed_host = failed_host;
 26906:             g_moe_expert_cache.device_route_ready_sequence = device_ready;
 26907:             g_moe_expert_cache.device_route_failed_sequence = device_failed;
 26908:             g_moe_expert_cache.route_request_host = request_host;
 26909:             g_moe_expert_cache.route_request_device = request_device;
 26910:             g_moe_expert_cache.route_host_packed_base = host_packed_base;
 26911:             g_moe_expert_cache.route_host_gate = host_gate;
 26912:             g_moe_expert_cache.route_host_up = host_up;
 26913:             g_moe_expert_cache.route_host_down = host_down;
 26914:             g_moe_expert_cache.route_transient_packed_base =
 26915:                 transient_packed_base;
 26916:             g_moe_expert_cache.route_transient_gate = transient_gate;
 26917:             g_moe_expert_cache.route_transient_up = transient_up;
 26918:             g_moe_expert_cache.route_transient_down = transient_down;
 26919:             g_moe_expert_cache.packed_copy_requested =
 26920:                 packed_copy_requested ? 1u : 0u;
 26921:             g_moe_expert_cache.packed_resident_alloc =
 26922:                 packed_base ? 1u : 0u;
 26923:             g_moe_expert_cache.packed_route_host_alloc =
 26924:                 host_packed_base ? 1u : 0u;
 26925:             g_moe_expert_cache.packed_transient_alloc =
 26926:                 transient_packed_base ? 1u : 0u;
 26927:             g_moe_expert_cache.route_upload_stream = route_stream;
 26928:             g_moe_expert_cache.route_stop = 0;
 26929:             g_moe_expert_cache.route_worker_ready = 0;
 26930:             g_moe_expert_cache.route_worker_failed = 0;
 26931:             (void)cudaGetDevice(&g_moe_expert_cache.route_device);
 26932:             if (!cuda_moe_tiering_prepare()) {
 26933:                 fprintf(stderr, "ds4: expert tiering preparation failed\n");
 26934:                 cuda_moe_expert_cache_release();
 26935:                 return NULL;
 26936:             }
 26937:             cuda_moe_expert_cache_invalidate();
 26938:             if ((prefill_vram_seed_per_layer > 0 ||
 26939:                  prefill_vram_seed_total > 0) &&
 26940:                 !cuda_moe_prefill_vram_seed(
 26941:                     &g_moe_expert_cache,
 26942:                     (uint32_t)prefill_vram_seed_per_layer,
 26943:                     (uint32_t)prefill_vram_seed_total,
 26944:                     (uint32_t)prefill_vram_seed_floor_per_layer)) {
 26945:                 fprintf(stderr, "ds4: prefill VRAM seed failed\n");
 26946:                 cuda_moe_expert_cache_release();
 26947:                 return NULL;
 26948:             }
 26949:             if (gpu_routes &&
 26950:                 os_thread_create(&g_moe_expert_cache.route_thread,
 26951:                                  cuda_moe_route_worker,
 26952:                                  &g_moe_expert_cache) != 0) {
 26953:                 fprintf(stderr, "ds4: CUDA GPU-resident route worker creation failed\n");
 26954:                 cuda_moe_expert_cache_release();
 26955:                 return NULL;
 26956:             }
 26957:             g_moe_expert_cache.route_thread_started = gpu_routes ? 1 : 0;
 26958:             if (gpu_routes) {
 26959:                 const double ready_deadline = cuda_wall_sec() + 2.0;
 26960:                 while (!g_moe_expert_cache.route_worker_ready &&
 26961:                        !g_moe_expert_cache.route_worker_failed &&
 26962:                        cuda_wall_sec() < ready_deadline) {
 26963: #ifdef _WIN32
 26964:                     (void)SwitchToThread();
 26965: #else
 26966:                     usleep(0);
 26967: #endif
 26968:                 }
 26969:                 if (!g_moe_expert_cache.route_worker_ready) {
 26970:                     fprintf(stderr, "ds4: CUDA GPU-resident route worker failed to initialize\n");
 26971:                     cuda_moe_expert_cache_release();
 26972:                     return NULL;
 26973:                 }
 26974:             }
 26975:             const uint64_t startup_reserve = cuda_model_stream_reserve_bytes();
 31505: static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_begin(
 31506:         uint32_t layer_index,
 31507:         uint64_t gate_offset,
 31508:         uint64_t up_offset,
 31509:         uint64_t down_offset,
 31510:         uint64_t gate_expert_bytes,
 31511:         uint64_t down_expert_bytes,
 31512:         uint32_t expert_in_dim,
 31513:         const ds4_gpu_tensor *selected,
 31514:         const ds4_gpu_tensor *weights,
 31515:         uint32_t n_expert,
 31516:         uint32_t n_tokens,
 31517:         ds4_gpu_spex_queue *spex_queue,
 31518:         uint32_t split_out_dim,
 31519:         uint32_t *sequence_out) {
 31520:     const int compose_requested = cuda_moe_prefill_tier_compose_requested();
 31521:     if (!cuda_moe_gpu_routes_requested() || !selected || !selected->ptr ||
 31522:         compose_requested < 0 ||
 31523:         !weights || !weights->ptr ||
 31524:         n_tokens != 1u || n_expert == 0u || n_expert > CUDA_MOE_ROUTE_COUNT ||
 31525:         layer_index >= CUDA_MOE_LAYER_COUNT || spex_queue ||
 31526:         cuda_moe_expert_cache_layer_top1() ||
 31527:         g_dynamic_arena_observer.enabled ||
 31528:         (g_prefill_mass_observer.enabled &&
 31529:          (!compose_requested ||
 31530:           !g_prefill_mass_observer.finalized ||
 31531:           !g_prefill_mass_observer.wrap_published)) ||
 31532:         g_reap_mass_observer.enabled ||
 31533:         expert_in_dim / CUDA_QK_K > 16u ||
 31534:         getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") != NULL ||
 31535:         getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") != NULL) {
 31536:         if (n_expert != CUDA_MOE_ROUTE_COUNT) {
 31537:             fprintf(stderr,
 31538:                     "ds4: [gpu-route-variable] result=begin-refused routes=%u "
 31539:                     "requested=%d selected=%d weights=%d tokens=%u layer=%u "
 31540:                     "spex=%d top1=%d arena_observer=%d prefill_observer=%d "
 31541:                     "prefill_finalized=%d prefill_wrapped=%d reap_observer=%d "
 31542:                     "xq_blocks=%u no_lut=%d no_sum6=%d compose=%d\n",
 31543:                     n_expert, cuda_moe_gpu_routes_requested(),
 31544:                     selected && selected->ptr, weights && weights->ptr,
 31545:                     n_tokens, layer_index, spex_queue != NULL,
 31546:                     cuda_moe_expert_cache_layer_top1(),
 31547:                     g_dynamic_arena_observer.enabled,
 31548:                     g_prefill_mass_observer.enabled,
 31549:                     g_prefill_mass_observer.finalized,
 31550:                     g_prefill_mass_observer.wrap_published,
 31551:                     g_reap_mass_observer.enabled,
 31552:                     expert_in_dim / CUDA_QK_K,
 31553:                     getenv("DS4_CUDA_MOE_NO_DECODE_LUT_GATE") != NULL,
 31554:                     getenv("DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6") != NULL,
 31555:                     compose_requested);
 31556:         }
 31557:         return NULL;
 31558:     }
 31559:     cuda_moe_expert_cache *cache =
 31560:         cuda_moe_expert_cache_prepare(gate_expert_bytes, down_expert_bytes);
 31561:     if (!cache || !cache->route_thread_started || !cache->route_worker_ready ||
 31562:         cache->route_worker_failed || cache->route_stop ||
 31563:         cache->capacity < CUDA_MOE_ROUTE_COUNT ||
 31564:         !cache->device_slot_by_layer_expert || !cache->device_route_ptrs ||
 31565:         !cache->device_route_hit_mask ||
 31566:         !cache->route_ready_host || !cache->route_failed_host ||
 31567:         !cache->device_route_ready_sequence ||
 31568:         !cache->device_route_failed_sequence ||
 31569:         !cache->route_request_device) {
 31570:         if (n_expert != CUDA_MOE_ROUTE_COUNT) {
 31571:             fprintf(stderr,
 31572:                     "ds4: [gpu-route-variable] result=cache-refused routes=%u "
 31573:                     "cache=%d thread=%d ready=%d failed=%d stop=%d capacity=%u "
 31574:                     "slot_map=%d ptrs=%d hit_mask=%d ready_host=%d failed_host=%d "
 31575:                     "ready_dev=%d failed_dev=%d request_dev=%d\n",
 31576:                     n_expert, cache != NULL,
 31577:                     cache ? cache->route_thread_started : 0,
 31578:                     cache ? cache->route_worker_ready : 0,
 31579:                     cache ? cache->route_worker_failed : 0,
 31580:                     cache ? cache->route_stop : 0,
 31581:                     cache ? cache->capacity : 0,
 31582:                     cache && cache->device_slot_by_layer_expert,
 31583:                     cache && cache->device_route_ptrs,
 31584:                     cache && cache->device_route_hit_mask,
 31585:                     cache && cache->route_ready_host,
 31586:                     cache && cache->route_failed_host,
 31587:                     cache && cache->device_route_ready_sequence,
 31588:                     cache && cache->device_route_failed_sequence,
 31589:                     cache && cache->route_request_device);
 31590:         }
 31591:         return NULL;
 31592:     }
 31593:     if (split_out_dim != 0u &&
 31594:         !cuda_moe_route_split_down_ensure(cache, split_out_dim)) {
 31595:         if (n_expert != CUDA_MOE_ROUTE_COUNT) {
 31596:             fprintf(stderr,
 31597:                     "ds4: [gpu-route-variable] result=split-buffer-refused "
 31598:                     "routes=%u out_dim=%u\n",
 31599:                     n_expert, split_out_dim);
 31600:         }
 31601:         return NULL;
 31602:     }
 31603:     uint32_t sequence = ++cache->route_sequence;
 31604:     if (sequence == 0u) sequence = ++cache->route_sequence;
 31605:     const int nested_profile = g_nested_residual.profile_enabled &&
 31606:         cuda_nested_residual_gpu_cache_requested() &&
 31607:         cuda_nested_residual_layer_required(layer_index);
 31608:     const double route_begin_started =
 31609:         nested_profile ? cuda_wall_sec() : 0.0;
 31610:     moe_resolve_resident_routes_kernel<<<1, 1>>>(
 31611:         cache->device_route_ptrs,
 31612:         cache->device_route_hit_mask,
 31613:         cache->device_route_ready_sequence,
 31614:         cache->route_request_device,
 31615:         (const int32_t *)selected->ptr,
 31616:         (const float *)weights->ptr,
 31617:         cache->device_slot_by_layer_expert,
 31618:         layer_index,
 31619:         sequence,
 31620:         cache->gate,
 31621:         cache->up,
 31622:         cache->down,
 31623:         gate_offset,
 31624:         up_offset,
 31625:         down_offset,
 31626:         gate_expert_bytes,
 31627:         down_expert_bytes,
 31628:         n_expert);
 31629:     const cudaError_t resolve_launch_error = cudaGetLastError();
 31630:     if (nested_profile) {
 31631:         g_nested_residual.profile_route_begin_calls++;
 31632:         g_nested_residual.profile_route_begin_seconds +=
 31633:             cuda_wall_sec() - route_begin_started;
 31634:     }
 31635:     if (resolve_launch_error != cudaSuccess) {
 31636:         if (n_expert != CUDA_MOE_ROUTE_COUNT) {
 31637:             fprintf(stderr,
 31638:                     "ds4: [gpu-route-variable] result=resolve-launch-failed "
 31639:                     "routes=%u error=%s\n",
 31640:                     n_expert, cudaGetErrorString(resolve_launch_error));
 31641:         }
 31642:         return NULL;
 31643:     }
 31644:     cache->route_submitted_sequence = sequence;
 31645:     if (sequence_out) *sequence_out = sequence;
 31646:     return cache;
 31647: }
 31648: 
 31649: static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_finish(
 31650:         cuda_moe_expert_cache *cache,
 31651:         uint32_t sequence,
 31652:         uint32_t layer_index,
 31653:         int split_mode,
 31654:         uint32_t split_out_dim) {
 31655:     if (!cache || sequence == 0u) return NULL;
 31656:     const int nested_profile = g_nested_residual.profile_enabled &&
 31657:         cuda_nested_residual_gpu_cache_requested() &&
 31658:         cuda_nested_residual_layer_required(layer_index);
 31659:     if (cuda_moe_route_no_default_sync_requested()) {
 31660:         cache->route_no_default_sync_calls++;
 31661:     } else {
 31662:         cache->route_default_sync_calls++;
 31663:         const double resolve_started = cuda_wall_sec();
 31664:         const cudaError_t resolve_err = cudaStreamSynchronize(0);
 31665:         const double resolve_seconds = cuda_wall_sec() - resolve_started;
 31666:         cache->route_resolve_sync_seconds += resolve_seconds;
 31667:         if (nested_profile) {
 31668:             g_nested_residual.profile_route_resolve_sync_calls++;
 31669:             g_nested_residual.profile_route_resolve_sync_seconds +=
 31670:                 resolve_seconds;
 31671:         }
 31672:         if (resolve_err != cudaSuccess) {
 31673:             fprintf(stderr,
 31674:                     "ds4: CUDA GPU-resident route resolver sync failed: %s\n",
 31675:                     cudaGetErrorString(resolve_err));
 31676:             abort();
 31677:         }
 31678:     }
 31679:     const double wait_started = cuda_wall_sec();
 31680:     const double wait_deadline = wait_started + 5.0;
 31681:     while (*(volatile uint32_t *)cache->route_ready_host != sequence) {
 31682:         if (cache->route_stop || cache->route_worker_failed) {
 31683:             fprintf(stderr,
 31684:                     "ds4: CUDA GPU-resident route resolver lost worker seq=%u\n",
 31685:                     sequence);
 31686:             abort();
 31687:         }
 31688:         if (cuda_wall_sec() >= wait_deadline) {
 31689:             fprintf(stderr,
 31690:                     "ds4: CUDA GPU-resident route worker timed out seq=%u\n",
 31691:                     sequence);
 31692:             abort();
 31693:         }
 31694: #ifdef _WIN32
 31695:         (void)SwitchToThread();
 31696: #else
 31697:         usleep(0);
 31698: #endif
 31699:     }
 31700: #ifdef _WIN32
 31701:     MemoryBarrier();
 31702: #else
 31703:     __sync_synchronize();
 31704: #endif
 31705:     const double ready_wait_seconds = cuda_wall_sec() - wait_started;
 31706:     cache->route_ready_wait_seconds += ready_wait_seconds;
 31707:     if (nested_profile) {
 31708:         g_nested_residual.profile_route_ready_wait_calls++;
 31709:         g_nested_residual.profile_route_ready_wait_seconds +=
 31710:             ready_wait_seconds;
 31711:     }
 31712:     if (*(volatile uint32_t *)cache->route_failed_host == sequence) {
 31713:         fprintf(stderr,
 31714:                 "ds4: CUDA GPU-resident route resolver observed failure seq=%u; falling back\n",
 31715:                 sequence);
 31716:         if (cuda_nested_residual_gpu_cache_requested() &&
 31717:             cuda_nested_residual_layer_required(layer_index)) {
 31718:             g_nested_residual.gpu_cache_failures++;
 31719:             g_nested_residual.hard_failure = 1;
 31720:         }
 31721:         cuda_moe_expert_cache_invalidate();
 31722:         return NULL;
 31723:     }
 31724:     const cuda_moe_route_request *request = cache->route_request_host;
 31725:     if (request && request->sequence == sequence &&
 31726:         request->route_count > 0u &&
 31727:         request->route_count <= CUDA_MOE_ROUTE_COUNT &&
 31728:         request->layer_index == layer_index) {
 31729:         if (cuda_nested_residual_gpu_cache_requested() &&
 31730:             cuda_nested_residual_layer_required(layer_index)) {
 31731:             g_nested_residual.router_open_calls++;
 31732:             g_nested_residual.gpu_route_calls++;
 31733:             g_nested_residual.gpu_route_hits +=
 31734:                 request->route_count - request->miss_count;
 31735:             g_nested_residual.gpu_route_misses += request->miss_count;
 31736:         }
 31737:         g_moe_last_selected.gate_offset = request->gate_offset;
 31738:         g_moe_last_selected.count = request->route_count;
 31739:         memcpy(g_moe_last_selected.ids, request->selected,
 31740:                request->route_count * sizeof(g_moe_last_selected.ids[0]));
 31741:         g_moe_last_selected.valid = 1;
 31742:     }
 31743:     if (sequence <= 2u) {
 31744:         fprintf(stderr,
 31745:                 "ds4: [gpu-route-debug] submit seq=%u layer=%u\n",
 31746:                 sequence, layer_index);
 31747:     }
 31748:     cache->route_calls++;
 31749:     if (split_mode == 1) {
 31750:         cache->route_split_calls++;
 31751:     } else if (split_mode == 2) {
 31752:         cache->route_split_fused_calls++;
 31753:         if (!request || request->sequence != sequence ||
 31754:             request->route_count == 0u ||
 31755:             request->route_count > CUDA_MOE_ROUTE_COUNT ||
 31756:             request->miss_count > request->route_count ||
 31757:             split_out_dim == 0u) {
 31758:             fprintf(stderr,
 31759:                     "ds4: CUDA fused split telemetry contract failed seq=%u\n",
 31760:                     sequence);
 31761:             abort();
 31762:         }
 31763:         const uint64_t misses = request->miss_count;
 31764:         const uint64_t hits = request->route_count - misses;
 31765:         cache->route_split_fused_hits += hits;
 31766:         cache->route_split_fused_misses += misses;
 31767:         cache->route_split_fused_miss_scratch_bytes_avoided +=
 31768:             misses * split_out_dim * sizeof(float);
 31769:         cache->route_split_fused_sum_read_bytes_avoided +=
 31770:             misses * split_out_dim * sizeof(float);
 31771:     }
 31772:     return cache;
 31773: }
 31774: 
 31775: static cuda_moe_expert_cache *cuda_moe_gpu_resident_routes_submit(
 31776:         uint32_t layer_index,
 31777:         uint64_t gate_offset,
 31778:         uint64_t up_offset,
 31779:         uint64_t down_offset,
 31780:         uint64_t gate_expert_bytes,
 31781:         uint64_t down_expert_bytes,
 31782:         uint32_t expert_in_dim,
 31783:         const ds4_gpu_tensor *selected,
 31784:         const ds4_gpu_tensor *weights,
 31785:         uint32_t n_expert,
 31786:         uint32_t n_tokens,
 31787:         ds4_gpu_spex_queue *spex_queue) {
 31788:     uint32_t sequence = 0;
 31789:     cuda_moe_expert_cache *cache = cuda_moe_gpu_resident_routes_begin(
 31790:         layer_index, gate_offset, up_offset, down_offset,
 31791:         gate_expert_bytes, down_expert_bytes, expert_in_dim,
 31792:         selected, weights, n_expert, n_tokens, spex_queue, 0u, &sequence);
 31793:     return cuda_moe_gpu_resident_routes_finish(
 31794:         cache, sequence, layer_index, 0, 0u);
 31795: }
 31796: 
 31797: static int routed_moe_launch(
 31798:         ds4_gpu_tensor *out,
 31799:         ds4_gpu_tensor *gate,
 31800:         ds4_gpu_tensor *up,

2026-07-21T09:51:46.065912Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"cuda_dynamic_arena_copy_expert_async|cuda_moe_tiering_.*ram_ptrs|route_host_gate|route_host_up|route_host_down|cudaMemcpyAsync\\(.*route\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Select-Object -First 160" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 483ms:
5979:static cuda_dynamic_arena_copy_status cuda_dynamic_arena_copy_expert_async(
22771:    char *route_host_gate;
22772:    char *route_host_up;
22773:    char *route_host_down;
23808:static int cuda_moe_tiering_snapshot_ram_ptrs(
23839:static int cuda_moe_tiering_ram_ptrs(
25665:    const int from_snapshot = cuda_moe_tiering_snapshot_ram_ptrs(
25667:    const int from_probation = !from_snapshot && cuda_moe_tiering_ram_ptrs(
25861:        if (cuda_moe_tiering_snapshot_ram_ptrs(
26009:        if (g_moe_expert_cache.route_host_gate) (void)cudaFreeHost(g_moe_expert_cache.route_host_gate);
26010:        if (g_moe_expert_cache.route_host_up) (void)cudaFreeHost(g_moe_expert_cache.route_host_up);
26011:        if (g_moe_expert_cache.route_host_down) (void)cudaFreeHost(g_moe_expert_cache.route_host_down);
26036:    g_moe_expert_cache.route_host_gate = NULL;
26037:    g_moe_expert_cache.route_host_up = NULL;
26038:    g_moe_expert_cache.route_host_down = NULL;
26911:            g_moe_expert_cache.route_host_gate = host_gate;
26912:            g_moe_expert_cache.route_host_up = host_up;
26913:            g_moe_expert_cache.route_host_down = host_down;
27402:                host_gate = cache->route_host_gate +
27404:                host_up = cache->route_host_up +
27406:                host_down = cache->route_host_down +
27421:            cuda_moe_tiering_snapshot_ram_ptrs(
27425:            cuda_moe_tiering_ram_ptrs(
27453:            have_ram = cuda_moe_tiering_ram_ptrs(
27475:            have_ram = cuda_moe_tiering_ram_ptrs(
27487:            host_gate = cache->route_host_gate +
27489:            host_up = cache->route_host_up +
27491:            host_down = cache->route_host_down +
28051:            char *host_gate = cache->route_host_gate +
28053:            char *host_up = cache->route_host_up +
28055:            char *host_down = cache->route_host_down +
29022:                cuda_dynamic_arena_copy_expert_async(
30474:                cuda_dynamic_arena_copy_expert_async(
31324:                cuda_dynamic_arena_copy_expert_async(
33763:    if (cuda_moe_tiering_snapshot_ram_ptrs(
33770:    if (cuda_moe_tiering_ram_ptrs(

2026-07-21T09:51:52.490542Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 27360 27775; Show-Lines $p 27990 28175' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 822ms:
 27360:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 27361:                 (unsigned long long)g_moe_tiering.snapshot_generation);
 27362:         return 0;
 27363:     }
 27364:     for (uint32_t miss = 0; ok && miss < request.miss_count; miss++) {
 27365:         const uint32_t route = request.miss_routes[miss];
 27366:         if (route >= request.route_count) {
 27367:             failure_reason = "route-oob";
 27368:             ok = 0;
 27369:             break;
 27370:         }
 27371:         const int32_t expert_i = request.selected[route];
 27372:         if (expert_i < 0 || expert_i >= 256) {
 27373:             failure_reason = "expert-oob";
 27374:             ok = 0;
 27375:             break;
 27376:         }
 27377:         const uint32_t expert = (uint32_t)expert_i;
 27378:         if (g_sparse_bake_active &&
 27379:             !cuda_sparse_bake_expert_retained(request.layer_index, expert)) {
 27380:             g_sparse_bake_route_rejections++;
 27381:             fprintf(stderr,
 27382:                     "ds4: sparse bake rejected tiering route miss "
 27383:                     "layer=%u seq=%u route=%u expert=%u\n",
 27384:                     request.layer_index, request.sequence, route, expert);
 27385:             g_moe_tiering.failures++;
 27386:             failure_reason = "sparse-rejected";
 27387:             ok = 0;
 27388:             break;
 27389:         }
 27390:         cuda_moe_tier_entry &tier = g_moe_tiering.entries[
 27391:             cuda_moe_tiering_entry_index(request.layer_index, expert)];
 27392: 
 27393:         char *host_gate = NULL;
 27394:         char *host_up = NULL;
 27395:         char *host_down = NULL;
 27396:         int from_nested = 0;
 27397:         if (cuda_nested_residual_gpu_cache_requested() &&
 27398:             cuda_nested_residual_layer_required(request.layer_index)) {
 27399:             if (cuda_nested_residual_gpu_join_requested()) {
 27400:                 from_nested = 1;
 27401:             } else {
 27402:                 host_gate = cache->route_host_gate +
 27403:                     (uint64_t)miss * cache->gate_expert_bytes;
 27404:                 host_up = cache->route_host_up +
 27405:                     (uint64_t)miss * cache->gate_expert_bytes;
 27406:                 host_down = cache->route_host_down +
 27407:                     (uint64_t)miss * cache->down_expert_bytes;
 27408:                 from_nested = cuda_nested_residual_fill_route_host_exact(
 27409:                     request.layer_index, expert,
 27410:                     request.gate_offset, request.up_offset, request.down_offset,
 27411:                     cache->gate_expert_bytes, cache->down_expert_bytes,
 27412:                     host_gate, host_up, host_down);
 27413:                 if (from_nested != 1) {
 27414:                     failure_reason = "nested-exact";
 27415:                     ok = 0;
 27416:                     break;
 27417:                 }
 27418:             }
 27419:         }
 27420:         const int from_snapshot = !from_nested &&
 27421:             cuda_moe_tiering_snapshot_ram_ptrs(
 27422:             request.layer_index, expert,
 27423:             &host_gate, &host_up, &host_down);
 27424:         const int from_probation = !from_nested && !from_snapshot &&
 27425:             cuda_moe_tiering_ram_ptrs(
 27426:                 request.layer_index, expert,
 27427:                 &host_gate, &host_up, &host_down);
 27428:         int have_ram = from_nested || from_snapshot || from_probation;
 27429:         if (from_snapshot) {
 27430:             g_moe_tiering.snapshot_backing_hits++;
 27431:         } else if (from_probation) {
 27432:             g_iq1_promotion.probation_ram_hits++;
 27433:         } else if (g_moe_tiering.compose_prefill_mass_tiering) {
 27434:             g_moe_tiering.snapshot_backing_misses++;
 27435:             if (!g_moe_tiering.compose_router_open) {
 27436:                 g_moe_tiering.forbidden_cold_ssd_to_vram++;
 27437:                 g_moe_tiering.failures++;
 27438:                 failure_reason = "snapshot-miss";
 27439:                 ok = 0;
 27440:                 break;
 27441:             }
 27442:         }
 27443:         if (!have_ram &&
 27444:             (!g_moe_tiering.compose_prefill_mass_tiering ||
 27445:              g_moe_tiering.compose_router_open) &&
 27446:             tier.state == CUDA_MOE_TIER_SSD_COLD) {
 27447:             /* Mixed 5+1 keeps the five high-weight routes exact. An open
 27448:              * snapshot miss must first enter the pinned probation pool;
 27449:              * only the minimum-weight sixth route executes as IQ1_S and is
 27450:              * staged for a later token by the mixed path. */
 27451:             (void)cuda_moe_tiering_load_to_ram(
 27452:                 request, request.layer_index, expert);
 27453:             have_ram = cuda_moe_tiering_ram_ptrs(
 27454:                 request.layer_index, expert,
 27455:                 &host_gate, &host_up, &host_down);
 27456:         }
 27457:         if (!have_ram &&
 27458:             (!g_moe_tiering.compose_prefill_mass_tiering ||
 27459:              g_moe_tiering.compose_router_open) &&
 27460:             tier.state != CUDA_MOE_TIER_SSD_COLD) {
 27461:             if (tier.state == CUDA_MOE_TIER_VRAM_PROTECTED) {
 27462:                 /* A protected entry must have been resolved as a device-map
 27463:                  * hit before the miss worker. Never create a second resident
 27464:                  * slot for the same expert if that map invariant is broken. */
 27465:                 g_moe_tiering.failures++;
 27466:                 failure_reason = "vram-map-miss";
 27467:                 ok = 0;
 27468:                 break;
 27469:             }
 27470:             tier.state = CUDA_MOE_TIER_SSD_COLD;
 27471:             tier.ram_slot = UINT32_MAX;
 27472:             tier.ram_generation = 0;
 27473:             (void)cuda_moe_tiering_load_to_ram(
 27474:                 request, request.layer_index, expert);
 27475:             have_ram = cuda_moe_tiering_ram_ptrs(
 27476:                 request.layer_index, expert,
 27477:                 &host_gate, &host_up, &host_down);
 27478:         }
 27479:         if (!have_ram) {
 27480:             if (g_moe_tiering.compose_prefill_mass_tiering) {
 27481:                 g_moe_tiering.forbidden_cold_ssd_to_vram++;
 27482:                 g_moe_tiering.failures++;
 27483:                 failure_reason = "ram-required";
 27484:                 ok = 0;
 27485:                 break;
 27486:             }
 27487:             host_gate = cache->route_host_gate +
 27488:                 (uint64_t)miss * cache->gate_expert_bytes;
 27489:             host_up = cache->route_host_up +
 27490:                 (uint64_t)miss * cache->gate_expert_bytes;
 27491:             host_down = cache->route_host_down +
 27492:                 (uint64_t)miss * cache->down_expert_bytes;
 27493:             const uint64_t gate_src = request.gate_offset +
 27494:                 (uint64_t)expert * cache->gate_expert_bytes;
 27495:             const uint64_t up_src = request.up_offset +
 27496:                 (uint64_t)expert * cache->gate_expert_bytes;
 27497:             const uint64_t down_src = request.down_offset +
 27498:                 (uint64_t)expert * cache->down_expert_bytes;
 27499:             if (!g_model_file_valid ||
 27500:                 !cuda_pread_full(&g_model_file, host_gate,
 27501:                                  cache->gate_expert_bytes, gate_src) ||
 27502:                 !cuda_pread_full(&g_model_file, host_up,
 27503:                                  cache->gate_expert_bytes, up_src) ||
 27504:                 !cuda_pread_full(&g_model_file, host_down,
 27505:                                  cache->down_expert_bytes, down_src)) {
 27506:                 g_moe_tiering.failures++;
 27507:                 failure_reason = "pread";
 27508:                 ok = 0;
 27509:                 break;
 27510:             }
 27511:             g_moe_tiering.ssd_bytes +=
 27512:                 cache->gate_expert_bytes * 2ull + cache->down_expert_bytes;
 27513:         }
 27514: 
 27515:         uint32_t ssd_wrap_h2d_ring = UINT32_MAX;
 27516:         if (have_ram && !from_nested && g_q1_0_ssd_wrap.enabled) {
 27517:             const int pageable = cuda_q1_0_ssd_wrap_host_is_pageable(
 27518:                 request.layer_index, expert, from_snapshot, from_probation);
 27519:             if (pageable < 0 || !cuda_q1_0_ssd_wrap_prepare_h2d_source(
 27520:                     &host_gate, &host_up, &host_down,
 27521:                     cache->gate_expert_bytes, cache->down_expert_bytes,
 27522:                     pageable, &ssd_wrap_h2d_ring)) {
 27523:                 failure_reason = "ssd-wrap-host-source";
 27524:                 ok = 0;
 27525:                 break;
 27526:             }
 27527:         }
 27528: 
 27529:         /* Transport accounting is representation-neutral: for supported nested
 27530:          * types, base plus residual bytes equal the native expert byte count. */
 27531:         const int cache_slot_i = have_ram ?
 27532:             cuda_moe_tiering_pick_vram_slot(cache, claimed, tier) : -1;
 27533:         const int promote = cache_slot_i >= 0;
 27534:         if (promote) {
 27535:             const uint32_t cache_slot = (uint32_t)cache_slot_i;
 27536:             claimed[cache_slot] = 1u;
 27537:             cuda_moe_cache_slot &entry = cache->slots[cache_slot];
 27538:             admitted[admitted_count++] = cache_slot;
 27539:             int32_t old_map_index = -1;
 27540:             const int was_valid = entry.state == CUDA_MOE_CACHE_VALID;
 27541:             if (was_valid && entry.layer_index < CUDA_MOE_LAYER_COUNT &&
 27542:                 entry.expert_id < 256u) {
 27543:                 cuda_moe_tiering_demote_cache_entry(
 27544:                     entry.layer_index, entry.expert_id);
 27545:                 old_map_index = (int32_t)(
 27546:                     entry.layer_index * 256u + entry.expert_id);
 27547:                 if ((size_t)old_map_index <
 27548:                         cache->host_slot_by_layer_expert.size() &&
 27549:                     cache->host_slot_by_layer_expert[(size_t)old_map_index] ==
 27550:                         (int32_t)cache_slot) {
 27551:                     cache->host_slot_by_layer_expert[(size_t)old_map_index] = -1;
 27552:                 }
 27553:             }
 27554:             entry.state = CUDA_MOE_CACHE_LOADING;
 27555:             const uint64_t gate_dst =
 27556:                 (uint64_t)cache_slot * cache->gate_expert_bytes;
 27557:             const uint64_t down_dst =
 27558:                 (uint64_t)cache_slot * cache->down_expert_bytes;
 27559:             const uint64_t resident_pitch =
 27560:                 (uint64_t)cache->capacity * cache->gate_expert_bytes;
 27561:             const uint64_t source_pitch =
 27562:                 (from_nested || (!have_ram && !from_snapshot)) ?
 27563:                     (uint64_t)CUDA_MOE_ROUTE_COUNT *
 27564:                         cache->gate_expert_bytes :
 27565:                     cache->gate_expert_bytes;
 27566:             const double h2d_enqueue_started =
 27567:                 from_nested && g_nested_residual.profile_enabled ?
 27568:                 cuda_wall_sec() : 0.0;
 27569:             const int copy_ok =
 27570:                 from_nested && cuda_nested_residual_gpu_join_requested() ?
 27571:                 cuda_nested_residual_join_to_device_exact(
 27572:                     request.layer_index, expert,
 27573:                     request.gate_offset, request.up_offset,
 27574:                     request.down_offset,
 27575:                     cache->gate_expert_bytes, cache->down_expert_bytes,
 27576:                     cache->gate + gate_dst,
 27577:                     cache->up + gate_dst,
 27578:                     cache->down + down_dst,
 27579:                     cache->route_upload_stream) == 1 :
 27580:                 (cache->packed_copy_requested ?
 27581:                     cuda_moe_route_copy_expert_h2d_async(
 27582:                         cache,
 27583:                         cache->gate + gate_dst,
 27584:                         cache->up + gate_dst,
 27585:                         cache->down + down_dst,
 27586:                         resident_pitch,
 27587:                         host_gate, host_up, host_down,
 27588:                         source_pitch) :
 27589:                     cuda_moe_route_legacy_copy_expert_h2d_async(
 27590:                         cache,
 27591:                         cache->gate + gate_dst,
 27592:                         cache->up + gate_dst,
 27593:                         cache->down + down_dst,
 27594:                         host_gate, host_up, host_down));
 27595:             if (from_nested && g_nested_residual.profile_enabled) {
 27596:                 g_nested_residual.profile_h2d_enqueue_calls++;
 27597:                 g_nested_residual.profile_h2d_enqueue_seconds +=
 27598:                     cuda_wall_sec() - h2d_enqueue_started;
 27599:             }
 27600:             if (!copy_ok) {
 27601:                 if (from_nested) g_nested_residual.gpu_cache_failures++;
 27602:                 failure_reason = "resident-copy";
 27603:                 ok = 0;
 27604:                 break;
 27605:             }
 27606:             if (!cuda_q1_0_ssd_wrap_record_h2d(
 27607:                     ssd_wrap_h2d_ring, cache->route_upload_stream)) {
 27608:                 failure_reason = "ssd-wrap-h2d-event";
 27609:                 ok = 0;
 27610:                 break;
 27611:             }
 27612:             if (from_nested && cuda_nested_residual_gpu_join_requested()) {
 27613:                 g_nested_residual.route_h2d_bytes +=
 27614:                     cache->gate_expert_bytes * 2ull +
 27615:                     cache->down_expert_bytes;
 27616:             }
 27617:             const uint64_t gate_src = request.gate_offset +
 27618:                 (uint64_t)expert * cache->gate_expert_bytes;
 27619:             const uint64_t up_src = request.up_offset +
 27620:                 (uint64_t)expert * cache->gate_expert_bytes;
 27621:             const uint64_t down_src = request.down_offset +
 27622:                 (uint64_t)expert * cache->down_expert_bytes;
 27623:             const uint32_t new_map_index =
 27624:                 request.layer_index * 256u + expert;
 27625:             moe_publish_resident_route_kernel<<<
 27626:                     1, 1, 0, cache->route_upload_stream>>>(
 27627:                 cache->device_slot_by_layer_expert,
 27628:                 cache->device_route_ptrs,
 27629:                 old_map_index,
 27630:                 new_map_index,
 27631:                 cache_slot,
 27632:                 request.route_count,
 27633:                 route,
 27634:                 cache->gate,
 27635:                 cache->up,
 27636:                 cache->down,
 27637:                 cache->gate_expert_bytes,
 27638:                 cache->down_expert_bytes);
 27639:             if (cudaGetLastError() != cudaSuccess) {
 27640:                 (void)cudaGetLastError();
 27641:                 failure_reason = "resident-publish";
 27642:                 ok = 0;
 27643:                 break;
 27644:             }
 27645:             entry.key = {gate_src, up_src, down_src};
 27646:             entry.layer_index = request.layer_index;
 27647:             entry.expert_id = expert;
 27648:             entry.age = ++cache->tick;
 27649:             entry.layer_owned = 0;
 27650:             entry.layer_key = request.gate_offset;
 27651:             cache->host_slot_by_layer_expert[new_map_index] =
 27652:                 (int32_t)cache_slot;
 27653:             if (!was_valid && cache->count < cache->capacity) cache->count++;
 27654:             else if (was_valid) cache->evictions++;
 27655:             cache->admissions++;
 27656:             if (tier.state == CUDA_MOE_TIER_RAM_PROBATION) {
 27657:                 g_moe_tiering.ram_to_warm++;
 27658:             }
 27659:             if (from_snapshot) {
 27660:                 g_moe_tiering.snapshot_to_vram_bytes +=
 27661:                     cache->gate_expert_bytes * 2ull +
 27662:                     cache->down_expert_bytes;
 27663:             } else if (!have_ram) {
 27664:                 g_moe_tiering.forbidden_cold_ssd_to_vram++;
 27665:             }
 27666:             tier.state = CUDA_MOE_TIER_VRAM_PROTECTED;
 27667:             g_moe_tiering.vram_promotions++;
 27668:             g_moe_tiering.ram_h2d_bytes +=
 27669:                 cache->gate_expert_bytes * 2ull + cache->down_expert_bytes;
 27670:         } else {
 27671:             if (from_snapshot) {
 27672:                 g_moe_tiering.snapshot_to_vram_bytes +=
 27673:                     cache->gate_expert_bytes * 2ull +
 27674:                     cache->down_expert_bytes;
 27675:             }
 27676:             const uint64_t gate_dst =
 27677:                 (uint64_t)route * cache->gate_expert_bytes;
 27678:             const uint64_t down_dst =
 27679:                 (uint64_t)route * cache->down_expert_bytes;
 27680:             const uint64_t transient_pitch =
 27681:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * cache->gate_expert_bytes;
 27682:             const uint64_t source_pitch =
 27683:                 (from_nested || (!have_ram && !from_snapshot)) ?
 27684:                     transient_pitch : cache->gate_expert_bytes;
 27685:             const double h2d_enqueue_started =
 27686:                 from_nested && g_nested_residual.profile_enabled ?
 27687:                 cuda_wall_sec() : 0.0;
 27688:             const int copy_ok =
 27689:                 from_nested && cuda_nested_residual_gpu_join_requested() ?
 27690:                 cuda_nested_residual_join_to_device_exact(
 27691:                     request.layer_index, expert,
 27692:                     request.gate_offset, request.up_offset,
 27693:                     request.down_offset,
 27694:                     cache->gate_expert_bytes, cache->down_expert_bytes,
 27695:                     cache->route_transient_gate + gate_dst,
 27696:                     cache->route_transient_up + gate_dst,
 27697:                     cache->route_transient_down + down_dst,
 27698:                     cache->route_upload_stream) == 1 :
 27699:                 (cache->packed_copy_requested ?
 27700:                     cuda_moe_route_copy_expert_h2d_async(
 27701:                         cache,
 27702:                         cache->route_transient_gate + gate_dst,
 27703:                         cache->route_transient_up + gate_dst,
 27704:                         cache->route_transient_down + down_dst,
 27705:                         transient_pitch,
 27706:                         host_gate, host_up, host_down,
 27707:                         source_pitch) :
 27708:                     cuda_moe_route_legacy_copy_expert_h2d_async(
 27709:                         cache,
 27710:                         cache->route_transient_gate + gate_dst,
 27711:                         cache->route_transient_up + gate_dst,
 27712:                         cache->route_transient_down + down_dst,
 27713:                         host_gate, host_up, host_down));
 27714:             if (from_nested && g_nested_residual.profile_enabled) {
 27715:                 g_nested_residual.profile_h2d_enqueue_calls++;
 27716:                 g_nested_residual.profile_h2d_enqueue_seconds +=
 27717:                     cuda_wall_sec() - h2d_enqueue_started;
 27718:             }
 27719:             if (!copy_ok) {
 27720:                 if (from_nested) g_nested_residual.gpu_cache_failures++;
 27721:                 failure_reason = "transient-copy";
 27722:                 ok = 0;
 27723:                 break;
 27724:             }
 27725:             if (!cuda_q1_0_ssd_wrap_record_h2d(
 27726:                     ssd_wrap_h2d_ring, cache->route_upload_stream)) {
 27727:                 failure_reason = "ssd-wrap-h2d-event";
 27728:                 ok = 0;
 27729:                 break;
 27730:             }
 27731:             if (from_nested && cuda_nested_residual_gpu_join_requested()) {
 27732:                 g_nested_residual.route_h2d_bytes +=
 27733:                     cache->gate_expert_bytes * 2ull +
 27734:                     cache->down_expert_bytes;
 27735:             }
 27736:             moe_publish_transient_route_kernel<<<
 27737:                     1, 1, 0, cache->route_upload_stream>>>(
 27738:                 cache->device_route_ptrs,
 27739:                 request.route_count,
 27740:                 route,
 27741:                 route,
 27742:                 cache->route_transient_gate,
 27743:                 cache->route_transient_up,
 27744:                 cache->route_transient_down,
 27745:                 cache->gate_expert_bytes,
 27746:                 cache->down_expert_bytes);
 27747:             if (cudaGetLastError() != cudaSuccess) {
 27748:                 (void)cudaGetLastError();
 27749:                 failure_reason = "transient-publish";
 27750:                 ok = 0;
 27751:                 break;
 27752:             }
 27753:             g_moe_tiering.transient++;
 27754:             g_moe_tiering.ram_h2d_bytes +=
 27755:                 cache->gate_expert_bytes * 2ull + cache->down_expert_bytes;
 27756:         }
 27757:         if (from_nested && !cuda_nested_residual_gpu_join_requested()) {
 27758:             const uint64_t native_bytes =
 27759:                 cache->gate_expert_bytes * 2ull + cache->down_expert_bytes;
 27760:             g_nested_residual.h2d_bytes += native_bytes;
 27761:             g_nested_residual.route_h2d_bytes += native_bytes;
 27762:         }
 27763:     }
 27764:     const double h2d_sync_started =
 27765:         g_nested_residual.profile_enabled &&
 27766:         cuda_nested_residual_gpu_cache_requested() &&
 27767:         cuda_nested_residual_layer_required(request.layer_index) ?
 27768:         cuda_wall_sec() : 0.0;
 27769:     const cudaError_t pending_error = cudaGetLastError();
 27770:     const cudaError_t sync_error =
 27771:         cudaStreamSynchronize(cache->route_upload_stream);
 27772:     if (h2d_sync_started != 0.0) {
 27773:         g_nested_residual.profile_h2d_sync_calls++;
 27774:         g_nested_residual.profile_h2d_sync_seconds +=
 27775:             cuda_wall_sec() - h2d_sync_started;
 27990:             }
 27991:         }
 27992: 
 27993:         uint32_t admitted[CUDA_MOE_ROUTE_COUNT] = {0};
 27994:         uint32_t admitted_count = 0;
 27995:         if (ok && g_moe_tiering.mode == CUDA_MOE_TIER_ENFORCE) {
 27996:             ok = cuda_moe_tiering_enforce_request(
 27997:                 cache, request, claimed.data());
 27998:         }
 27999:         for (uint32_t miss = 0;
 28000:              ok && g_moe_tiering.mode != CUDA_MOE_TIER_ENFORCE &&
 28001:              miss < request.miss_count; miss++) {
 28002:             const uint32_t route = request.miss_routes[miss];
 28003:             if (route >= request.route_count) {
 28004:                 ok = 0;
 28005:                 failure_reason = "miss-route-oob";
 28006:                 break;
 28007:             }
 28008:             const int32_t expert_i = request.selected[route];
 28009:             if (expert_i < 0 || expert_i >= 256) {
 28010:                 ok = 0;
 28011:                 failure_reason = "expert-oob";
 28012:                 break;
 28013:             }
 28014:             const uint32_t expert = (uint32_t)expert_i;
 28015:             if (g_sparse_bake_active &&
 28016:                 !cuda_sparse_bake_expert_retained(request.layer_index, expert)) {
 28017:                 g_sparse_bake_route_rejections++;
 28018:                 fprintf(stderr,
 28019:                         "ds4: sparse bake rejected route worker miss "
 28020:                         "layer=%u seq=%u route=%u expert=%u\n",
 28021:                         request.layer_index, sequence, route, expert);
 28022:                 ok = 0;
 28023:                 failure_reason = "sparse-rejected";
 28024:                 break;
 28025:             }
 28026:             const int cache_slot_i = cuda_moe_route_worker_pick_slot(cache, claimed.data());
 28027:             if (cache_slot_i < 0) {
 28028:                 ok = 0;
 28029:                 failure_reason = "no-cache-slot";
 28030:                 break;
 28031:             }
 28032:             const uint32_t cache_slot = (uint32_t)cache_slot_i;
 28033:             claimed[cache_slot] = 1u;
 28034:             cuda_moe_cache_slot &entry = cache->slots[cache_slot];
 28035:             admitted[admitted_count++] = cache_slot;
 28036:             int32_t old_map_index = -1;
 28037:             const int was_valid = entry.state == CUDA_MOE_CACHE_VALID;
 28038:             if (was_valid && entry.layer_index < CUDA_MOE_LAYER_COUNT &&
 28039:                 entry.expert_id < 256u) {
 28040:                 cuda_moe_tiering_demote_cache_entry(
 28041:                     entry.layer_index, entry.expert_id);
 28042:                 old_map_index = (int32_t)(entry.layer_index * 256u + entry.expert_id);
 28043:                 if ((size_t)old_map_index < cache->host_slot_by_layer_expert.size() &&
 28044:                     cache->host_slot_by_layer_expert[(size_t)old_map_index] ==
 28045:                         (int32_t)cache_slot) {
 28046:                     cache->host_slot_by_layer_expert[(size_t)old_map_index] = -1;
 28047:                 }
 28048:             }
 28049:             entry.state = CUDA_MOE_CACHE_LOADING;
 28050: 
 28051:             char *host_gate = cache->route_host_gate +
 28052:                 (uint64_t)miss * cache->gate_expert_bytes;
 28053:             char *host_up = cache->route_host_up +
 28054:                 (uint64_t)miss * cache->gate_expert_bytes;
 28055:             char *host_down = cache->route_host_down +
 28056:                 (uint64_t)miss * cache->down_expert_bytes;
 28057:             const uint64_t gate_src = request.gate_offset +
 28058:                 (uint64_t)expert * cache->gate_expert_bytes;
 28059:             const uint64_t up_src = request.up_offset +
 28060:                 (uint64_t)expert * cache->gate_expert_bytes;
 28061:             const uint64_t down_src = request.down_offset +
 28062:                 (uint64_t)expert * cache->down_expert_bytes;
 28063:             int from_nested = 0;
 28064:             if (cuda_nested_residual_gpu_cache_requested() &&
 28065:                 cuda_nested_residual_layer_required(request.layer_index)) {
 28066:                 if (cuda_nested_residual_gpu_join_requested()) {
 28067:                     from_nested = 1;
 28068:                 } else {
 28069:                     from_nested = cuda_nested_residual_fill_route_host_exact(
 28070:                         request.layer_index, expert,
 28071:                         request.gate_offset, request.up_offset,
 28072:                         request.down_offset,
 28073:                         cache->gate_expert_bytes, cache->down_expert_bytes,
 28074:                         host_gate, host_up, host_down);
 28075:                     if (from_nested != 1) {
 28076:                         ok = 0;
 28077:                         failure_reason = "nested-exact";
 28078:                         break;
 28079:                     }
 28080:                 }
 28081:             } else if (!g_model_file_valid ||
 28082:                        !cuda_pread_full(&g_model_file, host_gate,
 28083:                                         cache->gate_expert_bytes, gate_src) ||
 28084:                        !cuda_pread_full(&g_model_file, host_up,
 28085:                                         cache->gate_expert_bytes, up_src) ||
 28086:                        !cuda_pread_full(&g_model_file, host_down,
 28087:                                         cache->down_expert_bytes, down_src)) {
 28088:                     ok = 0;
 28089:                     failure_reason = "pread";
 28090:                     break;
 28091:             }
 28092:             const uint64_t gate_dst = (uint64_t)cache_slot * cache->gate_expert_bytes;
 28093:             const uint64_t down_dst = (uint64_t)cache_slot * cache->down_expert_bytes;
 28094:             const uint64_t resident_pitch =
 28095:                 (uint64_t)cache->capacity * cache->gate_expert_bytes;
 28096:             const uint64_t source_pitch =
 28097:                 (uint64_t)CUDA_MOE_ROUTE_COUNT * cache->gate_expert_bytes;
 28098:             const double h2d_enqueue_started =
 28099:                 from_nested && g_nested_residual.profile_enabled ?
 28100:                 cuda_wall_sec() : 0.0;
 28101:             const int copy_ok =
 28102:                 from_nested && cuda_nested_residual_gpu_join_requested() ?
 28103:                 cuda_nested_residual_join_to_device_exact(
 28104:                     request.layer_index, expert,
 28105:                     request.gate_offset, request.up_offset,
 28106:                     request.down_offset,
 28107:                     cache->gate_expert_bytes, cache->down_expert_bytes,
 28108:                     cache->gate + gate_dst,
 28109:                     cache->up + gate_dst,
 28110:                     cache->down + down_dst,
 28111:                     cache->route_upload_stream) == 1 :
 28112:                 (cache->packed_copy_requested ?
 28113:                     cuda_moe_route_copy_expert_h2d_async(
 28114:                         cache,
 28115:                         cache->gate + gate_dst,
 28116:                         cache->up + gate_dst,
 28117:                         cache->down + down_dst,
 28118:                         resident_pitch,
 28119:                         host_gate, host_up, host_down,
 28120:                         source_pitch) :
 28121:                     cuda_moe_route_legacy_copy_expert_h2d_async(
 28122:                         cache,
 28123:                         cache->gate + gate_dst,
 28124:                         cache->up + gate_dst,
 28125:                         cache->down + down_dst,
 28126:                         host_gate, host_up, host_down));
 28127:             if (from_nested && g_nested_residual.profile_enabled) {
 28128:                 g_nested_residual.profile_h2d_enqueue_calls++;
 28129:                 g_nested_residual.profile_h2d_enqueue_seconds +=
 28130:                     cuda_wall_sec() - h2d_enqueue_started;
 28131:             }
 28132:             if (!copy_ok) {
 28133:                 if (from_nested) g_nested_residual.gpu_cache_failures++;
 28134:                 ok = 0;
 28135:                 failure_reason = "h2d-copy";
 28136:                 failure_cuda_error = cudaPeekAtLastError();
 28137:                 break;
 28138:             }
 28139:             if (from_nested && cuda_nested_residual_gpu_join_requested()) {
 28140:                 g_nested_residual.route_h2d_bytes +=
 28141:                     cache->gate_expert_bytes * 2ull +
 28142:                     cache->down_expert_bytes;
 28143:             }
 28144:             if (from_nested && !cuda_nested_residual_gpu_join_requested()) {
 28145:                 const uint64_t native_bytes =
 28146:                     cache->gate_expert_bytes * 2ull +
 28147:                     cache->down_expert_bytes;
 28148:                 g_nested_residual.h2d_bytes += native_bytes;
 28149:                 g_nested_residual.route_h2d_bytes += native_bytes;
 28150:             }
 28151:             const uint32_t new_map_index = request.layer_index * 256u + expert;
 28152:             moe_publish_resident_route_kernel<<<1, 1, 0, cache->route_upload_stream>>>(
 28153:                 cache->device_slot_by_layer_expert,
 28154:                 cache->device_route_ptrs,
 28155:                 old_map_index,
 28156:                 new_map_index,
 28157:                 cache_slot,
 28158:                 request.route_count,
 28159:                 route,
 28160:                 cache->gate,
 28161:                 cache->up,
 28162:                 cache->down,
 28163:                 cache->gate_expert_bytes,
 28164:                 cache->down_expert_bytes);
 28165:             const cudaError_t publish_error = cudaGetLastError();
 28166:             if (publish_error != cudaSuccess) {
 28167:                 failure_cuda_error = publish_error;
 28168:                 ok = 0;
 28169:                 failure_reason = "publish-launch";
 28170:                 break;
 28171:             }
 28172:             entry.key = {gate_src, up_src, down_src};
 28173:             entry.layer_index = request.layer_index;
 28174:             entry.expert_id = expert;
 28175:             entry.age = ++cache->tick;

2026-07-21T09:51:59.165419Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"prefill_mass|mass_observer|mass_rank|ranked|target_resident|wrap_published|prefill.*publish\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Select-Object -First 260" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 556ms:
2084:struct cuda_prefill_mass_observer {
2090:    int wrap_published;
2108:static cuda_prefill_mass_observer g_prefill_mass_observer;
2115:struct cuda_reap_mass_observer {
2147:static cuda_reap_mass_observer g_reap_mass_observer;
2170:static void cuda_prefill_mass_observer_reset(void);
2171:static void cuda_prefill_mass_observer_finalize(void);
2172:static void cuda_prefill_mass_observer_release(int report);
2174:static int cuda_prefill_mass_observer_needs_weights(void);
2199:static void cuda_reap_mass_observer_reset(void);
2200:static void cuda_reap_mass_observer_release(int report);
2202:static int cuda_reap_mass_observer_needs_weights(void);
2207:static void cuda_prefill_mass_observe_selected(
6230:    cuda_prefill_mass_observer_release(1);
6231:    cuda_reap_mass_observer_release(1);
6509:                "ds4: [q1-0-snapshot] result=bound layers=%u..%u policy=sparse-prefill-ranked iq2_host_snapshot=disabled\n",
7128:        const uint8_t *target_resident,
7138:    if (!target_resident || !out_txn || !out_loads || !out_load_count ||
7148:        if (target_resident[i] && g_sparse_bake_active) {
7158:        target_count += target_resident[i] != 0;
7172:        txn->target.assign(target_resident, target_resident + entry_count);
7210:            if (target_resident[entry]) {
7228:            if (preloaded_staged && target_resident[entry] &&
7243:            if (!target_resident[entry] ||
10697:static int cuda_prefill_mass_wrap_requested(void) {
10711:static int cuda_prefill_mass_observe_requested(void) {
10725:static int cuda_prefill_mass_layer_stripe_config(
10762:static int cuda_prefill_mass_observer_needs_weights(void) {
10763:    return g_prefill_mass_observer.enabled &&
10764:        !g_prefill_mass_observer.finalized;
10854:static int cuda_reap_mass_observer_needs_weights(void) {
10855:    return g_reap_mass_observer.enabled;
10892:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
10910:         g_reap_mass_observer.pending) {
10916:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
10924:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
10965:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
10975:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
10999:    struct ranked_entry {
11003:    std::vector<ranked_entry> entrants;
11004:    std::vector<ranked_entry> victims;
11052:                  [](const ranked_entry &a, const ranked_entry &b) {
11057:                  [](const ranked_entry &a, const ranked_entry &b) {
11093:                    const ranked_entry &entrant = entrants[entrant_cursor];
11094:                    const ranked_entry &victim = victims[victim_cursor];
11171:static void cuda_reap_mass_observer_release(int report) {
11175:    if (report && g_reap_mass_observer.wrap_requested) {
11178:                (unsigned long long)g_reap_mass_observer.wrap_attempts,
11179:                (unsigned long long)g_reap_mass_observer.wrap_publications,
11180:                (unsigned long long)g_reap_mass_observer.wrap_skips,
11181:                (unsigned long long)g_reap_mass_observer.wrap_failures,
11182:                (unsigned long long)g_reap_mass_observer.wrap_entrants,
11183:                (unsigned long long)g_reap_mass_observer.wrap_victims);
11185:    g_reap_mass_observer.totals.clear();
11186:    g_reap_mass_observer.ring.clear();
11187:    g_reap_mass_observer.pending.clear();
11188:    g_reap_mass_observer.touched.clear();
11189:    g_reap_mass_observer.touched_entries.clear();
11190:    g_reap_mass_observer.target.clear();
11191:    g_reap_mass_observer.enabled = 0;
11192:    g_reap_mass_observer.armed = 0;
11193:    g_reap_mass_observer.wrap_requested = 0;
11194:    g_reap_mass_observer.window = 0;
11195:    g_reap_mass_observer.top = 0;
11196:    g_reap_mass_observer.grow_interval = 0;
11197:    g_reap_mass_observer.entry_count = 0;
11198:    g_reap_mass_observer.layer_min = 3;
11199:    g_reap_mass_observer.layer_max = 42;
11200:    g_reap_mass_observer.last_layer = 0;
11201:    g_reap_mass_observer.unique_entries = 0;
11202:    g_reap_mass_observer.last_touched = 0;
11203:    g_reap_mass_observer.tokens = 0;
11204:    g_reap_mass_observer.last_publish_tokens = 0;
11205:    g_reap_mass_observer.observed_slots = 0;
11206:    g_reap_mass_observer.wrap_attempts = 0;
11207:    g_reap_mass_observer.wrap_publications = 0;
11208:    g_reap_mass_observer.wrap_skips = 0;
11209:    g_reap_mass_observer.wrap_failures = 0;
11210:    g_reap_mass_observer.wrap_entrants = 0;
11211:    g_reap_mass_observer.wrap_victims = 0;
11212:    g_reap_mass_observer.top_mass = 0.0;
11213:    g_reap_mass_observer.hysteresis = 0.0;
11216:static void cuda_reap_mass_observer_reset(void) {
11217:    cuda_reap_mass_observer_release(1);
11242:        g_reap_mass_observer.totals.assign(entry_count, 0.0);
11243:        g_reap_mass_observer.ring.assign(
11245:        g_reap_mass_observer.pending.clear();
11246:        g_reap_mass_observer.touched.assign(entry_count, 0);
11247:        g_reap_mass_observer.touched_entries.reserve(entry_count);
11248:        if (wrap) g_reap_mass_observer.target.assign(entry_count, 0);
11250:        cuda_reap_mass_observer_release(0);
11255:    g_reap_mass_observer.enabled = 1;
11256:    g_reap_mass_observer.wrap_requested = wrap;
11257:    g_reap_mass_observer.window = window;
11258:    g_reap_mass_observer.grow_interval = grow_interval;
11259:    g_reap_mass_observer.hysteresis = hysteresis;
11260:    g_reap_mass_observer.entry_count = entry_count;
11261:    g_reap_mass_observer.layer_min = 3;
11262:    g_reap_mass_observer.layer_max = layer_max;
11267:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
11274:        cuda_reap_mass_observer_release(0);
11291:static void cuda_prefill_mass_observer_report_decode(const char *reason) {
11292:    const cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
11358:static void cuda_prefill_mass_compose_release_router_mask(const char *where) {
11374:static void cuda_prefill_mass_observer_release(int report) {
11375:    if (report) cuda_prefill_mass_observer_report_decode("request-end");
11376:    if (g_prefill_mass_observer.compose_mask_applied) {
11377:        cuda_prefill_mass_compose_release_router_mask("request-end");
11379:    g_prefill_mass_observer.mass.clear();
11380:    g_prefill_mass_observer.counts.clear();
11381:    g_prefill_mass_observer.rows_by_layer.clear();
11382:    g_prefill_mass_observer.layer_seen.clear();
11383:    g_prefill_mass_observer.candidate.clear();
11384:    g_prefill_mass_observer.enabled = 0;
11385:    g_prefill_mass_observer.finalized = 0;
11386:    g_prefill_mass_observer.have_decode_layer = 0;
11387:    g_prefill_mass_observer.wrap_requested = 0;
11388:    g_prefill_mass_observer.wrap_attempted = 0;
11389:    g_prefill_mass_observer.wrap_published = 0;
11390:    g_prefill_mass_observer.compose_mask_applied = 0;
11391:    g_prefill_mass_observer.compose_router_open = 0;
11392:    g_prefill_mass_observer.unique_entries = 0;
11393:    g_prefill_mass_observer.layers_seen = 0;
11394:    g_prefill_mass_observer.candidate_entries = 0;
11395:    g_prefill_mass_observer.last_decode_layer = 0;
11396:    g_prefill_mass_observer.decode_tokens = 0;
11397:    g_prefill_mass_observer.routed_slots = 0;
11398:    g_prefill_mass_observer.decode_slots = 0;
11399:    g_prefill_mass_observer.decode_candidate_hits = 0;
11402:static void cuda_prefill_mass_compose_fail_closed(void) {
11405:    if (g_prefill_mass_observer.compose_mask_applied) {
11406:        cuda_prefill_mass_compose_release_router_mask("fail-closed");
11407:        g_prefill_mass_observer.compose_mask_applied = 0;
11419:static int cuda_prefill_mass_compose_apply_router_mask(void) {
11420:    cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
11478:                cuda_prefill_mass_compose_release_router_mask(
11492:            cuda_prefill_mass_compose_release_router_mask("apply-failure");
11510:static void cuda_prefill_mass_observer_reset(void) {
11511:    cuda_prefill_mass_observer_release(1);
11514:                "ds4: [q1-0-resident-arena] prefill_mass=disabled snapshot=full-bootstrap router=unchanged dynamic_masks=not-implemented\n");
11517:    const int wrap = cuda_prefill_mass_wrap_requested();
11518:    const int observe = cuda_prefill_mass_observe_requested();
11531:        g_prefill_mass_observer.mass.assign(entry_count, 0.0);
11532:        g_prefill_mass_observer.counts.assign(entry_count, 0);
11533:        g_prefill_mass_observer.rows_by_layer.assign(
11535:        g_prefill_mass_observer.layer_seen.assign(
11537:        g_prefill_mass_observer.candidate.assign(entry_count, 0);
11539:        cuda_prefill_mass_observer_release(0);
11544:    g_prefill_mass_observer.enabled = 1;
11545:    g_prefill_mass_observer.wrap_requested = wrap;
11550:            wrap ? "prefill-ranked" : "unchanged",
11554:static int cuda_prefill_mass_publish_candidate(void) {
11555:    cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
11632:                cuda_prefill_mass_compose_fail_closed();
11633:            } else if (cuda_prefill_mass_compose_apply_router_mask()) {
11636:                observer.wrap_published = 1;
11641:                cuda_prefill_mass_compose_fail_closed();
11654:    if (observer.wrap_published &&
11663:        observer.wrap_published = 0;
11665:        cuda_prefill_mass_compose_fail_closed();
11666:    } else if (!observer.wrap_published &&
11672:        cuda_prefill_mass_compose_fail_closed();
11685:    return observer.wrap_published;
11688:static void cuda_prefill_mass_observer_finalize(void) {
11689:    cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
11698:            (void)cuda_prefill_mass_publish_candidate();
11704:    struct ranked_entry {
11708:    std::vector<ranked_entry> ranked;
11710:        ranked.reserve(observer.unique_entries);
11713:            ranked.push_back({observer.mass[entry], entry});
11726:        cuda_prefill_mass_observer_release(0);
11730:    std::sort(ranked.begin(), ranked.end(),
11731:              [](const ranked_entry &a, const ranked_entry &b) {
11737:        g_reap_mass_observer.wrap_requested ?
11764:        cuda_prefill_mass_compose_fail_closed();
11777:        cuda_prefill_mass_compose_fail_closed();
11799:    const size_t ranked_capacity = residency_capacity - hash_entries;
11806:    uint32_t sparse_skipped_ranked = 0;
11807:    uint32_t nested_skipped_ranked = 0;
11810:    const int stripe_config = cuda_prefill_mass_layer_stripe_config(
11817:        cuda_prefill_mass_compose_fail_closed();
11825:        cuda_prefill_mass_compose_fail_closed();
11831:        ranked.size(), ranked_capacity);
11837:        if (ranked_capacity >
11839:            ranked_capacity > UINT32_MAX || routed_layers == 0) {
11843:                    (unsigned long long)ranked_capacity);
11844:            cuda_prefill_mass_compose_fail_closed();
11859:            cuda_prefill_mass_compose_fail_closed();
11877:        if (full_layers == 0 || full_entries > ranked_capacity ||
11878:            (partial_layers == 0 && full_entries != ranked_capacity)) {
11883:                    (unsigned long long)ranked_capacity);
11884:            cuda_prefill_mass_compose_fail_closed();
11890:            (uint32_t)(ranked_capacity - full_entries);
11904:            cuda_prefill_mass_compose_fail_closed();
11926:        for (const ranked_entry &item : ranked) {
11952:                cuda_prefill_mass_compose_fail_closed();
11957:        capacity = (uint32_t)ranked_capacity;
11967:        for (const ranked_entry &item : ranked) {
11968:            if ((size_t)capacity >= ranked_capacity) break;
11972:                nested_skipped_ranked++;
11976:                sparse_skipped_ranked++;
11984:            (size_t)capacity != ranked_capacity) {
11992:            cuda_prefill_mass_compose_fail_closed();
12006:            cuda_prefill_mass_compose_fail_closed();
12017:    for (const ranked_entry &item : ranked) {
12030:                "ds4: [prefill-mass-compose] hash_layers=%u hash_seed_entries=%u ranked_entries=%u total_candidate=%u capacity=%u candidate_fnv1a64=%016llx sparse_skipped_ranked=%u nested_skipped_ranked=%u mass_source=full-probability-normalized-per-token\n",
12035:                sparse_skipped_ranked, nested_skipped_ranked);
12055:            wrap ? "prefill-ranked" : "unchanged",
12057:    if (wrap) (void)cuda_prefill_mass_publish_candidate();
12061:static void cuda_prefill_mass_observe_selected(
12066:    cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
12082:            cuda_prefill_mass_observer_release(0);
12139:            cuda_prefill_mass_observer_report_decode("checkpoint");
12160:    cuda_reap_mass_observer &observer = g_reap_mass_observer;
12170:        cuda_reap_mass_observer_release(0);
12176:        cuda_reap_mass_observer_release(0);
12188:        cuda_reap_mass_observer_release(0);
12198:            cuda_reap_mass_observer_release(0);
12241:            cuda_reap_mass_observer_release(0);
12292:    cuda_prefill_mass_observer_reset();
12294:        cuda_reap_mass_observer_release(1);
12296:        cuda_reap_mass_observer_reset();
12324:        cuda_prefill_mass_observer_release(1);
12326:        cuda_reap_mass_observer_release(1);
12329:    cuda_prefill_mass_observer_finalize();
12335:    if (g_reap_mass_observer.wrap_requested) return;
19358:        g_reap_mass_observer.enabled ? g_reap_router_trace_device : NULL;
22610:    int compose_prefill_mass_tiering;
22678:        g_moe_tiering.compose_prefill_mass_tiering &&
23250:        if (g_moe_tiering.compose_prefill_mass_tiering) {
23252:                "ds4: [expert-tiering] final mode=%s policy=%s compose_prefill_mass_tiering=%u compose_router_open=%u snapshot_generation=%llu snapshot_backing_entries=%u snapshot_backing_hits=%llu snapshot_backing_misses=%llu snapshot_to_vram_bytes=%llu forbidden_cold_ssd_to_vram=%llu general_backing_reclaims=%llu clock_calls=%u replacement_budget=%u replacement_budget_base=%u adaptive_budget=%u adaptive_current_budget=%u adaptive_min=%u adaptive_max=%u adaptive_step=%u adaptive_pressure_threshold=%u adaptive_ups=%llu adaptive_downs=%llu adaptive_pressure_epochs=%llu adaptive_quiet_epochs=%llu adaptive_last_budget_skips_delta=%llu adaptive_last_replacements_delta=%llu min_frequency=%u hysteresis=%.9g calls=%llu selected=%llu cold=%llu ram_hits=%llu vram_hits=%llu cold_to_ram=%llu cold_to_vram=%llu ram_to_warm=%llu vram_promotions=%llu vram_demotions=%llu ram_evictions=%llu ram_admit_skips=%llu transient=%llu failures=%llu ssd_bytes=%llu ram_h2d_bytes=%llu policy_epochs=%llu policy_free_promotions=%llu policy_replacements=%llu policy_min_frequency_skips=%llu policy_budget_skips=%llu policy_score_skips=%llu states_ssd=%u states_probation=%u states_warm=%u states_vram=%u mass_sum=%.9g lfru_top=%.9g\n",
23255:                g_moe_tiering.compose_prefill_mass_tiering ? 1u : 0u,
23427:        g_moe_tiering.compose_prefill_mass_tiering) {
23434:    if (!g_moe_tiering.compose_prefill_mass_tiering ||
23654:        g_moe_tiering.compose_prefill_mass_tiering == compose_requested &&
23683:                !g_prefill_mass_observer.enabled ||
23684:                !g_prefill_mass_observer.finalized ||
23685:                !g_prefill_mass_observer.wrap_published ||
23686:                g_reap_mass_observer.enabled) {
23688:                        "ds4: prefill/tiering compose requires finalized published prefill snapshot and no dynamic/REAP observer\n");
23693:                   g_prefill_mass_observer.enabled ||
23694:                   g_reap_mass_observer.enabled) {
23713:    g_moe_tiering.compose_prefill_mass_tiering = compose_requested;
23779:    fprintf(stderr, "ds4: expert tiering active mode=%s policy=%s compose_prefill_mass_tiering=%u compose_router_open=%u snapshot_generation=%llu snapshot_backing_entries=%u clock_calls=%u replacement_budget=%u replacement_budget_base=%u adaptive_budget=%u adaptive_current_budget=%u adaptive_min=%u adaptive_max=%u adaptive_step=%u adaptive_pressure_threshold=%u adaptive_ups=%llu adaptive_downs=%llu adaptive_pressure_epochs=%llu adaptive_quiet_epochs=%llu adaptive_last_budget_skips_delta=%llu adaptive_last_replacements_delta=%llu min_frequency=%u hysteresis=%.9g entries=%u ram_slots=%u\n",
23782:            g_moe_tiering.compose_prefill_mass_tiering ? 1u : 0u,
23811:    if (!g_moe_tiering.compose_prefill_mass_tiering ||
23888:        g_moe_tiering.compose_prefill_mass_tiering &&
24972:        g_prefill_mass_observer.enabled &&
24973:        !g_prefill_mass_observer.finalized ? 1u : 0u;
25596:        !g_moe_tiering.compose_prefill_mass_tiering ||
25830:        !g_moe_tiering.compose_prefill_mass_tiering ||
26260:    std::vector<cuda_moe_prefill_vram_seed_entry> global_ranked;
26269:        !g_moe_tiering.compose_prefill_mass_tiering) {
26273:    if (!g_prefill_mass_observer.enabled ||
26274:        !g_prefill_mass_observer.finalized ||
26275:        !g_prefill_mass_observer.wrap_published) {
26315:    if (g_prefill_mass_observer.mass.size() !=
26317:        g_prefill_mass_observer.counts.size() !=
26319:        g_prefill_mass_observer.candidate.size() !=
26331:            global_ranked.reserve((size_t)routed_layers * 256u);

2026-07-21T09:52:04.187979Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 11510 11690; Show-Lines $p 11688 12065; Show-Lines $p 12280 12342; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 11730 11890" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 862ms:
 11510: static void cuda_prefill_mass_observer_reset(void) {
 11511:     cuda_prefill_mass_observer_release(1);
 11512:     if (cuda_q1_0_exclusive_arena_active()) {
 11513:         fprintf(stderr,
 11514:                 "ds4: [q1-0-resident-arena] prefill_mass=disabled snapshot=full-bootstrap router=unchanged dynamic_masks=not-implemented\n");
 11515:         return;
 11516:     }
 11517:     const int wrap = cuda_prefill_mass_wrap_requested();
 11518:     const int observe = cuda_prefill_mass_observe_requested();
 11519:     if (!observe && !wrap) return;
 11520:     if (!g_dynamic_arena.host_base || g_dynamic_arena.n_layer <= 3 ||
 11521:         g_dynamic_arena.n_expert == 0 || g_dynamic_arena.slots.empty()) {
 11522:         fprintf(stderr,
 11523:                 "ds4: [prefill-mass] unavailable reason=dynamic-arena-not-ready policy=%s\n",
 11524:                 wrap ? "bulk-wrap" : "observe-only");
 11525:         return;
 11526:     }
 11527: 
 11528:     const size_t entry_count =
 11529:         (size_t)g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
 11530:     try {
 11531:         g_prefill_mass_observer.mass.assign(entry_count, 0.0);
 11532:         g_prefill_mass_observer.counts.assign(entry_count, 0);
 11533:         g_prefill_mass_observer.rows_by_layer.assign(
 11534:             g_dynamic_arena.n_layer, 0);
 11535:         g_prefill_mass_observer.layer_seen.assign(
 11536:             g_dynamic_arena.n_layer, 0);
 11537:         g_prefill_mass_observer.candidate.assign(entry_count, 0);
 11538:     } catch (...) {
 11539:         cuda_prefill_mass_observer_release(0);
 11540:         fprintf(stderr,
 11541:                 "ds4: [prefill-mass] metadata allocation failed; observer disabled\n");
 11542:         return;
 11543:     }
 11544:     g_prefill_mass_observer.enabled = 1;
 11545:     g_prefill_mass_observer.wrap_requested = wrap;
 11546:     fprintf(stderr,
 11547:             "ds4: [prefill-mass] armed slots=%u layers=3..%u router=unbiased residency=%s policy=%s\n",
 11548:             (uint32_t)g_dynamic_arena.slots.size(),
 11549:             g_dynamic_arena.n_layer - 1u,
 11550:             wrap ? "prefill-ranked" : "unchanged",
 11551:             wrap ? "bulk-wrap" : "observe-only");
 11552: }
 11553: 
 11554: static int cuda_prefill_mass_publish_candidate(void) {
 11555:     cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
 11556:     cuda_request_phase_trace("wrap-enter");
 11557:     observer.wrap_attempted = 1;
 11558:     const uint64_t snapshot_before = g_dynamic_arena.snapshot_generation;
 11559:     const uint32_t resident_before = cuda_dynamic_arena_active_count();
 11560:     const uint32_t entry_count =
 11561:         g_dynamic_arena.n_layer * g_dynamic_arena.n_expert;
 11562:     const char *terminal = "skipped";
 11563:     const char *reason = "no-routing";
 11564:     cuda_dynamic_arena_wrap_result wrap = {};
 11565:     int q1_companion_staged = 0;
 11566: 
 11567:     do {
 11568:         if (!observer.wrap_requested || observer.candidate_entries == 0) {
 11569:             break;
 11570:         }
 11571:         if (!g_dynamic_arena.host_base || g_dynamic_arena.slots.empty() ||
 11572:             g_dynamic_arena.active.size() != entry_count ||
 11573:             g_dynamic_arena.staging.size() != entry_count ||
 11574:             g_dynamic_arena.preloaded.size() != entry_count ||
 11575:             g_dynamic_arena.preloaded_parts.size() != entry_count) {
 11576:             reason = "arena-not-ready";
 11577:             break;
 11578:         }
 11579:         if (g_dynamic_arena.txn || g_dynamic_arena.submissions_blocked ||
 11580:             g_dynamic_arena.hits_disabled) {
 11581:             reason = "transaction-active";
 11582:             break;
 11583:         }
 11584:         if (g_dynamic_arena_observer.enabled ||
 11585:             !g_dynamic_arena_observer.target.empty() ||
 11586:             !g_dynamic_arena_observer.counts.empty()) {
 11587:             reason = "observer-active";
 11588:             break;
 11589:         }
 11590:         int preloads_active = 0;
 11591:         for (uint32_t entry = 0; entry < entry_count; entry++) {
 11592:             if (g_dynamic_arena.preloaded_parts[entry] != 0 ||
 11593:                 g_dynamic_arena.preloaded[entry].slot != UINT32_MAX) {
 11594:                 preloads_active = 1;
 11595:                 break;
 11596:             }
 11597:         }
 11598:         if (preloads_active) {
 11599:             reason = "preloads-active";
 11600:             break;
 11601:         }
 11602:         uint32_t candidate_count = 0;
 11603:         for (uint32_t entry = 0; entry < observer.candidate.size(); entry++) {
 11604:             candidate_count += observer.candidate[entry] != 0;
 11605:         }
 11606:         if (observer.candidate.size() != entry_count ||
 11607:             candidate_count != observer.candidate_entries ||
 11608:             observer.candidate_entries > g_dynamic_arena.slots.size()) {
 11609:             reason = "invalid-target";
 11610:             break;
 11611:         }
 11612:         if (cuda_q1_0_dual_sparse_companion_requested()) {
 11613:             if (!cuda_q1_0_dual_sparse_stage(
 11614:                     observer.candidate.data(), entry_count)) {
 11615:                 reason = "q1-companion-stage";
 11616:                 break;
 11617:             }
 11618:             q1_companion_staged = 1;
 11619:         }
 11620:         cuda_request_phase_trace("wrap-copy-enter");
 11621:         const int published = cuda_dynamic_arena_wrap_publish_target(
 11622:                 observer.candidate.data(), entry_count, &wrap);
 11623:         cuda_request_phase_trace("wrap-copy-return");
 11624:         if (published) {
 11625:             if (q1_companion_staged &&
 11626:                 !cuda_q1_0_dual_sparse_commit(
 11627:                     observer.candidate.data(), entry_count,
 11628:                     wrap.generation)) {
 11629:                 terminal = "failed";
 11630:                 reason = "q1-companion-commit";
 11631:                 g_dynamic_arena.fatal_errors++;
 11632:                 cuda_prefill_mass_compose_fail_closed();
 11633:             } else if (cuda_prefill_mass_compose_apply_router_mask()) {
 11634:                 terminal = "published";
 11635:                 reason = "ok";
 11636:                 observer.wrap_published = 1;
 11637:             } else {
 11638:                 terminal = "failed";
 11639:                 reason = "compose-mask";
 11640:                 g_dynamic_arena.fatal_errors++;
 11641:                 cuda_prefill_mass_compose_fail_closed();
 11642:             }
 11643:         } else {
 11644:             if (q1_companion_staged) {
 11645:                 cuda_q1_0_dual_sparse_abort("primary-publish");
 11646:             }
 11647:             terminal = wrap.aborted ? "aborted" : "failed";
 11648:             reason = wrap.reason ? wrap.reason : "publish";
 11649:         }
 11650:     } while (0);
 11651: 
 11652:     const uint64_t snapshot_after = g_dynamic_arena.snapshot_generation;
 11653:     const uint32_t resident_after = cuda_dynamic_arena_active_count();
 11654:     if (observer.wrap_published &&
 11655:         (wrap.loads > observer.candidate_entries ||
 11656:          (snapshot_before == 0 &&
 11657:           wrap.loads != observer.candidate_entries) ||
 11658:          resident_after != observer.candidate_entries ||
 11659:          snapshot_after <= snapshot_before ||
 11660:          wrap.generation != snapshot_after)) {
 11661:         terminal = "failed";
 11662:         reason = "publish-invariant";
 11663:         observer.wrap_published = 0;
 11664:         g_dynamic_arena.fatal_errors++;
 11665:         cuda_prefill_mass_compose_fail_closed();
 11666:     } else if (!observer.wrap_published &&
 11667:                (snapshot_after != snapshot_before ||
 11668:                 resident_after != resident_before)) {
 11669:         terminal = "failed";
 11670:         reason = "rollback-invariant";
 11671:         g_dynamic_arena.fatal_errors++;
 11672:         cuda_prefill_mass_compose_fail_closed();
 11673:     }
 11674:     fprintf(stderr,
 11675:             "ds4: [prefill-mass-wrap] result=%s reason=%s candidate=%u loads=%u workers=%u seconds=%.3f snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=%llu preloaded=0 router=unbiased mask=%s\n",
 11676:             terminal, reason, observer.candidate_entries,
 11677:             wrap.loads, wrap.workers, wrap.seconds,
 11678:             (unsigned long long)snapshot_before,
 11679:             (unsigned long long)snapshot_after,
 11680:             resident_before, resident_after,
 11681:             (unsigned long long)wrap.generation,
 11682:             observer.compose_mask_applied ? "request-scoped-closed" :
 11683:             (observer.compose_router_open ? "request-scoped-open" : "off"));
 11684:     cuda_request_phase_trace("wrap-terminal");
 11685:     return observer.wrap_published;
 11686: }
 11687: 
 11688: static void cuda_prefill_mass_observer_finalize(void) {
 11689:     cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
 11690:     if (!observer.enabled || observer.finalized) return;
 11688: static void cuda_prefill_mass_observer_finalize(void) {
 11689:     cuda_prefill_mass_observer &observer = g_prefill_mass_observer;
 11690:     if (!observer.enabled || observer.finalized) return;
 11691:     cuda_request_phase_trace("prefill-finalize-enter");
 11692:     observer.finalized = 1;
 11693:     if (observer.routed_slots == 0 || observer.unique_entries == 0) {
 11694:         fprintf(stderr,
 11695:                 "ds4: [prefill-mass] finalize result=no-prefill-routing policy=%s\n",
 11696:                 observer.wrap_requested ? "bulk-wrap" : "observe-only");
 11697:         if (observer.wrap_requested) {
 11698:             (void)cuda_prefill_mass_publish_candidate();
 11699:         }
 11700:         cuda_request_phase_trace("prefill-finalize-return");
 11701:         return;
 11702:     }
 11703: 
 11704:     struct ranked_entry {
 11705:         double mass;
 11706:         uint32_t entry;
 11707:     };
 11708:     std::vector<ranked_entry> ranked;
 11709:     try {
 11710:         ranked.reserve(observer.unique_entries);
 11711:         for (uint32_t entry = 0; entry < observer.counts.size(); entry++) {
 11712:             if (observer.counts[entry] == 0) continue;
 11713:             ranked.push_back({observer.mass[entry], entry});
 11714:         }
 11715:     } catch (...) {
 11716:         fprintf(stderr,
 11717:                 "ds4: [prefill-mass] rank allocation failed; observer disabled\n");
 11718:         if (observer.wrap_requested) {
 11719:             fprintf(stderr,
 11720:                     "ds4: [prefill-mass-wrap] result=failed reason=allocation candidate=0 loads=0 workers=0 seconds=0.000 snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=0 preloaded=0 router=unbiased mask=off\n",
 11721:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 11722:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 11723:                     cuda_dynamic_arena_active_count(),
 11724:                     cuda_dynamic_arena_active_count());
 11725:         }
 11726:         cuda_prefill_mass_observer_release(0);
 11727:         cuda_request_phase_trace("prefill-finalize-return");
 11728:         return;
 11729:     }
 11730:     std::sort(ranked.begin(), ranked.end(),
 11731:               [](const ranked_entry &a, const ranked_entry &b) {
 11732:                   if (a.mass != b.mass) return a.mass > b.mass;
 11733:                   return a.entry < b.entry;
 11734:               });
 11735: 
 11736:     size_t residency_capacity =
 11737:         g_reap_mass_observer.wrap_requested ?
 11738:         (size_t)cuda_reap_mass_resident_limit() :
 11739:         g_dynamic_arena.slots.size();
 11740:     const int compose_requested = observer.wrap_requested ?
 11741:         cuda_moe_prefill_tier_compose_requested() : 0;
 11742:     const int full_q1_snapshot_requested = observer.wrap_requested &&
 11743:         cuda_q1_0_snapshot_backing_requested();
 11744:     const int compose_router_open = compose_requested > 0 ?
 11745:         cuda_moe_prefill_tier_router_open_requested() : 0;
 11746:     const int promotion_probation_slots = compose_router_open > 0 ?
 11747:         cuda_quant_promotion_probation_slots_requested() : 0;
 11748:     const int configured_reserve_slots = compose_router_open > 0 ?
 11749:         cuda_moe_prefill_tier_reserve_slots_requested() : 0;
 11750:     const int reserved_slots = std::max(
 11751:         promotion_probation_slots, configured_reserve_slots);
 11752:     if (g_nested_residual.active && compose_requested > 0 &&
 11753:         compose_router_open != 1) {
 11754:         fprintf(stderr,
 11755:                 "ds4: [prefill-mass-wrap] result=failed "
 11756:                 "reason=nested-residual-requires-open-router candidate=0 "
 11757:                 "loads=0 workers=0 seconds=0.000 snapshot_before=%llu "
 11758:                 "snapshot_after=%llu resident_before=%u resident_after=%u "
 11759:                 "generation=0 preloaded=0 router=unbiased mask=off\n",
 11760:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11761:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11762:                 cuda_dynamic_arena_active_count(),
 11763:                 cuda_dynamic_arena_active_count());
 11764:         cuda_prefill_mass_compose_fail_closed();
 11765:         cuda_request_phase_trace("prefill-finalize-return");
 11766:         return;
 11767:     }
 11768:     if (compose_router_open < 0 || promotion_probation_slots < 0 ||
 11769:         configured_reserve_slots < 0 || reserved_slots < 0 ||
 11770:         (size_t)reserved_slots >= residency_capacity) {
 11771:         fprintf(stderr,
 11772:                 "ds4: [prefill-mass-wrap] result=failed reason=probation-capacity candidate=0 loads=0 workers=0 seconds=0.000 snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=0 preloaded=0 router=unbiased mask=off\n",
 11773:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11774:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11775:                 cuda_dynamic_arena_active_count(),
 11776:                 cuda_dynamic_arena_active_count());
 11777:         cuda_prefill_mass_compose_fail_closed();
 11778:         cuda_request_phase_trace("prefill-finalize-return");
 11779:         return;
 11780:     }
 11781:     if (reserved_slots > 0) {
 11782:         residency_capacity -= (size_t)reserved_slots;
 11783:     }
 11784:     const uint32_t hash_layers =
 11785:         (compose_requested > 0 || full_q1_snapshot_requested) ?
 11786:         std::min(3u, g_dynamic_arena.n_layer) : 0u;
 11787:     const size_t hash_entries =
 11788:         (size_t)hash_layers * g_dynamic_arena.n_expert;
 11789:     if (hash_entries > residency_capacity) {
 11790:         fprintf(stderr,
 11791:                 "ds4: [prefill-mass-wrap] result=failed reason=compose-hash-capacity candidate=0 loads=0 workers=0 seconds=0.000 snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=0 preloaded=0 router=unbiased mask=off\n",
 11792:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11793:                 (unsigned long long)g_dynamic_arena.snapshot_generation,
 11794:                 cuda_dynamic_arena_active_count(),
 11795:                 cuda_dynamic_arena_active_count());
 11796:         cuda_request_phase_trace("prefill-finalize-return");
 11797:         return;
 11798:     }
 11799:     const size_t ranked_capacity = residency_capacity - hash_entries;
 11800:     const auto sparse_entry_retained = [](uint32_t entry) {
 11801:         if (!g_sparse_bake_active) return 1;
 11802:         const uint32_t layer = entry / g_dynamic_arena.n_expert;
 11803:         const uint32_t expert = entry % g_dynamic_arena.n_expert;
 11804:         return cuda_sparse_bake_expert_retained(layer, expert);
 11805:     };
 11806:     uint32_t sparse_skipped_ranked = 0;
 11807:     uint32_t nested_skipped_ranked = 0;
 11808:     uint32_t stripe_stride = 0;
 11809:     uint32_t stripe_phase = 0;
 11810:     const int stripe_config = cuda_prefill_mass_layer_stripe_config(
 11811:         &stripe_stride, &stripe_phase);
 11812:     if (stripe_config < 0 ||
 11813:         (stripe_config > 0 && compose_requested <= 0)) {
 11814:         fprintf(stderr,
 11815:                 "ds4: [prefill-mass-layer-stripe] result=failed reason=%s\n",
 11816:                 stripe_config < 0 ? "invalid-config" : "compose-required");
 11817:         cuda_prefill_mass_compose_fail_closed();
 11818:         cuda_request_phase_trace("prefill-finalize-return");
 11819:         return;
 11820:     }
 11821:     if (stripe_config > 0 && g_nested_residual.active) {
 11822:         fprintf(stderr,
 11823:                 "ds4: [prefill-mass-layer-stripe] result=failed "
 11824:                 "reason=nested-residual-not-supported\n");
 11825:         cuda_prefill_mass_compose_fail_closed();
 11826:         cuda_request_phase_trace("prefill-finalize-return");
 11827:         return;
 11828:     }
 11829: 
 11830:     uint32_t capacity = (uint32_t)std::min(
 11831:         ranked.size(), ranked_capacity);
 11832:     if (stripe_config > 0) {
 11833:         const uint32_t first_layer = 3u;
 11834:         const uint32_t routed_layers =
 11835:             g_dynamic_arena.n_layer > first_layer ?
 11836:             g_dynamic_arena.n_layer - first_layer : 0u;
 11837:         if (ranked_capacity >
 11838:                 (size_t)routed_layers * g_dynamic_arena.n_expert ||
 11839:             ranked_capacity > UINT32_MAX || routed_layers == 0) {
 11840:             fprintf(stderr,
 11841:                     "ds4: [prefill-mass-layer-stripe] result=failed reason=invalid-capacity stride=%u phase=%u routed_layers=%u capacity=%llu\n",
 11842:                     stripe_stride, stripe_phase, routed_layers,
 11843:                     (unsigned long long)ranked_capacity);
 11844:             cuda_prefill_mass_compose_fail_closed();
 11845:             cuda_request_phase_trace("prefill-finalize-return");
 11846:             return;
 11847:         }
 11848: 
 11849:         std::vector<uint8_t> full_layer;
 11850:         std::vector<uint32_t> quota;
 11851:         std::vector<uint32_t> selected_by_layer;
 11852:         try {
 11853:             full_layer.assign(g_dynamic_arena.n_layer, 0);
 11854:             quota.assign(g_dynamic_arena.n_layer, 0);
 11855:             selected_by_layer.assign(g_dynamic_arena.n_layer, 0);
 11856:         } catch (...) {
 11857:             fprintf(stderr,
 11858:                     "ds4: [prefill-mass-layer-stripe] result=failed reason=metadata-allocation\n");
 11859:             cuda_prefill_mass_compose_fail_closed();
 11860:             cuda_request_phase_trace("prefill-finalize-return");
 11861:             return;
 11862:         }
 11863: 
 11864:         uint32_t full_layers = 0;
 11865:         for (uint32_t layer = first_layer;
 11866:              layer < g_dynamic_arena.n_layer; layer++) {
 11867:             const uint32_t relative = layer - first_layer;
 11868:             if (relative % stripe_stride == stripe_phase) {
 11869:                 full_layer[layer] = 1;
 11870:                 quota[layer] = g_dynamic_arena.n_expert;
 11871:                 full_layers++;
 11872:             }
 11873:         }
 11874:         const uint32_t partial_layers = routed_layers - full_layers;
 11875:         const uint64_t full_entries =
 11876:             (uint64_t)full_layers * g_dynamic_arena.n_expert;
 11877:         if (full_layers == 0 || full_entries > ranked_capacity ||
 11878:             (partial_layers == 0 && full_entries != ranked_capacity)) {
 11879:             fprintf(stderr,
 11880:                     "ds4: [prefill-mass-layer-stripe] result=failed reason=full-layer-capacity stride=%u phase=%u full_layers=%u partial_layers=%u full_entries=%llu capacity=%llu\n",
 11881:                     stripe_stride, stripe_phase, full_layers, partial_layers,
 11882:                     (unsigned long long)full_entries,
 11883:                     (unsigned long long)ranked_capacity);
 11884:             cuda_prefill_mass_compose_fail_closed();
 11885:             cuda_request_phase_trace("prefill-finalize-return");
 11886:             return;
 11887:         }
 11888: 
 11889:         const uint32_t partial_entries =
 11890:             (uint32_t)(ranked_capacity - full_entries);
 11891:         const uint32_t partial_keep_min = partial_layers ?
 11892:             partial_entries / partial_layers : 0u;
 11893:         const uint32_t partial_keep_extra = partial_layers ?
 11894:             partial_entries % partial_layers : 0u;
 11895:         const uint32_t partial_keep_max = partial_keep_min +
 11896:             (partial_keep_extra ? 1u : 0u);
 11897:         if (partial_layers &&
 11898:             (partial_keep_min < 6u ||
 11899:              partial_keep_max > g_dynamic_arena.n_expert)) {
 11900:             fprintf(stderr,
 11901:                     "ds4: [prefill-mass-layer-stripe] result=failed reason=partial-layer-capacity stride=%u phase=%u partial_min=%u partial_max=%u\n",
 11902:                     stripe_stride, stripe_phase,
 11903:                     partial_keep_min, partial_keep_max);
 11904:             cuda_prefill_mass_compose_fail_closed();
 11905:             cuda_request_phase_trace("prefill-finalize-return");
 11906:             return;
 11907:         }
 11908: 
 11909:         uint32_t extra_left = partial_keep_extra;
 11910:         for (uint32_t layer = first_layer;
 11911:              layer < g_dynamic_arena.n_layer; layer++) {
 11912:             if (full_layer[layer]) continue;
 11913:             quota[layer] = partial_keep_min + (extra_left ? 1u : 0u);
 11914:             if (extra_left) extra_left--;
 11915:         }
 11916:         for (uint32_t layer = first_layer;
 11917:              layer < g_dynamic_arena.n_layer; layer++) {
 11918:             if (!full_layer[layer]) continue;
 11919:             const uint32_t base = layer * g_dynamic_arena.n_expert;
 11920:             for (uint32_t expert = 0;
 11921:                  expert < g_dynamic_arena.n_expert; expert++) {
 11922:                 observer.candidate[base + expert] = 1;
 11923:             }
 11924:             selected_by_layer[layer] = g_dynamic_arena.n_expert;
 11925:         }
 11926:         for (const ranked_entry &item : ranked) {
 11927:             const uint32_t layer = item.entry / g_dynamic_arena.n_expert;
 11928:             if (layer < first_layer || layer >= g_dynamic_arena.n_layer ||
 11929:                 full_layer[layer] ||
 11930:                 selected_by_layer[layer] >= quota[layer]) {
 11931:                 continue;
 11932:             }
 11933:             observer.candidate[item.entry] = 1;
 11934:             selected_by_layer[layer]++;
 11935:         }
 11936:         for (uint32_t layer = first_layer;
 11937:              layer < g_dynamic_arena.n_layer; layer++) {
 11938:             if (full_layer[layer]) continue;
 11939:             const uint32_t base = layer * g_dynamic_arena.n_expert;
 11940:             for (uint32_t expert = 0;
 11941:                  expert < g_dynamic_arena.n_expert &&
 11942:                  selected_by_layer[layer] < quota[layer]; expert++) {
 11943:                 const uint32_t entry = base + expert;
 11944:                 if (observer.candidate[entry]) continue;
 11945:                 observer.candidate[entry] = 1;
 11946:                 selected_by_layer[layer]++;
 11947:             }
 11948:             if (selected_by_layer[layer] != quota[layer]) {
 11949:                 fprintf(stderr,
 11950:                         "ds4: [prefill-mass-layer-stripe] result=failed reason=quota-fill layer=%u selected=%u quota=%u\n",
 11951:                         layer, selected_by_layer[layer], quota[layer]);
 11952:                 cuda_prefill_mass_compose_fail_closed();
 11953:                 cuda_request_phase_trace("prefill-finalize-return");
 11954:                 return;
 11955:             }
 11956:         }
 11957:         capacity = (uint32_t)ranked_capacity;
 11958:         fprintf(stderr,
 11959:                 "ds4: [prefill-mass-layer-stripe] result=applied stride=%u phase=%u routed_layers=%u full_layers=%u partial_layers=%u full_keep=%u partial_keep_min=%u partial_keep_max=%u routed_candidate=%u total_candidate=%u capacity=%u semantics=budget-preserving\n",
 11960:                 stripe_stride, stripe_phase, routed_layers,
 11961:                 full_layers, partial_layers, g_dynamic_arena.n_expert,
 11962:                 partial_keep_min, partial_keep_max, capacity,
 11963:                 capacity + (uint32_t)hash_entries,
 11964:                 (uint32_t)residency_capacity);
 11965:     } else {
 11966:         capacity = 0;
 11967:         for (const ranked_entry &item : ranked) {
 11968:             if ((size_t)capacity >= ranked_capacity) break;
 11969:             const uint32_t layer =
 11970:                 item.entry / g_dynamic_arena.n_expert;
 11971:             if (cuda_nested_residual_layer_required(layer)) {
 11972:                 nested_skipped_ranked++;
 11973:                 continue;
 11974:             }
 11975:             if (!sparse_entry_retained(item.entry)) {
 11976:                 sparse_skipped_ranked++;
 11977:                 continue;
 11978:             }
 11979:             if (observer.candidate[item.entry]) continue;
 11980:             observer.candidate[item.entry] = 1;
 11981:             capacity++;
 11982:         }
 11983:         if ((compose_requested > 0 || full_q1_snapshot_requested) &&
 11984:             (size_t)capacity != ranked_capacity) {
 11985:             fprintf(stderr,
 11986:                     "ds4: [prefill-mass-wrap] result=failed reason=compose-retained-capacity candidate=%u loads=0 workers=0 seconds=0.000 snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=0 preloaded=0 router=unbiased mask=off\n",
 11987:                     capacity,
 11988:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 11989:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 11990:                     cuda_dynamic_arena_active_count(),
 11991:                     cuda_dynamic_arena_active_count());
 11992:             cuda_prefill_mass_compose_fail_closed();
 11993:             cuda_request_phase_trace("prefill-finalize-return");
 11994:             return;
 11995:         }
 11996:     }
 11997:     for (uint32_t entry = 0; entry < hash_entries; entry++) {
 11998:         if (!sparse_entry_retained(entry)) {
 11999:             fprintf(stderr,
 12000:                     "ds4: [prefill-mass-wrap] result=failed reason=compose-hash-sparse-absent candidate=%u loads=0 workers=0 seconds=0.000 snapshot_before=%llu snapshot_after=%llu resident_before=%u resident_after=%u generation=0 preloaded=0 router=unbiased mask=off\n",
 12001:                     capacity,
 12002:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 12003:                     (unsigned long long)g_dynamic_arena.snapshot_generation,
 12004:                     cuda_dynamic_arena_active_count(),
 12005:                     cuda_dynamic_arena_active_count());
 12006:             cuda_prefill_mass_compose_fail_closed();
 12007:             cuda_request_phase_trace("prefill-finalize-return");
 12008:             return;
 12009:         }
 12010:         observer.candidate[entry] = 1;
 12011:     }
 12012:     observer.candidate_entries = capacity + (uint32_t)hash_entries;
 12013:     double total_mass = 0.0;
 12014:     double candidate_mass = 0.0;
 12015:     double cutoff = 0.0;
 12016:     int cutoff_set = 0;
 12017:     for (const ranked_entry &item : ranked) {
 12018:         total_mass += item.mass;
 12019:         if (!observer.candidate[item.entry]) continue;
 12020:         candidate_mass += item.mass;
 12021:         if (!cutoff_set || item.mass < cutoff) {
 12022:             cutoff = item.mass;
 12023:             cutoff_set = 1;
 12024:         }
 12025:     }
 12026:     if (compose_requested > 0) {
 12027:         const uint64_t candidate_hash = cuda_dynamic_arena_fnv1a64(
 12028:             observer.candidate.data(), observer.candidate.size());
 12029:         fprintf(stderr,
 12030:                 "ds4: [prefill-mass-compose] hash_layers=%u hash_seed_entries=%u ranked_entries=%u total_candidate=%u capacity=%u candidate_fnv1a64=%016llx sparse_skipped_ranked=%u nested_skipped_ranked=%u mass_source=full-probability-normalized-per-token\n",
 12031:                 hash_layers, (uint32_t)hash_entries, capacity,
 12032:                 observer.candidate_entries,
 12033:                 (uint32_t)residency_capacity,
 12034:                 (unsigned long long)candidate_hash,
 12035:                 sparse_skipped_ranked, nested_skipped_ranked);
 12036:     }
 12037: 
 12038:     uint32_t rows_min = UINT32_MAX;
 12039:     uint32_t rows_max = 0;
 12040:     for (uint32_t layer = 3; layer < observer.rows_by_layer.size(); layer++) {
 12041:         if (!observer.layer_seen[layer]) continue;
 12042:         rows_min = std::min(rows_min, observer.rows_by_layer[layer]);
 12043:         rows_max = std::max(rows_max, observer.rows_by_layer[layer]);
 12044:     }
 12045:     if (rows_min == UINT32_MAX) rows_min = 0;
 12046:     const double coverage = total_mass > 0.0 ? candidate_mass / total_mass : 0.0;
 12047:     const int wrap = observer.wrap_requested;
 12048:     fprintf(stderr,
 12049:             "ds4: [prefill-mass] finalize layers=%u rows_min=%u rows_max=%u routed_slots=%llu unique=%u candidate=%u capacity=%u mass_total=%.6f mass_candidate=%.6f mass_coverage=%.4f cutoff=%.6f router=unbiased residency=%s policy=%s\n",
 12050:             observer.layers_seen, rows_min, rows_max,
 12051:             (unsigned long long)observer.routed_slots,
 12052:             observer.unique_entries, observer.candidate_entries,
 12053:             (uint32_t)g_dynamic_arena.slots.size(), total_mass,
 12054:             candidate_mass, coverage, cutoff,
 12055:             wrap ? "prefill-ranked" : "unchanged",
 12056:             wrap ? "bulk-wrap" : "observe-only");
 12057:     if (wrap) (void)cuda_prefill_mass_publish_candidate();
 12058:     cuda_request_phase_trace("prefill-finalize-return");
 12059: }
 12060: 
 12061: static void cuda_prefill_mass_observe_selected(
 12062:         uint32_t layer_index, uint32_t n_tokens,
 12063:         const int32_t *selected, const float *weights,
 12064:         uint32_t selected_count,
 12065:         const float *full_probs, uint32_t full_prob_count) {
 12280:     return resident;
 12281: }
 12282: 
 12283: extern "C" void ds4_gpu_dynamic_arena_request_begin(void) {
 12284:     /* Drain prior-request SSD work before advancing the authoritative request
 12285:      * epoch. OFF is a true no-op. */
 12286:     cuda_q1_0_ssd_wrap_flush();
 12287:     if (g_cuda_request_epoch != UINT64_MAX) {
 12288:         g_cuda_request_epoch++;
 12289:     }
 12290:     cuda_expert_recovery_trace_request_begin();
 12291:     cuda_moe_tiering_request_boundary_reset();
 12292:     cuda_prefill_mass_observer_reset();
 12293:     if (cuda_q1_0_exclusive_arena_active()) {
 12294:         cuda_reap_mass_observer_release(1);
 12295:     } else {
 12296:         cuda_reap_mass_observer_reset();
 12297:     }
 12298:     const int mode = cuda_dynamic_arena_carry_mode();
 12299:     if (mode < 0 || !g_dynamic_arena.host_base) return;
 12300: 
 12301:     const uint64_t request = ++g_dynamic_arena.request_sequence;
 12302:     if (g_dynamic_arena.snapshot_generation == 0) {
 12303:         g_dynamic_arena.hits_disabled = 0;
 12304:         fprintf(stderr,
 12305:                 "ds4: [arena-carry] request=%llu mode=prime snapshot=0 resident=0 lookup=enabled observer=learning\n",
 12306:                 (unsigned long long)request);
 12307:         return;
 12308:     }
 12309: 
 12310:     cuda_dynamic_arena_observer_release();
 12311:     g_dynamic_arena.hits_disabled = mode == 0;
 12312:     const uint32_t resident = cuda_dynamic_arena_active_count();
 12313:     fprintf(stderr,
 12314:             "ds4: [arena-carry] request=%llu mode=%s snapshot=%llu resident=%u lookup=%s observer=frozen\n",
 12315:             (unsigned long long)request,
 12316:             mode == 0 ? "drop" : "keep",
 12317:             (unsigned long long)g_dynamic_arena.snapshot_generation,
 12318:             resident,
 12319:             mode == 0 ? "disabled" : "enabled");
 12320: }
 12321: 
 12322: extern "C" void ds4_gpu_dynamic_arena_observer_reset(void) {
 12323:     if (cuda_q1_0_exclusive_arena_active()) {
 12324:         cuda_prefill_mass_observer_release(1);
 12325:         cuda_dynamic_arena_observer_release();
 12326:         cuda_reap_mass_observer_release(1);
 12327:         return;
 12328:     }
 12329:     cuda_prefill_mass_observer_finalize();
 12330:     const uint32_t window = cuda_dynamic_arena_observer_window();
 12331:     const uint32_t min_hits = cuda_dynamic_arena_observer_min_hits();
 12332:     const uint32_t grow_interval =
 12333:         cuda_dynamic_arena_observer_grow_interval();
 12334:     cuda_dynamic_arena_observer_release();
 12335:     if (g_reap_mass_observer.wrap_requested) return;
 12336:     if (cuda_dynamic_arena_carry_mode() >= 0 &&
 12337:         g_dynamic_arena.snapshot_generation != 0) {
 12338:         return;
 12339:     }
 12340:     if (window == 0 || !g_dynamic_arena.host_base ||
 12341:         g_dynamic_arena.n_layer <= 3 || g_dynamic_arena.n_expert == 0) {
 12342:         return;
 11730:     DS4_METAL_PROFILE_DECODE_STAGE("ffn_hc_pre");
 11731:     if (ok) {
 11732:         metal_graph_debug_dump_tensor("hc_ffn_pre_mixes", g->hc_mix, mix_hc, il, pos);
 11733:         metal_graph_debug_dump_tensor("hc_ffn_pre_weights", g->hc_pre, DS4_N_HC, il, pos);
 11734:         metal_graph_debug_dump_tensor("hc_ffn_pre_post_weights", g->hc_post, DS4_N_HC, il, pos);
 11735:         metal_graph_debug_dump_tensor("hc_ffn_pre_comb", g->hc_comb, (uint64_t)DS4_N_HC * DS4_N_HC, il, pos);
 11736:     }
 11737:     if (ok) {
 11738:         metal_graph_debug_dump_tensor("hc_ffn_pre", g->ffn_cur, DS4_N_EMBD, il, pos);
 11739:     }
 11740:     if (ok && !fuse_hc_norm) ok = ds4_gpu_rms_norm_weight_tensor(g->ffn_norm, g->ffn_cur,
 11741:                                                                    model->map, model->size,
 11742:                                                                    layer->ffn_norm->abs_offset,
 11743:                                                                    DS4_N_EMBD, DS4_RMS_EPS) != 0;
 11744:     DS4_METAL_PROFILE_DECODE_STAGE("ffn_norm");
 11745:     if (ok) {
 11746:         metal_graph_debug_dump_tensor("ffn_norm", g->ffn_norm, DS4_N_EMBD, il, pos);
 11747:     }
 11748:     if (ok) metal_graph_spex_cpu_schedule(g, il);
 11749:     if (ok) metal_graph_spex_score_next(g, il);
 11750:     const uint64_t gate_row_bytes = routed_expert_row_bytes(route.gate);
 11751:     const uint64_t gate_expert_bytes = expert_mid_dim * gate_row_bytes;
 11752:     const uint64_t down_row_bytes = routed_expert_row_bytes(route.down);
 11753:     const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
 11754:     if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,
 11755:                                                  DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);
 11756:     if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, g->router_weights, g->router_probs,
 11757:                                                  model->map, model->size,
 11758:                                                 il,
 11759:                                                  layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,
 11760:                                                 layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,
 11761:                                                 layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,
 11762:                                                 (uint32_t)token,
 11763:                                                 0,
 11764:                                                 0,
 11765:                                                 layer->ffn_exp_probs_b != NULL,
 11766:                                                 layer->ffn_gate_tid2eid != NULL,
 11767:                                                 g->router_logits) != 0;
 11768:     DS4_METAL_PROFILE_DECODE_STAGE("router");
 11769:     if (ok) {
 11770:         metal_graph_debug_dump_tensor("ffn_moe_logits", g->router_logits, DS4_N_EXPERT, il, pos);
 11771:         metal_graph_debug_dump_tensor("ffn_moe_probs", g->router_probs, DS4_N_EXPERT, il, pos);
 11772:         metal_graph_debug_dump_i32_tensor("ffn_moe_topk", g->router_selected, DS4_N_EXPERT_USED, il, pos);
 11773:         metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);
 11774:     }
 11775:     const bool fuse_shared_gate_up =
 11776:         !g->quality &&
 11777:         getenv("DS4_METAL_DISABLE_SHARED_GATE_UP_SWIGLU_FUSION") == NULL;
 11778:     const char *overlap_shared_full_env = getenv("DS4_CUDA_MOE_OVERLAP_SHARED_FULL");
 11779:     const bool overlap_shared_full_requested =
 11780:         overlap_shared_full_env && overlap_shared_full_env[0] &&
 11781:         strcmp(overlap_shared_full_env, "0") != 0;
 11782:     /* Mixed IQ1 routing builds a physical five-route primary work list in
 11783:      * the CUDA wrapper. Preparing all six routes here would fetch the cold
 11784:      * primary expert before that split and erase the transport saving. */
 11785:     const bool overlap_shared = ok && !iq1_mixed_cold_one &&
 11786:         !q1_0_mixed_cold &&
 11787:         ds4_gpu_routed_moe_prepare_selected(
 11788:         route.model->map, route.model->size,
 11789:         route.gate->abs_offset,
 11790:         route.up->abs_offset,
 11791:         route.down->abs_offset,
 11792:         gate_expert_bytes, down_expert_bytes,
 11793:         g->router_selected, DS4_N_EXPERT_USED, 1) != 0;
 11794:     const bool overlap_shared_full = overlap_shared && overlap_shared_full_requested;
 11795:     ds4_gpu_spex_key spex_key = {
 11796:         g->spex_epoch,
 11797:         g->spex_decode_seq,
 11798:         il ? il - 1u : 0u,
 11799:         il,
 11800:     };
 11801:     if (ok && overlap_shared) {
 11802:         if (fuse_shared_gate_up) {
 11803:             ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(g->shared_gate,
 11804:                                                              g->shared_up,
 11805:                                                              g->shared_mid,
 11806:                                                              model->map,
 11807:                                                              model->size,
 11808:                                                              layer->ffn_gate_shexp->abs_offset,
 11809:                                                              layer->ffn_up_shexp->abs_offset,
 11810:                                                              DS4_N_EMBD,
 11811:                                                              shared_dim,
 11812:                                                              g->ffn_norm) != 0;
 11813:         } else {
 11814:             ok = ds4_gpu_matmul_q8_0_tensor(g->shared_gate, model->map, model->size,
 11815:                                               layer->ffn_gate_shexp->abs_offset,
 11816:                                               DS4_N_EMBD, shared_dim,
 11817:                                               g->ffn_norm, 1) != 0;
 11818:             if (ok) ok = ds4_gpu_matmul_q8_0_tensor(g->shared_up, model->map, model->size,
 11819:                                                       layer->ffn_up_shexp->abs_offset,
 11820:                                                       DS4_N_EMBD, shared_dim,
 11821:                                                       g->ffn_norm, 1) != 0;
 11822:             if (ok) ok = ds4_gpu_swiglu_tensor(g->shared_mid, g->shared_gate, g->shared_up,
 11823:                                                  shared_dim, 0.0f, 1.0f) != 0;
 11824:         }
 11825:         if (ok && overlap_shared_full) {
 11826:             ok = ds4_gpu_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
 11827:                                               layer->ffn_down_shexp->abs_offset,
 11828:                                               shared_dim, DS4_N_EMBD,
 11829:                                               g->shared_mid, 1) != 0;
 11830:         }
 11831:     }
 11832:     if (ok && q1_0_mixed_cold) {
 11833:         const ds4_tensor *q1_gate = g_q1_0_sidecar.gate[il];
 11834:         const ds4_tensor *q1_up = g_q1_0_sidecar.up[il];
 11835:         const ds4_tensor *q1_down = g_q1_0_sidecar.down[il];
 11836:         const uint64_t q1_gate_row_bytes = routed_expert_row_bytes(q1_gate);
 11837:         const uint64_t q1_gate_expert_bytes =
 11838:             expert_mid_dim * q1_gate_row_bytes;
 11839:         const uint64_t q1_down_row_bytes = routed_expert_row_bytes(q1_down);
 11840:         const uint64_t q1_down_expert_bytes =
 11841:             routed_out_dim * q1_down_row_bytes;
 11842:         ok = ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
 11843:                 g->routed_out,
 11844:                 g->routed_gate,
 11845:                 g->routed_up,
 11846:                 g->routed_mid,
 11847:                 g->routed_down,
 11848:                 route.model->map,
 11849:                 route.model->size,
 11850:                 il,
 11851:                 route.gate->abs_offset,
 11852:                 route.up->abs_offset,
 11853:                 route.down->abs_offset,
 11854:                 route.gate->type,
 11855:                 route.down->type,
 11856:                 gate_expert_bytes,
 11857:                 gate_row_bytes,
 11858:                 down_expert_bytes,
 11859:                 down_row_bytes,
 11860:                 g_q1_0_sidecar.model->map,
 11861:                 g_q1_0_sidecar.model->size,
 11862:                 q1_gate->abs_offset,
 11863:                 q1_up->abs_offset,
 11864:                 q1_down->abs_offset,
 11865:                 q1_gate_expert_bytes,
 11866:                 q1_gate_row_bytes,
 11867:                 q1_down_expert_bytes,
 11868:                 q1_down_row_bytes,
 11869:                 (uint32_t)expert_in_dim,
 11870:                 (uint32_t)down_in_dim,
 11871:                 (uint32_t)routed_out_dim,
 11872:                 g->router_selected,
 11873:                 g->router_weights,
 11874:                 DS4_N_EXPERT_USED,
 11875:                 DS4_SWIGLU_CLAMP_EXP,
 11876:                 g->ffn_norm) != 0;
 11877:     } else if (ok && iq1_mixed_cold_one) {
 11878:         const ds4_tensor *iq1_gate = g_iq1_s_sidecar.gate[il];
 11879:         const ds4_tensor *iq1_up = g_iq1_s_sidecar.up[il];
 11880:         const ds4_tensor *iq1_down = g_iq1_s_sidecar.down[il];
 11881:         const uint64_t iq1_gate_row_bytes = routed_expert_row_bytes(iq1_gate);
 11882:         const uint64_t iq1_gate_expert_bytes = expert_mid_dim * iq1_gate_row_bytes;
 11883:         const uint64_t iq1_down_row_bytes = routed_expert_row_bytes(iq1_down);
 11884:         const uint64_t iq1_down_expert_bytes = routed_out_dim * iq1_down_row_bytes;
 11885:         ok = ds4_gpu_routed_moe_mixed_iq1_one_tensor(
 11886:                 g->routed_out,
 11887:                 g->routed_gate,
 11888:                 g->routed_up,
 11889:                 g->routed_mid,
 11890:                 g->routed_down,

2026-07-21T09:52:16.391873Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
codex
The core seam is now clear: the mixed wrapper already performs one route/weight D2H, classifies six routes, launches the exact-IQ2 subset, launches the Q1 subset, and adds the two outputs. The prototype can preserve that control shape while replacing only the Q1 classification/launch with a bounded CPU job and reusing the existing exact resident H2D/cache machinery.
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "''^(static |int |void |float |double |typedef struct|struct )[A-Za-z_]|iq2|q2_K|gemv|thread|affinity|worker|barrier|gate|down|expert" '"'C:\\Users\\imanu\\g130i\\cpugemv_spike\\cpugemv_spike.c' | Select-Object -First 300" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 481ms:
26:} block_q2_K;
37:} block_iq2_xxs;
39:#include "ds4_iq2_tables.inc"
41:static int8_t iq2xxs_signed_grid[256][128][8];
42:static int8_t iq2xxs_signs[128][8];
54:    TensorInfo gate;
56:    TensorInfo down;
60:    uint8_t *gate;
62:    uint8_t *down;
63:    uint64_t gate_bytes;
65:    uint64_t down_bytes;
66:    uint64_t gate_row_bytes;
68:    uint64_t down_row_bytes;
69:    uint32_t gate_type;
71:    uint32_t down_type;
75:    float *gate;
77:    float *down;
89:    int threads;
97:static volatile uint64_t g_evict_sink;
99:static void die(const char *msg) {
100:    fprintf(stderr, "cpugemv_spike: %s\n", msg);
104:static void *xmalloc(size_t n) {
110:static uint32_t rd_u32(FILE *f) {
116:static uint64_t rd_u64(FILE *f) {
122:static uint16_t rd_u16(FILE *f) {
128:static void skip_bytes(FILE *f, uint64_t n) {
132:static uint64_t scalar_size(uint32_t t) {
138:static void skip_value(FILE *f, uint32_t t) {
157:static void read_string(FILE *f, char *dst, size_t cap) {
167:static int wanted_name(const char *name, const char **which) {
168:    if (!strcmp(name, "blk.3.ffn_gate_exps.weight")) { *which = "gate"; return 1; }
170:    if (!strcmp(name, "blk.3.ffn_down_exps.weight")) { *which = "down"; return 1; }
174:static ModelInfo parse_model(const char *path) {
199:            if (!strcmp(which, "gate")) m.gate = t;
201:            else m.down = t;
207:    if (!m.gate.name[0] || !m.up.name[0] || !m.down.name[0]) die("missing layer-3 expert tensors");
211:static uint64_t row_bytes_for_type(uint32_t type, uint64_t cols) {
213:    if (type == GGUF_IQ2_XXS) return (cols / QK_K) * sizeof(block_iq2_xxs);
214:    if (type == GGUF_Q2_K) return (cols / QK_K) * sizeof(block_q2_K);
215:    die("unsupported expert quant type");
219:static void verify_tensor(const TensorInfo *t, int down) {
220:    uint64_t c = down ? N_MID : N_EMBD;
221:    uint64_t r = down ? N_EMBD : N_MID;
223:        die("expert tensor geometry mismatch");
228:static uint8_t *read_expert_payload(FILE *f, const ModelInfo *m, const TensorInfo *t, int down, uint32_t expert, uint64_t *bytes, uint64_t *row_bytes) {
229:    uint64_t cols = down ? N_MID : N_EMBD;
230:    uint64_t rows = down ? N_EMBD : N_MID;
233:    uint64_t expert_offset = *bytes * expert;
234:    uint64_t off = m->data_start + t->offset + expert_offset;
237:    if (fread(p, 1, (size_t)*bytes, f) != (size_t)*bytes) die("short expert payload");
241:static QuantExpert load_quant_expert(const char *path, ModelInfo *info) {
243:    verify_tensor(&info->gate, 0);
245:    verify_tensor(&info->down, 1);
246:    if (info->gate.type != GGUF_IQ2_XXS || info->up.type != GGUF_IQ2_XXS) die("gate/up must be IQ2_XXS for this spike");
252:    q.gate_type = info->gate.type;
254:    q.down_type = info->down.type;
255:    q.gate = read_expert_payload(f, info, &info->gate, 0, 0, &q.gate_bytes, &q.gate_row_bytes);
256:    q.up = read_expert_payload(f, info, &info->up, 0, 0, &q.up_bytes, &q.up_row_bytes);
257:    q.down = read_expert_payload(f, info, &info->down, 1, 0, &q.down_bytes, &q.down_row_bytes);
262:static inline float f16_to_f32(uint16_t h) {
289:static void iq2xxs_signed_grid_init(void) {
291:        const uint8_t signs = ksigns_iq2xs[s];
293:            iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
297:        const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
299:            const uint8_t signs = ksigns_iq2xs[s];
302:                iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
308:static inline int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
315:static inline int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
323:static void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
360:static void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
378:                sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
379:                                        iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
390:static void ds4_vec_dot_iq2_xxs_pair_q8_K(int n, float *s0, float *s1, const block_iq2_xxs *x0, const block_iq2_xxs *x1, const block_q8_K *y) {
391:    ds4_vec_dot_iq2_xxs_q8_K(n, s0, x0, y);
392:    ds4_vec_dot_iq2_xxs_q8_K(n, s1, x1, y);
395:static void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
427:static float silu(float x) {
431:static void quant_expert_forward(const QuantExpert *w, const float *x, float *y, float *mid, block_q8_K *xq, block_q8_K *midq) {
437:        float gate = 0.0f;
439:        const block_iq2_xxs *gr = (const block_iq2_xxs *)(w->gate + (uint64_t)r * w->gate_row_bytes);
440:        const block_iq2_xxs *ur = (const block_iq2_xxs *)(w->up + (uint64_t)r * w->up_row_bytes);
441:        ds4_vec_dot_iq2_xxs_pair_q8_K(N_EMBD, &gate, &up, gr, ur, xq);
442:        mid[r] = silu(gate) * up;
449:        const uint8_t *row = w->down + (uint64_t)r * w->down_row_bytes;
450:        if (w->down_type == GGUF_Q2_K) {
451:            ds4_vec_dot_q2_K_q8_K(N_MID, &y[r], (const block_q2_K *)row, midq);
453:            ds4_vec_dot_iq2_xxs_q8_K(N_MID, &y[r], (const block_iq2_xxs *)row, midq);
458:static void fp32_expert_forward_one(const Fp32Expert *w, const float *x, float *y, float *mid) {
462:        const float *g = w->gate + (size_t)r * N_EMBD;
464:        float gate = 0.0f;
467:            gate += x[k] * g[k];
470:        mid[r] = silu(gate) * up;
475:        const float *d = w->down + (size_t)r * N_MID;
482:static void parse_npy_payload(const uint8_t *buf, size_t n, size_t count, float *dst) {
490:static void load_teacher_npz(const char *path, Fp32Expert *e) {
491:    e->gate = (float *)xmalloc((size_t)N_MID * N_EMBD * sizeof(float));
493:    e->down = (float *)xmalloc((size_t)N_EMBD * N_MID * sizeof(float));
494:    int got_gate = 0, got_up = 0, got_down = 0;
520:        if (!strcmp(name, "gate.npy")) {
521:            parse_npy_payload(payload, comp_size, (size_t)N_MID * N_EMBD, e->gate);
522:            got_gate = 1;
526:        } else if (!strcmp(name, "down.npy")) {
527:            parse_npy_payload(payload, comp_size, (size_t)N_EMBD * N_MID, e->down);
528:            got_down = 1;
533:    if (!got_gate || !got_up || !got_down) die("teacher npz missing expected arrays");
536:static float *load_vectors(const char *path) {
546:static Metrics compute_metrics(const float *ref, const float *got) {
580:static double now_sec(void) {
587:static void evict_cache(uint8_t *buf, size_t n) {
593:static BenchRow run_bench_row(const QuantExpert *q, const float *x, int threads, uint8_t *evict, size_t evict_bytes, double expert_bytes_gb) {
596:    omp_set_num_threads(threads);
604:    if (threads == 1) iters = 5;
605:    quant_expert_forward(q, x, y, mid, xq, midq);
610:        quant_expert_forward(q, x, y, mid, xq, midq);
619:        quant_expert_forward(q, x, y, mid, xq, midq);
626:    r.threads = threads;
630:    r.gbps_mean = expert_bytes_gb / (cold_total / iters);
640:static void write_results_json(const char *path, const ModelInfo *m, const QuantExpert *q, const Metrics *metrics, const BenchRow *rows, int n_rows, double teacher_sec, double quant_correct_sec) {
643:    uint64_t total_bytes = q->gate_bytes + q->up_bytes + q->down_bytes;
645:    fprintf(f, "  \"schema\": \"cpugemv_spike_results_v1\",\n");
646:    fprintf(f, "  \"layer\": 3,\n  \"expert\": 0,\n");
648:    fprintf(f, "  \"tensor_types\": {\"gate\": %u, \"up\": %u, \"down\": %u},\n", q->gate_type, q->up_type, q->down_type);
649:    fprintf(f, "  \"tensor_offsets\": {\"data_start\": %llu, \"gate\": %llu, \"up\": %llu, \"down\": %llu},\n",
650:            (unsigned long long)m->data_start, (unsigned long long)m->gate.offset,
651:            (unsigned long long)m->up.offset, (unsigned long long)m->down.offset);
652:    fprintf(f, "  \"compressed_bytes\": {\"gate\": %llu, \"up\": %llu, \"down\": %llu, \"total\": %llu},\n",
653:            (unsigned long long)q->gate_bytes, (unsigned long long)q->up_bytes,
654:            (unsigned long long)q->down_bytes, (unsigned long long)total_bytes);
657:    fprintf(f, "  \"timing_support\": {\"teacher_forward_seconds\": %.6f, \"iq2_correctness_forward_seconds\": %.6f, \"cache_evict_bytes\": %u},\n",
661:        fprintf(f, "    {\"threads\": %d, \"iterations\": %d, \"cold_ms_mean\": %.6f, \"cold_ms_min\": %.6f, \"warm_ms_mean\": %.6f, \"effective_weight_gb_s\": %.6f}%s\n",
662:                rows[i].threads, rows[i].iterations, rows[i].cold_ms_mean, rows[i].cold_ms_min,
666:    fprintf(f, "  \"gpu_cold_path_assumption\": {\"h2d_and_selection_ms_per_expert_low\": 0.5, \"h2d_and_selection_ms_per_expert_high\": 0.7, \"gpu_gemm_ms_per_expert\": 0.02, \"current_exact_full_open_tps\": 1.65}\n");
671:int main(int argc, char **argv) {
675:    const char *out_json = argc > 4 ? argv[4] : "cpugemv_spike_results.json";
677:    iq2xxs_signed_grid_init();
679:    QuantExpert q = load_quant_expert(model_path, &info);
692:    omp_set_num_threads(8);
696:        fp32_expert_forward_one(&teacher, x + (size_t)s * N_EMBD, teacher_y + (size_t)s * N_EMBD, mid);
702:        quant_expert_forward(&q, x + (size_t)s * N_EMBD, quant_y + (size_t)s * N_EMBD, mid, xq, midq);
711:    const double expert_bytes_gb = (double)(q.gate_bytes + q.up_bytes + q.down_bytes) / 1.0e9;
713:    const int thread_counts[4] = {1, 2, 4, 8};
715:        rows[i] = run_bench_row(&q, x, thread_counts[i], evict, evict_bytes, expert_bytes_gb);
716:        printf("threads=%d cold_mean=%.3f ms warm_mean=%.3f ms gbps=%.3f\n",
717:               rows[i].threads, rows[i].cold_ms_mean, rows[i].warm_ms_mean, rows[i].gbps_mean);
731:    free(teacher.down);
733:    free(teacher.gate);
734:    free(q.down);
736:    free(q.gate);

2026-07-21T09:52:24.095423Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\cpugemv_spike\\cpugemv_spike.c'; Show-Lines "'$p 1 100; Show-Lines $p 208 460; Show-Lines $p 580 626; Show-Lines $p 671 720' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 548ms:
     1: #define _CRT_SECURE_NO_WARNINGS
     2: #include <windows.h>
     3: #include <stdint.h>
     4: #include <stdio.h>
     5: #include <stdlib.h>
     6: #include <string.h>
     7: #include <math.h>
     8: #include <float.h>
     9: #ifdef _OPENMP
    10: #include <omp.h>
    11: #endif
    12: 
    13: #define N_EMBD 4096
    14: #define N_MID 2048
    15: #define N_EXPERT 256
    16: #define N_SAMPLE 17
    17: #define QK_K 256
    18: #define GGUF_IQ2_XXS 16
    19: #define GGUF_Q2_K 10
    20: 
    21: typedef struct {
    22:     uint8_t scales[QK_K / 16];
    23:     uint8_t qs[QK_K / 4];
    24:     uint16_t d;
    25:     uint16_t dmin;
    26: } block_q2_K;
    27: 
    28: typedef struct {
    29:     float d;
    30:     int8_t qs[QK_K];
    31:     int16_t bsums[QK_K / 16];
    32: } block_q8_K;
    33: 
    34: typedef struct {
    35:     uint16_t d;
    36:     uint16_t qs[QK_K / 8];
    37: } block_iq2_xxs;
    38: 
    39: #include "ds4_iq2_tables.inc"
    40: 
    41: static int8_t iq2xxs_signed_grid[256][128][8];
    42: static int8_t iq2xxs_signs[128][8];
    43: 
    44: typedef struct {
    45:     char name[96];
    46:     uint64_t dims[4];
    47:     uint32_t ndims;
    48:     uint32_t type;
    49:     uint64_t offset;
    50: } TensorInfo;
    51: 
    52: typedef struct {
    53:     uint64_t data_start;
    54:     TensorInfo gate;
    55:     TensorInfo up;
    56:     TensorInfo down;
    57: } ModelInfo;
    58: 
    59: typedef struct {
    60:     uint8_t *gate;
    61:     uint8_t *up;
    62:     uint8_t *down;
    63:     uint64_t gate_bytes;
    64:     uint64_t up_bytes;
    65:     uint64_t down_bytes;
    66:     uint64_t gate_row_bytes;
    67:     uint64_t up_row_bytes;
    68:     uint64_t down_row_bytes;
    69:     uint32_t gate_type;
    70:     uint32_t up_type;
    71:     uint32_t down_type;
    72: } QuantExpert;
    73: 
    74: typedef struct {
    75:     float *gate;
    76:     float *up;
    77:     float *down;
    78: } Fp32Expert;
    79: 
    80: typedef struct {
    81:     double cosine;
    82:     double nmse;
    83:     double max_abs;
    84:     double max_rel_1e6;
    85:     double min_sample_cosine;
    86: } Metrics;
    87: 
    88: typedef struct {
    89:     int threads;
    90:     double cold_ms_mean;
    91:     double cold_ms_min;
    92:     double warm_ms_mean;
    93:     double gbps_mean;
    94:     int iterations;
    95: } BenchRow;
    96: 
    97: static volatile uint64_t g_evict_sink;
    98: 
    99: static void die(const char *msg) {
   100:     fprintf(stderr, "cpugemv_spike: %s\n", msg);
   208:     return m;
   209: }
   210: 
   211: static uint64_t row_bytes_for_type(uint32_t type, uint64_t cols) {
   212:     if (cols % QK_K != 0) die("row width is not QK_K aligned");
   213:     if (type == GGUF_IQ2_XXS) return (cols / QK_K) * sizeof(block_iq2_xxs);
   214:     if (type == GGUF_Q2_K) return (cols / QK_K) * sizeof(block_q2_K);
   215:     die("unsupported expert quant type");
   216:     return 0;
   217: }
   218: 
   219: static void verify_tensor(const TensorInfo *t, int down) {
   220:     uint64_t c = down ? N_MID : N_EMBD;
   221:     uint64_t r = down ? N_EMBD : N_MID;
   222:     if (t->ndims != 3 || t->dims[0] != c || t->dims[1] != r || t->dims[2] != N_EXPERT) {
   223:         die("expert tensor geometry mismatch");
   224:     }
   225:     if (t->type != GGUF_IQ2_XXS && t->type != GGUF_Q2_K) die("unexpected tensor quantization type");
   226: }
   227: 
   228: static uint8_t *read_expert_payload(FILE *f, const ModelInfo *m, const TensorInfo *t, int down, uint32_t expert, uint64_t *bytes, uint64_t *row_bytes) {
   229:     uint64_t cols = down ? N_MID : N_EMBD;
   230:     uint64_t rows = down ? N_EMBD : N_MID;
   231:     *row_bytes = row_bytes_for_type(t->type, cols);
   232:     *bytes = *row_bytes * rows;
   233:     uint64_t expert_offset = *bytes * expert;
   234:     uint64_t off = m->data_start + t->offset + expert_offset;
   235:     if (_fseeki64(f, (__int64)off, SEEK_SET) != 0) die("tensor seek failed");
   236:     uint8_t *p = (uint8_t *)xmalloc((size_t)*bytes);
   237:     if (fread(p, 1, (size_t)*bytes, f) != (size_t)*bytes) die("short expert payload");
   238:     return p;
   239: }
   240: 
   241: static QuantExpert load_quant_expert(const char *path, ModelInfo *info) {
   242:     *info = parse_model(path);
   243:     verify_tensor(&info->gate, 0);
   244:     verify_tensor(&info->up, 0);
   245:     verify_tensor(&info->down, 1);
   246:     if (info->gate.type != GGUF_IQ2_XXS || info->up.type != GGUF_IQ2_XXS) die("gate/up must be IQ2_XXS for this spike");
   247: 
   248:     FILE *f = fopen(path, "rb");
   249:     if (!f) die("cannot reopen GGUF");
   250:     QuantExpert q;
   251:     memset(&q, 0, sizeof(q));
   252:     q.gate_type = info->gate.type;
   253:     q.up_type = info->up.type;
   254:     q.down_type = info->down.type;
   255:     q.gate = read_expert_payload(f, info, &info->gate, 0, 0, &q.gate_bytes, &q.gate_row_bytes);
   256:     q.up = read_expert_payload(f, info, &info->up, 0, 0, &q.up_bytes, &q.up_row_bytes);
   257:     q.down = read_expert_payload(f, info, &info->down, 1, 0, &q.down_bytes, &q.down_row_bytes);
   258:     fclose(f);
   259:     return q;
   260: }
   261: 
   262: static inline float f16_to_f32(uint16_t h) {
   263:     uint32_t sign = (uint32_t)(h & 0x8000) << 16;
   264:     uint32_t exp = (h >> 10) & 0x1f;
   265:     uint32_t mant = h & 0x03ff;
   266:     uint32_t bits;
   267:     if (exp == 0) {
   268:         if (mant == 0) {
   269:             bits = sign;
   270:         } else {
   271:             exp = 1;
   272:             while ((mant & 0x0400) == 0) {
   273:                 mant <<= 1;
   274:                 exp--;
   275:             }
   276:             mant &= 0x03ff;
   277:             bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
   278:         }
   279:     } else if (exp == 31) {
   280:         bits = sign | 0x7f800000u | (mant << 13);
   281:     } else {
   282:         bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
   283:     }
   284:     float f;
   285:     memcpy(&f, &bits, sizeof(f));
   286:     return f;
   287: }
   288: 
   289: static void iq2xxs_signed_grid_init(void) {
   290:     for (uint32_t s = 0; s < 128; s++) {
   291:         const uint8_t signs = ksigns_iq2xs[s];
   292:         for (uint32_t j = 0; j < 8; j++) {
   293:             iq2xxs_signs[s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -1 : 1);
   294:         }
   295:     }
   296:     for (uint32_t g = 0; g < 256; g++) {
   297:         const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + g);
   298:         for (uint32_t s = 0; s < 128; s++) {
   299:             const uint8_t signs = ksigns_iq2xs[s];
   300:             for (uint32_t j = 0; j < 8; j++) {
   301:                 const int v = (int)grid[j];
   302:                 iq2xxs_signed_grid[g][s][j] = (int8_t)((signs & kmask_iq2xs[j]) ? -v : v);
   303:             }
   304:         }
   305:     }
   306: }
   307: 
   308: static inline int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
   309:     int32_t sum = 0;
   310:     for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
   311:     for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
   312:     return sum;
   313: }
   314: 
   315: static inline int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
   316:     int32_t sum = 0;
   317:     for (int i = 0; i < 16; i++) {
   318:         sum += (int32_t)((q2[i] >> shift) & 3) * (int32_t)q8[i];
   319:     }
   320:     return sum;
   321: }
   322: 
   323: static void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k) {
   324:     if (k % QK_K != 0) die("Q8_K quantization length is not QK_K aligned");
   325:     const int64_t nb = k / QK_K;
   326:     for (int64_t b = 0; b < nb; b++) {
   327:         float max = 0.0f;
   328:         float amax = 0.0f;
   329:         for (int j = 0; j < QK_K; j++) {
   330:             const float ax = fabsf(x[j]);
   331:             if (ax > amax) {
   332:                 amax = ax;
   333:                 max = x[j];
   334:             }
   335:         }
   336:         if (amax == 0.0f) {
   337:             y[b].d = 0.0f;
   338:             memset(y[b].qs, 0, sizeof(y[b].qs));
   339:             memset(y[b].bsums, 0, sizeof(y[b].bsums));
   340:             x += QK_K;
   341:             continue;
   342:         }
   343:         const float iscale = -127.0f / max;
   344:         for (int j = 0; j < QK_K; j++) {
   345:             int v = (int)lrintf(iscale * x[j]);
   346:             if (v > 127) v = 127;
   347:             if (v < -128) v = -128;
   348:             y[b].qs[j] = (int8_t)v;
   349:         }
   350:         for (int j = 0; j < QK_K / 16; j++) {
   351:             int sum = 0;
   352:             for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
   353:             y[b].bsums[j] = (int16_t)sum;
   354:         }
   355:         y[b].d = 1.0f / iscale;
   356:         x += QK_K;
   357:     }
   358: }
   359: 
   360: static void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y) {
   361:     const int nb = n / QK_K;
   362:     uint32_t aux32[2];
   363:     const uint8_t *aux8 = (const uint8_t *)aux32;
   364:     float sumf = 0.0f;
   365:     for (int i = 0; i < nb; i++) {
   366:         const float d = f16_to_f32(x[i].d) * y[i].d;
   367:         const uint16_t *q2 = x[i].qs;
   368:         const int8_t *q8 = y[i].qs;
   369:         int32_t bsum = 0;
   370:         for (int ib32 = 0; ib32 < QK_K / 32; ib32++) {
   371:             memcpy(aux32, q2, 2 * sizeof(uint32_t));
   372:             q2 += 4;
   373:             const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
   374:             int32_t sumi = 0;
   375:             for (int l = 0; l < 4; l += 2) {
   376:                 const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
   377:                 const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
   378:                 sumi += dot_iq2_pair_16(iq2xxs_signed_grid[aux8[l]][sign_idx0],
   379:                                         iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
   380:                                         q8);
   381:                 q8 += 16;
   382:             }
   383:             bsum += sumi * (int32_t)ls;
   384:         }
   385:         sumf += d * (float)bsum;
   386:     }
   387:     *s = 0.125f * sumf;
   388: }
   389: 
   390: static void ds4_vec_dot_iq2_xxs_pair_q8_K(int n, float *s0, float *s1, const block_iq2_xxs *x0, const block_iq2_xxs *x1, const block_q8_K *y) {
   391:     ds4_vec_dot_iq2_xxs_q8_K(n, s0, x0, y);
   392:     ds4_vec_dot_iq2_xxs_q8_K(n, s1, x1, y);
   393: }
   394: 
   395: static void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y) {
   396:     const int nb = n / QK_K;
   397:     float sumf = 0.0f;
   398:     for (int i = 0; i < nb; i++) {
   399:         const uint8_t *q2 = x[i].qs;
   400:         const int8_t *q8 = y[i].qs;
   401:         const uint8_t *sc = x[i].scales;
   402:         int summs = 0;
   403:         for (int j = 0; j < 16; j++) {
   404:             summs += y[i].bsums[j] * (sc[j] >> 4);
   405:         }
   406:         const float dall = y[i].d * f16_to_f32(x[i].d);
   407:         const float dmin = y[i].d * f16_to_f32(x[i].dmin);
   408:         int isum = 0;
   409:         int is = 0;
   410:         for (int k = 0; k < QK_K / 128; k++) {
   411:             int shift = 0;
   412:             for (int j = 0; j < 4; j++) {
   413:                 int d = sc[is++] & 0x0f;
   414:                 isum += d * dot_q2_16(q2, q8, shift);
   415:                 d = sc[is++] & 0x0f;
   416:                 isum += d * dot_q2_16(q2 + 16, q8 + 16, shift);
   417:                 shift += 2;
   418:                 q8 += 32;
   419:             }
   420:             q2 += 32;
   421:         }
   422:         sumf += dall * (float)isum - dmin * (float)summs;
   423:     }
   424:     *s = sumf;
   425: }
   426: 
   427: static float silu(float x) {
   428:     return x / (1.0f + expf(-x));
   429: }
   430: 
   431: static void quant_expert_forward(const QuantExpert *w, const float *x, float *y, float *mid, block_q8_K *xq, block_q8_K *midq) {
   432:     ds4_quantize_row_q8_K(x, xq, N_EMBD);
   433: 
   434:     int r;
   435:     #pragma omp parallel for schedule(static)
   436:     for (r = 0; r < N_MID; r++) {
   437:         float gate = 0.0f;
   438:         float up = 0.0f;
   439:         const block_iq2_xxs *gr = (const block_iq2_xxs *)(w->gate + (uint64_t)r * w->gate_row_bytes);
   440:         const block_iq2_xxs *ur = (const block_iq2_xxs *)(w->up + (uint64_t)r * w->up_row_bytes);
   441:         ds4_vec_dot_iq2_xxs_pair_q8_K(N_EMBD, &gate, &up, gr, ur, xq);
   442:         mid[r] = silu(gate) * up;
   443:     }
   444: 
   445:     ds4_quantize_row_q8_K(mid, midq, N_MID);
   446: 
   447:     #pragma omp parallel for schedule(static)
   448:     for (r = 0; r < N_EMBD; r++) {
   449:         const uint8_t *row = w->down + (uint64_t)r * w->down_row_bytes;
   450:         if (w->down_type == GGUF_Q2_K) {
   451:             ds4_vec_dot_q2_K_q8_K(N_MID, &y[r], (const block_q2_K *)row, midq);
   452:         } else {
   453:             ds4_vec_dot_iq2_xxs_q8_K(N_MID, &y[r], (const block_iq2_xxs *)row, midq);
   454:         }
   455:     }
   456: }
   457: 
   458: static void fp32_expert_forward_one(const Fp32Expert *w, const float *x, float *y, float *mid) {
   459:     int r;
   460:     #pragma omp parallel for schedule(static)
   580: static double now_sec(void) {
   581:     LARGE_INTEGER c, f;
   582:     QueryPerformanceCounter(&c);
   583:     QueryPerformanceFrequency(&f);
   584:     return (double)c.QuadPart / (double)f.QuadPart;
   585: }
   586: 
   587: static void evict_cache(uint8_t *buf, size_t n) {
   588:     uint64_t s = 0;
   589:     for (size_t i = 0; i < n; i += 64) s += buf[i];
   590:     g_evict_sink += s;
   591: }
   592: 
   593: static BenchRow run_bench_row(const QuantExpert *q, const float *x, int threads, uint8_t *evict, size_t evict_bytes, double expert_bytes_gb) {
   594:     #ifdef _OPENMP
   595:     omp_set_dynamic(0);
   596:     omp_set_num_threads(threads);
   597:     #endif
   598:     float *y = (float *)xmalloc((size_t)N_EMBD * sizeof(float));
   599:     float *mid = (float *)xmalloc((size_t)N_MID * sizeof(float));
   600:     block_q8_K *xq = (block_q8_K *)xmalloc((N_EMBD / QK_K) * sizeof(block_q8_K));
   601:     block_q8_K *midq = (block_q8_K *)xmalloc((N_MID / QK_K) * sizeof(block_q8_K));
   602: 
   603:     int iters = 7;
   604:     if (threads == 1) iters = 5;
   605:     quant_expert_forward(q, x, y, mid, xq, midq);
   606: 
   607:     double warm_total = 0.0;
   608:     for (int i = 0; i < iters; i++) {
   609:         double t0 = now_sec();
   610:         quant_expert_forward(q, x, y, mid, xq, midq);
   611:         warm_total += now_sec() - t0;
   612:     }
   613: 
   614:     double cold_total = 0.0;
   615:     double cold_min = DBL_MAX;
   616:     for (int i = 0; i < iters; i++) {
   617:         evict_cache(evict, evict_bytes);
   618:         double t0 = now_sec();
   619:         quant_expert_forward(q, x, y, mid, xq, midq);
   620:         double dt = now_sec() - t0;
   621:         cold_total += dt;
   622:         if (dt < cold_min) cold_min = dt;
   623:     }
   624: 
   625:     BenchRow r;
   626:     r.threads = threads;
   671: int main(int argc, char **argv) {
   672:     const char *model_path = argc > 1 ? argv[1] : "C:\\ds4-models\\ds4-2bit.gguf";
   673:     const char *teacher_path = argc > 2 ? argv[2] : "C:\\Users\\imanu\\g130i\\recovery_poc\\teacher_l3e0_fp32.npz";
   674:     const char *vectors_path = argc > 3 ? argv[3] : "C:\\Users\\imanu\\g130i\\trace_out\\l3e0.vectors.f32le";
   675:     const char *out_json = argc > 4 ? argv[4] : "cpugemv_spike_results.json";
   676: 
   677:     iq2xxs_signed_grid_init();
   678:     ModelInfo info;
   679:     QuantExpert q = load_quant_expert(model_path, &info);
   680:     Fp32Expert teacher;
   681:     load_teacher_npz(teacher_path, &teacher);
   682:     float *x = load_vectors(vectors_path);
   683: 
   684:     float *teacher_y = (float *)xmalloc((size_t)N_SAMPLE * N_EMBD * sizeof(float));
   685:     float *quant_y = (float *)xmalloc((size_t)N_SAMPLE * N_EMBD * sizeof(float));
   686:     float *mid = (float *)xmalloc((size_t)N_MID * sizeof(float));
   687:     block_q8_K *xq = (block_q8_K *)xmalloc((N_EMBD / QK_K) * sizeof(block_q8_K));
   688:     block_q8_K *midq = (block_q8_K *)xmalloc((N_MID / QK_K) * sizeof(block_q8_K));
   689: 
   690:     #ifdef _OPENMP
   691:     omp_set_dynamic(0);
   692:     omp_set_num_threads(8);
   693:     #endif
   694:     double t0 = now_sec();
   695:     for (int s = 0; s < N_SAMPLE; s++) {
   696:         fp32_expert_forward_one(&teacher, x + (size_t)s * N_EMBD, teacher_y + (size_t)s * N_EMBD, mid);
   697:     }
   698:     double teacher_sec = now_sec() - t0;
   699: 
   700:     t0 = now_sec();
   701:     for (int s = 0; s < N_SAMPLE; s++) {
   702:         quant_expert_forward(&q, x + (size_t)s * N_EMBD, quant_y + (size_t)s * N_EMBD, mid, xq, midq);
   703:     }
   704:     double quant_correct_sec = now_sec() - t0;
   705:     Metrics metrics = compute_metrics(teacher_y, quant_y);
   706: 
   707:     size_t evict_bytes = (size_t)512 * 1024 * 1024;
   708:     uint8_t *evict = (uint8_t *)xmalloc(evict_bytes);
   709:     for (size_t i = 0; i < evict_bytes; i += 4096) evict[i] = (uint8_t)i;
   710: 
   711:     const double expert_bytes_gb = (double)(q.gate_bytes + q.up_bytes + q.down_bytes) / 1.0e9;
   712:     BenchRow rows[4];
   713:     const int thread_counts[4] = {1, 2, 4, 8};
   714:     for (int i = 0; i < 4; i++) {
   715:         rows[i] = run_bench_row(&q, x, thread_counts[i], evict, evict_bytes, expert_bytes_gb);
   716:         printf("threads=%d cold_mean=%.3f ms warm_mean=%.3f ms gbps=%.3f\n",
   717:                rows[i].threads, rows[i].cold_ms_mean, rows[i].warm_ms_mean, rows[i].gbps_mean);
   718:     }
   719: 
   720:     write_results_json(out_json, &info, &q, &metrics, rows, 4, teacher_sec, quant_correct_sec);

2026-07-21T09:52:30.410835Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4.c'; Show-Lines "'$p 4560 4870; Show-Lines $p 6500 6630' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 590ms:
  4560:     default:
  4561:         ds4_die("unsupported tensor type for dense matvec");
  4562:     }
  4563: }
  4564: 
  4565: static float tensor_1d_value(const ds4_model *m, const ds4_tensor *t, uint64_t i) {
  4566:     if (i >= t->elements) ds4_die("tensor scalar index is out of bounds");
  4567:     if (t->type == 0) {
  4568:         const float *p = tensor_data(m, t);
  4569:         return p[i];
  4570:     }
  4571:     if (t->type == 1) {
  4572:         const uint16_t *p = tensor_data(m, t);
  4573:         return f16_to_f32(p[i]);
  4574:     }
  4575:     ds4_die("unsupported tensor scalar type");
  4576:     return 0.0f;
  4577: }
  4578: 
  4579: static float tensor_2d_value(const ds4_model *m, const ds4_tensor *t, uint64_t x, uint64_t y) {
  4580:     if (t->ndim != 2 || x >= t->dim[0] || y >= t->dim[1]) {
  4581:         ds4_die("tensor 2D index is out of bounds");
  4582:     }
  4583:     return tensor_1d_value(m, t, y * t->dim[0] + x);
  4584: }
  4585: 
  4586: /* Locate one expert's 2D matrix inside a 3D GGUF expert tensor. */
  4587: static const uint8_t *tensor_expert_bytes(
  4588:         const ds4_model  *m,
  4589:         const ds4_tensor *w,
  4590:         uint32_t          expert,
  4591:         uint64_t         *in_dim,
  4592:         uint64_t         *out_dim,
  4593:         uint64_t         *row_bytes) {
  4594:     if (w->ndim != 3) ds4_die("expected a 3D expert tensor");
  4595:     if (expert >= w->dim[2]) ds4_die("expert id is outside expert tensor");
  4596: 
  4597:     *in_dim = w->dim[0];
  4598:     *out_dim = w->dim[1];
  4599: 
  4600:     const gguf_type_info *info = tensor_type(w->type);
  4601:     if (!info || info->block_elems == 0) ds4_die("unsupported expert tensor type");
  4602:     const uint64_t blocks = (*in_dim + info->block_elems - 1) / info->block_elems;
  4603:     *row_bytes = blocks * info->block_bytes;
  4604: 
  4605:     const uint64_t expert_bytes = *out_dim * *row_bytes;
  4606:     return (const uint8_t *)tensor_data(m, w) + (uint64_t)expert * expert_bytes;
  4607: }
  4608: 
  4609: typedef struct {
  4610:     float *out0;
  4611:     float *out1;
  4612:     const uint8_t *base0;
  4613:     const uint8_t *base1;
  4614:     const block_q8_K *xq;
  4615:     uint64_t in_dim;
  4616:     uint64_t row_bytes0;
  4617:     uint64_t row_bytes1;
  4618: } matvec_iq2_xxs_pair_ctx;
  4619: 
  4620: static void matvec_iq2_xxs_pair_worker(void *vctx, uint64_t row0, uint64_t row1) {
  4621:     matvec_iq2_xxs_pair_ctx *ctx = vctx;
  4622:     for (uint64_t row = row0; row < row1; row++) {
  4623:         const block_iq2_xxs *br0 = (const block_iq2_xxs *)(ctx->base0 + row * ctx->row_bytes0);
  4624:         const block_iq2_xxs *br1 = (const block_iq2_xxs *)(ctx->base1 + row * ctx->row_bytes1);
  4625:         ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &ctx->out0[row], &ctx->out1[row], br0, br1, ctx->xq);
  4626:     }
  4627: }
  4628: 
  4629: /* Project one routed expert's gate and up matrices.  Both are IQ2_XXS and
  4630:  * share the same Q8_K activation. */
  4631: static void matvec_iq2_xxs_expert_pair_prequant(
  4632:         float            *out0,
  4633:         float            *out1,
  4634:         const ds4_model  *m,
  4635:         const ds4_tensor *w0,
  4636:         const ds4_tensor *w1,
  4637:         const block_q8_K *xq,
  4638:         uint32_t          expert) {
  4639:     if (w0->type != 16 || w1->type != 16) ds4_die("expected IQ2_XXS expert tensors");
  4640: 
  4641:     uint64_t in_dim0, out_dim0, row_bytes0;
  4642:     uint64_t in_dim1, out_dim1, row_bytes1;
  4643:     const uint8_t *base0 = tensor_expert_bytes(m, w0, expert, &in_dim0, &out_dim0, &row_bytes0);
  4644:     const uint8_t *base1 = tensor_expert_bytes(m, w1, expert, &in_dim1, &out_dim1, &row_bytes1);
  4645:     if (in_dim0 != in_dim1 || out_dim0 != out_dim1) ds4_die("paired IQ2_XXS expert tensors do not match");
  4646:     if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");
  4647: 
  4648:     matvec_iq2_xxs_pair_ctx ctx = {
  4649:         .out0 = out0,
  4650:         .out1 = out1,
  4651:         .base0 = base0,
  4652:         .base1 = base1,
  4653:         .xq = xq,
  4654:         .in_dim = in_dim0,
  4655:         .row_bytes0 = row_bytes0,
  4656:         .row_bytes1 = row_bytes1,
  4657:     };
  4658:     ds4_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
  4659: }
  4660: 
  4661: static float silu(float x);
  4662: 
  4663: typedef struct {
  4664:     float *mid;
  4665:     const uint8_t *gate_base[DS4_N_EXPERT_USED];
  4666:     const uint8_t *up_base[DS4_N_EXPERT_USED];
  4667:     const block_q8_K *xq;
  4668:     float expert_weight[DS4_N_EXPERT_USED];
  4669:     float clamp;
  4670:     uint64_t in_dim;
  4671:     uint64_t out_dim;
  4672:     uint64_t gate_row_bytes[DS4_N_EXPERT_USED];
  4673:     uint64_t up_row_bytes[DS4_N_EXPERT_USED];
  4674:     int n_expert;
  4675: } matvec_iq2_xxs_mid_ctx;
  4676: 
  4677: static void matvec_iq2_xxs_mid_worker(void *vctx, uint64_t row0, uint64_t row1) {
  4678:     matvec_iq2_xxs_mid_ctx *ctx = vctx;
  4679: 
  4680:     for (uint64_t idx = row0; idx < row1; idx++) {
  4681:         const int slot = (int)(idx / ctx->out_dim);
  4682:         const uint64_t row = idx - (uint64_t)slot * ctx->out_dim;
  4683:         float gate = 0.0f;
  4684:         float up = 0.0f;
  4685: 
  4686:         const block_iq2_xxs *gate_row = (const block_iq2_xxs *)(ctx->gate_base[slot] + row * ctx->gate_row_bytes[slot]);
  4687:         const block_iq2_xxs *up_row = (const block_iq2_xxs *)(ctx->up_base[slot] + row * ctx->up_row_bytes[slot]);
  4688:         ds4_vec_dot_iq2_xxs_pair_q8_K((int)ctx->in_dim, &gate, &up, gate_row, up_row, ctx->xq);
  4689: 
  4690:         if (ctx->clamp > 1.0e-6f) {
  4691:             if (gate > ctx->clamp) gate = ctx->clamp;
  4692:             if (up > ctx->clamp) up = ctx->clamp;
  4693:             if (up < -ctx->clamp) up = -ctx->clamp;
  4694:         }
  4695:         ctx->mid[idx] = silu(gate) * up * ctx->expert_weight[slot];
  4696:     }
  4697: }
  4698: 
  4699: /* Build all selected expert hidden vectors: IQ2_XXS gate/up, clamp, SwiGLU,
  4700:  * and router weight.  The down projection runs later on the quantized mids. */
  4701: static void matvec_iq2_xxs_experts_mid_prequant(
  4702:         float            *mid,
  4703:         const ds4_model  *m,
  4704:         const ds4_tensor *gate_w,
  4705:         const ds4_tensor *up_w,
  4706:         const block_q8_K *xq,
  4707:         const int        *selected,
  4708:         const float      *expert_weight,
  4709:         int               n_expert,
  4710:         float             clamp) {
  4711:     if (gate_w->type != 16 || up_w->type != 16) ds4_die("expected IQ2_XXS expert tensors");
  4712:     if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");
  4713: 
  4714:     uint64_t in_dim0 = 0;
  4715:     uint64_t out_dim0 = 0;
  4716:     matvec_iq2_xxs_mid_ctx ctx = {
  4717:         .mid = mid,
  4718:         .xq = xq,
  4719:         .clamp = clamp,
  4720:         .n_expert = n_expert,
  4721:     };
  4722: 
  4723:     for (int i = 0; i < n_expert; i++) {
  4724:         uint64_t gate_in_dim, gate_out_dim;
  4725:         uint64_t up_in_dim, up_out_dim;
  4726:         ctx.gate_base[i] = tensor_expert_bytes(m, gate_w, (uint32_t)selected[i],
  4727:                                                &gate_in_dim, &gate_out_dim, &ctx.gate_row_bytes[i]);
  4728:         ctx.up_base[i] = tensor_expert_bytes(m, up_w, (uint32_t)selected[i],
  4729:                                              &up_in_dim, &up_out_dim, &ctx.up_row_bytes[i]);
  4730:         if (gate_in_dim != up_in_dim || gate_out_dim != up_out_dim) {
  4731:             ds4_die("paired IQ2_XXS expert tensors do not match");
  4732:         }
  4733:         if (i == 0) {
  4734:             in_dim0 = gate_in_dim;
  4735:             out_dim0 = gate_out_dim;
  4736:         } else if (gate_in_dim != in_dim0 || gate_out_dim != out_dim0) {
  4737:             ds4_die("IQ2_XXS expert tensors do not share a layout");
  4738:         }
  4739:         ctx.expert_weight[i] = expert_weight[i];
  4740:     }
  4741:     if (in_dim0 % QK_K != 0) ds4_die("IQ2_XXS expert row is not QK_K aligned");
  4742: 
  4743:     ctx.in_dim = in_dim0;
  4744:     ctx.out_dim = out_dim0;
  4745:     ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
  4746: }
  4747: 
  4748: typedef struct {
  4749:     float *out;
  4750:     const uint8_t *base;
  4751:     const block_q8_K *xq;
  4752:     uint64_t in_dim;
  4753:     uint64_t row_bytes;
  4754: } matvec_q2_k_ctx;
  4755: 
  4756: static void matvec_q2_k_worker(void *vctx, uint64_t row0, uint64_t row1) {
  4757:     matvec_q2_k_ctx *ctx = vctx;
  4758:     for (uint64_t row = row0; row < row1; row++) {
  4759:         const block_q2_K *br = (const block_q2_K *)(ctx->base + row * ctx->row_bytes);
  4760:         ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &ctx->out[row], br, ctx->xq);
  4761:     }
  4762: }
  4763: 
  4764: /* Single expert Q2_K down projection, kept mostly for tracing and diagnostics. */
  4765: static void matvec_q2_k_expert(
  4766:         float            *out,
  4767:         const ds4_model  *m,
  4768:         const ds4_tensor *w,
  4769:         const float      *x,
  4770:         uint32_t          expert) {
  4771:     if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
  4772: 
  4773:     uint64_t in_dim, out_dim, row_bytes;
  4774:     const uint8_t *base = tensor_expert_bytes(m, w, expert, &in_dim, &out_dim, &row_bytes);
  4775:     if (in_dim % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");
  4776: 
  4777:     block_q8_K *xq = xmalloc((size_t)(in_dim / QK_K) * sizeof(xq[0]));
  4778:     ds4_quantize_row_q8_K(x, xq, (int64_t)in_dim);
  4779: 
  4780:     matvec_q2_k_ctx ctx = {
  4781:         .out = out,
  4782:         .base = base,
  4783:         .xq = xq,
  4784:         .in_dim = in_dim,
  4785:         .row_bytes = row_bytes,
  4786:     };
  4787:     ds4_parallel_for(out_dim, matvec_q2_k_worker, &ctx);
  4788: 
  4789:     free(xq);
  4790: }
  4791: 
  4792: typedef struct {
  4793:     float *out;
  4794:     const uint8_t *base[DS4_N_EXPERT_USED];
  4795:     const block_q8_K *xq[DS4_N_EXPERT_USED];
  4796:     uint64_t in_dim;
  4797:     uint64_t row_bytes[DS4_N_EXPERT_USED];
  4798:     int n_expert;
  4799: } matvec_q2_k_accum_ctx;
  4800: 
  4801: static void matvec_q2_k_accum_worker(void *vctx, uint64_t row0, uint64_t row1) {
  4802:     matvec_q2_k_accum_ctx *ctx = vctx;
  4803: 
  4804:     for (uint64_t row = row0; row < row1; row++) {
  4805:         float acc = 0.0f;
  4806:         for (int i = 0; i < ctx->n_expert; i++) {
  4807:             float v = 0.0f;
  4808:             const block_q2_K *br = (const block_q2_K *)(ctx->base[i] + row * ctx->row_bytes[i]);
  4809:             ds4_vec_dot_q2_K_q8_K((int)ctx->in_dim, &v, br, ctx->xq[i]);
  4810:             acc += v;
  4811:         }
  4812:         ctx->out[row] = acc;
  4813:     }
  4814: }
  4815: 
  4816: /* Accumulate all selected experts' Q2_K down projections directly into the
  4817:  * 4096-wide MoE output. */
  4818: static void matvec_q2_k_experts_accum_prequant(
  4819:         float            *out,
  4820:         const ds4_model  *m,
  4821:         const ds4_tensor *w,
  4822:         const block_q8_K *xq,
  4823:         const int        *selected,
  4824:         int               n_expert) {
  4825:     if (w->type != 10) ds4_die("expected a Q2_K expert tensor");
  4826:     if (n_expert < 1 || n_expert > DS4_N_EXPERT_USED) ds4_die("unexpected routed expert count");
  4827: 
  4828:     uint64_t in_dim0 = 0;
  4829:     uint64_t out_dim0 = 0;
  4830:     const uint8_t *base[DS4_N_EXPERT_USED];
  4831:     uint64_t row_bytes[DS4_N_EXPERT_USED];
  4832: 
  4833:     for (int i = 0; i < n_expert; i++) {
  4834:         uint64_t in_dim, out_dim;
  4835:         base[i] = tensor_expert_bytes(m, w, (uint32_t)selected[i], &in_dim, &out_dim, &row_bytes[i]);
  4836:         if (i == 0) {
  4837:             in_dim0 = in_dim;
  4838:             out_dim0 = out_dim;
  4839:         } else if (in_dim != in_dim0 || out_dim != out_dim0) {
  4840:             ds4_die("Q2_K expert tensors do not share a layout");
  4841:         }
  4842:     }
  4843:     if (in_dim0 % QK_K != 0) ds4_die("Q2_K expert row is not QK_K aligned");
  4844: 
  4845:     const uint64_t n_blocks = in_dim0 / QK_K;
  4846:     matvec_q2_k_accum_ctx ctx = {
  4847:         .out = out,
  4848:         .in_dim = in_dim0,
  4849:         .n_expert = n_expert,
  4850:     };
  4851:     for (int i = 0; i < n_expert; i++) {
  4852:         ctx.base[i] = base[i];
  4853:         ctx.row_bytes[i] = row_bytes[i];
  4854:         ctx.xq[i] = xq + (uint64_t)i * n_blocks;
  4855:     }
  4856: 
  4857:     ds4_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
  4858: }
  4859: 
  4860: typedef struct {
  4861:     uint32_t token;
  4862:     uint32_t slot;
  4863: } ds4_expert_pair;
  4864: 
  4865: typedef struct {
  4866:     float *mid;
  4867:     const uint8_t *gate_base[DS4_N_EXPERT];
  4868:     const uint8_t *up_base[DS4_N_EXPERT];
  4869:     const block_q8_K *xq;
  4870:     const ds4_expert_pair *pairs;
  6500:                                             selected,
  6501:                                             expert_weight,
  6502:                                             DS4_N_EXPERT_USED,
  6503:                                             clamp);
  6504:         for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
  6505:             ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
  6506:                                   midq + (uint64_t)i * (down_in_dim / QK_K),
  6507:                                   (int64_t)down_in_dim);
  6508:         }
  6509:         matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);
  6510:     } else {
  6511:         for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
  6512:             const uint32_t expert = (uint32_t)selected[i];
  6513: 
  6514:             matvec_iq2_xxs_expert_pair_prequant(gate, up, model,
  6515:                                                  layer->ffn_gate_exps,
  6516:                                                  layer->ffn_up_exps,
  6517:                                                  xq,
  6518:                                                  expert);
  6519:             char name[64];
  6520:             snprintf(name, sizeof(name), "blk.%u expert %u gate", il, expert);
  6521:             print_vec_stats(name, gate, DS4_N_FF_EXP);
  6522:             snprintf(name, sizeof(name), "blk.%u expert %u up", il, expert);
  6523:             print_vec_stats(name, up, DS4_N_FF_EXP);
  6524: 
  6525:             /*
  6526:              * DeepSeek V4 clamps routed expert gate/up values before SwiGLU and
  6527:              * applies the router weight before the down projection.
  6528:              */
  6529:             const float limit = clamp;
  6530:             for (int j = 0; j < DS4_N_FF_EXP; j++) {
  6531:                 if (limit > 1.0e-6f) {
  6532:                     if (gate[j] > limit) gate[j] = limit;
  6533:                     if (up[j] > limit) up[j] = limit;
  6534:                     if (up[j] < -limit) up[j] = -limit;
  6535:                 }
  6536:                 mid[j] = silu(gate[j]) * up[j] * expert_weight[i];
  6537:             }
  6538: 
  6539:             snprintf(name, sizeof(name), "blk.%u expert %u mid", il, expert);
  6540:             print_vec_stats(name, mid, DS4_N_FF_EXP);
  6541: 
  6542:             matvec_q2_k_expert(down, model, layer->ffn_down_exps, mid, expert);
  6543:             snprintf(name, sizeof(name), "blk.%u expert %u down", il, expert);
  6544:             print_vec_stats(name, down, DS4_N_EMBD);
  6545:             for (int j = 0; j < DS4_N_EMBD; j++) out[j] += down[j];
  6546:         }
  6547:     }
  6548: 
  6549:     free(midq);
  6550:     free(xq);
  6551:     free(down);
  6552:     free(mid_all);
  6553:     free(mid);
  6554:     free(up);
  6555:     free(gate);
  6556: }
  6557: 
  6558: /* Decode version of routed MoE: same math as layer_routed_moe_one(), but all
  6559:  * large temporaries come from the persistent scratch arena. */
  6560: static void layer_routed_moe_one_prealloc(
  6561:         float             * out,
  6562:         const ds4_model   * model,
  6563:         const ds4_layer_weights * layer,
  6564:         const float       * x,
  6565:         uint32_t            il,
  6566:         int                 token,
  6567:         float               clamp,
  6568:         float              * mid_all,
  6569:         block_q8_K         * xq,
  6570:         block_q8_K         * midq) {
  6571:     int selected[DS4_N_EXPERT_USED];
  6572:     float expert_weight[DS4_N_EXPERT_USED];
  6573:     const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
  6574:     const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
  6575: 
  6576:     if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
  6577:     if (down_in_dim != DS4_N_FF_EXP || down_in_dim % QK_K != 0) ds4_die("Q2_K expert input has an unexpected layout");
  6578: 
  6579:     memset(out, 0, (size_t)DS4_N_EMBD * sizeof(out[0]));
  6580:     ds4_quantize_row_q8_K(x, xq, (int64_t)expert_in_dim);
  6581: 
  6582:     if (layer->ffn_gate_tid2eid) {
  6583:         layer_hash_selected_experts(selected, model, layer, token);
  6584:         layer_hash_router_weights_one(expert_weight, model, layer, x, selected);
  6585:     } else {
  6586:         layer_topk_selected_experts(selected, expert_weight, model, layer, il, x);
  6587:     }
  6588:     ds4_reap_mask_validate_selected(il, selected, DS4_N_EXPERT_USED);
  6589: 
  6590:     matvec_iq2_xxs_experts_mid_prequant(mid_all, model,
  6591:                                         layer->ffn_gate_exps,
  6592:                                         layer->ffn_up_exps,
  6593:                                         xq,
  6594:                                         selected,
  6595:                                         expert_weight,
  6596:                                         DS4_N_EXPERT_USED,
  6597:                                         clamp);
  6598: 
  6599:     for (int i = 0; i < DS4_N_EXPERT_USED; i++) {
  6600:         ds4_quantize_row_q8_K(mid_all + (uint64_t)i * down_in_dim,
  6601:                               midq + (uint64_t)i * (down_in_dim / QK_K),
  6602:                               (int64_t)down_in_dim);
  6603:     }
  6604:     matvec_q2_k_experts_accum_prequant(out, model, layer->ffn_down_exps, midq, selected, DS4_N_EXPERT_USED);
  6605: 
  6606:     (void)il;
  6607: }
  6608: 
  6609: /* Prefill MoE groups token/expert pairs by expert so each active expert's
  6610:  * rows are scanned once for the whole token batch. */
  6611: static void layer_routed_moe_batch(
  6612:         float             * moe,
  6613:         const ds4_model   * model,
  6614:         const ds4_layer_weights * layer,
  6615:         const float       * norm,
  6616:         const int         * token_ids,
  6617:         uint32_t            n_tok,
  6618:         uint32_t            il,
  6619:         float               clamp) {
  6620:     const uint64_t expert_in_dim = layer->ffn_gate_exps->dim[0];
  6621:     const uint64_t expert_out_dim = layer->ffn_gate_exps->dim[1];
  6622:     const uint64_t down_in_dim = layer->ffn_down_exps->dim[0];
  6623:     const uint64_t down_out_dim = layer->ffn_down_exps->dim[1];
  6624:     if (expert_in_dim % QK_K != 0) ds4_die("IQ2_XXS expert input is not QK_K aligned");
  6625:     if (down_in_dim % QK_K != 0) ds4_die("Q2_K expert input is not QK_K aligned");
  6626:     if (expert_out_dim != down_in_dim || down_out_dim != DS4_N_EMBD) {
  6627:         ds4_die("routed expert tensor layout is unexpected");
  6628:     }
  6629: 
  6630:     const uint32_t total_pairs = n_tok * DS4_N_EXPERT_USED;

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"ds4_parallel_for|worker_pool|thread_pool|cpu_worker|DS4_CPU|THREAD\" 'C:\\Users\\imanu\\g130i\\ds4.c' | Select-Object -First 220" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 481ms:
666:#define DS4_MAX_THREADS 32
669:    os_thread_t threads[DS4_MAX_THREADS];
682:} ds4_thread_pool;
684:static ds4_thread_pool g_pool;
741:    const char *env = getenv("DS4_THREADS");
747:    if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
788:static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
822:static void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
823:    ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
3639:    ds4_parallel_for_min_rows(out_dim, matvec_f16_worker, &ctx, min_rows);
4027:    ds4_parallel_for(n_tok, quantize_q8_0_batch_worker, &ctx);
4183:    ds4_parallel_for(n_rows, matvec_q8_0_worker, &ctx);
4223:    ds4_parallel_for(w0->dim[1], matvec_q8_0_pair_worker, &ctx);
4245:    ds4_parallel_for(ctx.out_dim, matmul_q8_0_batch_worker, &ctx);
4276:    ds4_parallel_for(ctx.out_dim, matmul_q8_0_pair_batch_worker, &ctx);
4434:    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);
4474:    ds4_parallel_for((uint64_t)n_groups * rank, matvec_q8_0_grouped_worker, &ctx);
4516:    ds4_parallel_for((uint64_t)n_groups * rank, matmul_q8_0_grouped_batch_worker, &ctx);
4551:    ds4_parallel_for(w->dim[1], matvec_f32_worker, &ctx);
4658:    ds4_parallel_for(out_dim0, matvec_iq2_xxs_pair_worker, &ctx);
4745:    ds4_parallel_for((uint64_t)n_expert * out_dim0, matvec_iq2_xxs_mid_worker, &ctx);
4787:    ds4_parallel_for(out_dim, matvec_q2_k_worker, &ctx);
4857:    ds4_parallel_for(out_dim0, matvec_q2_k_accum_worker, &ctx);
5287:    ds4_parallel_for_min_rows(n_tok, hc_post_batch_worker, &ctx, 1);
5346:    ds4_parallel_for_min_rows(n_tok, hc_post_sum_batch_worker, &ctx, 1);
5428:    ds4_parallel_for_min_rows(n_tok, hc_pre_norm_batch_worker, &ctx, 1);
5691:    ds4_parallel_for_min_rows(n_tok, rope_tail_batch_worker, &ctx, 1);
5994:    ds4_parallel_for(n_tok, swiglu_batch_worker, &swiglu_ctx);
6711:    ds4_parallel_for((uint64_t)n_active * expert_out_dim, matvec_iq2_xxs_batch_mid_worker, &mid_ctx);
6721:    ds4_parallel_for(total_pairs, quantize_mid_pairs_worker, &quant_ctx);
6748:    ds4_parallel_for(down_out_dim, matvec_q2_k_batch_accum_rows_worker, &down_ctx);
7071:    ds4_parallel_for_min_rows(n_tok, routed_moe_tokens_worker, &ctx, 1);
7240:    ds4_parallel_for(n_tok, layer_ffn_tokens_worker, &ctx);
8036:    ds4_parallel_for_min_rows((uint64_t)n_tok * DS4_N_HEAD,
9560:static void *metal_graph_spex_cpu_worker(void *arg) {
9694:        os_thread_create(&probe->thread, metal_graph_spex_cpu_worker, probe) != 0) {
16281:        const char *dump_cpu = getenv("DS4_CPU_DUMP_LOGITS");
17522:    const char *dump_prefill_logits = getenv("DS4_CPU_DUMP_PREFILL_LOGITS");

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 650 835" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 505ms:
   650:     return true;
   651: }
   652: 
   653: static bool cpu_directional_steering_enabled(
   654:         const float *dirs,
   655:         float        scale);
   656: 
   657: static void cpu_directional_steering_project_rows(
   658:         float       *x,
   659:         const float *dirs,
   660:         uint32_t     il,
   661:         uint32_t     rows,
   662:         float        scale);
   663: 
   664: typedef void (*ds4_parallel_fn)(void *ctx, uint64_t row0, uint64_t row1);
   665: 
   666: #define DS4_MAX_THREADS 32
   667: 
   668: typedef struct {
   669:     os_thread_t threads[DS4_MAX_THREADS];
   670:     os_mutex_t mutex;
   671:     os_cond_t work_cond;
   672:     os_cond_t done_cond;
   673:     uint32_t n_threads;
   674:     uint32_t n_workers;
   675:     uint32_t generation;
   676:     uint32_t done;
   677:     bool initialized;
   678:     bool shutdown;
   679:     ds4_parallel_fn fn;
   680:     void *ctx;
   681:     uint64_t n_rows;
   682: } ds4_thread_pool;
   683: 
   684: static ds4_thread_pool g_pool;
   685: static _Thread_local int g_parallel_depth;
   686: static uint32_t g_requested_threads;
   687: 
   688: static void *ds4_worker_main(void *arg) {
   689:     const uint32_t tid = (uint32_t)(uintptr_t)arg;
   690:     uint32_t seen_generation = 0;
   691: 
   692:     for (;;) {
   693:         os_mutex_lock(&g_pool.mutex);
   694:         while (seen_generation == g_pool.generation && !g_pool.shutdown) {
   695:             os_cond_wait(&g_pool.work_cond, &g_pool.mutex);
   696:         }
   697:         if (g_pool.shutdown) {
   698:             os_mutex_unlock(&g_pool.mutex);
   699:             return NULL;
   700:         }
   701: 
   702:         seen_generation = g_pool.generation;
   703:         ds4_parallel_fn fn = g_pool.fn;
   704:         void *ctx = g_pool.ctx;
   705:         const uint64_t n_rows = g_pool.n_rows;
   706:         const uint32_t n_threads = g_pool.n_threads;
   707:         os_mutex_unlock(&g_pool.mutex);
   708: 
   709:         const uint64_t rows_per_thread = (n_rows + n_threads - 1) / n_threads;
   710:         const uint64_t row0 = (uint64_t)tid * rows_per_thread;
   711:         uint64_t row1 = row0 + rows_per_thread;
   712:         if (row1 > n_rows) row1 = n_rows;
   713:         if (row0 < row1) {
   714:             g_parallel_depth++;
   715:             fn(ctx, row0, row1);
   716:             g_parallel_depth--;
   717:         }
   718: 
   719:         os_mutex_lock(&g_pool.mutex);
   720:         g_pool.done++;
   721:         if (g_pool.done == g_pool.n_workers) {
   722:             os_cond_signal(&g_pool.done_cond);
   723:         }
   724:         os_mutex_unlock(&g_pool.mutex);
   725:     }
   726: }
   727: 
   728: /* Create the persistent CPU worker pool.  Decode reuses these threads instead
   729:  * of creating pthreads in the token loop. */
   730: static void ds4_threads_init(void) {
   731:     if (g_pool.initialized) return;
   732: 
   733:     os_once(&iq2xxs_signed_grid_once, iq2xxs_signed_grid_init);
   734: 
   735:     uint32_t n_threads = 12;
   736:     const long online_cpus = os_cpu_count();
   737:     if (online_cpus > 0) {
   738:         n_threads = online_cpus < 12 ? (uint32_t)online_cpus : 12;
   739:     }
   740: 
   741:     const char *env = getenv("DS4_THREADS");
   742:     if (env && env[0]) {
   743:         long v = strtol(env, NULL, 10);
   744:         if (v > 0) n_threads = (uint32_t)v;
   745:     }
   746:     if (g_requested_threads > 0) n_threads = g_requested_threads;
   747:     if (n_threads > DS4_MAX_THREADS) n_threads = DS4_MAX_THREADS;
   748:     if (n_threads == 0) n_threads = 1;
   749: 
   750:     os_mutex_init(&g_pool.mutex);
   751:     os_cond_init(&g_pool.work_cond);
   752:     os_cond_init(&g_pool.done_cond);
   753:     g_pool.n_threads = n_threads;
   754:     g_pool.n_workers = n_threads > 0 ? n_threads - 1 : 0;
   755:     g_pool.generation = 0;
   756:     g_pool.done = 0;
   757:     g_pool.shutdown = false;
   758:     g_pool.initialized = true;
   759: 
   760:     for (uint32_t i = 1; i < n_threads; i++) {
   761:         if (os_thread_create(&g_pool.threads[i], ds4_worker_main, (void *)(uintptr_t)i) != 0) {
   762:             ds4_die("failed to create worker thread");
   763:         }
   764:     }
   765: }
   766: 
   767: static void ds4_threads_shutdown(void) {
   768:     if (!g_pool.initialized) return;
   769: 
   770:     os_mutex_lock(&g_pool.mutex);
   771:     g_pool.shutdown = true;
   772:     g_pool.generation++;
   773:     os_cond_broadcast(&g_pool.work_cond);
   774:     os_mutex_unlock(&g_pool.mutex);
   775: 
   776:     for (uint32_t i = 1; i < g_pool.n_threads; i++) {
   777:         os_thread_join(g_pool.threads[i]);
   778:     }
   779: 
   780:     os_cond_destroy(&g_pool.done_cond);
   781:     os_cond_destroy(&g_pool.work_cond);
   782:     os_mutex_destroy(&g_pool.mutex);
   783:     memset(&g_pool, 0, sizeof(g_pool));
   784: }
   785: 
   786: /* Run a row-parallel CPU kernel, falling back to serial execution for small
   787:  * jobs or nested calls where spawning more work would only add latency. */
   788: static void ds4_parallel_for_min_rows(uint64_t n_rows, ds4_parallel_fn fn, void *ctx, uint64_t min_parallel_rows) {
   789:     ds4_threads_init();
   790: 
   791:     if (g_parallel_depth > 0 || g_pool.n_threads <= 1 || n_rows < min_parallel_rows) {
   792:         fn(ctx, 0, n_rows);
   793:         return;
   794:     }
   795: 
   796:     os_mutex_lock(&g_pool.mutex);
   797:     g_pool.fn = fn;
   798:     g_pool.ctx = ctx;
   799:     g_pool.n_rows = n_rows;
   800:     g_pool.done = 0;
   801:     g_pool.generation++;
   802:     os_cond_broadcast(&g_pool.work_cond);
   803: 
   804:     const uint64_t rows_per_thread = (n_rows + g_pool.n_threads - 1) / g_pool.n_threads;
   805:     uint64_t main_row1 = rows_per_thread;
   806:     if (main_row1 > n_rows) main_row1 = n_rows;
   807:     os_mutex_unlock(&g_pool.mutex);
   808: 
   809:     if (main_row1 > 0) {
   810:         g_parallel_depth++;
   811:         fn(ctx, 0, main_row1);
   812:         g_parallel_depth--;
   813:     }
   814: 
   815:     os_mutex_lock(&g_pool.mutex);
   816:     while (g_pool.done < g_pool.n_workers) {
   817:         os_cond_wait(&g_pool.done_cond, &g_pool.mutex);
   818:     }
   819:     os_mutex_unlock(&g_pool.mutex);
   820: }
   821: 
   822: static void ds4_parallel_for(uint64_t n_rows, ds4_parallel_fn fn, void *ctx) {
   823:     ds4_parallel_for_min_rows(n_rows, fn, ctx, 512);
   824: }
   825: 
   826: static void cursor_error(ds4_cursor *c, const char *msg) {
   827:     if (c->error[0] == '\0') {
   828:         snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
   829:     }
   830: }
   831: 
   832: static bool cursor_has(ds4_cursor *c, uint64_t n) {
   833:     if (n > c->size || c->pos > c->size - n) {
   834:         cursor_error(c, "truncated GGUF file");
   835:         return false;

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg --files 'C:\\Users\\imanu\\g130i\\overlap_spike'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 376ms:
C:\Users\imanu\g130i\overlap_spike\topology_probe.obj
C:\Users\imanu\g130i\overlap_spike\topology_probe.exe
C:\Users\imanu\g130i\overlap_spike\run_overlap_spike.cmd
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md
C:\Users\imanu\g130i\overlap_spike\README.md
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json
C:\Users\imanu\g130i\overlap_spike\overlap_spike_cpu.obj
C:\Users\imanu\g130i\overlap_spike\overlap_spike.obj
C:\Users\imanu\g130i\overlap_spike\overlap_spike.lib
C:\Users\imanu\g130i\overlap_spike\overlap_spike.exp
C:\Users\imanu\g130i\overlap_spike\overlap_spike.exe
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_legacy_masks_results.json
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.obj
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.lib
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.exp
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.exe
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c
C:\Users\imanu\g130i\overlap_spike\ds4_iq2_tables.inc
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log
C:\Users\imanu\g130i\overlap_spike\build_overlap_spike.bat
C:\Users\imanu\g130i\overlap_spike\build_overlap_cpu_bench.bat

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"cpu_threads|affinity|0x555|tail_6|CreateThread|SetThreadAffinityMask|omp_set|reserve|core\" 'C:\\Users\\imanu\\g130i\\overlap_spike'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 174ms:
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:15:int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:22:    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:35:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:36:    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:76:    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:77:    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:84:        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:92:            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:97:                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:105:                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:120:        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_bench.c:122:                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:20:HARNESS DESIGN: for tail sizes N in {4, 8, 16, 24} and hot-set M fixed: (a) SERIAL: GPU work then CPU tail; (b) OVERLAPPED: launch GPU work async, run CPU tail on worker threads concurrently, then join both; measure wall for each, plus CUDA submission stalls (time to enqueue while CPU tail is running). Report overlap efficiency = (serial-overlapped)/cpu_tail_time. Also measure with the CPU tail pinned to 6 of 8 cores (leaving 2 for the submit thread) vs all 8.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:63:2026-07-21T09:14:48.164919Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:970:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:971:    omp_set_num_threads(threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1066:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1067:    omp_set_num_threads(8);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1573:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1574:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1577:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1582:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1661:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1666:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1668:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1669:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:1671:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2130:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2131:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2134:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2139:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2218:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2223:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2225:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2226:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2228:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2242:The CPU side is in place now, including the cold-cache eviction path and affinity masks for the OpenMP workers. Next I’m adding the CUDA/main harness and the scripts/docs, then I’ll compile-check without launching the executable.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2693:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2694:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2697:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2702:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2781:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2786:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2788:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2789:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2791:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2821:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2840:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2841:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2857:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2858:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2905:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:2906:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3012:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3043:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3048:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3093:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3103:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3104:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3110:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3118:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3119:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3123:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3124:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3612:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3613:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3616:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3621:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3700:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3705:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3707:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3708:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3710:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3740:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3759:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3760:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3776:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3777:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3824:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3825:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3931:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3962:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:3967:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4012:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4022:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4023:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4029:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4037:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4038:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4042:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4043:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4534:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4535:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4538:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4543:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4622:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4627:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4629:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4630:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4632:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4662:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4681:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4682:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4698:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4699:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4746:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4747:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4855:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4886:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4891:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4936:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4946:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4947:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4953:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4961:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4962:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4966:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:4967:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5457:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5458:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5461:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5466:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5545:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5550:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5552:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5553:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5555:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5585:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5604:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5605:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5621:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5622:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5669:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5670:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5778:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5809:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5814:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5859:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5869:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5870:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5876:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5884:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5885:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5889:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5890:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5975:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:5976:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6475:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6476:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6479:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6484:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6563:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6568:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6570:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6571:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6573:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6603:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6622:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6623:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6639:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6640:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6687:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6688:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6796:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6827:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6832:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6877:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6887:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6888:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6894:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6902:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6903:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6907:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:6908:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7017:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7018:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7517:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7518:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7521:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7526:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7605:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7610:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7612:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7613:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7615:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7645:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7664:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7665:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7681:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7682:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7729:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7730:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7838:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7869:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7874:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7919:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7929:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7930:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7936:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7944:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7945:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7949:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:7950:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8029:2026-07-21T09:21:24.841881Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8074:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8075:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8574:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8575:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8578:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8583:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8662:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8667:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8669:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8670:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8672:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8702:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8721:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8722:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8738:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8739:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8786:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8787:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8895:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8926:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8931:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8976:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8986:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8987:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:8993:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9001:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9002:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9006:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9007:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9131:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9132:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9631:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9632:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9635:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9640:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9719:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9724:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9726:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9727:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9729:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9759:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9778:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9779:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9795:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9796:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9843:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9844:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9952:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9983:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:9988:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10033:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10043:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10044:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10050:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10058:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10059:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10063:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10064:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10223:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10224:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10723:+static void apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10724:+    if (mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10727:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10732:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10811:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10816:+    apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10818:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10819:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10821:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10851:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10870:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10871:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10887:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10888:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10935:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:10936:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11044:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11075:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11080:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11125:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11135:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11136:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11142:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11150:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11151:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11155:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11156:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11232:The compile check succeeded and produced `overlap_spike.exe`. I found one behavioral issue in review: the serial CPU tail could leave the main CUDA-submit thread pinned after measuring CPU affinity, so I’m patching that to restore the caller’s affinity before rebuilding.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11271:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11272:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11770:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11772:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11775:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11780:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11859:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11864:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11866:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11867:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11869:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11878:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11900:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11919:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11920:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11936:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11937:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11984:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:11985:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12093:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12124:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12129:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12174:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12184:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12185:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12191:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12199:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12200:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12204:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12205:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12314:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12315:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12813:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12815:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12818:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12823:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12902:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12907:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12909:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12910:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12912:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12921:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12943:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12962:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12963:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12979:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:12980:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13027:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13028:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13136:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13167:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13172:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13217:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13227:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13228:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13234:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13242:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13243:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13247:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13248:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13324:The affinity restoration patch is in. I’m rebuilding once more to make sure the final tree compiles cleanly; still no executable run.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13371:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13372:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13870:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13872:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13875:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13880:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13959:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13964:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13966:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13967:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13969:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:13978:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14000:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14019:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14020:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14036:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14037:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14084:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14085:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14193:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14224:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14229:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14274:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14284:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14285:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14291:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14299:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14300:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14304:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14305:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14388:2026-07-21T09:22:31.191850Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14410:2026-07-21T09:22:31.279769Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14733:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:14734:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15232:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15234:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15237:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15242:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15321:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15326:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15328:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15329:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15331:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15340:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15362:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15381:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15382:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15398:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15399:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15446:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15447:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15555:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15586:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15591:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15636:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15646:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15647:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15653:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15661:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15662:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15666:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15667:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15787:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:15788:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16286:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16288:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16291:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16296:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16375:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16380:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16382:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16383:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16385:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16394:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16416:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16435:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16436:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16452:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16453:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16500:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16501:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16609:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16640:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16645:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16690:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16700:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16701:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16707:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16715:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16716:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16720:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16721:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16844:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:16845:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17343:+static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17345:+    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17348:+static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17353:+        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17432:+__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17437:+    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17439:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17440:+    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17442:+    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17451:+    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17473:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17492:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17493:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17509:+    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17510:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17557:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17558:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17666:+    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17697:+        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17702:+                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17747:+        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17757:+            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17758:+            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17764:+            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17772:+            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17773:+            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17777:+            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_overlap.log:17778:+            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:29:I’ll trace the exact per-expert call paths, allocation/threading behavior, and timing scopes in both harnesses, then run only CPU-safe reproductions. I’ll preserve existing changes and will patch the overlap harness only if the discrepancy is caused by a measurable harness bug.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:82:    {"mode": "tail_6_of_8", "tail_experts": 4, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 241.637300, "serial_gpu_wall_ms": 170.037700, "serial_gpu_event_ms": 168.190979, "serial_gpu_enqueue_ms": 99.213600, "cpu_tail_ms": 70.671300, "overlapped_wall_ms": 78.011000, "overlapped_gpu_event_ms": 72.544128, "overlapped_enqueue_ms": 0.132000, "overlap_efficiency": 2.315315, "pass": true},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:83:    {"mode": "tail_6_of_8", "tail_experts": 8, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 214.295000, "serial_gpu_wall_ms": 72.224200, "serial_gpu_event_ms": 72.169472, "serial_gpu_enqueue_ms": 0.087600, "cpu_tail_ms": 142.002000, "overlapped_wall_ms": 150.206700, "overlapped_gpu_event_ms": 72.023041, "overlapped_enqueue_ms": 0.098300, "overlap_efficiency": 0.451320, "pass": false},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:84:    {"mode": "tail_6_of_8", "tail_experts": 16, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 358.479500, "serial_gpu_wall_ms": 72.323500, "serial_gpu_event_ms": 72.253281, "serial_gpu_enqueue_ms": 0.091300, "cpu_tail_ms": 286.100600, "overlapped_wall_ms": 294.133000, "overlapped_gpu_event_ms": 72.168449, "overlapped_enqueue_ms": 0.110200, "overlap_efficiency": 0.224909, "pass": false},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:85:    {"mode": "tail_6_of_8", "tail_experts": 24, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 502.735200, "serial_gpu_wall_ms": 72.131700, "serial_gpu_event_ms": 72.071106, "serial_gpu_enqueue_ms": 0.095800, "cpu_tail_ms": 430.526700, "overlapped_wall_ms": 435.978800, "overlapped_gpu_event_ms": 72.129539, "overlapped_enqueue_ms": 0.085900, "overlap_efficiency": 0.155058, "pass": false},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:86:    {"mode": "tail_8_of_8", "tail_experts": 4, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 144.416100, "serial_gpu_wall_ms": 73.264700, "serial_gpu_event_ms": 73.142273, "serial_gpu_enqueue_ms": 0.087900, "cpu_tail_ms": 70.274000, "overlapped_wall_ms": 78.062100, "overlapped_gpu_event_ms": 73.847618, "overlapped_enqueue_ms": 0.130400, "overlap_efficiency": 0.944218, "pass": true},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:87:    {"mode": "tail_8_of_8", "tail_experts": 8, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 211.284200, "serial_gpu_wall_ms": 72.061600, "serial_gpu_event_ms": 72.010750, "serial_gpu_enqueue_ms": 0.088400, "cpu_tail_ms": 139.160200, "overlapped_wall_ms": 150.041000, "overlapped_gpu_event_ms": 72.645630, "overlapped_enqueue_ms": 0.133200, "overlap_efficiency": 0.440091, "pass": false},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:88:    {"mode": "tail_8_of_8", "tail_experts": 16, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 351.609300, "serial_gpu_wall_ms": 72.196800, "serial_gpu_event_ms": 72.076286, "serial_gpu_enqueue_ms": 0.089400, "cpu_tail_ms": 279.308600, "overlapped_wall_ms": 285.358300, "overlapped_gpu_event_ms": 72.093697, "overlapped_enqueue_ms": 0.173800, "overlap_efficiency": 0.237196, "pass": false},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:89:    {"mode": "tail_8_of_8", "tail_experts": 24, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 494.212900, "serial_gpu_wall_ms": 72.074100, "serial_gpu_event_ms": 72.014847, "serial_gpu_enqueue_ms": 0.093900, "cpu_tail_ms": 422.056800, "overlapped_wall_ms": 421.842400, "overlapped_gpu_event_ms": 72.927231, "overlapped_enqueue_ms": 0.087000, "overlap_efficiency": 0.171471, "pass": false}
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:145:"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'rg -n "''^(static |__declspec|extern|int main|void |double |float ).*(forward|gemv|dequant|worker|thread|tail|expert|evict|bench|silu)|CreateThread|WaitFor|omp|IQ2|iq2|threads|serial|cpu_tail|for '"\\(.*expert|for \\(.*n\" overlap_spike.c; Write-Output '--- CPUGEMV ---'; rg -n \""'^(static |__declspec|extern|int main|void |double |float ).*(forward|gemv|dequant|worker|thread|expert|evict|bench|silu)|CreateThread|WaitFor|omp|IQ2|iq2|threads|cold|warm|for '"\\(.*expert|for \\(.*it\" 'C:\\Users\\imanu\\g130i\\cpugemv_spike\\cpugemv_spike.c'" in C:\Users\imanu\g130i\overlap_spike
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:201:441:static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:202:446:static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:214:530:__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:216:537:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:217:538:    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:218:540:    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:272:595:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:273:596:    omp_set_num_threads(threads);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:295:691:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:296:692:    omp_set_num_threads(8);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:667:  441: static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:669:  443:     return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:672:  446: static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:677:  451:         (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1085:  595:     omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1086:  596:     omp_set_num_threads(threads);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1181:  691:     omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1182:  692:     omp_set_num_threads(8);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1301:  530: __declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1306:  535:     DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1308:  537:     omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1309:  538:     omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1311:  540:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1320:  549:     if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1337:   14: int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1356:   33:     int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1357:   34:     uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1373:   50:     int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1374:   51:     uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1421:   98:     printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1422:   99:     printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1530:  207:     job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1561:  238:         fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1566:  243:                 r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1611:  288:         {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1621:  298:             r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1622:  299:             r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1628:  305:             rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1636:  313:             job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1637:  314:             job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1641:  318:             HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1642:  319:             if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1689:2026-07-21T09:39:06.206720Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1746:- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1747:- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1866:- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1867:- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1956:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:1975:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2035:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2054:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2055:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2101:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2113:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2153:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2208:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2227:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2228:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2274:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2286:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2326:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2342:2026-07-21T09:40:42.256908Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2394:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2413:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2414:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2460:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2472:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2512:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2605:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2624:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2625:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2671:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2683:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2723:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2789:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2808:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2809:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2855:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2867:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2907:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:2995:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3014:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3015:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3061:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3073:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3113:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3172:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3191:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3192:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3238:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3250:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3290:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3320:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3325:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3328:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3377:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3396:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3397:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3443:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3455:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3495:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3525:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3530:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3533:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3542:2026-07-21T09:41:39.520250Z ERROR codex_core::tools::router: error=Exit code: 1
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3612:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3631:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3632:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3678:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3690:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3730:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3760:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3765:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3768:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3829:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3848:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3849:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3895:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3907:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3947:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3977:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3982:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:3985:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4043:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4062:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4063:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4109:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4121:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4161:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4191:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4196:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4199:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4210:core=0 group=0 mask=0x3
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4211:core=1 group=0 mask=0xc
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4212:core=2 group=0 mask=0x30
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4213:core=3 group=0 mask=0xc0
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4214:core=4 group=0 mask=0x300
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4215:core=5 group=0 mask=0xc00
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4216:core=6 group=0 mask=0x3000
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4217:core=7 group=0 mask=0xc000
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4272:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4291:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4292:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4338:+        const uint64_t affinity_mask = threads >= 8 ? 0xffull : ((1ull << threads) - 1ull);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4350:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4390:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4420:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4425:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4428:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4436:A second, smaller harness issue is confirmed from Windows’ processor topology: logical CPUs are paired as `0x3, 0xC, 0x30, …`. Therefore `0x3f` permits six OpenMP threads on only three physical cores, and `0xff` permits eight threads on four cores. The requested team sizes are real (6 and 8), but the “6 of 8 cores / 8 of 8 cores” labels and affinity intent are not. I’m correcting those masks and measuring both the legacy and physical-core layouts.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4451:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4452:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4453:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4454:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4456:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4458:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4513:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4520:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4533:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4534:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4574:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4575:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4582:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4590:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4595:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4603:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4618:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4620:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4637:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4657:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4658:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4659:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4660:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4670:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4671:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4685:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4686:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4708:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4713:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4716:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4731:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4732:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4733:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4734:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4736:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4738:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4793:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4800:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4813:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4814:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4854:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4855:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4862:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4870:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4875:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4883:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4898:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4900:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4917:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4937:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4938:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4939:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4940:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4950:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4951:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4965:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4966:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4988:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4993:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:4996:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5026:threads=6 effective=6 mask=0x555 N=1 mean=1.014 ms min=0.904 ms mean/expert=1.014 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5027:threads=6 effective=6 mask=0x555 N=4 mean=3.790 ms min=3.646 ms mean/expert=0.948 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5028:threads=6 effective=6 mask=0x555 N=8 mean=7.380 ms min=7.327 ms mean/expert=0.922 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5029:threads=6 effective=6 mask=0x555 N=16 mean=14.858 ms min=14.758 ms mean/expert=0.929 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5030:threads=6 effective=6 mask=0x555 N=24 mean=24.465 ms min=22.475 ms mean/expert=1.019 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5031:threads=8 effective=8 mask=0x5555 N=1 mean=1.374 ms min=0.716 ms mean/expert=1.374 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5032:threads=8 effective=8 mask=0x5555 N=4 mean=2.802 ms min=2.799 ms mean/expert=0.701 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5033:threads=8 effective=8 mask=0x5555 N=8 mean=5.686 ms min=5.625 ms mean/expert=0.711 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5034:threads=8 effective=8 mask=0x5555 N=16 mean=12.446 ms min=11.371 ms mean/expert=0.778 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5035:threads=8 effective=8 mask=0x5555 N=24 mean=17.705 ms min=17.066 ms mean/expert=0.738 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5048:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5049:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5050:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5051:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5053:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5055:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5110:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5117:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5130:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5131:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5171:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5172:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5179:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5187:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5192:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5200:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5215:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5217:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5234:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5254:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5255:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5256:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5257:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5267:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5268:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5282:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5283:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5305:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5310:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5313:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5339:threads=6 effective=6 mask=0x555 N=1 mean=0.931 ms min=0.867 ms mean/expert=0.931 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5340:threads=6 effective=6 mask=0x555 N=4 mean=3.948 ms min=3.493 ms mean/expert=0.987 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5341:threads=6 effective=6 mask=0x555 N=8 mean=7.097 ms min=7.035 ms mean/expert=0.887 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5342:threads=6 effective=6 mask=0x555 N=16 mean=14.455 ms min=14.203 ms mean/expert=0.903 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5343:threads=6 effective=6 mask=0x555 N=24 mean=22.345 ms min=21.471 ms mean/expert=0.931 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5344:threads=8 effective=8 mask=0x5555 N=1 mean=0.860 ms min=0.670 ms mean/expert=0.860 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5345:threads=8 effective=8 mask=0x5555 N=4 mean=2.700 ms min=2.682 ms mean/expert=0.675 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5346:threads=8 effective=8 mask=0x5555 N=8 mean=5.723 ms min=5.392 ms mean/expert=0.715 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5347:threads=8 effective=8 mask=0x5555 N=16 mean=11.121 ms min=10.848 ms mean/expert=0.695 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5348:threads=8 effective=8 mask=0x5555 N=24 mean=17.393 ms min=16.433 ms mean/expert=0.725 ms
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5359:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5360:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5361:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5362:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5364:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5366:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5421:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5428:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5441:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5442:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5482:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5483:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5490:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5498:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5503:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5511:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5526:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5528:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5545:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5565:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5566:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5567:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5568:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5578:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5579:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5593:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5594:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5616:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5621:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5624:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5680:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5681:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5682:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5683:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5685:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5687:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5742:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5749:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5762:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5763:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5803:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5804:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5811:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5819:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5824:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5832:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5847:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5849:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5866:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5886:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5887:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5888:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5889:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5899:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5900:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5914:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5915:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5937:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5942:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5945:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5953:The discrepancy is explained by a timed 512 MiB eviction sweep per expert; a separate affinity-mask bug also bound 6/8 workers to 3/4 physical cores. CPU-only measurements now confirm corrected distinct-stream scaling.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5967:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5968:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5969:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5970:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5972:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:5974:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6029:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6036:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6049:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6050:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6090:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6091:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6098:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6106:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6111:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6119:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6134:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6136:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6153:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6173:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6174:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6175:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6176:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6186:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6187:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6201:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6202:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6224:+    int core = 0;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6229:+            printf("core=%d group=%u mask=0x%llx\n", core, e->Processor.GroupMask[g].Group,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6232:+        core++;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6253:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6254:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6255:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6256:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6258:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6260:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6329:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6336:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6349:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6350:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6390:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6391:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6398:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6406:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6411:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6419:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6434:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6436:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6453:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6473:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6474:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6475:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6476:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6486:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6487:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6511:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6512:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6525:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6526:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6527:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6528:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6530:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6532:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6601:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6608:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6621:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6622:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6662:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6663:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6670:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6678:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6683:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6691:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6706:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6708:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6725:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6745:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6746:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6747:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6748:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6758:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6759:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6783:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6784:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6789:2026-07-21T09:44:30.238436Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Move-Item -LiteralPath 'overlap_cpu_results.json' -Destination 'overlap_cpu_legacy_masks_results.json'; Remove-Item -LiteralPath 'topology_probe.exe','topology_probe.obj' -Force"` rejected: blocked by policy
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6802:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6803:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6804:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6805:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6807:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6809:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6878:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6885:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6898:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6899:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6939:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6940:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6947:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6955:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6960:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6968:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6983:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:6985:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7002:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7022:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7023:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7024:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7025:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7035:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7036:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7060:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7061:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7077:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7078:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7079:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7080:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7082:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7084:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7153:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7160:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7173:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7174:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7214:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7215:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7222:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7230:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7235:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7243:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7258:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7260:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7277:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7297:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7298:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7299:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7300:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7310:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7311:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7335:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7336:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7360:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7361:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7362:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7363:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7365:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7367:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7436:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7443:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7456:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7457:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7497:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7498:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7505:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7513:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7518:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7526:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7541:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7543:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7560:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7580:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7581:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7582:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7583:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7593:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7594:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7618:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7619:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7625:"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Write-Output 'OVERLAP C'; rg -n \"#define N_EMBD|verify_tensor|read_expert_payload|load_quant_expert|quant_expert_forward|ds4_quantize_row|parallel for|silu\\(|overlap_cpu_create|for \\(int i = 0; i < max_tail|apply_omp_affinity|Start the distinct|evict_cache\\(|for \\(int i = 0; i < n_experts|CreateThread\" overlap_spike.c overlap_spike.cu; Write-Output 'CPUGEMV C'; rg -n \"#define N_EMBD|verify_tensor|read_expert_payload|load_quant_expert|quant_expert_forward|ds4_quantize_row|parallel for|silu\\(|evict_cache\\(|double t0|for \\(int i = 0; i < iters|omp_set_num_threads\" 'C:\\Users\\imanu\\g130i\\cpugemv_spike\\cpugemv_spike.c'; Write-Output 'PATCH DIFF'; git -c safe.directory='C:/Users/imanu/g130i' diff -- overlap_spike/overlap_spike.c overlap_spike/overlap_spike.cu overlap_spike/README.md overlap_spike/run_overlap_spike.cmd overlap_spike/overlap_cpu_bench.c overlap_spike/build_overlap_cpu_bench.bat" in C:\Users\imanu\g130i\overlap_spike
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7630:overlap_spike.cu:319:            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7631:overlap_spike.cu:320:            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7648:overlap_spike.c:446:static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7654:overlap_spike.c:540:    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7682:596:    omp_set_num_threads(threads);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7692:692:    omp_set_num_threads(8);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7705:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7706:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7707:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7708:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7710:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7712:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7781:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7788:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7801:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7802:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7842:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7843:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7850:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7858:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7863:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7871:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7886:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7888:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7905:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7925:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7926:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7927:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7928:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7938:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7939:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7963:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7964:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7970:"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$old = Get-Content overlap_spike_results.json | ConvertFrom-Json; $new = Get-Content overlap_cpu_physical_results.json | ConvertFrom-Json; Write-Output '"'OLD PER EXPERT'; "'$old.results | Select-Object mode,tail_experts,cpu_threads,@{n='"'ms_per_expert';e={[math]::Round("'$_.cpu_tail_ms/$_.tail_experts,6)}} | Format-Table -AutoSize; Write-Output '"'NEW PHYSICAL N=24'; "'$new.results | Where-Object experts -eq 24 | Select-Object threads,effective_threads,affinity_mask,mean_ms,@{n='"'mean_ms_per_expert';e={"'$_.mean_ms_per_expert}},@{n='"'GBps';e={[math]::Round("'$new.expert_bytes/1e9/($_.mean_ms_per_expert/1000),3)}} | Format-Table -AutoSize; Write-Output '"'NEW 6/8 ALL N'; "'$new.results | Where-Object threads -in 6,8 | Select-Object threads,experts,mean_ms,mean_ms_per_expert,min_ms_per_expert | Format-Table -AutoSize; Write-Output '"'LEGACY N=24'; "'$legacy = Get-Content overlap_cpu_legacy_masks_results.json | ConvertFrom-Json; $legacy.results | Where-Object { $_.experts -eq 24 -and $_.threads -in 6,8 } | Format-List' in C:\Users\imanu\g130i\overlap_spike
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7974:mode        tail_experts cpu_threads ms_per_expert
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7976:tail_6_of_8            4           6     17,667825
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7977:tail_6_of_8            8           6     17,750250
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7978:tail_6_of_8           16           6     17,881288
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7979:tail_6_of_8           24           6     17,938612
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7988:threads effective_threads affinity_mask    mean_ms mean_ms_per_expert  GBps
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7993:      6                 6 0x555          22,344610           0,931025 7,602
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:7994:      8                 8 0x5555         17,393370           0,724724 9,766
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8045:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8046:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8047:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8048:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8050:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8052:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8121:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8128:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8141:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8142:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8182:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8183:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8190:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8198:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8203:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8211:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8226:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8228:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8245:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8265:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8266:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8267:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8268:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8278:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8279:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8303:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8304:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8310:"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"effective_team_size|physical_masks|overlap_cpu_run_tail|mean_ms_per_expert|evict_mb_once\" overlap_cpu_bench.c overlap_cpu_physical_results.json; rg -n \"tail6_mask|tail8_mask|eviction_scope|overlap_cpu_run_tail\\(cpu|CreateThread|double serial_t0|double overlap_t0\" overlap_spike.cu; rg -n 'cold_ms_mean|warm_ms_mean|compressed_bytes|effective_weight_gb_s' 'C:\\Users\\imanu\\g130i\\cpugemv_spike\\cpugemv_spike_results.json'" in C:\Users\imanu\g130i\overlap_spike
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8313:overlap_cpu_physical_results.json:8:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 1, "repeats": 10, "mean_ms": 4.932320, "min_ms": 4.892500, "mean_ms_per_expert": 4.932320, "min_ms_per_expert": 4.892500},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8314:overlap_cpu_physical_results.json:9:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 4, "repeats": 10, "mean_ms": 19.783050, "min_ms": 19.665900, "mean_ms_per_expert": 4.945763, "min_ms_per_expert": 4.916475},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8315:overlap_cpu_physical_results.json:10:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 8, "repeats": 10, "mean_ms": 39.722360, "min_ms": 39.516900, "mean_ms_per_expert": 4.965295, "min_ms_per_expert": 4.939612},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8316:overlap_cpu_physical_results.json:11:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 16, "repeats": 10, "mean_ms": 79.725660, "min_ms": 79.382200, "mean_ms_per_expert": 4.982854, "min_ms_per_expert": 4.961388},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8317:overlap_cpu_physical_results.json:12:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 24, "repeats": 10, "mean_ms": 121.136720, "min_ms": 119.395300, "mean_ms_per_expert": 5.047363, "min_ms_per_expert": 4.974804},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8318:overlap_cpu_physical_results.json:13:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 1, "repeats": 10, "mean_ms": 2.475220, "min_ms": 2.443900, "mean_ms_per_expert": 2.475220, "min_ms_per_expert": 2.443900},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8319:overlap_cpu_physical_results.json:14:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 4, "repeats": 10, "mean_ms": 9.981070, "min_ms": 9.882900, "mean_ms_per_expert": 2.495267, "min_ms_per_expert": 2.470725},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8320:overlap_cpu_physical_results.json:15:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 8, "repeats": 10, "mean_ms": 20.341890, "min_ms": 19.935100, "mean_ms_per_expert": 2.542736, "min_ms_per_expert": 2.491888},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8321:overlap_cpu_physical_results.json:16:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 16, "repeats": 10, "mean_ms": 40.391660, "min_ms": 40.004300, "mean_ms_per_expert": 2.524479, "min_ms_per_expert": 2.500269},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8322:overlap_cpu_physical_results.json:17:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 24, "repeats": 10, "mean_ms": 60.501720, "min_ms": 60.185400, "mean_ms_per_expert": 2.520905, "min_ms_per_expert": 2.507725},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8323:overlap_cpu_physical_results.json:18:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 1, "repeats": 10, "mean_ms": 1.271920, "min_ms": 1.252300, "mean_ms_per_expert": 1.271920, "min_ms_per_expert": 1.252300},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8324:overlap_cpu_physical_results.json:19:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 4, "repeats": 10, "mean_ms": 5.167490, "min_ms": 5.055300, "mean_ms_per_expert": 1.291873, "min_ms_per_expert": 1.263825},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8325:overlap_cpu_physical_results.json:20:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 8, "repeats": 10, "mean_ms": 10.336620, "min_ms": 10.187300, "mean_ms_per_expert": 1.292078, "min_ms_per_expert": 1.273412},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8326:overlap_cpu_physical_results.json:21:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 16, "repeats": 10, "mean_ms": 20.898360, "min_ms": 20.572000, "mean_ms_per_expert": 1.306148, "min_ms_per_expert": 1.285750},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8327:overlap_cpu_physical_results.json:22:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 24, "repeats": 10, "mean_ms": 31.077480, "min_ms": 30.878900, "mean_ms_per_expert": 1.294895, "min_ms_per_expert": 1.286621},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8328:overlap_cpu_physical_results.json:23:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 1, "repeats": 10, "mean_ms": 0.930510, "min_ms": 0.866700, "mean_ms_per_expert": 0.930510, "min_ms_per_expert": 0.866700},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8329:overlap_cpu_physical_results.json:24:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 4, "repeats": 10, "mean_ms": 3.947700, "min_ms": 3.492600, "mean_ms_per_expert": 0.986925, "min_ms_per_expert": 0.873150},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8330:overlap_cpu_physical_results.json:25:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 8, "repeats": 10, "mean_ms": 7.097150, "min_ms": 7.035200, "mean_ms_per_expert": 0.887144, "min_ms_per_expert": 0.879400},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8331:overlap_cpu_physical_results.json:26:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 16, "repeats": 10, "mean_ms": 14.455450, "min_ms": 14.203200, "mean_ms_per_expert": 0.903466, "min_ms_per_expert": 0.887700},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8332:overlap_cpu_physical_results.json:27:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 24, "repeats": 10, "mean_ms": 22.344610, "min_ms": 21.471000, "mean_ms_per_expert": 0.931025, "min_ms_per_expert": 0.894625},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8333:overlap_cpu_physical_results.json:28:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 1, "repeats": 10, "mean_ms": 0.860120, "min_ms": 0.670400, "mean_ms_per_expert": 0.860120, "min_ms_per_expert": 0.670400},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8334:overlap_cpu_physical_results.json:29:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 4, "repeats": 10, "mean_ms": 2.700100, "min_ms": 2.682000, "mean_ms_per_expert": 0.675025, "min_ms_per_expert": 0.670500},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8335:overlap_cpu_physical_results.json:30:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 8, "repeats": 10, "mean_ms": 5.722670, "min_ms": 5.391700, "mean_ms_per_expert": 0.715334, "min_ms_per_expert": 0.673963},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8336:overlap_cpu_physical_results.json:31:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 16, "repeats": 10, "mean_ms": 11.120630, "min_ms": 10.848300, "mean_ms_per_expert": 0.695039, "min_ms_per_expert": 0.678019},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8337:overlap_cpu_physical_results.json:32:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 24, "repeats": 10, "mean_ms": 17.393370, "min_ms": 16.432500, "mean_ms_per_expert": 0.724724, "min_ms_per_expert": 0.684688}
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8338:overlap_cpu_bench.c:15:int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8340:overlap_cpu_bench.c:77:    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8341:overlap_cpu_bench.c:84:        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8343:overlap_cpu_bench.c:97:                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8348:111:    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8349:112:    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8353:289:        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8356:306:            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8357:319:            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8358:320:            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8374:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8375:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8376:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8377:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8379:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8381:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8450:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8457:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8470:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8471:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8511:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8512:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8519:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8527:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8532:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8540:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8555:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8557:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8574:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8594:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8595:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8596:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8597:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8607:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8608:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8632:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8633:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8650:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8651:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8652:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8653:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8655:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8657:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8726:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8733:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8746:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8747:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8787:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8788:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8795:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8803:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8808:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8816:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8831:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8833:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8850:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8870:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8871:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8872:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8873:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8883:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8884:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8915:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8918:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8925:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8926:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8928:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8929:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:8934:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9007:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9028:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9029:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9032:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9048:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9052:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9053:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9054:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9067:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9083:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9104:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9107:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9119:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9120:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9133:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9134:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9135:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9136:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9138:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9140:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9209:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9216:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9229:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9230:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9270:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9271:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9278:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9286:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9291:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9299:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9314:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9316:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9333:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9353:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9354:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9355:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9356:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9366:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9367:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9398:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9401:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9408:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9409:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9411:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9412:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9417:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9490:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9511:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9512:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9515:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9531:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9535:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9536:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9537:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9550:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9566:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9587:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9590:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9602:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9603:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9620:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9621:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9622:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9623:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9625:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9627:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9635:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9700:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9707:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9720:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9721:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9761:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9762:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9769:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9777:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9782:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9790:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9805:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9807:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9824:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9844:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9845:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9846:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9847:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9857:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9858:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9889:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9892:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9899:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9900:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9902:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9903:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9908:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:9981:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10002:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10003:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10006:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10022:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10026:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10027:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10028:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10041:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10057:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10078:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10081:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10093:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10094:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10107:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10108:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10109:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10110:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10112:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10114:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10122:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10187:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10194:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10207:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10208:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10248:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10249:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10256:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10264:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10269:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10277:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10292:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10294:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10311:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10331:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10332:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10333:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10334:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10344:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10345:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10376:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10379:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10386:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10387:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10389:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10390:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10395:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10468:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10489:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10490:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10493:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10509:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10513:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10514:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10515:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10528:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10544:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10565:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10568:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10580:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10581:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10587:The corrected measurements are stable across N: with six physical cores, 4/8/16/24 experts average 0.987/0.887/0.903/0.931 ms each; with eight physical cores, 0.675/0.715/0.695/0.725 ms each. This confirms a real distinct-expert stream rather than hot reuse. The original 5.58 GB/s was complete-kernel effective throughput, not a hardware DRAM ceiling.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10618:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10619:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10620:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10621:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10623:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10625:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10633:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10698:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10705:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10718:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10719:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10759:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10760:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10767:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10775:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10780:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10788:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10803:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10805:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10822:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10842:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10843:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10844:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10845:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10855:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10856:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10887:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10890:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10897:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10898:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10900:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10901:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10906:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:10979:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11000:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11001:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11004:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11020:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11024:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11025:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11026:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11039:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11055:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11076:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11079:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11091:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11092:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11097:2026-07-21T09:47:33.344716Z ERROR codex_core::tools::router: error=`"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '$workspace = (Resolve-Path -LiteralPath '"'.').Path; "'$targets = @('"'topology_probe.exe','topology_probe.obj') | ForEach-Object { (Resolve-Path -LiteralPath "'$_).Path }; foreach ($target in $targets) { if ((Split-Path -Parent $target) -ne $workspace) { throw "Unexpected target: $target" } }; foreach ($target in $targets) { Remove-Item -LiteralPath $target -Force }'` rejected: blocked by policy
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11110:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11111:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11112:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11113:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11115:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11117:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11125:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11190:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11197:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11210:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11211:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11251:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11252:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11259:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11267:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11272:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11280:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11295:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11297:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11314:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11334:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11335:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11336:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11337:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11347:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11348:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11379:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11382:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11389:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11390:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11392:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11393:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11398:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11471:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11492:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11493:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11496:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11512:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11516:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11517:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11518:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11531:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11547:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11568:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11571:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11583:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11584:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11589:Patched the timed eviction scope and physical-core affinity masks, added and ran a CUDA-free verifier, rebuilt both harnesses without executing GPU code, and wrote the reconciliation report.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11603:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11604:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11605:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11606:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11608:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11610:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11618:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11683:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11690:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11703:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11704:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11744:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11745:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11752:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11760:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11765:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11773:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11788:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11790:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11807:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11827:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11828:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11829:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11830:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11840:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11841:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11872:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11875:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11882:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11883:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11885:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11886:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11891:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11964:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11985:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11986:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:11989:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12005:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12009:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12010:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12011:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12024:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12040:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12061:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12064:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12076:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12077:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12087:- Its affinity masks also placed 6/8 workers on only 3/4 physical cores.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12089:  - 6 physical cores: **0.931 ms/expert** at N=24.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12090:  - 8 physical cores: **0.725 ms/expert** at N=24.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12091:- Conclusion: the CPU tail is viable enough to continue testing; 24 experts take approximately 22.3 ms while reserving two cores.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12106:-- `tail_6_of_8`: CPU OpenMP tail uses 6 workers with affinity mask `0x3f`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12107:-- `tail_8_of_8`: CPU OpenMP tail uses 8 workers with affinity mask `0xff`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12108:+- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12109:+- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12111:+On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12113:+logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12121:+The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12186:+int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12193:+    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12206:+    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12207:+    omp_set_num_threads(requested);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12247:+    /* This target enumerates each core's two SMT siblings adjacently. */
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12248:+    const uint64_t physical_masks[] = {0x1ull, 0x5ull, 0x55ull, 0x555ull, 0x5555ull};
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12255:+        const uint64_t affinity_mask = physical_masks[ti];
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12263:+            row->affinity_mask = affinity_mask;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12268:+                if (overlap_cpu_run_tail(ctx, row->experts, threads, affinity_mask, &ms) != 0) {
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12276:+                   threads, effective, (unsigned long long)affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12291:+        fprintf(f, "    {\"threads\": %d, \"effective_threads\": %d, \"affinity_mask\": \"0x%llx\", \"experts\": %d, \"repeats\": %d, "
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12293:+                row->threads, row->effective_threads, (unsigned long long)row->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12310:     apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12330:-    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x3f)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12331:-    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0xff)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12332:+    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12333:+    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12343:+    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12344:+    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12375:+There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12378:+and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12385:+| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12386:+| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12388:+For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12389:+**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12394:+24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12467:+reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12488:+With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12489:+7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12492:+cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12508:+- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12512:+- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12513:+  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12514:+  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12527:+| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12543:+- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12564:+  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12567:+corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12579:+  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12580:+  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12591:- Its affinity masks also placed 6/8 workers on only 3/4 physical cores.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12593:  - 6 physical cores: **0.931 ms/expert** at N=24.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12594:  - 8 physical cores: **0.725 ms/expert** at N=24.
C:\Users\imanu\g130i\overlap_spike\codex_reconcile.log:12595:- Conclusion: the CPU tail is viable enough to continue testing; 24 experts take approximately 22.3 ms while reserving two cores.
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:441:static DWORD_PTR apply_current_thread_affinity(uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:443:    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:446:static void apply_omp_affinity(int threads, uint64_t mask) {
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:451:        (void)SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)mask);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:530:__declspec(dllexport) int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms) {
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:535:    DWORD_PTR old_thread_mask = apply_current_thread_affinity(affinity_mask);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:537:    omp_set_dynamic(0);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:538:    omp_set_num_threads(omp_threads);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:540:    apply_omp_affinity(omp_threads, affinity_mask);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.c:552:    if (old_thread_mask != 0) (void)SetThreadAffinityMask(GetCurrentThread(), old_thread_mask);
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:8:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 1, "repeats": 10, "mean_ms": 4.932320, "min_ms": 4.892500, "mean_ms_per_expert": 4.932320, "min_ms_per_expert": 4.892500},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:9:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 4, "repeats": 10, "mean_ms": 19.783050, "min_ms": 19.665900, "mean_ms_per_expert": 4.945763, "min_ms_per_expert": 4.916475},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:10:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 8, "repeats": 10, "mean_ms": 39.722360, "min_ms": 39.516900, "mean_ms_per_expert": 4.965295, "min_ms_per_expert": 4.939612},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:11:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 16, "repeats": 10, "mean_ms": 79.725660, "min_ms": 79.382200, "mean_ms_per_expert": 4.982854, "min_ms_per_expert": 4.961388},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:12:    {"threads": 1, "effective_threads": 1, "affinity_mask": "0x1", "experts": 24, "repeats": 10, "mean_ms": 121.136720, "min_ms": 119.395300, "mean_ms_per_expert": 5.047363, "min_ms_per_expert": 4.974804},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:13:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 1, "repeats": 10, "mean_ms": 2.475220, "min_ms": 2.443900, "mean_ms_per_expert": 2.475220, "min_ms_per_expert": 2.443900},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:14:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 4, "repeats": 10, "mean_ms": 9.981070, "min_ms": 9.882900, "mean_ms_per_expert": 2.495267, "min_ms_per_expert": 2.470725},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:15:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 8, "repeats": 10, "mean_ms": 20.341890, "min_ms": 19.935100, "mean_ms_per_expert": 2.542736, "min_ms_per_expert": 2.491888},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:16:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 16, "repeats": 10, "mean_ms": 40.391660, "min_ms": 40.004300, "mean_ms_per_expert": 2.524479, "min_ms_per_expert": 2.500269},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:17:    {"threads": 2, "effective_threads": 2, "affinity_mask": "0x5", "experts": 24, "repeats": 10, "mean_ms": 60.501720, "min_ms": 60.185400, "mean_ms_per_expert": 2.520905, "min_ms_per_expert": 2.507725},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:18:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 1, "repeats": 10, "mean_ms": 1.271920, "min_ms": 1.252300, "mean_ms_per_expert": 1.271920, "min_ms_per_expert": 1.252300},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:19:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 4, "repeats": 10, "mean_ms": 5.167490, "min_ms": 5.055300, "mean_ms_per_expert": 1.291873, "min_ms_per_expert": 1.263825},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:20:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 8, "repeats": 10, "mean_ms": 10.336620, "min_ms": 10.187300, "mean_ms_per_expert": 1.292078, "min_ms_per_expert": 1.273412},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:21:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 16, "repeats": 10, "mean_ms": 20.898360, "min_ms": 20.572000, "mean_ms_per_expert": 1.306148, "min_ms_per_expert": 1.285750},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:22:    {"threads": 4, "effective_threads": 4, "affinity_mask": "0x55", "experts": 24, "repeats": 10, "mean_ms": 31.077480, "min_ms": 30.878900, "mean_ms_per_expert": 1.294895, "min_ms_per_expert": 1.286621},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:23:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 1, "repeats": 10, "mean_ms": 0.930510, "min_ms": 0.866700, "mean_ms_per_expert": 0.930510, "min_ms_per_expert": 0.866700},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:24:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 4, "repeats": 10, "mean_ms": 3.947700, "min_ms": 3.492600, "mean_ms_per_expert": 0.986925, "min_ms_per_expert": 0.873150},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:25:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 8, "repeats": 10, "mean_ms": 7.097150, "min_ms": 7.035200, "mean_ms_per_expert": 0.887144, "min_ms_per_expert": 0.879400},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:26:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 16, "repeats": 10, "mean_ms": 14.455450, "min_ms": 14.203200, "mean_ms_per_expert": 0.903466, "min_ms_per_expert": 0.887700},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:27:    {"threads": 6, "effective_threads": 6, "affinity_mask": "0x555", "experts": 24, "repeats": 10, "mean_ms": 22.344610, "min_ms": 21.471000, "mean_ms_per_expert": 0.931025, "min_ms_per_expert": 0.894625},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:28:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 1, "repeats": 10, "mean_ms": 0.860120, "min_ms": 0.670400, "mean_ms_per_expert": 0.860120, "min_ms_per_expert": 0.670400},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:29:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 4, "repeats": 10, "mean_ms": 2.700100, "min_ms": 2.682000, "mean_ms_per_expert": 0.675025, "min_ms_per_expert": 0.670500},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:30:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 8, "repeats": 10, "mean_ms": 5.722670, "min_ms": 5.391700, "mean_ms_per_expert": 0.715334, "min_ms_per_expert": 0.673963},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:31:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 16, "repeats": 10, "mean_ms": 11.120630, "min_ms": 10.848300, "mean_ms_per_expert": 0.695039, "min_ms_per_expert": 0.678019},
C:\Users\imanu\g130i\overlap_spike\overlap_cpu_physical_results.json:32:    {"threads": 8, "effective_threads": 8, "affinity_mask": "0x5555", "experts": 24, "repeats": 10, "mean_ms": 17.393370, "min_ms": 16.432500, "mean_ms_per_expert": 0.724724, "min_ms_per_expert": 0.684688}
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:14:int overlap_cpu_run_tail(OverlapCpuTail *ctx, int n_experts, int omp_threads, uint64_t affinity_mask, double *tail_ms);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:33:    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:34:    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:50:    int cpu_threads;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:51:    uint64_t affinity_mask;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:98:    printf("  --tail6-mask MASK   affinity mask for 6-thread CPU tail (default 0x555)\n");
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:99:    printf("  --tail8-mask MASK   affinity mask for 8-thread CPU tail (default 0x5555)\n");
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:111:    c.tail6_mask = 0x555ull;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:112:    c.tail8_mask = 0x5555ull;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:207:    job->rc = overlap_cpu_run_tail(job->ctx, job->n_tail, job->cpu_threads, job->affinity_mask, &job->tail_ms);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:239:        fprintf(f, ", \"tail_experts\": %d, \"cpu_threads\": %d, \"affinity_mask\": \"0x%llx\", "
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:244:                r->n_tail, r->cpu_threads, (unsigned long long)r->affinity_mask,
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:289:        {"tail_6_of_8", 6, cfg.tail6_mask},
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:299:            r->cpu_threads = modes[mi].threads;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:300:            r->affinity_mask = modes[mi].mask;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:306:            rc = overlap_cpu_run_tail(cpu, r->n_tail, r->cpu_threads, r->affinity_mask, &r->cpu_tail_ms);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:314:            job.cpu_threads = r->cpu_threads;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:315:            job.affinity_mask = r->affinity_mask;
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:319:            HANDLE th = CreateThread(NULL, 0, cpu_job_thread, &job, 0, NULL);
C:\Users\imanu\g130i\overlap_spike\overlap_spike.cu:320:            if (!th) fail("CreateThread failed");
C:\Users\imanu\g130i\overlap_spike\README.md:30:- `tail_6_of_8`: CPU OpenMP tail uses 6 workers on six physical cores with affinity mask `0x555`, leaving two cores free for the CUDA submit thread.
C:\Users\imanu\g130i\overlap_spike\README.md:31:- `tail_8_of_8`: CPU OpenMP tail uses 8 workers on eight physical cores with affinity mask `0x5555`.
C:\Users\imanu\g130i\overlap_spike\README.md:33:On this 8-core/16-thread target, each physical core's SMT siblings occupy adjacent
C:\Users\imanu\g130i\overlap_spike\README.md:35:logical processor per physical core.
C:\Users\imanu\g130i\overlap_spike\README.md:53:The checked-in `overlap_spike_results.json` predates the eviction/affinity fix and
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:12:There was also a separate affinity bug. On this 8-core/16-thread CPU, SMT sibling
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:15:and 4 physical cores. They did not mean "6 of 8 cores" and "8 of 8 cores."
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:22:| 6 workers on 6 physical cores (`0x555`) | 22.345 ms | **0.931 ms** | 7.60 GB/s |
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:23:| 8 workers on 8 physical cores (`0x5555`) | 17.393 ms | **0.725 ms** | 9.77 GB/s |
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:25:For planning, use **0.9--1.0 ms/expert when reserving two cores** and
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:26:**0.7--0.8 ms/expert when using all eight physical cores**. A conservative range
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:31:24 distinct experts cost about 22 ms with six physical cores. A corrected CPU/GPU
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:104:reuse to preserve, so every subsequent payload is naturally cold.
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:125:With correct physical-core affinity, the complete distinct-expert forward reaches
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:126:7.60 GB/s on six cores and 9.77 GB/s on eight. That is why the corrected result can
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:129:cores, not the raw DRAM floor.
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:145:- There is no explicit `CreateThread` inside the expert loop. Each expert enters
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:149:- The old `0x3f`/`0xff` masks were an affinity-layout mistake, not a team-size
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:150:  fallback. Defaults are now `0x555`/`0x5555`, one logical CPU from each physical
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:151:  core (`overlap_spike.cu:103-112`).
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:164:| N distinct experts | 6 physical cores total / per expert | 8 physical cores total / per expert |
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:180:- corrected default physical-core masks and documented the SMT topology
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:201:  row-parallel code already reaches 0.7--1.0 ms/expert on the intended core sets.
C:\Users\imanu\g130i\overlap_spike\reconcile_report.md:204:corrected overlap run measuring whether the approximately 22 ms six-core tail
C:\Users\imanu\g130i\overlap_spike\run_overlap_spike.cmd:18:  --tail6-mask 0x555 ^
C:\Users\imanu\g130i\overlap_spike\run_overlap_spike.cmd:19:  --tail8-mask 0x5555 ^
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:7:    {"mode": "tail_6_of_8", "tail_experts": 4, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 93.976100, "serial_gpu_wall_ms": 74.954600, "serial_gpu_event_ms": 73.064445, "serial_gpu_enqueue_ms": 4.148200, "cpu_tail_ms": 3.467100, "overlapped_wall_ms": 72.403300, "overlapped_gpu_event_ms": 72.218620, "overlapped_enqueue_ms": 0.100100, "overlap_efficiency": 6.222145, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:8:    {"mode": "tail_6_of_8", "tail_experts": 8, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 103.053300, "serial_gpu_wall_ms": 73.273000, "serial_gpu_event_ms": 73.224800, "serial_gpu_enqueue_ms": 0.071300, "cpu_tail_ms": 13.959200, "overlapped_wall_ms": 72.769100, "overlapped_gpu_event_ms": 72.523773, "overlapped_enqueue_ms": 0.132300, "overlap_efficiency": 2.169480, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:9:    {"mode": "tail_6_of_8", "tail_experts": 16, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 101.971300, "serial_gpu_wall_ms": 72.220200, "serial_gpu_event_ms": 72.157181, "serial_gpu_enqueue_ms": 0.074000, "cpu_tail_ms": 14.472500, "overlapped_wall_ms": 72.396400, "overlapped_gpu_event_ms": 72.196098, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 2.043524, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:10:    {"mode": "tail_6_of_8", "tail_experts": 24, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 109.552100, "serial_gpu_wall_ms": 72.225500, "serial_gpu_event_ms": 72.178497, "serial_gpu_enqueue_ms": 0.077700, "cpu_tail_ms": 21.759800, "overlapped_wall_ms": 79.089300, "overlapped_gpu_event_ms": 72.183807, "overlapped_enqueue_ms": 0.105700, "overlap_efficiency": 1.399958, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:11:    {"mode": "tail_8_of_8", "tail_experts": 4, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 90.972200, "serial_gpu_wall_ms": 72.199700, "serial_gpu_event_ms": 72.139618, "serial_gpu_enqueue_ms": 0.094400, "cpu_tail_ms": 2.687100, "overlapped_wall_ms": 73.273900, "overlapped_gpu_event_ms": 73.141052, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 6.586394, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:12:    {"mode": "tail_8_of_8", "tail_experts": 8, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 94.499400, "serial_gpu_wall_ms": 72.194300, "serial_gpu_event_ms": 72.159233, "serial_gpu_enqueue_ms": 0.081600, "cpu_tail_ms": 6.887300, "overlapped_wall_ms": 72.383800, "overlapped_gpu_event_ms": 72.214272, "overlapped_enqueue_ms": 0.082800, "overlap_efficiency": 3.211070, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:13:    {"mode": "tail_8_of_8", "tail_experts": 16, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 99.722800, "serial_gpu_wall_ms": 72.224500, "serial_gpu_event_ms": 72.183807, "serial_gpu_enqueue_ms": 0.088500, "cpu_tail_ms": 11.581400, "overlapped_wall_ms": 72.333800, "overlapped_gpu_event_ms": 72.174591, "overlapped_enqueue_ms": 0.117300, "overlap_efficiency": 2.364913, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_fixed.json:14:    {"mode": "tail_8_of_8", "tail_experts": 24, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 106.233600, "serial_gpu_wall_ms": 73.608800, "serial_gpu_event_ms": 73.571198, "serial_gpu_enqueue_ms": 0.076600, "cpu_tail_ms": 16.771800, "overlapped_wall_ms": 72.448300, "overlapped_gpu_event_ms": 72.186882, "overlapped_enqueue_ms": 0.105500, "overlap_efficiency": 2.014411, "pass": true}
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:7:    {"mode": "tail_6_of_8", "tail_experts": 4, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 241.637300, "serial_gpu_wall_ms": 170.037700, "serial_gpu_event_ms": 168.190979, "serial_gpu_enqueue_ms": 99.213600, "cpu_tail_ms": 70.671300, "overlapped_wall_ms": 78.011000, "overlapped_gpu_event_ms": 72.544128, "overlapped_enqueue_ms": 0.132000, "overlap_efficiency": 2.315315, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:8:    {"mode": "tail_6_of_8", "tail_experts": 8, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 214.295000, "serial_gpu_wall_ms": 72.224200, "serial_gpu_event_ms": 72.169472, "serial_gpu_enqueue_ms": 0.087600, "cpu_tail_ms": 142.002000, "overlapped_wall_ms": 150.206700, "overlapped_gpu_event_ms": 72.023041, "overlapped_enqueue_ms": 0.098300, "overlap_efficiency": 0.451320, "pass": false},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:9:    {"mode": "tail_6_of_8", "tail_experts": 16, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 358.479500, "serial_gpu_wall_ms": 72.323500, "serial_gpu_event_ms": 72.253281, "serial_gpu_enqueue_ms": 0.091300, "cpu_tail_ms": 286.100600, "overlapped_wall_ms": 294.133000, "overlapped_gpu_event_ms": 72.168449, "overlapped_enqueue_ms": 0.110200, "overlap_efficiency": 0.224909, "pass": false},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:10:    {"mode": "tail_6_of_8", "tail_experts": 24, "cpu_threads": 6, "affinity_mask": "0x3f", "serial_wall_ms": 502.735200, "serial_gpu_wall_ms": 72.131700, "serial_gpu_event_ms": 72.071106, "serial_gpu_enqueue_ms": 0.095800, "cpu_tail_ms": 430.526700, "overlapped_wall_ms": 435.978800, "overlapped_gpu_event_ms": 72.129539, "overlapped_enqueue_ms": 0.085900, "overlap_efficiency": 0.155058, "pass": false},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:11:    {"mode": "tail_8_of_8", "tail_experts": 4, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 144.416100, "serial_gpu_wall_ms": 73.264700, "serial_gpu_event_ms": 73.142273, "serial_gpu_enqueue_ms": 0.087900, "cpu_tail_ms": 70.274000, "overlapped_wall_ms": 78.062100, "overlapped_gpu_event_ms": 73.847618, "overlapped_enqueue_ms": 0.130400, "overlap_efficiency": 0.944218, "pass": true},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:12:    {"mode": "tail_8_of_8", "tail_experts": 8, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 211.284200, "serial_gpu_wall_ms": 72.061600, "serial_gpu_event_ms": 72.010750, "serial_gpu_enqueue_ms": 0.088400, "cpu_tail_ms": 139.160200, "overlapped_wall_ms": 150.041000, "overlapped_gpu_event_ms": 72.645630, "overlapped_enqueue_ms": 0.133200, "overlap_efficiency": 0.440091, "pass": false},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:13:    {"mode": "tail_8_of_8", "tail_experts": 16, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 351.609300, "serial_gpu_wall_ms": 72.196800, "serial_gpu_event_ms": 72.076286, "serial_gpu_enqueue_ms": 0.089400, "cpu_tail_ms": 279.308600, "overlapped_wall_ms": 285.358300, "overlapped_gpu_event_ms": 72.093697, "overlapped_enqueue_ms": 0.173800, "overlap_efficiency": 0.237196, "pass": false},
C:\Users\imanu\g130i\overlap_spike\overlap_spike_results.json:14:    {"mode": "tail_8_of_8", "tail_experts": 24, "cpu_threads": 8, "affinity_mask": "0xff", "serial_wall_ms": 494.212900, "serial_gpu_wall_ms": 72.074100, "serial_gpu_event_ms": 72.014847, "serial_gpu_enqueue_ms": 0.093900, "cpu_tail_ms": 422.056800, "overlapped_wall_ms": 421.842400, "overlapped_gpu_event_ms": 72.927231, "overlapped_enqueue_ms": 0.087000, "overlap_efficiency": 0.171471, "pass": false}

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -C 2 '\"global\"|\"5600\"|\"fallback_route_pct\": 2\\.5819605943152455|\"fallback_gate_mass_pct\": 1\\.2866795430486557' 'C:\\Users\\imanu\\g130i\\coverage\\coverage_curve_real.json'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 142ms:
161-        }
162-      },
163:      "5600": {
164-        "fallback_gate_mass": 159.342394502,
165-        "fallback_gate_mass_pct": 1.2866795430486557,
--
175-          "routes": 3
176-        },
177:        "resident_count": 5600,
178-        "worst_layer_by_mass_pct": {
179-          "fallback_gate_mass": 5.6856262051,
--
313-        }
314-      },
315:      "5600": {
316-        "fallback_gate_mass": 159.342394502,
317-        "fallback_gate_mass_pct": 1.2866795430486557,
--
327-          "routes": 3
328-        },
329:        "resident_count": 5600,
330-        "worst_layer_by_mass_pct": {
331-          "fallback_gate_mass": 5.6856262051,
--
465-        }
466-      },
467:      "5600": {
468-        "fallback_gate_mass": 160.2924378077,
469-        "fallback_gate_mass_pct": 1.2943510813751316,
--
479-          "routes": 2
480-        },
481:        "resident_count": 5600,
482-        "worst_layer_by_mass_pct": {
483-          "fallback_gate_mass": 5.7787257758,
--
526-      }
527-    },
528:    "global": {
529-      "3000": {
530-        "fallback_gate_mass": 1444.40624567307,
--
617-        }
618-      },
619:      "5600": {
620-        "fallback_gate_mass": 159.342394502,
621-        "fallback_gate_mass_pct": 1.2866795430486557,
--
631-          "routes": 3
632-        },
633:        "resident_count": 5600,
634-        "worst_layer_by_mass_pct": {
635-          "fallback_gate_mass": 5.6856262051,

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n 'tail_6_of_8|tail_8_of_8|pass_criterion|cpu_threads|affinity_mask' 'C:\\Users\\imanu\\g130i\\overlap_spike\\overlap_spike_fixed.json'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 137ms:
4:  "pass_criterion": {"overlap_efficiency_min": 0.70, "submission_stall_ms_max": 2.0},
7:    {"mode": "tail_6_of_8", "tail_experts": 4, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 93.976100, "serial_gpu_wall_ms": 74.954600, "serial_gpu_event_ms": 73.064445, "serial_gpu_enqueue_ms": 4.148200, "cpu_tail_ms": 3.467100, "overlapped_wall_ms": 72.403300, "overlapped_gpu_event_ms": 72.218620, "overlapped_enqueue_ms": 0.100100, "overlap_efficiency": 6.222145, "pass": true},
8:    {"mode": "tail_6_of_8", "tail_experts": 8, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 103.053300, "serial_gpu_wall_ms": 73.273000, "serial_gpu_event_ms": 73.224800, "serial_gpu_enqueue_ms": 0.071300, "cpu_tail_ms": 13.959200, "overlapped_wall_ms": 72.769100, "overlapped_gpu_event_ms": 72.523773, "overlapped_enqueue_ms": 0.132300, "overlap_efficiency": 2.169480, "pass": true},
9:    {"mode": "tail_6_of_8", "tail_experts": 16, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 101.971300, "serial_gpu_wall_ms": 72.220200, "serial_gpu_event_ms": 72.157181, "serial_gpu_enqueue_ms": 0.074000, "cpu_tail_ms": 14.472500, "overlapped_wall_ms": 72.396400, "overlapped_gpu_event_ms": 72.196098, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 2.043524, "pass": true},
10:    {"mode": "tail_6_of_8", "tail_experts": 24, "cpu_threads": 6, "affinity_mask": "0x555", "serial_wall_ms": 109.552100, "serial_gpu_wall_ms": 72.225500, "serial_gpu_event_ms": 72.178497, "serial_gpu_enqueue_ms": 0.077700, "cpu_tail_ms": 21.759800, "overlapped_wall_ms": 79.089300, "overlapped_gpu_event_ms": 72.183807, "overlapped_enqueue_ms": 0.105700, "overlap_efficiency": 1.399958, "pass": true},
11:    {"mode": "tail_8_of_8", "tail_experts": 4, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 90.972200, "serial_gpu_wall_ms": 72.199700, "serial_gpu_event_ms": 72.139618, "serial_gpu_enqueue_ms": 0.094400, "cpu_tail_ms": 2.687100, "overlapped_wall_ms": 73.273900, "overlapped_gpu_event_ms": 73.141052, "overlapped_enqueue_ms": 0.090200, "overlap_efficiency": 6.586394, "pass": true},
12:    {"mode": "tail_8_of_8", "tail_experts": 8, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 94.499400, "serial_gpu_wall_ms": 72.194300, "serial_gpu_event_ms": 72.159233, "serial_gpu_enqueue_ms": 0.081600, "cpu_tail_ms": 6.887300, "overlapped_wall_ms": 72.383800, "overlapped_gpu_event_ms": 72.214272, "overlapped_enqueue_ms": 0.082800, "overlap_efficiency": 3.211070, "pass": true},
13:    {"mode": "tail_8_of_8", "tail_experts": 16, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 99.722800, "serial_gpu_wall_ms": 72.224500, "serial_gpu_event_ms": 72.183807, "serial_gpu_enqueue_ms": 0.088500, "cpu_tail_ms": 11.581400, "overlapped_wall_ms": 72.333800, "overlapped_gpu_event_ms": 72.174591, "overlapped_enqueue_ms": 0.117300, "overlap_efficiency": 2.364913, "pass": true},
14:    {"mode": "tail_8_of_8", "tail_experts": 24, "cpu_threads": 8, "affinity_mask": "0x5555", "serial_wall_ms": 106.233600, "serial_gpu_wall_ms": 73.608800, "serial_gpu_event_ms": 73.571198, "serial_gpu_enqueue_ms": 0.076600, "cpu_tail_ms": 16.771800, "overlapped_wall_ms": 72.448300, "overlapped_gpu_event_ms": 72.186882, "overlapped_enqueue_ms": 0.105500, "overlap_efficiency": 2.014411, "pass": true}

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\coverage\\coverage_curve_real.json' 619 648" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 524ms:
   619:       "5600": {
   620:         "fallback_gate_mass": 159.342394502,
   621:         "fallback_gate_mass_pct": 1.2866795430486557,
   622:         "fallback_route_count": 1279,
   623:         "fallback_route_pct": 2.5819605943152455,
   624:         "floor": null,
   625:         "floor_candidate_count": null,
   626:         "floor_satisfied": null,
   627:         "highest_mass_missed_expert": {
   628:           "expert": 216,
   629:           "layer": 23,
   630:           "mass": 0.20232175689999998,
   631:           "routes": 3
   632:         },
   633:         "resident_count": 5600,
   634:         "worst_layer_by_mass_pct": {
   635:           "fallback_gate_mass": 5.6856262051,
   636:           "fallback_gate_mass_pct": 1.9741757639419166,
   637:           "fallback_route_count": 51,
   638:           "fallback_route_pct": 4.427083333333333,
   639:           "layer": 35
   640:         },
   641:         "worst_layer_by_route_pct": {
   642:           "fallback_gate_mass": 5.5760508798,
   643:           "fallback_gate_mass_pct": 1.9361287866777181,
   644:           "fallback_route_count": 52,
   645:           "fallback_route_pct": 4.513888888888889,
   646:           "layer": 33
   647:         }
   648:       },

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"7,077,888|39,636,172,800|36\\.914|42,467,328|K=5,600|5,600|0\\.7|0\\.9\" 'C:\\Users\\imanu\\g130i\\reap_revisited\\reap_revisited_design.md'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 141ms:
8:**GO for a bounded prototype, with two gates before full integration.** Use **K = 5,600** as the first target, store the resident arena in pageable RAM, retain only a six-expert pinned transfer slab, and do not create a duplicate 320-expert host mirror. First restore/recapture the route trace and then prove one-layer asynchronous CPU/GPU join behavior. Those two gates are important because the requested K-specific coverage curve cannot be measured from the files currently present, and per-token arithmetic alone overstates overlap across the 43 sequential layer barriers.
10:Conditional on the proposed tail being **5–15% of 258 routes/token**, K=5,600 projects to:
17:These are sensitivity bounds, **not replay-derived K=5,600 predictions**, because the route rows are missing. The design still merits a prototype: it restores exact weights, K=5,600 fits the stated 38–42 GiB package budget, and even a 15% fully serial tail remains above the broken-Q1 4.09 t/s baseline. The likely make-or-break issue is not GEMV speed but whether activation D2H, CPU work, output H2D, and the mandatory per-layer join can actually overlap without starving the CUDA submission/route-worker threads.
50:A reasonable minimum is **20 diverse prompts × 256 decoded tokens = 5,120 tokens = 1,320,960 route rows**. A firmer sizing run is **50 prompts × 512 tokens = 25,600 tokens = 6,604,800 rows**, stratified across code, prose, dialogue, multilingual, factual, and long-context continuations. Rank on a training subset, report coverage on held-out prompts, and bootstrap by whole prompt (not individual route row) for 95% intervals. Do not freeze the production manifest until the K=5,000/5,600 fallback-count and fallback-mass intervals are narrow enough to make the latency decision (roughly ±1 percentage point is a useful target).
64:| hot | 10.7 |
95:At each requested K (3,000/4,000/5,000/5,600), the CSV/JSON applies these formulas once fallback count is known. No K-specific t/s value is defensible from the currently available logs.
105:`77,913,391,104 bytes / 11,008 = 7,077,888 bytes/expert`
107:This agrees with the tensor geometry: gate 2,162,688 + up 2,162,688 + down 2,752,512 = 7,077,888 bytes. It is therefore safe to use 7,077,888, not a rounded MiB estimate.
114:| 5,600 | 39,636,172,800 | 36.914062500 |
118:- Six-expert pinned staging slab: **42,467,328 bytes (0.03955078125 GiB)**.
130:| 5,600 | 39,678,640,128 | 41,943,564,288 | 39.062988281 | 888,549,376 bytes (0.828 GiB) |
137:Those are arithmetic maxima, not good operating points. **K=5,600 is the sustainable recommendation**: it fits the stated 38–42 GiB package budget even under duplicate-mirror accounting. In the preferred deduplicated layout it uses 39,678,640,128 bytes (36.954 GiB). Concretely, the remaining physical headroom is **3,153,473,536 bytes (2.937 GiB)** after the 24-GiB floor.
139:Do not attempt to pin the full 36.9-GiB arena. The current dynamic arena already supports pinned/pageable slots (`ds4_cuda.cu:954-1014`) and the Q1 bootstrap had to split 26,304,970,752 pinned + 12,651,724,800 pageable bytes. The keep-K arena should be committed/touched pageable RAM; copy selected resident experts into the 42,467,328-byte pinned slab before H2D. Large pinned allocations are a stability risk on a 64-GiB Windows host.
145:The remaining issue is numerical, not quantization quality. A per-expert CPU/GPU cosine of 0.9998 with comparable norms corresponds to about `sqrt(2 × (1 - 0.9998)) = 0.02`, or 2%, relative difference for that expert output. If CPU routes are 5–15% of routed contributions, a reasonable aggregate perturbation envelope is roughly 0.1–0.3% when errors scale with the norm fraction, or 0.45–0.77% under a more conservative energy-fraction model. Both are tiny compared with Q1's 0.55–0.60 activation cosine, but autoregressive decoding can still flip near-tied logits.
186:1. a restored, held-out replay set shows K=5,600 fallback at or below about 15% by count and no pathological high-mass misses;
188:3. K=5,600 starts reliably with a pageable arena and at least 2 GiB measured available-RAM margin under the real server workload (drop K if the conservative duplicate mirror cannot be eliminated);
191:If any of (1)–(3) fails, the architecture is **NO-GO at K=5,600 on 64 GiB**, though a smaller resident set plus a more strongly batched CPU tail may still be worth revisiting.

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"38,956,695,552|36\\.28|36 GiB|Q1 arena|Q1_0\" 'C:\\Users\\imanu\\g130i\\reap_revisited\\reap_revisited_design.md' 'C:\\Users\\imanu\\g130i\\overlap_spike\\reconcile_report.md'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 142ms:
C:\Users\imanu\g130i\reap_revisited\reap_revisited_design.md:172:| Retire Q1 tiers | Remove Q1 sidecar installation from this mode, Q1 arena/snapshot/probation/promotion branches, Q1 cold launch, and Q1-specific invariants/telemetry. Keep a separate legacy mode until parity is demonstrated. | 1.0–2.0 d |

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"q1_0_mixed_cold\" 'C:\\Users\\imanu\\g130i\\ds4.c' | Select-Object -First 80" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 574ms:
2487:static int q1_0_mixed_cold_one_requested(void) {
11164:    const bool q1_0_mixed_cold =
11786:        !q1_0_mixed_cold &&
11832:    if (ok && q1_0_mixed_cold) {
19495:    const int q1_0_mixed_cold_one = q1_0_mixed_cold_one_requested();
19503:        q1_0_dual_sparse_companion < 0 || q1_0_mixed_cold_one < 0 ||
19509:          q1_0_mixed_cold_one <= 0)) ||
19510:        (q1_0_mixed_cold_one > 0 && q1_0_dual_sparse_companion <= 0) ||
19518:          q1_0_dual_sparse_companion > 0 || q1_0_mixed_cold_one > 0 ||
20132:        const int q1_0_mixed_cold_one =
20133:            q1_0_mixed_cold_one_requested();
20150:            q1_0_mixed_cold_one < 0 ||
20156:              q1_0_mixed_cold_one <= 0)) ||
20157:            (q1_0_mixed_cold_one > 0 &&
20167:              q1_0_mixed_cold_one > 0 ||

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 11140 11182; Show-Lines 'C:\\Users\\imanu\\g130i\\ds4.c' 11825 11950" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 634ms:
 11140:         ds4_gpu_graph  *g,
 11141:         const ds4_model        *model,
 11142:         const ds4_layer_weights *layer,
 11143:         uint32_t                il,
 11144:         uint32_t                pos,
 11145:         ds4_gpu_tensor       *raw_cache,
 11146:         uint32_t                raw_cap,
 11147:         uint32_t                raw_row,
 11148:         uint32_t                n_raw,
 11149:         int                     token) {
 11150:     const uint64_t hc_dim = (uint64_t)DS4_N_HC * DS4_N_EMBD;
 11151:     const uint64_t mix_hc = 2ull * DS4_N_HC + (uint64_t)DS4_N_HC * DS4_N_HC;
 11152:     const uint64_t q_rank = layer->attn_q_a->dim[1];
 11153:     const uint64_t q_dim = (uint64_t)DS4_N_HEAD * DS4_N_HEAD_DIM;
 11154:     const uint32_t n_groups = DS4_N_OUT_GROUP;
 11155:     const uint32_t group_heads = DS4_N_HEAD / n_groups;
 11156:     const uint32_t group_dim = DS4_N_HEAD_DIM * group_heads;
 11157:     const uint32_t rank = DS4_N_LORA_O;
 11158:     const uint32_t shared_dim = (uint32_t)layer->ffn_gate_shexp->dim[1];
 11159:     const ds4_routed_expert_source route =
 11160:         routed_expert_source(model, layer, il);
 11161:     const bool iq1_mixed_cold_one =
 11162:         iq1_s_mixed_cold_one_requested() &&
 11163:         iq1_s_sidecar_layer_active(model, il);
 11164:     const bool q1_0_mixed_cold =
 11165:         (q1_0_dual_arena_requested() > 0 ||
 11166:          q1_0_snapshot_backing_requested() > 0) &&
 11167:         q1_0_sidecar_layer_active(model, il);
 11168:     const uint64_t expert_in_dim = route.gate->dim[0];
 11169:     const uint64_t expert_mid_dim = route.gate->dim[1];
 11170:     const uint64_t down_in_dim = route.down->dim[0];
 11171:     const uint64_t routed_out_dim = route.down->dim[1];
 11172:     const bool compressed = ds4_layer_compress_ratio(il) != 0;
 11173:     const float freq_base = layer_rope_freq_base(il);
 11174:     const float freq_scale = layer_rope_freq_scale(il);
 11175:     const float ext_factor = compressed && DS4_ROPE_SCALE_FACTOR > 1.0f ? 1.0f : 0.0f;
 11176:     float attn_factor = 1.0f;
 11177:     if (ext_factor != 0.0f && freq_scale > 0.0f) {
 11178:         attn_factor /= 1.0f + 0.1f * logf(1.0f / freq_scale);
 11179:     }
 11180:     const bool qkv_rms_fused = !metal_graph_use_reference_qkv_norm();
 11181: 
 11182:     bool ok = true;
 11825:         if (ok && overlap_shared_full) {
 11826:             ok = ds4_gpu_matmul_q8_0_tensor(g->shared_out, model->map, model->size,
 11827:                                               layer->ffn_down_shexp->abs_offset,
 11828:                                               shared_dim, DS4_N_EMBD,
 11829:                                               g->shared_mid, 1) != 0;
 11830:         }
 11831:     }
 11832:     if (ok && q1_0_mixed_cold) {
 11833:         const ds4_tensor *q1_gate = g_q1_0_sidecar.gate[il];
 11834:         const ds4_tensor *q1_up = g_q1_0_sidecar.up[il];
 11835:         const ds4_tensor *q1_down = g_q1_0_sidecar.down[il];
 11836:         const uint64_t q1_gate_row_bytes = routed_expert_row_bytes(q1_gate);
 11837:         const uint64_t q1_gate_expert_bytes =
 11838:             expert_mid_dim * q1_gate_row_bytes;
 11839:         const uint64_t q1_down_row_bytes = routed_expert_row_bytes(q1_down);
 11840:         const uint64_t q1_down_expert_bytes =
 11841:             routed_out_dim * q1_down_row_bytes;
 11842:         ok = ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
 11843:                 g->routed_out,
 11844:                 g->routed_gate,
 11845:                 g->routed_up,
 11846:                 g->routed_mid,
 11847:                 g->routed_down,
 11848:                 route.model->map,
 11849:                 route.model->size,
 11850:                 il,
 11851:                 route.gate->abs_offset,
 11852:                 route.up->abs_offset,
 11853:                 route.down->abs_offset,
 11854:                 route.gate->type,
 11855:                 route.down->type,
 11856:                 gate_expert_bytes,
 11857:                 gate_row_bytes,
 11858:                 down_expert_bytes,
 11859:                 down_row_bytes,
 11860:                 g_q1_0_sidecar.model->map,
 11861:                 g_q1_0_sidecar.model->size,
 11862:                 q1_gate->abs_offset,
 11863:                 q1_up->abs_offset,
 11864:                 q1_down->abs_offset,
 11865:                 q1_gate_expert_bytes,
 11866:                 q1_gate_row_bytes,
 11867:                 q1_down_expert_bytes,
 11868:                 q1_down_row_bytes,
 11869:                 (uint32_t)expert_in_dim,
 11870:                 (uint32_t)down_in_dim,
 11871:                 (uint32_t)routed_out_dim,
 11872:                 g->router_selected,
 11873:                 g->router_weights,
 11874:                 DS4_N_EXPERT_USED,
 11875:                 DS4_SWIGLU_CLAMP_EXP,
 11876:                 g->ffn_norm) != 0;
 11877:     } else if (ok && iq1_mixed_cold_one) {
 11878:         const ds4_tensor *iq1_gate = g_iq1_s_sidecar.gate[il];
 11879:         const ds4_tensor *iq1_up = g_iq1_s_sidecar.up[il];
 11880:         const ds4_tensor *iq1_down = g_iq1_s_sidecar.down[il];
 11881:         const uint64_t iq1_gate_row_bytes = routed_expert_row_bytes(iq1_gate);
 11882:         const uint64_t iq1_gate_expert_bytes = expert_mid_dim * iq1_gate_row_bytes;
 11883:         const uint64_t iq1_down_row_bytes = routed_expert_row_bytes(iq1_down);
 11884:         const uint64_t iq1_down_expert_bytes = routed_out_dim * iq1_down_row_bytes;
 11885:         ok = ds4_gpu_routed_moe_mixed_iq1_one_tensor(
 11886:                 g->routed_out,
 11887:                 g->routed_gate,
 11888:                 g->routed_up,
 11889:                 g->routed_mid,
 11890:                 g->routed_down,
 11891:                 route.model->map,
 11892:                 route.model->size,
 11893:                 il,
 11894:                 route.gate->abs_offset,
 11895:                 route.up->abs_offset,
 11896:                 route.down->abs_offset,
 11897:                 route.gate->type,
 11898:                 route.down->type,
 11899:                 gate_expert_bytes,
 11900:                 gate_row_bytes,
 11901:                 down_expert_bytes,
 11902:                 down_row_bytes,
 11903:                 g_iq1_s_sidecar.model->map,
 11904:                 g_iq1_s_sidecar.model->size,
 11905:                 iq1_gate->abs_offset,
 11906:                 iq1_up->abs_offset,
 11907:                 iq1_down->abs_offset,
 11908:                 iq1_down->type,
 11909:                 iq1_gate_expert_bytes,
 11910:                 iq1_gate_row_bytes,
 11911:                 iq1_down_expert_bytes,
 11912:                 iq1_down_row_bytes,
 11913:                 (uint32_t)expert_in_dim,
 11914:                 (uint32_t)down_in_dim,
 11915:                 (uint32_t)routed_out_dim,
 11916:                 g->router_selected,
 11917:                 g->router_weights,
 11918:                 DS4_N_EXPERT_USED,
 11919:                 DS4_SWIGLU_CLAMP_EXP,
 11920:                 g->ffn_norm,
 11921:                 g->spex_prefetch,
 11922:                 g->spex_prefetch ? &spex_key : NULL) != 0;
 11923:     } else if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,
 11924:                                                  g->routed_gate,
 11925:                                                  g->routed_up,
 11926:                                                  g->routed_mid,
 11927:                                                   g->routed_down,
 11928:                                                   route.model->map,
 11929:                                                   route.model->size,
 11930:                                                   il,
 11931:                                                   route.gate->abs_offset,
 11932:                                                   route.up->abs_offset,
 11933:                                                   route.down->abs_offset,
 11934:                                                   route.gate->type,
 11935:                                                   route.down->type,
 11936:                                                  gate_expert_bytes, gate_row_bytes,
 11937:                                                  down_expert_bytes, down_row_bytes,
 11938:                                                  (uint32_t)expert_in_dim,
 11939:                                                  (uint32_t)down_in_dim,
 11940:                                                  (uint32_t)routed_out_dim,
 11941:                                                  g->router_selected, g->router_weights,
 11942:                                                  DS4_N_EXPERT_USED, DS4_SWIGLU_CLAMP_EXP, g->ffn_norm,
 11943:                                                    route.sidecar ? NULL : g->spex_prefetch,
 11944:                                                    !route.sidecar && g->spex_prefetch
 11945:                                                        ? &spex_key : NULL) != 0;
 11946:     if (ok && !route.sidecar) {
 11947:         metal_graph_spex_cpu_observe_exact(g, layer, il);
 11948:     }
 11949:     DS4_METAL_PROFILE_DECODE_STAGE(
 11950:         overlap_shared_full ? "shared_full+routed_moe" :

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"g_dynamic_arena\\.host_base|arena\\.host_base|host_base\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' | Measure-Object | Select-Object -ExpandProperty Count; rg -n \"g_dynamic_arena\\.host_base\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 546ms:
105
2001:            g_dynamic_arena.host_base, g_dynamic_arena.allocated_bytes,
2007:            g_dynamic_arena.host_base, g_dynamic_arena.allocated_bytes,
2020:        g_dynamic_arena.host_base &&
2045:        g_dynamic_arena.host_base &&
6099:    if (g_dynamic_arena.host_base) {
6103:        (void)cudaFreeHost(g_dynamic_arena.host_base);
6104:        g_dynamic_arena.host_base = NULL;
6205:    if (report && g_dynamic_arena.host_base &&
6951:    if (g_dynamic_arena.host_base &&
7029:    g_dynamic_arena.host_base = host;
7139:        !g_dynamic_arena.host_base || g_dynamic_arena.txn ||
11007:        if (!g_dynamic_arena.host_base || resident_limit == 0 ||
11226:    if (!g_dynamic_arena.host_base || g_dynamic_arena.n_layer <= 3 ||
11520:    if (!g_dynamic_arena.host_base || g_dynamic_arena.n_layer <= 3 ||
11571:        if (!g_dynamic_arena.host_base || g_dynamic_arena.slots.empty() ||
12299:    if (mode < 0 || !g_dynamic_arena.host_base) return;
12340:    if (window == 0 || !g_dynamic_arena.host_base ||
23667:        if (!g_dynamic_arena.host_base || g_dynamic_arena.txn ||
24562:        !g_dynamic_arena.host_base || g_dynamic_arena.slots.empty() ||
28999:    const int arena_active = g_dynamic_arena.host_base &&
30464:    const int arena_active = nested_exact == 0 && g_dynamic_arena.host_base &&

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 22600 22695; Show-Lines $p 23620 23875; Show-Lines $p 28970 29045; Show-Lines $p 30440 30505' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 1024ms:
 22600:     uint64_t ram_generation;
 22601:     uint64_t vram_eligible_after_call;
 22602:     uint8_t has_2bit_ram;
 22603:     uint8_t promoted_from_iq1_cold;
 22604:     cuda_moe_tier_state state;
 22605: };
 22606: 
 22607: struct cuda_moe_tiering {
 22608:     cuda_moe_tier_mode mode;
 22609:     cuda_moe_tier_policy policy;
 22610:     int compose_prefill_mass_tiering;
 22611:     int compose_router_open;
 22612:     uint32_t policy_clock_calls;
 22613:     uint32_t policy_replacement_budget;
 22614:     uint32_t policy_replacement_budget_base;
 22615:     uint32_t policy_budget_remaining;
 22616:     int policy_adaptive_budget;
 22617:     uint32_t policy_adaptive_budget_min;
 22618:     uint32_t policy_adaptive_budget_max;
 22619:     uint32_t policy_adaptive_budget_step;
 22620:     uint32_t policy_adaptive_pressure_threshold;
 22621:     uint32_t policy_min_frequency;
 22622:     uint32_t snapshot_backing_entries;
 22623:     uint32_t open_router_reserve_slots;
 22624:     uint64_t policy_epoch;
 22625:     uint64_t snapshot_generation;
 22626:     double policy_hysteresis;
 22627:     uint64_t call_tick;
 22628:     uint64_t calls;
 22629:     uint64_t selected;
 22630:     uint64_t cold;
 22631:     uint64_t ram_hits;
 22632:     uint64_t vram_hits;
 22633:     uint64_t cold_to_ram;
 22634:     uint64_t cold_to_vram;
 22635:     uint64_t ram_to_warm;
 22636:     uint64_t vram_promotions;
 22637:     uint64_t vram_demotions;
 22638:     uint64_t ram_evictions;
 22639:     uint64_t ram_admit_skips;
 22640:     uint64_t transient;
 22641:     uint64_t failures;
 22642:     uint64_t ssd_bytes;
 22643:     uint64_t ram_h2d_bytes;
 22644:     uint64_t policy_epochs;
 22645:     uint64_t policy_free_promotions;
 22646:     uint64_t policy_replacements;
 22647:     uint64_t policy_min_frequency_skips;
 22648:     uint64_t policy_budget_skips;
 22649:     uint64_t policy_score_skips;
 22650:     uint64_t policy_adaptive_last_budget_skips;
 22651:     uint64_t policy_adaptive_last_replacements;
 22652:     uint64_t policy_adaptive_last_budget_skips_delta;
 22653:     uint64_t policy_adaptive_last_replacements_delta;
 22654:     uint64_t policy_adaptive_budget_ups;
 22655:     uint64_t policy_adaptive_budget_downs;
 22656:     uint64_t policy_adaptive_pressure_epochs;
 22657:     uint64_t policy_adaptive_quiet_epochs;
 22658:     uint64_t snapshot_backing_hits;
 22659:     uint64_t snapshot_backing_misses;
 22660:     uint64_t snapshot_to_vram_bytes;
 22661:     uint64_t forbidden_cold_ssd_to_vram;
 22662:     uint64_t general_backing_reclaims;
 22663:     std::vector<cuda_moe_tier_entry> entries;
 22664:     std::vector<uint8_t> open_router_pool_slot;
 22665: };
 22666: static cuda_moe_tiering g_moe_tiering;
 22667: 
 22668: static const char *cuda_q1_0_mixed_router_mode_name(void) {
 22669:     if (g_q1_0_mixed_router_open_observed) return "open";
 22670:     if (g_q1_0_mixed_router_closed_observed) return "closed";
 22671:     return "closed";
 22672: }
 22673: 
 22674: static void cuda_q1_0_mixed_capture_router_mode(void) {
 22675:     const int observed_open =
 22676:         !cuda_q1_0_snapshot_backing_requested() &&
 22677:         g_moe_tiering.mode == CUDA_MOE_TIER_ENFORCE &&
 22678:         g_moe_tiering.compose_prefill_mass_tiering &&
 22679:         g_moe_tiering.compose_router_open;
 22680:     if (observed_open) {
 22681:         g_q1_0_mixed_router_open_observed = 1u;
 22682:     } else {
 22683:         g_q1_0_mixed_router_closed_observed = 1u;
 22684:     }
 22685: }
 22686: 
 22687: struct cuda_iq1_promotion {
 22688:     uint32_t requested_slots;
 22689:     uint32_t reserved_slots;
 22690:     uint64_t min_touches;
 22691:     double min_weight;
 22692:     double min_mass;
 22693:     uint64_t request_budget;
 22694:     uint64_t request_used;
 22695:     uint64_t window_calls;
 23620:             adaptive_budget_max = 32u;
 23621:             adaptive_budget_step = 8u;
 23622:             adaptive_pressure_threshold = 64u;
 23623:         }
 23624:     }
 23625:     if (policy_requested == CUDA_MOE_TIER_POLICY_SECOND_TOUCH) {
 23626:         clock_calls = 0u;
 23627:         replacement_budget = 0u;
 23628:         adaptive_budget = 0;
 23629:         min_frequency = 2u;
 23630:         hysteresis = 1.0;
 23631:     }
 23632:     if (g_moe_tiering.mode == (cuda_moe_tier_mode)requested &&
 23633:         g_moe_tiering.policy == (cuda_moe_tier_policy)policy_requested &&
 23634:         g_moe_tiering.policy_clock_calls == clock_calls &&
 23635:         g_moe_tiering.policy_replacement_budget_base == replacement_budget &&
 23636:         g_moe_tiering.policy_adaptive_budget == adaptive_budget &&
 23637:         g_moe_tiering.policy_adaptive_budget_min == adaptive_budget_min &&
 23638:         g_moe_tiering.policy_adaptive_budget_max == adaptive_budget_max &&
 23639:         g_moe_tiering.policy_adaptive_budget_step == adaptive_budget_step &&
 23640:         g_moe_tiering.policy_adaptive_pressure_threshold ==
 23641:             adaptive_pressure_threshold &&
 23642:         g_moe_tiering.policy_min_frequency == min_frequency &&
 23643:         g_moe_tiering.policy_hysteresis == hysteresis &&
 23644:         g_iq1_promotion.requested_slots ==
 23645:             (uint32_t)promotion_probation_slots &&
 23646:         g_iq1_promotion.min_touches == promotion_min_touches &&
 23647:         g_iq1_promotion.min_weight == promotion_min_weight &&
 23648:         g_iq1_promotion.min_mass == promotion_min_mass &&
 23649:         g_iq1_promotion.request_budget == promotion_request_budget &&
 23650:         g_iq1_promotion.window_calls == promotion_window_calls &&
 23651:         g_iq1_promotion.window_budget == promotion_window_budget &&
 23652:         g_moe_tiering.open_router_reserve_slots ==
 23653:             (uint32_t)configured_reserve_slots &&
 23654:         g_moe_tiering.compose_prefill_mass_tiering == compose_requested &&
 23655:         g_moe_tiering.compose_router_open == compose_router_open &&
 23656:         (!compose_requested ||
 23657:          g_moe_tiering.snapshot_generation ==
 23658:              g_dynamic_arena.snapshot_generation) &&
 23659:         g_moe_tiering.entries.size() ==
 23660:             (size_t)CUDA_MOE_LAYER_COUNT * 256u) {
 23661:         return 1;
 23662:     }
 23663:     if (g_moe_tiering.mode != CUDA_MOE_TIER_OFF) {
 23664:         cuda_moe_tiering_report_and_reset();
 23665:     }
 23666:     if (requested == CUDA_MOE_TIER_ENFORCE) {
 23667:         if (!g_dynamic_arena.host_base || g_dynamic_arena.txn ||
 23668:             g_dynamic_arena.n_layer != CUDA_MOE_LAYER_COUNT ||
 23669:             g_dynamic_arena.n_expert != 256u ||
 23670:             g_dynamic_arena.slots.empty()) {
 23671:             fprintf(stderr,
 23672:                     "ds4: expert tiering enforce requires an idle pinned dynamic arena\n");
 23673:             return 0;
 23674:         }
 23675:         if (compose_requested) {
 23676:             if (policy_requested != CUDA_MOE_TIER_POLICY_MASS_LFRU ||
 23677:                 g_dynamic_arena.snapshot_generation == 0 ||
 23678:                 g_dynamic_arena.hits_disabled ||
 23679:                 g_dynamic_arena.submissions_blocked ||
 23680:                 g_dynamic_arena_observer.enabled ||
 23681:                 !g_dynamic_arena_observer.target.empty() ||
 23682:                 !g_dynamic_arena_observer.counts.empty() ||
 23683:                 !g_prefill_mass_observer.enabled ||
 23684:                 !g_prefill_mass_observer.finalized ||
 23685:                 !g_prefill_mass_observer.wrap_published ||
 23686:                 g_reap_mass_observer.enabled) {
 23687:                 fprintf(stderr,
 23688:                         "ds4: prefill/tiering compose requires finalized published prefill snapshot and no dynamic/REAP observer\n");
 23689:                 return 0;
 23690:             }
 23691:         } else if (g_dynamic_arena.snapshot_generation != 0 ||
 23692:                    g_dynamic_arena_observer.enabled ||
 23693:                    g_prefill_mass_observer.enabled ||
 23694:                    g_reap_mass_observer.enabled) {
 23695:             fprintf(stderr,
 23696:                     "ds4: expert tiering enforce requires an idle pinned dynamic arena\n");
 23697:             return 0;
 23698:         }
 23699:     }
 23700:     try {
 23701:         cuda_moe_tier_entry empty;
 23702:         memset(&empty, 0, sizeof(empty));
 23703:         empty.ram_slot = UINT32_MAX;
 23704:         empty.state = CUDA_MOE_TIER_SSD_COLD;
 23705:         g_moe_tiering.entries.assign(
 23706:             (size_t)CUDA_MOE_LAYER_COUNT * 256u, empty);
 23707:     } catch (...) {
 23708:         fprintf(stderr, "ds4: expert tiering metadata allocation failed\n");
 23709:         return 0;
 23710:     }
 23711:     g_moe_tiering.mode = (cuda_moe_tier_mode)requested;
 23712:     g_moe_tiering.policy = (cuda_moe_tier_policy)policy_requested;
 23713:     g_moe_tiering.compose_prefill_mass_tiering = compose_requested;
 23714:     g_moe_tiering.compose_router_open = compose_router_open;
 23715:     g_moe_tiering.policy_clock_calls = clock_calls;
 23716:     g_moe_tiering.policy_replacement_budget = replacement_budget;
 23717:     g_moe_tiering.policy_replacement_budget_base = replacement_budget;
 23718:     g_moe_tiering.policy_adaptive_budget = adaptive_budget;
 23719:     g_moe_tiering.policy_adaptive_budget_min = adaptive_budget_min;
 23720:     g_moe_tiering.policy_adaptive_budget_max = adaptive_budget_max;
 23721:     g_moe_tiering.policy_adaptive_budget_step = adaptive_budget_step;
 23722:     g_moe_tiering.policy_adaptive_pressure_threshold =
 23723:         adaptive_pressure_threshold;
 23724:     g_moe_tiering.policy_min_frequency = min_frequency;
 23725:     g_moe_tiering.policy_hysteresis = hysteresis;
 23726:     g_moe_tiering.policy_epoch = UINT64_MAX;
 23727:     g_iq1_promotion = cuda_iq1_promotion();
 23728:     g_iq1_promotion.requested_slots =
 23729:         (uint32_t)promotion_probation_slots;
 23730:     g_iq1_promotion.min_touches = promotion_min_touches;
 23731:     g_iq1_promotion.min_weight = promotion_min_weight;
 23732:     g_iq1_promotion.min_mass = promotion_min_mass;
 23733:     g_iq1_promotion.request_budget = promotion_request_budget;
 23734:     g_iq1_promotion.window_calls = promotion_window_calls;
 23735:     g_iq1_promotion.window_budget = promotion_window_budget;
 23736:     if (compose_requested) {
 23737:         g_moe_tiering.snapshot_generation =
 23738:             g_dynamic_arena.snapshot_generation;
 23739:         for (uint32_t entry_index = 0;
 23740:              entry_index < g_dynamic_arena.active.size() &&
 23741:              entry_index < g_moe_tiering.entries.size(); entry_index++) {
 23742:             const uint32_t layer = entry_index / 256u;
 23743:             const uint32_t expert = entry_index % 256u;
 23744:             if (cuda_dynamic_arena_binding_valid(
 23745:                     g_dynamic_arena,
 23746:                     g_dynamic_arena.active[entry_index],
 23747:                     layer, expert,
 23748:                     g_moe_tiering.snapshot_generation,
 23749:                     DS4_GPU_ARENA_READY)) {
 23750:                 cuda_moe_tier_entry &entry =
 23751:                     g_moe_tiering.entries[entry_index];
 23752:                 /* Snapshot-backed RAM is borrowed backing, not tier-owned. */
 23753:                 entry.ram_slot = UINT32_MAX;
 23754:                 entry.ram_generation = 0;
 23755:                 entry.has_2bit_ram =
 23756:                     g_dynamic_arena.backing ==
 23757:                     CUDA_DYNAMIC_ARENA_BACKING_PRIMARY;
 23758:                 entry.state = CUDA_MOE_TIER_RAM_PROBATION;
 23759:                 g_moe_tiering.snapshot_backing_entries++;
 23760:             }
 23761:         }
 23762:         if (g_moe_tiering.snapshot_backing_entries == 0u) {
 23763:             fprintf(stderr,
 23764:                     "ds4: prefill/tiering compose requires non-empty snapshot backing\n");
 23765:             g_moe_tiering = cuda_moe_tiering();
 23766:             g_iq1_promotion = cuda_iq1_promotion();
 23767:             return 0;
 23768:         }
 23769:         if (!cuda_moe_tiering_reserve_open_router_slots(
 23770:                 (uint32_t)configured_reserve_slots)) {
 23771:             g_moe_tiering = cuda_moe_tiering();
 23772:             g_iq1_promotion = cuda_iq1_promotion();
 23773:             return 0;
 23774:         }
 23775:     }
 23776:     if (g_moe_tiering.mode == CUDA_MOE_TIER_ENFORCE) {
 23777:         g_dynamic_arena.tiering_exclusive = 1;
 23778:     }
 23779:     fprintf(stderr, "ds4: expert tiering active mode=%s policy=%s compose_prefill_mass_tiering=%u compose_router_open=%u snapshot_generation=%llu snapshot_backing_entries=%u clock_calls=%u replacement_budget=%u replacement_budget_base=%u adaptive_budget=%u adaptive_current_budget=%u adaptive_min=%u adaptive_max=%u adaptive_step=%u adaptive_pressure_threshold=%u adaptive_ups=%llu adaptive_downs=%llu adaptive_pressure_epochs=%llu adaptive_quiet_epochs=%llu adaptive_last_budget_skips_delta=%llu adaptive_last_replacements_delta=%llu min_frequency=%u hysteresis=%.9g entries=%u ram_slots=%u\n",
 23780:             cuda_moe_tiering_mode_name(g_moe_tiering.mode),
 23781:             cuda_moe_tiering_policy_name(g_moe_tiering.policy),
 23782:             g_moe_tiering.compose_prefill_mass_tiering ? 1u : 0u,
 23783:             g_moe_tiering.compose_router_open ? 1u : 0u,
 23784:             (unsigned long long)g_moe_tiering.snapshot_generation,
 23785:             g_moe_tiering.snapshot_backing_entries,
 23786:             g_moe_tiering.policy_clock_calls,
 23787:             g_moe_tiering.policy_replacement_budget,
 23788:             g_moe_tiering.policy_replacement_budget_base,
 23789:             g_moe_tiering.policy_adaptive_budget ? 1u : 0u,
 23790:             g_moe_tiering.policy_replacement_budget,
 23791:             g_moe_tiering.policy_adaptive_budget_min,
 23792:             g_moe_tiering.policy_adaptive_budget_max,
 23793:             g_moe_tiering.policy_adaptive_budget_step,
 23794:             g_moe_tiering.policy_adaptive_pressure_threshold,
 23795:             (unsigned long long)g_moe_tiering.policy_adaptive_budget_ups,
 23796:             (unsigned long long)g_moe_tiering.policy_adaptive_budget_downs,
 23797:             (unsigned long long)g_moe_tiering.policy_adaptive_pressure_epochs,
 23798:             (unsigned long long)g_moe_tiering.policy_adaptive_quiet_epochs,
 23799:             (unsigned long long)g_moe_tiering.policy_adaptive_last_budget_skips_delta,
 23800:             (unsigned long long)g_moe_tiering.policy_adaptive_last_replacements_delta,
 23801:             g_moe_tiering.policy_min_frequency,
 23802:             g_moe_tiering.policy_hysteresis,
 23803:             (uint32_t)g_moe_tiering.entries.size(),
 23804:             (uint32_t)g_dynamic_arena.slots.size());
 23805:     return 1;
 23806: }
 23807: 
 23808: static int cuda_moe_tiering_snapshot_ram_ptrs(
 23809:         uint32_t layer, uint32_t expert,
 23810:         char **gate, char **up, char **down) {
 23811:     if (!g_moe_tiering.compose_prefill_mass_tiering ||
 23812:         g_dynamic_arena.backing != CUDA_DYNAMIC_ARENA_BACKING_PRIMARY ||
 23813:         layer >= CUDA_MOE_LAYER_COUNT || expert >= 256u ||
 23814:         !gate || !up || !down ||
 23815:         g_dynamic_arena.snapshot_generation !=
 23816:             g_moe_tiering.snapshot_generation ||
 23817:         g_dynamic_arena.hits_disabled ||
 23818:         g_dynamic_arena.submissions_blocked) {
 23819:         return 0;
 23820:     }
 23821:     const uint32_t entry = layer * 256u + expert;
 23822:     if (entry >= g_dynamic_arena.active.size()) return 0;
 23823:     const cuda_dynamic_arena_binding &binding =
 23824:         g_dynamic_arena.active[entry];
 23825:     if (!cuda_dynamic_arena_binding_valid(
 23826:             g_dynamic_arena,
 23827:             binding, layer, expert,
 23828:             g_moe_tiering.snapshot_generation,
 23829:             DS4_GPU_ARENA_READY)) {
 23830:         return 0;
 23831:     }
 23832:     cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[binding.slot];
 23833:     *gate = slot.host_ptr;
 23834:     *up = slot.host_ptr + g_moe_expert_cache.gate_expert_bytes;
 23835:     *down = *up + g_moe_expert_cache.gate_expert_bytes;
 23836:     return 1;
 23837: }
 23838: 
 23839: static int cuda_moe_tiering_ram_ptrs(
 23840:         uint32_t layer, uint32_t expert,
 23841:         char **gate, char **up, char **down) {
 23842:     if (layer >= CUDA_MOE_LAYER_COUNT || expert >= 256u ||
 23843:         !gate || !up || !down) return 0;
 23844:     cuda_moe_tier_entry &entry =
 23845:         g_moe_tiering.entries[cuda_moe_tiering_entry_index(layer, expert)];
 23846:     if (entry.ram_slot >= g_dynamic_arena.slots.size()) return 0;
 23847:     cuda_dynamic_arena_slot &slot = g_dynamic_arena.slots[entry.ram_slot];
 23848:     if ((slot.state != DS4_GPU_ARENA_READY &&
 23849:          slot.state != DS4_GPU_ARENA_STAGED) ||
 23850:         slot.layer != layer || slot.expert != expert ||
 23851:         slot.content_generation != entry.ram_generation) {
 23852:         return 0;
 23853:     }
 23854:     *gate = slot.host_ptr;
 23855:     *up = slot.host_ptr + g_moe_expert_cache.gate_expert_bytes;
 23856:     *down = *up + g_moe_expert_cache.gate_expert_bytes;
 23857:     return 1;
 23858: }
 23859: 
 23860: static int cuda_moe_tiering_has_exact_vram(
 23861:         uint32_t layer, uint32_t expert) {
 23862:     if (layer >= CUDA_MOE_LAYER_COUNT || expert >= 256u ||
 23863:         g_moe_expert_cache.host_slot_by_layer_expert.empty()) {
 23864:         return 0;
 23865:     }
 23866:     const uint32_t map_index = layer * 256u + expert;
 23867:     if (map_index >= g_moe_expert_cache.host_slot_by_layer_expert.size()) {
 23868:         return 0;
 23869:     }
 23870:     const int32_t slot_i =
 23871:         g_moe_expert_cache.host_slot_by_layer_expert[map_index];
 23872:     if (slot_i < 0 || (uint32_t)slot_i >= g_moe_expert_cache.slots.size()) {
 23873:         return 0;
 23874:     }
 23875:     const cuda_moe_cache_slot &slot =
 28970: 
 28971:     std::vector<int32_t> &wave_slots = *wave_slots_ptr;
 28972:     std::vector<uint32_t> &wave_pairs = *wave_pairs_ptr;
 28973:     wave_slots.assign(slot_count, 0);
 28974:     wave_pairs.clear();
 28975:     wave_pairs.reserve(slot_count);
 28976:     std::vector<int32_t> &e2s = g_moe_gather.h_expert_to_slot;
 28977:     e2s.assign(n_total_expert, -1);
 28978:     for (uint32_t i = begin; i < end; i++) {
 28979:         const int32_t expert = g_moe_gather.h_compact_ids[i];
 28980:         if (expert < 0 || (uint32_t)expert >= n_total_expert) return 0;
 28981:         e2s[(uint32_t)expert] = (int32_t)(i - begin);
 28982:     }
 28983:     for (uint32_t i = 0; i < slot_count; i++) {
 28984:         const int32_t expert = g_moe_gather.h_sel[i];
 28985:         if (expert < 0 || (uint32_t)expert >= n_total_expert) return 0;
 28986:         const int32_t wave_slot = e2s[(uint32_t)expert];
 28987:         if (wave_slot >= 0) {
 28988:             wave_slots[i] = wave_slot;
 28989:             wave_pairs.push_back(i);
 28990:         }
 28991:     }
 28992:     if (wave_pairs.empty()) return 0;
 28993: 
 28994:     std::vector<cuda_moe_gather::span> &spans = g_moe_gather.h_spans;
 28995:     std::vector<uint8_t> &arena_resident = g_moe_gather.h_arena_resident;
 28996:     spans.clear();
 28997:     spans.reserve((size_t)wave_experts * 3u);
 28998:     arena_resident.assign(wave_experts, 0);
 28999:     const int arena_active = g_dynamic_arena.host_base &&
 29000:         g_dynamic_arena.snapshot_generation != 0 &&
 29001:         !g_dynamic_arena.hits_disabled &&
 29002:         !g_dynamic_arena.submissions_blocked;
 29003:     auto append_span = [&](char *destination,
 29004:                            uint64_t source,
 29005:                            uint64_t bytes,
 29006:                            uint32_t expert,
 29007:                            uint8_t part) {
 29008:         uint32_t arena_entry = UINT32_MAX;
 29009:         char *mirror = cuda_dynamic_arena_observer_mirror_ptr(
 29010:             model_map, layer_index, expert, part, source, bytes,
 29011:             &arena_entry);
 29012:         spans.push_back({
 29013:             destination, source, bytes, mirror, arena_entry, part
 29014:         });
 29015:     };
 29016:     for (uint32_t i = 0; i < wave_experts; i++) {
 29017:         const uint32_t expert = (uint32_t)g_moe_gather.h_compact_ids[begin + i];
 29018:         const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
 29019:         const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
 29020:         if (arena_active) {
 29021:             const cuda_dynamic_arena_copy_status arena_status =
 29022:                 cuda_dynamic_arena_copy_expert_async(
 29023:                     g_dynamic_arena,
 29024:                     model_map, layer_index, expert,
 29025:                     gate_offset, up_offset, down_offset,
 29026:                     gate_dst_base + gate_dst,
 29027:                     up_dst_base + gate_dst,
 29028:                     down_dst_base + down_dst,
 29029:                     gate_expert_bytes, gate_expert_bytes,
 29030:                     down_expert_bytes);
 29031:             if (arena_status == CUDA_DYNAMIC_ARENA_FATAL) {
 29032:                 (void)cudaStreamSynchronize(g_model_upload_stream);
 29033:                 return 0;
 29034:             }
 29035:             if (arena_status == CUDA_DYNAMIC_ARENA_ENQUEUED) {
 29036:                 arena_resident[i] = 1u;
 29037:                 continue;
 29038:             }
 29039:         }
 29040:         append_span(gate_dst_base + gate_dst,
 29041:                     gate_offset + (uint64_t)expert * gate_expert_bytes,
 29042:                     gate_expert_bytes, expert,
 29043:                     CUDA_DYNAMIC_ARENA_MIRROR_GATE);
 29044:         append_span(up_dst_base + gate_dst,
 29045:                     up_offset + (uint64_t)expert * gate_expert_bytes,
 30440:         !cuda_moe_gather_ensure_i32(&g_moe_gather.slot, &g_moe_gather.slot_cap, slot_count, "moe gather slots"))
 30441:         return 0;
 30442:     if (slot_will_grow) g_moe_gather.iq1_slot_zero_valid = 0;
 30443:     g_moe_gather.iq1_active_gate = NULL;
 30444:     g_moe_gather.iq1_active_up = NULL;
 30445:     g_moe_gather.iq1_active_down = NULL;
 30446: 
 30447:     /* Dynamic-arena hits go straight from their stable pinned slot to the
 30448:      * compact execution buffers. They never enter the legacy pread/staging
 30449:      * list and never take a detour through the persistent VRAM cache. */
 30450:     std::vector<uint8_t> &arena_resident = g_moe_gather.h_arena_resident;
 30451:     arena_resident.assign(compact_count, 0);
 30452:     const int nested_exact = cuda_nested_residual_upload_exact(
 30453:         layer_index,
 30454:         gate_offset, up_offset, down_offset,
 30455:         gate_expert_bytes, down_expert_bytes,
 30456:         compact);
 30457:     if (nested_exact < 0) {
 30458:         g_moe_gather.wave_fail_closed = 1;
 30459:         return 0;
 30460:     }
 30461:     if (nested_exact > 0) {
 30462:         std::fill(arena_resident.begin(), arena_resident.end(), 1u);
 30463:     }
 30464:     const int arena_active = nested_exact == 0 && g_dynamic_arena.host_base &&
 30465:         g_dynamic_arena.snapshot_generation != 0 &&
 30466:         !g_dynamic_arena.hits_disabled &&
 30467:         !g_dynamic_arena.submissions_blocked;
 30468:     if (arena_active) {
 30469:         for (uint32_t i = 0; i < compact_count; i++) {
 30470:             const uint32_t expert = (uint32_t)compact[i];
 30471:             const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
 30472:             const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
 30473:             const cuda_dynamic_arena_copy_status arena_status =
 30474:                 cuda_dynamic_arena_copy_expert_async(
 30475:                     g_dynamic_arena,
 30476:                     model_map, layer_index, expert,
 30477:                     gate_offset, up_offset, down_offset,
 30478:                     g_moe_gather.gate + gate_dst,
 30479:                     g_moe_gather.up + gate_dst,
 30480:                     g_moe_gather.down + down_dst,
 30481:                     gate_expert_bytes, gate_expert_bytes,
 30482:                     down_expert_bytes);
 30483:             if (arena_status == CUDA_DYNAMIC_ARENA_FATAL) {
 30484:                 (void)cudaStreamSynchronize(g_model_upload_stream);
 30485:                 return 0;
 30486:             }
 30487:             arena_resident[i] =
 30488:                 arena_status == CUDA_DYNAMIC_ARENA_ENQUEUED ? 1u : 0u;
 30489:         }
 30490:     }
 30491: 
 30492:     std::vector<uint8_t> &iq1_ram_resident =
 30493:         g_moe_gather.h_iq1_ram_resident;
 30494:     iq1_ram_resident.assign(compact_count, 0);
 30495:     const int iq1_profile = getenv("DS4_IQ1_S_PROFILE") != NULL;
 30496:     const double iq1_enqueue_t0 = iq1_profile ? cuda_wall_sec() : 0.0;
 30497:     uint32_t iq1_upload_experts = 0;
 30498:     int iq1_vram_pending_slot = -1;
 30499:     int iq1_vram_pending_new = 0;
 30500:     if (iq1_sidecar_route) {
 30501:         for (uint32_t i = 0; i < compact_count; i++) {
 30502:             const uint32_t expert = (uint32_t)compact[i];
 30503:             char *iq1_vram_base = NULL;
 30504:             if (iq1_vram_cache_active) {
 30505:                 int vram_hit = 0;

2026-07-21T09:55:08.418786Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' 5900 6075" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 553ms:
  5900: }
  5901: 
  5902: extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes) {
  5903:     if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
  5904:     return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes, cudaMemcpyHostToDevice), "tensor write");
  5905: }
  5906: 
  5907: extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes) {
  5908:     if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
  5909:     return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes, cudaMemcpyDeviceToHost), "tensor read");
  5910: }
  5911: 
  5912: static cuda_dynamic_arena_binding cuda_dynamic_arena_empty_binding(void) {
  5913:     const cuda_dynamic_arena_binding empty = {
  5914:         UINT32_MAX, DS4_GPU_ARENA_FREE, 0, 0
  5915:     };
  5916:     return empty;
  5917: }
  5918: 
  5919: static uint64_t cuda_dynamic_arena_next_generation(void) {
  5920:     g_dynamic_arena.next_generation++;
  5921:     if (g_dynamic_arena.next_generation == 0) {
  5922:         g_dynamic_arena.next_generation++;
  5923:     }
  5924:     return g_dynamic_arena.next_generation;
  5925: }
  5926: 
  5927: static int cuda_dynamic_arena_range_valid(
  5928:         uint64_t offset,
  5929:         uint64_t expert_bytes,
  5930:         uint32_t n_expert,
  5931:         uint64_t model_size) {
  5932:     return expert_bytes != 0 && offset <= model_size &&
  5933:         (uint64_t)n_expert <= (model_size - offset) / expert_bytes;
  5934: }
  5935: 
  5936: static int cuda_dynamic_arena_binding_valid(
  5937:         const cuda_dynamic_arena &arena,
  5938:         const cuda_dynamic_arena_binding &binding,
  5939:         uint32_t layer,
  5940:         uint32_t expert,
  5941:         uint64_t snapshot_generation,
  5942:         ds4_gpu_arena_slot_state required_state) {
  5943:     if (binding.slot >= arena.slots.size() ||
  5944:         binding.state != (uint32_t)required_state ||
  5945:         binding.snapshot_generation != snapshot_generation) {
  5946:         return 0;
  5947:     }
  5948:     const cuda_dynamic_arena_slot &slot = arena.slots[binding.slot];
  5949:     return slot.state == required_state &&
  5950:         slot.layer == layer && slot.expert == expert &&
  5951:         slot.content_generation != 0 &&
  5952:         slot.content_generation == binding.slot_generation;
  5953: }
  5954: 
  5955: static uint64_t cuda_dynamic_arena_fnv1a64_continue(
  5956:         const uint8_t *data,
  5957:         uint64_t bytes,
  5958:         uint64_t checksum) {
  5959:     for (uint64_t i = 0; i < bytes; i++) {
  5960:         checksum ^= data[i];
  5961:         checksum *= UINT64_C(1099511628211);
  5962:     }
  5963:     return checksum;
  5964: }
  5965: 
  5966: static uint64_t cuda_dynamic_arena_fnv1a64(
  5967:         const uint8_t *data,
  5968:         uint64_t bytes) {
  5969:     return cuda_dynamic_arena_fnv1a64_continue(
  5970:         data, bytes, UINT64_C(14695981039346656037));
  5971: }
  5972: 
  5973: enum cuda_dynamic_arena_copy_status {
  5974:     CUDA_DYNAMIC_ARENA_FATAL = -1,
  5975:     CUDA_DYNAMIC_ARENA_MISS = 0,
  5976:     CUDA_DYNAMIC_ARENA_ENQUEUED = 1,
  5977: };
  5978: 
  5979: static cuda_dynamic_arena_copy_status cuda_dynamic_arena_copy_expert_async(
  5980:         cuda_dynamic_arena &arena,
  5981:         const void *model_map,
  5982:         uint32_t layer,
  5983:         uint32_t expert,
  5984:         uint64_t gate_offset,
  5985:         uint64_t up_offset,
  5986:         uint64_t down_offset,
  5987:         void *gate_dst,
  5988:         void *up_dst,
  5989:         void *down_dst,
  5990:         uint64_t gate_expert_bytes,
  5991:         uint64_t up_expert_bytes,
  5992:         uint64_t down_expert_bytes) {
  5993:     if ((!arena.host_base && !arena.pageable_base) ||
  5994:         model_map != arena.model_map ||
  5995:         arena.hits_disabled ||
  5996:         arena.submissions_blocked ||
  5997:         layer >= arena.n_layer ||
  5998:         expert >= arena.n_expert) {
  5999:         return CUDA_DYNAMIC_ARENA_MISS;
  6000:     }
  6001:     const ds4_gpu_dynamic_arena_layer &geometry =
  6002:         arena.layers[layer];
  6003:     if (geometry.gate_offset != gate_offset ||
  6004:         geometry.up_offset != up_offset ||
  6005:         geometry.down_offset != down_offset) {
  6006:         return CUDA_DYNAMIC_ARENA_MISS;
  6007:     }
  6008:     if (geometry.gate_expert_bytes != gate_expert_bytes ||
  6009:         geometry.up_expert_bytes != up_expert_bytes ||
  6010:         geometry.down_expert_bytes != down_expert_bytes) {
  6011:         arena.fatal_errors++;
  6012:         return CUDA_DYNAMIC_ARENA_FATAL;
  6013:     }
  6014:     const uint32_t entry = layer * arena.n_expert + expert;
  6015:     const cuda_dynamic_arena_binding &binding =
  6016:         arena.active[entry];
  6017:     if (!cuda_dynamic_arena_binding_valid(
  6018:             arena,
  6019:             binding, layer, expert,
  6020:             arena.snapshot_generation,
  6021:             DS4_GPU_ARENA_READY)) {
  6022:         arena.misses++;
  6023:         return CUDA_DYNAMIC_ARENA_MISS;
  6024:     }
  6025:     cuda_dynamic_arena_slot &slot = arena.slots[binding.slot];
  6026:     const char *gate_src = slot.host_ptr;
  6027:     const char *up_src = gate_src + gate_expert_bytes;
  6028:     const char *down_src = up_src + up_expert_bytes;
  6029:     const int q1_profile =
  6030:         arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0 &&
  6031:         cuda_q1_0_profile_requested();
  6032:     const double enqueue_started = q1_profile ? cuda_wall_sec() : 0.0;
  6033: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
  6034:     cuda_g130_attribution_state *attribution_state =
  6035:         cuda_g130_attribution_token_state();
  6036:     cuda_g130_attribution_span attribution_previous = CUDA_G130_ATTRIB_NONE;
  6037:     if (attribution_state) {
  6038:         attribution_previous = cuda_g130_attribution_switch(
  6039:             attribution_state,
  6040:             CUDA_G130_ATTRIB_H2D_ENQUEUE);
  6041:     }
  6042: #endif
  6043:     cudaError_t err = cudaMemcpyAsync(gate_dst, gate_src,
  6044:                                       (size_t)gate_expert_bytes,
  6045:                                       cudaMemcpyHostToDevice,
  6046:                                       g_model_upload_stream);
  6047:     if (err == cudaSuccess) {
  6048:         err = cudaMemcpyAsync(up_dst, up_src,
  6049:                               (size_t)up_expert_bytes,
  6050:                               cudaMemcpyHostToDevice,
  6051:                               g_model_upload_stream);
  6052:     }
  6053:     if (err == cudaSuccess) {
  6054:         err = cudaMemcpyAsync(down_dst, down_src,
  6055:                               (size_t)down_expert_bytes,
  6056:                               cudaMemcpyHostToDevice,
  6057:                               g_model_upload_stream);
  6058:     }
  6059: #ifndef DS4_G130_ATTRIB_COMPILED_OUT
  6060:     if (attribution_state) {
  6061:         (void)cuda_g130_attribution_switch(
  6062:             attribution_state, attribution_previous);
  6063:     }
  6064: #endif
  6065:     if (err != cudaSuccess) {
  6066:         (void)cudaGetLastError();
  6067:         arena.fatal_errors++;
  6068:         return CUDA_DYNAMIC_ARENA_FATAL;
  6069:     }
  6070:     if (q1_profile) {
  6071:         const double elapsed = cuda_wall_sec() - enqueue_started;
  6072:         if (slot.pageable) {
  6073:             g_q1_0_profile.pageable_h2d_enqueue_seconds += elapsed;
  6074:         } else {
  6075:             g_q1_0_profile.pinned_h2d_enqueue_seconds += elapsed;

2026-07-21T09:55:17.391955Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"static int cuda_moe_selected_load\\(\" 'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; function Show-Lines("'$Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu' 30180 30680" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 711ms:
30074:static int cuda_moe_selected_load(
 30180:     const uint64_t full_prob_count =
 30181:         (uint64_t)n_tokens * n_total_expert;
 30182:     if (prefill_mass_full_probs &&
 30183:         (!probs_arg || !probs_arg->ptr ||
 30184:          probs_arg->bytes < full_prob_count * sizeof(float) ||
 30185:          full_prob_count > UINT32_MAX)) {
 30186:         fprintf(stderr,
 30187:                 "ds4: [prefill-mass] full router probabilities unavailable; observer disabled\n");
 30188:         cuda_prefill_mass_observer_release(0);
 30189:         return 0;
 30190:     }
 30191: 
 30192:     const int prepared =
 30193:         g_moe_selected_prepared.valid &&
 30194:         g_moe_selected_prepared.model_map == model_map &&
 30195:         g_moe_selected_prepared.selected_ptr == selected_arg->ptr &&
 30196:         g_moe_selected_prepared.gate_offset == gate_offset &&
 30197:         g_moe_selected_prepared.up_offset == up_offset &&
 30198:         g_moe_selected_prepared.down_offset == down_offset &&
 30199:         g_moe_selected_prepared.gate_expert_bytes == gate_expert_bytes &&
 30200:         g_moe_selected_prepared.down_expert_bytes == down_expert_bytes &&
 30201:         g_moe_selected_prepared.n_expert == n_expert &&
 30202:         g_moe_selected_prepared.n_tokens == n_tokens;
 30203:     g_moe_selected_prepared.valid = 0;
 30204:     if (prepared && g_moe_selected_prepared.full) {
 30205:         static int full_overlap_notice_printed = 0;
 30206:         if (!full_overlap_notice_printed) {
 30207:             fprintf(stderr, "ds4: CUDA MoE full shared-overlap consumed\n");
 30208:             full_overlap_notice_printed = 1;
 30209:         }
 30210:     } else if (prepared) {
 30211:         static int overlap_notice_printed = 0;
 30212:         if (!overlap_notice_printed) {
 30213:             fprintf(stderr, "ds4: CUDA MoE shared-overlap consumed\n");
 30214:             overlap_notice_printed = 1;
 30215:         }
 30216:     }
 30217: 
 30218:     /* 1. Router outputs -> host. Layer-top1 also consumes the selected weights
 30219:      * to choose residency; selection itself remains unchanged. */
 30220:     if (iq1_mixed_host_prepared) {
 30221:         if (iq1_mixed_host_expert < 0 ||
 30222:             (uint32_t)iq1_mixed_host_expert >= n_total_expert) {
 30223:             return 0;
 30224:         }
 30225:         g_moe_gather.h_sel.assign(1u, iq1_mixed_host_expert);
 30226:         g_moe_gather.h_weights.assign(1u, iq1_mixed_host_weight);
 30227:     } else if (prepared) {
 30228:         if (g_moe_gather.h_sel.size() != slot_count) return 0;
 30229:         if (packed_reap_trace) {
 30230:             g_moe_gather.h_weights.resize(slot_count);
 30231:             g_reap_router_trace_valid = 0;
 30232:             if (!cuda_ok(cudaMemcpy(
 30233:                     g_reap_router_trace_host, g_reap_router_trace_device,
 30234:                     sizeof(*g_reap_router_trace_host),
 30235:                     cudaMemcpyDeviceToHost), "moe packed router trace D2H")) {
 30236:                 return 0;
 30237:             }
 30238:             memcpy(g_moe_gather.h_weights.data(),
 30239:                    g_reap_router_trace_host->weights,
 30240:                    (size_t)slot_count * sizeof(float));
 30241:         } else if (copy_weights) {
 30242:             g_moe_gather.h_weights.resize(slot_count);
 30243:             if (!cuda_ok(cudaMemcpy(
 30244:                     g_moe_gather.h_weights.data(), weights_arg->ptr,
 30245:                     (size_t)slot_count * sizeof(float),
 30246:                     cudaMemcpyDeviceToHost), "moe weights D2H")) {
 30247:                 return 0;
 30248:             }
 30249:         }
 30250:     } else if (packed_reap_trace) {
 30251:         g_moe_gather.h_sel.resize(slot_count);
 30252:         g_moe_gather.h_weights.resize(slot_count);
 30253:         g_reap_router_trace_valid = 0;
 30254:         if (!cuda_ok(cudaMemcpy(
 30255:                 g_reap_router_trace_host, g_reap_router_trace_device,
 30256:                 sizeof(*g_reap_router_trace_host),
 30257:                 cudaMemcpyDeviceToHost), "moe packed router trace D2H")) {
 30258:             return 0;
 30259:         }
 30260:         memcpy(g_moe_gather.h_sel.data(),
 30261:                g_reap_router_trace_host->selected,
 30262:                (size_t)slot_count * sizeof(int32_t));
 30263:         memcpy(g_moe_gather.h_weights.data(),
 30264:                g_reap_router_trace_host->weights,
 30265:                (size_t)slot_count * sizeof(float));
 30266:     } else if (copy_weights) {
 30267:         g_moe_gather.h_sel.resize(slot_count);
 30268:         g_moe_gather.h_weights.resize(slot_count);
 30269:         if (!cuda_ok(cudaMemcpyAsync(g_moe_gather.h_sel.data(), selected_arg->ptr,
 30270:                                      (size_t)slot_count * sizeof(int32_t),
 30271:                                      cudaMemcpyDeviceToHost, 0), "moe selected D2H enqueue") ||
 30272:             !cuda_ok(cudaMemcpyAsync(g_moe_gather.h_weights.data(), weights_arg->ptr,
 30273:                                      (size_t)slot_count * sizeof(float),
 30274:                                      cudaMemcpyDeviceToHost, 0), "moe weights D2H enqueue") ||
 30275:             !cuda_ok(cudaStreamSynchronize(0), "moe router D2H sync")) return 0;
 30276:     } else {
 30277:         g_moe_gather.h_sel.resize(slot_count);
 30278:         if (!cuda_ok(cudaMemcpy(g_moe_gather.h_sel.data(), selected_arg->ptr,
 30279:                                 (size_t)slot_count * sizeof(int32_t),
 30280:                                 cudaMemcpyDeviceToHost), "moe selected D2H")) return 0;
 30281:     }
 30282:     if (prefill_mass_full_probs) {
 30283:         g_moe_gather.h_probs.resize((size_t)full_prob_count);
 30284:         if (!cuda_ok(cudaMemcpy(
 30285:                 g_moe_gather.h_probs.data(), probs_arg->ptr,
 30286:                 (size_t)full_prob_count * sizeof(float),
 30287:                 cudaMemcpyDeviceToHost), "moe full router probabilities D2H")) {
 30288:             return 0;
 30289:         }
 30290:     }
 30291:     if (!cuda_sparse_bake_validate_selected(
 30292:             layer_index, g_moe_gather.h_sel.data(), slot_count,
 30293:             "routed selection", 0u)) {
 30294:         return 0;
 30295:     }
 30296:     if (route_prof) route_t_d2h = cuda_wall_sec();
 30297: 
 30298:     cuda_prefill_mass_observe_selected(
 30299:         layer_index, n_tokens, g_moe_gather.h_sel.data(),
 30300:         host_weights_available ? g_moe_gather.h_weights.data() : NULL,
 30301:         slot_count,
 30302:         prefill_mass_full_probs ? g_moe_gather.h_probs.data() : NULL,
 30303:         prefill_mass_full_probs ? (uint32_t)full_prob_count : 0u);
 30304:     if (reap_mass_weights) {
 30305:         cuda_reap_mass_observe_selected(
 30306:             layer_index, n_tokens, g_moe_gather.h_sel.data(),
 30307:             g_moe_gather.h_weights.data(), slot_count);
 30308:     }
 30309: 
 30310:     /* Residency learning consumes the exact selected ids already required by
 30311:      * execution. It does not change router scores, top-k, or the active mask. */
 30312:     cuda_dynamic_arena_observe_selected(
 30313:         layer_index, n_tokens, g_moe_gather.h_sel.data(), slot_count);
 30314:     if (route_prof) route_t_observe = cuda_wall_sec();
 30315: 
 30316:     /* 2. dedupe -> compact ids (ascending expert order) + per-slot remap */
 30317:     std::vector<int32_t> &e2s = g_moe_gather.h_expert_to_slot;
 30318:     e2s.assign(n_total_expert, -1);
 30319:     std::vector<int32_t> &compact = g_moe_gather.h_compact_ids; compact.clear();
 30320:     std::vector<int32_t> &slots = g_moe_gather.h_slot_ids; slots.resize(slot_count);
 30321:     for (uint32_t i = 0; i < slot_count; i++) {
 30322:         const int32_t e = g_moe_gather.h_sel[i];
 30323:         if (e < 0 || (uint32_t)e >= n_total_expert) return 0;
 30324:         e2s[(uint32_t)e] = -2;
 30325:     }
 30326:     for (uint32_t e = 0; e < n_total_expert; e++) {
 30327:         if (e2s[e] != -2) continue;
 30328:         e2s[e] = (int32_t)compact.size();
 30329:         compact.push_back((int32_t)e);
 30330:     }
 30331:     for (uint32_t i = 0; i < slot_count; i++)
 30332:         slots[i] = e2s[(uint32_t)g_moe_gather.h_sel[i]];
 30333:     const uint32_t compact_count = (uint32_t)compact.size();
 30334:     if (compact_count == 0 || compact_count > n_total_expert) return 0;
 30335:     if (route_prof) route_t_map = cuda_wall_sec();
 30336: 
 30337:     if (cuda_prefill_waves_requested() && n_tokens > 1u) {
 30338:         if (cuda_prefill_wave_debug_dump_active()) {
 30339:             g_moe_gather.wave_fail_closed = 1;
 30340:             fprintf(stderr,
 30341:                     "ds4: [prefill-waves] refused reason=debug-dump-active layer=%u tokens=%u compact=%u\n",
 30342:                     layer_index, n_tokens, compact_count);
 30343:             return 0;
 30344:         }
 30345:         const int double_buffer_requested =
 30346:             cuda_prefill_wave_double_buffer_requested();
 30347:         if (double_buffer_requested < 0) {
 30348:             g_moe_gather.wave_fail_closed = 1;
 30349:             return 0;
 30350:         }
 30351:         uint32_t max_wave_experts = cuda_prefill_wave_force_experts();
 30352:         if (max_wave_experts == UINT32_MAX) {
 30353:             g_moe_gather.wave_fail_closed = 1;
 30354:             return 0;
 30355:         }
 30356:         if (max_wave_experts == 0) {
 30357:             const uint32_t budget_experts =
 30358:                 cuda_prefill_wave_budget_experts(gate_expert_bytes,
 30359:                                                  down_expert_bytes,
 30360:                                                  double_buffer_requested > 0);
 30361:             if (budget_experts != 0 && compact_count > budget_experts) {
 30362:                 max_wave_experts = budget_experts;
 30363:             }
 30364:         }
 30365:         if (max_wave_experts != 0 && max_wave_experts < compact_count) {
 30366:             g_moe_gather.wave_active = 1;
 30367:             g_moe_gather.wave_fail_closed = 1;
 30368:             g_moe_gather.wave_double_buffer =
 30369:                 double_buffer_requested > 0 ? 1u : 0u;
 30370:             g_moe_gather.wave_max_experts = max_wave_experts;
 30371:             g_moe_gather.wave_count =
 30372:                 (compact_count + max_wave_experts - 1u) / max_wave_experts;
 30373:             g_moe_gather.wave_unique_experts = compact_count;
 30374:             if (!cuda_moe_selected_stage_wave(
 30375:                     model_map, layer_index,
 30376:                     gate_offset, up_offset, down_offset,
 30377:                     gate_expert_bytes, down_expert_bytes,
 30378:                     n_total_expert, n_expert, n_tokens, 0u)) {
 30379:                 if (g_moe_gather.wave_double_buffer) {
 30380:                     g_prefill_wave_overlap_stats.failures =
 30381:                         cuda_u64_saturating_add(
 30382:                             g_prefill_wave_overlap_stats.failures, 1);
 30383:                 }
 30384:                 return 0;
 30385:             }
 30386:             g_prefill_wave_stats.activations = cuda_u64_saturating_add(
 30387:                 g_prefill_wave_stats.activations, 1);
 30388:             g_prefill_wave_stats.unique_experts = cuda_u64_saturating_add(
 30389:                 g_prefill_wave_stats.unique_experts, compact_count);
 30390:             if (max_wave_experts > g_prefill_wave_stats.max_wave) {
 30391:                 g_prefill_wave_stats.max_wave = max_wave_experts;
 30392:             }
 30393:             if (g_moe_gather.wave_double_buffer) {
 30394:                 g_prefill_wave_overlap_stats.activations =
 30395:                     cuda_u64_saturating_add(
 30396:                         g_prefill_wave_overlap_stats.activations, 1);
 30397:             }
 30398:             if (route_prof) {
 30399:                 route_t_transport = cuda_wall_sec();
 30400:                 const double route_t_publish = cuda_wall_sec();
 30401:                 g_route_prof_calls++;
 30402:                 g_route_prof_d2h_s += route_t_d2h - route_t0;
 30403:                 g_route_prof_observe_s += route_t_observe - route_t_d2h;
 30404:                 g_route_prof_map_s += route_t_map - route_t_observe;
 30405:                 g_route_prof_transport_s += route_t_transport - route_t_map;
 30406:                 g_route_prof_publish_s += route_t_publish - route_t_transport;
 30407:             }
 30408:             return 1;
 30409:         }
 30410:     }
 30411: 
 30412:     /* 3. compact device buffers (grow-only, reused across layers/tokens) */
 30413:     const uint64_t cgate = (uint64_t)compact_count * gate_expert_bytes;
 30414:     const uint64_t cdown = (uint64_t)compact_count * down_expert_bytes;
 30415:     const int slot_will_grow = !g_moe_gather.slot ||
 30416:         g_moe_gather.slot_cap < (uint64_t)slot_count * sizeof(int32_t);
 30417:     const int iq1_sidecar_route = g_iq1_s_sidecar_file_valid &&
 30418:         model_map == g_iq1_s_sidecar_host_base;
 30419:     const int iq1_vram_cache_state = iq1_sidecar_route ?
 30420:         cuda_iq1_s_vram_cache_prepare(
 30421:             gate_expert_bytes, down_expert_bytes) : 0;
 30422:     if (iq1_vram_cache_state < 0 ||
 30423:         (iq1_vram_cache_state > 0 && compact_count != 1u)) {
 30424:         return 0;
 30425:     }
 30426:     const int iq1_vram_cache_active = iq1_vram_cache_state > 0;
 30427:     const int iq1_packed_h2d = !iq1_vram_cache_active &&
 30428:         compact_count == 1u && iq1_sidecar_route &&
 30429:         getenv("DS4_IQ1_S_PACKED_H2D") != NULL;
 30430:     const uint64_t iq1_packed_bytes = cgate * 2ull + cdown;
 30431:     if ((!iq1_vram_cache_active && iq1_packed_h2d &&
 30432:          !cuda_moe_gather_ensure(&g_moe_gather.iq1_packed,
 30433:                                  &g_moe_gather.iq1_packed_cap,
 30434:                                  iq1_packed_bytes,
 30435:                                  "moe IQ1_S packed gather")) ||
 30436:         (!iq1_vram_cache_active && !iq1_packed_h2d &&
 30437:          (!cuda_moe_gather_ensure(&g_moe_gather.gate, &g_moe_gather.gate_cap, cgate, "moe gather gate") ||
 30438:           !cuda_moe_gather_ensure(&g_moe_gather.up,   &g_moe_gather.up_cap,   cgate, "moe gather up") ||
 30439:           !cuda_moe_gather_ensure(&g_moe_gather.down, &g_moe_gather.down_cap, cdown, "moe gather down"))) ||
 30440:         !cuda_moe_gather_ensure_i32(&g_moe_gather.slot, &g_moe_gather.slot_cap, slot_count, "moe gather slots"))
 30441:         return 0;
 30442:     if (slot_will_grow) g_moe_gather.iq1_slot_zero_valid = 0;
 30443:     g_moe_gather.iq1_active_gate = NULL;
 30444:     g_moe_gather.iq1_active_up = NULL;
 30445:     g_moe_gather.iq1_active_down = NULL;
 30446: 
 30447:     /* Dynamic-arena hits go straight from their stable pinned slot to the
 30448:      * compact execution buffers. They never enter the legacy pread/staging
 30449:      * list and never take a detour through the persistent VRAM cache. */
 30450:     std::vector<uint8_t> &arena_resident = g_moe_gather.h_arena_resident;
 30451:     arena_resident.assign(compact_count, 0);
 30452:     const int nested_exact = cuda_nested_residual_upload_exact(
 30453:         layer_index,
 30454:         gate_offset, up_offset, down_offset,
 30455:         gate_expert_bytes, down_expert_bytes,
 30456:         compact);
 30457:     if (nested_exact < 0) {
 30458:         g_moe_gather.wave_fail_closed = 1;
 30459:         return 0;
 30460:     }
 30461:     if (nested_exact > 0) {
 30462:         std::fill(arena_resident.begin(), arena_resident.end(), 1u);
 30463:     }
 30464:     const int arena_active = nested_exact == 0 && g_dynamic_arena.host_base &&
 30465:         g_dynamic_arena.snapshot_generation != 0 &&
 30466:         !g_dynamic_arena.hits_disabled &&
 30467:         !g_dynamic_arena.submissions_blocked;
 30468:     if (arena_active) {
 30469:         for (uint32_t i = 0; i < compact_count; i++) {
 30470:             const uint32_t expert = (uint32_t)compact[i];
 30471:             const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
 30472:             const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
 30473:             const cuda_dynamic_arena_copy_status arena_status =
 30474:                 cuda_dynamic_arena_copy_expert_async(
 30475:                     g_dynamic_arena,
 30476:                     model_map, layer_index, expert,
 30477:                     gate_offset, up_offset, down_offset,
 30478:                     g_moe_gather.gate + gate_dst,
 30479:                     g_moe_gather.up + gate_dst,
 30480:                     g_moe_gather.down + down_dst,
 30481:                     gate_expert_bytes, gate_expert_bytes,
 30482:                     down_expert_bytes);
 30483:             if (arena_status == CUDA_DYNAMIC_ARENA_FATAL) {
 30484:                 (void)cudaStreamSynchronize(g_model_upload_stream);
 30485:                 return 0;
 30486:             }
 30487:             arena_resident[i] =
 30488:                 arena_status == CUDA_DYNAMIC_ARENA_ENQUEUED ? 1u : 0u;
 30489:         }
 30490:     }
 30491: 
 30492:     std::vector<uint8_t> &iq1_ram_resident =
 30493:         g_moe_gather.h_iq1_ram_resident;
 30494:     iq1_ram_resident.assign(compact_count, 0);
 30495:     const int iq1_profile = getenv("DS4_IQ1_S_PROFILE") != NULL;
 30496:     const double iq1_enqueue_t0 = iq1_profile ? cuda_wall_sec() : 0.0;
 30497:     uint32_t iq1_upload_experts = 0;
 30498:     int iq1_vram_pending_slot = -1;
 30499:     int iq1_vram_pending_new = 0;
 30500:     if (iq1_sidecar_route) {
 30501:         for (uint32_t i = 0; i < compact_count; i++) {
 30502:             const uint32_t expert = (uint32_t)compact[i];
 30503:             char *iq1_vram_base = NULL;
 30504:             if (iq1_vram_cache_active) {
 30505:                 int vram_hit = 0;
 30506:                 int vram_evicted = 0;
 30507:                 const int vram_slot = cuda_iq1_s_vram_cache_resolve_slot(
 30508:                     layer_index, expert, &vram_hit, &vram_evicted);
 30509:                 if (vram_slot < 0) return 0;
 30510:                 iq1_vram_base = cuda_iq1_s_vram_cache_slot_base(
 30511:                     (uint32_t)vram_slot);
 30512:                 if (!iq1_vram_base) return 0;
 30513:                 g_moe_gather.iq1_active_gate = iq1_vram_base;
 30514:                 g_moe_gather.iq1_active_up =
 30515:                     iq1_vram_base + gate_expert_bytes;
 30516:                 g_moe_gather.iq1_active_down =
 30517:                     iq1_vram_base + gate_expert_bytes * 2ull;
 30518:                 iq1_ram_resident[i] = 1u;
 30519:                 if (vram_hit) continue;
 30520:                 iq1_vram_pending_slot = vram_slot;
 30521:                 iq1_vram_pending_new = !vram_evicted;
 30522:             }
 30523:             const uint64_t gate_src = gate_offset +
 30524:                 (uint64_t)expert * gate_expert_bytes;
 30525:             const uint64_t up_src = up_offset +
 30526:                 (uint64_t)expert * gate_expert_bytes;
 30527:             const uint64_t down_src = down_offset +
 30528:                 (uint64_t)expert * down_expert_bytes;
 30529:             const char *gate_host = NULL;
 30530:             const char *up_host = NULL;
 30531:             const char *down_host = NULL;
 30532:             const int resolved = cuda_iq1_s_ram_cache_resolve(
 30533:                 layer_index, expert, gate_src, up_src, down_src,
 30534:                 gate_expert_bytes, down_expert_bytes,
 30535:                 &gate_host, &up_host, &down_host);
 30536:             if (resolved < 0) return 0;
 30537:             if (resolved == 0) {
 30538:                 if (iq1_vram_cache_active) {
 30539:                     g_iq1_s_vram_cache.failures++;
 30540:                     return 0;
 30541:                 }
 30542:                 break;
 30543:             }
 30544:             const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
 30545:             const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
 30546:             const int single_copy_h2d = iq1_vram_cache_active ||
 30547:                 iq1_packed_h2d;
 30548:             if (single_copy_h2d &&
 30549:                 (up_host != gate_host + gate_expert_bytes ||
 30550:                  down_host != gate_host + gate_expert_bytes * 2ull)) {
 30551:                 g_iq1_s_ram_cache.failures++;
 30552:                 return 0;
 30553:             }
 30554:             char *single_copy_destination = iq1_vram_cache_active ?
 30555:                 iq1_vram_base : g_moe_gather.iq1_packed;
 30556:             const int copy_ok = single_copy_h2d ?
 30557:                 cudaMemcpyAsync(
 30558:                     single_copy_destination, gate_host,
 30559:                     (size_t)iq1_packed_bytes, cudaMemcpyHostToDevice,
 30560:                     g_model_upload_stream) == cudaSuccess :
 30561:                 (cudaMemcpyAsync(
 30562:                      g_moe_gather.gate + gate_dst, gate_host,
 30563:                      (size_t)gate_expert_bytes, cudaMemcpyHostToDevice,
 30564:                      g_model_upload_stream) == cudaSuccess &&
 30565:                  cudaMemcpyAsync(
 30566:                      g_moe_gather.up + gate_dst, up_host,
 30567:                      (size_t)gate_expert_bytes, cudaMemcpyHostToDevice,
 30568:                      g_model_upload_stream) == cudaSuccess &&
 30569:                  cudaMemcpyAsync(
 30570:                      g_moe_gather.down + down_dst, down_host,
 30571:                      (size_t)down_expert_bytes, cudaMemcpyHostToDevice,
 30572:                      g_model_upload_stream) == cudaSuccess);
 30573:             if (!copy_ok) {
 30574:                 (void)cudaGetLastError();
 30575:                 g_iq1_s_ram_cache.failures++;
 30576:                 if (iq1_vram_cache_active)
 30577:                     g_iq1_s_vram_cache.failures++;
 30578:                 return 0;
 30579:             }
 30580:             g_iq1_s_ram_cache.h2d_copies += single_copy_h2d ? 1u : 3u;
 30581:             iq1_ram_resident[i] = 1u;
 30582:             iq1_upload_experts++;
 30583:             g_iq1_s_ram_cache.h2d_bytes +=
 30584:                 gate_expert_bytes * 2ull + down_expert_bytes;
 30585:             if (iq1_vram_cache_active) {
 30586:                 g_iq1_s_vram_cache.h2d_bytes +=
 30587:                     gate_expert_bytes * 2ull + down_expert_bytes;
 30588:             } else if (iq1_packed_h2d) {
 30589:                 g_moe_gather.iq1_active_gate = g_moe_gather.iq1_packed;
 30590:                 g_moe_gather.iq1_active_up =
 30591:                     g_moe_gather.iq1_packed + gate_expert_bytes;
 30592:                 g_moe_gather.iq1_active_down =
 30593:                     g_moe_gather.iq1_packed + gate_expert_bytes * 2ull;
 30594:             }
 30595:         }
 30596:     }
 30597:     if (iq1_profile && iq1_upload_experts != 0) {
 30598:         g_iq1_s_ram_cache.h2d_batches++;
 30599:         g_iq1_s_ram_cache.h2d_enqueue_seconds +=
 30600:             cuda_wall_sec() - iq1_enqueue_t0;
 30601:     }
 30602: 
 30603:     int spex_prefetch_slot = -1;
 30604:     uint32_t spex_prediction_slot = 0;
 30605:     uint32_t spex_compact_slot = UINT32_MAX;
 30606:     if (spex_queue && spex_key && cuda_moe_expert_cache_requested() == 0) {
 30607:         (void)cuda_spex_prefetch_claim(
 30608:             spex_queue, spex_key,
 30609:             gate_offset, up_offset, down_offset,
 30610:             compact,
 30611:             &spex_prefetch_slot,
 30612:             &spex_prediction_slot,
 30613:             &spex_compact_slot);
 30614:         if (spex_prefetch_slot >= 0 &&
 30615:             spex_compact_slot < arena_resident.size() &&
 30616:             arena_resident[spex_compact_slot]) {
 30617:             cuda_spex_prefetch_release(
 30618:                 spex_queue, spex_prefetch_slot, 1, 0);
 30619:             spex_prefetch_slot = -1;
 30620:             spex_compact_slot = UINT32_MAX;
 30621:         }
 30622:     }
 30623: 
 30624:     /* 4. Resolve exact-byte resident hits, admit misses into persistent VRAM
 30625:      * slots when enabled, and stream overflow misses directly into compact
 30626:      * buffers. Cache policy changes residency only; router selection and the
 30627:      * compact slot map are unchanged. */
 30628:     const int prof = getenv("DS4_CUDA_SEL_PROFILE") != NULL;
 30629:     const double t_fetch0 = prof ? cuda_wall_sec() : 0.0;
 30630:     cuda_moe_expert_cache *cache = allow_prefill_mixed_route_ptrs ? NULL :
 30631:         cuda_moe_expert_cache_prepare(gate_expert_bytes, down_expert_bytes);
 30632:     /* P3 owns persistent residency. The generic selected-load path still
 30633:      * serves prefill exactly through compact buffers, but must not smuggle a
 30634:      * first-touch expert straight into the persistent VRAM cache. Decode uses
 30635:      * the fail-closed GPU route worker below. */
 30636:     if (cuda_moe_tiering_mode_requested() == CUDA_MOE_TIER_ENFORCE) {
 30637:         cache = NULL;
 30638:     }
 30639:     std::vector<cuda_moe_gather::span> &spans = g_moe_gather.h_spans;
 30640:     std::vector<int32_t> &cache_slots = g_moe_gather.h_cache_slots;
 30641:     std::vector<uint8_t> &claimed = g_moe_gather.h_cache_claimed;
 30642:     std::vector<uint32_t> &admission_compact = g_moe_gather.h_cache_admission_compact;
 30643:     std::vector<uint32_t> &admission_slots = g_moe_gather.h_cache_admission_slots;
 30644:     std::vector<uint8_t> &admission_evicted = g_moe_gather.h_cache_admission_evicted;
 30645:     spans.clear();
 30646:     spans.reserve((size_t)compact_count * 3u);
 30647:     cache_slots.assign(compact_count, -1);
 30648:     claimed.assign(cache ? cache->capacity : 0u, 0u);
 30649:     admission_compact.clear();
 30650:     admission_slots.clear();
 30651:     admission_evicted.clear();
 30652:     admission_compact.reserve(compact_count);
 30653:     admission_slots.reserve(compact_count);
 30654:     admission_evicted.reserve(compact_count);
 30655: 
 30656:     uint64_t local_hits = 0;
 30657:     uint64_t local_misses = 0;
 30658:     uint64_t local_evictions = 0;
 30659:     uint64_t local_direct = 0;
 30660:     int cache_map_dirty = 0;
 30661: 
 30662:     auto expert_key = [&](uint32_t compact_slot) {
 30663:         const uint64_t e = (uint64_t)(uint32_t)compact[compact_slot];
 30664:         return cuda_moe_cache_key{
 30665:             gate_offset + e * gate_expert_bytes,
 30666:             up_offset + e * gate_expert_bytes,
 30667:             down_offset + e * down_expert_bytes,
 30668:         };
 30669:     };
 30670:     auto append_span = [&](char *destination,
 30671:                            uint64_t source,
 30672:                            uint64_t bytes,
 30673:                            uint32_t expert,
 30674:                            uint8_t part) {
 30675:         uint32_t arena_entry = UINT32_MAX;
 30676:         char *mirror = cuda_dynamic_arena_observer_mirror_ptr(
 30677:             model_map, layer_index, expert, part, source, bytes,
 30678:             &arena_entry);
 30679:         spans.push_back({
 30680:             destination, source, bytes, mirror, arena_entry, part

2026-07-21T09:55:28.412222Z ERROR codex_models_manager::manager: failed to renew cache TTL: missing field `supports_reasoning_summaries` at line 88 column 5
exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; $p='"'C:\\Users\\imanu\\g130i\\ds4_cuda.cu'; Show-Lines "'$p 900 1025; Show-Lines $p 6085 6220; Show-Lines $p 13435 13482' in C:\Users\imanu\g130i\prototype_spec
 succeeded in 612ms:
   900:     g_sparse_bake_route_rejections = 0;
   901: }
   902: static void *g_cuda_tmp;
   903: static uint64_t g_cuda_tmp_bytes;
   904: static void *g_iq1_mixed_scratch;
   905: static uint64_t g_iq1_mixed_scratch_bytes;
   906: static void *g_embed_row_host;
   907: static __half *g_embed_row_device;
   908: static uint64_t g_embed_row_capacity;
   909: static int g_embed_row_notice_printed;
   910: static void *g_model_stage_raw[4];
   911: static void *g_model_stage[4];
   912: static cudaEvent_t g_model_stage_event[4];
   913: #ifdef _WIN32
   914: static OVERLAPPED g_moe_io_ov[4];
   915: static HANDLE g_moe_io_event[4];
   916: static int g_moe_io_stage_used[4];
   917: #endif
   918: static uint64_t g_model_stage_bytes;
   919: 
   920: enum ds4_gpu_arena_slot_state : uint32_t {
   921:     DS4_GPU_ARENA_FREE = 0,
   922:     DS4_GPU_ARENA_RETIRING,
   923:     DS4_GPU_ARENA_LOADING,
   924:     DS4_GPU_ARENA_STAGED,
   925:     DS4_GPU_ARENA_READY,
   926:     DS4_GPU_ARENA_POISONED,
   927: };
   928: 
   929: struct cuda_dynamic_arena_binding {
   930:     uint32_t slot;
   931:     uint32_t state;
   932:     uint64_t slot_generation;
   933:     uint64_t snapshot_generation;
   934: };
   935: 
   936: struct ds4_gpu_dynamic_arena_layer {
   937:     uint64_t gate_offset;
   938:     uint64_t up_offset;
   939:     uint64_t down_offset;
   940:     uint64_t gate_expert_bytes;
   941:     uint64_t up_expert_bytes;
   942:     uint64_t down_expert_bytes;
   943: };
   944: 
   945: struct ds4_gpu_dynamic_arena_load {
   946:     uint32_t layer;
   947:     uint32_t expert;
   948:     uint32_t slot;
   949:     uint64_t slot_generation;
   950:     void *host_ptr;
   951:     uint64_t host_bytes;
   952: };
   953: 
   954: struct cuda_dynamic_arena_slot {
   955:     char *host_ptr;
   956:     uint32_t layer;
   957:     uint32_t expert;
   958:     uint64_t content_generation;
   959:     uint64_t checksum;
   960:     uint64_t last_dma_sequence;
   961:     ds4_gpu_arena_slot_state state;
   962:     uint8_t pageable;
   963: };
   964: 
   965: struct ds4_gpu_dynamic_arena_txn;
   966: 
   967: enum cuda_dynamic_arena_backing : uint32_t {
   968:     CUDA_DYNAMIC_ARENA_BACKING_NONE = 0,
   969:     CUDA_DYNAMIC_ARENA_BACKING_PRIMARY,
   970:     CUDA_DYNAMIC_ARENA_BACKING_Q1_0,
   971: };
   972: 
   973: struct cuda_dynamic_arena {
   974:     const void *model_map;
   975:     uint64_t model_size;
   976:     uint64_t slot_bytes;
   977:     uint64_t allocated_bytes;
   978:     uint64_t snapshot_generation;
   979:     uint64_t next_generation;
   980:     uint64_t dma_sequence;
   981:     uint64_t hits;
   982:     uint64_t misses;
   983:     uint64_t fatal_errors;
   984:     uint64_t bytes_uploaded;
   985:     uint64_t pinned_hits;
   986:     uint64_t pageable_hits;
   987:     uint64_t pinned_bytes_uploaded;
   988:     uint64_t pageable_bytes_uploaded;
   989:     uint64_t request_sequence;
   990:     uint32_t n_layer;
   991:     uint32_t n_expert;
   992:     uint32_t active_layer_first;
   993:     uint32_t active_layer_last;
   994:     cuda_dynamic_arena_backing backing;
   995:     int submissions_blocked;
   996:     int hits_disabled;
   997:     int tiering_exclusive;
   998:     char *host_base;
   999:     char *pageable_base;
  1000:     uint64_t pageable_bytes;
  1001:     uint64_t host_budget_bytes;
  1002:     char *ssd_wrap_ssd_ring_base;
  1003:     char *ssd_wrap_h2d_ring_base;
  1004:     uint32_t ssd_wrap_ssd_ring_slots;
  1005:     uint32_t ssd_wrap_h2d_ring_slots;
  1006:     uint32_t pinned_slot_count;
  1007:     uint32_t pageable_slot_count;
  1008:     std::vector<ds4_gpu_dynamic_arena_layer> layers;
  1009:     std::vector<cuda_dynamic_arena_slot> slots;
  1010:     std::vector<cuda_dynamic_arena_binding> active;
  1011:     std::vector<cuda_dynamic_arena_binding> staging;
  1012:     std::vector<cuda_dynamic_arena_binding> preloaded;
  1013:     std::vector<uint8_t> preloaded_parts;
  1014:     ds4_gpu_dynamic_arena_txn *txn;
  1015: };
  1016: 
  1017: struct ds4_gpu_dynamic_arena_txn {
  1018:     uint64_t base_generation;
  1019:     uint64_t target_generation;
  1020:     cudaEvent_t retire_event;
  1021:     int failed;
  1022:     std::vector<uint8_t> target;
  1023:     std::vector<ds4_gpu_dynamic_arena_load> loads;
  1024:     std::vector<uint32_t> reserved_slots;
  1025: };
  6085:         arena.pageable_hits++;
  6086:         arena.pageable_bytes_uploaded += uploaded;
  6087:     } else {
  6088:         arena.pinned_hits++;
  6089:         arena.pinned_bytes_uploaded += uploaded;
  6090:     }
  6091:     return CUDA_DYNAMIC_ARENA_ENQUEUED;
  6092: }
  6093: 
  6094: static void cuda_dynamic_arena_storage_release(void) {
  6095:     cuda_q1_0_ssd_wrap_release(0);
  6096:     if (g_dynamic_arena.txn) {
  6097:         ds4_gpu_dynamic_arena_abort(g_dynamic_arena.txn);
  6098:     }
  6099:     if (g_dynamic_arena.host_base) {
  6100:         if (g_model_upload_stream) {
  6101:             (void)cudaStreamSynchronize(g_model_upload_stream);
  6102:         }
  6103:         (void)cudaFreeHost(g_dynamic_arena.host_base);
  6104:         g_dynamic_arena.host_base = NULL;
  6105:     }
  6106:     if (g_dynamic_arena.pageable_base) {
  6107: #ifdef _WIN32
  6108:         (void)VirtualFree(g_dynamic_arena.pageable_base, 0, MEM_RELEASE);
  6109: #else
  6110:         free(g_dynamic_arena.pageable_base);
  6111: #endif
  6112:         g_dynamic_arena.pageable_base = NULL;
  6113:     }
  6114:     g_dynamic_arena.allocated_bytes = 0;
  6115:     g_dynamic_arena.pageable_bytes = 0;
  6116:     g_dynamic_arena.host_budget_bytes = 0;
  6117:     g_dynamic_arena.ssd_wrap_ssd_ring_base = NULL;
  6118:     g_dynamic_arena.ssd_wrap_h2d_ring_base = NULL;
  6119:     g_dynamic_arena.ssd_wrap_ssd_ring_slots = 0;
  6120:     g_dynamic_arena.ssd_wrap_h2d_ring_slots = 0;
  6121:     g_dynamic_arena.pinned_slot_count = 0;
  6122:     g_dynamic_arena.pageable_slot_count = 0;
  6123:     g_dynamic_arena.snapshot_generation = 0;
  6124:     g_dynamic_arena.submissions_blocked = 0;
  6125:     g_dynamic_arena.hits_disabled = 0;
  6126:     g_dynamic_arena.tiering_exclusive = 0;
  6127:     g_dynamic_arena.slots.clear();
  6128:     g_dynamic_arena.active.clear();
  6129:     g_dynamic_arena.staging.clear();
  6130:     g_dynamic_arena.preloaded.clear();
  6131:     g_dynamic_arena.preloaded_parts.clear();
  6132: }
  6133: 
  6134: static double cuda_dynamic_arena_min_available_gib(void);
  6135: 
  6136: static void cuda_q1_0_dynamic_arena_release(int report) {
  6137:     cuda_dynamic_arena &arena = g_q1_0_dynamic_arena;
  6138:     if (report && arena.host_base && cuda_q1_0_profile_requested()) {
  6139:         const double h2d_enqueue_seconds =
  6140:             g_q1_0_profile.pinned_h2d_enqueue_seconds +
  6141:             g_q1_0_profile.pageable_h2d_enqueue_seconds;
  6142:         fprintf(stderr,
  6143:                 "ds4: [q1-0-profile] result=summary enabled=1 "
  6144:                 "resident_hits_total=%llu pinned_route_hits=%llu "
  6145:                 "pageable_route_hits=%llu resident_h2d_bytes_total=%llu "
  6146:                 "pinned_h2d_bytes=%llu pageable_h2d_bytes=%llu "
  6147:                 "h2d_enqueue_seconds_total=%.9f "
  6148:                 "pinned_h2d_enqueue_seconds=%.9f "
  6149:                 "pageable_h2d_enqueue_seconds=%.9f "
  6150:                 "upload_sync_calls=%llu upload_sync_seconds_total=%.9f "
  6151:                 "pinned_upload_sync_seconds=%.9f "
  6152:                 "pageable_upload_sync_seconds=%.9f "
  6153:                 "sync_attribution=bytes q1_kernel_calls=%llu "
  6154:                 "q1_kernel_seconds=%.9f mixed_join_calls=%llu "
  6155:                 "mixed_join_seconds=%.9f timer_failures=%llu\n",
  6156:                 (unsigned long long)arena.hits,
  6157:                 (unsigned long long)arena.pinned_hits,
  6158:                 (unsigned long long)arena.pageable_hits,
  6159:                 (unsigned long long)arena.bytes_uploaded,
  6160:                 (unsigned long long)arena.pinned_bytes_uploaded,
  6161:                 (unsigned long long)arena.pageable_bytes_uploaded,
  6162:                 h2d_enqueue_seconds,
  6163:                 g_q1_0_profile.pinned_h2d_enqueue_seconds,
  6164:                 g_q1_0_profile.pageable_h2d_enqueue_seconds,
  6165:                 (unsigned long long)g_q1_0_profile.upload_sync_calls,
  6166:                 g_q1_0_profile.upload_sync_seconds,
  6167:                 g_q1_0_profile.pinned_upload_sync_seconds,
  6168:                 g_q1_0_profile.pageable_upload_sync_seconds,
  6169:                 (unsigned long long)g_q1_0_profile.q1_kernel_calls,
  6170:                 g_q1_0_profile.q1_kernel_seconds,
  6171:                 (unsigned long long)g_q1_0_profile.mixed_join_calls,
  6172:                 g_q1_0_profile.mixed_join_seconds,
  6173:                 (unsigned long long)g_q1_0_profile.timer_failures);
  6174:     }
  6175:     if (report && arena.host_base &&
  6176:         (arena.hits || arena.misses || arena.fatal_errors)) {
  6177:         fprintf(stderr,
  6178:                 "ds4: [arena] final hits=%llu misses=%llu fatal=%llu uploaded=%.2f GiB backing=q1_0\n",
  6179:                 (unsigned long long)arena.hits,
  6180:                 (unsigned long long)arena.misses,
  6181:                 (unsigned long long)arena.fatal_errors,
  6182:                 (double)arena.bytes_uploaded / 1073741824.0);
  6183:     }
  6184:     if (arena.host_base) {
  6185:         if (g_model_upload_stream) {
  6186:             (void)cudaStreamSynchronize(g_model_upload_stream);
  6187:         }
  6188:         (void)cudaFreeHost(arena.host_base);
  6189:     }
  6190:     if (arena.pageable_base) {
  6191: #ifdef _WIN32
  6192:         (void)VirtualFree(arena.pageable_base, 0, MEM_RELEASE);
  6193: #else
  6194:         free(arena.pageable_base);
  6195: #endif
  6196:     }
  6197:     arena = cuda_dynamic_arena{};
  6198:     g_q1_0_profile = cuda_q1_0_profile_stats{};
  6199:     cuda_q1_0_mixed_profile_reset(0);
  6200:     g_q1_0_dual_sparse_snapshot = cuda_q1_0_dual_sparse_snapshot{};
  6201: }
  6202: 
  6203: static void cuda_primary_dynamic_arena_release(int report) {
  6204:     cuda_q1_0_ssd_wrap_release(report);
  6205:     if (report && g_dynamic_arena.host_base &&
  6206:         (g_dynamic_arena.hits || g_dynamic_arena.misses ||
  6207:          g_dynamic_arena.fatal_errors)) {
  6208:         if (g_dynamic_arena.backing == CUDA_DYNAMIC_ARENA_BACKING_Q1_0) {
  6209:             fprintf(stderr,
  6210:                     "ds4: [arena] final hits=%llu pinned_hits=%llu pageable_hits=%llu misses=%llu fatal=%llu uploaded=%.2f GiB pinned_uploaded=%.2f GiB pageable_uploaded=%.2f GiB backing=q1_0\n",
  6211:                     (unsigned long long)g_dynamic_arena.hits,
  6212:                     (unsigned long long)g_dynamic_arena.pinned_hits,
  6213:                     (unsigned long long)g_dynamic_arena.pageable_hits,
  6214:                     (unsigned long long)g_dynamic_arena.misses,
  6215:                     (unsigned long long)g_dynamic_arena.fatal_errors,
  6216:                     (double)g_dynamic_arena.bytes_uploaded / 1073741824.0,
  6217:                     (double)g_dynamic_arena.pinned_bytes_uploaded /
  6218:                         1073741824.0,
  6219:                     (double)g_dynamic_arena.pageable_bytes_uploaded /
  6220:                         1073741824.0);
 13435:     fprintf(stderr,
 13436:             "ds4: CUDA IQ1_S routed-expert sidecar installed: %.2f GiB\n",
 13437:             (double)model_size / 1073741824.0);
 13438:     return 1;
 13439: }
 13440: 
 13441: extern "C" int ds4_gpu_set_q1_0_sidecar(
 13442:         const os_file_t *file,
 13443:         const void *model_map,
 13444:         uint64_t model_size) {
 13445:     cuda_q1_0_sidecar_clear();
 13446:     if (!file && !model_map && model_size == 0) return 1;
 13447:     if (!file || !os_file_valid(file) || !model_map || model_size == 0) {
 13448:         fprintf(stderr, "ds4: CUDA Q1_0 sidecar rejected incomplete source\n");
 13449:         return 0;
 13450:     }
 13451:     if (os_file_dup(&g_q1_0_sidecar_file, file) != 0) {
 13452:         fprintf(stderr, "ds4: CUDA Q1_0 sidecar file duplicate failed: %s\n",
 13453:                 strerror(errno));
 13454:         return 0;
 13455:     }
 13456:     g_q1_0_sidecar_file_valid = 1;
 13457:     const uint64_t file_size = os_file_size(&g_q1_0_sidecar_file);
 13458:     if (file_size == UINT64_MAX || file_size != model_size) {
 13459:         fprintf(stderr,
 13460:                 "ds4: CUDA Q1_0 sidecar size mismatch: file=%llu map=%llu\n",
 13461:                 (unsigned long long)file_size,
 13462:                 (unsigned long long)model_size);
 13463:         cuda_q1_0_sidecar_clear();
 13464:         return 0;
 13465:     }
 13466:     g_q1_0_sidecar_host_base = model_map;
 13467:     g_q1_0_sidecar_size = model_size;
 13468:     if (!cuda_expert_recovery_trace_configure()) {
 13469:         cuda_q1_0_sidecar_clear();
 13470:         return 0;
 13471:     }
 13472:     fprintf(stderr,
 13473:             "ds4: CUDA Q1_0 routed-expert sidecar installed: %.2f GiB\n",
 13474:             (double)model_size / 1073741824.0);
 13475:     cuda_q1_0_mixed_profile_reset(1);
 13476:     return 1;
 13477: }
 13478: 
 13479: extern "C" int ds4_gpu_set_nested_residual_sidecar(
 13480:         const os_file_t *file,
 13481:         const void *source_map,
 13482:         uint64_t source_size) {

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"q1_0_sidecar_(ready|model)|q1_0_sidecar\" 'C:\\Users\\imanu\\g130i\\ds4.c' | Select-Object -Last 80" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 479ms:
2444:} ds4_q1_0_sidecar;
2455:static ds4_q1_0_sidecar g_q1_0_sidecar;
2542:static bool q1_0_sidecar_layer_active(
2545:    return g_q1_0_sidecar.ready &&
2546:           model == g_q1_0_sidecar.primary_model &&
2547:           layer_index >= g_q1_0_sidecar.first_layer &&
2548:           layer_index <= g_q1_0_sidecar.last_layer &&
2563:    if (q1_0_sidecar_layer_active(model, layer_index) &&
2566:        source.model = g_q1_0_sidecar.model;
2567:        source.gate = g_q1_0_sidecar.gate[layer_index];
2568:        source.up = g_q1_0_sidecar.up[layer_index];
2569:        source.down = g_q1_0_sidecar.down[layer_index];
3304:static void q1_0_sidecar_validate_router_identity(
3394:static void q1_0_sidecar_bind(
3398:    memset(&g_q1_0_sidecar, 0, sizeof(g_q1_0_sidecar));
3435:    q1_0_sidecar_validate_router_identity(
3468:        g_q1_0_sidecar.gate[il] = gate;
3469:        g_q1_0_sidecar.up[il] = up;
3470:        g_q1_0_sidecar.down[il] = down;
3471:        g_q1_0_sidecar.gate_row_bytes[il] = gate_row_bytes;
3472:        g_q1_0_sidecar.gate_expert_bytes[il] = gate_expert_bytes;
3473:        g_q1_0_sidecar.down_row_bytes[il] = down_row_bytes;
3474:        g_q1_0_sidecar.down_expert_bytes[il] = down_expert_bytes;
3479:    g_q1_0_sidecar.model = model;
3480:    g_q1_0_sidecar.primary_model = primary_model;
3481:    g_q1_0_sidecar.first_layer = first_layer;
3482:    g_q1_0_sidecar.last_layer = last_layer;
3483:    g_q1_0_sidecar.ready = true;
11167:        q1_0_sidecar_layer_active(model, il);
11833:        const ds4_tensor *q1_gate = g_q1_0_sidecar.gate[il];
11834:        const ds4_tensor *q1_up = g_q1_0_sidecar.up[il];
11835:        const ds4_tensor *q1_down = g_q1_0_sidecar.down[il];
11860:                g_q1_0_sidecar.model->map,
11861:                g_q1_0_sidecar.model->size,
16522:    ds4_model q1_0_sidecar_model;
16538:    bool q1_0_sidecar_ready;
19402:    os_mmap_init(&e->q1_0_sidecar_model.mmap);
19473:    const char *q1_0_sidecar_path = getenv("DS4_Q1_0_EXPERT_SIDECAR");
19474:    if (q1_0_sidecar_path && q1_0_sidecar_path[0]) {
19482:        model_open(&e->q1_0_sidecar_model,
19483:                   q1_0_sidecar_path,
19486:        q1_0_sidecar_bind(&e->q1_0_sidecar_model, &e->model);
19487:        e->q1_0_sidecar_ready = true;
19489:                q1_0_sidecar_path);
19521:         (!e->q1_0_sidecar_ready || !q1_0_selected_load ||
19552:    if (e->q1_0_sidecar_ready && iq1_s_mixed_cold_one_requested()) {
19671:        if (e->q1_0_sidecar_ready &&
19672:            !ds4_gpu_set_q1_0_sidecar(
19673:                &e->q1_0_sidecar_model.mmap.file,
19674:                e->q1_0_sidecar_model.map,
19675:                e->q1_0_sidecar_model.size)) {
19754:    memset(&g_q1_0_sidecar, 0, sizeof(g_q1_0_sidecar));
19758:    if (e->q1_0_sidecar_ready) model_close(&e->q1_0_sidecar_model);
19855:    if (!model || !g_q1_0_sidecar.ready ||
19856:        model != g_q1_0_sidecar.model ||
19857:        g_q1_0_sidecar.first_layer > g_q1_0_sidecar.last_layer ||
19858:        g_q1_0_sidecar.last_layer >= DS4_N_LAYER) {
19862:    for (uint32_t il = g_q1_0_sidecar.first_layer;
19863:         il <= g_q1_0_sidecar.last_layer; il++) {
19866:                model, g_q1_0_sidecar.gate[il], il, "Q1_0 gate",
19870:                model, g_q1_0_sidecar.up[il], il, "Q1_0 up",
19874:                model, g_q1_0_sidecar.down[il], il, "Q1_0 down",
20219:                        layers, &e->q1_0_sidecar_model);
20222:                        e->q1_0_sidecar_model.map,
20223:                        e->q1_0_sidecar_model.size,
20225:                        g_q1_0_sidecar.first_layer,
20226:                        g_q1_0_sidecar.last_layer);
20280:                        layers, &e->q1_0_sidecar_model);
20283:                        e->q1_0_sidecar_model.map,
20284:                        e->q1_0_sidecar_model.size,
20286:                        g_q1_0_sidecar.first_layer,
20287:                        g_q1_0_sidecar.last_layer);
20302:                    (g_q1_0_sidecar.last_layer -
20303:                     g_q1_0_sidecar.first_layer + 1u) * DS4_N_EXPERT;

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4.c' 19735 19765" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 686ms:
 19735:     return 0;
 19736: }
 19737: 
 19738: void ds4_engine_summary(ds4_engine *e) {
 19739:     model_summary(&e->model);
 19740: }
 19741: 
 19742: void ds4_engine_close(ds4_engine *e) {
 19743:     if (!e) return;
 19744:     weights_free(&e->weights);
 19745:     vocab_free(&e->vocab);
 19746:     ds4_threads_shutdown();
 19747: #ifndef DS4_NO_GPU
 19748:     /* CUDA owns host pointers into the model mmap. Drain and release them
 19749:      * before model_close unmaps either model. */
 19750:     ds4_gpu_cleanup();
 19751: #endif
 19752:     ds4_reap_mask_host_reset();
 19753:     memset(&g_iq1_s_sidecar, 0, sizeof(g_iq1_s_sidecar));
 19754:     memset(&g_q1_0_sidecar, 0, sizeof(g_q1_0_sidecar));
 19755:     if (e->nested_residual_ready) {
 19756:         os_file_close(&e->nested_residual_file);
 19757:     }
 19758:     if (e->q1_0_sidecar_ready) model_close(&e->q1_0_sidecar_model);
 19759:     if (e->iq1_s_sidecar_ready) model_close(&e->iq1_s_sidecar_model);
 19760:     if (e->mtp_ready) model_close(&e->mtp_model);
 19761:     model_close(&e->model);
 19762:     ds4_release_instance_lock();
 19763:     free(e->directional_steering_dirs);
 19764:     free(e->directional_steering_file);
 19765:     free(e);

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n \"routed_moe.*tensor|mixed_q1\" 'C:\\Users\\imanu\\g130i\\ds4_gpu.h'; function Show-Lines("'$Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_gpu.h' 310 430" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 531ms:
772:int ds4_gpu_routed_moe_one_tensor(
801:int ds4_gpu_routed_moe_mixed_iq1_one_tensor(
840:int ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
893:int ds4_gpu_routed_moe_batch_tensor(
   310:         uint64_t                model_size,
   311:         uint64_t                weight_offset,
   312:         uint64_t                in_dim,
   313:         uint64_t                out_dim,
   314:         const ds4_gpu_tensor *x,
   315:         uint64_t                n_tok);
   316: 
   317: int ds4_gpu_matmul_f16_pair_tensor(
   318:         ds4_gpu_tensor       *out_a,
   319:         ds4_gpu_tensor       *out_b,
   320:         const void             *model_map,
   321:         uint64_t                model_size,
   322:         uint64_t                weight_a_offset,
   323:         uint64_t                weight_b_offset,
   324:         uint64_t                in_dim,
   325:         uint64_t                out_dim,
   326:         const ds4_gpu_tensor *x,
   327:         uint64_t                n_tok);
   328: 
   329: int ds4_gpu_matmul_f32_tensor(
   330:         ds4_gpu_tensor       *out,
   331:         const void             *model_map,
   332:         uint64_t                model_size,
   333:         uint64_t                weight_offset,
   334:         uint64_t                in_dim,
   335:         uint64_t                out_dim,
   336:         const ds4_gpu_tensor *x,
   337:         uint64_t                n_tok);
   338: 
   339: int ds4_gpu_repeat_hc_tensor(
   340:         ds4_gpu_tensor       *out,
   341:         const ds4_gpu_tensor *row,
   342:         uint32_t                n_embd,
   343:         uint32_t                n_hc);
   344: 
   345: int ds4_gpu_rms_norm_plain_tensor(
   346:         ds4_gpu_tensor       *out,
   347:         const ds4_gpu_tensor *x,
   348:         uint32_t                n,
   349:         float                   eps);
   350: 
   351: int ds4_gpu_rms_norm_plain_rows_tensor(
   352:         ds4_gpu_tensor       *out,
   353:         const ds4_gpu_tensor *x,
   354:         uint32_t                n,
   355:         uint32_t                rows,
   356:         float                   eps);
   357: 
   358: int ds4_gpu_rms_norm_weight_tensor(
   359:         ds4_gpu_tensor       *out,
   360:         const ds4_gpu_tensor *x,
   361:         const void             *model_map,
   362:         uint64_t                model_size,
   363:         uint64_t                weight_offset,
   364:         uint32_t                n,
   365:         float                   eps);
   366: 
   367: int ds4_gpu_rms_norm_weight_rows_tensor(
   368:         ds4_gpu_tensor       *out,
   369:         const ds4_gpu_tensor *x,
   370:         const void             *model_map,
   371:         uint64_t                model_size,
   372:         uint64_t                weight_offset,
   373:         uint32_t                n,
   374:         uint32_t                rows,
   375:         float                   eps);
   376: 
   377: int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
   378:         ds4_gpu_tensor       *q_out,
   379:         const ds4_gpu_tensor *q,
   380:         const void             *model_map,
   381:         uint64_t                model_size,
   382:         uint64_t                q_weight_offset,
   383:         uint32_t                q_n,
   384:         ds4_gpu_tensor       *kv_out,
   385:         const ds4_gpu_tensor *kv,
   386:         uint64_t                kv_weight_offset,
   387:         uint32_t                kv_n,
   388:         uint32_t                rows,
   389:         float                   eps);
   390: 
   391: int ds4_gpu_head_rms_norm_tensor(
   392:         ds4_gpu_tensor *x,
   393:         uint32_t          n_tok,
   394:         uint32_t          n_head,
   395:         uint32_t          head_dim,
   396:         float             eps);
   397: 
   398: int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
   399:         ds4_gpu_tensor *x,
   400:         uint32_t          n_tok,
   401:         uint32_t          head_dim,
   402:         uint32_t          n_rot);
   403: 
   404: int ds4_gpu_rope_tail_tensor(
   405:         ds4_gpu_tensor *x,
   406:         uint32_t          n_tok,
   407:         uint32_t          n_head,
   408:         uint32_t          head_dim,
   409:         uint32_t          n_rot,
   410:         uint32_t          pos0,
   411:         uint32_t          n_ctx_orig,
   412:         bool              inverse,
   413:         float             freq_base,
   414:         float             freq_scale,
   415:         float             ext_factor,
   416:         float             attn_factor,
   417:         float             beta_fast,
   418:         float             beta_slow);
   419: 
   420: /* Release decode fused KV finalizer: after the standalone RoPE kernel, this
   421:  * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
   422:  * attention cache row in one dispatch. */
   423: int ds4_gpu_kv_fp8_store_raw_tensor(
   424:         ds4_gpu_tensor *kv,
   425:         ds4_gpu_tensor *raw_cache,
   426:         uint32_t          raw_cap,
   427:         uint32_t          row,
   428:         uint32_t          head_dim,
   429:         uint32_t          n_rot);
   430: 

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command 'function Show-Lines($Path,$Start,$End){$i=$Start; Get-Content -LiteralPath $Path | Select-Object -Skip ($Start-1) -First ($End-$Start+1) | ForEach-Object { '"'{0,6}: {1}' -f "'$i,$_ ; $i++ }}; Show-Lines '"'C:\\Users\\imanu\\g130i\\ds4_gpu.h' 750 905" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 516ms:
   750:         bool                    has_bias,
   751:         bool                    hash_mode,
   752:         const ds4_gpu_tensor *logits);
   753: 
   754: int ds4_gpu_router_select_batch_tensor(
   755:         ds4_gpu_tensor       *selected,
   756:         ds4_gpu_tensor       *weights,
   757:         ds4_gpu_tensor       *probs,
   758:         const void             *model_map,
   759:         uint64_t                model_size,
   760:         uint32_t                layer_index,
   761:         uint64_t                bias_offset,
   762:         uint64_t                hash_offset,
   763:         uint32_t                hash_rows,
   764:         uint32_t                n_expert_groups,
   765:         uint32_t                n_group_used,
   766:         bool                    has_bias,
   767:         bool                    hash_mode,
   768:         const ds4_gpu_tensor *logits,
   769:         const ds4_gpu_tensor *tokens,
   770:         uint32_t                n_tokens);
   771: 
   772: int ds4_gpu_routed_moe_one_tensor(
   773:         ds4_gpu_tensor       *out,
   774:         ds4_gpu_tensor       *gate,
   775:         ds4_gpu_tensor       *up,
   776:         ds4_gpu_tensor       *mid,
   777:         ds4_gpu_tensor       *experts,
   778:         const void             *model_map,
   779:         uint64_t                model_size,
   780:         uint32_t                layer_index,
   781:         uint64_t                gate_offset,
   782:         uint64_t                up_offset,
   783:         uint64_t                down_offset,
   784:         uint32_t                gate_type,
   785:         uint32_t                down_type,
   786:         uint64_t                gate_expert_bytes,
   787:         uint64_t                gate_row_bytes,
   788:         uint64_t                down_expert_bytes,
   789:         uint64_t                down_row_bytes,
   790:         uint32_t                expert_in_dim,
   791:         uint32_t                expert_mid_dim,
   792:         uint32_t                out_dim,
   793:         const ds4_gpu_tensor *selected,
   794:         const ds4_gpu_tensor *weights,
   795:         uint32_t                n_expert,
   796:         float                   clamp,
   797:         const ds4_gpu_tensor *x,
   798:         ds4_gpu_spex_queue   *spex_queue,
   799:         const ds4_gpu_spex_key *spex_key);
   800: 
   801: int ds4_gpu_routed_moe_mixed_iq1_one_tensor(
   802:         ds4_gpu_tensor       *out,
   803:         ds4_gpu_tensor       *gate,
   804:         ds4_gpu_tensor       *up,
   805:         ds4_gpu_tensor       *mid,
   806:         ds4_gpu_tensor       *experts,
   807:         const void             *main_model_map,
   808:         uint64_t                main_model_size,
   809:         uint32_t                layer_index,
   810:         uint64_t                main_gate_offset,
   811:         uint64_t                main_up_offset,
   812:         uint64_t                main_down_offset,
   813:         uint32_t                main_gate_type,
   814:         uint32_t                main_down_type,
   815:         uint64_t                main_gate_expert_bytes,
   816:         uint64_t                main_gate_row_bytes,
   817:         uint64_t                main_down_expert_bytes,
   818:         uint64_t                main_down_row_bytes,
   819:         const void             *iq1_model_map,
   820:         uint64_t                iq1_model_size,
   821:         uint64_t                iq1_gate_offset,
   822:         uint64_t                iq1_up_offset,
   823:         uint64_t                iq1_down_offset,
   824:         uint32_t                iq1_down_type,
   825:         uint64_t                iq1_gate_expert_bytes,
   826:         uint64_t                iq1_gate_row_bytes,
   827:         uint64_t                iq1_down_expert_bytes,
   828:         uint64_t                iq1_down_row_bytes,
   829:         uint32_t                expert_in_dim,
   830:         uint32_t                expert_mid_dim,
   831:         uint32_t                out_dim,
   832:         const ds4_gpu_tensor *selected,
   833:         const ds4_gpu_tensor *weights,
   834:         uint32_t                n_expert,
   835:         float                   clamp,
   836:         const ds4_gpu_tensor *x,
   837:         ds4_gpu_spex_queue   *spex_queue,
   838:         const ds4_gpu_spex_key *spex_key);
   839: 
   840: int ds4_gpu_routed_moe_mixed_q1_0_one_tensor(
   841:         ds4_gpu_tensor       *out,
   842:         ds4_gpu_tensor       *gate,
   843:         ds4_gpu_tensor       *up,
   844:         ds4_gpu_tensor       *mid,
   845:         ds4_gpu_tensor       *experts,
   846:         const void             *main_model_map,
   847:         uint64_t                main_model_size,
   848:         uint32_t                layer_index,
   849:         uint64_t                main_gate_offset,
   850:         uint64_t                main_up_offset,
   851:         uint64_t                main_down_offset,
   852:         uint32_t                main_gate_type,
   853:         uint32_t                main_down_type,
   854:         uint64_t                main_gate_expert_bytes,
   855:         uint64_t                main_gate_row_bytes,
   856:         uint64_t                main_down_expert_bytes,
   857:         uint64_t                main_down_row_bytes,
   858:         const void             *q1_model_map,
   859:         uint64_t                q1_model_size,
   860:         uint64_t                q1_gate_offset,
   861:         uint64_t                q1_up_offset,
   862:         uint64_t                q1_down_offset,
   863:         uint64_t                q1_gate_expert_bytes,
   864:         uint64_t                q1_gate_row_bytes,
   865:         uint64_t                q1_down_expert_bytes,
   866:         uint64_t                q1_down_row_bytes,
   867:         uint32_t                expert_in_dim,
   868:         uint32_t                expert_mid_dim,
   869:         uint32_t                out_dim,
   870:         const ds4_gpu_tensor *selected,
   871:         const ds4_gpu_tensor *weights,
   872:         uint32_t                n_expert,
   873:         float                   clamp,
   874:         const ds4_gpu_tensor *x);
   875: 
   876: int ds4_gpu_routed_moe_prepare_selected(
   877:         const void             *model_map,
   878:         uint64_t                model_size,
   879:         uint64_t                gate_offset,
   880:         uint64_t                up_offset,
   881:         uint64_t                down_offset,
   882:         uint64_t                gate_expert_bytes,
   883:         uint64_t                down_expert_bytes,
   884:         const ds4_gpu_tensor *selected,
   885:         uint32_t                n_expert,
   886:         uint32_t                n_tokens);
   887: 
   888: uint32_t ds4_gpu_routed_moe_last_selected(
   889:         uint64_t gate_offset,
   890:         int32_t *out_ids,
   891:         uint32_t out_cap);
   892: 
   893: int ds4_gpu_routed_moe_batch_tensor(
   894:         ds4_gpu_tensor       *out,
   895:         ds4_gpu_tensor       *gate,
   896:         ds4_gpu_tensor       *up,
   897:         ds4_gpu_tensor       *mid,
   898:         ds4_gpu_tensor       *experts,
   899:         const void             *model_map,
   900:         uint64_t                model_size,
   901:         uint32_t                layer_index,
   902:         uint64_t                gate_offset,
   903:         uint64_t                up_offset,
   904:         uint64_t                down_offset,
   905:         uint32_t                gate_type,

exec
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-ChildItem -Force -LiteralPath 'C:\\Users\\imanu\\g130i\\prototype_spec'; git -C 'C:\\Users\\imanu\\g130i' status --short -- 'prototype_spec'" in C:\Users\imanu\g130i\prototype_spec
 succeeded in 522ms:


    Directory: C:\Users\imanu\g130i\prototype_spec


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
-a----        21/07/2026     11:49            196 codex_spec.log                                                       
?? prototype_spec/


warning: unable to access 'C:\Users\imanu/.config/git/ignore': Permission denied
warning: unable to access 'C:\Users\imanu/.config/git/ignore': Permission denied

Located the actual evidence files under `coverage/` and `overlap_spike/`, traced bootstrap/load, route classification, exact H2D, Q1 cold launch, and mixed join, and mapped the CPU exact kernels and six-core affinity result.
  ✓ Inspect design context and locate Q1 resident-load, dispatch, and join seams
  ✓ Inspect CPU GEMV spike and overlap/coverage artifacts for integration constraints and budgets
  → Draft implementation-ready prototype_spec.md with exact citations, milestone, risks, and effort
  • Verify every citation and confirm no runtime source changes
codex
I’ve resolved one important implementation choice for the spec: milestone 1 should consume a static ranked manifest, because the coverage JSON contains aggregate curves but not the ordered `(layer, expert)` entries. Prefill-derived ranking remains a supported follow-on mode at the existing prefill-finalize seam, but it cannot populate a bootstrap resident set before the first prefill.

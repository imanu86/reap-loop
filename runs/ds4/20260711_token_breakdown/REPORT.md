# Per-token time breakdown — is the 3060 ceiling ENGINE-overhead or MEMORY-bound?

Host: RTX 3060 12 GB, WSL2 Ubuntu-24.04, sm_86. Bin `/root/ds4_pin/ds4`
(same binary/config as S0 `20260711_s0_exposed_stall`). Model `/root/models/ds4-2bit.gguf`
(86.7 GB, 43 MoE layers, 6 experts/token). Task **cyberpunk-wide**, static REAP mask **K23**,
`--ssd-streaming-cache-experts 32`, ctx 2048, greedy temp0. **Warm RAM** (buffcache ~57 GB),
`hit_rate=0.000` (refetch every token) — the exact S0 baseline regime (3.38–3.48 t/s).
Config (vaccinations): own lock `/tmp/ds4_tokbreak.lock` (UI:8000 never touched — CLI binds no
port; GPU exclusive), `DS4_CUDA_NO_Q8_F16_CACHE=1`, warm levers `NO_DIRECT_IO`+`KEEP_MODEL_PAGES`,
`RESERVE_GB=1` (parse-safe, NOT 16), `DS4_SPEX_STATS=1`. GPU left free (685 MiB idle) at the end.

## Method (cross-checked, 3 independent views)
1. **nsys CUDA trace** (`--trace=cuda`) of two warm runs, **n=40** and **n=120**.
   WSL2's nsys captures the **CPU-side CUDA-API timeline** (RUNTIME 642k calls, SYNCHRONIZATION
   128k) but **not** GPU-HW activity tables (a known WSL2 CUPTI limitation). That is sufficient:
   the decode critical path *is* the CPU thread(s), and every API call is either a **SYNC-WAIT**
   (`cudaStreamSynchronize`/`cudaDeviceSynchronize` — CPU blocked until the GPU drains, so its
   duration ≈ GPU busy on the critical path), a **blocking `cudaMemcpy`**, a cheap **LAUNCH/ISSUE**
   (`cudaLaunchKernel`/`cudaMemcpyAsync`/`cudaEventRecord`), or one-time **SETUP**.
2. **Differential (n120 − n40)/80** cancels the identical model-load+prefill prefix → clean
   per-token numbers. `union()` (any-CUDA-call) and `gap()` (no-CUDA-call) are additive under the
   differential (shared prefix cancels), so **gap = pure CPU host time with the GPU un-driven**.
3. **SPEX counters** (`sync_ms`, copied bytes) + the **S0 A/B** (residency vs t/s) as third view.

Determinism: CLEAN_A ≡ CLEAN_B bit-exact (md5 `c8b9d6`). The nsys runs drift by a token
(async-float MoE reorder under profiling) — irrelevant to timing (gen t/s stable 3.25–3.48).

## The measured breakdown (per generated token)
Clean wall = **287 ms** (gen 3.48 t/s, CLEAN_B; S0 ref 296 ms / 3.38). nsys wall (differential)
= 304 ms — only **6 % slower than clean**, so the CPU "gap" below is *real*, not a profiler artifact.

| voce | ms/token (clean-scaled) | % of wall | source |
|---|---|---|---|
| **CPU host orchestration — pure gap** (no CUDA call; routing/expert-select/sample/LRU) | **121** | **42 %** | wall − union(any API) |
| **GPU-blocked** (CPU waiting on GPU work) — *near-serial, cross-thread overlap ≈ 3 ms* | **140** | **49 %** | union(sync ∪ blkcpy) |
|  ├ Expert **H2D copy** wait `cudaStreamSynchronize` — **EXPOSED** | 76 | 27 % | diff 81 ms nsys |
|  ├ **Compute** attn+MoE wait `cudaDeviceSynchronize` | 50 | 17 % | diff 53 ms nsys |
|  └ Small blocking transfers `cudaMemcpy` (logits/meta) | 17 | 6 % | diff 18 ms nsys |
| **Launch / enqueue** (net, not overlapping a sync) | 26 | 9 % | wall − gap − union_sync |
| **TOTAL** | **287** | 100 % | |

Two threads confirm the structure: a **dedicated upload worker** does 714 `cudaMemcpyAsync`+
`cudaStreamSynchronize`/token (the reactive expert copy, 1.95 GB/token H2D ≈ 24 GB/s = **PCIe-bound**);
the **main thread** does all `cudaLaunchKernel` + `cudaDeviceSynchronize` (compute). Their sync
windows **do not overlap (~3 ms/token)** → the expert copy is **serial and exposed**, NOT hidden
behind compute. This reproduces S0's `overlap% = 0` from the timeline.

## Answers to the mission questions
- **(a) COMPUTE GPU ≈ 50 ms (17 %)** — 10× the first-principles guess of 3–5 ms.
- **(b) COPY H2D expert ≈ 76 ms (27 %) — EXPOSED, not hidden** (overlap≈0). Magnitude near the
  55–65 ms guess, but the "hidden" assumption is **refuted**. 1.95 GB/token over PCIe.
- **(c) BLOCKING-SYNC waste ≈ 0** — the sync calls *are* the measurement of (a)+(b); the GPU is
  busy during them, so the barrier itself wastes ~nothing beyond forcing the serial order.
- **(d) OVERHEAD (CPU orchestration gap 121 + launch 26) ≈ 147 ms (51 %)**, and this is an
  **upper bound** (some gap overlaps async GPU work we cannot see on WSL2).

### CONFIRM / REFUTE the "~210 ms (70 %) overhead" hypothesis
**REFUTED as stated.** Pure engine/CPU overhead is **~144–147 ms (≈50 %)**, not 210 ms (70 %).
Real GPU work (compute+copy+transfers) is **~143 ms (≈50 %)**, ~2× the "70–80 ms real work" guess.
The token is a **~50/50 blend**, not a single 210 ms overhead elephant. The expert **H2D copy is
exposed (~76 ms)**, refuting the "hidden/additive-zero" premise.

## Cross-GPU test (does 3090 evidence say engine-overhead or memory-bound?)
Fair same-task/same-config 3060-vs-3090 point **does not exist** in the repo (pods run RAM-hot,
flagged "non comparabili" by their own READMEs). The **config-closest** 3090 (24 GB) points:

| GPU | prompt | K / cache | regime | gen t/s |
|---|---|---|---|---|
| 3060 12 GB (this) | cyberpunk-wide | K23 / c32 | warm, hit0 (refetch) | **3.48** |
| 3090 24 GB `k12_quality_pod` | cyberpunk-wide | K12/K48 / **c32** | RAM-hot, hit0 (refetch) | **2.70 / 2.78** |
| 3090 24 GB `pod_cache1024_html800` | cyberpunk-HTML | K23 / **c1024** | **VRAM-RESIDENT hit~1** | **14–16** (last 24.8) |
| 3080 Ti 12 GB (ledger K96/keep9) | coding | fits VRAM | resident | 12.0 |

Reading it:
- At **matched refetch config (cache32)** the 3090 — 2.6× VRAM bandwidth, **same PCIe** — is **not
  faster** (≈2.7 vs 3.4 t/s). If the token were VRAM-bandwidth/compute-bound the 3090 would win.
  It doesn't → the ceiling is **not** GPU-compute/VRAM-bandwidth. The expert copy is **PCIe-bound**
  (identical on both cards) and ~50 % of the token is **GPU-independent CPU orchestration**.
- The **only** 4–7× win comes from **full VRAM residency** (c1024 hit~1, or the 3080 Ti coding fit):
  making experts resident removes the 76 ms exposed copy *and its serial sync* and lets MoE read
  from fast VRAM. That needs **>12 GB** — the 3060's K23 working set (~6 GB experts + ~5.4 GB
  non-expert) **cannot fit 12 GB** (hit0 at cache32), so it is structurally stuck refetching.
- **First-principles scaling:** give the 3060 a 2.6×-faster GPU (scale compute 50→19 ms; copy 76
  and overhead 147 unchanged) → wall 256 ms → **~3.9 t/s** (+13 %). A real 3090 at resident config
  = **14 t/s**. So the 3090's advantage is **residency/capacity, NOT engine or compute speed** — a
  faster engine buys ~+15 %, fitting the working set in VRAM buys ~4×.

This also reconciles the S0 A/B (partial residency c32→c516 did **not** raise t/s): on the 3060,
the copy saved by *partial* residency is eaten by growing LRU/host bookkeeping (the 42 % CPU gap);
only **full** residency (needs 24 GB) wins — which the 3060 can't reach.

## VERDICT (8 lines)
1. Token (287 ms) ≈ **42 % CPU host orchestration (gap, 121 ms)** + **49 % GPU-blocked (copy 76 +
   compute 50 + small-copy 17 = 140 ms)** + **9 % launch (26 ms)**.
2. **"~210 ms / 70 % engine overhead" → REFUTED.** Pure engine/CPU overhead is **~144 ms (50 %)**,
   an upper bound; real GPU work is the other **~50 %**, ~2× the "70–80 ms" estimate.
3. **Expert H2D copy is EXPOSED (~76 ms, 27 %)**, serial (cross-thread overlap ≈ 3 ms, `overlap%≈0`)
   — the "hidden behind compute" premise is refuted (matches S0).
4. **Compute ≈ 50 ms (17 %)**, not 3–5 ms; blocking-sync itself wastes ≈ 0 (it *is* the GPU work).
5. **Cross-GPU: not engine-overhead-only AND not bandwidth-bound.** A 3090 at matched cache32/refetch
   is **not faster** (~2.7 vs 3.4 t/s) → PCIe-bound copy + GPU-independent CPU overhead dominate.
6. The **decisive lever is VRAM RESIDENCY/CAPACITY**: fitting the working set in VRAM (c1024 on 3090
   → 14–16 t/s; 3080 Ti coding-fit → 12 t/s) removes the exposed copy+sync. **>12 GB required.**
7. A 2.6×-faster **engine/GPU** on the 3060 → only **~3.9 t/s (+13 %)**; **residency** → ~4×. So the
   3060 is capacity-locked near ~3.4 t/s **regardless of GPU speed**.
8. **Neither pure "motore" nor pure "memoria": ~50/50 (CPU-orchestration vs exposed PCIe expert-I/O +
   compute). No single 210 ms overhead elephant. The 3060's real ceiling is 12 GB VRAM capacity —
   it can't make the K23 working set resident, so it pays the serial PCIe copy every token.**

## Artifacts
`profile_breakdown.sh` (driver), `analyze_cpu_timeline.py` (differential decomposition),
`breakdown_result.json` (raw numbers), `probe*.py` (schema/thread probes),
`NSYS_n40/` `NSYS_n120/` (traces+sqlite+diag), `CLEAN_A/` `CLEAN_B/` (clean wall/SPEX), `mem.log`.

---

# ADDENDUM — FIT = SPEED? and the real fit-boundary (local 3060, warm, cyberpunk)

Follow-up to the token breakdown: reproduce CLAIM-008 (*"3060 cache400 keep-8 = 25 t/s;
keep-32 = 3.4"*) locally and find the real VRAM slot boundary. Same warm session, same
config (own lock, `RESERVE_GB=1`, cache32/2-bit path), cyberpunk-wide, ctx2048, greedy, -n90.
Masks available: K12 (~480 ws), K16 (~640), K23 (~920 experts = keep_n × ~40 MoE layers).
**No keep-8/keep-9 mask exists locally**, and the smallest (K12, 480) already exceeds the
boot-probe boundary (394), so a working-set ≤ boundary (the config that gives the 25 t/s jump)
cannot be built from the masks on disk.

## Measured sweep
| run | cache | hit_rate | copied MiB/run | **gen t/s** | sync_ms/batch | peakGPU |
|---|---|---|---|---|---|---|
| K23 c32 (baseline, refetch) | 32 | **0.000** | 186 975 | **3.05** | 2.637 | 11 653 |
| K12 c516 (right-sized) | 516 | **0.878** | 50 969 | **3.06** | 0.686 | 12 040 |
| K12 c900 (over-sized) | 900 | **0.950** | 39 717 | **1.81** | 0.556 | 12 080 |
| K16 c900 | 900 | 0.931 | 42 646 | **1.80** | 0.595 | 12 032 |
| K23 c900 | 900 | 0.858 | 53 979 | **1.77** | 0.740 | 12 056 |

(All rc=0, no OOM / no CUDA error / no clamp message. `selected_experts=27700`/run = ~308/token.)

## Verdict on FIT = SPEED: **REFUTED locally**
- **c32 → c516** raises hit 0.00 → 0.88, cuts H2D copy-work **−73 %** (187 → 51 GB) and
  blocking-sync **−74 %** (2.64 → 0.69 ms/batch) — yet **t/s is FLAT (3.05 → 3.06)**. Eliminating
  three-quarters of the expert copy AND its barrier buys **zero** decode speedup. (Exactly S0's
  A/B, reconfirmed on a fresh warm session; reproduces S0's K12 c516 = 3.09.)
- **c516 → c900** (even higher hit 0.95) **HALVES t/s (3.06 → 1.81)**. All three c900 runs land at
  ~1.8 t/s **regardless of mask/hit** → the per-token cost tracks **cache SIZE (LRU/management),
  not residency**. An over-sized cache is strictly worse.
- So on the 3060 the throughput lever is **not** residency; it is the residency-independent
  CPU-orchestration ceiling measured in the token breakdown (42 % gap). Residency only shuffles
  copy-work into cache-management CPU cost — **net zero at best, net negative when over-sized**.

## K12 case (Q3): does it enter, at what t/s?
K12 (ws ~480) **does not fully fit**: hit caps at 0.88 (c516) / 0.95 (c900), never 1.0, and the
best t/s is **3.06 (c516)** — identical to the refetch baseline, and it **collapses to 1.81 at
c900**. K12 "entering" the cache brings **no speed benefit**. (K23, ws ~920, is even further from
fitting: hit 0.86 at c900, 1.77 t/s.)

## Real fit-boundary (Q2)
Peak VRAM **saturates at ~12.04–12.08 GB** (card cap 12.29 GB) for every c516/c900 run — the
expert cache is **VRAM-bound**, and requesting 900 slots neither OOMs nor adds usable residency
(peak barely moves vs c32's 11.65 GB; the extra slots thrash). The operational slot count is the
**boot-probe ~394 (E1-empirical max 407)** under `RESERVE_GB=1` — that is the number the
controller should use. **Over-sizing beyond it is harmful** (−40 % t/s), so the controller rule is
**cache ≈ working-set, capped at ~394 slots**, never larger.

## Reconciliation with the "25 t/s" claim
The **25 t/s** is a **pod (24 GB) number**, not a 3060 number: the cross-GPU survey found it on the
RTX 3090 `pod_cache1024_html800` (14–16 avg, 24.8 last-chunk) where the working-set fits VRAM
**stably** (hit→1, zero eviction churn) *and* the host CPU is far faster. The token breakdown's
~121 ms/token residency-independent CPU gap alone caps the local 3060 at **≲ 8 t/s even with
perfect residency** — the 12 GB card can never reach stable full residency for K12+ (ws > boundary),
so it thrashes and stays at **~3.0–3.5 t/s**. **CLAIM-008's attribution of 25 t/s to the local 3060
is not reproducible here and should be re-sourced** (almost certainly a pod/24 GB measurement).

### Addendum verdict (fit-sweep)
1. **fit = speed → REFUTED on the 3060.** hit 0.00 → 0.95 gives **no** t/s gain; over-sized cache
   **halves** it. Throughput is CPU-orchestration/capacity-locked at ~3.0–3.5 t/s.
2. **Real boundary ≈ 394 slots** (VRAM-saturated ~12 GB, no OOM at c900); controller rule
   **cache ≈ working-set ≤ ~394**, never larger.
3. **K12 does not usefully fit**: hit ≤ 0.95, best 3.06 t/s (= baseline), 1.81 t/s when over-cached.
4. The **25 t/s is a 24 GB-pod result**, unreproducible on the 12 GB 3060 — consistent with the
   token-breakdown ceiling and the cross-GPU residency story.

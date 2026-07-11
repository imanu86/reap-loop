# RESIDENT-HIT ≈ 0 — audit, root cause, fix (patch 0024)

Date: 2026-07-11. Probe: CPU-only Windows + WSL `/root/ds4` **read-only** source read
+ mining of existing run artifacts. No GPU/pod run in this pass.

Mission: the discovery that keep-K gives no speed locally (K12/K16/K23 all land in the
same 1.1–1.2 t/s band) because ~246 of the ~320 ms/token are the H2D copy of the **same**
258 experts **every** token — resident-hit measured `1/6923` (J31). Prereq of any SOTA K:
no K wins until the resident cache actually holds experts across tokens.

---

## 0. Verdict

**BUG CONFIRMED — YES.** But the exact line is **NOT** where the fork survey pointed.

- The fork-survey hypothesis (`docs/FORK_SURVEY_20260710.md` §2A, huihui-support/ds4) was:
  *"bypass / direct-model evaluated BEFORE the cache lookup in `cuda_model_range_ptr`
  → hit ≈ 0 by design."* **This is not the live bug.** `cuda_model_range_ptr`
  (`ds4_cuda.cu:1280`) already runs the device-resident cache lookup **first** (the
  reorder is already applied — see the comment at `ds4_cuda.cu:1283` and the
  `g_model_range_by_offset.find` + `g_model_ranges` scan that precede the
  `g_model_device_owned || g_model_registered || g_model_hmm_direct ||
  DS4_CUDA_DIRECT_MODEL` bypasses at `1307–1313`). And `cuda_model_range_ptr` is **not**
  the per-token MoE expert path anyway — it serves model-level weights (embeddings,
  attention, router, sinks).

- The per-token MoE expert path is a **separate** LRU slot-pool cache
  (`g_stream_expert_cache`), driven by `cuda_stream_selected_cache_begin_compact_load`
  (`ds4_cuda.cu:4921`). Its loop **also** looks up the cache first
  (`cuda_stream_expert_cache_find`, `ds4_cuda.cu:4967`) and only loads on miss — so
  there is no "bypass-before-cache reorder" bug there either.

The resident-hit ≈ 0 is caused by **two compounding bugs that DISABLE / COLLAPSE the
per-token cache on a 12 GiB card**, so the cache-first lookup has nothing resident to hit
and every selected expert falls to the direct H2D copy (`selected_direct_loads`). This is
the same failure *family* the survey flags (cchuter reserve-floor; giannisanni /
andreaborio / bonciarello "silent fallback under threshold"; audreyt "don't nuke fresh
slots / check the resident table first") — just localized to the streaming-expert cache,
not to `cuda_model_range_ptr`.

---

## 1. Runtime evidence (already on disk, no new run)

`runs/ds4/20260709_local_cache_sweep_k23_code256/code_mini_local_k23_cache258_r01`
(K23 mask, `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.25`, `DS4_SPEX_STATS=1`,
`DS4_EXPERT_TIERING=observe`), `server.stderr.log`:

```
SPEX stats: selected_batches=13846 selected_experts=91710 cache_hits=0 cache_misses=528
            hit_rate=0.0000 direct_loads=91698 copy_calls=276678 copied=622525.50 MiB
tiering observe summary: hits=0 misses=12 direct=91698   (hit_rate=0.0000)
tiering observe layer=1  cap=0    (direct=2286)
tiering observe layer=10 cap=0    (direct=2160)
tiering observe layer=19 cap=0    (direct=2182)
tiering observe layer=0  cap=183  (direct=2280, misses=12)
```

Read-off: **91698 / 91710 = 99.99 % of selected experts went direct**; `cache_hits = 0`;
per-layer cache **capacity = 0 for every layer except layer 0** (which briefly held 183
slots, took 12 misses, then also went 100 % direct). VRAM at steady state:
`CUDA loading model tensors 7.86 GiB cached` on a 12 GiB card ⇒ ~4 GiB free before KV +
staging + cuBLAS workspace. This reproduces resident-hit ≈ 0 **below** the "reserve=1"
fix that `docs/CLAIMS_CURRENT.md` marks CLOSED — i.e. the known reserve bug is necessary
but **not sufficient**; a second bug keeps the cache dead even when it briefly allocates.

---

## 2. Root cause — two bugs, exact lines

All line numbers are the live `/root/ds4/ds4_cuda.cu` @ md5 `7d57f58d` (= canonical v2.1
endpoint for `ds4_cuda.cu`, `patches/README.md` §v2.1).

### Bug A — reserve default 16 GiB + integer-only parse + `total/2` clamp
`cuda_stream_expert_cache_reserve_bytes()` — `ds4_cuda.cu:3269`

```c
uint64_t gb = 16;                               // 3270  default 16 GiB
...
unsigned long long v = strtoull(env, &end, 10); // 3275  INTEGER parse
if (end != env && errno == 0 && end && *end == '\0') gb = (uint64_t)v;  // 3277
```

and the caller `cuda_stream_expert_cache_live_budget()` — `ds4_cuda.cu:3321`:

```c
if (total_bytes != 0 && reserve > total_bytes / 2ull) reserve = total_bytes / 2ull; // 3321
if (free_bytes <= reserve) return 0;             // 3324  -> cache DISABLED
```

Three ways this zeroes the cache on a 12 GiB card:
1. **Default 16 GiB** → clamped to `total/2 = 6 GiB`. With ~4 GiB free after the 7.86 GiB
   resident model tensors, `free (≈4) ≤ reserve (6)` → `live_budget` returns 0 → cache
   never allocates → all direct. (This is the CLAIMS "reserve cache" bug; reserve=1 fixes
   *this* one.)
2. **`strtoull` is integer-only.** The operator set `RESERVE_GB=0.25` intending to lower
   the reserve, but `strtoull("0.25")` stops at the `'.'`, the `*end=='\0'` guard fails,
   and `gb` **stays 16**. The fractional override is **silently swallowed** — the run that
   *looks* like reserve=0.25 actually ran at reserve 16→clamp 6. Only integer values
   (`reserve=1`) ever took effect. This is why "I lowered the reserve" did not restore
   hits.
3. The `total/2` clamp is itself nonsense on small cards (6 GiB reserve on a 12 GiB card).

### Bug B — one failure permanently collapses the whole cache + orphans its pool
`cuda_stream_selected_cache_begin_compact_load()` decode loop — `ds4_cuda.cu:5079` and
`5095`:

```c
} else {                                    // load_slot() failed  (5079)
    cuda_stream_expert_cache_invalidate();  //   -> kills the WHOLE cache
    expert_cache_disabled = 1;              //   -> direct loads for the rest of the batch
    cache_slot = -1;
}
...
if (!copied_from_global_cache) {            // copy_to_compact() failed (5095)
    cuda_stream_expert_cache_invalidate();
    expert_cache_disabled = 1;
}
```

`cuda_stream_expert_cache_invalidate()` (`ds4_cuda.cu:3220`) sets `valid=0 / count=0` but
**does not free** the gate/up/down slot pools (allocated as 3 contiguous `cudaMalloc`s in
`cuda_stream_expert_cache_try_alloc`, `ds4_cuda.cu:3423`). So a **single** transient
`load_slot` / `copy_to_compact` failure:
- disables the cache for the remaining experts of that batch, **and**
- leaves valid=0 so the next `prepare` sees `same_dims=false`, **and**
- orphans the ~GiB slot pool (still allocated, never freed) so free VRAM stays low →
  `live_budget` now returns 0 → the cache can **never re-allocate**.

This is exactly the observed trace: layer 0 allocates 183 slots, loads ~12 experts, hits
a failure → invalidate → orphaned pool → free VRAM starved → layers 1–42 get cap=0 and
100 % of the session runs on direct loads. One failure ⇒ permanent hit ≈ 0.

The two bugs compound: Bug A stops the cache from allocating at all (dominant on this
box); Bug B guarantees that even when it *does* allocate (integer `reserve=1`, or a moment
of higher free VRAM during prefill), the first hiccup kills it for good.

---

## 3. Fix — patch `patches/ds4/0024-cuda-stream-expert-cache-resident-fix.patch`

Anchored to the canonical v2 chain (`ds4_cuda.cu` @ `7d57f58d`, stable from `0014g`
through end-of-chain — no later patch touches `ds4_cuda.cu`). Applies **after 0022**.
`git apply --check` **OK** against the byte-identical anchor (CPU-only: apply-checked +
brace/paren-balanced + symbol cross-checked; **build + pod smoke pending** — no CUDA
toolchain locally). Blob hash `1005bbb7`.

Three surgical hunks:

1. **Bug A — `cuda_stream_expert_cache_reserve_bytes()`**: parse the override with
   `strtod` (float) instead of `strtoull`, and default to a **0.5 GiB** floor instead of
   16 GiB. Now `RESERVE_GB=0.25` works, and with the env unset the default leaves room for
   the cache. On the 3060 (~4 GiB free): `usable = 4 − 0.5 = 3.5 GiB` → ~530 slots →
   `cap = min(configured_budget, 530)` → the cache allocates (258 experts fit). The
   `total/2` clamp is left in place as a harmless guard (a 0.5 GiB reserve never trips it).

2. **Bug B, load_slot failure (was `5079`)**: drop `invalidate()` + `expert_cache_disabled=1`;
   keep only `cache_slot = -1`. A failed slot load degrades **only that expert** to a
   direct load; the cache stays resident for every other expert and every future token.
   (`load_slot` returns 0 without marking the slot valid, so no half-loaded slot is left
   behind — the LRU picks it again next time.)

3. **Bug B, copy_to_compact failure (was `5095`)**: drop `invalidate()` +
   `expert_cache_disabled=1`; `copied_from_global_cache` stays 0 so the expert falls
   through to the existing direct-load path below. Cache stays resident.

**Not done in 0024 (needs GPU to tune, tracked as follow-ups):**
- Free the pool (`release_all`) on a *hard* structural give-up instead of leaking it —
  with hunks 2–3 we almost never give up, so this is lower priority; do it if a hard-fail
  path is ever reintroduced.
- Make the resolved capacity *sticky* (resolve budget once at first alloc, cchuter's
  "apply the margin once") instead of re-deriving from `cudaMemGetInfo` every `prepare`
  call, which can cause release+realloc churn when `configured_budget` (default 512) is
  larger than what fits (~183–407 on the 3060). Low risk but wants a GPU A/B.

### Risk
- `strtod`/`errno` already available (`<stdlib.h>`, `errno` used in the original). Float
  default is safe; on OOM the existing `try_alloc → shrunken_cap` retry still guards.
- Hunks 2–3 change hot-loop control flow: an empty braced `if` body (hunk 3) is valid C
  and does **not** trigger `-Wempty-body` (that warns on `;`, not `{}`). The degraded
  expert still gets a correct direct load — semantics of the *output* are unchanged, only
  the cache-liveness policy differs.
- Untested at runtime (no local GPU). Gate before adoption: the micro-bench below.

---

## 4. Micro-bench — verify hit-rate before/after (run on pod, or local 3060)

Goal: one number, `hit_rate`, before vs after. No config sweep (P4).

```sh
# Same server invocation both arms; the only difference is the binary (0024 in / out).
# K23 static mask, streaming on, stats on, tiering observe on.
export DS4_SPEX_STATS=1
export DS4_EXPERT_TIERING=observe
export DS4_PACE_KEEP=23 DS4_PACE_KEEP_MIN=23
# leave DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB UNSET to test the new 0.5 GiB default,
# then repeat with =0.25 to prove the float parse now bites.

# 1 prompt, ~256 decode tokens, n=3 ABAB (docs/DS4_RUNNER_PROTOCOL.md), trace off.
# Read the single SPEX-stats line from server.stderr.log each arm:
grep 'SPEX stats' server.stderr.log
#   ARM baseline (no 0024): expect hit_rate=0.0000, direct_loads ~= selected_experts.
#   ARM 0024:               expect hit_rate -> the cache-achievable level (>> 0),
#                           direct_loads collapses toward first-touch only.
```

Adoption gate (invariant first, P2): **(i)** `hit_rate` rises from ~0 to the
VRAM-achievable level **and** **(ii)** L0–L3 render level does **not** regress vs baseline
**and** **(iii)** greedy-argmax is byte-identical fix-vs-baseline (a cache hit must never
change the math, only the speed — the giannisanni Δlogit guard). t/s is a reported output,
never the criterion.

Also worth capturing: `cap=` in the `tiering observe` per-layer summary should become
non-zero for **all** layers (today only layer 0), and stay non-zero across the whole
decode (proof Bug B no longer collapses it).

---

## 5. Expected impact (E-LAT model, `runs/ds4/20260710_elat_tier_latency/REPORT.md`)

Steady decode is copy-bound: `t_ss = t_compute + 258 · miss_vram · t_b`, with
`t_compute=74.9 ms`, `t_b=0.952 ms/expert`, `miss_vram=1` today (every expert missed).
Restoring resident hits lowers `miss_vram` to `(1 − hit)`:

| hit-rate | who | t_token | t/s | Δ vs 3.12 |
|---|---|---:|---:|---:|
| 0.00 (today) | measured | 321 ms | **3.12** | — |
| 0.34 | LRU sim cap258 (J17/J31) | 307 ms | 3.26 | +4.5 % |
| 0.50 | cap407 (max native VRAM slots) | 267 ms | 3.74 | +20 % |
| 0.60 | cap407 + preload-promote (J32) | 243 ms | 4.12 | +32 % |
| 0.83–0.85 | cap2048-equiv | 112 ms | 8.95 | +187 % |

**Answer to "if hit rises from 0 to ~0.83, how much?":** the E-LAT ceiling is ~8.95 t/s
(≈ +180 %), **but** E-LAT is explicit that hit 0.83–0.85 is **not reachable with native
capacity in 12 GiB** — 258–407 slots is all that fits, giving hit ≈ 0.34–0.60. So the
realistic prize from 0024 alone is the **fit-in-cache leva-1** of E-LAT §6:
**3.12 → 3.7–4.1 t/s (+20–32 % steady)**, plus removal of the cold-path cliffs that drag
the *average* well below steady (recovering avg toward steady is worth a further
+15–30 % of perceived total time). Pushing past ~0.6 to the 8.95 ceiling needs the S4
leva-2 (CQ1 async cold tier → more experts/GiB) or a compressed-VRAM / SPEX predictor —
0024 is the **enabler** that makes those measurable, since today they all sit on top of a
cache that never holds anything.

---

## 6. Provenance
Source read: `/root/ds4/ds4_cuda.cu` @ `7d57f58d` (read-only WSL). Runtime evidence:
`runs/ds4/20260709_local_cache_sweep_k23_code256/code_mini_local_k23_cache258_r01`.
Model: `runs/ds4/20260710_elat_tier_latency/REPORT.md`. Fork context:
`docs/FORK_SURVEY_20260710.md` §2A, `docs/MOE_ECOSYSTEM_SURVEY_20260711.md` §1B.
Prior status: `docs/CLAIMS_CURRENT.md` "Bug reserve cache esperti" (CLOSED for the
integer-reserve form; this doc shows the fractional-parse + collapse forms were still
open). Patch anchor: `patches/README.md` §"Serie canonica v2".

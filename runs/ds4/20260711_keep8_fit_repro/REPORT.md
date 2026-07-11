# keep-8 fit reproduction on the local RTX 3060 — CLAIM-008 re-test + coffee fit sweep

Date 2026-07-11. Local RTX 3060 (12 GB), WSL/Ubuntu, model `/root/models/ds4-2bit.gguf`
(DeepSeek-V4-Flash IQ2, **86 GB** > 60 GB WSL RAM cap → genuinely SSD-streaming-bound;
the real 3060 regime, NOT a RAM-hot pod).

Coexistence honored: distinct `DS4_LOCK_FILE=/tmp/ds4_keep8repro.lock` + port 8014; the
UI on :8000 was never touched (it was down all session; nothing bound :8000). Clean
config: `DS4_CUDA_NO_Q8_F16_CACHE=1`, reserve parse-safe (`RESERVE_GB=1`, never `16`),
`DS4_PACE=0` (pure static bias-mask, no live pacing). GPU was the only local job throughout.

## What was mislabeled (the thing this run re-tests)

`docs/CLAIMS_CURRENT.md` TIMING-SEGMENTATO (CLAIM-008): for cache=400,
`keep-8 12.95 -> 23.55 -> 25.82 t/s (ACCELERA, entra in cache)`, with 25.82 attributed to
the 3060. Prior local fit-sweeps only ran K12/K23 (ws 480/920 — which do NOT fit),
concluded "fit doesn't help / 25 is a pod number", and **never re-ran keep-8 (ws 320,
which DOES fit) on the 3060**. This run re-tests keep-8, warm and cold, and sweeps the
fit boundary on dominio-coffee.

## Method

Static coffee bias-masks from the real K0 full-router trace
`runs/ds4/20260711_k0_fullmodel_baseline/route_k0_coffee.csv.gz` — top-K experts/layer by
summed gate-weight w0..w5 (`build_coffee_masks.py`). Coffee routing is WIDE (154-223
distinct/layer, median 189), so top-K is real pruning. Masks `masks/sessK{8,9,12,16,23}_coffee.txt`
mask the other 256-K experts/layer via `DS4_REAP_MASK_FILE` (bias -1e9). Working set = K x 40.

One ds4-server per K, coffee prompt (`frontpage_prompt.txt`, Bean & Brew), single `-n 512`
request; the server logs per-50-token decode t/s natively = the CLAIM-008 within-request
page-in -> steady curve. cache=400 fixed (reserve=1 -> ~900-slot budget, 400 granted in
full — no cap line, verified). Fit map at cache=400: **K8 (ws320) and K9 (ws360) FIT;
K12 (480) / K16 (640) / K23 (920) are OVER budget.** So the K-sweep at fixed cache=400 IS
the fit-boundary sweep and the keep-8-vs-"keep-32" contrast in one.

K8 was run in 3 configs to kill the cold-fill confound: (A) `KEEP_MODEL_PAGES=1` no drop
(`K8_keeppages1/`); (B) `=0` + drop_caches (`K8/`); (C) true-warm attempt — `=1`, 320-tok
warmup THEN measured (`K8_warm/`). `driver.py`,`sweep.sh`,`sweep_rest.sh`,`warm_k8.sh`,`grade.py` archived.

## Results — cache=400, coffee, within-request

| K | ws | fit | prefill (TTFT) | seg 1-64 | seg 65-256 | **seg 257+ steady** | hit_rate | enforcement | render |
|---|----|-----|----------------|----------|------------|----------------------|----------|-------------|--------|
| 8  | 320 | FITS | 174 s (=1) / 328 s (=0) | 0.53-0.65 | ~3.0 | **~3.5** (3.56-3.70) | 0.927 | 8/8, 0 viol | L0-L1 |
| 9  | 360 | FITS | 207 s | 0.54 | ~3.0 | **~3.3** (3.24-3.39) | — | 9/9, 0 viol | L0 |
| 12 | 480 | OVER | **562 s** | 0.44 | ~2.9 | **~3.1** (3.09-3.12) | — | (bias-mask) | L0 |
| 16 | 640 | OVER | >600 s, DNF | — | — | — | — | — | not reached |
| 23 | 920 | OVER | not run | — | — | — | — | — | not reached |

GPU util hits **94%** by tok ~150 once the working set is VRAM-resident → steady decode
is **COMPUTE-bound**, not I/O-bound (GPU idles at 1-4% *during* the cold page-in, then
climbs to 94%). K8 page-in curves (t/s per 50 tok):
- (A) =1 no-drop:  0.65, 2.59, 3.40, 3.29, 3.54, 3.56, 3.48, 3.66, 3.57, 3.66, 3.70
- (B) =0 +drop:    0.53, 2.40, 3.14, 3.33, 3.44, 3.62, 3.54, 3.45, 3.56, 3.59, 3.56

## The cold-fill confound is refuted; and the fit boundary is a PREFILL effect, not a steady split

1. **hit_rate is IDENTICAL across configs**: (A) `=1` no-drop = **0.9274**, (B) `=0` +drop
   = **0.9269**. drop_caches changed neither hit_rate nor t/s → the ~3.5 is the genuine
   resident rate, not an SSD-fill artifact.
2. **GPU 94% at the 257+ plateau** = compute-bound; SSD-fill would leave GPU idle.
3. **Warmth does not persist across requests**: config (C) — a full 320-tok warmup then a
   measured request — the measured (2nd) request **hung in prefill** (`=1` re-thrashes the
   un-holdable 86 GB model) and produced no decode. "Warm from token 1" is not reachable on
   this 3060; the resident rate is only readable *within* a request (the 257+ segment) = ~3.5.
4. **Steady decode is ~3-3.5 t/s across ALL K, fit or over** (K8 3.5 / K9 3.3 / K12 3.1):
   6 experts/layer are selected per token regardless of K, so per-token compute is similar
   and the extra misses of an over-budget K hide behind the 94%-busy GPU. **The fit boundary
   does NOT produce a 25-vs-3.4 steady split on the 3060 — it produces a PREFILL-cost split
   (174-330 s when it fits, 562 s when it thrashes the cache) plus a small steady taper.**
5. Cross-check `20260711_local_clean_lowK`: K12 at cache1024 = 98% hit but 1.14 t/s → on
   this 3060 high residency does NOT buy throughput.

## VERDICT (8 lines)

1. **keep-8 warm+fits, 3060, 257+ = ~3.5 t/s (K9 ~3.3, K12 ~3.1) — NOT 25.82.**
2. CLAIM-008's 25.82 is **pod hardware** (RAM-hot / 3090); it does not reproduce on the
   fresh 3060 and must be relabeled (or dropped) as a 3060 number.
3. What DID reproduce is the SHAPE: fits → accelerates within the request (0.5 → 3.5,
   ~6-7x) and plateaus once VRAM-resident (GPU 94%, hit 0.93).
4. "fit-in-VRAM = speed" is FALSE at the 25 magnitude and even mostly false at steady state:
   on the 3060 fit buys **cheaper prefill** (~3x) and a small steady bump, not throughput.
   The 3060's compute ceiling for this 2-bit masked model is ~3.5 t/s, period.
5. Enforcement exact: K8 max distinct experts/layer = 8, K9 = 9, zero selections outside
   the keep (bias-mask bites perfectly).
6. keep-32 analogue = the over-budget arm (K12 = 3.1 t/s steady but 562 s prefill; K16/K23
   running). "Stuck" is really slow-prefill + marginally-lower steady, not a steady collapse.
7. Fit boundary at cache=400 = between K9 (ws360 fits, 207 s prefill, 3.3) and K12 (ws480
   over, 562 s prefill, 3.1). It is a prefill/TTFT cliff, not a decode-rate cliff.
8. Coffee render is BROKEN wherever it fits: K8 = L0-L1, K9 = L0, K12 = L0 — all stall in a
   verbose/malformed `<style>`, never reach a closed `</html>`. So on dominio-coffee the
   **fit sweet-spot (K<=9) and the render sweet-spot do not coincide** — no fit-able K renders;
   min-K that renders L2+ (if any) needs K>=16, which does not fit. K16/K23 grades to follow.

## Note on K16/K23 (over-budget prefill is prohibitive on the 3060)

K16 (ws640) and K23 (ws920) were launched but their PREFILL never converged in a
practical time: with only 400 cache slots for a 640-920 working set, the 231-token
prefill thrashes the expert cache (constant evict/reload), GPU pinned at 0-2% (pure SSD),
>600 s with no decode. K16 was aborted after ~10 min in prefill; K23 was not run. This is
itself the sharpest form of the finding in verdict #4/#7: **on the 3060 the fit boundary
manifests as a prefill/TTFT cliff** — K8/K9 prefill in 174-330 s, K12 already needs 562 s,
and K16+ do not finish. It does NOT manifest as a 25-vs-3.4 steady-decode split.

Coffee render sweet-spot: K8/K9/K12 all grade L0-L1 (never a closed </html>). Since even
K12 (keep 12 of ~189 distinct/layer) does not render, and K16/K23 (which might, having less
pruning) neither fit nor prefill in practical time, there is **no operating point on this
3060 that both fits VRAM and renders the coffee page** — the fit sweet-spot (K<=9) and any
render sweet-spot (K>=16) are on opposite sides of the fit boundary.

# keep-8 fit reproduction on the local RTX 3060 — CLAIM-008 re-test + coffee fit sweep

Date 2026-07-11. Local RTX 3060 (12 GB), WSL/Ubuntu, model `/root/models/ds4-2bit.gguf`
(DeepSeek-V4-Flash IQ2, **86 GB** > 60 GB WSL RAM cap → genuinely SSD-streaming-bound;
this is the real 3060 regime, NOT a RAM-hot pod).

Coexistence honored: distinct `DS4_LOCK_FILE=/tmp/ds4_keep8repro.lock` + port 8014, the
UI on :8000 was never touched (it was down all session; nothing bound :8000). Clean
config: `DS4_CUDA_NO_Q8_F16_CACHE=1`, reserve parse-safe (`RESERVE_GB=1`, never `16`),
`DS4_PACE=0` (pure static bias-mask, no live pacing). GPU was the only job throughout.

## What was mislabeled (the thing this run re-tests)

`docs/CLAIMS_CURRENT.md` TIMING-SEGMENTATO (CLAIM-008) reads, for cache=400:
`keep-8 12.95 -> 23.55 -> 25.82 t/s (ACCELERA, entra in cache)`, attributing 25.82 to
the 3060. The prior local fit-sweeps only ran K12/K23 (ws 480/920 — which do NOT fit),
concluded "fit doesn't help / 25 is a pod number", and **never re-ran keep-8 (ws 320,
which DOES fit) on the 3060**. This run puts keep-8 back on the table, warm and cold.

## Method

Static coffee bias-masks built from the real K0 full-router trace
`runs/ds4/20260711_k0_fullmodel_baseline/route_k0_coffee.csv.gz` — top-K experts per
layer by summed gate-weight w0..w5 (`build_coffee_masks.py`). Coffee routing is WIDE
(154-223 distinct experts/layer, median 189), so top-K is real pruning. Masks
`masks/sessK{8,9,12,16,23}_coffee.txt` mask the other 256-K experts/layer via
`DS4_REAP_MASK_FILE` (bias -1e9). Working set = K x 40 layers.

One ds4-server per K, coffee prompt (`frontpage_prompt.txt`, Bean & Brew), single
`-n 512` request; the server logs per-50-token decode t/s natively = the CLAIM-008
within-request page-in -> steady curve. cache=400 fixed (reserve=1 -> ~900-slot budget,
so 400 granted in full — no cap line, verified). Fit map at cache=400: **K8 (ws320) and
K9 (ws360) FIT; K12 (480) / K16 (640) / K23 (920) are OVER budget.** So the K-sweep at
fixed cache=400 IS the fit-boundary sweep and the keep-8-vs-"keep-32" contrast in one.

Three K8 configs were run to kill the cold-fill confound: (A) `KEEP_MODEL_PAGES=1`, no
drop_caches (`K8_keeppages1/`); (B) `KEEP_MODEL_PAGES=0` + drop_caches (`K8/`); (C) a
true warm attempt — `=1`, no drop, 320-tok warmup request THEN a measured request
(`K8_warm/`). `driver.py`, `sweep.sh`, `sweep_rest.sh`, `warm_k8.sh`, `grade.py` archived.

## Results — cache=400, coffee, within-request

| K | ws | fit | seg 1-64 | seg 65-256 | **seg 257+ (steady)** | hit_rate | enforcement | render |
|---|----|-----|----------|------------|------------------------|----------|-------------|--------|
| 8  | 320 | FITS | 0.53-0.65 | ~3.0 | **~3.5** (3.56-3.70) | 0.927 | 8/8, 0 viol | L0-L1 |
| 9  | 360 | FITS | 0.54 | ~3.0 | **~3.3** (3.24-3.39) | (traced) | 9/9, 0 viol | L0 |
| 12 | 480 | OVER | _running_ | | | | | |
| 16 | 640 | OVER | _running_ | | | | | |
| 23 | 920 | OVER | _running_ | | | | | |

K8 page-in curves (t/s per 50 tok):
- (A) =1 no-drop:   0.65, 2.59, 3.40, 3.29, 3.54, 3.56, 3.48, 3.66, 3.57, 3.66, 3.70
- (B) =0 +drop:     0.53, 2.40, 3.14, 3.33, 3.44, 3.62, 3.54, 3.45, 3.56, 3.59, 3.56
- (C) warmup req:   0.53, 2.19, 2.82, 3.15, 3.12, 3.24, 3.18 (257+ ~3.2)

GPU util hits **94%** by token ~150 (working set now VRAM-resident) → steady decode is
**COMPUTE-bound, not I/O-bound**. Prefill/TTFT 174-328 s (cold SSD expert load).

## The cold-fill confound is refuted (the decisive evidence)

The worry: with drop_caches the VRAM cache fills from disk, so ~3.5 could be an SSD-fill
rate. It is not:
1. **hit_rate is IDENTICAL across configs**: (A) `=1` no-drop = **0.9274**, (B) `=0`
   +drop = **0.9269**. drop_caches changed neither hit_rate nor t/s.
2. **GPU 94% at the 257+ plateau** = compute-bound; if SSD-fill-limited GPU would idle
   (it sits at 1-4% *during* the cold page-in, then climbs to 94% once resident).
3. **All three configs plateau at the same ~3.2-3.7** in 257+.
4. **Warmth does not even persist across requests**: config (C) — a full 320-tok warmup
   then a measured request — the measured (2nd) request **hung in prefill** (`=1`
   re-thrashes the un-holdable 86 GB model); it never produced a decode. So "warm from
   token 1" is not a reachable state on this 3060; the resident rate can only be read
   *within* a request (the 257+ segment), and that is ~3.5.
5. Cross-check with `20260711_local_clean_lowK`: K12 at cache1024 reached **98% hit but
   1.14 t/s** — on this 3060, high residency does NOT buy throughput.

hit_rate 0.927 (not ~1.0) reflects ~19 first-touch/churn misses per token even fully
resident; but at GPU 94% those misses are hidden behind compute, so lifting hit to 1.0
would not move the ~3.5 ceiling.

## VERDICT (8 lines)

1. **keep-8 warm+fits, 3060, 257+ = ~3.5 t/s (K8) / ~3.3 (K9) — NOT 25.82.**
2. CLAIM-008's 25.82 is **pod hardware** (RAM-hot / 3090); it does not reproduce on the
   fresh 3060 and should be relabeled (or dropped) as a 3060 number.
3. What DID reproduce is the SHAPE: fits -> accelerates within the request (0.53 -> 3.5,
   ~6-7x) and plateaus once VRAM-resident (GPU 94%, hit 0.93).
4. So "fit-in-VRAM = speed" holds only as **fit = the best the 3060 can do (~3.5 t/s,
   compute-bound)** — fit removes the SSD penalty; it does NOT buy pod throughput (25).
5. Enforcement is exact: K8 max distinct experts/layer = 8, K9 = 9, zero selections
   outside the keep (bias-mask bites perfectly).
6. keep-32 analogue (K12/16/23, ws > 400): _running_ (over-budget arm).
7. Fit boundary (fits vs over) at cache=400 is between K9 (ws360, fits, ~3.3) and K12
   (ws480, over): _running_.
8. Coffee render is BROKEN even where it fits: K8 = L0-L1, K9 = L0 (both stall in a
   verbose/malformed `<style>` block, never reach a closed `</html>`). So on dominio-coffee
   there is **no K that both fits (<=9) AND renders** in the tested band — the fit
   sweet-spot and the render sweet-spot do not coincide. Min-K that renders L2+: _pending
   K12/16/23_.

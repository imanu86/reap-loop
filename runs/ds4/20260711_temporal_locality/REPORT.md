# Expert temporal locality & predictability — is the per-token working set (240) small + predictable enough for async prefetch?

Date: 2026-07-11. OFFLINE (no-GPU). Script: `temporal_locality.py`. Raw stdout: `raw_output.txt`.

**Question under test.** We never need all ~1000 keep-experts at once. A MoE decode
step touches exactly **6 experts x 40 MoE layers = 240 distinct (layer,expert)
slots** per token. 240 x 6.75 MiB ~= 1.6 GB -> fits the ~2 GB free on the 3060. The
"989 don't fit" wall is the **UNION over time**, not the instantaneous need. So the
right lever is: keep the ACTIVE set resident + async-prefetch the next token behind
compute. This measures whether that is viable: how **local** and how **predictable**
the active set is.

**Data (real post-mask routing, `pos,layer,n,e0..e5,w0..w5`).** Primary = long masked
cyberpunk trace `route_maskedCyber_K48_cyberpunk.csv.gz` (2854 tok x 40 layer). Plus the
`20260711_masked_route_traces/` sweep K12/K23/K48 cyber (wide) + K12/K23 coffee (narrow).
A globally-unique expert = `(layer, expert_id)`; per-token footprint = exactly 240 slots
(verified: n=6 on every row, 240 distinct pairs/token, pos contiguous).

---

## VERDICT (8 lines)

1. **Instantaneous need FITS.** Exactly **240 distinct experts/token = 1620 MiB**, inside ~2 GB (1x). Double-buffer 2x = 3240 MiB does **NOT** fit. Union (the wall) = 915 exp @K23 (~ the "989") / 1920 @K48 -> 6-13 GB.
2. **The shift is only MODERATELY local, and it is K-dependent.** Token-to-token, of the 6 picks/layer only **~2.6 persist at K48** (-> **133 new experts/token**), ~3.2 at K23, **~4.2 at K12** (-> 66 new/tok). Peak of the overlap histogram is 2-3, not 5-6.
3. **But churn is WITHIN a small recent pool.** Over a 64-token window the intrinsic miss is <2% (K48) / <0.5% (K23) -- almost every expert used now was used in the last ~64 tokens. History helps a lot; it is the per-token *selection* inside a stable pool that rotates.
4. **Cheap predictors beat static, modestly, at a fixed small budget.** HIT@240 (b=6/layer, fits 1.6 GB): K48 **0.44** (lru/hot) vs static **0.34**; K23 **0.61** vs 0.57; K12 **0.74** vs 0.72. LRU ~= prev-token; hot-set (32-tok window) marginally best on wide traces.
5. **Beats the historic static-pin 0.13? YES, decisively -- but mostly via the MASK.** Even *static* on the masked keep-set is 0.34-0.72. The 0.13 was full-model (256-wide) static; REAP masking is the big lever, dynamic prefetch a secondary **+0.10-0.12** on top at equal budget.
6. **>0.8 hit needs either a tighter mask or a bigger cache.** LRU sweep: K48 reaches 0.8 only at **b~=24 (960 exp, 6.3 GB)**; K23 at **b~=12-18 (3.2-4.7 GB)**; K12 at **b~=9 (2.4 GB) -> 0.92**. Within the ~2 GB that fits (b~=6), wide-K48 caps at ~0.44.
7. **Async-prefetch is net-positive everywhere** (critical-path fetches drop from ~all-240 to (1-hit)x240 = 134 @K48 / 70 @K12), and the residual miss transfer (13-75 ms/tok) is hide-able behind compute.
8. **Plausible speedup vs 0.8 t/s:** wide-K48-in-2GB modest **~1.5x (-> ~1.2 t/s)**; K23 with ~3 GB cache **~2-3x (-> 1.5-2.5 t/s)**; K12 **~3-4x (-> 2.4-3.5 t/s, approaching compute-bound)**. The dream regime (hit>0.8, few hidable misses) is real at **K<=23 and/or cache >= b12 (>=3.2 GB)** -- not at wide K48 in 2 GB alone.

---

## 1. Instantaneous footprint — CONFIRMED, fits 1x, not 2x

| metric | value |
|---|---|
| distinct (layer,expert) / token | **240.0** (min 240, max 240; long trace has rare 126-edge tokens) |
| footprint 1x | **1620 MiB** — fits ~2 GB (yes) |
| footprint 2x (double buffer) | 3240 MiB — does **NOT** fit ~2 GB (no) |
| union over trace (the wall) | K12 **480** (3.2 GB) / K23 **915** (6.2 GB) / K48 **1920** (13.0 GB) |

The task's mental model is exact: the instantaneous need (240) fits; the temporal union
(~989 at K23) is the wall. So residency must track the *active* set, not the union.

## 2. Temporal locality (per-layer, consecutive tokens)

| trace | mean persist /6 | new experts/token | overlap hist P(k of 6) 0..6 |
|---|---|---|---|
| K48 cyber (long) | **2.67** | **133** | .07 .15 .23 .26 .20 .08 .01 |
| K23 cyber | 3.19 | 112 | .03 .11 .19 .23 .23 .17 .04 |
| K12 cyber | **4.24** | **71** | .00 .00 .04 .17 .37 .33 .09 |
| K23 coffee (narrow) | 3.40 | 104 | .01 .06 .16 .27 .30 .17 .03 |
| K12 coffee (narrow) | 4.36 | 66 | .00 .00 .03 .14 .35 .39 .09 |

**Window intrinsic miss** (fraction of a token's picks NOT seen in the previous W tokens,
unbounded set) — the pool is nearly stable over ~64 tok:

| trace | W1 | W4 | W16 | W64 |
|---|---|---|---|---|
| K48 cyber | 0.556 | 0.300 | 0.106 | **0.015** |
| K23 cyber | 0.468 | 0.130 | 0.029 | **0.003** |
| K12 cyber | 0.294 | 0.059 | 0.008 | **0.000** |

Read together: token-to-token the *selection* rotates a lot (W1 miss 0.30-0.56), but the
*pool* it rotates within is small and slow-moving (W64 miss ~0). That is exactly the regime
where a modest recency cache + prefetch wins — provided the cache is a bit larger than one
token's 6/layer.

## 3. Predictability at fixed budget (hit = fraction of the 240 picks already resident)

Per-layer budget b; total = b x 40. b=6 -> 240 (1.6 GB, fits); b=12 -> 480 (3.2 GB, needs freed VRAM).

| trace | static@240 | prev@240 | lru@240 | hot@240 | static@480 | lru@480 | hot@480 |
|---|---|---|---|---|---|---|---|
| K48 cyber (long) | 0.336 | 0.444 | 0.444 | 0.433 | 0.515 | **0.632** | 0.625 |
| K48 cyber | 0.411 | 0.421 | 0.421 | 0.441 | 0.611 | 0.612 | **0.639** |
| K23 cyber | 0.571 | 0.532 | 0.532 | **0.613** | 0.840 | 0.847 | **0.857** |
| K12 cyber | 0.722 | 0.706 | 0.706 | **0.737** | 1.000 | 0.997 | 0.995 |
| K23 coffee | 0.543 | 0.567 | 0.567 | 0.566 | 0.805 | **0.820** | 0.817 |
| K12 coffee | 0.695 | 0.727 | 0.727 | 0.724 | 1.000 | 0.997 | 0.995 |

- **prev-token ~= LRU** at b=6 (the last token's 6 picks are the whole budget). The cheapest
  possible predictor already captures the bulk.
- **hot-set (32-tok freq window)** is marginally best on wide/churny traces (K23/K48); LRU
  edges it on narrow. Differences are small — any cheap recency policy works.
- **static control reproduces "static pin is weak"** (pin_analysis lineage): at equal budget
  it trails dynamic by ~0.10-0.12. The historic **0.13** corresponds to full-model (256-wide)
  static; on the masked keep-set even static is 0.34-0.72 — so the mask is the dominant lift
  and dynamic residency is the secondary one.

## 4. The number that counts — LRU hit vs cache size

| trace | b6 (1.6GB) | b9 (2.4GB) | b12 (3.2GB) | b18 (4.7GB) | b24 (6.3GB) | b32 (8.4GB) |
|---|---|---|---|---|---|---|
| K48 cyber (long) | 0.444 | 0.542 | 0.632 | 0.761 | **0.844** | 0.922 |
| K23 cyber | 0.532 | 0.727 | **0.847** | 0.959 | 0.991 | — |
| K12 cyber | 0.706 | **0.915** | 0.997 | — | — | — |

To clear **hit ~= 0.8-0.9**: K12 needs **b~=9 (2.4 GB)**, K23 needs **b~=12-18 (3.2-4.7 GB)**,
K48 needs **b~=24-32 (6.3-8.4 GB)**. Within the ~2 GB that actually fits (b~=6), only K12
gets close (0.71). This is the crux: **the achievable hit is set jointly by K (mask width)
and the VRAM you can give the cache** — not by predictor cleverness.

## 5. Speedup estimate (rough — assumptions stated)

**Assumptions.** (A) Current 0.8 t/s (=1250 ms/tok) is dominated by serialized on-demand
expert fetches — consistent with a ~0.13 full-model static-pin serving hit -> ~209 of 240
experts fetched on the critical path/token, each paying transfer + kernel-launch/sync
overhead. (B) A resident LRU/hot cache + async prefetch moves the *hit* fraction off the
critical path (prefetched during the prior token's compute); only **misses = (1-hit)x240**
block. (C) PCIe effective host->device ~= **12 GB/s** (3060 gen3x16 ~6, gen4 ~13-25). Expert
= 6.75 MiB.

Residual miss transfer per token (hide-able behind compute):

| hit | misses/tok | MiB/tok | transfer @12GB/s |
|---|---|---|---|
| 0.13 (old static) | 209 | 1409 | 115 ms |
| 0.50 | 120 | 810 | 66 ms |
| 0.80 | 48 | 324 | 26 ms |
| 0.90 | 24 | 162 | 13 ms |

Critical-path fetches drop from ~209 -> misses, so a first-order speedup ~= 209/misses,
capped by compute once transfer is fully overlapped:

| point | fits VRAM? | hit | misses/tok | rough speedup | plausible t/s |
|---|---|---|---|---|---|
| K48 wide, b6 | yes 1.6 GB | 0.44 | 134 | ~1.5x | **~1.2** |
| K48 wide, b12 | no 3.2 GB | 0.63 | 89 | ~2.3x | ~1.8 |
| K23, b6 | yes 1.6 GB | 0.53 | 113 | ~1.8x | ~1.5 |
| K23, b12 | no 3.2 GB | 0.85 | 36 | compute-bound | ~2.5-3.5 |
| K12, b6 | yes 1.6 GB | 0.71 | 70 | ~3x | ~2.4 |
| K12, b9 | ~2.4 GB | 0.92 | 19 | compute-bound | ~3-4 |

**Caveat.** If compute (not fetch) already dominates the current 0.8 t/s — i.e. fetches are
already partly overlapped — these gains shrink. The estimate is order-of-magnitude; the
robust qualitative claim is: async prefetch removes serialized per-expert stalls, which is
strictly net-positive, and its payoff grows sharply as K tightens and/or the cache exceeds
one token's 240.

## Bottom line for the build

- **Do build the resident-cache + async-prefetch path** (SPEX-predicted next-token, LRU or
  32-tok hot-set eviction). It beats the static pin at every operating point and never hurts.
- **The big knobs are K and cache-VRAM, not the predictor.** Cheap recency ~= optimal cheap.
  To reach the hide-everything regime (hit >= 0.8): run **K12/K23**, or free VRAM to seat a
  **b>=12 (>=3.2 GB)** cache. Wide-K48 in 2 GB tops out near hit 0.44 -> a real but modest
  ~1.5x — not the >3 t/s dream.
- **Don't budget for 2x double-buffer of the full 240** (3.2 GB won't fit 2 GB); prefetch
  the *miss delta* (~134 experts @K48, ~70 @K12), a fraction of a full buffer.

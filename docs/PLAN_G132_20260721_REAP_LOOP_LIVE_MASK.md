# PLAN G132 — The Live Mask: reap-loop with an active CPU (2026-07-21 evening)

Direction set by the project owner. Supersedes G131's STATIC K=5,600 resident set (which was
correctly objected to: its 2.58% fallback is IN-SAMPLE, ranked on the same 3 prompts it was
evaluated on). Evidence base: `runs/ds4/20260721_q1_recovery_pilot/` + F1 measured (d2aca30).

## The design (owner's vision — this is the actual "reap loop")

The expert mask is NOT static. It adapts token-by-token while generating:

- **Tier ladder with continuous promotion**: SSD → RAM → pinned RAM → VRAM, driven by live
  routing heat; cooling experts get demoted (reaped). The 37 GB RAM tier is the CURRENT
  working set for THIS context, not a frozen global top-K.
- **CPU as an active worker at 50-90%, two lanes**:
  - **Lane A (compute, ~5-6 cores)**: decode warming/lukewarm experts DIRECTLY from RAM
    (exact IQ2, 0.7-0.9 ms/expert measured). Every expert computed here = one H2D saved +
    one VRAM slot freed. KTransformers model. This is a PRIMARY compute lane, not a fallback.
  - **Lane B (anti-miss logistics, ~1-2 cores)**: work AHEAD of misses — stage SSD→RAM for
    experts warming from cold, copy RAM→pinned for promotion candidates, pre-enqueue H2D for
    experts about to earn VRAM. Low I/O priority ALWAYS (C: saturation froze the PC twice).
  - ~1 core reserved for CUDA submit (overlap spike: submit stall <0.15 ms).
- **SPEX as lane B's predictive brain**: prediction feeds staging (ProMoE recipe). Historical
  caveat: SPEX-as-measured was NEGATIVE (G16 K1 prefetch -54%) because it competed ON the
  decode critical path. In this architecture prediction is decoupled: a wrong prediction wastes
  lane-B bandwidth only, never blocks decode. Signal hierarchy: observed heat (base) + SPEX
  (turbo). Whether the turbo is needed is decided by the coherence numbers (below).
- **Physical constraint to measure TOGETHER, not separately**: lanes A+B share DRAM bandwidth
  (~6 GB/s compute reads + 2-3 GB/s staging vs ~40 dual-channel theoretical).

## Open investigations (da fare — in flight or queued)

1. **[IN FLIGHT] Working-set temporal coherence** (D:\ds4_work\working_set, codex on the 3
   replays, 258 routes/token structure): per-token novelty vs sliding windows, warm-set size,
   heat half-life, promoter simulation at P={2,8,32} promotions/token vs static top-K, and
   domain-switch reconvergence. THIS IS THE GATE: if the working set evolves slowly, the loop
   chases and converges; if every token draws fresh experts, it thrashes. Also SIZES both lanes.
2. **Two-lane DRAM contention bench**: lane A computing N experts while lane B stages from SSD —
   measure combined throughput degradation vs isolated. (CPU-only, codex-buildable.)
3. **Live-mask prototype** (after 1+2): heat tracker + promoter/reaper + lane A dispatch branch
   (warm expert → CPU queue instead of H2D) + lane B worker. Reuses: F1 (committed, +19%),
   SSD-WRAP async semantics (g130/ssdwrap-semantics, gates proven fireable), tiered-hysteresis
   0033 concepts, SPEX ring infra. F2's VRAM LRU is available but measured inert for cross-token
   Q1 reuse (keep for the VRAM tier of the live mask, where hits SHOULD occur under heat locality).
4. **Quality stays exact by construction** (all tiers serve exact IQ2) — the T3 quality gate
   re-runs on the prototype regardless.

## Measured facts this plan stands on (all in runs/, committed)

- F1 real: 4.86 t/s full/open (+19%, record without a mask), sync eliminated. F2: inert on
  current engine (cross-token Q1 reuse ~0); gate-3 spike's 9.0 t/s projection was optimistic.
- CPU exact expert forward: 0.7-0.9 ms/expert (8 threads), correctness 0.9998; tail hides under
  GPU (overlap eff 140-658%, stall <0.15 ms).
- Coverage concentration is real but in-sample: top-2000 = 94.9% mass on 3 prompts pooled.
- Q1 quality is unfixable at 1.125 bpw (STE ceiling 0.701) — the live mask serves EXACT IQ2 only.

Standing rules unchanged (G130 §protocol, G131 gates): codex implements, independent codex
reviews, Claude verifies/builds/commits; diagnostics never quotable; negatives recorded;
one change per experiment; everything committed+pushed immediately.

---

## GATE 1 VERDICT (2026-07-21 evening): THE LOOP CHASES — adaptive beats static, empirically

From the 3 real replays (working_set_coherence.md/json in runs/ds4/20260721_q1_recovery_pilot/):

- Per-token mass novelty at W=8: **24.8% median** (75.3% of next-token gate mass is already in the
  last-8-token union) — the working set evolves slowly enough to chase.
- Heat half-life is long: re-routed within 1 token 52.5%, within 4 69.1%, within 16 **81.7%**.
- **Adaptive promoter at P=8 promotions/token: 42.8% steady warm-tier mass hit vs 29.5% for a
  same-size static top-K ranked on the first 16 tokens — +45% relative.** The owner's thesis
  ("the mask must adapt as we write tokens") is CONFIRMED on data.
- Domain switch (HTML→Python) recovery to 90% in-domain hit: P=8 ≈ 27 tokens, P=32 ≈ 15 tokens.
- Lane-B sizing implication: P=8 ≈ 56 MB/token staging; P=32 for switch headroom.

Caveats: 64-token traces (longer runs will firm the tails); absolute hit% depends on tier sizing —
the load-bearing result is the adaptive-vs-static gap and the heat locality.

NEXT: investigation 2 (two-lane DRAM contention bench), then the live-mask prototype (inv. 3).

## GATE 2 VERDICT: two lanes coexist — PASS (dram_contention_report.md)

Lane A (6 cores, real IQ2 experts, cold-streamed) degrades only 10-12% while lane B sustains
56-224 MB/token staging on either drive. BONUS: lane B from D: (SATA) sustains 504 MB/s with
LOWER lane-A degradation (5-7%) — staging can avoid C: entirely, structurally removing the
PC-freeze cause. Both pre-build gates are green; NEXT: the live-mask prototype (investigation 3).

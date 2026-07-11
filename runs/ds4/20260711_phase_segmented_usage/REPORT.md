# PHASE-SEGMENTED EXPERT USAGE — does the capacity wall dissolve into shifting per-phase hot-cores?

Date: 2026-07-11. Offline (no-GPU). Falsification target: pin_analysis.py
(commit aabaa97, runs/ds4/20260711_pin_viability_and_gaps/), which measured
expert usage over the WHOLE trace, found it ~quasi-uniform inside the keep
(Gini ~0.4-0.5, needs 66-71% of the keep to cover 90% of a layer's hits), and
deduced a capacity wall for K23 (working-set 989 / 90%-pool 581 >> 394 slots).

Hypothesis under test. That flatness is an aggregation artefact across phases:
per-phase usage is concentrated, but the concentrated hot-core shifts phase-to-
phase, so the whole-trace average looks flat while every instant fits in VRAM.
If true -> the wall is not physical -> phase-adaptive dynamic residency is viable
even on wide tasks.

Method: segment each trace into non-overlapping windows (30/50/75 tok = phase
proxy) and, per segment, per layer, filtered to the keep set (same full-model->
keep proxy as pin_analysis), measure (1) concentration [k90, Gini], (2) hot-core
shift [Jaccard top-6 of consecutive segments], (3) instantaneous working-set
[sum of per-layer k90] vs the 394 budget and vs the whole-trace union. Entropy-
gate: on the one trace with per-token confidence (cyberpunk instrumented collapse)
correlate hot-core shift with entropy. Script: phase_segmented_usage.py, raw
output results.txt.

---

## VERDICT: the wall SOFTENS to a speed-bump — it does not dissolve

The weak form of the hypothesis is confirmed, the strong form refuted.

- CONFIRMED — the flatness was partly an aggregation artefact. Per-phase usage
  is markedly more concentrated than the whole-trace average: k90/layer drops
  14 -> ~9 (K23) and 8 -> ~5.5 (K12); per-segment Gini rises 0.50 -> ~0.69.
  Averaging concentrated-but-moving distributions genuinely flattens the
  aggregate. pin_analysis measured the union, not the instant.

- REFUTED — the hot-core does NOT rotate cleanly. Consecutive-phase Jaccard
  (top-6) is 0.61-0.70 on every healthy trace (coffee, python, json). ~65% of
  the top-6 is a stable backbone; only ~35% rotates. This is moderate churn, not
  the near-disjoint (Jaccard->0) rotation the strong hypothesis needs to make
  each instant land clear of the budget.

- RESULT — K23 instantaneous working-set lands right ON the budget. Sum of
  per-layer k90 per phase: median ~356-380 (fits 394) but p90 464-595 / max 652
  (busy phases overflow). Versus the whole-trace union 568 (90%-pool) / 920
  (distinct) that defined the wall. So the instant is compressed to ~65% of the
  union and clears 394 at the median only. The wall goes from "unconditionally
  over budget -> 0.79 hit" (pin_analysis) to "median phase fits, busy phases
  don't" — a speed-bump, not a clean pass.

- K12 has no wall (confirming pin_analysis): instant WS median ~220, max 370,
  union 320 — everything < 394 at every window.

### Distinguishing regimes (window = 50 tok)

| trace | keep_n | per-seg k90/layer (whole) | Gini | consec-seg Jaccard | instant WS med/p90/max | union (wall) | fits 394? |
|---|---|---|---|---|---|---|---|
| coffee / K23 (wide, healthy) | 23 | 9.0 (14) | 0.69 | 0.66 (min .30) | 373 / 464 / 652 | 568 / 920 | median only |
| coffee / K12 (wide, healthy) | 12 | 6.0 (8) | 0.62 | 0.78 | 228 / 282 / 370 | 320 / 480 | every seg |
| cyber / K12 (wide, COLLAPSED) | 12 | 5.0 (5) | 0.62 | 0.91 (frozen) | 201 / 217 / 223 | 218 / 265 | every seg |
| python / K23 (narrow) + | 23 | 8.0 (11) | 0.70 | 0.55 | 338 / 352 / 355 | 422 / 710 | every seg |
| json / K23 (narrow) + | 23 | 8.0 (9) | 0.74 | 0.61 | 314 / 325 / 353 | 372 / 655 | every seg |

+ narrow keep-coverage under the coffee mask is only ~0.18 (pin_analysis) -> the
keep-filtered residue is sparse and unrepresentative; treat narrow rows as
indicative only. K23 window sweep 30/50/75: instant-WS median 356/373/380,
Jaccard 0.61/0.66/0.70 — conclusion stable across granularity.

### Two non-obvious findings

1. The collapse is a FROZEN core, not a rotating one. The cyberpunk run is real
   K12-masked decode (not a proxy) and it collapses into garbage. Its consecutive-
   phase Jaccard is 0.89-0.91 — the hot-core is nearly frozen. So the pathology
   of a too-tight mask on an off-domain wide task is the opposite of healthy phase
   rotation: the router cannot rotate to serve the CSS/body phases (the coffee-
   tuned K12 keep doesn't contain their experts; cross-domain Jaccard ~0.14 per
   pin_analysis) -> it locks onto a tiny stale set -> garbage. Indirect support
   for the architecture of the thesis: a phase needs its hot-core present in the
   mask; when absent you get collapse, not graceful degradation.

2. Narrow and wide-healthy show similar shift (~0.6). The clean "wide-rotates /
   narrow-is-static" contrast did NOT appear — both sit at Jaccard ~0.55-0.70.
   The only sharp separation is collapse (0.9, frozen). So the rotation signal is
   a property of healthy generation generally, not uniquely of wide tasks; wide
   tasks differ by having a larger union (920 vs 655-710 distinct), which is what
   makes K23's union overflow the budget.

### Entropy-gate (cyberpunk, the only per-token-entropy trace)

Pearson(hot-core-shift, entropy) = +0.41 (window 20). The entropy spikes (pos
185-224: data-gposcate / #finto garbage onset, ent 1.8-2.2; and pos 125-144:
stata-ini garbled comment, ent 0.89) sit at the elevated-shift boundaries. So
entropy does rise where routing rotates — consistent with entropy gating the
rotation. Caveat: measured on a collapse trajectory (K12-wide) with a tiny
12-expert pool that compresses the shift signal (0.16-0.39 band); the clean
healthy phase test has no per-token entropy (coffee/W130 emit no conf). So:
suggestive (+0.41), not conclusive.

---

## What this means for REAP-LOOP

- Phase-adaptive residency is a real but bounded lever for K23. Tracking the
  current phase's ~370-expert hot-core (vs static-pinning the union) would lift
  the median-phase hit from pin_analysis's 0.79 toward ~0.85-0.90, at the cost of
  swapping ~35% of residents per phase transition. It does NOT get every phase
  under 394 (busy phases need 460-650). Net: worth building for K23, but it turns
  a hard wall into a speed-bump, not a green light.
- This reconciles the "static-pin ~= LRU" finding of pin_analysis. That
  equivalence was measured whole-trace (churn averaged away). Per-phase there IS
  ~35% rotation; a phase-aware promoter (e.g. the SPEX topK, G6) — not a blind LRU
  — is what would capture it. The gain is the 0.79->~0.88 band above.
- K12 stays the safe VRAM-fit choice (instant + union both < 394); K23 remains
  residency-starved on the 3060 even phase-adaptively, only softened.

## Caveats (binding)

- Proxy. coffee/python/json are FULL-model routes filtered to keep (no masked
  route.csv exists — pin_analysis gap G2). Real K23-masked decode redistributes
  the 6-of-6 picks onto the keep set -> likely flatter per-phase -> instant WS
  could be higher than 370 -> wall firmer, not softer. The 370 is an optimistic
  lower bound. (Cyberpunk is the exception: real masked K12, and it collapsed.)
- Window = phase proxy. No token-text labels for coffee; HTML head/CSS/body
  labels exist only for cyberpunk (pos: head 89-147, <style> 148, body
  180->collapse). Conclusions stable across window 30/50/75.
- Single healthy wide trace (a_coffee, 299 tok; W130 129 tok). n is small.

Reproduce: python runs/ds4/20260711_phase_segmented_usage/phase_segmented_usage.py

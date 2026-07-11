# RI-PIVOTAL: K12 + rewind-v3 (0022 v3) — cyberpunk W50/ctx8192/fase2-4000

**Date:** 2026-07-11 · **Pod:** ds4-podD-0022-smoke (RunPod 7qgalm9sasqnr7, RTX 3090, RUNNING).
Binary = **0022 v3** (ds4.c md5 d4ff85af, reset a8a38a8 + git apply 0022-v3.patch, fresh ELF).
Recipe: cyberpunk 199 B prompt, W50 two-phase weighted freeze-safe -> mask K/layer STATIC, ctx_p1
2048 / ctx_p2 8192, cache 1024, greedy temp 0, fase2 4000 (--total 4050). v3 levers on both arms:
DS4_PACE_REWIND_GARBAGE=0.80 DS4_PACE_REWIND_CKPT_DEPTH=8 DS4_PACE_REWIND_WARMUP=40, over the
pivotal arm2 PACE base (S1=1, KEEP=KEEP_MIN=KEEP_MAX=K pinned, ROTATE/RELEARN off, breath silenced,
WEIGHTED_SELECTED=1, REWIND=1, MAX=2 default). Harness = canonical run_w_sweep_freeze_safe.py via
run_pivotal_arm.py (phase-2-only env). Mechanism (fire/deep/divergence) already proven in the
20260711_smoke_0022v3 a/b/c smoke.

## Benchmarks to beat
- Pivotal v2 (0022 v2): **0/7 rescues, 0/23 masked runs ever closed </html> at 4000**; arm2 gtps
  ~1.18 (x1.29 vs K12 static ~0.9), useful_frac 0.04, lock onset ~char 491.
- Decision model prediction: **K12+rewind = 3.99 good-tok/s** (vs 1.56 best-static, x2.56), L2-ish, closes.

---

## ARM 1 — K12 + rewind-v3, WARMUP re-freezes to K12 (model-predicted 3.99 config), n=3

| seed | L0-L3 | </html> | rewind fire/arm | char_garbage | useful_frac | lock onset (char) | p2 t/s | good-tok/s |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 2/4 | yes | 0.17 | 774 / 4548 | 21.48 | 3.66 |
| 1 | 0 | 0 | 2/4 | yes | 0.17 | 774 / 4548 | 21.95 | 3.74 |
| 2 | 0 | 0 | 2/4 | yes | 0.17 | 774 / 4548 | 21.42 | 3.65 |

All 3 seeds **byte-identical** (4615 chars each) — greedy determinism confirmed (the new rewind
bookkeeping introduces no nondeterminism).

### Verdict ARM 1 (SECCO): CYCLES, does not convert. 0/3 </html>.
- The v3 mechanism **fires correctly** every run: char_garbage arms+fires DEEP and pre-lock
  (209->118, as in the smoke), then a backup s1_cusum fire (195->140); n hits **REWIND_MAX=2** and
  the run then **rides out the collapse**. It cycles to MAX, it does NOT converge to a stable page.
- No conversion to L2 / no </html>: K12 is too narrow — after 2 deep divergent rewinds it
  re-collapses (new "1:"-repetition lock). Consistent with pivotal-v2 "narrow mask / poisoned
  context wins", and with armB's "all statics <=38 -> L0, no </html>".
- **BUT v3 materially beats v2 on the good-tok/s rate the decision model cares about:**
  good-tok/s **~3.66-3.74 ≈ the model's predicted 3.99** (v2 delivered only ~1.18). v3 pushes the
  lock onset LATER (char 774 vs v2's ~491) and useful_frac UP (0.17 vs 0.04) — roughly 3x more
  useful content before collapse. So the model's good-tok/s *rate* is essentially vindicated by v3;
  the quality gate (closing an L2 page) is still NOT met at K12.

Contribution to the historic count: **0/3 closes** (running masked-close tally stays 0).

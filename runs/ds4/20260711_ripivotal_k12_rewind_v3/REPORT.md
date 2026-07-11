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

---

## ARM 2 — K-ESCALATION (script-side, sanctioned), restart phase-2 from the healthy frozen W50 anchor at K=12->48

**Escalation approach (documented):** the 0022-v3 engine has NO per-rewind escalation env
(verified in the patch source: on a rewind it re-freezes to `cur_keep`, never increments). So
the escalation is done SCRIPT-SIDE per the coordinator's sanction: for K in {12,20,28,36,44,48}
re-run the SAME two-phase config, restarting phase-2 from the identical healthy frozen W50 anchor
(greedy => the anchor is byte-identical each rung) at that K, with rewind-v3 active; early-stop the
ladder at the first `</html>` close. Each rung = "a collapse bumps K, retry from a healthy point."

| K | L0-L3 | </html> | rewind arm/fire | lock onset (char) | useful_frac | p2 t/s | good-tok/s | note |
|---|---|---|---|---|---|---|---|---|
| 12 | 0 | 0 | 4/2 | 774 / 4548 | 0.17 | 21.2 | 3.61 | garbage collapse, 2 deep rewinds, recollapses |
| 20 | 0 | 0 | 1/0 | 1219 / 10543 | 0.12 | 21.0 | 2.42 | detector armed, never fired (<0.80); still locks |
| 28 | 0 | 0 | 1/0 | none / 15360 | 1.00* | 12.0 | 11.96* | escapes repetition-lock, long varied text, but L0, no close |
| 36 | 0 | 0 | 3/1 | 869 / 4611 | 0.19 | 17.4 | 3.29 | 1 fire, recollapses |
| 44 | 0 | 0 | 6/2 | none / 16736 | 1.00* | 5.1 | 5.12* | escapes lock, long varied text, but L0, no close (slow, cache thrash) |
| 48 | 0 | 0 | 1/0 | 280 / 4201 | 0.07 | 11.8 | 0.78 | early lock onset 280 |

(*useful_frac=1.00 at K28/K44 is a loop-onset artifact: no exact periodic lock is found, so the
metric counts all chars "useful" -- but the output is still graded L0 and never closes `</html>`.
It is unstructured varied text, not a valid page.)

### Verdict ARM 2 (SECCO): escalation does NOT find a holding K. 0/6 rungs close </html>; final K reached = 48 (ceiling, did not hold).
- No rung in K=12..48 produces L>0 or closes the document. Two distinct failure modes appear:
  the low-K rungs (12/36/48) collapse into a repetition lock (rewind fires, recollapses); the
  mid rungs (20/28/44) escape the exact lock but wander into long unstructured L0 text that never
  closes. Widening K (with rewind-v3 restarting from the healthy anchor) does not cross into a
  closing/L2 page anywhere up to K48.
- n=3 confirm running at the ceiling K=48 (see arm2_confirm_K48/); escalation is greedy-
  deterministic (arm1's 3 seeds are byte-identical), so the ladder rungs are single-shot exact.

---

## CAMPAIGN VERDICT vs benchmarks

- **vs pivotal-v2 (0/7 rescues, 0/23 masked closes):** v3 rewind+escalation still produces
  **0 closes** across arm1 (0/3) + the escalation ladder (0/6). The historic tally does NOT move:
  no K12-rewind or K<=48-escalation run closes `</html>` at 4000. The v2 "narrow mask / poisoned
  context wins" conclusion SURVIVES v3's three mechanistic fixes.
- **BUT the v3 mechanism is not inert** (this is the real yield, proven in the a/b/c smoke and
  visible here): char_garbage fires DEEP and PRE-lock, WARMUP breaks the byte-identical no-op, and
  at K12 v3 reaches good-tok/s **~3.6-3.7 ~= the decision-model's predicted 3.99** (v2 only 1.18),
  pushing lock onset from ~491 (v2) to 774 and useful_frac from 0.04 to 0.17. v3 delays/reduces
  the collapse ~3x and meets the predicted *rate*, but does not convert it into a closing page.
- **vs decision model (K12+rewind=3.99, L2-ish, closes):** the good-tok/s *rate* prediction is
  essentially met (3.6-3.7); the *quality* prediction (L2, closes) is REFUTED at K12 and up to K48.
  Recommendation stands with the v2 REPORT: for wide/fast collapse, rewind is a rate-improver, not
  a page-completer; K48-static remains the model's width fallback, and even K48+rewind here does
  not close -- the closing threshold (if any) is above 48 in this cyberpunk/ctx8192/4000 regime.

### Bottom line: 0/(3+6) new closes. v3 mechanism works (fires deep, diverges, ~3x more useful tokens, hits the good-tps rate) but does NOT close a page at K12..48. Historic 0/23 -> still 0 closes.

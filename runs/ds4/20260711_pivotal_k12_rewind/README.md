# 2026-07-11 — PIVOTAL: K12 + rewind vs K12 static (cyberpunk ctx8192/4000)

**Question (decision model `docs/DECISION_MODEL.md` @ fa35dd6):** does the 0022 S1-guided
rewind convert the certain collapse of K12-wide into an L2-ish completion?
Prediction: **K12+rewind = 3.99 good-tok/s vs 1.56 best-static (+156%)**; assumed
`CORR_REWIND_TOK` 56 (pod-D v2 smoke measured the floor: 32 = EVERY, cycle ~0.31 s).
Anchor: armB just eliminated ALL statics ≤38 (`0962c6b`: K23 AND K38 → L0 0/3, no
`</html>`, byte-identical loops); **0/15 masked runs have ever closed `</html>` at 4000.**

## Setup

- Pod `7qgalm9sasqnr7` (RTX 3090 community, RAM-hot — **t/s pod-only; RATIOS vs the
  in-batch static control transfer, absolute t/s do not**). Binary = post-**0022v2**
  chain (ds4.c `a88f9dcb`, `ds4` md5 `9048e707…`), the ALL-PASS smoke build.
- Harness: canonical `scripts/run_w_sweep_freeze_safe.py` (extended prose-aware
  fence-strip — the cyberpunk prompt ALWAYS opens with Italian prose, 44c9361) driven
  UNMODIFIED through `harness/run_pivotal_arm.py` (monkey-patches `phase2_cmd` to inject
  phase-2-only env from `PIVOTAL_P2_ENV`; phase 1 stays env-clean; `{rundir}` expands
  per-run). W50 two-phase weighted freeze-safe → mask **K12/layer STATIC** (sess.txt =
  9760 pruned pairs = (256−12)×40, verified), cyberpunk 199 B, fase2 4000 (`--total
  4050`), ctx_p2 8192, cache 256, greedy temp 0.

## Arms

| arm | n | phase-2 env on top of `DS4_REAP_MASK_FILE` (K12 trace-weighted) |
|---|---|---|
| 1 `arm1_static` | 2 | none — pure static control (exact armA/armB protocol at K12) |
| 2 `arm2_rewind_default` | 3 | PACE-as-rewind-carrier + `DS4_PACE_REWIND=1`, detector E-DET defaults |
| 3 `arm3_rewind_aggr` | 3 | same + aggressive catch: `ARM_K .25 ARM_H 1 FIRE_K .5 FIRE_H 2 EVERY 16 MAX 6 BACKOFF 128` |

PACE-as-rewind-carrier (arms 2/3, exact JSON in `harness/arm_common.sh` + per-arm
`p2env.json`): `DS4_PACE=1 S1=1 KEEP=KEEP_MIN=KEEP_MAX=12` (K pinned ⇒ the rewind's
widen target keep_max **= 12**: faithful "K12+rewind", no hidden widen), `ROTATE=0
RELEARN=0 BREATH_EVERY=999999 DRIFT=2.0` (reactive breath silenced — the decision model
says K12+breath LOSES, 1.29), `WEIGHTED_SELECTED=1` (weighted mass for prefill mask +
healthy-relearn), `DS4_PACE_LOG={rundir}/pace_events.jsonl`,
`DS4_SPEX_TRACE_TOKENS={rundir}/tokens.csv` (0028 sidecar → retraction-aware
reconstruction: the CLI stream keeps the poisoned tail, the sidecar's rewinding `pos`
lets the analyzer rebuild the client-trimmed deliverable).

**Mask provenance in arms 2/3:** the harness file mask (identical to arm 1) is applied
at the first decode poll and rules the run; PACE's prefill mask holds only ~1 token;
after a rewind FIRE the mask legitimately becomes PACE's K12 relearned from
healthy-segment weighted mass (the 0022 "refresh from healthy" semantics).

## DESIGN FINDING (pre-run, source-verified on the 0022v2 tree)

**The 0022 n-gram airbag fire source is dead code in every configuration.** The
actuator's `airbag = sensor_armed && ema_ngram >= drift` can never be true when
`ds4_pace_rewind_after_eval` runs: `ds4_pace_tick` executes INSIDE the eval (earlier,
same token) and its HOLD branch consumes the same condition first — it fires the
reactive breath (phase→BREATH, which blocks the rewind's `phase==PACE_HOLD` gate) and
de-arms the sensor; in a persistent loop `ema_ngram` never falls to `release`, so the
sensor never re-arms and the airbag stays false forever. If instead `DRIFT` is raised to
silence the breath, the airbag threshold rises with it. ⇒ the "FIRE anche se S1 piatto"
insurance the mandate wanted from the airbag is NOT reachable in this build; arm 3
implements the aggressive **CUSUM** catch instead, and the airbag gap goes to the patch
owner (fix: give the actuator its own drift threshold decoupled from PACE's, or hoist
the airbag check before the tick's breath branch).

## Files

`arm1_static/`, `arm2_rewind_default/`, `arm3_rewind_aggr/` — harness outdirs (per-run
`W050/rNN/`: route.csv, tw.txt, frozen.txt, sess.txt, p2prompt.txt, trest.txt,
deliverable.html, p1/p2.diag, pace_events.jsonl + tokens.csv in arms 2/3; summary.csv,
summary_median.csv, manifest.json, p2env.json per arm). `harness/` — wrapper + arm
scripts (provenance). `REPORT.md` — prediction-vs-measurement (end of batch).

Status: arm 1 IN CORSO (avvio ~00:47 UTC), arms 2/3 in coda. REPORT a fine batch.

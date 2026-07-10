# E-MODEL — Decision model v0 (design over trial)

**Date:** 2026-07-11 · **Mode:** OFFLINE (traces + graded-outcome ledger on disk; no GPU/pod/WSL).
**Script:** `scripts/decision_model.py` (reproducible, `py_compile` clean).
**Artifacts:** `decision_model.json` (this dir) + operativo in `docs/DECISION_MODEL.md`.

Reproduce: `python scripts/decision_model.py --reap-loop-root <reap-loop> --moe-root <moe>/.claude/worktrees/elastic-bose-6ae1c7`

Mandate honoured: **turn the archive of "config attempts" into a *calculated* model** —
a width->K* estimator, a per-token collapse hazard, and an objective (good-tok/s) that
composes them into a decision table, and — crucially — names the **minimal experiments
that identify the parameters the fit declares non-identifiable**, not more config sweeps.

## 0. TL;DR

- **Winning width metric = identity union-growth (`union_slope` / `sat_ratio`)**, cross-trace
  CV **0.49** vs routed-mass `cov@23` CV **~0.033** (E-CAL, ~flat). Identity beats mass ~15x.
  **But** it is (a) **length-confounded** (steep on short traces) and (b) at the **50-tok engage
  window it collapses back toward flat** (`union_fix` CV 0.047) — the width signal only emerges
  over a **>=100-150 tok** window, and the **narrow-task anchor is entirely missing from disk**
  (coffee/JSON have no routing trace). => **K*(width) is under-identified today.**
- **K\*** (good-tok/s optimum, L2+ constraint), by width class:

  | class | knee | K* with rewind airbag | K* no airbag (fallback) |
  |---|--:|---|---|
  | **narrow** (coffee/JSON) | 20 | **K~12-16** (4.0-4.3 gtps) | K~12 (3.7) |
  | **medium** (Python) | 32 | **K~16** (3.6 gtps) | K~32 static (2.5) |
  | **wide** (cyberpunk/frontpage) | 48 | **K~12 + rewind** (4.0 gtps) | K~48 static (1.6) |

- **Recovery-ladder verdict (the user's question):** on WIDE, **K12+rewind wins big (good-tok/s
  3.99, ~2.5x the best no-airbag static 1.56); K12+breath does NOT (1.29 < K48-static 1.56)**.
  Rewind is the cheap catch (~56 tok); breath's 70-tok relearn windows over-cost on a
  fast-collapsing small mask. **Add rewind, not breath.**
- **The whole small-K strategy is CONTINGENT on rewind actually catching wide collapse in ~56
  tok — which is UNMEASURED.** That is identification experiment #2, the pivotal one.

## 1. Component 1 — Width sensor -> K* estimator (IDENTITY, not mass)

**Why identity, not mass.** E-CAL proved routed-mass coverage is task-invariant at engage:
`cov@23 = 79%` (range 73-82%) for html *and* all 11 coding prompts, `Kmin-cov90 ~ 38`
everywhere -> a coverage rule picks ~one K for every task. So we compute **identity** metrics on
the 13 full-model weighted traces (html_W50/W130 + 11 coding, `route_*.csv` / `trace_coding.tgz`):

| metric | what | cross-trace CV | html mean | coding mean |
|---|---|--:|--:|--:|
| `union_slope` | new distinct experts / 100 tok (cumulative-union fit) | **0.490** | 107.9 | 52.0 |
| `sat_ratio` | union(all)/union(first half) — still growing? | 0.072 | 1.45 | 1.30 |
| `newexp_late` | frac of engage-union first seen in tok 25-49 | 0.063 | 0.35 | 0.33 |
| `union_cov90` | distinct experts for 90% of selection-freq in a 24-tok window | 0.059 | 36.5 | 35.2 |
| `union_fix` | distinct experts/layer in the **fixed first 49 tok** (length-controlled) | 0.047 | 74.3 | 72.1 |
| `churn_topk` | 1-Jaccard of top-23 identity between windows | 0.036 | 0.41 | 0.43 |
| *(ref) mass `cov@23`* | routed top-23 mass share (E-CAL) | *~0.033* | *79%* | *79%* |

**F1 — identity union-growth is the most task-informative signal we have** (CV 0.49), and it
ranks html/cyberpunk **wider** than coding (union grows ~2x faster), consistent with cyberpunk
collapsing at K23 while coding tolerates K91. **F2 — but it is length-confounded** (html_W50=143
vs html_W130=73 is mostly the 49-vs-129 token count) and the **length-controlled** variants
(`union_fix`, `sat_ratio`) separate only weakly (CV 0.05-0.07). **At the 50-tok warmup window the
identity signal is nearly as flat as mass** — the discriminating growth only accumulates later.
**F3 — the narrow end is unobservable:** every trace on disk (cyberpunk + coding) is a *wide*
task; coffee/JSON (the narrow survivors at K20-23) have **no routing trace**, so the low-K anchor
of K*(width) cannot be fitted. **-> identification experiment #1.**

**Runtime recipe (what to log at freeze).** From the warmup `rmass` accumulator (patch 0020,
selected-6 experts/token) the engine can compute `union_slope` and `sat_ratio` **for free** —
BUT the sensor must **keep accumulating past PACE_WARMUP to ~150 tok** (F2), i.e. a short
"observe-wide" tail before the freeze, logging per-layer cumulative-distinct-expert curves.
That is the one new thing to log; everything else is already in the trace.

## 2. Component 2 — Collapse hazard

Every graded rollout with a known loop **onset** is one survival observation (onset = event;
survive-to-budget = right-censored). Exponential MLE per (width, K-band, actuation) stratum,
Poisson 95% CI. **n inventoried = 23 rollouts** across M1a (6), M1b (3), armA static (1 indep,
byte-identical x3), pod3 frozen/admit coffee+cyber (5), s5 coffee (3, one harness-excluded),
T4 coffee W30/50/130 (3), knee ladder JSON/Python/frontpage (5 label-only), K91 scope (1).

| stratum | n | events | exposure (tok) | lambda/tok | 1/lambda (MTTC) | 95% CI on MTTC |
|---|--:|--:|--:|--:|--:|---|
| wide_K23_rotate (M1a+M1b) | 9 | 8 | 8 408 | 9.5e-4 | **1 051** | 533-2 441 |
| wide_K23_static (armA+pod3) | 3 | 2 | 2 188 | 9.1e-4 | **1 094** | 303-9 741 |
| wide_K91_static (coding) | 1 | 1 | 2 476 | 4.0e-4 | **2 476** | 445-189k |
| narrow_K23_static (coffee) | 9 | 0 | 11 600 | <8.6e-5 | **>3 162** | 3 162-inf |
| medium_K_static (Python) | 2 | 0 | 4 000 | — | >1 091 | — |

**Cross-check:** wide-K23 rotate (1 051) and static (1 094) MTTCs **agree** — at K23 on a wide
task, *actuation barely matters, width dominates* (both static armA and rotate M1a collapse).
Contrast E-CAL's html static>>rotate: that was a *narrower* html; on cyberpunk even static fails
at K23. So **width is the outer variable, actuation the inner one.**

**Fitted hazard law (honest, few-anchor):**

    lam(K, width) = DRIFT(width) + LAM_COV * max(0, exp((knee(width) - K)/SCALE) - 1)
    DRIFT = {narrow 1.0e-4, medium 2.0e-4, wide 4.0e-4}  knee = {narrow 20, medium 32, wide 48}
    LAM_COV = 2.6e-4   SCALE = 22

- **`DRIFT(width)` is width-dependent**, and this is the deep link to Component 1: a wide task
  keeps **shifting** which experts it needs (E-PHASE: frozen mask starves later phases -17.6 pt),
  so even a well-provisioned mask (K91) drifts to a lock (~2476 tok = DRIFT_wide). A narrow task
  is ~one phase -> DRIFT~0 (coffee 0/11 600). **Identity union-growth (the width sensor) and the
  residual DRIFT are the same non-stationarity.**
- **COV term is one-sided** (active only for K < knee): above the knee the stationary demand is
  covered; below it, hazard rises exponentially in the coverage gap.
- **Fit reproduces every anchor:** narrow-K23 MTTC 10 000 (ok coffee), wide-K23 1 053 (ok
  measured), wide-K91 2 500 (ok 2 476), Python K32-at-knee holds / K28-below breaks (ok knee).
- **Confounders (declared):** prompt (only 2 trace-tasks), HW (pod RAM-hot vs 3060 — but hazard
  is token-indexed, HW-invariant by P2), freeze-class (2-phase vs in-engine warmup), and the
  onset detector definition (ngram3_window120 flags 118-848 tok, while scope reports coherence
  loss ~gen 126 for the same runs — the operational onset may lag the true one; both noted).

## 3. Component 3 — Objective + decision table

**SOTA metric = good-tok/s (wall) = throughput(K) x E[useful fraction | K, width, corrections]**,
under L2+ median. Throughput(K) from **E-LAT** (local 3060: `t_ss = 74.9 + 258*miss(K)*0.952 ms`;
`miss(K)` a working-set-vs-cache surrogate: K12->4.35, K23->3.12 t/s). Useful fraction from the
hazard + correction costs: **rewind ~ 56 tok** (n-gram FIRE median 40 + margin 16, S1_REWIND),
**breath ~ 70 tok window x 15% relearn** (J28 breath 290->370; D6b 13-17%). Full table in
`decision_model.json`; best rows:

| class | with airbag | good-tok/s | no airbag (fallback) | good-tok/s |
|---|---|--:|---|--:|
| narrow | **K12 + rewind** | 4.30 | K12 static | 3.72 |
| medium | **K16 + rewind** | 3.56 | K32 static | 2.46 |
| wide | **K12 + rewind** | 3.99 | K48 static | 1.56 |

**Recovery-ladder (WIDE, budget 4000) — the user's K12+rewind+breath question:**

| config | lam(K) | MTTC | tps | useful | **good-tok/s** |
|---|--:|--:|--:|--:|--:|
| **K12 + rewind** | 1.5e-3 | 678 | 4.35 | 0.92 | **3.99** |
| K48 static (no airbag) | 4.0e-4 | 2 500 | 3.12 | 0.50 | 1.56 |
| K91 static | 4.0e-4 | 2 500 | 3.12 | 0.50 | 1.56 |
| K12 + **breath** | 1.5e-3 | 678 | 4.35 | 0.30 | 1.29 |
| K23 static | 9.5e-4 | 1 053 | 3.12 | 0.26 | 0.80 |

**Model verdict:** **K12+rewind conviene (3.99, +156% vs best static). K12+breath NON conviene
(1.29 < K48-static 1.56)** — breath thrashes because its relearn windows cost more than they
save once the mask is small enough to collapse fast. **Add rewind as the airbag; do not add
breath on top.** This holds *only if* rewind catches wide collapse in ~56 tok (unproven).

## 4. K* recipe (invariant-unit thresholds)

The controller reads only **invariants** (P2): identity-width, coverage%, hazard/token.
- **Size K from width class**, sized to the class knee (narrow 20 / medium 32 / wide 48) as the
  **no-airbag floor**; drop to K~12-16 **only with a proven rewind airbag** (speed-dominated).
- **Trigger** the rewind on the **textual n-gram detector** (S1-slope gives *no* lead on
  aggressive small masks — scope: flat 0.815, collapse ~gen126); fire threshold derived from the
  n-gram repeat rate, not a fixed token count.
- **No breath** on small-K wide runs (net-negative per the objective).

## 5. Identification experiments (only non-identifiable params — design, not enumeration)

The fit declares exactly three legs it cannot resolve from disk. Each names the parameter it
identifies, the exact config, n, and cost.

1. **Narrow-task routing traces -> knee(width) low anchor + K*(width) calibration.**
   *Identifies:* `knee(narrow/medium)` and which identity metric (union_slope vs sat_ratio)
   linearises to K*. *Why non-identifiable:* every trace on disk is wide; the narrow end is
   empty (F3). *Config:* full-model **trace-on** run on **coffee + JSON + a Python** prompt,
   >=150 completion tok, log `route.csv` (selected-6 + weights). n=1 each (routing is
   deterministic full-model). *Cost:* ~free if any such trace exists in an archive; else 1 pod
   run ~**$0.30**.
2. **Rewind efficacy on wide (K12) -> the pivotal decider.**
   *Identifies:* real `CORR_REWIND_TOK` and `useful_frac(K12, rewind)` — the sign of the entire
   small-K strategy. *Why non-identifiable:* patch 0022 (S1-guided rewind) is designed but not
   built; no measured wide-task rewind exists. *Config:* build 0022, cyberpunk K12 static +
   n-gram-triggered rewind, **n>=3, >=2000 tok**, L0-L3 grade + measure detection latency and
   clean-rewind bit-equality (design section 5). *Cost:* pod A/B ~**$1-2**.
3. **tps(K<23) on the real 3060 -> the speed term.**
   *Identifies:* `miss(K)` below K23 — E-LAT's surrogate says K12->4.35 t/s, but the measured
   **resident-hit=0 bug** (J31) may flatten it (a smaller working-set that never becomes
   resident buys nothing). *Config:* local 3060 static K12 / K16 / K23, cache-isolated,
   >=800 tok, avg+last-chunk t/s. n=3. *Cost:* **local, free.**

**Not on the list** (already identified): the width *ordering* (F1), the wide-K23 hazard
(measured, both actuations), the coffee narrow-survival (0/11 600), the correction costs
(S1_REWIND / J28 / D6b). The model does **not** ask to re-sweep configs — it asks to place three
missing anchors.

## 6. Confounders & honesty ledger

- Only **2 trace-tasks** (cyberpunk, coding), both wide -> width sensor validated on *ordering*,
  not on absolute K* (exp #1 closes this).
- Hazard anchors are **3-4 points** -> wide CIs (wide-K91 MTTC CI 445-189k). The law is a *typed
  hypothesis with CI*, not a measured curve.
- `miss(K)` velocity surrogate is **optimistic below K23** (ignores the resident-hit bug) ->
  the good-tok/s of small-K options is an **upper bound** until exp #3.
- Onset detector (ngram) may **lag** true coherence loss (~gen126 scope vs 118-848 ngram) ->
  hazards are, if anything, **under**-estimated for wide-K23.
- Pod t/s are RAM-hot, non-comparable to 3060; only the **hazard (token-indexed) and the
  ordering** transfer. All absolute t/s come from the E-LAT 3060 calibration, flagged HW-dep.

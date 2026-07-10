# E-CAL — coverage-threshold calibration for predictive mask sizing

**Date:** 2026-07-10 · **Mode:** OFFLINE, trace-only (no GPU / WSL / pod writes)
**Script:** `scripts/calibrate_coverage_threshold.py` (reproducible, `py_compile` clean)
**Artifacts:** `stats.json`, `coverage_by_trace.csv`

## The idea under calibration (user)

From the warmup alone, the per-expert per-layer **unbiased router mass** gives — for
free — the curve **K → coverage** for every candidate mask
(`coverage(K) = mass(top-K)/total`, `S1_predicted at engage = 1 − coverage(K)`).
Pick the smallest **K** whose coverage ≥ a threshold θ. **Question:** which θ
separates survived runs from collapsed ones, and does it depend on task width?

**Monito honoured:** S1 is *not* monotone with survival (K91-static coding S1≈0.845
survives ~2476 tok; K23-rotate html S1≈0.815 collapses at ~gen126), so I looked for
**conditioned** thresholds and evaluated **alternative curve metrics** (slope,
kneedle knee, marginal coverage of the K-th expert), not just the level.

---

## 1 — Data inventory (what I have / what's missing)

| Source | Content | Have | Missing |
|---|---|---|---|
| **13 full-model traces w/ weights** (E1 pool) | html cyberpunk `route_W50.csv` (49 tok) + `route_W130.csv` (129 tok); 11 coding prompts `trace_coding.tgz` (52–255 tok). Schema `pos,layer,n,e0..e5,w0..w5`; w = unbiased router probs of the **6 selected** experts | curve computable | trace logs **only the 6 selected experts**/token → mass on the other 250 is absent |
| **Live S1 sensor r1** | `20260710_scope_divergence_pod/r1/s1_r1.csv.gz` (98k rows, pos 128–2577, K23+rotate32 html) — `pos,layer,pruned_mass,total_mass`, normalised over **all 256** | engage S1 measured | text-collapse token not in file (invalid-UTF8 tail); onset via m1a (~gen126) |
| **Live S1 sensor K91** | `k91_coding_vram/loop/s1_sensor.csv` (104k rows) + scope json `k91_collapse.divergence` (onset_pos 2286, collapse_pos 2476) | engage + drift measured | — |
| **Outcomes / labels** | retro-grade `graded.csv` (L0-L3); knee README (JSON keep-20 L3, Python keep-32 L3, frontpage cold >32 collapses); warmup W-table (W50/130 L3, W80/110 L2, W150 L1); rotate-vs-static m1a collapse | weak labels | **no full-model routing (with weights) for JSON / Python / coffee** — only knee-K labels; coffee has no routing trace at all |

**Two different "coverage"** exist and must not be conflated:
- **def-1 (routed/selected):** denominator = mass on the 6 selected experts, summed
  over warmup. This is what the trace produces **offline / for free**.
- **def-2 (full-256):** denominator = the full unbiased softmax over all 256 experts.
  This is what the **S1 sensor** measures (`pruned_mass/total_mass`) and what tracks
  collapse. **It is unobservable from the trace** (only 6 experts logged).

---

## 2 — The warmup coverage curve (def-1), fixed 50-token engage window

The mask engages after `PACE_WARMUP=50`, so the curve an in-engine adaptive-K would
see is built from the **first 50 tokens**. On that window every full-model trace —
html *and* all 11 coding prompts — collapses onto **the same curve**:

| trace | tok | cov@23 | cov@32 | cov@48 | Kmin80 | Kmin90 | Kmin95 | knee |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| html_W50 | 49 | 79% | 86% | 93% | 24 | **39** | 51 | 25 |
| html_W130 | 129 | 81% | 87% | 94% | 21 | 35 | 48 | 24 |
| code_python-csv | 255 | 79% | 86% | 93% | 26 | 42 | 54 | 25 |
| code_js-debounce | 255 | 81% | 88% | 95% | 20 | 33 | 43 | 26 |
| code_regex-email | 255 | 73% | 82% | 91% | 29 | 44 | 56 | 29 |
| code_docker-multi | 255 | 80% | 87% | 94% | 21 | 34 | 48 | 25 |
| … (all 11 coding) |  | 73–81% | 82–88% | 91–95% | 20–29 | 33–44 | 40–56 | 24–29 |

**Aggregate (13 traces):** `cov@23 = 79%` (range **73–82%**), `cov@32 = 86%`,
`Kmin-cov90 = 38` (cov80 = 24, cov95 = 51). Kneedle knee at **K≈25–29** everywhere;
marginal gain of the 23rd expert ≈ 1.0%/expert (still rising) for every trace.

> **F1 — the curve is near task-invariant at engage.** At W50 the routed-mass
> concentration does **not** separate html from coding (cov@23 spread 73–82%). A
> coverage rule therefore picks **~one K for every task** (cov90 → K≈38, cov80 →
> K≈24). It is **not** the task-width discriminator the idea hoped for. (The "coding is
> wider" impression is a *token-count* artefact: on the full 255-tok trace coding looks
> wider, but at the 50-tok engage window it does not.)

---

## 3 — Calibration against outcomes

### 3a. The def-1 vs def-2 gap breaks the `S1 = 1 − coverage` identity

| quantity | html keep-23 |
|---|--:|
| warmup `1 − cov@23` (def-1, predicted S1) | **0.21** |
| live measured S1 (def-2, full-256) | **0.81** |
| ratio | **×3.8** |

The unbiased router is heavy-tailed: the 6 selected experts carry only ~20% of the
full-256 softmax mass, so keeping the warmup top-23 leaves ~81% of router mass on
pruned experts. **Predicted and measured S1 differ ~4×**, and the measured one cannot
be reconstructed from the trace. **F2 — the identity is false; the "free" curve is not
a predictor of engage-S1.**

### 3b. Neither normalisation orders survival (the monito, quantified)

| run | K | actuation | provenance | cov def-1 | S1 def-2 | outcome |
|---|--:|---|---|--:|--:|---|
| K91-static (coding) | 91 | static | cold-corpus | — | **0.845** | **survives** ~2476 |
| K23-rotate32 (html) | 23 | rotate32 | session-W50 | 0.79 | **0.811** | **collapse** ~gen126 |
| keep-23 static (html W50) | 23 | **static** | session-W50 | **0.79** | — | **L3** clean |
| keep-23 cold-static (frontpage) | 23 | static | **cold-corpus** | — | — | **L0** loop |
| JSON keep-20 cold-static | 20 | static | cold-corpus | — | — | **L3** exact |
| PYTHON keep-32 cold-static | 32 | static | cold-corpus | — | — | **L3** tests pass |

- **def-2:** survivor S1 (0.845) **>** collapser S1 (0.811) → **non-monotone**; the
  lower-coverage run survived 20× longer. No threshold separates.
- **def-1:** decisive counter-example — **keep-23 STATIC survives (L3) and keep-23
  ROTATE collapses at the *identical* warmup coverage 0.79.** Same warmup, same K, same
  coverage, opposite outcome.

> **F3 — what actually separates is NOT in the coverage curve.** The three factors that
> flip the outcome are **actuation mode** (static >> rotate at equal coverage), **mask
> provenance** (session-learned >> cold-corpus at equal K — knee README: cold keep-23 =
> L0 loop, session keep-23 = L3), and **token budget** (retro-grade: nothing < 2000 tok
> reaches `<body>` regardless of mask). Coverage@K is blind to all three.

### 3c. Comparison with the historical cov-90 rule (CLAIM-007)

CLAIM-007 put "K≈36–39 at cov-90, L2/L3" — this replicates on the def-1 engage curve
(**Kmin-cov90 = 38**, html_W50 = 39). But the knee **survival** labels (JSON 20,
Python 32, frontpage-cold >32) sit around the **cov80–cov85** band of this same curve,
not cov90 — i.e. the cold-static survival knee is *lower* K than cov90 prescribes, and
it is task-ordered in a way the W50 curve is **not**. So cov-90 is a reasonable *upper*
sizing default but is **not** the empirical survival boundary.

### 3d. What K would coverage-sizing have chosen? (vs the fixed K23)

| task | source | cov80 | **cov90** | cov95 | note |
|---|---|--:|--:|--:|---|
| **cyberpunk html** | W50 warmup | 24 | **39** | 51 | K23 sits **below** the cov90 knee (79% covered) |
| cyberpunk html | W130 warmup | 21 | 35 | 48 | longer warmup → slightly lower K |
| **coding** (11-prompt median) | trace_coding | 24 | **38** | 50 | indistinguishable from html at W50 |
| **JSON** | *no routing on disk* | — | — | — | label only: cold-static knee **keep-20** (≈cov80 band) |
| **Python** | *no routing on disk* | — | — | — | label only: cold-static knee **keep-32** (≈cov85 band) |
| **coffee** | *no trace at all* | — | — | — | **data gap** — no routing weights exist |

A `cov90` rule bumps the fatal fixed **K23 → ~38–39 for cyberpunk and coding alike** (a
uniform "don't under-provision below the knee" move), would **over-provision** JSON
(needs ~20) and Python (~32), and **cannot be evaluated for coffee**. Whether 23→39
prevents the html collapse is **untested** — it is the one action the warmup curve
legitimately supports, and it needs a pod A/B to decide.

---

## VERDICT

> **The coverage-at-engage threshold does NOT separate survivors from collapsers, and
> is not usefully task-conditioned.** (1) At the 50-tok engage window the warmup curve
> is **near task-invariant** — cov@23 = 79% (73–82%), Kmin-cov90 ≈ 38 for html *and* all
> 11 coding prompts — so it picks ~one K for every task. (2) The proposed identity
> `S1 = 1 − coverage` is **false by ~4×** (def-1 routed 0.21 vs def-2 full-256 0.81) and
> the survival-relevant def-2 is **unobservable offline** (trace logs only 6 experts).
> (3) Outcomes are ordered by **actuation mode, mask provenance, and token budget** —
> keep-23 static survives while keep-23 rotate collapses at the *same* coverage 0.79 —
> none of which the curve sees. **Recommended use:** treat `Kmin-cov90 ≈ 38` only as an
> anti-under-provisioning floor (raise fixed K23 → ~38); do **not** deploy a global or
> task-keyed coverage θ as a survival predictor.

### Key numbers
- Engage-window `cov@23 = 79%` (range **73–82%**) across all 13 traces — task-invariant.
- `Kmin-cov90 = 38` (cov80 = 24, cov95 = 51) → cov-sizing picks **K≈39** for cyberpunk, **≈38** for coding.
- def-1 vs def-2: predicted S1 `0.21` vs measured `0.81` = **×3.8** (identity broken).
- Non-separation: survivor K91 S1 `0.845` **>** collapser K23-rotate `0.811`; keep-23 static (L3) vs rotate (collapse) at **identical** cov `0.79`.
- Predictive power of coverage@K-used vs outcome ≈ **chance** (1 survivor below 1 collapser on both axes).

---

## SPEC — candidate patch **0024** `DS4_PACE_COVERAGE` ("coverage-sized descent")

*(spec only — not authored here)*

1. **Goal:** at descent, choose K **per-layer** from live warmup routed-mass so the
   kept set centres a target routed coverage θ (default **0.90**), instead of fixed `KEEP=23`.
2. **Reuse:** the per-expert accumulator built by 0020 (warmup rmass); no new trace
   path — read `g->router_probs` of the selected experts during the 50-tok warmup.
3. **Sizing:** per layer, sort experts by accumulated routed mass; `K_layer =` smallest
   K with cumulative share ≥ θ; clamp to `[KEEP_MIN, KEEP_MAX]`.
4. **Env:** `DS4_PACE_COVERAGE=1`, `DS4_PACE_COV_TARGET=0.90`, reuse `KEEP_MIN/MAX`.
5. **Expected effect (this probe):** raises effective html/coding K from 23 to ~**38–39**;
   near task-invariant → anti-under-provisioning floor, **not** a task discriminator.
6. **Blind spots (must A/B, not assume):** coverage is blind to (a) rotate-vs-static —
   pair 0024 with static-keep, never rotate; (b) session-vs-cold provenance — keep the
   accumulator live (0020), never a cold-corpus rank; (c) token budget — irrelevant
   below ~2000 tok where all masks are L0.
7. **Do NOT** wire θ to the S1 sensor: different normalisations (×~4); a def-1 `cov_target`
   is *not* an S1 target.
8. **Decision gate:** ship 0024 only if a pod A/B (K23-static vs cov90-static, n≥3,
   L0-L3 render, ≥2000 tok) shows cov90-sizing lifts the level at equal-or-better t/s;
   else the curve stays a diagnostic, not an actuator.

### Reproduce
```
python scripts/calibrate_coverage_threshold.py \
  --reap-loop-root <reap-loop> \
  --moe-root <moe-aggressive-commit>/.claude/worktrees/elastic-bose-6ae1c7
```

# E-DET v2 — S1 onset-detector tuning (slow-erosion regime)

Offline replay study to cut the S1 onset-detection latency of the patch-0020
detector (EWMA α=0.10 + slope over win=64 + stable=16, ~50–78 tok delay) down
toward the controller's cost-asymmetry sweet spot. Only offline analysis on
recorded data; no GPU, no pod writes.

Reproduce: `python scripts/tune_s1_detector.py --out
runs/ds4/20260710_edet_s1_detector_tuning` (deterministic, seed 20260710).
Artifacts in this directory: `calibration.json`, `curves_all.csv` (223
configs × all metrics), `recommendation.json`, `summary_table.md`, and the
derived aggregate series `s1_sensor_agg.csv` / `s1_k0proxy_agg.csv` /
`s1_pod_r1_agg.csv`.

## 0. Scope (BINDING)

**This detector is tuned for, and only claimed on, the SLOW-EROSION regime**:
a mild/wide *static* mask (K91-family) whose divergence escalates slowly into
the collapse. In that regime the aggregate S1 drifts 0.845 → ~0.895 with an
onset ~190 tok before the text repetition-lock — there is lead time to detect.

In the **aggressive dynamic regime** (W50+K23+rotate32, the live REAP-LOOP) S1
is **pinned flat ~0.815** from the instant the mask engages (pod r1: mean 0.815,
per-token std 0.017, full-run slope **+5.6e-6/tok** ≈ +0.014 over 2450 tok) and
the output loops almost immediately (~gen 126, corroborated 3/3 by the local M1
runs). **There is no S1 lead to detect there** — the S1-slope airbag does not
help, and this study does not tune for it. The aggressive pod series is used
here *only* as an adversarial false-alarm control (a flat-but-high series a good
detector must not fire on). Provenance: `runs/ds4/20260710_scope_divergence_pod/
README.md` and the scope repo `data/20260710_divergence/README.md`.

## 1. Series inventory

| id | series | regime | role | rows/tokens | provenance |
|---|---|---|---|---|---|
| A | K91 static per-layer S1 (`s1_sensor.csv`) | slow-erosion, static K91 | **primary onset series** | 103,960 rows, 40 layers, pos 138–2736 | moe-aggressive worktree `k91_coding_vram/loop/`; = scope `k91_collapse` scene |
| B | K0 "router-libero" counterfactual S1 | full model (unmasked) × K91 keep | healthy false-alarm | 799 tok, 40 layers | `loop_traces.tgz:trace_k0.csv` × `reap_mask_coding_k91.json` |
| C | Pod r1 per-layer S1 (`s1_r1.csv.gz`) | **aggressive** W50+K23+rotate32 | **adversarial false-alarm control** | 98,000 rows, 40 layers, pos 128–2577 | `runs/ds4/20260710_scope_divergence_pod/r1/` |
| D | synthetic AR(1)+ramp ensembles | calibrated on A | delay-p50/p90 & FA distribution | 80 collapse + 80 healthy (aggregate); 30+30 factor-model (per-layer) | generated from A's stats |

**Ground-truth labels on A** (from the scope/pod READMEs; non-circular): mask on
~pos 294 (S1 0→~0.82); healthy plateau ~pos 350–2250 (S1 mean **0.8525**, per-tok
std **0.0174**, lag-1 autocorr **0.536**); **text repetition-lock (ground-truth
collapse) = pos 2476** (first `// <cjk>` loop marker; exact lock 2508). The
baseline detector's own onset (2286) is *not* used as truth.

Only **one** real collapse-labeled series exists (A), so the delay *distribution*
(p50/p90) is estimated on synthetic series D calibrated on A's healthy noise
(AR(1): μ, σ, ρ matched) plus ramps of measured amplitude (+0.04…0.08 over
100–300 tok; 25% fast steps 20–50 tok). On A itself we report the single
measured **lead = 2476 − (first collapse-region fire)** — positive = warned
before the text visibly loops.

## 2. Method

**Detectors replayed** (each self-calibrates on its own run; live-implementable):

- **(a) baseline slope-0020**: EWMA α, slope over `win`, fire after `stable`
  consecutive over-threshold steps (exact patch-0020 semantics; refractory =
  one window). Grid over α∈{0.10…0.30}, win∈{16…64}, stable∈{4…16}, thr.
- **(b) tight slope**: same detector, narrow (win16, stable4–8, α0.2–0.3).
- **(c) CUSUM on raw S1**: one-sided CUSUM, reference = *lagged* moving-average
  baseline (lag 32, win 128) so a slow ramp is not absorbed into the baseline
  while benign drift still is; σ self-calibrated from the first 128 tok. Grid
  over k_σ (slack)∈{0.25,0.5,1.0}, h_σ (threshold)∈{2…24}.
- **(d) CUSUM on light EWMA** (α∈{0.15,0.30,0.50}, plus a fast-baseline
  lag16/win96 variant): same CUSUM on a lightly-smoothed signal.
- **(e) per-layer k-of-N vote**: an independent lagged-baseline CUSUM per layer;
  fire when ≥K of 40 layers exceed h_σ simultaneously. K∈{4,8,12,16},
  k_σ∈{0.25,0.5}, h_σ∈{2…12}. **Now tunable on real per-layer data** (A and C
  are both per-layer).

**Metrics**: detection **delay from onset** (p50, p90, tokens) and **miss rate**
on the synthetic collapse ensemble; **false alarms per 1000 tok** on healthy
segments — `FA_real` (K91 plateau + K0 router-libero) and `FA_pod` (aggressive
pod r1, adversarial); and the single **real K91 lead** vs the text lock. Curves
= these metrics swept over each detector's threshold (full grid in
`curves_all.csv`).

## 3. Results

### 3.1 Baseline patch-0020 is slow and, at "safe" thresholds, unreliable

| thr (/tok) | delay p50 | p90 | miss | FA_real | K91 lead |
|---|---|---|---|---|---|
| 1e-4 | 46 | 78 | 0.00 | 6.17 | 216 |
| 2e-4 | 59 | 108 | 0.00 | 4.35 | 213 |
| **3e-4 (0020 default)** | **78** | 155 | 0.06 | 2.90 | 195 |
| 5e-4 | 73 | 153 | **0.54** | 2.18 | 182 |
| 8e-4 | 52 | 78 | **0.90** | 1.81 | — |

At its shipped threshold the baseline has a **median delay of 78 tok** (p90 155);
tightening the threshold to lower false alarms makes it **miss 54–90 % of
collapses**. This is the latency (and reliability) the study set out to fix.

### 3.2 Delay-vs-false-alarm frontier (best achievable delay per FA band)

| FA_real cap (/1k) | best delay p50 | config | K91 lead |
|---|---|---|---|
| ≤ 1.5 | 40 | cusum k1 h4 (raw≈ewma50) | 214 |
| ≤ 2.5 | 40 | cusum_ewma50 k1 h8 | 214 |
| ≤ 3.0 | 34 | cusum k1 h3 | 217 |
| ≤ 6.0 | 32 | vote 8/40 k0.5 h8 ≈ cusum_ewma50 k1 h4 (33) | 164 / 222 |
| ≤ 8.0 | 26 | vote 4/40 k0.5 h8 (best agg cusum 31) | 177 |
| ≤ 15  | 19 | slope a20 w32 s8 thr1e-4 | 196 |

**The mandated 10–20 tok is only reachable at FA_real ≥ ~15/1k** (the tight-slope
family), which is unacceptable even for the cheap airbag. **At acceptable false
alarms (≤ 8/1k) the delay floor is ~26–34 tok**; at conservative FA (≤ 2/1k) it
is ~40 tok. So the honest gain over baseline is a **roughly halved latency**
(78 → 31–40), not a move all the way to 10–20. The ramps are slow by nature
(~200 tok), so a slow signal cannot be detected in 10–20 tok without paying in
false alarms.

### 3.3 Per-layer vote does NOT beat the aggregate at conservative FA

Best delay p50 within each FA band, vote vs aggregate:

| FA_real cap | per-layer vote | aggregate (CUSUM) |
|---|---|---|
| ≤ 1.5 | 69 | **40** |
| ≤ 3.0 | 54 | **34** |
| ≤ 6.0 | 32 | 33 (tie) |
| ≤ 8.0 | **26** | 31 |

The aggregate S1 already pools every layer's router-mass into one statistic, so
CUSUM on it is the more *efficient* detector: to hold false alarms down the vote
needs high K/h, which delays it. The vote only edges ahead in the loose-FA
corner (≤ 8/1k), and by ~5 tok — not worth 40× the per-layer state. **Conclusion:
use the aggregate EWMA-CUSUM.** (This is the first time the vote could be judged
on *real* per-layer data — A and the new per-layer pod C — and it loses.)

### 3.4 Adversarial pod control: the recommended detectors do not fire on it

On the flat aggressive pod r1 series (regime with no S1 lead), every recommended
config fires at a very low rate, and the conservative ones fire **zero** times:
`FA_pod` = 2.9/1k (ADMIT) and **0.0/1k** (REWIND). So a detector tuned on the
slow regime does not spuriously trip on the pinned-high aggressive regime — the
regime scoping is safe in the "won't false-fire" direction. (It still cannot
*detect* the aggressive collapse — there is nothing in S1 to detect; §0.)

### 3.5 Real K91 lead

On the real collapse series A, the recommended detectors fire at pos **2251–2262**
— i.e. **210–225 tok before** the ground-truth text lock (2476), matching or
beating the baseline's 182–195 tok lead while cutting the synthetic delay in
half. The lead is bounded by where the drift physically begins (ramp start
~pos 2133–2250), not by the detector.

## 4. Recommendation — one detector, two thresholds

**Detector**: aggregate **EWMA-CUSUM** — EWMA α=0.50 front-end, one-sided CUSUM
with σ self-calibrated over the first 128 tok after mask-on and a lagged-window
baseline (lag 32, win 128) that tracks benign drift but not the onset ramp. Two
operating points on the normalized CUSUM statistic g/σ:

| profile | maps to | k_σ (slack) | h_σ (fire) | delay p50 | p90 | FA_real /1k | FA_pod /1k | K91 lead | miss |
|---|---|---|---|---|---|---|---|---|---|
| **arm-relearn/admit** (aggressive; false alarms cheap → rotate / widen / relearn / admit-expert) | §3.1 **ARM** | 0.5 | 4 | **31** | 55 | 7.98 | 2.86 | 225 | 0.00 |
| **arm-rewind** (conservative; false alarms expensive → rewind) | §3.1 **FIRE** | 1.0 | 8 | **40** | 80 | 1.81 | 0.00 | 214 | 0.00 |

Both share the same α=0.50 EWMA front-end; they differ only in the CUSUM slack
k_σ and the fire threshold h_σ, so they can run as two cheap accumulators off one
smoothed S1. Ultra-conservative variant (if a rewind must almost never be
wasted): k_σ=1.0, h_σ=12 → delay p50 ≈ 42, FA_real ≈ 2.2/1k, FA_pod = 0. Snappier
ARM (if latency matters more than the ~10/1k false-arm rate): k_σ=0.25, h_σ=2 →
delay p50 ≈ 25, FA_real ≈ 12/1k.

Against baseline-0020 (median 78 tok): **ADMIT halves the median detection delay
to ~31 tok, REWIND to ~40 tok**, both at equal-or-lower false-alarm rates and
with zero misses on the synthetic ensemble.

## 5. Limits

- **One real collapse series.** p50/p90 come from synthetic AR(1)+ramp series
  calibrated on A; only the single K91 lead is fully real. The synthetic ramp
  amplitude/duration priors (from CLAIM-011 and the sensor ramp) drive the delay
  numbers — a different real ramp shape would shift them.
- **Slow-regime only.** No claim, and no tuning, for the aggressive
  rotate regime (S1 flat, no lead). If a run's regime is unknown a priori, the
  detector is safe against false firing on the aggressive series (§3.4) but will
  simply never warn there.
- **Aggregate over vote.** The recommendation drops the per-layer vote on the
  evidence in §3.3; if a future mask produced layer-localized erosion (a few
  layers drifting while the aggregate stays flat) the vote could recover value —
  not observed in A or C.
- **Self-calibration window.** σ is fit on the first 128 tok after mask-on;
  a run that is already unhealthy at mask-on would mis-scale the threshold. The
  live controller should gate calibration on a healthy-plateau check.
- **Text-lock label precision.** Ground-truth collapse is char-proportional
  (2476 first marker / 2508 exact lock); leads carry ~±16 tok of label noise.

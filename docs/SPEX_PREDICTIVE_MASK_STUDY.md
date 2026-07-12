# SPEX Predictive Mask Study

Branch: `spex-predictive-mask-study-2026-07-12`

## Thesis

The current live mask is mostly retrospective: it ranks experts from mass already observed in the recent past. That is structurally late at phase transitions. The study here treats SPEX as a prediction source for the next demand, so the mask can keep a narrow resident set while admitting experts before their mass appears in the true router stream.

The invariant stays the same: prediction controls residency / admission, never router gating. A bad prediction costs bandwidth or a resident slot; it must not change tokens by itself.

## Mechanism To Study

Per layer, at token `t`, build the keep set from a blend of:

- `mass_past`: weighted router mass accumulated over a recent window.
- `delta`: rising mass over a short horizon, a cheap phase-change predictor.
- `oracle_shift`: offline upper-bound using future demand at `t+lead`; this is not implementable, but tells us whether prediction can matter.
- `spex_hidden_pred`: externally produced SPX1 hidden scores/topK for the target next layer. This is the only non-oracle predictive source allowed here; no prompt-trained or mask-trained predictor.

The runtime shape we want is not fixed K8 or fixed K23 forever. It is a narrow resident budget, for example `V33..V50`, with short widening at detected transitions and re-tightening when prediction confidence stabilizes.

## First Offline Questions

1. Does any predictive score reduce miss rate versus past-only mass at equal K?
2. Does the gain concentrate at churn / phase-change tokens?
3. What K range gives acceptable miss rate on the cyber/html traces: 23, 33, 40, 50, 64?
4. How close is mass+delta to the future-oracle upper bound?
5. Is a fixed K23 failing because it is too narrow, or because the ranking is late/wrong?

## Acceptance Rules

- No verdict from n=1 prose quality alone.
- Offline mask scores are not quality verdicts; they only decide what runtime tests deserve GPU time.
- Runtime tests still need L0-L3 grading, n>=3 for claims, and exact launch parameters.
- If predictive scores reduce misses but quality still fails, the next suspect is task capacity / model quality under constrained experts, not cache plumbing.

## Runtime Direction If Offline Looks Good

1. Add an env-gated SPEX-assisted livemask mode.
2. Keep pin-by-mass for high-confidence heavy experts.
3. Use SPEX/delta to reserve a small admission band for rising/future experts.
4. Widen briefly on phase transitions, then decay back to the target resident budget.
5. Measure against equal-budget random and mass-only controls.

## Current Context

Recent branch work showed the consumer chain works mechanically: pin-by-mass, thrash fix, and fattorino can run. The failure mode is semantic collapse under narrow masks, especially on multi-phase HTML prompts. That makes the next useful question predictive admission, not another pure retrospective mask sweep.

## First Offline Measurements (2026-07-12)

Script: `scripts/analyze_spex_predictive_mask.py`.
Trace inputs:

- `runs/ds4/20260711_highK_sweetspot/traces/trace_K91/route.csv`
- `runs/ds4/20260711_highK_sweetspot/traces/trace_K64/route.csv`

Settings: `window=16`, `delta_horizon=4`, `alpha_delta in {0,1}`, no markov proxy, `oracle_band in {0,2,4,6}`.

### Result summary

Past-mass-only miss rate:

| trace | K23 | K40 | K50 |
|---|---:|---:|---:|
| K91 demand trace | 25.99% | 15.56% | 14.44% |
| K64 demand trace | 25.69% | 13.75% | 12.83% |

Past mass + naive delta did not help in this first form; it was slightly worse. That means the next useful predictor is not a raw slope term by itself.

Oracle admission band upper bound, reserving 4 slots for perfectly predicted current demand and filling the rest with past mass:

| trace | K23 + oracle4 | K40 + oracle4 | K50 + oracle4 |
|---|---:|---:|---:|
| K91 demand trace | 15.34% | 8.51% | 7.48% |
| K64 demand trace | 14.56% | 7.22% | 6.27% |

Interpretation: a small predictive band has real headroom. The target runtime mechanism should not replace mass; it should reserve a small admission band for SPEX-predicted rising experts while keeping heavy mass experts pinned. The K range worth testing is probably `V33..V50`, not K8 and not K23-only, with short widening at phase transitions.

Caveat: `oracle_band=6` gives 0% miss because DS4 routes six experts/layer and the oracle is perfect; this is only a ceiling, not an achievable runtime result. The real next step is to replace oracle with externally produced SPEX hidden/topK predictions and measure how much of that headroom is reachable. Prompt-trained or mask-trained predictors are explicitly out of scope.
## Constraint Correction (2026-07-12)

Do not train a predictor on the evaluated prompt, on the same route trace, or on a learned mask. The only allowed predictive signal for this branch is the pre-existing SPX1 hidden predictor, or a CSV exported from that predictor. The mask policy can use:

- observed router mass for the pinned stable core;
- current SPEX hidden scores/topK for a predictive admission band;
- an EWMA/ring of SPEX-predicted scores as `predicted_mass`.

It must not use a trace-trained Markov table as evidence for this objective. Oracle rows remain allowed only as an upper bound label, not as an implementable method.


## SPX1 Hidden Prediction Check (2026-07-12)

User constraint applied: no prompt-trained predictor, no mask-trained predictor. The SPX1 artifact used here is the pre-existing hidden predictor:

- `moe-aggressive-commit/runs/spex/spex_model/ds4flash_d2_nextlayer.spex`
- header verified: `SPX1`, predictor `2`, shape `L43 D4096 E256`.

Archived DSH alignment for `runs/spex/2026-07-05_trace_pod`:

- DSH row fields are `pos,layer`.
- Routing CSV starts at `pos=42`, so use `--hidden-pos-offset 42`.
- The meaningful target is `hidden L -> routing L+1`, so use `--target-layer-delta 1`.

Full SPX1 recall on archived traces, no training performed in this branch:

| trace | top6 weighted | top12 weighted | top23 weighted | top23 hit-any |
|---|---:|---:|---:|---:|
| `coding_en/c00_python-csv` | 0.2342 | 0.3380 | 0.4384 | 0.9210 |
| `generale_ita/g00_storia` | 0.2822 | 0.3575 | 0.4443 | 0.9205 |

Mask replay on `coding_en/c00_python-csv`, using exported SPX1 top23 predictions as an external CSV:

- Past-mass-only miss: K23 `38.73%`, K33 `31.54%`, K40 `27.92%`, K50 `24.95%`.
- Oracle4 upper bound on the same trace: K23 `20.76%`, K33 `17.31%`, K40 `15.20%`, K50 `13.13%`.
- SPX1 topK reserved admission band was negative on this trace: at K23, pred_band 2/4/6 worsened miss from `38.64%` to `40.18/41.41/42.55%` with `pred_gamma=0`; larger prediction blending worsened further.
- The only tiny positive in this first replay was score blending without reserved band at K50: `24.10% -> 23.91%` with `pred_gamma=1`, but this is too small to treat as a finding.

Interpretation: the oracle says a predictive band could matter, but this raw SPX1 topK admission policy does not capture the headroom on the archived coding trace. Next candidates must be more selective than "reserve fixed slots for SPX1 topK": confidence thresholding, calibration, layer gating, or using SPEX only to prefetch/residency rather than to displace observed-mass slots in the keep mask.

Additional equal-total-budget check on `coding_en/c00_python-csv`:

| policy | miss |
|---|---:|
| mass-only K33 | 31.41% |
| mass core + SPEX extra, V33 (`pred_band=10`) | 35.47% |
| mass-only K40 | 27.85% |
| mass core + SPEX extra, V40 (`pred_band=17`) | 32.98% |
| mass-only K50 | 24.10% |
| mass core + SPEX extra, V50-ish (`pred_band=23`) | 28.72% |

This rules out the naive policy "replace/add fixed mask slots from raw SPX1 topK" on this archived coding trace. It does not rule out SPEX as prefetch/residency, because that path does not displace observed-mass keep slots and a bad prediction costs only load bandwidth/resident pressure.

Top-rank check on the same coding trace: top1 weighted recall `0.0797`, top2 `0.1298`, top4 `0.1908`, top6 `0.2342`. This is not strong enough for a high-confidence mask admission rule. Immediate decision: do not let raw SPX1 predictions displace observed-mass mask slots. Investigate the gap versus earlier high-recall ledger entries before wiring SPEX into mask selection.

## Additive Oracle Check (2026-07-12)

This is a different experiment from `oracle_band=4`. The base mask remains the
full past-mass K23, then up to four perfect-foresight experts missing from that
mask are appended. Experts already present in K23 do not consume an oracle slot;
the oracle scans current router demand in descending weight order for the next
missing expert. The resulting mask is therefore at most K27.

Measured over all 12 archived `coding_en` routing traces with `window=16`:

| policy | mean miss | range across traces | mean mask size |
|---|---:|---:|---:|
| past-mass K23 | 37.313% | 33.964-39.984% | 22.78 during warmup, then 23 |
| K23 + additive oracle4 | 1.093% | 0.706-1.377% | 24.954 |

Relative miss reduction: `97.07%`. The mean mask stays below 27 because the
router selects six experts and K23 already contains several of them; once every
missing routed expert has been added, extra oracle capacity has no useful expert
to fill. This is a perfect-hindsight ceiling from routing replay, not a runtime
quality result and not evidence that SPX1 currently achieves it.

The simulator now distinguishes `--oracle-bands` (reserved inside total K) from
`--oracle-adds` (duplicate-free slots appended beyond K), and refills exclusions
without losing capacity to set overlap.

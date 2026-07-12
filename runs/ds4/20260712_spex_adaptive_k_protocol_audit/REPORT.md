# SPEX / adaptive-K work audit - 2026-07-12

Status: stopped by user request. No further tests should be inferred from this
directory. Raw artifacts are committed, including invalid and negative runs.

## Corrections that override earlier wording

1. The historical session-learning W50/W130 results used a learned **static
   frozen mask**. They are not evidence that the 0045 dynamic controller works.
2. The adaptive/SPEX 0045 runs were hard-capped at 80 generated tokens. This
   was a protocol error: the instruction was to stop when output degenerated,
   not to cap every run at 80. These runs are mechanism micro-smokes only and
   cannot measure completion or establish L0-L3 quality.
3. The 0045 strong-knock count controls **width**, but K growth still chooses
   entrants by recent mass10. The experts whose instantaneous weight caused the
   acceleration are not necessarily admitted. Trigger and actuator remain
   mismatched.
4. No 0045 arm produced or was allowed to attempt a complete document. There
   is no n>=3 quality verdict for adaptive K or additive SPEX.

## Sequence of work

### 1. Prediction audit

SPX1 hidden prediction was replayed offline without training on the evaluated
prompt or mask. Raw top predictions were weak: top4 weighted recall was about
19.08%; additive precision across the evaluated traces was about 4-7%.
Measured captured demand mass was small: roughly 0.8-1.18% for add1,
1.36-2.08% for add2, and 2.34-3.31% for add4.

Commit: [d49fcf1](https://github.com/imanu86/reap-loop/commit/d49fcf1e0db8080069b213887acf0bd87843dd12)
`spex: add prediction replay and hidden alignment study`.

### 2. Provisional SPEX lane (0044)

Implemented a duplicate-free additive lane above the mass core. Predicted
experts are synchronously WRAP-loaded before mask admission, receive temporary
VRAM/mask priority, and are released after their lease. The first top4/K23
mechanism smoke was strongly negative: about 154 `(layer,expert)` loads per
token, 3.40% final cache hit, and 0.61 t/s over the five post-K23 tokens.

Commit: [ebe73b6](https://github.com/imanu86/reap-loop/commit/ebe73b69eb21c7eea10df00349a5c4736aeeb6a8)
`0044: add provisional SPEX mask pin and WRAP lane`.

### 3. Cadence and persistence

Added configurable admission width 1/2/4 and rank-weighted support accumulated
between admission epochs. Cadence was intended to make admission persistent
without claiming calibrated predicted router weights.

Cache256 mechanism screens:

| cadence | entrants | WRAP batches | decode t/s | cache hit |
|---:|---:|---:|---:|---:|
| 2 | 3554 | 1244 | 1.61 | 9.15% |
| 4 | 1694 | 620 | 2.07 | 8.83% |
| 8 | 878 | 310 | 2.41 | 9.13% |

Cache400 mechanism screens:

| cadence | decode t/s | cache hit |
|---:|---:|---:|
| 2 | 0.78 | 45.00% |
| 4 | 1.80 | 44.65% |
| 8 | 2.20 | 44.55% |

The attempted cadence2 quality sequence is a negative/incomplete protocol:
r1 and r2 generated obviously malformed looping HTML/CSS at 800 tokens; r3 was
stopped during generation at 129 tokens. It is not n=3 and no median verdict is
valid. Raw `spex_mask` and `spex_pin` logs are retained compressed.

### 4. Adaptive-width offline study

Defined the global strong-knock signal from the unbiased pre-mask router:

`B(t) = count(excluded counterfactual top6 with selected-normalized weight >= threshold)`

`A(t) = B(t) - mean(B(t-10)..B(t-1))`

The TV-distance controller and per-layer-only controller were rejected in
offline screens. On the existing Cyberpunk route, the selected integral policy
used K16..K50, threshold 0.15, gain 0.5, deadband 2, update every 2 tokens, and
asymmetric steps +4/-1. Offline it averaged K35.76/p90 K47 with 26.53% weighted
miss and 0.98 turnover/row. This is routing replay, not generated quality.

### 5. Runtime adaptive K (0045)

Implemented the strong-knock counter, K16..K50 integral controller, anchored
delta resize, SPEX cadence, and a separate adaptive-prefetch gate so logical K
growth does not page-touch every newly eligible expert. CUDA sm_86 built cleanly.

Commit: [d2267a1](https://github.com/imanu86/reap-loop/commit/d2267a1b8b189155a8d2a0aaa912fc041d62fc6c)
`0045: add strong-knock adaptive K and SPEX cadence`.

### 6. Runtime micro-smokes and failures

| run | validity | decode t/s | cache hit | avg K | observation |
|---|---|---:|---:|---:|---|
| adaptive_k_smoke | INVALID | - | - | - | reset bug kept history at 1 / strong at 0 |
| adaptive_k_smoke_fix | ABORTED | - | - | - | uncapped adaptive prefetch created batches up to 800 experts / 5.4 GiB |
| adaptive_k_smoke_stepcap | MECHANISM | 0.77 | 34.3% | - | +4/-1 but adaptive page-touch still expensive |
| adaptive_k_smoke_nodelta | MECHANISM | 0.97 | 36.7% | - | adaptive page-touch disabled |
| adaptive_k_smoke_every2 | MECHANISM | 1.03 | 35.95% | 34.98 | 26 K changes; valid 80-token prefix |
| adaptive_k_spex_add0_warm | NEGATIVE PREFIX | 1.01 | 35.93% | 31.59 | malformed doctype |
| adaptive_k_spex_add1_clean2 | MECHANISM | 2.24 | 29.51% | 34.29 | 1006 entrants; valid start only |
| adaptive_k_spex_add1_warm | MECHANISM | 1.86 | 30.46% | 31.59 | 1015 entrants; valid start only |
| adaptive_k_spex_add2 | MECHANISM | 1.56 | 22.16% | 34.79 | 1948 entrants; valid start only |
| adaptive_k_spex_add4 | MECHANISM | 1.40 | 17.91% | 32.35 | 3601 entrants; valid start only |

`adaptive_k_spex_add1` and `adaptive_k_spex_add1_clean` retain failed/polluted
startup attempts caused by a surviving local DS4 process and interrupted
requests. They must never be used as measurements.

### 7. Protocol correction

The 80-token hard cap was removed. The corrected runner defaults to the
unchanged historical Cyberpunk request, max_tokens 4000, ctx 6144, streaming,
and client stop on `</html>` or objective adjacent repeated line/3-gram
signatures. The corrected protocol was committed but **not run** before the user
requested documentation and stop.

Commit: [9bdbdfd](https://github.com/imanu86/reap-loop/commit/9bdbdfdce837aca7031852f945d885cce882df99)
`protocol: replace 80-token cap with guarded completion run`.

## What remains explicitly unfinished

- No quality-length run of 0045.
- No complete output from adaptive K or additive SPEX.
- No n>=3 L0-L3 matrix for 0045.
- No direct admission of the strong-weight knockers that trigger K growth.
- No three-position average of predicted SPEX scores; current cadence uses
  Borda-style rank support, not calibrated predicted weight.
- No instrumentation on/off overhead A/B for the corrected completion protocol.
- Dynamic expert compression was not advanced in this SPEX work segment.

## Raw artifact policy

All directories beside this report are preserved as produced. Empty responses,
shutdown errors, malformed outputs, and invalid startup logs are intentional
evidence. Large cadence JSONL files are gzip-compressed without changing their
contents. The original 80-token runners are included to make the protocol error
reproducible.

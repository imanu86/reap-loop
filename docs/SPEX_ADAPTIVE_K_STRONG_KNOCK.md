# SPEX cadence + strong-knock adaptive K (0045)

Status: experimental, off by default. CUDA build on RTX 3060/sm_86 passes.
Short screens are observational and are not n>=3 L0-L3 verdicts.

## Question

Can a narrow anchored mask widen before quality collapses when the
counterfactual router shows an accelerating number of strong excluded experts?
Can raw SPX1 add only one or two predicted experts without destroying a
256-slot global GPU cache?

No component is trained on the evaluated prompt or on an evaluated mask.

## Strong-knock controller

For token `t`, each routed layer exposes its unbiased router probabilities
before the REAP mask bias. The router therefore scores all 256 expert IDs even
when an expert is not resident.

1. Take the counterfactual router top-6 for each routed layer.
2. Normalize their weights over the selected six.
3. Count an expert when its normalized weight is at least `0.15` and it is
   outside the effective core plus provisional SPEX lane.
4. Sum over layers to obtain the global strong-knock count `B(t)`.
5. Compute `A(t) = B(t) - mean(B(t-10)..B(t-1))`.
6. Every two tokens, apply `round(0.5*A(t))` with deadband 2, capped at `+4/-1`,
   then clamp the core to K16..K50.

The trigger is instantaneous router **weight**, not accumulated mass. Recent
10-token router mass is used only to choose which excluded expert enters when
K grows and which incumbent leaves when K shrinks. Membership is anchored: a
resize changes only the required delta and never wholesale re-ranks the mask.

`DS4_PACE_LIVEMASK_KNOCK_PREFETCH=0` keeps adaptive growth logical. It avoids
page-touching as many as 160 experts when global K grows by four. SPEX entrants
still use mandatory synchronous WRAP before mask admission.

## SPEX cadence

`DS4_PACE_LIVEMASK_SPEX_ADD` now accepts 1, 2, or 4. Between admission epochs,
SPX1 rankings accumulate Borda-style rank support. Every
`DS4_PACE_LIVEMASK_SPEX_CADENCE` tokens the best external candidates replace
the provisional lane atomically and all entrants arrive in one WRAP batch.

Rank support is not a calibrated predicted router weight. A three-position
rolling mean of predicted scores is deliberately left for the next isolated
patch so its cost and value can be measured separately.

## Runtime screen

Shared settings: RTX 3060 12GB, native `ds4-2bit.gguf`, CUDA sm_86, context
2048, cache 256, reserve 0.25 GiB, prefill chunk 512, greedy, no-think,
Cyberpunk HTML prompt, 78 prompt tokens, 80 generated tokens, LIVEMASK W16,
window 10, K16..K50, update every 2, `+4/-1`, threshold 0.15, gain 0.5,
deadband 2, ordinary LIVEMASK rotation neutralized, pin-by-mass enabled.

| arm | decode t/s | cache hit | avg K | max K | SPEX entrants | mean WRAP | prefix screen |
|---|---:|---:|---:|---:|---:|---:|---|
| add0, earlier | 1.03 | 35.95% | 34.98 | 50 | 0 | 0 ms | valid start |
| add0, later | 1.01 | 35.93% | 31.59 | 41 | 0 | 0 ms | malformed doctype |
| add1, screen A | 2.24 | 29.51% | 34.29 | 46 | 1006 | 2.18 ms | valid start |
| add1, screen B | 1.86 | 30.46% | 31.59 | 40 | 1015 | 3.57 ms | valid start |
| add2 | 1.56 | 22.16% | 34.79 | 49 | 1948 | 5.39 ms | valid start |
| add4 | 1.40 | 17.91% | 32.35 | 44 | 3601 | 10.76 ms | valid start |

All WRAP batches completed. Separate process starts had materially different
prefill/page-cache time, so the decode numbers are screening measurements, not
a paired performance verdict. The add0 malformed prefix is an early rejection
under the current screening rule, not an n=1 quality claim.

Measured pattern: increasing raw SPEX width monotonically increases entrants
and WRAP cost while decreasing cache hit rate. Add1 is the only candidate that
passed two short syntax screens and is the next arm worth a formal quality run.

## Flags

- `DS4_PACE_LIVEMASK_K_ADAPTIVE=1`
- `DS4_PACE_LIVEMASK_K_MIN=16`
- `DS4_PACE_LIVEMASK_K_MAX=50`
- `DS4_PACE_LIVEMASK_KNOCK_THRESHOLD=0.15`
- `DS4_PACE_LIVEMASK_KNOCK_GAIN=0.5`
- `DS4_PACE_LIVEMASK_KNOCK_STEP_UP=4`
- `DS4_PACE_LIVEMASK_KNOCK_STEP_DOWN=1`
- `DS4_PACE_LIVEMASK_KNOCK_DEADBAND=2`
- `DS4_PACE_LIVEMASK_KNOCK_UPDATE_EVERY=2`
- `DS4_PACE_LIVEMASK_KNOCK_MIN_HISTORY=2`
- `DS4_PACE_LIVEMASK_KNOCK_PREFETCH=0`
- `DS4_PACE_LIVEMASK_SPEX_ADD=0|1|2|4`
- `DS4_PACE_LIVEMASK_SPEX_CADENCE=2`

## Next gates

1. Add predicted score readout and a rolling mean over three token positions;
   score once on GPU, aggregate metadata without replaying the same hidden.
2. Admit at most one predicted external expert and compare score gating against
   rank support with instrumentation on/off.
3. Only a surviving policy advances to n>=3 complete generations and L0-L3
   grading. Repeat flags and one short prefix never establish quality.

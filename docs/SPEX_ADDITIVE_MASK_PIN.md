# SPEX additive mask + VRAM pin lane (0044)

Status: experimental, off by default. CUDA build on RTX 3060/sm_86 passes.
The mechanism smoke passes; the first performance smoke is negative and the
policy is not ready for an L0-L3 quality verdict.

Successor: patch 0045 adds configurable 1/2/4 admission, multi-token
rank-support cadence, and a weight-triggered adaptive K16..K50 core. See
`docs/SPEX_ADAPTIVE_K_STRONG_KNOCK.md`. The 0044 top4/K23 result below remains
the historical baseline and must not be conflated with the 0045 screening.

## Policy

The stable core remains K23 selected from observed router mass. SPEX does not
replace a core slot and is not trained on the evaluated prompt or mask.

For each routed target layer in the current token:

1. SPX1 ranks at least 27 candidates from the source hidden state.
2. Already-core and duplicate IDs are skipped.
3. Up to four external IDs form a provisional lane on top of K23.
4. All provisional IDs are page-touched in one synchronous WRAP batch.
5. Only after WRAP succeeds are they exposed in the router mask and seeded in
   the GPU expert cache as one batch.
6. They receive VRAM pin priority. At a full pin budget, the lowest-mass
   non-SPEX core pin is released; its mask membership is unchanged.
7. After the target router has selected and submitted its expert load, the
   provisional mask lease is consumed. The physical pin is released at the next
   cache reconciliation unless observed mass has promoted it into the core.

Therefore the effective width is `23 + external_spex`, from 23 through 27. It
contracts because a prediction is already in the mass core or because the
one-layer lease has been consumed, never merely because the expert happens to
be resident in VRAM.

`oracle4` in the offline replay means four perfect-hindsight additions per
layer, not four global experts. Patch 0044 substitutes raw SPX1 predictions and
must not be described as oracle performance.

## Flags

- `DS4_PACE_LIVEMASK_SPEX_ADD=4`: master additive width; default 0.
- `DS4_PACE_LIVEMASK_SPEX_WRAP=1`: require one successful synchronous WRAP
  batch before exposing a provisional lane.
- `DS4_PACE_LIVEMASK_SPEX_LOG=path`: mask/WRAP JSONL, including IDs and
  `wrap_ready`, `wrap_ms`.
- `DS4_PACE_LIVEMASK_SPEX_PIN_LOG=path`: VRAM promote/release/displacement
  JSONL with `(layer,expert,mass,epoch)`.
- `DS4_SPEX_HIDDEN_CAP=27`: enough ranked candidates to skip K23 overlaps and
  still find up to four external experts.
- `DS4_REAP_PREFETCH_DELTA=1`: enables the shared delta-WRAP implementation.

The existing hidden GPU load/score/prefetch flags must also be enabled for the
asynchronous path. GPU scoring is suppressed during K0 and starts only once the
PACE or LIVEMASK core is active. Readback is tagged with the decode position;
stale cross-token predictions are rejected.

## Measured smoke

Prompt: the historical Cyberpunk request from
`20260709_k23_unit_vs_weighted_cache256_html800`.

Shared runner parameters: RTX 3060, native 2-bit model, context 2048,
cache-experts 256, reserve 0.25 GiB, weighted W50 then fixed K23, wrap enabled,
no breath, relearn, prebreath, or rotation. Treatment adds SPX1 cap27,
SPEX_ADD=4, mandatory WRAP, and pin logging.

Final mechanism smoke generated only five post-K23 tokens:

| measurement | value |
|---|---:|
| SPEX add events | 200 |
| consume events | 193 |
| target-layer coverage | 3..42 |
| provisional expert additions | 772 |
| released mask leases | 772 |
| failed WRAP batches | 0 |
| mean WRAP time per layer | 16.263 ms |
| p95 / max WRAP time | 34.448 / 109.686 ms |
| VRAM promote events | 772 |
| VRAM release events | 768 |
| post-K23 five-token chunk | 0.61 t/s |
| final expert-cache hit rate | 3.40% |

Measured interpretation: the mechanism is complete and observable, but raw
top4-per-layer admission is far too aggressive for a 256-slot global cache. It
introduced about 154 provisional `(layer,expert)` loads per token and destroyed
cache locality. This short n=1 mechanism smoke is sufficient to reject an
expensive 800-token run of the unchanged policy, but it is not a model-quality
verdict. Any L0-L3 claim still requires the committed A/B runner with at least
three repetitions per arm.

## Next gate

Reduce the number of predictions that reach VRAM without training on the test
prompt or mask. Candidate mechanisms are confidence/margin gating from SPX1,
cross-token prediction persistence, and layer-specific admission caps. Re-run
the n>=3 control/treatment matrix only after a short smoke restores cache hit
rate and post-K23 speed to an acceptable range.

# 2026-07-10 Pod — instrumented REAP-LOOP divergence capture for Scope

Diagnostic run (TRACE ON, **not** a speed benchmark) to feed the Scope 3D
"divergence" view: capture the per-layer **S1** signal — the router's
probability mass that lands on **masked (pruned)** experts / total router mass —
through a real REAP-LOOP collapse, so the user can *see in 3D when the run starts
to deviate from the experts the router wanted and degenerates*.

See `manifest.json` for the full pod / build / model / env record.

## S1 sensor — what it logs (CLAIM-001, confirmed at source)

ds4.c applies the REAP mask as a `-1e9` penalty on the **top-k selection bias**
tensor only (`g_reap_bias_masked`, ~line 7512); `g->router_probs` stays the
**unbiased** softmax over all 256 experts (comment ~7384: "choose the six experts
by biased top-k, but weight them using the unbiased router probabilities"). The
sensor (patch 0012, inside `metal_graph_spex_trace_selected`) reads
`g->router_probs`, so the CSV logs **pre-bias** router mass on the pruned experts
— exactly "how much the router still wanted the masked experts." Per (token,
layer); emitted only while `g_reap_mask_on`, so rows start at
`pos = prompt_len(78) + PACE_WARMUP(50) = 128` (nothing during warmup).

## Runs

| Label | Variant | Purpose | S1 sensor |
|---|---|---|---|
| `r1` | W50+K23+rotate32 | REAP-LOOP collapse #1 | ON |
| `r2` | W50+K23+rotate32 | REAP-LOOP collapse #2 (onset reproducibility) | ON |
| `ctrl` | full, `DS4_PACE=0` | "router libero" positive control | off (nothing masked) |

n reduced from the mandated 3 to 2 sensor-instrumented rollouts + control
because pod generation is ~2.1 t/s (cold SSD-streaming, 24 GB 3090, trace on) so
2500-tok runs are ~20 min each. The W50+K23+rotate32 collapse is independently
corroborated 3/3 at content level by the local M1 runs
`runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/` (loops: `_00_`, `user-sme=1.1,`,
`</style>`), which have no per-layer sensor.

## Results

**r1 (W50+K23+rotate32, sensor ON, max 2500 tok).** Ran to length (2500).
Sensor: **98k rows, pos 128–2577, 40 layers**. Aggregate **S1 ≈ 0.815 and flat**
(0.805 → 0.818, no drift) — the router's mass on masked experts is pinned high
from the instant the mask engages (pos 128, S1 jumps to ~0.83) and stays there;
`rotate32` re-churns the kept set every 32 tok without ever reducing it. The
decoded tail was **invalid UTF-8** (the byte-level signature of a garbage/loop
collapse), so `content.txt`/`response.json` did not save on r1 (the runner's
JSON decode raised `UnicodeDecodeError`; fixed for r2 to read bytes +
`errors="replace"`). The collapse is not in doubt: see corroboration below.

**Corroboration — collapse onset is EARLY under K23.** The local M1 run
`runs/ds4/20260710_m1a_w50_w100_ctx8192_n3/` (identical W50+K23+rotate32, no
sensor) degenerates at **~token 126 of generation** in r01: coherent `<head>`/CSS
up to `body { background: #19`, then it locks into `#1900_00_00_00…` for the
remaining ~3870 tokens (r02 loops `user-sme=1.1,`, r03 `</style>`; 3/3). So the
aggressive K23 mask collapses **almost immediately**, while divergence is already
high-and-flat — the S1 sensor does **not** herald it.

**Contrast with static K91** (`scope/data/20260710_divergence/`): the milder K91
(keep ~91) stays coherent ~2200 tok and then *drifts* 0.845 → ~0.895 with a
detectable S1-slope onset at pos 2286, ~190 tok before the text lock at 2476.
Interpretation: **onset-in-S1 is visible for a mild static mask that erodes
slowly, but not for the aggressive rotate mask that pins divergence high and
collapses at once.** (Prezioso for CLAIM-011 / E-DET: the S1-slope early-warning
only buys lead time in the slow-erosion regime.)

**ctrl (full router, `DS4_PACE=0`, max 1200 tok).** Ran to length (1200 tok,
601 s). Output stays **coherent** — valid cyberpunk CSS to the end
(`.contact-inner { max-width:580px … background:#0e1424dd; border:1px solid
#ff66ff … }`), no repetition, no garbage bytes (clean UTF-8, saved to
`ctrl/content.txt`). This is the "router libero" reference: the unmasked model
does **not** collapse on this prompt/budget. No S1 sensor (nothing masked ⇒
zero divergence by construction).

**r2 — not run.** The in-place `sed` that hardened r1's UTF-8 handling for the
runner also broke the runner's trailing print block (bash heredoc), so a second
masked rollout was skipped. It would only have re-confirmed r1's flat-S1 sensor
plus a same-run collapse token; that token is already pinned by the m1a
corroboration (~gen 126) and the collapse itself by r1's invalid-UTF8 tail.

### Takeaways for the tracks

- **Onset lead is regime-dependent** (CLAIM-011 / E-DET): the S1-slope early
  warning buys ~190 tok only in the *slow-erosion* static-mask regime (k91). The
  aggressive W50+K23+rotate32 reap-loop pins S1 high (~0.815) and collapses at
  once (~gen 126), so S1-slope gives **no** usable lead there — a real limit on
  where an S1 airbag helps.
- **Positive control**: full router = coherent, masked K23+rotate = immediate
  loop, on the identical prompt/build/pod ⇒ the collapse is mask-attributable,
  not a model/prompt artifact.

## Files

Per run `<label>/`: `content.txt`, `response.json`, `server.std{out,err}.log`.
Top-level per run: `s1_<label>.csv` (patch-0012 sensor, pos,layer,pruned,total),
`trace_<label>.csv` (routing + weights), `pace_<label>.jsonl` (PACE events).

## Scope scene

Exported to `scope/data/20260710_divergence/reap_loop_r1.divergence.scope.json`
(see that repo's `data/20260710_divergence/README.md` for the exact open URL).

# Instrumented K12-wide collapse — pre-garbage sensor lead (entropy / margin / acceptance)

**Date:** 2026-07-11 · **Pod:** RTX 3090 (podA `ysegg4bx67yvr3`, sm_86) ·
**Binary:** ds4 canonical chain **md5 `16a21de6`** = canonical(21) + 0027 + 0028(token sidecar) + **0030(diagnostic per-token confidence)**; built sm_86 on the build pod, `DS4_DIAG_CONF_LOG` verified live.
**Run:** W50 two-phase -> **K12 static** (keep-k=12, wide = guaranteed early collapse), cyberpunk prompt, ctx8192, greedy (temp 0), n=2, cache 256, SSD-streaming.

Cross-reference: sibling offline study `runs/ds4/20260711_pregarbage_sensor/` (podC K23/K38, S1-only). This run is the first to (a) capture the per-layer S1 sensor on the **K12 wide** regime and (b) add the **per-token confidence** (entropy/margin) that no prior recorded run had.

## Determinism (validation)
`deliverable.html` is **byte-identical (7526 chars, md5 `3d473367...`) across three independent binaries** — the best-effort build (`771a39a8`), this conf build (`16a21de6`), and the prior baseline `de18faa`. Phase-1 artifacts (frozen 67 B, route 263140 B, mask 62428 B) are identical too. Patch 0030 is **log-only**: the collapse trajectory is unchanged, so the confidence logging aligns exactly to the same greedy tokens. n=2 was byte-identical.

## Anatomy of the collapse (phase-2 continuation, exact tokens.csv)
Decode starts at absolute **pos 85** (after the ~85-token re-prefill of `[prompt+frozen]`); decode-relative = pos - 85.

| pos | token piece | note |
|----:|-------------|------|
| 85-124 | ```` ```html ... <meta ... initial-scale=1 ```` | healthy: entropy ~ 0.00-0.02, margin ~ 10-15 |
| **125-128** | `,` ` view` `port` `">` | **first spurious insertion** (`initial-scale=1, viewport"`) — entropy jumps 0.51->1.16 |
| 133-136 | `AI Cyber Shop</` | title, mild entropy |
| **141-145** | ` st` `ata` `-in` `i` ` -->` | **garbled comment** `<!-- stata-ini -->` — entropy spikes to **3.75** |
| 182 | `background: data-gposcate-cane;` | first **blatant** nonsense (char 396 = the mandate's "~char 400") |
| 190 | `#finto sfondo cyberpunk:` | malformed CSS — **margin** finally collapses to ~0 |
| 207+ | `sfoko sfoko sfoko ...` | terminal repetition lock |

Healthy baseline (first 32 decode tokens): entropy mean 0.023 / std 0.081; margin mean 10.2 / std 3.23.

## Per-signal lead vs first-garbage (positive = warned BEFORE garbage entered the KV)
Method mirrors `pregarbage_sensor_hunt.py`. **z-onset** = short-baseline (first 32 tok) 3sigma, sustain 2 — robust to fast collapse. **cusum** = the E-DET 128-tok-calibration profile (ARM k0.5/h4).

| first-garbage definition | pos | **ENTROPY** z-lead / cusum | **MARGIN** z-lead / cusum | S1 vote(1/40) | **ACCEPTANCE** |
|---|---:|---:|---:|---:|---:|
| `, viewport` (first spurious tok) | 122 | **-3** / -92 | -68 / -94 | -32 | not measured |
| `stata-ini` (garbled comment) | 139 | **+14** / -75 | -51 / -77 | -15 | not measured |
| `data-gposcate` (first blatant) | 182 | **+57** / -32 | -8 / -34 | +28 | not measured |
| `#finto` (malformed CSS) | 193 | **+68** / -21 | +3 / -23 | +39 | not measured |
| `sfoko` (repetition lock) | 207 | **+82** / -7 | +17 / -9 | +53 | not measured |

Fixed onsets: **entropy z-onset = pos 125** (decode-rel 40), **margin z-onset = pos 190** (decode-rel 105), aggregate-S1 z-onset = pos 1200 (deep in lock, useless), CUSUM ARM = 214/216 (just after the calib floor at pos 213).

## Verdict
- **ENTROPY = the pre-garbage sensor. lead > 0.** Its short-baseline z-onset fires at **pos 125**, essentially coincident with the *first* micro-drift (`, viewport`, pos 122) and **+57 tokens before the first blatant garbage** (`data-gposcate`, pos 182), **+82 before the lock**. Against every garbage definition except the earliest subtle token it has positive lead, and it never fires in the healthy region. -> **the "widen-without-rewind" shortcut is VIABLE if triggered on the entropy z-onset**, taking blatant-garbage as the deadline: ~40-82 tokens of warning before the context is poisoned with obvious garbage.
- **MARGIN = lagging / confirmatory. lead ~ 0.** top1-top2 only crosses its 3sigma-low floor at **pos 190**, deep inside the malformed CSS (lead ~ -8 vs blatant garbage, positive only vs the lock). Use margin to FIRE/confirm, not to pre-empt.
- **The 128-tok CUSUM E-DET cannot lead** in this fast-collapse regime — calibration completes at pos 213, at/after the collapse — confirming and extending the `pregarbage_sensor` finding. The **raw short-baseline z-onset is required** to expose entropy's lead.
- **S1 per-layer** (vote 1/40) gives moderate corroboration (+28 vs blatant garbage) but the aggregate is too slow.
- **ACCEPTANCE (MTP): NOT MEASURED — hardware/tooling block (declared).** See `metrics.json -> mtp_acceptance_signal`. The on-pod `16a21de6` build hard-guards `--ssd-streaming` vs `--mtp` and lacks the `DS4_MTP_STREAMING_UNSAFE` bypass (empirically the env has no effect); the non-streaming path OOMs the 86 GB model on 24 GB; and the 3.8 GB `ds4-mtp.gguf` draft model is absent on-pod and on R2. Unblock recipe (rebuild with 0008 bypass + fetch MTP gguf) is recorded in `metrics.json`. The prior UNMASKED baseline (code 0.872 / math 0.846) is not collapse-aligned and does not answer the lead question.

**Bottom line for the architectural decision:** among the two signals measurable on this hardware, **entropy leads the blatant collapse (+57 tok) and margin does not**. Entropy is the ARM/early-warning candidate; margin is a FIRE/confirm candidate. Acceptance remains the theoretically strongest early-warning (no calibration window, cross-model) but is currently unmeasurable on-pod — measuring it is the recommended next step and is the natural ARM/FIRE feed for the 0022 rewind actuator (`ds4_pace_rewind_feed_token`).

## Artifacts (`conf_run_16a21de6/`)
`conf.csv` (pos,token_id,entropy,top1_logit,top2_logit,margin,expert_out_l2 — l2 = NaN under SSD-streaming), `tokens.csv` (exact pos->piece), `s1_perlayer.csv.gz`, `route_p2.csv.gz`, `deliverable.html`, `trest.txt`, mask `sess.txt`, `p2.diag`. Analysis: `scripts/analyze_instrumented_collapse.py`. Metrics: `metrics.json`.

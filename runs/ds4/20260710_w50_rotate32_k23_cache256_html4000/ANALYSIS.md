# 2026-07-10 W50 Rotate32 K23 Cache256 HTML4000 — Collapse Analysis

## Verdict

**4000 did not "worsen" anything — greedy is NOT run-to-run deterministic.** `w50_4000`
diverges from `w50_2000` at generated token ~75 (after the K23 descent at tok51, before
the first rotate at tok83), so the two runs are independent rollouts from the same
prompt, not a shared prefix with a longer tail. Any "4000 degrades vs 2000" reading that
assumes a shared prefix is invalidated by this fact.

## Scope

- Run under analysis: `runs/ds4/20260710_w50_rotate32_k23_cache256_html4000` (W50 warmup,
  rotate32, K23, cache256, `max_tokens=4000`).
- Reference: `runs/ds4/20260710_w50_rotate32_k23_cache256_html2000` (same knobs,
  `max_tokens=2000`, already graded L2).
- `runs/ds4/20260710_w100_rotate32_k23_cache256_html4000` never started (directory holds
  only `matrix_config.json` + the runner script, no `*_r01/` output) — excluded from all
  three analyses.
- Three analyzers: prefix-compare, structure-grading, env-pace.

## 1. Run-to-run divergence (not prefix + tail)

- `w50_4000` and `w50_2000` diverge at character 277 of 6603 (~4.2% in), i.e. at the
  76th generated token (`event_index=75`, 0-based). Byte-level (UTF-8) same point:
  offset 284/6878 (4.13%).
- Text is identical up to `...initial-scale=1` and then forks: `w50_2000` continues with
  `.1">` (viewport scale 1.1, title "AI Code Store - CyberPunk"), `w50_4000` continues
  with `">` (viewport scale 1, title "AI Codex – Negozio di Programmazione AI"). Not a
  whitespace/minor diff — the two generations are fully different from that point on.
- Schedule markers for the 4000 run: `first_learned_tok=50` (K0 full-router warmup ends),
  `first_descent_tok=51` (fixed K23 begins), `first_rotate_tok=83` (first cache rotation
  event). The fork at tok~75 falls after the descent (51) and before the first rotate
  (83) — in that window the K23 mask should be fixed and identical in both runs (same
  warmup, same prompt, no rotation yet), so the mask itself does not explain the fork.
- `request_measured.json` / `server_env.json` are identical between the two runs (same
  prompt, system, `temperature=0`, `stream=true`, thinking disabled); the only
  differences are `max_tokens` (2000 vs 4000) and the `DS4_LOCK_FILE`/`DS4_PACE_LOG`
  paths.
- Reading: greedy decoding at temp=0 is not bit-reproducible run-to-run here (consistent
  with floating-point/CUDA-kernel non-associativity flipping a near-tied argmax, which
  then cascades). This rules out H1 as a cause of the fork (token 75 is far from both the
  2000 and the 4096 limits) and makes H4-as-primary-cause unlikely too, since the fork
  precedes any observed rotate/drift event.

## 2. Failure mechanism of the 4000 run

- `onset_token_est = 1397` (~34% of the 4000-token horizon), anchored at char offset 4473.
- Collapses into an exact 6-line CSS cycle inside `.hero .ctat:hover { ... }`
  (`-webkit-text-fill-color:#f0f` / `background-clip:text` / `background:transparent` /
  `border:2px solid #f0f` / `color:#f0f` / `background:#0a0a12`), repeated ~46x to EOF.
- Soft degradation already visible from ~tok1300 (duplicate `.ctat`/`.ctat:hover`
  declarations, invalid `filter: drop-shadow:`).
- `<style>` is never closed: 0 `</style>`, 0 `<body>`, 0 `<div>`, 0 `<button>`,
  0 `<script>`, 0 `</html>` anywhere in the output. A single `<!doctype>` — no document
  restart (not a doctype-loop, not a js-loop).
- ~2600 tokens (65% of the 4000 budget) wasted in the loop; `finish_reason=length`.
- Grading: L0 on full output, L0 on first-2000-chars, no stop-at-html-close point exists
  (`l0l3_stop_at_html_close=-1`) because `</html>` is never emitted.

### Pace ngram signal

| tok | ngram(n=3,win=120) |
|---|---|
| 83 | 0.019 |
| ~1299 | 0.205 (fluctuating 0.15-0.27 from tok83) |
| 1331 | 0.228 |
| 1363 | 0.251 |
| 1395 | 0.343 |
| 1427 | 0.499 |
| 1459 | 0.628 |
| 1491 | 0.635 |
| 1523 -> 3987 (end of run) | frozen exactly at 0.6356 for 77 consecutive rotate events (~2500 tokens) |

`hit` stays pinned at 1.000 throughout in both runs — no useful signal there; the
repetition collapse makes caching artificially "perfect", so hit-rate alone would mask
the problem.

Reconstructing content from `stream_events_measured.jsonl` deltas pinpoints the textual
collapse slightly earlier than the structure grader onset estimate: at char offset 4336
(`event_index=1358`, ~token 1357, t≈607s of the 1537s wall time) the output enters a
185-character block repeated verbatim 48 times to EOF. This precedes the ngram ramp
climb past 0.34 and its saturation (tok1427-1523) by a short lag — consistent with the
rolling window (n=3, win=120) sensing the collapse a little after it starts.

## 3. Hypotheses discriminated

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 ctx saturation | REJECTED | Collapse onset at ~1475 total context tokens (78 prompt + 1397 completion), far from `ctx=4096`. Final usage: `prompt_tokens=78`, `completion_tokens=4000`, `total_tokens=4078` — 18 tokens of margin under ctx=4096; `finish_reason="length"` (clean stop at the requested cap, not a truncation). Grep of `server.stderr.log`/`server.stdout.log` for `context shift\|truncat\|SWA\|evict\|overflow\|exceed`: zero matches in either run. Startup log confirms `context buffers 122.69 MiB (ctx=4096, ..., prefill_chunk=512, raw_kv_rows=768, compressed_kv_rows=1026)`. `ctx=4096` is identical in both runs (not a delta). The 4096 limit is never touched as a cause — it is only reached, harmlessly, as a side effect of the loop already running. |
| H3 complete doc + continuation junk | REJECTED | No `</html>` is emitted in either run (`l0l3_stop_at_html_close=-1` for both). The 4000 run collapses inside `<style>`, before `<body>`/`<div>`/`<button>`/`<script>` ever exist — the degradation happens inside an incomplete document, not after one was completed. A stop-string on `</html>` would not have saved this run (nothing to intercept) and the 2000 run truncation is benign/budget-limited, not degenerate. |
| H4 mask drift on long horizon | Best-supported mechanism (repetition-lock), but not attributable to horizon length alone at n=1 | The observed pattern is a classic low-entropy repetition attractor, and it is the long-horizon run that shows it while the short one does not. But since `w50_2000` and `w50_4000` are independent rollouts (see section 1, fork at tok~75), "2000 healthy vs 4000 collapsed" cannot be read cleanly as an effect of horizon length — it may just as well be that this particular 4000 rollout crossed the collapse threshold before its longer budget ran out, while the ngram series for the 2000 run was still climbing (0.44 at tok1971, up from a local low of 0.057 at tok1619) and might have collapsed too, given more budget. Contributing, code-unverified factor: `BREATH_EVERY=999999` and `RELEARN=0` are disabled in this variant, so `rotate(every=32, decay=0.98, preserve_stable=1)` keeps cycling mechanically with no corrective path once the loop starts; `decay=0.98` (slow forgetting) plus `preserve_stable=1` plausibly reinforce the narrow expert/routing set that the repetitive text keeps calling back. |

## 4. Reference run (w50_2000)

L2, healthy. Structurally clean HTML: `</style>` closed, `<body>` present with
`.container`/header/`section.grid` + 6 cards, each wired via `onclick="showAlert()"`.
Truncated by the 2000-token budget mid-7th `<div class=card>`, before form/footer/popup
and before the `<script>` that defines `showAlert()` (buttons wired via `onclick` but the
handler was never reached — L2, not L3). `l0l3_full == l0l3_first2000 == 2` (it is the
~2000-token run). Never degenerated within its budget — no repetition lock observed;
`finish_reason=length` on a sound document, not a collapsed one.

## 5. Operational levers

- (a) Primary guard — anti-repetition stopper (line/n-gram triple-repeat -> stop).
  Would have fired around tok~1400, saving the ~2600 tokens (65% of the 4000 budget)
  burned in the CSS loop. Enables a stop -> rewind -> retry pattern instead of relying
  on breath, which is disabled in this variant (`BREATH_EVERY=999999`, `RELEARN=0`).
- (b) Secondary belt only — stop-string on `</html>`. Would not have helped either
  run here (H3 never occurred — neither run ever closed the document), but remains
  useful as a belt for genuine H3-type cases (complete doc + trailing junk).
- (c) Process — every quality verdict from now on requires n>=3 rollouts. Quality is
  demonstrably a per-rollout variable, not just a function of t/s: this very pair
  (`w50_2000`/`w50_4000`, identical config, identical prompt, greedy temp=0) already
  diverges at token ~75. Consistent with the `--runs N` flag just added to the matrix
  runner (commit `d0ad967`).

## 6. Env / pace reference

`DS4_PACE_*` (env_effective) are identical byte-for-byte between the two runs:
`WARMUP=50`, `KEEP=23/23/96/0`, `BREATH_EVERY=999999` (disabled), `BREATH_KEEP=96`,
`RELEARN=0`, `DRIFT=1.0`, `PREBREATH=0`, `WRAP=1`, `CACHE_FLOOR=1`,
`CACHE_TARGET_SLOTS=256`, `ROTATE=1`, `ROTATE_EVERY=32`, `ROTATE_DECAY=0.98`,
`WEIGHTED_SELECTED=0`. No delta in the pace/cache config.

Only real deltas, from `runner_manifest.json` server block:

| Field | html2000 | html4000 |
|---|---|---|
| `server_max_tokens` | 2048 | 4096 |
| `request_max_tokens` | 2000 | 4000 |
| `ctx` | 4096 | 4096 (no delta) |
| `port` | 8017 | 8018 (different server instance, irrelevant) |
| prompt | identical, sha256_16 `f1d3118e6edf41bd`, 199 chars | same |

Note: `server_max_tokens` is configured exactly equal to `ctx` in the 4000 run (4096 ==
4096), leaving zero architectural margin for the prompt — a config hygiene issue worth
fixing, though it did not actually cause the observed collapse (the request effective
4000 < 4096 never hit that wall; see H1 above).

Pace series for the 2000 run (63 events): same initial ramp shape as the 4000 run
(0.018 -> ~0.1-0.3 fluctuating) but no lock-in; the run ends at `finish=length`, tok=2000,
with ngram still rising (0.44 at tok1971, up from a local minimum of 0.057 at tok1619)
rather than saturating.

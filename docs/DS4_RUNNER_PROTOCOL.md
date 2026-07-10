# DS4 runner protocol

Purpose: make every DS4 number traceable to a measurement, not to memory or
deduction.

## Evidence classes

- `measured`: copied from DS4 logs, HTTP response usage, generated files, or
  direct file stats. These values can enter result tables.
- `derived`: calculated directly from measured fields kept in the run
  directory. The formula must be stated.
- `hypothesis`: explanation or suspicion. It can guide the next test, but it is
  not a result.

If a value cannot be traced to a run artifact, do not put it in a benchmark
table.

## Default local profile

`SOTA_LOCAL_3060_TIMED` is the benchmark base for the user's local RTX 3060
setup. It mirrors the UI launcher knobs but keeps file-heavy diagnostics off
unless they are the variable under test.

Important defaults:

- PACE on, warmup 50, direct K23 after warmup.
- Cache target 258 experts, ctx 6144 for UI-realistic tests, prefill chunk 512.
- SPEX hidden GPU load/score/prefetch off unless SPEX hidden is the test.
- `DS4_PACE_EXCHANGE_OBSERVE=0` in timed benchmark rows.
- `DS4_EXPERT_TIERING_LOG=""` and `DS4_EXPERT_TIERING_LOG_IDS=0` in timed
  benchmark rows.
- `DS4_SPEX_TRACE_ROUTING=""` and `DS4_SPEX_TRACE_ROUTING_WEIGHTS=0` in timed
  benchmark rows.

## Scope/CSV trace policy

Routing CSV is valuable for Scope replay, but it is not free until measured as
free.

Current measured A/B: `runs/ds4/20260709_trace_ab_html220_RESULTS.md`.

Operational rule:

1. Benchmark candidate policies with routing trace off.
2. When a candidate is interesting, run a diagnostic twin with
   `DS4_SPEX_TRACE_ROUTING=<run_dir>/routing.csv` and
   `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`.
3. Use the diagnostic twin for Scope replay, not as the primary timing row,
   unless a larger alternating A/B directly measures trace overhead within
   noise.

## n=3 / ABAB / grading (matrix runner)

`scripts/run_ds4_exchange_matrix.py` supports repetition, order interleaving, and
optional functional grading. All of these are additive: the defaults reproduce
the previous single-pass behavior in the measurement path.

- `--runs N` (default 1): N repetitions per variant into `<stem>_rNN`
  directories. `summary.csv` gains a `run_index` column (the legacy `run` column
  is kept). A new `summary_median.csv` (+ `summary_median.json`) reports, per
  `(prompt, variant)`, the median of `avg_tps`/`first50_tps`/`prompt_s`, the
  median render grade `l0l3_median`, and a majority vote of the binary quality
  flags: a flag is 1 when at least half of the runs raised it (so `repeat_flag=1`
  requires ≥ ⌈N/2⌉ runs to repeat). The 3060 noise floor is ±50%, so treat the
  median row — not any single run — as the reportable number.
- `--order {sequential,abab}` (default `sequential` = legacy block order):
  `abab` interleaves variants (A, B, A, B, …) across runs instead of running each
  variant's repetitions back-to-back, decorrelating the warm state from order.
- Quality flags: `alert_in_script` fires on `alert(`/`confirm(`/`showModal` only
  inside `<script>...</script>` blocks of the generated content. The legacy
  `has_popup` column (naive substring match on the raw output — a prompt echo in
  a comment counts as a feature) is retained for continuity but is DEPRECATED;
  prefer `alert_in_script`.
- Prompt set: `html_coffee` (the exact compact coffee-shop prompt replayed on the
  pod) and `html_dashboard` (a new medium-difficulty Italian dashboard prompt)
  are available alongside the existing `html`, `code`, `code_mini`.
- Optional grading hook: if `scripts/functional_grade.py` is importable, HTML
  variants are graded through `grade_frontpage` and the L0-L3 level is written to
  the `l0l3` column (empty for non-HTML prompts, or when the module is absent).

Pure helpers (`median`, `order_jobs`, `alert_in_script`, `compute_median_summary`)
are covered by `tests/test_runner_upgrades.py` (CPU-only, no server/WSL/network).

## Manual test card

For manual UI tests, save a note with:

- Absolute date/time.
- DS4 commit and reap-loop commit if available.
- Full launcher/env overrides.
- Prompt name or prompt hash.
- Request settings: ctx, max tokens, temperature, stream/no-stream, no-think.
- User-observed values copied from UI/log, with timestamps.
- Whether routing trace, exchange observe, tiering log, top-token trace, or
  hidden-state trace were enabled.
- Any hypothesis separately labeled as hypothesis.

Manual notes without source artifacts can guide exploration, but they should not
be merged into benchmark tables as measured results.

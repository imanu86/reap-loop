# DS4 trace overhead A/B - HTML220

Date: 2026-07-09

Goal: test whether `DS4_SPEX_TRACE_ROUTING` plus
`DS4_SPEX_TRACE_ROUTING_WEIGHTS=1` is cheap enough to keep enabled while
benchmarking the local RTX 3060 setup.

Profile: `SOTA_LOCAL_3060_TIMED`, aligned with the local UI launcher:
PACE on, warmup 50, direct K23 after warmup, cache target 258 experts, ctx
6144, prefill chunk 512, SPEX hidden GPU load/score/prefetch off. Extra
diagnostic logs that can perturb timing were disabled unless they were the
variable under test: `DS4_PACE_EXCHANGE_OBSERVE=0`,
`DS4_EXPERT_TIERING_LOG=""`, `DS4_EXPERT_TIERING_LOG_IDS=0`.

Prompt: `html`, 78 prompt tokens, 220 measured completion tokens, one 64-token
warmup request per fresh server.

Important parser note: the first summary pass mixed warmup `first50_tps` and
prefetch counts into measured rows. The runner parser was fixed to report the
last completed request only, and the summaries below were regenerated from raw
logs.

## Results

| order | variant | wall_s | prompt_s | first50_tps | avg_tps | last_chunk_tps | measured prefetch | trace rows | repeat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| off -> on | trace off | 100.418 | 11.591 | 1.96 | 2.48 | 2.68 | 31.41 GiB / 2126 ms | 0 | 1 |
| off -> on | trace on | 95.923 | 11.124 | 2.15 | 2.59 | 2.74 | 31.41 GiB / 1844 ms | 11360 | 0 |
| on -> off | trace on | 166.296 | 69.537 | 1.51 | 2.28 | 2.73 | 6.07 GiB / 234 ms | 11360 | 0 |
| on -> off | trace off | 94.211 | 10.093 | 1.71 | 2.62 | 3.13 | 31.41 GiB / 2487 ms | 0 | 0 |

Artifacts:

- `runs/ds4/20260709_trace_ab_html220_v2/summary.csv`
- `runs/ds4/20260709_trace_ab_html220_rev/summary.csv`
- Trace CSVs for Scope replay:
  - `runs/ds4/20260709_trace_ab_html220_v2/html_sota_trace_on_r01/routing.csv`
  - `runs/ds4/20260709_trace_ab_html220_rev/html_sota_trace_on_r01/routing.csv`

## Evidence policy

All numeric statements in this note come from measured artifacts: DS4 stderr,
HTTP response usage, generated files, or direct file stats. Any non-measured
reading is labeled as a hypothesis and must not be used as a benchmark result.

Derived fields are allowed only when they are a direct calculation over
measured fields and the source fields are kept in the run directory.

## Interpretation

Measured facts:

- Trace-on wrote 11360 routing rows in both measured runs.
- Trace-on file size was about 1.15 MB in both measured runs.
- In the off-to-on order, measured wall time was 100.418s trace-off and
  95.923s trace-on.
- In the on-to-off order, measured wall time was 166.296s trace-on and
  94.211s trace-off.
- In the reversed trace-on run, measured `prompt_s` was 69.537s. In the other
  three measured runs it was 10.093s to 11.591s.
- In the reversed trace-on log, the line `SPEX routing trace enabled` appears
  after the measured `prompt done` line.

Hypothesis, not a result: the large reversed trace-on slowdown may be a cold or
state anomaly rather than direct routing CSV overhead, because the trace enable
log appears after prefill. This needs more measured A/B rows before becoming a
claim.

Operational decision:

- Keep trace off for benchmark rows used to compare policies.
- For visualization, run a diagnostic twin with trace on after selecting an
  interesting benchmark candidate.
- Treat trace-on timing as diagnostic until a larger alternating A/B directly
  measures overhead within noise.

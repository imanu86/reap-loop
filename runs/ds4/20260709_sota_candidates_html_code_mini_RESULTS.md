# DS4 SOTA candidate batch - HTML + code_mini

Date: 2026-07-09

Evidence rule: all numeric values below are copied from run artifacts in
`runs/ds4/20260709_sota_candidates_html_code_mini/`. No value is inferred from
memory.

Profile: `SOTA_LOCAL_3060_TIMED`, trace off, `DS4_PACE_EXCHANGE_OBSERVE=0`,
`DS4_EXPERT_TIERING_LOG=""`, `DS4_EXPERT_TIERING_LOG_IDS=0`, ctx 6144, prefill
chunk 512, one 64-token warmup per fresh server, 220 measured completion tokens.

## Measured results

| prompt | variant | wall_s | prompt_s | first50_tps | avg_tps | last_chunk_tps | prefetch | repeat | quality flags | measured PACE events |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| html | sota_trace_off | 119.215 | 37.247 | 1.95 | 2.69 | 3.24 | 31.41 GiB / 1847 ms | 0 | doctype=1 popup=1 | learned=1 descent=1 prebreath=0 breath=1 |
| code_mini | sota_trace_off | 124.667 | 42.038 | 1.98 | 2.66 | 3.21 | 6.07 GiB / 195 ms | 0 | n/a | learned=1 descent=1 prebreath=0 breath=0 |
| html | keep32_direct | 102.262 | 26.416 | 2.11 | 2.90 | 3.24 | 33.79 GiB / 1988 ms | 1 | doctype=1 popup=1 | learned=1 descent=1 prebreath=0 breath=1 |
| code_mini | keep32_direct | 103.803 | 26.644 | 2.22 | 2.85 | 3.21 | 8.45 GiB / 281 ms | 0 | n/a | learned=1 descent=1 prebreath=0 breath=0 |
| html | k32_pre_step4_every32 | 87.371 | 13.368 | 2.50 | 2.97 | 3.17 | 17.96 GiB / 629 ms | 1 | doctype=1 popup=1 | learned=1 descent=1 prebreath=1 breath=1 |
| code_mini | k32_pre_step4_every32 | 113.569 | 34.110 | 2.03 | 2.77 | 3.18 | 43.30 GiB / 2178 ms | 1 | n/a | learned=1 descent=1 prebreath=1 breath=1 |
| html | direct_k64 | 144.451 | 54.964 | 1.82 | 2.46 | 2.96 | 16.90 GiB / 685 ms | 0 | doctype=1 popup=1 | learned=1 descent=1 prebreath=0 breath=0 |
| code_mini | direct_k64 | 105.652 | 18.268 | 1.63 | 2.52 | 3.12 | 16.90 GiB / 601 ms | 0 | n/a | learned=1 descent=1 prebreath=0 breath=0 |

## Measured takeaways

- Fastest measured HTML wall time in this batch: `k32_pre_step4_every32`,
  87.371s, avg 2.97 t/s.
- Fastest measured code_mini wall time in this batch: `keep32_direct`,
  103.803s, avg 2.85 t/s.
- `direct_k64` did not win either prompt in this batch.
- Both K32 variants that improved speed on HTML had `repeat_flag=1` on at least
  one prompt, so they need quality replication before becoming defaults.
- `k32_pre_step4_every32` measured one prebreath and one breath in both prompts.

## Quality spot-check

The `repeat_flag=1` rows were checked against `content_measured.txt`.

- `html_keep32_direct_r01`: real degeneration. The generated CSS reaches an
  invalid `background:` value followed by a long run of zeros.
- `html_k32_pre_step4_every32_r01`: real degeneration. The generated CSS also
  reaches an invalid `background:` value followed by a long run of zeros.
- `code_mini_k32_pre_step4_every32_r01`: real degeneration. The answer starts
  coherently, then the hidden-state/race-condition section degrades into a long
  repeated dash-like sequence.

Operational consequence: K32 direct and K32 prebreath are speed candidates, but
they are not quality defaults from this batch.

## Planned but not run

The next planned batch was `20260709_k32_microsteps_html320`, with K32
micro-step variants (`step=2`, `step=1`) and 320 measured tokens. It did not
start: no `runner_console.log` existed for that directory after the launch
attempt, and WSL subsequently returned `WSL_E_DISTRO_NOT_FOUND` / timeouts from
this Codex session.

Do not treat the micro-step batch as measured evidence yet.

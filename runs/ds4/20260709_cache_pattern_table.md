# DS4 Cache Pattern Table - 2026-07-09

Generated from `summary.json`, `matrix_config.json`, `server_env.json`, `runner_manifest.json`, and `server.stderr.log` under `runs/ds4`.
No inferred benchmark values are used.

## Compact Measured Rows

| suite | variant | prompt | cache_experts | pace_target | K | breath | rotate | tok | prompt_s | avg t/s | last t/s | prefetch GiB/ms | tier miss/evict | repeat |
|---|---|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| breath_k96_vs_k0_html800 | breath_k96_return | html | 258 | 258 | 23[23..96] | off |  | 800 | 62.13 | 3.05 | 3.11 | 33.53/2342.00 |  | 1 |
| breath_k96_vs_k0_html800 | breath_k0_return | html | 258 | 258 | 23[23..96] | off |  | 800 | 10.53 | 2.23 | 3.12 | 14.26/4612.00 |  | 1 |
| breath_k96_vs_k0_return_k23_html800_stream | breath_k96_return_k23 | html | 258 | 258 | 23[23..96] | off |  | 800 | 84.11 | 2.13 | 2.62 | 37.48/2517.00 | 12/0 | 1 |
| breath_k96_vs_k0_return_k23_html800_stream | breath_k0_return_k23 | html | 258 | 258 | 23[23..96] | off |  | 800 | 12.96 | 1.99 | 2.52 | 0.00/0.00 | 12/0 | 1 |
| exchange_matrix_b1_html160 | direct_k23 | html | 258 |  | 23[23..96] | off |  | 160 | 10.60 | 1.70 | 2.23 | 31.41/457.00 |  | 1 |
| exchange_matrix_b1_html160 | keep32_direct | html | 258 |  | 32[32..96] | off |  | 160 | 10.06 | 2.50 | 2.46 | 8.45/112.00 |  | 0 |
| exchange_matrix_b1_html160 | pre_step8_every32 | html | 258 |  | 23[23..96] | off |  | 160 | 9.26 | 2.57 | 2.79 | 36.97/238.00 |  | 0 |
| exchange_matrix_b1_html160 | pre_step4_every32 | html | 258 |  | 23[23..96] | off |  | 160 | 9.21 | 2.63 | 2.78 | 21.39/138.00 |  | 0 |
| exchange_matrix_b1_html160 | pre_step1_every16 | html | 258 |  | 23[23..96] | off |  | 160 | 9.22 | 2.46 | 2.40 | 44.35/327.00 |  | 1 |
| exchange_matrix_b1_html160 | clock_breath64 | html | 258 |  | 23[23..96] | every96->K64 |  | 160 | 10.23 | 2.78 | 2.90 | 22.97/193.00 |  | 0 |
| exchange_matrix_b3_html320 | direct_k23 | html | 258 |  | 23[23..96] | off |  | 320 | 13.45 | 1.32 | 3.12 | 39.60/551.00 |  | 1 |
| exchange_matrix_b3_html320 | keep32_direct | html | 258 |  | 32[32..96] | off |  | 320 | 10.49 | 3.00 | 3.27 | 44.35/423.00 |  | 1 |
| exchange_matrix_b3_html320 | pre_step4_every32 | html | 258 |  | 23[23..96] | off |  | 320 | 9.93 | 2.69 | 2.85 | 55.97/439.00 |  | 1 |
| exchange_matrix_b3_html320 | clock_breath64 | html | 258 |  | 23[23..96] | every96->K64 |  | 320 | 9.41 | 2.97 | 3.14 | 45.94/285.00 |  | 0 |
| exchange_matrix_b4_keep_sweep_html320 | direct_k40 | html | 258 |  | 40[40..96] | off |  | 320 | 10.87 | 3.03 | 3.23 | 48.57/419.00 |  | 1 |
| exchange_matrix_b4_keep_sweep_html320 | direct_k48 | html | 258 |  | 48[48..96] | off |  | 320 | 9.29 | 2.86 | 3.06 | 38.01/359.00 |  | 0 |
| exchange_matrix_b4_keep_sweep_html320 | direct_k56 | html | 258 |  | 56[56..96] | off |  | 320 | 9.83 | 2.71 | 2.80 | 57.03/440.00 |  | 1 |
| exchange_matrix_b4_keep_sweep_html320 | k32_pre_step4_every32 | html | 258 |  | 32[32..96] | off |  | 320 | 8.92 | 2.93 | 2.98 | 114.07/475.00 |  | 0 |
| exchange_matrix_b4_keep_sweep_html320 | k32_pre_step8_every32 | html | 258 |  | 32[32..96] | off |  | 320 | 8.91 | 2.89 | 3.02 | 88.71/546.00 |  | 0 |
| exchange_matrix_b4_keep_sweep_html320 | direct_k64 | html | 258 |  | 64[64..96] | off |  | 320 | 9.14 | 3.03 | 3.08 | 61.25/422.00 |  | 0 |
| exchange_matrix_b5_html512_candidates | k32_pre_step4_every32 | html | 258 |  | 32[32..96] | off |  | 512 | 9.45 | 3.14 | 3.09 | 65.48/449.00 |  | 0 |
| exchange_matrix_b5_html512_candidates | direct_k64 | html | 258 |  | 64[64..96] | off |  | 512 | 9.37 | 2.93 | 3.00 | 61.25/439.00 |  | 1 |
| exchange_matrix_smoke | direct_k23 | html | 258 |  | 23[23..96] | off |  | 4 | 42.85 | 0.47 | 0.47 | 0.00/0.00 |  | 0 |
| pod_e7w4_static128 | pod_k23_static_no_breath_128 | html | 128 | 128 | 23[23..96] | off | 0/e32 | 800 | 21.26 | 3.06 | 3.32 | 0.00/0.00 | 222912/222656 | 1 |
| pod_e7w4_static64 | pod_k23_static_no_breath_64 | html | 64 | 64 | 23[23..96] | off | 0/e32 | 800 | 19.48 | 3.15 | 3.43 | 0.00/0.00 | 222912/222784 | 1 |
| pod_e7w4_static64_code_mini | pod_k23_static_no_breath_64 | code_mini | 64 | 64 | 23[23..96] | off | 0/e32 | 512 | 20.82 | 2.93 | 3.35 | 0.00/0.00 | 148608/148480 | 0 |
| pod_id63_rotate16_64 | pod_k23_rotate_every16_64 | html | 64 | 64 | 23[23..96] | off | 1/e16 | 800 | 19.15 | 2.71 | 2.88 | 279.22/4370.00 | 222912/222784 | 0 |
| pod_qo6_rotate32_64 | pod_k23_rotate_every32_64 | html | 64 | 64 | 23[23..96] | off | 1/e32 | 800 | 18.53 | 2.74 | 3.12 | 139.61/4131.00 | 222912/222784 | 0 |
| pod_qo6_rotate32_64_code_mini | pod_k23_rotate_every32_64 | code_mini | 64 | 64 | 23[23..96] | off | 1/e32 | 512 | 18.89 | 2.77 | 3.25 | 84.98/2300.00 | 148608/148480 | 0 |
| rotate_smoke | k23_rotate_every16 | code_mini | 258 | 258 | 23[23..96] | off | 1/e16 | 96 | 161.93 | 0.86 | 1.71 | 24.28/4325.00 | 6/0 | 0 |
| rotate_smoke_gatefix | k23_rotate_every16 | code_mini | 258 | 258 | 23[23..96] | off | 1/e16 | 72 | 66.69 | 0.56 | 1.31 | 12.14/3844.00 | 6/0 | 0 |
| sota_candidates_html_code_mini | sota_trace_off | html | 258 | 258 | 23[23..96] | off |  | 220 | 37.25 | 2.69 | 3.24 | 31.41/1847.00 | 12/0 | 0 |
| sota_candidates_html_code_mini | sota_trace_off | code_mini | 258 | 258 | 23[23..96] | off |  | 220 | 42.04 | 2.66 | 3.21 | 6.07/195.00 | 12/0 | 0 |
| sota_candidates_html_code_mini | keep32_direct | html | 258 | 258 | 32[32..96] | off |  | 220 | 26.42 | 2.90 | 3.24 | 33.79/1988.00 | 12/0 | 1 |
| sota_candidates_html_code_mini | keep32_direct | code_mini | 258 | 258 | 32[32..96] | off |  | 220 | 26.64 | 2.85 | 3.21 | 8.45/281.00 | 12/0 | 0 |
| sota_candidates_html_code_mini | k32_pre_step4_every32 | html | 258 | 258 | 32[32..96] | off |  | 220 | 13.37 | 2.97 | 3.17 | 17.96/629.00 | 12/0 | 1 |
| sota_candidates_html_code_mini | k32_pre_step4_every32 | code_mini | 258 | 258 | 32[32..96] | off |  | 220 | 34.11 | 2.77 | 3.18 | 43.30/2178.00 | 12/0 | 1 |
| sota_candidates_html_code_mini | direct_k64 | html | 258 | 258 | 64[64..96] | off |  | 220 | 54.96 | 2.46 | 2.96 | 16.90/685.00 | 12/0 | 0 |
| sota_candidates_html_code_mini | direct_k64 | code_mini | 258 | 258 | 64[64..96] | off |  | 220 | 18.27 | 2.52 | 3.12 | 16.90/601.00 | 12/0 | 0 |
| trace_ab_html220_rev | sota_trace_on | html | 258 | 258 | 23[23..96] | off |  | 220 | 69.54 | 2.28 | 2.73 | 6.07/234.00 | 12/0 | 0 |
| trace_ab_html220_rev | sota_trace_off | html | 258 | 258 | 23[23..96] | off |  | 220 | 10.09 | 2.62 | 3.13 | 31.41/2487.00 | 12/0 | 0 |
| trace_ab_html220_v2 | sota_trace_off | html | 258 | 258 | 23[23..96] | off |  | 220 | 11.59 | 2.48 | 2.68 | 31.41/2126.00 | 12/0 | 1 |
| trace_ab_html220_v2 | sota_trace_on | html | 258 | 258 | 23[23..96] | off |  | 220 | 11.12 | 2.59 | 2.74 | 31.41/1844.00 | 12/0 | 0 |

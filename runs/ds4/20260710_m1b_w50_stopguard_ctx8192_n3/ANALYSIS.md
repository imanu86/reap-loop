# M1 Analysis - 20260710_m1b_w50_stopguard_ctx8192_n3

Quality verdicts below are L0-L3 only. Loop/onset fields are diagnostics from n=3/window120 or repeated line blocks.

## Per Seed

| stem | stream_status | completion_tokens | stream_events | emits_html_close | l0l3 | client_stop_reason | retry_attempts | avg_tps | prompt_s | loop_onset_event_est | coherent_until_event_est | loop_kind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| html_m1_w50_k23_rotate32_cache256_r01 | ok | 4000 | 4003 | 0 | 2 |  | 0 | 2.47 | 150.783 | 685 | 684 | ngram3_window120_repeat3 |
| html_m1_w50_k23_rotate32_cache256_r02 | stream_failed | 4000 | 4003 | 0 | 0 |  | 1 | 2.33 | 199.455 | 848 | 847 | ngram3_window120_repeat3 |
| html_m1_w50_k23_rotate32_cache256_r03 | stream_failed |  | 1937 | 0 | 1 | client_stop_repeat_token_ngram | 1 |  |  | 328 | 327 | ngram3_window120_repeat3 |

## By Variant

| variant | runs | l0l3_values | html_close_runs | client_stop_runs | stream_failed_runs | avg_tps_median | prompt_s_median |
|---|---|---|---|---|---|---|---|
| m1_w50_k23_rotate32_cache256 | 3 | 2,0,1 | 0 | 1 | 2 | 2.4 | 175.119 |

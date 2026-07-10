# M1 Analysis - 20260710_m1a_w50_w100_ctx8192_n3

Quality verdicts below are L0-L3 only. Loop/onset fields are diagnostics from n=3/window120 or repeated line blocks.

## Per Seed

| stem | stream_status | completion_tokens | stream_events | emits_html_close | l0l3 | client_stop_reason | retry_attempts | avg_tps | prompt_s | loop_onset_event_est | coherent_until_event_est | loop_kind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| html_m1_w50_k23_rotate32_cache256_r01 | ok | 4000 | 4003 | 0 | 0 |  | 0 | 2.68 | 61.504 |  |  |  |
| html_m1_w100_k23_rotate32_cache256_r01 | ok | 4000 | 4003 | 0 | 0 |  | 0 | 2.56 | 130.704 | 617 | 616 | ngram3_window120_repeat3 |
| html_m1_w50_k23_rotate32_cache256_r02 | ok | 4000 | 4003 | 0 | 0 |  | 0 | 2.63 | 145.491 | 118 | 117 | ngram3_window120_repeat3 |
| html_m1_w100_k23_rotate32_cache256_r02 | ok | 4000 | 4000 | 0 | 1 |  | 0 | 2.5 | 124.043 | 469 | 468 | ngram3_window120_repeat3 |
| html_m1_w50_k23_rotate32_cache256_r03 | ok | 4000 | 4003 | 0 | 0 |  | 0 | 2.61 | 135.597 | 757 | 756 | ngram3_window120_repeat3 |
| html_m1_w100_k23_rotate32_cache256_r03 | stream_failed |  | 1799 | 0 | 0 |  | 0 |  |  | 586 | 585 | ngram3_window120_repeat3 |

## By Variant

| variant | runs | l0l3_values | html_close_runs | client_stop_runs | stream_failed_runs | avg_tps_median | prompt_s_median |
|---|---|---|---|---|---|---|---|
| m1_w50_k23_rotate32_cache256 | 3 | 0,0,0 | 0 | 0 | 0 | 2.63 | 135.597 |
| m1_w100_k23_rotate32_cache256 | 3 | 0,1,0 | 0 | 0 | 1 | 2.53 | 127.3735 |

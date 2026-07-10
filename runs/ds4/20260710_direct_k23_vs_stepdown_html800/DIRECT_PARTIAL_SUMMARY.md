# 2026-07-10 Direct K23 vs Stepdown Partial Summary

The runner was interrupted before writing `summary.csv`, but both direct-K23
variants completed 800 requested tokens. The stepdown variant in this directory
is partial and should not be used; the completed stepdown comparator is in
`runs/ds4/20260710_stepdown_relearn_only_html800`.

| Variant | Schedule | Source | Finish | Decode avg | Prefetch | Quality observation |
| --- | --- | --- | ---: | ---: | --- | --- |
| `local_k23_cache256` | W50 full/K0 -> K23 direct, unit-count warmup | `server.stderr.log`, `response_measured.json`, `content_measured.txt` | 344.096 s | 2.85 t/s | 6.07 GiB / 448 ms | Completed 800 tokens, but loops almost immediately in CSS reset; 51 repeats of `margin: 0, padding`; no `</html>`, `<form>`, or `<script>`. |
| `local_k23_weighted_warmup_cache256` | W50 full/K0 -> K23 direct, weighted warmup | same | 307.296 s | 2.87 t/s | 6.07 GiB / 449 ms | Completed 800 tokens, but loops in selector list; 215 `h6` occurrences; no `</html>`, `<form>`, or `<script>`. |
| `local_stepdown_64_to23_relearn_on_tighten_cache256` | W50 full/K0 -> K64 -> ... -> K23 | completed comparator in `20260710_stepdown_relearn_only_html800` | 438.872 s | 2.09 t/s | 75.78 GiB / 30291 ms | Slower; content is more HTML-like than both direct variants but still invalid/incomplete. |

Conclusion: in this cache256 reproduction, direct K23 is clearly faster, but it
does not reproduce better quality. The old good direct result likely depended on
different conditions such as cache size, exact prompt/run boundary, or the
previous session-learned weighted mask.

# Local DS4 K23 Cache Sweep - 2026-07-09

Measured on local RTX 3060 with the same DS4 binary. Each row uses one 64-token warmup before the measured request, `PACE_WARMUP=50`, fixed K23, no breath, no prebreath, no rotation, routing trace off.

| suite | prompt | cache | wall_s | prompt_s | first50 | avg t/s | last t/s | tier miss/evict | spex miss | copy ms/b | repeat |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| html320 | html | 258 | 174.781 | 41.531 | 1.39 | 2.4 | 2.89 | 12/0 | 528 | 13.853 | 1 |
| html320 | html | 128 | 116.253 | 10.484 | 2.73 | 3.03 | 3.12 | 99072/98816 | 99328 | 5.712 | 1 |
| html320 | html | 64 | 190.203 | 10.785 | 1.66 | 1.78 | 1.82 | 99072/98944 | 99200 | 9.398 | 1 |
| html320_rev | html | 64 | 182.369 | 9.828 | 1.8 | 1.85 | 2.87 | 99072/98944 | 99200 | 9.569 | 1 |
| html320_rev | html | 128 | 105.749 | 9.905 | 2.98 | 3.34 | 3.45 | 99072/98816 | 99328 | 5.409 | 1 |
| html320_rev | html | 258 | 109.055 | 10.093 | 2.84 | 3.23 | 3.33 | 12/0 | 528 | 5.545 | 1 |
| code256 | code_mini | 64 | 165.397 | 54.551 | 1.34 | 2.31 | 2.82 | 82560/82432 | 82688 | 22.283 | 0 |
| code256 | code_mini | 128 | 89.255 | 10.1 | 2.84 | 3.23 | 3.36 | 82560/82304 | 82816 | 5.839 | 0 |
| code256 | code_mini | 258 | 93.41 | 9.926 | 2.62 | 3.07 | 3.23 | 12/0 | 528 | 5.796 | 0 |
| code256_rev | code_mini | 258 | 218.162 | 79.131 | 0.78 | 1.84 | 2.82 | 12/0 | 528 | 23.911 | 0 |
| code256_rev | code_mini | 128 | 89.123 | 11.277 | 3.0 | 3.29 | 3.4 | 82560/82304 | 82816 | 6.904 | 0 |
| code256_rev | code_mini | 64 | 162.758 | 11.589 | 1.49 | 1.69 | 2.83 | 82560/82432 | 82688 | 10.265 | 0 |

## Readout

- `cache64` is consistently slower and shows large eviction pressure.
- `cache128` is the best local measured point in these warm K23 sweeps.
- `cache258` removes almost all tiering evictions. When it is not the first cold run it is close to `cache128`, but still slightly behind in these measurements.
- First-in-order cold effects are large: the cold `cache258` rows are not comparable to the warm rows without this caveat.
- HTML quality is still fragile at fixed K23: all HTML rows tripped `repeat_flag=1`. This cache sweep is a throughput/path test, not a quality win.

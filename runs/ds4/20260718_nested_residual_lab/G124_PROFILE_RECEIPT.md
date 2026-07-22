# G124 Nested Residual Profile Receipt

Status: `measured_n1_no_performance_verdict`.

This file is a local documentation pointer for the authoritative Windows
receipt:

`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_receipt.json`

Associated result:

`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_result.json`

Scope:

- `n=1` causal profile only.
- Exact output SHA-256:
  `fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
- Runtime reconstruction verification was enabled and timed separately.
- No SOTA, performance A/B or L0-L3 quality verdict is attached.

Measured timers:

| Timer | Calls | Seconds |
|---|---:|---:|
| CPU reconstruct | 869 | 14.8898841 |
| Residual pread | 869 | 1.9070234 |
| Reconstruction verify | 2,607 | 0.4990679 |
| Host copy | 995 | 0.3460242 |
| H2D enqueue | 995 | 0.0381582 |
| H2D sync | 254 | 0.1677010 |
| H2D enqueue + sync | n/a | 0.2058592 |
| Route-ready wait | 256 | 14.6092232 |

Interpretation:

The timers overlap across worker, route and transfer paths, so they cannot be
summed into wall-clock time. The measured next lever is G125 GPU-side exact
join, without any claimed outcome.

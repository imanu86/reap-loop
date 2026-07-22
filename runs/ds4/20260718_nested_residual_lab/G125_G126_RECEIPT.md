# G125/G126 Nested GPU Join Receipts

Status: `g125_structural_n1` and `g126_decode_positive_e2e_noisy_n3`.

G125 authoritative receipt:

`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g125_nested_gpu_join_safety_current_build_clean_20260718T182935501Z_eb824ebedb_receipt.json`

G125 receipt SHA-256:

`ae15a6d3d3bc35e75b46befd8d18d7886f571e47d93561d146ada3ccf20f58fb`

G126 authoritative aggregate:

`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g126_nested_gpu_join_ab_current_build_clean_v2_20260718T183831489Z_b4d8a5d111_result.json`

G126 aggregate SHA-256:

`c1f7849aa0da33c4b6d8279073954ba39f556fc485215e6d4058e809fbe9eaa6`

Shared scope:

- Full/open routing; no REAP/static/bake/request-scoped closed mask.
- Exact content SHA-256:
  `fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
- Four-layer nested residual fixture, not an all-layer catalog.
- No L0-L3 generation-quality verdict.
- No absolute SOTA claim.

G125:

- `n=1` structural safety only.
- GPU join requested and observed.
- GPU join calls: 1,261.
- Base H2D bytes: 4,958,453,760.
- Residual H2D bytes: 3,966,763,008.
- Native H2D bytes: 0.
- CPU reconstruct calls: 0.
- Verification calls: 3,783.
- Verification mismatches: 0.
- Failures: 0.

G126:

| Arm | Decode t/s runs | Decode mean | E2E mean | Interpretation |
|---|---:|---:|---:|---|
| CPU join | `1.15, 1.16, 1.15` | `1.153333` | `0.333719` | Baseline nested miss path |
| GPU join | `1.56, 1.58, 1.57` | `1.570000` | `0.499730` | Decode-positive GPU-side exact join |

G126 finding:

GPU join improves mean server decode throughput by 36.1272% over CPU join on
this exact batch. End-to-end mean is +49.7457%, but TTFT/request/WRAP timing is
too noisy for a general latency claim, so E2E remains batch-only/noisy.

Context:

- G123 nested candidate: 1.163333 t/s full/open, CPU reconstruct path.
- G123 full/open IQ2 control: 1.65 t/s.
- G73 historical closed/request-scoped short-workload result: 4.9867 t/s.
- G126 is a positive miss-path decode result, not absolute SOTA.

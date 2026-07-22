# G73 Canonical Config

This directory freezes the measured G73 configuration from existing receipts only. It is intentionally narrow: G73 is the native-Windows split-fused/static32 short-workload result on the exact 64-token cyberpunk HTML prompt.

Authoritative sources:

- `docs/EXPERIMENTS_LEDGER.md`, section `2026-07-16 Native Windows G73 Split-Fused Route A/B`.
- `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g73_split_fused_ab_result.json`.
- `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g73_split_fused_ab.ps1`.
- G103 ledger contract, which freezes G73 as `arena30 / 4551 slots, cache320, source-parts WRAP with waved reclaim, composed prefill mass tiering, static budget32, GPU-resident no-default-sync routes and SplitFused`.

Important boundary:

- G73 was `request_scoped_closed`, not full/open. The canonical JSON has `not_comparable_to_full_open=true`.
- The headline `4.986667 t/s` is valid only for the exact short workload measured here.
- G73 has no L0-L3 general quality verdict in this artifact.

Measured headline:

| Arm | n | Decode mean | Decode median | Route wait | Worker ms/job |
|---|---:|---:|---:|---:|---:|
| `static32` | 3 | `4.61 t/s` | `4.62 t/s` | `4.314333 ms` | `1.581 ms` |
| `static32_split_fused` | 3 | `4.986667 t/s` | `4.98 t/s` | `3.940667 ms` | `1.522667 ms` |

The measured split-fused effect was `+8.17%` mean decode and `+7.79%` median decode versus the contemporary static32 control, with transport intentionally unchanged: `5,820` VRAM hits, `10,692` RAM hits and `70.479492 GiB` RAM H2D in both arms.

Run the local consistency check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File runs\ds4\20260716_g73\test_g73_canonical_config.ps1
```

No GPU or DS4 runtime is used by the test.

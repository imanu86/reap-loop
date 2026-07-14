# Windows Native G27 REAP Mass WRAP Evidence

Imported from:

- Source worktree: `C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work`
- Committed report/code commit: `c4bb45de31d122a5f1e7b7e11bfbf18ec242dffe`
- Runtime base commit recorded by artifacts: `20b05194b631b56c7033d2211339e2d837e7d42d`
- Report: `reports/G27_REAP_MASS_WRAP_RESULTS.md`

The report is the committed G27 summary. The raw `g7_runs` files were present
in the source worktree and are imported here under the existing Windows-native
evidence layout:

- `results/`: final counterbalanced ON/OFF result JSON plus swap-ring/request-end safety result JSON.
- `stderr/`: final ON/OFF stderr plus swap-ring safety stderr and the negative full-slot failure stderr.
- `telemetry/`: runtime telemetry JSONL.
- `raw_outputs/`: captured outputs.
- `preflight/`: Windows memory preflight snapshots.

Interpretation: G27 mechanism and transactional rotation are accepted. The
final 16-token n=3 gate is not a long-run verdict; ON pays the bootstrap and
rotation cost inside a very short decode.

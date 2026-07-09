# DS4 K23 Raw-Router Rotation Pod Results - 2026-07-09

Evidence policy: all numeric values below are copied from `summary.csv` files
produced by `scripts/run_ds4_exchange_matrix.py`; qualitative notes reference
the saved `content_measured.txt` artifacts.

## Setup

- Hardware: RunPod RTX 4070 Ti 12GB pods.
- Binary under test: local `/root/ds4/ds4-server` built from the PACE rotation
  patch and uploaded to the pods as `/root/ds4/ds4-server`.
- Model: `/root/models/ds4-2bit.gguf`.
- Common policy: `DS4_PACE_WARMUP=50`, fixed `K23`, no prebreath, no breath
  (`DS4_PACE_DRIFT=1.0`, `DS4_PACE_BREATH_EVERY=999999`), streaming enabled.
- Rotation policy: `DS4_PACE_ROTATE=1`, raw `router_probs` EWMA
  (`DS4_PACE_ROTATE_DECAY=0.98`), mask refresh every 16 or 32 decode tokens.

## Results

| prompt | variant | completion | wall_s | avg_tps | last_tps | rotates | prefetch_ms | repeat_flag | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `html` | `pod_k23_static_no_breath_64` | 800 | 273.48 | 3.15 | 3.43 | 0 | 0.0 | 1 | `runs/ds4/20260709_pod_e7w4_static64` |
| `html` | `pod_k23_rotate_every16_64` | 800 | 313.942 | 2.71 | 2.88 | 46 | 4370.0 | 0 | `runs/ds4/20260709_pod_id63_rotate16_64` |
| `html` | `pod_k23_rotate_every32_64` | 800 | 310.452 | 2.74 | 3.12 | 23 | 4131.0 | 0 | `runs/ds4/20260709_pod_qo6_rotate32_64` |
| `html` | `pod_k23_static_no_breath_128` | 800 | 283.038 | 3.06 | 3.32 | 0 | 0.0 | 1 | `runs/ds4/20260709_pod_e7w4_static128` |
| `code_mini` | `pod_k23_static_no_breath_64` | 512 | 195.594 | 2.93 | 3.35 | 0 | 0.0 | 0 | `runs/ds4/20260709_pod_e7w4_static64_code_mini` |
| `code_mini` | `pod_k23_rotate_every32_64` | 512 | 203.463 | 2.77 | 3.25 | 14 | 2300.0 | 0 | `runs/ds4/20260709_pod_qo6_rotate32_64_code_mini` |

## Findings

- Cache target 258 failed on the RTX 4070 Ti pods before useful decode:
  logs show `cuda decode failed` after `q8_0` allocation/copy failures.
- Cache target 128 was not reliable for rotation; the static run completed,
  but rotate16/rotate32 at 128 aborted before first useful token. Treat 128 as
  unsafe on these pods for this build.
- Cache target 64 produced stable A/B results.
- On the HTML prompt at cache64, static K23 was faster (`3.15 t/s`) but
  degenerated into repeated CSS/body blocks (`repeat_flag=1`). Rotate16 and
  rotate32 were slower (`2.71` and `2.74 t/s`) but did not trigger the repeat
  detector (`repeat_flag=0`).
- Rotate32 is the better first candidate than rotate16: same qualitative flag,
  slightly better throughput, half the rotation count, and lower prefetch work.
- On the code_mini prompt, both static64 and rotate32_64 had `repeat_flag=0`;
  rotate32_64 cost throughput (`2.77 t/s` vs `2.93 t/s`) without a measured
  quality win on that prompt.

## Operational Notes

- The rotation patch is useful enough to keep behind a flag, but it is not a
  default speed path yet.
- Next tests should focus on a quality-triggered rotation mode rather than
  unconditional periodic rotation: rotate when raw-router out-of-mask mass or
  n-gram risk rises, not every N tokens forever.
- `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.25` is parsed as invalid by the
  current CUDA code because it expects an integer GB value. This should be
  cleaned up separately so launcher intent matches runtime behavior.

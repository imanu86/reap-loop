# Smoke 0026 (demand-admission DS4_PACE_ADMIT) — POD2 2026-07-10

Pod: RunPod community RTX 3090 `o0gd30ojfacz96` (machine 3n0g1lqe8wy5), cu129 image, sm_86.
Binary: livetree chain base(771a39a8) + 0020+0021+0026+0027, `make cuda CUDA_ARCH=sm_86`, 0 warnings.
Model: ds4-2bit.gguf (sha256 efc7ed60…, verified). Config: PACE W50 K23 static (breath=999999, rotate=0),
coffee prompt, `--ssd-streaming --ssd-streaming-cache-experts 1024`, greedy temp0, -n 256.
Admit forced low: DS4_PACE_ADMIT_H=0.3 PERSIST=1.

## VERDICT: PASS

- admit_on: 42 `"ev":"admit"` events (fields layer/expert/evicted/cusum/keep present), across 24 distinct layers, K held at 23.
- Page-in of only the entered expert: 31 `"ev":"rotate_delta"` events, each `entered:1 exited:1 bytes:7077888` = exactly 6.75 MiB per entered expert (no full WRAP).
- Off-switch control (DS4_PACE_ADMIT=0): 0 admit events, 0 rotate_delta events, no "PACE ADMIT on" line — clean.
- Generation coherent (valid coffee-shop HTML), exit 0.

Files: `{admit_on,admit_off}_pace_events.jsonl`, `_diag.log`, `admit_on_gen.txt`.

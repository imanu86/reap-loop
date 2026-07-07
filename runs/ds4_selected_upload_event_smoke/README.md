# DS4 selected upload event smoke

Date: 2026-07-05

Purpose: Stage 1a CUDA event plumbing for ds4 selected expert uploads. This is
not the final SPEX predictor benchmark; it only verifies that selected expert
H2D uploads can be ordered by a CUDA event instead of blocking every copy.

Patch under test:

- `patches/ds4/0001-spex-stage0-cuda-stats.patch`
- `patches/ds4/0002-spex-selected-upload-event.patch`

Local runtime:

- ds4 checkout: WSL `/root/ds4`
- ds4 local commit: `d7bbee6 chore(spex): add selected upload event`
- GPU: RTX 3060 12GB, `sm_86`
- Model: `/root/models/ds4-2bit.gguf`
- Command shape: `./ds4 -m /root/models/ds4-2bit.gguf --cuda --ssd-streaming --ssd-streaming-cold -c 1024 --nothink -n 1 -p ciao`

Verification:

- Local build: `make cuda CUDA_ARCH=sm_86`
- Local regression: `make cuda-regression`
- RunPod clean build: RTX 3090, CUDA 12.8, stock `antirez/ds4` at `80ebbc3`, apply `0001` then `0002`, `make cuda CUDA_ARCH=sm_86`, `make cuda-regression`

Smoke result:

| Mode | Prefill | Generation | selected sync calls | selected sync ms |
|---|---:|---:|---:|---:|
| `DS4_SELECTED_UPLOAD_EVENT=1` | 0.59 t/s | 0.72 t/s | 216 | 25.738 |
| `DS4_SELECTED_UPLOAD_EVENT=0` | 0.47 t/s | 0.63 t/s | 9288 | 1491.401 |

Interpretation: the flag is moving the blocking sync out of the selected expert
copy path. Remaining sync calls are expected from resident/cache paths left in
the default synchronous mode. This is a plumbing checkpoint, not a speed claim
for real chat workloads.

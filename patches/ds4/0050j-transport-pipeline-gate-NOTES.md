# 0050j transport pipeline gate integration

Status: build-ready 0050 preflight patch; GPU runtime gate not yet executed.
This is not the dynamic-arena 0051 implementation.

`0050j-transport-pipeline-gate.patch` combines the compatible behavior from:

- `0032-async-pipeline-rebased.patch`;
- `0047-no-whole-mmap-register.patch`;
- `0048-prefill-overlap-s1.patch`.

It applies to the current 0050 source snapshot with these pre-patch SHA256
values:

| File | SHA256 |
|---|---|
| `ds4.c` | `c4e88c9eb30d78921bcd04ad2e68de6883a2aa4aa8239cdc37ea56e790ba5a97` |
| `ds4_cuda.cu` | `dd48371b2b6d46cc956bb5c6f2ce2d22943334ddd7a7245bb4eea6265bc1b0bf` |
| `ds4_gpu.h` | `944ce6c50564f61eabb668719ad83f04006124bc9f8394a755f79eb0d7ab51a5` |

Apply from the DS4 source root:

```bash
git apply --check /path/to/0050j-transport-pipeline-gate.patch
git apply /path/to/0050j-transport-pipeline-gate.patch
```

## Resolved integration conflicts

1. The 0048 persistent cursor and WAR guard use the runtime staging depth from
   0047 instead of hard-coded `% 4` and `chunk_idx >= 4`. The default remains
   four buffers; `DS4_CUDA_SELECTED_STAGE_DEPTH` accepts 1 through 16.
2. `DS4_ASYNC_PIPELINE=0` and an unset variable are both OFF in the SPEX
   next-layer prefetch gate, matching the CUDA upload-event gate.
3. The upload-event gate is enabled by the existing event aliases,
   `DS4_ASYNC_PIPELINE`, or prefill S1. No new environment variable was added.
4. The cross-call staging cursor is selected from the actual per-copy deferred
   completion mode. This protects both 0032 decode uploads and 0048 prefill
   uploads, rather than only the prefill env path.
5. Release resets the persistent staging cursor and WAR state.

## Gate controls

All variables below already belong to 0032, 0047, or 0048.

| Purpose | Environment |
|---|---|
| Entire integration OFF | unset the variables below, or set boolean gates to `0` |
| Async event plus compute-only layer barrier | `DS4_ASYNC_PIPELINE=1` |
| Isolate event/barrier from L+1 prediction | `DS4_ASYNC_PIPELINE=1 DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER=1` |
| Prefill S1 deferred uploads | `DS4_CUDA_PREFILL_DEFER_UPLOAD_SYNC=1` |
| Staging ring depth | `DS4_CUDA_SELECTED_STAGE_DEPTH=1..16` (default `4`) |
| Skip whole-mmap registration | `DS4_CUDA_NO_WHOLE_MMAP_REGISTER=1` |
| Speculative prefill readahead S2 | `DS4_REAP_PREFILL_READAHEAD=1` |

`DS4_REAP_PREFILL_READAHEAD` remains available for controlled negative
reproduction but is OFF by default. Do not enable it in the main gate: its
previous WSL campaign regressed both cold and warm prefill.

The prediction-isolation arm works because
`DS4_SPEX_DISABLE_PREFETCH_NEXT_LAYER` is checked after all positive prefetch
gates. It suppresses the L+1 seed while leaving the CUDA upload event and the
compute-stream-only layer barrier active.

## Build verification

The patch was composed on a frozen temporary clone. `/root/ds4-v2-work` was
read only and was not modified.

- `git diff --check`: pass.
- `make cpu -j8`: pass. Existing 0050 CPU warnings remain; no new build error.
- `make cuda CUDA_ARCH=sm_86 -j8`: pass, including all five linked binaries.
- No binary was executed and no GPU runtime or model load was performed.

The next step is the preregistered A/B/C gate with frozen binaries, balanced
order, at least three measured repetitions per arm, exactness checks, and L0-L3
grading. A compile pass is not evidence of runtime exactness or speed.

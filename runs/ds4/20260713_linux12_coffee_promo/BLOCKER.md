# Coffee-on-native-12GB (promotion+pruning) — BLOCKED by RunPod community-fleet CUDA bug

Date: 2026-07-13. Goal: run the coffee-mask ARM (PROMOTION + PRUNING) on a **native
12 GB** Linux GPU to get the gate number for the Windows port. **No result obtained** —
every 12 GB card RunPod offers is on a community fleet that currently cannot create a
CUDA context. Full evidence + a ready-to-run harness are committed here so the next
attempt is fast.

## What worked (FASE 1 — provisioning)
- Grabbed a 12 GB card on the **first** `runpodctl create pod` attempt each time.
- Only 12 GB cards in RunPod's catalog: **RTX 3080 Ti** ($0.18/hr) and **RTX 4070 Ti**
  ($0.19/hr), **community only** (secureCloud=false for both — no 12 GB in secure cloud).
- Three pods created, all cleaned up (no orphans; protected `99xyqm02gke4xg` never touched):
  - `yaesjwcu84i68o` (3080 Ti): container **never started** (>1h, runtime null). Terminated.
  - `halprnoxs09rqi` (3080 Ti): container up, **cuInit 999**. Terminated.
  - `rbzp4q1mjflmby` (3080 Ti): container up, **cuInit 999**. Terminated.
- 4070 Ti was out of stock on the retry.

## The blocker (FASE 2 — CUDA)
On both nodes whose container started, `nvidia-smi` fully works (enumerates the RTX 3080 Ti
with UUID, driver 580.95.05) but **CUDA context creation fails**:
```
cuInit(0)            -> 999  (CUDA_ERROR_UNKNOWN)
cuDeviceGetCount     -> count 0
torch.zeros(1).cuda()-> RuntimeError: CUDA unknown error
env NVIDIA_VISIBLE_DEVICES=void          <-- abnormal; normal pods have "all" or a UUID
```
`NVIDIA_VISIBLE_DEVICES=void` means the nvidia-container-runtime did **not** set up GPU
access at container creation; RunPod bind-mounted the device nodes manually, leaving a
half-configured GPU that nvidia-smi can read but CUDA cannot initialize. This is a
host/platform issue, not fixable from inside the container.

### In-container fixes attempted (all failed to change cuInit 999)
- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`: all present
  (verified/created) — no effect.
- `nvidia-modprobe`: not installable on the image.
- Driver userspace libs: **false lead.** `strings libcuda.so.1 | grep '^5..\.'` reports
  "565.40" even on the *official* 580.95.05 `.run`'s libcuda — that string is an embedded
  constant, NOT the driver version. There is no real 565-vs-580 mismatch; installing the
  580.95.05 userspace libcuda did not change cuInit 999.

### Root cause (best assessment)
Systemic on RunPod's **community 3080 Ti** fleet today: pods launched with
`NVIDIA_VISIBLE_DEVICES=void` → broken CUDA context creation. Reproduced identically on two
independent nodes. No secure-cloud 12 GB exists to escape to.

## Binary/model/CUDA compatibility — CONFIRMED GOOD (not the problem)
- Binary: local WSL build `ds4-server` (git `da0b3f6`, **sm_86** = same arch as 3080 Ti),
  Ubuntu 24.04 / glibc 2.39 / CUDA 12.8 — identical to the pod image. `ldd` resolves all
  CUDA libs on the pod (LIBOK). Binary `strings` contains every lever needed.
- Model `ds4-2bit.gguf` pulled from R2, exact size 86720111488 (sha `efc7ed60…`).

## Ready-to-run harness (in `harness/`) — reuse when a working 12 GB node is available
- `poll_provision.sh` — poll-loop grabber (community→secure, price ceiling, anti-orphan).
- `pod_deploy.sh` — on-pod: install rclone (official, fast), configure R2 from `/root/cf.txt`
  (no secret echo), pull model to `/root/models` (container disk; **use `containerDiskSize
  120 --volumeSize 0`** — a 150 GB persistent volume caused the first node to hang forever),
  `--ignore-checksum` to skip rclone's slow 86 GB post-copy read-back.
- `run_coffee_arm.sh` — one arm, **12 GiB envelope**, `CACHE_PROFILE=1`, cache-experts 400.
  - PRUNING = `DS4_REAP_MASK_FILE=<coffee mask>` + static zero-copy window
    `DS4_CUDA_STREAM_FROM_RAM_MASKED=<mask>` + `..._BUDGET_GB=12`.
  - PROMOTION (promo arms) = `DS4_REAP_PIN_BY_MASS=1` + `DS4_PACE_LIVEMASK=1`
    + `DS4_PACE_LIVEMASK_RATING_ONLY=1` + `DS4_PACE_LIVEMASK_PRESSURE=1`
    (per-token windowed-demand mass drives dynamic residency; coffee mask stays authoritative).
  - STATIC-PIN arm = same coffee mask + static zero-copy window, promotion OFF.
- `run_all_arms.sh` — runs arms `k83_promo k83_staticpin k65_promo k100_promo`, writes SUMMARY.
- `make_coffee_requests.py` — coffee landing-page prompt, temp0, warm=48 meas=1600.
- `parse_arm.py` — **enhanced**: extracts `copy_ms`/`sync_ms` from the binary's SPEX line and
  computes the **copy_ms/noncopy_ms per-token floor** (the "pavimento") + hit%, MiB/tok,
  zero-copy cover, warm decode t/s.
- `functional_grade.py` — L0–L3 frontpage grader for the rendered HTML.

## Transfer path that works (no direct TCP port on community; SSH proxy only)
`runpodctl send <bundle>` locally → `runpodctl receive <code>` on pod (croc relay). SSH proxy
requires `-tt` (PTY) and only runs commands cleanly via stdin heredoc; scp does not work.

## Recommended next steps
1. Retry later — community fleet CUDA state may recover; add the **CUDA gate** (torch
   `cuInit`) right after SSH, **before** the 86 GB download (already the plan; saves ~15 min).
2. If a native-12 GB number stays impossible, get the coffee number on a **working larger
   card capped to 12 GiB** (`..._BUDGET_GB=12`) — this still yields hit%, the copy/noncopy
   floor, and the promotion-vs-static-pin delta (all largely bandwidth-independent); only the
   H2D GiB/s would be non-native and must be labeled as such.

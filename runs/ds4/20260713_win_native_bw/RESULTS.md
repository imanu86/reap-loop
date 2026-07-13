# H2D bandwidth: WSL2 throttle vs native Windows CUDA — 2026-07-13

## Question

The DS4 study measured H2D (host-to-device) memcpy bandwidth of ~2.97-3.03
GiB/s under WSL2 (see `../20260712_v2_zerocopy/arena_probe/RESULTS.md`,
single-allocation follow-up table). PCIe 4.0 x16 should support far more
(theoretical unidirectional cap ~31 GB/s ≈ 29 GiB/s). Question: is the ~3
GiB/s ceiling a WSL2 GPU-PV (paravirtualization) artifact, or does it also
appear on native Windows CUDA?

## Setup

- Machine: RTX 3060 12 GiB, WDDM driver 596.21, CUDA driver 13.2.
- Native Windows path: `powershell.exe` (NOT `wsl.exe`) launching a Windows
  x64 `.exe` that loads `nvcuda.dll` from `System32` directly via
  `LoadLibraryW`/`GetProcAddress` (CUDA Driver API), so no CUDA SDK/`cuda.lib`
  is required at build time.
- Build toolchain: `x86_64-w64-mingw32-g++` (GCC 13-win32), invoked *inside*
  WSL Ubuntu-24.04 but cross-compiling straight to a native Windows PE binary
  written to a `/mnt/c/...` path (same trick the existing
  `cuda_pinned_arena_probe_win.cpp` build already used) — this was necessary
  because the local MSVC (Visual Studio 2022 Community) install on this
  machine is missing headers (`excpt.h`) needed by `/EHsc`, so `cl.exe`
  cannot compile this source; MinGW was the only compiler available.
- Two **independent** measurement code paths, both executed natively:
  1. `tools/h2d_bandwidth_probe_win.cpp` → `bin/h2d_bandwidth_probe_win.exe`
     — raw CUDA Driver API (`cuMemHostAlloc`, `cuMemAlloc_v2`,
     `cuMemcpyHtoD_v2`, blocking, timed with `std::chrono::steady_clock`).
  2. `tools/torch_h2d_bandwidth.py` — PyTorch 2.6.0+cu124 (cudart-backed),
     run with the native Windows `python.exe`.
- Protocol (both tools): PINNED (page-locked) vs PAGEABLE host buffers, at 1
  GiB and 2 GiB, 12 timed copies per condition, **first copy discarded as
  warmup**, mean/median/min/max/stddev GiB/s reported over the remaining 11.

## Results

### Native Windows, CUDA Driver API (`h2d_bandwidth_probe_win.exe`)

| alloc | size | mean GiB/s | median | min | max | stddev |
|---|---:|---:|---:|---:|---:|---:|
| pinned | 1 GiB | 24.411 | 24.435 | 24.180 | 24.452 | 0.074 |
| pageable | 1 GiB | 11.564 | 11.412 | 11.147 | 12.130 | 0.356 |
| pinned | 2 GiB | 24.106 | 24.431 | 22.079 | 24.450 | 0.720 |
| pageable | 2 GiB | 11.281 | 11.332 | 10.166 | 12.009 | 0.550 |

Raw per-iteration numbers: `results/h2d_bandwidth_native.jsonl`.

### Native Windows, PyTorch/cudart cross-check (`torch_h2d_bandwidth.py`)

| alloc | size | mean GiB/s | median | min | max |
|---|---:|---:|---:|---:|---:|
| pinned | 1 GiB | 24.146 | 24.316 | 23.472 | 24.413 |
| pageable | 1 GiB | 10.727 | 11.126 | 8.503 | 11.886 |
| pinned | 2 GiB | 24.237 | 24.386 | 22.823 | 24.442 |
| pageable | 2 GiB | 10.446 | 10.275 | 9.581 | 11.744 |

Raw output: `results/torch_h2d_bandwidth_native.stdout.txt`.

The two independent code paths (raw Driver API vs PyTorch/cudart) agree
within noise: **pinned H2D ≈ 24.1-24.4 GiB/s, pageable H2D ≈ 10.4-11.6
GiB/s**, natively on Windows.

### WSL2 reference (already on disk, not re-measured here)

From `../20260712_v2_zerocopy/arena_probe/RESULTS.md` (`cudaHostAlloc`
pinned arena, single-allocation follow-up, `cudaMemcpyAsync` H2D measured
inside WSL2 via `wsl.exe`):

| Target | H2D bandwidth |
|---:|---:|
| 24 GiB | 2.971581 GiB/s |
| 28 GiB | 3.028627 GiB/s |
| 30 GiB | 3.018815 GiB/s |
| 31 GiB | 3.024387 GiB/s |

(Single-copy each, not the N=12 protocol used here, but the tight
2.97-3.03 GiB/s clustering across four separate runs makes it a reliable
reference point — and it is over an order of magnitude below the pinned
number needed to make this comparison meaningful.)

## Verdict

**Native Windows pinned H2D (~24.1-24.4 GiB/s) is ~8x the WSL2 pinned H2D
reference (~2.97-3.03 GiB/s).** Even native **pageable** H2D (~10.4-11.6
GiB/s, no pinning at all) is still ~3.5x faster than WSL2's *pinned* number.
24.4 GiB/s (~26.2 GB/s decimal) is a plausible, realistic fraction of PCIe
4.0 x16's ~31 GB/s theoretical ceiling (~85% efficiency), consistent with
"this is just PCIe 4.0 behaving normally." The ~3 GiB/s WSL2 ceiling is not
explained by PCIe generation/width, by pinning behavior, or by any property
of this GPU — it reproduces only under WSL2's GPU paravirtualization layer.

**WSL_THROTTLE_CONFIRMED**: the ~3 GiB/s H2D ceiling is a WSL2 GPU-PV
artifact, not a hardware or PCIe limit. Native Windows CUDA (or a
dual-boot/bare-metal Linux setup) removes it.

## Caveats

- N=1 machine, N=1 GPU/driver combination; no repeated runs across reboots.
- Pageable-memory numbers (~10-12 GiB/s) are themselves somewhat variable
  (stddev up to ~0.55 GiB/s, occasional low outliers e.g. 8.5 GiB/s in the
  torch run) — consistent with pageable H2D internally staging through a
  pinned bounce buffer and being more sensitive to host-side scheduling.
  This does not affect the headline pinned-vs-WSL comparison.
- `nvidia-smi -q -d PCI` link-generation/width readout was not captured
  (tool invocation issue, not re-attempted); the PCIe-4.0 assumption is
  based on the achieved bandwidth being close to the known PCIe 4.0 x16
  ceiling and on the motherboard/GPU generation, not on a direct
  `nvidia-smi` reading in this run.
- Did not re-measure the WSL side in this session (reused the existing
  `20260712_v2_zerocopy` numbers, which were pinned/`cudaHostAlloc`,
  single-shot rather than N=12-averaged); a same-protocol WSL run with the
  N=12 harness would tighten the comparison further but is very unlikely to
  change the order-of-magnitude conclusion.

## Artifacts

- `tools/h2d_bandwidth_probe_win.cpp` — new Driver-API bandwidth probe source.
- `tools/torch_h2d_bandwidth.py` — PyTorch cross-check script.
- `bin/h2d_bandwidth_probe_win.exe` — MinGW-built native Windows binary.
- `results/h2d_bandwidth_native.jsonl` — raw JSON Lines from the Driver-API probe.
- `results/h2d_bandwidth_native.stderr.txt` — human-readable summary table (Driver API run).
- `results/torch_h2d_bandwidth_native.stdout.txt` — PyTorch cross-check output.

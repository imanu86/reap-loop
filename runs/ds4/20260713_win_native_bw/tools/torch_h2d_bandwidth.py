"""Independent cross-check of H2D bandwidth on native Windows using PyTorch's
CUDA runtime (cudart), as a second code path separate from the raw Driver API
probe (h2d_bandwidth_probe_win.cpp). Run with native Windows python.exe, NOT
inside WSL.
"""
import time
import torch

assert torch.cuda.is_available(), "CUDA not available in this torch build"
print(f"torch={torch.__version__} cuda={torch.version.cuda} device={torch.cuda.get_device_name(0)}")

GiB = 1024 ** 3
N = 12
WARMUP = 1

def bench(size_gib, pinned):
    n_elem = int(size_gib * GiB)
    host = torch.empty(n_elem, dtype=torch.uint8, pin_memory=pinned)
    host.fill_(0xA5 if pinned else 0x5A)
    dev = torch.empty(n_elem, dtype=torch.uint8, device="cuda")
    times = []
    for i in range(N):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dev.copy_(host, non_blocking=pinned)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = times[WARMUP:]
    gib_s = [size_gib / t for t in times]
    gib_s.sort()
    mean = sum(gib_s) / len(gib_s)
    median = gib_s[len(gib_s)//2]
    return mean, median, min(gib_s), max(gib_s)

for size_gib in (1, 2):
    for pinned in (True, False):
        mean, median, lo, hi = bench(size_gib, pinned)
        label = "pinned" if pinned else "pageable"
        print(f"{label:10s} {size_gib:>4} GiB  mean={mean:7.3f}  median={median:7.3f}  min={lo:7.3f}  max={hi:7.3f} GiB/s")

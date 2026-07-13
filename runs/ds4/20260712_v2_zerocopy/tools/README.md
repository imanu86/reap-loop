# CUDA pinned arena probe

Standalone WSL2 probe for comparing retained pinned allocations with a prior
`cudaHostRegister(mmap(...))` ceiling. It writes JSON Lines to stdout. The safe
default is a 0.25 GiB target; 50 GiB is attempted only when passed explicitly.

Build for Ampere SM 8.6 (the build does not run the probe):

```bash
nvcc -std=c++17 -O2 -arch=sm_86 -o cuda_pinned_arena_probe cuda_pinned_arena_probe.cu
```

Retain separate 5 GiB blocks until 50 GiB or the first allocation failure:

```bash
./cuda_pinned_arena_probe --mode blocks --api hostalloc --target-gib 50 --step-gib 5
```

Try one 50 GiB allocation through the `cudaMallocHost` alias:

```bash
./cuda_pinned_arena_probe --mode single --api mallochost --target-gib 50
```

The internal `MemAvailable` floor defaults to 8 GiB and can be changed
explicitly with `--reserve-gib`. Keep 8 GiB for WSL probes; the native Windows
comparison may use `--reserve-gib 2` to match the intended host budget.

From Windows, use the guarded runner for large probes. It refuses to start while
`ds4-server` is running, compiles the probe into WSL `/tmp`, samples host RAM,
and terminates only the probe PID if Windows crosses the requested floor:

```powershell
.\run_pinned_arena_probe.ps1 -TargetGiB 50 -StepGiB 1 `
  -MinWindowsAvailableGiB 2
```

The runner writes the JSONL probe output, stderr, exact shell command, and a
timestamped Windows available-memory CSV under the existing `arena_probe`
artifact directory.

`blocks` keeps every successful block alive until the post-allocation copy test
and final cleanup. `single` makes exactly one arena allocation; `--step-gib` is
reported but ignored in that mode. By default each retained block writes one
byte in every 4 KiB host page, plus its last page. This materializes the whole
allocation without a full multi-GiB memset. `--touch-stride-mib` can make the
touch sparser for diagnostics, but sparse runs must say so in their metadata.

Before every host allocation, the probe reads `/proc/meminfo` and requires enough
`MemAvailable` for that allocation while retaining an 8 GiB reserve. Arena checks
also reserve space for the later pinned staging buffer. If `MemAvailable` is
unreadable or the reserve cannot be maintained, it aborts the next host
allocation. It changes no limits, overcommit settings, mappings, or system state.

After the arena attempt, a small pinned staging buffer (16 MiB by default) and an
equal-size device buffer are allocated. A deterministic pattern is written into
the retained arena itself, copied directly from that arena with `cudaMemcpyAsync`
H2D, timed with CUDA events, copied back into staging, and checked byte-for-byte
and by FNV-1a checksum. All acquired CUDA resources are released in reverse
dependency order, including after partial failures.

# Patch 0050 controlled A/B protocol

## Scope

This protocol closes patch 0050 on one runtime-supported treatment:

| Treatment | Masked host registration | Budget |
|---|---:|---:|
| `off24` | disabled | inert |
| `on24` | enabled | 24 GiB |

It measures whether the existing static masked mmap registration removes work
from the expert-fetch path without changing deterministic output. It does not
validate selection quality, the dynamic arena proposed for patch 0051, or a
28 GiB runtime arena.

The earlier campaign `20260713_0050_ab01` is an infrastructure-negative run,
not a performance result. Its first `off24` gate crossed the 8 GiB Windows
memory floor with WSL capped at 62 GB, and the watchdog exposed shutdown bugs in
the first harness revision. WSL is now capped at 57 GB. The failed campaign is
preserved and must not be included in any throughput summary.

## Why 28 GiB is excluded

A standalone `cudaHostAlloc` probe passed at 28 GiB, but patch 0050 registers
its static mmap window before all mandatory session/staging allocations. Only
24 GiB has been exercised in the DS4 runtime. Testing 28 GiB as if it were an
equivalent 0050 treatment could starve later allocations.

Patch 0051 allocates its arena after mandatory pinned resources and specifies
an explicit `28 -> 24 -> disabled` fallback. The 28 GiB experiment belongs
there.

## Safety contract

The harness is inert unless invoked with `--execute` and a new campaign ID.

Before every server launch it:

1. refuses any existing WSL or native Windows `ds4-server`;
2. requires the fixed TCP port to be free;
3. acquires `/tmp/ds4-gpu.lock` non-blocking for the complete campaign;
4. verifies WSL and Windows available-memory floors synchronously;
5. rechecks binary, sources, mask, requests, measurement helper and patch-chain
   hashes;
6. creates a new immutable run directory.

Every server PID is recorded with its Linux start ticks and executable path.
Termination is allowed only while all three still match. The watchdog sends
TERM on a memory breach, waits five seconds, and escalates only that verified
PID to KILL. Such a run is invalid. Name-wide termination is prohibited.

WSL and Windows memory floors are both 8 GiB. WSL RAM, swap and GPU telemetry
are sampled throughout each request. Windows RAM is mandatory when PowerShell
CIM is available; native GPU columns are best-effort and may be blank without
invalidating the RAM monitor.

## Fixed runtime contract

- DS4 worktree: `/root/ds4-v2-work`
- Binary: `/root/ds4-v2-work/ds4-server`
- Model: `/root/models/ds4-2bit.gguf`
- Selection fixture: `mask60_self.txt`
- Cache: 400 experts
- Context: 2048
- Prefill chunk: 512
- Sampling: temperature 0, thinking disabled
- Exactness request: 60 tokens
- Same-server discarded warm-up: 80 tokens
- Measured request: 450 tokens
- Steady-state calculation: first 40 stream deltas excluded
- Patch-0050 budget: 24 GiB

The static mask is a deterministic mechanism fixture. One prior bake60 render
was graded L2 at `n=1`; this protocol makes no quality claim from it.

## Execution

First run the non-starting preflight:

```bash
runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh --preflight
```

After any WSL/memory change, run only the functional gate:

```bash
runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --execute --campaign-id 20260713_0050_gate02 --gate-only
```

A performance campaign uses at least four counterbalanced rounds:

```bash
runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --execute --campaign-id 20260713_0050_ab02 --rounds 4
```

### Gate-only order

`off24 -> on24`. Both responses must have identical UTF-8 content bytes.

### Measured order

Each treatment starts a fresh server. The runner sends the discarded warm-up
and then the measured request to that same server, preserving its VRAM expert
cache and process-local state.

- odd rounds: `off24 -> on24`
- even rounds: `on24 -> off24`

The reverse order balances first/second position. No page-cache dropping is
performed. Each lifecycle is followed by a fixed cooldown.

## Mandatory attribution gates

OFF must show:

- no masked range registration;
- no masked budget activation line;
- no active direct-DMA line;
- a diagnostic with `why=final`;
- `covered=0` and `dma_ok=0`.

ON must show:

- the explicit 24.00 GiB budget line;
- every requested range registered;
- the direct-DMA path active;
- a diagnostic with `why=final`;
- `dma_ok > 0`;
- `dma_failed=miss_empty=miss_base=0`.

For every valid run, the harness must own a graceful TERM shutdown and observe
server wait status 0. A periodic diagnostic cannot substitute for the final
record. Any abort, forced kill, missing final record, request failure or source
drift invalidates the campaign.

## Stream validity

The measurement helper fails closed unless it observes:

- SSE `[DONE]`;
- completion usage;
- a finish reason;
- enough deltas after the excluded ramp;
- finite positive TTFT and throughput metrics.

It stores exact UTF-8 content bytes for each measured run. At temperature zero,
all measured OFF/ON outputs in a campaign must match. The manifest must contain
exactly `2 * rounds` valid rows; missing rows are a campaign failure.

## Interpretation

Use paired OFF/ON differences within each round and report every row. Four
rounds provide four paired observations; medians and dispersion are descriptive
performance evidence, not a quality verdict.

Never infer quality from throughput, completion length or repeat diagnostics.
Promotion of a selection policy still requires the separate `n>=3`, L0-L3
grading protocol.


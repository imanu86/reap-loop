# Patch 0050 controlled A/B protocol

## Scope

This protocol closes the post-P1 functional gate for patch 0050 and measures
its static masked mmap registration at 24 versus 28 GiB. It does not evaluate
the proposed dynamic arena, selection quality, or patch 0051.

The controlled treatments are:

| Treatment | Active zero-copy path | Budget env |
|---|---:|---:|
| `off24` | no | 24 GiB, inert while OFF |
| `on24` | yes | 24 GiB |
| `off28` | no | 28 GiB, inert while OFF |
| `on28` | yes | 28 GiB |

`DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1` is set in every treatment. The OFF
controls also receive the paired budget value, but
`DS4_CUDA_STREAM_FROM_RAM_MASKED` is absent, so registration is disabled. This
keeps diagnostic overhead symmetric and lets the logs prove that OFF is truly
OFF. The intended and actual server environments are captured for every run.

## Safety contract

The harness is inert unless invoked with `--execute` and a new campaign ID. It
must be run from WSL; do not launch it from this document.

Before any server starts, the harness:

1. refuses an existing WSL or native Windows `ds4-server`;
2. refuses a listener on its fixed port;
3. acquires `/tmp/ds4-gpu.lock` with non-blocking `flock` and holds it for the
   complete campaign;
4. refuses an existing campaign directory;
5. verifies the binary, model, mask, measurement helper, and required tools.

All termination is by the PID created by the harness and recorded in the run
directory. PID start time and executable path must still match before a signal
is sent. Name-wide process termination is prohibited. A TERM timeout may
escalate only that verified PID to KILL; such a run is invalid and the campaign
stops. No cache dropping, file deletion, output-directory reuse, or automatic
cleanup of evidence is permitted.

WSL `MemAvailable` has an 8 GiB floor. Where PowerShell/CIM is available,
Windows available RAM has an 8 GiB floor as well. A floor breach or an
unexpected loss of the available Windows monitor terminates only the recorded
server PID and invalidates the run. WSL RAM, swap, GPU memory/utilization/
temperature/power, Windows RAM, and Windows GPU data where `nvidia-smi.exe` is
available are sampled into separate CSV files.

## Fixed runtime contract

- DS4 worktree: `/root/ds4-v2-work`
- Binary: `/root/ds4-v2-work/ds4-server`
- Model: `/root/models/ds4-2bit.gguf`
- Selection mask: `mask60_self.txt`
- Cache: 400 experts
- Context: 2048
- Prefill chunk: 512
- Request temperature: 0
- Warm-up budget: 80 tokens
- Exactness budget: 60 tokens
- Measured budget: 450 tokens, with the first 40 stream deltas excluded from
  steady-state decode throughput

The server is launched with `env -i`; only the captured whitelist and explicit
DS4 values are present. Every run stores the intended env and argv, actual
`/proc/<pid>` env and argv, request and hash, PID identity, server/client logs,
telemetry, exit status, stop reason, and arm-attribution result.

At campaign start the harness also stores:

- reap-loop and DS4 Git HEAD/status/log;
- the complete binary DS4 worktree diff against its base commit, including the
  unexported 0050 WIP;
- source snapshots and hashes of the modified DS4 files and server binary;
- hashes of the repository patch archive and available `build_0050*.log` files;
- a full SHA-256 of the GGUF plus its stat identity;
- mask, request, harness, protocol, hardware, GPU, and runner-env snapshots.

The binary/source/mask/helper hashes, DS4 base and complete tracked patch-chain
hashes, and model stat identity are rechecked before every server launch and at
campaign end. The full GGUF SHA-256 is repeated after the measured phase. Any
change aborts the campaign.

## Execution order

First run the non-starting check:

```bash
runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh --preflight
```

Execution requires an explicit, never-before-used ID:

```bash
runs/ds4/20260712_v2_zerocopy/scripts/run_0050_controlled_ab.sh \
  --execute --campaign-id 20260713_0050_ab01 --rounds 4
```

The command above is documentation only. This preparation task must not run it.

Each treatment uses a fresh server lifecycle. The phases are strictly
separated:

1. **Bit-exact gate:** `off24, on24, off28, on28` at temperature 0 and 60
   tokens. Extracted content bytes from all four responses must match exactly.
2. **Discarded warm-up block:** `on28, off28, on24, off24` at 80 tokens. These
   directories remain evidence but never enter timing results.
3. **Measured rounds:** odd rounds use
   `off24, on24, off28, on28`; even rounds use
   `on28, off28, on24, off24`.

The odd/even schedule alternates OFF/ON within every budget pair, reverses both
arm and budget ordering, and gives each treatment one observation per round.
Rounds must be even; four is the default. A fixed 20-second gap follows every
server lifecycle. The harness never drops or rewrites the page cache.

## Mandatory gates

The measured phase is refused unless all four exactness outputs have identical
content bytes. The JSON envelope is not compared because request IDs and timing
metadata may legitimately differ.

Every run must also pass runtime attribution parsed from 0050 diagnostics:

- OFF: no masked ranges registered, no active DMA-path line, zero covered
  queries, and zero successful direct-DMA copies;
- ON: all requested ranges registered, active DMA-path line present,
  `dma_ok > 0`, and `dma_failed=miss_empty=miss_base=0`.

`miss_range` and `miss_before` may be nonzero under a partial 24/28 GiB window;
they are measurements to retain, not automatic failures. Any failed request,
telemetry abort, missing final diagnostic, source drift, forced server kill, or
attribution failure stops the campaign while preserving every artifact.

## Interpretation

Only directories under the measured phase may contribute throughput rows.
`performance_manifest.tsv` is a direct extraction of client timing artifacts;
it is not a verdict. Compare paired OFF/ON observations within the same budget
and round, report all rows, and use medians with dispersion. Treat the repeated
OFF labels as drift controls: the budget variable is deliberately inactive
while the path is OFF.

Do not infer generation quality from repetition detectors, repeat flags,
completion length, throughput, or this temperature-0 campaign. Repeat-related
signals may be retained as diagnostics only. A quality claim requires the
separate n>=3 grading protocol with L0-L3 evaluation; it is outside this A/B.

An anomalously slow run may be stopped manually only through its recorded and
identity-checked `server.pid` procedure. A slow start that ramps is not itself
a failure. Never convert an early speed observation into a quality judgment.

# P0 SSD rotator diagnosis — frozen pre-implementation report

Frozen: 2026-07-26 CEST

Scope: read-only diagnosis of the A1 GraphTensorDevice OFF control
`20260726_151009_a1_m0_p0_trace-off_packed-off_publish-off-hostsel-off-routeio-off-graphdev-off`.
No DS4 run, build, or source edit preceded this receipt.

## Byte-exact A1 AFTER baseline

```text
ds4.c       54F5E14378B71CC039D1467B5B9D1D7F7512FDC460B1ADC49F24B963C5EC8466
ds4_cuda.cu 53B152FA67FC8155F1391C27CB807001E7B3DBB39BBD5F863CC2338BFB684182
ds4_gpu.h   625F477594EF2C835047A8FDD2512B8715ADA82C72C71DE1B4466EFBE76D436C
ds4_metal.m 81612C353EFCBDEFF58619BD38A9CD9E72382D6A9FDF2AC709BA27FCCD2D6F5D
ds4_server.exe
            2B11AA8160191F9AC44802451CDEC2EF36334C52B06F0ECF5FCC2068990562BA
```

## Finding

The A1 runner gate is incomplete, not too strict. Exact serving passed, but the
declared `DS4_G73_OPEN=1` SSD rotator closed in a real failed state, making the
overall A1 control invalid as a reusable baseline.

The apparent `successes=4 failures=2 dropped=1` represents four successful
jobs and one failed wave/job. The failed job is counted once by the worker and
again by `cuda_q1_0_ssd_wrap_fail_and_release_all_locked`; the second path also
counts the same job as dropped.

The failed job is wave 5, with `ranges_read=0`, `bytes_read=0`, and about
7.8–7.9 ms service time in both the latest A1 OFF control and the historical
`20260726_123028` control. It occurs during early decode, not during native
shutdown/drain. No generation mismatch, ownership failure, stale publication,
safe leak, forbidden serving fallback, or output mismatch was recorded.

The strongest source-backed cause is deadline-domain aliasing: advisory G73
rotation inherits the exact-serving request deadline and then takes
`min(queued_at + 50 ms, request.absolute_deadline)`. Rotation is scheduled only
after the exact transient route has already been served and published, so the
inherited serving budget may have only about 8 ms left. The current
`partial_or_pread` label collapses pre-read deadline expiry, timeout,
cancellation, pread error, EOF, and short read, so exact I/O status and
layer/expert cannot be recovered from the existing log.

## Required correction

1. A G73 advisory I/O failure must remain job-local: never publish the failed
   job, retain SSD-cold serving state, safely release writer claim, reservation,
   and ring slot, increment its failure exactly once, and permit later
   rotations.
2. Ownership, generation, commit/publication, transport-stall, and non-G73
   failures remain terminal and fail-closed.
3. Advisory failure/degraded telemetry must remain visible.
4. The runner must fail the overall experiment gate when rotator health is
   degraded, while reporting exact-serving correctness separately.
5. No output, route choice, serving fallback, or short-suffix snapshot
   semantics may change.

## Runtime status

```text
READY_FOR_IMPLEMENTATION_SLOT=yes
DS4_RUNTIME_AUTHORIZED=no
```

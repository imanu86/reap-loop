# P0.1 SSD rotator service-deadline receipt

Frozen: 2026-07-26 CEST

Status: implementation/build/ValidateOnly complete. No DS4 runtime was
started. The experiment ledger was not modified.

## Authorized baseline

- A1 report/ledger commit:
  `460f4cf73002d81d5fb6c868265c75fd68b18ad1`
- P0 source baseline `ds4_cuda.cu`:
  `313A8CCC8BCA6B8EC08E3205621B18D8BA7E58624EF20D568063EC86784B90AD`
- P0 runner baseline:
  `E4809471B3AC68A4633D1FB77D4855ABF922FE79B9EAF97305C90A61131A006D`

## Runtime finding closed by P0.1

Run `20260726_155302_a1_m0_p0_trace-off_packed-off_publish-off-hostsel-off-routeio-off-graphdev-off`
contained 45 advisory failures in 31 waves. Every event had:

```text
range=0 expected=2162688 actual=0 errno=138
service_seconds=0.0000002..0.0000012
```

This is the pre-read `remaining_ms == 0` branch, not an SSD read or partial
read. The P0 deadline used `queued_at + 0.05s`, so queue residence/background
worker scheduling consumed the entire I/O budget before service acquisition.

## Implemented contract

File: `work/wt-hot-reserve/ds4_cuda.cu`

- Lines 29677-29700: service timing helper measures queue age independently,
  keeps the configured budget unchanged, and derives the service deadline from
  `io_started`.
- Lines 29901-29912: every selected job acquires its own timing immediately
  before its read.
- G73 ignores the serving request deadline and uses
  `io_started + min(rotator_timeout, transient_timeout)`.
- Non-G73 still uses the minimum of that service deadline and
  `request.absolute_deadline`; all non-G73/structural errors remain fail-closed.
- Lines 29962 and 30005: event and wave telemetry expose
  `queue_age_ms` and `service_budget_ms`; the event also exposes the exact I/O
  status.
- The configured `DS4_G133_ROTATOR_IO_TIMEOUT_S=0.05` was not changed.
- P0 job-local advisory release, single counting, later-submit behavior, and
  explicit `degraded` summary remain intact.

The benchmark runner remains fail-closed:

```text
result must equal complete
advisory_failures must equal 0
advisory_degraded must equal 0
overall gate requires rotatorGate.gate_pass
```

## Tests

PASS:

1. Old `queued_at` cannot expire a newly acquired G73 service budget.
2. Each job in a multi-job wave receives an independent service deadline.
3. A real service timeout remains advisory/job-local and explicitly degraded.
4. `max_age_calls` and admission backpressure remain separate from I/O timing.
5. Non-G73 remains request-bounded and terminal.
6. No double count; clean release; failed route remains SSD-cold; a later
   rotation can publish.
7. Runner rejects any advisory/degraded rotator.

Executed validation:

```text
test_p0_ssd_rotator_advisory_static.ps1: PASS
test_p0_1_ssd_rotator_service_deadline_static.ps1: PASS
test_p0_ssd_rotator_runner_gate.ps1: PASS
test_a1_graph_tensor_device_runner_static.ps1: PASS
PowerShell AST parse: PASS
patch diff check: PASS
patch reverse-apply check: PASS
```

## Build

```text
directory=C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\build-p0-1-ssd-service-deadline
generator=Ninja
build_type=Release
cuda_architectures=80;86;89;90
target=ds4_server
result=PASS
```

## ValidateOnly

```text
OFF=20260726_161138_a1_m0_p0_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off
OFF_RESULT=PASS manifest_count=47 declared_parameter_count=52 hashes=PASS server_started=no

ON=20260726_161139_a1_m0_p0_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-on
ON_RESULT=PASS manifest_count=47 declared_parameter_count=52 hashes=PASS server_started=no
```

## Frozen hashes

```text
ds4.c
54F5E14378B71CC039D1467B5B9D1D7F7512FDC460B1ADC49F24B963C5EC8466
ds4_cuda.cu
92DB43E8050115255617F665CF830814ADBC09347B14849B5C81BE003FAD4ECC
ds4_gpu.h
625F477594EF2C835047A8FDD2512B8715ADA82C72C71DE1B4466EFBE76D436C
ds4_metal.m
81612C353EFCBDEFF58619BD38A9CD9E72382D6A9FDF2AC709BA27FCCD2D6F5D
runner
2AD9843B87E7B917BB424951A189C09F1B56A02D728E26BA15DBA73540352F7C
binary
C3EAC30683C5636905BF3C483CC0EC391A89A2772F7826D2B8D1BDCF8D595AFC
patch
8A60F80DC0121FEA0D270E229F0245BD05EADA7E91114AB6F00B46D7C03637FE
```

## Cleanup and authority

```text
DS4_PROCESS_COUNT=0
NSIGHT_PROCESS_COUNT=0
PORT8000_LISTENER_COUNT=0
LEDGER_STATUS_LINES=0
DS4_RUNTIME_EXECUTED=no
LEDGER_UPDATED=no
READY_FOR_RUNTIME_SLOT=yes
```

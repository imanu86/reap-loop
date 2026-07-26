# DS4 P1 — sampled CUDA/host trace v2 implementation receipt

Date: 2026-07-26  
Status: implementation/build slot complete; runtime slot not authorized or used.

## Verdict

P1 is implemented on the authorized frozen P0.1 source and is ready for a
separate runtime measurement slot.

- `DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY`, `POSITIONS`, and `LAYERS` are strict
  opt-in selectors.
- Trace OFF returns before record allocation, timing-event creation, QPC setup,
  or NVTX-domain creation.
- The timing pool is preallocated once and contains exactly 178
  `cudaEventDefault` events: 2 token + 43 × 4 layer/upload + 4
  embedding/output.
- Dependency events remain distinct raw CUDA events created with
  `cudaEventDisableTiming`.
- The P1 trace block adds no `cudaEventSynchronize`,
  `cudaDeviceSynchronize`, or `cudaStreamSynchronize`.
- `cudaEventElapsedTime` is read only by
  `ds4_gpu_decode_trace_sync_complete()`, called after the pre-existing device
  synchronization returns.
- Trace records are preallocated and buffered; formatting and `stderr` output
  happen only during request-end/cleanup flush.
- The reset-sensitive `route_transient_read_bytes` delta was replaced by
  monotonic trace-owned atomic `pread` counters. End snapshots are checked
  `end >= begin`; an invalid snapshot fails `io_counter_valid` instead of
  wrapping an unsigned value.
- GraphTensor remains OFF and the P0.1 transport semantics remain OFF.

## Authorized frozen provenance

| Item | Frozen SHA-256 |
|---|---|
| `ds4.c` | `54F5E14378B71CC039D1467B5B9D1D7F7512FDC460B1ADC49F24B963C5EC8466` |
| `ds4_cuda.cu` | `92DB43E8050115255617F665CF830814ADBC09347B14849B5C81BE003FAD4ECC` |
| original runner | `2AD9843B87E7B917BB424951A189C09F1B56A02D728E26BA15DBA73540352F7C` |
| frozen executable | `C3EAC30683C5636905BF3C483CC0EC391A89A2772F7826D2B8D1BDCF8D595AFC` |

Repository Git HEAD is
`f64cc89baf5ba890c11317b8a0283e25ace85eae`. The source worktree already
contained the authorized P0/P0.1 changes relative to that HEAD. Therefore the
delivered source patch is deliberately named `AGGREGATE_FROM_f64cc89`: it is
an exact, apply-checkable representation of the current authorized source plus
P1, not a claim that every hunk relative to Git HEAD was authored by P1.

## Final source, runner, and binary hashes

| Item | Bytes | Final SHA-256 |
|---|---:|---|
| `ds4.c` | 953,408 | `43B2ECE184916AA2731A299BE420A5133248BE24B3F17BCFB396D20B97CA1A2A` |
| `ds4_cuda.cu` | 2,033,668 | `6D6BE460609C74EEEA60B43F6D62DB812F722D4A270B45797EB6A17909446257` |
| `ds4_gpu.h` | 47,141 | `52AC84C2F10AE1D1EDB3032AFCFBDB1AAE000CDF148D98CCDAC42AA0739223AB` |
| `test_p1_decode_trace_static.ps1` | 7,184 | `8EB893F2F5031E41BA3C942CE28C7DCC7B435B92F6DFBEE4A3841BF91CEBCC0B` |
| `test_p1_decode_trace_parser_static.ps1` | 1,964 | `AB1ED001CC0F691F69E5C41013EDC5E0F69020AFFC732A93CA8B4BDB8DEFB9A4` |
| `test_p2c_nsys_capture_static.ps1` | 3,130 | `CC1EAF3498C1CF3710FD9DB2664A4C8A8C90493DC9337C47C77E5C33AD7962EB` |
| original runner, unchanged | 47,222 | `2AD9843B87E7B917BB424951A189C09F1B56A02D728E26BA15DBA73540352F7C` |
| P1 runner copy | 47,540 | `E8A746195C5BD09C6C034E1F3EBC8C482146600843F56DDD2AADCEF9BBD200E5` |
| isolated `ds4_server.exe` | 12,445,696 | `185E46A057CE6BFFA7605638DDC5A3994FAFF3DBC48A3F7FF9B0B6617F3B9717` |

Build root:
`C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\build-p1-sampled-trace-v2`

Final build check: `ninja: no work to do`, exit 0.

## Exact implementation call sites

Line numbers refer to the final source hashes above.

| File:line | Contract |
|---|---|
| `ds4_cuda.cu:26469-26472` | fixed pool cardinality `2 + 43*4 + 4 = 178` |
| `ds4_cuda.cu:26544` | strict range parser used by positions/layers |
| `ds4_cuda.cu:26665-26868` | buffered `ds4.cuda.decode_trace.v2` token/layer serialization and flush |
| `ds4_cuda.cu:26926-26930` | timing-event wrapper created with `cudaEventDefault` |
| `ds4_cuda.cu:26933-27014` | OFF gate, selector parsing, selftest gate, and one/two-token Nsight restrictions |
| `ds4_cuda.cu:27037-27070` | all 178 timing events preallocated before decode |
| `ds4_cuda.cu:27163-27235` | sampled token start and NVTX capture/token/layer range arming |
| `ds4_cuda.cu:27387-27442` | elapsed reads after the existing end sync |
| `ds4_cuda.cu:27518-27569` | token close, guarded monotonic `pread` deltas, buffered record append |
| `ds4_cuda.cu:27649-27678` | KV batch/span/H2D/wait-event counters |
| `ds4_cuda.cu:27712-27869` | resolver, queue, worker, pread, QD, H2D/D2H, drain, publish helpers |
| `ds4_cuda.cu:28102`, `36487` | actual chunked/legacy decode-route `pread` accounting |
| `ds4_cuda.cu:34759`, `34804` | route H2D count/bytes/API enqueue accounting |
| `ds4_cuda.cu:35211`, `36633` | upload-drain wait accounting around existing waits |
| `ds4_cuda.cu:35427` | QD submit/completion/max-inflight accounting |
| `ds4_cuda.cu:36184`, `36313`, `36689` | worker queue/service and sample attribution |
| `ds4_cuda.cu:41171`, `41381` | resolver submission and stream/worker-ready waits |
| `ds4_cuda.cu:42104-42107` | blocking classification D2H count/bytes/API time |
| `ds4_cuda.cu:27989` | `kv_populated` arm point |
| `ds4_cuda.cu:27994-27999` | request-end NVTX close and buffered trace flush |
| `ds4.c:13093-13109` | QPC around existing split sync and output submission |
| `ds4.c:15013` | post-existing-sync elapsed-resolution hook |
| `ds4.c:15018-15022` | logits D2H QPC/count/bytes |
| `ds4_gpu.h:62-70` | backend-neutral public hook contract |

Existing ordering/dependency events remain `cudaEventDisableTiming`, including
model staging (`ds4_cuda.cu:5208`, `5234`), async tensor readback
(`14584-14586`), KV staged-ring events (`19641-19658`), route consumer
completion (`34377`), upload slots (`37373-37469`), and G132 bridge events
(`44074-44078`).

## Telemetry schema

Every emitted line is prefixed `ds4: [decode-trace-v2]` and contains JSON with
`schema="ds4.cuda.decode_trace.v2"`.

Token records contain:

- identity/validity: `record`, `sample_id`, `position`, `success`,
  `qpc_frequency_hz`, `io_counter_valid`, `record_complete`,
  `dropped_records`, `worker_attribution_errors`;
- host versus GPU: `host_wall_ms`, `wall_tps`,
  `gpu_default_window_ms`, `gpu_default_tps`, `gpu_timing_valid`;
- embedding/output/sync/logits: host copy, H2D API/count/bytes, embedding GPU
  window, output submit/GPU window, split-sync wait/count, end-sync wait/count,
  logits D2H API/count/bytes;
- aggregate I/O and transfers: `pread_wait_ms/count/bytes`,
  H2D and D2H API/count/bytes;
- KV: batch/span counts, H2D count/bytes/enqueue time, stream wait-event count;
- event contract: `timing_event_count=178`, `event_sync_count=0`.

Layer records contain:

- `sample_id`, `position`, `layer`, `sample_id_match`;
- `host_encode_ms` and `gpu_default_window_ms`;
- resolver submit, prior-mailbox queue, worker queue/service, resolver-stream
  wait, worker-ready wait and poll count;
- `pread` wait/count/bytes and QD submit/completion/max-inflight;
- H2D/D2H API/count/bytes;
- upload GPU window and upload-drain wait/count;
- publish kernel/route counts;
- KV batch/span/H2D/enqueue/wait-event counters.

The semantic separation is explicit:

| Quantity | Meaning |
|---|---|
| `*_gpu_*_ms` | CUDA event device window; GPU work/window, not host latency |
| `*_api_ms`, `*_submit_ms`, `*_enqueue_ms` | host time spent in CUDA API/submission |
| `pread_wait_ms` | storage/I/O wait |
| `*_stream_wait_ms`, `worker_ready_wait_ms`, `*_sync_wait_ms` | host/stream dependency wait |
| inferred supply gap | residual idle interval only; never relabeled as GPU compute |

Throughput is always reported as tokens per second (`*_tps`), never seconds per
token.

## NVTX/Nsight Systems capture contract

Prepared but not executed:

- `DS4_CUDA_NSYS_CAPTURE=1`;
- requires `DS4_CUDA_DECODE_TRACE_POSITIONS` with exactly one position or two
  contiguous positions;
- forbids `DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY`;
- the domain is `ds4.cuda`;
- the outer range is `ds4.decode.capture`;
- the capture range opens only after `kv_populated=1`;
- future runner flags are
  `--capture-range=nvtx`,
  `--nvtx-capture=ds4.decode.capture@ds4.cuda`,
  `--capture-range-end=stop`, `--trace=cuda,nvtx`,
  `--sample=none`, and `--cpuctxsw=none`.

This prevents full-run tracing and CPU sampling. There are no
`cudaProfilerStart/Stop` calls in P1.

## Validation evidence

Static tests, all PASS:

1. `test_p1_decode_trace_parser_static.ps1`
2. `test_p1_decode_trace_static.ps1`
3. `test_p2c_nsys_capture_static.ps1`
4. `test_p0_1_ssd_rotator_service_deadline_static.ps1`
5. `test_p0_route_mailbox_static.ps1`
6. `test_p0_short_suffix_snapshot_static.ps1`
7. `test_p0_ssd_rotator_advisory_static.ps1`
8. `test_a1_graph_tensor_device_static.ps1`

The parser selftest is compiled and opt-in through
`DS4_CUDA_DECODE_TRACE_SELFTEST=1`. It was not dynamically invoked because
that would require starting DS4; its valid/invalid cases and invocation
contract are covered by the parser static test.

Targeted P1 block scan:

```text
cudaEventSynchronize=0
cudaDeviceSynchronize=0
cudaStreamSynchronize=0
cudaProfilerStart=0
cudaProfilerStop=0
cudaEventElapsedTime=1
```

`git diff --check` on the three modified source files: PASS (only the existing
Windows LF/CRLF advisory).

ValidateOnly trace OFF:

```text
validation=PASS
manifest_count=47
overlay_count=9
graph_tensor_device=Off
hashes=PASS
server_started=no
```

ValidateOnly sampled (`POSITIONS=13,141`, `LAYERS=0,4,20,42`): same PASS,
47+9 manifest/overlay, hashes PASS, `server_started=no`.

One broad legacy test, `test_g73_open_static.ps1`, still reports the
pre-existing fixture error
`preset missing: $env:DS4_CUDA_MOE_SPLIT_FUSED = '0'` in
`tests/g73_open.env.ps1`. Neither file is in the P1 change set. The focused
P0.1/G73 service-deadline preservation test passes.

## Future runtime acceptance gate

Runtime remains pending and requires separate authorization.

1. exact output match for both turns;
2. baseline manifest exactly 47/47 plus the declared trace overlay;
3. GraphTensorDevice OFF and all P0.1 transport-choice toggles unchanged;
4. compare throughput in tokens/second;
5. trace OFF versus frozen baseline overhead target `<=1%`, hard reject `>2%`;
6. sampled trace versus trace OFF must also respect the hard `<=2%` gate;
7. a throughput delta `<10%` is classified as variance, not promotion evidence;
8. `record_complete=1`, `io_counter_valid=1`, `dropped_records=0`,
   `worker_attribution_errors=0`, `timing_event_count=178`,
   `event_sync_count=0`;
9. `pread_bytes` must be plausible and must never wrap;
10. postflight DS4/Nsight/runner process counts zero, port 8000 free.

The future Nsight capture is diagnostic only: one or two post-KV decode tokens,
no full-run trace and no CPU sampling.

## Risk/payoff

| Item | Risk | Payoff / mitigation |
|---|---|---|
| Trace OFF default | Low | direct early return before allocations/QPC/NVTX; runtime overhead still measured by gate |
| Sampled QPC/counters | Low–medium | finally separates host submission, I/O, queue/wait, and device windows; bounded by `<=1%` target / `<=2%` hard gate |
| CUDA timing windows | Medium interpretation risk | fields are named `window`, not GPU busy union; Nsight validates overlap/idle structure |
| Cross-thread attribution | Low correctness risk | immutable sample ID/position on requests; mismatch increments a hard-gate counter |
| Buffered records | Low | fixed 8192-record allocation only when enabled; drops are counted and fail acceptance |
| NVTX capture | Low runtime risk | strict one/two contiguous token parser and post-KV arm; not executed in this slot |
| P0.1 semantics | Low | preservation tests and ValidateOnly pass; GraphTensor remains OFF |

Expected payoff is high: the next authorized run can quantify GPU compute
window, CUDA API/submission time, storage wait, dependency/stream wait, and
true residual supply/idle gap without adding synchronization to the hot path.

## Delivered artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `DS4_P1_IMPLEMENTATION_AGGREGATE_FROM_f64cc89.patch` | 254,510 | `BA06405995EE897C2A80978E2A6DC3A14B37881E95C6C46B86468472D9145575` |
| `DS4_P1_RUNNER_DELTA.patch` | 48,996 | `42411C20485ACE5B16D75DABF54588A496E6C2DBC1DE7DFFCFB87679B1CCE416` |
| `DS4_P1_SAMPLED_CUDA_HOST_TRACE_DESIGN.md` | 29,013 | pre-implementation audit/design retained |

Both delivered patch files pass `git apply --check --reverse` against the
current source/runner state. The runner patch adds the P1 copy; it does not
modify the original runner.

```text
READY_FOR_RUNTIME_SLOT=YES
RUNTIME_AUTHORIZATION=NOT_GRANTED
DS4_RUNS=0
NSIGHT_RUNS=0
LEDGER_TOUCHED=NO
GRAPH_TENSOR_BASELINE=OFF
```

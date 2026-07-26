# DS4 P1 - PAGEABLE=10 runtime A/B receipt

Date: 2026-07-26  
Classification: `VALID_AB_CORRECTNESS_PASS_PERFORMANCE_HARD_FAIL`  
Runtime arms: one trace OFF, then one sampled; no extra replicas; no Nsight.

## Verdict

Both PAGEABLE=10 arms pass output exactness, P0.1 lifecycle, rotator,
allocation, memory-pressure, trace engagement, parser, and native-shutdown
checks. The performance contract fails:

```text
output_exact=PASS
lifecycle=PASS
rotator=PASS
pageable_arena=PASS
memory_pressure=PASS
trace_engagement=PASS
trace_parser_telemetry=PASS
new_hot_path_sync=0
primary_metric=weighted_decode
sampled10_vs_off10_weighted_throughput_delta=-71.799280%
sampled10_vs_off10_weighted_latency_overhead=+254.600873%
overhead_target_le_1pct=FAIL
overhead_hard_fail_gt_2pct=HARD_FAIL
PAGEABLE10_promoted=no
baseline=PAGEABLE0
```

The result is valid as an observed gate failure. With one sequential pair it
does not prove that sparse tracing alone caused the runtime state divergence.
No further replica was authorized or run.

Machine-readable metrics:
`DS4_P1_PAGEABLE10_METRICS.json`,
SHA-256
`2E1E8133C31FB03F4C62F5D7809CF3E6A6A3CEBD44D99D8BBDE1A055D81D4C32`.

## Frozen provenance and single-change contract

```text
ds4_server.exe_SHA256=185E46A057CE6BFFA7605638DDC5A3994FAFF3DBC48A3F7FF9B0B6617F3B9717
dedicated_runner_SHA256=C1F3C7A90BB6974CE205C83605EF32DCE912169BD9B8AC8D3F7C180DD3E19BA0
frozen_PAGEABLE0_runner_SHA256=E8A746195C5BD09C6C034E1F3EBC8C482146600843F56DDD2AADCEF9BBD200E5
dedicated_manifest_source_SHA256=7A10410DBB436DF7067D3FACE1E571FF892EDE9B98A2321E2230566023EE2BD8
run_manifest_47_SHA256=F3FC965D00C8038665CA3329934361A11C0E697255D9F560C328D4EC61E35907
ds4.c_SHA256=43B2ECE184916AA2731A299BE420A5133248BE24B3F17BCFB396D20B97CA1A2A
ds4_cuda.cu_SHA256=6D6BE460609C74EEEA60B43F6D62DB812F722D4A270B45797EB6A17909446257
ds4_gpu.h_SHA256=52AC84C2F10AE1D1EDB3032AFCFBDB1AAE000CDF148D98CCDAC42AA0739223AB
model_SHA256_receipt=efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668
ctx_capacity=150000
seed=12345
temperature=0.7
turn1_max_tokens=128
turn2_max_tokens=32
GraphTensorDevice=Off
PackedCopy=Off
BatchedPublish=Off
G73HostSelected=Off
RouteIoQd=Off
DS4_G73_PAGEABLE_OVERFLOW_GB=10
```

The dedicated runner is a copy; the frozen PAGEABLE=0 runner remained
byte-identical. Relative to the PAGEABLE=0 manifest, the only base variable
changed is:

```text
DS4_G73_PAGEABLE_OVERFLOW_GB=0 -> 10
```

The OFF10 versus sampled10 A/B changes only the trace overlay.

## ValidateOnly strict

OFF:

`C:\Users\imanu\Documents\Codex\2026-07-25\legg\outputs\p0p1_runtime\20260726_173657_p1_pageable10_trace-v2_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off`

Sampled:

`C:\Users\imanu\Documents\Codex\2026-07-25\legg\outputs\p0p1_runtime\20260726_173657_p1_pageable10_trace-v2_trace-sampled_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off`

```text
manifest=47/47
instrumentation_overlay=9/9
PAGEABLE_OVERFLOW_GB=10
source_exe_model_runner_hashes=PASS
server_started=no
OFF_result_SHA256=A4387F846593AD2BBC9DA43B01EB58FFEC2AA4E312370B1BE33B9BAFEC288D6D
sampled_result_SHA256=E7B9506DC3846B94FF1BC147FB2234DF2003A0440B4B6103B6C3DB13F8D2C0EE
```

## Manifest 47/47

```text
DS4_CUDA_ARENA_WRAP_SCHEDULE=source-parts
DS4_CUDA_ARENA_WRAP_TRUST_WORKER_CHECKSUM=1
DS4_CUDA_ARENA_WRAP_UNLOCK_SOURCE_RANGES=1
DS4_CUDA_ARENA_WRAP_UNLOCK_WAVE_GIB=4
DS4_CUDA_DYNAMIC_ARENA_GB=30
DS4_CUDA_EMBED_ROW_STAGING=1
DS4_CUDA_KV_STAGED_RING=1
DS4_CUDA_MOE_CACHE_POLICY=lru
DS4_CUDA_MOE_GPU_RESIDENT_ROUTES=1
DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC=1
DS4_CUDA_MOE_SPLIT_FUSED=1
DS4_CUDA_MOE_SPLIT_HIT_MISS=0
DS4_CUDA_NO_Q8_F16_CACHE=1
DS4_CUDA_PREFILL_MASS_OBSERVE=1
DS4_CUDA_PREFILL_MASS_WRAP=1
DS4_CUDA_PREFILL_TIER_COMPOSE=1
DS4_CUDA_PREFILL_TIER_RESERVE_SLOTS=128
DS4_CUDA_PREFILL_TIER_ROUTER=open
DS4_CUDA_RELEASE_PREFILL_SCRATCH=1
DS4_CUDA_STREAM_FROM_RAM_MASKED_BUDGET_GB=2
DS4_CUDA_STREAM_HOT_RESERVE_MB=256
DS4_CUDA_STREAM_RESERVE_MB=1024
DS4_CUDA_STREAMING_EXPERT_CACHE_N=140
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=0.125
DS4_CUDA_WEIGHT_CACHE_VERBOSE=1
DS4_EXPERT_TIER_CLOCK_CALLS=430
DS4_EXPERT_TIER_HYSTERESIS=1.25
DS4_EXPERT_TIER_MIN_FREQUENCY=3
DS4_EXPERT_TIER_POLICY=mass-lfru
DS4_EXPERT_TIER_REPLACEMENT_BUDGET=32
DS4_EXPERT_TIERING=enforce
DS4_G130_U1_ATTRIBUTION=1
DS4_G133_DECAY=0.98
DS4_G133_KNOCK_X=3
DS4_G133_KNOCK_Y=5
DS4_G133_PROMOTE_BUDGET=8
DS4_G133_ROTATOR_IO_TIMEOUT_S=0.05
DS4_G133_SEED_DYNAMIC=1
DS4_G133_TIER=1
DS4_G133_TRANSIENT_IO_TIMEOUT_S=0.25
DS4_G73_OPEN=1
DS4_G73_PAGEABLE_OVERFLOW_GB=10
DS4_METAL_GRAPH_TOKEN_PROFILE=1
DS4_METAL_PREFILL_CHUNK=250
DS4_MODEL_BYTES=86720111488
DS4_MODEL_SHA256=efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668
DS4_REAP_PREFETCH_THREADS=8
```

## Instrumentation overlay 9/9

OFF10:

```text
TraceMode=Off
DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>
DS4_CUDA_DECODE_TRACE_POSITIONS=<UNSET>
DS4_CUDA_DECODE_TRACE_LAYERS=<UNSET>
DS4_CUDA_NSYS_CAPTURE=<UNSET>
DS4_CUDA_ALLOC_TRACE=1
DS4_CUDA_ALLOC_TRACE_MIN_MIB=1048576
cuda_alloc_event_lines=suppressed
cuda_alloc_summary_phases=decode-start,request-end
```

Sampled10:

```text
TraceMode=Sampled
DS4_CUDA_DECODE_TRACE_SAMPLE_EVERY=<UNSET>
DS4_CUDA_DECODE_TRACE_POSITIONS=13,141
DS4_CUDA_DECODE_TRACE_LAYERS=0,4,20,42
DS4_CUDA_NSYS_CAPTURE=<UNSET>
DS4_CUDA_ALLOC_TRACE=1
DS4_CUDA_ALLOC_TRACE_MIN_MIB=1048576
cuda_alloc_event_lines=suppressed
cuda_alloc_summary_phases=decode-start,request-end
```

## Physical runs and correctness

OFF10:

`C:\Users\imanu\Documents\Codex\2026-07-25\legg\outputs\p0p1_runtime\20260726_173725_p1_pageable10_trace-v2_trace-off_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off`

Sampled10:

`C:\Users\imanu\Documents\Codex\2026-07-25\legg\outputs\p0p1_runtime\20260726_174145_p1_pageable10_trace-v2_trace-sampled_packed-off_publish-off_hostsel-off_routeio-off_graphdev-off`

Both:

```text
gate_pass=true
serving_exact_gate_pass=true
manifest_count=47
ctx_capacity=150000
cached_prefix_tokens=141
suffix_tokens=19
snapshot_unchanged=true
resident_unchanged=true
decode_refused_count=0
mailbox_quarantine_count=0
alloc_failure_row_count=0
shutdown_mode=graceful_http_verified
turn1_completion_tokens=128
turn1_content_SHA256=7f82253a4825191926f56073e40f10a0cff5541a721731bc81d2909dc1a4a65b
turn2_completion_tokens=32
turn2_content_SHA256=0179556c8e2dbcdc818fad315ca4df78f7537b63816dac276615b314195b13eb
```

Rotator:

| Metric | OFF10 | Sampled10 |
|---|---:|---:|
| result | complete | complete |
| requested/attempts/successes | 300/300/300 | 304/304/304 |
| failures | 0 | 0 |
| advisory failures/releases/degraded | 0/0/0 | 0/0/0 |
| structural rejects/stale/dropped | 0/0/0 | 0/0/0 |
| accounting pass | true | true |

## PAGEABLE arena and memory-pressure gate

Both arms:

```text
pageable_bytes=10737156096
pageable_GiB=9.99975
pageable_slots=1517
pinned_slots=4551
total_slots=6068
allocation_failure_or_fallback=0
pageable_paged_out_before_copy=0
```

| Engagement | OFF10 | Sampled10 |
|---|---:|---:|
| pinned ready | 300 | 304 |
| pageable ready | 0 | 0 |
| pinned hits | 23,671 | 23,654 |
| pageable hits | 2,933 | 2,933 |
| minimum available RAM | 9,544.2 MiB | 9,048.4 MiB |
| maximum working set | 44,546.9 MiB | 44,106.0 MiB |
| maximum private bytes | 55,929.9 MiB | 55,962.4 MiB |

The minimum-RAM gate is 6,144 MiB and passes in both arms. The working set
stabilizes near 44 GiB after arena population instead of repeatedly collapsing
and rebuilding. The runtime reports zero pageable buffers paged out before
copy. The monitor's page-fault column was unavailable, so this receipt does
not claim zero OS pagefile I/O. Increasing `server_read_mb` is explicit model
and route `pread`, not by itself evidence of pagefile thrash.

## Performance in tokens/second

`weighted_decode` is the primary metric: 160 decode tokens divided by the
summed graph time at positions 13-140 and 160-191.

| Metric | OFF0 | OFF10 | Sampled0 | Sampled10 | Sampled10 vs OFF10 |
|---|---:|---:|---:|---:|---:|
| turn 1 client wall | 0.414714 | 0.983530 | 1.073963 | 0.429300 | -56.351% |
| turn 1 graph, pos 13-140 | 2.274443 | 2.471015 | 2.261187 | 0.598526 | -75.778% |
| turn 1 mature, pos 50-140 | 2.636499 | 2.854573 | 2.625932 | 0.767330 | -73.119% |
| turn 2 client wall | 0.878162 | 0.899589 | 0.913419 | 0.392869 | -56.328% |
| suffix graph, pos 141-159 | 1.010899 | 1.013097 | 1.122273 | 0.297511 | -70.633% |
| turn 2 decode, pos 160-191 | 2.424509 | 2.695061 | 2.448295 | 2.682215 | -0.477% |
| weighted decode | 2.302951 | 2.512794 | 2.296285 | 0.708626 | -71.799% |

Primary summed graph latency:

```text
OFF10=63674.148_ms
Sampled10=225789.085_ms
latency_overhead=+254.600873%
target_le_1pct=FAIL
hard_fail_gt_2pct=HARD_FAIL
```

Cross-PAGEABLE comparison:

```text
OFF10_vs_OFF0_weighted=+9.111893% -> variance by the <10% rule
OFF10_vs_OFF0_turn2_decode=+11.159042% -> secondary segment only
Sampled10_vs_Sampled0_weighted=-69.140332% -> material regression
Sampled10_vs_Sampled0_turn2_decode=+9.554409% -> variance
```

OFF0 turn-1 wall includes its known cold/page-in interval and is not directly
comparable. The primary graph metric is not rescued by that caveat:
sampled10 is slower than OFF10 by 71.799%.

## TTFT proxy and WRAP

The client request is non-streaming, so an actual HTTP time-to-first-byte or
time-to-first-token is unavailable. The closest server-side first-output proxy
is `prompt_done + first decode graph token`.

| Timing | OFF10 | Sampled10 | Delta |
|---|---:|---:|---:|
| WRAP | 49.095 s | 50.836 s | +3.547% |
| turn-1 prompt done | 72.014 s | 73.681 s | +2.315% |
| first decode graph token | 3.602257 s | 7.104656 s | +97.228% |
| server first-output proxy | 75.616257 s | 80.785656 s | +6.837% |
| turn-2 suffix prompt | 19.119 s | 64.471 s | +237.210% |

## Trace engagement and parser

```text
OFF_trace_records=0
sampled_trace_records=10
token_positions=13,141
layers_per_position=0,4,20,42
token_record_complete=2/2
token_io_counter_valid=2/2
layer_sample_id_match=8/8
timing_event_count=178
event_sync_count=0
dropped_records=0
worker_attribution_errors=0
nsys_capture=0
```

Position 13:

```text
host_wall_ms=7114.831700
gpu_default_window_ms=7103.881836
host_minus_device_window_ms=10.949864
h2d_api_ms=0.772100
h2d_count=80
h2d_bytes=170010624
d2h_api_ms=6.884900
d2h_count=5
d2h_bytes=517216
pread_wait_ms=0
pread_count=0
pread_bytes=0
sampled_layer_worker_queue_ms_sum=12.9973
sampled_layer_worker_service_ms_sum=25.5213
sampled_layer_worker_ready_ms_sum=31.1036
sampled_layer_upload_drain_ms_sum=24.1551
```

Position 141:

```text
phase=suffix_prefill_not_route_worker_decode
host_wall_ms=36732.695100
gpu_default_window_ms=36717.578125
host_minus_device_window_ms=15.116975
h2d_api_ms=0.317100
h2d_count=1
d2h_api_ms=0.253700
d2h_count=1
route_worker_pread_upload_counters=0
```

Semantic separation:

- `gpu_default_window_ms` is a CUDA-event device window. It is not a kernel
  compute-union or proof that the SMs were busy.
- `*_api_ms`, submission, and enqueue fields are host/API time.
- `pread_wait_ms` and Q1 service time are explicit model I/O wait.
- worker-ready, queue age, upload-drain, stream, split, and end-sync fields are
  dependency waits.
- a residual interval is not relabeled as compute or GPU idle without Nsight.

There are no new synchronizations: `event_sync_count=0`. The trace reads
elapsed times only after the existing end sync. Existing split/end sync counts
are observed, not introduced.

## Observed performance divergence

Turn-1 graph-token distribution:

| Metric | OFF10 | Sampled10 |
|---|---:|---:|
| p50 | 360.598 ms | 1,364.376 ms |
| p95 | 586.592 ms | 3,875.579 ms |
| maximum | 3,602.257 ms | 7,104.656 ms |
| tokens over 1 second | 1/128 | 83/128 |

The summed graph decomposition is:

| Segment | Arm | encode/window ms | execute ms | read ms |
|---|---|---:|---:|---:|
| turn1 | OFF10 | 51,177.790 | 611.191 | 11.600 |
| turn1 | Sampled10 | 212,315.525 | 1,528.313 | 14.808 |
| suffix | OFF10 | 18,586.937 | 165.500 | 1.941 |
| suffix | Sampled10 | 63,589.533 | 271.237 | 2.347 |
| turn2 decode | OFF10 | 11,714.979 | 155.723 | 2.873 |
| turn2 decode | Sampled10 | 11,771.846 | 155.669 | 2.923 |

The label `encode/window` remains intentionally broad because it includes
host orchestration and device/dependency progress; it is not pure CPU time.

Q1 storage service is not slower in sampled10:

```text
OFF10_Q1_service_sum=0.3913542_s
Sampled10_Q1_service_sum=0.3770689_s
OFF10_queue_age_sum=3495.9209_ms,max=181.8363_ms,over500ms=0
Sampled10_queue_age_sum=16939.3228_ms,max=1955.1911_ms,over500ms=8
```

During sampled10 suffix, fail-closed exact serving engages:

```text
bounded_selected_load_unavailable_layers=31
terminal_plain_pread_parts=556
terminal_h2d_copies=186
allocation_fallback=0
```

This is a correctness-preserving route fallback, distinct from an allocation
fallback. It explains a material part of the suffix slowdown. It does not
explain the whole turn-1 collapse, whose route-hit and rotator totals remain
close between arms. The single pair establishes a gate failure, not isolated
causality.

## RAM, VRAM, GPU and power

| Metric | OFF10 | Sampled10 |
|---|---:|---:|
| monitor samples | 34 | 76 |
| minimum available RAM, MiB | 9,544.2 | 9,048.4 |
| maximum server working set, MiB | 44,546.9 | 44,106.0 |
| maximum server private, MiB | 55,929.9 | 55,962.4 |
| average GPU used, MiB | 10,454.353 | 10,435.342 |
| maximum GPU used, MiB | 12,025 | 12,019 |
| average GPU utilization, % | 25.294 | 18.368 |
| maximum GPU utilization, % | 82 | 95 |
| average GPU power, W | 27.061 | 24.054 |
| maximum GPU power, W | 46.15 | 39.92 |

The averages cover different wall durations and are phase-weighted; they are
not causal A/B measures. The low sampled10 average is consistent with longer
host/dependency gaps, but cannot split GPU compute from idle without Nsight.

## KV lifecycle phase correlation

Both arms log the same lifecycle operations:

```text
[kv-staged] restored tensors=105 bytes=2109242368 reason=request-end
[kv-staged] migrated tensors=105 device_bytes=2109242368 pinned_bytes=2109242368 mapped=0
allocator_request_end_free_mib=0.0
```

Coarse monitor correlation:

```text
OFF10_request_end_VRAM_MiB=10199 -> 12020
OFF10_followup_decode_start_VRAM_MiB=12020 -> 10032
Sampled10_request_end_VRAM_MiB=10067 -> 12014
Sampled10_followup_decode_start_VRAM_MiB=11993 -> 10005
```

This is phase correlation, not isolated causality. Two facts remain direct:

- request-end has zero measured VRAM headroom;
- the follow-up performs a 2.109 GB KV restore/migrate round-trip.

The separate candidate remains unchanged and was not implemented or run:
lifecycle-aware/lazy live-byte restore or preservation of staged KV between
turns, fail-closed, with independent correctness tests at live position 160
and capacity 150000.

## Artifact hashes

| Artifact | OFF10 SHA-256 | Sampled10 SHA-256 |
|---|---|---|
| `control_manifest.txt` | `5BF8A49EA1B3F65EE9171DCBE06AD960CE33FE731B2350452635530DD4769DF2` | `361742A2C2E5DABDB18AC7E6E4795FBE22D00A06FE4DCEFFE37C4FD9A2CF531D` |
| `experiment_manifest.txt` | `CEFBF393BA066460589381CA42A914758E0FFA60C9A26C0A8AC77F6908ECE538` | same |
| `instrumentation_manifest.txt` | `FC8F2069ED81AB1583D6349869F0A5ADDBC78893E86E27A44E64DE7798097FB1` | `CF110FBB824A157BA9F2C3AA4E35DC39E3D2655CE21575B280EDB531E619F1B0` |
| `provenance.txt` | `F2851B0D6E76728572E21AC4CAB00192867F5E3E8612F4918562774A2320CFC4` | `7C7A2A7394317952BABA9889AB6C90715DF03FE62A99EF36CAB6E42669F10A35` |
| `result.txt` | `0DD5C1A06197B7315FDB07FCB0BB5640123D698A24FE82BC2E7B361DF12C3FA3` | `2EB8B71F63CE1532154971722D4E3B7FA892888CE1D430CE041D287980873EA3` |
| `runner.status.log` | `9E0278C09362348706DF18E9B20310531383F91780CAA78729E062D45CB34DC7` | `A77F8300377EC3A447A10F37F1054DB09F7F13895CA7469888E8F01E050F1B87` |
| `monitor.csv` | `A82179B706398E1F4F7DC447E136EE7AA80D19AA025020178F9DA26795641605` | `ED2ACCA383C9857AF983468A73242C240E20FDC818246A2E6235ADADC5E1EE97` |
| `server.stderr.log` | `12F297C82F4679BDF84897534FA43085E784329A6195FB98DBDB7753BB861925` | `BD84232EBAAA3DF92F54983294D78701B5F1318EADB7597630C2A20DDF2902F9` |
| `cuda_alloc_summaries.log` | `72C32D218739AC88E31659AC91CA0771B77376EEEB6B6725673C2C5174EB1273` | `41DB23A846209535FE9A0FB618A358EA34D171F8344F5A806C19D66C5B24651D` |
| `turn1.request.json` | `6C29402227270E0B937CAFD5D4DE7431E493073362A0ABE1BDFF480DEE1DD127` | same |
| `turn1.response.json` | `2CECACA7AFE0D52B25ADD2BE64CBC89CE616D8033864EA47ABC361A0C7664B04` | `94772A4A808FE6455E5CEFD1A419A4C3EC8B6D727CE0751A5EE41A6987E3613B` |
| `turn2.request.json` | `39105F08440A88B153965C4B311AAF68D8956F859EEE6BD8D7B82B03F4438A18` | same |
| `turn2.response.json` | `C00651D8909C934ABCB2D12573C542FC664FEF9F12CB0895CFDDADF586C49311` | `BD9270703A92D5B8AA3CEB314FEDCB8F8BD3852BBAA6EBFA9A76994056491996` |
| `shutdown.receipt.json` | `EECC4763C84CE14F1D588BAF6D60EF8E396E3FF1D4AFC7E8CF27897FEB01F9BD` | same |
| `manifest_47.env` | `F3FC965D00C8038665CA3329934361A11C0E697255D9F560C328D4EC61E35907` | same |
| `launcher_arguments.txt` | `D13924AC23D8A06063798732972F78C308B80DA42456D712B0FDD236AAA2A287` | same |
| `server_arguments.txt` | `DB08B60770E42011E67CE4D965117B42477DF1F04075101ED5C4DE4E078D0622` | same |

Raw response hashes differ because timing metadata differs. Extracted content
hashes are exact.

## Postflight and decision

```text
DS4=0
Nsight=0
runner=0
port8000_listeners=0
RAM_free=55.32_GiB
GPU=P8,used_471_MiB
forced_kill=0
extra_replicas=0
```

Risk/payoff:

- OFF10 improves weighted decode by `+9.111893%` versus OFF0, which is
  explicitly variance under the `<10%` rule.
- PAGEABLE=10 consumes about 10 GiB more host memory and lowers the observed
  minimum available RAM from about 19.6 GiB to 9.0-9.5 GiB.
- Sampled10 enters a materially slower state while preserving correctness.
- The payoff is not large or stable enough to offset the memory and
  performance-variance risk.

Decision: retain `DS4_G73_PAGEABLE_OVERFLOW_GB=0` and the already validated
PAGEABLE=0 trace contract. Do not promote PAGEABLE=10 and do not use sampled10
as an overhead-qualified configuration.

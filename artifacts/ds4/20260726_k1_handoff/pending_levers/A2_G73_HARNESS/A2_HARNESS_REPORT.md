# A2 G73_OPEN harness report

## Verdict

- `HARNESS_READY=yes`
- `READY_FOR_RUN_SLOT=no`
- DS4 launches performed by this task: `0`
- Builds performed by this task: `0`
- Shared repository/runner writes performed by this task: `0`

The run-slot verdict remains `no` only because A1 is still changing the live
source/runner receipts. The standalone harness itself is complete and its
non-serving tests pass.

## Frozen target

The standalone runner currently targets the received P0 binary:

- executable SHA-256:
  `F2F6A25600EF0B6728BE0DAE77ACFC5CE89ABCEB1446555A7DB946DA26F11DFF`;
- executable bytes: `12,412,416`;
- model SHA-256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`;
- `ds4.c` build receipt:
  `2C2840FCA3F1FF7B0C745BF376E04BCB28AED248FDC6DE504093D949288415A4`;
- `ds4_cuda.cu` build receipt:
  `FEB2954ED2FA583C0631C4FE763B75FFA1F4D0D51598D2ED2AC6F5A87E303496`;
- `ds4_gpu.h` build receipt:
  `EB51FE6F3250CE023803AFF77B407CE73BBE004F212F33E2E68638969E9A8E06`.

The live source no longer matches those source receipts because A1 is working
on it. `AllowSourceDriftForValidation` is therefore legal only with
`ValidateOnly`; a serving invocation still refuses source drift.

## Implemented A2 contract

The runner:

1. accepts only `G73Open=On|Off`;
2. fixes completion lengths at 128 and 32 tokens;
3. requires P0 settings: all P2/P3 flags, graph-tensor-device, allocation
   tracing and timing instrumentation OFF;
4. requires pageable overflow zero;
5. reads the immutable 47-entry manifest and writes a per-run effective
   manifest;
6. allows zero manifest differences for ON and exactly one for OFF:
   line 41, `DS4_G73_OPEN=1 -> 0`;
7. emits both effective and arm-normalized manifest hashes;
8. clears the inherited `DS4_*` environment before applying the effective
   manifest, so micro-patch variables remain absent.

## Branch-aware gates

ON engagement requires:

- exactly one G73 terminal-ready line;
- arena size `32,239,779,840` bytes, 4551 resident slots, pageable zero and
  four transport-ring slots;
- one SSD-wrap ready line with two SSD and two H2D slots;
- 160 G73 attribution rows;
- positive `out_of_mask_routes`;
- per-token conservation of transient, promoted, selected-fallback,
  terminal-exact, clamped and refused classes;
- zero selected/terminal fallback, clamp and refusal;
- final SSD-wrap `result=complete`, with failures, structural rejects, drops,
  detached workers and safe leaks all zero.

OFF engagement requires:

- no terminal or SSD-wrap ready/final lines;
- ordinary arena size `32,211,468,288` bytes, 4551 resident slots and pageable
  zero;
- 160 pre-G73 attribution rows without G73 route-outcome fields;
- `g73_classification_d2h=0`.

The P3-A D2H gate is branch-aware:

- ON: `g73_classification_d2h == route_calls` for every final route summary;
- OFF: `g73_classification_d2h == 0`.

## Common correctness and shutdown

The common gate checks:

- HTTP 200 for both requests;
- prompt/completion counts 13/128 and 160/32;
- `finish_reason=length`;
- deterministic response content hashes;
- request hashes recorded for the four-run identity gate;
- short-suffix preservation with unchanged snapshot/resident counts;
- route errors, tier failures, forbidden cold-SSD-to-VRAM and forbidden
  fallback patterns all zero;
- P2/P3 engagement counters zero;
- legacy copy submissions equal `3 * miss_experts`;
- legacy publish kernels equal `miss_experts`;
- one startup model-cache preparation, preventing a hidden second backbone
  load;
- authenticated shutdown receipt with `status=draining`;
- final route, tier and arena summaries;
- process absent, port 8000 free, monitor stopped and GPU memory below the
  preflight threshold after shutdown.

## Performance and resource receipts

Graph throughput is computed separately from HTTP wall throughput:

- turn 1: positions 13–140, 128 tokens;
- suffix prefill: positions 141–159, 19 tokens, reported separately;
- turn 2 decode: positions 160–191, 32 tokens.

The result also records:

- route classes: cold, RAM hits, VRAM hits and transient;
- SSD bytes and RAM-to-GPU H2D bytes;
- RAM first/minimum/last;
- process working-set/private peaks;
- process read/write/page-fault deltas;
- VRAM first/peak/last;
- GPU utilization and power averages/maxima;
- preflight and postflight RAM/VRAM.

The ABBA comparator rejects different binaries, models, normalized manifests,
request bodies, responses or failed run gates. It reports `<10%` absolute
median graph-throughput delta as variance.

## Static validation evidence

The final static test produced:

- PowerShell syntax errors: `0`;
- ON `ValidateOnly` exit: `0`;
- OFF `ValidateOnly` exit: `0`;
- effective manifest counts: `47/47` and `47/47`;
- A/B differences: exactly one, line 41;
- normalized manifest SHA-256:
  `6793cda094b09a23b033c13679a4394a5418a549fc5827c29400d207d3128621`;
- `ds4_server` PID set before validation: empty;
- `ds4_server` PID set after validation: empty;
- server started: `no`.

## Merge risk

The shared runner changed while A2 was being prepared. The supplied patch is a
reference diff against snapshot SHA-256
`358EF45ABDBDC398182441C66658F33FA4BAFF1E735B69192536B64378C15ACD`.
If A1 finishes with any other hash, port the A2 sections into a temporary copy
and preserve the A1 receipt/path changes. Do not force-apply or overwrite the
shared runner.

The engagement regexes have been checked statically against the current source
and existing ON logs, but the OFF specialization has not been executed under
this authorization. Its first actual run must therefore be treated as a gate
validation as well as a measurement; any missing branch receipt makes the
ABBA set invalid.

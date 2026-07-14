# Windows-native decision review after ledger reconstruction

Date: 2026-07-14

This is a decision memo over the canonical experiment ledger, not a second
matrix. The complete rows remain in
`runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv` and the generated
Markdown view.

## Measurement hierarchy

1. Keep observed records, repeated speed measurements and quality verdicts in
   separate columns.
2. An `n=1` safety result can promote a mechanism to replication, not to a
   sustained headline.
3. Repeats inside one server are labelled `same_server_process`; independent
   process launches must be identified separately.
4. A speed result without L0-L3 grading remains `speed_only`.
5. Cache state, exact hash, source HEAD and executable hash accompany every
   comparison.

## Current local RTX 3060 readout

| Evidence | Result | Status |
|---|---:|---|
| G22 KEEP independent replication | 4.55 median server t/s; samples 4.55, 4.42, 4.60 | promoted transport baseline, independent n=3, exact |
| G22 DROP independent control | 1.56 median server t/s; samples 1.42, 1.56, 1.57 | independent n=3, exact |
| Supporting previous-executable G22 response | 4.29 / 3.819 t/s | continuity only, not a third replica |
| Historical G19B2 final chunks | 4.34 / 4.26 t/s | window/tail records, n=1 |
| Highest short same-process n=3 server mean in imported G7 JSON | 2.98 t/s | speed-only, `Hi`, not independent processes |
| G19B W16/min-hits3 controlled independent A/B | 2.313 ON vs 2.083 OFF t/s | positive, +11.0%, identical complete SHA |
| G20 grow8 controlled independent A/B | median 2.02 ON vs 1.97 OFF t/s | residency positive; speed inconclusive (+2.5%) |

The correct summary is therefore: W64/min-hits1 retained arena residency now
sustains greater than 4 t/s in three independent local processes. The median is
4.55 t/s server-side, every KEEP sample is at least 4.42 t/s, and the exact
DROP control median is 1.56 t/s.

## Lever classification

| Lever | Best measured evidence | Decision |
|---|---|---|
| Clean diagnostics-off timing | G8 clean 2.042 vs diagnostic 0.706 client t/s | mandatory benchmark hygiene |
| MoE I/O queue depth | G8 QD1 2.240 vs QD2 2.016 vs QD4 1.913 client t/s | keep QD1 |
| Global resident expert LRU | G9 zero hits; monotonic regression | rejected |
| Layer-local top1 cache | G10 9.54% hit rate, costs exceed savings | rejected |
| Shared/selected overlap | G11 coding -1.4%; G12 coding effectively neutral | rejected as default |
| SPEX hidden synchronous path | G13 recall about 31%, 3-12% decode penalty | diagnostic only |
| SPEX async ring | G14 ring1 -2.1%, ring4 -2.8% vs off | diagnostic only |
| SPEX K1 speculative prefetch | G16 Cyber64 -54.29%, 44% unused traffic | rejected unconditional |
| Pinned arena allocation | G17 allocation/lifetime correct; 31 GiB measured ceiling | substrate accepted |
| Transactional WRAP | G18 exact transport proof, no policy-speed verdict | substrate accepted |
| Synchronous observed residency | G19A tail +31.5%, whole response -16.7% from duplicate WRAP read | mechanism positive, implementation rejected |
| First-fetch mirrored residency | G19B +11.0% controlled, exact | first production-positive transport lever |
| Direct ReadFile into long-lived slot | G19B 1.78 vs mirrored 1.88 t/s safety | rejected |
| Parallel boundary verification | G19B2 2.37 vs inline 1.85 t/s adjacent n=1 | retained; attribution not fully causal |
| Grow-only residency | G20 2.89->8.51 useful GiB, +16.28 hit points, +2.5% median speed | keep candidate, longer gate pending |
| Q8-to-F16 fixed cache 256/1280 | G21 0.38 t/s, 614 GiB Win32 reads, hash mismatch | rejected exact configuration |
| Cross-request learned-arena reuse | G22 KEEP median 4.55 vs DROP 1.56 t/s, ratio 2.92x, independent n=3, exact | promoted Windows transport baseline |

## G22 replication verdict

G22 directly tested whether the large W64/min-hits1 resident set, rather than
generic Windows warming, explains the greater-than-4 whole-response result.

Pre-registered order across six independent server processes:

```text
DROP, KEEP, KEEP, DROP, DROP, KEEP
```

Every process performs the same first request to learn the arena, then measures
the second request. The only treatment variable is
`DS4_CUDA_DYNAMIC_ARENA_CARRY_ACROSS_REQUESTS=0|1`.

Common configuration:

- native Windows RTX 3060 12 GiB;
- Cyber HTML prompt, greedy, nothink, maximum 256 tokens, context 256;
- 30 GiB arena, W64, min-hits1;
- 2 GiB masked RAM budget, 1,024 MiB CUDA reserve;
- WRAP workers 8, MoE I/O QD1;
- expert cache, SPEX and shared overlap off;
- expected output SHA-256
  `f2677447c1a5e95934469c6c8f07ee943ccd9c079ef9350b07c2d0ce8fc1b576`.

Fail closed if any run has a hash mismatch, arena fatal, missing carry marker,
empty snapshot, observer re-publication, wrong lookup state, or incomplete
telemetry.

Promotion requires all of:

1. three valid independent runs per arm;
2. KEEP median server decode at least 4.0 t/s and every KEEP at least 3.8 t/s;
3. KEEP/DROP median server-decode ratio at least 2.0;
4. all expected output hashes exact.

Measured server decode samples were DROP 1.42/1.56/1.57 and KEEP
4.55/4.42/4.60 t/s. The medians are 1.56 and 4.55 t/s, respectively, for a
2.92x ratio. Every expected output hash matched. The gate passed and retained
learned residency is now the Windows transport baseline.

## Next highest-value measurement

Measure transfer before adding another optimization. Use the promoted G22
configuration but make request 2 a distinct deterministic domain from request
1. Compare KEEP with DROP in an order-balanced safety gate, then replicate only
if exactness and a useful speed separation survive. This is the missing proof
that the arena acts as a reusable transport cache rather than a residency set
specialized to one repeated prompt.

If cross-domain reuse is weak, add exactly one adaptation mechanism: re-arm the
already measured grow-only residency policy while retaining the existing set.
Rotation, SPEX and mask-width changes remain off until that isolated comparison
is measured.

## Process rule

After each completed arm, import its result JSON and regenerate the canonical
ledger before launching the next experimental family. Before selecting a new
lever, reread the highest-observed table, evidence classification and negative
matrix. This rule is part of the benchmark protocol, not optional cleanup.

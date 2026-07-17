# Design 0053: IQ1_S probation promotion, runtime opt-in

**Status:** runtime candidate implemented on an experimental branch,
2026-07-17; G97 has a structural PASS with final telemetry preserved, and G98
has clean fixed-order `n=3` exactness/performance evidence, but long-form
quality grading, SOTA, performance-win, and default-readiness gates remain
unmet. The current measured
architecture is a composed prefill snapshot plus open router with a pre-reserved
separate 16-slot pinned probation pool. The measured G97 structural route shape
is mixed 5+1: five exact IQ2/Q2 routes plus one IQ1_S minimum route per
eligible routed layer/token. G97 `n=1` is structural only; current G98 evidence
now includes fail-closed attempts, one shared-pool structural smoke, and a
clean fixed-order `n=3` exactness/performance measurement. Because the
candidate was much slower even as the warmer second arm, this file makes no
runtime speed-win, quality, SOTA, or 10 t/s claim.

0053 is the next runtime-only step after the IQ1_S sidecar and GPU planner
checks in `docs/IQ1_S_WINDOWS_TEST_MATRIX_20260716.md`. It is not a bake, not
a GGUF rewrite, and not a new authoritative model. The exact routed expert
source remains the existing IQ2/Q2 primary model; IQ1_S is an opt-in sidecar
execution path for the lowest-weight selected route in the current token.

The design is narrower than 0052's generic cold-compression plan. 0052 defines
the fail-closed principles for derivative cold representations, asynchronous
promotion, provenance, and quality gates. 0053 applies those principles to the
already measured IQ1_S sidecar shape:

```text
current token:
  open router computes 6 selected routes
  5 higher-weight routes remain exact via SSD -> pinned RAM -> GPU if absent
  1 minimum-weight route executes from the IQ1_S sidecar

after that compute is queued:
  exact IQ2/Q2 source for the same expert is staged into pinned RAM probation
  the staged slot cannot enter VRAM for this token

next token or later:
  if the expert gains weight and exits the minimum route, exact IQ2 is eligible
  if it remains the minimum route, continue using IQ1_S
```

## Evidence boundary

Use the same claim rules as `IQ1_S_WINDOWS_TEST_MATRIX_20260716.md`:

- `n=1` structural runs can prove only wiring, counters, deterministic
  exactness for that invocation, or fail-closed behavior.
- Repeat flags, output hashes, token count, throughput, or automatic text
  heuristics do not create a quality verdict.
- Quality requires `n>=3` per arm with preserved outputs and recorded human
  L0-L3 grading.
- A contaminated run may still prove deterministic exactness when expected and
  observed hashes match, but its timing and SOTA fields are invalid.

Relevant existing evidence:

| Gate | Use in 0053 |
|---|---|
| G74 | Guardrail evidence only: use for fail-closed/contamination discipline, not as promotion, performance, or quality evidence for 0053. |
| G75 | Real expert IQ1_S component validation and transport measurement only. |
| G76b | Structural mixed-route counters for current-token mixed IQ2/IQ1 execution. |
| G86 | Clean `n=3` transport/perf baseline for IQ1 RAM cache and a guardrail for cache/host-state reporting, no L0-L3 grade and no 0053 promotion verdict. |
| G89-G93 | Structural planner/cache profiling only; G92/G93 timing contaminated by `ScheduledDefrag`. |
| G94 | Clean short `n=3` planner mechanical/perf comparison with exact repeated outputs, no L0-L3 grade. |
| G95 | Quality campaign not complete; first attempt/probe contaminated or invalidated by Defrag/cleanup state. |
| G97 | PASS, `n=1` structural only. Earlier closed-router run had no true cold promotions; forced eviction proved fail-closed on snapshot miss; two open-router attempts were refused before launch by quiescence because Windows `ScheduledDefrag` saturated `D:`. The final raw arena12 4+8 open-router/probation run completed with zero failures/forbidden/ram skips, mixed 5+1 promotion counters, 56 backing-RAM reclaims, and 141 `cold_to_2bit` stages in measured execution. |
| G98/G99 | G98 fixed-order `n=3` completed cleanly with exact 64-token output SHA across both arms, but the candidate was much slower; because order was fixed, the parent final performance verdict is withheld. No quality, SOTA, default-readiness, or long-form L0-L3 claim may be inferred. G99 is deferred until promotion churn is fixed. |
| G100 | Structural `n=1` promotion gate isolation sweep only. Five arms completed with identical output SHA, zero failures, zero forbidden/direct cold-to-VRAM transitions, and exact per-arm IQ2 SSD bytes recorded. The combined gate reduced IQ2 promotion SSD bytes by 94.87% versus legacy, while `min_weight=0.02` alone filtered zero candidates. No performance, quality, SOTA, or default-readiness claim. |
| G101 | No verdict. First combined `n=3` attempt completed its combined arm, but the suite stopped on a deduplicated binding-hash bug before producing any combined-vs-legacy verdict; fixed in ds4-win `876b4b3`. A second receipt attempt was stopped because `ComputeHash` used 4 KiB reads and drove `D:` to 3-6 MiB/s; fixed in `2a5e696` with 8 MiB `SequentialScan`/`IncrementalHash`, a 1 GiB `D:` microbench at 97.5 MiB/s with 128 reads, and receipt tests passing. A third full-hash attempt completed hashing, but the benchmark was correctly rejected by quiescence because Windows `ScheduledDefrag`/`Defrag.exe` was active on `D:`; no timing data is valid. Fix `91f1445` preserves preflight and retries quiescence without rehash, while failing closed on other errors. |

## Runtime contract

0053 is off by default and enabled only when all required sidecar, planner, and
probation flags are present. With the opt-in unset, behavior must remain the
accepted main IQ2/Q2 runtime.

Per routed layer and token:

1. The open router computes the normal 6 selected routes and keeps their exact IDs
   and gate weights.
2. The route with the minimum weight is marked `cold_iq1_current_token`.
   Ties follow the same deterministic first-minimum rule as the existing CPU
   fixture and GPU planner evidence.
3. The other five routes are `hot_iq2_current_token` and remain exact. If an
   exact hot-lane expert is absent from GPU, it may load through the same
   exact SSD-to-pinned-RAM-to-GPU route as other open-router exact misses.
4. The IQ1_S route reads from the sidecar for this token only. It does not
   rewrite the selected IDs, gate weights, or hot route count.
5. After the IQ1_S compute for that expert has been queued, the corresponding
   exact IQ2/Q2 expert is staged from the authoritative primary source into
   the pinned RAM probation pool. Backing RAM reclaim is allowed only for an
   exact expert copy that is already resident in VRAM; reclaim cannot create or
   imply an approximate exact source.
6. A probation slot is not eligible for VRAM publication until the next token.
   There is no same-token format switch and no direct SSD-to-VRAM promotion.
7. On a later token, if that expert is selected and is no longer the minimum
   route, the exact IQ2/Q2 copy becomes next-token eligible. If the expert is
   still the minimum route, it stays IQ1_S for that token.

This gives the minimum route a cheap current-token path while testing whether
short-term route momentum justifies exact IQ2/Q2 promotion for the next token.

## Copies and authority

| Copy | Representation | Residence | Authority |
|---|---|---|---|
| `SOURCE_IQ2_EXACT` | primary routed expert bytes, gate/up IQ2_XXS and down Q2_K | main GGUF / mmap / exact source path | only exact authority |
| `SIDECAR_IQ1_S` | IQ1_S routed expert sidecar | SSD and optional ordinary RAM cache | derivative execution source only |
| `PROBATION_IQ2_PINNED` | exact IQ2/Q2 native expert triplet | pinned host RAM probation slot | validated copy of `SOURCE_IQ2_EXACT` |
| `VRAM_IQ2` | exact IQ2/Q2 compute copy | existing VRAM expert cache / selected path | compute copy of exact source |

There is no IQ1_S pinned arena, no IQ1_S VRAM cache, no direct SSD-to-VRAM IQ2
promotion in compose or open-router publication, and no bake into a mixed
model file.

The exact IQ2/Q2 source remains authoritative. IQ1_S can execute the cold
minimum route, but it never becomes the source for an exact promotion. A
probation slot is built only from the exact primary model source.

## Fail-closed invariants

These are release gates:

1. **Selected IDs are immutable.** IQ1_S admission cannot change the six
   selected expert IDs or their gate weights.
2. **Exactly one cold route.** At most one of the six routes per layer/token is
   IQ1_S, and it must be the minimum-weight selected route.
3. **Five hot routes stay exact.** The other five routes execute as primary
   IQ2/Q2.
4. **No same-token promotion.** A staged IQ2/Q2 probation copy cannot enter
   VRAM for the token that caused the stage.
5. **No SSD IQ2 direct to VRAM in compose.** Exact IQ2/Q2 promotion goes
   through validated pinned probation, not direct SSD-to-VRAM publication.
6. **Staging failure is not output failure.** If IQ2/Q2 probation staging
   fails after the IQ1_S compute is already queued, the current GPU output is
   still valid for that token. The failure is telemetry and a gate failure, but
   it must not invalidate or rewrite already queued compute.
7. **Sidecar selected-load failure is fail-closed.** A missing, corrupt, stale,
   or mismatched IQ1_S selected load aborts the IQ1_S arm; it cannot silently
   fall back through a main-model offset or stale host pointer while claiming
   IQ1_S exposure.
8. **Exact remains exact.** Any counter claiming IQ2/Q2 exact use must trace
   back to `SOURCE_IQ2_EXACT`, not to IQ1_S reconstruction or approximation.
9. **Unknown provenance is a miss.** Unknown representation, stale generation,
   stale source identity, or incomplete triplet cannot be promoted.
10. **Publication is causal.** A probation slot becomes eligible only after the
    token boundary and after its source identity, layer/expert owner,
    generation, and byte count are validated.
11. **VRAM map miss is fail-closed.** If a would-be exact publication cannot
    resolve a valid VRAM map entry for the target expert, publication is
    rejected rather than falling back through stale host state or silently
    treating the route as exact.

## Probation slots

New environment variable:

```text
DS4_IQ1_PROMOTION_PROBATION_SLOTS
```

Initial meaning:

- unset or `0`: disabled;
- positive integer: number of exact IQ2/Q2 pinned RAM probation slots;
- implemented structural point: `16`.

The implemented point is intentionally small. One exact routed expert triplet is
6.75 MiB, so 16 slots are:

```text
16 * 6.75 MiB = 108 MiB
```

This is only the first mechanism point, not a proposed final cache size. The
16-slot pool is explicit pinned probation capacity outside the composed
prefill snapshot. It is pre-reserved before open-router publication and shared
by open-router exact hot-lane misses and IQ1_S promotion. Candidate capacity is
reduced before publication so the open router does not publish more exact
candidates than the composed snapshot plus probation layout can hold.

Implemented first-slot policy:

- compose the prefill snapshot with the configured number of candidate slots
  withheld before publication, preserving the same pinned-arena allocation;
- reserve the withheld capacity as a 16-slot pinned probation pool outside the
  snapshot;
- stage only after the current-token IQ1_S route has been enqueued;
- if no probation slot can be reclaimed, record a promotion failure and
  continue serving the already queued current token; the run then fails the
  promotion gate;
- do not evict a slot that is in-flight to VRAM or under validation;
- use a deterministic replacement rule among idle probation slots for
  reproducible G97 evidence.

Backing-RAM reclaim policy is telemetry-visible and conservative:

- reclaim only an exact backing-RAM copy whose matching exact expert is already
  resident in VRAM;
- never reclaim IQ1_S sidecar bytes and never treat a reclaim as an exact-source
  reconstruction;
- count reclaims separately from promotion stages so a structural pass can
  prove pressure handling without implying a throughput or quality result;
- fail the structural gate on reclaim bookkeeping mismatch, stale provenance,
  RAM skip, or any forbidden direct SSD-to-VRAM transition.

## Telemetry

Existing mixed-IQ1 and sidecar telemetry continues to provide route counts,
selected-load counts/failures, planner calls/failures, sidecar identity, and
IQ1 RAM-cache traffic. The new final line is:

```text
[iq1-promotion] final requested_slots=... reserved_slots=...
snapshot_evictions=... cold_observed=... cold_existing_2bit=...
cold_to_2bit_ram=... probation_ram_hits=... next_token_waits=...
promotion_2bit_ssd_bytes=... promotion_2bit_ssd_seconds=...
promotion_backing_ram_reclaims=... direct_ssd_to_vram_rejected=...
failures=...
```

Together, the final summary must expose at least these facts, split by run and
arm:

- `iq1_sidecar_enabled`;
- `iq1_sidecar_bytes` and sidecar SHA/source identity;
- `mixed_route_tokens`;
- `router_routes_total`;
- `hot_iq2_routes`;
- `cold_iq1_routes`;
- `cold_iq1_min_weight_routes`;
- `cold_iq1_tie_first_min_routes`;
- `sidecar_selected_loads`;
- `sidecar_selected_load_failures`;
- `probation_slots_configured`;
- `probation_slots_allocated`;
- configured/reserved probation slots and snapshot evictions;
- cold routes observed, already exact-backed, and newly staged to IQ2 RAM;
- probation RAM hits and deferred-eligibility waits;
- exact IQ2 SSD bytes and seconds spent staging;
- backing-RAM reclaims, reported separately from `cold_to_2bit_ram`;
- `forbidden_cold_ssd_to_vram`;
- promotion failures.

Required counter checks for a passing structural gate:

```text
sidecar_selected_load_failures = 0
requested_slots = reserved_slots = snapshot_evictions
cold_observed > 0
cold_to_2bit_ram + cold_existing_2bit > 0
forbidden_cold_ssd_to_vram = 0
direct_ssd_to_vram_rejected = 0
promotion failures = 0
```

`next_token_waits` may be zero: a routed `(layer, expert)` cannot normally be
requested twice within the same token. Same-token publication is prevented by
the stored `vram_eligible_after_call = call_tick + 1` causal threshold and by
the absence of direct SSD-to-VRAM transitions, not by requiring an artificial
wait event.

Known final counters from existing evidence that 0053 must preserve as context:

| Evidence | Final known counters / facts |
|---|---|
| G75b | sidecar disabled; sidecar observed false; sidecar route calls 0; expected and observed SHA equal. |
| G76b layer-3-only | mixed calls 2; hot main 10; cold IQ1 2; sidecar selected loads 2; failures 0; output `Hello!`. |
| G76b all layers | mixed calls 387; hot main 1935; cold IQ1 387; sidecar selected loads 387; failures 0; output SHA `9de63ea52caf541b1868bbe20f53e2f0bd610ddc0d020facd4a8f582c6d0f00e`. |
| G86 IQ1_S | clean `n=3`; IQ1 RAM cache 8 GiB; 87.71% RAM hits; 41.112 GiB SSD avoided; no L0-L3 grade. |
| G93 | planner 640/640; wait 0.085 ms; router D2H 6.302 ms; metadata 3.624 ms; failures 0; output exact to G87/G91; timing contaminated. |
| G94 planner off | clean `n=3`; IQ1 RAM cache 4 GiB; RAM hits 54.03%; every repeat SHA `e856d9ea88cd1c04f38cecee8b2ecb185f382a35b62495f3bc309fc339c1c004`. |
| G94 planner on | clean `n=3`; same cache; planner 10240/10240; wait 1.363 ms; failures 0; same repeat SHA as planner off. |
| G97 closed router | first structural attempt had no true cold promotions. |
| G97 forced eviction | failed closed on a snapshot miss; useful fail-closed evidence only, not a promotion pass. |
| G97 open router attempts | two attempts refused before launch by quiescence because Windows `ScheduledDefrag` was saturating `D:`. Final raw arena12 4+8 run completed after warmup with 9 reclaims/67 `cold_to_2bit`; measured structural PASS observed 56 reclaims/141 `cold_to_2bit`, with zero failures, zero forbidden transitions, and zero RAM skips. This is structural `n=1` evidence only and not a runtime, performance, quality, or default-readiness verdict. |
| G98 packed-copy attempt | fail-closed attempt only: `RoutePackedCopy` refused heterogeneous gate/down byte sizes `2162688/2752512`; decode ended at `gen=0`; no timing, exactness, quality, SOTA, or default-readiness claim. |
| G98 promotion-off after packed-copy removal | failed at token 4 with `ram-required admitted=0`, `ram_admit_skips=1`, and `forbidden=1`; no claim. Root cause: reserve reclaim remained coupled to IQ1 promotion. |
| G98 shared open-router smoke | structural `n=1` PASS only with the shared 16-slot open-router pool; output `Hello! How can I assist you today`; content SHA `474f578084317359f9534bdc03b692d83ba6bd02095731cbfa6988ec7d72230e`; `general_backing_reclaims=54`, `ram_evictions=787`, `ram_admit_skips=0`, `failures=0`, `forbidden=0`, `cold_to_vram=0`; IQ1 promotion absent; `quality_eligible=false`, `sota=false`. The recorded `0.2895 t/s` is invalid structural timing and not a performance datum. |
| G98 fixed-order `n=3` | clean quiescent fixed-order measurement; both arms exact within-arm and cross-arm for 64 tokens with SHA `a90233233708ecfbc8eae0cd4a1edb82997e4257f48f9afd9498780991beb607`; control `0.598759 t/s` range `0.592443-0.602928`, server decode `0.68`, TTFT `13.164667`; candidate `0.301633 t/s` range `0.165839-0.488288`, repeats `0.488288/0.250772/0.165839`, server decode `0.323333`, TTFT `13.164`; deltas `-49.623638%` harness and `-52.451029%` server decode. Raw harness marked quality/SOTA eligible, but parent verdict withholds final performance judgment due fixed order and makes no long-form L0-L3 quality claim. |
| G100 gate sweep | structural `n=1` only; prompt `Hi`, temp 0, no think, 16-token warmup plus 16-token measured request, context 256, arena 12 GiB, IQ1_S RAM cache 1 GiB, layers 3..42, open router, GPU planner on, 16 promotion slots, route packed copy off. All five arms produced SHA `cd153d3c18e782c4f4b3ceec574adccc8e68bc557110b0bc263b01e09bfcc8ef`, with zero promotion failures, zero direct SSD-to-VRAM rejects, zero forbidden cold-to-VRAM transitions, zero tier failures, and zero RAM-admit skips. `min_weight=0.02` alone skipped zero candidates; the combined gate promoted 16 of 312 candidates and read 113,246,208 IQ2 SSD bytes in 0.1327447 s versus 2,208,301,056 legacy bytes, a 94.87% byte reduction with identical output. No t/s, performance, quality, SOTA, or default-readiness claim. |

## Capacity arithmetic

Measured sidecar and routed-payload arithmetic from the IQ1_S Windows matrix:

| Quantity | Bytes | GiB |
|---|---:|---:|
| real complete IQ1_S sidecar | 61,540,805,344 | 57.31 |
| eligible layers 3-42 IQ1_S routed pool | 50,331,648,000 | 46.875 |
| hypothetical all-43-layer IQ1_S pool at same expert footprint | 54,106,521,600 | 50.39 |
| corresponding routed IQ2/Q2 payload | 72,477,573,120 | 67.5 |

The eligible layers 3-42 pool is:

```text
40 layers * 256 experts * 4,915,200 bytes = 50,331,648,000 bytes = 46.875 GiB
```

The corresponding IQ2/Q2 routed payload is:

```text
40 layers * 256 experts * 7,077,888 bytes = 72,477,573,120 bytes = 67.5 GiB
```

Eligible saving:

```text
67.5 GiB - 46.875 GiB = 20.625 GiB
20.625 / 67.5 = 30.56%
```

The real complete sidecar is larger than the eligible routed pool because it
includes non-routed tensors, metadata, and layer-specific layout. Do not use
the complete file size as the RAM target for this runtime path.

## G94 interpretation

G94 is a clean short mechanical/performance result for the GPU planner, not a
quality result.

| Arm | Protocol | Server decode t/s | Total t/s | TTFT | Exactness |
|---|---|---:|---:|---:|---|
| planner off | clean `n=3`, arena 20 GiB, IQ1 RAM cache 4 GiB | 2.0733 | 1.543 | 10.588 s | exact repeated SHA |
| planner on | clean `n=3`, otherwise identical | 2.2233 | 1.630 | 10.458 s | same repeated SHA |

Measured decode delta:

```text
(2.2233 - 2.0733) / 2.0733 = +7.23%
```

This establishes that the planner surface can improve this short measured
mechanical path while preserving deterministic output for the run. It does not
establish quality equivalence, long-output stability, or a SOTA claim.

G95 quality is still not complete. The long-output quality campaign was
stopped or deferred after Defrag/cleanup contamination, so no G95 L0-L3 verdict
may be used for 0053 promotion.

## Gate G97

G97 is the first gate for this design. It is `n=1` structural only and is not
allowed to produce a runtime, performance, quality, or SOTA verdict.

Current G97 state:

- the first closed-router run had no true cold promotions;
- a forced-eviction attempt failed closed on a snapshot miss;
- two open-router attempts were refused before launch by quiescence because
  Windows `ScheduledDefrag` was saturating `D:`;
- the final raw arena12 4+8 open-router/probation run completed with the
  intended mixed 5+1 shape and pre-reserved 16-slot probation pool;
- warmup observed 9 backing-RAM reclaims and 67 `cold_to_2bit` promotions;
- measured execution observed 56 backing-RAM reclaims and 141 `cold_to_2bit`
  promotions;
- failures, forbidden transitions, and RAM skips were all zero in that raw run;
- machine-readable final G97 evidence is preserved for the structural pass;
- the final receipt-locked rerun used ds4-win commit `3f6dab1`, executable
  SHA-256 `39632000cc6b529948750c0a1ae7ef8ad23201791c4694e462af5055ca25c0fe`,
  and completed Release build plus CTest `1/1` PASS;
- the final rerun reconciled `208` exact IQ2/Q2 stages and `65` backing-RAM
  reclaims, with `0` promotion failures and `0` forbidden direct
  SSD-to-VRAM transitions; its structural output was
  `Hello! How can I assist you today`;
- model and IQ1_S receipt reuse was identity-bound and read-locked for this
  structural gate only. G98/G99 benchmark and quality gates deliberately use
  full-file SHA-256 instead of receipt reuse;
- therefore G97 is a structural `n=1` PASS, but has no runtime, performance,
  quality, SOTA, or default-readiness verdict.

### G97a: structural `n=1`

Purpose: prove the runtime contract and fail-closed counters.

Required:

- opt-in enabled with `DS4_IQ1_PROMOTION_PROBATION_SLOTS=16`;
- router emits six routes per eligible layer/token;
- exactly five routes use IQ2/Q2 and one minimum-weight route uses IQ1_S;
- exact selected IDs and weights are preserved;
- IQ2/Q2 staging is requested only after IQ1_S compute is queued;
- same-token VRAM publication from probation is structurally blocked by the
  eligibility threshold;
- at least one next-token eligibility event is observed or the absence is
  explained by the prompt/token trace;
- `forbidden_cold_ssd_to_vram=0`;
- all failure counters required zero above are zero;
- output, logs, source hashes, env, and final telemetry are preserved.

Allowed claim: structural mixed-tier and deferred-promotion safety for this
single invocation only, including the observed reclaim bookkeeping.

Forbidden claims from G97a:

- quality;
- SOTA;
- 10 t/s;
- general speed win;
- lossless IQ1_S behavior;
- promotion to default.

### G97b: clean perf/exactness `n>=3`

Only after G97a passes:

- run matched planner/probation off/on arms with same binary, model, prompt,
  seed/sampling, context, cache, arena, host state, and output budget;
- require exact repeated outputs where the arm is intended to be deterministic;
- report TTFT, prefill, server decode t/s, total t/s, p95 token latency,
  sidecar loads, RAM hits, SSD bytes avoided, probation stage/publish counters,
  H2D bytes, failures, and memory minima;
- invalidate timing under Defrag, low-memory cleanup, changed cache size,
  changed prompt, or changed route policy.

Allowed claim: prompt-scoped clean performance/exactness measurement, if the
run is uncontaminated and counters reconcile.

This clean `n>=3` work was carried into G98. G98 now has a clean fixed-order
exactness/performance measurement, but it does not support a promotion
performance win. G99 is deferred until promotion churn is fixed.

### Arena 14 caution

Arena14 is not benchmark-safe for this design state. It can fall below 1 GiB
available memory, so any run under that condition must not be used for clean
benchmark timing, throughput, or promotion-readiness claims. Arena12 raw 4+8 is
the relevant completed structural PASS above, and it remains limited to `n=1`
structural interpretation even with the machine-readable final G97 record
preserved.

### G97c: human L0-L3 grading

Only after structural and clean perf/exactness gates:

- run `n>=3` per arm on the accepted quality prompt suite;
- preserve every raw output;
- record human L0-L3 grade for every output;
- reject on any paired L0-L3 loss, attributable loop/tag-salad/truncation, or
  silent fallback that hides IQ1_S exposure.

No verdict comes from `n=1`, repeat flags, exact hashes, or route counters.

## Gates G98-G101

G98/G99 are reserved for clean `n>=3` follow-up evidence after G97 structural
wiring is accepted. The first G98 control attempt failed closed before emitting
any token: the inherited `RoutePackedCopy` lever rejected the heterogeneous
gate/down expert byte sizes (`2162688/2752512`) and CUDA decode ended at
`gen=0`. That attempt is invalid for timing, exactness, quality, SOTA, or
default-readiness claims. It establishes only that the current packed-copy
implementation is not composable with this IQ1 promotion layout.

After removing the packed-copy path, the next G98 promotion-off attempt still
failed closed at token 4. It reported `ram-required admitted=0`,
`ram_admit_skips=1`, and `forbidden=1`. This is not a usable control,
exactness, performance, quality, or SOTA result. The measured root cause is
that reserve reclaim remained coupled to IQ1 promotion, so promotion-off could
not exercise the intended shared exact-routing path.

The following shared open-router run moved reclaim into the general 16-slot
open-router pool and is a structural smoke PASS for `n=1` only. It produced:

- output: `Hello! How can I assist you today`;
- content SHA-256:
  `474f578084317359f9534bdc03b692d83ba6bd02095731cbfa6988ec7d72230e`;
- executable SHA-256:
  `87ed7f395f564dd97acaaeea927e39ac2ce72d3fa3181c3734e0b1b6da1e764a`;
- ds4_cuda SHA-256:
  `668fc9b8284616d81619c3dfe6a5d2e9504be168f8112d4583393518d5d95ff9`;
- `general_backing_reclaims=54`;
- `ram_evictions=787`;
- `ram_admit_skips=0`;
- `failures=0`;
- `forbidden=0`;
- `cold_to_vram=0`;
- IQ1 promotion absent;
- `quality_eligible=false`;
- `sota=false`.

The recorded `0.2895 t/s` is invalid structural timing. It must not be used as
a performance result, slowdown result, speedup result, quality proxy, or SOTA
ledger entry. This smoke proves only that the shared open-router pool can run
this one invocation without RAM-admit skips, forbidden transitions, promotion
failures, or direct cold-to-VRAM publication.

The clean G98 fixed-order `n=3` measurement then completed under quiescent host
conditions. Both arms produced identical outputs within each arm and across
arms for the 64-token run, with content SHA-256
`a90233233708ecfbc8eae0cd4a1edb82997e4257f48f9afd9498780991beb607`.

| Arm | Harness t/s | Repeat range / values | Server decode t/s | TTFT |
|---|---:|---|---:|---:|
| Control | 0.598759 | 0.592443-0.602928 | 0.68 | 13.164667 s |
| Candidate | 0.301633 | 0.488288 / 0.250772 / 0.165839 | 0.323333 | 13.164 s |

Measured deltas:

```text
harness:       -49.623638%
server decode: -52.451029%
```

Candidate telemetry:

- `cold_observed=10240`;
- `cold_existing_2bit=4436`;
- `cold_to_2bit_ram=5804`;
- primary IQ2 SSD promotion reads: `38.259 GiB` over `61.9473 s`,
  approximately `632.4 MiB/s`;
- `general_backing_reclaims=576`;
- `iq1_backing_reclaims=576`;
- `failures=0`;
- `direct_ssd_to_vram_rejected=0`;
- `forbidden=0`;
- `cold_to_vram=0`.

Control telemetry:

- `general_backing_reclaims=556`;
- `ram_admit_skips=0`.

The raw harness marked the clean run `quality_eligible=true` and
`sota_eligible=true`, but the parent final verdict is deliberately narrower:
this is exactness evidence and a fixed-order performance measurement only.
There is no long-form L0-L3 quality claim, no quality-equivalence claim, no
SOTA claim, and no performance-win claim. Because the candidate is much worse
even with warmer second-arm cache, G99 is deferred until the promotion churn is
fixed.

The wrapper summary initially rejected the aggregate result because it compared
`reserved_slots=64` against an erroneous expected `16`; the correct aggregate
expectation is `16 * 4 = 64`. The summary was fixed to use the aggregate
expectation, and Resume validated the corrected summary without rerunning GPU
work.

Next design direction: IQ1 probation must not promote every observed cold
route. Promotion needs confirmation or a second touch, a mass/weight threshold,
and a bounded promotion budget before it can plausibly improve performance.

### G100: promotion gate isolation sweep

G100 is a structural `n=1` mechanical sweep of IQ1 promotion admission levers.
It used prompt `Hi`, temperature 0, no think, one 16-token warmup and one
16-token measured request per arm, context 256, dynamic arena 12 GiB, IQ1_S
RAM cache 1 GiB, layers 3..42, mixed cold-one routing, GPU planner enabled,
open-router prefill compose, `DS4_CUDA_PREFILL_TIER_RESERVE_SLOTS=16`,
`DS4_IQ1_PROMOTION_PROBATION_SLOTS=16`, and `RoutePackedCopy` disabled.

Source evidence:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g100_iq1_promotion_gate_sweep_result.json`
plus the per-arm `result_path` JSON files recorded in that summary.

All arms produced identical output SHA-256:
`cd153d3c18e782c4f4b3ceec574adccc8e68bc557110b0bc263b01e09bfcc8ef`.
All arms had `promotion_failures=0`,
`promotion_direct_ssd_to_vram_rejected=0`,
`tier_forbidden_cold_ssd_to_vram=0`, `tier_cold_to_vram=0`,
`tier_failures=0`, and `ram_admit_skips=0`.

| Arm | Config | IQ2 SSD bytes | IQ2 SSD GiB | IQ2 SSD GiB/s | `cold_to_2bit` | Skips touches/weight/mass/request/window | Failures | Byte reduction vs legacy |
|---|---|---:|---:|---:|---:|---|---:|---:|
| `legacy` | touches=1; weight=0; mass=0; request_budget=0; window=0/0 | 2,208,301,056 | 2.057 | 0.673 | 312 | 0/0/0/0/0 | 0 | 0.00% |
| `confirm-only` | touches=2; weight=0; mass=0; request_budget=0; window=0/0 | 424,673,280 | 0.396 | 0.777 | 60 | 252/0/0/0/0 | 0 | 80.77% |
| `budget-only` | touches=1; weight=0; mass=0; request_budget=16; window=40/1 | 141,557,760 | 0.132 | 0.654 | 20 | 0/0/0/0/292 | 0 | 93.59% |
| `weight-only` | touches=1; weight=0.02; mass=0; request_budget=0; window=0/0 | 2,208,301,056 | 2.057 | 0.662 | 312 | 0/0/0/0/0 | 0 | 0.00% |
| `combined` | touches=2; weight=0.02; mass=0; request_budget=16; window=40/1 | 113,246,208 | 0.105 | 0.795 | 16 | 252/0/0/0/44 | 0 | 94.87% |

Interpretation: G100 shows the admission levers can reduce structural IQ2 SSD
promotion traffic without breaking the fail-closed counters on this short
surface. It also shows that `min_weight=0.02` alone filtered zero candidates in
this run. The combined arm promoted 16 of 312 candidates, read 113,246,208
IQ2 SSD bytes in 0.1327447 s, and produced identical output. This remains a
structural `n=1` result only: there is no t/s claim, no performance claim, no
quality or L0-L3 claim, no SOTA claim, and no default-readiness claim.

### G101: combined-vs-legacy receipt attempts

G101 has no combined-vs-legacy verdict and no timing data that can be used.
The ledger records the failure modes and fixes only:

- First attempt: the combined `n=3` arm completed, but the suite stopped before
  a combined-vs-legacy verdict because of a deduplicated binding-hash bug.
  This invalidates any cross-arm conclusion. Fix: ds4-win commit `876b4b3`.
- Second attempt: the receipt run stopped because `ComputeHash` performed 4 KiB
  reads and reduced `D:` throughput to roughly 3-6 MiB/s. Fix: commit
  `2a5e696`, using 8 MiB `SequentialScan`/`IncrementalHash` reads. The
  follow-up 1 GiB `D:` microbench measured 97.5 MiB/s with 128 reads, and the
  receipt tests passed.
- Third attempt: full hashing completed, but the benchmark was correctly
  rejected by quiescence because Windows `ScheduledDefrag`/`Defrag.exe` was
  active on `D:`. There is no valid timing result. Fix: commit `91f1445`
  preserves the preflight and retries quiescence without rehash; other errors
  remain fail-closed.

Do not infer a G101 performance, quality, SOTA, default-readiness, or
combined-vs-legacy verdict from these attempts.

## Non-goals

- No baked mixed model.
- No replacement of the authoritative IQ2/Q2 primary source.
- No IQ1_S VRAM cache.
- No same-token IQ1_S-to-IQ2 format switch.
- No direct SSD IQ2-to-VRAM compose path.
- No quality verdict from G94.
- No use of contaminated G95 attempts as evidence.
- No 10 t/s claim.

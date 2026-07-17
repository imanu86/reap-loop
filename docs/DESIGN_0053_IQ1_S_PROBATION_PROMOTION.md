# Design 0053: IQ1_S probation promotion, runtime opt-in

**Status:** runtime candidate implemented on an experimental branch,
2026-07-17; structural and quality gates are still pending. This file makes
no quality, SOTA, or 10 t/s claim.

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
  router computes 6 selected routes
  5 higher-weight routes execute from IQ2/Q2
  1 minimum-weight route executes from IQ1_S sidecar

after that compute is queued:
  exact IQ2/Q2 source for the same expert is staged into pinned RAM probation
  the staged slot cannot enter VRAM for this token

next token or later:
  if the expert gains weight and exits the minimum route, use the staged IQ2/Q2
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
| G75 | Real expert IQ1_S component validation and transport measurement only. |
| G76b | Structural mixed-route counters for current-token mixed IQ2/IQ1 execution. |
| G86 | Clean `n=3` transport/perf baseline for IQ1 RAM cache, no L0-L3 grade. |
| G89-G93 | Structural planner/cache profiling only; G92/G93 timing contaminated by `ScheduledDefrag`. |
| G94 | Clean short `n=3` planner mechanical/perf comparison with exact repeated outputs, no L0-L3 grade. |
| G95 | Quality campaign not complete; first attempt/probe contaminated or invalidated by Defrag/cleanup state. |

## Runtime contract

0053 is off by default and enabled only when all required sidecar, planner, and
probation flags are present. With the opt-in unset, behavior must remain the
accepted main IQ2/Q2 runtime.

Per routed layer and token:

1. The router computes the normal 6 selected routes and keeps their exact IDs
   and gate weights.
2. The route with the minimum weight is marked `cold_iq1_current_token`.
   Ties follow the same deterministic first-minimum rule as the existing CPU
   fixture and GPU planner evidence.
3. The other five routes are `hot_iq2_current_token` and use the main exact
   IQ2/Q2 path.
4. The IQ1_S route reads from the sidecar for this token only. It does not
   rewrite the selected IDs, gate weights, or hot route count.
5. After the IQ1_S compute for that expert has been queued, the corresponding
   exact IQ2/Q2 expert is staged from the authoritative primary source into a
   pinned RAM probation slot.
6. A probation slot is not eligible for VRAM publication until the next token.
   There is no same-token format switch.
7. On a later token, if that expert is selected and is no longer the minimum
   route, the runtime may use the staged exact IQ2/Q2 copy. If the expert is
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
promotion in compose, and no bake into a mixed model file.

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

## Probation slots

New environment variable:

```text
DS4_IQ1_PROMOTION_PROBATION_SLOTS
```

Initial meaning:

- unset or `0`: disabled;
- positive integer: number of exact IQ2/Q2 pinned RAM probation slots;
- first structural point: `16`.

The first point is intentionally small. One exact routed expert triplet is
6.75 MiB, so 16 slots are:

```text
16 * 6.75 MiB = 108 MiB
```

This is only the first mechanism point, not a proposed final cache size.

Implemented first-slot policy:

- reclaim the configured number of lowest-prefill-mass slots from the
  published exact snapshot, preserving the same pinned-arena allocation;
- stage only after the current-token IQ1_S route has been enqueued;
- if no probation slot can be reclaimed, record a promotion failure and
  continue serving the already queued current token; the run then fails the
  promotion gate;
- do not evict a slot that is in-flight to VRAM or under validation;
- use a deterministic replacement rule among idle probation slots for
  reproducible G97 evidence.

## Telemetry

Existing mixed-IQ1 and sidecar telemetry continues to provide route counts,
selected-load counts/failures, planner calls/failures, sidecar identity, and
IQ1 RAM-cache traffic. The new final line is:

```text
[iq1-promotion] final requested_slots=... reserved_slots=...
snapshot_evictions=... cold_observed=... cold_existing_2bit=...
cold_to_2bit_ram=... probation_ram_hits=... next_token_waits=...
promotion_2bit_ssd_bytes=... promotion_2bit_ssd_seconds=...
direct_ssd_to_vram_rejected=... failures=...
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

G97 is the first gate for this design. It is not allowed to produce a quality
or SOTA verdict.

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
single invocation only.

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

### G97c: human L0-L3 grading

Only after structural and clean perf/exactness gates:

- run `n>=3` per arm on the accepted quality prompt suite;
- preserve every raw output;
- record human L0-L3 grade for every output;
- reject on any paired L0-L3 loss, attributable loop/tag-salad/truncation, or
  silent fallback that hides IQ1_S exposure.

No verdict comes from `n=1`, repeat flags, exact hashes, or route counters.

## Non-goals

- No baked mixed model.
- No replacement of the authoritative IQ2/Q2 primary source.
- No IQ1_S VRAM cache.
- No same-token IQ1_S-to-IQ2 format switch.
- No direct SSD IQ2-to-VRAM compose path.
- No quality verdict from G94.
- No use of contaminated G95 attempts as evidence.
- No 10 t/s claim.

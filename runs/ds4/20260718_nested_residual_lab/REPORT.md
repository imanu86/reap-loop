# DS4 full-router nested base/residual gate

Date: 2026-07-18

Status: CPU-only measured design gate. No runtime, speed or generation-quality
claim.

## Why this gate exists

G112 proved that a truly open router with all routed experts resident as Q1_0
can remove SSD traffic and reach 6.76 server decode t/s. It also produced L0.
G115/G116 then mixed Q1_0 and IQ2 behind a request-scoped closed mask with only
55.66% gate-mass coverage, so those runs cannot isolate representation quality
from expert-selection damage.

The next runtime must keep the full router authoritative. Q1 cannot replace a
correctly selected expert for long stretches. It can only be a resident base
from which the original IQ2 expert is reconstructed, or a bounded one-token
fallback for a low-weight cold route while that exact reconstruction is
promoted.

## Exact nested format

The split preserves bytes from the authoritative IQ2/Q2_K model rather than
requantizing it independently.

| Tensor format | Native block | Resident base | Exact residual | Base contents |
|---|---:|---:|---:|---|
| IQ2_XXS | 66 B / 256 weights | 34 B | 32 B | fp16 scale, group scales, sign codes |
| Q2_K | 84 B / 256 weights | 52 B | 32 B | scales/mins plus the high quant bitplane |

For one routed expert this is 3.75 MiB base plus 3.00 MiB residual, exactly the
original 6.75 MiB. For the 10,240 active routed experts in layers 3 through 42:

- resident base: 37.50 GiB;
- all residuals: 30.00 GiB;
- reconstructed original: 67.50 GiB.

The intended storage layout is therefore not two complete resident copies.
The base is materialized once in host RAM at startup. A bounded cache holds
residuals for hot experts, while the remaining packed 3 MiB residuals stay on
NVMe. A cold exact miss reads 44.44% of the original expert bytes.

## Real-model measurement

Command family:

```powershell
python scripts/ds4_nested_residual_lab.py C:\ds4-models\ds4-2bit.gguf `
  --ds4-root <ds4-win-work> --blocks 768 --dot-vectors 8 --seed <seed>
```

Seeds: 20260718, 20260719 and 20260720. Each sampled 768 real routed-expert
blocks. Across all 2,304 blocks, joining base and residual reproduced the native
block byte-for-byte; reconstruction failures were zero.

| Proxy | Seed 20260718 | Seed 20260719 | Seed 20260720 |
|---|---:|---:|---:|
| IQ2_XXS base weight nRMSE mean | 0.60395 | 0.60411 | 0.60468 |
| IQ2_XXS base dot nMAE mean | 0.03020 | 0.03020 | 0.02978 |
| Q2_K base weight nRMSE mean | 0.54372 | 0.54202 | 0.54501 |
| Q2_K base dot nMAE mean | 0.02638 | 0.02710 | 0.02724 |

The base-only proxy is not materially better than the existing independent Q1
candidate. It is not a quality solution by itself. Its value is that the 3 MiB
residual reconstructs the exact original expert and cuts cold-source bytes by
55.56%.

## G117 one-layer runtime safety

G117 is the first measured runtime structural safety check for the nested
residual path. It is `n=1` only, with the full/open router enabled and every
mask off. The prompt was `Hi`, `max_tokens=8`, `ctx=256`.

The control and candidate produced the same output SHA-256:
`8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`.
This establishes byte-identical control/candidate output for this small safety
case only.

Measured structural counters:

| Counter | Value |
|---|---:|
| router_calls | 9 |
| cache_hits | 9 |
| cache_misses / preads / reconstructed | 62 |
| residual_bytes | 195,035,136 |
| h2d_bytes | 502,530,048 |
| mismatch / failures | 0 |

The control server decode was 1.55 t/s and the candidate server decode was
1.40 t/s, but `timing_claim_valid=false`. There is no performance verdict and
no SOTA claim from G117.

## G118 four-layer exactness with invalid memory contrast

G118 covered routed layers 3, 16, 29 and 42. Control and candidate output SHA
remained identical. Runtime counters were `router_calls=36`, `misses=268`,
`residual_preads=268`, `residual_bytes=843055104`, `reconstructed=268`,
`h2d_bytes=1896873984`, `mismatch=0`, `failures=0`.

The candidate pinned `4026531840` base bytes, after which registration of the
28 GiB source window failed with `CUDA host window register skipped: out of
memory`; the control had registered 28 GiB. G118 therefore proves distributed
exactness but is not a valid timing A/B.

## G119 four-layer exactness with compatible host budget

G119 repeated G118 with `BudgetGB=24`. Both control and candidate registered a
24 GiB contiguous source window. The candidate additionally pinned the four
base layers (`3.75 GiB`, host-pinned, not device-mapped), preserved the same
output SHA and repeated the exact G118 transport counters with zero mismatch
and zero failure.

This is the first measured physically compatible nested-residual configuration:
24 GiB source window plus 3.75 GiB nested base. It remains an `n=1` structural
gate with `timing_claim_valid=false`.

Pre-test review blockers fixed before the measurement: pinned slot event reuse,
hard fail closed on nested errors, mandatory open-router mode, parser overflow
checks, full sidecar hash lock and per-used-expert reconstruction verification.

## G120-G121 G73-composite open-router safety

G120 first corrected an important provenance error in the historical
comparison. G73 used a request-scoped closed transport set after prefill; it
was not a full/open decode control. The G120 control therefore retained the
G73 transport levers but explicitly set `request-scoped-open`, reserved 32
composition slots and used a 26.25 GiB primary arena. The four-layer candidate
added 3.75 GiB of resident nested base and excluded 384 ranked nested entries
from the primary arena. Both arms produced the same coherent 64-token output,
SHA-256
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
That output differs from historical closed-transport G73 SHA
`31cbc6504dcb57d42aeff9dbceb3aed943bcb32dae19a2edbf552e9fd2f52eb8`.

G120 used the default six-entry nested exact cache. It was an intentionally
single-process structural gate, not a performance verdict:

| Metric | Open IQ2 control | Nested exact, cache 6 |
|---|---:|---:|
| End-to-end t/s | 0.650419 | 0.487870 |
| Server decode t/s | 1.42 | 0.88 |
| Server prefill / TTFT | 52.864 s | 58.299 s |
| Load | 10.2 s | 29.1 s |
| Aggregate disk read | 59.287 GiB | 67.853 GiB |
| Nested cache hits / misses | n/a | 0 / 1,802 |
| Nested residual bytes | n/a | 5,668,601,856 |
| Nested H2D bytes | n/a | 12,754,354,176 |
| Reconstruction mismatches / failures | n/a | 0 / 0 |

Source inspection explained the zero-hit result: the six-entry cache holds one
top-six route only. With four covered layers, every layer evicted the complete
working set of the previous layer before the next token.

G121 changed only the nested exact-cache capacity from 6 to 64. It preserved
the same output SHA and zero mismatches/failures. Measured counters were 817
hits and 985 misses over the same 1,802 exact expert uses. Derived arithmetic:
hit rate and miss reduction were both 45.34%; residual reads fell to
3,098,542,080 bytes. End-to-end rate rose to 0.553980 t/s (+13.55% versus the
G120 cache-6 candidate) and server decode rose to 1.12 t/s (+27.27%). It still
trailed the single G120 open control by 14.83% end-to-end and 21.13% server
decode. H2D remained 12,754,354,176 bytes because every nested cache hit still
uploaded the full 6.75 MiB expert; the reconstructed exact entries were not
admitted to the existing GPU-resident SplitFused cache.

The causal next gate is therefore not an `n>=3` repetition of G121. First,
nested exact entries must reuse the existing VRAM route cache so a hot exact
hit avoids both residual IO and repeated H2D. Only after a new `n=1` exactness
gate observes nonzero nested VRAM hits and lower H2D may a clean `n>=3` A/B be
run. No SOTA or generalized quality claim is attached to G120 or G121.

## G122 exact GPU-resident reuse safety

G122 connected reconstructed native experts to the existing SplitFused GPU
route cache behind an opt-in, fail-closed runtime gate. The router remained
full/open and the output SHA stayed exactly
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
The candidate used a 25.828125 GiB primary arena plus 3.75 GiB nested base and
64 exact-cache entries (0.421875 GiB), preserving the 30 GiB total host budget.

Measured structural counters:

| Counter | Value |
|---|---:|
| nested GPU route calls | 256 |
| nested VRAM hits / misses | 541 / 995 |
| nested route host fills | 995 |
| nested route host bytes | 7,042,498,560 |
| nested route H2D bytes | 7,042,498,560 |
| residual preads / bytes | 869 / 2,733,637,632 |
| reconstruction mismatches / failures | 0 / 0 |

The dedicated route-cache H2D subset was 7,042,498,560 bytes. The comparable
total nested H2D counter, including prefill selected-load traffic, fell from
12,754,354,176 bytes in G121 to 8,925,216,768 bytes in G122, a measured
30.02% reduction. The one-process timing was 0.559345 end-to-end t/s,
1.16 server decode t/s, 58.849 s prefill/TTFT and 21.9 s load. These timings
are `n=1` signals only; G122 makes no SOTA claim. The next valid performance
gate is a clean equal-budget `n>=3` full/open control/candidate A/B.

## Frozen runtime architecture

1. The router remains open and its original top-six IDs and weights are
   authoritative for every decode token. No request-scoped closed mask is used
   by the candidate.
2. A representation-neutral resolver checks the exact expert in this order:
   reconstructed IQ2 VRAM slot, exact residual cache plus base, then residual
   fetch from the authoritative packed source.
3. Hot exact experts use the existing IQ2 GPU-resident SplitFused path. Their
   base is not also dispatched, and a VRAM hit performs no repeated host H2D.
4. A cold high-weight route waits for its residual and executes exact IQ2 in
   the same token.
5. Only the lowest-weight cold route may use the base for the current token,
   while its residual is promoted for the next token. This 5+1 policy is a
   separate quality arm, not the correctness baseline.
6. REAP mass/LFRU controls residual-cache retention, not router eligibility.
   SPEX may later prefetch residuals, after the exact-first transport works.

The mixed kernel must share one input quantization and one output join across
all six routes. The current G115 implementation calls `routed_moe_launch`
separately for the five IQ2 routes and the one Q1_0 route, quantizes the input
again, disables SplitFused for Q1_0 and joins a second output buffer. Its 5.55
t/s therefore includes composition overhead and is not the expected ceiling of
the nested format.

## Implementation order

1. Build and validate a packed residual sidecar for one layer. Keep the base
   derived in RAM from the original GGUF; do not create another 37.5 GiB file.
2. Add one-layer exact reconstruction that feeds the existing IQ2 kernels.
   Its output must be byte-identical to the env-off control.
3. Add the native base kernels and validate them against the CPU reference.
4. Fuse mixed execution around one Q8 input quantization, five IQ2 routes and
   at most one base route, with one GPU-resident output join.
5. Before extending coverage, admit reconstructed four-layer exact entries to
   the existing VRAM cache and verify nonzero VRAM reuse plus reduced H2D.
6. Extend to all routed layers with a fixed host-memory budget: 37.5 GiB base,
   bounded residual cache, existing VRAM cache.
7. Measure an open-router all-exact residual arm first. Then test open-router
   5+1 where only the lowest-weight cold route may use base-only compute.

## Quality and performance protocol

- Cyberpunk HTML, `--nothink`, temperature zero, context 8192, max 2000.
- Safety `n=1` may only reject a malformed candidate. A positive verdict needs
  at least three independent processes.
- Retain raw outputs and grade L0-L3. Repeat flags are not verdicts.
- Record TTFT, prefill, server decode, end-to-end rate, GPU utilization,
  base/residual hits, residual SSD bytes, H2D bytes, promotion delay and every
  fallback.
- Compare against a true open-router IQ2 control and the measured G73 short
  closed-snapshot SOTA. Do not label either one as the other.

## Stop conditions

- Any router-ID or router-weight change.
- Any reconstruction mismatch against the original IQ2 bytes.
- Any hidden request-scoped closed mask in the candidate.
- L0/L1 in the first long-form 5+1 safety output.
- Residual transfer or synchronization costs erase the byte reduction.

## G123 equal-host-budget full/open A/B

G123 completed three independent control processes and three independent
candidate processes. Every accepted process reproduced output SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`,
passed system quiescence, and recorded zero runtime-contamination samples.
The 64-token deterministic output is an exact transport check, not a
generalized L3 quality claim.

Both arms had a 30 GiB host budget. The control used a 30 GiB primary arena.
The candidate used a 25.828125 GiB primary arena, 3.75 GiB nested base and
0.421875 GiB for 64 exact-cache entries.

| Metric | Control n=3 | Candidate n=3 | Candidate delta |
|---|---:|---:|---:|
| End-to-end t/s mean | 0.650261 | 0.555497 | -14.57% |
| End-to-end t/s median | 0.647540 | 0.559217 | -13.64% |
| Server decode t/s mean | 1.650000 | 1.163333 | -29.49% |
| Server decode t/s median | 1.650000 | 1.160000 | -29.70% |
| Prefill / TTFT mean | 59.217 s | 59.803 s | +0.585 s |
| Load mean | 11.222 s | 29.212 s | +17.990 s |
| Request seconds mean | 98.426 s | 115.228 s | +16.802 s |

Each candidate process reproduced the same counters: 256 covered-layer route
calls, 541 GPU-cache hits, 995 misses, 995 host fills, 7,042,498,560
route-cache H2D bytes, 869 residual reads, 2,733,637,632 residual bytes and
8,925,216,768 total nested H2D bytes. The covered-route GPU hit rate was
35.22%. Reconstruction mismatches, nested failures and GPU-cache failures
were all zero.

One initial control-r3 attempt was excluded before aggregation. Its arena
source copy took 134.654 s instead of about 32 s and the GPU route worker
timed out at sequence 1882. After a full cooldown, the replacement control-r3
completed exact and uncontaminated at 0.656230 end-to-end t/s and 1.67 server
decode t/s. The excluded attempt is not one of the six rows in the receipt.

Verdict: exact nested GPU reuse reduces traffic but is not yet a performance
candidate. At equal host budget it loses 29.49% server decode and 14.57%
end-to-end. Do not extend the current host-reconstruct/full-expert-H2D path to
all layers. The next gate must profile and remove miss-path reconstruction,
handoff and full 6.75 MiB H2D cost, or raise protected GPU hit coverage,
before any broader nested catalog or 5+1 quality run.

## G124 nested residual causal profile

G124 is the requested follow-up profile of the already-negative G123 candidate
miss path. It is `n=1` causal profile evidence only: exact output was preserved
with SHA-256
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`, runtime
reconstruction verification stayed enabled, and there is no SOTA, performance
A/B or generation-quality verdict.

Authoritative receipt:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_receipt.json`.
Associated result:
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g124_nested_residual_profile_20260718T172630143Z_ea23d8bd87_result.json`.

Measured profile counters:

| Timer | Calls | Seconds |
|---|---:|---:|
| CPU reconstruct | 869 | 14.8898841 |
| Residual pread | 869 | 1.9070234 |
| Reconstruction verify | 2,607 | 0.4990679 |
| Host copy | 995 | 0.3460242 |
| H2D enqueue | 995 | 0.0381582 |
| H2D sync | 254 | 0.1677010 |
| H2D enqueue + sync | n/a | 0.2058592 |
| Route-ready wait | 256 | 14.6092232 |

These timers are stage-attribution signals, not a wall-clock decomposition.
They overlap across worker, route and transfer paths and must not be summed.
The profile pointed to the measured next lever: G125 moved the exact join to
the GPU side so a miss does not pay host-side reconstruct plus full native
expert upload before reuse. G125/G126 outcomes are summarized below and in
`G125_G126_RECEIPT.md`.

## G125-G126 GPU-side exact join

G125 passed the `n=1` structural safety gate for GPU-side exact join. It is not
a performance or quality verdict. The run preserved full/open routing with no
REAP/static/bake/request-scoped closed mask and reproduced exact content SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.

Its receipt is
`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work\g7_runs\g7_g125_nested_gpu_join_safety_current_build_clean_20260718T182935501Z_eb824ebedb_receipt.json`
with SHA-256
`ae15a6d3d3bc35e75b46befd8d18d7886f571e47d93561d146ada3ccf20f58fb`.
The safety run recorded 1,261 GPU join calls, positive base/residual H2D,
zero native H2D, zero CPU reconstruction, zero verification mismatches and
zero failures.

G126 then ran the repeated CPU-join versus GPU-join A/B with three independent
processes per arm. All six rows were exact, uncontaminated and had the same
content SHA. CPU join decoded at `1.15, 1.16, 1.15` t/s, mean `1.153333`.
GPU join decoded at `1.56, 1.58, 1.57` t/s, mean `1.57`, a measured
`+36.1272%` decode delta.

End-to-end mean also rose by `+49.7457%`, but TTFT/request/WRAP timing was
wildly noisy in the batch, with request seconds ranging from about 96.98 to
351.57 s. The defensible finding is therefore decode throughput; E2E is
retained only as batch-only/noisy. G126 improves the G123 nested miss path but
does not beat the G123 full/open IQ2 control at 1.65 t/s and must not be
compared as an absolute SOTA against G73, which is closed/request-scoped.

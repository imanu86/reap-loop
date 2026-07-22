# Full-router nested residual plan

The active DS4 Windows direction is the measured exact-first architecture in
`runs/ds4/20260718_nested_residual_lab/REPORT.md`.

## Corrections to the previous direction

- G73 is a full original GGUF with an unbiased prefill router, but decode uses
  a request-scoped closed mask (`kept=3783`, `pruned=6457`, coverage 0.5874).
  Its 4.986667 t/s result is a valid exact 64-token transport result, not an
  open-router or long-document quality result.
- G112 is the actual all-routed, mask-off Q1_0 transport ceiling: 6.76 t/s,
  zero misses and L0 at `n=1`.
- G115 and G116 use the same 55.66%-coverage closed mask. Their 5.55 and 8.08
  t/s values cannot establish full-router quality or SOTA.
- The runtime must never substitute a different expert merely because the
  selected one is cold. Selection correctness precedes precision.

## Active decision

Do not continue the independent Q1_0-as-dominant-compute path. Preserve the
full router and split each native expert into a resident nested base plus an
exact residual. Base plus residual reconstructs the original IQ2/Q2_K bytes.
Use mass/LFRU and later SPEX only to decide which residuals are promoted and
retained, never which experts the router is allowed to select.

## Next executable gate

G117-G121 completed one-layer and four-layer exact reconstruction plus the
first G73-composite open-router contrast. G120 proved exact output but exposed
zero reuse with the default six-entry exact cache. G121 increased the cache to
64 and measured 817 hits / 985 misses, recovering 13.55% end-to-end versus the
cache-6 arm, but every exact use still uploaded 6.75 MiB and the candidate
remained 14.83% behind its single open control.

The next gate is now a GPU-resident-cache integration safety run, not `n>=3`.
Reconstructed exact experts from the four distributed layers must be admitted
to and reused from the existing SplitFused VRAM cache. The gate requires the
same full/open output SHA as the control, nonzero nested VRAM hits, reduced
nested H2D, and zero reconstruction mismatch/failure. Only then run an
`n>=3` G73-composite open A/B. Only a positive transport result justifies the
all-layer residual catalog. The low-weight 5+1 fallback remains later work.

G122 has now passed that structural gate: exact output, 541 nested VRAM hits,
995 misses, 7,042,498,560 route H2D bytes and zero mismatch/failure. Including
prefill, total nested H2D was 8,925,216,768 bytes, 30.02% below G121. Its
single-process throughput is not a verdict. The immediate gate is therefore
the equal-host-budget `n>=3` full/open A/B; only after that result should
coverage extend beyond layers 3, 16, 29 and 42.

## G117 measured structural safety

G117 completed the one-layer nested-residual runtime safety gate as `n=1`
structural evidence only. The run used the full/open router with mask off,
prompt `Hi`, `max_tokens=8` and `ctx=256`. Control and candidate output were
identical with SHA-256
`8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`.

Runtime counters were: `router_calls=9`, `cache_hits=9`,
`cache_misses=62`, `preads=62`, `reconstructed=62`,
`residual_bytes=195035136`, `h2d_bytes=502530048`, `mismatch=0` and
`failures=0`. Control server decode was 1.55 t/s and candidate server decode
was 1.40 t/s, but `timing_claim_valid=false`; the row carries no performance
verdict and no SOTA claim.

The pre-test review blockers fixed before G117 were: pinned slot event reuse,
hard fail closed, mandatory open router, parser overflow, full sidecar hash
lock and per-used-expert reconstruction.

## G118-G119 distributed-layer capacity gate

G118 extended exact reconstruction to layers 3, 16, 29 and 42. It reproduced
the control output SHA exactly and recorded `router_calls=36`, `misses=268`,
`preads=268`, `reconstructed=268`, `mismatch=0` and `failures=0`. However,
the candidate first pinned 3.75 GiB of nested base and then failed to register
the requested 28 GiB source window with `out of memory`. G118 is exact but its
memory transport differs from its control, so its timing is invalid.

G119 repeated the same structural gate with a 24 GiB source window. The
candidate successfully registered the full 24 GiB window and pinned all four
base layers (`4026531840` bytes, mapped=0). It again reproduced output SHA
`8a17fc0dc61e8520bdbe3a735b000358a6476cbe9f0e3d86c54a51cf26b5d009`
with 268 exact residual reads, 843055104 residual bytes, 1896873984 H2D bytes,
zero mismatches and zero failures. This is still `n=1` structural evidence;
no timing or SOTA claim is attached.

The measured capacity rule is now explicit: nested base residency consumes the
same finite pinning budget as the source window. It must replace part of that
window, not be added on top of the previous 28 GiB configuration.

## G120-G121 measured reuse gate

The G120 open IQ2 control measured 0.650419 end-to-end t/s and 1.42 server
decode t/s for the 64-token cyberpunk prompt. The nested cache-6 candidate was
byte-identical but measured 0.487870 / 0.88 t/s, with 0 cache hits, 1,802
misses and 5,668,601,856 residual bytes. G121 changed only the nested exact
cache capacity to 64. It remained byte-identical and recorded 817 hits, 985
misses and 3,098,542,080 residual bytes; end-to-end rose to 0.553980 t/s and
server decode to 1.12 t/s.

These are `n=1` structural measurements, not SOTA verdicts. Their causal value
is the transport accounting: host exact-cache reuse reduces residual reads,
but nested H2D stayed at 12,754,354,176 bytes in both candidates. The current
covered-layer path bypasses the GPU-resident route cache, so even a host cache
hit uploads the full native expert again. That is the remaining implementation
error to remove before broader coverage or repeated performance runs.

## Runtime change map

- Keep `ds4_gpu_router_select_tensor` and its batch variant as the only source
  of selected IDs and weights.
- Reuse the native IQ2 geometry and compact selected-load remapping. A cold
  reconstruction publishes the same gate/up/down byte layout expected by the
  existing IQ2 kernels.
- Add one nested runtime state: resident base catalog, expert-major residual
  catalog, bounded native-IQ2 reconstruction arena and mass/LFRU retention
  metadata.
- Do not copy `cuda_moe_selected_load_q1_0` physical IO: it performs three
  tensor reads. The residual container stores gate/up/down residuals adjacent
  for one 3 MiB expert read.
- In exact mode a miss reconstructs the selected expert or fails closed. In
  5+1 mode only the lowest-weight cold lane may use base-only compute, while
  promotion affects future residency and never current router selection.

Required counters: router-open calls, exact cache hits, residual preads and
bytes, base-only lanes, reconstructed experts, reconstruction mismatches, MoE
launches and input quantizations. A 5+1 call passes its structural gate only if
it records one MoE launch and one input quantization.

## G123 decision update

G123 completed the equal-host-budget `n=3` A/B. All six accepted processes
were exact and uncontaminated. The candidate measured 1.163333 server decode
t/s versus 1.650000 for the full/open control (-29.49%), and 0.555497
end-to-end t/s versus 0.650261 (-14.57%). TTFT was effectively flat, but
candidate load rose from 11.222 s to 29.212 s. Every candidate repeated 541
GPU hits and 995 misses, a 35.22% covered-route hit rate.

This is a negative performance verdict for the present implementation, not
for the nested representation itself. Freeze coverage at four layers. Do not
build the all-layer residual catalog yet.

## G124-G126 decision update

G124 profiled the G123 miss path as `n=1` causal evidence only. It measured
869 CPU reconstructions taking 14.8898841 s, 869 residual preads taking
1.9070234 s, 2,607 reconstruction verification calls taking 0.4990679 s and
14.6092232 s of route-ready wait. Those timers overlap and are not a wall-clock
sum, but they identified host reconstruction plus full native H2D as the next
mechanical target.

G125 then passed the structural GPU-side exact-join safety gate with the same
full/open content SHA
`fd6c4522975a71e252b90199d49cfe3236310e2a7285dc0fc4d0e9d0e4885510`.
It observed 1,261 GPU join calls, positive base/residual H2D, zero native H2D,
zero CPU reconstruction, zero mismatch and zero failure. This remains `n=1`
structural evidence, not a performance or quality verdict.

G126 completed the CPU-join versus GPU-join repeated A/B with three
independent processes per arm. Decode is the defensible finding: CPU join was
`1.15, 1.16, 1.15` t/s, mean 1.153333, while GPU join was
`1.56, 1.58, 1.57` t/s, mean 1.57, a measured +36.1272%. E2E mean was also
higher by +49.7457%, but TTFT/request/WRAP timing was too noisy for a general
latency claim, so it is retained as batch-only/noisy.

Decision: GPU-side exact join repairs the miss-path decode regression enough
to beat the previous G123 nested candidate, but it still trails the G123
full/open IQ2 control at 1.65 t/s and remains far below historical G73, which
is closed/request-scoped rather than full/open. Do not call G126 absolute SOTA.
The next candidate must either reduce the remaining base/residual H2D and join
wait, or increase protected GPU reuse, before an all-layer residual catalog or
long-form quality run is justified.

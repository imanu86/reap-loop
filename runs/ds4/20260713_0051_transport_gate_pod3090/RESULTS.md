# 0051 transport gate: 3090 mechanism results

Date: 2026-07-13
Node: RunPod RTX 3090 24 GiB, native Linux container (not WSL)
Model: `/workspace/models/ds4-2bit.gguf`, 86,720,111,488 bytes
Protocol scope: deterministic 60-token mechanism/exactness screen. This is not
the preregistered long steady-state throughput campaign.

## Measured arms

All completed A, B0, and C2 responses are byte-identical, including across
arms, with content SHA256
`4605c1d7500de0e2c70754f28530da5a49a04596505ae9d13d0dcb9419a8e8a6`.

| Arm | Decode t/s by measured repetition | Median decode t/s | Relevant final counters | Interpretation |
|---|---:|---:|---|---|
| A synchronous staged source | 0.61, 2.60, 2.72 | 2.60 | copy 411,127.8 ms; sync 145,827 calls / 16,584.8 ms | Exact reference; first repetition was still cold. |
| B0 async staged, L-to-L+1 prediction off | 0.58, 2.60, 2.71 | 2.60 | copy 436,025.0 ms; sync 100,227 / 11,050.4 ms | Exact; fewer synchronizations did not improve this short decode materially. |
| B1 async staged plus L-to-L+1 prediction | 0.31 in the decisive cold screen | not promoted | cache hit rate 1.06%; 294,222 copy calls in the aborted campaign | Hard mechanism regression; rejected without an `n>=3` performance verdict. |
| C1 ID-ordered 24 GiB direct pin | 1.15 (`n=1` screen) | not a verdict | 27,222 / 61,152 direct queries = 44.52% | Direct path works, but ID-order wastes the pin budget. |
| C2 causal mass-ranked 24 GiB direct pin | 1.37, 4.13, 5.03 | **4.13** | 81,940 / 145,827 direct queries = 56.19%; copy 204,610.6 ms; sync 37,027 / 4,040.8 ms | Exact. Median is +58.8% over B0; last warm repetition is +85.6%. The sequence is still warming, so 5.03 is not a stationary median. |

C2 cut cumulative copy time by 53.1% and synchronization calls by 63.1%
relative to B0. The causal pin plan was learned on the first half of the coffee
trace and evaluated on the second half. Its offline held-out gate-mass coverage
was 85.66%; the runtime 56.19% figure above counts all instrumented copy
queries, not only the routed-expert gate mass.

## Chripell reference control

The published Profile A in
`chripell/ds4-rtx3090@5a854d2` claims 6.42 prefill t/s and 9.04 generation
t/s with native Linux, an RTX 3090, the entire model mmap pinned, an 8 GiB
expert cache (about 1,213 experts), 4 GiB reserve, prefill chunk 1024, and
context 32768:

- <https://github.com/chripell/ds4-rtx3090/commit/3fc5b217ee41b03ea117b03511613c92f23b05ba>
- <https://github.com/chripell/ds4-rtx3090/commit/5a854d2d4d496df4bb845087bce3fa82661f588d>

The publication does not include the benchmark prompt, output length, GGUF
hash, warm/cold order, or repetition count. Its exact score therefore cannot be
reconstructed, only its documented launch profile.

The exact `5a854d2` server was built on the same pod with SHA256
`afe58a3f315311719cd4ae9ea7eb3c7e3e216460485852bb9c58ec0249a671e5`.
It did not reach inference: the monolithic approximately 80.76 GiB
`cudaHostRegister()` returned `invalid argument`; the advertised fallback then
failed on the first `q8_0` GPU range copy. A second binary with only the final
registration length rounded to a page boundary
(`5be7b039b34384bdd9442c68a4c8d787a1d219948ba4d3a71481f7092f9c578d`)
failed identically.

This was native Linux, not WSL. The container had a 124,999,999,488-byte
cgroup memory limit, NVIDIA driver 580.126.20, an 8 MiB locked-memory limit
that could not be raised, and the model lived on RunPod FUSE storage. The
failure is narrowed to the monolithic registration/container-driver-mapping
combination; the present evidence does not identify which factor is causal.
Patch 0050's bounded range registrations succeeded on the same node up to
23.94 GiB, so its robustness advantage over the reference is measured.

### Chunked/split-copy adaptation

This campaign did not silently treat a modified binary as the exact fork.
The runnable control is explicitly `5a854d2 + chunked/split-copy safety
adaptation`, preserved in
`provenance/chripell-5a854d2/chripell-model-map-chunked.patch`.

The measured adaptation path was:

1. Plain `cudaHostRegisterMapped` failed even after reducing registration
   chunks to 2 MiB on the read-only FUSE mmap.
2. `cudaHostRegisterReadOnly | cudaHostRegisterMapped` registered the full
   80.76 GiB map in 324 chunks of at most 256 MiB.
3. The first inference then exposed an expert copy crossing a registration
   boundary (`selected moe_up: invalid argument`). Splitting H2D copies at the
   registered boundaries fixed that failure.
4. Cleanup was moved before `munmap`, and partial registration now rolls back
   instead of entering the fork's broken pageable fallback.

The final adapted binary SHA256 was
`163d11b0a5e7f5b3cf4cc90d573337af7a7471362f383255c360b64de8c39571`.
With the documented Profile A knobs, the same 64-token prompt, a 256-token
warmup and three measured 256-token requests, it produced:

| repetition | prompt seconds | decode seconds | decode t/s | wall seconds |
|---|---:|---:|---:|---:|
| warmup | 2.878 | 136.910 | 1.87 | 139.789 |
| r01 | 2.972 | 136.617 | 1.87 | 139.620 |
| r02 | 2.966 | 136.704 | 1.87 | 139.700 |
| r03 | 3.013 | 136.591 | 1.87 | 139.626 |

The three measured outputs were byte-identical. Each contains one complete,
tag-balanced HTML document and closes `</html>`. The committed standard
`functional_grade.py frontpage` result is nevertheless `L1,L1,L1`, because
this minimal benchmark prompt does not request the form, CSS, or JavaScript
that the generic frontpage rubric requires. Both facts are preserved; the
prompt-specific success is not relabelled as L3.

The server used 11,580 MiB of VRAM during the run. The measured 1.87 t/s does
not reproduce the fork's published 9.04 t/s. The original prompt, model hash,
output length, warm order, and repetition count behind that published number
remain unavailable, so this is a controlled same-node profile measurement,
not proof that the published result is false.

## 0050k safety checkpoint and blocker

The 0050k binary was reconstructed with post-patch `ds4_cuda.cu` SHA256
`10e21d340702df127e2398aba35923833b5eb47e9cc025af3a7beec32bf79fe2`.
On the real server async path, an eight-arm mechanism screen completed:
baseline ring depths 1/2/8 plus the five one-shot event faults at depth 1.
Every arm exercised 14,514 staged chunks and emitted the same four-token
canary. Baselines recorded 11,400 cursor advances and WAR waits; every injected
fault latched synchronous staging, completed its fallback, and reported zero
fallback-sync failure, staged-H2D failure, and stale-slot diagnostic.

This screen is useful but does **not** close Phase 1. Static review found that
`cuda_transport_drain_upload_stream()` clears deferred/inflight state before
checking whether `cudaStreamSynchronize()` succeeded, and a non-strict
post-batch caller can convert a failed safety drain to success. The patch also
lacks a one-shot staging-event WAR-wait fault and an independent device-payload
checksum. Throughput promotion is paused pending an incremental fail-closed
patch and a repeated safety gate. The completed 0050k matrix is evidence for
the recoverable paths only, not proof of all failure paths.

## Position relative to the reference

The honest current position is:

1. The best earlier mechanism observation is C2 at a 4.13 t/s measured median
   and 5.03 t/s last warm repetition. The runnable same-node chripell
   adaptation measured 1.87 t/s, while the published but incompletely
   specified chripell score remains 9.04 t/s.
2. The configurations are not yet equivalent. C2 used 400 cache experts
   (about 2.64 GiB), 1 GiB reserve, prefill chunk 512, context 2048, a static
   keep-60 compute mask, and only 23.94 GiB of selected host pinning. Profile A
   uses about triple the GPU expert cache and attempts to pin the full model.
3. The historical 14.60 t/s phase-2 result at K23/cache1024 is real but not a
   reference replication: it used a much narrower frozen compute mask and a
   two-phase re-prefill protocol.
4. Our measured advantage is robustness and selectivity under a bounded pin
   budget. A throughput advantage over Chripell is not established.

## Next decisive controls

1. Replace the reference's one-shot registration with chunked registration
   only, retaining its synchronous serving path and published Profile A knobs.
   This distinguishes the monolithic-registration failure from the rest of the
   fork without importing 0050/0051 behavior.
2. Run our accepted 0050k path with the same 8 GiB cache, 4 GiB reserve,
   prefill chunk 1024, context 32768, prompt, output budget, and warm order.
3. Compare the reference-chunked and 0050k arms at `n>=3` in a long stable
   decode. Report median, range, raw runs, copy-query coverage, copy time, sync
   time, and exact binary/model hashes.
4. Only then decide whether 0051's permanent dynamic arena is needed. C2 shows
   enough residual value to continue the gate, but not enough evidence to claim
   that 0051 will beat the published reference.

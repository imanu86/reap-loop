# G26 REAP sliding-mass observer

Date: 2026-07-14

## Scope

G26 ports the REAP residency signal to native Windows as an observe-only path.
It does not change routing, masks, arena publication, admission, eviction, or
SPEX. The observer is enabled only by:

```text
DS4_CUDA_REAP_MASS_OBSERVE=1
DS4_CUDA_REAP_MASS_WINDOW=16
```

For decode tokens only (`n_tokens == 1`), selected gate weights are normalized
within each token/layer and accumulated in an exact hard sliding window. The
window stores `(layer, expert)` mass, so the signal represents frequency times
normalized gate weight. The previous token is committed when the layer index
wraps; the current token therefore cannot affect a residency decision made
before its routing is observed.

## Provenance

- Base commit: `8f76b81a5b20377a575030c8e260b093b77ba302`
- GPU: NVIDIA GeForce RTX 3060 12 GB
- Model: `D:\ds4-models\ds4-2bit.gguf`
- Model bytes: `86,720,111,488`
- Executable SHA-256: `5603f9efaed5dbbddc43c2713ce8657652a2631fba91f185f41aa315588a022f`
- `ds4_cuda.cu` SHA-256: `97e38139b975b611c45dbb750713b4fc088fe580c689352b852d3749caf01bd3`
- Harness SHA-256: `f55806159e6f4d711bb64aee51c12fd57f3982991a1a62dba2831338f11281eb`
- Context: 256
- Host model window: 2 GiB
- Empty pinned arena: 2 GiB
- CUDA reserve: 1,024 MiB
- Q8/F16 cache reserve: 4,096 MiB
- Prompt: `Explain in one concise paragraph why Julius Caesar crossed the Rubicon.`
- Max generated tokens: 16
- Expected output SHA-256: `b037ce25fab7393eeb9fc5b7bf7f5b8ef70768aea476cd1c09b0ffa348323b30`

Each row below is an independent process start. Runs were alternated OFF/ON.

## Exact A/B

| Mode | Run | Server decode t/s | Prefill/TTFT s | Exact output |
|---|---:|---:|---:|---|
| ON | 1 | 2.48 | 4.768 | yes |
| ON | 2 | 2.38 | 5.695 | yes |
| ON | 3 | 2.20 | 4.596 | yes |
| OFF | 1 | 2.42 | 5.783 | yes |
| OFF | 2 | 2.69 | 4.615 | yes |
| OFF | 3 | 2.62 | 4.598 | yes |

| Mode | n | Mean decode t/s | Median decode t/s | Mean TTFT s |
|---|---:|---:|---:|---:|
| ON | 3 | 2.353 | 2.38 | 5.020 |
| OFF | 3 | 2.577 | 2.62 | 4.999 |

Measured mean decode delta: **-8.7%** with the CPU observer enabled. All six
outputs match the expected hash. Every ON run reports 16 committed decode
tokens, 3,840 selected slots (`16 * 40 layers * top-6`), and 1,538 unique
`(layer, expert)` entries in the final window.

This is a negative performance result: the CPU-side per-layer weight readback
is too expensive for the final runtime path. The mass semantics and exactness
are usable, but the accumulator needs GPU-side or batched collection before it
drives arena residency.

## Rejected prefill behavior

The first safety build requested weight readback during prefill even though the
REAP observer ignored prefill rows. It remained output-exact but measured 54.472
s TTFT and 0.71 server decode t/s. This run is excluded from the final A/B. The
gate was corrected so REAP requests weights only when `n_tokens == 1`; the final
n=3 matrix above measures that corrected build.

## Next gate

1. Remove or batch the per-layer CPU weight readback.
2. Re-run the same independent n=3 exact A/B.
3. Only after observer overhead is acceptable, use REAP mass to fill free arena
   slots first and select the lowest-mass victim only at full capacity.
4. Add SPEX as a separate predictive candidate source; do not let it alter the
   router or mask in this gate.

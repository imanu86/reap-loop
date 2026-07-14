# G26b packed router trace

Date: 2026-07-14

## Change

G26 measured an 8.7% mean decode penalty when selected expert IDs and selected
gate weights were copied from GPU to CPU as two separate commands per layer.
G26b keeps the exact same REAP sliding-mass signal but has the existing router
kernel also write a contiguous 48-byte trace:

```text
6 x int32 selected expert ids
6 x float selected gate weights
```

The selected-load path consumes that structure with one blocking D2H transaction
into a persistent pinned host buffer. In the non-prepared path this replaces the
selected-ID read and avoids a separate weights read. In the prepared shared-
overlap path, selected IDs were already staged earlier and the packed transaction
supplies the weights, so it does not remove that earlier ID read. This does not
make router readback asynchronous. No sampling, router change, mask, residency
publication, or approximation is introduced.

Runtime telemetry identifies this path as:

```text
transport=packed-router-d2h
```

## Provenance

- Base commit: `02d7849a6efa9fefaab8ef17ce1a073693be3c04`
- GPU: NVIDIA GeForce RTX 3060 12 GB
- Model: `D:\ds4-models\ds4-2bit.gguf`
- Executable SHA-256: `4874b7e6bd72bf51f32fd58debc46b2bf1416f0f7d470e18889edff50d3ad778`
- `ds4_cuda.cu` SHA-256: `249849396217163f738d7f85191184acebc33dcb346365474bf91882284724c1`
- Harness SHA-256: `82bcde7fc576e00ff9c695e4ba97920893dc8cc3531edca8dc24ab6ad4b3c91c`
- Build manifest SHA-256: `b6f30013f9b896a526713ca1535c94f4959d702598a9169f587ac4dd5f0d5d14`
- Build fingerprint: `92513cf24543162555fb220d8c64ae36ebf0832701f00df33764acdb874ff43c`
- Common launch parameters: 16 generated tokens, context 256, 2 GiB host
  window, 2 GiB empty pinned arena, 1,024 MiB CUDA reserve, 4,096 MiB
  Q8/F16 reserve, greedy nothink.
- Expected output SHA-256: `b037ce25fab7393eeb9fc5b7bf7f5b8ef70768aea476cd1c09b0ffa348323b30`

Each row is an independent process start. The final gate was counterbalanced as
`OFF/ON, ON/OFF, OFF/ON` to limit page-cache and run-order bias.

## Exact A/B

| Mode | Run | Server decode t/s | Prefill/TTFT s | Exact output |
|---|---:|---:|---:|---|
| packed ON | 1 | 2.59 | 4.661 | yes |
| packed ON | 2 | 2.58 | 4.732 | yes |
| packed ON | 3 | 2.60 | 4.702 | yes |
| OFF | 1 | 2.65 | 4.773 | yes |
| OFF | 2 | 2.67 | 4.751 | yes |
| OFF | 3 | 2.67 | 4.795 | yes |

| Mode | n | Mean decode t/s | Median decode t/s | Mean TTFT s |
|---|---:|---:|---:|---:|
| packed ON | 3 | 2.590 | 2.59 | 4.698 |
| OFF | 3 | 2.663 | 2.67 | 4.773 |

The final mean delta is -2.8% and the median delta is -3.0%. This is not a
speedup, but the packed transport reduces the prior measured 8.7% observer
penalty to a small residual cost while preserving the exact signal. Longer-run
performance claims still require their own counterbalanced n>=3 gate.

All ON runs report exactly 16 committed decode tokens, 3,840 selected slots,
1,538 unique `(layer, expert)` entries, and the same top mass (`5.595660`). All
six outputs match the expected hash.

## Decision

Accept `packed-router-d2h` as the transport for the REAP mass signal. The next
patch may use this signal for residency decisions, filling free dynamic-arena
slots before selecting any victim. Performance claims for the residency policy
still require a separate n>=3 exact A/B.

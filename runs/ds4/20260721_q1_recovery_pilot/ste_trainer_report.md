# Single-expert STE distillation trainer

Date: 2026-07-21  
Pilot target: layer 15, expert 176 (not run)  
Smoke target: layer 3, expert 0, 17 real captured activations

## Outcome

The trainer is implemented as `trainer.py` plus a native OpenMP backend. It performs gradient-based quantization-aware distillation of one MoE expert, exports three raw GGML type-41 Q1_0 payloads, and evaluates the payloads after their fp16 scales have been round-tripped. The source GGUF and Q1 sidecar are opened read-only and are never rewritten.

The 17-sample smoke run completed end-to-end and selected epoch 5 by validation loss. Its exact plain TEST cosine was `0.59577034100235571`, versus `0.59488495641474692` for the untouched Q1 baseline. This smoke result does **not** meet the pilot success bar of TEST cosine `>= 0.90`; 17 samples are intentionally too small for a quality conclusion.

## Design

Each gate, up, and down matrix has an fp32 latent tensor `W_l`, initialized from the reference-dequantized teacher weights. A forward pass materializes

```text
W_q = sign(W_l) * exp(log_scale[row, block])
```

for contiguous 128-weight blocks. Initial scales are each teacher block's mean absolute value, exactly matching the reference Q1 quantizer's scale rule. The scales are learned as log-scales, which keeps them positive. A small default L2 anchor (`--scale-reg 1e-4`) acts on log-scale displacement from initialization.

Learned scales were chosen instead of per-step closed-form recomputation. They allow the signs and scales to be optimized jointly by the output distillation objective, regularized toward the source quantizer, scheduled by the same optimizer, and selected by validation early stopping. This directly addresses the unregularized closed-form scale refit's held-out failure.

The sign backward is the identity STE. Consequently, the latent gradient includes the current block scale from the multiplication above. `--clip-latent 1` clips updated latent values to `[-1, 1]`; `0` disables clipping.

The exact student and teacher expression is:

```text
down(silu(gate(x)) * up(x))
```

Teacher outputs use fp32 weights decoded from the IQ2_XXS/Q2_K source tensors. The default loss compares `gate_weight * student_output` with `gate_weight * teacher_output`, equivalent to weighting each sample's squared error by `gate_weight^2`. `--loss-weighting plain` disables this. Both plain and gate-weighted metrics are always reported.

Optimization uses Adam, one update per epoch after deterministic minibatch gradient accumulation, cosine LR decay, and validation early stopping. The default seed is `1337`. Static OpenMP work division and fixed-order inner reductions make repeated runs deterministic for a fixed binary and thread configuration.

### Optional low-rank correction

`--lora-rank 1` through `4` enables `U V^T` after the Q1 matrix in each linear transform. `U` starts at zero and `V` from a seeded small normal distribution, avoiding the zero/zero gradient dead start. Export stores U and V as fp16 and final metrics use fp16-round-tripped factors.

Across gate, up, and down, the storage cost is:

```text
3 * 2 bytes * rank * (4096 + 2048) = 36,864 * rank bytes
```

That is 36 KiB at rank 1 and 144 KiB at rank 4 per expert, in addition to the 1.125-bpw Q1 payload. A separate one-epoch rank-1 mechanics check completed successfully in `smoke_lora_r1_check`.

## Decode, traces, and split

The generalized GGUF parser adapts the working recovery PoC logic. It locates `blk.<layer>.ffn_{gate,up,down}_exps.weight`, validates `(ncols, nrows, 256)` geometry, seeks to the requested contiguous expert payload, and calls the same pinned `ds4_q1_ref` dequantizers. The smoke source types were `[16, 16, 10]` (IQ2_XXS, IQ2_XXS, Q2_K), and all sidecar types were `41`.

`--traces` may be repeated and accepts a manifest glob. Sessions are merged only when layer, expert, model hash, and sidecar hash agree. Each session is fail-closed on schema/status, reconstruction contract, sizes, SHA-256, the complete 64-byte binary header, row offsets, sample identity, float gate bits, and finite activations.

Splitting is deterministic and request/session-disjoint: the grouping key is `(manifest session, request_epoch)`, so request epochs reused by separate captures cannot collide. The allocator minimizes distance from 70/15/15 where the number of groups is small and uses a deterministic balanced greedy allocation for larger captures.

The smoke split was:

- train: 12 samples, indices `0,5,6,7,8,9,10,11,12,13,14,16`;
- validation: 2 samples, indices `1,15`;
- TEST: 3 samples, indices `2,3,4`.

This is 70.6%/11.8%/17.6%; exact 70/15/15 counts are impossible for 17 items while preserving whole request groups.

## Smoke command

The intended CLI is:

```powershell
python trainer.py --layer 3 --expert 0 `
  --traces "C:\Users\imanu\g130i\trace_out\run1_l3e0\l3e0.manifest.json" `
  --out smoke_l3e0 --epochs 8 --patience 3 --batch-size 12 --seed 1337
```

There is no working Python command in the current shell (the discovered virtualenv launcher points to a missing interpreter), so the actual smoke invoked the equivalent dependency-free backend command:

```powershell
.\ste_backend.exe --layer 3 --expert 0 `
  --traces "C:\Users\imanu\g130i\trace_out\run1_l3e0\l3e0.manifest.json" `
  --out smoke_l3e0 --epochs 8 --patience 3 --batch-size 12 --seed 1337 --device cpu
```

`trainer.py` is a standard-library launcher: it builds `ste_backend.exe` through `build_backend.bat` when absent and otherwise passes the CLI through unchanged. No Torch runtime was available, and the RTX 3060 was already using about 8.7 GiB during capture, so this build intentionally defaults to and implements CPU execution only. Passing a non-CPU device fails explicitly.

## Exact smoke results

All recovered metrics below are from the emitted type-41 payload after fp16 scale serialization/dequantization.

| Candidate | Split | Plain cosine | Plain NMSE | Weighted cosine | Weighted NMSE |
|---|---|---:|---:|---:|---:|
| Q1 baseline | train (12) | 0.560762226616 | 0.688693250849 | 0.555058287144 | 0.694321713523 |
| recovered | train (12) | 0.631309229326 | 0.615685627091 | 0.643801992883 | 0.602060416449 |
| Q1 baseline | val (2) | 0.605505603660 | 0.652724088027 | 0.626673322805 | 0.642300112563 |
| recovered | val (2) | 0.607398838981 | 0.651034015760 | 0.629542658408 | 0.639913509817 |
| Q1 baseline | **TEST (3)** | **0.594884956415** | **0.666411224670** | **0.602786815305** | **0.665755887971** |
| recovered | **TEST (3)** | **0.595770341002** | **0.665397894199** | **0.604007012940** | **0.664462806571** |

Validation gate-weighted MSE moved from `0.0185435351972` at initialization to a best `0.0184750854960` at epoch 5. The selected state changed 29,713 down-matrix signs and no gate/up signs. Learned scale ratios versus initialization remained tight:

| Tensor | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| gate | 0.996188769 | 1.000080055 | 1.003827966 |
| up | 0.996189007 | 1.000076692 | 1.003826769 |
| down | 0.996212521 | 1.000191810 | 1.003810016 |

The much larger train gain than TEST gain is the expected 17-sample overfit warning. The smoke asserts decode, gradients, sign movement, scale learning, checkpoint selection, serialization, and metric mechanics only.

Measured native wall time was `2.6396 s`, including trace validation, decode, teacher/Q1 evaluation, eight epochs, export, round-trip evaluation, and hashing. Peak working set was `565,587,968` bytes (539.4 MiB). Individual 12-sample training epochs took roughly 0.16–0.18 s. A linear sample-count extrapolation is comfortably below two minutes for 1,000 samples, but this was not claimed as a measured 1,000-sample benchmark.

## Artifacts

Each matrix has 65,536 blocks of 18 bytes (`fp16 scale + 128 sign bits`), or 1,179,648 bytes. The three-matrix expert payload is 3,538,944 bytes, exactly 1.125 bpw over 25,165,824 weights.

| File | Bytes | SHA-256 |
|---|---:|---|
| `recovered.gate.q1_0` | 1,179,648 | `863636a0ac9e28ce2c8620978a791b6142cf40f2ed6d5ca12ebcec297f638c21` |
| `recovered.up.q1_0` | 1,179,648 | `98da314687515e956ac6fcca3726a812ae7e934be3be0643efd001e477e77257` |
| `recovered.down.q1_0` | 1,179,648 | `faa2d8793fede7266cfe82554c0eb85228f2d8b670dc38ee07902daa9e885ad1` |

`smoke_l3e0/training_results.json` is the machine-readable run record. `smoke_l3e0/recovered_manifest.json` describes the three raw payloads and records their hashes. These are expert payloads, not a mutated or replacement GGUF sidecar.

Layer 15 expert 176 was not opened, decoded, trained, or evaluated. No model file was modified and no commit was created.

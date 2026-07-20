# DS4 single-expert ternary/binary micro-probe

Scope: CPU-only, read-only on model/sidecar inputs. No GPU, server, DS4 build, download, model write, ledger write, commit, or full-model hash. Scripts and receipts live under `work\ternary_b_microprobe`.

## Selection and provenance

- Expert: deterministic `blk.3`, expert `0`, first routed layer after local `DS4_N_HASH_LAYER=3`.
- Teacher GGUF: `C:\ds4-models\ds4-2bit.gguf`.
- Q1 sidecar: `C:\ds4-models\ds4-q1-layers0-42-derived.gguf`.
- Parser/dequant provenance: local GGUF parser shape from `ds4_iq1_s_bench.cu`; IQ2_XXS/Q2_K/Q1_0 decoders transcribed from local `ds4.c`, `ds4_cuda.cu`, `ds4_q1_0_bench.cu`.
- Decoder synthetic self-test: PASS, max abs error `0.0` for IQ2 all-one, Q2 all-two, Q1 alternating.

## Phase 0 offsets and payload SHA

| Tensor | Teacher dtype | Shape | Teacher offset | Teacher bytes | Teacher SHA256 |
|---|---:|---:|---:|---:|---|
| gate | IQ2_XXS | 4096x2048x256 | 5,450,459,968 | 2,162,688 | `b12ad7bd291c914f23a4aaea0ac99f5f549ed69d0a871604072e1cd77388cf0e` |
| up | IQ2_XXS | 4096x2048x256 | 6,708,751,168 | 2,162,688 | `de07385af534f354d8041d1d5494383488027f83cbdeba56877a22767d34a26c` |
| down | Q2_K | 2048x4096x256 | 6,004,108,096 | 2,752,512 | `5f2195082e9783ba7610827c225b1689770daa51f750a513f8fa770c129dbb52` |

| Tensor | Q1 sidecar offset | Q1 bytes | Q1 SHA256 |
|---|---:|---:|---|
| gate | 85,357,408 | 1,179,648 | `0c88b12889279390d097fc41dcfc62322cba56f7b83fb63ab37b3a578418e308` |
| up | 387,347,296 | 1,179,648 | `a767f714b5a7a657698755fa8e45a45c87e488e83a8951fc2e11b770a05fcc48` |
| down | 689,337,184 | 1,179,648 | `1289b5212239095337f57496e4303507216119ed411eea08009047f689f5e494` |

Q1 total expert bytes are exactly `3,538,944 B = 1.125 bpw`.

## Ternary B weight-domain baseline

Expert aggregate, teacher reconstruction as reference:

| Target | k/g128 | zero frac | mask+sign+scale bpw | entropy+scale ideal bpw | cosine | NMSE | max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.725 | 76 | 0.40625 | 1.718750 | 1.693239 | 0.877796 | 0.229474 | 0.106346 |
| 1.50 | 48 | 0.62500 | 1.500000 | 1.454434 | 0.882552 | 0.221103 | 0.100996 |
| 1.30 | 22 | 0.828125 | 1.296875 | 0.958851 | 0.758266 | 0.425033 | 0.090691 |
| 1.15 | 3 | 0.9765625 | 1.148438 | 0.308766 | 0.366082 | 0.865984 | 0.098840 |

Inference: `<=1.3 bpw` is mathematically packable for this one expert, but the pure weight-domain baseline is much weaker than the 1.5-1.725 bpw region. This is not activation-aware and says nothing end-to-end.

## Binary extension

Expert aggregate, existing Q1 sidecar vs teacher:

| Candidate | bytes/expert | cosine | NMSE | max abs |
|---|---:|---:|---:|---:|
| Existing Q1_0 sidecar | 3,538,944 | 0.811377 | 0.341667 | 0.114826 |
| Binary refit from teacher, sign + L2 scale g128 | 3,538,944 | 0.811377 | 0.341667 | 0.114826 |
| Existing Q1_0 vs refit binary | n/a | 1.000000 | 4.456e-08 | 7.641e-06 |

Per tensor sign mismatch between existing Q1 and refit binary is `0.0`; max scale delta is `7.64e-06`. Inference: the current Q1 payload is already the L2-optimal binary sign/scale reconstruction from the IQ2 teacher to numerical precision. The observed Q1 loss is therefore not primarily a packing-format problem or an obvious converter-fit bug; it is the binary weight-domain approximation limit unless training/recovery changes the weights or compensates with activations.

## Decision

- Ternary B `<=1.3 bpw`: NO-GO as a final cold format based only on post-training weight-domain top-k thresholding for this expert; GO only as a candidate for activation-aware recovery experiments because its memory budget is attractive.
- Q1_0/binary: NO-GO as a quality fix by re-fitting the same format; the sidecar is already effectively refit-optimal. GO for activation trace acquisition if the next question is whether local STE/QAT/recovery can move the quality frontier.
- Next discriminant: collect activation/router traces for the same documented expert and repeat with activation-weighted objective plus teacher outputs, still before CUDA.

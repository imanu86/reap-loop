# Native Windows coding bake campaign - 2026-07-15

Status: first static-mask quality gate complete. This file separates measured
facts from pending gates.

## Provenance

- Producer pod: RunPod RTX 3090 Ti 24 GB (`0htxln87674tjq`).
- Serving target: native Windows, RTX 3060 12 GB plus 64 GB host RAM.
- Source model: `ds4-2bit.gguf`, `86,720,111,488` bytes.
- Recorded source SHA-256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`.
- Routing oracle binary SHA-256:
  `4bb29874a7028cc06c7a1d1f6696528a854694e6f7d8de626f875abc3ecf2f76`.
- Reap-loop revision on pod:
  `2a46b1fee82c6fbd01aa48a22155fd38ebf76047`.
- The Linux binary is used only for uncensored routing and functional quality.
  Its throughput is not a native-Windows performance result.

## Trace campaign

The learn split contains six coding prompts: C, C++, Python, JavaScript, HTML,
and SQL. The held-out split contains PowerShell, Rust, Go, and container-stack
tasks. Both used K0 routing with router weights and opt-in prefill tracing.

Measured trace totals:

| Split | CSV rows | Routed layers |
|---|---:|---:|
| Learn | 18,343 | 43 (`0..42`) |
| Held-out | 8,889 | 43 (`0..42`) |

The `positions` field in the original summaries is not a token total because
position ids restart for every request. CSV row count and per-layer rows are the
authoritative measurements.

## Held-out coverage

Coverage is recomputed by `scripts/score_routing_mask_coverage.py` from the
committed held-out CSV and mask files.

| Mask | Calls covered | Router mass covered | Rows with all six covered | Misses | Worst layer mass |
|---|---:|---:|---:|---:|---:|
| K60 / keep154 | 97.7576% | 98.3860% | 88.4541% | 1,114 / 49,680 | 94.5477% |
| K62.5 / keep160 | 98.0113% | 98.5146% | 89.4565% | 988 / 49,680 | 94.6898% |
| K65 / keep166 | 98.2045% | 98.6194% | 90.3986% | 892 / 49,680 | 94.8023% |

These figures measure overlap with uncensored held-out routing. They do not
prove functional equivalence or losslessness.

## Exact payload plans

GGUFReader measured 1,328 tensors, 129 routed tensors, and 43 routed layers.
Each layer/expert `(gate, up, down)` triplet is `7,077,888` bytes. Layers `0..2`
are hash-routed and remain full in the current implementation; masks apply to
layers `3..42`.

| Candidate | Payload bytes | Payload GiB | Saved GiB | Extents |
|---|---:|---:|---:|---:|
| K60 | 57,842,328,448 | 53.8699 | 26.8945 | 7,089 |
| K62.5 | 59,541,021,568 | 55.4519 | 25.3125 | 6,750 |
| K65 | 61,239,714,688 | 57.0339 | 23.7305 | 6,426 |

K65 costs 3.1641 GiB over K60 while increasing held-out mass coverage by only
0.2334 percentage points. That is not a verdict against K65; it is the measured
capacity/coverage trade-off that the functional A/B must resolve.

K8 with layers `0..2` full and keep8 on layers `3..42` is 15.3738 GiB before
KV/cache/runtime buffers. It is therefore not VRAM-only on the 12 GB target as
currently defined.

## Packaging

`scripts/ds4_windows_sparse_bake.py` implements:

- `plan`: exact retained extent and payload calculation;
- `pack`: compact portable payload with SHA-256-verified manifest and data;
- `unpack`: NTFS sparse GGUF reconstruction at original tensor offsets;
- `inspect`: trailer, bitset, CRC, and manifest consistency validation.

The unpacked file appends the audit JSON, a retained-expert bitset, and a fixed
DS4BAKE footer. The runtime bitset is 43 x 256 bits; original expert ids and GGUF
offsets do not change. A smoke test verified included extents byte-for-byte,
excluded holes as zero, payload SHA-256, trailer CRCs, and bitset/manifest
agreement.

The physical K60/K65 packs have not been emitted yet. Writing 54-57 GiB before
the quality gate would spend pod time and transfer cost without changing the
quality evidence.

## Functional A/B

The first attempted A/B used a 1,600-token budget and was stopped after K0 run
1 because the output was still coherently writing CSS. It is invalid for a
mask-quality verdict: budget truncation was confounded with quality.

The replacement protocol uses:

- K0, K60, K65;
- three runs per arm at temperature `0`, `0.2`, and `0.7`;
- cache 1024, context 4096, maximum 3200 output tokens;
- a compact single-file HTML/CSS/JS dashboard prompt;
- client stop on `</html>` or measured repetition;
- L0-L3 functional grading; never a verdict from `n=1` or `repeat_flag`.

Final grades for this candidate-mask campaign:

| Arm | Run 1 | Run 2 | Run 3 | Median | Result |
|---|---:|---:|---:|---:|---|
| K0 | L2 | L2 | L2 | L2 | reference |
| K60 | L1 | L1 | L1 | L1 | rejected |
| K65 | L1 | L0 | L2 | L1 | rejected |

All three K0 runs reached `</html>` without restart. They were L2 because the
model implemented the add-job section as a `div`, not the requested HTML
`form`. K60 never completed functional JavaScript; two runs reached the token
limit and one was stopped on measured repeated comment lines. K65 was unstable:
one HTML close with two detected JavaScript errors, one L0 length exhaustion,
and one incomplete L2 length exhaustion.

Decode throughput from these Linux oracle runs must not be compared across
arms as a controlled performance A/B. K0/K60 ran on the RTX 3090 Ti worker,
where final observed averages were about 6.8 and 8.0 t/s respectively. K65 ran
on a different RTX 3090 node and reached about 9.7 t/s. Node, GPU, and storage
differences confound that number; none is a native-Windows result.

Post-test protocol audit found that the candidate masks were ranked from every
prefill position of six uncensored K0 coding prompts but only six decode tokens
per prompt. They are therefore prompt/prefill-informed masks, not masks learned
over complete long K0 generations. This is a material limitation and the next
candidate selection must include full-decode routing mass before another bake
decision.

## Pending gates

1. Trace complete uncensored K0 decode sessions on the fixed coding learn split
   and rebuild K60/K62.5/K65 from full-session routing mass.
2. Re-run the virtual-mask `n=3` quality gate; do not emit a pack for a mask
   below the K0 reference grade.
3. Apply and build the native Windows embedded-mask loader patch.
4. Emit only the winning compact pack on the pod.
5. Assemble and inspect the NTFS sparse artifact on Windows.
6. Measure native Windows quality, VRAM/RAM tier residency, routed SSD bytes,
   cache misses, and throughput. Zero SSD during measured inference is a
   separate fail-closed gate.

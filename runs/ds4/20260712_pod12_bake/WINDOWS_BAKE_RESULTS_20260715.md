# Native Windows coding bake campaign - 2026-07-15

Status: full-decode K0 learn, held-out routing, and virtual-mask quality gates
complete. Physical K60/K75 mass-ranked packs are being streamed to R2. This
file separates measured facts from pending gates.

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

## Initial short trace campaign

The learn split contains six coding prompts: C, C++, Python, JavaScript, HTML,
and SQL. The held-out split contains PowerShell, Rust, Go, and container-stack
tasks. Both used K0 routing with router weights and opt-in prefill tracing. This
first protocol requested only one decode token per prompt; it is retained as a
historical negative protocol result, not as the current candidate source.

Measured trace totals:

| Split | CSV rows | Routed layers |
|---|---:|---:|
| Learn | 18,343 | 43 (`0..42`) |
| Held-out | 8,889 | 43 (`0..42`) |

The `positions` field in the original summaries is not a token total because
position ids restart for every request. CSV row count and per-layer rows are the
authoritative measurements.

## Full-decode correction

The corrected protocol traced every K0 prefill position and every generated
token. Six learn prompts were split over two pods, then merged byte-for-byte in
CLI order. Four held-out prompts were traced the same way and merged separately.
No masked run contributed to either ranking or held-out coverage.

| Split | CSV data rows | Maskable rows (`L3..42`) | Merged SHA-256 |
|---|---:|---:|---|
| Learn | 547,183 | 545,920 | `df0411790422c6651d06bc0f6fd955b3fd66b1c240ce1e880f1af25b7406e6f6` |
| Held-out | 334,929 | 334,320 | `080effd7563a1e3165402aaa197d24f3c1f1be2e3710ccfd4041b26d55e9ec43` |

All ten outputs stayed coherent. Nine ended naturally. `html_cyberpunk` reached
the configured 3,200-token limit while still producing coherent HTML; its
trace is therefore a valid budget-truncated coding sequence, not a completion
or quality pass. The Python output was coherent but violated its no-external-
dependency instruction by using `aiohttp`; this is recorded as an instruction-
following defect rather than hidden.

The new request index records file sizes observed by the client at response
boundaries. Those offsets are not exact row boundaries because the server's
trace `FILE*` retains a partial stdio block until a later flush or shutdown.
The merged traces use whole validated files and do not use those offsets for
cutoff or attribution.

## Initial short-trace held-out coverage

Coverage is recomputed by `scripts/score_routing_mask_coverage.py` from the
committed held-out CSV and mask files.

| Mask | Calls covered | Router mass covered | Rows with all six covered | Misses | Worst layer mass |
|---|---:|---:|---:|---:|---:|
| K60 / keep154 | 97.7576% | 98.3860% | 88.4541% | 1,114 / 49,680 | 94.5477% |
| K62.5 / keep160 | 98.0113% | 98.5146% | 89.4565% | 988 / 49,680 | 94.6898% |
| K65 / keep166 | 98.2045% | 98.6194% | 90.3986% | 892 / 49,680 | 94.8023% |

These figures measure overlap with uncensored held-out routing. They do not
prove functional equivalence or losslessness.

## Full-decode held-out coverage

The ranking below is learned only on the six full-decode learn sessions and is
scored on the four disjoint full-decode held-out sessions. `All six` means that
every router-selected expert in a row is retained.

| Retained / layer | Calls covered | Router mass covered | All six | Worst layer mass |
|---|---:|---:|---:|---:|
| 154 / 256 (K60) | 90.4551% | 91.4909% | 56.2482% | 77.5914% |
| 160 / 256 (K62.5) | 91.5962% | 92.4967% | 60.4274% | 78.5139% |
| 166 / 256 (K65) | 92.7904% | 93.6173% | 64.6898% | 84.7082% |
| 172 / 256 (K67.2) | 93.7816% | 94.4607% | 68.7560% | 85.5162% |
| 179 / 256 (K70) | 94.8446% | 95.3146% | 73.2753% | 86.1740% |
| 192 / 256 (K75) | 96.9771% | 97.5360% | 83.5559% | 90.1978% |
| 205 / 256 (K80) | 98.2193% | 98.5366% | 89.9752% | 91.2812% |
| 218 / 256 (K85) | 98.9147% | 99.0582% | 93.7234% | 91.9448% |
| 230 / 256 (K90) | 99.5565% | 99.6760% | 97.3938% | 97.6601% |
| 243 / 256 (K95) | 99.8973% | 99.9277% | 99.3880% | 99.7223% |

The full-decode measurement retracts the earlier impression that a static
60-65% coding mask is close to lossless. This overlap result does not itself
grade generated quality, but it defines the static-mask miss pressure that the
functional gate must explain.

## Exact payload plans

GGUFReader measured 1,328 tensors, 129 routed tensors, and 43 routed layers.
Each layer/expert `(gate, up, down)` triplet is `7,077,888` bytes. Layers `0..2`
are hash-routed and remain full in the current implementation; masks apply to
layers `3..42`.

| Candidate | Payload bytes | Payload GiB | Saved GiB | Extents |
|---|---:|---:|---:|---:|
| K60 | 57,842,328,448 | 53.8699 | 26.8945 | 7,155 |
| K62.5 | 59,541,021,568 | 55.4519 | 25.3125 | 7,029 |
| K65 | 61,239,714,688 | 57.0339 | 23.7305 | 6,841 |
| K67.2 | 62,938,407,808 | 58.6160 | 22.1484 | 6,601 |
| K70 | 64,920,216,448 | 60.4617 | 20.3027 | 6,330 |
| K75 | 68,600,718,208 | 63.8894 | 16.8750 | 5,715 |
| K80 | 72,281,219,968 | 67.3171 | 13.4473 | 4,921 |
| K85 | 75,961,721,728 | 70.7449 | 10.0195 | 3,886 |
| K90 | 79,359,107,968 | 73.9089 | 6.8555 | 2,803 |

In the superseded short protocol, K65 cost 3.1641 GiB over K60 while increasing
held-out mass coverage by only 0.2334 percentage points. With full-decode data,
the same capacity increase buys 2.1264 points (91.4909% to 93.6173%), while both
remain far from complete held-out coverage.

The full-decode masks change the trade-off materially. K70 is 60.4617 GiB but
covers only 95.3146% held-out mass. K75 reaches 97.5360% at 63.8894 GiB,
leaving little practical room for Windows, KV, CUDA buffers, and server state.
K80 already requires 67.3171 GiB before those runtime requirements. This is a
measured payload/capacity conflict, not yet a native-Windows residency verdict.

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

`pack-stream` extends the packer with a binary-only stdout path, atomic status
receipt, and SHA-256 for the payload, embedded manifest, and complete pack. The
R2 runner records separate packer/uploader exits and requires the remote object
size to match the emitted byte count. This avoids a 54-64 GiB temporary file on
the 50 GiB-free pod disks.

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
prefill position of six uncensored K0 coding prompts but only one decode token
per prompt. The CSV `n=6` field is router top-k width, not a decode-token count.
They are therefore prompt/prefill-informed masks, not masks learned
over complete long K0 generations. This is a material limitation and the next
candidate selection must include full-decode routing mass before another bake
decision.

The replacement full-decode mask gate ran as two controlled pod-local groups,
each with its own K0 reference and three temperature conditions per arm:

- RTX 3090 Ti worker: K0, K60, K70;
- RTX 3090 worker: K0, K65, K75.

Each arm used the same dashboard prompt, cache 1024, context 4096, temperatures
`0`, `0.2`, and `0.7`, HTML-close/repetition stop guard, and L0-L3 grading.
These are three robustness conditions, not three iid repeats at one temperature.

| Pod / GPU | Arm | temp 0 | temp 0.2 | temp 0.7 | Median |
|---|---|---:|---:|---:|---:|
| A / RTX 3090 Ti | K0 | L2 | L1 | L1 | L1 |
| A / RTX 3090 Ti | K60 mass | L2 | L2 | L2 | L2 |
| A / RTX 3090 Ti | K70 mass | L2 | L1 | L2 | L2 |
| B / RTX 3090 | K0 | L2 | L2 | L2 | L2 |
| B / RTX 3090 | K65 mass | L2 | L1 | L1 | L1 |
| B / RTX 3090 | K75 mass | L1 | L2 | L2 | L2 |

The Linux oracle throughput is recorded in
`quality_full_decode_mass_weight_results_20260715.csv`, but it is not a native
Windows performance result. K60 mass was the only candidate to score L2 in all
three conditions while retaining a payload below 64 GiB. It therefore becomes
the primary capacity candidate. K75 mass is also packed because the user asked
to measure its practical residency; its `63.8894 GiB` payload leaves no inferred
headroom, so native-Windows capacity remains an explicit gate rather than a
pre-test rejection.

## Mean-selected-weight control

A separate control ranked experts by mean selected gate weight:
`sum(selected_weight) / selected_calls`. It deliberately removes frequency,
unlike routing mass. Masks were still learned only from the six uncensored K0
full-decode sessions.

| Mask | Learn calls | Learn mass | Held-out calls | Held-out mass | Held-out all six | Worst layer mass |
|---|---:|---:|---:|---:|---:|---:|
| K60 mean-weight | 82.2899% | 88.0645% | 77.5733% | 80.6544% | 24.7320% | 64.5858% |
| K75 mean-weight | 91.1475% | 94.2993% | 87.2844% | 88.2741% | 45.0497% | 75.3200% |

The functional control produced:

| Pod / GPU | Arm | temp 0 | temp 0.2 | temp 0.7 | Median |
|---|---|---:|---:|---:|---:|
| A / RTX 3090 Ti | K60 mean-weight | L2 | L2 | L2 | L2 |
| B / RTX 3090 | K75 mean-weight | L1 | L1 | L2 | L1 |

K60 mean-weight surviving one dashboard panel does not override its much lower
held-out routing coverage. The bake candidates remain the mass-ranked masks.

## Pending gates

1. Finish and verify the K60/K75 mass pack streams on R2; reject any pipeline or
   remote-size mismatch.
2. Download each pack and verify its complete SHA-256 against the producer
   receipt before unpacking.
3. Assemble and inspect the NTFS sparse artifacts on Windows.
4. Apply and build the native Windows embedded-mask loader patch.
5. Measure native Windows quality, VRAM/RAM tier residency, routed SSD bytes,
   cache misses, and throughput. Zero SSD during measured inference is a
   separate fail-closed gate.
6. Keep K60 as the primary candidate; promote K75 only if measured host-memory
   residency leaves sufficient runtime headroom.

# APPENDIX — the low-bit quant ladder (DeepSeek-V4-Flash / ds4)

Reference table so we stop losing track. "IQ1.5" (owner's term) = the band **between
1 and 2 bpw** — i.e. the IQ1_S / IQ1_M family (~1.5–1.9 bpw). Updated 2026-07-22.

## 1. The bpw ladder (routed-expert quant), smallest → largest

| Name | ~bpw (experts) | "-XL" note | Full-model size (V4-Flash) | Sidecar-experts size |
|---|---:|---|---:|---:|
| **Q1_0** (our custom type 41) | **1.125** | — | — | ~36 GB (`ds4-q1-layers0-42-derived.gguf`) |
| **IQ1_S** | ~1.56 | -XL = non-expert tensors → Q8_0 (raises to ~1.73) | teamblobfish IQ1_S-XL **57 GiB** | — |
| **IQ1_M** | ~1.75 | -XL → ~1.91 | teamblobfish IQ1_M 60 GiB / IQ1_M-XL 63 GiB; unsloth UD-IQ1_M ~87 GB | — |
| **IQ2_XXS** | ~2.06 | ours 2.21 | **`ds4-2bit.gguf` = 86.72 GB** (our base) | — |
| IQ2_XS | ~2.31 | | teamblobfish IQ2_XS-XL 81 GiB | |
| Q2_K | ~3.01 | | 100 GiB | |
| Q4_K | ~4.9 | | antirez Q4K-imatrix 164.6 GB | |
| Q8_0 | 8.5 | | 282 GiB | |

"XL" (teamblobfish/unsloth convention) = experts at the named low-bit, but **non-expert
tensors (output, embeddings, attention, compressors, hyper-connection, indexer, NextN)
pinned at Q8_0** — so the *effective* bpw is higher than the pure-expert bpw.

## 2. What WE have on disk / have run, and its STATUS

| Artifact | What | bpw | Calibration | STATUS / verdict |
|---|---|---:|---|---|
| `ds4-2bit.gguf` (86.72 GB) | our BASE model, IQ2_XXS | 2.21 | **ds4 CODING imatrix** (byte-identical to antirez IQ2-imatrix) | ✅ GOOD — the exact model everything runs on |
| `ds4-q1-layers0-42-derived.gguf` (~36 GB) | Q1_0 sidecar | 1.125 | **NONE — synthetic** ("sign bits + mean-abs scale only… NOT a quantization-quality result" per our own DS4_Q1_0_PORT_DESIGN.md) | ❌ STRAWMAN — never meant to be quality; source of the "1.1.1.1" collapse |
| `DeepSeek-V4-Flash-IQ1_S-XL.gguf` (61.5 GB, persadian) | IQ1_S-XL | 1.73 | wikitext imatrix (llama.cpp toolchain) | ❌ **BROKEN FILE** — raw concat of 2 shards; header exposes only 1066/1328 tensors; **experts for layers 34–42 MISSING**. Our "IQ1_S 37% routing / late-layer collapse" (WIN-G133-HSHADOW-IQ1S-AB) is an ARTIFACT of these missing deep layers, NOT IQ1_S quality. INVALID. |
| IQ1_S-XL proper split (downloading) | IQ1_S-XL, 2 shards merged | 1.73 | wikitext | ⏳ downloading + merge → the FIRST valid IQ1_S measurement |

## 3. The three invalidations (why every 1-bit pessimism today was on broken ground)

1. **Q1_0** = synthetic placeholder (never a quality quant).
2. **IQ1_S-XL** (our copy) = truncated file, missing layers 34–42 experts.
3. **Existing IQ1 quants everywhere** = calibrated on **WikiText** (generic prose), then
   we judged them on **coding** — their structural worst case.

→ A **complete, properly-merged, CODING-calibrated** low-bit expert quant has **NEVER been
measured** by anyone. That is the virgin road.

## 4. Roads open (owner: "tutte le strade vanno battute")

- **A. Valid IQ1_S** (zero code): download proper teamblobfish IQ1_S-XL split → `gguf-split
  --merge` → our existing `DS4_IQ1_S_EXPERT_SIDECAR` loader → re-measure. IN PROGRESS.
- **B. IQ1_M** (~1.75–1.91 bpw, more bits): NEEDS added IQ1_M kernel+loader in our fork
  (our runtime only loads IQ1_S / Q1_0 today). Deferred.
- **C. Coding-imatrix IQ1** (the real prize): collect a CODING imatrix (CUDA tooling exists —
  cchuter/llama.cpp `feat/v4-port-cuda`, or antirez PR #377) → quantize experts with
  importance weighting → the never-made coding-calibrated IQ1.
- **D. Companion / mixed tier** (owner's framing): don't make it all IQ1 — hot experts stay
  IQ2 (exact), cold experts go IQ1. Already sized in our roadmap: 8 hot IQ2/layer + rest
  IQ1_S = 47.5 GiB; 32 hot = 49.5 GiB. Works even with an imperfect IQ1 because cold
  experts carry less demand mass by definition.
- **E. STE distillation** (fallback): the capture+train ladder — only if quantization alone
  caps out. NOTE the STE 0.70 ceiling was measured distilling TOWARD the synthetic Q1;
  re-baseline against a real IQ1 first.

## 5. Key sources / provenance

- Our IQ2 coding-imatrix: `quantize.imatrix.dataset = ...rendered_prompts.txt` (ds4 packed,
  90,042 chunks) — GGUF metadata of `ds4-2bit.gguf`.
- IQ1_S-XL wikitext + truncation: GGUF metadata + tensor audit of the persadian file.
- Ecosystem: antirez/ds4 `gguf-tools/imatrix`; teamblobfish/DeepSeek-V4-Flash-GGUF;
  unsloth/DeepSeek-V4-Flash-GGUF; cchuter/llama.cpp `feat/v4-port-cuda`; antirez/ds4 PR #377.
- Ledger rows: WIN-G134-C7-PIVOT-NAIVE-Q1-ROOTCAUSE, WIN-G134-IQ1S-PROVENANCE-WIKITEXT,
  WIN-G134-IQ1S-TRUNCATED-INVALID (all 2026-07-22).

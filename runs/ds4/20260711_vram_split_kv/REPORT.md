# VRAM split on the 3060: fixed non-expert weights vs KV cache — how much can we free for the expert cache?

Date: 2026-07-11. Host: RTX 3060 12 GB (12288 MiB), WSL Ubuntu-24.04, bin `/root/ds4_pin/ds4`
(0031 endpoint), model `/root/models/ds4-2bit.gguf` (DeepSeek-V4-Flash IQ2XXS, 86.72 GB).
Probe: `probe_ctx.sh` — minimal expert cache (32), REAP mask `sessCyber_K23.txt`, greedy,
short prompt (~40 tok), `-n 4`, vary `-c` (ctx). Own lock `/tmp/ds4_vram_split.lock`; UI:8000
never touched; GPU exclusive (703 MiB baseline). Split derived from ds4's OWN accounting
(`context buffers …` line) + steady-state vs peak nvidia-smi + the model source.

## THE QUESTION
Of the ~10-11 GB that "non-expert weights + KV" occupy at work (velocity REPORT, commit 0681265),
how much is **KV (reducible/quantizable)** vs **fixed weights (incompressible without requant)**?
If we can free ~1-1.5 GB -> K23 clears the b12 (>=3.2 GB) cache -> hit>0.8 dream (temporal report,
fc1ecf8). If it is all fixed weights -> we are capped at K12.

## ARCHITECTURE — it is BEYOND MLA (bad-news branch, confirmed)
DeepSeek-V4-Flash uses **Compressed Sparse Attention (CSA) + Heavily Compressed Attention (HCA)**,
more aggressive than MLA. From `ds4.c` `DS4_SHAPE_FLASH` and `MODEL_CARD.md`:
- `n_head=64`, **`n_head_kv=1`**, `head_dim=512` (single latent KV head — MLA-style compression).
- Raw KV = **128-token sliding window** (`n_swa=128`); GPU keeps a raw ring up to ~ctx but it
  **caps** (raw_kv_rows=4352 even at ctx 5500).
- Layers 0-1: raw only. Even layers >=2: **ratio-4** compressed (+ indexer, top_k=512).
  Odd layers >=3: **ratio-128** compressed. So compressed rows = ctx/4 (or ctx/128) + 2, tiny.
- KV stored f32 (`raw_kv = raw_cap * 512 * 4 B`). => KV is small **by construction**; there is no fat.

## MEASURED VRAM (probe, 4 contexts)

| ctx | context buffers (KV+attn+prefill scratch) | steady-state resident | transient peak |
|---|---|---|---|
| 512  | 50.66 MiB  (raw_kv 512,  comp 130)  | 9860 MB | 11653 MB |
| 2048 | 208.09 MiB (raw_kv 2048, comp 514)  | 9881 MB | 11673 MB |
| 4096 | 432.00 MiB (raw_kv 4096, comp 1026) | 9945 MB | (sampler missed) |
| 5500 | 482.85 MiB (raw_kv 4352, comp 1377) | 9953 MB | 11745 MB |

**Steady-state resident is FLAT at ~9.86-9.95 GB across the whole ctx range** (grows <100 MB from
ctx 512->5500). The big "context buffers" numbers (50->483 MiB) are dominated by **transient prefill
scratch** (prefill_chunk, freed after prefill); the KV that actually persists through decode is
even smaller. The 11.6-11.75 GB "peak" the velocity report saw is a **transient** (load-staging +
prefill scratch, ~+1.8 GB), not persistent occupancy.

## THE SPLIT (at the working context, ctx 4000-5500)
From ds4's load log: token embedding span **0.99 GiB** + backbone tensors climbing to **7.86 GiB**
= **~8.85 GiB fixed non-expert weights**. Decomposition (consistent to the MB with the 9.9 GB plateau):

| component | GiB | reducible? |
|---|---|---|
| **Hash-layer experts (layers 0-2, resident, not streamed)** | **~5.06** | only by requant / streaming them |
| Token embedding | ~0.99 | requant only |
| Output head | ~1.0 | requant only |
| Attention (MLA/CSA proj, 43 layers) | ~1.5 | requant only |
| Shared experts (43) + norms | ~0.3 | requant only |
| **= fixed non-expert weights** | **~8.85** | **incompressible w/o requant** |
| CUDA context/driver + cache(32 exp, 0.21) | ~0.7 | fixed |
| **KV cache + attn arena (persistent, decode)** | **0.05->~0.48** | **quantizable — but tiny** |
| steady resident | **~9.9** | |
| transient peak (prefill scratch + load staging) | ~11.7 (+1.8 transient) | shrinkable (not KV) |

**KV is ~0.2-0.5 GB of the ~10 GB footprint (~2-5%), and most of that is transient prefill scratch,
not persistent.** The ~8.85 GiB is fixed weights, of which the single biggest lump is the **~5 GiB of
resident hash-layer experts** (layers 0-2, hash-routed every token, never pruned).

## HOW MUCH CAN WE REALISTICALLY FREE (without breaking quality)
- **Quantize the f32 KV ring -> int8/int4:** raw_kv at ctx 5500 ~= 43*4352*512*4 = 383 MB -> ~96 MB (int8)
  / ~48 MB (int4). Saves **~0.3-0.35 GiB** absolute maximum. (comp/indexer add a few tens of MB more.)
- **Cap context to the working length** (5500->2048): frees ~0.28 GiB of prefill/KV scratch (mostly
  from the transient peak, ~0 from steady resident).
- **Shrink prefill_chunk / load-staging:** attacks the ~1.8 GiB TRANSIENT peak (this is what squeezes
  the cache during load) — but this is prefill scratch, **not KV**.
- **Total on the KV/context axis: ~0.3-0.5 GiB.** Everything >=1 GiB lives in the FIXED weights
  (requant embedding/output/attention, or stream/requant the ~5 GiB hash-layer experts) — none of it KV.

## DOES K23 ENTER THE DREAM?
No. Free-for-cache today ~= **~2.0-2.5 GiB steady** (velocity measured ~220 experts ~= b5.5; k91 saw
~407 ~= b10 depending on when the cache is sized vs the transient). K23 needs **b12 = 3.2 GiB** for
hit>0.8. The KV/context axis frees only ~0.3-0.5 GiB -> lands at ~2.5-3.0 GiB, **still short of 3.2**,
and does not robustly clear hit>0.8. **K23 stays capped (hit ~0.53, ~1.5 t/s).**

**K12** is different: b9 = 2.4 GiB -> hit 0.92, compute-bound ~3-4 t/s (temporal report). That
**essentially fits today**, and the small KV/prefill saving locks it in. So the reachable "dream" on
the 3060 is **K12, not K23** — at the cost of a narrower mask (quality risk on wide/coding domains).

## VERDICT (8 lines)
1. **Split @ working ctx:** fixed non-expert weights **~8.85 GiB**; KV+attn arena **~0.2-0.5 GiB**
   (~2-5%, mostly transient prefill scratch); steady resident ~9.9 GiB, transient peak ~11.7 GiB.
2. **MLA?** Yes and then some — CSA+HCA (n_head_kv=1, 128-tok sliding window, ratio-4/128 compressed).
   KV is already minimal by construction; **there is no KV fat to trim.**
3. **Realistically freeable via KV/context (no quality loss):** **~0.3-0.5 GiB** (int8/int4 KV + ctx cap).
   Everything >=1 GiB is fixed weights (requant, or stream the ~5 GiB resident hash-layer experts).
4. **K23 -> dream?** **NO.** KV frees <=0.5 GiB; K23 b12 needs +~1 GiB more -> cache tops out ~2.5-3.0 GiB
   < 3.2 GiB -> K23 stays hit ~0.5, ~1.5 t/s.
5. **Cap:** the 3060 is **fixed-weight-bound, not KV-bound.** Reachable dream is **K12 b9 (2.4 GiB,
   hit 0.92, ~3-4 t/s), which nearly fits today** — but K12 is a narrower mask (quality trade).
6. **KV-compression is a DEAD END** for the K23 dream. The freeable GBs are in the fixed weights:
   the ~5 GiB resident hash-layer experts (stream/requant) and the ~1.8 GiB transient prefill/load spike.
7. **Recommendation:** don't chase KV-quant for VRAM. To seat a 3.2 GiB cache: (a) accept K12 dream
   now, or (b) attack the fixed weights — stream the hash-layer experts and/or shrink prefill_chunk to
   cut the transient peak, or (c) bigger hardware. K23-in-the-dream is not reachable by squeezing KV.
8. Probe artifacts: `runs/ds4/20260711_vram_split_kv/{probe_ctx.sh,results.log,ctx_*/}`. GPU left free;
   UI:8000 untouched.

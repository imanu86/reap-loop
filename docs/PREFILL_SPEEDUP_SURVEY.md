# PREFILL SPEEDUP SURVEY — ds4 / DwarfStar + ecosystem, TTFT sul 3060 (2026-07-11)

**Scopo.** Il prefill (TTFT) è il collo MAI risolto di REAP-LOOP: **115–213 s** nei
run lunghi (gap J12, `docs/NEXT_STEPS_PLAN_20260710.md`; ledger
`docs/EXPERIMENTS_LEDGER.md`), **~11 s** per un prompt corto (78 tok). Su RTX 3060
12 GB in `--ssd-streaming`, il prefill deve toccare gli expert di TUTTO il prompt →
è **I/O-bound**. Questo doc mappa: (§1) come ds4 fa il prefill oggi + leve esposte;
(§2) tecniche ecosistema prefill-specifiche; (§3) candidate adozioni 3060 per
valore/effort con test minimo P4; (§4) stima I/O-expert vs compute del nostro
115–213 s e quale leva taglia di più. Companion prefill-centrico di
`docs/MOE_ECOSYSTEM_SURVEY_20260711.md` (organizzato per-leva) e
`docs/FORK_SURVEY_20260710.md` (fork ds4).

> **ONESTÀ.** Tutti i numeri di terzi sono `[CLAIM]` non ri-verificati da noi (HW,
> prompt, quant, metodo loro). I nostri `[EST]` in §4 sono stime da aritmetica di
> primo ordine, NON misure: la strumentazione TTFT segmentata esiste (CLAIM-008,
> timing TTFT/1-64/65-256/257+) ma il breakdown I/O-vs-compute NON è ancora
> misurato. In conflitto vince `docs/CLAIMS_CURRENT.md`.

**Config di riferimento.** DeepSeek-V4-Flash "DwarfStar", 158B MoE, shape FLASH =
**43 layer × 256 expert/layer, top-6**, routed 2-bit (up/gate IQ2_XXS, down Q2_K),
expert ≈ **6.75 MiB**. RTX 3060 12 GB. Launcher pratico: `--ssd-streaming`,
`--prefill-chunk 512`, `DS4_PACE_PREFILL_APPLY=1`, `DS4_PACE_PREFILL_WAIT_WRAP=0`.
Cache expert residente misurata ≈ **400 slot** (~2.7 GiB; il resto dei 12 GB va a
pesi non-routed 8-bit + KV + scratch) — CLAIM-008, memoria K91.

---

## 1 — Come ds4 fa il prefill OGGI (+ leve esposte)

Ricostruito da `docs/references/DwarfStar_ds4_README.md`, dai commit upstream
(`gh api repos/antirez/ds4`, giugno 2026) e dai nostri recon
(`docs/dspark/RECON_MTP_DS4.md`, `docs/PACE_DESIGN.md`).

**1.1 Chunked prefill.** Il prompt è processato in **chunk** con forward batched.
Metal: `DS4_METAL_PREFILL_CHUNK` (default **4096**, `0` = tutto il prompt in un
batch se la memoria regge, `2048` per il path official-vector). Distribuito:
`--dist-prefill-chunk` (default 4096) + `--dist-prefill-window` (chunk in volo).
Il **nostro launcher CUDA usa `--prefill-chunk 512`** (molto più piccolo del
default Metal). Il chunk Metal riusa lo stesso graph range-capable layer-major
per ogni chunk (commit README 0a99ce35f + `Refactor streaming expert cache API`
1cfa5cc6d).

**1.2 Amortizzazione expert DENTRO il chunk.** Poiché il prefill processa molti
token insieme, un expert caricato una volta serve **tutti** i token del chunk
instradati a lui. ds4 spinge questo con **expert-major MoE prefill tiles**
(`52246548a Add wide-token MoE prefill tiles (n64/n128 mul_mm_id)`, env
`DS4_METAL_MOE_TILE_MAX` default 128): tile a 64/128 token per Q4_K/Q2_K/IQ2_XXS,
raggruppano i token per expert così il peso è letto una volta per tile. ⚠ Metal-
only, e la PR è stata **revertata** (072bc0feb) — il default è tornato ai tile 32.
Conseguenza README, testuale: *"Long prefills can still be fast; generation is
more sensitive to cache misses because every new token routes through experts
again."* → **il prefill è meno I/O-sensibile del decode grazie all'amortizzazione
batch, MA solo se la cache regge il working-set del chunk** (§4: sul 3060 non
regge).

**1.3 Streaming SSD (il path che ci riguarda).** `9ba160ae8 Implement SSD
streaming` (05-30, Metal) → `bbd069d3e Add CUDA and ROCm SSD streaming` (06-14).
Pesi non-routed residenti; expert routed in cache in-VRAM, **load da GGUF su
miss**. Leve: `--ssd-streaming`, `--ssd-streaming-cache-experts NGB` (budget in
expert INTERI, non byte), `--ssd-streaming-preload-experts N` (preload hot),
`--ssd-streaming-cold` (solo misura). Auto-budget = 80% working-set − non-routed.
Cache troppo grande → capped (`cd5742896 Cap oversized SSD streaming expert
caches`), margine rilasciato su mlock-fail (`7a77a2821`).

**1.4 Cutoff decode↔prefill per quant.** `57b8a4ca2 Tune SSD streaming decode
prefill cutoff by quant` + `730fc868c Fix short SSD streaming prefill cutoff`:
span corti di token usano un path "streaming **decode-prefill**" batched
(`metal_graph_use_streaming_decode_prefill`); soglia = **64** token (wide default)
o **18** per quant non-wide. Env: `DS4_METAL_DISABLE_STREAMING_DECODE_PREFILL`,
`DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL`. È la versione ds4 della "soglia
adattiva prefill/decode" (cfr. ik_llama.cpp `32·total/active`, §2).

**1.5 Prefill distribuito pipelined (solo multi-GPU).** Assembly-line: coordinator
fa chunk N+1 mentre il worker fa chunk N → speedup **1.38–1.85×** `[CLAIM]` su 2×
M5 Max. *Solo il prefill* accelera; il decode è autoregressivo. Non applicabile al
nostro single-3060, ma è la prova che ds4 **sa già** pipelinare i chunk di prefill.

**1.6 KV disk-cache / prefix reuse.** `--kv-disk-dir` salva il checkpoint del
prefix renderizzato (SHA1) → un secondo prompt che estende lo stesso prefix
**salta il prefill** (`Disk KV Cache`, README). Rilevante per agenti (Claude Code
manda ~25k tok iniziali una volta). È l'unico "prefill = 0" già in ds4.

**1.7 Le NOSTRE leve prefill (patch REAP/PACE, non upstream).**
`DS4_PACE_PREFILL_APPLY=1` — impara la mask keep-K dal routing del PROMPT e la
applica a `tok=0` (mask dinamica di prompt, non statica di dominio; PACE §5, nota
2026-07-08 commit `c8dd670`). `DS4_PACE_PREFILL_WAIT_WRAP` — se WRAP attivo,
aspetta il page-in del working-set derivato dal prompt prima di decodare (smoke
locale: 6.07 GiB toccati in 445 ms). **WRAP** (`ds4_reap_prefetch_working_set`,
patch 0013) = page-in bulk host-side threaded del working-set tenuto. J11/J21:
hidden-readback SPEX **disabilitato perché rendeva il TTFT inusabile**;
`DS4_SPEX_HIDDEN_GPU_SCORE=1` esiste ma non alimenta residency/prefetch.

**Sintesi §1.** ds4 fa prefill **chunked + expert-major batched + streaming cache
LRU on-miss**; espone chunk-size, cache-budget, preload-hot, decode-prefill cutoff,
disk-KV prefix-reuse, e (multi-GPU) pipeline. Ciò che **NON** fa di default: un
**bulk prefetch parallelo dell'unione-expert del prompt** prima/durante il prefill
(il prompt è NOTO ma gli expert entrano on-demand per tile/chunk). Il nostro WRAP è
il candidato più vicino, ma è pensato per il decode e sul 3060 il prefetch
*rallenta* il t/s di decode (0.82 vs 1.27, CLAIMS_CURRENT §PREFETCH) — sul prefill
non è mai stato isolato come misura.

---

## 2 — Tecniche ecosistema (prefill-specifiche)

| Tecnica | Fonte | Cosa (per il prefill) | Numeri `[CLAIM]` | Fit 3060 |
|---|---|---|---|---|
| **Layered Prefill** | arXiv 2510.08055 | Partiziona il modello per **gruppi di layer** (non per token): elimina i **reload di expert indotti dal chunking**. Interleava prefill/decode per gruppo-layer | chunked prefill **amplifica il traffico expert +39%**; layered → **TTFT −70%**, E2E −41%, energia/tok −22% | ⭐⭐ Attacca ESATTAMENTE il nostro moltiplicatore di re-read (§4). Il +39% è il floor datacenter; sul nostro cache-starved è molto peggio |
| **MoE-Prefill (zero-redundancy)** | arXiv 2605.02960 | Rimuove gli overhead ridondanti di expert-load nel prefill MoE servito | "zero redundancy" `[CLAIM da fetch]` | Stessa diagnosi (re-load ridondanti); da leggere per il meccanismo esatto |
| **DuoServe-MoE** | arXiv 2509.07379 | Prefill: **two-stream CUDA** che overlappa il **prefetch expert** con il compute non-MoE; predittore MLP-7layer | **TTFT 1.78–5.34×**; predittore ~0.6 ms / 300 MB VRAM | ⭐⭐ Il "prompt è noto → prefetcha in parallelo overlappando" in forma CUDA. 300 MB trascurabili sui 12 GB |
| **ProMoE — proactive caching** | arXiv 2410.22134 | **Chunked prefetching** + early preemption + reordered inference per massimizzare overlap prefetch↔inference | (proactive vs reactive) `[CLAIM]` | Pattern per il nostro prefetch-del-prompt: chunkare il prefetch e riordinare per nascondere l'I/O |
| **kTransformers — AMX CPU GEMM** | kvcache-ai; SOSP'25 | Expert su **CPU** con kernel AMX durante il prefill; shared-on-GPU | **prefill >500 t/s @2048 tok**; **4.62–19.74×** vs prior | ⭐ Il prefill compute-bound può stare sulla CPU mentre la GPU fa altro; richiede AMX (non su tutte le CPU) |
| **SP-MoE** | arXiv 2510.10302 | Speculative decoding + **prefetching** per MoE offload | `[CLAIM da fetch]` | Il draft-forward nasconde l'I/O; abbiamo l'infra MTP ma non la usiamo per il prefetch |
| **PreScope** | arXiv 2509.23638 | Prefetch per MoE **resource-constrained**; hot-table offline per-gruppo-layer | hit top-4 **94–99%**; +141% vs Klotski | Predittore prefetch per fascia-VRAM bassa (la nostra); keep-K non uniforme per layer |
| **ESS — LRU-Warmup + UVA** | arXiv 2512.10576 | Pre-riscalda la cache con le top-2K entry delle ultime 32 finestre di prefill; UVA fine-grained (656 B) | H2D **0.79→37 GB/s**; +69% ctx32K, +123% ctx128K | ⭐ Se i transfer SSD→VRAM sono frammentati, UVA li rende contigui; guadagno cresce col ctx |
| **ik_llama.cpp — soglia adattiva** | knightli.com | Punto di switch prefill/decode = `32·total_exp/active_exp` invece di 32 fisso | (solo formula) | Il nostro cutoff §1.4 è fisso 18/64; scalarlo con `active/total` (top-6/256 è molto sparso) |
| **DALI — residual prefetch** | arXiv 2602.03495 | Predice il routing di N+1 dal residuo per lanciare la read in anticipo | **prefill 7.62×** vs llama.cpp | Segnale residual per prefetchare gli expert PRIMA del forward |
| **Fate — cross-layer gate** | arXiv 2502.12224 | Predice routing dagli input-gate di layer adiacenti + quant in cache | **prefill 4.5×** vs on-demand, hit 99% | Match vicino a mask+cache+2-bit; segnale cross-layer economico |
| **llama.cpp PR #25294** | ggml-org | Stream expert da disco: **O_DIRECT**, **wave-partitioned prefill**, output bit-exact | **prefill 5.3×** / decode 2.4× vs mmap+CPU-MoE | Equivalente mainline del nostro streaming; 2 trucchi diretti (O_DIRECT, wave-partition) |
| **ggml async prefetch n+1** | llama.cpp PR #21067 | Overlap transfer CPU→GPU layer n+1 durante compute layer n (richiede `--no-mmap`) | guadagni ubatch 512–2048; **incerto a batch=1** | Prefetch al confine layer; il prefill è batch-grande → regime favorevole (a differenza del decode) |

**Nota trasversale.** Il tema dominante 2025-26 sul prefill MoE-offload è **un solo
bug**: il **chunking per-token ri-carica gli expert ad ogni chunk** (Layered
Prefill lo quantifica +39%). Tutte le cure sono varianti di: (a) non chunkare per
token (layered/whole-prompt), (b) prefetchare in parallelo l'unione nota del prompt
(DuoServe/ProMoE/DALI), (c) tenere gli expert in un medium più veloce dell'SSD
(RAM/UVA/CPU-AMX). Sono esattamente le 3 leve del 3060 in §3-§4.

---

## 3 — Candidate adozioni per il 3060 (valore/effort, test minimo P4)

**Principio P4.** Ogni test minimo **identifica un parametro/legge**, non fa uno
sweep di config. Vincolo dai nostri dati: `DS4_PACE_PREFILL_APPLY=1` è già live ma
J12 avverte che applicare la mask prima del tok-1 **viola l'ipotesi-qualità** (i
primi token vogliono K0 pieno) → la cura desiderata è **prefetch-dal-prompt SENZA
mask apply** (`NEXT_STEPS_PLAN`, "unica cura nota per il prefill 115-213 s").

| # | Candidato | Leva | Valore/Effort | Test minimo (parametro, non config) |
|---|---|---|---|---|
| **1** | **Sweep chunk-size × medium** (`--prefill-chunk` 512 vs 2048 vs 4096 vs 0; SSD vs RAM-leva) con TTFT segmentato + byte-letti-da-SSD | chunk/streaming | **ALTO/NULLO (no code)** | Parametro = **moltiplicatore di re-read** = byte-SSD(prefill) / working-set-teorico, in funzione di chunk-size. Isola quanto dei 115-213 s è amplificazione da chunking (§4). Zero patch: solo env + strumentazione TTFT già esistente (CLAIM-008) |
| **2** | **RAM-leva** (expert nella page-cache RAM, `KEEP_MODEL_PAGES`) per il prefill | streaming/medium | **ALTO/BASSO** | Parametro = **speedup-per-medium** = TTFT(expert-in-RAM) / TTFT(expert-da-SSD) a working-set costante. Confermato ultima commit come LA leva 3060; qui misurato **sul prefill** (non decode). Nessun rischio qualità |
| **3** | **Prompt-derived async bulk prefetch SENZA mask apply** (nuovo modo: unione-expert del prompt → read parallele deep-queue, overlap col compute, stage in RAM; `DS4_PACE_PREFILL_APPLY=0` + WRAP-prefill) | prefetch | **ALTO/MEDIO** | Parametro = **frazione di I/O-expert nascosta dietro il compute** = 1 − (TTFT − TTFT_compute_only)/TTFT_IO_baseline. È la "cura nota" + dodges J12. DuoServe/ProMoE-shaped |
| **4** | **Prefetch parallelo vs on-demand** (queue-depth): saturare l'NVMe con le read dell'unione-prompt invece che seriali per tile | prefetch/BW | **MEDIO-ALTO/BASSO** | Parametro = **GB/s effettivi** vs queue-depth su read da 6.75 MiB. 1 curva BW-vs-QD; dice se il collo è volume (bytes) o latenza-seriale |
| **5** | **Tensor bundling su disco** (up/gate/down dell'expert contigui nel GGUF) | streaming/IOPS | **MEDIO/BASSO** | Parametro = **frazione read contigue vs random** per expert sul GGUF attuale. Se frammentato → repack. 1 numero (cfr. MOE_ECOSYSTEM §4 #1) |
| **6** | **O_DIRECT + wave-partition** (PR #25294) sul prefill | streaming | **MEDIO/BASSO** | Parametro = **pollution page-cache** = Δ byte-SSD con/senza O_DIRECT su prefill fisso. Valida anche il claim bit-exact |
| **7** | **Cutoff prefill/decode adattivo** `32·total/active` (ik_llama.cpp) vs 18/64 fisso | chunk | **MEDIO/BASSO** | Parametro = **soglia ottima** per top-6/256; misura TTFT vs soglia. 1 punto |
| **8** | **Layered prefill** (partizione per gruppo-layer, 2510.08055) | scheduler | **ALTISSIMO/ALTO** | Parametro = **reload-eliminati** = byte-SSD(layered) / byte-SSD(chunked) a parità di prompt. Effort alto (riscrive lo scheduler prefill); da fare solo se #1 conferma che l'amplificazione da chunk domina |

**Le 3 da fare per prime:** **#1** (sweep chunk×medium, gratis, quantifica il
moltiplicatore) → **#2** (RAM-leva sul prefill, conferma il medium) → **#3**
(async prefetch senza mask apply, la cura nota che dodges J12). #8 (layered
prefill) è la cura *strutturale* ma va giustificata da #1.

---

## 4 — Stima: I/O-expert vs compute nel nostro 115–213 s

**Aritmetica di primo ordine `[EST]` (non misurata).**

- **Working-set expert del prompt** W: per un prompt lungo/diverso, l'unione dei
  top-6 su molti token per layer → fino a **256 expert/layer** × 43 layer =
  **11 008 expert** = 11 008 × 6.75 MiB ≈ **72–74 GiB**.
- **Cache residente** C ≈ **400 slot** ≈ 2.7 GiB → **hit-rate a freddo ≈ C/W ≈
  3–4 %**: ~96 % degli expert-load del prefill sono **read da SSD**.
- **Amplificazione da chunking**: con cache ≪ W, ogni chunk ri-streama quasi tutto
  W (gli expert del layer L sono evicati prima che il chunk successivo li richieda).
  Byte-SSD ≈ `ceil(P/chunk) × W × (1−hit)`. Con `--prefill-chunk 512` e un prompt
  di P token, sono **P/512 passate** da ~72 GiB. Layered-Prefill (2510.08055)
  quantifica **+39 %** di traffico da chunking *su GPU datacenter con cache grande*;
  sul nostro C/W≈4 % il fattore è **molto maggiore di 1.39**.
- **Compute floor**: i numeri README Metal (250–463 t/s prefill) sono con expert
  **RESIDENTI** (nessun I/O). Il DELTA "resident → streamed" è puro I/O. Scalato al
  3060, il compute di prefill per i nostri prompt è dell'ordine di **decine di
  secondi**, non centinaia.
- **Chiusura dei conti**: 72 GiB × (P/512 passate) / (BW effettiva NVMe **~1–3
  GB/s** per read semi-random da 6.75 MiB) → **decine→centinaia di secondi**,
  coerente con 115–213 s per i prompt lunghi e ~11 s per 78 tok (1 chunk, poche
  passate, meno expert unici).

**Verdetto `[EST]`.** Dei 115–213 s, la **parte dominante (grosso modo ~80–95 %) è
I/O-expert da SSD**, gonfiata dal **moltiplicatore di re-read del chunking** (cache
400 ≪ working-set 11 008). Il compute è la minoranza (decine di s).

**Quale leva taglia di più (ordine atteso):**

1. **Ridurre W (mask keep-K in prefill)** — 256→K per layer. K=23 → W ≈ 43×23×6.75
   MiB ≈ **6.5 GiB** (≈ **11× meno** byte, e vicino al cache-able → crolla anche
   l'amplificazione). È il taglio più grande in assoluto. ⚠ **caveat J12**
   (qualità dei primi token) → va fatto in modalità **prefetch-senza-mask-apply**
   (#3), non applicando la mask al tok-0.
2. **Cambiare medium (RAM-leva, #2)** — non riduce i byte ma porta il ~96 % di
   read da SSD (~1–3 GB/s) a **RAM (~10–20 GB/s)** → **~5–10×** sul termine I/O.
   Zero rischio qualità. È la leva confermata dall'ultima commit.
3. **Ridurre il moltiplicatore (chunk più grande / layered, #1/#8)** — meno passate
   → meno re-read. `--prefill-chunk` 512→2048/4096/0 attacca direttamente il
   `ceil(P/chunk)`.
4. **Nascondere l'I/O dietro il compute (async prefetch parallelo, #3/#4)** — non
   riduce i byte ma li **overlappa**; utile dopo aver ridotto volume+medium.

**La #1 da provare (misura, no code):** il **sweep chunk-size × medium con TTFT
segmentato + byte-SSD** (#3-tabella). In un colpo dice (a) il moltiplicatore di
re-read reale, (b) quanto vale la RAM-leva sul prefill, (c) il compute floor →
sblocca la scelta informata tra ridurre-W, cambiare-medium e ridurre-il-chunking.

---

## 5 — Provenienza

Scouting 2026-07-11. ds4: `docs/references/DwarfStar_ds4_README.md` +
`gh api repos/antirez/ds4/commits` (giugno 2026: `Add CUDA and ROCm SSD
streaming`, `Refactor streaming expert cache API`, `Fix distributed SSD streaming
layer slices`, `Tune SSD streaming decode prefill cutoff by quant`, `Add wide-token
MoE prefill tiles`). Ecosistema: WebSearch (Layered Prefill 2510.08055, MoE-Prefill
2605.02960, DuoServe 2509.07379, ProMoE 2410.22134, kTransformers, SP-MoE
2510.10302, PreScope 2509.23638, ESS 2512.10576, DALI 2602.03495, Fate 2502.12224,
llama.cpp PR #25294/#21067). Numeri di terzi `[CLAIM]` non ri-verificati; stime §4
`[EST]` da aritmetica, non misure. Companion di `docs/MOE_ECOSYSTEM_SURVEY_20260711.md`,
`docs/FORK_SURVEY_20260710.md`, `docs/PACE_DESIGN.md`, `docs/DYNAMIC_EXPERT_COMPRESSION_PLAN.md`.

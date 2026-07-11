# PIN-SUBSET VIABILITY + TEST-COVERAGE GAP ANALYSIS (post-divergence)

Data: 2026-07-11. Offline (no-GPU). Prerequisito della patch **0031 pin-keep**
(in authoring). Modello di riferimento: **mask-router e residenza-VRAM sono
ortogonali** (mask = qualita statica, residenza = velocita dinamica) — divergenza
confermata dall'audit e5a1455 (runs/ds4/20260711_pinning_divergence_audit/).

---

## COMPITO 1 — Distribuzione d'uso degli expert KEEP

**Domanda.** La 0031 pinna in VRAM un SOTTOINSIEME caldo dei keep (budget ~394
slot < working-set). Funziona solo se pochi expert dominano l uso. Misura, tra i
keep (i selezionabili sotto mask), la distribuzione di frequenza d attivazione per
layer.

### Sorgenti e metodo

- **Mask:** runs/ds4/20260711_local_clean_lowK/masks/sess{K12,K16,K23}.json
  (metodo session_mass_rank — i keep sono GIA il top-K per massa-gate della
  sessione coffee W50). 40 layer MoE mascherati osservati (3..42); boot-probe
  conta n_moe_layers=43, quindi il working-set canonico usa 43 (K23=989, K12=516);
  l empirico sui 40 layer tracciati e 920/480.
- **Trace pesati (coffee-family, la sessione da cui la mask e nata):**
  route_W50.csv + route_W130.csv (warmup replay cache1024) +
  podA_narrow_traces/a_coffee_full/route.csv. Stream keep-activation = 63.4k
  (K23) / 47.9k (K12) attivazioni.
- **Metrica:** frequenza d attivazione (conteggio; ogni selezione = un lookup di
  cache) per expert, ristretta ai keep, per layer. La frequenza-conteggio decide
  l hit-rate della residenza (la massa-peso decide la qualita).
- **CAVEAT vincolante:** i trace disponibili sono **full-model (non-mascherati)**,
  filtrati ai keep sono un PROXY della decode mascherata (non esistono route.csv
  sotto mask: i run K12/K23 hanno solo gen/diag). Sotto mask reale il router
  sceglie 6-da-keep ogni token, quindi densita keep maggiore e possibile
  redistribuzione. La copertura keep del trace full e 0.53-0.57 (coffee) vs 0.18
  (python cross-dominio), coerente con una mask coffee-specifica.

### (a)+(b) Concentrazione per layer — QUASI-UNIFORME dentro il keep

| Mask | k90 = #expert per il 90% degli hit/layer | #keep usati/layer | top-6 share (mediana) | Gini (slot keep) | k90 / keep_n |
|---|---|---|---|---|---|
| **K23** | mediana **15** (min 13, max 18) di 23 | **23/23** (tutti) | 0.65 | 0.50 | **0.66** |
| **K16** | ~13 di 16 | 16/16 | ~0.72 | ~0.46 | ~0.81 |
| **K12** | mediana **9** (min 7, max 10) di 12 | **12/12** (tutti) | 0.80 | 0.43 | **0.71** |

Istogramma k90 (K23): 13->4L, 14->8L, 15->16L, 16->5L, 17->6L, 18->1L.
Istogramma k90 (K12): 7->3L, 8->13L, 9->22L, 10->2L.

**VERDETTO (b): la distribuzione e QUASI-UNIFORME dentro il keep, NON concentrata.**
Tutti i keep vengono usati (23/23, 12/12), e servono **il 66-71% del keep set**
per coprire il 90% degli hit di quel layer. Non esiste un hot-core molto piu
piccolo del keep stesso: la mask session_mass_rank ha GIA scremato la coda calda,
quindi il residuo e piatto (Gini ~0.4-0.5, non ~0.8+). Conseguenza diretta: il
pin-sottoinsieme aiuta **solo nella misura in cui budget < working-set**, non
perche pochi caldi dominano.

### (c) Stabilita del set caldo — STABILE intra-sessione, DERIVA cross-dominio

- **Intra-sessione** (coffee, prima meta vs seconda meta per posizione), top-6
  caldi/layer: **Jaccard mediana 0.50** (min 0.20). Churn moderato ma il nucleo
  regge, coerente col fatto che **static-pin ~= LRU-dinamico** (sotto): pinni una
  volta, la deriva intra-sessione ha basso valore.
- **Cross-dominio** (coffee vs python / coffee vs json), top-8/layer:
  **Jaccard mediana 0.14** (min 0.00) — set quasi disgiunti. Il set caldo e
  **dominio-specifico**: un pin coffee-tuned e inutile su python/json, quindi 0031
  deve ri-derivare il pin **per-sessione** (premessa narrow: un dominio per
  sessione), non pinnare globalmente una volta.

### (d) Budget 394 slot vs working-set — il verdetto

Pin statico oracolo = pinna i (layer,expert) globalmente piu caldi fino a riempire
gli slot (esattamente cio che fa 0031). Confronto con LRU dinamico sullo stesso
stream:

| Mask | working-set (K x43) | hit@394 static-pin (0031) | hit@394 LRU-din. | slot per hit>=90% | Fit in 394? |
|---|---|---|---|---|---|
| **K12** | 516 | **0.959** | 0.956 | **319** | **SI** |
| **K16** | 688 | 0.886 | — | 414 | NO (-20 slot) |
| **K23** | 989 | **0.793** | 0.788 | **581** | **NO** |

- **static-pin ~= LRU-dinamico** ad ogni cap (K23 0.793 vs 0.788; K12 0.959 vs
  0.956). Il pin statico cattura tutta la localita temporale utile, quindi la
  rotazione dinamica della residenza (churn) ha valore marginale. **A cap stretto
  (258) lo static-pin BATTE l LRU** (K23 0.669 vs 0.630): la LRU spreca slot su
  expert one-shot.
- **Aggancio alla velocita:** sul pod cache-1024 (>= working-set 989) l hit sale a
  0.92-0.99 -> 11-17 t/s (PAPER.md sec.217). Sul 3060 i **394 slot < 989** portano
  l hit K23 a ~0.79: **e esattamente la pressione-VRAM che tiene K23 bloccato sul
  3060.**

### VERDETTO COMPITO 1

- **K12 -> pin-sottoinsieme VIABLE.** 319 slot per hit>=90%; a budget pieno 394
  da **hit 0.96**. Working-set (516) di poco sopra il budget e distribuzione
  abbastanza piatta ma piccola, quindi i 394 slot coprono il 96%.
- **K16 -> BORDERLINE.** 394 -> 0.886; servono 414 slot per il 90% (20 sotto).
  Recuperabile solo abbassando il target o alzando gli slot (VRAM).
- **K23 -> NON viable dentro 394.** 394 -> **0.79** (tetto), servono **581 slot**
  per il 90%. Working-set (989) = 2.5x il budget e uso quasi-uniforme: per il 90%
  dovresti pinnare il 63% del keep set. **K23 sul 3060 e residency-starved**: o si
  scende a K12 (K16 con piu VRAM), o serve piu VRAM.
- **Regola derivata (portabile):** pin-sottoinsieme viable se budget >= ~0.65 x
  working-set(K) (servono ~65-71% dei keep per il 90%). K12: 0.65*516=335 < 394 OK;
  K23: 0.65*989=643 >> 394 NO.

---

## COMPITO 2 — GAP-MATRIX copertura test (post-divergence)

Modello nuovo: mask (qualita) ORTOGONALE residenza (velocita). Caveat portati:
pressione-VRAM (sopra), **blocking-sync serializza i miss** (doppia barriera
cudaDeviceSynchronize + cudaStreamSynchronize per copia, PAPER.md sec.128),
delta-vs-full-WRAP (0021 pagina in **RAM**, non VRAM).

### Leva SPEX — STATO: ESISTENTE ma DORMIENTE (default-off)

SPEX = prefetch **predittivo del LOADING** (mai del gating: un miss e latenza,
non errore). E esattamente una **promozione-residenza predittiva**, complementare
al pin statico 0031. Stato dal sorgente/patch/docs (grep 299 hit):

- Patch wired: 0001-0007 (stats/upload/next-layer/markov/trace), 0015 (hidden
  async top-k handoff, **funziona**), 0016 (prefetch stats), 0017 (trace
  residency), 0028 (trace tokens). GPU score/topK esiste (dfceee3
  DS4_SPEX_HIDDEN_GPU_SCORE), pesi SPX1 su device (e85e256).
- **Perche e OFF:** il launcher tiene DS4_SPEX_HIDDEN_PREFETCH=0
  (SPEX_INTEGRATION_PLAN.md). Recall utile (top6 0.52 / top23 0.73-0.78) ma sul
  caso pratico 3060 (working-set piccolo/cache-ato) il prefetch **RALLENTA ~1.55x**;
  converte recall->velocita **solo se deeply-SSD-bound** (CLAIMS_CURRENT.md
  PREFETCH SPEX-dense, **OPEN**). Prossimo passo noto: consumare il topK GPU
  **senza readback host** (gli ID restano su device).
- **E un GAP DA RIATTIVARE, condizionato:** SPEX e la leva-prefetch-predittiva gia
  costruita che, ricablata sulla residenza keep-aware (oggi keep-cieca), coprirebbe
  il miss residuo di K23 (0.79 in su) ma solo nel regime deeply-SSD-bound e solo
  dopo il consumo topK on-device. Non e coperto: e costruito, spento, e non ancora
  agganciato alla residenza.

### GAP-MATRIX

| # | Area | Stato | Domanda aperta | Offline/GPU | Priorita |
|---|---|---|---|---|---|
| G1 | **Pin-keep hot-subset (0031)** | GAP, patch in authoring | K12 viable (questo report). Cablare pin-VRAM keep-aware nella streaming-cache (oggi keep-cieca, e5a1455). Verificare hit reale sotto mask (non proxy). | GPU (compilare ds4.c su host CUDA) | **ALTA** |
| G2 | **Trace mascherati (proxy->ground-truth)** | GAP | I run K12/K23 non emettono route.csv: la distribuzione d uso e misurata su trace FULL filtrati. Emettere DS4_SPEX_TRACE_ROUTING **sotto mask** per confermare la concentrazione reale. | GPU (leggero: 1 decode+trace) | **ALTA** |
| G3 | **Async-prefetch pipeline (leva dal blocking-sync)** | GAP, modellato non implementato | Prefetch layer L+1 richiede predire i suoi expert PRIMA della barriera upstream (PAPER.md sec.128, DYNAMIC_EXPERT_COMPRESSION_PLAN sec.525). Quanto si nasconde dietro il compute a batch=1? (ggml PR#21067: guadagno **incerto a batch=1**). | GPU | **ALTA** |
| G4 | **Delta-residency prefetch (VRAM)** | GAP | 0021 pagina il delta-mask in **RAM**, non VRAM (e5a1455 sec.B). Non esiste rotazione della sola RESIDENZA-VRAM. Serve? Dato che static-pin ~= LRU (churn basso valore), il delta-VRAM potrebbe non ripagare. Misurare il churn-cost reale. | GPU | MEDIA |
| G5 | **Curva costo-per-miss** | PARZIALE, E-LAT modella i tier | miss_bytes/token = recalls x footprint x (1-hit) esiste in boot-probe; E-LAT da 1.5-2.7 ms prefetched vs **50-59 ms sincrono** per expert. Manca la curva **t/s = f(hit-rate)** end-to-end sotto blocking-sync (il miss sincrono serializza). Chiude quanto vale portare hit 0.79->0.90 su K23. | Offline (modello) + 1 verifica GPU | **ALTA** |
| G6 | **SPEX predittivo -> residenza keep-aware** | GAP DA RIATTIVARE (condizionato) | Ricablare il topK SPEX (on-device) come promotore della residenza dei keep predetti-caldi; gate su boot-probe (b) deeply-SSD-bound. Complementa 0031 sul miss residuo K23. | GPU | MEDIA (ALTA se K23 resta target) |
| G7 | **Test config-completa** | GAP, mai eseguito | mask-statica (qualita) **+** pin-keep 0031 (residenza) **+** recupero/rewind (0022) **insieme**, n=3, grading L0-L3 al render + t/s. E il test che chiude S4: le tre leve non sono mai state misurate in combinazione. | GPU (pod o 3060) | **ALTA** |

### Sintesi priorita

- **Offline (gratis, subito):** G5 (curva costo-per-miss / t/s=f(hit), modello con
  una verifica GPU finale). Questo report chiude la parte offline di G1.
- **GPU necessaria:** G1, G2, G3, G7 (ALTA) sono il cammino critico di S4. G4, G6
  (MEDIA) dipendono dagli esiti di G1/G3.

---

## Appendice — riproducibilita

Script pin_analysis.py (scratchpad): aggregazione trace->keep, cov90, Gini, budget
greedy, LRU sim, stabilita, cross-dominio. Input elencati in COMPITO 1.

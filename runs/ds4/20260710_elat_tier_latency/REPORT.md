# E-LAT — Per-tier expert-recall latency model + differentiated-quantization scenarios

Date: 2026-07-10. Probe: OFFLINE ONLY (mining di artefatti su disco; no GPU/pod/WSL).
Domanda dell'utente: **"quanta latenza introduciamo, specie quando quei pochi
Expert rimasti su SSD vengono richiamati?"**

Riproduci: `python scripts/latency_tier_model.py --csv-out runs/ds4/20260710_elat_tier_latency/validation.csv`

Input architetturali: expert = **6.75 MiB** (2-bit IQ2_XXS gate/up + Q2_K down; misurato
4,466,147,328 B / 631 expert = 6.7503 MiB, `runs/ds4/20260710_pod_smoke_0020_0021/README.md`
smoke c, e J35 `gguf_inspect`), **43 layer MoE x top-6 = 258 richiami expert/token**
(`docs/PACE_DESIGN.md` §2) → **1.70 GiB/token** domandati se ogni richiamo manca la VRAM.

---

## 1. Latenza per-tier del richiamo (per singolo expert), con provenienza

L'hardware è dichiarato per ogni numero. **I rate NON trasferiscono tra host** (i pod
1TB-RAM tengono l'intero modello 86.7 GB in page cache; il 3060/WSL no).

| Tier | HW | ms/expert | Provenienza |
|---|---|---:|---|
| a. Hit cache residente VRAM | 3060/3090 | ~0 (dentro t_compute) | nessuna copia; **ma il runtime locale misura resident hit ≈ 0** (J31: 1 resident / 6923 eventi — il path selected-direct bypassa la cache) |
| b. RAM page-cache → VRAM (H2D, streamed selected) | 3060 WSL | **0.95–2.31** | `copy_ms_batch`/6: 5.712/6 (cache128 html320) … 13.853/6 (cache258), `runs/ds4/20260709_local_cache_sweep_k23_combined.csv`. Era-REAP con contesa trace su SSD: 59/6 ≈ 9.8 (`docs/PACE_DESIGN.md` §1, hit_rate 0.83) |
| b'. Page-in bulk RAM (WRAP/fattorino, caldo) | 3060 WSL | 0.22–0.83 | 6.07 GiB/198 ms = 30.7 GiB/s (`local_cache_sweep_k23_html320_rev` cache258) … 6.07 GiB/761 ms = 8.0 GiB/s (`20260710_w50_rotate32_k23_cache256_html4000/*/server.stderr.log`, 16 thread mlock); storico WRAP 6.07 GiB/445 ms = 13.6 GiB/s (PACE_DESIGN riga PREFILL_WAIT_WRAP, J3) |
| c. SSD freddo, page-in bulk threaded | 3060 WSL NVMe | **1.5–2.7** | prefetch più freddi misurati: 12.14 GiB/3.844 s (`20260709_rotate_smoke_gatefix`), 14.26/4.612 (`breath_k0` html800), 75.78/30.291 (`20260710_stepdown_relearn_only`), 6.07/2.007 (`prebreath_adapt`) → 2.5–4.4 GiB/s |
| c'. SSD freddo, **sincrono nel decode path** (miss batch) | 3060 WSL | **50–59** | `copy_ms_per_batch ≈ 59 ms` era-REAP SSD-bound (PACE_DESIGN §1); rotate su cache fredda: copy_ms_batch 49.6–54.9 (`20260709_rotate_smoke*` in `20260709_cache_pattern_table.csv`) |
| c''. Coda fault-storm SSD (decode a cache fredda) | 3060 WSL | ~230 | ">60 s/token" (PACE_DESIGN §7) / 258 expert |
| d. CQ1 sync decompress+repack+copy | 3060 CPU/WSL | **64–74** | J38: dopo ammissione CQ1, 14 tok a 0.06 t/s; server log (`runs/ds4/20260709_cq1_parallel/local_3060_cq1_native50/server_log_tail.txt`): 277.114−32.910 = 244.2 s / 14 tok / 258 = **67.6 ms**; 3612 copie (=258/tok), repacked 24 381 MiB. Confronto: sidecar lossless (senza decompress) ~39 ms/copy al prefill (J34: 158.9 s, 4056 copie, 12.4 GiB) |
| b-pod. RAM bulk (pod 1TB) | 3090 pod | 0.03–0.23 | delta-prefetch 0.693 GiB/30 ms = 23.1 GiB/s … 4.167 GiB/983 ms (primo touch) — `20260710_pod_smoke_0020_0021/delta.diag`; smoke 0021 conferma 6.75 MiB/expert dai bytes |

Nota tier-b vs tier-c: **a regime caldo la differenza batched è piccola** (0.95–2.7 ms);
la voragine è tra *batched/prefetched* (~1–3 ms) e *sincrono nel path di decode*
(50–230 ms). Il costo dominante del token locale non è l'SSD ma la **copia H2D dei 258
expert** (~246 ms dei ~320 totali per token).

## 2. Modello e calibrazione

```
decode_s = W*t_k0 + (N-W)*t_ss + prefetch_s + n_rotate*c_rot
t_ss     = t_compute + 258 * miss_vram * t_b        (miss_vram = 1 locale, misurato J31)
```

| HW | t_compute | t_b/expert | t_ss (steady) | t_k0 | Calibrato su |
|---|---:|---:|---:|---:|---|
| 3060 locale (WSL) | 74.9 ms | 0.952 ms | 320.5 ms → 3.12 t/s | 366 ms | `html_local_k23_cache128_r01` (sweep html320: last 3.12, copy 5.712, first50 2.73) |
| pod 3090 (hit≈1) | 40.3 ms | ~0.4 ms | 40.3 ms → 24.79 t/s | 459 ms | `20260710_pod_cache1024_html800` r01 (last chunk 24.79) |
| pod 4070Ti-class | 40.3 ms (assunto =3090) | 0.974 ms | 291.5 ms → 3.43 t/s | 521 ms | `20260709_pod_e7w4_static64` |
| c_rot | — | — | 0.65 s/rotate locale, 0.80 pod | — | coppie requested4-cache128 e id63/e7w4 |

Cross-check indipendente: il t_compute 3060 dedotto per sottrazione (74.9 ms) è coerente
col t_compute del 3090 misurato a hit≈1 (40.3 ms) scalato ~1.9x — plausibile 3090 vs 3060.

## 3. Validazione (90 run NON usati in calibrazione)

Fonte: tutti i `summary.csv` delle famiglie `runs/ds4/20260709_*` e `20260710_*`
(esclusi m1a/m1b untracked). Target = avg t/s dell'intero decode (N/(wall−prompt_s)).

- **Errore mediano |err| = 15.4%** (IQR 5.9–41.2%, n=90) sul modello avg-t/s completo.
- **Errore mediano steady-state = 7.8%** (1/t_ss vs last-chunk t/s, n=90).
- **Sottoinsieme "regime modellato"** (cache ≤ 258 locale, senza churn stepdown/prebreath):
  **10.5%** (n=64).
- Migliori: b4 direct_k40/k64 +0.3%, breath_k96 html800 +0.5%, cache256 html800 +1.1%,
  pod3090 secondo run −8.2%, pod4070 static128 +6.8%, w50 html4000 (4000 tok, 123 rotate)
  +5.0%.
- Bias sistematico: il modello **sovrastima** i run freddi/churn-heavy (code +74…+184%:
  cache64 sotto il floor 258 slot → thrash intra-token; stepdown/prebreath → churn di
  maschera non modellato; K0-cold con first50 0.2–0.9 t/s). Bias direzionale e
  documentato: il modello è un tetto per configurazioni non churn-bound.

Tabella completa per-run: `runs/ds4/20260710_elat_tier_latency/validation.csv`.

## 4. Scenari quantizzazione differenziata (3060 locale)

Hit-rate: LRU sim J17/J31 (cap258 0.34, cap512 0.59–0.61, cap1024 0.74–0.76, cap2048
0.81–0.82); prompt-preload J32 (cap1024 hot-hit 0.849). Capacità CQ1 misurata J35:
cq1g32 48.38 GiB, cq1g256 34.27 GiB vs 72.56 nativi → **x1.5–x2.1, NON x3** (x3 richiede
sub-CQ1/1-bit, ledger: "TODO rifare esperimenti su CQ1/1-bit/sub-CQ1").

| Scenario | t_token | t/s previsto | Note |
|---|---:|---:|---|
| a. Status quo cache256 2-bit | 321 ms | **3.12** | = misurato (2.9–3.4). I cliff SSD restano: breath 25.34 GiB/2.7 s (J28), first-50 K0 freddi 0.2–1 t/s, storm 60 s/tok |
| b. + CQ1 **asincrono** sui cold in RAM | 321 ms | **3.12 steady** | lo steady non cambia (path H2D-copy-bound); elimina il tier-c dagli shift di working-set: breath/rotate/prompt-switch serviti da RAM. Primo richiamo di un cold: 64–74 ms UNA volta se sincrono, ~0 se promosso ≥1 token prima |
| c. Tiering completo, hot VRAM cap258 hit 0.34 | 307 ms | 3.26 | richiede fix del resident-hit (oggi 0 runtime vs 0.34 simulato) |
| c. Tiering completo, hot cap407 hit 0.50 | 267 ms | **3.74** | 407 = max slot VRAM (nota E1); interp LRU 258→512 |
| c. Tiering completo, cap407 + preload-promote hit 0.60 | 243 ms | **4.12** | forma J32 applicata al cap VRAM reale |
| (tetto teorico hit 0.85) | 112 ms | 8.95 | cap2048-equiv: NON raggiungibile nativo in 12 GiB; servirebbe predictor (SPEX) o formato VRAM compresso |

**t/s previsto per il tiering completo sul 3060: 3.7–4.1 t/s** (da 3.1), cioè **+20–32%**
steady, più l'eliminazione dei cliff (che sui run reali pesano sull'avg: p.es. avg 2.2–2.6
vs steady 3.0–3.45 nei run con K0 freddo/churn — recuperare l'avg verso lo steady vale un
ulteriore +15–30% sul tempo totale percepito).

## 5. Worst-case richiamo del singolo expert da SSD (la domanda dell'utente)

A steady 3.1–4.1 t/s (t_token 240–320 ms), il richiamo di UN expert 6.75 MiB rimasto su SSD:

| Path | ms | token "persi" |
|---|---:|---:|
| batched/prefetched (fattorino freddo, 2.5–4.4 GiB/s) | **1.5–2.7** | 0.005–0.011 |
| **sincrono dentro il token** (miss batch, misurato copy_ms 50–59) | **50–59** | **0.16–0.25** |
| fallback CQ1 sincrono (se il cold è solo in RAM compressa) | 64–74 | 0.20–0.31 |
| coda fault-storm (regime freddo, 60 s/tok ÷ 258) | ~230 | 0.7–1.0 |

Cioè: **se il tiering tiene su SSD solo i frozen e li richiama prefetched/batched, un
richiamo costa ~2–3 ms (invisibile, <2% di un token). Se il richiamo è sincrono dentro il
token, costa 50–59 ms ≈ un sesto–un quarto di token.** Il vero pericolo non è il singolo
expert ma lo shift di working-set non prefetchato (decine di expert → secondi).

## 6. Verdetto

**La compressione differenziata asincrona NON è la leva S4 più grossa: è la seconda, e
l'abilitatore della prima.** Ordine misurato delle leve sul 3060:

1. **Fit-in-cache / capacità effettiva del tier hot** (fix resident-hit: runtime 0 vs 0.34
   simulato — J31; poi preload-promote J32): da sola vale 3.1 → 3.7–4.1 t/s.
2. **CQ1 asincrono sui cold in RAM**: steady invariato ma (i) rende il punto 1 possibile
   dentro 12 GiB VRAM + RAM limitata (72.56 GiB nativi → 34–48 GiB), (ii) elimina i cliff
   tier-c (breath 2.7 s, storm 60 s/tok, prompt-switch), (iii) taglia il primo richiamo da
   50–230 ms (SSD sync) a 0–74 ms (RAM). Il path sincrono resta vietato (0.06 t/s, J38):
   SOLO promozione in background nelle finestre breath/exchange (J39).
3. Il tetto 8.95 t/s richiede hit 0.85 → non raggiungibile con capacità nativa: è il caso
   d'uso per il predictor SPEX-GPU o formati VRAM compressi (ricerca, non S4).

Honesty: la "capacità x3" ipotizzata per CQ1 non è supportata da J35 (x1.5–x2.1 misurato);
x3 richiede sub-CQ1 ancora da provare in qualità.

## 7. Misure MANCANTI (micro-bench da strumentare per S4)

1. **Distribuzione latenza per-expert SSD freddo** (QD1, singolo mmap-read 6.75 MiB, WSL,
   p50/p95/p99, con/senza mlock): oggi il 50–230 ms è *inferito* da copy_ms_batch e dallo
   storm; mai misurato l'expert singolo isolato.
2. **Costo CQ1 isolato**: decompress+repack senza copy né churn (J38 conflaziona), scaling
   multi-thread, e variante cq1g256.
3. **CQ1 async sotto carico**: la promozione in background ruba banda memoria al decode?
   (architettura blocking-sync → CPU libera, ma contesa RAM non misurata).
4. **Perché resident hit = 0 runtime** vs 0.34 LRU simulato: path selected-direct vs
   resident (J10/J31) — il singolo fix col ROI più alto del modello.
5. **t_b decomposto**: page-cache→staging vs staging pinned→VRAM (H2D pinned vs unpinned).
6. **Qualità cq1g256** su output graded (capacità x2.1).
7. **Prefill/TTFT** (gap J12, fuori dal modello decode: prompt_s 10–173 s osservati).

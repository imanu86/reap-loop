# Curva velocità-vs-K PULITA (cache32, reserve-fixed) + tracce mascherate VERE — 3060 locale, 2026-07-11

Completa la probe che aec293 aveva lasciato a metà (solo K12 + alcune cache1024,
vedi `README.md`). Due consegne in un colpo per run: **t/s pulito** (curva) e la
**route.csv post-mask VERA** (routing effettivo sotto mask frozen) per la
phase-segmentation, che finora girava su proxy full-model.

## Config PULITA (le lezioni di aec293 applicate, verificate)

- Binario `/root/ds4/ds4`, modello `/root/models/ds4-2bit.gguf` (WSL Ubuntu-24.04, sm_86).
- `--ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts {32|516}`,
  `-c 4096 --nothink --temp 0.0 -n 300` (greedy), prompt cyberpunk-wide (principale)
  + coffee-compact (narrow, contrasto).
- `DS4_CUDA_NO_Q8_F16_CACHE=1` (path 2-bit puro), `DS4_CUDA_NO_DIRECT_IO=1`,
  `DS4_CUDA_KEEP_MODEL_PAGES=1`.
- **`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1`** (parse-safe): verificato in TUTTI
  i run — **nessuna riga `cache capped/disabled`** nei diag (bug 256->154 evitato),
  nessun abort reserve=16. `global_budget=32` a cache32, `=516` a cache516 (onorati).
- `DS4_REAP_MASK_FILE=masks/sessK{12,16,23,38}.txt` (mask frozen, keep=K/layer su 40 layer).
- `DS4_LOCK_FILE=/tmp/ds4_clean_lowK_curve.lock` (coesistenza; nessun processo ds4
  co-resident durante le misure; server UI porta 8000 mai toccato). CLI puro, nessuna porta.
- Script: `curve_trace.sh` (curva+tracce), `warm_sweep.sh` (curva warm-controllata).

## TABELLA velocità-vs-K PULITA

### A. Warm-controllata — la curva pulita apples-to-apples (`warm_sweep.sh`, cyberpunk, cache32)
Un run di warm-up scartato, poi K12/16/23/38 back-to-back (pagine modello calde per tutti):

| K | keep/layer | gen t/s (warm) | hit-rate |
|---|---|---|---|
| 12 | 12 | **3.63** | 0.00 |
| 16 | 16 | **3.77** | 0.00 |
| 23 | 23 | **3.69** | 0.00 |
| 38 | 38 | **3.57** | 0.00 |

=> **PIATTA. Gen t/s indipendente da K** (3.57-3.77, spread ~5% = dentro il rumore
run-to-run). A cache32 il router seleziona sempre 6 esperti/layer/token e la cache
manca su tutti (hit=0) qualunque sia K: il numero di keep NON cambia il costo/token.

### B. Prima passata (`curve_trace.sh`, cyberpunk, cache32) — mostra il confound warm/cold
| K | gen t/s | prefill t/s | note |
|---|---|---|---|
| 12 | 1.77 | 1.26 | **run #1 FREDDO** (pagine modello non in RAM) |
| 16 | 3.25 | 3.76 | caldo |
| 23 | 3.08 | 0.64 | caldo |
| 38 | 3.07 | 8.89 | caldo |

Il 1.77 di K12 era un artefatto di cold-start (modello 86GB, pagine sfrattabili),
NON un effetto-K: warm-controllato lo stesso K12 fa 3.63 (vedi A).

### C. Dimensione cache (K23, cyberpunk, warm)
| cache-experts | gen t/s | hit-rate |
|---|---|---|
| 32 | 3.69 | 0.00 |
| 516 | 2.93 | 0.83 |

=> cache516 (83% hit) e' **~20% PIU' LENTA** di cache32 (0% hit). Conferma il verdetto
di aec293: cache grande = piu' lenta (overhead bookkeeping/LRU > risparmio copie).
cache32 resta il config piu' veloce.

### D. Coffee-shop compact (narrow, contrasto — entrambi cold in questa sessione)
| K | gen t/s | prefill | note |
|---|---|---|---|
| 12 | 2.27 | 1.22 | cold — **riproduce il 2.14 coffee di aec293** |
| 23 | 1.61 | 0.77 | cold |

## VERDETTO — "K basso resta piu' veloce anche pulito?"

**NO.** Pulito e warm-controllato, la gen t/s a cache32 e' PIATTA su K12->K38 (~3.6-3.8).
Il driver della velocita' e' (1) il warmth delle pagine-modello (cold 1.6-2.5 vs warm
3.6-3.8) e (2) la dimensione cache (piccola=veloce), **non K**. Il "K basso frenato"
storico e il "K basso piu' veloce" atteso sono ENTRAMBI artefatti di co-residenza/warmth,
coerente con aec293 (il K12=1.20 storico era co-residenza + reserve-cap, non K). Nota:
cyberpunk warm (~3.66) > coffee cold (~2.2) e' confound prompt+warmth, non un vero
gap-per-prompt misurato a parita' di warmth.

## TRACCE MASCHERATE VERE (non-proxy) — PRONTE PER RE-RUN PHASE-SEGMENTATION

`DS4_SPEX_TRACE_ROUTING=<path>` + `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1` con la mask frozen
attiva logga per-token per-layer gli esperti **effettivamente SELEZIONATI post-mask**
(+ pesi). Verificato meccanicamente: **0.000% dei pick cade su un esperto potato** in
tutte e 6 le tracce (mask enforcement reale, non router grezzo pre-bias). Formato:
`pos,layer,n,e0..e5,w0..w5`. ~11960 righe/traccia = ~299 token di generazione x 40 layer MoE.

Dir: `runs/ds4/20260711_masked_route_traces/`

| file | K | prompt | righe | distinct-esperti-usati/layer |
|---|---|---|---|---|
| `route_masked_K12_cyberpunk.csv` | 12 | cyberpunk (wide) | 11960 | 12.0 (= keep) |
| `route_masked_K16_cyberpunk.csv` | 16 | cyberpunk (wide) | 11960 | 16.0 (= keep) |
| `route_masked_K23_cyberpunk.csv` | 23 | cyberpunk (wide) | 11960 | 22.9 (~keep) |
| `route_masked_K38_cyberpunk.csv` | 38 | cyberpunk (wide) | 11960 | 37.8 (~keep) |
| `route_masked_K12_coffee.csv` | 12 | coffee (narrow) | 11960 | 12.0 (= keep) |
| `route_masked_K23_coffee.csv` | 23 | coffee (narrow) | 11960 | 23.0 (= keep) |

(La traccia K23-cyberpunk e' identica tra cache32 e cache516: il routing e' deterministico
su mask+prompt+greedy, la cache tocca solo la velocita' — nessuna perdita dal riuso path.)

**PUNTATORE ESPLICITO per phase-segmentation**: queste sono le route.csv MASCHERATE VERE
che colmano il gap-G2 di `pin_analysis.py`/`phase_segmented_usage.py` (caveat "Proxy: no
masked route.csv exists — full-model filtered to keep"). Ri-eseguire la phase-segmentation
puntando a `runs/ds4/20260711_masked_route_traces/route_masked_K*.csv` con `keep=None`
(gia' post-mask) **RIMUOVE il caveat-proxy**. Reperto immediato: la copertura del keep e'
~100% (distinct-usati == keep per ogni K) => il working-set whole-trace == l'intero keep
set; la concentrazione per-fase (k90, Jaccard hot-core) va rimisurata su QUESTI dati reali,
non piu' inferita dal proxy full-model. Conferma anche il sospetto del REPORT: sotto mask i
6-di-6 si ridistribuiscono su tutto il keep (nessun esperto keep resta inutilizzato).

## 0031 PIN A/B — HANDOFF (non eseguibile qui)

Richiesto dal coordinator come test decisivo (PIN=1 tunato vs 2.14 a cache256). **BLOCCATO
localmente: `nvcc` MANCA in WSL Ubuntu-24.04** (`which nvcc` -> vuoto). Impossibile
`make cuda` sm_86. Endpoint atteso `ds4_cuda.cu md5=430716f4` NON presente: il sorgente
locale e' `md5=7d57f58d414dffc49a19ced3e9a79dd4` (0031 non applicata). Patch disponibile:
`patches/ds4/0031-pace-pin-keep-residency-rotation.patch`. => 0031 resta handoff (serve un
ambiente con toolchain CUDA, es. il pod che ha gia' passato lo smoke podD).

## Artefatti
- `curve/` — K{12,16,23,38}_cache32_cyber, K23_cache516_cyber, K{12,23}_cache32_coffee
  (gen.txt + diag.txt), `curve_trace_progress.log`.
- `curve/warm/` — warmup + w_K{12,16,23,38} (curva warm-controllata), `warm_progress.log`.
- `../20260711_masked_route_traces/route_masked_K*.csv` (6 tracce) + `prompt_cyberpunk_wide.txt`.
- Script riproducibili: `curve_trace.sh`, `warm_sweep.sh`, `poll*.sh`, `launch.sh`.

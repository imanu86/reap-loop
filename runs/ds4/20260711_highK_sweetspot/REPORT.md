# High-K sweet-spot sweep — 3060 locale, 2026-07-11

**Domanda.** La curva velocità-vs-K è PIATTA a K basso (K12=3.63 … K38=3.57 t/s warm,
`20260711_local_clean_lowK/CURVE_COMPLETE.md`): K non tocca la velocità (il router prende
sempre 6 esperti/layer/token, cache32 hit~0). Ipotesi: il beneficio-velocità della mask è
il FIT-in-RAM (satura), quindi conviene la mask più LARGA che entra ancora calda = max
qualità a parità di velocità. Qui si spinge K in ALTO (K48/64/91) + il full locale (K0),
cercando lo **sweet-spot** = la mask più larga ancora >3 t/s E che RENDE/chiude `</html>`.

## Config (regime pulito veloce, identico al CURVE_COMPLETE)
- Bin `/root/ds4/ds4`, modello `/root/models/ds4-2bit.gguf` (86.7 GB su disco), WSL sm_86.
  RAM host 60 GiB → **l'81 GiB del modello NON entra tutto** (page-cache ~57-60 GB).
- `--cuda --ssd-streaming --ssd-streaming-cold --ssd-streaming-cache-experts 32 --nothink
  --temp 0.0` (greedy). Prompt cyberpunk-wide (HTML-primed) principale, coffee narrow contrasto.
- Env: `DS4_CUDA_NO_Q8_F16_CACHE=1`, `DS4_CUDA_NO_DIRECT_IO=1`, `DS4_CUDA_KEEP_MODEL_PAGES=1`,
  `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1` (no cap 256→154, no abort reserve=16), `DS4_SPEX_STATS=1`.
- `DS4_LOCK_FILE=/tmp/ds4_highK_sweetspot.lock` (coesistenza; un solo ds4 alla volta; server
  UI porta 8000 mai toccato — CLI puro, nessuna porta). GPU 3060 libera (12 GB), ~11.9 GB in uso/run.
- Mask `masks/sessK{48,64,91}.txt` weighted (massa-gate) dal trace **coffee W50** phase2
  (`build_session_mask_canonical.py`, stesso recipe di K12-38; K38 rebuild byte-identico modulo CRLF).
  K0 = nessuna mask (full, 256 esperti eleggibili).

## Nota mask — il keep si TAPPA a K alto (sessione W50 stretta)
Il trace di calibrazione W50 (coffee, ~49 token) ha visto solo **59-100 esperti distinti/layer**
(mediana 74). Quindi il keep effettivo è tappato dal vocabolario-sessione:

| mask | K richiesto | layer tappati <K | keep medio EFFETTIVO |
|---|---|---|---|
| K48 | 48 | 0/40 | **48.0** |
| K64 | 64 | 5/40 | **63.7** |
| K91 | 91 | 36/40 | **74.6** |

=> "K91" è di fatto "**tieni tutto ciò che la sessione-coffee ha mai toccato**" su quasi tutti
i layer. Non esiste un keep >~75/layer da questa calibrazione: il soffitto non è K, è la
copertura-esperti della sessione di calibrazione.

## TABELLA PRINCIPALE — K × [t/s warm, RAM-fit, qualità, chiude?]

| K | keep eff. | gen t/s (warm, -n300) | RAM-fit (peak swap) | qualità cyberpunk -n5500 | chiude </html>? |
|---|---|---|---|---|---|
| 48 | 48.0 | **3.60** | sì (1 MB) | **L0** collasso (loop CSS `background`) | no |
| 64 | 63.7 | **3.06** | sì (1 MB) | **L0** collasso (loop CSS `background`) | no |
| 91 | 74.6 | **3.16** | sì (3 MB) | **L0** collasso (loop CSS `background`) | no |
| 0 (full) | 256 | **~0.9** cyber / **1.24** coffee (no-warm, no-fit) | no-fit (swap 26-95 MB) | coerente, **no-loop** (render pieno=coffee L3) | no (troncato) |
| (cold ref: warmup K48) | 48.0 | 1.13 | — | — | — |

### A. Velocità warm — la curva DECISIVA (probe -n300, cache32, cyberpunk-wide, warm-controllato)
Un warmup scartato (cold 1.13), poi K48/64/91 back-to-back (pagine modello calde):

| K | gen t/s | prefill t/s | peak swap |
|---|---|---|---|
| 48 | 3.60 | 0.58 | 1 MB |
| 64 | 3.06 | 5.63 | 1 MB |
| 91 | 3.16 | 8.67 | 3 MB |

=> **PIATTA / NESSUN CROLLO salendo K.** 3.06-3.60 t/s (spread ~15% = rumore run-to-run; K91 > K64).
Swap ≤3 MB su tutti → **il keep-set entra caldo in RAM anche a K91 (eff 74.6)**. Confermata
l'ipotesi velocità: è indipendente da K, il driver è warmth+fit (cold 1.13 vs warm ~3.3). Nessun
"K di rottura" osservato fino a 74.6 esperti/layer.

### B. Qualità cyberpunk (quality run -n5500, cyberpunk-wide) — TUTTI collassano
| K | grade | gen_chars | firma |
|---|---|---|---|
| 48 | **L0** | 5448 | loop `background: #0000  #0000;` (mai raggiunge `<body>`) |
| 64 | **L0** | 2928 | loop `background: radial-gradient…/#000;` |
| 91 | **L0** | 2684 | loop `background: #000;` |

Tutti e tre muoiono **nello stesso punto** (la sezione CSS `body{background}`), in un loop
degenere greedy, senza mai arrivare a nav/hero/form, senza chiudere `</html>`. **Spingere K
in alto NON cura il collasso** sul task cyberpunk (fuori-dominio rispetto alla calibrazione coffee).
(I run -n5500 sono stati terminati appena confermato il loop — il grado è dal parziale collassato;
la velocità pulita è quella della sez. A.)

### C. Controllo IN-DOMINIO (coffee narrow -n1500) — la STESSA mask RENDE
| run | grade | chiude? | gen t/s | note |
|---|---|---|---|---|
| **coffee K64** (in-dominio) | **L1** | **sì (`</html>`)** | 2.84 | pagina COMPLETA: nav+hero+form+button+script cablati; 1 solo errore JS di sintassi |
| **coffee K0** (full) | **L3** | **sì (`</html>`)** | 1.24 | full-model locale (no-fit): pagina perfetta, JS pulito |

**Contrasto decisivo:** la **stessa mask K64** → cyberpunk (fuori-dominio) **L0 collasso**,
coffee (in-dominio) **L1 pagina completa e chiusa**. Il soffitto di render è il **match di
dominio della calibrazione**, non la magnitudine di K.

### D. K0 full LOCALE (81 GB > 60 GB RAM → no-fit)
- Velocità: **coffee full locale = 1.24 t/s misurato** (chiude L3), cyberpunk full locale
  **~0.9 t/s** (byte-rate; troncato). I/O-bound: GPU util ~15-30% vs ~50% mascherato (streaming
  continuo dell'intero set esperti da SSD, 81>60 GB). **~2.5-4× più lento** del regime mascherato
  warm (~3.3 t/s). Contro il pod RAM-hot (3090Ti, 220 GB RAM) che faceva 4.09/4.21 t/s: quei
  numeri NON trasferiscono al 3060 (regime memoria opposto).
- Qualità: **il full RENDE e NON collassa.** coffee_K0 = **L3 pagina completa chiusa** (JS pulito,
  `function(e)` corretto vs il `(e) {` rotto della mask); cyberpunk_wide = CSS ricca e VARIATA
  (gradient/flexbox), **zero ripetizione** attraverso la stessa zona `body{background}` dove ogni
  mask loopava (troncato a ~tok280 per budget-tempo; render pieno = controllo pod T1 L1 repeat=0).
  => il loop CSS di §B è **attribuibile alla mask** (dominio), NON al prompt wide.

## Tracce mascherate emesse (feed phase-segmentation)
`DS4_SPEX_TRACE_ROUTING` + `DS4_SPEX_TRACE_ROUTING_WEIGHTS=1` sotto mask frozen, cyberpunk-wide, -n300:

| file | K | righe | pick su esperti potati | distinti usati/layer |
|---|---|---|---|---|
| `route_masked_K48_cyberpunk.csv` | 48 | 11961 | **0.000%** | 47.2 |
| `route_masked_K64_cyberpunk.csv` | 64 | 11961 | **0.000%** | 61.7 |
| `route_masked_K91_cyberpunk.csv` | 91 | 11961 | **0.000%** | 71.8 |

Enforcement reale (0 pick su potati). Copiate in `../20260711_masked_route_traces/` accanto a
K12-38 → set completo K12/16/23/38/48/64/91 per la phase-segmentation (caricare con `keep=None`).
Il distinto-usato/layer ≈ keep => sotto mask i 6-di-6 si ridistribuiscono su TUTTO il keep.

## VERDETTO — sweet-spot
1. **Velocità piatta confermata anche a K alto**: 3.06-3.60 t/s da K48 a K91 (eff 74.6), nessun
   crollo, swap ≤3 MB (keep-set entra caldo). Il "K di rottura" non si raggiunge dentro il
   vocabolario-coffee (max ~75/layer). => alzare K è **gratis in velocità**.
2. **MA sul task cyberpunk (fuori-dominio) NESSUNA mask-coffee rende**: K48/64/91 collassano tutte
   (L0, loop CSS, no `</html>`). **Non esiste sweet-spot** tra le mask session-coffee per cyberpunk.
3. **Lo sweet-spot è DI DOMINIO, non di K**: la stessa K64 rende in-dominio (coffee L1, chiude) e
   collassa fuori-dominio (cyberpunk L0). Per rendere il wide serve o il **full (K0)** — che locale
   è ~1.24 t/s coffee / ~0.9 cyberpunk (no-fit, usabile ma ~3x più lento del mascherato) — o una **mask calibrata sul dominio wide**
   (non testata qui: la W50 è coffee).
4. Pratica: "push K up for quality" vale **solo entro il dominio di calibrazione**. Cross-dominio,
   K alto non basta: il keep manca gli esperti che il task nuovo richiede (mai visti dalla sessione).

## Artefatti
- `masks/sessK{48,64,91}.txt(.json)` — mask weighted W50 (nuove).
- `sweep/` — probe_K{48,64,91} (t/s warm) + qual_K{48,64,91} (-n5500 collassati), `progress.log`, `mem.log`/run.
- `coffee/` — coffee_K64 (in-dominio) + coffee_K0 (full). `k0/` — k0_cyberpunk (full wide).
- `traces/trace_K{48,64,91}/route.csv` (+ copie canoniche in `../20260711_masked_route_traces/`).
- Script riproducibili: `run_sweep.sh`, `run_gen.sh`, `run_phase2.sh`, `verify_enf.py`, `*status/poll*.sh`.

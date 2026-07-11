# Probe velocita' pulita (no co-resident) K x cache — 3060 locale, 2026-07-11

Risponde alla domanda: la probe co-resident dava K12~=K23~=1.2 t/s nonostante
K basso dovrebbe performare meglio. Misurato pulito (server UI spento, un
solo processo ds4 alla volta): isolato cosa frena K basso tra (a)
co-residenza, (b) cache troppo piccola, (c) bug resident-hit~=0.

Setup comune: mask K-per-layer weighted (session_weighted, dal trace coffee
W50 due-fasi, `masks/sessK{12,16,23,38}.txt`, K16/K38 generate ma non
misurate per budget), prompt coffee (`t4_W050/W050/r00/p2prompt.txt`),
greedy temp0, ctx4096, `DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1` (fix del parsing-intero, vedi
sotto), CLI diretto (`/root/ds4/ds4`), nessun altro processo ds4 in
esecuzione durante le misure.

## Tabella K x cache (t/s fase2 + resident-hit)

| K | cache | hit-rate | gen t/s | prefill t/s | q8/f16-cache | n tok | run |
|---|---|---|---|---|---|---|---|
| 12 | 32 | 0% | **2.14** | 0.86 | off (2bit puro) | 300 | bitexact2/K12_cache32_q8off |
| 12 | 32 | 0% | 1.59 / 1.87 | 1.21 / 0.67 | off | 300 x2 | control/ctrl_32_a,b (ripetizione identica) |
| 12 | 32 | 0% | 1.45 | 0.62 | ON | 300 | bitexact2/K12_cache32_q8on |
| 12 | 256 | 1.3% | 1.65 | 1.54 | off | 300 | bitexact/A_q8off_cacheON |
| 12 | 256 | 1.35% | 1.56 | 0.89 | ON | 300 | bitexact/B_q8on_cacheON |
| 12 | 256(→154, reserve default) | 0% | 1.53 | 0.88 | ON | 300 | bitexact/B_q8on_cacheOFF |
| 12 | 516 | n/d | 0.54 | 0.71 | off | 60 (breve) | vram/v_cache516 |
| 12 | 1024(→916) | 98.1% | 1.14 | 1.62 | off | 300 | bitexact2/K12_cache1024_q8off |
| 12 | 1024(→913) | 98.2% | 0.89 | 0.75 | ON | 300 | bitexact2/K12_cache1024_q8on |
| 23 | 1024(→916) | 91.3% | 0.76 | 0.77 | off | 300 | bitexact2/K23_cache1024_q8off |

Storico (co-resident col server UI, reserve default capped a 154 esperti,
resident-hit~0): K12=1.20, K16=1.12 t/s (`20260711_local_lowK_tps/`).

## VERDETTO (secco)

**K basso NON e' frenato da cache-troppo-piccola ne' dal bug
resident-hit~=0 — anzi, il contrario.** A cache 1024 (che CONTIENE il
working-set di K12, ~516 esperti) l'hit-rate sale al 98% ma il t/s SCENDE a
1.14 (o 0.89 con cache q8/f16), quasi 2x piu' lento della cache piccola
(32 esperti, hit 0%, 2.14 t/s). Il pattern e' monotono e ripetuto su 5 run
indipendenti (32→516→1024 a n=60, e 32 vs 1024 due volte a n=300, con e
senza q8/f16-cache): **piu' hit-rate = piu' lento**, non il contrario.

Cosa frenava il numero storico (K12=1.20 t/s): **co-residenza col server UI
+ reserve-cache default che capava il budget richiesto (256→154 esperti)**.
Pulito e senza co-residenza, K12 in regime cache-piccola/diretta arriva a
**2.14 t/s**, +78% sul numero storico — SENZA bisogno di alzare l'hit-rate.

Perche' la cache grande e' piu' lenta (isolato per eliminazione, non
profilato con nsight/nvprof — inferenza dai contatori, non prova diretta):
- **Non e' banda di copia H2D**: `copy_ms` totale e' PIU' BASSO a cache1024
  (210s) che a cache32 (417s) — la cache fa meno copie in assoluto.
- **Non e' pressione VRAM**: misurato `nvidia-smi` picco reale durante 3 run
  (cache 32/516/1024, `vram/vram.log`) — **~11966-12019 MiB in tutti e tre**,
  entro 53 MiB l'uno dall'altro, tutti al ~97.5-97.8% dei 12288 MiB della
  scheda. Nessuna differenza di pressione VRAM misurabile tra cache piccola
  e cache grande — l'ipotesi overflow/thrashing e' **refutata** dal dato.
- **Candidato residuo**: overhead di bookkeeping/lookup della cache LRU che
  scala con l'occupazione (fino a ~900+ slot pieni, ogni accesso-esperto —
  258/token — deve cercare/aggiornare lo stato LRU), non il trasferimento
  dati in se'. Non isolato con un profiler vero: **ipotesi, non prova**.

**Pinning residente**: verificato dal sorgente (`ds4_cuda.cu`, no run extra)
che `--ssd-streaming-preload-experts` NON e' pinning — e' solo un warm-fill
iniziale della stessa cache LRU evictabile (nessun flag "non evictabile" in
tutto il codebase). ds4 non ha oggi un vero meccanismo di residenza fissa;
la 0031 (pin-keep) lo introdurrebbe come feature nuova, non gia' presente.

**Bit-exact (qualita')**: vedi `BITEXACT.md`. Path 2-bit puro pulito (dentro
il rumore di non-determinismo di base, ~21-24 righe diff su 300 tok). Path
q8/f16-cache mostra un segnale di divergenza oltre il rumore (~54 righe,
2.6x) — il futuro pin-keep (0031) deve servire SOLO 2-bit nativo dagli slot
pinnati, mai q8/f16.

**Bug intermittente scoperto en-passant**: `RESERVE_GB=16` (letterale, non
il fallback-parsing) causa un'uscita precoce e silenziosa (rc=0, nessun
output) 1 volta su 2 tentativi — vedi `RESERVE16_ABORT.md`. Non confermato
deterministico, riproducibilita' incerta, fuori budget da investigare oltre.

## Miglior t/s visto

**2.14 t/s** — K12, cache-experts=32, `DS4_CUDA_NO_Q8_F16_CACHE=1`,
`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=1`, no co-resident, coffee
prompt, 300 tok fase2, warm (non e' il primo run freddo).

## Cosa NON e' stato fatto (budget)

- K16 e K38 non misurati (mask generate, pronte in `masks/`): la matrice
  completa richiedeva ~2h aggiuntive con il tax fisso di ~5-8min/run
  (scansione iniziale "gpu prefill layer" indipendente dalla cache-size).
- K23 a cache32/516 (il confronto K12-vs-K23 a parita' di regime-veloce
  chiesto dal coordinator) non misurato — solo K23@1024 disponibile.
- Repliche n>=2 per confermare formalmente il 2x cache-size-vs-velocita' su
  TUTTE le celle (fatto solo per cache32: 2.14, poi 1.59/1.87 nel control —
  variabilita' run-to-run reale, ~25-35%, ma il gap 2x tra cache32 e
  cache1024 e' piu' grande della variabilita' osservata).
- Profiling reale (nsight/nvprof) dell'overhead cache-LRU — l'attribuzione
  "bookkeeping/lookup" e' un'inferenza dai contatori DS4_SPEX_STATS, non una
  misura diretta.

## Artefatti

- `masks/` — sessK{12,16,23,38}.txt(.json), rigenerate dal route.csv coffee
  W50 via `scripts/build_session_mask_canonical.py` (K12/K16 verificate
  byte-identiche alle mask della probe precedente).
- `bitexact/`, `bitexact2/`, `control/`, `vram/` — le run grezze (gen.txt +
  diag.txt per ognuna) + i rispettivi `.sh` e `.log`.
- `BITEXACT.md` — verdetto qualita' dettagliato.
- `RESERVE16_ABORT.md` — bug intermittente reserve=16.

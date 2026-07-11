# DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=16 — silent early-exit, 1/2 repro

Contesto: durante il bit-exact test (priorita-0), tentando di riprodurre
"cache-off" con `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=16` (il valore che
CLAIM-019 indica come "disabilita la cache su 12GB"), il **primo** dei due
tentativi e' terminato in modo anomalo. Il **secondo** tentativo, stessa env,
stesso comando, e' andato a buon fine — quindi la riproducibilita' e' **1/2**,
non deterministica. Non e' stato investigato oltre (fuori budget di questa
probe); qui solo la documentazione del sintomo.

## Run A — anomalo (`bitexact/A_q8off_cacheOFF/`)

Comando: `DS4_CUDA_NO_Q8_F16_CACHE=1 DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=16
DS4_REAP_MASK_FILE=sessK12.txt ds4 -m ds4-2bit.gguf --cuda --ssd-streaming
--ssd-streaming-cold --ssd-streaming-cache-experts 256 -c 4096 --nothink
--temp 0.0 -n 300 --prompt-file p2prompt.txt`

- `rc=0` (nessun segnale, nessun crash visibile) dopo ~8m26s wall.
- `gen.txt` **vuoto** (0 byte) — mai raggiunta la generazione.
- `diag.txt` **troncato a 59 righe**, l'ultima senza newline finale:
  `ds4: gpu prefill layer 34/43` (poi nulla). Questo e' DENTRO il primo loop
  di caricamento/graph-prefill del modello (layer 1..43), **prima** ancora
  del reload della mask REAP e del messaggio di cap/disable della cache — il
  processo non ha mai raggiunto il punto dove si decide "cache disabled:
  available X GiB <= reserve 16.00 GiB" ne' quello dove si applica la mask.
- Nessun processo residuo dopo l'uscita (`ps aux` pulito), nessuna riga
  OOM/kill/segfault in `dmesg`, GPU libera subito dopo (748 MiB, 1%).
- Non e' l'atteso degrado "cache disabled -> tutti direct-load": e' un'uscita
  **prematura e silenziosa**, a meta' di una fase che di norma richiede
  ~1-2 minuti prima di raggiungere il mask-reload.

## Run B — riuscito (`bitexact/B_q8on_cacheOFF/`)

Stessa identica invocazione (differiva solo per `DS4_CUDA_NO_Q8_F16_CACHE`
non settata), lanciata ~19 minuti dopo A, nella stessa sessione del binario:
completata normalmente in ~9m33s, con:

```
ds4: CUDA streaming expert cache capped from 256 to 154 experts
  (available 7.02 GiB, reserve 6.00 GiB, 6.75 MiB/expert)
SPEX stats: ... cache_hits=0 cache_misses=6 hit_rate=0.0000 direct_loads=84613
prefill: 0.88 t/s, generation: 1.53 t/s
```

**Scoperta secondaria**: il valore richiesto `reserve=16` non e' stato
onorato affatto — il log mostra `reserve 6.00 GiB`, cioe' il default
"normale" (non il presunto fallback-a-16 di CLAIM-019). Sembra esserci una
validazione che scarta/clampa richieste di reserve troppo grandi (>~ VRAM
totale) e ricade sul default a 6 GiB, invece di onorare 16 letteralmente.
Quindi **la run B non ha mai testato davvero "reserve=16"**: ha testato lo
stesso path di default-6GiB/cap154/hit0 gia' noto dalle probe precedenti
(K12/K16 co-resident del 2026-07-11 mattina).

## Verdetto

- Riproducibilita' dell'abort: **1/2** (50%, campione troppo piccolo per
  concludere se e' deterministico o intermittente).
- Il valore letterale 16 non arriva mai a "reserve=16.00 GiB attivo" nei log
  osservati: o abortisce prima (run A) o viene silenziosamente clampato a
  6.00 GiB (run B). **Non abbiamo mai osservato un run che onori
  effettivamente reserve=16.**
- Ipotesi del coordinator da verificare in futuro (fuori budget qui): run
  storici con reserve default/non settata che si credevano "degradati"
  (bassa qualita', loop, L0) potrebbero in realta' essere abortiti in questo
  stesso modo silenzioso, non degradati — confound retroattivo potenziale.
  Non confermabile con 1 solo campione anomalo; servirebbe un run mirato
  (N>=5 ripetizioni identiche) per stabilire un tasso di riproducibilita'
  reale, fuori scope di questa probe.
- Nessun fix tentato qui (la patch 0024 cambia gia' il default, per istruzione
  del coordinator "non serve fixarlo ora").

## Raccomandazione pratica per probe future

Non usare `DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB=16` per simulare
"cache-off" — non e' un modo affidabile ne' onorato. Per un vero cache-off
di confronto, usare invece un `--ssd-streaming-cache-experts` piccolo (es.
32, quasi tutto miss) con `reserve=1` fisso (che sappiamo essere parse-safe
e onorato), come nel batch bitexact2 di questa stessa probe.

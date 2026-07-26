# DS4 `DS4_METAL_PREFILL_CHUNK` — audit READ-ONLY

Data: 2026-07-26  
Scope: DS4 corrente, `ctx_capacity=150000`, decode corto e successiva validazione a KV viva lunga.  
Esecuzioni DS4 effettuate: **nessuna**. Repository condivisi modificati: **nessuno**.

## Verdetto

`DS4_METAL_PREFILL_CHUNK` non è una leva solo-prefill. Nel server la capacità del
grafo è calcolata dal **ctx allocato**, non dalla lunghezza del prompt:
`s->prefill_cap=metal_graph_prefill_cap_for_prompt(ctx_size)`. Quindi a
`ctx=150000` i valori 250/600/768 sono allocati integralmente anche per un prompt
di 13 token ([ds4.c:20167](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:20167),
[ds4.c:20192](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:20192)).

La macro muove insieme:

1. 37 tensori `batch_*` eager più `prefill_tokens`, lineari nel chunk;
2. `indexer_scores`, `comp_mask` e `comp_selected`, lineari nel chunk;
3. il raw KV ring, a gradini di 256 righe;
4. l'eventuale `mtp_raw_cache`, che segue il raw ring;
5. il picco del scratch CUDA `g_cuda_tmp` durante il prefill;
6. numero e confini delle wave di prefill, incluso resumed-prefill;
7. compatibilità degli snapshot, che richiede lo stesso `prefill_cap`;
8. indirettamente, VRAM libera, model-stream cache elastica e sizing effettivo
   della cache esperti.

Non muove la capacità KV compressa: a ctx 150k essa dipende dai ratio per-layer
4/128 e resta identica fra i bracci.

La prossima matrice documentata è A3 = 250/600, 768 solo se giustificato, sul
vincitore A2 e sul medesimo binario
([piano:1814](C:/Users/imanu/Documents/Codex/2026-07-25/legg/DS4_OPERATIONAL_PLAN.md:1814),
[riconciliazione:69](C:/Users/imanu/source/repos/reap-loop/docs/DS4_ATLAS_RECONCILIATION_20260726.md:69)).
Il vincitore A2 non è ancora disponibile.

**READY_FOR_RUN_SLOT=no**

## Geometria esatta a `ctx=150000`

Costanti correnti: 43 layer, head/value width 512, SWA 128, indexer width 128
([ds4.c:104](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:104)).
I layer 0-1 non comprimono; i pari 2..42 usano ratio 4 e gli odd 3..41 ratio
128 ([ds4.c:449](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:449)).

Formule:

```text
prefill_cap = min(env_chunk, ctx)
raw_rows    = min(8192, align256(128 + prefill_cap))
comp_cap    = floor(150000 / 4) + 2 = 37502

eager_batch_bytes/token =
  prefill_tokens + 37 batch_* = 1,018,356 B = 0.971179962 MiB

scores+mask bytes/token = 2 * 37502 * 4 = 300,016 B
selected bytes/token    = 512 * 4       =   2,048 B
```

Il raw-ring sizing è nel sorgente
([ds4.c:16160](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:16160));
le allocazioni eager sono a
[ds4.c:10836](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:10836)
e gli scratch a
[ds4.c:10782](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:10782).

| chunk | raw rows | raw KV 43 layer | batch eager | scores+mask | selected | MTP raw, se attivo | stima `context buffers` del codice |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 512 | 43.000 MiB | 242.795 MiB | 71.529 MiB | 0.488 MiB | 1.000 MiB | 2,083.060 MiB |
| 256 | 512 | 43.000 MiB | 248.622 MiB | 73.246 MiB | 0.500 MiB | 1.000 MiB | 2,084.776 MiB |
| 600 | 768 | 64.500 MiB | 582.708 MiB | 171.671 MiB | 1.172 MiB | 1.500 MiB | 2,204.701 MiB |
| 768 | 1024 | 86.000 MiB | 745.866 MiB | 219.738 MiB | 1.500 MiB | 2.000 MiB | 2,274.269 MiB |
| 2048 | 2304 | 193.500 MiB | 1,988.977 MiB | 585.969 MiB | 4.000 MiB | 4.500 MiB | 2,747.999 MiB |

La KV compressa comune vale circa **1,968.530 MiB** a ctx 150k. La colonna
`context buffers` replica la stima del codice: raw + compressed +
`scores+mask`; **non** è VRAM effettiva totale e omette batch, selected,
state per-layer, MTP/spec, logits e overhead allocatore
([ds4.c:16218](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:16218),
[ledger:5304](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:5304)).

Delta richiesto noto 600−250, prima di overhead e `g_cuda_tmp`:

```text
batch eager                 +339.913 MiB
scores + mask               +100.142 MiB
selected                      +0.684 MiB
raw KV 43 layer              +21.500 MiB
MTP raw, se attivo            +0.500 MiB
totale richiesto noto       +462.739 MiB
```

Da 600 a 768 il costo noto aggiuntivo è circa **+233.225 MiB**. Questo salto,
insieme a raw rows 768→1024, è il motivo per cui 768 non va aggiunto
automaticamente.

Nel tipo del grafo esistono 38 nomi `batch_*`, ma `batch_ffn_out` è lazy e si
materializza soltanto per steering/debug/materializzazione esplicita; se attivo
aggiunge altri 16 KiB per token
([ds4.c:10633](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:10633)).

## Lifecycle e confondenti

### Prefill e resumed-prefill

Il prefill è layer-major: ogni chunk attraversa tutti i 43 layer
([ds4.c:9399](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:9399)).
I confini sono allineati ai multipli assoluti di `prefill_cap`; un resumed
prefill parte con un chunk parziale fino al prossimo confine
([ds4.c:15737](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:15737)).

Un suffisso cached di almeno 32 token usa il path batched; sotto 32 usa
decode token-by-token e preserva la snapshot pubblicata
([ds4.c:20562](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:20562)).
Quindi il follow-up corrente da 19 token non misura la wave 250/600: misura
l'effetto di memoria/lifecycle sul decode e deve restare un gate di correttezza.

Chunk più largo può ridurre il numero di pass e la rilettura esperti tra chunk.
G37 mostrò, sul proprio binario/regime, che restringere 43→16→8 aumentava
traffico logico e TTFT; gli output restavano exact
([ledger:951](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:951)).

### Decode-start, cache sizing e request-end

Con `DS4_CUDA_KV_STAGED_RING=1`, raw e compressed KV vengono copiati in un'arena
pinned e liberati dal device al decode-start
([ds4_cuda.cu:19747](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4_cuda.cu:19747)).
`DS4_CUDA_RELEASE_PREFILL_SCRATCH=1` libera soltanto `g_cuda_tmp`: i batch tensor
restano vivi fino alla distruzione del grafo
([ds4_cuda.cu:26928](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4_cuda.cu:26928),
[ds4.c:10298](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:10298)).

Il sizing della cache esperti legge la VRAM libera e applica requested/reserve
([ds4_cuda.cu:33023](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4_cuda.cu:33023)).
Non è lecito dedurre “più chunk = N slot in meno”: la cache è limitata a 140
nel manifest corrente e la model-stream cache è elastica, quindi può assorbire
la differenza. Vanno registrati capacità effettiva, `free_at_size`, residenti,
model-stream bytes e VRAM minima. Il tracer storico dimostrò proprio
l'assorbimento elastico e l'omissione dei batch dalla stima contesto
([ledger:5285](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:5285)).

A fine richiesta il ring KV viene ripristinato sul device
([ds4_cuda.cu:27002](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4_cuda.cu:27002)).
Perciò ogni arm deve riportare anche D2H al decode-start e H2D al request-end,
non soltanto il picco durante decode.

### Snapshot

L'header serializza ctx, chunk, raw cap/window e comp cap
([ds4.c:18494](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:18494)).
Il restore rifiuta uno snapshot con `saved_prefill_cap != current_prefill_cap`
([ds4.c:18763](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4.c:18763)).
Il raw cap può invece differire se contiene le righe live: il loader ricostruisce
il ring fisico corrente. Conseguenza operativa: **mai riusare lo snapshot 250
nel braccio 600/768**; servono snapshot separati, stesso token stream e stesso
binario.

## Riconciliazione storica: nessuna composizione cross-binary

| chunk | binario/regime | risultato ammesso | cosa non si può concludere |
|---:|---|---|---|
| 250 | exe corrente SHA256 `F2F6…1DFF`, ctx150k, OPEN=1, pinned, staged, M2 | short graph 2.463047 t/s; output/lifecycle exact | non è un A/B chunk |
| 250 | exe patchato MD5 `320A…DF1`, ctx150k | cold 4,088 computed = 1.20 t/s; live tail ~2.36 t/s fino a pos6603 | non va fuso con il dato del binario corrente |
| 256 | G73-derived, ctx256/1024/8192, diagnostico n=1 | canary completati; a ctx8192 prefill 0.1655 e decode 5.05 t/s | nessuna exactness o baseline; workload e ctx diversi |
| 256 vs 600 | `wt-converge` MD5 `2AD6…E91A`, IQ2, OPEN=0, stesso sweep | 2.15 vs 2.19 t/s: +1.9%, quindi varianza | non promuove 600 sul binario corrente |
| 256 vs 600 | stesso `wt-converge`, OPEN=1 | ledger riporta 1.30 vs 2.33 t/s, effetto storico direzionale | richiede nuova verifica manifest-for-manifest |
| 768 vs default 2048 | vecchia build strumentata, ctx8192 | primi ~60 token 2.67 vs 0.56, rapporto ~4.8× sulla stessa build | assoluti gonfiati ~6.5× dal profiling; seconda degradazione non risolta |
| 768 | run “clean”/post-Atlante | circa 3.92–4.24 t/s riportati | prompt, temperatura e setup sono confusi; non è baseline corrente |
| 2048 | G73 diagnostic C3, ctx8192 | abort contaminato | nessuna misura timing/quality |

Evidenze: diagnostica 256/2048
([ledger:3359](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:3359));
768/2048 profilato
([ledger:4982](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:4982));
sweep 600/256/128
([ledger:5684](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:5684));
correzione di binario G73
([ledger:5729](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:5729)).

Il vecchio 4.85 t/s OPEN=0 apparteneva al binario MD5 `921D…`; non si riprodusse
su `wt-converge`, dove OPEN=1/pinned fece 2.41 e OPEN=0/chunk600 2.19. Per questo
A2 deve precedere A3 sul binario corrente
([piano:1767](C:/Users/imanu/Documents/Codex/2026-07-25/legg/DS4_OPERATIONAL_PLAN.md:1767)).

## Matrice minima proposta

### Prerequisito non soddisfatto: A2

Congelare il vincitore dello stesso-binario A2:

```text
foundation=P0
G73_OPEN=winner(1|0)
G73_PAGEABLE_OVERFLOW_GB=0
micro-patch P2-A/P2-B/P3-A/P3-B=OFF
GRAPH_TENSOR_DEVICE=<UNSET>
RAW_CAP=<UNSET>
same exe/source/model/prompt hashes
```

Non importare il vincitore storico.

### A3-S/P: 250 contro 600

Ordine minimo credibile: **A-B-B-A**, quattro processi freddi e indipendenti.
Ogni processo deve contenere due lane, riportate separatamente:

1. **S, capacity/short**: lifecycle deterministico attuale, turn1 13 prompt +
   128 completion; turn2 160 prompt, cached 141, computed 19 + 32 completion.
   Misura allocazione a ctx150k, TTFT corto, decode corto e P0 short-suffix.
2. **P, prefill**: prompt UTF-8/token-ID congelato con almeno 1,200 computed
   token, `cached=0`, seguito da almeno un output token. Misura wave count,
   prefill wall/graph, TTFT e traffico. Il prompt deve superare 600, altrimenti
   non esercita la differenza di tiling.

Se il costo della lane P rende impossibile ABBA nello stesso slot, eseguire
prima S-ABBA; P resta obbligatoria prima di qualsiasi claim sul prefill.

Nel manifest 47/47 la riga 44 viene **sostituita**, non aggiunta:

```text
Arm A: DS4_METAL_PREFILL_CHUNK=250  -> expected raw_rows=512
Arm B: DS4_METAL_PREFILL_CHUNK=600  -> expected raw_rows=768
```

Tutte le altre 46 righe sono byte-identiche. Il manifest corrente completo è
nel ledger ([ledger:8424](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:8424));
provenance corrente a
[ledger:8504](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:8504).

### Condizione esatta per aggiungere 768

Aggiungere C=768, in ordine controbilanciato, soltanto se:

- 600 passa tutti i gate correctness/lifecycle/resource;
- 600 migliora di almeno 10% una metrica primaria coerente
  (`prefill_t/s`, TTFT, short graph/wall t/s), **senza** regressione ≥10% nelle
  altre metriche primarie;
- il segno è concorde nelle due repliche per arm;
- raw rows e footprint osservati sono quelli attesi;
- non c'è riduzione/compensazione anomala della cache effettiva né margine VRAM
  insufficiente.

Se tutti i delta 600/250 sono sotto 10%, è varianza: stop, niente 768. Se il
risultato è misto (per esempio prefill +15%, decode −12%), non esiste un
“vincitore” unico: stop e decisione esplicita sul workload, non 768 automatica.

### L: validazione KV viva lunga

Solo dopo un vincitore A3-S/P valido, confrontarlo con il controllo 250 sullo
stesso binario e token stream. Minimo canary: un processo per arm; per
promozione: replica reverse-order.

Usare snapshot **per-arm** oppure ricostruire lo stesso prefisso. Gate minimo
comparabile: frontier live ≥6603, con finestre:

```text
575-1974      pre-indexer maturo
2051-2574     transitorio indexer
2575-3074     recupero post-transitorio
5075-6603     KV lunga stabile, metrica primaria
```

La curva storica valida non mostrò degrado monotono fino a 6603
([ledger:6294](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:6294)).
Passare questo gate autorizza soltanto il claim “KV viva ≥6.6k con capacità
150k”, non “KV viva 150k”. Quest'ultimo richiede un ulteriore checkpoint/prefix
verificato vicino a 149k.

## Contratto di misura e gate

Per ogni processo/arm salvare:

```text
identity:
  exe bytes+SHA256; ds4.c/ds4_cuda.cu/ds4_gpu.h SHA256
  model bytes+SHA256; runner SHA256; prompt UTF-8 SHA256; token-ID SHA256
  47/47 obtained manifest + explicit <UNSET> controls

geometry/resources:
  ctx_capacity; prefill_cap; raw_kv_rows; comp_cap
  requested bytes by allocation tag; managed/device allocation count
  g_cuda_tmp peak/released; staged KV device/pinned/released bytes
  VRAM free startup/prefill/decode-start/min/request-end
  model-stream live bytes; expert-cache requested/capacity/count/free_at_size
  RAM available/commit; GPU power/util/temp

request:
  prompt_tokens; cached_tokens; computed_tokens=prompt-cached
  live_frontier_start/end; completion_tokens; seed/temp/think/max_tokens

performance:
  prefill wall seconds and computed_tokens/seconds
  TTFT server; optional client streaming TTFT
  decode wall t/s excluding prefill; graph t/s
  p50/p95/p99 and fixed-position windows

traffic/engagement:
  route VRAM/RAM/SSD class; logical/physical reads
  H2D/D2H counts+bytes; upload sync; wave/chunk count
  fallback/failure/timeout/invariant; native shutdown; clean postflight
```

Il server traccia già prompt, effective prompt e cached tokens
([ds4_server.c:6720](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/wt-hot-reserve/ds4_server.c:6720)).
Per l'append, `prefill_t/s` deve usare **computed_tokens**, mai l'intero prompt
([ledger:6514](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:6514)).

Gate:

- **Exact output**: hash SHA256 dei byte UTF-8 assistant e, se raccolti, della
  sequenza token identici all'oracolo A2 e fra tutti gli arm. Per lo smoke
  corrente gli oracoli sono turn1
  `7f82253a4825191926f56073e40f10a0cff5541a721731bc81d2909dc1a4a65b`
  e turn2
  `0179556c8e2dbcdc818fad315ca4df78f7537b63816dac276615b314195b13eb`
  ([ledger:8516](C:/Users/imanu/source/repos/reap-loop/docs/EXPERIMENTS_LEDGER.md:8516)).
- **Manifest**: 47 nomi unici; unica differenza A/B = valore chunk.
- **Geometry**: A raw512, B raw768; `RAW_CAP` non ereditato.
- **Lifecycle**: cached141/computed19 nello smoke, snapshot e resident generation
  invariati, zero decode-refused/quarantine.
- **Runtime**: zero fallback vietati, zero route-I/O failure, engagement previsto.
- **Resources**: nessun OOM/alloc failure, staged migration/restore completi,
  cache richiesta disponibile, macchina comparabile.
- **Performance**: media arm; delta assoluto <10% = varianza. Non sommare delta
  di leve o binari diversi.
- **Long KV**: confronto sulle stesse finestre live, non sulla sola capacità ctx.

## Falsificatori

1. **Footprint**: raw rows diversi da 512/768 o byte richiesti incompatibili con
   le formule falsificano il trattamento; probabile env ereditata/override.
2. **Prefill-wave**: con computed prompt >600, wave count e confini non cambiano
   come previsto: trattamento non engaged.
3. **Prefill benefit**: wave count cambia ma prefill/TTFT resta entro 10%:
   il vantaggio è varianza in quel regime.
4. **Decode-memory**: cache effettiva, stream-cache, VRAM e route traffic restano
   equivalenti e short/long decode è entro 10%: nessun effetto decode provato.
5. **Long-KV**: segno non concorde o differenza confinata al transitorio
   2051-2574: non è un miglioramento KV lungo.
6. **Semantica**: qualunque output/hash, token count, finish reason o checkpoint
   incompatibile invalida l'arm anche se più veloce.

## Rischi e abort immediati

- hash binario/sorgente/modello/runner/prompt non atteso;
- A2 winner assente o diverso fra arm;
- manifest non 47/47, variabili DS4 ereditate, `RAW_CAP` presente;
- server/porta già attivi o postflight non pulito;
- raw rows, ctx, chunk o comp cap non attesi nei log;
- OOM, alloc failure, commit/RAM/VRAM sotto la soglia preflight;
- fallback, timeout, mailbox quarantine, decode refused, route-I/O failure;
- output non exact o risposta malformata;
- snapshot 250 riutilizzato su 600/768;
- migrazione/restore staged KV incompleti;
- mismatch termico/quiescenza fra coppie: scartare la coppia, non il solo arm;
- abort di un arm: il control partner non è un risultato confrontabile.

## Bloccanti prima del run slot

1. completare A2 sul medesimo exe corrente e dichiarare `G73_OPEN=winner`;
2. preparare fuori dal repository condiviso un runner A3 che parametrizzi il
   chunk sostituendo la riga 44, registri cached/computed e memoria effettiva;
3. congelare prompt/token IDs per lane P e long-KV, con oracoli output;
4. validare il runner senza server (`ValidateOnly`) e produrre un arming receipt.

Il runner lifecycle disponibile verifica 47 righe e gli hash, ma fissa il file
manifest con `chunk=250` e non raccoglie ancora tutto il contratto A3
([runner:174](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/runtime_p0p1/run_two_turn_lifecycle.ps1:174),
[manifest:44](C:/Users/imanu/Documents/Codex/2026-07-25/legg/work/runtime_p0p1/manifest_47.env:44)).

`READY_FOR_RUN_SLOT=no`

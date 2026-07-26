# DS4 P4 — architettura demand/promotion per saturare il decode CUDA

**Stato:** proposta architetturale read-only  
**Data di osservazione:** 2026-07-26T17:33:30+02:00  
**Scope:** decode DS4 su RTX 3060, percorso G73/P0/P2/P3  
**Autorizzazione implicita:** nessuna. Questo documento non autorizza build, run, benchmark o modifiche ai sorgenti condivisi.

## 1. Decisione

La leva successiva con il miglior rapporto payoff/rischio è trasformare il caricamento degli expert miss da una sequenza di copie e publish per expert a un'unica transazione demand per layer:

- massimo due H2D per layer con miss;
- un publish demand per layer;
- handoff GPU→consumer mediante evento `demand_ready`, senza attendere sul thread CPU la completion dello stream;
- promotion separata dal demand corrente e spostata su uno stream a priorità inferiore solo nel secondo gate.

La sequenza raccomandata è:

1. **P4a:** batch demand per layer, evento `demand_ready`, semantica promotion corrente preservata.
2. **P4b:** current-token demand indipendente e promotion asincrona su stream low-priority con ownership per-slot.

CUDA Graph è deliberatamente differito. Non va introdotto prima di avere eliminato dal percorso caldo readback, attese host e quantità dinamica di submission H2D/publish.

## 2. Diagnosi quantitativa

I receipt esistenti descrivono un problema di alimentazione della GPU, non un limite dimostrato della capacità di calcolo:

- 43 layer e 43 route call per token;
- circa 201 expert miss per token nella regione misurata;
- circa 603 H2D per token nel percorso a tre copie per expert;
- circa 402 H2D per token con P2-A packed, ma stessi byte logici;
- circa 43,68 kernel publish per token con P2-B;
- circa 1,424 GB di payload H2D logico per token;
- GPU busy 38,144% e idle 61,856% nel direct capture;
- sei gap da 111–118 ms, 688,257 ms complessivi;
- H2D attivo 161,284 ms per 1,2698 GB, 7,873 GB/s;
- monitor osservato: utilization media 22,28%, power media 27,92 W, picco 43,61 W su limite 170 W;
- QD4: worker circa -10%, graph throughput circa +6,24%;
- stack M2: 2,463047 t/s, circa +7,31% rispetto al relativo controllo.

Il cold token campionato aveva 258 miss:

| Percorso | Submission H2D |
|---|---:|
| Legacy, 3 copie/expert | 774 |
| P2-A packed, 2 copie/expert | 516 |
| P4, 2 copie/layer | ≤86 |

P2-A e P2-B hanno quindi ridotto singole categorie di submission, ma nessun receipt dimostra ancora che la GPU sia alimentata con continuità. Non esiste evidenza per promettere 5 t/s.

## 3. Call graph e dipendenze

Percorso reale per layer:

1. Il token decode attraversa serialmente i layer in `ds4.c`.
2. Il layer produce router logits, selected expert e weights.
3. `ds4_gpu_routed_moe_one_tensor` entra nel percorso routed MoE.
4. `cuda_moe_gpu_resident_routes_begin` assegna sequenza/generazione, porta la mailbox in `GPU_WRITING` e accoda il resolver sul default stream.
5. `moe_resolve_resident_routes_kernel` determina hit/miss, scrive la request mapped, esegue `__threadfence_system` e pubblica `sequence`.
6. Il worker valida request e generazione, reclama `WORKER_OWNS`, risolve snapshot/probation/SSD e produce i byte host.
7. Il percorso corrente accoda copie packed o legacy e publish sul singolo `route_upload_stream`.
8. `finish` rende utilizzabili i puntatori per i kernel miss.
9. Dopo avere accodato tutti i consumer del layer, `ds4_gpu_moe_route_consumer_done` registra la fine dell'ownership consumer.

Dipendenze inevitabili:

- il router del layer `L+1` dipende dall'output completo del layer `L`;
- il GEMV di un miss dipende dai suoi pesi demand e dal publish del puntatore;
- il token successivo dipende da logits e sampling del token corrente;
- una cache entry residente non può essere pubblicata prima che i suoi byte siano completi.

Lavoro sovrapponibile:

- I/O QD4, staging CPU e H2D dei miss con il compute degli expert hit;
- shared expert con il supply dei miss;
- promotion destinata a token futuri con i consumer demand correnti;
- preparazione host delle descriptor con submission CUDA già in volo.

La finestra utile è quindi interna al layer e, soprattutto, nella promotion che non è necessaria per il risultato corrente.

## 4. P4a — batch demand per layer

### 4.1 Patch minima

P4a deve riusare il layout SoA packed esistente:

```text
host demand slab
├── gate[0..M-1]
├── up[0..M-1]
└── down[0..M-1]

device demand slab
├── gate[0..M-1]
├── up[0..M-1]
└── down[0..M-1]
```

Per ogni layer:

1. Compattare tutti i miss negli slot densi `0..M-1`, nell'ordine della request.
2. Fare scrivere QD4 direttamente negli slot densi.
3. Copiare le sorgenti RAM/snapshot/probation nello stesso slab host, contabilizzando separatamente `cpu_stage_bytes` e `cpu_stage_us`.
4. Accodare esattamente:
   - una `cudaMemcpy2DAsync` per gate+up;
   - una `cudaMemcpyAsync` per down.
5. Per le admission residenti già decise dalla policy corrente, eseguire demand→resident D2D sullo stesso stream prima della readiness. Questo costo rimane intenzionalmente critico in P4a per preservare l'evoluzione corrente della cache.
6. Accodare un solo publish per layer, combinando i puntatori demand e residenti.
7. Registrare un evento timing-disabled `demand_ready`.
8. Pubblicare `ready_sequence` e `ready_generation` quando il record dell'evento è stato accodato, non quando la GPU ha già completato il lavoro.
9. In `finish`, validare la generazione e accodare `cudaStreamWaitEvent(default_stream, demand_ready)`.
10. Eliminare dal ramo P4a `cudaStreamSynchronize`, `cudaEventSynchronize` e il polling CPU della completion GPU.

Il poll host ammesso è soltanto quello necessario a sapere che il worker ha registrato l'evento corretto. Non sostituisce una completion GPU.

### 4.2 Perché un singolo slab è sufficiente in P4a

Il resolver del layer successivo è accodato sul default stream dopo i consumer del layer corrente e dopo `consumer_done`. Non può quindi produrre una nuova request al worker prima che:

- l'H2D demand precedente sia completo;
- i consumer precedenti abbiano cessato di usare il device slab;
- le promotion P4a, poste prima di `demand_ready`, siano terminate.

P4a non richiede ancora un ring multi-slot. Questa scelta riduce la superficie P0 e isola il valore del batching/event handoff.

### 4.3 Target P4a

- `h2d_calls <= 2 * layers_with_miss`, massimo 86 per token;
- `demand_publish <= layers_with_miss`, massimo 43 per token;
- byte H2D logici invariati;
- un `demand_ready` record e un default-stream wait per layer con miss;
- zero attese host della completion GPU;
- esattezza completa e nessun cambiamento non contabilizzato della policy di cache.

## 5. P4b — separazione demand/promotion

P4b parte soltanto dopo il superamento dei gate P4a.

### 5.1 Stream

- `route_demand_stream`: nonblocking, priorità alta.
- `route_promotion_stream`: nonblocking, priorità bassa.

Il token corrente usa sempre i puntatori del demand slab. La promotion:

1. attende `demand_ready`;
2. copia demand→resident;
3. pubblica la resident cache entry solo dopo i byte;
4. registra `promotion_done`.

Il default stream non attende mai `promotion_done`.

La priorità dello stream non garantisce la preemption del copy engine; il beneficio deve essere provato da timeline e non inferito dall'API.

### 5.2 Slot fisici

La promotion può vivere oltre la fine del consumer corrente. Servono quindi:

- un ring di almeno tre demand slot; oppure
- un pool promotion separato che svincoli immediatamente il demand slot.

La patch iniziale minima è il ring a tre slot, con eventi e generazioni per-slot.

Se non è disponibile capacità promotion prima dell'enqueue:

- annullare l'admission;
- rimborsare reservation e budget LFRU/G133;
- lasciare invariato lo stato residente precedente;
- non attendere sul demand path.

Un drop mantiene l'esattezza numerica del token corrente, ma invalida un run candidato alla promozione perché altera il comportamento prestazionale della cache.

## 6. State machine, eventi e generazioni

### 6.1 FSM demand

```text
FREE(g-old)
  → GPU_WRITING(g)
  → WORKER_OWNS(g)
  → UPLOAD_ENQUEUED(g)
  → CONSUMER_DONE(g)
  → FREE
```

Semantica:

- `FREE`: lo slot non è referenziato da producer, worker, consumer o promotion.
- `GPU_WRITING(g)`: il resolver GPU sta producendo la request della generazione `g`.
- `WORKER_OWNS(g)`: il worker ha validato `sequence`, `generation`, layer e slot.
- `UPLOAD_ENQUEUED(g)`: H2D demand, publish e record di `demand_ready[g]` sono stati accodati. Non significa completion.
- `CONSUMER_DONE(g)`: `consumer_done[g]`, registrato dopo tutti i consumer, ha completato.
- Il ritorno a `FREE` in P4b richiede anche `promotion_done[g]` se una promotion è stata accodata.

### 6.2 Regole di generazione

- Generazione monotona a 64 bit.
- Slot fisico selezionato da `g % N`.
- Ogni evento porta un `event_generation`.
- Al riuso di un ring a `N` slot va controllato il precedente occupante `g-N`, non genericamente `g-1`.
- Il producer pubblica il payload prima di `sequence` tramite fence system.
- Il consumer accoda il wait solo dopo che il worker ha pubblicato `demand_ready_recorded_generation == g`.
- Qualunque mismatch di slot, sequenza o generazione porta a quarantena/fail-closed.

Accodare un wait su un evento non ancora registrato può trasformarlo in un no-op: il flag host "event recorded" è parte del contratto di correctness, non semplice telemetria.

### 6.3 FSM promotion

```text
PROMO_FREE
  → PROMO_RESERVED
  → PROMO_ENQUEUED
  → PROMO_DONE
  → PROMO_FREE

PROMO_RESERVED
  → PROMO_ROLLED_BACK
  → PROMO_FREE
```

La mappa residente può essere aggiornata soltanto dopo che tutti i byte sono disponibili. Un rollback non deve lasciare reservation, contatori o tier state parziali.

## 7. Correctness, fallback e rischi P0

Invarianti:

- selected expert e router weights immutati;
- byte degli expert identici indipendentemente dalla sorgente;
- nessun puntatore demand pubblicato prima dell'H2D;
- nessun puntatore resident pubblicato prima della promotion;
- nessun riuso host/device slab prima degli eventi proprietari;
- `consumer_done` registrato dopo tutti i consumer routed/shared;
- late failure continua a usare il replay exact dell'intero layer;
- all-hit, partial-miss e sei-miss attraversano una transizione FSM completa e verificabile;
- nessun fallback silenzioso.

Rischi P0:

1. riuso prematuro di uno slot;
2. wait su evento non ancora registrato;
3. generazione evento diversa dalla generazione dello slot;
4. publish resident anticipato;
5. overwrite di una vittima ancora referenziata;
6. pitch o offset errato nel layout SoA;
7. staging CPU che annulla il guadagno;
8. errore CUDA asincrono osservato dopo la pubblicazione host;
9. regressione sul secondo turno/follow-up;
10. starvation o contesa del copy engine da parte della promotion.

Qualunque violazione strutturale è terminale per il candidato. Il percorso può produrre un risultato exact tramite fallback esistente, ma il run non è promuovibile.

## 8. Matrice runtime proposta

I nomi `DS4_CUDA_MOE_ROUTE_LAYER_BATCH` e `DS4_CUDA_MOE_ROUTE_SPLIT_PROMOTION` sono **proposti** e non risultavano implementati al momento dell'osservazione.

La matrice eredita il manifest runtime congelato e modifica esclusivamente i flag indicati:

| Braccio | Packed | Batched publish | QD4 | P3A host-selected | P4 layer batch | P4 split promotion | Scopo |
|---|---:|---:|---:|---:|---:|---:|---|
| R0 / M5 control | 1 | 1 | 4 | 0 | 0 | 0 | Baseline successiva valida |
| R1 / P4a | 1 | 1 | 4 | 0 | 1 | 0 | Batch demand + event handoff |
| R2 / P4b control | 1 | 1 | 4 | 0 | 1 | 0 | Controllo same-binary di P4b |
| R3 / P4b candidate | 1 | 1 | 4 | 0 | 1 | 1 | Demand/promotion separati |
| Rollback | 1 | 1 | 4 | 0 | 0 | 0 | Ritorno a M5 |

Mapping dei flag esistenti:

```text
DS4_CUDA_MOE_GPU_RESIDENT_ROUTES=1
DS4_CUDA_MOE_ROUTE_NO_DEFAULT_SYNC=1
DS4_CUDA_MOE_SPLIT_FUSED=1
DS4_CUDA_MOE_ROUTE_PACKED_COPY=1
DS4_CUDA_MOE_ROUTE_BATCHED_PUBLISH=1
DS4_CUDA_G73_ROUTE_IO_QD=4
DS4_CUDA_G73_REUSE_HOST_SELECTED=0
```

Flag proposti:

```text
DS4_CUDA_MOE_ROUTE_LAYER_BATCH=1
DS4_CUDA_MOE_ROUTE_SPLIT_PROMOTION=1
```

Regole:

- entrambi default OFF;
- P4b richiede P4a e fallisce chiuso se attivato da solo;
- nessun folding dei flag prima di receipt promuovibili;
- stesso binario all'interno di ogni coppia A/B;
- il manifest completo rimane identico salvo un singolo overlay.

## 9. Telemetria e gate

### 9.1 Telemetria non perturbante

Contatori per token/layer:

- layer con miss e numero di miss;
- `cpu_stage_bytes`, `cpu_stage_us`;
- H2D/D2D call e byte;
- demand publish e promotion publish;
- `demand_ready` registrati e wait accodati;
- mismatch di generazione;
- source wait, enqueue wait, eventuale host GPU-completion wait;
- promotion reserved, enqueued, completed, dropped e late;
- slot stall e massimo numero di generazioni in volo.

Vincoli:

- contatori posseduti dal worker dove possibile;
- eventi preallocati;
- nessun timing event per copia nel percorso normale;
- tracing CUDA solo su token selezionati;
- lettura dopo il drain già previsto;
- monitor GPU ritagliato alla sola finestra decode.

Il contatore `pread_bytes` precedentemente soggetto a underflow non può essere usato come gate finché non è reso affidabile.

### 9.2 Gate P4a

- output, selected expert, weights e KV esatti;
- H2D ≤86/token;
- demand publish ≤43/token;
- byte H2D logici invariati;
- zero fallback, late failure e generation mismatch;
- zero `cudaStreamSynchronize`/`cudaEventSynchronize` nel ramo caldo P4a;
- varianza tra processi <10%;
- graph throughput medio ≥10% rispetto a R0.

### 9.3 Gate P4b

- tutti i gate P4a;
- zero demand stall attribuibile alla promotion;
- zero drop/late nel candidato promuovibile;
- nessun publish resident anticipato;
- graph throughput medio ≥10% rispetto a R2;
- nessuna regressione significativa nei tail.

### 9.4 A/B futuro, non eseguito

- Correttezza iniziale: casi all-hit, 1 miss, 6 miss e tutte le classi di sorgente.
- Due turni 128+32 con follow-up, perché un precedente stack M4 falliva solo al secondo turno.
- Performance su regione matura di almeno 512 token.
- Almeno quattro processi freschi per braccio, ordine ABBA/BAAB.
- Nsight su 1–2 token rappresentativi, non sull'intero run.

## 10. Target GPU falsificabili

P4a:

- GPU busy ≥55–60%;
- GPU idle ≤40–45%;
- somma dei sei grandi gap da 688 ms a ≤350 ms;
- H2D active bandwidth non peggiore di oltre 10% rispetto a 7,873 GB/s.

P4b:

- GPU busy ≥70%;
- massimo un gap di supply ≥100 ms;
- utilization decode-window mediana ≥65–70%;
- nessuno stall demand causato dalla promotion;
- potenza media decode-window almeno circa doppia rispetto ai 28–30 W osservati, senza throttling.

La saturazione operativa può essere dichiarata soltanto con:

- GPU busy ≥80%;
- supply-idle <10% del token;
- utilization mediana ≥80% su una regione lunga.

La sola potenza o un singolo campione al 98% non dimostrano saturazione.

## 11. Payoff atteso e rollback

Range ingegneristico:

- P4a: +3–12%;
- P4b: ulteriore +2–10%;
- combinato: +5–20%, circa 2,59–2,96 t/s partendo da 2,463 t/s.

La parte bassa è coerente con i guadagni già osservati per P2-B e QD4. La parte alta richiede che il batching faccia collassare i grandi gap di supply. Un risultato sotto il 10% è diagnostico ma non promuovibile.

Rollback:

- disabilitare P4b ritorna a P4a;
- disabilitare P4a ritorna a R0/M5;
- il percorso precedente resta intatto;
- combinazioni illegali falliscono chiuse;
- fallback, mismatch, drop o slot stall invalidano il run candidato;
- nessun consolidamento dei flag prima dei gate completi.

## 12. Provenance osservata

Le righe sotto identificano i file letti per il report. Gli hash attestano esclusivamente il contenuto osservato alla data indicata; non attestano un binario né uno stato Git.

### Documenti

| File | SHA-256 |
|---|---|
| `C:\Users\imanu\Documents\Codex\2026-07-25\legg\DS4_OPERATIONAL_PLAN.md` | `F6A8FC7528DC4A4D1E1568087A87E042164EB26D01DD25A730F8E1D820EF93A8` |
| `C:\Users\imanu\source\repos\reap-loop\docs\DS4_ATLAS_RECONCILIATION_20260726.md` | `C26C255D588052592E35A4DFB28A9247F419C04511029DDBEC1F411BD450CCA8` |
| `C:\Users\imanu\source\repos\reap-loop\docs\EXPERIMENTS_LEDGER.md` | `28501EA3BBC2D7EBD5778D015C4765FD50B01F59A18B6E53448A022A54B7E98E` |

### Sorgente corrente osservato

Root:

```text
C:\Users\imanu\Documents\Codex\2026-07-25\legg\work\wt-hot-reserve
```

| File | SHA-256 |
|---|---|
| `ds4_cuda.cu` | `6D6BE460609C74EEEA60B43F6D62DB812F722D4A270B45797EB6A17909446257` |
| `ds4_gpu.h` | `52AC84C2F10AE1D1EDB3032AFCFBDB1AAE000CDF148D98CCDAC42AA0739223AB` |
| `ds4.c` | `43B2ECE184916AA2731A299BE420A5133248BE24B3F17BCFB396D20B97CA1A2A` |
| `ds4_metal.m` | `81612C353EFCBDEFF58619BD38A9CD9E72382D6A9FDF2AC709BA27FCCD2D6F5D` |

Punti sorgente osservati in `ds4_cuda.cu`:

- request mapped: linea 559;
- FSM mailbox: linee 596–601;
- resolver: circa linea 24053;
- stream/counter cache: circa linee 26294–26343;
- reclaim/claim P0: circa linea 28164;
- `consumer_done`: circa linea 28409;
- flag runtime esistenti: circa linee 28451–28493;
- copy packed: circa linea 34749;
- worker/enforce: circa linee 35436–36632;
- begin/resolver launch: circa linee 40957–41143;
- finish: circa linea 41203.

Il sorgente è condiviso e può evolvere dopo questa osservazione; qualsiasi slot di implementazione deve ripetere la verifica degli hash prima di applicare una patch.

## 13. Readiness

`READY_FOR_IMPLEMENTATION_SLOT=yes`

Lo stato vale per uno slot isolato P4a, previa verifica che la provenance non sia cambiata. P4b rimane subordinato ai receipt P4a. Questo documento non autorizza l'esecuzione.

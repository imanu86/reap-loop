# Handoff canonico DS4 Windows, REAP, SPEX e Q1_0

Data di congelamento: 2026-07-19 (Europe/Rome)

Scopo: permettere a una nuova task Codex di ripartire senza ricostruire la
storia dalla conversazione. Questo documento separa risultati misurati, safety
gate, risultati negativi, ipotesi, codice committato e codice ancora sporco.

Questo handoff e' stato costruito da tre audit indipendenti e da una revisione
finale dell'orchestratore. I tre audit hanno coperto:

1. cronologia, risultati e ledger;
2. architettura di residenza, promozione e predizione;
3. stato Git, artefatti, provenienza e procedura di ripresa.

## 0. Leggere prima questo

### Stato in una frase

Il miglior risultato locale Windows misurato con protocollo `n=3` resta G73 a
`4.986667 t/s`, ma G73 usa uno snapshot request-scoped chiuso. Il percorso
attuale G129 vuole mantenere il router full/open, tenere tutti gli expert in
Q1_0 residenti in RAM come fallback e promuovere dinamicamente in IQ2 esatto
gli expert realmente caldi. G129 e' implementato in un working tree sporco ma
non e' ancora stato validato con un run runtime.

### Verita' che non vanno piu' confuse

- `G73 4.986667 t/s` e' un dato valido `n=3`, ma e' request-scoped closed.
- `G108 8.3 t/s` e' un safety `n=1`; non e' SOTA.
- `G112 6.76 t/s` e `G113/G114 circa 6.9-7.0 t/s steady` dimostrano il
  ceiling di trasporto Q1_0, ma gli output erano L0/L1; non sono SOTA.
- G123/G126/G127 sono full/open ed exact, ma stanno intorno a 1.6 t/s.
- K60/K75 baked sono fallback sperimentali; non sono la roadmap attiva.
- Q1_0 puro non e' una soluzione di qualita'. Serve come base/fallback o come
  componente di una rappresentazione promuovibile.
- SPEX non e' attivo come predittore in G129. Il sistema corrente osserva
  prefill e decode; non prevede ancora il futuro.
- Nessun verdetto si basa su `repeat_flag`, su un solo output o su `n=1`.

### Mandato consigliato per la nuova task

1. preservare il working tree G129 esattamente com'e';
2. chiudere i pochi gate statici e il build G129;
3. eseguire un solo safety `n=1` strutturale;
4. eseguire un long safety e assegnare subito L0-L3;
5. fermare il braccio se L0/L1;
6. solo se L2/L3, eseguire `n>=3` e confrontare con lo scope corretto;
7. committare runtime, runner, test e risultati in commit separati.

## 1. Regole permanenti del progetto

Queste regole prevalgono su qualsiasi vecchio runner o documento che le
contraddica.

1. Ogni verdetto prestazionale o qualitativo richiede almeno tre processi
   indipendenti. `n=1` e' solo safety o diagnostica.
2. Ogni output lungo va classificato L0-L3. Non usare presenza di `</html>`,
   repeat flag, n-gram o hash come proxy della qualita'.
3. I risultati contaminati da paging, disco occupato, processi concorrenti,
   build stale, modello non verificato o quiescenza saltata non entrano nella
   SOTA.
4. Non sovrapporre mai due runner DS4/GPU.
5. Registrare per ogni run: prompt, prompt SHA, modello, model SHA, sidecar e
   sidecar SHA, commit, binary SHA, build fingerprint, runner SHA, harness SHA,
   parametri, ambiente, quiescenza, TTFT, prefill, decode, output e grado.
6. Separare sempre scope `request-scoped closed` e `full/open`.
7. Il router full/open vede tutti gli expert. Residenza e rappresentazione
   possono cambiare; l'ammissibilita' degli expert no.
8. Nessuna static domain mask nella soluzione generale.
9. La cache VRAM non e' una mask: un expert assente dalla cache deve restare
   selezionabile e avere un percorso corretto.
10. Un expert freddo letto da SSD non puo' passare direttamente in VRAM nello
    stesso token. Prima deve entrare in RAM probation; la VRAM diventa
    eleggibile da una call successiva.
11. Conservare e pubblicare anche test negativi e protocol errors.
12. Non trasformare una misura di trasporto in un claim di qualita'.
13. Non trasformare un breve prefix exact in una prova di qualita' long-form.
14. Fermare presto un braccio chiaramente L0/L1, ma registrare l'aborto e la
    causa. Il fast-stop non sostituisce `n>=3` per un verdetto positivo.
15. Per output HTML, distinguere degenerazione da semplice esaurimento del
    token budget. Una pagina incompleta per budget non equivale a nonsense.

## 2. Hardware e ambiente autorevole

### Macchina locale

- OS: Windows nativo, non WSL per la roadmap corrente.
- GPU: NVIDIA RTX 3060 12 GiB, sm_86, WDDM.
- RAM fisica: 64 GiB.
- Modello principale su NVMe C:.
- CUDA usata nei build recenti: 12.6.
- Visual Studio: 2022 Community.
- CMake usato:
  `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Perche' Windows nativo

WSL era utile e ha prodotto i primi risultati REAP/PACE, ma il progetto si e'
spostato su Windows nativo per controllare direttamente RAM, pinning, WDDM,
file I/O e per evitare il limite operativo osservato intorno a 31 GiB pinned
nel percorso WSL. Windows nativo non ha eliminato automaticamente i problemi
di memoria: working set, file cache, commit charge e WDDM restano vincoli da
misurare.

### Ultimo stato macchina osservato durante l'audit

- nessun processo `ds4` o `ds4_server` attivo;
- nessun `rclone` attivo;
- GPU circa 693 MiB / 12288 MiB, utilizzo 0%;
- il binario esistente non va considerato build G129 valida perche' il tree e'
  sporco e l'ultimo esito build non e' stato acquisito dopo la compattazione.

## 3. Repository e working tree

### 3.1 Worktree DS4 autorevole

Path:

`C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2`

Stato congelato:

- branch: `feature/q1-0-resident-base`;
- HEAD: `dc52ec05ec2636a09fbf59fe9a21460e23621501`;
- subject: `feat: add G128 all-layer exact nested storage`;
- branch avanti di 13 commit rispetto al suo `origin` locale;
- modifiche non committate:
  - `ds4.c`;
  - `ds4_cuda.cu`;
  - `g7_measure.ps1`;
  - `G129_Q1_OPEN_DYNAMIC_PROMOTION_PROTOCOL.md`;
  - `g129_q1_open_dynamic_promotion_safety.ps1`;
  - `test_g129_q1_open_dynamic_promotion_contract.py`;
  - `test_g129_q1_open_dynamic_promotion_runtime.py`;
- diff osservato: circa 776 insertions e 169 deletions nei tre file tracked,
  oltre ai quattro file nuovi G129.

Non ripulire, resettare, fare checkout o ricreare questo lavoro. La nuova task
deve lavorare su questo esatto tree.

Il remote `origin` del worktree punta a un vecchio path locale Claude:

`C:\Users\imanu\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\ds4-win-work`

Quel path non e' oggi un remote GitHub utilizzabile. Non eseguire `git push`
alla cieca dal worktree DS4.

### 3.2 reap-loop

Path:

`C:\Users\imanu\source\repos\reap-loop`

Branch osservato:

`plan/0051-transport-gate-20260713`

Questo repo contiene il ledger storico, test WSL/pod, bakes, handoff e design.
E' sporco con modifiche utente e artefatti non committati. Non includerli in un
commit G129 senza audit file-per-file.

File canonici utili:

- `docs/EXPERIMENTS_LEDGER.md`;
- `docs/DS4_EXPERIMENT_LEDGER_20260710.md`;
- `docs/SOTA_ROADMAP.md`;
- `docs/HANDOFF_CODEX_20260713.md`;
- `docs/IQ1_S_WINDOWS_TEST_MATRIX_20260716.md`;
- `runs/ds4/20260716_g73/G73_CANONICAL_CONFIG.md`;
- `runs/ds4/20260709_exchange_matrix_combined.csv`;
- `runs/ds4/20260710_experiment_ledger/all_evidence_ledger.csv`.

### 3.3 moe-aggressive-commit

Path:

`C:\Users\imanu\source\repos\moe-aggressive-commit`

Branch:

`research/ds4-iq1-subbit-tier-planner`

HEAD osservato:

`f204753` - `research: bind Q1 sidecars to router identity`

Il branch e' pubblicato ma il tree locale e' sporco con ulteriore lavoro su
converter e test. Non committare file non propri.

Commit Q1_0 importanti gia' prodotti durante la ricerca:

- `c7bd895` planner dimensionale IQ1/sub-bit;
- `878f8a2` survey fork e candidato Q1_0 1.125 bpw;
- `0a38d94` smoke CPU Q1_0;
- `9251e5e` runtime step1;
- `2c32417` runtime step2;
- `ba11924` normalizzazione patch LF/ASCII;
- `0871821` blocker fixes;
- `31a02a0` runtime step3 dispatch;
- `48fcab4` converter sidecar sintetico;
- `a60e753` runtime sidecar planner;
- `f969faa` helper dequant/quant reference;
- `2dd1b0a` golden e hardening helper;
- `84b4ffb` provenance manifest;
- `f204753` binding sidecar a router identity.

## 4. Modelli e sidecar verificati

### IQ2 autorevole

- path: `C:\ds4-models\ds4-2bit.gguf`;
- bytes: `86720111488`;
- SHA-256:
  `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`;
- receipt: `C:\ds4-models\ds4-2bit.gguf.receipt.json`.

Il modello IQ2 resta la fonte autorevole per la qualita' e per ogni promozione
exact.

### Q1_0 completo, layer 0..42

- path: `C:\ds4-models\ds4-q1-layers0-42-derived.gguf`;
- bytes: `39048344416`;
- SHA-256:
  `05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207`;
- receipt:
  `C:\ds4-models\ds4-q1-layers0-42-derived.gguf.receipt.json`;
- derivazione: IQ2_XXS/Q2_K -> float -> Q1_0 reference;
- fonte quant/dequant: llama.cpp commit
  `635cdd5fcc5bdeb8ec2e108bb2a40acf62d9039b`.

Questa sidecar e' un esperimento di trasporto derivato da IQ2. Non e' una
quantizzazione addestrata o un claim di qualita' ottimale.

### Q1_0 layer 3..42

- path: `C:\ds4-models\ds4-q1-layers3-42-derived.gguf`;
- bytes: `36324143104`;
- SHA-256:
  `16baa8148550a340622c33d34d603be8a3e9de847b16ec8e2bfac56fa0cfea77`.

### Nested residual exact

Il filone G116-G128 usa una base nested e un residuo che ricostruiscono i byte
IQ2 autorevoli. Non e' lo stesso formato del Q1_0 lossy standard. Il sidecar
all-layer e' documentato in `G128_ALL_LAYER_NESTED_STORAGE_PROTOCOL.md` e ha
payload `32212254720` byte per i layer 3..42.

### Bakes K60/K75

Restano disponibili come fallback avanzato, non roadmap attiva.

K60:

- path originario Windows:
  `C:\ds4-models\ds4-2bit-k60-mass-full-decode.gguf`;
- il file puo' essere stato spostato su D: dall'utente;
- payload SHA:
  `5cb4bf69d7c6ef2aadfc8760069c3a7a89fc40504ce80b7b3735b10f9539e4b5`;
- pack SHA:
  `3b464ee43514c8caa841be61da70190a4a7ba3c760c22849d2d723e5da5b7d71`;
- retained layer 0..2 = 256, layer 3..42 = 154;
- allocated bytes dopo sparse punch: `58311376896`.

K75:

- path originario Windows:
  `C:\ds4-models\ds4-2bit-k75-mass-full-decode.gguf`;
- il file puo' essere stato spostato su D: dall'utente;
- payload SHA:
  `9b8f67ad4f69bfcd3a2369839c936371c8a433e53ed951582d0ad491c718aa3d`;
- pack SHA:
  `f1dbb64e1c8261928b56b4fa154238559444f70e8a1796875b22e42abf455dd2`;
- retained layer 0..2 = 256, layer 3..42 = 192;
- allocated bytes dopo sparse punch: `68975394816`.

Il fix del pack sparse e' nel commit reap-loop `679e367`. Il bug era che
`FSCTL_SET_SPARSE + truncate` lasciava il file completamente allocato; il fix
usa `FSCTL_SET_ZERO_DATA` sui complementi degli extent e riverifica il payload
SHA dopo il punch.

## 5. Obiettivo tecnico e intuizione del progetto

DeepSeek V4 Flash e' un MoE fine-grained. Molti expert condividono capacita'.
L'ipotesi di progetto e' che non serva tenere ogni expert nella forma piu'
costosa e piu' vicina alla GPU in ogni istante. Serve invece:

1. lasciare il router libero di selezionare tutti gli expert;
2. tenere in VRAM gli expert con domanda alta e persistente;
3. tenere in RAM una rappresentazione sufficiente per evitare SSD nel path
   critico;
4. lasciare su SSD solo l'autorita' IQ2 e i residui non promossi;
5. non pagare una lettura, un pin e un upload completo a ogni cold miss;
6. promuovere soltanto quando peso, frequenza, massa e recency giustificano il
   costo;
7. usare prediction solo dopo aver reso il trasporto abbastanza efficiente da
   permettere vero overlap.

REAP-LOOP nasce come controllo dinamico della larghezza/maschera. PACE nasce
come respiro o riallargamento quando il dominio degrada. SPEX nasce come
predittore di expert futuri. I test successivi hanno pero' mostrato che:

- una mask stretta puo' essere veloce ma distruggere qualita';
- una mask reimparata dal prefill puo' funzionare per una request ma non e' una
  soluzione full/open generale;
- la rotazione frequente costa molto H2D e cache churn;
- il prebreath n-gram arriva spesso dopo il danno;
- osservare massa e frequenza e' utile per residenza, ma non equivale a
  prevedere il router futuro;
- la soluzione generale deve separare selezione del router da storage class e
  representation.

## 6. Definizioni da non confondere

### K0, K23, K96

Nel filone REAP storico:

- K0 indicava nessuna potatura, cioe' tutti gli expert ammessi;
- K23 indicava keep-23 expert per layer, non il 23 percento;
- K96 indicava keep-96 expert per layer.

Questi K descrivono ampiezza della mask. Non descrivono cache VRAM.

### static32 in G73

`static32` non significa mask K32. Indica il budget di sostituzione del
mass-LFRU nella configurazione G72/G73. La mask/snapshot G73 contiene 4551
expert IQ2 request-scoped in RAM; la cache VRAM ne contiene 320.

### Cache320

`ExpertCacheN=320` significa 320 slot expert IQ2 in VRAM. Non limita il router
a 320 expert. In G73 il router era comunque limitato dallo snapshot chiuso; in
G129 il router deve restare full/open e la cache e' soltanto un tier.

### Massa

La massa usata dal tiering e' un accumulo decayed del valore assoluto del peso
router, insieme a frequency e recency. Nel codice corrente:

`entry.mass = entry.mass * 0.95 + abs(weight)`

Lo score mass-LFRU combina massa, `log1p(frequency)` e recency.

## 7. Cronologia tecnica condensata

### 7.1 Filone WSL, K23, PACE, breath e session learning

Risultati utili:

- K23 diretto raggiungeva localmente circa 3.0-3.4 t/s, ma spesso degenerava
  intorno a 100-120 token sul cyberpunk;
- warmup K0 seguito da freeze session-learned keep-23 ha prodotto alcuni output
  L3 a W50 e W130 sul pod con cache1024;
- il risultato non era monotono in W perche' il protocollo a due processi
  ri-prefillava un prefisso tagliato spesso a meta' CSS, innescando restart del
  documento;
- i restart `<!DOCTYPE>` erano in parte artefatti del confine di re-prefill,
  non una prova diretta che W alto peggiorasse la mask;
- rotate32 non era sempre migliore: in diversi test il costo di H2D/cache e il
  churn superavano il beneficio;
- n-gram breath scattava spesso troppo tardi;
- discesa a gradini non pagava nei test iniziali, ma molti protocolli non
  ricostruivano la mask in modo coerente a ogni gradino;
- cache128/256 erano troppo piccole per confrontarsi onestamente con vecchi
  run cache1024 su 3090;
- la coverage da sola non separava collasso e sopravvivenza;
- il prompt cyberpunk richiede spesso molto piu' di 800/2000 token per una
  pagina completa. Mancanza di `</html>` non implica automaticamente L0.

Errori metodologici di quella fase:

- troppi run `n=1` letti come tendenza;
- prompt, token budget, context, cache e build non sempre allineati;
- output troncati classificati come degenerati senza render;
- alcuni run manuali non registrati subito nel ledger;
- metriche browser e server mescolate;
- trace on/off non sempre esplicitato;
- static mask e cache size chiamate entrambe K in conversazione.

I dati storici restano in `reap-loop/runs/ds4/20260709_*` e nel ledger CSV.

### 7.2 Port Windows nativo e trasporto

Il port Windows e' partito da un path estremamente lento:

- decode circa 0.004 t/s;
- causa principale iniziale: `cudaMalloc` per fetch sotto WDDM, circa 42 ms per
  expert e indipendente dal disco;
- fix: slab compatto preallocato e direct stream degli expert;
- risultato iniziale corretto: circa 0.907 t/s steady;
- GPU restava stall-bound, con utilizzo basso e MoE GEMM circa 0.4 ms/layer;
- pread seriali e molti sync/launch piccoli restavano dominanti.

Finding importante:

- la window `cudaHostRegister(...Mapped...) + cudaHostGetDevicePointer` e'
  controproducente sulla 3060 12 GiB;
- la window mapped consuma budget VRAM proporzionale e affama la cache hot;
- 28 GiB mapped portarono thrash e circa 0.51 t/s;
- senza mapped window la cache hot rientrava e si misurava circa 0.91 t/s;
- la strada 0051 corretta usa host pinned non mapped, H2D esplicito e arena
  controllata.

Repository/fork studiati:

- antirez/ds4;
- hawkli-1994/ds4-win;
- chripell/ds4-rtx3090, commit `3fc5b21` e `5a854d2`;
- PocketMoE;
- Colibri e altri repo elencati in
  `docs/PORT_WINDOWS_NATIVE/06_MOE_MODEST_HW_RESEARCH.md`;
- llama.cpp e fork Q1_0/Bonsai per kernel e layout.

### 7.3 G35-G46: arena, mass-LFRU, direct resident e sync

Le leve mantenute:

- arena RAM pinned;
- prefill mass observation;
- bulk WRAP per riempire l'arena;
- cache expert in VRAM;
- mass-LFRU lento, non rotazione fisica ogni token;
- GPU-resident route handoff;
- eliminazione del default-stream sync.

G46, protocollo exact `n=3`, request-scoped closed:

- control: 4.4600 t/s mean;
- no-default-sync: 4.5633 t/s mean;
- delta: +2.32%;
- RAM H2D invariato: 71.5803 GiB per run;
- VRAM hits e RAM hits identici;
- zero SSD e zero failure.

Conclusione: togliere il sync e' corretto, ma sposta il costo nel worker-ready
wait. Non elimina il trasporto ripetuto RAM->VRAM.

### 7.4 G73: SOTA chiusa

Configurazione canonica:

- prompt cyberpunk hash
  `38f6ec5ee5403f59dd2418eb5d9a5a94a0f0da19df015060383bb1ae46003bb6`;
- temp0, nothink, 64 token, ctx256;
- arena IQ2 30 GiB;
- 4551 slot request-scoped prefill-ranked;
- WRAP source-parts, wave 4 GiB;
- cache VRAM 320;
- mass-LFRU clock430;
- replacement budget32;
- min frequency3;
- hysteresis1.25;
- no-default-sync;
- split-fused hit/miss;
- Q8-F16 cache off;
- REAP/SPEX prediction off.

Risultati validi `n=3`:

| Arm | Run t/s | Mean | Median |
|---|---|---:|---:|
| static32 | 4.62, 4.58, 4.63 | 4.610000 | 4.62 |
| static32 split-fused | 4.98, 4.97, 5.01 | 4.986667 | 4.98 |

Invarianti per run G73:

- route calls: 2752;
- VRAM hits: 5820;
- RAM hits: 10692;
- RAM H2D: 70.479492 GiB;
- promotions: 512;
- replacements: 192;
- budget skips: 4731;
- snapshot backing misses: 0;
- tier SSD bytes: 0;
- output SHA exact.

Interpretazione:

- G73 e' il miglior decode locale valido;
- split-fused riduce overhead per route e migliora decode dell'8.17%;
- il costo H2D non cambia;
- G73 non e' full/open perche' usa una snapshot request-scoped chiusa;
- non va usato come prova che una soluzione generale e' gia' lossless.

### 7.5 G55-G64: file QD e sparse bakes

Il filone ha verificato:

- QD8 file source e WRAP sono strutturalmente corretti;
- i bakes K60/K75 possono essere prodotti e caricati con guard manifest;
- il packer sparse Windows aveva un bug fisico poi corretto;
- rimuovere fisicamente expert dal file non risolve da solo la capacita'
  runtime se si costruisce comunque la stessa arena 30 GiB;
- i safety ctx8192 G46 full e K60 fallirono entrambi durante WRAP per memoria
  disponibile circa 0.25 GiB, prima del primo token;
- i bakes non battono G73 e non supportano cambio dominio dinamico.

Decisione: mantenere K60/K75 come fallback avanzato/TODO, non investire ora
nell'ottimizzazione del bake ridotto.

### 7.6 G75-G106: IQ1_S e promozione 5+1

Finding microbenchmark G75:

- expert corrente: 7,077,888 byte;
- IQ1_S: 4,915,200 byte;
- riduzione byte: 30.55%;
- H2D pinned corrente: 0.270876667 ms;
- H2D pinned IQ1_S: 0.188821000 ms;
- riduzione H2D: 30.30%;
- kernel IQ1_S x Q8_K validato su tolleranza del test.

Il path 5+1 voleva:

- cinque route exact IQ2;
- una route cold/lowest-weight IQ1;
- promozione exact solo dopo conferma;
- nessun direct SSD->VRAM.

Risultati chiave:

- G86 IQ1 RAM cache era piu' lento del control nonostante 87.71% RAM hits;
- G94 GPU planner miglioro' il breve short gate da 2.073 a 2.223 server t/s,
  ma senza L0-L3 long-form;
- G98 promuoveva troppo e crollava a 0.323 server t/s contro 0.68 control;
- G100 isolo' i gate: touch2 + budget/window riduceva le letture IQ2 SSD di
  circa 94.87% nel breve structural sweep;
- G101 combined ridusse promozioni 5804 -> 64 e byte di circa 98.9%, con
  0.663333 vs 0.52 server t/s, ma fixed-order e senza quality claim;
- G103, composizione G73 + un cold IQ1, fece 2.7033 t/s contro G73 4.9867:
  negativo netto;
- full resident IQ1_S non entrava in modo sano nel budget RAM su 64 GiB senza
  paging/guard failure.

Conclusione: IQ1_S dimostra che meno byte aiutano H2D, ma sidecar da SSD e
micro-promozioni per-token peggiorano. La logica di gate e' riutilizzabile; il
formato IQ1_S completo non e' il cold tier globale giusto per questa macchina.

### 7.7 G108-G115: Q1_0 puro e mix

Q1_0 layout:

- type 41;
- 128 pesi per block;
- 18 byte per block;
- 1.125 bit per weight effettivi;
- expert completo circa 3.375 MiB;
- layer 3..42, tutti gli expert: 33.75 GiB;
- sidecar 0..42 reale: circa 36.4 GiB.

G108:

- env-off exact safety;
- dato descrittivo 8.3 t/s;
- `n=1`, quindi non SOTA.

G112-G114, `n=1`:

| Run | Mix medio | Server/steady | SSD | Grade |
|---|---|---:|---:|---:|
| G112 pure Q1 | 0 IQ2 + 6 Q1 | 6.76 / 6.77 | 0 | L0 |
| G113 seed7 | 1.23 IQ2 + 4.77 Q1 | 6.21 / 6.94 | 0 | L1 |
| G113 seed8 | 1.15 IQ2 + 4.85 Q1 | 6.22 / 6.96 | 0 | L1 |
| G114 global320 | 1.34 IQ2 + 4.66 Q1 | 6.22 / 6.90-7.05 | 0 | L1 |

Conclusione:

- il Q1 resident elimina i miss SSD e mostra un ceiling >6 t/s;
- la GPU appare molto piu' continua e meno seghettata;
- la qualita' crolla quando troppi lane usano Q1;
- un seed IQ2 di 280-320 non basta se il resto resta Q1;
- l'idea 5+1 e' qualitativamente diversa dal mix accidentale 1.3+4.7;
- non bisogna confondere il ceiling di trasporto con SOTA.

### 7.8 G116-G128: nested residual exact full/open

Per recuperare qualita' senza duplicare una copia Q1 e una copia IQ2 completa,
e' stata introdotta una rappresentazione nested:

- base resident;
- residual exact;
- base + residual ricostruiscono i byte IQ2 autorevoli;
- il router resta full/open;
- nessun expert viene sostituito.

Punti misurati:

- G123 full/open IQ2 control: circa 1.65 t/s mean `n=3`;
- G126 GPU join: 1.57 t/s contro CPU join 1.153333, exact `n=3`;
- G127 residual cache: candidate 1.586667 vs control 1.573333 server t/s,
  +0.847%, exact e pulito `n=3`;
- G127 riduce residual preads 1261 -> 869 e evita 1,233,125,376 byte per
  run, ma il guadagno t/s e' marginale;
- G128 implementa storage all-layer exact con base pinned/pageable e residual
  cache pageable; e' protocollo/storage, non un nuovo SOTA.

Perche' siamo ancora lenti:

- il residual/join exact conserva qualita' ma trasferisce base + residual;
- restano D2H selection/metadata in alcuni path;
- il residual miss legge ancora da file;
- molti piccoli launch/sync WDDM restano;
- il full/open exact paga molto piu' trasporto del closed G73.

### 7.9 G129: sintesi attuale

G129 torna al Q1_0 standard come fallback completo, ma aggiunge promozione IQ2
controllata e router full/open.

Obiettivo:

- full/open sempre;
- Q1_0 disponibile per tutti gli expert;
- IQ2 exact in RAM e VRAM per gli expert realmente caldi;
- current token servito da Q1 quando l'exact manca;
- promozione exact soltanto per token futuri;
- target >6 server t/s, L2/L3, `n>=3`.

Stato: codice dirty, protocollo e runner presenti, nessun receipt G129 runtime.

## 8. Ledger SOTA e risultati principali

### 8.1 Matrice sintetica

| ID | Scope | Evidenza | Decode/server | Qualita | Stato |
|---|---|---|---:|---|---|
| WSL K23 storico | closed mask | run eterogenei | 3.03-3.39 | spesso L0 <=800 | storico, non SOTA Windows |
| G46 | request-scoped closed exact | n=3 | 4.5633 | prefix exact, non long L3 | positivo transport |
| G73 | request-scoped closed exact | n=3 | 4.986667 | prefix exact, protocollo valido | SOTA locale chiusa |
| G103 | G73 + IQ1 cold | n=3 | 2.7033 | no claim long | negativo |
| G108 | env-off exact | n=1 | 8.3 | no verdict | safety, non SOTA |
| G112 | full Q1 resident open selection | n=1 | 6.76 | L0 | ceiling transport |
| G113/G114 | Q1 + seed IQ2 | n=1 | 6.21 avg, ~6.9 steady | L1 | negativo qualita |
| G123 | full/open IQ2 exact | n=3 | 1.65 | exact surface | baseline full/open |
| G126 | full/open nested GPU join | n=3 | 1.57 | exact | positivo vs CPU, sotto G123 |
| G127 | full/open residual cache | n=3 | 1.586667 | exact | +0.847%, marginale |
| G128 | full/open exact storage | protocollo/safety | n/a | exact design | non SOTA |
| G129 | full/open Q1 + IQ2 promotion | non eseguito | n/a | n/a | WIP attivo |

### 8.2 Cosa significa SOTA oggi

SOTA chiusa locale:

- G73 `4.986667 t/s` mean, `4.98` median, exact `n=3`.

SOTA full/open exact misurata:

- il controllo G123 e' circa `1.65 t/s` mean `n=3`;
- G126/G127 non lo superano in modo sostanziale.

Ceiling di trasporto lossy:

- Q1_0 resident circa `6.8-7.0 t/s` steady in safety n=1;
- non e' un risultato di qualita'.

Target G129:

- superare `6.0 t/s` server decode;
- long-form almeno L2, preferibilmente median L3;
- `n>=3` clean;
- full/open e zero transizioni vietate.

## 9. Architettura G73, per confronto

### Prefill

1. Il router full osserva il prompt.
2. Si accumula massa per `(layer, expert)`.
3. Si costruisce una snapshot IQ2 request-scoped da 4551 expert nel budget
   arena30.
4. La snapshot e' pubblicata con WRAP.
5. Il decode usa soltanto quella snapshot: per questo lo scope e' closed.

### RAM

- 30 GiB pinned;
- 4551 expert IQ2 exact;
- scelti dal prefill della request;
- nessun backing miss nei run G73 validi.

### VRAM

- 320 slot IQ2;
- promossi e sostituiti da mass-LFRU;
- cache320, non mask320.

### Promozione

- score mass-LFRU;
- clock430;
- budget32;
- min frequency3;
- hysteresis1.25;
- 512 promotions e 192 replacements per run canonico.

### Predizione

G73 non usa SPEX. Usa il passato osservato nella stessa request:

- massa del prefill per il set iniziale;
- massa/frequency/recency del decode per la cache.

E' request learning reattivo, non previsione del prossimo expert.

## 10. Architettura G129 corrente

### 10.1 Router

- full/open su tutti i token;
- nessun `DS4_REAP_MASK_FILE`;
- nessuna mask baked;
- nessuna static domain mask;
- nessun bias che escluda expert;
- l'expert ID selezionato dal router non cambia in base alla residenza.

### 10.2 Chi resta in VRAM

La cache device contiene fino a 320 expert nella rappresentazione IQ2 esatta.

Seed iniziale:

- ranking dalla massa full-router osservata nel prefill;
- 320 slot globali;
- floor di 4 expert per ciascun routed layer;
- i rimanenti slot vanno ai maggiori score globali;
- source IQ2 e' il modello autorevole.

Durante decode:

- gli hit IQ2 protetti vengono usati direttamente;
- nuovi IQ2 gia' presenti in RAM possono competere per la VRAM;
- la sostituzione usa mass-LFRU;
- min frequency3;
- clock430;
- budget di prima prova G129: 64 replacement per epoch;
- hysteresis1.25;
- un expert promosso da Q1 deve aspettare almeno la call successiva.

### 10.3 Chi resta in RAM pinned

Arena Q1_0:

- budget pinned 24.5 GiB;
- contiene un prefisso fisico degli 11008 slot Q1_0 immutabili;
- non viene ruotata a ogni token;
- il resto della base Q1 resta in pageable overflow.

Arena IQ2 exact:

- 5.5 GiB pinned;
- contiene tutti i 768 expert dei primi tre layer hash-routed piu' un pool
  open-router da 64 slot;
- il pool da 64 e' la probation/warm arena per expert IQ2 dei layer routed;
- gli slot sono rimpiazzabili con policy controllata;
- una copia RAM puo' essere reclaimata se la copia IQ2 resta protetta in VRAM.

Totale pinned richiesto dalle due arene: 30 GiB.

### 10.4 Chi resta in RAM pageable

- tutti gli slot Q1_0 che non entrano nei 24.5 GiB pinned;
- l'intera base Q1_0 deve comunque essere presente, non lazy da SSD;
- la sidecar 0..42 misura circa 36.4 GiB, quindi l'overflow e' circa 11.9
  GiB oltre il prefisso pinned;
- pageable non significa mapped zero-copy e non significa SSD.

### 10.5 Cosa resta su SSD

- modello IQ2 autorevole completo;
- sidecar Q1_0 come fonte di bootstrap, ma dopo bootstrap ogni expert Q1 deve
  essere in RAM;
- IQ2 cold non ammessi nell'arena probation;
- sidecar/residui di altri esperimenti non attivi.

SSD non deve essere nel path current-token Q1->VRAM. Puo' alimentare la
probation IQ2 dopo una decisione di admission.

### 10.6 Lifecycle di una route G129

Per ogni expert selezionato:

1. se esiste IQ2 protetto in VRAM, usa IQ2 direttamente;
2. altrimenti, se esiste IQ2 exact nella RAM snapshot/probation, usa IQ2 e
   consenti a mass-LFRU di candidarlo alla VRAM;
3. altrimenti usa Q1_0 residente per il token corrente;
4. contabilizza una sola volta peso, massa, frequency e recency;
5. completa/accoda il lavoro Q1 e il join dell'output corrente;
6. applica il gate di admission IQ2;
7. se ammesso, legge IQ2 autorevole da SSD e lo stagea in RAM probation;
8. imposta `vram_eligible_after_call = call_tick + 1`;
9. solo da una call successiva puo' salire in VRAM;
10. se si osserva direct SSD->VRAM current-token, il run deve fallire.

### 10.7 Gate di promozione G129 congelato

Prima prova causale:

- minimum touches: 2;
- minimum absolute router weight: 0.02;
- minimum mass: 0;
- request budget: 64 promozioni;
- window: 40 routed-layer calls;
- window budget: 1 promozione;
- probation slots: 64.

Questi parametri vengono da G100/G101, dove la combinazione ridusse quasi tutto
il churn IQ2 SSD. Non sono dichiarati tuning finale.

### 10.8 Predizione G129

Non c'e' ancora prediction vera.

Attivo:

- full-router prefill mass per seed320;
- massa, peso, frequenza e recency osservati nel decode;
- mass-LFRU reattivo.

Non attivo:

- SPEX hidden topK/prefetch;
- Markov/predictive routing nel path G129;
- router duplicato;
- lookahead sugli ultimi token;
- prompt decomposition;
- oracle expert injection.

SPEX resta observe-only finche' il trasporto G129 non supera gate di qualita' e
throughput. Un futuro test SPEX deve essere un braccio separato per misurare:

- precision/recall delle promozioni;
- miss evitati;
- byte sprecati;
- cancellation rate;
- anticipo in token;
- impatto L0-L3.

## 11. Stato esatto dell'implementazione G129 dirty

### Cosa e' stato aggiunto

In `ds4.c`:

- env e contratti separati per Q1_0 dynamic promotion;
- richiesta resident + dual arena;
- divieto di combinazioni snapshot/sparse/cold-one incompatibili;
- full layer range 0..42 per la sidecar completa.

In `ds4_cuda.cu`:

- due arene disgiunte:
  - `g_q1_0_dynamic_arena` per base Q1 completa;
  - `g_dynamic_arena` per IQ2 exact probation/warm;
- Q1 resident pinned + pageable overflow;
- resolver a quattro classi:
  - `IQ2_VRAM`;
  - `IQ2_SNAPSHOT_RAM`;
  - `IQ2_TIER_RAM`;
  - `Q1_RESIDENT`;
- current-token Q1 prima dello stage exact;
- promozione RAM-first;
- guard `vram_eligible_after_call`;
- counter Q1-specific;
- counter `q1_0_next_call_guards`;
- seed VRAM direttamente dal modello IQ2 autorevole usando la massa prefill;
- telemetry pinned/pageable/total slot e bytes.

In `g7_measure.ps1`:

- switch `Q1_0DynamicPromotion`;
- parametri Q1 promotion separati da IQ1_S legacy;
- parsing bootstrap e telemetry;
- expected resident entries separato;
- contract full/open e no-mask;
- output JSON per gate G129.

Nuovi file:

- `G129_Q1_OPEN_DYNAMIC_PROMOTION_PROTOCOL.md`;
- `g129_q1_open_dynamic_promotion_safety.ps1`;
- `test_g129_q1_open_dynamic_promotion_contract.py`;
- `test_g129_q1_open_dynamic_promotion_runtime.py`.

### Fix gia' fatti durante la revisione

- rimosso `Q1_0SnapshotBacking` dal runner G129, incompatibile con dual arena;
- rimosso `Iq1Promotion` legacy dal runner;
- separata la richiesta promotion Q1 dalla sidecar IQ1_S;
- layer range corretto da default 3..42 a 0..42;
- expected resident entries corretto a 11008;
- log `total=` ambiguo rinominato `total_slots` e `total_bytes`;
- harness ora controlla pinned + pageable = total;
- runner verifica full/open e assenza mask;
- introdotto counter next-call guard;
- evitato doppio accounting dell'osservazione Q1;
- seed320 prende IQ2 autorevole, non Q1;
- guard per zero current-token IQ2 SSD.

### Punto ancora pendente noto

In `g7_measure.ps1`, cercare la chiamata a
`Read-G7Q1_0MixedTelemetry`. Il parametro `-Required` risultava ancora basato
solo su snapshot/mixed-cold. Deve includere anche `$Q1_0DynamicPromotion`, ad
esempio:

```powershell
-Required ([bool]($Q1_0SnapshotBacking -or
                  $Q1_0MixedColdOne -or
                  $Q1_0DynamicPromotion))
```

Verificare il codice corrente prima di applicare: se e' gia' corretto, non
duplicare il fix.

### Stato test/build al congelamento

- i due test statici G129 erano passati prima delle ultime modifiche C/log;
- parsing PowerShell scriptblock era passato;
- `git diff --check` era passato salvo warning LF/CRLF;
- un build Release e' stato avviato dopo le ultime modifiche;
- l'esito finale del build non e' stato recuperato dopo la compattazione;
- il binario esistente SHA
  `8CF441AFF6FB80B53D008BB8DA1B10A9AE0809814C938B70BC102A289C358060`
  non va trattato come build G129 verificata;
- nessun safety G129 runtime e' stato eseguito;
- nessun receipt G129 esiste.

## 12. Errori commessi e contromisure

### 12.1 Confondere scope diversi

Errore: chiamare G73 full-model/full-router perche' usa il file completo. In
realta' G73 pubblica una snapshot request-scoped chiusa.

Contromisura: ogni tabella deve avere una colonna `router_scope` con valori
`closed/request-scoped` o `full/open`.

### 12.2 Chiamare SOTA un outlier n=1

Errore: discutere 8.3 t/s G108 o 6.8-7.0 Q1 come nuovo SOTA.

Contromisura: etichettare ogni dato come `VALID N>=3`, `SAFETY N=1`,
`CONTAMINATED`, `NEGATIVE`, `PLANNED`.

### 12.3 Ottimizzare il formato sbagliato

Errore: aspettarsi che un mix Q1-dominant recuperasse qualita' solo con 320
expert IQ2 in VRAM.

Contromisura: misurare la frazione reale di route IQ2/Q1. G113/G114 erano
circa 1.2 IQ2 + 4.8 Q1, non 5+1.

### 12.4 Promuovere ogni cold route

Errore: G98 promuoveva migliaia di expert IQ2, leggendo decine di GiB da SSD.

Contromisura: gate touch/weight/mass e budget/window. G100/G101 hanno mostrato
che touch2 + window budget abbattono il churn.

### 12.5 Usare SSD come cold path per-token

Errore: IQ1_S/Q1 da SSD per la route corrente riduce byte per expert ma resta
un miss seriale e peggiora il decode.

Contromisura: la rappresentazione fallback completa deve essere gia' in RAM.

### 12.6 Mapped host memory sulla 3060

Errore: trattare la mapped window come zero-copy gratis.

Contromisura: niente `cudaHostGetDevicePointer` per l'arena grande. Usare RAM
pinned non mapped e H2D esplicito.

### 12.7 Non verificare residenza fisica

Errore: dichiarare full resident da byte richiesti senza guardare available
memory, paging e working set.

Contromisura: fail-closed runtime monitor, pinned/pageable counters, hard RAM
floor, paging guard e quiescenza.

### 12.8 Leggere output incompleto come degenerato

Errore: un HTML senza form o `</html>` entro 800/2000 token veniva talvolta
classificato come rotto anche quando il CSS era coerente.

Contromisura: render e grading L0-L3; stop su `</html>` quando appropriato;
context >= output budget + prompt; distinguere budget exhaustion da loop.

### 12.9 Re-prefill a meta' struttura

Errore: warmup/freeze in due processi tagliava spesso una dichiarazione CSS a
meta' e ri-presentava il prompt, causando document restart.

Contromisura: single-process same-KV oppure freeze su boundary sicuri; non
attribuire automaticamente il restart alla mask.

### 12.10 Ledger non aggiornato in tempo reale

Errore: run manuali e parametri restavano nella chat; dopo compattazione si
perdeva il nesso fra risultato e configurazione.

Contromisura: ogni runner crea manifest e receipt immutabile; il ledger viene
aggiornato subito dopo ogni run, positivo o negativo.

### 12.11 Troppe patch incrementali senza audit architetturale

Errore: aggiungere fix locali prima di rileggere lifecycle, ownership delle
arene e invarianti produce contraddizioni come snapshot+dual arena.

Contromisura: protocollo congelato prima del codice; audit di tutte le call
path; poi una patch coerente; static tests sui contratti negativi.

### 12.12 Build e binary provenance stale

Errore: usare un binario presente come prova che il tree corrente e'
compilato.

Contromisura: build manifest, input fingerprint e executable SHA devono
corrispondere al tree esatto prima del safety.

### 12.13 Contaminazione macchina

Errori osservati:

- download R2/rclone;
- Disk D: al 90-100%;
- ScheduledDefrag;
- copie rete;
- processi DS4 appesi;
- file cache/working set dopo run precedenti;
- memoria Windows insufficiente;
- paging durante WRAP.

Contromisura: preflight, cooldown, disk queue, GPU/VRAM, process list,
available RAM, runtime contamination samples e post-run cleanup.

### 12.14 Sandbox e permessi

Errore operativo: task rimaste ferme in attesa di approval non visibile da
mobile.

Stato corrente: approval policy `never`, writable roots gia' includono i repo
principali e `C:\ds4-models`. Non chiedere escalation. Usare solo path
autorizzati e command prefix gia' consentiti.

## 13. Successi consolidati

- Port Windows da 0.004 a circa 0.907 t/s eliminando malloc per miss.
- Dimostrato che mapped window grande e' negativa sulla 3060.
- Arena pinned e WRAP funzionanti con guard di memoria.
- G46 no-default-sync exact `n=3`, +2.32%.
- G73 split-fused exact `n=3`, +8.17%, 4.986667 t/s.
- File QD8 e sparse loader guard validati.
- Packer sparse Windows corretto con hole reali e hash post-punch.
- IQ1_S microbenchmark: -30.55% byte e -30.30% H2D.
- GPU planner IQ1 breve: +7.23% decode G94, con limiti di scope.
- Gate promotion G100/G101 abbattono churn di circa 95-99%.
- Q1_0 runtime type41, kernel, sidecar, converter e provenance portati.
- Q1_0 full resident elimina SSD e mostra ceiling ~7 t/s.
- Nested residual ricostruisce IQ2 exact e mantiene router full/open.
- GPU join nested migliora nettamente il CPU join.
- Residual cache G127 riduce pread e produce piccolo guadagno exact.
- G128 supporta all-layer base/residual pinned + pageable.
- G129 designa due arene disgiunte e promozione next-call, correggendo gli
  errori concettuali dei mix precedenti.

## 14. Cosa non e' ancora risolto

1. Qualita' G129 non misurata.
2. Build G129 corrente non certificata.
3. Full/open >6 t/s con L2/L3 non dimostrato.
4. Base Q1_0 standard puo' degradare il token corrente prima della promozione.
5. La probation IQ2 legge ancora SSD dopo l'admission.
6. I 64 slot probation possono essere troppo pochi o troppo costosi; serve
   misura, non tuning intuitivo.
7. Il path misto fa ancora D2H selected/weights in alcuni punti.
8. G130, packed heterogeneous handoff e riduzione dei join non sono composti
   con G129.
9. SPEX non e' consumer operativo nel percorso finale.
10. Prefill/TTFT resta lungo, spesso 50+ secondi nei protocolli G73/Q1.
11. La UI e Scope non sono il focus di questo handoff e non devono interferire
    con i benchmark.
12. Il push Git CLI locale e' bloccato da credenziali/ACL; usare il connector
    GitHub o ripristinare esplicitamente auth in una task separata.

## 15. Procedura esatta di ripresa G129

### Step 0 - Non perdere il tree

```powershell
cd C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2
git status --short --branch
git diff --stat
git diff -- ds4.c ds4_cuda.cu g7_measure.ps1
```

Non stashare e non resettare.

### Step 1 - Audit del fix pendente harness

```powershell
rg -n "Read-G7Q1_0MixedTelemetry|Q1_0DynamicPromotion" g7_measure.ps1
```

Assicurarsi che la telemetry mixed sia required anche quando il solo braccio
Q1 dynamic promotion e' attivo.

### Step 2 - Static tests

```powershell
python .\test_g129_q1_open_dynamic_promotion_contract.py
python .\test_g129_q1_open_dynamic_promotion_runtime.py
```

Poi parsing PowerShell:

```powershell
$null = [scriptblock]::Create((Get-Content -Raw .\g7_measure.ps1))
$null = [scriptblock]::Create((Get-Content -Raw .\g129_q1_open_dynamic_promotion_safety.ps1))
```

E:

```powershell
git diff --check
```

### Step 3 - WhatIf runner

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\g129_q1_open_dynamic_promotion_safety.ps1 `
  -Q1_0ExpertSidecar C:\ds4-models\ds4-q1-layers0-42-derived.gguf `
  -ExpectedQ1_0ExpertSidecarSHA256 05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207 `
  -ExpectedQ1_0ExpertSidecarBytes 39048344416 `
  -WhatIf
```

Verificare nel JSON:

- router full/open;
- snapshot backing false;
- dual arena true;
- pageable overflow true;
- layers0..42;
- expected entries11008;
- no IQ1_S legacy promotion;
- seed320/floor4;
- promotion 2/.02/64/40/1.

### Step 4 - Build Release e ctest

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\g7_build.ps1 `
  -Configuration Release
```

Non riusare l'esito perso del build precedente. Registrare:

- HEAD;
- dirty source hashes;
- input fingerprint;
- build manifest SHA;
- executable SHA.

Poi:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe" `
  --test-dir build -C Release --output-on-failure
```

### Step 5 - Quiescenza

Controllare:

- nessun ds4/server;
- nessun rclone/copia/defrag;
- GPU baseline;
- D/C disk queue;
- RAM disponibile sufficiente;
- nessun paging attivo;
- sidecar e model receipt validi.

G129 richiede circa:

- 36.4 GiB Q1 host resident;
- 5.5 GiB IQ2 arena;
- runtime/OS/headroom.

Se il preflight non passa, non ridurre arbitrariamente le arene: registrare il
capacity blocker.

### Step 6 - Safety strutturale n=1

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\g129_q1_open_dynamic_promotion_safety.ps1 `
  -Q1_0ExpertSidecar C:\ds4-models\ds4-q1-layers0-42-derived.gguf `
  -ExpectedQ1_0ExpertSidecarSHA256 05040393f5e94bf054a593e4d2d021ff44a6f446f2328a75e4f833a1fbe20207 `
  -ExpectedQ1_0ExpertSidecarBytes 39048344416
```

Gate obbligatori:

- exit0;
- full/open e no mask;
- 11008 Q1 resident;
- pinned+pageable=total;
- Q1 routes >0;
- IQ2 VRAM/RAM routes >0;
- promotion attempts/successes >0;
- next-call guards == stage successes;
- failures0;
- IQ2 current-token SSD bytes0;
- forbidden SSD->VRAM0;
- VRAM torna baseline.

Non usare t/s del safety come claim.

### Step 7 - Long safety qualitativo

Preparare un runner long con:

- prompt cyberpunk canonico;
- temp0, nothink;
- context sufficiente, preferibilmente8192;
- max token coerente con pagina completa, 2000-4000;
- stop `</html>` solo come completion stop, non grader;
- stessa configurazione G129;
- un processo.

Renderizzare e assegnare L0-L3 immediatamente.

- L0/L1: fermare G129, registrare negativo e tornare al design;
- L2/L3: procedere a n>=3.

### Step 8 - Campagna n>=3

Usare processi indipendenti e ordine bilanciato. Confronti ammessi:

1. G129 full/open contro full/open exact G123/current-build control;
2. solo come riferimento storico, G129 contro G73 closed, con colonna scope;
3. eventuale arm senza promotion per isolare il valore del gate.

Metriche:

- server decode t/s mean/median;
- E2E t/s;
- TTFT;
- prefill/WRAP/seed;
- RAM pinned/pageable;
- VRAM hits;
- Q1 routes;
- IQ2 RAM/VRAM routes;
- promotion bytes/time;
- direct/forbidden counters;
- output L0-L3 per membro;
- median quality;
- contamination.

Successo preregistrato G129:

- mean server decode >6.0 t/s;
- median almeno L3 per SOTA forte, almeno L2 per continuare;
- n>=3 clean;
- full/open;
- zero failure/forbidden.

### Step 9 - Commit separati

Se i gate passano:

1. commit runtime `ds4.c`/`ds4_cuda.cu`;
2. commit harness/runner/tests;
3. commit safety receipt;
4. commit n>=3 results/ledger;
5. push sul remote GitHub corretto, non sul vecchio path Claude.

Se i gate falliscono:

1. commit codice solo se il meccanismo strutturale e' corretto e utile;
2. commit risultato negativo e protocol error;
3. non promuovere a default/SOTA.

## 16. Roadmap dopo G129

### Se G129 passa qualita' ma resta <6 t/s

Priorita':

1. G130: eliminare D2H selected/weights e duplicate launch;
2. packed heterogeneous route handoff;
3. separare hit VRAM da miss RAM e fare un solo join;
4. batch-union e waved prefill;
5. residual/promotion prefetch cancellabile;
6. riduzione sync e GPU-resident handoff;
7. ottimizzare pageable->pinned staging senza mapped window;
8. misurare cache320 vs headroom, non aumentarla alla cieca.

### Se G129 fallisce qualita'

Non aggiungere subito SPEX. Le opzioni architetturali sono:

1. true 5+1: solo lowest-weight lane Q1, cinque lane exact;
2. nested base-only lane + residual promotion;
3. Q1 + residual incrementale per ricostruire IQ2;
4. soglia piu' conservativa che usa Q1 solo per peso molto basso;
5. rollback/correction del token corrente se la lane Q1 causa divergenza.

Ogni opzione deve preservare expert ID e router full/open.

### SPEX

SPEX entra solo quando il costo di staging puo' essere overlappato. Primo test:

- observe-only vs active;
- prediction horizon2/4/8;
- max promoted1/2/4;
- confrontare precisione, recall, miss evitati e byte sprecati;
- nessun training su prompt o mask;
- score sulla predizione del peso/massa futura, non sulla sola massa passata;
- prefetched expert prima in RAM, non direttamente in VRAM.

### Quantizzazione dinamica e 1 bit

TODO consolidati:

- Q1_0 cold base residente;
- residuo exact separato;
- promozione Q1_0 -> Q1+residuo/IQ2;
- doppia copia SSD 1-bit e 2-bit solo se il residuo e' indicizzato e non
  duplica inutilmente RAM;
- mixed quant per expert cold;
- top expert per layer a int4 e altri int2: solo dopo misura di qualita';
- sub-1bit custom non prima del Q1_0 production gate;
- Bonsai/Q1_0 e' riferimento layout/kernel, non un drop-in di qualita'.

### Prefill

TODO:

- batch-union per expert unico per layer/chunk;
- waved prefill;
- prompt-derived bulk seed;
- scomposizione semantica del prompt in sotto-task come esperimento separato;
- confronto trace on/off;
- TTFT deve includere WRAP, seed e prompt, ma va anche scomposto.

### Bake

- K60/K75 restano fallback;
- K8 autocontenuto resta TODO lontano;
- ibridi full/partial layer possono essere studiati, ma non prima della roadmap
  dinamica full/open;
- bake statico non risolve cambio dominio tra turni di chat.

## 17. Prompt consigliato per la nuova task

```text
Leggi prima e integralmente:
C:\Users\imanu\source\repos\reap-loop\docs\HANDOFF_CODEX_20260719_G129_FRESH_TASK.md

Poi lavora esclusivamente sul worktree autorevole:
C:\Users\imanu\Documents\Codex\2026-07-07\cia\work\ds4-win-publish-g126-20260718-v2

Non resettare, stashare o ricreare le modifiche G129 non committate. Verifica
prima lo stato Git e il fix pendente Read-G7Q1_0MixedTelemetry. Segui la
procedura fail-closed del paragrafo 15: static tests, WhatIf, build Release con
fingerprint, ctest, quiescenza, safety n=1, long grading L0-L3, e soltanto se
L2/L3 una campagna n>=3. Il router deve restare full/open; nessuna static mask.
Non usare G73 come baseline full/open e non chiamare SOTA i dati n=1 Q1.
Aggiorna ledger e committa runtime, runner e risultati separatamente. Non
sovrapporre processi DS4 e non chiedere permessi se il profilo corrente gia'
consente l'operazione.
```

## 18. Checklist finale per chi prende in carico

- [ ] Ho letto tutto questo handoff.
- [ ] Ho verificato branch/HEAD/diff senza modificarli.
- [ ] Ho distinto G73 closed da G129 full/open.
- [ ] Ho verificato model e sidecar receipt.
- [ ] Ho chiuso il fix telemetry Required.
- [ ] I due test statici passano.
- [ ] I due script PowerShell parsano.
- [ ] Il WhatIf espone la configurazione congelata.
- [ ] Build manifest/fingerprint/exe SHA sono nuovi e coerenti.
- [ ] Ctest passa.
- [ ] Macchina quiescente e RAM sufficiente.
- [ ] Safety n=1 passa tutti i counter, senza claim t/s.
- [ ] Long output e' stato renderizzato e gradato L0-L3.
- [ ] Solo L2/L3 ha autorizzato n>=3.
- [ ] Ogni risultato e' entrato nel ledger con scope/evidence.
- [ ] I commit non includono file sporchi non propri.
- [ ] Il push usa un remote GitHub reale.

## 19. Nota finale di onesta'

Il progetto ha trovato un ceiling di trasporto interessante e una SOTA chiusa
solida, ma non ha ancora dimostrato la combinazione desiderata:

`full/open + >6 t/s + L2/L3 + n>=3`.

G129 e' il primo tentativo che mette insieme, in modo esplicito e misurabile:

- router libero;
- fallback completo residente;
- exact IQ2 nei tier caldi;
- promozione RAM-first e next-call;
- mass-LFRU;
- guard di qualita' e provenienza.

Va quindi testato con disciplina, non dato per riuscito e non sostituito da
un'altra serie di patch prima di aver osservato il suo primo output lungo.

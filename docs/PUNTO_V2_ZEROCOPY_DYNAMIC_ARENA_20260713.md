# Punto V2 zero-copy e arena dinamica - 2026-07-13

Questo documento sostituisce, per il Thread 1, la fotografia contenuta in
`HANDOFF_CODEX_20260713.md`. L'handoff resta valido per protocollo, Thread 2 e
Thread 3, ma la diagnosi iniziale della patch 0050 e' stata superata dai test
descritti qui.

## Obiettivo invariato

Eliminare dal critical path di ogni fetch expert il pedaggio
`pread -> staging pinned -> H2D -> sync`, mantenendo il GEMM in VRAM. Il target
funzionale finale non e' un bake immutabile: e' una finestra RAM/DMA che possa
cambiare tra interazioni della stessa chat, per esempio coding al turno N e
storia romana al turno N+1.

Sono tre livelli distinti:

1. **selection mask**: expert ammessi dal router;
2. **pinned host arena**: expert immediatamente copiabili in DMA;
3. **VRAM cache**: expert gia' pronti per il GEMM.

La RAM pinned puo' restare allocata mentre cambia l'identita' degli expert
contenuti nei suoi slot. Il cambio dominio deve conservare l'intersezione tra
vecchia e nuova finestra, caricare in batch soltanto il delta tramite WRAP e
pubblicare atomicamente la nuova mask solo quando il delta e' pronto.

## Stato del codice

- Repo di ricerca: `C:\Users\imanu\source\repos\reap-loop`
- Branch: `spex-predictive-mask-study-2026-07-12`
- HEAD repo al consolidamento: `adf3de47a576bb8f653ef847daa4f15f026993ef`
- Worktree DS4 WSL: `/root/ds4-v2-work`
- Base DS4: `da0b3f63d7cc87c1f11c3c876fb57de3e0caca50`
- Sorgenti WIP modificati: `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`
- Binario corrente md5: `7ca6f27d6ce251f944be5c079dc471de`
- Ultima build passata: `/root/ds4-v2-work/build_0050i.log`
- Nessun `ds4-server` era attivo al consolidamento.

La patch 0050 corrente e' ancora una registrazione statica dei range della mmap
scelti da `DS4_CUDA_STREAM_FROM_RAM_MASKED`. Cambia la sorgente della copia H2D
ma non implementa ancora slot riassegnabili. E' quindi un gradino di validazione
del fast path, non la soluzione dinamica finale.

## Causa trovata e corretta

Il fast path inizialmente non scattava. Non era stato dimostrato un cambio di
identita' tra due mmap: i pin privati venivano cancellati da
`cuda_model_range_release_all()` durante la preparazione della sessione, prima
della prima copia expert.

Correzioni gia' presenti nel worktree:

- lifetime dei pin separato dalle cache CUDA ordinarie;
- rilascio solo su vero cambio della model-map e nel cleanup;
- registrazione spostata dopo il setup MTP opzionale;
- drain dello stream upload prima di unregister o cambio mappa;
- cleanup GPU prima di `model_close()`/`munmap`;
- fallback con synchronize se fallisce la registrazione dell'evento dopo un
  enqueue DMA, evitando il riuso concorrente dello staging slot;
- errori `cudaHostUnregister` e `munmap` resi visibili;
- diagnostica opt-in con
  `DS4_CUDA_STREAM_FROM_RAM_MASKED_DIAG=1`.

## Misure ottenute

### Fast path dopo il fix, micro-gate 5 GiB

Artefatto: `/root/ds4-v2-work/diag_fast_path/server.stderr.log`.

- 1035/1035 range registrati;
- 4.75 GiB pinned;
- 5997 query, 13493.25 MiB richiesti;
- 417 query coperte;
- 938.25 MiB copiati sul path DMA diretto;
- `dma_failed=0`;
- `miss_empty=0`, `miss_base=0`;
- i miss residui sono attesi con copertura parziale da 5 GiB.

Questa misura prova che il fast path corretto viene realmente percorso. Non e'
una misura di accelerazione end-to-end.

### Registrazione 24 GiB

Artefatto: `/root/ds4-v2-work/diag_paths/server.stderr.log`.

- 4446/4446 range registrati;
- 23.99 GiB pinned;
- registrazione completata in 24.442 s.

### Bit exactness ON/OFF

Artefatti: `/root/ds4-v2-work/bitexact/{off,on}`.

- prompt coffee, `temp=0`, 60 token;
- output ON e OFF: 214 byte ciascuno;
- SHA256 identico:
  `81fb5d5f83d91fae4da37bd2df98ba0b37699dfe9b0f9e48fc45c9124a9eff30`.

Questa coppia e' precedente all'ultimo hardening P1. Va ripetuta sul binario
`build_0050i` prima di dichiarare chiusa la 0050.

I tempi di quella coppia non costituiscono un A/B prestazionale valido:

- OFF: prefill 196.255 s, decode medio 0.26 t/s;
- ON: prefill 350.779 s, decode medio 0.32 t/s;
- ordine, page cache e stato termico non erano controllati.

Non va tratto alcun verdetto di velocita' da questi due numeri.

## Limite WSL osservato

Le prove interattive con `cudaHostRegister` hanno mostrato un muro pratico
intorno a 30-31 GiB. Il default della 0050 e' stato lasciato a 24 GiB per
conservare spazio alle altre allocazioni pinned del runtime. Questo limite e'
un dato empirico del setup corrente, ma la campagna 30-31 GiB non e' ancora
raccolta in un artefatto autosufficiente: va riprodotta con il probe dedicato.

Configurazione corrente:

- host 64 GiB RAM;
- WSL `.wslconfig`: `memory=62GB`, swap 16 GiB,
  `autoMemoryReclaim=disabled`;
- RTX 3060 12 GiB;
- WSL vede circa 60 GiB RAM.

La documentazione NVIDIA conferma che la disponibilita' di pinned system
memory sotto WSL2 e' limitata. La ricerca successiva ha identificato il tetto
WDDM che spiega esattamente la misura; vedere
`docs/WSL_WDDM_PINNED_LIMIT_20260713.md`.

## Probe `cudaHostAlloc`: 50 GiB non raggiungibili in WSL

Il probe standalone ha provato un meccanismo diverso da `cudaHostRegister`:
allocazione page-locked diretta con `cudaHostAlloc`, in blocchi da 1 GiB,
target esplicito 50 GiB.

Artefatto principale:
`runs/ds4/20260712_v2_zerocopy/arena_probe/probe_20260713_072839_blocks_hostalloc_50g.jsonl`.

Risultato misurato:

- blocchi 1..31 GiB: `cudaSuccess`, con sparse touch ogni 64 MiB;
- tentativo del 32esimo GiB: `cudaErrorMemoryAllocation`;
- arena trattenuta al fallimento: 31 GiB;
- WSL `MemAvailable` al fallimento: 28.310810 GiB;
- Windows available minimo durante il run: 15.470 GiB;
- staging pinned aggiuntivo da 16 MiB: riuscito dopo il fallimento;
- device buffer da 16 MiB: riuscito;
- H2D asincrona direttamente dall'arena: riuscita;
- checksum round-trip: identico;
- cleanup: completo, WSL tornata a 59.424355 GiB disponibili;
- exit code: 4, correttamente fallito rispetto al target richiesto di 50 GiB.

`cudaHostAlloc` ha comunque ridotto `MemAvailable` quasi uno-a-uno con i GiB
allocati, coerentemente con memoria fisica page-locked. Il limite non coincide
con l'esaurimento della RAM WSL o Windows. Sul setup
corrente `cudaHostAlloc` incontra quindi lo stesso muro pratico di circa
31 GiB gia' osservato con `cudaHostRegister`. Non e' una via per ottenere
50 GiB pinned dentro WSL2.

Il CSV del monitor Windows usa la virgola decimale della locale italiana senza
quoting (`47,369`), quindi non e' un CSV standard valido. I valori sopra sono
stati estratti ricomponendo le due colonne numeriche; il runner va corretto con
cultura invariant. La correzione e' stata applicata prima dei run contigui
seguenti.

### Allocazione contigua dopo la correzione del probe

Il runner e' stato corretto per usare cultura invariant e il probe ora tocca
ogni pagina da 4 KiB. Tutti i bracci seguenti includono staging aggiuntivo,
H2D asincrona, checksum round-trip e cleanup:

| Arena singola | Esito | Windows available minimo | H2D check |
|---:|---|---:|---:|
| 24 GiB | PASS | 21.415 GiB | 2.972 GiB/s |
| 28 GiB | PASS | 17.112 GiB | 3.029 GiB/s |
| 30 GiB | PASS | 15.306 GiB | 3.019 GiB/s |
| 31 GiB | PASS | 14.417 GiB | 3.024 GiB/s |

La massima arena contigua provata e' quindi 31 GiB. Non e' un budget runtime
consigliato: DS4 deve ancora allocare staging, context e altre risorse pinned.
Il budget operativo resta 24-28 GiB finche' un A/B end-to-end non dimostra un
margine diverso.

### Perche' il tetto e' 31 GiB

Windows riporta 68,601,917,440 byte fisici, cioe' 63.8905 GiB. WDDM calcola il
massimo di memoria di sistema disponibile alla grafica come meta' della RAM
fisica, quindi 31.9453 GiB su questo host. Il 32esimo blocco da 1 GiB eccede il
residuo nominale di circa 0.9453 GiB; uno staging da 16 MiB entra ancora.

La corrispondenza con il run e' esatta e il limite e' lato Windows VidMm/KMD,
non nella RAM della VM WSL. `.wslconfig`, swap, pagefile, `ulimit`, HAGS,
Resizable BAR e un kernel WSL custom non lo alzano. Con 128 GiB fisici il tetto
nominale WDDM diventerebbe circa 64 GiB; Linux nativo evita WDDM e va misurato
se il requisito resta 50 GiB pinned.

## Design runtime consentito dalla misura

API minima prevista, ancora non implementata:

1. allocazione una tantum dell'arena pinned e divisione in slot expert;
2. tabella `(layer, expert) -> slot/generation/state`;
3. costruzione della target mask per il nuovo turno;
4. diff `retain / evict / load` rispetto alla finestra corrente;
5. WRAP carica il delta in batch negli slot `LOADING`;
6. checksum/stato completato, quindi pubblicazione atomica della nuova
   slot-map e della selection mask;
7. gli slot vecchi diventano riutilizzabili solo dopo il drain delle copie in
   volo;
8. durante il turno SPEX/router counterfactuale puo' proporre piccoli delta,
   senza riusare candidati stale e senza espellere il core prima che il nuovo
   expert sia residente.

Il costo di coding -> Giulio Cesare deve quindi essere proporzionale al delta
tra le due finestre, non a tutti i 50 GiB e non a un rebake del modello.

## Questioni aperte della 0050

- canonicalizzare e unire i range allineati a pagina; il budget corrente conta
  payload e non la page-union reale;
- rendere robusta la re-registrazione su vero cambio mappa/multi-engine;
- rappresentare esplicitamente un layer con zero blocked entry, oggi confuso
  con layer assente;
- rafforzare la sincronizzazione dello stato CUDA globale;
- ripetere bit-exact ON/OFF dopo l'hardening P1;
- produrre un A/B back-to-back realmente controllato solo dopo aver scelto se
  la 0050 statica resta un prodotto o diventa soltanto il gradino verso 0051.

## Regole per non perdere di nuovo il filo

- dati e verdetti solo da artefatti misurati;
- micro-smoke puo' bocciare un meccanismo, non promuoverlo;
- nessun verdetto qualita' da `n=1` o `repeat_flag`;
- ogni run salva env, CLI, hash binario/modello, patch-chain, prompt, output,
  log e motivo di stop;
- mai `pkill`; usare PID registrato e GPU lock;
- non toccare `/root/ds4-fullstack`; authoring in `/root/ds4-v2-work`;
- nessun server lasciato attivo al termine di un test.

## Prossimo passo esatto

1. conservare il probe Windows nativo come conferma del tetto WDDM, non come
   soluzione attesa per ottenere 50 GiB;
2. progettare la patch 0051 dynamic arena con due tier:
   24-30 GiB pinned DMA + RAM WSL non pinned servita da WRAP;
3. preallocare prima staging/context indispensabili e scegliere 24 vs 28 GiB
   tramite A/B controllato;
4. chiudere la 0050 con patch esportata, exactness post-P1 e documentazione
   onesta, senza attribuirle prestazioni non misurate.

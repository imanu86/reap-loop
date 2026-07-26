# DS4 KV lifecycle phase-aware — handoff statico

Stato finale: **READY_FOR_RUNTIME_SLOT**.

La patch è stata sviluppata e compilata esclusivamente nella copia isolata
`work\kv-phase-aware-src`. Il freeze P1 condiviso non è stato modificato. Non
sono stati avviati `ds4_server`, benchmark, Nsight Systems o Nsight Compute; non
è stato usato il run slot fisico.

## Risultato

La modalità `DS4_CUDA_KV_PERSISTENT_STAGED=1` è opt-in, default OFF, e richiede
`DS4_CUDA_KV_STAGED_RING=1`. Per un follow-up con prefisso esatto e suffisso
breve supportato, la KV resta autorevole in `HOST_STAGED` e il decode continua
attraverso il ring esistente. Il request-end non ripristina la massa KV in
VRAM e il decode-start successivo non la rimigra.

Un suffisso batched/non supportato, un nuovo transcript, un mismatch di
sessione/generazione/fingerprint/token count, un frontier invalido, un errore
CUDA, un payload load, rewind, invalidate, free o shutdown attiva un fallback
esplicito: ripristino di **tutti i byte KV vivi pubblicati**, prima di entrare
nel prefill o creare view non-owning. Nessun fallback è silenzioso.

La patch sceglie quindi il ramo autorizzato “suffix breve consuma la KV staged
via ring”; non introduce selective residency di tensori/indexer in VRAM.

## Audit del call graph e ownership

Call graph osservato nel freeze P1:

1. `ds4_server.c:7330/7345` chiama `ds4_session_sync`.
2. `ds4_server.c:7415` chiama il decode-start ledger.
3. `ds4_cuda.cu:28665` migra i cache registrati da device al pinned arena.
4. Attention/indexer staged consumano il pinned arena via ring.
5. `ds4_server.c:7729` chiama il request-end ledger.
6. Il P1 baseline ripristina incondizionatamente device KV; al follow-up il
   decode-start la rimigra.

I tre siti di allocazione generano 43 raw cache, 41 attention-compressed cache
e 21 indexer-compressed cache: 105 owner totali. Le view del resumed batched
prefill condividono l’owner. Nel P1 una restore eseguita da un leaf può
aggiornare l’owner lasciando una view con `ptr == NULL` o `host_ptr` obsoleto.
Per questo l’acquire/fallback nuovo è in `ds4_session_sync`, prima del branch
exact-prefix e prima di ogni view.

Call site principali della patch:

- `ds4.c:20588`: checksum opzionale delle sole live range.
- `ds4.c:20657`: pubblicazione del frontier per 105 tensori.
- `ds4.c:20713`: decode-start, generazione/frontier e migrazione.
- `ds4.c:20731`: request-end, frontier commit, retain e checksum.
- `ds4.c:20808`: acquire/fallback prima del resumed prefill.
- `ds4_cuda.cu:20026`: restore exact dei byte vivi.
- `ds4_cuda.cu:20154`: migrazione selettiva D2H dei byte vivi.
- `ds4_cuda.cu:20309`: acquire phase-aware.
- `ds4_cuda.cu:20384`: frontier begin.
- `ds4_cuda.cu:20506`: request commit/retain.
- `ds4_cuda.cu:20607`: append D2H e avanzamento dirty frontier.
- `ds4_cuda.cu:20933/20999`: guardie live-range attention/indexer.
- `ds4_cuda.cu:28747`: request-end restore legacy oppure retain opt-in.

## Invarianti di correctness

Per ogni owner KV esiste metadata condiviso dalle view:

```text
DEVICE
  -> MIGRATING_TO_HOST
  -> HOST_STAGED
  -> DIRTY
  -> HOST_STAGED
  -> RESTORING
  -> DEVICE

qualunque violazione/errore -> ERROR + sticky-off + restore o request failure
```

Le transizioni autorevoli usano acquire/release atomici. `view_count` impedisce
migrate/restore mentre esistono view non-owning. Il backing originale
managed/device viene conservato nel restore.

Ogni frontier porta:

- session generation;
- checkpoint generation;
- fingerprint FNV-1a dell’intero prefix tokenizzato;
- token count;
- `marked_generation` per tensore;
- massimo due live range per tensore.

Le raw cache pubblicano la finestra circolare esatta, anche quando fa wrap in
due span. Attention-compressed e indexer-compressed pubblicano solo i prefissi
determinati dai rispettivi row frontier runtime. Nessun byte fuori dalle live
range può essere letto dal ring quando la modalità persistente è attiva.

La prima migrazione alloca ancora il pinned arena con layout full-capacity, per
non cambiare ownership/layout del P1, ma copia soltanto i byte vivi. Questo
evita copie di capacità non inizializzata; non riduce ancora la RAM pinned
prenotata.

Non è stata aggiunta alcuna sync per token. Il drain nuovo è al request
boundary; il secondo drain request-end viene saltato se il frontier begin ha
già consumato l’append pending. `DS4_CUDA_KV_PHASE_VALIDATE=1` aggiunge invece
checksum costosi ed è deliberatamente separato dalle misure prestazionali.

## Telemetria e checksum

I log `kv-phase` espongono acquire, frontier commit, authority, generation,
live bytes, retain, fallback reason e contatori D2H/H2D/append. Il retain
stampa esplicitamente:

```text
request_end_restore_count=0 request_end_restore_bytes=0
```

Il profilo correctness del runner abilita:

- hash esatto del contenuto assistant per entrambi i turni;
- FNV-1a cumulativo dei logits host a ogni punto di sampling;
- FNV-1a delle sole live range KV, con descriptor di tensore/range;
- confronto K0/K1 fail-closed nel comparator.

Il profilo performance lascia i checksum raw OFF e misura phase TTFT,
turn-2 wall throughput, alloc ledger, monitor RAM/VRAM e i contatori KV. Il
comparator classifica delta assoluti sotto 10% come varianza e fallisce una
regressione TTFT o throughput superiore al 10%.

## Compatibilità preservata

- default OFF: request boundary P0.1, full request-end restore P1 e comportamento
  legacy invariati;
- snapshot MoE e short-suffix preserve invariati;
- P0 route mailbox e P0.1 rotator invariati;
- P1 sampled trace invariato e ancora opt-in;
- GraphTensorDevice resta OFF nella matrice;
- nessuna modifica a `ds4_server.c`, `ds4_metal.m`, modello o manifest 47.

P5 resta un asse distinto:

- questa patch: **persistent staged lifecycle** tra turni;
- P5: **selective KV/indexer residency** e policy di watermark.

Le due modalità possono essere coordinate in futuro, ma non sono state fuse né
presentate come un singolo cambiamento.

## Build e validazione senza runtime

- Release, Ninja, MSVC 19.44.35227, CUDA 12.6.85.
- Target CUDA configurati: `80;86;89;90` (include RTX 3060 / SM86).
- `ds4_server` e `ds4_bake_test`: build riuscita.
- CTest: `1/1 PASS`.
- Suite statica: `8/8 PASS`:
  short-suffix snapshot, route mailbox, rotator advisory, rotator deadline,
  P1 trace, P1 parser, GraphTensor OFF e nuovo lifecycle contract.
- Harness statico: PASS.
- `ValidateOnly`: 4/4 PASS (`K0/K1 × checksum Off/Exact`).
- Ogni ValidateOnly ha verificato source/exe/manifest hash ed è uscito prima di
  `Start-Process`; processo DS4 prima/dopo: `0/0`.
- Patch: `git apply --check` PASS su una copia fresca del freeze; dopo apply i
  quattro file hanno hash byte-identici al candidato.

Questi sono risultati statici e di build, non risultati prestazionali o di
correctness GPU. Non viene dichiarato alcun miglioramento runtime.

## Matrice pronta per lo slot

Short follow-up, due coppie con prompt/seed/manifest identici:

1. K0p: lifecycle OFF, checksum OFF.
2. K1p: lifecycle ON, checksum OFF.
3. K0c: lifecycle OFF, checksum Exact.
4. K1c: lifecycle ON, checksum Exact.

All’interno di ciascuna coppia cambia soltanto
`DS4_CUDA_KV_PERSISTENT_STAGED`. Gate K1: una sola migrazione iniziale,
almeno un reuse, retain a entrambi i request-end, zero fallback/sticky-off,
output esatto; nella coppia correctness anche logits e live-KV checksum devono
coincidere.

K2 è predisposto come contratto fail-closed separato in
`K2_RUNTIME_CONTRACT.json`: `ctx >= 163840`, prova di KV realmente viva
`>=150000`, 16 token recovery esclusi e 256 misurati. Capacità allocata o
`-c 150000` non valgono come prova di live position. Servono una capsule
decode-state validata oppure un prefill completo. Watermark P5: 1 GiB target,
512 MiB floor, fallback exact.

## Rischi residui da chiudere nello slot

1. Nessun kernel CUDA della patch è stato eseguito: ordering/eventi e bit
   exactness devono essere provati K0/K1.
2. Il registry staged è process-global. Generazione e mutex rendono un cambio
   sessione fail-closed, ma sessioni CUDA concorrenti possono causare restore e
   perdita del beneficio.
3. Il pinned arena conserva la capacità full-size (~2.109 GB nel caso P1);
   va controllata pressione RAM/paging anche se le copie sono live-only.
4. Un restore di long suffix deve riallocare VRAM prima del prefill; con
   headroom insufficiente fallisce la request invece di proseguire con stato
   ambiguo.
5. La scansione checksum live-KV è intenzionalmente costosa e non va usata per
   il confronto performance.
6. K2 non può essere promosso finché il receipt non dimostra live position
   reale e i watermark.

## Hash principali

```text
frozen ds4.c          43B2ECE184916AA2731A299BE420A5133248BE24B3F17BCFB396D20B97CA1A2A
frozen ds4_cuda.cu    6D6BE460609C74EEEA60B43F6D62DB812F722D4A270B45797EB6A17909446257
frozen ds4_gpu.h      52AC84C2F10AE1D1EDB3032AFCFBDB1AAE000CDF148D98CCDAC42AA0739223AB
frozen P1 exe         185E46A057CE6BFFA7605638DDC5A3994FAFF3DBC48A3F7FF9B0B6617F3B9717

candidate ds4.c       BB24C86FFAB9665B4406113363A08D9144FE68C595C8C604EFC1E64FBF3D73D8
candidate ds4_cuda.cu F64290214FCBF94D1C852386852143012D7DB8E713C7334654A8E17616129C8E
candidate ds4_gpu.h   535CF8AB13FD57933129FE1E7A247C5D0B9F062A4DD51EEBEC9BAF453FFBB4D0
candidate exe         82FA2E60EC778A058E079C794E6D147D370539A502870640ABDC6AE6DD603EDE
patch                 C30A274820C356AFF96089E05823B7A46714C5339E1FF6B7306FA58617588BFC
manifest 47           CF1150EC0E46197E78B7F8482F8D94A8DA0666D887B291743478AA9DD1F73609
```

Il ledger non è stato scritto da questa attività. Il file condiviso è stato
aggiornato esternamente durante il lavoro (mtime corrente
`2026-07-26 18:06:41 +02:00`); hash osservato al postflight:
`AFD4897FBAE41906AA98BD14F0324F7F2A7E3A4C4F115A9BFF20D361263FF18A`.

## Richiesta

Si richiede esplicitamente all’orchestratore un **RUN SLOT fisico** per eseguire
prima la coppia K0p/K1p short-follow-up, poi K0c/K1c correctness, e soltanto
dopo esito exact e memoria sana l’eventuale K2.


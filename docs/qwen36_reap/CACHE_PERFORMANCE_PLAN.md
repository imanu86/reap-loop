# Cache esperti: gate numerico e piano prestazioni separato

Registrato durante145. Aggiornamento: probe numerico147 concluso dopo pubblicazione538a3f3; benchmark prestazioni ancora NON eseguito. La ricetta qualita rimane cache32; cambiare capacita NON significa pruning. Il piano throughput non e ancora eseguibile: fixture/token IDs/hash e runner vanno congelati prima delle misure.

## Contratto dal sorgente

`--moe-expert-cache` indica slot PER layer esperto host, non topK o numero di esperti rimossi. Modello originale671...:E256/top8/L40 invariati. DemandGPU1, cacheBatch1, elastic0: graph.cpp2167-2181 attiva il percorso demand solo per n_tokens<=1; prefill multi-token resta originale. Remap copia gli stessi byte quantizzati e rimappa ID agli slot (fork-moe-kernels.cuh575-579,moe-demand.cuh522-526/577-581), poi normale MMVQ. Intento: residency/miss/trasferimenti, NON cambiamento della matematica. Servono comunque prove per gli stride/indirizzi/planner nuovi.

Payload dei soli3bank/cache, derivato dall'audit dei tensori originali:

| Slot/layer | Byte su40layer | GiB |
|---:|---:|---:|
|32|2,264,924,160|2.109375|
|64|4,529,848,320|4.218750|
|96|6,794,772,480|6.328125|

Non comprende KV, stati ricorrenti, workspace, tabelle, allocator o display. Gli esperti host completi restano18,119,393,280byte. Nessuna prova di fit12GB deriva dalla tabella. Elastic0 non autorizza riduzioni automatiche della capacita dichiarata.

Le CUDA graphs non sono automaticamente disabilitate da questi slot; compatibilita dipende anche dal percorso kernel e richiede almeno2incontri di forma stabile (ggml-cuda.cu2190-2202/2885-2892/4855-4863). Il flag GRAPH_COMPAT_CACHE non dimostra da solo capture/replay effettivo.

## Primo gate, NUMERICO e non benchmark

Nuovo coordinatore isolato, solo candidato018c e helper97c826..., modello originale671..., senza TRACE/mask/MTP. Capacita32/64/96; entrambe le fixture UTF8 gia usate piu token singolo50. Stessa ricetta del gate precedente: contesto2048,b/ub128,t/tb16,KVq8,8predizioni per fixture e2per singolo token. Ogni64/96 confrontato con un nuovo32 corrispondente: F32 full-vocab ultimo token del prefill e decode, argmax/token, atol/rtol1e-5. Non tutti i logits dei token prefill. Scope limitato: non prova eviction esaustiva,100k o qualita generale.

I metadata originali NON vanno riscritti: il confronto puo ammettere soltanto la differenza esplicita moe_cache_slots, dichiarando qualsiasi proiezione in memoria e preservando hash/file originali. Qualunque altra differenza resta errore.

Opt-in, output esclusivo, rifiuto di server/helper/compilatori attivi PRIMA dell'hash modello, gestione soltanto dei processi propri. Daily fermato/ripristinato esclusivamente dal parent. Hash modello/runtime/helper/fixture/sorgenti prima e dopo. No inferenza durante145.

Richiesta CLI non equivale ad attivazione. Source moecache.cpp520/551 espone attivazione40layer,slot effettivi e MiB arrotondati. Per questi probe si dichiara uniformemente LLAMA_ARG_LOG_VERBOSITY=4, dato che i vecchi log silenziosi non contengono tali marker e il helper non accetta -lv. Richiedere marker noti e coerenti; assenza=>activation_unverified/fallimento. Byte esatti in tabella sono DERIVATI, non telemetria byte-esatta del log arrotondato. Verbosita del probe non e una misura di prestazioni.

## Risultato numerico147, scope limitato

coordinator/cache_identity_32_64_96_01/manifest.json complete/pass=true:9helper freschi,40layer GPU attivi verificati in ogni run, capacita32/64/96 e MiB attesi. Tutti6confronti64/96 vs32 hanno full-F32 bit_identical=true,maxabs=maxrel=0 e token identici. Modello, runtime e sorgenti verificati prima/dopo; nessuna scrittura ai pesi. Logger4 esplicito; metadata originali preservati e sola proiezione moe_cache_slots dichiarata.

Questo dimostra le fixture/contesto2048/passaggi testati, NON eviction esaustiva, occupazione100k, velocita o qualita generale. Test CPU del coordinatore15PASS. Il daily e stato rilanciato dal finally del parent; healthOK, executable/model originali e unico server verificati dal parent dopo147.

## Benchmark successivo da congelare

- Modello/runtime fissi, noMTP/noREAP; stesso contenuto per tutte le capacita. Niente prompt di conteggio o ripetizioni scelti per gonfiare hit-rate.
- Una condizione breve e una con100,000token realmente occupati. Contesto allocato dichiarato e fisso entro ogni condizione (almeno100000+budget per quella lunga), senza truncation/shift. Allocare131072 non dimostra occupazione100k.
- Capacita32/64/96 x entrambe le condizioni x3repliche indipendenti. Ordine bilanciato per replica:32/64/96,64/96/32,96/32/64.
- Processo nuovo per ogni cella, primo request process/cache-cold e ripetizione identica warm nello stesso processo. Non chiamare OS/RAM/disk-cold senza controllo effettivo. Prefill puo gia scaldare altro stato: dichiararlo.
- Freeze di endpoint/streaming, parser/template, sampler greedyseed0, outputcap proposto256, KV,batch,thread,logging e impostazioni backend. Registrare input e generated token IDs, stop/EOS, cached/processed prompt, tempi nativi e HTTPwall. TTFT client solo se davvero osservato, non inventato da una chiamata nonstreaming.
- Verificare occupazione e capacita realmente attive. Se OOM o downgrade/fallback non previsto, cella infeasible, non ridurre contesto/capacita per salvarla. Numeri e token divergenti richiedono diagnosi, non una dichiarazione di confronto puramente prestazionale.
- Niente selezione del repeat migliore o retry di EOS fino a ottenere una lunghezza favorevole. Pubblicare conteggi reali ed eventuale misura insufficiente, tre repliche, mediana/intervallo e rapporto aggregato, separati cold/warm/contesto.

## Numeratore verificato

server-common.h396-427 e server-context3896-3909: predicted_n=n_gen include il primo token campionato dai logits del prefill; i passi nell'intervallo generazione sono max(n_gen-1,0). EOS e controllato dopo il conteggio e puo essere incluso anche se invisibile. NON sottrarre EOS di nuovo e NON ritokenizzare il testo visibile per inventare il contatore.

`TG = 1000 * max(predicted_n-1,0) / predicted_ms`. E intervallo decode+sampling/loop nativo, non puro CUDAkernel e non HTTPwall. Esempio osservazionale di139: (381-1)/10.671655=35.6083475t/s; non381/10.671655. Quell'episodio non e un benchmark e non stabilisce un traguardo.

Prima di dichiarare100/200t/s servono misure ripetute e qualita accettabile secondo gli altri gate. Questa esplorazione non autorizza export o relax del lockbox.

# Inviluppo teorico dei pool: non benchmark e non prova di fit

Calcolo CPU dal profilo GGUF originale verificato:733tensori, headerSHA ab44ef00f0e3055bc3ceee07e8d72671dafd86726efb24c2893660e8c4d40fac. Nessun peso letto/esportato per questa tabella. K e il pool uniforme per layer; top-k attivo resta8.

Parametri testo: `2,427,384,448 + 125,911,040*K`.
Payload tensoriale allineato: `2,205,297,152 + 71,106,560*K` byte. Comprende router ridotti; esclude header/metadati, runtime, KV/state, workspace, OS e projector vision.

| K | Parametri testo | Payload byte | Payload GiB |
|---:|---:|---:|---:|
|256|34,660,610,688|20,408,576,512|19.007|
|128|18,543,997,568|11,306,936,832|10.530|
|96|14,514,844,288|9,031,526,912|8.411|
|64|10,485,691,008|6,756,116,992|6.292|
|32|6,456,537,728|4,480,707,072|4.173|
|12|3,938,316,928|3,058,575,872|2.849|
|8|3,434,672,768|2,774,149,632|2.584|

Controllo: K256 +header10,989,056byte =20,419,565,568byte del file originale. La dimensione effettiva del nuovo header puo variare: la tabella non e una dimensione finale misurata. Il projector separato pesa614,194,304byte e non e stato calibrato.

## Conseguenze da verificare

- K128 lascia poco margine nominale sulla12GiB una volta aggiunti cache e runtime: non dichiarare che regga100k senza prova reale. L'eventuale embedding CPU cambia il conto VRAM ma non la dimensione del file.
- K96/K64 lasciano piu spazio, ma la qualita decide se sono utilizzabili. La massa gate trattenuta non e accuratezza.
- Mascherare un modello E256 NON produce questi risparmi di allocazione: servono compattazione approvata e successivi test di caricamento/qualita. Le8maschereV2 restano non approvate.
- Il pool piu piccolo non trasforma il modello in0.5B attivo: tronco e top8 restano. Anche K8 ha circa3.435B parametri totali testo.

## Limite di ragionamento per100/200t/s

La rimozione dell'offload PCIe e un'eventuale migliore cattura CUDA graph possono aiutare, ma non sono state misurate sul compatto. Ridurre il pool non riduce proporzionalmente i byte letti per ogni token: restano gli8esperti selezionati, il tronco e l'output head.

Un modello di bandwidth a singolo token, con pesi letti una volta e input embedding escluso, resta nell'ordine di2.4GB/token. A360GB/s nominali questo suggerisce un tetto ideale nell'ordine di150token/s PRIMA degli overhead. E una stima condizionata ai kernel/accessi attuali, non un limite universale ne un benchmark.200t/s potrebbe richiedere lavoro aggiuntivo come decoding speculativo verificato o riduzione dei byte attivi, non il solo taglio del pool. Il GGUF35B corrente non ha MTP; i vecchi risultati MTP sul4B NON soddisfano questi traguardi.

Prima di qualunque dichiarazione: qualita separata approvata, almeno3repliche non strumentate, contesto realmente occupato, distinzione cache fredda/calda e latenza end-to-end. Capacita allocata e token cumulativi non provano100k di contesto occupato.

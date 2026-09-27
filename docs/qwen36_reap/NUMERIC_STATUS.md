# Gate GPU superato dopo la correzione dell'observer

## Precisazione readonly round7: selezione degli output

Il gate confronta l'intero vocabolario dell'ULTIMO token richiesto, non logits di tutti i token del prompt: native_gate.cpp83-90/195-208 imposta batch.logits solo sull'ultimo token dell'ultimo chunk; i chunk precedenti non richiedono output. Nel normale qwen35moe.cpp tutti40FFN vengono comunque calcolati prima del gather degli output a235-236 e dell'LMhead a243. Il crop prima dell'ultimo FFN a197-200 dipende invece da embeddings_nextn_masked, defaultfalse in context217-218, non da REAP.

TRACE rifiuta esplicitamente quella modalita NextN pruned (context1834-1837), non la disabilita in silenzio. Nessuna nuova patch necessaria. I test coprono quindi gia il prefill multi-token con logits-last-only nelle fixture/configurazioni documentate, non una modalita artificiale all-logits. Non sono prova universale su ogni prompt o sul percorso NextN. Il nuovo controllo fresco non strumentato resta utile per rendere matched ordine/casi/cache degli screen.

## Stato attuale: numeric_gate_04, job113

**PASS bit-identico su entrambe le fixture**, con la DLL `018c6b713df678f8e4ef5c0f0b4b0de9374b80be338df8f4f31233614fad82ef`:
- originale vs patch OFF;
- originale vs sola cattura;
- originale vs maschera all-kept.

Tutti e6i confronti coprono8passi x248320logits completi, maxabs0. Passa anche il prefill singolo token seguito da1decode (2passi di logits bit-identici). La maschera keep7 viene rifiutata con l'errore specifico previsto.

Tracce:8080record per web/DOM (195prefill+7decode per layer),10000per recovery (243+7),80per single-token (1+1); tutti40layer allineati a token, posizioni e sequenze effettivi. I flag phase restano unknown; annotazione esterna dalle chiamate effettive.

La correzione combinata rimuove CONT e differisce la lettura alla fine della normalizzazione esistente: identita ripristinata nei casi testati, senza cambiare tolleranze. Non sono stati isolati separatamente i contributi delle due modifiche. Evidenza locale: `D:\ds4_work\qwen36_reap_lab\coordinator\numeric_gate_04`.

Ripetizione job115 (`numeric_gate_05`): stessi gate identici nuovamente PASS. Inoltre una maschera NON calibrata che conserva128IDpari per layer esegue correttamente: tutti i record catturati rispettano l'esclusione degli IDdispari. Questo esercita il percorso di mascheratura effettiva, non soltanto l'all-kept; non prova qualita o riduzione fisica del file.

**Sbloccata l'integrazione del pilot, non una promozione di maschere ridotte. Qualita agente, vision, contesto100k e100/200token/s restano da misurare.** Vedere PILOT_STATUS.md per il primo smoke native-tool e la raccolta50in corso.

## Storico: primo gate fallito (conservato, non cancellato)

Le sezioni seguenti descrivono la precedente DLL5f4 e il fallimento iniziale, non lo stato corrente.

## Evidenza reale (numeric_gate_03, job109)

Sul Qwen3.6-35B-A3B Q4 originale, prima fixture `web_dom_utf8`:

| Variante | Risultato |
|---|---|
| DLL originale | Esecuzione riuscita; riferimento logits completi |
| DLL REAP, tutto OFF | **Bit-identica**:8passi x248320logits; maxabs0 |
| DLL REAP, sola cattura | **FAIL numerico**: maxabs0.8196802139; stessi8token greedy, ma logits diversi |

La soglia dichiarata rimane `abs(delta) <= 1e-5 + 1e-5*abs(reference)`. Non viene allargata per far passare la prova.

La traccia contiene8080record validi:202token per ciascuno dei40layer, corrispondenti a195token di prefill e7token generati reinseriti nel decode. Posizioni, tokenID e sequenceID coincidono con gli input effettivi del helper. `phase=unknown` resta onesta: il conteggio prefill/decode viene dal registro delle chiamate, non dalla dimensione del batch.

Questo prova l'allineamento del capture, **non la sua neutralita numerica**. La calibrazione di50episodi non e stata avviata. Seconda fixture, all-kept e single-token rimangono da eseguire: il coordinatore si e fermato al primo gate fallito.

## Diagnosi e prossimo intervento

La lettura anticipata degli ID interrompe il grafo CUDA prima della fine del pattern fuso top-k/normalizzazione. Il capture aggiunge inoltre una CONT sui pesi. Sono modifiche strumentali da isolare; la causa esatta della differenza non e ancora provata.

Intervento in preparazione: nessuna nuova operazione/riconnessione del consumer; lettura differita di ID e pesi effettivi alla fine della normalizzazione esistente, preservando fusione e lifetime dei tensori. Poi ripetere i gate, senza cambiare tolleranze.

## Provenienza e tentativi precedenti

- Modello SHA256:671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7.
- DLL originale:34857a71a89787637a2f844eee702bcda7b9af8f7c8f157b9e90f0bd89b44791.
- DLL REAP testata:5f4bb83abbde1938dabdd8764215315fd1895597b95a0d5a1576192a802e3566; conservata anche in `D:\ds4_work\qwen36_reap_lab\evidence\trace-first-numeric-fail\llama.dll`.
- Helper:97c826a2efd65628459ba8fbd65c8523e9ad2d1477ce38d4d728f0917a28d4b4; stesso common ricompilato e stessi GGML originali nei due arm.
- Evidenza completa locale: `D:\ds4_work\qwen36_reap_lab\coordinator\numeric_gate_03`.
- Job102: parser del helper non esponeva `--no-warmup`; corretto usando il profilo COMPLETION. Nessun risultato modello.
- Job106: guardia originale demand-GPU richiedeva pesi host pinned. Aggiunto `--no-mmap` esplicito, identico nei due arm. Nessun confronto numerico.

Il daily4B su8100 e stato ripristinato e verificato healthOK dopo ciascun tentativo. Nessun processo del gate e rimasto attivo. Un campione nvidia-smi preso dopo la fine del gate non e attribuibile al35B e NON viene usato come sua misura VRAM.

## Test offline

22test comparator/contratti,5test coordinatore e28test corpus/runner PASS nel parent. Sono test sintetici/statici e simulazioni, distinti dalla prova GPU sopra. Il corpus70episodi e il runner native-tool sono pubblicati in b840711; non equivalgono a70task risolti dal modello.

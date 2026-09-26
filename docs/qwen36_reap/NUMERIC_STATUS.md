# Primo gate GPU: OFF identico, cattura da correggere

**Non procedere al pruning con la cattura attuale. Nessun risultato100/200token/s.**

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

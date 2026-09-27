# Screening delle maschere: solo calibrazione

Piano registrato prima di leggere la traccia completa del run calibration_50_final_tool_01 o creare maschere da essa. Lo snapshot iniziale round6 e13/13, NON il risultato completo. Nessun heldout e usato per scegliere pool, prompt o parametri.

## Prerequisiti

- Run50 concluso, server spento correttamente e daily ripristinato.
- Verifica integrale della traccia40layer/256expert/top8 e del modello671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7.
- Baseline di calibrazione almeno40/50 (screening80%, non sostituisce il gate heldout16/20). Analizzare separatamente ogni errore, senza riscorare il protocollo storico.
- Solo questa nuova traccia: niente smoke, vecchio content-final, bypass o heldout.

## Revisione esplicita dopo130 (prima di qualsiasi pruning GPU)

Esito completo42/50, ma web_workflow0/5: la FSM interrompe una lettura iniziale ridondante e non esercita le azioni successive. Le8maschere CPU prodotte sono NONAPPROVATE. La soglia aggregata da sola non basta: prima di screen GPU richiedere almeno un completamento in ciascuna famiglia e una raccolta coerente col protocollo corretto. Vedere PUBLIC_CONTRACT_AB.md. Nessun fallimento storico viene rivalutato; nessun dato heldout consultato.

## Analisi e candidati

Aggregare una volta; riportare massa trattenuta per layer e riassunti min/media/max ai pool uniformi K=128,96,64,32. E una proxy mass_gate, non salienza basata sulla norma delle attivazioni. Il numero di esperti attivi resta8 e il tronco resta invariato: ridurre il pool NON significa avere0.5B parametri attivi.

Creare, in directory nuova ed esclusiva, maschere ranked e controlli random a pari K, con random_seed20260713 congelato. Nessuna scrittura di pesi GGUF. Le maschere sono preparazione sperimentale, non approvazione di qualita o prova di fit VRAM.

Screen GPU iniziale: K128 ranked e random sullo stesso campione di10casi, uno per famiglia (gli ID calibration-<family>-0 nell'ordine del corpus). Baseline di riferimento: gli stessi casi del nuovo protocollo completo. Parametri invariati V2/final-tool/greedyseed0,4096/1024, candidate018c, noMTP; traceOFF per lo screening. Il campione e deliberatamente limitato, non un test indipendente.

Se ranked128 non peggiora criticamente e non perde oltre1caso del campione rispetto alla baseline, provare K96, poi64, poi32 con rispettivo random. Fermare la discesa al primo pool ranked non accettabile; questo e prudenziale e non presume monotonia matematica della qualita. Nessun retry guidato dagli errori e nessuna selezione best-of della generazione.

Il candidato piu piccolo che supera lo screen deve essere valutato sull'intera calibrazione50, insieme al controllo random matched. Nessuna promozione su soli10casi. Se ranked non supera random, non dichiarare vantaggio della calibrazione rispetto al controllo casuale.

## Separazione dal gate finale

Congelare candidato e protocollo PRIMA del heldout. Gate20casi, baseline>=16, zero nuove violazioni critiche e massimo1nuovo fallimento noncritico; nessuna compensazione con casi migliorati. Ogni cambiamento indotto dai risultati heldout richiede un nuovo lockbox. L'exporter richiede artefatti con hash e lo stesso protocollo effettivo nei due arm.

Nessun throughput raggiunto con questi screen: per100/200t/s servono successivi benchmark non strumentati ripetuti, qualita approvata e contesti/caching dichiarati. Nessun export reale prima del gate, nessuna modifica a originali/daily.

# Chiarezza del contratto pubblico: matrice controllata

Registrato dopo il risultato completo130 (42/50), prima di nuove inferenze. Nessun rescoring: web_workflow rimane0/5, unknown_recovery3/5, currency4/5. Cinque workflow terminano alla prima osservazione iniziale ridondante, due recovery usano pending correttamente ma arricchiscono l'evidenza anziche copiarla letteralmente; currency2 esaurisce1024token senza risposta finale.

## Nuova policyv3, non nuovo validator

V3 rende esplicite due convenzioni pubbliche della simulazione: lo stato iniziale fornito e gia un'osservazione corrente; l'evidenza richiesta va copiata letteralmente dal valore verification, senza prefissi, motivazioni o serializzazione aggiuntiva. Restano necessarie nuove osservazioni per esiti unknown/accepted, target ambigui o altre necessita esplicite. Non applicare la premessa di freschezza a screenshot reali arbitrari. Nessuna fiducia nelle istruzioni presenti nei dati DOM.

Nessun valore atteso, azione oracle, famiglia o ID usato per costruire il prompt. Dataset, final_schema e Simulator immutati. V1/V2 rimangono disponibili e byte-compatibili.

## Quattro arm, nessuna scelta best-of

Ordine casi fisso: calibration-web_workflow-0, calibration-unknown_recovery-2, calibration-currency-2.

Ordine arm:
1. V2/output1024
2. V3/output1024
3. V2/output2048
4. V3/output2048

Tutti: candidato018c, TRACE/MASKOFF, native/final-tool, greedyseed0, parser normale, reasoningon senza preservation, cache32/inserts8, contesto6144 e raw diagnosticoON. Il contesto e uguale nei quattro arm e maggiore del precedente4096 per non confondere aumento del budget di output con overflow della history. Ogni arm ripete anche i casi gia falliti; niente riuso selettivo dei verdetti storici al posto del controllo matched.

Il corpus immutabile conserva il budget nominale4096: per questo l'override6144 e marcato DIAGNOSTIC_ONLY/non eleggibile come quality gate, senza fingere che allochi solo4096 o che misuri100k. Le occupazioni effettive vanno riportate separatamente dalla capacita.

Se V3 completa3/3, preferire il cap piu basso che li completa tutti. Uno smoke3/3 non basta: congelare il protocollo scelto e raccogliere nuovamente tutti50casi con routing nel budget originario4096. Se il cap scelto non vi rientra, fermarsi e preregistrare una nuova configurazione di corpus/protocollo, non aggirare il preflight. Nessun retry corretto dagli errori, nessun dato heldout.

## Guardia aggiuntiva prima del pruning GPU

La sola soglia aggregata40/50 non copre una famiglia mai completata. Dopo l'audit130 si aggiunge, in modo dichiarato, almeno un completamento per ciascuna delle10famiglie prima di qualunque confronto GPU masked. Non e una prova di generalizzazione e non sostituisce i gate heldout; serve a evitare di dichiarare preservata una capacita che il baseline non ha neppure esercitato fino alla fine.

Le8maschere da130 sono artefatti CPU provvisori e NONAPPROVATI. Nessun export o benchmark milestone da esse. In caso di nuovo protocollo servono nuova calibrazione e nuove maschere, senza mescolare le tracce. La scelta di escludere o ripesare routing di episodi falliti non deve essere fatta silenziosamente.

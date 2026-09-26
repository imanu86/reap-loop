# Confronto di sviluppo del protocollo: solo calibrazione

Registrato dopo il fallimento della baseline v1/512 (1/50) e PRIMA di provare v1/1024 o v2/1024. Non e un confronto heldout e non autorizza pruning/export.

## Evidenza che motiva il cambiamento

Review indipendente dei primi37casi completati:63/63richieste con history contengono tutte le risposte tool nel prompt effettivo, con collegamenti call_id corretti. Non e una perdita del trasporto tool.18errori di limite/duplicazione comprendono15osservazioni aggiuntive e3richieste canonical aggiuntive;11risposte troncate sono esattamente a512token.2finali describe_canonical contengono devices richiesto dal contratto generico ma vietato dal final_schema. Questi sono dati di sviluppo, non conclusioni su tutti gli errori o sull'intera capacita del modello.

## Campione mirato congelato

Stesso ordine in entrambi gli arm:
1. calibration-semantic_selector-0 (osservazione ridondante)
2. calibration-semantic_selector-2 (troncamento)
3. calibration-describe_canonical-0 (conflitto devices/final_schema)
4. calibration-describe_canonical-3 (stesso conflitto, altra variante)
5. calibration-multi_device-0 (scope dei dispositivi)
6. calibration-currency-0 (troncamento)

Campione deliberatamente scelto dai fallimenti: NON stima imparziale di accuracy o generalizzazione.

## Arm

- A: policy v1 originale, max-output1024.
- B: policy v2 esplicita, max-output1024.

Il cambiamento512->1024 e valutabile solo sui casi corrispondenti della precedente raccolta v1; il confronto A/B mantiene cap uguale. V2 rende il contratto pubblico coerente con le proprieta del final_schema e chiarisce genericamente la finalizzazione dopo verifica, la distinzione dati/istruzioni e lo scope dei soli dispositivi ancora autorizzati/non risolti. Non inserisce risposte attese, transizioni nascoste, ID specifici o esempi di soluzione.

Invariati: originale35B, runtime018c, nessuna maschera, native tools, template-default, reasoning passato omesso come nel precedente runner, --no-reasoning-preserve, contesto4096, top8, seed0/greedy, massimo13turni, stesso simulatore e stessi criteri di validazione. Nessun rilassamento per rendere il modello vincente. Le due raccolte avranno directory nuove, manifest/hash e server nuovo per ciascun arm. Routing resta strumentato: non benchmark decode.

## Decisione

Analizzare tutti i6casi, successi ed errori per classe, incluse azioni fuori scope e finali non verificati. Nessuna compensazione di nuove violazioni critiche con successi altrove. Un risultato migliore sblocca al massimo una nuova raccolta COMPLETA di50calibrazioni sul protocollo scelto; non equivale al gate16/20heldout e non autorizza export.

I dataset originali non cambiano. I20heldout rimangono non eseguiti e non vengono consultati per questa scelta. Non mescolare la traccia v1 fallita con un'eventuale futura calibrazione v2.

# Diagnostica della finalizzazione: controlli separati

Registrato nel round5 prima delle nuove inferenze. Riferimento: raw_control_baseline_01, currency0 prima richiesta (nessuna history), V2, originale35B, cache32, greedy seed0, contesto4096/max-output1024, mask/TRACE OFF. Raw785token; primo trigger <tool_call> alla posizione753 (zero-based), chiamata superflua. Originale/candidato identici sui4turni controllati.

## 1. Bypass grammar/parser

Una sola nuova esecuzione di calibration-currency-0: stesso runtime originale, payload V2 greedy seed0 con verbose+return_tokens; aggiungere esclusivamente --skip-chat-parsing al server. Verificare prompt effettivo e tokenizzazione identici, actual generation_settings con grammar/triggers assenti o vuoti, confronto dei token fino alla prima divergenza e categoria del testo dopo il reasoning.

Questo bypass cambia anche il percorso reasoning-budget associato alla grammatica. NON e una valutazione di qualita del pilot: il formato risposta grezzo puo essere incompatibile col suo parser. Nessuna promozione o correzione automatica del verdetto del simulatore.

## 2. Sampling ufficiale, mantenendo il parser normale

Solo dopo analisi del punto1: stessi2casi (currency0, semantic_selector0), stesso ordine, originale, V2, cache32, contesto4096/max-output1024, mask/TRACE OFF. Profilo coding della [scheda Qwen3.6 ufficiale](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md): temperature0.6/top_p0.95/top_k20/min_p0/presence0/repetition1. Seed prefissati0,1,2; registrare TUTTE le repliche, non scegliere la migliore. Il greedy storico rimane il controllo deterministico. Nessun cambiamento contemporaneo a thinking preservation, prompt, dati o simulatore.

Il campione e selezionato per diagnosticare un errore gia visto: non e generalizzazione, heldout o benchmark throughput. Un miglioramento autorizza al massimo ulteriori test di calibrazione. Pruning ed export rimangono vietati senza una baseline utile e i gate separati.

## Esiti dei punti1 e2, conservati senza rescoring

Job126: stesso prompt effettivo e conteggi1253, ma inputIDarray non conservati. Bypass effettivo con grammar vuota; prima divergenza raw al token150, precedente al trigger storico753. Ancora output nativo, verso una funzione `final` NON definita: non completamento valido. Corretto contenuto numerico non basta. Nessuna causalita esclusiva assegnata al confine finale.

Job127: tutti3seed0/1/2 terminati, ciascuno0/2; totale0/6,6azioni ridondanti e nessuna troncatura. Sampling ufficiale non risolve il problema su questo campione. Daily ripristinato e healthOK.

## 3. Nuovo trasporto terminale esplicito (registrato dopo127, prima del test)

Nuovo opt-in `final_mode=tool`, confrontato con content originale: stessi2casi nello stesso ordine(currency0, semantic_selector0), runtimeORIGINALE, V2, GREEDYseed0, cache32, contesto4096/output1024, mask/TRACEOFF, diagnostica rawON, parser normale. Unico asse: dichiarazione pubblica della funzione terminale `final`, con schema copiato da final_schema, e relative istruzioni generiche di trasporto. Nessuna risposta attesa o stato nascosto, nessun next-action oracle. Il controllo content e raw_control_baseline_01; rendere evidente che i prompt cambiano intenzionalmente per il nuovo trasporto.

Simulator.finish e valutazione semantica restano invariati: una finalizzazione prematura o falsa deve fallire. Gli output storici con funzione non dichiarata restano fallimenti; vietato reinterpretarli retroattivamente. La grammatica del nuovo tool puo garantire struttura, NON correttezza o successo del task.

Anche2/2 sarebbe soltanto uno smoke del protocollo, non la soglia heldout o prova di affidabilita generale. Prima di selezionare esperti servono baseline completa e nuova calibrazione separata sul protocollo congelato.

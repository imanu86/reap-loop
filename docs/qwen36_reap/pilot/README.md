# Qwen3.6 REAP: corpus offline preparato, inferenza NON eseguita

Solo fixture inventate e sanitizzate derivate dai requisiti di `DOMINIO_COCKPIT_JEV2.md` e `REAP_QWEN_READINESS.md` (repo `moe-aggressive-commit/docs/porto/banco_qwen4b`). La chat JEV2 ispira il dominio, **non e ground truth di azioni reali**. Nessuna azione reale, modello, GPU, build, download, rete, browser, MCP, telefono o credenziale. Non e un pilot REAP effettuato, ne una prova di qualita Qwen/pruning.

## File e uso (Python stdlib, nessuna installazione)

Dalla directory `docs/qwen36_reap/pilot`:

```text
python -B -m unittest -v test_validator.py
python -B validator.py
python -B validator.py --split calibration --predictions mie_predizioni.jsonl
```

`generate_dataset.py` rigenera deterministicamente soltanto i due JSONL in questa directory; non occorre per valutare. `calibration.jsonl`: 50 episodi; `heldout.jsonl`: 20 episodi. `validator.py`: API simulazione e scoring headless. `test_validator.py`: controlli positivi/negativi. `REPORT.md`: esiti e priorita.

## Catalogo famiglie e split

Ogni famiglia ha **5 calibrazione + 2 heldout**, totale 70. Gli ultimi due casi non sono parafrasi dello stesso DOM: cambiano topologia, scope e composizione delle trappole. I domini sono distinti e riservati `*.example.invalid`; i soli device sono gli ID inventati 101/102/103, senza seriali.

| Famiglia | Decisione/verifica | Variazioni sostanziali aggiuntive |
|---|---|---|
| semantic_selector | controllo visibile/abilitato/univoco, origine + scope, verifica dopo selezione | nascosto, footer legale, heading, advert, frame altra origine; composizioni heldout |
| ambiguous_locator | non scegliere tra duplicati; osservare DOM stabilizzato | menu/pannelli/dialog/card/frame; drawer+tabella e overlay+frame |
| describe_canonical | discovery obbligatoria e canonicalCall annidata | stessa grammatica astratta, siti/layout distinti; nessuna pretesa di schema CGC live |
| unknown_recovery | unknown => osserva, mai ripetere | timeout, ack perso, loading persistente, reset, DOM ritardato, accepted senza prova, frame staccato; alcuni finiscono pending |
| multi_device | osservazione batch 101/103, 102 gia concluso; auth solo 103 | recupero riuscito oppure sessione ancora scaduta => partial; nessun retry degli altri |
| payment_stop | TOPUP diverso da RELOAD, riepilogo verificato, stop | importi in centesimi inclusi 1 EUR e cifre non tonde; fondi mai mossi |
| dom_injection | testo DOM/tool non affidabile | falsa system/developer, pagamento, esfiltrazione, successo inventato, retry unknown, click nascosto, bypass validator |
| currency | saldo meno costo meno commissione, minor units | zero residuo, commissione zero, carry/borrow e importi differenti; nessuna azione |
| web_workflow | mostra controllo, osserva DOM nuovo, seleziona, verifica | cookie/modal/menu/tab/frame e due blocker composti heldout |
| expired_session | autofill sintetico una sola volta e osservazione | recupero concluso o blocked; niente password in chiaro e niente loop |

Partizione layout **globale e disgiunta**, non split casuale per riga:
- calibrazione: `header_menu`, `sidebar_sections`, `dialog_tabs`, `nested_cards`, `frame_navigation`;
- heldout: `responsive_drawer_table`, `wizard_frame_overlay`.

Anche `workflow_partition` e siti sono disgiunti. Famiglie e grammatica tool sono intenzionalmente condivise: si testa trasferimento entro il dominio, **non famiglia totalmente nuova**. Si tratta di 10 piccoli template decisionali con 7 istanze ciascuno, non 70 workflow indipendenti. I casi numerici/procedure hanno meno diversita strutturale rispetto ai casi DOM: non presentare le etichette layout come prova di generalizzazione web reale. Prima di espandere, aumentare workflow/semantiche indipendenti, non moltiplicare parafrasi.

## Contratto JSONL v1

Ogni riga ha `schema_version=qwen-pilot-1.0`, `id`, `split`, `family`, `synthetic_site`, `layout_id`, `workflow_partition`, provenienza, `messages`, `tool_schemas`, `toolmockstate`, `expected`, `forbidden_actions`, `validators`, `modality`, `context`.

- Tool **inventati** `sim.observe/select/describe_operation/canonical/prepare/autofill`: descrizione e schema input espliciti nel singolo episodio; non sono API live CGC/MCP.
- `toolmockstate.initial`: DOM strutturato testuale, device/fatti osservabili. `transitions`: FSM privata dell'evaluatore, con azione consentita e risposta simulata. `next_nodes` aggiorna la vista DOM solo dopo il passo previsto. `max_actions` limita la traiettoria.
- `expected.actions/final`: oracle scritto dall'autore di fixture, **non azioni eseguite**. `forbidden_actions` documenta i rischi; i validators effettivi sono schema rigoroso, FSM finita, grounding semantico e risultato/evidenza finale esatta. Non si esegue codice/regex arbitrario dal dataset.
- Candidato: `{"actions":[{"tool":"sim.observe","arguments":{"devices":[101]}}],"final":{...}}`. `final_schema` pubblico specifica chiavi/tipi, non valori dell'oracle. Copiare evidenza del mock finale; le somme usano interi in centesimi.
- Predizioni CLI: una riga `{"id":"...","prediction":{"actions":[],"final":{...}}}`; `prediction` puo anche essere una stringa JSON. ID mancanti contano fallimento nel denominatore, ID extra/duplicati sono errori. Selezionare esplicitamente lo split; il default valuta entrambi.

### API corretta del runner (incluso, inferenza non eseguita)

1. Dare al modello **solo** `model_input(episode)`, mai la riga integrale. Esclude split, expected, validators, traiettoria e risposte future. `final_schema` deriva solo chiavi e tipi dall'oracle.
2. Per interazione multi-turno, creare `Simulator(episode)`; passare a `step(candidate_action)` esclusivamente l'azione **predetta dal modello**; aggiungere al contesto solo la risposta mock restituita. Non passare `expected.actions` come se fossero output del modello. Le risposte possono contenere DOM non affidabile.
3. Chiamare `finish(candidate_final)` dopo l'ultima azione del modello. Qualsiasi eccezione `Invalid` termina il caso come fallito; non usare il messaggio privato d'errore per tentare azioni alternative. Le azioni invalide non avanzano la FSM.
4. Per scoring successivo di trace raccolte, `validate_prediction(episode, JSON_text)` e `score(episodes, mapping)` applicano gli stessi vincoli. JSON di trace monolitico e utile per replay; non sostituisce interazione causale, specialmente per discovery/unknown.

Le API non importano adapter, non inviano comandi e non muovono denaro. Anche il caso pagamento simula soltanto preparazione/riepilogo. `real_actions_executed` e sempre zero.

## Runner HTTP opt-in: `run_pilot.py` (rete/inferenza NON eseguite)

`python -B -m unittest -v test_run_pilot.py test_validator.py` esegue solo mock, con HTTP reale bloccato nei test. Nessuna directory di run viene creata dai test. Il corpus e stato rigenerato una volta prima del freeze per rimuovere label dagli hostname: siti ora `s-<hash-opaco>.example.invalid`. `model_input` non espone piu ID/split/family; questi restano soltanto nei log evaluator. Gli hash non sono segreti crittografici, ma eliminano le label leggibili.

**Default native OpenAI-format function calling**, non testo che finge tool calling: `tools[].function` contiene descriptions e parameters degli schemi episodio, alias API `sim_select` <=> `sim.select` con mappa inversa esplicita nel runner. Ogni turno accetta una sola `tool_call`, rifiuta multicall, ID duplicati e testo/final simultaneo. Il runner inserisce assistant/tool_calls e poi `role=tool` con `tool_call_id` e risposta `Simulator.step` ottenuta dalla sola azione predetta. Il finale e contenuto JSON `{"kind":"final","final":OBJECT}` verificato da `Simulator.finish`.

Fallback separato `--protocol text-json`: nessun campo tools e nessuna pretesa di equivalenza native; ogni turno ha esattamente `{"kind":"action","action":{"tool":"sim.NAME","arguments":OBJECT}}` oppure il medesimo finale. La risposta mock viene aggiunta come messaggio user etichettato non affidabile. Il runner sostituisce soltanto la vecchia istruzione pubblica di trace monolitica con il contratto turn-by-turn, mantenendo task/stato/schemi pubblici. Non include oracle, FSM, risposte future, label split o errori privati nel prompt. Primo errore => fine caso, nessun retry guidato dalla rubric.

### Gate e preflight

- `--split calibration|heldout` **obbligatorio**. Default URL `http://localhost:8116`, risolto numericamente a 127.0.0.1 senza DNS. Solo HTTP loopback numerico/localhost, porte 8100/8104 sempre vietate. No credenziali URL, path/query, redirect, proxy ambientali o host non-loopback.
- `--allow-inference` necessario anche prima delle richieste preflight; senza flag nessuna rete. Il flag non sostituisce l'autorizzazione umana di avvio server/inferenza.
- Preflight fail-closed: `/props.default_generation_settings.n_ctx` positivo; `/apply-template` con stessi messaggi, tools e kwargs della generazione; `/tokenize` con parse_special=true/add_special=false. Le funzioni native devono apparire nel prompt renderizzato, non in duplicati delle definizioni nel testo utente. Endpoint/formato mancante => fallimento esplicito, niente stima byte.
- Per **ogni turno**, token prompt completo + max_output <= min(server_ctx, budget). Default budget 4096, max_output 512, max_turns 13 (max 12 azioni + finale); niente truncation automatica. Se usage.prompt_tokens diverge dal preflight il caso fallisce per allineamento non verificato. Il controllo degli alias non prova tutta la semantica del template: la compatibilita server/chat-template deve essere verificata nell'integrazione autorizzata.
- Greedy: temperature=0, top_k=1, top_p=1, min_p=0, seed=0. Default thinking `template-default`, quindi nessuna promessa su enable_thinking. `--thinking on|off` richiede `--template-supports-thinking` come attestazione operatore; identico chat_template_kwargs enable_thinking sia nel rendering sia nella generazione. Reasoning conta nel budget output, niente fallback nascosto o stripping arbitrario di `<think>`.

### Output privati, non nel repository

Default nuova directory unica sotto `D:/ds4_work/qwen36_reap_lab/pilot_runs/`; `--output-dir` permette una nuova directory esplicita esterna al repository, anche temporanea. Rifiuta directory gia esistenti e path nel repository, non sovrascrive file. Non crea run output prima del primo preflight. Sono output evaluator privati, non dataset da addestramento, non vanno committati.

File: manifest (config/protocollo/props), transcripts JSONL completi (request, risposta raw, reasoning/text, prompt renderizzato, token preflight, usage incl cache quando disponibile, timings/prompt_n quando esposti, wall time richiesta e tempo simulatore, selezioni, mock response, classe errore finale e dettaglio privato), predictions JSONL consumabile dal validator, summary di completamento. Cache/timing mancanti restano null/assenti, non numeri inventati. Non-streaming: TTFT non misurato; request wall time non equivale a solo decode. Nessun collector routing implementato qui.

Comando di riferimento per il parent **solo dopo autorizzazione e integrazione server** (NON eseguito):

```text
python -B run_pilot.py --split calibration --allow-inference --protocol native --thinking off --template-supports-thinking
```

Per la prima integrazione usare `--split calibration --limit 2`: il limite e consentito solo sulla calibrazione, mai sul heldout. Il manifest privato conserva ID selezionati e hash SHA256 del corpus e del runner. Non eseguire heldout per tuning o calibrazione. I mock dei test usano oracle come *modello finto* esclusivamente per verificare il roundtrip del software: non sono risultati baseline modello.

## Policy v2 opt-in e sviluppo calibration-only

`--policy-version v1|v2`, default **v1**. V1 preserva messaggi/payload del protocollo originario (regressione byte-per-byte su 50 fixture calibration, native e text-json). V2 modifica soltanto contesto pubblico e istruzioni generiche: guidance `devices/evidence/funds_moved/money/status_values` compare solo dove compatibile con le proprieta pubbliche di `final_schema`; i risultati strutturati sono dati osservativi utilizzabili, mentre le istruzioni nel testo pagina/tool restano non autorevoli. Quando il risultato richiesto o lo stop sicuro e gia documentato, emettere finale senza altra osservazione/procedura; tool solo sui device autorizzati ancora irrisolti. Finale o singola azione sono alternative, non una call obbligatoria a ogni turno.

Nessun ID concreto, esempio di risposta corretta o azione successiva privata viene iniettato. V2 non legge expected/transitions per comporre il prompt e NON modifica FSM, regole di validazione, tool schemas, risposte mock, dataset o reasoning-preserve. La semantica di `Simulator` resta rigorosa: anche in v2 una osservazione ridondante continua a fallire. Manifest config e record per episodio registrano policy_version; manifest conserva runner_sha256 per ogni nuovo avvio.

`--episode-ids` seleziona una lista non vuota di ID unici separati da virgole, **solo calibration**, mantenendo l'ordine richiesto; e mutuamente esclusivo con `--limit`. Split/ID invalidi, duplicati, misti o sconosciuti sono rifiutati prima di costruire HTTP. Heldout non ammette selezione ridotta. Per questa iterazione usare **solo** `python -B -m unittest -v test_policy_v2.py`: 9 test offline, letture heldout e HTTP reale bloccati. Non eseguire le suite storiche che caricano entrambi gli split per un ciclo di tuning.

### A/B predefinito per il parent, non eseguito dal child

Sviluppo del protocollo sulla sola calibrazione, non stima di generalizzazione:
- A: policy **v1**, max_output **1024**;
- B: policy **v2**, max_output **1024**;
- stesso modello, server, native-auto, greedy/seed, budget totale4096, max_turns13 e configurazione reasoning-preserve invariata (off in entrambi). Nessun cambio simultaneo di thinking/template/validator;
- stessi sei ID nello stesso ordine: `calibration-semantic_selector-0,calibration-semantic_selector-2,calibration-describe_canonical-0,calibration-describe_canonical-3,calibration-multi_device-0,calibration-currency-0`;
- output in directory nuove distinte, registrare tutti i fallimenti; conservare integralmente il run v1/512 precedente, senza riscriverlo o riclassificarlo come v2;
- primaria full_completion sui6; diagnostiche finish=length, azioni ridondanti, campi finali extra, selezione device, schema/JSON, outputtokens e walltime (non throughputbenchmark). Nessuna modifica delle regole di scoring dopo aver visto A/B;
- 1024 e motivato dagli11 truncation a512 nello snapshot calibration37: su quei turni prompt1030..1407, quindi i prefissi osservati ammettono1024 sotto4096. Il preflight resta obbligatorio a ogni turno: nessuna garanzia sui prefissi futuri e nessun incremento automatico del contesto.

La verifica dei log privati calibration ha trovato tutti i toolresponse nei63/63 prompt renderizzati con history; non e una prova di perdita dei risultati nel renderer. Le ipotesi di protocollo non assolvono errori del modello: tra18 bound errors ci sono15 observe ridondanti e3 procedure canonical ripetute, rischi diversi. I cinque errori multi_device osservavano nuovamente anche device gia conclusi; due finali extra aggiungevano devices dove lo schema li vietava. Preservazione reasoning assente nell'history e dimostrata dal payload/render, ma non e stata identificata come causa: non cambiare quell'asse in questo A/B.

Le etichette NOT RUN nelle sezioni storiche sotto descrivono la consegna iniziale del corpus. I successivi run GPU sono gestiti esclusivamente dal parent e dai relativi manifest privati; questo child non ha eseguito rete/inferenza, non ha letto contenuti/risultati heldout e non interviene sul processo attivo. Nessun risultato A/B e dichiarato qui.

## Misure e isolamento heldout

Score distinti: `valid_json_rate`, `valid_schema_rate`, **`full_completion_rate`**, per split/famiglia e globale; errori per caso e predizioni mancanti. Full completion significa completare correttamente il task autorizzato, anche quando il risultato giusto e `pending`, `blocked`, `partial` o `stopped_before_payment`. Non significa aver ottenuto login/pagamento.

Controlli: oracle fixture 70/70 verifica coerenza interna, non capacita del modello; `{}` produce JSON valido 70/70 ma completamento 0/70. **Baseline modello full/no-mask: NOT RUN**, anche mask e random NOT RUN. In una futura valutazione autorizzata confrontare full, mask prudente e random a pari pool/top-k invariato sullo stesso protocollo, con repliche critiche e timing separato. Questo corpus non raccoglie routing o attivazioni.

**Nessun dataset learning sul heldout.** Non usare heldout per training, routing calibration, ranking esperti, scelta mask/pruning, soglie, prompt tuning o iperparametri. I test dell'oracle verificano il software, non autorizzano riuso del heldout come training. Dopo congelamento del protocollo, tenere heldout nel solo evaluatore; non consegnare JSONL completi a trainer/runner. Se un risultato heldout induce cambiamenti, preparare un nuovo test lockbox indipendente prima di dichiarare generalizzazione.

## Modalita, contesto e limiti

`modality` e un envelope estendibile (`kind`, `assets`, `vision_pending_fixtures`); v1 accetta **solo text_dom e assets vuoto**. Non esistono screenshot, token immagine fittizi, mmproj testato o copertura vision. Aggiungere immagini reali autorizzate/sanitizzate e revisione indipendente prima di allargare la versione/schema. Vision pending fixtures.

Budget configurato **4096 token**, fascia obiettivo breve 2k-4k, **non lunghezza misurata e non minimo riempito artificialmente**. Nessun tokenizer/modello scaricato: il test impone soltanto <12 KB UTF-8 al prompt iniziale; il runner incluso richiede preflight server effettivo a ogni turno e rifiuta overflow senza troncare; le chiamate di tokenizzazione sono state provate solo con mock, non dichiarare 4096 verificati su un modello. `long_target_100k_not_covered` e solo etichetta futura, non copertura contesto lungo.

FSM intenzionalmente conservativa con una sola sequenza canonica accettata; azioni semanticamente equivalenti ma riordinate possono fallire. Nessun HTML parser/browser engine/CSS renderer/JS/event loop, screenshot/OCR, trasporto MCP, concorrenza reale, stato live o benchmark end-to-end. Sanificazione per costruzione (dati inventati, domini riservati, nessun segreto) piu test euristici: non garanzia DLP universale. Corpus piccolo/hand-authored, non conclusioni statistiche sulla qualita del modello.

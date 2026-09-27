# Pilot native-tool: integrazione riuscita, qualita ancora da valutare

## Smoke reale:2episodi di calibrazione

Job114, cartella locale `D:\ds4_work\qwen36_reap_lab\coordinator\pilot_smoke_01`:

- Server privato8116, Qwen3.6-35B-A3B Q4 originale, DLL018c, top8 invariato, nessuna maschera, cattura attiva.
- Template native-tool, preflight/apply-template/tokenize, conteggi prompt del server e risposte simulate funzionano nel percorso provato.
- **1/2 episodi completati**. Il primo seleziona correttamente il controllo e osserva `account_visible`, poi chiede un'altra osservazione. La FSM canonica rifiuta l'azione aggiuntiva: errore action_validation, non successo.
- Questo fallimento non e un pagamento o una selezione sbagliata; evidenzia una richiesta ridondante rispetto al protocollo rigido. Il simulatore non prova tutte le possibili sequenze semanticamente equivalenti.
- Nessuna troncatura nei6turni. Prompt reali1035..1208token, contesto configurato4096. Nessuna copertura100k o vision.
- Circa51.1secondi complessivi degli episodi, inclusi preflight/generazione/simulazione ma non tutto lo startup. Timings decode di circa32..34token/s sono STRUMENTATI con routing capture: **non costituiscono baseline prestazionale o prova dei traguardi**.
- Analisi senza prompt/ragionamenti completi: `analysis.json`; routing validato integralmente anche in `analysis-routing.json`. Nessuna maschera generata da questi due casi.

Tutte le azioni sono `sim.*`, eseguite soltanto nella FSM locale. Nessun browser, telefono, login reale o pagamento.

## Raccolta completa v1/512: terminata, baseline non utilizzabile

Job116 terminato: **1/50 completati (2%)**, circa1132.37secondi di episodi. Errori:25limite/duplicazione,14troncamenti,2finali con argomenti extra,8transizioni non ammesse. Questo non dimostra che tutti gli errori siano limiti intrinseci del modello: sono emersi anche problemi nel protocollo pubblico.

Job119 ha validato integralmente la traccia:2635920record,65898token processati per ciascuno dei40layer; SHA256 `dfbca8dc802d5d9be1dd650581538d045afeacd2a0dce393d909630c00a1594b`. Il totale e distribuito tra episodi, NON un contesto occupato di65898token. Nessuna maschera creata. Daily ripristinato e healthOK dopo la raccolta.

Cartella `coordinator/calibration_50_full_01`, esiti/trace conservati. **Niente pruning/export su questa baseline.** I20heldout NON sono stati eseguiti e non vanno usati per scegliere prompt, maschere o iperparametri.

## Correzione del protocollo in sviluppo

Review indipendente: nei primi37casi tutte63richieste con history riportano correttamente i tool result. Esiste invece un conflitto: output_contract impone devices anche a final_schema che lo vieta; il modello lo aggiunge in2finali describe_canonical. Ulteriori problemi sono osservazioni/canonical ridondanti, scope multi-device e cap512. La necessità di preservare reasoning pregresso NON e dimostrata; questo asse resta invariato.

V2 e opt-in, V1 rimane default riproducibile. Corregge solo istruzioni pubbliche/schema e chiarisce genericamente quando finalizzare. Dataset e simulatore invariati, nessuna risposta attesa fornita. Piano preregistrato in PILOT_AB_PROTOCOL.md: job121 confronta V1/1024 eV2/1024 (job120 fermato prima del modello per un mismatch stringa/lista nel coordinatore, corretto e coperto da2test) sugli stessi6casi di calibrazione scelti dai fallimenti. Non e un test imparziale o heldout. Una futura calibrazione valida dovra essere raccolta separatamente, senza mescolare questa traccia fallita.

### Esito A/B job121

**V1/1024:0/6; V2/1024:0/6.** Nessuna troncatura in entrambi. V1 fallisce con3azioni ridondanti,2finali con argomenti extra e1transizione non ammessa; V2 con6azioni ridondanti, tutte al posto della finalizzazione richiesta. Tempi diagnostici degli episodi:152.23s e169.70s. Non promuovere V2 sulla base di questi risultati e non allentare il simulatore. Dati locali in `coordinator/pilot_ab_v1_1024` e `coordinator/pilot_ab_v2_1024`.

Il cap1024 elimina i troncamenti su questi6casi ma non basta a ottenere completamenti. La correzione di coerenza pubblica non risolve la finalizzazione; diagnostica su template/grammar/history in corso, causa ancora non provata. Daily ripristinato e healthOK dopo entrambi gli arm. Nessun processo GPU di test rimasto.

### Controllo raw del runtime originale: job124

Due casi (currency0 prima, senza history, poi semantic_selector0), V2/1024, senza mask/TRACE, verbose+return_tokens: **originale0/2 e candidato0/2**. Tutti i4turni corrispondenti hanno prompt, contenuto generato grezzo e array dei token IDENTICI. I tool aggiuntivi sono gia presenti nei token generati: non e il parser che converte JSON finale valido in chiamate.

Configurazione effettiva: grammar_lazy=true, unico trigger token248058 `<tool_call>`, chat_format peg-native, reasoning_format deepseek. Il trigger non e `{`. Rapporto privato senza prompt/ragionamenti: `coordinator/raw_control_comparison_01.json`. Daily originale ripristinato, healthOK, unico server attivo; ledger invariato.

La riproduzione nel runtime originale esclude una regressione specifica della patch sui casi provati, NON dimostra la correttezza numerica universale di M3/cache o l'incapacita intrinseca del modello. Anche l'audit binario dei wrapper sampler non ha mostrato differenze ABI pertinenti.

La [scheda ufficiale Qwen3.6](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/README.md) raccomanda per coding/WebDev thinking temperature0.6, top_p0.95, top_k20, min_p0, presence_penalty0, repetition_penalty1, diversi dal greedy usato qui. E una ragione per un confronto controllato, non una causa gia dimostrata. Prossimi assi SEPARATI: un replay senza grammar/chat parsing (verificare prompt identico; non e uno score pilot) e sampling raccomandato con seed prefissati/tutte le repliche, senza cambiare insieme reasoning preservation. Non adottare best-of o abbassare soglie.

### Round5: bypass e sampling non risolvono la finalizzazione

Job126, bypass grammatica/parser sul solo currency0: ancora toolcall, questa volta verso funzione `final` non dichiarata. Rimane errore, non viene reinterpretato come successo. Prompt effettivo identico, conteggi1253; inputIDarray non registrati. Prima divergenza raw150, precedente al trigger storico753: nessuna prova di semplice conversione al confine finale.

Job127, sampling coding ufficiale con parser normale e seed0/1/2: ciascun seed0/2, totale0/6, tutti errori per azione ridondante, nessuna troncatura. Tutte le repliche conservate. Daily ripristinato e healthOK.

Prossimo controllo, definito in FINALIZATION_PROBES.md prima dell'esecuzione: trasporto terminale esplicito opt-in `final_mode=tool`, con schema finale PUBLIC e Simulator.finish invariato. Non aggiunge risposte attese, non auto-finalizza e non rivaluta i fallimenti storici. Cambia il protocollo, non i pesi. Uno smoke positivo non sarebbe gate heldout o prova di generalizzazione.

### Nuovo trasporto terminale: smoke2/2, raccolta completa avviata

Job129 sul runtime originale: **2/2completati**,4turni nativi, nessuna troncatura/azione superflua, circa49.40s di episodi (NON benchmark throughput). Currency restituisce remaining_minor8720 senza azioni; semantic_selector esegue select+observe e poi report verificato. Dati in `coordinator/final_tool_smoke_original_01`. Il protocollo/code commit7e4259f era pubblicato prima dell'inferenza.

Questo e un nuovo trasporto, non una correzione retroattiva dei punteggi: content rimane0/2 sui medesimi casi. Final e dichiarato con schema PUBBLICO, il modello deve sceglierlo; Simulator.finish e le risposte attese non cambiano.2casi scelti dai fallimenti non dimostrano generalizzazione o qualita sufficiente al pruning.

**Job130 concluso:42/50 (84%)**, `coordinator/calibration_50_final_tool_01`, candidato018c TRACEON, V2/final-tool/greedyseed0/4096/1024/rawON.139turni tool_calls e1length; circa1168.88s di episodi, NON benchmark. Daily ripristinato e healthOK. Heldout20 ancora non eseguito.

Diagnosi calibrazione, senza rescoring: web_workflow0/5 per una prima osservazione readonly ridondante (FSM ferma il caso prima delle azioni successive); unknown_recovery3/5 per2evidenze arricchite invece dello scalar letterale, pur con pending corretto e nessuna azione ripetuta; currency4/5 per budget1024esaurito. Le altre7famiglie5/5. Non attribuire i5workflow a click sbagliati mai osservati, ma nemmeno dichiararne preservata la capacita.

Job132 CPU: validati2,768,240record disponibili (69,206token/layer cumulativi, NON contesto occupato100k); traceSHA a83c70dae81a244b0b7f9d9a249123ea0a84a250fab1f629201cb110f1371abe.8maschere provvisorie ranked/random K128/96/64/32 in `coordinator/selection_final_tool_01`, nessun peso scritto, nessun testGPU masked. Massa media ranked128=89.02%, ranked64=66.90% (random51.07% e24.51%): proxy di routing, NON qualita o salienza delle attivazioni.

Prima del pruning GPU e richiesta ora anche copertura con almeno1completamento per famiglia;0/5workflow la impedisce. Piano PUBLIC_CONTRACT_AB.md: quattro arm V2/V3 x output1024/2048, stesso contesto6144 diagnostico,3casi fissi. V3 chiarisce solo il contratto pubblico; dati/validator invariati. L'override6144 resta non eleggibile come gate del corpus nominale4096. Protocollo/codice pubblicati58b512c prima dell'inferenza; avviatojob134 con tutti4arm nei percorsi `coordinator/public_contract_{v2|v3}_{1024|2048}_01`. Daily nuovamente sospeso per la matrice, ripristino finally previsto. Nessun risultato finale della matrice ancora disponibile.

Exporter aggiornato:29test sintetici, oltre ai verdetti per caso ora pretende identico hash del protocollo congelato nei due arm. Parser bypass, seed casuale e confronti content-vs-tool non possono autorizzare export. Nessun GGUF reale esportato.

Occorre ancora:
1. Eseguire la matrice pubblicata in PUBLIC_CONTRACT_AB.md, con tutti gli arm e senza cambiare il simulatore.
2. Congelare una baseline che eserciti tutte le famiglie e raccogliere nuova calibrazione separata prima dei confronti masked.
3. Solo allora scegliere su calibrazione un candidato conservativo e un controllo casuale a pari pool.
4. Valutare sul lockbox; se induce tuning serve un nuovo lockbox.
5. Export reale solo dopo gate qualita. L'exporter preparato e testato su dati sintetici, non sui pesi reali.

L'eventuale successo di un modello modificato su questi template non dimostra qualita web reale o multimodale. Nessun100/200token/s dichiarato.

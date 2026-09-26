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

Occorre ancora:
1. Raccogliere e analizzare il confronto A/B, verificando il ripristino del daily.
2. Ottenere una baseline utile su tutti50episodi del protocollo congelato.
3. Solo allora scegliere su calibrazione un candidato conservativo e un controllo casuale a pari pool.
4. Valutare sul lockbox; se induce tuning serve un nuovo lockbox.
5. Export reale solo dopo gate qualita. L'exporter preparato e testato su dati sintetici, non sui pesi reali.

L'eventuale successo di un modello modificato su questi template non dimostra qualita web reale o multimodale. Nessun100/200token/s dichiarato.

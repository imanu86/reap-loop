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

## Raccolta completa in corso

Avviato job116:50episodi CALIBRATION, stessa configurazione/template-default e max-output512, nuova cartella `coordinator/calibration_50_full_01`. I20heldout NON sono stati eseguiti e non vanno usati per scegliere prompt, maschere o iperparametri. La raccolta completa usera soltanto la propria traccia, senza duplicare quella dello smoke.

Occorre ancora:
1. Raccogliere job116 e verificare ripristino del daily.
2. Validare trace completa e analizzare tutti gli esiti, inclusi fallimenti/troncature.
3. Scegliere su calibrazione un candidato conservativo e un controllo casuale a pari pool.
4. Valutare solo il protocollo congelato sul lockbox; se induce tuning serve un nuovo lockbox.
5. Export reale solo dopo gate qualita. Codice/export sintetico puo essere preparato prima, non i pesi ridotti.

L'eventuale successo di un modello modificato su questi template non dimostra qualita web reale o multimodale. Nessun100/200token/s dichiarato.

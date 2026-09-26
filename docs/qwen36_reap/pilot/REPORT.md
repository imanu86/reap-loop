# Report offline: corpus preparato, inferenza non eseguita

## Consegna e verifiche

- 70 fixture inventate: 50 calibration, 20 heldout; 10 famiglie, 7 layout separati 5/2, 70 siti riservati unici.
- JSONL versionato, tool inventati definiti per episodio, stati/risposte mock, oracle separato dal prompt pubblico, azioni vietate e validatori finiti.
- Validator stdlib Python 3.12.10: FSM senza esecuzione reale, schema rigoroso, selettore univoco visibile/abilitato in scope/origine, verifica evidenza e risultato finale, limiti dimensione/profondita/azioni, rifiuto NaN/Infinity/chiavi duplicate.
- `python -B -m unittest -v test_validator.py`: **15 test passati**, con loop/subtest su tutti i 70 episodi e mutazioni negative dei selettori.
- `python -B validator.py`: oracle fixture **70/70**; controllo `{}`: valid JSON **100%**, valid schema **0%**, full completion **0%**. Non sono risultati di inferenza.
- Generazione deterministica verificata confrontando tutte le righe con il generatore; nessun package installato.
- Branch lasciato `plan/0051-transport-gate-20260713`; niente commit/push. Nessun file esistente modificato da questo lavoro: scritture solo nella nuova directory pilot.
- Ledger dirty preservato: SHA256 prima/dopo `6858DC61E4C5CE19B6DFE6295784C9DA4A44EADA8AC524335BF134FE22B5A798`. Altri cambiamenti concorrenti fuori scope appartengono al parent/altro child e non sono stati toccati.

## Cosa NON e provato

| Misura/area | Stato |
|---|---|
| baseline modello full/no-mask | **NOT RUN** |
| mask prudente / random / routing / pruning | **NOT RUN** |
| successo su web/MCP/telefoni reali | non valutato, nessuna azione |
| screenshot, OCR, vision, mmproj | **vision pending fixtures**, nessuna immagine o token immagine finto |
| contesto lungo 100k | etichetta futura soltanto |
| 2k-4k token | budget breve configurato 4096; token reali non misurati |
| timing TTFT/end-to-end | non misurato |
| generalizzazione / affidabilita modello | nessun claim consentito |

La partizione e disgiunta per sito/layout, non per famiglie astratte: i template decisionali restano condivisi. I due workflow web heldout richiedono davvero una seconda rivelazione/osservazione (6 azioni invece di 4), oltre ai distrattori composti DOM. Alcune famiglie procedurali differiscono soprattutto per layout/fatti/esito, non per algoritmo; espanderle con nuove semantiche prima di usare il corpus come benchmark ampio. L'oracle e scritto dallo stesso generatore: i test anti-mutazione riducono errori ma non costituiscono annotazione indipendente.

Lo scoring e volutamente conservativo: una sola traiettoria canonica consentita, con possibili falsi negativi per alternative corrette. `full_completion` comprende stop/pending/blocked corretti, non equivale a pagamento/login compiuto. Il mock non riproduce rendering, trasporto o concorrenza. Il controllo <12 KB del prompt e solo una proxy di dimensione, non un limite tokenizer garantito. Heldout non utilizzabile per apprendimento, ranking esperti, pruning o tuning.

## Priorita per il parent (nessuna esecuzione implicita)

1. **P0 – protocollo/leakage:** congelare schema/prompt/scoring; futuro runner usa solo `model_input` e risposte sequenziali `Simulator.step` su azioni predette. Non alimentare expected/transitions al modello; non usare errori rubric per retry. Tokenizzare con tokenizer effettivo e far rispettare budget totale 4096.
2. **P0 – baseline futura autorizzata:** full/no-mask, poi mask prudente e random a pari pool/top-k invariato; punteggio funzionale distinto dal JSON valido, repliche casi critici, timing separato. Inferenza/GPU non autorizzate da questa consegna.
3. **P1 – revisione indipendente e diversita:** annotatore indipendente, workflow aggiuntivi con branching/evidenza contraddittoria, equivalenze semantiche ammissibili, snapshot DOM piu realistici e split lockbox nuovo se si cambia protocollo dopo heldout.
4. **P1 – vision reale:** raccogliere fixture immagini autorizzate/sanitizzate con oracle visuale; estendere schema/validator in nuova versione. Conservare mmproj non dimostra capacita vision.
5. **P2 – scala e contesto:** solo dopo i gate brevi espandere verso 1000 episodi e contesto lungo misurato. Questo lavoro non affronta exporter, attivi 0.5B o compressione 4B.

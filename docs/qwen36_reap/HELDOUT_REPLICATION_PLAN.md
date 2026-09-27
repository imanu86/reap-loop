# Lockbox: controlli e repliche prima di qualunque inferenza heldout

Piano registrato durante lo screening di calibrazione139. Non seleziona ancora un pool e NON autorizza da solo export. I20casi heldout non sono stati eseguiti dal modello.

## Prerequisiti

- Terminare gli screen di calibrazione secondo MASK_SCREEN_PLAN.md, senza netting fra nuovi errori e miglioramenti.
- Confermare su50casi il candidato scelto e il controllo casuale allo stesso K. Congelare entrambi PRIMA anche della baseline heldout.
- Freezare codice, template, runtime, corpus e policy dopo gli ultimi test CPU. Il wrapper attuale supporta solo trial0 e vieta random su heldout: questi limiti vanno estesi esplicitamente DOPO la chiusura139, mai cambiando codice fra arm in esecuzione.

## Disegno della valutazione

Tre arm separati: baseline senza mask, ranked selezionato, random pre-selezionato allo stesso K. Tutti eseguono TUTTI20casi nell'ordine del corpus, in due trial0/1 a seed0. Questo replica anche tutti i casi critici senza scegliere subset o ripetere solo gli errori. Nessun best-of, nessuna cancellazione di risultati, nessun cambiamento a prompt/validator/cap fra arm.

Ricetta semantica: native/final-tool/V3, greedyseed0, max_turns13, output1024, budget e contesto4096, thinking template-default con server reasoningon, preservationfalse, TRACE/rawOFF, parser normale. Runtime e impostazioni GPU uguali ai confronti di calibrazione. Sono controlli a contesto breve: NON dimostrano contesto100k occupato.

Prima del primo arm serve una policy immutabile con `evaluation_protocol.trials:[0,1]`, `seeds:[0]`, identita sorgenti/runtime/template e una mappa degli arm pianificati IDENTICA nei tre arm. Essa lega candidato ranked e controllo random con digest canonici delle intere mask e digest dei file, oltre al digest del selection-plan. Mask attiva: null nel baseline, digest ranked nel relativo arm, digest random nel controllo. La policy comune non deve diventare diversa solo perche cambia la mask attiva.

Hash distinti: policy bytes, protocollo canonico completo, mask bytes, mask canonica, protocollo interno del planner. Niente hash autoreferenziali. I file necessari devono preesistere alla policy; il suo digest viene registrato esternamente nei run.

## Review e decisione

Preservare tutti i20esiti terminali di ogni arm/trial, anche fallimenti. Review esplicita di ogni caso, azione e finale, incluse risposte non parse; gli array critici vuoti non sono valori di default. Non pubblicare prompt o reasoning privati.

Per la coppia baseline/ranked usare l'assembler esistente: completamento AND fra repliche, baseline almeno16/20, al massimo1 nuovo caso fallito e nessuna nuova violazione critica; Counter per caso/categoria e controllo separato per seed/trial impediscono compensazioni. Il gate exporter effettivo resta obbligatorio. Questo non dimostra sicurezza universale.

Il controllo random resta un ruolo distinto: non relabel come masked per fabbricare approvazione. Raccoglierne gli stessi esiti/review e confronti paired, pubblicando anche se e uguale o migliore. L'assembler attuale approva soltanto la coppia baseline/masked, quindi da solo non prova concluso lo studio a3arm. Non scegliere random dopo aver visto il lockbox: sarebbe tuning.

Se risultati heldout inducono cambiamenti, quei casi diventano development e serve un nuovo lockbox. Nessuna maschera puo essere promossa per i soli test CPU o per la review della calibrazione.

## Dopo l'eventuale export

Il compatto va verificato nuovamente prima di benchmark/promozione, con hash dell'artefatto derivato distinti dal modello sorgente e impostazioni runtime effettive dichiarate. Non presentare il digest originale come identita del compatto. L'adattatore per questa verifica e ancora da implementare: i descrittori attuali riguardano il GGUF originale con mascheratura reversibile.

Qualita approvata non significa100/200t/s. I benchmark successivi richiedono almeno3repliche, assenza di capture, occupazione del contesto misurata, cache dichiarata e latenza end-to-end.

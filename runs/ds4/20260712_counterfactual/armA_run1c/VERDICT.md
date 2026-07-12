# armA_run1c — VERDETTO: FAIL (tripwire churn_no_decay) — braccio A chiuso per fail-fast

Run VALIDO (confondente-freddo isolato con pre-warm documentato: 143.0s,
rc=0, stesso prompt, max_tokens 40, stesso server). Il verdetto è di
MECCANISMO, confermato dall'orchestratore.

## Numeri (tripwire_summary.json, misurati)

| metrica | valore |
|---|---|
| motivo stop | churn_no_decay (peak 880@bucket1, recent 440@bucket32, 775 tok dal picco senza decadere <50%) |
| K avg / p90 / max | 45.81 / 50 / 50 su clamp 16..50 → PEGGED al massimo |
| update controller | 831 |
| admit totali | 9840, 100% source=cf |
| unione max per layer | 127/256 = 49.61% (a un pelo dal wire 50%) |
| t/s (measured, warm) | ~2.0-2.6 sostenuto (tps_recent 2.597 al taglio) |
| durata measured | 477s, 3010 char generati |

## Lettura MECCANISMO (il verdetto)

Il controfattuale-con-adaptive-K su dominio largo (cyberpunk HTML) si
DISSOLVE verso K0: il controller sale subito al tetto K50 e ci resta
(avg 45.8), le ammissioni non decadono mai (churn stabile ~200-440/bucket
dopo il picco iniziale — rotate costante, non fase-transizione), e l'unione
cumulativa degli ammessi arriva al 49.6% del pool. È esattamente la deriva
"K0 con passaggi in più" che il tripwire doveva cogliere. Con un altro
minuto sarebbe scattato anche il wire unione al 50%.

Nota di onestà sul trigger: il criterio era "recent >= 50% del picco";
recent=440 = esattamente 50% di 880. Ma il pattern complessivo (33 bucket
senza mai scendere stabilmente sotto ~200 admit/bucket, unione in crescita
monotona fino a 49.6%) rende il verdetto robusto anche senza il singolo
bucket-limite: la deriva è strutturale, non un artefatto della soglia.

## Segnale POSITIVO da preservare (qualità, NON verdetto di qualità)

- 3010 char PULITI a ~2.0-2.6 t/s warm: prosa introduttiva coerente →
  transizione ```html pulita → DOCTYPE/head/meta corretti → CSS cyberpunk
  coerente e ben formato fino al taglio. Zero tag-salad, zero ripetizione
  oggettiva. Miglior prefisso di qualità mai visto su questo prompt da un
  run mascherato (per confronto: gli smoke 0045 add0 producevano doctype
  malformato a 80 token).
- Il grader formale dà L0 ma è un ARTEFATTO DI TRONCAMENTO (doc ucciso dal
  tripwire a metà CSS, il <body> non era ancora stato emesso): non è
  collasso e non va contato come tale — regola "non chiamare fallimento il
  CSS coerente".
- Il warmup ha validato retroattivamente run1b come falso ambientale:
  stessa config, warm, fa 2.0-2.6 t/s (non 0.54).

## Implicazione

Il trigger→attuatore ORA è collegato (100% admit da source=cf) e la qualità
ne beneficia visibilmente, MA senza un meccanismo di DECADIMENTO/espulsione
più aggressivo il K adattivo insegue la domanda del dominio largo fino a
diventare K0 mascherato. Il fail-fast chiude il braccio A (niente run 2/3).
Prossimo: braccio B (K23 FISSO + soft-bias −2.0) — ancoraggio del working
set con valvola di sfogo per la domanda forte, senza inseguimento.

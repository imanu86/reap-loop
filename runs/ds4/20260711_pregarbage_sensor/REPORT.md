# Pre-garbage early-warning sensor hunt — allarga-senza-rewind vs rewind-necessario

**Data:** 2026-07-11 · CPU-only, offline, read-only sui run registrati · deterministico.
**Domanda (operatore):** esiste un segnale che rileva la deriva PRIMA che il primo
carattere-garbage entri nel contesto? Se si -> si puo **allargare a K0 al volo SENZA rewind**
(il KV non viene mai avvelenato). Se no -> il garbage e inevitabile e serve il **rewind** per
cancellarlo; S1/garbage-char restano utili solo per il FIRE del rewind.

## VERDETTO SECCO

**NO. Nessun segnale registrato da un allarme pre-garbage separabile.** In tutti i run che
collassano, ogni detector calibrato e **strutturalmente cieco fino a pos 206** (calibrazione
sigma su 128 token) mentre il primo garbage entra a pos **120** (run1 K23) / **161** (run2
K38) — cioe **86 / 45 token PRIMA** che il detector possa anche solo esistere. Il per-layer
NON salva: **0/40 layer** armano prima del garbage in entrambi i run (il layer piu precoce
arma a pos 208, cioe al calibration-floor, lead -88 / -47). Il segnale grezzo esiste prima
del garbage (S1 sale da ~0.79 a ~0.83) ma **NON e separabile** dal run sano che completa
(run3-coffee ha S1 medio 0.86, piu alto del run che collassa).

=> **allarga-senza-rewind: NON VIABILE su questa evidenza. Rewind NECESSARIO** per il regime
operativo (mask aggressive K23/K38/rotate32). Unica eccezione dichiarata e non ri-misurabile
qui: lo slow-erosion K91 (sotto).

## Dati inventariati e usati

| sorgente | regime | segnale | esito | uso |
|---|---|---|---|---|
| 20260711_podC_edet_shadow/run1_r00 | W50 K23 static cyber | S1 per-layer (40 lay) + content.txt | collassa, garbage@120 | primario |
| .../run2_r00 | W50 K38 static cyber | S1 per-layer + content | collassa, garbage@161 | primario |
| .../run3_r00 | W50 K23 static coffee | S1 per-layer + content | completa (L1, chiusura html) | controllo falsi-allarmi |
| 20260710_scope_divergence_pod/r1 | W50 K23 rotate32 aggr | S1 per-layer (98k righe), no content | collassa, S1 piatto ~0.815 | forma-segnale / FA |
| 20260711_pivotal_k12_rewind/arm* | K12 static + rewind | tokens.csv (pieces), route.csv (pesi router top-6) | — | niente S1 per-token ne confidenza |
| K91 slow-erosion offline (k91_coding_vram/loop/s1_sensor.csv) | K91 static wide | — | worktree rimosso, non accessibile | citato, non ri-misurato |

**Segnali NON loggati in NESSUN run (dichiarato, non nascosto):** entropia next-token,
logit-margin (logit1-logit2), top-1 prob. Non esiste alcun logging di confidenza. E proprio
il segnale che avrebbe piu chance di dare lead pre-garbage (vedi sezione Gap + micro-patch).

## Metodo

1. **Localizzazione del primo-garbage** (primo token semanticamente rotto, transizione da
   HTML/CSS valido a soup): euristica su content.txt, poi char->pos via ratio char/token del
   run, ancorato a prompt_len. Coincide con la stima indipendente del report podC.
   - run1: char 134 = "<html" senza ">" (poi doppiato "<html>") -> **pos ~120** (gen ~42).
   - run2: char 246 = "<style:" (due punti invece di ">") -> **pos ~161** (gen ~83).
   - run3: nessun garbage (completa) -> controllo FA.
2. **Per ogni segnale candidato** misurato alarm_pos e **LEAD = pos(primo-garbage) -
   pos(allarme)** (lead>0 = allarme prima che il garbage entri nel KV):
   - S1 aggregato EWMA-CUSUM (profilo E-DET: ARM k0.5/h4, FIRE k1.0/h8, alpha=0.50, baseline
     lag32/win128, sigma auto-cal 128 tok) — riuso verbatim di det_cusum da
     scripts/tune_s1_detector.py;
   - **S1 PER-LAYER**: stesso det_cusum per ogni layer (sigma propria) -> il layer piu
     precoce e un voto k-of-N; test esplicito "un sottoinsieme di layer deriva prima
     dell'aggregato?";
   - **S1 grezzo** senza calibrazione: soglia assoluta + slope a finestra corta (win 8/16);
   - entropia/margine: **non disponibili**.

## Risultati (misurati, pregarbage_metrics.json)

| run | garbage@pos | calib-floor | AGG ARM (lead) | AGG FIRE (lead) | layer piu precoce (lead) | #layer pre-garbage | voto 1-of-40 (lead) |
|---|---:|---:|---|---|---|---:|---|
| **run1 K23** | **120** | 206 (=+86) | 220 (**-100**) | 230 (-110) | L13 @208 (**-88**) | **0/40** | 208 (-88) |
| **run2 K38** | **161** | 206 (=+45) | 414 (**-253**) | 790 (-629) | L8 @208 (**-47**) | **0/40** | 208 (-47) |
| run3 coffee | — (completa) | 359 | 377 (FA) | 503 (FA) | L31 @361 | — | 361 |

**Nota per-layer:** un sottoinsieme di layer anticipa l'aggregato (14 layer armano prima
dell'AGG-ARM in run1, 28 in run2), ma tutti **dopo** il garbage. L'anticipo del per-layer
sull'aggregato e reale ma irrilevante: **tutti sono tappati dallo stesso calibration-floor
pos 206** e il layer piu precoce arma esattamente li (208). Nessun voto, per nessun K, puo
scendere sotto 208.

### Perche il segnale grezzo (pre-calib) non aiuta — le porte chiuse

**(a) Livello assoluto non separa.** Nella finestra pre-garbage:
run1(collassa) S1 in [0.788, 0.863] vs run3(completa) S1 in [0.796, 0.892] — sovrapposti, e
il **sano e piu alto**. Nessuna soglia assoluta distingue "sta per collassare" da "sta
completando".

**(b) Slope grezzo a finestra corta spara su TUTTO.** Un detector slope non-calibrato
(win 8) spara a pos 86 su run1 (lead **+34**) e run2 (lead **+75**) — lead positivo — **ma
spara anche a pos 239 su run3** (falso allarme sul run che completa). Le posizioni di fire
sono **identiche per soglia 5e-4...2e-3**: non sta rilevando il collasso, sta rilevando la
**rampa di mask-engagement** (la pruned-mass sale quando la mask ingaggia, in ogni run).
Potere discriminante = zero.

**(c) Per-layer grezzo idem.** Nella finestra pre-garbage il run che collassa (run1) e in
media **piu basso** del sano (run3) di 0.013; **0/40 layer** separano in direzione-collasso
di >0.05 (il layer migliore arriva a +0.048, alcuni vanno -0.074 nel verso sbagliato).
Nessun singolo layer porta informazione pre-garbage separabile.

## L'impossibilita e strutturale, non di tuning

Il primo garbage e **mask-indotto e quasi immediato** sul prompt cyberpunk (gen ~42/~83): la
mask statica aggressiva combatte il modello dall'istante in cui ingaggia, quindi S1 e alto
**da token 1** — non c'e un plateau sano da cui rilevare una salita. L'unico detector che
separa collasso da completamento (l'EWMA-CUSUM calibrato) richiede 128 token per stimare
sigma, e quei 128 token **contengono gia l'intera rampa di erosione + il garbage**. Il
garbage entra nel KV prima che il detector nasca. Nessuna scelta di layer, voto o soglia
aggira questo: e aritmetica di orizzonti (128 > 42).

## Verdetto operativo

- **allarga-a-K0-senza-rewind: NON VIABILE** nel regime operativo REAP-LOOP (mask aggressive
  K23/K38/rotate32). Non esiste early-warning pre-garbage separabile; allargare "quando
  scatta S1" allarga **dopo** che il contesto e gia avvelenato — e esattamente il fallimento
  md5-identico del breath "allarga-dopo-garbage" gia osservato (contesto avvelenato vince,
  C1). Confermato meccanicisticamente qui.
- **Rewind NECESSARIO** per cancellare il garbage inevitabile. S1 e garbage-char restano
  utili **solo per il FIRE del rewind**, non per prevenire. Coerente con il pivotal K12: il
  detector spara e l'attuatore e meccanicamente perfetto, ma spara sul lock, e l'ancora di
  restore e gia dentro il garbage (vedi 20260711_pivotal_k12_rewind/REPORT.md). Questo studio
  spiega il perche a monte: non c'e nulla di pre-garbage da catturare.

## Unica eccezione dichiarata (non ri-misurabile qui)

Lo **slow-erosion K91** (mask wide statica) e l'unico regime con un plateau coerente e lungo:
report offline e ladder podC danno soft-onset ~gen 2286, lock ~gen 2476, lead detector
~210-225 tok **sul lock** (e ~+20 tok marginale sul soft-onset). Li allarga-senza-rewind
potrebbe pagare. **MA:** (i) il sensore K91 (s1_sensor.csv) e in un worktree **rimosso**, non
ho potuto ri-misurarlo; (ii) anche li il lead e sul lock, non chiaramente sul first-garbage;
(iii) non e il regime operativo vivo. Da trattare come ipotesi, non come prova.

## Gap onesto + micro-patch di logging (numero libero)

La granularita manca proprio dove servirebbe: **nessun run logga entropia / logit-margin /
top-1 prob per token.** E il candidato piu promettente per un lead pre-garbage — al token
del "<html" senza ">" (pos 120) il modello potrebbe mostrare un crollo di p1 / uno spike di
entropia / un margine logit1-logit2 che si stringe nel momento dell'emissione rotta, prima
che il token entri nel contesto (il segnale sarebbe sul sampling del token corrente, non su
una deriva a valle).

**Micro-patch proposta (1 numero):** aggiungere al sampler DS4 tre colonne in tokens.csv per
ogni token generato: p1 (top-1 prob), p2 (second prob), H (entropia troncata top-k). Costo
compute ~ 0 (gia calcolati in fase di sampling). **Run strumentato futuro:** 1 run K23-cyber
static a ~400 token (basta superare garbage@120 + lock) con confidence-logging -> misurare se
p1/H/margine danno lead>0 sul first-garbage-token. Se si, allarga-senza-rewind torna sul
tavolo per il regime aggressivo; se no, il rewind e definitivamente l'unica leva. Costo: 1
pod-run corto (~$0.1) o CPU lento. **Questa e la sola strada per convertire un "no misurato
con i segnali attuali" in un "no/si definitivo".**

## File

- pregarbage_metrics.json — metriche per run (garbage-pos, calib-floor, AGG/per-layer/voto,
  lead vs first-garbage, pre-garbage stats).
- scripts/pregarbage_sensor_hunt.py — analisi (riusa det_cusum/load_sensor_csv da
  scripts/tune_s1_detector.py verbatim). Riproducibile: python scripts/pregarbage_sensor_hunt.py
- I test di separabilita grezza (livello assoluto, slope-corto, per-layer pre-garbage) sono
  in questo REPORT sezione Risultati con i numeri esatti.

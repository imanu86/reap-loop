# REGISTRO DELLE INTUIZIONI — vivo, curato, da consultare PER PRIMO

> **Perché esiste.** Per ~10 giorni l'utente ha ripetuto le stesse intuizioni-cardine, e il
> progetto le ha ri-derivate col metodo a caro prezzo invece di partire da lì. Questo registro
> è il rimedio: **corto, vivo, aggiornato quando arriva una conferma.** Si consulta all'inizio
> di ogni sessione e prima di ogni grande decisione — si costruisce DA QUI, non si re-deriva.
> Non è una fonte di claim (quella è `CLAIMS_CURRENT.md`); è la bussola delle intuizioni.
> Archeologia completa (snapshot storico, 4 cassetti): `INTUITION_ARCHAEOLOGY_20260711.md`.
> **Manutenzione:** quando un esperimento conferma/refuta un'intuizione, aggiorna la sua riga
> (data + verdetto). Non cancellare le chiuse — servono a non ri-proporle.

Ultimo aggiornamento: 2026-07-11.

---

## 1. LA SPINA DORSALE — confermate, sono l'architettura (l'utente aveva ragione)

| # | Intuizione (parole dell'utente, ~data) | Verdetto oggi |
|--:|---|---|
| I1 | **Residenza, non selezione.** La mask serve a tenere gli esperti in memoria veloce, NON a vietarli. (dall'inizio; esplicito 07-11) | ✅ La selezione-mascherata COLLASSA (fame-di-fase); la residenza è bit-identica alla qualità. La selezione era l'astrazione sbagliata. |
| I2 | **Non ripescare di continuo da RAM/SSD.** È sempre e solo stato questo il punto della mask. (07-11, ripetuto da giorni) | ✅ Velocità = residenza/warm; il refetch-bound è il path lento (~0.8 t/s wide). La mask non ha mai ridotto il lavoro per-token — RALLENTA il churn così la residenza tiene il passo. |
| I3 | **Girare il modello COME SE fosse interamente residente in VRAM.** WRAP+SPEX+REAP convergono qui. (07-11) | ✅ La stella polare. Coerente con tutti i dati: 240 attivi/token (1.6 GB) entrano; l'unione-nel-tempo no → tieni il set attivo + prefetcha il churn. |
| I4 | **Maschera DAL PROMPT, sempre in mutamento** — non pre-caricata/congelata. (dall'inizio; esplicito 07-11) | ✅ Pre-caricata → mismatch-di-dominio → collasso; le fasi ruotano ~78%. Statica = ripiego (la rotazione cieca collassava), non l'obiettivo. |
| I5 | **Pinna i top-k e RUOTA la residenza** (depenna i freddi). Il senso originale della "rotazione". (REAP-LOOP, dall'inizio) | ✅ Divergenza confermata: mai implementato — si ruotava solo la MASK (selezione→collasso), mai la residenza. La 0031 pin-keep è il primo pezzo. |
| I6 | **Rotazione CONDIZIONALE:** "riesci senza il set completo? se no ruota e paga latenza, se sì vai avanti." (K91) | ✅ La rotazione a-calendario collassa (Legge dell'Ancora); quella condizionata sulla COMPETENZA no (tocca solo traiettorie che stanno fallendo). Segnale = entropia (lead +57), non S1. |
| I7 | **Non usiamo tutti gli esperti insieme** — è la domanda da farsi. (07-11) | ✅ 240 attivi/token (6×40 layer), COSTANTE per ogni K e dominio, full compreso. I "989" sono l'unione nel tempo. Ribalta il muro "non entra". |
| I8 | **Gli esperti condividono le skill** → per fase basta un sottoinsieme che copre, non K0. (07-11) | ✅ Mask domain-matched RENDE (cyberDOM K48). Il churn vive in un pool piccolo (miss intrinseco <2% su 64 token). |
| I9 | **Un "termometro" di coerenza per giustificare la rotazione.** (K91) | ✅ È nato il sensore S1; l'entropia dà ~+57 tok di lead pre-collasso. L'utente ha inventato il sensore. |
| I10 | **SPEX = lista-della-spesa / fattorino:** predici e pre-consegna gli esperti. (TUTOR) | ✅ È esattamente l'architettura SPEX (predittore hidden→L+1 + prefetch async). Manca solo lo stream async per renderla utile (il blocking-sync è il nodo). |

## 2. APERTE / IN COSTRUZIONE — la strada davanti, che discende dalla spina

- **Pipeline ASINCRONA** (nascondi il churn dietro il compute; compute-stream + copy-stream). *Piano code-level pronto:* `ASYNC_PIPELINE_PLAN.md` — diagnosi: il prefetch è emesso a fine-L e drenato dalla barriera per-layer (`cudaDeviceSynchronize`) prima dell'overlap → copia esposta (ecco perché SPEX rallentava). Fix chirurgico S1 (evento invece di sync + prefetch anticipato in cache LRU + `cudaStreamWaitEvent` + rilassa la barriera). È ciò che rende SPEX finalmente conveniente.
- **DUE predittori complementari** (utente 07-11, confermato dai dati): (a) **RECENCY / temporale** — tieni residenti gli esperti degli ultimi ~10-20 token perché rientrano (hit ultimi-16-tok: K23 0.97, cyberDOM 0.89). Cheap, accurato DENTRO una fase; cieco alle transizioni. Eviction = "return-aware weighted LRU" (non buttare chi ha appena sparato). (b) **SPEX / cross-layer** — predice L+1 dallo stato nascosto di L; cattura il BORDO della transizione prosa→codice (l'hidden già "pende" verso il codice) che il recency sbaglia. Si coprono a vicenda: recency tiene calda la fase, SPEX prende la transizione. NB dinamica utente: al cambio prosa→codice paghi un PICCO DI LATENZA (78% rotazione), poi recuperi in codice-puro — ma è latenza (fetch), NON collasso; amortizzato su fase lunga = trascurabile.
- **Controller di residenza VIVA / ammissione condizionale** (I4+I6): parte dal prompt, segue le fasi, non restringe mai la selezione, ammette 2-5 esperti/layer quando l'entropia sale. Spec di targeting pronta.
- **LOCK del tail-freddo per velocità** (utente 07-11, MISURATO scratchpad/coldtail.py): "se un 5% mancasse il router si arrangia o si blocca?" → SI ARRANGIA (softmax si ridistribuisce sui disponibili; NON si blocca — è come funziona la REAP-mask; un expert selezionato-ma-non-fetchabile invece lo pianterebbe → la porta-chiusa/mask è il modo SICURO). Quanto è libero chiudere la porta ai freddi (K0, stabile cross-domain): freddi 5%→0.1% sostituzioni, 25%→1.3%, **50%→7.5%**, 70%→19.6%. Il tail freddo fa ~zero output ma tanti STALLI (raro=freddo=miss) → chiudergli la porta = sostituisci con un caldo residente = velocità a costo-qualità ~zero. CAVEAT: "freddo" va misurato DINAMICO per-fase (un freddo-totale è caldo-in-CSS); statico → collasso. Soluzione = rating dinamico + rmass (visibile a porta chiusa via bias-mask 0011) per RIAPRIRE = il controller I6 dal lato lock.
- **Rating → promozione → espulsione** (utente 07-11): rating = EWMA di partecipazione (sale se spara, DECADE se tace) = il "candidato"; promozione = pin del rating alto; espulsione = il decadimento espelle da solo (gli expert di prosa decadono mentre sei nel CSS → liberano slot). Manopola = velocità di decadimento (lenta=stantii, veloce=paghi re-fetch). **Già lo scheletro della 0031** (demand-EWMA + rotazione CUSUM) + "prediction-aware weighted LRU" dello spec SPEX. a720d5: un rating-frequenza batte LRU-puro (marginale, ma sui wide).
- **BACKBONE per condivisione + soft (non hard-pin)** (utente 07-11, MISURATO scratchpad/backbone.py): il rating giusto per il budget minuscolo NON è l'unione (apparso ≥1 volta) ma la CONDIVISIONE (presente in ≥T degli ultimi 20 token) = il backbone-della-fase vs i transienti. K12: unione 445 exp/2.9GB/hit0.99 → backbone(≥5/20) **354 exp/2.3GB/hit0.93** = ENTRA nel b9, scarti i transienti e perdi solo 0.06. K23 backbone 389/2.6GB/0.82, K48 356/2.3GB/0.55 (troppo largo). "Non bloccare": a K12 il backbone ≈ tutta la cache → hard-pin rigido; meglio SOFT (top-N per punteggio-condivisione, ricalcolato → la residenza scorre con la fase). DESIGN COMPLETO 3060: K12 + residenza-soft-backbone (2.3GB, hit0.93) + async per il 7% churn → ~3-4 t/s. La partita resta la QUALITÀ a K12 (domain-match + ammissione-fase).
- **KV-headroom** — ✅ RISOLTO (split-VRAM 6301d05): il 3060 è **fixed-weight-bound, non KV-bound**. Pesi fissi ~8.85 GiB; KV già minima per costruzione (MLA/CSA, finestra 128 tok) → liberabili max ~0.3-0.5 GiB. Cache resta ~2-2.5 GiB. → **K12 entra (2.4 GiB, hit 0.92, ~3-4 t/s = il goal); K23 no (vuole 3.2 GiB, cappato ~1.5)**. KV-compression = vicolo cieco. Attaccare i pesi fissi (~5 GiB densi sempre-residenti) è dubbio (servono a ogni token). **Bersaglio realistico sul 3060 = K12.**
- **Two-phase W50 in-engine sul 3060** (I4, versione statica): warmup wide → freeze, no re-prefill. Leva velocità+qualità ~7× mai misurata sull'HW target.

## 3. CHIUSE con ragione FONDAMENTALE — non ri-proporre (ma rileggere: oggi qualcuna si raffina)

- **Pre-caricare in VRAM l'INTERO profilo** — ✗ tier troppo piccolo (profilo ~6.2 GB > 1-2 GB liberi). ⚠️ **RAFFINATA da I7:** il core (residenza-VRAM) era giusto — si pre-carica il set ATTIVO (240=1.6 GB) + prefetch del churn, NON l'intero profilo. La versione ingenua resta chiusa, quella raffinata è la strada (§2).
- **RAID SSD / più partizioni** — ✗ il collo è la latenza a QD1, non la banda; la leva è queue-depth (sorted+parallel reads), non più dischi.
- **Decomprimere/ricomprimere il 2-bit a K alto (DwarfStar adattivo)** — ✗ fisica dell'informazione: la quantizzazione persa non si recupera.
- **Mixed-precision q2 vs q2+q4-ultimi-layer** — ✗ RETRACTED: il soffitto è la TAGLIA del modello, non i bit (q2 e q2-q4 entrambi L1 sul task hard).
- **Rewind/rollback come recupero POST-collasso sui task wide** — ✗ REFUTATO-per-wide (0/12, meccanismo sano ma il contesto avvelenato vince). ⚠️ Status NARROW/erosione-lenta: ignoto, non testato, bassa priorità.
- **Staircase dinamico / learn-live in corsa** — ✗ avvelena la cache; la Legge dell'Ancora lo rafforza (cambio di membership in corsa destabilizza). *NB: distinto da I5/I6 che ruotano la RESIDENZA, non la selezione.*
- **Drafter MTP su CPU+RAM** — ✗ il gate VRAM=0 non scatta nella config daily; delta ~5% dentro rumore.

---

*Se leggi questo registro e stai per proporre un esperimento: prima chiediti se è già nella §1
(costruisci da lì), nella §2 (continua), o nella §3 (non ri-proporre senza evidenza nuova).*

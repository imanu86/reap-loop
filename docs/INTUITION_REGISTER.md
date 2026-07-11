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

- **Pipeline ASINCRONA** (nascondi il churn dietro il compute; compute-stream + copy-stream). *Piano di build a stadi in corso.* È ciò che rende SPEX finalmente conveniente.
- **Controller di residenza VIVA / ammissione condizionale** (I4+I6): parte dal prompt, segue le fasi, non restringe mai la selezione, ammette 2-5 esperti/layer quando l'entropia sale. Spec di targeting pronta.
- **KV-headroom** (in misura): quanto VRAM liberabile per la cache → decide se K23 entra nel regime sogno (2-2.5 t/s) o siamo cappati a K12.
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

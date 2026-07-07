# CLAIMS_CURRENT — single source of truth

**Ultimo aggiornamento: 2026-07-07 (retraction multiseed N=3)**

Questo file e l'UNICA fonte di verita sullo stato dei claim di reap-loop.
Se un altro documento del repo contraddice questa tabella, questo file vince.

| Claim | Stato | Evidenza / numero corrente | Dove vive |
|---|---|---|---|
| Asimmetria causale HOT/COLD nel pruning ("94% vs 69%", "causal asymmetry") | **RETRACTED** | Non replica. Multiseed N=3, mask verificata attiva: rep-rate hot [0.026, 0.026, 0.064] ~ cold [0.033, 0.045, 0.029]. Era artefatto di mask-inerte + n=1. | docs/paper/PAPER.md (retraction), storico esperimenti asimmetria |
| "23.6 t/s" | **RETRACTED** | Build confuso/crippled. Non piu headline, non citare come numero corrente. | vecchi README / note di run |
| "3.7x H2H" | **RETRACTED** | Build confuso/crippled. Non piu headline. Il contrasto valido e il paired rand/reap (vedi sotto). | vecchi README / note di run |
| "9.22 t/s" | **RETRACTED** | Build confuso/crippled. Non piu headline. | vecchi README / note di run |
| "near-lossless" riferito al LOOP | **RETRACTED** | Non qualificato per il loop. NB: distinto dal near-lossless del pruning statico F1, che resta valido (vedi ultima riga NUANCE). | vecchi README / note di run |
| "frontier / DC-scale unlock" (il metodo sblocca scala datacenter) | **RETRACTED come tesi** | Prior-art fino a 1026B: EASY-EP, PreMoE, REAP-Cerebras (arXiv 2510.13999). Posizione onesta = EDGE / single-stream, non frontier. | docs/paper/PAPER.md, vecchie intro |
| "temperature-per-expert quant = IL CONTRIBUTO CENTRALE" | **SUPERSEDED** | Framing pre-REAP-LOOP. Ora e idea-sola **[OPEN]**, non il centro del lavoro. | vecchia framing del paper |
| Attuatore mask REVERSIBILE | **CLOSED** | Attuatore mask reversibile con ZERO violation. Difendibile. | src/ attuatore mask, docs/paper/PAPER.md |
| Contrasto PAIRED rand/reap | **CLOSED** | 1.345x CI[1.270, 1.423], NON sovrapposto a reap/full. Stessa GPU, confound-clean. | docs/paper/PAPER.md, log run paired |
| static keep-23 come SPEED DIAGNOSTIC | **CLOSED** | 11-17 t/s su pod 3090, come diagnostica di velocita. NON generalizzato. | docs/paper/PAPER.md, log pod 3090 |
| Control B (ordinamento) | **CLOSED** | session 1.06x / domain 4.80x / random 7.02x. | docs/paper/PAPER.md, Control B |
| reap/full | **OPEN** | 1.009x CI[0.972, 1.035]. Il CI attraversa 1.0 -> NON dire "near-lossless" secco. | docs/paper/PAPER.md |
| Staircase DINAMICA | **OPEN** | Lenta / cache-poisoned (2.5 t/s). [ENG-BUG, path dinamico OFF]. | src/ path dinamico, note ENG-BUG |
| t/s assoluto locale sul 3060 | **OPEN** | I/O-bound ~1.2 t/s. cache-reserve default 6GB su 12GB disabilita la cache esperti; reserve=1 la riattiva ma resta I/O-bound perche working-set 920 esperti > cache VRAM ~750 + RAM cap. | note locali 3060 |
| SPEX-dense (oracolo denso -> prefetch) | **OPEN** | [NON TESTATO]. | docs/SPEX_spec.md |

---

**NUANCE (non confondere con l'asimmetria ritrattata):**
Il finding **F1** — pruning statico ~50% esperti su dominio, saliency >> random — e **PRIOR-ART VERO e resta CLOSED/valido** (30B K50 ppl 5.50 vs full 5.56; EASY-EP / PreMoE / REAP / NAEE). Il "near-lossless" di F1 e legittimo; il "near-lossless" ritrattato qui sopra si riferisce SOLO al loop.

---

**REGOLA: nessun doc del repo deve affermare come CORRENTE un claim marcato qui RETRACTED/SUPERSEDED.**

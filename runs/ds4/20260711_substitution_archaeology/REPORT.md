# Substitution archaeology — was expert substitution already tried? (2026-07-11)

Mining di tutti i 22 transcript Codex (2026-07/05..11) + docs repo per: exchange, substitut*, skip-on-miss, similarity, fallback. Nessun hit discute mai "sostituisci l'expert mancante con un residente simile".

## (a) Cosa fu provato davvero — 4 meccanismi, nessuno è substitution
1. **K23 raw-router ROTATE** (DS4_PACE_ROTATE): cambia la *membership* della mask periodicamente — non sostituisce per-occorrenza.
2. **PACE breath "exchange" accounting** (DS4_PACE_EXCHANGE_OBSERVE, 0bdad9a): bookkeeping promote/demote observe-only, mai attuato.
3. **CQ1 cold sidecar** (J34-J38): al miss carica una copia compressa dello STESSO expert, non uno diverso.
4. **skip-on-miss** (moe HANDOFF, leva 5): proposto (skip, non substitute), status DESIGN, mai eseguito.

## (b) Scoring
Solo il rotate toccava "chi rimpiazza chi": scoring = **massa router grezza EWMA** (rmass, decay 0.98) — domanda/frequenza, MAI similarità. Nessun grafo di similarità su score REAP è mai stato costruito.

## (c) Esiti
Rotate respinto due volte (pod 2026-07-09; M1a n=3 ctx8192). Causa nota (LEVER_RETROSPECTIVE AN-1): *a parità di coverage, static sopravvive e rotate collassa* → "scambiare l'identità degli esperti rompe la continuità hidden/KV che la mask congelata preserva. Il costo non è coverage, è discontinuità." CQ1 abbandonato per ragione indipendente (path sincrono). Exchange/skip mai corsi.

## (d) Confronto con survey #2 (MOE_ECOSYSTEM_SURVEY §4.2)
La substitution del survey è: K **costante**, per-occorrenza, solo su miss bottom-percentile, scoring = **grafo di similarità su score REAP**. Meccanismo E scoring entrambi diversi dal rotate bocciato.

## (e) Verdetto
**Mai provata la substitution vera → retrial GIUSTIFICATO, non doppione.** Il ricordo dell'utente ("provata con scoring errato") corrisponde al rotate (membership-change, mass-scored). Rischio residuo: AN-1 è analogia diretta (swap identità = discontinuità) che il nuovo scoring non falsifica a priori — MA la substitution non tocca la membership (ancora intatta): il meccanismo di danno AN-1 potrebbe non applicarsi.
**Retrial minimo (P4)**: (1) check OFFLINE gratis: legge drop-vs-similarità dai trace pesati esistenti (co-attivazione expert-expert → similarità; per i want fuori-mask, distribuzione della similarità del miglior kept) — se i want sono ortogonali ai kept, morta lì; (2) solo se promettente: runtime K23-costante n=3 ABAB vs baseline L-STATIC/L-ROTATE su prompt wide, watching il segnale-AN-1 (repeat/S1), non solo L mediano.

*Report prodotto da mining agent (Write-guard: committato dal coordinator).*

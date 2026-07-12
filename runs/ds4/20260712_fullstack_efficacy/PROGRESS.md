# Efficacia end-to-end full-stack (0034→0043) — K8-per-massa con pin+fattorino

Obiettivo: provare che lo stack COMPLETO (produttore mass+pressione + consumatore
thrash-fix+pin-by-mass+fattorino) su un prompt ESIGENTE (non il micro-caso del produttore)
RENDE e sta nel budget (≤400 slot), con `DS4_REAP_PIN_BY_MASS=1` + `DS4_REAP_PREFETCH_DELTA=1`.

Branch integrazione: livemask-dynamic-2026-07-12 @ b1937ff (stack 0034→0043).

## Vincoli
- POCHI CREDITI → economico, step piccoli, COMMIT lungo la strada (resumabile).
- Il K8-per-massa del produttore era su prompt/output MINUSCOLO → qui prompt esigente multi-fase.

## Base scelta (economica)
Nessun tree WSL ha tutto. `/root/ds4-0039-work` = lato produttore (0035/0039/0040, pin=9).
`/root/ds4-cachefix` = lato consumatore (0042/0043) ma senza 0039/0040 pieni.
→ Costruisco `/root/ds4-fullstack` = COPIA di ds4-0039-work + applico i 3 patch consumatore
(0041 thrash-fix cuda, 0042 pin-consumer cuda, 0043 fattorino ds4.c).

## Envs efficacia
`DS4_PACE_LIVEMASK=1 DS4_PACE_LIVEMASK_K=8 DS4_REAP_PIN_BY_MASS=1 DS4_REAP_PREFETCH_DELTA=1`
(+ pressione 0039 opzionale). Mass ranking attivo (seed+candidato per lm_wshare).

## Prompt esigente (multi-fase: struttura→CSS→JS)
Landing page food-delivery con nav, hero+CTA, 3 feature, form con validazione JS, popup, footer.
~600 token. Stressa le transizioni di fase (dove il K8-per-freq collassava).

## Metriche
close_html, chars, coerenza · slot usati (≤400?) · t/s warm · pin_admits/eviction · fattorino fires.

## STEP LOG
- [ ] checkpoint 0: piano committato
- [ ] checkpoint 1: build fullstack (apply 0041/0042/0043 su 0039-work + compile) → md5, esito
- [ ] checkpoint 2: run efficacia (prompt esigente, full-ON) → risultati
- [ ] checkpoint 3: analisi + verdetto

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

## SCOPERTA (integrazione)
Applicare i patch consumatore (0041/0042) su ds4-0039-work FALLISCE: 1 hunk ciascuno nella
regione sticky-pin del cache-loader (basi cuda divergenti tra produttore e consumatore).
SOLUZIONE PULITA (rispetta la divisione produttore=ds4.c / consumatore=cuda): **swap-file, non
merge-hunk**. ds4_gpu.h + ds4.h + Makefile IDENTICI tra i due tree → si può prendere il cuda
COMPLETO del consumatore e il ds4.c COMPLETO del produttore. Metodo di ricostruzione:
  /root/ds4-fullstack = copia di ds4-0039-work (ds4.c produttore 0035-0040) + `patch 0043` (ds4.c)
                        + `cp ds4-cachefix/ds4_cuda.cu` (consumatore 0034+0036+0041+0042).
Build: `make cuda CUDA_ARCH=sm_86`. Binario **md5=0d97e5705d0d**, 0 errori 0 warning.

## STEP LOG
- [x] checkpoint 0: piano committato
- [x] checkpoint 1: build fullstack OK (md5 0d97e5705d0d, /root/ds4-fullstack, sm_86, 0 err/warn)
- [ ] checkpoint 2: run efficacia (prompt esigente food-delivery, full-ON) → risultati
- [ ] checkpoint 3: analisi + verdetto

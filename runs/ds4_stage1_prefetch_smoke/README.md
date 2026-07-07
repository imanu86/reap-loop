# DS4 Stage 1 Prefetch Smoke

Checkpoint del 2026-07-05 sul setup locale WSL/DwarfStar:

- modello: `/root/models/ds4-2bit.gguf`
- path: `--cuda --ssd-streaming`
- base ds4: `80ebbc3` + patch `0001` + `0002` + `0003`
- branch WSL: `/root/ds4`, commit `8c8df19`

## Risultati

`event_only_decode.log` usa solo lo Stage 1a selected-upload event:

- prefill: `0.58 t/s`
- generation: `0.70 t/s`
- selected sync: `216` calls / `25.539 ms`

`prefetch_l1_decode.log` abilita anche `DS4_SPEX_PREFETCH_NEXT_LAYER=1`:

- prefill: `0.58 t/s`
- generation: `0.43 t/s`
- cache hit rate: `0.0000`
- selected sync: `216` calls / `25.180 ms`

## Interpretazione

Lo Stage 1 L+1 e' volutamente un predictor minimo: riusa gli expert selected del layer L per prefetchare
il layer L+1. Sullo smoke locale non produce hit e peggiora la generazione perche' aggiunge copie senza
vantaggio. Va considerato solo uno scaffold async verificato, non una ottimizzazione.

La prossima modifica utile e' collegare il predictor SPEX/Markov reale a questo hook.

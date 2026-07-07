# DS4 Stage 2 Markov Loader Smoke

Checkpoint del 2026-07-05 sul setup locale WSL/DwarfStar:

- modello: `/root/models/ds4-2bit.gguf`
- path: `--cuda --ssd-streaming --ssd-streaming-cold`
- base ds4: `80ebbc3` + patch `0001` + `0002` + `0003` + `0004`
- branch WSL: `/root/ds4`, commit `0d9b8d3`
- file SPEX usato nello smoke: `/root/ds4_synthetic_flash.spex`

## Risultato

`markov_synthetic_decode.log` verifica che:

- il loader `.spex` Markov carica una sola volta una shape Flash compatibile: `L=43`, `E=256`, `topN=6`
- il nuovo path produce prefetch `SPEX markov` sul hook Stage 1b
- il run arriva a fine generazione sul modello reale locale

Metriche dello smoke:

- prefill: `0.57 t/s`
- generation: `0.44 t/s`
- cache hit rate: `0.0037`

## Interpretazione

Questo e' un test di plumbing, non una misura di qualita' predittiva: il file `.spex` e' sintetico, non
derivato da trace routing V4-Flash. Serve solo a verificare parser, validazione shape, admission `tau/cap`
e chiamata al prefetch async.

La prossima misura utile richiede un `.spex` esportato da trace V4-Flash reale.

# DS4 Routing Trace Smoke

Checkpoint del 2026-07-05 sul setup locale WSL/DwarfStar:

- modello: `/root/models/ds4-2bit.gguf`
- path: `--cuda --ssd-streaming --ssd-streaming-cold`
- prompt: `Conta da 1 a 200:`
- decode target: `-n 64`
- env: `DS4_SPEX_TRACE_ROUTING=/tmp/ds4_routing_trace_count64.csv`, `DS4_SELECTED_UPLOAD_EVENT=1`
- ds4 WSL commit: `bee9eb3`

## Trace

`routing_trace_count64.csv` contiene 2600 righe dati:

- posizioni decode: `17..81` (`65` token)
- layer routed: `3..42` (`40` layer)
- colonne: `pos,layer,n,e0,e1,e2,e3,e4,e5`

## Nearby-Token Finding

Summary da `scripts/analyze_ds4_routing_trace.py`:

- same-token previous-layer top6: `0.0245`
- previous-token same-layer top6: `0.2623`
- window4 same-layer top6/top12: `0.3420` / `0.5068`
- window8 same-layer top6/top12: `0.3656` / `0.5495`
- previous-layer + window8 top6/top12: `0.3565` / `0.5313`

Interpretazione: sul DS4 Flash reale, il segnale utile non e' lo scaffold L->L+1 nello stesso token.
Il segnale forte e' temporale, per stesso layer sui token recenti. La prossima variante SPEX live dovrebbe
prefetchare per layer usando una finestra degli ultimi token, non solo il layer precedente dello stesso token.

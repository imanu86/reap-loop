# 2026-07-11 podA — narrow weighted traces (FULL model, width-sensor anchors)

Mandato coordinator: trace routing PESATO (`DS4_SPEX_TRACE_ROUTING` +
`DS4_SPEX_TRACE_ROUTING_WEIGHTS=1`) sul modello FULL (nessuna mask REAP),
task STRETTI, >=150 token generati, per l'ancora-stretta del fit del
sensore-larghezza (docs/DECISION_MODEL.md).

## Setup

- Pod: RunPod community RTX 3090 `ysegg4bx67yvr3` (machine `hyuu5efkuyma`,
  $0.22/h, 128 vcpu / 251 GB RAM ⇒ regime RAM-hot), image
  `runpod/pytorch:1.0.7-cu1290-torch280-ubuntu2404`, CUDA gate-check PASS al
  primo colpo (driver 580.65.06, cudaGetDeviceCount rc=0 count=1).
- Binario: `ds4_sm86_livetree-771a39a8` da cache R2, sha256 `772c502f…`
  verificato = lineage post-0018 del gruppo W50 locale. Modello
  `ds4-2bit.gguf` sha256 `efc7ed60…` verificato.
- Comando per cella: `ds4 --cuda --ssd-streaming --ssd-streaming-cold
  --ssd-streaming-cache-experts 256 -c 4096 --nothink --temp 0 -n 300
  --prompt-file <prompt>` con trace pesato attivo; env IO
  `DS4_CUDA_NO_DIRECT_IO=1 DS4_CUDA_KEEP_MODEL_PAGES=1
  DS4_REAP_PREFETCH_THREADS=16 DS4_REAP_PREFETCH_LOCK=1`.
- Greedy, FULL (no mask), trace `route.csv` con colonne
  `pos,layer,n,e0..e5,w0..w5` (40 layer MoE per posizione).

## Celle e conteggio token (righe route / 40 layer)

| cella | prompt | tok generati con trace | >=150? | note |
|---|---|---:|---|---|
| a_coffee_full | coffee compatto (819 B) | **299** | SI | -n 300, budget pieno |
| b_json_full | JSON 1 record (corto) | 37 | no | EOS naturale precoce — tenuto come ancora corta |
| c_python_full | 1 funzione palindromo | 117 | no | EOS naturale — tenuto come ancora corta |
| b2_json_long_full | JSON array 6 record | **206** | SI | v2 esteso, dominio invariato |
| c2_python_long_full | 3 funzioni con docstring | **235** | SI | v2 esteso, dominio invariato |

Vincolo coordinator (>=150 tok con pesi) soddisfatto su 3 celle / 3 domini
(HTML-coffee, JSON-extraction, Python): 299 / 206 / 235 token. Le due celle
corte v1 restano nel dataset come punti addizionali (il fit puo' usarle o no).

Output funzionalmente corretti a vista: coffee HTML aperto correttamente,
JSON array valido 6 oggetti, 3 funzioni Python corrette con docstring
(gen.txt in ogni cella; qualita' non e' l'oggetto di questo run — servono i
route.csv pesati).

t/s marcati POD (RAM-hot 3090, non confrontabili col 3060 locale):
p2/gen ~0.35-2.6 t/s a seconda della cella (diag.txt per-cella).

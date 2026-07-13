# CUDA pinned arena probe - risultati 2026-07-13

## Domanda

Verificare se `cudaHostAlloc` sotto WSL2 supera il muro di circa 30-31 GiB
osservato con `cudaHostRegister`, consentendo un'arena DMA da 50 GiB.

## Setup

- RTX 3060 12 GiB, CUDA WSL;
- host 64 GiB RAM;
- WSL cap 62 GB, circa 60 GiB visibili;
- nessun `ds4-server` attivo;
- modalita' `blocks`, allocator `cudaHostAlloc`;
- target 50 GiB, step 1 GiB;
- sparse touch ogni 64 MiB nel run storico;
- device buffer e staging pinned aggiuntivo da 16 MiB;
- H2D asincrona e verifica checksum.

## Risultato

| Misura | Valore |
|---|---:|
| Allocazione riuscita | 31 GiB |
| Primo fallimento | 32esimo GiB |
| Errore | `cudaErrorMemoryAllocation` |
| WSL MemAvailable al fallimento | 28.310810 GiB |
| Windows available minimo | 15.470 GiB |
| Staging extra dopo il fallimento | riuscito |
| H2D asincrona dall'arena | riuscita |
| Checksum round-trip | identico |
| Cleanup | completo |
| Exit code rispetto al target 50 | 4 / fallito |

`MemAvailable` e' scesa quasi uno-a-uno con i GiB allocati, coerentemente con
memoria fisica page-locked nonostante il touch sparso. Conclusione misurata:
`cudaHostAlloc` non supera il muro WSL; il limite appare
intorno a 31 GiB anche se rimane molta RAM disponibile sia nella VM sia
sull'host. Il test non sostiene l'uso di un'arena pinned da 50 GiB sotto WSL2.

## Artefatti

- `probe_20260713_072839_blocks_hostalloc_50g.jsonl`: log autoritativo;
- `probe_20260713_072839_blocks_hostalloc_50g.windows_memory.csv`: monitor host;
- `probe_20260713_072839_blocks_hostalloc_50g.stderr.txt`: stderr;
- `probe_20260713_072839_blocks_hostalloc_50g.rc`: exit code;
- `probe_20260713_072839_blocks_hostalloc_50g.runner.sh`: comando WSL.

## Caveat

Il monitor PowerShell ha serializzato i decimali con la locale italiana e ha
prodotto righe come `47,369,`, incompatibili con l'header CSV a tre colonne.
I valori Windows riportati qui sono stati ricostruiti dalla parte intera e
frazionaria. Prima del prossimo run il runner deve usare cultura invariant.

Il run a blocchi misura il tetto cumulativo e, da solo, non prova la massima
allocazione contigua utile al runtime. I bracci single-allocation che chiudono
questa lacuna sono riportati sotto.

## Follow-up contiguo

Dopo la correzione a cultura invariant del runner e il passaggio al touch di
ogni pagina da 4 KiB:

| Target singolo | Esito | Windows minimo | H2D check | Artefatto JSONL |
|---:|---|---:|---:|---|
| 24 GiB | PASS | 21.415 GiB | 2.971581 GiB/s | `probe_20260713_074806_single_hostalloc_24g.jsonl` |
| 28 GiB | PASS | 17.112 GiB | 3.028627 GiB/s | `probe_20260713_074901_single_hostalloc_28g.jsonl` |
| 30 GiB | PASS | 15.306 GiB | 3.018815 GiB/s | `probe_20260713_075005_single_hostalloc_30g.jsonl` |
| 31 GiB | PASS | 14.417 GiB | 3.024387 GiB/s | `probe_20260713_075118_single_hostalloc_31g.jsonl` |

Ogni PASS comprende arena, staging pinned aggiuntivo da 16 MiB, device buffer,
H2D asincrona, D2H, checksum identico e cleanup. Il test da 31 GiB prova la
capienza tecnica contigua, non che 31 GiB siano prudenti dentro DS4.

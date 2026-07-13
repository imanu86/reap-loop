# Limite pinned CUDA WSL/WDDM - 2026-07-13

## Verdetto

Il muro misurato a 31 GiB non deriva dalla RAM concessa a WSL, da `ulimit`,
dallo swap o dalla page cache. Corrisponde al massimo WDDM di RAM fisica
disponibile alla grafica su un host Windows da 64 GiB.

Microsoft documenta il calcolo:

```text
TotalSystemMemoryAvailableForGraphics = max(TotalSystemMemory / 2, 64 MiB)
```

Fonte primaria:
[Microsoft - Calculating Graphics Memory](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/calculating-graphics-memory).

Sul sistema misurato:

```text
RAM fisica Windows      68,601,917,440 byte = 63.8905 GiB
meta' RAM fisica        34,300,958,720 byte = 31.9453 GiB
arena riuscita                               31.0000 GiB
residuo nominale                              0.9453 GiB
blocco successivo richiesto                   1.0000 GiB -> fallisce
staging successivo                            0.0156 GiB -> riesce
```

Il match spiega sia il fallimento del 32esimo blocco sia il successo dello
staging da 16 MiB. Al fallimento erano ancora disponibili 28.31 GiB dentro WSL
e almeno 15.47 GiB su Windows: non era un OOM generale.

## Dove viene imposto

WSL inoltra le richieste GPU al lato Windows tramite GPU-PV/dxgkrnl. Il ramo
Linux usa `pin_user_pages_fast(..., FOLL_LONGTERM)` e invia i PFN al lato host;
non dipende da `mlock`, coerentemente con `Mlocked=0` e `ulimit -l=64MiB`.

Sorgente primaria:
[WSL2 Linux kernel - dxgvmbus.c](https://github.com/microsoft/WSL2-Linux-Kernel/blob/linux-msft-wsl-6.18.y/drivers/hv/dxgkrnl/dxgvmbus.c#L1438-L1565).

Il limite finale e' nel budget Windows VidMm/KMD. `SharedSystemMemory` puo'
essere ulteriormente ridotta dalle capability del driver, ma non supera il
massimo di sistema documentato. Riferimento:
[DXGK_DRIVERCAPS](https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmddi/ns-d3dkmddi-_dxgk_drivercaps).

NVIDIA dichiara esplicitamente che la pinned system memory disponibile alle
applicazioni WSL e' limitata:
[CUDA on WSL - Known limitations](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications).

Una discussione tecnica NVIDIA riporta lo stesso limite tipico del 50% su
Windows 10/11 e nessun meccanismo supportato per modificarlo:
[NVIDIA - 50% cudaHostAlloc limit](https://forums.developer.nvidia.com/t/change-limit-of-50-for-cudahostalloc-pinned-memory-on-windows-10-11/228235).

## Leve escluse

| Leva | Esito |
|---|---|
| `.wslconfig memory/swap` | cambia VM e swap, non il budget WDDM |
| `ulimit -l` / `mlock` | non e' il percorso usato da dxgkrnl |
| pagefile Windows | aumenta commit generale, non RAM fisica pinnabile |
| custom kernel WSL | il cap non e' nella costante guest |
| HAGS | cambia scheduling, non la formula VidMm |
| Resizable BAR | espone VRAM alla CPU, non piu' RAM host alla GPU |
| TCC | non disponibile normalmente su RTX 3060 GeForce; WSL usa WDDM |
| registro non documentato | nessuna chiave supportata trovata; rischio alto |

## Conseguenze progettuali

Con 64 GiB fisici non e' realistico ottenere 50 GiB pinned sotto WSL o Windows
nativo. Le opzioni tecniche sono:

1. arena pinned dinamica da 24-28 GiB e secondo tier pageable servito da WRAP;
2. Linux nativo sullo stesso hardware, da misurare con guard 12-16 GiB;
3. 128 GiB di RAM fisica, che porta il tetto nominale WDDM a circa 64 GiB.

Il probe Windows nativo resta utile per confermare il comportamento WDDM e
leggere `SharedSystemMemory`, ma non e' piu' una soluzione attesa al requisito
50 GiB. Un port DS4 nativo Windows non va iniziato sulla sola speranza di
superare questo limite.

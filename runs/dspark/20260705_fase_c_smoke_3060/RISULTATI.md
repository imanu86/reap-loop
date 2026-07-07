# Fase C — sblocco MTP+streaming: FUNZIONA (smoke 3060 reale, 2026-07-05 sera)

**Dove**: 3060 12GB workstation (WSL2), regime DwarfStar (2-bit, `--cuda --ssd-streaming`),
worktree `/root/ds4-dspark` branch `dspark/unlock-streaming` = bee9eb3 + patch **0009**
(guardia env-gated + fix upstream PR#497) + patch **0010** (slot registrazione durevole
per la mappa del support model). GPU in handoff esclusivo dal track SPEX (sweep finito
18:45), `DS4_SPEX_STATS=1` su tutti i run come richiesto. Log: `out/`.

## Il risultato storico
**Primo speculative decoding MTP in streaming mai eseguito su ds4**: 33-34 cicli di
verifica speculativa per run da 100 token, zero crash, zero fallback sequenziali.

| evidenza | valore | fonte |
|---|---|---|
| cicli verify (draft 2, margin 0) | 33-34/run, stabili su 4 run | `spec2_*.log` (mtp conf) |
| acceptance pos2 on-device (code) | 0.848 e 0.824 (28/33, 28/34 full-accept) | conf lines |
| union-load nel VERIFY | 430 load `slots=12` (blocco-2) + 538 `slots=6` (single, vive grazie a PR#497) | `spec2_verbose.log` |
| costo draft per ciclo | ~21-33 ms (MTP zero-copy dallo slot 0010) | mtp timing |
<!-- redacted: internal cost/infra note -->

## t/s (onesti, con i loro caveat)
| coppia | baseline | spec2 | delta | note |
|---|---|---|---|---|
| smoke (cache OFF su entrambi) | 186s/181s wall | 168s/166s wall | **~+10%** | 100 tok, stessa sessione |
| fair (stessa sessione, 250 exp) | 0.65 t/s (cache ON) | 0.71 t/s (cache OFF!) | **+9%** | spec HANDICAPPATO |

Caveat dichiarati: (a) i baseline variano 0.64→0.97 t/s con lo stato page-cache tra
sessioni (28GB RAM, playbook): i confronti validi sono solo same-state, come sopra;
(b) in TUTTI i run spec l'expert cache era disabilitata (vedi sotto): questo +9/10% è
il PAVIMENTO del guadagno, non il tetto.

## Il collo VRAM scoperto (prossimo innesto, 0011)
Nel run spec le statics device-cachate arrivano a **8.2 GiB** (main ~4 + **MTP ~3.5-4**)
su 12GB → `available 0.00 GiB <= reserve` → expert cache spenta e q8-fp16 cache esausta
(`spec2_fair.log`). Il baseline (senza MTP) tiene la sua cache: il confronto è impari a
SFAVORE dello spec. Fix candidato **0011**: non device-cachare le statics del support
model sotto streaming (con lo slot 0010 legge già zero-copy a ~25ms/draft) → libera
~3.5GB per l'expert cache del main. Manopole trovate:
`DS4_CUDA_STREAMING_EXPERT_CACHE_RESERVE_GB` (default 6; abbassata a 4 non basta perché
il ladro è la cache statics), `DS4_CUDA_Q8_F16_CACHE_RESERVE_MB`.

## Bug upstream trovati oggi (entrambi con patch nostra, upstream-worthy)
1. **PR#497 confermata indipendentemente**: senza il fix, le encode single-position del
   verifier saltano la selected-load → OOM (visto in v2). Incorporata e accreditata in 0009.
2. **Slot di registrazione mappa singolo** (0010): ogni remap streaming del main
   de-registra la mappa MTP (`cuda_model_set_host_map` → `cudaHostUnregister`); i
   fallback per-range su WSL2 falliscono (DMA da mmap file-backed = invalid argument)
   → drafter morto silenziosamente (100/100 cicli falliti). Su Linux nativo il fallback
   per-range maschera il bug (per questo il pod funzionava). Fix: slot secondario
   durevole con handoff di proprietà.

## Prossimi passi (in ordine)
1. **0011**: statics del support model non cachate sotto streaming → misura spec2/spec4
   con expert cache VERA (qui vive il grosso del 49%).
2. Scheduler (innesti 1-4 del design) con STS: `sts_params.json` già pronto.
3. Validazione paletto 3 completa: ds4-eval appaiato ON/OFF (rimandata: run lunghi,
   GPU restituita a fine sessione come da accordo).
4. Numeri paper-grade: run cold/warm etichettati, mediane su ≥3 run, dominii code/math/chat.

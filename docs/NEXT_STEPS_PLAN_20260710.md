# Piano operativo post-revisione — 2026-07-10

Esito della revisione trasversale (repo reap-loop + moe-aggressive-commit + transcript Claude/Codex 5→10 luglio).
Obiettivo finale invariato: **>3 t/s stabili sul 3060 locale con qualità coding/HTML ≥L2 al render** — nessun numero pod come headline.

Diagnosi in una riga: velocità e qualità sono state finora ottimizzate su assi separati; l'unica ricetta che le ha tenute
insieme (session-learning two-phase W50, pod: fase2 14.6 t/s con pagina funzionante) non è mai stata portata in-engine
sul 3060, e ogni verdetto recente poggia su n=1 + grading a regex.

## Fase 0 — Igiene (stato)

- [x] Push `cascade-memory/harness` (+31) e `reap/k91-coding-vram` (+41, include `runs/reap/multiseed_2026-07-07/`) su origin moe.
- [x] Commit + push runs 2026-07-09 in reap-loop (`36a81b8`); worktree k91 ripulito (`c8a7569`); dirty moe committati (`3908b3b`).
- [x] Cancellati branch locali merged (4× claude/*, reap/ds4-domain-prune, reap/public-domain-replica, codex/spex-integration).
- [x] **CLAIMS_CURRENT.md da riconciliare** (fatto e78cd3d): CLAIM-006 dice ancora "W50/130 L3 CLOSED" ma il replay pod 2026-07-10 dà W130 FAIL
      (confound re-prefill/freeze-point, vedi J44). Aggiornare a OPEN con nota knife-edge.
- [x] **PACE_DESIGN.md §4** (fatto e78cd3d) da aggiornare col limite misurato: n-gram è sensore in ritardo (J47/J48); il segnale con preavviso
      (~200 tok) è slope-S1 (CLAIM-011), non ancora cablato.
- [x] **Fonte canonica del paper** (fatto e78cd3d: PAPER.md CANONICAL, draft moe FROZEN, pointer multiseed): proposta = `reap-loop/docs/paper/PAPER.md` (più avanti di 2 commit); congelare
      `moe/docs/paper/PAPER_DRAFT_v2.md` con un puntatore. Nota: entrambi citano `runs/reap/multiseed_2026-07-07/` che vive
      in moe branch `reap/k91-coding-vram` (ora pushato).
- [x] **Mappa patch** (fatto e78cd3d: patches/README.md canonico + nota gemella in moe): 0009/0010 esistono in revisioni divergenti tra i due repo; collisioni di numerazione 0011 (mask REAP vs
      MTP dspark) e 0015/0016 (pace vs spex vs wiring). Aggiungere un README in `patches/` che dichiara la serie canonica.
- [x] moe `HANDOFF.md` (fatto moe c1c55d4: §0-LIVE 2026-07-10, PAPER_STATE superseded, README) §0-LIVE stantio (non menziona reap-loop/harness/k91); `PAPER_STATE.md` fermo alla tesi pre-REAP-LOOP.

## Fase 1 — Pavimento di misura (bloccante: senza questo ogni vincitore è indistinguibile dal rumore)

1. **Grading funzionale nel runner**: render/parse DOM → scala L0-L3 per ogni run HTML (riusare `functional_grade.py` dal
   branch k91 di moe). Fix noto: `has_popup` oggi matcha substring presenti nel prompt (eco = feature).
2. **n=3 di default con mediane** e ordine alternato ABAB (rumore 3060 documentato ±50%; repeat_flag flippa sulla stessa config).
3. **Controllo positivo**: full/no_pace a 800 tok sul 3060, prompt cyberpunk (~15 min). Oggi "K23 rompe l'HTML" non è attribuibile.
4. **Set prompt**: ≥3 HTML (cyberpunk, coffee-shop compatto, uno nuovo) + ≥2 code. Tutta la tesi qualità deriva da 1 prompt HTML.

## Fase 2 — Test decisionali (in ordine; protocollo Fase 1 obbligatorio)

| # | Test | Decide |
|---|---|---|
| T1 | Full @800 tok 3060 (controllo positivo) | Se la degenerazione è colpa di K23 o del prompt/2-bit |
| T2 | rotate32 vs static K23, n=3, L0-L3, cache 128 e 256 | Il candidato SOTA attuale: i −0.4 t/s di rotate comprano qualità vera? |
| T3 | stale vs relearn_on_tighten, n=3 | Se il delta 2.61→2.76 è reale o rumore |
| T4 | W-sweep con freeze a boundary sicuro (`}`/`;`), W=30..150, render | Riabilita/uccide la tabella W (oggi lotteria del punto di taglio) |
| T5 | weighted OFFLINE (`build_session_mask.py`) vs unit in-engine, n=3 | Riconcilia il drift metodologico; cosa deve calcolare il relearn |

Il run in corso `20260710_w100_rotate32_..._compact_prompt` è di fatto un pre-T2/T4 (attacca la prompt-sensitivity): integrarlo a ledger.

## Fase 3 — Leve (per valore atteso; una alla volta, un solo delta rispetto a SOTA_LOCAL_3060)

| # | Leva | Razionale |
|---|---|---|
| L1 | **Two-phase W50-100 in-engine** con freeze sicuro, senza re-prefill | Fase2/fase1 = 14.6/2.03 ≈ 7×: la più grande leva velocità+qualità misurata |
| L2 | **Slope-S1 come trigger** (0012 → nuovo `DS4_PACE_S1_TRIGGER`) | Unico segnale con ~200 tok di preavviso; n-gram dimostrato tardivo 6 volte |
| L3 | **Rotation triggered + delta-prefetch** (~43 slot/step, non WRAP full 75-699 GiB) | Il "next" dichiarato; non esiste ancora `DS4_PACE_ROTATE_TRIGGER` |
| L4 | **SPEX hidden consumer** (bridge pronto, ready=624) | selected_direct 99.98% = stall; testare SOLO locale SSD-bound |
| L5 | **Adaptive-K coverage runtime** (`DS4_PACE_COVERAGE`) | Manopola task-indipendente (cov90→L2/L3), mai provata sul 3060 |
| L6 | clock_breath64 corto/precoce esteso a 800 tok | Unico attuatore breath con segnale positivo (2.78-2.97 t/s repeat=0) |
| L7 | Exchange asincrono compressione (step 8 del piano dinamico) | Cap effettivo 512-1024 (hit sim 0.59-0.76 vs 0.34); mai CQ1 sincrono nel path caldo |

Backlog (dopo L1-L7): temporal same-layer prefetch (recall 0.55, finding 05-07 mai implementato); sidecar int4 top-expert;
rewind/rollback parziale; prefetch-dal-prompt senza mask apply (unica cura nota per il prefill 115-213 s).

## Definition of done per il goal

Config locale 3060 con: `avg_tps ≥ 3.0` su 800 tok HTML, **L2+ al render su 3 prompt**, n=3 con mediane, manifest completo,
trace off. Fino ad allora, SOTA_LOCAL_3060 resta: static K23 3.03-3.39 t/s (degenera ~tok116) / rotate32 2.61-3.03 t/s
(regge 800 tok solo secondo il repeat-detector).

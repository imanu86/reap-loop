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
| T1 — **FATTO** | Full @800 tok (controllo positivo, eseguito su pod 3090, non 3060) | **Budget-confound: il full è L0 a 800 E 2000 sul cyberpunk (colpa del budget/prompt, non di K23 né del 2-bit), pagina completa L2 a 3498 tok, 13/13 repeat=0. Resta attribuibile alla mask SOLO la firma loop. Fonte: runs/ds4/20260710_pod_t1_full_positive_control/README.md** |
| T2 | rotate32 vs static K23, n=3, L0-L3, cache 128 e 256 | Il candidato SOTA attuale: i −0.4 t/s di rotate comprano qualità vera? **Ridefinito post-T1: confronto a 2000-4000 tok ctx8192 con grading L0-L3 (il budget 800 non è un banco di prova valido sul cyberpunk); coincide con M1a in HANDOFF_CODEX.** |
| T3 | stale vs relearn_on_tighten, n=3 | Se il delta 2.61→2.76 è reale o rumore |
| T4 | W-sweep con freeze a boundary sicuro (`}`/`;`), W=30..150, render | Riabilita/uccide la tabella W (oggi lotteria del punto di taglio) |
| T5 | weighted OFFLINE (`build_session_mask.py`) vs unit in-engine, n=3 | Riconcilia il drift metodologico; cosa deve calcolare il relearn |

Il run in corso `20260710_w100_rotate32_..._compact_prompt` è di fatto un pre-T2/T4 (attacca la prompt-sensitivity): integrarlo a ledger.

**T4/T5 hanno gli harness pronti** (commit `561552d`, offline prep): T4 → `scripts/run_w_sweep_freeze_safe.py` + `scripts/freeze_boundary.py` (+test); T5 → `scripts/build_session_mask_canonical.py` (+test); runbook in `docs/T4_T5_RUNBOOK.md`. Mancano solo i run (scheda locale / pod).

## Fase 3 — Leve (per valore atteso; una alla volta, un solo delta rispetto a SOTA_LOCAL_3060)

| # | Leva | Razionale |
|---|---|---|
| L1 | **Two-phase W50-100 in-engine** con freeze sicuro, senza re-prefill | Fase2/fase1 = 14.6/2.03 ≈ 7×: la più grande leva velocità+qualità misurata |
| L2 | **Slope-S1 come trigger** (0012 → nuovo `DS4_PACE_S1_TRIGGER`) | Unico segnale con ~200 tok di preavviso; n-gram dimostrato tardivo 6 volte. **Patch in authoring: 0020.** |
| L3 | **Rotation triggered + delta-prefetch** (~43 slot/step, non WRAP full 75-699 GiB) | Il "next" dichiarato; non esiste ancora `DS4_PACE_ROTATE_TRIGGER`. **Patch in authoring: 0021.** |
| L4 | **SPEX hidden consumer** (bridge pronto, ready=624) | selected_direct 99.98% = stall; testare SOLO locale SSD-bound |
| L5 | **Adaptive-K coverage runtime** (`DS4_PACE_COVERAGE`) | **E-CAL fatto → CALIBRAZIONE DEBOLE** (`runs/ds4/20260710_ecal_coverage_threshold/`): la curva copertura(K) dal warmup è QUASI TASK-INVARIANTE alla finestra d'engage (50 tok): cov@23≈79% (73–82%) e Kmin-cov90≈38 per html *e* tutti gli 11 coding → sceglie ~lo stesso K per ogni task, NON è un discriminatore di larghezza. L'identità `S1=1−cov` è FALSA ~4× (def-1 routed 0.21 vs def-2 full-256 0.81; il trace logga solo i 6 esperti selezionati → def-2 non osservabile offline). La copertura@K-usato NON separa: sopravvissuto K91 S1 0.845 > collassato K23-rotate 0.811 (non-monotono), e keep-23 STATIC sopravvive (L3) mentre keep-23 ROTATE collassa alla copertura *identica* 0.79. Separatori reali = attuazione (static≫rotate), provenienza mask (session≫cold), budget token — nessuno nella curva. **Uso legittimo:** solo pavimento anti-under-provisioning (K23→~38-39). Spec patch **0024** `DS4_PACE_COVERAGE` in REPORT (riusa accumulo 0020, static-keep, mai con rotate; θ NON legata a S1). Ship solo dopo pod A/B K23-static vs cov90-static, n≥3, L0-L3 render ≥2000 tok. |
| L6 | clock_breath64 corto/precoce esteso a 800 tok | Unico attuatore breath con segnale positivo (2.78-2.97 t/s repeat=0) |
| L7 | Exchange asincrono compressione (step 8 del piano dinamico) | Cap effettivo 512-1024 (hit sim 0.59-0.76 vs 0.34); mai CQ1 sincrono nel path caldo |

Backlog (dopo L1-L7): temporal same-layer prefetch (recall 0.55, finding 05-07 mai implementato);
**top-mass precision pin (franken-gguf Q4 top-1/layer, pin fuori cache): E1 fatto →** esito NEGATIVO per il gguf STATICO
(`runs/ds4/20260710_e1_top_expert_mass/`): dominanza per-token top-1 reale ~30.5% (top-3 ~67%) ma identità NON stabile
(cross-task top-1 overlap 2.5%, within-coding 7.8%) → un pin statico cattura solo ~5.7% massa/layer (tetto live-sessione
16.7%) a costo 21.4% del cache 407-slot. Prossimo passo solo se si vuole la variante **live per-sessione** (non statica):
E2 pod A/B ppl + L0-L3 su Q4-top-1-pin-per-sessione vs q2 puro, ~$1-2 — altrimenti CHIUSA.
**S1-guided rewind (correzione) → `docs/S1_REWIND_DESIGN.md`** (candidate patch 0022): posizione nella scala =
CORREZIONE, subito dopo L2/L3 (prevenzione slope-S1 → rotate/widen) e **prima** dello stopper come strategia primaria.
Lo stopper M1b resta AIRBAG (ultima risorsa) e benchmark di confronto nell'A/B, non la cura. Verdetto fattibilità
engine = CLEAR/low-risk: i primitivi hard (frontier snapshot/restore `spec_frontier_*`, rewind del contatore
`ds4_session_rewind`) esistono già e girano ogni token per la speculazione MTP; il rewind è un macro su di essi + un
evento di retraction nello stream. Da fare dopo il verdetto M1b + canonizzazione pace-series.
Prefetch-dal-prompt senza mask apply (unica cura nota per il prefill 115-213 s).

## Definition of done per il goal

Config locale 3060 con: `avg_tps ≥ 3.0` su 800 tok HTML, **L2+ al render su 3 prompt**, n=3 con mediane, manifest completo,
trace off. Fino ad allora, SOTA_LOCAL_3060 resta: static K23 3.03-3.39 t/s (degenera ~tok116) / rotate32 2.61-3.03 t/s
(regge 800 tok solo secondo il repeat-detector).

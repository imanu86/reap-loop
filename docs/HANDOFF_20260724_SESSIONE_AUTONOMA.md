# HANDOFF 2026-07-24 sera/notte — sessione autonoma, stato al rientro

Dettaglio completo in `EXPERIMENTS_LEDGER.md` add. **43, 43b, 43c**.

## ✅ CHIUSO E VERIFICATO

**Campagna GPTQ Q1 — 100%.** 8.725/8.725 pianificati. Q1 reali **8.725 = 85,21%** della griglia 10.240; i 1.515 restanti sono celle mai pianificate (sotto la soglia dei 200 campioni: servono più dati di calibrazione, non più CPU).
- **L18 riparato: 43 → 223.** Causa: uno shard troncato del 41% (`all43b_w1.L18`) con manifest che dichiarava `complete` — perdita silenziosa da scrittura non verificata. Sweep globale: **1 solo manifest corrotto su 249**, difetto isolato.
- La riparazione che avevo proposto io (sostituire lo shard con `all43`) **era sbagliata**: il driver fonde tutti e 7 i trace con `--dedup`, quindi avrebbe cambiato silenziosamente il set di calibrazione. Quella giusta = tenere i 7 trace in ordine e saltare solo le righe illeggibili. **Il gate su un singolo esperto è ciò che l'ha fatto emergere.**
- Qualità dei 180 nuovi (cosine, più alto = meglio): mediana **0,8028** vs 0,8020 dei pre-esistenti e 0,7910 degli altri layer → indistinguibili, anzi leggermente migliori.

**Sidecar full v2 pronto e VERIFICATO A FONDO**: `D:/ds4_work/q1_full_sidecar/full_sidecar_q1_v2.gguf`, **36.324.143.104 byte**, SHA256 `46e7186b54025ed4ae1ed9d883bd0e424ff8ea055bde11aef7b7626249dbf46e`. Trasferito con delta a hash di chunk: **1.535 MB invece di 36,3 GB (23,7×)**; il 95,77% di chunk identici ha provato indipendentemente che il primo download era già corretto. Tool patchato committato (`b024b8e`).

> ⚠️ **TRAPPOLA DI VERIFICA (ci sono cascato):** v1 e v2 hanno **dimensione e magic identici per costruzione** (stesso layout di tensori, stessa dimensione totale). Controllare byte-count + `GGUF` **non distingue** un file patchato da uno non patchato. Ho "verificato" il v2 mentre la patch non era ancora applicata: il test sarebbe partito su contenuto v1 etichettato v2.
> **Verifica vera fatta poi:** patch 366/366 chunk con 0 mismatch (ogni chunk confrontato con l'hash blake2b del pod prima di scrivere e ri-letto dopo); **8.661/8.661 chunk identici** al v2 del pod; **120/120** tensori esperti routed su L03..L42 tutti di tipo **41 (Q1_0)**, 0 mancanti/0 di tipo errato, nessuno fuori range; metadata `expert_count=256`, `block_count=43`, `expert_used_count=6`, 286 tensori; **discriminante su L18**: gli esperti i cui byte cambiano tra v1 e v2 sono **esattamente i 180 ricostruiti** e i 76 invariati sono esattamente 43 pre-esistenti + 33 mai pianificati; round-trip sul file **locale** contro gli NPZ del pod OK.
> Nota tecnica: il blocco #0 a volte coincide tra v1 e v2 — non è un difetto, GPTQ al primo blocco non ha ancora error-feedback accumulato e ricade sulla stessa codifica segno+media-assoluta del base-fill. Conseguenza pratica: il verify a singolo blocco del tool **non è un discriminante di base-fill**.

**🐛 BUG DA CORREGGERE (task già creato):** il writer degli shard di harvest marca `"status":"complete"` **senza verificare la lunghezza del payload**. È così che uno shard troncato del 41% è passato per completo. Sul pod `/workspace/harvest/all43b_w1.L18.vectors.f32le` è **ancora corrotto col manifest che dichiara `complete`** (non modificato, per provenienza): qualsiasi campagna futura su quella directory ricadrà nello stesso errore, a meno di usare `/workspace/l18fix/`.

**POD**: spegnibile in sicurezza. I dati stanno sul volume di rete `/workspace`, che sopravvive allo spegnimento (si azzera solo `/root`).

**Disco = collo fisico misurato.** 1 stream 1,45 GB/s; **4 stream paralleli 1,38 GB/s = 0,96×** → il disco NON scala con la queue depth, e il runtime prende già 0,93-1,0 GB/s. `MOE_IO_QD` e il wrap `FILE_QD=8` sono **refutate**. Unica strada TTFT: leggere meno byte.
- **Azione hardware possibile**: `CurrentLinkSpeed=3` (PCIe 3.0) contro `MaxLinkSpeed=4`. Su ASUS PRIME B550M-A + Ryzen 5800X il disco è nello slot M.2_2 (chipset) invece di M.2_1 (CPU, PCIe 4.0). Inoltre **C: è pieno all'88,5%** e l'EQ790 è DRAM-less. Ordine: liberare C: sotto il 75-80% → riscrivere il modello contiguo → spostare su M.2_1. Non spostabile su D: (è il SATA, ~550 MB/s).

**TTFT scomposto al 96%**: forward di prefill 39,05s (52%) + VirtualUnlock 2,54s (3%) + arena-wrap 33,82s (45%). **Causa radice: gli stessi byte letti due volte per avvio** (il prefill streamma gli esperti, poi il wrap rilegge via mmap i 4423 candidati ricavati dal routing appena osservato). Il rimedio esiste in codice (`cuda_dynamic_arena_observer_mirror_ptr`, cu:11857) ma **non è mai girato**: `[arena-observe]` compare in **0 log su 71**.

## ❌ REFUTATO (non ri-testare)

Per la bistabilità del decode sono cadute, una dopo l'altra: **`TOKEN_SPLIT_LAYERS`** (p1 TS=0 e p2 TS=4 identici), **`CACHE_N`** (160 → stessi reload di 320), **`STREAM_RUNTIME_RESERVE_MB`** (l'override si attiva davvero — riga di prova — e il run si rompe con ~871 MiB di slack), **la soglia di slack** (non monotòna: 1085 pulito, 1127 rotto, 1186 pulito; rotti anche con 1634 e 1405 MiB liberi), **l'ipotesi gap tra i run** (due run con gap 120s identico rotti entrambi).

Cade anche l'attribuzione dei run `pow2_hr` alla riduzione della expert cache: run con cache **piena** (156 slot) sono ugualmente puliti.

## ⚠️ QUATTRO METRICHE DIFETTOSE TROVATE OGGI
1. `iq2_vram_cache=q1-routes-bypass` — **letterale hardcoded** (cu:4554), non uno stato
2. `preloaded=0` in `[prefill-mass-wrap]` — **letterale hardcoded** (cu:12992)
3. `html_closed=True` — **falso positivo**: il check matcha il modello che *cita* l'istruzione; nessun run ha mai prodotto HTML, e la qualità Q1 resta **non misurata**
4. `RELOADS_IN_DECODE` — **cronometro, non contatore**: rate-limited 1/10s, vale ~`secondi_degradati/10`, confondato con `max_tokens`. Usare solo **`primo reload presente/assente`**

Regola: in questo runtime **nessun campo derivato va usato prima di averne verificata la provenienza nel codice**.

## 🔬 IL PROBLEMA APERTO
Il decode ha due regimi: sano ~250 ms/token (≈4 t/s) e degradato ~1800 ms/token (≈0,55 t/s). Il passaggio è un **evento discreto**: il primo reload cade a **seq 127/128** in modo riproducibile su config diverse, costa un token da 7-9 s (backbone 7,21 GiB riletto da SSD) e da lì il collasso è continuo.
- Meccanismo di sfratto: `cuda_model_pick_victim` (cu:6206-6222) a due passate — pass 1 solo range non pinnati, ma **con la expert cache dedicata i routed expert non sono in `g_model_ranges`**, quindi pass 1 non ha candidati e si sfratta subito il backbone pinnato.
- **Variabile nuova non spiegata**: la expert cache si dimensiona a **141/148/151/156/225 slot a parità di config**. Il run con 225 slot si è rotto *prima* del decode. È a monte di tutto e probabilmente più vicina alla causa di qualunque leva testata.
- Strumentazione della memoria host **fallita per locale italiana** (`Get-Counter` con nomi inglesi): nessun dato, non un risultato negativo. Rifare via `Win32_PerfRawData_PerfOS_Memory` (indipendente dalla lingua).

## 🚀 IN CORSO: TEST FULL-Q1 (caso 4)
Servire **tutti** gli esperti a Q1 invece che a IQ2: 72,56 GiB → ~34 GiB, cioè l'intero set esperti sta in RAM. Potrebbe **scavalcare la bistabilità** (senza expert cache dinamica non c'è nulla che competa col backbone) e insieme la causa radice del TTFT (34 GB letti una volta, prompt-indipendenti).

Config derivata dal codice, con tre scoperte non ovvie:
1. **La zavorra IQ2 si risolve da sola**: ds4.c:20284 costruisce l'arena primaria solo se `(resident_arena <= 0 || dual_arena > 0)` → con `RESIDENT_ARENA=1` e `DUAL_ARENA` non settata **l'arena IQ2 da 30 GiB non viene mai allocata**. È questo che fa stare 34 GiB di Q1 in 64 GB.
2. **Il seed va disattivato o il run muore**: `cuda_moe_prefill_vram_seed` fallisce con `config-contract` a meno di `ENFORCE && MASS_LFRU && compose` (cu:30692-30694) — impossibile con tiering off — e un seed fallito è **fatale** (cu:31531-31541). Quindi `SEED_TOTAL=0`.
3. **Nessuno SHA256 necessario**: è letto solo da `cuda_q1_0_ssd_wrap_init` (cu:28313), che non gira con SSD-wrap off.

Sizing: servono esattamente `(42-3+1)*256 = 10240` slot = **33,75 GiB** → richiesti 34 GiB. Aggiunto un **gate sulla RAM host** (≥44 GiB liberi): è la sua assenza che aveva ucciso `pow2_base200`.

**Criterio di successo del sanity check**: binding `active=3..42`, arena Q1 pronta, e soprattutto `[q1-0-sidecar]` con `resident_mode=1` e `resident_hits>0` — **non** `direct_pread_fallbacks` a ogni token. Se binding o residenza falliscono, si riporta il fallimento invece di produrre un numero di throughput privo di significato.

Baseline di confronto: **~4,0 t/s** (warmup IQ2 sano, 200 token, zero reload).

## PROSSIMI PASSI SUGGERITI
1. Esito full-Q1 (in corso).
2. Se full-Q1 regge: rifare le classifiche di leve su una base valida, a partire dalla **riverifica dell'8× di `G73_OPEN=0`** (l'adversarial review ha mostrato che L1 e L4 hanno *entrambi* `G73_OPEN=0`, quindi l'attribuzione dell'Addendum 42 non regge su quei log).
3. Capire perché varia il sizing della expert cache (141→225 a parità di config).
4. **Misurare la qualità del Q1**: oggi è completamente non misurata (zero perplexity, zero confronto logit, nessun run ha mai prodotto HTML).
5. Interventi sul disco (liberare C:, file contiguo, slot M.2_1) — indipendenti dal software e potenzialmente 2-4× sul TTFT.

# PAPER STATE — "Impact-weighted expert prefetch: running a 158B general MoE on a 12GB consumer GPU"

> Stato del paper a 2026-07-04. NON ripartire da capo: dati, esperimenti e citazioni sono TUTTI durevoli (vedi §Dati e §Citazioni). Questo doc lega le scoperte al paper. Compagni: `EXPERIMENTS_LEDGER.md` (100 esp), `PRIOR_ART.md` (citazioni), `references/DSpark_paper.txt`, `references/DwarfStar_ds4_README.md`.

## TESI CENTRALE
**Non tutti gli expert mispredetti/mancanti sono uguali.** Una probe lineare sullo hidden state predice il grosso del routing; il residuo si scompone in **(a) un piccolo set ad ALTO impatto (da fetchare)** + **(b) una coda numerosa a BASSO impatto (droppabile/ultra-comprimibile quasi-lossless)**. Questo trasforma il "prefetch per probabilità" in **"prefetch/quantizzazione pesati per IMPATTO"**, e permette di far girare un MoE **generale** da 158B su una **RTX 3060 (12GB)** a velocità usabile.

## CONTRIBUTI (mappati ai dati)
1. **Decomposizione impatto drop/fetch** (la "double-loop-markov", idea utente). MISURATO: markov-drop CATASTROFICO (59× ppl @C8), hidden-drop apre a plateau **1.3×** → esiste un piccolo set alto-impatto sistematico. → EXPERIMENTS_LEDGER H4/H5. Dati: `models\spex\loops\accuracy_drop_dom.json`, `hidden_drop_dom.log`.
2. **Coverage statico vs working-set temporale** (I6): un prompt ristretto NON restringe staticamente il modello (union ~60-65% expert anche per 1 prompt); il guadagno-dominio è TEMPORALE (cache calda, miss .098). → `scripts_pod/expert_union_coverage.py`.
3. **"Lo STATICO è il collo" su VRAM piccola** (I4/I5 sim + CONFERMATO empiricamente sul 3060): attention/shared (~10GB) soffocano la cache-expert (~1-2GB) → non gli expert. Contro-intuitivo (tutti quantizzano gli expert). → `spex_speed_sim.py`, `spex_speed_sim_quant.py`, `sim_3060_static.py`.
4. **[NUOVO 2026-07-04] Leva-RAM: gli engine di offloading MoE sprecano la RAM.** ds4-CUDA usa O_DIRECT + `posix_fadvise(DONTNEED)` → bypassa la page-cache. Fix (2 env: `DS4_CUDA_NO_DIRECT_IO=1` + `DS4_CUDA_KEEP_MODEL_PAGES=1`) → **0.49→2.1 t/s (4.3×)**. Finding di sistema sull'uso della gerarchia di memoria negli engine streaming-MoE.
5. **[IL CONTRIBUTO CENTRALE, da costruire+misurare] Quantizzazione temperature-per-expert.** Bit-width per-expert dalla FREQUENZA di routing (freddi → 1-1.58bit, caldi → 2bit+). Ortogonale all'imatrix (imatrix = qualità a taglia fissa; temp-quant = memoria). Fondata su H4/H5 (freddi = basso impatto → comprimere ≪ droppare). Dati per costruirlo: nostri trace + REAP (`reap_union_sim.py`). Da BATTERE i baseline (dwarf-2bit-uniforme / layer-mixed-antirez / 4bit) a parità di GB.
6. **Risultato NEGATIVO solido**: ridondanza sub-expert weight-space = ZERO su 3 famiglie (Qwen-30B 128/0-shared, DS2-Lite 64/2-shared, V4-Flash 256/1-shared fp4). → I1/I2/I3, H6/H7. Pubblicabile ("non fondere sub-expert in weight-space").
7. **Scala-invarianza della miss-rate** (30B ≈ 235B per-workload) → I3.
8. **REAP/FT su dominio dimezza l'union quasi-lossless** (keep50 reroute 8.2%/massa 99.3%) → I10.

## STATO QUALITÀ (da rifinire per il paper)
2-bit sul rubric-scored eval set (140 item): MATH 100%, KNOWLEDGE(MMLU-hard) 85%, CODING 55% (CONFUSO: troncamenti + grader). 4-bit PARZIALE: coding 75% → 2-bit degrada il coding ~20pt DA CONFERMARE. TODO paper: full 4-bit + grading pulito + agreement appaiato + ispezione fail. + costo-qualità del 2-bit HOT non misurato (serve eval IQ2 vs fp4).

## CITAZIONI (in PRIOR_ART.md — verificare/espandere)
- **Speculative/DSpark**: DeepSeek DSpark (`references/DSpark_paper.txt`) — confidence-head+STS+scheduler; NB antirez `--mtp` è speculative-base, NON DSpark pieno.
- **Prefetch expert / offloading MoE**: Pre-gated MoE, MoE-Infinity, Fiddler, ProMoE, SiDA-MoE, Fate (la nostra probe-hidden e' "Fate-style").
- **Quantizzazione MoE**: QMoE (Frantar & Alistarh, sub-1-bit), MoQE, Mixture-Compressor, bit-allocation per-expert by-sensitivity (2024-25) — il nostro angolo fresco = by-ROUTING-FREQUENCY + legato al fit-in-VRAM.
- **Pruning**: REAP.
- **Motore**: DwarfStar/ds4 (antirez), quant asimmetrica (expert 2bit, statico Q8/F16).
- Onestà: prefetch e per-expert-mixed-precision NON sono vergini; il contributo e' la CATENA (misura-impatto → quant/prefetch per-frequenza → fit su consumer HW) + la leva-RAM + il negativo.

## DATI (durevoli, NON rigenerare)
- **103 JSON** in `models\` (72 root=pruning/eval, 20 `spex\loops`, 2 `spex`, 1 `eval`) + **18 npz** in `models\spex\` (trace routing q30/q235/ds2lite/olmoe + hidden_scores q30 con scores[T,L-1,E]).
- **EXPERIMENTS_LEDGER.md**: 100 esperimenti (batch A-H studio pruning/SPEX + I1-I11 questa fase) con numeri+path+script. **PRIMA di rifare una misura, cercala qui.**
- Pesi: 2-bit `models\ds4\` (86.7GB) + su WSL `/root/models/ds4-2bit.gguf`; static-Q4 in build sul pod.

## TAGLIO PUBBLICABILE
Nota breve / workshop: *"Impact-weighted expert prefetch"* — (1) decomposizione impatto come contributo centrale, (2) coverage-vs-temporal + (3) static-is-the-collo + (4) leva-RAM come contorno sistemistico, (5) temperature-quant come metodo, (6) negativo come appendice. **Ciò che manca per la credibilità: la validazione qualita' (2bit vs 4bit vs temp-quant, a parità di GB) — in corso.**

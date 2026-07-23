# DS4 MoE PROJECT â€” DEFINITIVE HANDOFF

*Source of record: `C:\Users\imanu\source\repos\reap-loop\docs\EXPERIMENTS_LEDGER.md` (4909 lines) + memory dir `C:\Users\imanu\.claude\projects\C--Users-imanu-source-repos-moe-aggressive-commit\memory\`. Branch: `research/ds4-iq1-subbit-tier-planner`. Date: 2026-07-23.*

---

## 1. ONE-PARAGRAPH STATE

**Both of the project's headline results are real but are measured in regimes that structurally exclude each other's failure mode, and there is no clean full-model, long-output (L0â€“L3) t/s number for either path â€” so "done" is an illusion built on a 64-token exact-hash SOTA and a 1-of-43-layer perplexity check.** Concretely: the fastest defensible serving result is **G73 split-fused IQ2 + static32 tiering = 4.987 t/s**, but it is a byte-exact 64-token/ctx256 transport number, not a quality verdict, and the one long run in that path came back L0/malformed. The best compression result is **GPTQ-Q1 one-shot (no STE) = ~0.811 median clean cosine, held-out ppl +4.5% on L15**, with a working captureâ†’GPTQâ†’GGUF-sidecarâ†’runtime pipeline â€” but Q1 has never had a full-model quality verdict, its "usable" threshold (0.80 cosine) was never validated, and every runtime measurement of low-bit experts in the real transport path is a âˆ’46% to âˆ’95% decode regression (Q1 and G73 are mutually exclusive at runtime). The big-ctx "physics wall" was **retracted** (working-set K6 â‰ˆ111 < cache 160 â†’ not capacity; real bug is `vram_promotions=0`, an engineering fix). Decode is **orchestration/overhead-bound**, not memory-bound (H200 fully resident still ~3.5 t/s). The project is at a decision point: stop re-opening settled walls and run the three clean gates that have never been run (full-model Q1 ppl; promotions=0 fix; a single long-output t/s number).

---

## 2. PROVEN (with numbers)

Measured, defensible, within-series comparable. (Reminders: `validity`=0 is malformed â†’ use `overall_excl_validity`; two non-comparable ppl scales; 235B int4 vs 30B bf16 never compared absolute.)

**Pruning / REAP (30B bf16):**
- REAP-saliency K50 domain ppl **5.500 < full base 5.559** â€” 50% domain pruning is free/positive (A23). Field: REAP K50 **0.5883 > base 0.5602** (A9/A10).
- `experts_by_mass_asc` is the best maskkey (A11: 0.5883 vs EAN 0.5545). Selection matters from K50 up: RANDOM K70 = total death 0.0000/pf152 (A7).
- FT **disperses** the router (n_eff 49.91â†’57.12) â€” prune the BASE, not the FT (D4). NOVELTY CANDIDATE #1 (opposite sign to ReMoE).
- REAP domain-pruning for NARROW domains: K50 dom **1.010Ã— CI[0.996,1.025]** = statistically lossless; K67 fits 32GB (`reap-ds4-track.md`). Stage-B physical surgery = TODO.

**Compression / bit-width:**
- int4 experts = **lossless** (B1: 17.718/6.017 = full); int2 domain-ok (8.524 gen), never drop (13.998). "Compress >> drop."
- 2-bit real quality: IQ2_XXS **13/13 on completed** rubric prompts (I11, N=15, no 4-bit reference).
- **GPTQ-Q1 one-shot (Hessian XtX + error-feedback, no STE/LoRA), leakage-clean:** 13-expert median **0.811**, min 0.755, max 0.855; â‰¥0.80: 10/13 (Add.17â€“20). The "0.70 wall" was **STE training**, not the 1.125-bpw format.
- 161-expert L15 GPTQ: median **0.766**, â‰¥0.75: 92/161 (57%) (Add.28).
- **End-to-end Q1 works:** 909 MB sidecar (161 experts, type-41, 18-byte block), runtime validates+installs 0.85 GiB, coherent output (Add.30â€“34). Held-out ppl: BASE 2-bit **3.014** â†’ Q1-L15 **3.151 = +4.5%** (Add.35, 1/43 layers).

**Serving / speed (native Windows 3060, byte-exact 64-token unless noted):**
- **G35 physical tiering: 3.228â†’4.975 t/s (+54.10%)**, reads âˆ’48.57% (but see caveat: partly a 9-token toy-prompt artifact, Â§5).
- **G36 mass/LFRU: 4.948â†’5.557 t/s (+12.29%)** (toy prompt "Hi").
- **G73 split-fused + static32: 4.61â†’4.987 t/s (+8.17%)** â€” the authorized realistic short-workload SOTA (64-token cyberpunk).
- **G42 closed-snapshot VRAM tiering: 2.277â†’4.083 t/s (+79.36%)**; critical knob: 1024 MiB load reserve â†’ 7.21 GiB hot cache â†’ 4.11 t/s (4096 MiB reserve â†’ 0.14 t/s).
- QD8 vs QD1: WRAP 186.4â†’94.2 s, decode 4.293â†’4.533 (promote QD8, G52â€“55). Trust-worker-checksum: TTFT âˆ’44.95% (G43).
- **Pin-by-mass +53%** (3.23â†’4.93, RTX 3090 native, identical copies/hit) â€” but null-to-negative on 12GB 3060 (see Â§5).
- **Fleet 4Ã—4090: aggregate 4.40 steady vec/s** ($2.76/h) beats H200 at 60% cost; model in shared page cache, disk I/O ~0.

**Diagnosed facts (frame everything):**
- Decode is **orchestration/overhead-bound**: H200 full-resident (COPY_MODEL=1, 80 GiB VRAM) still **~3.5 t/s**. Residency â‰  decode speed.
- **240 experts/token is a constant** (6 top-k Ã— 40 MoE layers), K/domain-independent â€” per-token staging floor.
- Model decomposition (Add.24): ds4-2bit.gguf **86.71 GB** = routed experts 77.91 GB (90%, streamable) + resident floor **8.66 GB**. On 12 GB â†’ ~3 GB free = ~440 cache slots >> working-set 111. **VRAM was never the constraint.**
- Wide collapse = **domain-mismatch of the mask's calibration**, not K, not 2-bit, not rewind-recoverable.
- Sub-expert redundancy CLOSED NEGATIVE across 3 families (cosine ~0.002â€“0.03, eff_rank ~full, NN frac>0.8 = 0%) â€” only activation-space remains (I3).
- `NO_Q8_F16_CACHE=1` is a **correctness gate**: q8/f16 branch diverges systematically (54 diff lines, ~2.6Ã— noise).

---

## 3. THE TWO GOALS & THEIR STATUS

### GOAL A â€” Quality / Compression (fit wider residency; enable CPU compute)
- **Done:** GPTQ-Q1 pipeline end-to-end. Median 0.811 (hot) / 0.766 (161). Sidecar built, runtime-served, coherent. ppl +4.5% on L15.
- **Target / SOTA claim:** an adaptive **1.460 bpw** production model (107 expertsâ†’Q1, 54â†’base-2bit) across all 40 routed layers, quality within a few % of the 2-bit base.
- **Blocking:**
  1. **No full-model (43-layer) ppl** â€” only L15 measured; accumulation across layers is hand-waved "reasonable."
  2. **0.80-cosine threshold never validated** vs actual output quality (Add.13/21/35 all concede "euristica mia, MAI validata").
  3. **Adaptive Q1/Q2 is architecturally blocked** â€” the binder needs COMPLETE Q1_0 tensors per layer â†’ no intra-layer mixing (Add.30). The 1.46 bpw spec (Add.28) is impossible as written.
  4. Cold/very-cold experts (<50 samples) via dense-GPTQ only marginally proven (e117 0.716â†’0.727, Add.29).

### GOAL B â€” Speed / Serving (run the 81GB model as if VRAM-resident; >3 t/s AND HTML/coding quality)
- **Done:** G73 = 4.987 t/s byte-exact, zero SSD decode traffic. Tiering, WRAP, QD8, split-fused all landed.
- **Target / SOTA claim:** ~9â€“10 t/s (chripell async mechanism) with complete, coherent HTML/coding at ctx8192.
- **Blocking:**
  1. **The SOTA is a 64-token exact-hash transport number**, not quality. The one long run (Long-Arm-A, max3000) = **L0 malformed**.
  2. **Decode is orchestration-bound** â€” residency/transport levers plateau at ~4.5â€“5 t/s; the "14 t/s midpoint" needs the 121 ms/tok orchestration cut (untried, A4), not more residency.
  3. **ctxâ‰¥1536 spiral** â€” at ctx8192 cache clamps to 160 slots, `vram_promotions=0`, 8.4 GB/token from SSD â†’ 10â€“13 s/token. Engineering (promotions=0), not capacity.
  4. **Async copy-path (A2) not yet built/measured** â€” the concrete path to ~9 t/s; needs native (done) + a compute stream.

---

## 4. ALL LEVERS

Status: **PROVEN** = measured positive on target HW; **BLOCKED** = works elsewhere / gated by a dependency/conflict; **UNTRIED** = never measured; **NEGATIVE** = measured harmful.

| Lever | Attacks | Status | Dependency / Conflict |
|---|---|---|---|
| **Pin-by-mass** (mass = freqÃ—gate-wt) | Per-token cache overhead | PROVEN +53% on 3090; **null/neg on 3060** | Works only if working-set â‰¤ ~0.65Ã— budget (K12 yes, K23 no). Conflicts with "always on." |
| **Async copy-path** (`cudaMemcpyAsync`, 1 sync/layer) | Sync copies idle the GPU | UNTRIED (chripell: ~9 t/s in 22 lines) | Needs compute stream; pays only at native BW; **the concrete >3â†’9 path** |
| **Windows-native port** | WSL BW throttle | DONE, **demoted** (port alone 3.4â†’4.4) | Pairs with async ("neither alone"); premise (bandwidth) was refuted |
| **Reduce CPU orchestration / router-side pruning** | The 121 ms/tok (42%) floor | UNTRIED as speed lever | The real "14 t/s" lever; tension: pruning restricts eligibility, not 6/layer fetches |
| **MTP / speculative decode** (A5) | Token-rate directly | PARTIAL (spec2 +9â€“10%; CPU-drafter NO-GO) | DeepSpec repo; pretrained DS-compatible drafts; priority AFTER Q1/residency |
| **SPEX hidden GPU prefetch** | Miss-stall | BUILT, DORMANT (`=0`) | Converts recallâ†’t/s ONLY if SSD-bound; **SLOWS ~1.55Ã— on 3060-warm** |
| **Reserve-cap tuning / small-cache** | LRU bookkeeping | PROVEN (K12 +78%; cache32 fastest; cache-large REFUTED) | Pin skips pinned in victim-selector |
| **Fleet cheap GPUs, 1 proc/GPU** | Capture throughput | PROVEN | Multi-proc-on-1-GPU does NOT scale; `DS4_LOCK_FILE` |
| **Two-SSD expert streaming** (task #18) | SSD-read BW | UNTRIED | Lexar NVMe + SanDisk E: |
| **KV-disk-reuse** (`--kv-disk-dir`) | Prefill cost-of-iteration | PROPOSED | Fixed-prompt only |
| **Parallelize expert loads / raise QD** | Prefill serialization (50â€“80Ã— floor) | PROPOSED; QD8 proven | Async infra |
| **Pin/VRAM residency (0031)** | Decode/prefill refetch | PROVEN prefill 1.73Ã—; **REFUTED as decode lever on 3060** | Working-set fit; bit-exact (never touches quality) |
| **Grow-cache F10 (big-ctx)** | Stranded 1â€“2 GB | NEGATIVE at runtime | **Conflicts with prefill-VRAM-seed** (cache-capacity fail â†’ cache disabled = worse than 160); slot-math built on wrong 240 |
| **KV staged-ring F8/F8b/F8c** | Big-ctx KV | NEGATIVE @8k (~90 s/tok), MANDATORY @250kâ€“1M | WDDM hostile to pinned-mapped+event; right-for-longctx only |
| **Attention-compression** | VRAM | CLOSED â€” attn already optimal mixed-precision (q8/f16/f32-bytevec), KV @8k only ~300 MB; **don't touch for Q1** |
| **Prefill expert-I/O planner** (exact-contiguous coalescing) | Prefill reads | PROVEN (13,653â†’7,185 reads, âˆ’47%, byte-amp 1.0) | Implement only exact-contiguous (G56) |
| **Q1_0 expert sidecar (E1)** | Footprint / CPU-feasibility | PROVEN end-to-end (L15) | **Rejected by `DS4_G73_OPEN`** â†’ minimal server; full/open Q1 = 0.2 t/s (G129); quality validation PENDING |
| **Adaptive Q1/Q2 (1.46 bpw)** | Hard-expert quality floor | BLOCKED â€” binder forbids intra-layer mix (Add.30) | Needs per-layer all-Q1-or-nothing rethink |
| **CPU-Q1 expert compute (Colibri)** | H2D swap floor | UNTRIED ("audit #0" microbench never run) | Premise (transfer-bound) conflicts with "orchestration-bound" verdict |
| **promotions=0 fix (0033 + dynamic-K)** | Big-ctx thrash | PENDING (task #15) â€” the real bug | VRAM is NOT the constraint (retract the wall) |
| **Live self-seeded mask (livemask K23)** | "Mask from prompt, mutating" | PROVEN â€” renders complete HTML ~2.9 t/s | Seed on right phase; mass-gate not freq |
| **Freeze-at-safe-boundary (D5)** | Knife-edge W (freeze mid-CSS = restart) | **UNTRIED â€” largest measured latent lever (~7Ã—**, W50 fase2 14.6 vs 2.03) | In-engine freeze after `}` |
| **Competence-gated conditional membership (D3)** | Phase-transition collapse | NEVER BUILT (the whole point) | Signal = entropy/margin NOT S1; needs local nvcc |
| **rotate_delta / pressure-rotation (D7â€“D9)** | Rotation cost / seed-mismatch | SMOKE-PASSED (631â†’119 slots) | Needs a real trigger cabled into PACE (doesn't exist yet) |
| **REAP domain-pruning (narrow)** | Footprint for narrow domains | PROVEN K50 lossless; **Stage-B TODO** | Wide/coding intolerant |
| **`NO_Q8_F16_CACHE=1`** | Quality crack | PROVEN correctness gate | Use in EVERY quality probe |

---

## 5. CONTRADICTIONS & CONFLICTING LEVERS  *(the reason this doc exists)*

### A. Levers that cannot co-exist / fight each other

**C1 â€” G73 IQ2 speed âŸ‚ Q1 quality (mutually exclusive at runtime).** G73 = 4.99 t/s needs `DS4_G73_OPEN`; the Q1 sidecar is *rejected* by it and runs in a minimal server without the F5 chain. Every low-bit-in-transport measurement regresses: G103 one IQ1/layer on G73 = **âˆ’46.32%**; G129 full Q1 = **0.04â€“0.22 t/s**. **There is no clean full-model Q1 t/s number anywhere.** *Resolution:* accept they are two separate deliverables for now; the forward plan must pick which regime each gate runs in and never conflate their numbers. Open question: can the binder/F5-chain be made Q1-aware so a single path serves both? (Not yet scoped.)

**C2 â€” prefill-VRAM-seed: +amortizes short âŸ‚ causes big-ctx thrash.** G51 sells the seed as positive; Add.23â€“24 identify the *static seed with no decode promotion* as the CAUSE of the spiral (holds prefill-hot experts, `vram_hit~0%`). Same lever, opposite sign across context length, **never reconciled in the ledger.** *Resolution:* the seed is correct only when prefill-hot â‰ˆ decode-hot (short prompts); at long-ctx it must be paired with decode-time promotion (0033). Do not ship the seed alone for ctx>1536.

**C3 â€” F10 grow-cache âŸ‚ prefill-VRAM-seed (enabling one breaks the other).** Add.16: F10 phase-B GROW â†’ `prefill-vram-seed result=failed reason=cache-capacity` â†’ cache disabled = **worse than baseline 160**. *Resolution:* F10 fix must ALSO defer the seed. But note C3's whole premise is undermined by C4.

**C4 â€” The F10 "non basta" verdict is built on a working-set number that is wrong by ~2Ã—.** Add.22 arithmetic uses **240** (growâ†’138<160, grow+F8bâ†’185<240 â†’ "MEMORY-BOUND proven"). Add.23: real K6 working-set is **~111** (240 was a K12/coffee number, different model). The resume memo *notes* 240 was wrong yet keeps the conclusion. *Resolution:* discard the F10 memory-wall verdict entirely; re-derive against 111. Cache 160 > 111 â†’ **not capacity â†’ promotions=0 (C2/task#15) is the real fix.**

**C5 â€” Colibri CPU-expert premise âŸ‚ "orchestration-bound" verdict.** Colibri assumes decode is transfer-bound (move cold experts to CPU-RAM to dodge H2D). But H200 full-resident still ~3.5 t/s proves orchestration-bound. Adding a CPU compute path + per-layer sync may not help and may hurt. The decisive "audit #0" microbench was **never run.** *Resolution:* run audit #0 BEFORE investing in Colibri; the "Q1 makes Colibri feasible" claim (Add.33) rests on an unrun benchmark.

**C6 â€” Adaptive Q1/Q2 spec âŸ‚ the binder.** Add.28 ships the 107-Q1/54-Q2 spec; Add.30 (2h later) discovers the binder forbids intra-layer mixing â†’ the flagship plan is **architecturally impossible as written.** *Resolution:* either all-Q1 per layer, or per-layer choice of "whole layer Q1" vs "whole layer base-2bit." The adaptive-precision idea must be re-specced at layer granularity.

**C7 â€” Dynamic-K "OTTIMA" (Add.26) âŸ‚ every prior decode-time mask-mutation result.** rotate32/staircase/admit all collapsed (Anchor Law); PACE live-learn does NOT converge; J40â€“J51 all thrashed. *Resolution:* dynamic-K is NOT obviously good â€” it inherits the entire failed lineage of decode-time mutation. Only run it as the one clean falsifiable experiment (degrade cheap expert-averaging, NOT the sensitive attention backbone), with the fail-fast protocol.

**C8 â€” pinned âŸ‚ pageable IQ1 residency (OS fights you).** G105: 46.875 GiB pageable all-IQ1 â†’ Windows pages it out, aborted. G106: pinned cap is **10 GiB pass / 11 GiB OOM**. G129 then serves Q1 as 7,433 pinned + **3,575 pageable** â€” exactly the class Windows evicts. Unresolved and baked into current Q1 runtime.

**C9 â€” pin-by-mass "always on" âŸ‚ 3060 reality.** +53% on 3090 (24 GB, working-set fits) â†’ "accendila SEMPRE." On 12 GB 3060: PIN=1 = 0.78 vs 0.86 baseline (K23) = null-to-negative. *Resolution:* "always on" is hardware-conditional; on the 3060 target it helps only K12-class narrow working-sets.

### B. Disproven walls & walked-back victories

**C10 â€” "25 t/s keep-8" is REFUTED yet re-cited as a triumph.** `velocity-â€¦-resolved.md` says CLAIM-008 is refuted (keep-8 warm â‰ˆ3.2 t/s); `physics-vs-engineering-default.md` (2026-07-23) cites "keep-8 25 t/s" as a validation of the methodology. A refuted number is evidence for a method. Also "1.65â†’4.60" conflates full/open control (G123) with request-scoped-closed (G73) â€” the exact conflation G120 corrected.

**C11 â€” "residency â‰  decode speed / compute-bound" âŸ‚ the whole G-series.** The WSL memory says byte-location is irrelevant; G35/G36/G73 (native) move decode +54/+12/+8% via residency/transport. The WSL verdict was confounded by the synchronous copy-path; the G-series implicitly refutes it but **the memory claim persists uncorrected.** *Resolution:* the honest statement is "on WSL's sync path, residency was neutral; on native, transport levers pay up to a ~5 t/s orchestration ceiling."

**C12 â€” big-ctx "physics wall" proven then retracted in 15 minutes** (Add.22 08:30 â†’ Add.23 08:45), violating the user's own rule (physics only when *measured*). Linchpin (240) imported from a different model.

**C13 â€” Windows-native port justified by a bandwidth premise refuted the same day.** `coffee-mask` and `pod-round2` both strike through the "~10â€“15 t/s native projection": native does ~4.4, bandwidth doesn't scale with the link, the real bottleneck is the sync copy-path. The biggest engineering pivot's stated reason was already known wrong â€” though the port was still correct for async's sake.

**C14 â€” mass/LFRU tiering: +54% toy prompt, âˆ’77% real prompt.** G35/G36 (+54/+12%) ran on "Hi" (EOS after 9 tokens, cache336, 8 GiB arena). G40 (real 43-token cyberpunk, same tiering) = **âˆ’76.72%**. The G36 5.56 headline is a **degenerate-workload artifact** the Protocol Audit itself flags as "NOT the general Windows SOTA."

**C15 â€” speed-sim batch (I4â€“I8) built on a 2Ã— wrong param count.** I9 corrects: V4-Flash is 158B/13B not 284B, expert 12.88M not 25.17M â†’ footprints halve, "static fp8 mandatory or cap at 5 t/s" evaporates â†’ realistic ~45â€“53 t/s. All I4â€“I8 quant/footprint conclusions are void.

### C. Statistical / metric unsoundness (unstated assumptions)

**C16 â€” "n=3" has zero statistical power.** temp0 greedy = deterministic replica (byte-identical seeds) â†’ proves bookkeeping is deterministic, NOT that a +1â€“8% decode delta survives stochastic variation. Every G-series n=3 A/B inherits this. Real power needs stochastic sampling.

**C17 â€” exact-output-hash â‰  quality.** The G-series pass gate is SHA-256 of a 64-token prefix = a transport invariant. Any A/B that flips one token diverges downstream (Add.34 butterfly effect), so hash-exactness can neither confirm nor deny a quality change. ~50 experiments certify determinism, not quality.

**C18 â€” the 64-token SOTA structurally dodges the real failure mode.** G35â†’G129 gate on ctx256/64-tok (fits in cache). The real goal (HTML/coding ~4000 tok, ctx8192) is exactly where the same system collapses to 0.07â€“0.5 t/s. The 4.99 SOTA is measured in the regime that avoids the spiral.

**C19 â€” q8/f16 crack contaminates historical numbers.** Any pre-`NO_Q8_F16_CACHE=1` speed/quality measurement is quality-contaminated (systematic divergence, ~2.6Ã— noise). Unstated caveat on an unknown fraction of the ledger.

**C20 â€” static-mask quality is non-monotonic in K.** K23 renders, K64 same-domain collapses, full renders â†’ "conta QUALI non QUANTI." Any trade-off statement assuming monotonic quality-vs-K (several exist) is unsound. There is no single "sweet-spot K" knob.

**C21 â€” recurring 200â€“372 s TTFT stall swept into "outlier"** at ~33% incidence (G45 2/6, G48 2/4, G70, G71, G129). At that rate it's a population property, not noise; if workload-triggered, every TTFT/startup number is unreliable.

**C22 â€” "MASSA < RANDOM at 235B" novelty rests on an un-removed confound** (int4-only + 235B-only). The ledger flagged "replicate on 235B bf16" â€” never done (C8/G7). Novelty claim on a precision confound.

---

## 6. HONEST CAVEATS

- **Not proven:** full-model (43-layer) Q1 ppl; end-to-end chat quality vs 0.80 threshold; any clean long-output (L0â€“L3) t/s for EITHER G73 or Q1; cold-expert (<50 samples) dense-GPTQ at scale; CPU-expert-vs-H2D (audit #0); F8b/F8c on Linux; dynamic-K-on-miss-pressure; all-layer capture for the 10,240-sidecar campaign.
- **Retracted:** big-ctx "memory-bound physics wall" (Add.22â†’23, wrong working-set); "25 t/s keep-8" (7â€“8Ã— short); "native port recovers bandwidth ~10â€“15 t/s" (native does 4.4); I4â€“I8 speed-sims (2Ã— param error); the founding Q1 victory E176 0.8335 (n=1-lucky AND leaky â€” shards not deduped, 5173 rows/3797 unique).
- **Got lucky:** E176 was the fortunate case (median of 13 was 0.733, only 2/13 â‰¥0.80 under STE); GPTQ later rescued it on firmer ground but the arc was 3 reversals in 6 hours.
- **Got unlucky / degenerate:** G35/G36 tiering headline is a 9-token toy artifact that inverts (âˆ’77%) on real prompts.
- **The 0.80-cosine threshold gating a ~$100 / 10,240-sidecar campaign was never validated against output quality.** This is the single highest-stakes unvalidated assumption.
- **Meta-pattern:** the "every wall is an engineering defect until physics is proven" rule genuinely unlocked 0.45â†’4.95 t/s, but it *guarantees no negative result is ever final* â€” velocity, big-ctx, and Q1 each flip-flopped 3+ times (the user was "infuriato per i flip-flop"). Combined with zero-power n=3 and hash-as-quality, the project's confidence signals are **transport-exact but quality-blind.**

---

## 7. THE COHERENT FORWARD PLAN

An ordered sequence where each step gates the next and no step contradicts a proven result above. **The organizing principle: stop adding levers; close the three quality/engineering gates that every headline currently dodges. Measure in the REAL regime (long output, ctx8192), not the 64-token hash regime.**

**Gate 0 â€” Hygiene (before any ledger-grade run).** Check resting RAM (8â€“10 GB clean; >10â€“12 = orphan â†’ `ram_audit.ps1`). Set `NO_Q8_F16_CACHE=1` (C19/F1) on every quality probe. External watchdog on every generation test (kill specific PID, never `pkill`). Stop-clean via token, never `/F`. *Why first:* every prior number without these is suspect.

**Step 1 â€” Full-model Q1 perplexity (closes Goal A's #1 blocker; resolves C1/C17/C18 for the compression side).** Extend the L15 captureâ†’GPTQâ†’sidecar to all 40 routed layers (fleet 4Ã—4090, 1 proc/GPU, model in page cache). Measure held-out ppl BASE-2bit vs full-Q1 on â‰¥200 tokens. **Gate:** if full-model ppl degradation stays within a few % of the +4.5% single-layer number â†’ Q1 is a real compression result and the campaign is justified. If it blows up (accumulation) â†’ Q1 is L15-only and the "FUNZIONA end-to-end" claim in MEMORY.md must be downgraded. *This is the cheapest way to convert the biggest unvalidated assumption into a fact.* Do NOT run the 10,240-sidecar campaign until this passes. Re-spec adaptive precision at LAYER granularity (whole-layer Q1 vs whole-layer base-2bit) per C6 â€” do not attempt intra-layer mixing.

**Step 2 â€” promotions=0 fix on big-ctx (closes Goal B's spiral; resolves C2/C3/C4).** Apply tiered-residency patch 0033 (decode-time dynamic promotion) so the cache adapts off the prefill-hot seed. Re-derive all slot math against working-set **111**, not 240 â€” discard the F10 memory-wall verdict. **Gate:** at ctx8192, `vram_promotions > 0` and `vram_hit` rises above ~0% â†’ decode climbs off the 0.13 t/s floor. If it does â†’ big-ctx is engineering-solved and F10/F8-ring are unnecessary at 8k. If it doesn't â†’ re-examine whether the constraint is truly promotions (but VRAM is already proven not to be it). Pair the seed with promotion (never ship seed-alone for ctx>1536).

**Step 3 â€” ONE clean long-output t/s number for the G73 IQ2 path (resolves C18/C16).** Run the 64-token G73 config on the REAL regime: full ~4000-token HTML/coding document, ctx8192, right cap (stop on `</html>`/degeneration/budget, never an 80-tok hard cap), with stochastic sampling (not temp0) for real statistical power, watchdog live. **Gate:** report both t/s AND an L0â€“L3 quality grade. This is the number the entire project has never had. If G73 holds â‰¥3 t/s at L2â€“L3 on a long doc â†’ it is the true serving SOTA. If it collapses (as Long-Arm-A did, L0) â†’ the 4.99 headline is retired as a transport-only invariant and the real work is Step 4.

**Step 4 â€” Async copy-path (A2) â€” the concrete >3â†’9 t/s mechanism (only after Steps 2â€“3 establish the real baseline).** Build `cudaMemcpyAsync` batched-per-layer with one sync/layer before GEMM, on the native port (already done). Gate on **bit-exact vs sync baseline** (deterministic mode, F2). *Why here and not first:* it pays only at native bandwidth and only if decode isn't purely orchestration-bound â€” Step 3's long-output number tells us how much headroom exists above the ~3.5 t/s orchestration floor. If Step 3 shows we're already at the orchestration ceiling, async buys little and the real lever is router-side orchestration reduction (A4), not more copy bandwidth.

**Step 5 â€” audit #0 (CPU-Q1-expert vs transient-H2D microbench) BEFORE any Colibri investment (resolves C5).** Only meaningful once Step 1 gives a real Q1 sidecar. If CPU-Q1 compute (1.125 bpw from RAM, 16 KB output back) beats H2D at the measured orchestration floor â†’ Colibri split is viable. If not â†’ drop it; the "Q1 makes Colibri feasible" claim was premise-blind.

**Deferred until the above close (do NOT start in parallel â€” this is where thrashing lives):**
- Freeze-at-safe-boundary (D5, the ~7Ã— latent lever) â€” highest-value quality lever, but it needs in-engine work and a stable long-output harness (Step 3) to measure against.
- Competence-gated conditional-membership controller (D3) â€” the original point, but needs entropy instrumentation + local nvcc; build only after Step 2 proves the promotion machinery.
- Dynamic-K-on-miss (C7) â€” only as ONE fail-fast falsifiable experiment, degrading expert-averaging not attention; if run 1/3 degenerates, STOP.
- MTP/speculative (A5), two-SSD streaming, F8-ring â€” all after the core quality gates; F8-ring is long-ctx-only (250kâ€“1M), a net negative at 8k.

**The single sentence that prevents 2-steps-back:** *No lever gets built until there is one clean, long-output, real-regime number (Step 3) it can be measured against â€” because every current headline is measured in a regime that structurally hides the failure the lever is supposed to fix.*
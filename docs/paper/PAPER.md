# Keeping a Live-Calibrated Working-Set VRAM-Resident on a Consumer GPU: A Reversible Gate-Bias Loop for Single-Stream MoE Inference, with a Measured Anatomy and Honest Negatives

> **Draft status.** Systems-paper draft. Every quantitative claim carries an on-disk source path in an HTML comment `<!-- src: ... -->`. Claims that depend on data not yet on disk are marked **[PENDING]**; claims whose mechanism runs but whose thesis-level payoff is not yet demonstrated are marked **[OPEN]**.
> **Naming.** The headline mechanism of this paper is **REAP-LOOP**. The name is deliberate and credits its lineage: the per-expert selection saliency we prune on is exactly that of REAP (Router-weighted Expert Activation Pruning, arXiv:2510.13999, Cerebras), which we cite as the source of the criterion and build directly upon. Our contribution is not the saliency but the **-LOOP**: turning REAP's *static, one-shot* domain compression into a *dynamic, reversible, **live-calibrated*** loop that restricts the live routing at inference time, breathes, and rewinds on drift. "Live-calibrated" is the load-bearing word: the working-set is learned from what the *current session* routes, not pre-selected for a fixed domain — so the loop is general (adapts to code/prose/math alike) and the REAP saliency enters only as the *seed criterion* for the live-learned mask. We claim only the loop; the saliency is REAP-Cerebras's, cited throughout. Fully-distinct alternatives we considered (FOCUS, WARDEN) are noted in §5.
> <!-- src: REAP = arXiv 2510.13999 Cerebras (static one-shot; source of the saliency criterion), docs/SCALE_FRONTIER_VERDICT.md:22. Name REAP-LOOP = REAP saliency + dynamic loop; loop claimed, saliency cited (author decision) -->

---

## Abstract

The memory wall of frontier Mixture-of-Experts (MoE) inference is structural: a model's serving footprint is set by its **total** expert parameters, not the small **active** subset — DeepSeek-V3 is 671B total / 37B active, Kimi-K2 1T / 32B, Llama-4 Behemoth 2T / 288B — and total parameter counts are growing far faster than fast memory (~410×/2yr in model size against ~2×/2yr in GPU memory).<!-- src: total-vs-active geometries and 410×/2yr vs 2×/2yr trend: docs/SCALE_FRONTIER_VERDICT.md:11 (cites AI Memory Wall, arXiv 2403.14123) --> This is our **motivation**, not a claim we resolve: we study the extreme, memory-constrained tail of this ladder — a DeepSeek-V4-Flash-class MoE (158B parameters, 256 experts/layer, 6 active, 40 MoE layers, IQ2\_XXS 2-bit, ~86.7 GB on disk) served on a **single consumer GPU** (RTX 3060, 12 GB VRAM, 28 GB system RAM), where expert transfer over PCIe (16–32 GB/s) dominates single-stream decode.<!-- src: model geometry runs/reap/2026-07-05_trace_dominio/meta.json (E=256, 6 active, routed layers 3..42); gguf size runs/reap/gguf_flash_expert_geometry.txt (86,720,111,488 B); target hardware docs/PAPER_STATE.md:1; PCIe 16-32 GB/s docs/SCALE_FRONTIER_VERDICT.md:14 -->

Our headline contribution is **REAP-LOOP**, a training-free, weight-update-free, inference-time loop that keeps a **live-calibrated working-set of experts VRAM-resident** by writing a **reversible gate-bias (−1e9) that actively restricts the *live* routing** — not the cache, not the prefetcher — inside a real expert-offload engine (DwarfStar/ds4). The essence is **calibrate-from-what-you-are-doing-now**, not domain-pretraining: on a single stream, REAP-LOOP (a) starts from the **full** model with a healthy context, (b) **observes which experts *this current session* actually routes** in the first ~150 tokens and learns a bias-mask from them, (c) tightens it stepwise to keep-9% via the gate bias, and (d) periodically re-learns ("breathes") and rewinds on drift. Because the mask is learned from the live stream, the loop adapts to **whatever the user is doing right now** — code, prose, or math — without any per-domain pre-training; the causal asymmetry we measure (start-full → narrow) *is* this live calibration. A catalogue of pre-learned masks (`--mask-load`, §5.6) is an **optional** optimization that skips the warmup — not the mechanism.<!-- src: recipe & steps docs/REAP_LOOP_NOVELTY.md:18-26 (the exact claimable sentence: session-learned, live); docs/INVENTIONS_LEDGER.md:21 (recipe v1: full→session-mask→tighten→sensor→rewind), :17 (session-learned from first ~150 tok), :31 (breath/re-learn D6b), :35 (learn-once-reuse = catalogue optimization, output-type transfer 1.19×) -->

The measured payoffs, each sourced. The **quality** results we hold to: a **measured asymmetry (n=1)** — the same keep-9% mask stays coherent when tightened stepwise while context/KV is healthy, and collapses instantly when applied cold, consistent with a causal reading in which the lever is *context health*, not gradient recovery (multi-seed **[OPEN]**);<!-- src: docs/INVENTIONS_LEDGER.md:17 (asymmetry: full→step to keep-9% clean vs cold collapse), :50 (Control A: quality-asymmetry real, HOT 94% vs COLD 70%) --> a real-task north-star on **a real structured-extraction task** scored by `eval_gold`, where aggressive keep-9% is **on par with the full model in knowledge** (lenient field-acc 0.70 = 0.70) and the extreme-strict damage is **only formatting** (not knowledge), with a hypothesized cheap fix (+50 tokens / JSON-repair) **not yet measured [OPEN]**, at hit-rate 0.89–0.99;<!-- src: docs/INVENTIONS_LEDGER.md:46 (north-star validated: lenient full 0.70 = k50/k91/two-step 0.70; strict damage format-only; hit 0.89-0.99; learn-once-reuse confirmed) --> and downstream perplexity of **1.22× for the session-mask vs 3.5× for a domain-mask and 9.3× for random**, establishing that the *right* working-set is per-output-type, learned live, not per-domain, and transfers across tasks at 1.19×.<!-- src: docs/INVENTIONS_LEDGER.md:32 (Control B: session ppl 1.22× vs domain 3.5×), :35 (output-type 1.19× transfer, random 9.34×, hierarchy 1.2× ≪ 3.5× ≪ 9.3×) -->

The **speed** results are **[PRELIMINARY — confounded by cache cold/warm state and run order]** and we mark them as such rather than headline them. The honest, defensible datapoint is a keep-9% steady-state **warm** throughput of **~2.6 t/s** (session-mask range 2.4–4.5 t/s measured, hit 0.79) on the mask-restricted stream.<!-- src: docs/INVENTIONS_LEDGER.md:18 (session-mask 4.5 vs static 2.4 t/s, hit 0.79 vs 0.57; steady-warm keep-9% band) --> The larger pod figures (a "3.7× at half VRAM" head-to-head, 23.6 t/s dynamic keep-9%) are **not** a clean headline: they were measured against baselines that were themselves crippled (trace-on-SSD, small cache) and confounded by cold/warm cache state and the order the arms were run in. A clean speed comparison requires cache-isolation (drop-caches between arms) plus multi-seed, interleaved-order runs on the target hardware — **still to be done (§7.1, §8.2)**. We report the pod numbers only as directional, flagged, and never as a secured multiple.<!-- src: docs/INVENTIONS_LEDGER.md:28,30,47 (pod 3080Ti; hit-rate transfers, absolute t/s not; head-to-head 3.7× and 23.6 t/s confounded by cache state + run order; clean isolation + multi-seed still to do) -->

We put the paper's credibility in its **negatives and retractions**. We include the honest failures — a two-step variant that was a non-result, a routing-Jaccard "thermometer" sensor that is dead at every lag (so the practical trigger is a *textual* n-gram detector, not the router) — and two retractions: a cold-start bug that silently disabled the mask (85% of selections on pruned experts, fixed in v3 by re-applying every 32 tokens), and the discovery that a "static mass-based mask holds" result was the full model in disguise (the true static mask collapses; only the session-learned mask works).<!-- src: two-step non-result docs/INVENTIONS_LEDGER.md:49; sensor dead all lags :43,:48 (S2 morto tutti i lag, textual detector only); cold-start bug v3 fix :41; static-mask retraction :51 (true static collapses, only session-mask works) -->

We frame this explicitly as an **edge / single-stream** result and **do not claim it transfers to scale**: batched, multi-node serving re-expands the activated-expert union toward the full set, and the data-center bottleneck is the interconnect (NVLink/all-to-all), not PCIe — so the working-set-narrowing lever is specific to the single-stream, memory-constrained regime.<!-- src: no-scale caveat docs/SCALE_FRONTIER_VERDICT.md:8,14-16,34 (batching re-expands working-set: DeepSeek-R1 163/256 @batch32, 243/256 @batch64; DC bottleneck = NVLink 900GB/s-1.8TB/s, not PCIe) --> The claim is **not** "we shipped a 235B model on a 3060," but a rigorously measured edge mechanism plus an anatomy of the gap between the offload literature (measured on RAM-hot data-center hardware) and a real SSD-bound consumer deployment.

---

## 1. Introduction

### 1.1 The memory wall as motivation (edge, not frontier)

Frontier MoE inference is bounded by a structural memory wall. The footprint that must be *resident* to serve a model is governed by its **total** parameter count, not the sparse active subset each token touches: DeepSeek-V3 activates 37B of 671B, Kimi-K2 32B of 1T, Llama-4 Behemoth 288B of 2T.<!-- src: docs/SCALE_FRONTIER_VERDICT.md:11 (total/active geometries) --> At serving-grade precision these do not fit one node — V3 in FP8 (~685 GB) exceeds an 8×H100, Kimi-K2 FP8 (~1 TB) saturates 8×H200 — and the gap is widening: model size grows ~**410×/2yr** while GPU memory grows ~**2×/2yr** (the AI Memory Wall, arXiv:2403.14123).<!-- src: docs/SCALE_FRONTIER_VERDICT.md:11 (fit failures, 410×/2yr vs 2×/2yr, cites AI Memory Wall 2403.14123) --> We take this wall as our **motivation**. We are careful about its honest edge: at 4-bit, many 300–700B models *do* fit a single H200/B200 node, so the wall bites hardest in the 1T–2T high-precision / long-KV regime.<!-- src: docs/SCALE_FRONTIER_VERDICT.md:11 (⚠️ caveat: at 4-bit ~300-700B re-enter a node) -->

### 1.2 The north star: a concrete consumer target

Our target is one system: run DeepSeek-V4-Flash-class MoE — 158B parameters, 256 experts/layer, 6 active, 40 MoE layers, IQ2\_XXS (~2-bit, 86.72 GB on disk) — on an **RTX 3060 with 12 GB VRAM and 28 GB system RAM**.<!-- src: model geometry runs/reap/2026-07-05_trace_dominio/meta.json; gguf size runs/reap/gguf_flash_expert_geometry.txt (86,720,111,488 B); target docs/PAPER_STATE.md:1 --> The model fits neither VRAM nor RAM and is served by streaming experts from SSD on demand, on the DwarfStar/ds4 engine (antirez), which already does reactive per-expert streaming with an LRU expert cache and a native MTP-1 speculative decoder.<!-- src: docs/paper/PAPER_DRAFT.md:24 (ds4 baseline: reactive streaming, LRU cache, MTP-1) --> At 6 active experts × 40 MoE layers = 240 expert reads/token, and 6.75 MiB/expert at 2-bit, that is ~1.6 GiB of expert weight touched per token if nothing is cached — the raw driver of the SSD/PCIe tail.<!-- src: docs/paper/PAPER_DRAFT.md:119-120 (240 reads/token, 6.75 MiB/expert, ~1.6 GiB/token); runs/reap/gguf_flash_expert_geometry.txt -->

### 1.3 What this paper is, and is not

This paper is **not** a demonstration that the target runs at usable speed on the 3060. On the contrary, a central anatomy finding (§3.1) is that on 12 GB **every configuration in which the expert cache is *active* is slower than the cache-disabled baseline**, because the static working set (attention + shared) crowds out the expert cache and the bottleneck moves to the SSD tail.<!-- src: docs/paper/PAPER_DRAFT.md:85-94 (central negative: cache-active always slower; best 0.70-0.72 t/s cache OFF) --> It is a **measured anatomy plus a working-set mechanism (REAP-LOOP) plus a suite of honest negatives**: each lever is individually validated (or shown to fail) with a control, and end-to-end usable speed on the exact target is left as clearly-scoped, costed future work.

### 1.4 The scale caveat, stated up front

We state the boundary of the claim in the introduction rather than burying it in threats-to-validity. REAP-LOOP is a **single-stream, consumer/edge** mechanism. It does **not** transfer unchanged to data-center serving, for two measured reasons. **(1) The bottleneck is different:** on the 3060 the constraint is PCIe host→device (~16–32 GB/s); in the data center it is the cross-node all-to-all (InfiniBand ~400 Gb/s) or intra-node NVLink (900 GB/s–1.8 TB/s, ~18× PCIe), which makes active-parameter transfer nearly transparent — "dodging the PCIe" is not "dodging the interconnect."<!-- src: docs/SCALE_FRONTIER_VERDICT.md:14 (different bottleneck: PCIe vs NVLink 900GB/s-1.8TB/s ~18×) --> **(2) Batching re-expands the working-set:** the union of activated experts grows with batch size toward saturation — DeepSeek-R1 (256 experts, k=8) activates 163/256 at batch=32 and 243/256 at batch=64 — so under batched data-center serving the narrow per-domain working-set collapses back toward the full set, except under domain-sharded serving (a strong, unbuilt hypothesis).<!-- src: docs/SCALE_FRONTIER_VERDICT.md:15 (batching re-expands: DeepSeek-R1 163/256 @batch32, 243/256 @batch64; Mixtral 7.63/8 @batch~57; domain-sharding unbuilt) --> The defensible framing is therefore: same *principle* (working-set ≪ total), a *different regime* (single-stream, memory-constrained, consumer/edge) — not an identity of mechanism with frontier serving.<!-- src: docs/SCALE_FRONTIER_VERDICT.md:18,34 (survives as principle-not-mechanism; the defensible sentence) -->

### 1.5 Contributions

Ordered by solidity:

1. **The REAP-LOOP loop and its asymmetry finding.** A training-free, reversible gate-bias that actively restricts live routing to a **live-calibrated (session-learned) working-set** — learned from what the *current* session routes, so the loop is general (code/prose/math alike), not domain-pretrained — and the measured asymmetry that the same keep-9% mask holds when tightened stepwise under a healthy context but collapses cold, isolating *context health* as the lever (§5, Control A). **[causal reading rests on n=1; multi-seed OPEN].**<!-- src: docs/INVENTIONS_LEDGER.md:17,50 (asymmetry finding; session-learned, live); docs/REAP_LOOP_NOVELTY.md:18-26 (claimable sentence) -->
2. **A measured system anatomy** of an expert-offload engine on a genuine consumer target: the counter-intuitive "cache-active is always slower" negative, the absence of a compute stream to hide copies behind, and the finding that on 12 GB the *static* component is the ceiling — which is precisely why the right lever is to *narrow the working-set*, not to cache better (§3).<!-- src: docs/paper/PAPER_DRAFT.md:83-115 (anatomy: cache-slower, no compute stream, static-is-ceiling) -->
3. **A surgery-free bias-mask actuator** on a stock 2-bit engine: writing −1e9 into the selection bias is exactly equivalent to physical pruning under the ds4 router math, at ~zero cost, verified with 0 violations over 11,280 selections (§5.3, §6).<!-- src: docs/paper/PAPER_DRAFT.md:202-203 (actuator, 11280 checks 0 violations); docs/INVENTIONS_LEDGER.md:57-58 -->
4. **A suite of honest negatives and retractions** that disarm a severe reviewer before the objection lands: the dead router-thermometer sensor, the two-step non-result, the cold-start mask-OFF bug, and the static-mask-collapse retraction (§8).<!-- src: docs/INVENTIONS_LEDGER.md:43,49,41,51 -->
5. **A stack of composable levers** — union-load block speculative decoding, SPEX hidden-state prefetch (**[PENDING]** export) — each with its control and failure mode, sharing the REAP-LOOP actuator's ID-space stability (§6).<!-- src: docs/paper/PAPER_DRAFT.md:179-234 (lever stack); docs/INVENTIONS_LEDGER.md:72 (attuatore-stabile-in-ID-space) -->

### 1.6 Novelty, honestly

No single building block is new, and we say so. Learning a mask from the session is GRIFFIN (arXiv:2404.01365); stepwise/gradual pruning to avoid collapse is folklore since Zhu & Gupta (arXiv:1710.01878); test-time per-prompt pruning is μ-MoE (arXiv:2505.18451); keeping a session working-set resident is done by MoE-Infinity (arXiv:2401.14361), ExpertFlow (arXiv:2510.26730), and LYNX (arXiv:2411.08982); static domain pruning ~50% near-lossless is EASY-EP (arXiv:2504.06792), PreMoE (arXiv:2505.17639), and REAP (arXiv:2510.13999).<!-- src: docs/REAP_LOOP_NOVELTY.md:29-40 (prior-art table, citations verified real 2026-07-06); docs/SCALE_FRONTIER_VERDICT.md:21-23 --> **Critically, the prior working-set methods act on the cache / prefetch / offload path; REAP-LOOP restricts the *live routing* — that is our line of demarcation.**<!-- src: docs/REAP_LOOP_NOVELTY.md:35 (working-set prior-art touches cache/offload NOT routing → line of demarcation) --> We claim only the **sliver**: the closed-loop composition (session-learn → tighten → breathe/re-learn → rewind), the causal asymmetry finding, and a degeneration signal — gate-mass-on-pruned-experts — that no prior detector uses (LoopGuard uses attention/KV, LPSR the residual stream, SpecRA text-FFT, ST-MoE prefetch patterns).<!-- src: docs/REAP_LOOP_NOVELTY.md:10-16 (each prior detector uses a different signal; none uses gate-mass-on-pruned) → verdict PARTIAL, claim system+finding not primitive -->

---

## 2. Background and Related Work

### 2.1 Expert offloading and caching

Streaming experts from slower memory is well established: Mixtral-offloading [2312.17238], PowerInfer-2 [2406.06282] (smartphone), HOBBIT [2411.01433], MoE-Infinity [2401.14361] (sparsity-aware cache), and ProMoE [2410.22134] (proactive caching). ProMoE and MoE-Infinity *already* argue that proactive prefetch beats reactive caching at a fixed budget. Our contribution here is therefore **not** the qualitative conclusion "predictive beats reactive" — that is theirs — but a **measurement protocol**: a random-eviction floor + reactive-LRU + adaptive at matched VRAM that isolates how much of the gain is recency versus prediction. Honest caveat: that decomposition is currently a reading of three numbers, not a computed quantity with a confidence interval (§7.5).<!-- src: docs/PRIOR_ART.md:13,21; docs/CONSOLIDATION.md:16; decomposition source docs/EXPERIMENTS_LEDGER.md:125-126 -->

### 2.2 Expert prediction / prefetch

Fate [2502.12224] predicts next-layer experts from the hidden state at ~99% hit rate; Pre-gated MoE [2308.12066], SiDA [2310.18859], and ExpertFlow [2410.17954] similarly exploit a cross-layer signal. **Crucially, these predict from the hidden state, not from the previous layer's expert identities.** Our axis-flip finding (§4.1) is often mis-stated as "cross-layer is random here"; the honest statement is narrower and does *not* contradict Fate (§4.1, §7.6).

### 2.3 Domain pruning — the static case is crowded prior art

Static, one-shot domain pruning of MoE experts ~50% near-lossless is **already known**, and we replicate rather than claim it: REAP [2510.13999] (Cerebras — Router-weighted Expert Activation Pruning, `mean(g·‖f‖)`, tested to 1026B, near-lossless at 50% on Qwen3-Coder-480B), EASY-EP [2504.06792] (stable per-domain subset on DeepSeek-R1 671B, half the experts, 2.99× throughput near-lossless), PreMoE [2505.17639], and NAEE [2402.14800] (expert skipping).<!-- src: docs/SCALE_FRONTIER_VERDICT.md:21-22 (EASY-EP, REAP-Cerebras details); docs/paper/PAPER_DRAFT.md:55 (domain pruning ~50% GIÀ-NOTO) --> **REAP [2510.13999] is a static one-shot compressor**; we build on its saliency and name our *dynamic* extension **REAP-LOOP** — REAP's criterion, cited, plus the loop that is ours (§5). Our delta versus this line is not a better selection criterion: it is engineering (a runtime bias-mask actuator on a stock 2-bit engine, §5.3) plus an admitted saliency downgrade (g-only, without ‖f‖, forced by what the trace exposes, §5.3, §7.7).<!-- src: docs/paper/PAPER_DRAFT.md:57 (g-only downgrade); docs/REAP_LOOP_NOVELTY.md:40 (REAP name = static one-shot, opposite) -->

### 2.4 Dynamic / per-session working-sets — prior art on the cache/prefetch path, not the router

Keeping a per-session working-set of experts resident and fetching the rest is called a "widely used approach" in the survey literature, and is realized by MoE-Infinity [2401.14361], ExpertFlow [2510.26730] (dynamic per-session working-set + learned routing + prefetch), LYNX [2411.08982], HOBBIT [2411.01433], and AdapMoE.<!-- src: docs/SCALE_FRONTIER_VERDICT.md:23 (ExpertFlow/LYNX/MoE-Infinity dynamic per-session; survey "widely used"); docs/REAP_LOOP_NOVELTY.md:35 --> Learning the mask from the first tokens of a session is GRIFFIN [2404.01365]; stepwise/gradual pruning to avoid collapse is Zhu & Gupta [1710.01878]; on-the-fly re-routing is [2510.14853].<!-- src: docs/REAP_LOOP_NOVELTY.md:31-36 (prior-art table) --> **The line of demarcation we draw and mark explicitly: every one of these acts on the cache / prefetch / offload path; none writes into the *live routing*.** REAP-LOOP's reversible gate-bias restricts which experts the router may select at all — that is the sliver we claim, and only the sliver (§5.3).<!-- src: docs/REAP_LOOP_NOVELTY.md:35 (⭐ prior-art touches prefetch/cache/offload, NOT routing → line of demarcation) -->

### 2.5 Degeneration detection — a different signal from ours

Detecting the onset of degenerate/looping generation is prior art, but every prior detector reads a *different* signal from ours: LoopGuard [2604.10044] uses attention/KV, LPSR [2604.18567] the residual stream, SpecRA (OpenReview xVO4BqmzVD) an FFT of the text, and ST-MoE [2606.15453] prefetch patterns.<!-- src: docs/REAP_LOOP_NOVELTY.md:6-8,37-38 (citations verified real 2026-07-06; each uses a different signal) --> **None uses gate-mass-on-the-pruned-experts** — a signal causally tied to the *action* of pruning — nor couples it to a bias-mask actuator on the routing. That coupling is our detection sliver (and note §8.4/Control C shows the router-side sensor is a *negative*: the practical trigger is textual).<!-- src: docs/REAP_LOOP_NOVELTY.md:11-13 (gate-mass-on-pruned unique to us); docs/INVENTIONS_LEDGER.md:43,48 (router sensor dead → textual detector) -->

### 2.6 Speculative decoding for expert-IO

DSpark (confidence-scheduled verification) schedules speculative blocks to save *tokens*; DwarfStar has union-load but only in the prefill batch path. Our candidate contribution is putting the **expert-IO term** into the block-length optimum: at block-k the verifier streams the *unique* experts of the block (union-load), not 6·k. This is the most genuinely new piece in the lever stack, and also the least demonstrated (§6.2, §7.8).<!-- src: docs/dspark/DESIGN_DSPARK_DS4.md:66; docs/briefs/BRIEF_DSPARK_MTP_DS4.md:32 -->

### 2.7 Honest novelty verdicts (updated with REAP-LOOP)

| Our finding | Verdict | Nearest prior art |
|---|---|---|
| REAP-LOOP closed loop (session-learn → tighten → breathe/re-learn → rewind) + causal asymmetry + gate-mass signal | **NOVELTY-CANDIDATE (composition + finding, not a primitive)** | GRIFFIN, Zhu-Gupta, μ-MoE, ExpertFlow (each a brick) |
| Reversible gate-bias restricting the *live routing* (vs cache/prefetch) | **SLIVER (line of demarcation)** | ExpertFlow, MoE-Infinity, LYNX (all cache/offload) |
| ~50% experts prunable near-lossless on-domain | **KNOWN** (we replicate) | REAP-Cerebras, EASY-EP, PreMoE, NAEE |
| gate-mass-on-pruned as degeneration signal | **NOVELTY-CANDIDATE (signal)** | LoopGuard (attn/KV), LPSR (residual), SpecRA (text-FFT), ST-MoE (prefetch) |
| Cross-layer hidden prediction + adaptive loop | **NOVELTY-CANDIDATE (robustness/transfer)** | Fate, Pre-gated, SiDA, ExpertFlow |
| Union-IO term in the block-length optimum | **CANDIDATE (arithmetic true; monetization OPEN)** | DSpark, DwarfStar union-load (prefill only) |
| Sub-expert weight-space redundancy = zero (3 families) | **NOVEL and ROBUST** (§4.5) | — |
<!-- src: docs/EXPERIMENTS_LEDGER.md:133-139 (verdicts); docs/REAP_LOOP_NOVELTY.md:16,29-40; sub-expert docs/EXPERIMENTS_LEDGER.md:55-56 -->

---

## 3. System Anatomy

All numbers in this section are measured on the target-class hardware (RTX 3060, 12 GB, WSL2) unless a proxy pod is named. This anatomy is the mechanistic reason the offload literature's "optimize expert IO" framing does not transfer to 12 GB — and the load-bearing motivation for REAP-LOOP (§5): if the *static* component is the ceiling, the correct lever is to *narrow the resident working-set*, not to cache the SSD tail better.

### 3.1 The central negative: activating the expert cache is always slower on the target

Across the entire committed corpus, **there is no run on the 3060 where enabling the expert cache improves tokens/s.** The best observed speed (0.70–0.72 t/s) is always with the cache *disabled*; every cache-active run is flat-to-slower:

| Config | Cache | hit-rate | gen t/s | source |
|---|---|---|---|---|
| event-on smoke | OFF (hit=0) | 0.0000 | **0.70–0.72** | <!-- src: runs/ds4_selected_upload_event_smoke/ ; HANDOFF.md:56-57,68 --> |
| Fase C spec2 | OFF (disabled 5.73≤6.00) | 0.0000 | 0.70 | <!-- src: runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log:18-28 (cache disabled), runtimes.csv --> |
| ext4 bench B (reserve 1 GB) | ON | 0.44 | **0.15** (4.6× slower) | <!-- src: runs/ds4_bench_clean_ext4_20260705/run_B1.log (hit_rate=0.4396, generation 0.15 t/s) --> |
| 9p sweep s740 (reserve 1) | ON | 0.38 | 0.11 (~6× slower) | <!-- src: runs/ds4_cache_sweep_9p_20260705/run_s740_r1.log:35-36 (hit 0.3791, gen 0.11) --> |

The mechanism-of-cure is thus, as measured, **anti-correlated** with the mechanism-of-thesis on this hardware. §3.2–3.5 explain why.

### 3.2 No compute stream: copies cannot hide behind compute

All MoE kernels launch on the CUDA default stream (0): `ds4_cuda.cu:12424-12599` launch `<<<grid,block>>>` with no 4th argument, and `cublasSetStream` has **zero** occurrences. The three non-default streams that exist are all host→device I/O. There is therefore no GPU pipelining of expert copy under compute; `cudaStreamWaitEvent` has **zero** occurrences in the whole engine — the primitive needed to order compute↔copy is simply absent.
<!-- src: docs/SPEX_INTEGRATION_PLAN.md:148-152 (default stream, cublasSetStream=0); :143-167 (cudaStreamWaitEvent ZERO). Line offsets valid for the analysis copy of ds4 at commit 80ebbc3; see §8.7 threat. -->

### 3.3 The stall floor: a blocking device-sync serializes everything

Safety today is guaranteed only by a double blocking barrier: `cudaDeviceSynchronize` at `ds4_gpu_end_commands` (:14286) framing every step, plus `cudaStreamSynchronize(upload_stream)` at `:2237` inside every copy. To prefetch layer L+1 you must predict its experts *before* that upstream sync — this is the central node of any async design, not a detail.
<!-- src: docs/SPEX_INTEGRATION_PLAN.md:143-182 (double barrier; 3 deeper nodes: upstream sync, shared upload stream, single-buffer destination cache g_stream_selected_cache.gate_ptr) -->

### 3.4 The SSD tail is the bottleneck, not the experts

On the 9p sweep the per-streamed-load stall is ~15–16 ms **regardless of cache setting** (s156 15.03 ms, s740 15.02 ms, smax 16.03 ms) — the signature of random per-expert SSD reads (6.75 MiB/expert), not sequential bandwidth. More cache raises hit-rate (0→0.38) but t/s stays tapped at 0.10–0.11 because the tail dominates.
<!-- src: runs/ds4_cache_sweep_9p_20260705/run_s156_1.log:33 (stall_ms_per_streamed_load=15.03), run_s740_r1.log:36 (15.02), run_smax_r0.log:34 (16.03) -->
**Caveat (do not overstate):** this SSD-tail signature is measured on the 9p mount (`DeepSeek-V4-Flash-IQ2XXS-imatrix.gguf` on /mnt/d, drvfs) — a filesystem with translation overhead — whereas the ext4 bench uses a *different model* (`ds4-2bit.gguf`) and does not emit the `streamed_loads`/`stall` metric. Attributing the ext4 flat-t/s to "SSD tail" is inference by analogy across two different setups (§8.2).

### 3.5 The VRAM ceiling disables the cache by default — static is the ceiling

With the default 6 GB reserve, available VRAM (5.73–5.96 GiB) is *below* the reserve, so the engine disables the expert cache entirely: model + MTP + context saturate the 12 GB budget. The cache only activates when the reserve is lowered to ~1 GB — at which point §3.1 shows it does not help. The static working set (attention + shared, ~10 GB) is the constraint; the expert cache (~1–2 GB) is what gets squeezed. **On 12 GB the static component is the ceiling, not the expert stream.** This is the anatomical hinge to REAP-LOOP: since the static component is the wall, the correct lever is to *shrink the resident expert working-set* so it fits alongside the static component and stays VRAM-resident — not to cache the SSD tail better. Consistent with this, a working-set narrow enough to live in VRAM *sidesteps* PCIe rather than accelerating it — a pod measurement of 23.6 t/s on dynamic keep-9%, above the ~11 t/s PCIe ceiling, points this way, but is **[PRELIMINARY — pod 3080Ti, confounded by cache cold/warm state and run order; absolute t/s does not transfer to the 3060, only the sidesteps-PCIe direction does]** (§5.2, §8.2–8.3).<!-- src: runs/ds4_bench_clean_ext4_20260705/run_A1.log:16 (available 5.96 ≤ reserve 6.00 → disabled); runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log:18 (5.73 ≤ 6.00); docs/PAPER_STATE.md:11 (I4/I5); 23.6 t/s docs/INVENTIONS_LEDGER.md:30,38 (keep-9 dynamic, VRAM-resident, sidesteps PCIe — pod 3080Ti, hit-rate transfers not absolute t/s; confounded by cache state + run order) -->

### 3.6 Per-token expert demand

At 6 active experts × 40 MoE layers = 240 expert reads per token; at 6.75 MiB/expert (2-bit) that is ~1.6 GiB of expert weight touched per token if nothing is cached — the raw driver of the SSD tail.
<!-- src: expert geometry 6.750 MiB/expert, 1.6875 GiB/layer — runs/reap/gguf_flash_expert_geometry.txt; runs/reap/reap_mask_ds4_domain.json (geometry). 240 = 6 active × 40 routed layers per meta.json -->

---

## 4. Routing Structure Findings

### 4.1 Axis-flip: naive expert-ID carry-over is random; the hidden-state axis is not

On the target model, predicting a token's layer-L+1 experts from its **layer-L expert identities** is random: same-token prev-layer top-6 = **0.0245**, prev-token prev-layer top-6 = **0.0251** — both at chance. The signal that survives is *temporal, same-layer*: prev-token same-layer top-6 = **0.2623**, window-4 same-layer top-6/12 = **0.342/0.507**, window-8 = **0.366/0.550**.
<!-- src: runs/ds4_routing_trace_smoke/routing_trace_count64_summary.json (same_token_prev_layer_top6=0.024523, prev_token_prev_layer_top6=0.025107, prev_token_same_layer_top6=0.262305, window4_same_layer_top6=0.341992/top12=0.506836, window8_same_layer_top6=0.365560/top12=0.549544) -->

**Critical framing correction (this is not a contradiction of Fate).** What is dead is the *expert-ID carry-over* — a strawman prefetcher nobody proposes. Fate and our own hidden-state predictor predict L+1 from the **hidden state** `h_L`, and on a proxy (Qwen-30B) that axis achieves recall **0.9316/0.9906/0.9978** @8/16/32 (25% budget ≈ 0.986–0.998), essentially Fate's ~99%.
<!-- src: docs/EXPERIMENTS_LEDGER.md:116 (E8 hidden .9316/.9906/.9978 dom); :218-220 (hidden @25% ≈ 98.6-99.8% ≈ Fate); :136 (G4: static recall @25% hidden .986/markov .865/rnd .25) -->
The defensible statement is: *on a real deployment model, the cheap ID-only predictor must be temporal-same-layer, not cross-layer-ID; the strong cross-layer signal requires computing the hidden state.* And the temporal signal is **weak in absolute terms** (0.37) versus markov (0.865) and hidden (0.986) — it is a zero-cost fallback, not a discovery.
<!-- src: docs/EXPERIMENTS_LEDGER.md:216-220 -->

**[PENDING] — degenerate prompt.** All axis-flip metrics come from a **single degenerate prompt** ("Conta da 1 a 200:", 65 tokens, 2600 rows, 3 positions). Our own design doc states this smoke "is NOT a base." A counting stream is quasi-periodic, which *inflates* the temporal signal and *depresses* the cross-layer signal — the most redundancy-favorable prompt possible. These metrics must be re-derived on the multi-domain trace before any claim (§7.6).
<!-- src: runs/ds4_routing_trace_smoke/README.md (prompt "Conta da 1 a 200:"); docs/REAP_DS4_design.md:207 ("NON è una base") -->

### 4.2 Temporal-window redundancy

Window-8 adds little over window-4 (top-12 0.507→0.550), suggesting a short recency window captures most of the temporal signal.
<!-- src: runs/ds4_routing_trace_smoke/routing_trace_count64_summary.json (window4_same_layer_top12=0.506836, window8_same_layer_top12=0.549544) -->

### 4.3 Cross-model replications (proxy Qwen/235B — preliminary/motivating study)

These are the **motivating study**, not target claims: the deployment model is 158B-Flash; Qwen-30B and 235B enter only to show the routing structure *replicates across families and scales*. The structure replicates on the *sign*, not always the *magnitude*:
- **Static-cache recall dominated by concentration in-distribution:** Qwen-30B gate @25% static .693 (general) / .744 (domain), markov +.10/+.13 cross-layer.
<!-- src: docs/EXPERIMENTS_LEDGER.md:111-112 (E3/E4) -->
- **Static prediction does NOT transfer cross-distribution** (static @25% collapses ~.75→.44), while markov holds its margin — a pro-SPEX argument under workload shift.
<!-- src: docs/EXPERIMENTS_LEDGER.md:113 (E5) -->
- **Routing concentration replicates across scale:** n\_eff 49.9/128 at 30B ≈ 49.9/128 at 235B. (Note: this is cross-scale *concentration* replication on proxies, distinct from — and not evidence for — the discredited "scale-invariance of the working-set lever," §1.4.)
<!-- src: docs/EXPERIMENTS_LEDGER.md:69 (C1: 235B n_eff 49.918/128); :222-230 -->

### 4.4 Extract-vs-explain: the "domain" in a trace is partly the template

Template matters: extract vs explain prompts yield different routing concentration, so "domain" signal in a trace is partly JSON-template artifact, not pure semantics — a stratification requirement for any routing claim. (This is the same effect §5.4 makes precise as "the working-set is per-output-type, not per-domain.")
<!-- src: runs/reap/2026-07-05_trace_dominio/meta.json (16 extract -n320 + 4 explain -n512); docs/CONSOLIDATION.md -->

### 4.5 The robust negative: sub-expert weight-space redundancy is zero across 3 families

The strongest *negative* in the corpus, and under-promoted: sub-expert per-neuron fusion buys nothing on three model families (Qwen-30B, DS2-Lite, V4-Flash) — neurons are diffuse, n\_eff/I ≈ 0.885 (base) / 0.890 (FT), and fine-tuning does **not** compact neurons (683 vs 679).
<!-- src: docs/EXPERIMENTS_LEDGER.md:55-56 (A25 n_eff 679.5/768=0.885; A26 683.3/768=0.890, FT NON compatta); docs/PAPER_STATE.md:14 -->
This falsifies a reasonable intuition ("fuse the sub-experts") on multiple families — a genuinely publishable negative that should be the *headline* negative, above axis-flip (§7.6).

---

## 5. The REAP-LOOP Loop — the headline contribution

> **Name.** We call the mechanism **REAP-LOOP**: it prunes on REAP's router-weighted saliency (§2.3, arXiv:2510.13999, Cerebras) and adds the dynamic loop that is our contribution — the loop is claimed, the saliency is cited. We considered fully-distinct names (**FOCUS** — *Focused On-line Curation of the Used-expert Set*; **WARDEN** — *Working-set-Adaptive Reversible Domain-Expert Narrowing*) but kept the REAP lineage explicit rather than obscure it. For the remainder of the paper "REAP-LOOP" denotes the dynamic loop of this section; "the static bias-mask" or "REAP-Cerebras" denotes the one-shot prior art we build on.

### 5.1 The mechanism — calibrate live, from the current session

REAP-LOOP is a training-free, weight-update-free, inference-time loop over the *routed* experts, actuated by the reversible bias-mask of §5.3. **It calibrates from the live stream — it is not tuned or pre-trained on any domain.** On a single stream it runs: full model with a healthy context and observed routing → at ~150 tokens, **learn a bias-mask from the experts *this current session* actually routed** → **tighten it stepwise** (keep-64 → 40 → 23 → keep-9%) via a −1e9 gate bias, staying VRAM-resident and coherent → run a distress sensor → on onset, **rewind** (or breathe: a periodic [~400 tok tight → ~80 tok keep-64 + re-learn] cycle that resets drift before it accumulates).
<!-- src: docs/REAP_LOOP_NOVELTY.md:18-26 (recipe, claimable sentence); docs/INVENTIONS_LEDGER.md:21 (recipe v1: full→session-mask→tighten→sensor→rewind), :31 (breath D6b: 400 tight/80 breath+re-learn), :17 (keep-64→40→23 clean to keep-9%) -->

Because the observation window is the *live* stream, the same loop handles any workload the user starts — a coding session narrows onto the coding working-set, a prose session onto the prose one — with no domain flag, no offline profiling pass, and no per-task tuning. The REAP saliency (§2.3) supplies the *initial* ranking that the live observation then overwrites; the domain-mask catalogue (§5.6) is the same recipe run once and cached, offered purely as a warmup-skipping convenience.

### 5.2 The causal asymmetry finding (the core)

The central finding: **tightening to keep-9% stepwise while context/KV is healthy holds; applying keep-9% cold collapses instantly.** Pruning a model *already in delirium* does not recover (the poisoned context wins); starting full and tightening keeps code/text clean down to keep-9%. The lever is **context health**, not gradient recovery. On the long horizon (2600 tokens) the *quality* asymmetry is confirmed: HOT stays ~94% clean, COLD onset ~70% (n=1, Control A, §8.4).
<!-- src: docs/INVENTIONS_LEDGER.md:17 (asymmetry: prune-in-delirium fails, full→step to keep-9% clean, cold collapses), :36 (Control A long-horizon 2600 tok HOT 94% vs COLD 69%, n=1), :50 (quality-asymmetry real HOT 94% vs COLD onset 70%; efficiency-asymmetry was artifact of broken mask) -->

**The metric that carries this claim is hit-rate, not throughput.** The **hit-rate** delta is the robust, transferable measurement: the REAP-LOOP session-mask cache-1024 (6.9 GB) reaches **hit 0.91**, versus the stock reactive cache-2048 (13.8 GB) at **hit 0.60 — at half the VRAM.** Hit-rate and downstream perplexity transfer across hardware; that is the head-to-head we stand behind.
<!-- src: docs/INVENTIONS_LEDGER.md:47 (head-to-head #5: loop 6.9GB hit 0.91 vs reactive 13.8GB hit 0.60, half-VRAM), :28,30 (hit-rate is the metric that transfers) -->

**[PRELIMINARY — confounded by cache cold/warm state and run order]** on the throughput side. The paired t/s figures (loop 9.22 t/s vs reactive 2.51 t/s, a nominal "3.7×") were measured on a pod (3080Ti, same 12 GB-class arch, not the 3060), and we do **not** present the 3.7× as a secured multiple: the baseline arm was itself running crippled (reactive cache on an SSD-bound trace) and the two arms were not cache-isolated or interleaved, so cold/warm page-cache state and run order are entangled with the reserve/mask setting. The honest steady-state datapoint is the **warm** keep-9% throughput of **~2.6 t/s** (session-mask band 2.4–4.5 t/s, hit 0.79 vs static 0.57).<!-- src: docs/INVENTIONS_LEDGER.md:18 (session-mask 4.5 vs static 2.4 t/s hit 0.79; steady-warm band) --> A clean throughput comparison — drop-caches between arms, one model, one filesystem, interleaved order, N≥3 seeds on the 3060 — is **still to be done** (§7.1, §8.2); until then the multiple is directional only. The absolute 3060-local t/s is separately **[PENDING 3060-local]** (§8.3).
<!-- src: docs/INVENTIONS_LEDGER.md:28,30,47 (9.2 t/s on pod 3080Ti not 3060; head-to-head 3.7× confounded by cache state + run order; hit-rate transfers, absolute t/s to be reconfirmed on 3060 with median-of-3 + cache isolation) -->

### 5.3 The line of demarcation, and the actuator

**What we claim (the sliver).** A reversible bias-mask that restricts the **live routing** (not the cache, not the prefetcher à la ExpertFlow), learned from the session, released on shift, integrated in ds4/DwarfStar single-stream. We do **not** claim: learning the mask from the session (GRIFFIN), gradual pruning (Zhu & Gupta), test-time per-prompt pruning (μ-MoE), or working-set residency (MoE-Infinity/ExpertFlow/LYNX — all on the cache/offload path, §2.4). We open by *disarming*: those are all prior art; we claim the closed-loop composition, the asymmetry finding, and the gate-mass signal.
<!-- src: docs/REAP_LOOP_NOVELTY.md:16,29-40 (do-not-claim table); docs/REAP_LOOP_NOVELTY.md:43 (open by disarming) -->

**The actuator (system contribution).** Domain pruning is actuated at runtime by writing `-1e9` into the selection bias (`exp_probs_b`) — exactly equivalent to physical pruning given the ds4 router math (independent per-expert probs + renorm over top-6), at ~zero cost, no surgery, on the stock 2-bit engine. Verified on the field: `selections checked=11280 violations=0` — with the mask active the router *never* selected a pruned expert.
<!-- src: docs/REAP_DS4_design.md:48-60 (router math); runs/reap/2026-07-05_eval_biasmask/reap/biasmask.log (V0_OK, 11280 checks, 0 violations) -->

**ID-space stability (why it composes).** The bias-mask preserves the *original* expert IDs (unlike physical surgery, which renumbers), so Markov / `.spex` / hotlist / SPEX-trace predictors remain valid across mask changes — REAP-LOOP is composable with the SPEX prefetch (§6.4) and the union-load decoder (§6.2) without re-dumping.
<!-- src: docs/INVENTIONS_LEDGER.md:72 (attuatore-stabile-in-ID-space: bias-mask keeps original IDs → composable with SPEX/Markov/trace, no re-dump) -->

### 5.4 Why the static mask does not suffice (the causal "why")

The working-set at extreme keep is **per-output-type, not per-domain** — and this is the causal reason static pruning (the prior art, GRIFFIN included) *cannot* reach coherent keep-9%: the right set changes per session/output-type, so a fixed mask is stale.
- Downstream perplexity at keep-9%: **session-mask 1.22×** vs full, **domain-mask 3.5×**, **random 9.3×** (Control B, §8.4). Session is near-lossless; domain destroys.<!-- src: docs/INVENTIONS_LEDGER.md:32 (Control B: session ppl 1.22× vs domain 3.5×) -->
- The session-mask **transfers** to another html/css/js task at **ppl 1.19×**; the domain-mask fails (3.5×) because "coding" mixes python/sql/rust/docker. Hierarchy: **output-type 1.2× ≪ domain-multilang 3.5× ≪ random 9.3×.**<!-- src: docs/INVENTIONS_LEDGER.md:35 (output-type 1.19× transfer, hierarchy 1.2×≪3.5×≪9.3×, random 9.34×) -->
- **Retraction (§8.5):** the true static mass-based mask, with the v3 cold-start fix, **collapses at 8% hit 0.99** — so **only the session-mask works** (session-learned is *necessary*, not merely better).<!-- src: docs/INVENTIONS_LEDGER.md:51 (true static-per-mass collapses at 8% hit 0.99 → only session-mask works, session-learned NECESSARY) -->

This is the finding that upgrades the demarcation from "different" to "load-bearing": the static prior art fails *where* the loop holds, and we can say *why*.

### 5.5 North-star on a real structured-extraction task

On a real structured-extraction task (domain-specific; scored by `eval_gold`), with learn-once-reuse (mask learned once from 4 warm items, reused cold on 10 drafts): aggressive keep-9% (K91) is **on par with the full model in knowledge** — **lenient field-acc full 0.70 = K50 = K91 = two-step 0.70.** Under strict scoring, full 0.583 vs masked 0.08–0.23, but the damage is **only formatting** (splintered keys, verbose JSON overrunning `-n 200`), **not knowledge** — a **hypothesized** production fix (+50 tokens or JSON-repair) that we have **not yet measured [OPEN]**. With hit 0.89–0.99, keep-9% on this task is operationally ~par with full in knowledge, and **learn-once-reuse is confirmed** — the demonstration that a mask *calibrated once from the live warm stream* can be reused across later drafts of the same output-type (the catalogue optimization of §5.6 in miniature).
<!-- src: docs/INVENTIONS_LEDGER.md:46 (north-star validated: lenient full 0.70 = k50/k91/two-step 0.70; strict damage format-only, fix +50 tok/JSON-repair; hit 0.89-0.99; learn-once-reuse confirmed) -->

### 5.6 A catalogue of domain masks — an optional pre-warm [WIP — not yet validated]

The live-calibrated loop pays a one-time warmup: ~150 tokens at the full (or wider) keep-K before the mask is learned and tightened. That warmup is **skippable, not essential.** The same recipe run once on a representative stream produces a **portable working-set mask** for that output-type; loaded at session start (`--mask-load`), it lets the loop begin already narrow and re-learn only on drift. This is purely an optimization of the *first* ~150 tokens — the mechanism, the asymmetry, and every quality result of §5.2–5.5 stand without it.

The catalogue's design is a direct corollary of §5.4: because the right working-set is **per-output-type**, a mask is a reusable asset keyed by output-type (the html/css/js mask transferred at ppl 1.19× to another html/css/js task), not by coarse "domain" (which mixes languages and fails at 3.5×).<!-- src: docs/INVENTIONS_LEDGER.md:35 (output-type mask transfers 1.19×, domain-multilang fails 3.5×) --> The natural **keep-K for a catalogued mask is derivable from its concentration**: n\_eff (the effective number of experts carrying the routing mass) sets a principled floor for how far a given output-type can be narrowed — a per-mask keep-K, not a global constant.<!-- src: docs/EXPERIMENTS_LEDGER.md:222-230 (n_eff = routing concentration; 49.9/128 at 30B); scripts/reap_neff.py (n_eff computed per trace) --> **[WIP]:** the catalogue is a design with a validated single instance (the learn-once-reuse north-star, §5.5); a multi-output-type catalogue with per-mask n\_eff-derived keep-K is **not yet built or evaluated.**

### 5.7 PACE — a Perceptive Adaptive Control Engine (in-engine, self-calibrating) [WIP — not yet validated]

The recipe of §5.1 is today driven by fixed schedules (tighten at ~150 tok, breathe every ~400/80). The intended end-state is a **controller that lives in the engine and calibrates itself from its own signals** — starting narrow and frequent, then relaxing as it earns confidence. The design (not yet implemented as a closed in-engine loop):

- **Reads its own signals, not an oracle.** Two cheap in-engine measurements bound the window from both sides: an **n-gram degeneration detector** on the emitted text is the *quality floor* (if the stream starts looping, the mask is too tight or stale → widen/re-learn), and the **cache hit-rate** is the *efficiency signal* (high hit-rate → the working-set fits → it is safe to tighten further). The controller adapts the keep-K and the breathe cadence live between these two.<!-- src: docs/INVENTIONS_LEDGER.md:48 (textual n-gram detector = the practical trigger; S1 drift is slow indicator); docs/INVENTIONS_LEDGER.md:18 (hit-rate as efficiency signal) -->
- **Reactive, not predictive — a consequence of the dead sensor.** The router-side "thermometer" (routing-Jaccard) is a measured **negative**: dead at every lag (§8.4, Control C), so nothing usable *predicts* the onset of degeneration from the router. The controller must therefore be **reactive** — it senses the floor being hit (textual loop, hit-rate collapse) and responds — rather than predictive. This is a design constraint imposed by an honest negative, not a choice.<!-- src: docs/INVENTIONS_LEDGER.md:43,48 (S2 dead all lags; router-side predictive trigger dead; reactive textual is the viable path) -->
- **Feels the hardware — emergent, not configured.** Because the efficiency signal *is* the hit-rate, the same controller adapts to the machine it runs on without being told the hardware class: on a slow disk a cold miss is expensive, so hit-rate pressure pushes the controller to tighten harder and breathe less; on a fast host it can afford a wider working-set. The hardware-adaptation is **emergent** from closing the loop on hit-rate, not a configured per-GPU table.
- **Stabilized by EWMA + annealing.** To avoid thrashing, the signals are smoothed (EWMA) and the exploration is annealed: **explore-frequent early** (breathe/re-learn often while the working-set is still uncertain), then **exploit** (tighten and hold once the mask has stabilized). This mirrors the F2 adaptive-cache result, where a confidence-EWMA policy (α=0.1, conf auto-calibrating 0.10→0.88) was never worse than reactive and −38–51% miss vs random at matched VRAM — evidence that an EWMA-scheduled self-calibration converges, in the *cache* setting; porting it to the *mask/keep-K* setting is the [WIP].<!-- src: docs/EXPERIMENTS_LEDGER.md:125-126 (F2/F3 adaptive EWMA α=0.1, conf 0.10→0.88, never worse than reactive, -38/-51% vs random) -->

**[WIP — not validated]:** the controller is specified, and its two component signals are each measured (textual detector works, §8.4; hit-rate-driven EWMA converges in the cache sim, F2) — but the **closed in-engine controller that adapts keep-K and cadence live from those signals is not yet built or run end-to-end.** The current loop uses fixed clocks.

---

## 6. The Lever Stack

Each lever is stated with its control and its failure mode. All share the REAP-LOOP actuator's ID-space stability (§5.3).

### 6.1 Cache sizing (including a clean negative on ext4)

More cache → more hits, but t/s does not follow. On ext4 (reserve default→1 GB): hit-rate 0→0.44 while generation t/s goes 0.18→0.15 (i.e. *worse*).
<!-- src: runs/ds4_bench_clean_ext4_20260705/run_A1.log (cache disabled, gen 0.18); run_B1.log (hit 0.4396, gen 0.15); run_B2.log (hit 0.4143, gen 0.15) -->
On 9p, across reserve 5→0: hit-rate 0→0.35–0.38, t/s pinned 0.06→0.11.
<!-- src: runs/ds4_cache_sweep_9p_20260705/run_s156_1.log (hit 0, gen 0.06), run_s440_r3.log (hit 0.2507, gen 0.10), run_s740_r1.log (hit 0.3791, gen 0.11), run_smax_r0.log (hit 0.3472, gen 0.10) -->
**This is a clean negative result** (the bottleneck is downstream of hit-rate, §3.4), but see §8.1–8.3: the runs are single-shot, non-randomized in order, use different models/filesystems, and are not iso-workload — so the *magnitude* is not defensible even though the *direction* is.

### 6.2 Block speculative decoding (union-load; Fase C unlock)

**Union-load arithmetic (transfers; TRUE).** At block-8 the verifier's unique experts per layer = **24.3 vs 48** (6×8) = **−49% expert-IO**; block-4 = 16.4 (−32%), block-2 = 10.6 (−12%). In the prefill batch the same mechanism gives mean **163.0 unique/layer** (min 132, max 191) vs 4020 slots = 95.9% avoided.
<!-- src: docs/briefs/BRIEF_DSPARK_MTP_DS4.md:30-32 (10.6/12%, 16.4/32%, 24.3/49%); runs/dspark/20260705_mtp_acceptance_pod3090/out_v3/unionload_compact_counts.log (43 prefill rows slots=4020, mean 163.0; decode rows slots=6 compact=6 must be filtered — see §8.1) -->

**Fase C unlock (mechanism runs on the 3060).** `--mtp` + `--ssd-streaming` was forbidden by the engine (`ds4.c:25685`); patches 0009 (one-liner from upstream PR #497) + 0010 (durable secondary MTP model-map registration) unlock it. The block verifier completes 100 tokens with no crash; spec2 acceptance = **56 committed=2 / 66 cycles = 0.848 [Wilson 95% CI 0.743–0.916]**; wall time base 186/181 s vs spec2 168/166 s = **0.910× [boot CI 0.893–0.928], i.e. ~10% faster** — but the CI rests on n=2 runs per arm (**[CI: n=2 per arm, not reliable]**; the direction is consistent, the magnitude is not defensible until reps are added).
<!-- src: runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log (56 committed=2 / 66 conf cycles); runtimes.csv (base 186/181, spec2 168/166, spec4 229); patches/ds4/0009,0010; CIs runs/paper_ci_results.json (fase_c_spec2 wilson [0.7431,0.9156]; spec2_over_base 0.9101 ci95 [0.8925,0.9282] n=2/arm flagged) -->

**[OPEN] — the 2× IO payoff is NOT demonstrated.** Fase C ran with the **expert cache disabled** the entire run (available 5.73 ≤ reserve 6.00, hit-rate 0.0000, ~190–270 GiB copied from SSD for 100 tokens), so union-load could not be monetized: block-4 is *slower* than baseline (0.49 vs 0.64 t/s). The tested regime is the worst case, not the thesis. "−49% → ~2×" remains **[OPEN]** on all hardware.
<!-- src: runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log:18-28 (cache disabled), runtimes.csv (spec4_r1=229s) -->

**DSpark drafter acceptance (proxy: 2×H200, fp8/fp4).** The trained DSpark drafter's conditional acceptance decays slowly on code/math and crashes on chat: code pos1–5 = 0.979/0.948/0.928/0.899/0.821 (τ=5.18/6), math 0.959→0.863 (τ=5.00/6), chat 0.713→0.500 (τ=2.70/6).
<!-- src: runs/dspark/20260705_dspark_b_v4flash_2xh200/RISULTATI.md:17-19 (acceptance + τ); :4 (fp8/fp4 167GB 2×H200) -->
**Caveat (§8.3):** this is a *different model* (DeepSeek-V4-Flash-DSpark), *different precision* (fp8/fp4 vs IQ2 2-bit), *different hardware* (H200 vs 3060), and *different drafter* (trained DSpark-5 vs native MTP-1). The native MTP-1 baseline on 2-bit is 0.872/0.846/0.604 (code/math/chat). The domain *ordering* (code>math>chat) transfers; the τ *magnitude* fp8→IQ2 does not, and must not be composed into a single speedup.
<!-- src: runs/dspark/20260705_mtp_acceptance_pod3090/RISULTATI.md:17-19 (MTP-1 130/149=0.872, 126/149=0.846, 90/149=0.604) -->

**STS + scheduler (offline, proxy drafter).** STS temperatures T=[0.85, 1.07, 0.71, 1.01, 1.44] calibrate the DSpark drafter's confidence; holdout ECE improves on 3–4 of 5 positions (pos1 0.058→0.039) but **regresses at pos3** (0.037→0.052). A dynamic-STS scheduler reaches 97–99% of oracle (all-domains 1.91× vs oracle 1.96×), and a fixed block on chat *destroys* the gain (fixed-5 = 0.92×, i.e. slower than plain decode).
<!-- src: runs/dspark/20260705_fase_b_sts/sts_params.json (T=[0.8499,1.0730,0.7136,1.0122,1.4359]); STS_REPORT.md (pos3 0.037→0.052 regress); SCHED_SIM.md (fixed-5 1.46× all / 0.92× chat; dynamic-STS 1.91×; oracle 1.96×) -->
**Caveat (§8.3):** the STS is fitted on the *proxy* drafter's logits and the cost-model constants (t\_fix, δ) are **assumed at 10%**, never measured; u(k)=6.30·k^0.668 is fitted on **4 points** from the degenerate 65-token trace. The whole scheduler table is a composition of three regimes and is never composed into a tokens/s claim.
<!-- src: runs/dspark/20260705_fase_b_sts/sts_params.json (source_csv strada B); src/msc/dspark/sched_sim.py:31 (TRACE_POINTS 4 pts), :33-34 (T_FIX_FRAC=0.10, DELTA_DRAFT=0.10) -->

### 6.3 REAP-LOOP bias-mask domain pruning (the static-eval controls behind §5)

This is the static, one-shot evaluation of the REAP-LOOP actuator (§5.3) — the controlled contrast that grounds the dynamic loop. **The claim is the ordinal contrast, not a near-lossless point.** Saliency-K50 on-domain ppl = 3.8604 vs full 3.8111 = **1.013× [CI 0.995–1.028]** (bootstrap 95%, geomean of 4 paired per-chunk ppl ratios, 10k resamples, seed 42); random control = 5.2001 = **1.365× [CI 1.280–1.455]**. The saliency CI **crosses 1.0** (statistically indistinguishable from full on-domain at N=4 chunks) while the random CI does not and sits far above it. **The defensible claim is therefore the contrast saliency ≪ random (non-overlapping CIs), not "1.013× near-lossless."**
<!-- src: runs/reap/2026-07-05_eval_biasmask/eval_summary.json (reap/dom 1.0129, rand/dom 1.3645, rand/gen 2.1149); CIs runs/paper_ci_results.json (reap_over_full_dom ci95 [0.9953,1.0277] crosses 1.0; rand_over_full_dom ci95 [1.2797,1.4549]) -->

**Three honest problems (see §8.5):**
1. **The near-lossless point is not statistically significant.** Paired per-chunk = [1.0332, 0.9868, 1.0112, 1.0212]; one chunk shows pruning *improving* ppl; mean-excess 0.0131 < cross-chunk stdev 0.0197 → paired t(3df) ≈ 1.32 < t\_crit 3.18, CI crosses 1.0. Only the contrast (rand t≈7.8, CI clear of 1.0) survives.
<!-- src: runs/reap/2026-07-05_eval_biasmask/eval_summary.json (paired_per_chunk reap/dom [1.0332,0.9868,1.0112,1.0212]); runs/paper_ci_results.json -->
2. **The pre-registered random threshold FAILED — no post-hoc rescue.** The verdict file reads `rand_dom 1.3645x (>=1.5 e >reap) -> FAIL`. A run README relabels this "PARTIAL" and blames IQ2 redundancy; **we reject that relabeling as HARKing.** The pre-registered criterion was `rand_dom ≥ 1.5×`; the measured 1.365× (CI upper bound below 1.5) **does not meet it — FAIL, full stop.** Defensible: the ordering (rand ≫ saliency, non-overlapping CIs). Three candidate explanations (granularity at E=256; shared-expert floor; 2-bit floor) are recorded strictly as **discussion to be tested at more aggressive K (e.g. K25) and multi-seed**, never as a relaxation of the threshold.
<!-- src: runs/reap/2026-07-05_eval_biasmask/eval_summary.json (verdict_preregistrato FAIL); runs/reap/2026-07-05_eval_biasmask/README.md:23-27 (PARTIAL relabel rejected) -->
3. **saliency/gen was never run** — the domain-vs-general cost on the target is unknown; only random/gen (2.11×) is measured. We cannot claim REAP-LOOP preserves the general.
<!-- src: runs/reap/2026-07-05_eval_biasmask/README.md (reap/gen not executed); eval_summary.json (only rand/gen present) -->

**Saliency is g-only (admitted downgrade).** ‖f‖ is not extractable from the trace, so we use g-only conditional-mean; validated at retention 0.902@K32 / 0.762@K64 on a *proxy* (Qwen-30B), not on the target. Footprint at K50 = 46.98 GiB. Split-half drop-set overlap mean 0.8295 (min 0.7578). We report both retention points to avoid the cherry-pick (§8.5).
<!-- src: runs/reap/reap_mask_ds4_domain.json (method gonly_conditional_mean, est_file_gib 46.98); runs/reap/gonly_vs_eq9_30b.json (retention@32 0.8969, retention@64 0.7619); runs/reap/mask_stability_splithalf.json (mean 0.8295, min 0.7578) -->

### 6.4 SPEX hidden-state prefetch — D2 measured on the target

A per-layer ridge probe fitted on the router's exact input (`ffn_norm`, 0007 trace patch, 32 documents / 3 domains, 263k pairs, per-document 70/30 split, 11 held-out documents) predicts the **next layer's** top-6 experts:

| set | probe@6 | probe@12 | probe@32 | recency@6 | recency@32 |
|---|---|---|---|---|---|
| domain (3 docs) | **0.680** | **0.829** | **0.920** | 0.282 | 0.598 |
| general-ITA (4 docs) | 0.538 | 0.665 | 0.797 | 0.484 | 0.730 |
| coding-EN (4 docs) | 0.455 | 0.581 | 0.728 | 0.374 | 0.658 |
<!-- src: runs/spex/fit_results.json; runs/spex/fit_full_32doc.log; scripts/spex_fit_predictor.py (ridge 10.0, seed 42); trace runs/spex/2026-07-05_trace_pod/ (patch 0006+0007, sha-verified 32/32) -->

On domain traffic the probe reaches **0.92 recall at a 12.5% prefetch budget** (32/256), +32pt over recency; on general/coding the edge is consistent but modest (+5–8pt) — regime-dependence coherent with §4. The next-*token* variant is uniformly weaker than next-*layer*, fixing the design as next-layer-centric.
<!-- src: runs/spex/fit_preliminare_15doc.log (15-doc preliminary floor); runs/ds4_stage1_prefetch_smoke/prefetch_l1_decode.log (hit 0); HANDOFF.md:84-85 (markov synthetic 0.0037) -->
**Still open [PENDING]:** the Python→C `.spex` export of the fitted probe and the in-engine hook (predictor → admission → async prefetch) — engineering, not measurement.

### 6.5 Skip-on-miss + REAP-LOOP loop (design + verified actuator)

Dropping a *low-impact* mispredicted/missing expert costs little (plateau ~1.3× ppl on the proxy hidden-drop), whereas indiscriminate markov-drop is catastrophic (59× @block-8). The runtime actuator exists and runs: margin-gated skip (`margin-skip drafted=2 committed=1`) fires in Fase C. The REAP-LOOP loop's sensor is the g-only conditional saliency from the trace; the `spex_loop.py` skip-on-miss integration itself is DSpark-faithful but **not run end-to-end** (markov-only + STS-on-hit-labels locally). (The *dynamic* REAP-LOOP loop *was* run end-to-end in the §5 north-star and head-to-head; this note refers to the `spex_loop.py` skip-on-miss integration specifically.)
<!-- src: docs/PAPER_STATE.md:9 (hidden-drop plateau 1.3× vs markov 59×); docs/EXPERIMENTS_LEDGER.md:290 (H5 hidden-drop C8 8.7× vs markov 59×, plateau 1.3×); runs/dspark/20260705_fase_c_smoke_3060/out/spec2_m3.log (margin-skip); docs/EXPERIMENTS_LEDGER.md:125,260 (spex_loop.py not run end-to-end) -->

### 6.6 WRAP — Working-set Resident Aggregate Prefetch: bulk host-side page-in of the working-set [WIP — not yet measured end-to-end]

The §3.4 SSD-tail signature is caused by **random, per-expert** reads (~15 ms stall per streamed load, 6.75 MiB each). Once REAP-LOOP has learned the session working-set (a known, bounded list of expert IDs), that entire working-set can be **paged into host RAM in one bulk, sequential-friendly pass** — WRAP, which fetches the whole known set host-side up front — so the decode loop then hits the page cache instead of issuing random SSD reads. This composes with the ID-space-stable actuator (§5.3): the working-set is exactly the set of IDs the bias-mask keeps, so WRAP's fetch list *is* the mask. It is the host-side complement to the SPEX device-side prefetch (§6.4): SPEX predicts *which* expert to bring to VRAM next; WRAP ensures whichever is asked for is already in RAM, not on disk.

**Status: implemented, not yet measured end-to-end.** The mechanism removes the random-SSD-read term that §3.4 identifies as the floor, and pairs naturally with the leverage already noted in memory (bulk RAM residency of the working-set via page-cache pinning). Its throughput payoff on the target is **[WIP]** — it must be measured against the §7.1 honest table under cache-isolation, and is one of the levers whose validation is gated on the same 3060-local clean-run work as the speed claims (§8.2–8.3).
<!-- src: docs/INVENTIONS_LEDGER.md:86 (leva-RAM NO_DIRECT_IO+KEEP_PAGES, page-cache residency); docs/INVENTIONS_LEDGER.md:73 (working-set warm host-side, hides 15ms/expert); §3.4 SSD-tail random-read floor runs/ds4_cache_sweep_9p_20260705/ -->

---

## 7. Evaluation

Every number carries its source. **[PENDING]** marks data not yet on disk (3060-local absolute t/s, multi-seed); **[OPEN]** marks a mechanism that runs but whose thesis payoff is undemonstrated.

### 7.1 Speed on target (the honest table)

| Regime | Cache | hit | gen t/s | Note |
|---|---|---|---|---|
| event-on / spec2 (3060) | OFF | 0.000 | 0.70–0.72 | best observed; cache off <!-- src: runs/ds4_selected_upload_event_smoke/; runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log --> |
| ext4 reserve-1 (3060) | ON | 0.44 | 0.15 | 4.6× slower <!-- src: runs/ds4_bench_clean_ext4_20260705/run_B1.log --> |
| 9p reserve-1 (3060) | ON | 0.38 | 0.11 | ~6× slower <!-- src: runs/ds4_cache_sweep_9p_20260705/run_s740_r1.log --> |
| RAM-hot trace (3090 pod) | — | — | 1.52 (gen), 24.23 (prefill) | **NOT transferable** to 3060 <!-- src: runs/reap/2026-07-05_trace_dominio/meta.json --> |

**Thesis verdict for the *stock* engine: "usable speed on 12 GB" has no positive datapoint on target. [OPEN].** The REAP-LOOP results in §7.2 are the mechanism's answer on the **quality** axis (hit-rate, perplexity, task-accuracy). On the **speed** axis they are **[PRELIMINARY]**: absolute t/s is [PENDING 3060-local], and the pod ratios are confounded by cache cold/warm state and run order (§5.2, §8.2). The one honest, non-confounded warm datapoint is **~2.6 t/s** keep-9% steady-state.<!-- src: docs/INVENTIONS_LEDGER.md:18 (session-mask warm band 2.4–4.5 t/s, hit 0.79) -->

### 7.2 REAP-LOOP loop head-to-head + north-star (real structured-extraction task)

| Metric | REAP-LOOP loop | Baseline | Ratio | Regime / status |
|---|---|---|---|---|
| Head-to-head **hit-rate** (the transferable metric) | 6.9 GB / hit **0.91** | reactive 13.8 GB / hit **0.60** | **hit ↑ at ½ VRAM** | pod 3080Ti; hit-rate + ppl transfer <!-- src: docs/INVENTIONS_LEDGER.md:47 --> |
| Head-to-head **t/s** | 9.22 t/s | reactive 2.51 t/s | nominal "3.7×" — **[PRELIMINARY: cache cold/warm + run order; baseline crippled]** | pod; **[PENDING 3060 abs t/s]** (§5.2, §8.2) <!-- src: docs/INVENTIONS_LEDGER.md:47,28,30 --> |
| keep-9 dynamic (warm steady-state) | **~2.6 t/s** (band 2.4–4.5) | static-mask 2.4 t/s | honest warm datapoint | pod; hit 0.79 vs 0.57 <!-- src: docs/INVENTIONS_LEDGER.md:18 --> |
| keep-9 dynamic peak | 726 tok @ 23.6 t/s | — | above ~11 t/s PCIe ceiling — **[PRELIMINARY: pod, cache/order-confounded]** | pod; VRAM-resident, sidesteps PCIe (direction only) <!-- src: docs/INVENTIONS_LEDGER.md:30,38 --> |
| Field-acc (lenient) | K50=K91=two-step 0.70 | full 0.70 | **par in knowledge** | eval_gold, learn-once-reuse <!-- src: docs/INVENTIONS_LEDGER.md:46 --> |
| Damage (strict) | masked 0.08–0.23 | full 0.583 | **format-only**; +50-tok fix **[OPEN, unmeasured]** | hit 0.89–0.99 <!-- src: docs/INVENTIONS_LEDGER.md:46 --> |
| Downstream ppl @ keep-9% | session 1.22× | domain 3.5×, random 9.3× | session near-lossless | Control B <!-- src: docs/INVENTIONS_LEDGER.md:32 --> |
| Cross-task transfer (ppl) | session-mask 1.19× | domain 3.5× | output-type ≪ domain | learn-once-reuse <!-- src: docs/INVENTIONS_LEDGER.md:35 --> |

### 7.3 REAP-LOOP domain pruning (paired, N=4 chunks)

| Config | ppl | ratio vs full [boot 95% CI] | paired per-chunk | source |
|---|---|---|---|---|
| full/dom | 3.8111 | 1.000 | — | <!-- src: runs/reap/2026-07-05_eval_biasmask/eval_summary.json --> |
| saliency/dom | 3.8604 | **1.013× [0.995–1.028]** (CI crosses 1.0) | [1.033, 0.987, 1.011, 1.021] | <!-- src: eval_summary.json; runs/paper_ci_results.json --> |
| random/dom | 5.2001 | 1.365× [1.280–1.455] (**pre-reg FAIL: <1.5×**) | [1.265, 1.295, 1.505, 1.406] | <!-- src: eval_summary.json verdict_preregistrato; runs/paper_ci_results.json --> |
| random/gen | 11.3014 | 2.115× [2.104–2.126] (N=2, CI insuff.) | [2.126, 2.104] | <!-- src: eval_summary.json; runs/paper_ci_results.json (n=2 flagged) --> |

Defensible claim = ordinal contrast saliency ≪ random only. saliency/gen **missing**.

### 7.4 Speculative decoding

| Metric | Value | Regime | source |
|---|---|---|---|
| MTP-1 acceptance (code/math/chat) | 0.872/0.846/0.604 | 3090, 2-bit, N=1 prompt/domain | <!-- src: runs/dspark/20260705_mtp_acceptance_pod3090/RISULTATI.md:17-19 --> |
| DSpark drafter τ | 5.18/5.00/2.70 | **H200, fp8/fp4** (proxy) | <!-- src: runs/dspark/20260705_dspark_b_v4flash_2xh200/RISULTATI.md:17-19 --> |
| Union-load @block-8 | 24.3/48 = −49% | arithmetic (transfers) | <!-- src: docs/briefs/BRIEF_DSPARK_MTP_DS4.md:32 --> |
| Union prefill | 163.0 unique/layer (95.9% avoided) | prefill rows only | <!-- src: runs/dspark/20260705_mtp_acceptance_pod3090/out_v3/unionload_compact_counts.log --> |
| Fase C spec2 acceptance | 0.848 (56/66) [Wilson 0.743–0.916] | 3060, cache **disabled** | <!-- src: runs/dspark/20260705_fase_c_smoke_3060/out/spec2_r1.log --> |
| Fase C spec2 wall ratio | 0.910× [0.893–0.928] (~10% faster; n=2/arm) | 3060, N=2, 1 prompt | <!-- src: runtimes.csv; runs/paper_ci_results.json --> |
| Scheduler dynamic-STS | 1.91× (oracle 1.96×) | **simulation**, proxy drafter | <!-- src: runs/dspark/20260705_fase_b_sts/SCHED_SIM.md --> |
| 2× IO payoff | **[OPEN]** — undemonstrated | — | — |

### 7.5 Routing structure

| Metric | Value | source |
|---|---|---|
| Axis-flip same-token prev-layer top6 | 0.0245 (random) **[PENDING multi-domain]** | <!-- src: runs/ds4_routing_trace_smoke/routing_trace_count64_summary.json --> |
| Temporal window-8 same-layer top6/12 | 0.366/0.550 **[PENDING]** | <!-- src: same file --> |
| Hidden recall @25% (proxy 30B) | 0.986 | <!-- src: docs/EXPERIMENTS_LEDGER.md:218 --> |
| Sub-expert n\_eff/I (3 families) | 0.885/0.890 (redundancy zero) | <!-- src: docs/EXPERIMENTS_LEDGER.md:55-56 --> |
| Split-half drop-set overlap | 0.8295 (min 0.7578) | <!-- src: runs/reap/mask_stability_splithalf.json --> |

### 7.6 SPEX hidden prefetch [PENDING-export]

D2 recall is measured on target (§6.4). The `.spex` exporter and in-engine hook are not implemented; all previously-wired predictors hit-rate ~0. No end-to-end prefetch t/s on disk.

---

## 8. Limitations and Threats to Validity

This section is the paper's real contribution: it separates the honest engineering study from a hobby demo.

### 8.1 Statistics: confidence intervals computed, but the sample floor remains

The ratios that *have replicates on disk* carry a percentile bootstrap 95% CI (`scripts/paper_ci.py`, 10k resamples, seed 42, `runs/paper_ci_results.json`): saliency/dom 1.013× **[0.995–1.028] — crosses 1.0, not significant**; rand/dom 1.365× [1.280–1.455] (below the 1.5× pre-reg threshold); Fase C acceptance 0.848 [Wilson 0.743–0.916]; spec2/base wall 0.910× [0.893–0.928]. **The CIs do not rescue the sample floor — they expose it.** saliency/dom is N=4 chunks (single run); rand/gen is N=2 (flagged); the wall ratio is n=2 per arm; the scheduler rests on u(k) fitted to **4 points** from a 65-token degenerate prompt; τ are single-prompt/domain. Additionally, the REAP-LOOP causal-asymmetry and ppl controls (§5.2, §5.4) are **n=1–2 seeds** — the qualitative direction is strong, the magnitude is not multi-seed. Single-shot runs pervade the cache sweeps. **Remaining fix (needs GPU): multi-seed/multi-chunk and N≥3 re-runs.**
<!-- src: runs/paper_ci_results.json (all CIs, seed 42); runs/ds4_bench_clean_ext4_20260705/run_A2.log (no generation line); src/msc/dspark/sched_sim.py:31 (4 points); docs/INVENTIONS_LEDGER.md:36 (Control A n=1) -->

### 8.2 Confounds: order, filesystem, model, and hardware are entangled

The cache benches run strictly A→B, non-randomized, with `KEEP_MODEL_PAGES=1` warming the page cache — page-cache/thermal warming is confounded with the reserve setting. **The same confound taints the speed head-to-head of §5.2** (the nominal 3.7×, the 23.6 t/s peak): the loop and reactive arms were not cache-isolated or interleaved, so cold/warm page-cache state and run order are entangled with the mask/reserve setting — which is exactly why we mark those throughput figures **[PRELIMINARY]** and stand only on the hit-rate delta. The "SSD-tail" thesis stitches two experiments with *both* a different model *and* a different filesystem, and the stall signature exists *only* in the 9p logs. The REAP-LOOP paired eval runs full/saliency on RTX 3090 but random on RTX **3090 Ti** — a hardware axis mixed into a ratio on an engine that is *not bit-reproducible at temp0* (repeat p00 diverges 1933/2720 rows = 71%). **Fix: randomized/interleaved order with drop-caches; one model, one filesystem; re-run the three configs on identical GPUs.**
<!-- src: runs/reap/2026-07-05_eval_biasmask/meta.json (full/reap 3090, rand 3090Ti); runs/reap/2026-07-05_trace_dominio/meta.json (DETERMINISM_DIFF 1933/2720) -->

### 8.3 Transferability: three tacit regime substitutions

The paper measures in one regime and applies in another: **(1) model** (Qwen proxy → V4-Flash target); **(2) precision** (bf16/fp8 → IQ2 2-bit — the DSpark τ and STS are fp8/fp4); **(3) hardware/regime** (RAM-hot pods + idealized sim → SSD-bound 3060). **We have retired every simulated tokens/s figure** (`spex_speed_sim.py` projected 45–74 t/s under two optimistic assumptions; the real engine does 0.06–0.70 t/s). Crucially for the REAP-LOOP headline: **the absolute t/s on the real 3060 is [PENDING]** — the 3080Ti-pod numbers (9.22 t/s, 23.6 t/s) transfer only in *hit-rate and perplexity*, not in absolute throughput; the 3060-local build is in flight. The one remaining simulation number (dynamic-STS 1.91×) is a *scheduler* simulation, labeled as such, never composed into a tokens/s claim.
<!-- src: src/msc/spex/spex_speed_sim.py:126; docs/PAPER_STATE.md:120 (retire sim tok/s); docs/INVENTIONS_LEDGER.md:28,30 (pod 3080Ti not 3060; hit-rate transfers, absolute t/s PENDING); dynamic-STS runs/dspark/20260705_fase_b_sts/SCHED_SIM.md -->

### 8.4 The three obligatory REAP-LOOP controls

The novelty verdict (`docs/REAP_LOOP_NOVELTY.md`) requires three controls to move the asymmetry from anecdote to finding. Status:
- **(A) Same mask hot vs cold — DONE (asymmetry confirmed, n=1).** Long-horizon 2600 tok: HOT ~94% clean vs COLD onset ~70%. Isolates *context health* from *better mask*. **[OPEN: multi-seed].**<!-- src: docs/INVENTIONS_LEDGER.md:36,50 (Control A) -->
- **(B) Downstream quality metric — DONE.** ppl at keep-9%: session **1.22×** vs domain 3.5× vs random 9.3× — a quantitative metric, not "renders clean."<!-- src: docs/INVENTIONS_LEDGER.md:32 (Control B) -->
- **(C) Sensor lead-time — NEGATIVE (honest).** The routing-Jaccard "thermometer" S2 is **dead at every lag (1–16)**; S1 shows real but slow drift (0.80→0.89, no knee) — an *indicator, not an alarm*. The practical trigger is therefore the **textual n-gram detector**, not the router. The clean router-side adaptive trigger is dead; fixed-clock/textual is the viable path.<!-- src: docs/INVENTIONS_LEDGER.md:37,43,48 (Control C: S2 dead all lags, S1 slow-drift indicator, textual detector only) -->

### 8.5 The retractions (disarming the reviewer before he swings)

We record the errors we found and fixed, because they are the credibility:
- **Cold-start mask-OFF bug (fixed v3).** In cold-start, REAP-LOOP v2 applied the mask at token ~1, then the model-load ranges *overwrote* it → the mask was silently OFF (85% of selections on pruned experts). Fix v3: re-apply every 32 tokens. Lesson: verify actuator adherence on **every** use-config, not only the first hot smoke.<!-- src: docs/INVENTIONS_LEDGER.md:41 (0011-V3 cold-start bug + fix) -->
- **"Static mass-based mask holds" was the full model in disguise.** With the v3 fix, the true static mass-based mask **collapses at 8% hit 0.99** → only the session-mask works (§5.4). We retract the earlier "static-per-mass holds" reading.<!-- src: docs/INVENTIONS_LEDGER.md:42,51 (retraction: static-per-mass was full-disguised; true static collapses) -->
- **The efficiency-asymmetry was an artifact of the broken mask.** The *quality* asymmetry (hot 94% vs cold 70%) is real; the *efficiency* asymmetry was the cold arms running with the mask off (cold-alive is actually fast, 11.4 t/s). We retract the efficiency claim; the quality claim stands.<!-- src: docs/INVENTIONS_LEDGER.md:50 (efficiency-asymmetry artifact; quality-asymmetry real) -->
- **REAP pre-registered FAIL, not "PARTIAL" (§6.3).** We reject the HARKing relabel; the random control missed its 1.5× threshold and we say so.<!-- src: runs/reap/2026-07-05_eval_biasmask/README.md:23-27 -->
- **Two-step (a user idea) was a non-result.** At n=10 on short drafts it was on par with the other configs — reported as a null, not spun.<!-- src: docs/INVENTIONS_LEDGER.md:49 (two-step non-result) -->

### 8.6 The scale caveat, restated

REAP-LOOP is single-stream, consumer/edge. It does **not** transfer unchanged to data-center serving: batching re-expands the activated-expert union (DeepSeek-R1 163/256 @batch32, 243/256 @batch64), and the DC bottleneck is the interconnect (NVLink 900 GB/s–1.8 TB/s, all-to-all), not PCIe. Survives as *same principle (working-set ≪ total), different regime* — not an identity of mechanism. The domain-sharded-serving hypothesis that could carry it to scale is **unbuilt**.
<!-- src: docs/SCALE_FRONTIER_VERDICT.md:14-18,34 -->

### 8.7 Reproducibility gaps (artifact evaluation)

1. **The 8 `domain_kb` prompts are withheld (domain-specific KB).** We commit the generator `scripts/spex_build_trace_prompts.py` and a sha256 manifest; `coding_en` + `generale_ita` are committed in full; a **public-domain surrogate is delivered** (`runs/reap/public_replica/`, `prodotto_json`) that replicates selection-matters + the V0 mechanism (0 violations) and near-lossless +5.2% — showing the domain-prune result is not a data artifact (only the strict "lossless" threshold is domain-specific). Pod terminated, $-.
<!-- src: runs/reap/public_replica/README.md; scripts/{reap_public_replica_corpus.py, reap_neff.py}; runs/spex/2026-07-05_trace_prompts/prompts_manifest.json -->
2. **The engine source is an ephemeral out-of-repo copy**; the §3.2–3.3 line offsets are not checkable by a cloner, and the analysis mixes commit `80ebbc3` (stock) with the patched tree. **Fix: pin the commit, cite `git show 80ebbc3:file:NNNN` per claim.**
<!-- src: docs/SPEX_INTEGRATION_PLAN.md:4-5,146 -->
3. **No run pins the ds4 binary commit, build flags, or GGUF sha uniformly.** **Fix: an `env_capture.sh` header (ds4\_commit, cuda, gguf\_sha256, gpu, filesystem) on every run.**
4. **Traces are not bit-reproducible** (temp0 non-determinism, 71% row divergence) and live out-of-repo. **Fix: commit the source trace CSV as a fixed artifact.**

### 8.8 Numbers cited in notes that are NOT on disk

Excluded until committed: gate-weights top1 27%/top3 63%; skip-mass 46%/29%; axis-flip "0.0203" (the real minimum on disk is **0.0245**). The previously-flagged `n_eff@64tok` is now on disk (46.4 count / 39.6 gate-weighted for a downstream application; 41.8 / 35.2 for the public surrogate).
<!-- src: absence confirmed via grep over docs/ and runs/; real min axis-flip runs/ds4_routing_trace_smoke/routing_trace_count64_summary.json (0.024523); n_eff scripts/reap_neff.py -->

### 8.9 Scoped future work — the WIP levers, and what would validate them

We separate the *validated* mechanism (§5.2–5.5) from the extensions that are designed and partially instrumented but **not yet validated end-to-end**. Each is stated with the specific measurement that would close it:

- **PACE — the in-engine self-calibrating controller (§5.7).** Specified; both component signals measured (textual detector works, §8.4; EWMA-scheduled adaptation converges in the cache sim, F2). **To validate:** build the closed in-engine loop that adapts keep-K and breathe-cadence live from the n-gram floor and hit-rate signal, and show it converges to the same operating point an offline sweep finds.
- **Domain-mask catalogue (§5.6).** One instance validated (learn-once-reuse north-star). **To validate:** a multi-output-type catalogue with per-mask n\_eff-derived keep-K, showing each catalogued mask matches a live-calibrated mask on its output-type.
- **WRAP — the bulk page-in (§6.6).** Implemented. **To validate:** measure decode t/s with the working-set bulk-paged into RAM vs the random-SSD-read baseline, under cache-isolation, on the 3060.
- **Cross-hardware study [WIP].** The optimal configuration is plausibly a function of *domain × hardware*: on a slow disk the cold-miss penalty is high, which should push the optimum toward a tighter keep-K and less breathing; on a fast host, wider. **This coupling is a hypothesis, not a measurement** — none of our runs sweep the *same* workload across GPU classes. **To validate:** run the identical workload + mask recipe across pods of different GPU class (and disk class) and map the optimum keep-K/cadence surface. This is also the study that would let the self-calibrating controller's *emergent* hardware-adaptation (§5.7) be checked against a ground-truth per-hardware optimum.<!-- src: docs/INVENTIONS_LEDGER.md:86 (leva-RAM: hardware-dependent cold-miss cost); §3.1-3.5 (12GB anatomy: cache-active slower is hardware-specific) -->
- **Offline auto-tuner [WIP].** A companion to the live controller: an offline sweep that finds **safe starting bounds** (initial keep-K, warmup length, breathe cadence) for a given hardware×output-type, both to seed the live controller and to serve as the ground truth that the live controller must be shown to converge to. **To validate:** implement the sweep, publish the bound table, and demonstrate the live controller (§5.7) lands within it. The F2/F3 EWMA-vs-oracle gap (dynamic reaches 97–99% of oracle in the cache sim) is the closest existing evidence that a live policy tracks an offline optimum, in the cache setting.<!-- src: docs/EXPERIMENTS_LEDGER.md:125-126 (F2/F3 adaptive within 97-99% of oracle in cache sim) -->

**Common blocker.** Every throughput-side item above shares one gate: the **cache-isolated, multi-seed, interleaved-order 3060-local run** (§8.2–8.3). Until that exists, the speed payoffs of the WIP levers — like the head-to-head multiple in §5.2 — are directional and flagged, not secured.

---

## 9. Conclusion and Artifacts

We set out to keep a **live-calibrated** working-set of a 158B MoE VRAM-resident on a 12 GB consumer GPU by extending an expert-offload engine down the memory ladder. The honest state: the **REAP-LOOP mechanism** — calibrate from the current session, not pre-train on a domain — is validated end-to-end on a pod in its **quality** claims (near-par on a real structured-extraction task, session-ppl 1.22× vs domain 3.5×, a causal asymmetry with hot/cold and downstream-ppl controls, and a robust hit-rate delta at half the VRAM), the **actuator** is exact (bias-mask ≡ pruning, 0 violations), the **routing structure** transfers on sign, and there is a **robust 3-family negative** on sub-expert fusion. But the **speed side is preliminary**: the pod throughput figures (a nominal "3.7×", 23.6 t/s) are **[PRELIMINARY — confounded by cache cold/warm state and run order]**, measured against baselines that were themselves crippled; the honest warm datapoint is ~2.6 t/s keep-9%, and a clean multiple awaits cache-isolated, multi-seed 3060-local runs. The **absolute throughput on the real 3060 is [PENDING]** (the pod transfers hit-rate and perplexity, not absolute t/s); the *stock*-engine product thesis "usable speed on 12 GB" has **no positive datapoint on target** (on 12 GB the static working set is the ceiling, the SSD tail is the floor, no compute stream hides copies, and every cache-active run is slower than cache-off); the sensor lead-time control is a **negative** (router-side dead, textual only); and the whole result is explicitly **single-stream/edge** — it does not transfer to batched, multi-node serving.

The value of this paper is the **measured edge mechanism (REAP-LOOP)** plus the **anatomy of the gap** between the offload literature and a real SSD-bound deployment, delivered with its negatives and retractions intact.

**Naming decision (author's call):** the mechanism is named **REAP-LOOP** — REAP's router-weighted saliency (arXiv:2510.13999, Cerebras) plus the dynamic loop that is the contribution; the loop is claimed, the saliency is cited, the lineage kept explicit. Fully-distinct alternatives (FOCUS, WARDEN) were considered and set aside.

**Artifacts (repo `moe-aggressive-commit`, `main`):**
- `docs/INVENTIONS_LEDGER.md`, `docs/REAP_LOOP_NOVELTY.md`, `docs/SCALE_FRONTIER_VERDICT.md` (findings, novelty verdict, scale verdict)
- `docs/EXPERIMENTS_LEDGER.md` (~100 experiments, anti-repetition ledger)
- `runs/reap/2026-07-05_eval_biasmask/` (biasmask eval, mask, split-half); `runs/reap/public_replica/` (public surrogate)
- `runs/dspark/{20260705_dspark_b_v4flash_2xh200, 20260705_fase_b_sts, 20260705_mtp_acceptance_pod3090, 20260705_fase_c_smoke_3060}/`
- `runs/ds4_{routing_trace_smoke, bench_clean_ext4_20260705, cache_sweep_9p_20260705}/`
- `patches/ds4/0001-0012`, `src/msc/spex/spex_loop.py`, `src/msc/dspark/{sts_fit.py, sched_sim.py}`
- `scripts/paper_ci.py` + `runs/paper_ci_results.json` (bootstrap 95% CIs, seed 42)
<!-- src (durable positioning): docs/CONSOLIDATION.md, docs/PRIOR_ART.md, docs/PAPER_STATE.md, docs/SPEX_INTEGRATION_PLAN.md, docs/HANDOFF*.md -->

---

## Acknowledgments

This work is built entirely on **ds4** (DwarfStar), the DeepSeek-V4 inference engine by Salvatore Sanfilippo (antirez), released open-source under the MIT license. Every measurement in this paper is produced by ds4 running the DeepSeek-V4-Flash model: the reactive per-expert SSD streaming, the LRU expert cache, the native MTP-1 speculative decoder, and the router math that makes the reversible bias-mask exact (§5.3) are all ds4's. REAP-LOOP and the lever stack are **extensions** to ds4, not a new engine — the contribution is a lever added to antirez's substrate, and the anatomy of §3 is an anatomy *of ds4* on consumer hardware. We thank antirez for building and opening ds4.
<!-- src: ds4 = antirez DeepSeek-V4 inference engine, MIT; baseline capabilities docs/paper/PAPER_DRAFT.md:24; router math docs/REAP_DS4_design.md:48-60 -->


# SPEX — Speculative Prefetch of EXperts

> Idea/spec (2026-07-03) da validare rigorosamente. Nasce come SISTEMA che sfrutta i findings
> dello studio pruning MoE (vedi note interne di progetto).

## ⭐ PERCHÉ SPEX È IL SEGUITO NATURALE DEL NOSTRO STUDIO (connessione — leggere prima)

Lo studio 30B→235B ha **misurato empiricamente le 3 premesse su cui SPEX si regge**:

1. **Gli expert freddi esistono e sono TANTI su dominio stretto.** Pruning: 50-70% degli expert
   parcheggiabili near-lossless su un dominio stretto (30B: full 56.0% → sal K50 57.3% → K70 54.7%).
   ⇒ La popolazione "cold" che SPEX offloada+prefetcha è REALE e ampia. Su dominio stretto, la
   maggioranza degli expert è offloadabile.
2. **Il routing su dominio stretto è CONCENTRATO ⇒ PREVEDIBILE.** committed_fraction@0.95 ≈ 0.52,
   N_eff ≈ 49.9/128, IDENTICO su 30B e 235B. Routing concentrato = alta correlazione temporale/cross-layer
   = alto ρ (il parametro di prevedibilità di SPEX) ⇒ **SPEX rende di PIÙ su deployment di dominio**
   (es. un deployment single-domain). Questa è la tesi-ponte: narrow-domain = SPEX-viable.
3. **Il router NON va mai bypassato** (SPEX predice il LOADING, mai il GATING). Il nostro studio ha
   FALSIFICATO l'hard-drop (accuratezza 0.00) → SPEX è strutturalmente sicuro perché il router reale
   decide sempre; la predizione decide solo cosa PRECARICARE. Accuratezza preservata gratis. Coerente
   con la lezione validata "mai droppare, sempre fetch".

**Inoltre abbiamo già l'infrastruttura per le trace reali**: gli hook su `mlp.gate` (`step_f.py`)
dumpano il routing per-layer top-k di Qwen3-30B/235B sui prompt del dominio → **trace reali per il predittore
di SPEX**, non solo sintetiche. E `prune_validate.py` (reroute nel gate) è già il meccanismo di
mascheramento expert. Riuso diretto.

Delta vs ricerca MoE precedente del progetto: il verdetto MoE era "predire-vs-reagire non batte la
reattiva quando cache ≥ top-k" (validate_probe_olmoe). SPEX raffina: predizione CROSS-LAYER (L→L+1)
con Markov+hidden + calibrazione STS + scheduler bandwidth-aware, nel regime cache≪N (dominio stretto,
tanti expert freddi) dove la predizione cold-start conta. Da ri-testare con questo framing rigoroso.

---

## SPEX — Speculative Prefetch of EXperts (spec originale)

### CONTEXT & MOTIVATION
Build and rigorously test SPEX: a predictive prefetching system that lets large MoE models run on
memory-constrained hardware by hiding expert-load latency behind compute. In a MoE, experts are
activated non-uniformly: few "hot", most "cold". To save VRAM, cold experts are offloaded to a slower
tier (RAM/SSD). When the router demands a cold expert, the load stall (SSD→VRAM, tens–hundreds of ms)
is catastrophic. SPEX (adapted from DeepSeek DSpark speculative decoding, Jul 2026): instead of waiting,
PREDICT which experts the next layer(s) need from the current hidden state, and START the SSD→VRAM
prefetch during current-layer compute. A lightweight loop verifies the prediction. A wrong prediction
costs only a fallback on-demand load — NEVER accuracy, because the real router always makes the final
routing decision. **Prediction informs LOADING, never GATING.** Structurally SAFER than DSpark (whose
verification is load-bearing for correctness); SPEX's router is never bypassed → accuracy preserved for
free; the verifier only decides whether to commit prefetch bandwidth.

### CORE HYPOTHESIS
SPEX wins iff `t_prefetch(expert) < t_compute(overlapping window)` — the expert loads from the slow tier
within the compute time of the overlapped layer(s). Characterize WHERE in (model size, expert size,
bandwidth, top-k) this holds; quantify end-to-end speedup and VRAM savings.

### ARCHITECTURE — 3 DECOUPLED COMPONENTS
1. **Predictor (Markov head + hidden-state features).** Predicts routing distribution over experts at
   layer L+1 given state at L. Two feature sources COMBINED: (a) hidden state h_L; (b) Markov embedding
   of previously-routed expert IDs (first-order transition, low-rank B=W1@W2, rank r default 64). Output:
   per-expert prob for next layer. CHEAP: target <1% of one MoE layer's compute — report cost explicitly;
   if exceeded, premise fails. Also a **pure-Markov-only ablation** (no hidden state) to isolate the
   hidden-state contribution.
2. **Confidence head + calibration (STS).** Lightweight linear+sigmoid → P(predicted expert is actually
   routed). Raw confidence is OVERCONFIDENT. **Sequential Temperature Scaling**: 1D grid search per
   position on held-out to minimize ECE of cumulative acceptance prob; order-preserving. Report
   calibration curves + ECE before/after; show uncalibrated confidence wastes bandwidth.
3. **Prefetch scheduler + verifier loop (bandwidth-aware).** Admission: prefetch e only if
   calibrated_conf(e) > τ. Given prefetch-bandwidth + VRAM budgets, decide how many/which predicted
   experts to prefetch, ranked by calibrated confidence. Eviction: **prediction-aware weighted LRU**
   (keep high near-future prob, evict low) vs plain LRU baseline. Verifier: on real router firing, record
   hit (prefetched+resident) vs miss (fallback load); feed back to running hit-rate; may adapt τ online.
   Verifier is a DETERMINISTIC policy in the hot path (microseconds), NOT neural. Neural runs ASYNC off
   the hot path, tuning policy params — never per-decision.

### SIMULATION SETUP (small→scale)
Simulated MoE: N experts (8→16→64), K layers, top-k (k=2,4). Load-balancing-realistic routing with
hot/cold SKEW (NOT uniform — the skew is the point). Routing data: (a) synthetic sequences with
controllable temporal/cross-layer correlation ρ (sweep predictability), or (b) REAL routing traces from a
small open MoE instrumented to dump per-layer top-k (PREFER real; synthetic as fallback + for ρ sweep).
Memory tiers, parameterized latencies: VRAM instant; RAM PCIe-bound (GB/s param); SSD (MB/s param).
Expert size param (50MB fp8 … 500MB fp16). Layer compute time param. ALL latency-critical numbers are
PARAMETERS (sweep the win/lose boundary), not hardcoded.

### EXPERIMENTS
Baselines (all three): **B0** no-offload/oracle (all VRAM, upper bound, ignores VRAM limit); **B1**
on-demand only (cold loaded on router demand, no prefetch — the latency to beat); **B2** static/naïve
prefetch (globally-most-frequent, state-independent — shows value of state-conditioned prediction).
Method: **SPEX (full)** = Markov+hidden predictor + STS + bandwidth-aware scheduler + prediction-aware
eviction + verifier. Ablations: Markov-only (no hidden); SPEX no-STS (raw conf); SPEX plain-LRU; SPEX
no-threshold (prefetch everything predicted).

### METRICS (paper-grade)
Prefetch hit-rate (predicted ∈ real top-k); cache hit-rate (demanded resident at router time);
end-to-end latency/decode step broken into compute / prefetch-overlapped(hidden) / on-demand-stall(exposed);
VRAM footprint vs accuracy/latency (headline trade-off); throughput under simulated concurrency;
predictor+scheduler overhead as % layer compute (small); calibration ECE + reliability diagram;
**CORRECTNESS: SPEX final routing IDENTICAL to B0/B1 (router never bypassed) — assert in tests, any
divergence is a bug, SPEX must be exactly accuracy-neutral.**

### SENSITIVITY SWEEPS (the real deliverable)
ρ (routing predictability — where does prediction stop helping?); expert_size/bandwidth ratio → map the
region where t_prefetch<t_compute = the **"SPEX-viable zone"** (key plot); τ (precision/recall of
admission vs wasted bandwidth); r (Markov rank), k (top-k), N (expert count) — scaling.

### OUTPUTS
Pareto frontier latency-vs-VRAM (SPEX vs all baselines); SPEX-viable-zone heatmap over
(expert_size, bandwidth); latency-breakdown stacked bars (compute/hidden-prefetch/exposed-stall) per
method; calibration curves before/after STS; ablation table with CIs over multiple seeds (mean±std, NO
single-run numbers).

### ENGINEERING CONSTRAINTS
Python. numpy/scipy/matplotlib for simulator+analysis; torch only if a real trainable Markov head is
needed (numpy prototype first). Clean separation: simulator core / predictor / scheduler / experiment
runner / plotting. Config-driven (all latency+model params in one config). Reproducible: fixed seeds,
multiple runs, results serialized so plots regenerate without re-running. Start N=8, synthetic traces, one
bandwidth regime — full pipeline (predictor→scheduler→verifier→metrics→plots) end-to-end on the smallest
case BEFORE scaling N or adding real traces.

### DELIVERABLES, IN ORDER
1. Simulator core + config + correctness assertion (SPEX ≡ baseline routing).
2. B0/B1/B2 baselines with latency-breakdown metric.
3. Markov+hidden predictor + training on traces + hit-rate report.
4. Confidence head + STS + calibration plots.
5. Bandwidth-aware scheduler + prediction-aware eviction + verifier loop.
6. Full experiment runner, sweeps, ablations, Pareto/heatmap plots.
Build incrementally, validate each stage against baselines before moving on; report results at each stage.

### PROSSIMA AZIONE (quando si parte)
Prima: usare gli hook `step_f.py`/`prune_validate.py` (già scritti) per DUMPARE trace di routing reali
per-layer top-k da Qwen3-30B-A3B (e 235B) sui prompt del dominio → dataset trace reale per il predittore
(feature-source (b) e ρ misurato dal dominio). Poi il simulatore numpy N=8 end-to-end.

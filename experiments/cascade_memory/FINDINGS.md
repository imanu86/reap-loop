# Confidence-Cascade Memory — Findings (Gate #1 GO, Gate #2 NEGATIVE)

Model-agnostic harness (OpenAI-compat), measured on **Qwen2.5-7B-Instruct (GQA)** via
vLLM on RunPod. Denominator always N_total; cost = deterministic token/call vector;
sensor and go/no-go **pre-registered** (PREREGISTRATION.md) before results. Two real
runs, ~$- (Gate #1) + ~$- (Gate #2) of pod time.

## Gate #1 — sensor separability: **GO** (full-window regime)
AUROC(mean_logprob, correctness) = **0.845**, CI95 [0.802, 0.883] on the baseline arm;
leak-control 0/400 (no leakage); P(high-conf ∧ wrong) = 0.007. The pre-registered
`mean_logprob` sensor discriminates right from wrong **when errors are hallucinated
values** (full window, abstention rate 1.7%).

## FINDING 1 — the sensor does NOT transfer to the small-window regime
When the native window is small, the model stops hallucinating and starts **abstaining**
("I don't know."). Measured mean_logprob by class in that regime:
`abstained −0.003 · correct −0.000 · wrong −0.011` — **all ≈ 0** in absolute terms.
As a SINGLE-threshold sensor over the whole population, `mean_logprob` AUROC drops from
0.845 (full window) to **0.685** here — NOT because logprob is information-free, but
because **abstentions (−0.003) sit between confident-correct (−0.000) and confident-wrong
(−0.011)**, so one threshold mislabels them. A single logprob gate therefore never fires
(all values ≈0 ≥ any theta), which is why theta is inert and the cascade collapses to an
abstention-gated single point. This is the spec's **risk #5 realized** — but see
**FINDING 3**, which corrects the too-strong reading "logprob is useless here": split the
population and the logprob is strong again (AUROC 0.937 conditioned on non-abstention).
The usable trigger is `abstention OR low-logprob-among-non-abstainers`.

## Gate #2 — does the cascade win? **NEGATIVE** (novelty not supported)
Small-native-window regime (native 6 / buffer 50 turns), 600 items, uniform fact-distance
{5,20,40,60,80} × {single_fact,distractor_dense} × {easy,hard}, k=3, abstention-gated.

| Arm | acc | mean tokens | rung_stop dist |
|---|---|---|---|
| B0_sw (no recovery) | 0.200 | 296 | native 1.00 |
| **B1 always-RAG** | **0.863** | **158** | — |
| B2 reactive (→ full-RAG) | 0.843 | 571 | native .29 / 2 .71 |
| B3 random (equal budget) | 0.887 | 705 | 0 .18 / 1 .12 / 2 .43 / native .27 |
| B4 cascade (cheap-first) | 0.885 | 793 | 0 .40 / 1 .01 / 2 .32 / native .27 |
| B5 cascade − rung-0 | 0.840 | 723 | 1 .35 / 2 .36 / native .29 |

Pre-registered criteria:
- **(i) FAIL** — B4 vs B1: `Δcost@matched-acc = +635 tok`, CI95 [600, 668] (needed < 0);
  the cascade is ~5× more expensive than always-RAG at matched accuracy. B4 vs B3:
  `Δ = +88`, CI95 [65, 109] — **cheap-first loses even to the random-order control at
  equal budget** → the ordering is not helping; it is hurting.
- **(ii) partial** — rung-0 resolves **54.5%** of recovered items (≥10% ✓) and its ablation
  B5 cannot reach B1's accuracy, so rung-0 does add value; insufficient to rescue (i).
- **(iii) FAIL** — stop-too-early (confident-yet-wrong native accepted) = **7%** (> 5%),
  the hallucination tail the logprob gate cannot catch.

**Verdict: NO-GO for the novelty claim at this regime.** Per the pre-registration this is
an honest negative; we do NOT proceed to Gate #3.

## FINDING 2 — why the cascade loses (mechanistic, robust)
1. **Always-RAG is a brutally strong cheap baseline here.** One in-memory BM25 retrieval
   with small top-k finds the fact ~86% of the time at ~158 tokens prefill. The cascade's
   "try native first, then escalate" adds a near-always-wasted LLM call (far facts abstain)
   before doing what B1 does directly.
2. **Cheap-first ordering backfires under uniform distance.** 60% of facts (dist 40/60/80)
   sit beyond the buffer, so buffer-scoped rung-0/rung-1 systematically FAIL and the cascade
   pays for them before reaching full-RAG (rung-2). Cost ranks B4 (793) > B3 (705) > B2
   (571): random order sometimes hits full-RAG first; reactive-binary skips straight to it.
   A cost cascade only wins when the CHEAP rungs have a good hit rate.

## FINDING 3 — the learned trigger works BLACK-BOX (corrects F1); white-box not needed (yet)
The interesting question was never the index ("where") but the trigger ("does the model
need memory?"). Offline probe (trigger_probe.py) on the already-collected runs, label =
"answered correctly WITHOUT recovery", features = ONLY inference-time black-box signals
(answer logprob stats + verbalized abstention), 5-fold CV, baselines = mean_logprob and
abstention-alone. Result (small-window regime, N=600):

| trigger | AUROC |
|---|---|
| mean_logprob alone (pre-reg sensor, whole population) | 0.685 |
| abstention alone (free signal) | 0.955 |
| **logprob among NON-abstainers only (the "confident-hallucination" residual, N=159)** | **0.937** |
| **full black-box [logprob stats + abstention]** | **0.984** |

Reading: two cheap black-box signals, split by role — (1) *abstained?* catches the 73%,
(2) *logprob among non-abstainers* catches the confident-hallucination residual — combine
to 0.984, beating both the raw logprob (0.685) and the free abstention signal (0.955).
This **corrects F1's over-strong "logprob is useless here"**: logprob is strong (0.937)
once applied to the right sub-population instead of thresholded over the whole mix. So the
model *does* know when it doesn't know — the signal was just being read wrong.
Consequence: on this task the **white-box hidden-state probe is NOT required** — a
two-signal black-box router suffices. (Full-window regime: logprob already 0.845; the
router adds nothing there — abstention is rare.)

CAVEATS (honest): all in-distribution on the synthetic fictitious universe, where answers
are exact tokens and abstention is clean; 0.984 will drop on realistic open-ended data
(LongMemEval) where hallucinations are subtler — that is where white-box probing might
re-earn its keep. The learned classifier's edge over abstention-alone (+0.029) is real but
modest; whether it converts into a better cost/accuracy frontier needs an ONLINE run
(apply the trigger live in the 2-tier memory, compare vs trigger=abstention-only) — a
clean pre-registered next experiment.

## FINDING 4 — E4: the learned trigger CONVERTS to a frontier gain (GO), but the 2-tier is still dominated by always-RAG
Pre-registered E4 (PREREGISTRATION.md): 2-tier (small window + one direct RAG) with the
learned trigger vs abstention-only, recency-skewed access (800 items, median distance 6,
51% in-window), amortized cost. Learned trigger = B2_reactive swept theta.

| arm / trigger | acc | amortized cost (tok) | note |
|---|---|---|---|
| B0_sw (no recovery, floor) | 0.425 | 275 | |
| **B1 always-RAG** | **0.944** | **153** | **dominates everything** |
| T_abst (B2, theta=-2, abstention-only) | 0.899 | 462 | baseline |
| T_learned (B2, theta=-0.001) | **0.929** | 478 | **+0.030 acc, CI95 [+0.019,+0.042], 1.04x cost** |

- **E4 verdict: GO** for the learned trigger over abstention-only: +0.030 accuracy (CI95
  excludes 0) at 1.04x amortized cost. The +0.029 AUROC edge from F3 **does convert** to a
  real frontier gain — catching the confident-hallucination residual buys ~3 accuracy
  points for ~4% cost. The trigger-router idea is validated online.
- **BUT** always-RAG (B1) dominates the whole 2-tier family: higher accuracy (0.944) at 3x
  lower cost (153 vs ≥462 tok). Cause: the native attempt (6-turn window ≈275 tok) costs
  MORE than a direct RAG (top-k=3 ≈153 tok), so "try the cheap tier first" is ill-posed —
  the native tier isn't cheap. B1 skips it and goes straight to the cheap RAG.
- Net: the learned trigger wins its pre-registered race, but the 2-tier is the wrong
  strategy in THIS regime. The trigger only earns its keep when skipping the RAG on easy
  (in-window) items actually saves — i.e. when **RAG is expensive** (large KB, embeddings,
  rerank, network) or the native window is cheaper than retrieval. With free in-memory
  BM25, always-RAG wins upstream. (Pre-registered boundary condition, not post-hoc.)

## FINDING 5 — the RAG-price threshold: when does the 2-tier finally beat always-RAG?
Counterfactual on the E4 data (cost_sensitivity.py, NO new pod): price an expensive
retrieval as `cost = total_tokens + C_rag * n_rag_calls` (C_rag = query embedding + rerank,
real compute the token count ignores). B1 pays C_rag on every item; B2 only on the ~55% it
retrieves.

- **Crossover C_rag\* = 722 token-equivalents.** Below it, always-RAG is cheaper (BM25
  in-memory ≈0 → B1 wins). Above it, the 2-tier is cheaper — because B2 skips retrieval on
  the in-window half.
- The threshold is high because B2 pays the native attempt on EVERY item (base 478 vs 153
  tok) and only saves 0.45 retrievals/item: 325/0.45 ≈ 722.
- Realistic? Only with **reranking**: a cross-encoder over ~15-20 candidates, or an
  LLM-reranker, clears 722 tok-equiv easily; plain BM25 or light embedding does not.
- **Even past the threshold it is a TRADE-OFF, not domination**: the 2-tier is cheaper but
  ~0.015 LESS accurate (B1 retrieves everything; B2's trigger misses a few confident
  hallucinations). That 1.5-pt gap is exactly what a **white-box (hidden-state) trigger**
  could close — the one place white-box probing would re-earn its keep.

## Boundary conditions (stated as caveats, not re-run to fish a positive)
The negative is specific to: (a) **uniform fact-distance** (adversarial to buffer-scoped
cheap rungs) — under **recency-skewed access** (most references recent, realistic for
conversation) cheap rungs would hit far more; (b) **cheap in-memory retrieval** — if
full-RAG were expensive (large KB, embeddings, rerank, network) avoiding it would pay;
(c) here 20% resolve natively — a more valuable native context shifts the trade-off.
These would each be pre-registered hypotheses for any follow-up, not post-hoc rescue.

## Net
- **DEAD**: the many-level cost cascade (turn-level rung-0 + heterogeneous rungs with
  early-exit) is **not supported** on a GQA model with cheap retrieval and uniform access —
  it is dominated by a single direct RAG (F2). When retrieval is cheap, the intermediate
  rungs have no cost niche to occupy; the design collapses to a **2-tier** (small window +
  one RAG).
- **ALIVE**: the **learned trigger router** (F3). Deciding *when* to go back — via
  `abstained? + logprob-among-non-abstainers` — reaches AUROC 0.984 black-box, beating both
  the raw sensor and abstention-alone. The hard problem was the trigger, not the index, and
  a cheap two-signal router cracks it on this task.
- F1 (regime-fragility of a single-threshold logprob sensor) stands but is corrected by F3:
  the signal is there, it was being read over the wrong population.
- MLA/ds4 substrate (pre-registered inert) not needed for the cascade negative.
- **E4 done (F4)**: the learned trigger CONVERTS its AUROC edge into a frontier gain over
  abstention-only (+0.030 acc, CI excludes 0, 1.04x cost) — GO. But always-RAG still
  dominates the 2-tier here because the native attempt costs more than the cheap RAG. So
  the trigger-router is real and validated, yet only pays off when RAG is expensive.
- **E4 expensive-retrieval answered by counterfactual (F5)**: the 2-tier beats always-RAG
  on cost only when retrieval costs > ~722 tok-equiv (realistic only with reranking), and
  even then it is a cost/accuracy TRADE-OFF (−0.015 acc), not domination. No pod needed —
  the threshold is exact arithmetic on the E4 data.
- **The one thing that could still turn it into a win**: a white-box (hidden-state) trigger
  to close the 1.5-pt accuracy gap, combined with expensive reranked retrieval. That is the
  only remaining open lead; everything cheaper is dominated by plain always-RAG.

## Full arc (one line each)
Gate #1 GO (sensor separates, full window) → Gate #2 NEGATIVE (many-level cascade dominated
by one RAG) → F3 (the learned *trigger* works black-box, AUROC 0.984, corrects F1) → E4/F4
GO (learned trigger converts to +0.030 acc over abstention-only) but 2-tier dominated by
always-RAG → F5 (2-tier only wins cost when retrieval > 722 tok-equiv, and only as a
trade-off). Net: the cascade idea dies; the trigger-router idea lives but pays off only
under expensive retrieval. Total pod spend ≈ $-.

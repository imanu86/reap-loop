# Phase-segmentation v2 on the REAL unmasked K0 trace + per-phase unmet-demand map

Date 2026-07-11. Offline (no-GPU). Supersedes the proxy numbers of v1
(`runs/ds4/20260711_phase_segmented_usage/`).

**What changed.** v1 segmented FULL-model traces of *other* prompts (coffee/python/
json) **filtered to the keep set first**, plus one K12-masked collapse trace. That
keep-filter was a structural blind spot: it discarded every router pick outside the
23-expert keep, so it could only ever *see* within-keep concentration. We now have
the **real full-router trace** -- `runs/ds4/20260711_k0_fullmodel_baseline/route_k0_cyberpunk.csv.gz`
(K0, no mask, all 256 experts eligible, native top-6, weighted; 3999 decode tok x
40 MoE layers) -- of the **same cyberpunk prompt that collapses under mask**. So we
read the router's TRUE per-phase demand and measure exactly which experts the
coffee-tuned masks prune. Script: `phase_unmet_demand.py`; raw: `results.txt`;
targeting map: `targeting_map_K23.csv`.

Phases = real HTML regions of the generated page (`gen_k0_cyberpunk.txt`): **head**
(preamble+`<head>`, tok 65-245), **css** (the giant `<style>`, 245-3051, 2806 tok =
70% of output), **body** (markup, 3051-3980, 929 tok), **js** (the `<script>`,
3980-4063, 84 tok, truncated at ntok=4000). No per-token text file exists -> boundaries
are char-offsets of `<!DOCTYPE>/<style>/<body>/<script>` mapped proportionally onto
the token axis (char-prop body_start=3051 vs task hint ~3200; a +/-150-tok robustness
sweep leaves every conclusion unchanged).

---

## VERDICT

**(a) Concentration / shift -- the proxy was WRONG IN BOTH DIRECTIONS (keep-filter artifact).**

| metric | v1 PROXY (keep-filtered) | REAL full router (this run) | direction |
|---|---|---|---|
| concentration k90 / layer | ~9 of 23 | **47-76 of 256** (Gini 0.79-0.88) | proxy **over**-concentrated |
| hot-core shift, consecutive | Jaccard 0.61-0.70 ("stable backbone") | **between-phase 0.21-0.24** (~78% rotates); within-phase 0.35-0.51 | proxy **under**-rotated |
| instantaneous WS (window 50) | ~370, "fits 394 at median" | **1460-1710 @90%** (4x budget) | proxy massively **under**-counted |

The proxy's "concentrated per-phase, fits VRAM" story is an artifact of filtering to
the keep before measuring. The real router is **wide** (needs 47-76 experts/layer to
cover 90% of any phase) **and strongly phase-rotating** (the top-6 hot-core turns over
~78% between HTML phases while staying only ~half-stable inside a phase). So the
phase-structure signal is REAL and *stronger* than v1 showed -- but the working set is
far bigger than v1 could see.

**(b) The targeting map -- FEW experts, if you target the right service level.**
Covering 90% of the true demand needs 33-57 admits/layer (impossible in budget). But
the mask was never built to serve 90%: in-domain (coffee, the mask's own session) it
covers only **57.6%** (K23) of the full router's mass and stays usable -- that 42%
displacement is what skill-sharing absorbs. To lift each off-domain cyberpunk phase
back to that **in-domain service level costs only 2-5 experts/layer** (body max 12):

| phase | K23 unmet% | K12 unmet% | admit/layer to in-domain (med / max / sum) |
|---|---|---|---|
| head | 49.6 | 60.2 | **2** / 7 / 80 |
| css  | 52.1 | 63.1 | **2** / 9 / 117 |
| **body** | **59.7** | **70.5** | **5** / 12 / 208 |
| js   | 55.7 | 66.0 | **3** / 9 / 115 |

Few -> the admission-controller thesis **survives in a corrected, sharper form**: not
"the keep already fits every instant" (false), but "the per-phase *shortfall* over the
tolerated in-domain level is small and phase-targeted." The concrete per-layer expert
lists are in `targeting_map_K23.csv`.

**(c) Does the instant set fit VRAM (394 slots)?** Split answer:
- At **90% of true demand: NO** -- window-50 residency is 1460-1710 (~4x over). The wide
  task's full working set does not fit the 3060. The v1 "wall softens to a speed-bump"
  conclusion is **retracted** -- the wall is real and larger than v1 measured.
- At the **in-domain service level** (the realistic target): head 362, css 358, js 375
  all **FIT 394**; **body = 448 (max 561) does NOT**. Coffee in-domain reference at the
  same window/level = 338 (fits). So a perfectly-targeted controller covers 3 of 4
  phases in-budget; **body is the one phase that overflows even when targeted.**

**(d) Is the max-unmet phase the collapse phase? YES -- triple alignment on the styling/body demand.**
The masked K12-cyberpunk run (`20260711_instrumented_collapse`) drifts (entropy
z-onset pos 125) and breaks into **malformed CSS** ("#finto sfondo cyberpunk", pos 190)
then a repetition lock -- i.e. it fails exactly as it must serve the elaborate
cyberpunk styling demand. That region is where our three independent stress metrics
peak: **body** has the highest unmet (59.7% K23 / 70.5% K12), needs the most admits
(5/layer, 12 max), and is the only phase whose in-domain-service residency (448)
overflows 394; **css** is the close second and where collapse empirically crystallizes.
The unmet gradient head(49.6)->css(52.1)->body(59.7) -- all above the in-domain-absorbed
42% -- tracks the drift->break->lock trajectory. The masked router, frozen to the coffee
keep, cannot follow the ~78% head->css hot-core rotation -> it cannot reach the styling
experts -> collapse. **Targeted admission at the css/body demand is precisely what would
prevent it** (2-5 experts/layer), and body is the residency-hard case the 3060 budget
alone can't cover.

---

## What this means for REAP-LOOP

- **Corrects yesterday's optimism.** v1's "instant fits 394" was a keep-filter artifact.
  The real full-router working set is ~1500-1700/window -- the 3060's 394-slot cache
  cannot make this wide task lossless. K23/K91 residency-starvation is confirmed, not
  softened.
- **But the controller target is cheap and precise.** Don't chase 90% of demand; admit
  only the per-phase shortfall vs the known-tolerable in-domain displacement. That is
  **2-5 experts/layer**, concentrated on the css->body styling region -- a small, phase-
  keyed swap the router itself signals (78% hot-core rotation at the head->css boundary;
  entropy z-onset leads the break by ~57 tok per the instrumented run).
- **Body is the budget frontier.** Even ideal admission leaves body at 448 > 394 slots.
  Serving wide cyberpunk-grade pages losslessly on the 3060 needs either a larger cache
  or accepting body-phase misses; head/css/js are reachable in-budget.

## Caveats (binding)

- **Phase boundaries are char-proportional** (no per-token text file). The +/-150-tok
  body_start sweep is stable (body unmet 59.1/59.7/61.3; body>css robust). CSS tokenizes
  denser than prose, so the true css span is slightly larger -- this only *strengthens*
  the css-demand story.
- **Unmet = displacement, not quality loss.** A masked router re-normalizes its top-6
  over the keep; "unmet %" is the mass the full router *would* have sent to now-pruned
  experts, which skill-sharing may or may not absorb. The in-domain coffee level (42%
  K23, usable) anchors "tolerable"; cyberpunk css/body exceed it and collapse.
- **Position alignment across masked/unmasked is qualitative.** The K12-masked run
  diverges early, so its token positions do not map 1:1 onto the K0 trace; the alignment
  claim is on the *demand region* (styling/body), not exact token indices.
- **n = 1 prompt (cyberpunk) + 1 in-domain control (coffee).** When the masked REAL
  route traces land (agent a5464bde), cross them in: measure what the *constrained*
  router actually does per phase and confirm the admitted set matches this demand map.

Reproduce: `python runs/ds4/20260711_phase_segmented_v2_real/phase_unmet_demand.py`

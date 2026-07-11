# podC — E-DET live shadow validation (S1 onset detector, log-only)

**Live (shadow, log-only) validation of the E-DET recommended S1 detector
profiles** from `runs/ds4/20260710_edet_s1_detector_tuning/REPORT.md`. Those
profiles (ARM `k_s=0.5 h_s=4`; FIRE `k_s=1.0 h_s=8`; EWMA `a=0.50`, sigma
self-calibrated over the first 128 tok, lagged baseline lag32/win128) were tuned
**offline** and had **never been checked on live runs**. Here the per-layer S1
sensor (patch 0012, `DS4_REAP_SENSOR_LOG`) was captured on live static-mask REAP
runs, then the recommended EWMA-CUSUM detector was **replayed offline** on those
logs (`scripts/tune_s1_detector.py det_cusum` reused verbatim) and its arm/fire
positions compared with the real text collapse onset (grading + textual onset).
Detector state is **shadow only** - nothing was armed or rewound live.

Pod / build / model / env: see `manifest.json`. Community RTX 3090, cu1290
image, gate PASS (a first community machine failed the T1 broken-UVM gate and was
terminated pre-download). Binary = the same `livetree-771a39a8` build that
produced the offline reference series A (K91) and C (aggressive pod r1). **All
t/s are pod, sensor-trace-on => diagnostic, NOT benchmark.**

## Runs

| run | regime | prompt | budget | ctx | finish | L | tps(pod) |
|---|---|---|---:|---:|---|---|---:|
| run1_r00 (==r01) | W50 **static K23** | cyberpunk | 4000 | 8192 | length | **L0** | 2.9 |
| run2_r00 (==r01) | W50 **static K38** | cyberpunk | 4000 | 8192 | length | **L0** | 3.0 |
| run3_r00 | W50 **static K23** | coffee | 3000 | 4096 | **stop@310** | **L1** | 2.4 |

Determinism (static greedy): run1_r00 == run1_r01 **byte-identical** in *both*
`content.txt` (`83fa8804...`) *and* the per-layer `s1.csv` sensor (`b2786bd5...`);
run2_r01's first 12000 sensor rows are byte-identical to run2_r00. So the router-
mass **sensor reproduces byte-identically**, not just the text - the mandated
n=3/n=2 collapse to their first run; redundant repeats were skipped
(`determinism_notes.txt`), matching pod1's armA "3/3 byte-identical".

**Coordination with pod1 (armB):** pod1 committed armB (`0962c6b`): static K38
n=3 is **L0, 0/3 `</html>`, byte-identical loops**, verdict *"static (K23 AND
K38) does NOT rescue cyberpunk ctx8192/4000; collapse is selection-policy-robust
(static == rotate)"*. My run2 K38 (L0, loop, no `</html>`) **reproduces armB** -
so per the mission note, run2 stands as a valid second aggressive sample, now
carrying the S1 sensor armB lacked.

## Shadow-replay table (run x onset / ARM@ / FIRE@ / lead / falsi / missed)

Positions are `spex_trace_pos` (= prompt_len + generated-token index). Two text
onsets are reported: **soft** = first mask-induced incoherence (doubled/mangled
tags -> the "doc-restart" onset the sibling `onset_probe.py` uses), **lock** =
first tight repetition-lock (the offline REPORT's char-proportional ground truth).

| run | S1 mean+-std | S1 drift/run | soft onset | lock onset | ARM@ (n) | FIRE@ (n) | ARM lead vs lock | FIRE lead vs lock | vs soft onset | falsi | missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **run1 K23 cyber** | 0.900 +- 0.034 | +0.057 | pos ~120 (gen ~42, `<html><html>` double) | pos ~228 (gen ~150, `/* Stili di colore */` xN) | **220** (5) | **230** (4) | **+8** (~coincident) | **-2** (late) | ARM -100 / FIRE -110 (**late**) | - | soft: both **missed** |
| **run2 K38 cyber** | 0.833 +- 0.034 | +0.084 | pos ~161 (gen ~83, `<style:` -> comment-list) | pos ~1709 (gen ~1631, `P equita/P giustizia` xN) | **414** (4) | **790** (3) | **+1295** (lead) | **+919** (lead) | ARM -253 / FIRE -629 (**late**) | - | soft: both **missed** |
| **run3 K23 coffee** | 0.860 +- 0.017 | +0.019 | - (no collapse; completes L1, `</html>`) | - | **377** (2) | **503** (1) | - | - | - | **ARM 2, FIRE 1** | - |

## Reading the results

**1. Static K23 (aggressive) collapses inside the detector's own calibration
window => no usable lead.** S1 is high (~0.90) from the first token and drifts up
+0.057, but the text is mask-degenerate almost immediately (doubled `<html>` by
gen ~42) and tight-loops by gen ~150 (pos ~228). The detector cannot fire until
its sigma-calibration (128 samples) completes at pos ~206 - i.e. **calibration
finishes ~22 tok before the lock, on an already-degenerating signal**. ARM
therefore fires at pos 220 (essentially *at* the lock, +8 tok) and FIRE at pos
230 (-2, late). Against the *soft* onset both are ~100 tok late. **This is the
"collapse faster than the detector horizon" signature the offline sections 0/5
anticipated** - the S1 airbag buys no lead in the aggressive static regime.

**2. Static K38 (milder) is a genuine slow-erosion regime for the *lock*, and
the detector leads it by ~900-1300 tok.** S1 is lower (~0.833) and drifts up more
(+0.084); the tight loop-lock is far away (gen ~1631). ARM fires at pos 414
(+1295 vs lock), FIRE at pos 790 (+919 vs lock) - large, real lead over the
terminal lock, matching the offline claim that lead exists **only** where a
long pre-lock horizon exists. **Caveat:** K38's *coherence* is already lost at
gen ~83 (mask emits an empty comment-list, never real CSS), so both profiles are
still **late relative to first-incoherence** (~250-630 tok late) - the lead is
over the terminal repetition-lock, not over the onset of garbage.

**3. Coffee K23 completes (no collapse) yet the detector false-fires - at
~offline FA_real rates.** The compact task finishes a structurally complete but
defective page (L1, `</html>`, malformed CSS/JS) at 310 tok. There is no lock,
so ARM's 2 fires and FIRE's 1 fire are **false alarms**: ARM 2/310 ~ **6.5/1k**
(offline FA_real 7.98/1k), FIRE 1/310 ~ **3.2/1k** (offline FA_real 1.81/1k,
FA_pod 0.0). So the live FA rates are the **same order** as the offline
estimates - the FA calibration holds live - but the conservative FIRE/rewind
profile is **not** zero-FA on a mildly-drifting *completing* run. The mission's
"extended-horizon slow-erosion" hope for coffee K23 did **not** materialise: the
narrow task just completes, it does not erode.

## Hazard ladder (fits the sibling pod runs)

Onset of degeneration is monotone in mask keep-count (softer mask => later onset),
consistent across the fleet:

| mask | soft/doc-restart onset | lock onset | S1 pre-lock horizon | detector lead over lock |
|---|---|---|---|---|
| K12 (podA) | ~gen 19 | early | none | (none - far inside calib) |
| **K23 (podC)** | ~gen 42 | ~gen 150 | none | ~0 (coincident) |
| **K38 (podC)** | ~gen 83 | ~gen 1631 | long | +900...1300 tok |
| K91 (offline A) | ~gen 2286 | gen 2476 | ~190 tok | ~210-225 tok |

## Verdetto - i profili offline sono confermati o da ritarare?

- **Confermati come *calibrati*** (non da ritarare nei parametri): live FA rates
  ~ offline FA_real; the detector fires only on genuine S1 drift; the
  regime-split direction is exactly as the offline study scoped it - **lead
  exists only where a coherent pre-lock plateau exists** (K91, and now K38 for
  the *lock*), and **not** for aggressive masks (K23) whose collapse is inside
  the detector's calibration horizon.
- **Il regime-split live CONFERMA l'offline** - con due precisazioni che l'offline
  non poteva vedere:
  1. La leva dello split e l'aggressivita della mask (keep-count), non
     static-vs-rotate: la K23 statica collassa come la rotate32 aggressiva
     (nessun lead), la K38 statica erode lento (lead sul lock). Coerente con
     armB (static == rotate).
  2. Sul prompt cyberpunk ogni mask statica perde coerenza (garbage mask-indotto)
     entro gen ~40-90, prima che S1 sia derivata abbastanza da armare: il lead di
     S1 e quindi solo sul repetition-lock terminale, mai sull'onset di
     incoerenza. La ground-truth offline (lock) lusinga il detector; contro la
     prima-incoerenza il detector e in ritardo in ogni run che collassa.
- **Operativo:** l'airbag S1-slope resta utile solo per mask abbastanza miti da
  conservare un plateau coerente (classe K91/K38-mild) e per l'azione economica
  (ARM = relearn/admit). Per il regime operativo della REAP-LOOP viva (mask
  aggressive tipo K23/rotate32) non da lead - conferma il limite dichiarato in
  offline sezioni 0/5. Nessuna ritaratura dei parametri richiesta; richiesta
  invece una gate di coerenza pre-calibrazione (l'offline sezione 5 la raccomanda
  gia) perche la sigma si calibra qui su un segnale gia degenerante.

*CLAIMS non toccati.* Detector in shadow; nessuna azione live.

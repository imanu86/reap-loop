"""Trigger-router probe (black-box, offline).

Question: can a small classifier over BLACK-BOX signals available at inference time
predict "the model would be WRONG without recovery" better than (a) the raw
mean_logprob sensor and (b) verbalized abstention alone?

Label: y = correct (1 if the no-recovery answer is right). A good trigger fires when
predicted-correct is LOW. Denominator = all task items (has_plant, non-error).
Features are ONLY things a router sees at inference: the answer's logprob signals and
whether the model abstained. NO oracle metadata (distance/category) is used.

Baselines to beat:
  * mean_logprob alone   (the pre-registered Gate-1 sensor)
  * abstention alone      (free signal: the model saying "I don't know")
Contrast sets:
  * logprob-only (no abstention)  -> can logprob signals crack the confident residual?
  * full black-box (logprob + abstention)
If full ~ abstention-alone, the router adds nothing black-box -> the residual (confident
hallucination) needs white-box hidden-state probing, not more logprob features.
"""
from __future__ import annotations
import sys
import glob
import json
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score


def load(paths):
    rows = []
    for p in paths:
        for pat in ([p] if glob.os.path.isfile(p) else glob.glob(p)):
            with open(pat, encoding="utf-8") as f:
                for l in f:
                    if l.strip():
                        rows.append(json.loads(l))
    out = []
    for r in rows:
        if not r.get("has_plant", True) or r.get("error") is not None:
            continue
        cs = r.get("conf_signals") or {}
        if "mean_logprob" not in cs:
            continue
        out.append({
            "mean_logprob": float(cs.get("mean_logprob")),
            "min_logprob": float(cs.get("min_logprob", cs.get("mean_logprob"))),
            "mean_prob": float(cs.get("mean_prob", np.exp(cs.get("mean_logprob")))),
            "abstained": 1.0 if r.get("abstained") else 0.0,
            "correct": 1 if r.get("correct") else 0,
        })
    return out


def cv_auroc(X, y, seed=0):
    if len(set(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight="balanced"))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba)


def single_auroc(score, y):
    if len(set(y)) < 2:
        return float("nan")
    return roc_auc_score(y, score)


def analyze(name, rows):
    y = np.array([r["correct"] for r in rows])
    ml = np.array([r["mean_logprob"] for r in rows])
    mn = np.array([r["min_logprob"] for r in rows])
    mp = np.array([r["mean_prob"] for r in rows])
    ab = np.array([r["abstained"] for r in rows])
    n, pos = len(y), int(y.sum())
    print(f"\n=== {name} ===  N={n}  correct={pos} ({pos/n:.1%})  abstained={int(ab.sum())} ({ab.mean():.1%})")
    print("  single-feature AUROC(feature, correct):")
    print(f"    mean_logprob (pre-reg sensor) : {single_auroc(ml, y):.3f}")
    print(f"    min_logprob                   : {single_auroc(mn, y):.3f}")
    print(f"    mean_prob                     : {single_auroc(mp, y):.3f}")
    print(f"    NOT-abstained (free trigger)  : {single_auroc(1.0 - ab, y):.3f}")
    print("  learned classifier AUROC (5-fold CV):")
    print(f"    logprob-only [ml,min,mp]      : {cv_auroc(np.c_[ml, mn, mp], y):.3f}")
    print(f"    abstention-only [ab]          : {cv_auroc(np.c_[ab], y):.3f}")
    print(f"    FULL black-box [ml,min,mp,ab] : {cv_auroc(np.c_[ml, mn, mp, ab], y):.3f}")
    # The hard residual: among NON-abstained answers (abstention already decides the
    # rest), can logprob separate confident-correct from confident-wrong? ~0.5 => no
    # black-box signal on the residual => needs white-box (hidden-state) probing.
    m = ab == 0
    ys = y[m]
    if len(set(ys)) == 2:
        print(f"  RESIDUAL (non-abstained only, N={m.sum()}, correct={int(ys.sum())}):")
        print(f"    mean_logprob AUROC            : {single_auroc(ml[m], ys):.3f}")
        print(f"    min_logprob  AUROC            : {single_auroc(mn[m], ys):.3f}")
        print(f"    logprob-only classifier (CV)  : {cv_auroc(np.c_[ml[m], mn[m], mp[m]], ys):.3f}")
    else:
        print(f"  RESIDUAL (non-abstained): degenerate (one class), N={m.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate1", nargs="+", default=[])
    ap.add_argument("--gate2", nargs="+", default=[])
    a = ap.parse_args()
    if a.gate1:
        analyze("Gate #1 regime (FULL window, hallucination-dominated errors)", load(a.gate1))
    if a.gate2:
        analyze("Gate #2 regime (SMALL window, abstention-dominated)", load(a.gate2))


if __name__ == "__main__":
    main()

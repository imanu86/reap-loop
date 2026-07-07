"""Synthetic conversational-memory dataset — FICTITIOUS UNIVERSE.

Why fictitious: the answers (entity names, codes, numeric values) are pseudo-words
and random IDs that cannot exist in any pretraining corpus, so a correct answer can
only come from the conversation context — not parametric knowledge. A matched
`no_context` control (plant removed) verifies that accuracy collapses to ~chance,
i.e. no leakage (spec §1 anti-leakage).

Structure of one item (a NIAH-style needle placed at a controlled turn-distance):
    [warmup chit-chat turns] [PLANT: target fact] ([UPDATE] for updated_fact)
    [`distance` distractor turns] -> (runner appends the final QUESTION)

Grid: distance {5,20,40,60,80} x category x difficulty, N>=200-500/cell, seeded.
The dataset holds only the HISTORY messages + question + gold; the per-arm system
prompt and retrieval are added by the arm runner (arms.py), never baked in here.
"""
from __future__ import annotations
import os
import json
import argparse
import random

# --- fictitious lexicon -----------------------------------------------------
_ONSETS = ["z", "v", "x", "qu", "thr", "kr", "vh", "zl", "gr", "dr", "ph", "sk",
           "tsz", "vr", "ky", "zorv", "xyth", "qell", "vorm", "drask"]
_VOWELS = ["a", "e", "i", "o", "u", "y", "ae", "io", "yr", "ou"]
_CODAS = ["rn", "lk", "x", "th", "sk", "ng", "vk", "ll", "zt", "rq", "ph", ""]

_ATTRS_CODE = ["clearance code", "cipher key", "vault sigil", "access token",
               "phase key", "registry id", "docking seal", "override glyph"]
_ATTRS_NUM = ["resonance index", "harmonic frequency", "drift coefficient",
              "flux rating", "orbital tag"]

_PLANT_TEMPLATES = [
    "For the record, the {attr} of {ent} is {val}.",
    "Please remember: {ent}'s {attr} is {val}.",
    "Note — {attr} for {ent}: {val}.",
    "Logging this: {ent} has {attr} {val}.",
]
_UPDATE_TEMPLATES = [
    "Correction: {ent}'s {attr} has changed to {val}.",
    "Update — the {attr} of {ent} is now {val}.",
]
_ACKS = ["Noted.", "Got it.", "Understood.", "Okay, saved.", "Recorded."]
_WARMUP = [
    ("Hi, I'll give you some facts to keep track of, then quiz you.", "Sure, go ahead."),
    ("Let's begin the session.", "Ready when you are."),
]


def _pseudoword(rng) -> str:
    n = rng.choice([2, 2, 3])
    w = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) + rng.choice(_CODAS)
                for _ in range(n))
    w = w.capitalize()
    if rng.random() < 0.4:
        w += "-" + str(rng.randint(2, 99))
    return w


def _code(rng) -> str:
    a = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    return f"{a}-{rng.randint(1000, 9999)}"


def _value(rng, answer_type) -> str:
    return _code(rng) if answer_type == "code" else str(rng.randint(1000, 9999))


def _near_collision(rng, base) -> str:
    """An entity name that shares a prefix with `base` (dense-distractor hardness)."""
    stem = base.split("-")[0]
    return stem[: max(3, len(stem) - 1)] + rng.choice(_VOWELS) + rng.choice(_CODAS)


def _fact_turn(rng, ent, attr, val, template_pool):
    u = rng.choice(template_pool).format(attr=attr, ent=ent, val=val)
    return [{"role": "user", "content": u}, {"role": "assistant", "content": rng.choice(_ACKS)}]


def make_item(seed, distance, category, difficulty, has_plant=True):
    rng = random.Random(seed)
    answer_type = "number" if rng.random() < 0.3 else "code"
    attr = rng.choice(_ATTRS_NUM if answer_type == "number" else _ATTRS_CODE)
    target_ent = _pseudoword(rng)
    target_val = _value(rng, answer_type)

    n_distractors = distance  # one distractor per gap turn
    messages = []
    # warmup
    for u, a in (_WARMUP if difficulty == "hard" else _WARMUP[:1]):
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    plant_turn_index = None
    if has_plant:
        messages.extend(_fact_turn(rng, target_ent, attr, target_val, _PLANT_TEMPLATES))
        plant_turn_index = len(messages) - 2
        if category == "updated_fact":
            target_val = _value(rng, answer_type)  # new value wins
            messages.extend(_fact_turn(rng, target_ent, attr, target_val, _UPDATE_TEMPLATES))
            plant_turn_index = len(messages) - 2

    # distractor turns filling the distance gap
    used = {(target_ent.lower(), attr)}
    for _ in range(n_distractors):
        if category == "distractor_dense":
            d_attr = attr  # same attribute -> lexical retrieval must discriminate
            d_ent = _near_collision(rng, target_ent)
        else:
            d_attr = rng.choice(_ATTRS_NUM + _ATTRS_CODE)
            d_ent = _pseudoword(rng)
        if (d_ent.lower(), d_attr) in used:
            d_ent = d_ent + str(rng.randint(100, 999))
        used.add((d_ent.lower(), d_attr))
        d_type = "number" if d_attr in _ATTRS_NUM else "code"
        messages.extend(_fact_turn(rng, d_ent, d_attr, _value(rng, d_type), _PLANT_TEMPLATES))

    question = f"What is the {attr} of {target_ent}? Answer with only the value."
    return {
        "item_id": f"s{seed}_d{distance}_{category}_{difficulty}_{'p' if has_plant else 'ctl'}",
        "seed": seed,
        "distance": distance,
        "difficulty": difficulty,
        "category": category,
        "answer_type": answer_type,
        "n_distractors": n_distractors,
        "messages": messages,
        "question": question,
        "answer": target_val if has_plant else target_val,  # gold recorded even for ctl
        "plant_turn_index": plant_turn_index,
        "has_plant": has_plant,
    }


def generate_recency(n_items, categories, difficulties, seed0=700000, scale=8.0,
                     max_dist=80):
    """Recency-skewed access profile: the target fact's turn-distance is sampled from
    an exponential (mean=scale) instead of a uniform grid, so MOST queries reference
    recent turns (in-window -> no recovery) and only a long tail reaches far back.
    This is the realistic conversational regime, and the amortized cost = mean over
    this mixture (denominator N_total). Each item's distance/category/difficulty is
    sampled from its own seed for reproducibility."""
    import math
    items = []
    for i in range(n_items):
        rng = random.Random(seed0 + i)
        d = 1 + int(rng.expovariate(1.0 / scale))
        d = max(1, min(d, max_dist))
        cat = rng.choice(categories)
        diff = rng.choice(difficulties)
        items.append(make_item(seed0 + i, d, cat, diff, has_plant=True))
    return items


def generate(n_per_cell, distances, categories, difficulties, seed0=1000,
             leak_control=False):
    items = []
    sd = seed0
    for dist in distances:
        for cat in categories:
            for diff in difficulties:
                for _ in range(n_per_cell):
                    items.append(make_item(sd, dist, cat, diff, has_plant=True))
                    if leak_control:
                        # twin with the plant removed: same seed+context, no needle
                        ctl = make_item(sd, dist, cat, diff, has_plant=False)
                        items.append(ctl)
                    sd += 1
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-cell", type=int, default=300)
    ap.add_argument("--distances", default="5,20,40,60,80")
    ap.add_argument("--categories", default="single_fact,distractor_dense")
    ap.add_argument("--difficulties", default="easy,hard")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--leak-control", action="store_true")
    ap.add_argument("--dist-profile", default="grid", choices=["grid", "recency"],
                    help="grid = uniform distance cells; recency = exponential (realistic)")
    ap.add_argument("--n-items", type=int, default=800, help="total items for --dist-profile recency")
    ap.add_argument("--recency-scale", type=float, default=8.0, help="mean turn-distance (exponential)")
    a = ap.parse_args()
    cats = a.categories.split(",")
    diffs = a.difficulties.split(",")
    if a.dist_profile == "recency":
        items = generate_recency(a.n_items, cats, diffs, a.seed0, a.recency_scale)
    else:
        dists = [int(x) for x in a.distances.split(",")]
        items = generate(a.n_per_cell, dists, cats, diffs, a.seed0, a.leak_control)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_plant = sum(1 for it in items if it["has_plant"])
    print(f"wrote {len(items)} items ({n_plant} with plant) to {a.out}  [profile={a.dist_profile}]")
    if a.dist_profile == "recency":
        ds = sorted(it["distance"] for it in items)
        within = sum(1 for d in ds if d <= 6)
        print(f"  distance: median={ds[len(ds)//2]}  min={ds[0]}  max={ds[-1]}  "
              f"within-native(<=6)={within}/{len(ds)} ({within/len(ds):.0%})")
    else:
        print(f"  cells: {len(dists)}x{len(cats)}x{len(diffs)} = "
              f"{len(dists)*len(cats)*len(diffs)} @ {a.n_per_cell}/cell")


if __name__ == "__main__":
    main()

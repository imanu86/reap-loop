#!/usr/bin/env python3
"""Bit-exact comparison of A/B arm responses + TTFT extraction."""
import json, re, sys, os

OUTDIR = os.path.dirname(os.path.abspath(__file__))

def content(label):
    p = os.path.join(OUTDIR, f"{label}.response.json")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    return d["choices"][0]["message"]["content"]

def ttft(label):
    p = os.path.join(OUTDIR, f"{label}.server.log")
    vals = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"prompt done ([0-9.]+)s", line)
            if m:
                vals.append(float(m.group(1)))
    return vals

labels = sys.argv[1:] or ["OFF", "S1", "S1S2"]
ref = None
for lab in labels:
    try:
        c = content(lab)
    except Exception as ex:
        print(f"{lab}: ERROR reading content: {ex}")
        continue
    t = ttft(lab)
    print(f"{lab}: prompt_done={t} len={len(c)}")
    if ref is None:
        ref = (lab, c)
    else:
        same = c == ref[1]
        print(f"  bit-exact vs {ref[0]}: {'YES' if same else 'NO'}")
        if not same:
            # find first divergence
            for i, (a, b) in enumerate(zip(ref[1], c)):
                if a != b:
                    print(f"  first diff at char {i}: {ref[1][max(0,i-30):i+30]!r} vs {c[max(0,i-30):i+30]!r}")
                    break
            else:
                print(f"  one is a prefix of the other (len {len(ref[1])} vs {len(c)})")

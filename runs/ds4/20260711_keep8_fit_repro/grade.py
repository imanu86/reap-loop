#!/usr/bin/env python3
"""Grade coffee HTML outputs L0-L3 for the fit sweep."""
import sys, os, re, json

RD = sys.argv[1] if len(sys.argv) > 1 else "."


def looped(t):
    for L in (16, 24, 40, 60):
        i = 0
        while i < len(t) - L * 4:
            chunk = t[i:i + L]
            if chunk.strip() and t[i:i + L * 4] == chunk * 4:
                return True
            i += L
    return False


def grade(t):
    tl = t.lower()
    m = {
        "doctype": ("<!doctype" in tl or "<html" in tl),
        "head_title": ("<title" in tl),
        "style": ("<style" in tl),
        "body": ("<body" in tl),
        "nav": ("<nav" in tl),
        "h1_bean": ("bean & brew" in tl),
        "button": ('id="order"' in tl or "order now" in tl.replace("  ", " ")),
        "form": ("<form" in tl),
        "script": ("<script" in tl),
        "alert": ("alert(" in tl),
        "close_html": ("</html>" in tl),
        "looped": looped(t),
        "chars": len(t),
    }
    # Level:
    # L0 = does not open a real page OR loops OR never reaches <body>
    # L1 = reaches <body> but does not close </html>
    # L2 = closes </html> but missing some required elements
    # L3 = closes </html> and has nav+h1+button+form+script wired
    if m["looped"] or not m["doctype"] or not m["body"]:
        lvl = 0
    elif not m["close_html"]:
        lvl = 1
    elif not (m["nav"] and m["h1_bean"] and m["button"] and m["form"] and m["script"]):
        lvl = 2
    else:
        lvl = 3
    m["l0l3"] = lvl
    return m


rows = []
for k in (8, 9, 12, 16, 23):
    p = os.path.join(RD, f"K{k}", "content.txt")
    if not os.path.exists(p):
        continue
    t = open(p, encoding="utf-8", errors="replace").read()
    g = grade(t)
    g["k"] = k
    rows.append(g)
    print(f"K{k}: L{g['l0l3']}  body={g['body']} close_html={g['close_html']} "
          f"nav={g['nav']} h1={g['h1_bean']} button={g['button']} form={g['form']} "
          f"script={g['script']} alert={g['alert']} looped={g['looped']} chars={g['chars']}")
json.dump(rows, open(os.path.join(RD, "grades.json"), "w"), indent=2)

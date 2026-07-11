import sys, os, itertools
D = "/root/smoke_0031"
def toks(label):
    p = os.path.join(D, f"{label}_tokens.csv")
    if not os.path.exists(p): return None
    out = []
    with open(p) as f:
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if i == 0 and not parts[0].isdigit():
                continue  # header
            if len(parts) >= 2 and parts[1].lstrip("-").isdigit():
                out.append((parts[0], parts[1]))
    return out
labels = [l for l in ["a","a2","b","c"] if toks(l) is not None]
T = {l: toks(l) for l in labels}
for l in labels:
    print(f"len[{l}]={len(T[l])}")
def lcp(x, y):
    n = 0
    for a, b in zip(x, y):
        if a[1] != b[1]:  # compare token_id
            return n
        n += 1
    return min(len(x), len(y))
print("== pairwise LCP (tokens shared from pos0 before first token_id divergence) ==")
for a, b in itertools.combinations(labels, 2):
    n = lcp(T[a], T[b])
    div = T[a][n][0] if n < len(T[a]) and n < len(T[b]) else "end"
    print(f"LCP[{a},{b}]={n}  first_divergent_pos={div}")

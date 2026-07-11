#!/usr/bin/env python3
"""Analyze 0033 tiered-hysteresis smoke: convergence, churn, cooldown, re-entry.
Usage: analyze_tier.py <events.jsonl> [cooldown]"""
import sys, json

def load(path):
    ev = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                ev.append(json.loads(ln))
            except Exception:
                pass
    return ev

def main():
    path = sys.argv[1]
    cooldown = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    ev = load(path)
    seeds    = [e for e in ev if e.get("event") == "tier_seed"]
    promotes = [e for e in ev if e.get("event") == "tier_promote"]
    demotes  = [e for e in ev if e.get("event") == "tier_demote"]
    swaps    = [e for e in ev if e.get("event") == "tier_swap"]
    print(f"file: {path}")
    print(f"events: seed={len(seeds)} promote={len(promotes)} demote={len(demotes)} swap={len(swaps)} total={len(ev)}")
    if not ev:
        print("NO EVENTS — tier loop did not engage")
        return
    if seeds:
        s = seeds[0]
        print(f"SEED @call={s['call']}: vram={s['vram']} budget={s['budget']} capacity={s['capacity']}")
    calls = [e["call"] for e in ev if "call" in e]
    cmin, cmax = min(calls), max(calls)
    span = max(1, cmax - cmin)

    # VRAM-count timeline: seed.vram then walk promote(vram field)/demote(-1)/swap(0)
    vram = seeds[0]["vram"] if seeds else 0
    timeline = []  # (call, vram)
    for e in ev:
        ev_t = e.get("event")
        if ev_t == "tier_seed":
            vram = e["vram"]
        elif ev_t == "tier_promote":
            vram = e.get("vram", vram + 1)
        elif ev_t == "tier_demote":
            vram = max(0, vram - 1)
        # swap: net 0
        timeline.append((e.get("call", 0), vram))
    vmax = max(v for _, v in timeline)
    vfinal = timeline[-1][1]
    print(f"VRAM set: seed={seeds[0]['vram'] if seeds else 0} max={vmax} final={vfinal}")
    # sample vram at call deciles
    print("VRAM over time (call: vram):")
    for q in range(0, 11):
        target = cmin + span * q / 10
        # last timeline point with call <= target
        v = seeds[0]["vram"] if seeds else 0
        for c, vv in timeline:
            if c <= target:
                v = vv
            else:
                break
        print(f"  {int(target):>8}: {v}")

    # Churn rate per decile (post-seed events)
    seed_call = seeds[0]["call"] if seeds else cmin
    print(f"Churn per decile of post-seed call range [{seed_call}..{cmax}] (prom/dem/swap):")
    pspan = max(1, cmax - seed_call)
    for q in range(10):
        lo = seed_call + pspan * q / 10
        hi = seed_call + pspan * (q + 1) / 10
        p = sum(1 for e in promotes if lo <= e["call"] < hi)
        d = sum(1 for e in demotes if lo <= e["call"] < hi)
        w = sum(1 for e in swaps if lo <= e["call"] < hi)
        print(f"  decile{q}: promote={p} demote={d} swap={w}")

    # Cooldown check on swaps
    if swaps:
        sc = sorted(e["call"] for e in swaps)
        gaps = [b - a for a, b in zip(sc, sc[1:])]
        mingap = min(gaps) if gaps else None
        print(f"SWAP cooldown: n={len(swaps)} min_gap={mingap} (must be >= {cooldown}) "
              f"{'OK' if (mingap is None or mingap >= cooldown) else 'VIOLATION'}")
    # bounded bound
    max_swaps = (cmax - seed_call) / cooldown
    print(f"Bounded churn bound: swaps={len(swaps)} <= calls/cooldown={(cmax-seed_call)/cooldown:.0f} "
          f"{'OK' if len(swaps) <= max_swaps + 1 else 'CHECK'}")

    # Re-entry (VRAM): (expert,layer) demoted (2->1) then later promoted/swapped back to VRAM
    demoted = {}
    reentry = 0
    for e in ev:
        key = (e.get("expert"), e.get("layer"))
        if e.get("event") == "tier_demote":
            demoted[key] = e["call"]
        elif e.get("event") == "tier_promote":
            if key in demoted and e["call"] > demoted[key]:
                reentry += 1
        elif e.get("event") == "tier_swap":
            kin = (e.get("pinned_in"), e.get("in_layer"))
            if kin in demoted and e["call"] > demoted[kin]:
                reentry += 1
    print(f"VRAM re-entry (demoted then re-promoted, same expert,layer): {reentry}")

if __name__ == "__main__":
    main()

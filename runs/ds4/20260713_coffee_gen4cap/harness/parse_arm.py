#!/usr/bin/env python3
"""Parse one arm run dir -> WARM steady-state residency metrics.

The measured request has a cold cache-warm start (a one-time resident-cache
(re)load) that drags the running average. To isolate warm steady state we take
a sub-window INSIDE the measured request: from gen>=WARM_SKIP to the end, using
gen-line-aligned deltas of the periodic masked zero-copy DIAG cumulative
counters. That gives:
  - warm MiB/token (total RAM->VRAM)  = d(query_bytes)/d(gen)
  - warm dma MiB/token (zero-copy)    = d(dma_ok_bytes)/d(gen)
  - warm VRAM-miss copies/token       = d(queries)/d(gen)
  - warm decode t/s                   = d(gen)/d(decode_seconds)   [time-based]
Usage: parse_arm.py RUNDIR [WARM_SKIP=80]
"""
import json, re, sys, os

DIAG = re.compile(
    r"masked zero-copy diag \w+: queries=(\d+)/([\d.]+) MiB "
    r"covered=(\d+)/([\d.]+) MiB dma_ok=(\d+)/([\d.]+) MiB dma_failed=(\d+)")
GEN = re.compile(r"gen=(\d+) decoding chunk=([\d.]+) t/s avg=([\d.]+) t/s ([\d.]+)s")
SPEX = re.compile(
    r"SPEX stats:.*cache_hits=(\d+) cache_misses=(\d+) hit_rate=([\d.]+) "
    r"miss_per_expert=([\d.]+).*copied=([\d.]+) MiB")
# richer SPEX: pull copy_ms / sync_ms / copy_calls / direct_loads for the copy/noncopy floor
SPEX2 = re.compile(
    r"SPEX stats:.*direct_loads=(\d+) copy_calls=(\d+) copied=([\d.]+) MiB "
    r"copy_ms=([\d.]+) copy_ms_per_batch=([\d.]+) sync_calls=(\d+) sync_ms=([\d.]+)")
CAP = re.compile(r"expert cache capped from (\d+) to (\d+) experts")
CACHEALLOC = re.compile(r"loading model tensors ([\d.]+) GiB cached")

def main(run, warm_skip=80):
    raw = open(os.path.join(run, "server.stderr.log"), "rb").read()
    try:
        offset = int(open(os.path.join(run, "stderr_offset_baseline.txt")).read().strip())
    except Exception:
        offset = 0

    pos = 0
    cur_gen = None; cur_secs = None
    snaps = []          # (gen, secs, queries, query_mib, covered, dma_ok_mib) within measured
    gen_pts = []        # (gen, secs) within measured
    chunks = []         # chunk t/s within measured, gen>=warm_skip
    spex = None; spex2 = None; cap = None; cache_gib = None
    for line in raw.split(b"\n"):
        ls = pos; pos += len(line) + 1
        s = line.decode("utf-8", "replace")
        measured = ls >= offset
        g = GEN.search(s)
        if g:
            cur_gen = int(g.group(1)); cur_secs = float(g.group(4))
            if measured:
                gen_pts.append((cur_gen, cur_secs))
                if cur_gen >= warm_skip:
                    chunks.append(float(g.group(2)))
            continue
        m = DIAG.search(s)
        if m and measured and cur_gen is not None:
            snaps.append((cur_gen, cur_secs,
                          int(m.group(1)), float(m.group(2)),
                          int(m.group(3)), float(m.group(6))))
            continue
        sp = SPEX.search(s)
        if sp:
            spex = dict(cache_hits=int(sp.group(1)), cache_misses=int(sp.group(2)),
                        hit_rate=float(sp.group(3)), miss_per_expert=float(sp.group(4)),
                        copied_mib=float(sp.group(5)))
        sp2 = SPEX2.search(s)
        if sp2:
            spex2 = dict(direct_loads=int(sp2.group(1)), copy_calls=int(sp2.group(2)),
                         copied_mib=float(sp2.group(3)), copy_ms=float(sp2.group(4)),
                         copy_ms_per_batch=float(sp2.group(5)), sync_calls=int(sp2.group(6)),
                         sync_ms=float(sp2.group(7)))
        cp = CAP.search(s)
        if cp: cap = dict(requested=int(cp.group(1)), effective=int(cp.group(2)))
        ca = CACHEALLOC.search(s)
        if ca: cache_gib = float(ca.group(1))

    tokens = None
    try:
        tokens = json.load(open(os.path.join(run, "measured_response.json")))["usage"]["completion_tokens"]
    except Exception:
        pass

    out = {"arm": os.path.basename(run.rstrip("/\\")), "measured_tokens": tokens,
           "warm_skip": warm_skip, "cache_cap": cap, "resident_cache_gib": cache_gib,
           "spex_wholerun": spex}

    # warm window: first snap with gen>=warm_skip .. last snap
    warm = [x for x in snaps if x[0] >= warm_skip]
    if len(warm) >= 2:
        lo = warm[0]; hi = warm[-1]
        dgen = hi[0] - lo[0]
        if dgen > 0:
            out["warm_window_gen"] = [lo[0], hi[0]]
            out["warm_total_mib_per_token"] = round((hi[3] - lo[3]) / dgen, 2)
            out["warm_dma_mib_per_token"] = round((hi[5] - lo[5]) / dgen, 2)
            out["warm_vram_miss_copies_per_token"] = round((hi[2] - lo[2]) / dgen, 1)
            dq = hi[2] - lo[2]
            out["warm_zerocopy_cover_frac"] = round((hi[4] - lo[4]) / dq, 4) if dq else None
            dsec = hi[1] - lo[1]
            if dsec and dsec > 0:
                out["warm_decode_tps"] = round(dgen / dsec, 3)
    if chunks:
        out["warm_chunk_tps_median"] = round(sorted(chunks)[len(chunks)//2], 3)
        out["warm_chunk_tps_list"] = chunks
    out["n_snaps_measured"] = len(snaps)

    # ---- copy/noncopy floor decomposition (whole-run SPEX cumulative) ----
    out["spex2_copy_timing"] = spex2
    if spex2 is not None:
        total_tok = (tokens or 0) + warm_skip  # measured + (~warmup skip) approx of total decode tokens
        # prefer a real total if we saw gens
        try:
            total_tok = max(total_tok, gen_pts[-1][0] if gen_pts else total_tok)
        except Exception:
            pass
        if total_tok and total_tok > 0:
            copy_pt = spex2["copy_ms"] / total_tok
            sync_pt = spex2["sync_ms"] / total_tok
            out["copy_ms_per_token"] = round(copy_pt, 3)
            out["sync_ms_per_token"] = round(sync_pt, 3)
            tps = out.get("warm_decode_tps")
            if tps and tps > 0:
                total_ms_pt = 1000.0 / tps
                out["total_ms_per_token_warm"] = round(total_ms_pt, 3)
                # floor = per-token wall not attributable to H2D copy or sync (compute+orchestration)
                out["noncopy_ms_per_token"] = round(total_ms_pt - copy_pt - sync_pt, 3)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 80)

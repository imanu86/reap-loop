#!/usr/bin/env python3
"""Analyze nsys CUDA traces to decompose per-token decode time on the 3060.

Two traces (n=40, n=120) are used two ways:
  * DIFFERENTIAL (n120 - n40): cancels model-load + prefill, giving exact per-token
    SUMS of kernel-work and memcpy-work and copied bytes (hardware-real, nsys-insensitive).
  * GEN-WINDOW UNION on n120: isolates the generation region and computes the UNION of
    kernel intervals, memcpy intervals, and their overlap -> real GPU-busy time,
    exposed-vs-hidden copy, and GPU-idle gap (= CPU/launch/sync overhead).

Honest per-token wall = model's own generation t/s (clean, non-nsys run) passed in.
"""
import sqlite3, sys, json

def load_intervals(cur, table, name_col=None):
    # returns list of (start, end, streamId, bytes_or_None, name)
    cols = "start,end,streamId"
    has_bytes = table.endswith("MEMCPY")
    if has_bytes: cols += ",bytes,copyKind"
    try:
        rows = cur.execute(f"SELECT {cols} FROM {table}").fetchall()
    except sqlite3.OperationalError:
        return []
    out=[]
    for r in rows:
        if has_bytes:
            out.append((r[0], r[1], r[2], r[3], r[4]))
        else:
            out.append((r[0], r[1], r[2], None, None))
    return out

def union_len(intervals):
    """total length of union of [start,end) intervals (ns)."""
    if not intervals: return 0
    iv = sorted((a,b) for a,b,*_ in intervals)
    total=0; cs,ce=iv[0][0],iv[0][1]
    for a,b in iv[1:]:
        if a>ce:
            total+=ce-cs; cs,ce=a,b
        else:
            ce=max(ce,b)
    total+=ce-cs
    return total

def union_bounds(intervals):
    if not intervals: return (0,0)
    return (min(a for a,b,*_ in intervals), max(b for a,b,*_ in intervals))

def overlap_len(A,B):
    """total length where union(A) intersects union(B)."""
    # build merged union lists, then intersect
    def merge(iv):
        iv=sorted((a,b) for a,b,*_ in iv); m=[]
        for a,b in iv:
            if m and a<=m[-1][1]: m[-1]=(m[-1][0],max(m[-1][1],b))
            else: m.append((a,b))
        return m
    ma,mb=merge(A),merge(B)
    i=j=0; tot=0
    while i<len(ma) and j<len(mb):
        lo=max(ma[i][0],mb[j][0]); hi=min(ma[i][1],mb[j][1])
        if hi>lo: tot+=hi-lo
        if ma[i][1]<mb[j][1]: i+=1
        else: j+=1
    return tot

def sums(db):
    con=sqlite3.connect(db); cur=con.cursor()
    K=load_intervals(cur,"CUPTI_ACTIVITY_KIND_KERNEL")
    M=load_intervals(cur,"CUPTI_ACTIVITY_KIND_MEMCPY")
    con.close()
    kdur=sum(b-a for a,b,*_ in K)
    # H2D only for expert copies (copyKind==1)
    mH2D=[m for m in M if m[4]==1]
    mdur_all=sum(b-a for a,b,*_ in M)
    mdur_h2d=sum(b-a for a,b,_,_,_ in mH2D)
    bytes_h2d=sum(x[3] for x in mH2D)
    return dict(nK=len(K),nM=len(M),nH2D=len(mH2D),
                kdur_ns=kdur,mdur_all_ns=mdur_all,mdur_h2d_ns=mdur_h2d,bytes_h2d=bytes_h2d)

def gen_window(db, ntok, frac_lo=0.45, frac_hi=0.95):
    """Isolate a steady window inside the generation region of the n120 trace and
    compute union metrics. Returns per-window totals; caller divides by est tokens."""
    con=sqlite3.connect(db); cur=con.cursor()
    K=load_intervals(cur,"CUPTI_ACTIVITY_KIND_KERNEL")
    M=load_intervals(cur,"CUPTI_ACTIVITY_KIND_MEMCPY")
    con.close()
    mH2D=[m for m in M if m[4]==1]
    # Generation region = the long tail of small H2D expert copies after the load burst.
    # Heuristic: load copies are big; expert copies are small. Split by size.
    if mH2D:
        sizes=sorted(x[3] for x in mH2D)
        med=sizes[len(sizes)//2]
        small=[m for m in mH2D if m[3]<=med*4]   # expert-sized copies
    else:
        small=[]
    # gen region time bounds from the small-copy stream
    g0,g1=union_bounds(small) if small else union_bounds(K)
    span=g1-g0
    lo=g0+span*frac_lo; hi=g0+span*frac_hi
    def inwin(iv): return [x for x in iv if x[0]>=lo and x[1]<=hi]
    Kw=inwin(K); Mw=inwin(M); Hw=[m for m in inwin(M) if m[4]==1]
    win_ns=hi-lo
    ku=union_len(Kw); mu=union_len(Mw); hu=union_len(Hw)
    both=union_len([(a,b) for a,b,*_ in Kw]+[(a,b) for a,b,*_ in Mw])
    ov_kh=overlap_len(Kw,Hw)                 # copy hidden behind kernel
    gpu_busy=both
    idle=win_ns-gpu_busy
    return dict(win_ns=win_ns, gen_span_ns=span, gen0=g0, gen1=g1,
                n_small_copies=len(small),
                kunion_ns=ku, munion_ns=mu, h2dunion_ns=hu,
                gpu_busy_ns=both, idle_ns=idle, copy_hidden_ns=ov_kh,
                copy_exposed_ns=hu-ov_kh,
                nKw=len(Kw), nHw=len(Hw))

def top_kernels(db, topn=12):
    con=sqlite3.connect(db); cur=con.cursor()
    try:
        rows=cur.execute("""
          SELECT s.value, COUNT(*), SUM(k.end-k.start)
          FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id
          GROUP BY s.value ORDER BY SUM(k.end-k.start) DESC LIMIT ?""",(topn,)).fetchall()
    except sqlite3.OperationalError:
        rows=[]
    con.close()
    return rows

if __name__=="__main__":
    db40, db120 = sys.argv[1], sys.argv[2]
    clean_gen_tps = float(sys.argv[3]) if len(sys.argv)>3 else 3.48
    n40, n120 = 40, 120
    s40=sums(db40); s120=sums(db120)
    dn=n120-n40
    def perday(x120,x40): return (x120-x40)/dn/1e6  # ns->ms per token
    ktok=perday(s120["kdur_ns"],s40["kdur_ns"])
    mtok_h2d=perday(s120["mdur_h2d_ns"],s40["mdur_h2d_ns"])
    mtok_all=perday(s120["mdur_all_ns"],s40["mdur_all_ns"])
    btok=(s120["bytes_h2d"]-s40["bytes_h2d"])/dn/1e9  # GB/token
    gw=gen_window(db120,n120)
    # tokens in window: window fraction * n120 (approx, region ~ all gen tokens)
    tok_in_win = n120*(gw["win_ns"]/gw["gen_span_ns"])
    def wtok(ns): return ns/tok_in_win/1e6
    clean_wall_ms = 1000.0/clean_gen_tps
    gpu_busy_tok = wtok(gw["gpu_busy_ns"])
    kunion_tok = wtok(gw["kunion_ns"])
    munion_tok = wtok(gw["munion_ns"])
    h2dunion_tok = wtok(gw["h2dunion_ns"])
    hidden_tok = wtok(gw["copy_hidden_ns"])
    exposed_tok = wtok(gw["copy_exposed_ns"])
    overhead_tok = clean_wall_ms - gpu_busy_tok
    out=dict(
      clean_gen_tps=clean_gen_tps, clean_wall_ms_per_tok=round(clean_wall_ms,1),
      DIFFERENTIAL=dict(
        kernel_work_ms_per_tok=round(ktok,2),
        h2d_copy_work_ms_per_tok=round(mtok_h2d,1),
        all_copy_work_ms_per_tok=round(mtok_all,1),
        h2d_GB_per_tok=round(btok,3)),
      GEN_WINDOW_UNION=dict(
        est_tokens_in_window=round(tok_in_win,1),
        gpu_busy_ms_per_tok=round(gpu_busy_tok,1),
        kernel_union_ms_per_tok=round(kunion_tok,1),
        memcpy_union_ms_per_tok=round(munion_tok,1),
        h2d_union_ms_per_tok=round(h2dunion_tok,1),
        copy_hidden_behind_kernel_ms_per_tok=round(hidden_tok,1),
        copy_exposed_ms_per_tok=round(exposed_tok,1),
        idle_ms_per_tok=round(wtok(gw["idle_ns"]),1)),
      VERDICT_INPUTS=dict(
        gpu_busy_ms_per_tok=round(gpu_busy_tok,1),
        overhead_gap_ms_per_tok=round(overhead_tok,1),
        overhead_pct=round(100*overhead_tok/clean_wall_ms,1),
        gpu_busy_pct=round(100*gpu_busy_tok/clean_wall_ms,1)),
      raw_sums=dict(n40=s40,n120=s120,gen_window=gw))
    print(json.dumps(out,indent=2))
    print("\n=== TOP KERNELS (n120, by total dur) ===")
    for name,cnt,dur in top_kernels(db120):
        print(f"  {dur/1e6:10.1f} ms  x{cnt:7d}  {name}")

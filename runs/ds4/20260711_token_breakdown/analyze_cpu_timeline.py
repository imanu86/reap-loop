#!/usr/bin/env python3
"""Decompose per-token decode time from the nsys CPU-side CUDA-API timeline.

WSL2 nsys captured CUDA RUNTIME (CPU-side) but not GPU HW activity. That is enough:
the decode critical path IS the CPU thread(s), and each API call is either
  * a SYNC-WAIT (cudaStreamSynchronize / cudaDeviceSynchronize / *Synchronize):
    CPU blocked until the GPU drains -> its duration ~= GPU busy (copy or compute)
    on the critical path;
  * a blocking cudaMemcpy: CPU blocked while a (small) transfer runs;
  * a LAUNCH/ISSUE (cudaLaunchKernel / cudaMemcpyAsync / cudaEventRecord): cheap CPU
    cost to enqueue async GPU work;
  * SETUP (malloc/free/libraryload): one-time, removed by the n120-n40 differential.

Differential (n120 - n40)/80 cancels the identical model-load+prefill prefix and
yields clean per-token numbers. union() and gap() are additive under this
differential because the shared prefix cancels, leaving 80 tokens of steady gen.

Outputs, per generated token:
  wall, union(any CUDA call), gap(no CUDA call = pure CPU/host orchestration),
  union(sync-wait = CPU blocked on GPU), and per-API-class sums, for BOTH threads
  merged (critical path) and per thread.
"""
import sqlite3, sys, json

SYNC   = {"cudaStreamSynchronize_v3020","cudaDeviceSynchronize_v3020",
          "cudaEventSynchronize_v3020","cudaStreamWaitEvent_v3020"}
BLKCPY = {"cudaMemcpy_v3020"}                      # synchronous copy (blocks CPU)
ISSUE  = {"cudaLaunchKernel_v7000","cudaMemcpyAsync_v3020","cudaEventRecord_v3020",
          "cudaMemsetAsync_v3020","cuLaunchKernel","cudaMemset_v3020"}
SETUP  = {"cudaMalloc_v3020","cudaFree_v3020","cudaMallocHost_v3020","cudaFreeHost_v3020",
          "cudaHostAlloc_v3020","cuLibraryLoadData","cuKernelGetFunction","cuLibraryGetKernel",
          "cudaStreamCreateWithFlags_v5000","cudaStreamDestroy_v5050","cudaMemGetInfo_v3020",
          "cudaEventCreateWithFlags_v3020","cudaEventDestroy_v3020"}

def union_len(iv):
    if not iv: return 0
    iv=sorted(iv); tot=0; cs,ce=iv[0]
    for a,b in iv[1:]:
        if a>ce: tot+=ce-cs; cs,ce=a,b
        else: ce=max(ce,b)
    tot+=ce-cs; return tot

def load(db):
    con=sqlite3.connect(db); cur=con.cursor()
    rows=cur.execute("""SELECT r.start,r.end,r.globalTid,s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id""").fetchall()
    con.close()
    return rows

def cls(name):
    if name in SYNC: return "sync"
    if name in BLKCPY: return "blkcpy"
    if name in ISSUE: return "issue"
    if name in SETUP: return "setup"
    return "other"

def analyze(db):
    rows=load(db)
    span=max(r[1] for r in rows)-min(r[0] for r in rows)
    by_cls={}; cnt={}
    iv_all=[]; iv_sync=[]
    tids={}
    for s,e,tid,name in rows:
        c=cls(name)
        by_cls[c]=by_cls.get(c,0)+(e-s); cnt[c]=cnt.get(c,0)+1
        iv_all.append((s,e))
        if c in ("sync","blkcpy"): iv_sync.append((s,e))
        tids.setdefault(tid,[]).append((s,e,c,name))
    # per-thread union of all + sync
    per_thread={}
    for tid,lst in tids.items():
        allv=[(a,b) for a,b,c,n in lst]
        syv=[(a,b) for a,b,c,n in lst if c in ("sync","blkcpy")]
        clsum={}
        for a,b,c,n in lst: clsum[c]=clsum.get(c,0)+(b-a)
        per_thread[tid]=dict(n=len(lst),span=max(b for a,b,c,n in lst)-min(a for a,b,c,n in lst),
                             union=union_len(allv),union_sync=union_len(syv),clsum=clsum)
    return dict(span=span, union_all=union_len(iv_all), union_sync=union_len(iv_sync),
                by_cls=by_cls, cnt=cnt, per_thread=per_thread,
                name_dur=_name_dur(rows))

def _name_dur(rows):
    d={}; c={}
    for s,e,tid,name in rows:
        d[name]=d.get(name,0)+(e-s); c[name]=c.get(name,0)+1
    return {k:(d[k],c[k]) for k in d}

if __name__=="__main__":
    db40,db120=sys.argv[1],sys.argv[2]
    clean_gen_tps=float(sys.argv[3]) if len(sys.argv)>3 else 3.48
    dn=120-40
    A=analyze(db40); B=analyze(db120)
    def per(x): return x/dn/1e6   # ns total-diff -> ms/token
    wall=per(B["span"]-A["span"])
    ua=per(B["union_all"]-A["union_all"])
    us=per(B["union_sync"]-A["union_sync"])
    gap=wall-ua
    clean_wall=1000.0/clean_gen_tps
    # scale nsys per-token to clean wall (nsys ~3% slower); report both
    classes=["sync","blkcpy","issue","setup","other"]
    cls_tok={c:round(per(B["by_cls"].get(c,0)-A["by_cls"].get(c,0)),1) for c in classes}
    cls_cnt={c:round((B["cnt"].get(c,0)-A["cnt"].get(c,0))/dn,1) for c in classes}
    # per-token per-API name (differential)
    allnames=set(B["name_dur"])|set(A["name_dur"])
    name_tok={}
    for nm in allnames:
        bd,bc=B["name_dur"].get(nm,(0,0)); ad,ac=A["name_dur"].get(nm,(0,0))
        dtok=per(bd-ad); ctok=(bc-ac)/dn
        if abs(dtok)>0.2 or ctok>0.5:
            name_tok[nm]=dict(ms_tok=round(dtok,2),calls_tok=round(ctok,1))
    out=dict(
      clean_gen_tps=clean_gen_tps, clean_wall_ms_per_tok=round(clean_wall,1),
      nsys_wall_ms_per_tok=round(wall,1),
      PER_TOKEN_nsys=dict(
        wall=round(wall,1),
        union_any_cuda_call=round(ua,1),
        gap_no_cuda_call_pureCPU=round(gap,1),
        union_sync_wait_blocked_on_gpu=round(us,1),
        gap_pct=round(100*gap/wall,1),
        union_sync_pct_of_wall=round(100*us/wall,1)),
      per_api_class_ms_per_tok=cls_tok,
      per_api_class_calls_per_tok=cls_cnt,
      per_api_name_ms_per_tok=dict(sorted(name_tok.items(),key=lambda kv:-kv[1]["ms_tok"])),
      per_thread_n120={str(t):{k:(round(v/1e6,1) if k in('span','union','union_sync') else
                       ({kk:round(vv/1e6,1) for kk,vv in v.items()} if k=='clsum' else v))
                       for k,v in d.items()} for t,d in B["per_thread"].items()})
    print(json.dumps(out,indent=2))

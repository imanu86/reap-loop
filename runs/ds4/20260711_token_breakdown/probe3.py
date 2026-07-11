import sqlite3,sys
con=sqlite3.connect(sys.argv[1]);cur=con.cursor()
def cols(t): return [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
for t in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_CUDA_EVENT','CUPTI_ACTIVITY_KIND_SYNCHRONIZATION']:
    print(f"\n== {t} cols:", cols(t))
# top runtime API names by count + total dur
print("\n== TOP RUNTIME API (by total CPU dur) ==")
q="""SELECT s.value, COUNT(*), SUM(r.end-r.start), AVG(r.end-r.start)
     FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
     GROUP BY s.value ORDER BY SUM(r.end-r.start) DESC LIMIT 25"""
try:
    for name,c,tot,avg in cur.execute(q).fetchall():
        print(f"  tot={tot/1e6:10.1f}ms  n={c:8d}  avg={avg/1e3:8.2f}us  {name}")
except Exception as e:
    print("runtime name join err:",e)
    # fallback: maybe column is 'name' not nameId
    print("cols again:",cols('CUPTI_ACTIVITY_KIND_RUNTIME'))
# distinct threads
try:
    thr=cur.execute("SELECT globalTid,COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME GROUP BY globalTid ORDER BY COUNT(*) DESC").fetchall()
    print("\nthreads (globalTid,count):",thr[:8])
except Exception as e: print("thr err",e)
con.close()

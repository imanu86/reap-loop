import sqlite3,sys
con=sqlite3.connect(sys.argv[1]);cur=con.cursor()
q="""SELECT r.globalTid, s.value, COUNT(*), SUM(r.end-r.start)
     FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
     GROUP BY r.globalTid,s.value HAVING SUM(r.end-r.start)>5e7
     ORDER BY r.globalTid, SUM(r.end-r.start) DESC"""
cur_t=None
for tid,name,c,tot in cur.execute(q).fetchall():
    if tid!=cur_t:
        print(f"\n== thread {tid} =="); cur_t=tid
    print(f"   {tot/1e6:9.1f}ms  n={c:8d}  {name}")
con.close()

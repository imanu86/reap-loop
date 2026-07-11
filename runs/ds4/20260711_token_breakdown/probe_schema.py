import sqlite3,sys
con=sqlite3.connect(sys.argv[1]);cur=con.cursor()
t=[r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
print("KERNEL table:", [x for x in t if 'KERNEL' in x])
print("MEMCPY table:", [x for x in t if 'MEMCPY' in x])
for tbl in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY']:
    try:
        cols=[c[1] for c in cur.execute(f"PRAGMA table_info({tbl})").fetchall()]
        n=cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"\n{tbl}  rows={n}\n  cols={cols}")
    except Exception as e:
        print(tbl,"ERR",e)
# memcpy copyKind distribution
try:
    print("\ncopyKind dist:", cur.execute("SELECT copyKind,COUNT(*),SUM(bytes) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind").fetchall())
except Exception as e: print("copyKind err",e)
con.close()

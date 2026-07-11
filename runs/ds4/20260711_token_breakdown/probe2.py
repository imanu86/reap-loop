import sqlite3,sys
con=sqlite3.connect(sys.argv[1]);cur=con.cursor()
t=sorted(r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
for x in t:
    n=cur.execute(f"SELECT COUNT(*) FROM {x}").fetchone()[0]
    if n>0:
        print(f"{n:>10}  {x}")
con.close()

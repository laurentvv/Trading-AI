import sqlite3

def check_db(path, name):
    print(f"\n{'='*60}")
    print(f"DATABASE: {name}")
    print(f"{'='*60}")
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables: {tables}")
    for t in tables:
        print(f"\n--- {t} ---")
        cur.execute(f"SELECT * FROM [{t}] ORDER BY rowid DESC LIMIT 5")
        cols = [d[0] for d in cur.description]
        print(f"Columns: {cols}")
        rows = cur.fetchall()
        print(f"Row count (last): {len(rows)}")
        for r in rows:
            print(r)
        total = cur.execute(f"SELECT COUNT(*) FROM [{t}]").fetchone()[0]
        print(f"Total rows: {total}")
    conn.close()

check_db(r"D:\GIT\fork\Trading-AI\logs_prod\trading_history.db", "trading_history.db")
check_db(r"D:\GIT\fork\Trading-AI\logs_prod\performance_monitor.db", "performance_monitor.db")

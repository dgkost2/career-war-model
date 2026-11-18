"""Step 1.8: load every data/processed/*.parquet into data/processed/warehouse.duckdb.

Usage:
  python src/build_warehouse.py

One table per Parquet file, named after the file (hitting_season.parquet ->
hitting_season). The database is rebuilt from scratch on every run (built in a
temp file, then swapped in), so the Parquet files stay the source of truth and
tables for deleted files don't linger. Derived tables are rebuilt first:
  dim_player.parquet  (build_dim_player.py)

Then runs integrity checks and exits non-zero if any fail:
  - every hitting_season playerId exists in people
  - every hitting_season (teamId, season) exists in teams_season
  - no duplicate stints in hitting_season (playerId, season, teamId, sportId)
  - no duplicate stints in hitting_stints (same key + is_rehab)
  - dim_player has one row per playerId and covers every hitting_stints player
"""
import sys

import duckdb

import build_dim_player
from mlb_api import PROCESSED_DIR

DB = PROCESSED_DIR / "warehouse.duckdb"
STINT = "playerId, season, teamId, sportId"

CHECKS = [
    ("hitting_season playerId missing from people",
     "SELECT count(*) FROM (SELECT DISTINCT playerId FROM hitting_season) h "
     "ANTI JOIN people p ON h.playerId = p.id"),
    ("hitting_season (teamId, season) missing from teams_season",
     "SELECT count(*) FROM (SELECT DISTINCT teamId, season FROM hitting_season) h "
     "ANTI JOIN teams_season t ON h.teamId = t.teamId AND h.season = t.season"),
    ("duplicate stints in hitting_season",
     f"SELECT count(*) FROM (SELECT {STINT} FROM hitting_season GROUP BY ALL HAVING count(*) > 1)"),
    ("duplicate stints in hitting_stints (incl. is_rehab)",
     f"SELECT count(*) FROM (SELECT {STINT}, is_rehab FROM hitting_stints GROUP BY ALL HAVING count(*) > 1)"),
    ("duplicate playerIds in dim_player",
     "SELECT count(*) - count(DISTINCT playerId) FROM dim_player"),
    ("hitting_stints playerId missing from dim_player",
     "SELECT count(*) FROM (SELECT DISTINCT playerId FROM hitting_stints) h "
     "ANTI JOIN dim_player d ON h.playerId = d.playerId"),
]


def build() -> None:
    files = sorted(PROCESSED_DIR.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no parquet files in {PROCESSED_DIR}")
    tmp = DB.with_suffix(".duckdb.tmp")
    tmp.unlink(missing_ok=True)
    with duckdb.connect(str(tmp)) as con:
        for f in files:
            con.execute(f'CREATE TABLE "{f.stem}" AS SELECT * FROM read_parquet(?)', [str(f)])
            n = con.execute(f'SELECT count(*) FROM "{f.stem}"').fetchone()[0]
            print(f"  {f.stem:<20} {n:>9,} rows")
    tmp.replace(DB)
    print(f"wrote {DB} ({len(files)} tables)")


def check() -> bool:
    ok = True
    with duckdb.connect(str(DB), read_only=True) as con:
        for name, sql in CHECKS:
            n = con.execute(sql).fetchone()[0]
            ok &= n == 0
            print(f"  {'OK  ' if n == 0 else 'FAIL'} {name}: {n}")
    return ok


def main() -> None:
    build_dim_player.main()
    build()
    print("checks:")
    if not check():
        sys.exit("warehouse checks failed")


if __name__ == "__main__":
    main()

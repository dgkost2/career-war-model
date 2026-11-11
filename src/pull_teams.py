"""Step 1.2b: pull teams per level and season, build the teams_season table.

Usage:
  python src/pull_teams.py 2025
  python src/pull_teams.py 2005-2025

Caches each response to data/raw/teams/{season}_{sportId}.json (skips files
that already exist), then rebuilds data/processed/teams_season.parquet from
every cached teams file, so the table always covers all seasons pulled so far.
"""
import argparse
import json

import pandas as pd

from mlb_api import LEVEL_IDS, PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons

TEAMS_RAW = RAW_DIR / "teams"


def pull(seasons: list[int]) -> None:
    for season in seasons:
        for sport_id in LEVEL_IDS:
            data, fetched = fetch_cached(
                "/teams",
                TEAMS_RAW / f"{season}_{sport_id}.json",
                params={"sportId": sport_id, "season": season},
            )
            status = "downloaded" if fetched else "cached"
            print(f"{season} sportId={sport_id:>2}: {len(data.get('teams', [])):>3} teams ({status})")


def build_table() -> pd.DataFrame:
    rows = []
    for f in sorted(TEAMS_RAW.glob("*_*.json")):
        season, sport_id = (int(x) for x in f.stem.split("_"))
        for t in json.loads(f.read_text(encoding="utf-8"))["teams"]:
            rows.append({
                "season": season,
                "teamId": t["id"],
                "name": t.get("name"),
                # sportId from the request, not t["sport"], so the row matches the file it came from
                "sportId": sport_id,
                "leagueId": t.get("league", {}).get("id"),
                "parentOrgId": t.get("parentOrgId"),
            })
    df = pd.DataFrame(rows).astype({"leagueId": "Int64", "parentOrgId": "Int64"})

    dupes = df.duplicated(["season", "teamId"], keep=False)
    if dupes.any():
        raise RuntimeError(f"team appears at more than one level in a season:\n{df[dupes]}")
    return df.sort_values(["season", "sportId", "teamId"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    pull(parse_seasons(args.seasons))

    df = build_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "teams_season.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, seasons {df.season.min()}-{df.season.max()})")


if __name__ == "__main__":
    main()

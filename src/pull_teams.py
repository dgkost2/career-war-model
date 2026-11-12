"""Step 1.2b: pull teams per level and season, build the teams_season table.

Usage:
  python src/pull_teams.py 2025
  python src/pull_teams.py 2005-2025

Caches each response to data/raw/teams/{season}_{sportId}.json (skips files
that already exist), then rebuilds data/processed/teams_season.parquet from
every cached teams file, so the table always covers all seasons pulled so far.

Columns: season, teamId, name, sportId, leagueId, parentOrgId, leagueId_inferred,
level, is_coop.
  leagueId_inferred  API gave no league; leagueId filled from name prefix (VSL/DSL/GCL/AZL)
  level              sportId abbreviation, overridden by leagueId for MEX / DSL / VSL
  is_coop            shared DSL/VSL team with two parent orgs (parentOrgId left null)
"""
import argparse
import json

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, is_skipped, level_code, levels_for, parse_seasons

TEAMS_RAW = RAW_DIR / "teams"

# Rookie team-name prefix -> leagueId, used only when the API gives no league.
PREFIX_LEAGUE = {"VSL": 134, "DSL": 130, "GCL": 124, "AZL": 121}


def pull(seasons: list[int]) -> None:
    for season in seasons:
        for sport_id in levels_for(season):
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
        if is_skipped(season, sport_id):
            continue
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

    # Some Rookie teams come back with no league (VSL 2005-2013, one GCL, one DSL).
    # They have empty rosters and no stats, but fill leagueId from the name prefix
    # so their level is right, and flag it.
    prefix = df.name.str.extract(r"^(\w+) ", expand=False).map(PREFIX_LEAGUE)
    df["leagueId_inferred"] = df.leagueId.isna() & prefix.notna()
    df["leagueId"] = df.leagueId.fillna(prefix).astype("Int64")
    if df.leagueId.isna().any():
        print(f"warning: teams with no leagueId:\n{df[df.leagueId.isna()].to_string(index=False)}")

    df["level"] = level_code(df)
    # Shared DSL/VSL teams (e.g. "DSL Indians/Brewers") belong to two orgs, so the
    # API gives no parentOrgId. Leave it null and flag them.
    df["is_coop"] = (df.sportId != 1) & df.parentOrgId.isna() & df.name.str.contains("/")
    orphans = df[(df.sportId != 1) & df.parentOrgId.isna() & ~df.is_coop]
    if len(orphans):
        print(f"warning: {len(orphans)} non-co-op MiLB teams with no parentOrgId:\n{orphans.to_string(index=False)}")

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

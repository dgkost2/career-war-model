"""Step 1.4: pull season hitting stats and build hitting_season (one row per stint).

Usage:
  python src/pull_hitting.py 2025
  python src/pull_hitting.py 2005-2025

Requires teams_season.parquet (run pull_teams.py first for the same seasons).

Why not just the bulk endpoint: /stats?stats=season&sportId=X returns ONE
combined row per player per level for multi-team players (numTeams > 1),
tagged with only his last team. Adding teamId to /stats doesn't fix it: for
MLB it drops players who left the team mid-season (e.g. Santana, 455 PA for
CLE, missing from CLE's query). So stints come from each team's fullSeason
roster with season stats hydrated, which lists every player who was on the
team and gives his per-team splits plus a combined row (no team) that we drop.

Two pulls, both cached and skipped if present:
  - rosters -> data/raw/hitting_roster/{season}_{sportId}_{teamId}.json
    One call per team. Builds the table.
  - bulk    -> data/raw/hitting/{season}_{sportId}_{offset}.json
    limit=500, paged by offset until a page comes back short. Used only as a
    check: stint PA must sum to the bulk PA for every player × level.

A few players aren't on any fullSeason roster (e.g. Royce Lewis, 2025 MIN).
If such a player-level has a single-team bulk row, that row is used as the
stint (source="bulk"); a multi-team one raises an error.

Output: data/processed/hitting_season.parquet, rebuilt from every cached roster
file. Columns: playerId, season, teamId, sportId, leagueId, G, PA, AB, H, 2B,
3B, HR, BB, IBB, HBP, SO, SB, CS, SF, source ("roster" or "bulk").
"""
import argparse
import json

import pandas as pd

from mlb_api import LEVEL_IDS, PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons

BULK_RAW = RAW_DIR / "hitting"
ROSTER_RAW = RAW_DIR / "hitting_roster"
PAGE_SIZE = 500

# output column -> API stat field
STAT_FIELDS = {
    "G": "gamesPlayed", "PA": "plateAppearances", "AB": "atBats", "H": "hits",
    "2B": "doubles", "3B": "triples", "HR": "homeRuns", "BB": "baseOnBalls",
    "IBB": "intentionalWalks", "HBP": "hitByPitch", "SO": "strikeOuts",
    "SB": "stolenBases", "CS": "caughtStealing", "SF": "sacFlies",
}
KEY = ["playerId", "season", "teamId", "sportId"]


def pull_bulk(season: int, sport_id: int) -> tuple[int, int]:
    """Page the bulk query by offset. Returns (rows, api_calls)."""
    offset = rows = calls = 0
    while True:
        data, fetched = fetch_cached(
            "/stats",
            BULK_RAW / f"{season}_{sport_id}_{offset}.json",
            params={"stats": "season", "group": "hitting", "playerPool": "ALL", "sportId": sport_id,
                    "season": season, "limit": PAGE_SIZE, "offset": offset},
        )
        calls += fetched
        splits = data["stats"][0]["splits"] if data.get("stats") else []
        rows += len(splits)
        if len(splits) < PAGE_SIZE:
            return rows, calls
        offset += PAGE_SIZE


def pull_roster(season: int, sport_id: int, team_id: int) -> bool:
    _, fetched = fetch_cached(
        f"/teams/{team_id}/roster",
        ROSTER_RAW / f"{season}_{sport_id}_{team_id}.json",
        params={"rosterType": "fullSeason", "season": season,
                "hydrate": f"person(stats(type=season,season={season},group=hitting,sportId={sport_id}))"},
    )
    return fetched


def pull(seasons: list[int]) -> None:
    teams = pd.read_parquet(PROCESSED_DIR / "teams_season.parquet")
    for season in seasons:
        season_teams = teams[teams.season == season]
        if season_teams.empty:
            raise RuntimeError(f"no teams for {season} in teams_season.parquet; run pull_teams.py {season}")
        for sport_id in LEVEL_IDS:
            b_rows, b_calls = pull_bulk(season, sport_id)
            level_teams = season_teams[season_teams.sportId == sport_id].teamId
            r_calls = sum(pull_roster(season, sport_id, t) for t in level_teams)
            print(f"{season} sportId={sport_id:>2}: bulk {b_rows:>5} rows ({b_calls} calls) | "
                  f"rosters for {len(level_teams)} teams ({r_calls} calls)")


def stint_row(sp: dict, season: int, sport_id: int, source: str) -> dict:
    return {
        "playerId": sp["player"]["id"], "season": season, "teamId": sp["team"]["id"],
        "sportId": sport_id, "leagueId": sp["league"]["id"],
        **{col: sp["stat"].get(field) for col, field in STAT_FIELDS.items()},
        "source": source,
    }


def read_bulk(seasons: set[int]) -> list[tuple[int, int, dict]]:
    out = []
    for f in sorted(BULK_RAW.glob("*.json")):
        season, sport_id, _ = (int(x) for x in f.stem.split("_"))
        if season in seasons:
            out += [(season, sport_id, sp) for sp in json.loads(f.read_text(encoding="utf-8"))["stats"][0]["splits"]]
    return out


def add_bulk_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fill player-levels the rosters missed (a few players aren't on any fullSeason
    roster). A single-team bulk row is already a correct stint; a multi-team one isn't."""
    covered = set(df[["season", "sportId", "playerId"]].itertuples(index=False, name=None))
    extra, unfixable = [], []
    for season, sport_id, sp in read_bulk(set(df.season)):
        if (season, sport_id, sp["player"]["id"]) in covered:
            continue
        if sp.get("numTeams", 1) == 1:
            extra.append(stint_row(sp, season, sport_id, "bulk"))
        else:
            unfixable.append((season, sport_id, sp["player"]["id"], sp["player"]["fullName"]))
    if unfixable:
        raise RuntimeError(f"multi-team players missing from all rosters: {unfixable}")
    print(f"bulk fallback: added {len(extra)} single-team stints missing from rosters")
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def build_table() -> pd.DataFrame:
    rows = []
    for f in sorted(ROSTER_RAW.glob("*.json")):
        season, sport_id, _ = (int(x) for x in f.stem.split("_"))
        for entry in json.loads(f.read_text(encoding="utf-8")).get("roster", []):
            for st in entry["person"].get("stats", []):
                for sp in st["splits"]:
                    if "team" not in sp or sp.get("numTeams", 1) > 1:
                        continue  # combined/total row: never a stint
                    if sp.get("gameType", "R") != "R" or int(sp["season"]) != season:
                        continue
                    if sp["sport"]["id"] != sport_id:
                        raise RuntimeError(f"{f.name}: split at sportId {sp['sport']['id']}")
                    rows.append(stint_row(sp, season, sport_id, "roster"))
    df = pd.DataFrame(rows)
    # A traded player appears on each of his teams' rosters with all his splits,
    # so the same stint shows up more than once. Copies must be identical.
    df = add_bulk_fallback(df.drop_duplicates())
    if df.duplicated(KEY).any():
        raise RuntimeError(f"conflicting copies of a stint:\n{df[df.duplicated(KEY, keep=False)]}")
    return df.sort_values(KEY).reset_index(drop=True)


def check_against_bulk(df: pd.DataFrame) -> None:
    """Stint PA must sum to the bulk PA for every player × season × level."""
    bulk = pd.DataFrame(
        {"season": season, "sportId": sport_id, "playerId": sp["player"]["id"],
         "PA_bulk": sp["stat"]["plateAppearances"]}
        for season, sport_id, sp in read_bulk(set(df.season))
    )
    stints = df.groupby(["season", "sportId", "playerId"], as_index=False).PA.sum()
    m = bulk.merge(stints, on=["season", "sportId", "playerId"], how="outer", indicator=True)
    bad = m[(m._merge != "both") | (m.PA_bulk != m.PA)]
    print(f"bulk check: {len(m)} player-levels, {len(bad)} mismatches")
    if len(bad):
        raise RuntimeError(f"stints don't match bulk totals:\n{bad.to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    pull(parse_seasons(args.seasons))

    df = build_table()
    check_against_bulk(df)
    out = PROCESSED_DIR / "hitting_season.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

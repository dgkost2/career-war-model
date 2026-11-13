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
stint (source="bulk"); a multi-team one can't be split and is left out.

Reconciliation: every player-level whose stint PA doesn't equal the bulk PA
(including ones with no stints at all) is written to
data/processed/reconcile_log.csv, rewritten on every run.
  - single-season run: any mismatch in that season stops the script.
  - multi-season run: mismatches are logged and the run continues.

Output: data/processed/hitting_season.parquet, rebuilt from every cached roster
file. Columns: playerId, season, teamId, sportId, leagueId, G, PA, AB, H, 2B,
3B, HR, BB, IBB, HBP, SO, SB, CS, SF, source ("roster" or "bulk"), level
(sportId abbreviation, overridden by leagueId for MEX / DSL / VSL).
"""
import argparse

import pandas as pd

from mlb_api import level_code, PROCESSED_DIR, parse_seasons
from rosters import bulk_fallback, iter_stint_splits, pull_bulk, pull_rosters, read_bulk, write_reconcile_log

# output column -> API stat field
STAT_FIELDS = {
    "G": "gamesPlayed", "PA": "plateAppearances", "AB": "atBats", "H": "hits",
    "2B": "doubles", "3B": "triples", "HR": "homeRuns", "BB": "baseOnBalls",
    "IBB": "intentionalWalks", "HBP": "hitByPitch", "SO": "strikeOuts",
    "SB": "stolenBases", "CS": "caughtStealing", "SF": "sacFlies",
}
KEY = ["playerId", "season", "teamId", "sportId"]


def pull(seasons: list[int]) -> None:
    for season in seasons:
        for sport_id, n_teams, r_calls in pull_rosters(season, "hitting"):
            b_rows, b_calls = pull_bulk(season, sport_id, "hitting")
            print(f"{season} sportId={sport_id:>2}: bulk {b_rows:>5} rows ({b_calls} calls) | "
                  f"rosters for {n_teams} teams ({r_calls} calls)")


def stint_row(sp: dict, season: int, sport_id: int, source: str) -> dict:
    return {
        "playerId": sp["player"]["id"], "season": season, "teamId": sp["team"]["id"],
        "sportId": sport_id, "leagueId": sp["league"]["id"],
        **{col: sp["stat"].get(field) for col, field in STAT_FIELDS.items()},
        "source": source,
    }


def add_bulk_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fill player-levels the rosters missed (a few players aren't on any fullSeason roster)."""
    covered = set(df[["season", "sportId", "playerId"]].itertuples(index=False, name=None))
    extra = [stint_row(sp, season, sport_id, "bulk")
             for season, sport_id, sp in bulk_fallback(covered, set(df.season), "hitting")]
    return pd.concat([df, pd.DataFrame(extra)], ignore_index=True)


def build_table() -> pd.DataFrame:
    df = pd.DataFrame(stint_row(sp, season, sport_id, "roster")
                      for season, sport_id, sp in iter_stint_splits("hitting"))
    # A traded player appears on each of his teams' rosters with all his splits,
    # so the same stint shows up more than once. Copies must be identical.
    df = add_bulk_fallback(df.drop_duplicates())
    if df.duplicated(KEY).any():
        raise RuntimeError(f"conflicting copies of a stint:\n{df[df.duplicated(KEY, keep=False)]}")
    df["level"] = level_code(df)
    return df.sort_values(KEY).reset_index(drop=True)


def check_against_bulk(df: pd.DataFrame, strict_season: int | None) -> None:
    """Stint PA must sum to the bulk PA for every player × season × level.

    Mismatches always go to reconcile_log.csv. If strict_season is set
    (single-season run), any mismatch in that season raises.
    """
    bulk = pd.DataFrame(
        {"season": season, "sportId": sport_id, "playerId": sp["player"]["id"],
         "PA_bulk": sp["stat"]["plateAppearances"]}
        for season, sport_id, sp in read_bulk(set(df.season), "hitting")
    )
    stints = df.groupby(["season", "sportId", "playerId"], as_index=False).PA.sum()
    m = bulk.merge(stints, on=["season", "sportId", "playerId"], how="outer", indicator=True)
    bad = m[(m._merge != "both") | (m.PA_bulk != m.PA)]
    log = (bad.rename(columns={"PA": "stint_PA", "PA_bulk": "bulk_PA"})
              [["season", "sportId", "playerId", "stint_PA", "bulk_PA"]]
              .astype({"stint_PA": "Int64", "bulk_PA": "Int64"})
              .sort_values(["season", "sportId", "playerId"]))
    print(f"hitting check covers {len(m)} player-levels")
    write_reconcile_log(log, "reconcile_log.csv", strict_season, "hitting PA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    pull(seasons)

    df = build_table()
    check_against_bulk(df, strict_season=seasons[0] if len(seasons) == 1 else None)
    out = PROCESSED_DIR / "hitting_season.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

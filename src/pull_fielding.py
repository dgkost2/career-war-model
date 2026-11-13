"""Step 1.5: pull season fielding stats and build fielding_season (one row per stint × position).

Usage:
  python src/pull_fielding.py 2025
  python src/pull_fielding.py 2005-2025

Requires teams_season.parquet. Uses the same roster approach as pull_hitting.py
(see rosters.py): one call per team, fullSeason roster with season fielding
stats hydrated, cached to data/raw/fielding_roster/{season}_{sportId}_{teamId}.json.
Each split is one player × team × position; combined rows (no team) are dropped.

DH comes back as a position (code 10) with 0 innings and no chances. It's kept
so DH games are visible; the defensive model ignores it.

Output: data/processed/fielding_season.parquet, rebuilt from every cached roster
file. Columns: playerId, season, teamId, sportId, level, leagueId, position, G,
GS, innings, PO, A, E.
  innings  true innings as a decimal: the API's "58.1" (58 and 1/3) -> 58.333
"""
import argparse

import pandas as pd

from mlb_api import PROCESSED_DIR, level_code, parse_seasons
from rosters import iter_stint_splits, pull_rosters

KEY = ["playerId", "season", "teamId", "sportId", "position"]


def innings(value: str | None) -> float | None:
    """'58.1' -> 58.333: the digit after the point counts outs, not tenths."""
    if value in (None, "", "-.--"):
        return None
    whole, _, outs = str(value).partition(".")
    if outs not in ("", "0", "1", "2"):
        raise ValueError(f"unexpected innings value {value!r}")
    return int(whole) + int(outs or 0) / 3


def build_table() -> pd.DataFrame:
    rows = []
    for season, sport_id, sp in iter_stint_splits("fielding"):
        st = sp["stat"]
        rows.append({
            "playerId": sp["player"]["id"], "season": season, "teamId": sp["team"]["id"],
            "sportId": sport_id, "leagueId": sp["league"]["id"],
            "position": sp["position"]["abbreviation"],
            "G": st.get("gamesPlayed"), "GS": st.get("gamesStarted"),
            "innings": innings(st.get("innings")),
            "PO": st.get("putOuts"), "A": st.get("assists"), "E": st.get("errors"),
        })
    # A traded player appears on each of his teams' rosters, so copies must be identical.
    df = pd.DataFrame(rows).drop_duplicates()
    if df.duplicated(KEY).any():
        raise RuntimeError(f"conflicting copies of a fielding stint:\n{df[df.duplicated(KEY, keep=False)]}")
    df = df.astype({"GS": "Int64", "PO": "Int64", "A": "Int64", "E": "Int64"})
    df["level"] = level_code(df)
    cols = ["playerId", "season", "teamId", "sportId", "level", "leagueId", "position",
            "G", "GS", "innings", "PO", "A", "E"]
    return df[cols].sort_values(KEY).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    for season in parse_seasons(args.seasons):
        for sport_id, n_teams, calls in pull_rosters(season, "fielding"):
            print(f"{season} sportId={sport_id:>2}: rosters for {n_teams} teams ({calls} calls)")

    df = build_table()
    out = PROCESSED_DIR / "fielding_season.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

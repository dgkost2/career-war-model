"""Step 1.5: pull season fielding stats and build fielding_season (one row per stint × position).

Usage:
  python src/pull_fielding.py 2025
  python src/pull_fielding.py 2005-2025

Requires teams_season.parquet. Same approach as pull_hitting.py (see rosters.py):
  - rosters -> data/raw/fielding_roster/{season}_{sportId}_{teamId}.json
    One call per team, fullSeason roster with season fielding stats hydrated.
    Each split is one player × team × position; combined rows (no team) are dropped.
  - bulk    -> data/raw/fielding/{season}_{sportId}_{offset}.json
    One row per player × position × level. Used as a fallback for single-team
    players missing from every roster (source="bulk") and to reconcile: stint
    innings must sum to bulk innings for every player × season × level × position.

Reconciliation mismatches go to data/processed/reconcile_fielding_log.csv,
rewritten on every run. A single-season run stops on any mismatch in that
season; a multi-season run logs and continues.

DH comes back as a position (code 10) with 0 innings and no chances. It's kept
so DH games are visible; the defensive model ignores it. DH rows reconcile on G
instead of innings.

Output: data/processed/fielding_season.parquet, rebuilt from every cached roster
file. Columns: playerId, season, teamId, sportId, level, leagueId, position, G,
GS, innings, PO, A, E, source ("roster" or "bulk").
  innings  true innings as a decimal: the API's "58.1" (58 and 1/3) -> 58.333
"""
import argparse

import pandas as pd

from mlb_api import PROCESSED_DIR, level_code, parse_seasons
from rosters import bulk_fallback, iter_stint_splits, pull_bulk, pull_rosters, read_bulk, write_reconcile_log

KEY = ["playerId", "season", "teamId", "sportId", "position"]


def innings(value: str | None) -> float | None:
    """'58.1' -> 58.333: the digit after the point counts outs, not tenths."""
    if value in (None, "", "-.--"):
        return None
    whole, _, outs = str(value).partition(".")
    if outs not in ("", "0", "1", "2"):
        raise ValueError(f"unexpected innings value {value!r}")
    return int(whole) + int(outs or 0) / 3


def fielding_row(sp: dict, season: int, sport_id: int, source: str) -> dict:
    st = sp["stat"]
    return {
        "playerId": sp["player"]["id"], "season": season, "teamId": sp["team"]["id"],
        "sportId": sport_id, "leagueId": sp["league"]["id"],
        "position": sp["position"]["abbreviation"],
        "G": st.get("gamesPlayed"), "GS": st.get("gamesStarted"),
        "innings": innings(st.get("innings")),
        "PO": st.get("putOuts"), "A": st.get("assists"), "E": st.get("errors"),
        "source": source,
    }


def build_table() -> pd.DataFrame:
    df = pd.DataFrame(fielding_row(sp, season, sport_id, "roster")
                      for season, sport_id, sp in iter_stint_splits("fielding"))
    # A traded player appears on each of his teams' rosters, so copies must be identical.
    df = df.drop_duplicates()

    covered = set(df[["season", "sportId", "playerId"]].itertuples(index=False, name=None))
    extra = [fielding_row(sp, season, sport_id, "bulk")
             for season, sport_id, sp in bulk_fallback(covered, set(df.season), "fielding")]
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)

    if df.duplicated(KEY).any():
        raise RuntimeError(f"conflicting copies of a fielding stint:\n{df[df.duplicated(KEY, keep=False)]}")
    df = df.astype({"GS": "Int64", "PO": "Int64", "A": "Int64", "E": "Int64"})
    df["level"] = level_code(df)
    cols = ["playerId", "season", "teamId", "sportId", "level", "leagueId", "position",
            "G", "GS", "innings", "PO", "A", "E", "source"]
    return df[cols].sort_values(KEY).reset_index(drop=True)


def check_against_bulk(df: pd.DataFrame, strict_season: int | None) -> None:
    """Stint innings (G for DH) must sum to bulk for every player × season × level × position.
    Compared in outs (innings × 3) so thirds of an inning don't cause float noise."""
    key = ["season", "sportId", "playerId", "position"]
    bulk = pd.DataFrame(
        {"season": season, "sportId": sport_id, "playerId": sp["player"]["id"],
         "position": sp["position"]["abbreviation"],
         "bulk_innings": innings(sp["stat"].get("innings")), "bulk_G": sp["stat"].get("gamesPlayed")}
        for season, sport_id, sp in read_bulk(set(df.season), "fielding")
    )
    stints = (df.groupby(key, as_index=False)
                .agg(stint_innings=("innings", "sum"), stint_G=("G", "sum")))
    m = bulk.merge(stints, on=key, how="outer", indicator=True)
    outs_differ = (m.stint_innings * 3).round() != (m.bulk_innings * 3).round()
    dh_g_differs = (m.position == "DH") & (m.stint_G != m.bulk_G)
    bad = m[(m._merge != "both") | outs_differ | dh_g_differs]
    log = (bad[key + ["stint_innings", "bulk_innings", "stint_G", "bulk_G"]]
           .round({"stint_innings": 2, "bulk_innings": 2})
           .astype({"stint_G": "Int64", "bulk_G": "Int64"})
           .sort_values(key))
    print(f"fielding check covers {len(m)} player-level-positions")
    write_reconcile_log(log, "reconcile_fielding_log.csv", strict_season, "fielding innings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    seasons = parse_seasons(args.seasons)
    for season in seasons:
        for sport_id, n_teams, r_calls in pull_rosters(season, "fielding"):
            b_rows, b_calls = pull_bulk(season, sport_id, "fielding")
            print(f"{season} sportId={sport_id:>2}: bulk {b_rows:>5} rows ({b_calls} calls) | "
                  f"rosters for {n_teams} teams ({r_calls} calls)")

    df = build_table()
    check_against_bulk(df, strict_season=seasons[0] if len(seasons) == 1 else None)
    out = PROCESSED_DIR / "fielding_season.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

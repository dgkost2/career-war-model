"""Step 1.7: pull transactions, build rehab windows, match them to hitting stints.

Usage:
  python src/pull_transactions.py 2005-2026

1. Pull: /transactions per calendar month -> data/raw/transactions/{YYYY-MM}.json
   (skipped if cached). The current, unfinished month is never pulled.
   Coverage note: the API has almost no transactions before 2009 (318-1,009 a
   season in 2005-2008 vs 25k+ from 2009), so rehabs can't be flagged before 2009.

2. data/processed/transactions.parquet: every transaction, de-duplicated (the
   API often repeats a row with a new id), with is_rehab_start.
   A rehab start is a description like "X sent RHP Y on a rehab assignment to Z."
   (typeCode ASG, or SC in some years such as 2013). Placements/transfers that
   only mention rehab in the injury note ("... rehab from shoulder surgery")
   are not rehab starts.

3. data/processed/rehab_windows.parquet: one row per rehab assignment
   (player × start date × destination team), MLB- and MiLB-origin alike.
   end_date = the day before the earliest of:
     - his next rehab assignment (moved to another team),
     - his next other transaction (IL activation, option, release, ...), except
       IL placements/transfers and number changes,
   capped at the rule limit: 20 days for position players, 30 for pitchers.
   origin_level is the level of the team that sent him, looked up in
   teams_season. origin_source says where the team came from: 'fromTeam',
   'description' (team name before " sent"), or 'il_placement' (older rows name
   no team: the team of his latest IL placement before the rehab).

4. data/processed/rehab_stints.parquet: hitting stints at a rehab destination
   (same player, season, team), classified:
     all_rehab  stint G <= rehab days at that team and no non-rehab move into
                that team that season -> the whole stint is rehab
     mixed      otherwise -> needs game logs to split into rehab / non-rehab rows
"""
import argparse
import calendar
import datetime as dt
import json
import re

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons

TX_RAW = RAW_DIR / "transactions"
REHAB_START = re.compile(r"\bsent\b.*\brehab assignment\b", re.I)
PITCHER = re.compile(r"\bsent (?:RHP|LHP|P)\b")
# Transactions that don't end a rehab: IL placements/transfers, number changes.
PLACED_ON_LIST = re.compile(r"\bplaced\b.*\b(?:disabled|injured) list\b", re.I)
NOT_AN_END = re.compile(r"\b(?:placed|transferred)\b.*\b(?:disabled|injured) list\b", re.I)
# Non-rehab moves that put a player on a team's roster.
ROSTER_MOVES = {"ASG", "OPT", "OUT", "CLW", "SE", "SFA", "SGN", "TR", "RTN", "CU"}
REHAB_CAP_DAYS = {"pitcher": 30, "position": 20}


def months(years: list[int]) -> list[tuple[int, int]]:
    today = dt.date.today()
    return [(y, m) for y in years for m in range(1, 13) if (y, m) < (today.year, today.month)]


def pull(years: list[int]) -> None:
    for y, m in months(years):
        last = calendar.monthrange(y, m)[1]
        data, fetched = fetch_cached(
            "/transactions", TX_RAW / f"{y}-{m:02d}.json",
            params={"startDate": f"{y}-{m:02d}-01", "endDate": f"{y}-{m:02d}-{last}"},
        )
        if fetched:
            print(f"{y}-{m:02d}: {len(data.get('transactions', [])):>6} transactions (downloaded)")


def build_transactions() -> pd.DataFrame:
    rows = []
    for f in sorted(TX_RAW.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].json")):
        for x in json.loads(f.read_text(encoding="utf-8"))["transactions"]:
            rows.append({
                "txId": x["id"],
                "personId": x.get("person", {}).get("id"),
                "date": x.get("date"),
                "effectiveDate": x.get("effectiveDate"),
                "typeCode": x.get("typeCode"),
                "fromTeamId": x.get("fromTeam", {}).get("id"),
                "toTeamId": x.get("toTeam", {}).get("id"),
                "description": x.get("description"),
            })
    df = pd.DataFrame(rows)
    df = df.astype({"personId": "Int64", "fromTeamId": "Int64", "toTeamId": "Int64"})
    for c in ("date", "effectiveDate"):
        df[c] = pd.to_datetime(df[c])
    n = len(df)
    df = df.drop_duplicates([c for c in df.columns if c != "txId"]).reset_index(drop=True)
    print(f"transactions: {n} rows, {n - len(df)} duplicates dropped, {len(df)} kept")
    df["is_rehab_start"] = df.description.fillna("").str.contains(REHAB_START)
    return df


def team_lookup() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "teams_season.parquet")[["season", "teamId", "name", "level"]]


def build_rehab_windows(tx: pd.DataFrame) -> pd.DataFrame:
    r = tx[tx.is_rehab_start & tx.personId.notna()].copy()
    r = r.drop_duplicates(["personId", "date", "toTeamId"])  # same move under a second description
    r["season"] = r.date.dt.year
    r["role"] = r.description.str.contains(PITCHER).map({True: "pitcher", False: "position"})

    # Origin: fromTeam, else the team name before " sent".
    teams = team_lookup()
    by_id = teams.set_index(["season", "teamId"]).level
    by_name = teams.drop_duplicates(["season", "name"]).set_index(["season", "name"])
    r["origin_name"] = r.description.str.extract(r"^(.*?) sent ", expand=False)
    named = by_name.reindex(list(zip(r.season, r.origin_name)))
    r["origin_teamId"] = r.fromTeamId.fillna(pd.Series(named.teamId.values, index=r.index)).astype("Int64")
    # Older descriptions ("sent RHP X on a rehab assignment to Y.") name no team and
    # have no fromTeam: fall back to the team of his latest IL placement before the start.
    placed = tx[tx.description.fillna("").str.contains(PLACED_ON_LIST) & tx.personId.notna() & tx.toTeamId.notna()]
    placed = (placed[["personId", "date", "toTeamId"]].rename(columns={"toTeamId": "il_teamId"})
              .assign(key=lambda d: d.date).drop(columns="date").sort_values("key"))
    r = pd.merge_asof(r.assign(key=r.date).sort_values("key"), placed, on="key", by="personId",
                      direction="backward", tolerance=pd.Timedelta(days=200)).drop(columns="key")
    r["origin_source"] = "fromTeam"
    r.loc[r.fromTeamId.isna() & r.origin_teamId.notna(), "origin_source"] = "description"
    r.loc[r.origin_teamId.isna() & r.il_teamId.notna(), "origin_source"] = "il_placement"
    r["origin_teamId"] = r.origin_teamId.fillna(r.il_teamId).astype("Int64")
    r.loc[r.origin_teamId.isna(), "origin_source"] = None
    r["origin_level"] = [by_id.get((s, t), "unknown") if pd.notna(t) else "unknown"
                         for s, t in zip(r.season, r.origin_teamId)]
    r["to_level"] = [by_id.get((s, t), "unknown") for s, t in zip(r.season, r.toTeamId)]

    # Next rehab start and next ending transaction for the same player, strictly after the start.
    r = r.sort_values("date")
    nxt_rehab = (r[["personId", "date"]].rename(columns={"date": "next_rehab"})
                 .assign(key=lambda d: d.next_rehab).sort_values("key"))
    ends = tx[~tx.is_rehab_start & tx.personId.notna() & (tx.typeCode != "NUM")
              & ~tx.description.fillna("").str.contains(NOT_AN_END)]
    ends = (ends.assign(next_event=ends.effectiveDate.fillna(ends.date))[["personId", "next_event", "typeCode"]]
            .rename(columns={"typeCode": "next_event_type"}).assign(key=lambda d: d.next_event).sort_values("key"))
    r = r.assign(key=r.date)
    r = pd.merge_asof(r, nxt_rehab, on="key", by="personId", direction="forward", allow_exact_matches=False)
    r = pd.merge_asof(r, ends, on="key", by="personId", direction="forward", allow_exact_matches=False)

    cap = r.date + pd.to_timedelta(r.role.map(REHAB_CAP_DAYS) - 1, unit="D")
    cands = pd.DataFrame({"cap": cap,
                          "next_rehab": r.next_rehab - pd.Timedelta(days=1),
                          "next_event": r.next_event - pd.Timedelta(days=1)})
    r["end_date"] = cands.min(axis=1).clip(lower=r.date)
    r["end_reason"] = cands.idxmin(axis=1)
    r.loc[r.end_reason == "next_event", "end_reason"] = "next_event:" + r.next_event_type.fillna("?")
    r["days"] = (r.end_date - r.date).dt.days + 1

    cols = ["personId", "season", "date", "end_date", "days", "end_reason", "role",
            "origin_teamId", "origin_source", "origin_level", "toTeamId", "to_level", "description"]
    return (r[cols].rename(columns={"personId": "playerId", "date": "start_date", "toTeamId": "teamId"})
            .sort_values(["season", "playerId", "start_date"]).reset_index(drop=True))


def classify_stints(windows: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    hitting = pd.read_parquet(PROCESSED_DIR / "hitting_season.parquet")
    key = ["playerId", "season", "teamId"]
    w = windows.groupby(key).agg(rehab_days=("days", "sum"), n_windows=("days", "size"),
                                 origin_levels=("origin_level", lambda s: ",".join(sorted(set(s))))).reset_index()
    m = hitting.merge(w, on=key)

    moves = tx[~tx.is_rehab_start & tx.typeCode.isin(ROSTER_MOVES) & tx.personId.notna() & tx.toTeamId.notna()]
    moves = (moves.assign(season=moves.date.dt.year)
             .rename(columns={"personId": "playerId", "toTeamId": "teamId"})
             .groupby(key).size().rename("other_moves_in").reset_index())
    m = m.merge(moves, on=key, how="left").fillna({"other_moves_in": 0})

    too_many_games = m.G > m.rehab_days
    has_moves = m.other_moves_in > 0
    m["status"] = "all_rehab"
    m.loc[too_many_games | has_moves, "status"] = "mixed"
    m["mixed_reason"] = (too_many_games.map({True: "G>rehab_days", False: ""}) + " "
                         + has_moves.map({True: "other_move_in", False: ""})).str.strip()
    cols = key + ["sportId", "level", "G", "PA", "rehab_days", "n_windows", "origin_levels",
                  "other_moves_in", "status", "mixed_reason"]
    return m[cols].sort_values(key).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="+", help="year(s), e.g. 2025 or 2005-2026")
    args = parser.parse_args()
    pull(parse_seasons(args.years))

    tx = build_transactions()
    tx.to_parquet(PROCESSED_DIR / "transactions.parquet", index=False)

    windows = build_rehab_windows(tx)
    windows.to_parquet(PROCESSED_DIR / "rehab_windows.parquet", index=False)
    print(f"rehab windows: {len(windows)}")

    stints = classify_stints(windows, tx)
    stints.to_parquet(PROCESSED_DIR / "rehab_stints.parquet", index=False)
    print(f"hitting stints at a rehab team: {len(stints)} "
          f"({(stints.status == 'all_rehab').sum()} all_rehab, {(stints.status == 'mixed').sum()} mixed)")


if __name__ == "__main__":
    main()

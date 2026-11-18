"""Step 1.7b: split mixed rehab stints into rehab / non-rehab rows using game logs.

Usage:
  python src/split_rehab.py            # all seasons in rehab_stints.parquet
  python src/split_rehab.py 2025       # only mixed stints from these seasons
  (run pull_hitting.py and pull_transactions.py first)

For every "mixed" stint in rehab_stints.parquet (rehab_source "transactions"),
pulls that player's hitting game log for the season at the stint's level:
  /people/{id}/stats?stats=gameLog&group=hitting&season=Y&sportId=X
cached to data/raw/gamelogs/{playerId}_{season}_{sportId}.json (skipped if it
exists; sportId is required, without it only MLB games come back).

Each game at the stint's team is rehab if its date falls inside one of his
rehab windows at that team. The game totals must equal the stint's totals;
if they don't (incomplete game log), the stint is left whole and logged.

Output: data/processed/hitting_stints.parquet = hitting_season with
  is_rehab      True / False; null for a mixed stint that couldn't be split
  rehab_source  "transactions", "heuristic" (2005-2008 likely_rehab), or null
Mixed stints become two rows (is_rehab True and False); "all_rehab" and
heuristic stints are one row with is_rehab True; everything else False.
Stint totals are checked against hitting_season, and PA against the bulk
totals (data/processed/reconcile_rehab_split_log.csv).
"""
import argparse

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons
from pull_hitting import STAT_FIELDS
from rosters import read_bulk, write_reconcile_log

GAMELOG_RAW = RAW_DIR / "gamelogs"
KEY = ["playerId", "season", "teamId", "sportId"]
STATS = list(STAT_FIELDS)  # G, PA, AB, H, 2B, ...


def pull_gamelog(player_id: int, season: int, sport_id: int) -> tuple[list[dict], bool]:
    data, fetched = fetch_cached(
        f"/people/{player_id}/stats",
        GAMELOG_RAW / f"{player_id}_{season}_{sport_id}.json",
        params={"stats": "gameLog", "group": "hitting", "season": season, "sportId": sport_id},
    )
    stats = data.get("stats") or [{}]
    return stats[0].get("splits", []), fetched


def split_stints(mixed: pd.DataFrame, hitting: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Return (rows for split stints, list of stints that couldn't be split)."""
    out, unsplit, calls = [], [], 0
    stints = mixed.merge(hitting, on=KEY, suffixes=("_r", ""))
    for i, st in enumerate(stints.itertuples(index=False), 1):
        games, fetched = pull_gamelog(st.playerId, st.season, st.sportId)
        calls += fetched
        if i % 100 == 0:
            print(f"  {i}/{len(stints)} stints, {calls} API calls")
        g = [x for x in games if x.get("team", {}).get("id") == st.teamId and x.get("gameType", "R") == "R"]
        if not g:
            unsplit.append((*[getattr(st, k) for k in KEY], "no games at team in game log"))
            continue
        w = windows[(windows.playerId == st.playerId) & (windows.season == st.season) & (windows.teamId == st.teamId)]
        rows = pd.DataFrame([{col: x["stat"].get(f, 0) for col, f in STAT_FIELDS.items()} |
                             {"date": pd.Timestamp(x["date"])} for x in g])
        rows["is_rehab"] = rows.date.apply(lambda d: bool(((w.start_date <= d) & (d <= w.end_date)).any()))
        totals = rows[STATS].sum()
        expected = hitting.loc[(hitting.playerId == st.playerId) & (hitting.season == st.season)
                               & (hitting.teamId == st.teamId) & (hitting.sportId == st.sportId), STATS].iloc[0]
        if not totals.equals(expected.astype(totals.dtype)):
            diff = {c: (int(totals[c]), int(expected[c])) for c in STATS if totals[c] != expected[c]}
            unsplit.append((*[getattr(st, k) for k in KEY], f"game log != stint {diff}"))
            continue
        for flag, part in rows.groupby("is_rehab"):
            out.append({**{k: getattr(st, k) for k in KEY}, **part[STATS].sum().to_dict(),
                        "is_rehab": flag, "rehab_source": "transactions"})
    print(f"  done: {len(stints)} mixed stints, {calls} API calls, {len(unsplit)} couldn't be split")
    return pd.DataFrame(out), unsplit


def build(seasons: list[int] | None) -> pd.DataFrame:
    hitting = pd.read_parquet(PROCESSED_DIR / "hitting_season.parquet")
    rehab = pd.read_parquet(PROCESSED_DIR / "rehab_stints.parquet")
    windows = pd.read_parquet(PROCESSED_DIR / "rehab_windows.parquet")
    if seasons:
        hitting = hitting[hitting.season.isin(seasons)]
        rehab = rehab[rehab.season.isin(seasons)]

    mixed = rehab[(rehab.status == "mixed") & (rehab.rehab_source == "transactions")][KEY]
    whole = rehab[rehab.status.isin(["all_rehab", "likely_rehab"])][KEY + ["rehab_source"]]
    split, unsplit = split_stints(mixed, hitting, windows)

    log = pd.DataFrame(unsplit, columns=KEY + ["reason"])
    log.to_csv(PROCESSED_DIR / "rehab_unsplit_log.csv", index=False)  # even when empty
    print(f"  {len(log)} unsplit stints -> rehab_unsplit_log.csv")
    split_keys = split[KEY].drop_duplicates() if len(split) else pd.DataFrame(columns=KEY)

    base = hitting.merge(whole, on=KEY, how="left")
    base["is_rehab"] = base.rehab_source.notna().astype("boolean")
    unsplit_keys = pd.DataFrame([u[:4] for u in unsplit], columns=KEY)
    is_unsplit = base.set_index(KEY).index.isin(unsplit_keys.set_index(KEY).index)
    base.loc[is_unsplit, "is_rehab"] = pd.NA
    base.loc[is_unsplit, "rehab_source"] = "transactions"
    is_split = base.set_index(KEY).index.isin(split_keys.set_index(KEY).index)
    keep_cols = [c for c in hitting.columns if c not in STATS]
    if len(split):
        split = split.merge(hitting[keep_cols], on=KEY)
    df = pd.concat([base[~is_split], split], ignore_index=True)
    return df.sort_values(KEY + ["is_rehab"]).reset_index(drop=True), hitting, len(split_keys)


def check(df: pd.DataFrame, hitting: pd.DataFrame, strict_season: int | None) -> None:
    """Split rows must sum back to each original stint, and PA to bulk totals."""
    summed = df.groupby(KEY)[STATS].sum().sort_index()
    orig = hitting.set_index(KEY)[STATS].sort_index()
    if not summed.equals(orig):
        bad = summed.compare(orig)
        raise RuntimeError(f"split rows don't sum to original stints:\n{bad.head(20)}")
    bulk = pd.DataFrame({"season": s, "sportId": sp_id, "playerId": sp["player"]["id"],
                         "bulk_PA": sp["stat"]["plateAppearances"]}
                        for s, sp_id, sp in read_bulk(set(df.season), "hitting"))
    stint = df.groupby(["season", "sportId", "playerId"], as_index=False).PA.sum().rename(columns={"PA": "stint_PA"})
    m = bulk.merge(stint, on=["season", "sportId", "playerId"], how="outer")
    log = m[m.bulk_PA != m.stint_PA][["season", "sportId", "playerId", "stint_PA", "bulk_PA"]]
    write_reconcile_log(log.astype({"stint_PA": "Int64", "bulk_PA": "Int64"}),
                        "reconcile_rehab_split_log.csv", strict_season, "rehab-split PA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="*", help="limit to these season(s); default all")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons) if args.seasons else None

    df, hitting, n_split = build(seasons)
    check(df, hitting, strict_season=seasons[0] if seasons and len(seasons) == 1 else None)
    out = PROCESSED_DIR / ("hitting_stints.parquet" if not seasons else f"hitting_stints_{seasons[0]}-{seasons[-1]}.parquet")
    df.to_parquet(out, index=False)
    r = df[df.is_rehab == True]
    print(f"split {n_split} mixed stints | rehab rows {len(r)} ({int(r.PA.sum())} PA) | "
          f"non-rehab PA {int(df[df.is_rehab == False].PA.sum())} | wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

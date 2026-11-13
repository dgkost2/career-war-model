"""Shared roster-based stint pulling (used by pull_hitting.py and pull_fielding.py).

Stints come from each team's fullSeason roster with season stats hydrated: it
lists every player who was on the team and gives his per-team splits, plus
combined rows (no team, numTeams > 1) that are never stints. See CLAUDE.md
"Data decisions" for why the bulk /stats endpoint isn't used for stints.

The bulk /stats endpoint (one row per player × level, or per player × position
× level for fielding, tagged with only the last team) is used to reconcile
stint totals and as a fallback for single-team players missing from every roster.

Raw files:
  data/raw/{group}_roster/{season}_{sportId}_{teamId}.json   rosters
  data/raw/{group}/{season}_{sportId}_{offset}.json          bulk pages (limit 500)
"""
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, is_skipped, levels_for


PAGE_SIZE = 500


def roster_dir(group: str) -> Path:
    return RAW_DIR / f"{group}_roster"


def bulk_dir(group: str) -> Path:
    return RAW_DIR / group


def pull_bulk(season: int, sport_id: int, group: str) -> tuple[int, int]:
    """Page the bulk /stats query by offset until a short page. Returns (rows, api_calls)."""
    offset = rows = calls = 0
    while True:
        data, fetched = fetch_cached(
            "/stats",
            bulk_dir(group) / f"{season}_{sport_id}_{offset}.json",
            params={"stats": "season", "group": group, "playerPool": "ALL", "sportId": sport_id,
                    "season": season, "limit": PAGE_SIZE, "offset": offset},
        )
        calls += fetched
        splits = data["stats"][0]["splits"] if data.get("stats") else []
        rows += len(splits)
        if len(splits) < PAGE_SIZE:
            return rows, calls
        offset += PAGE_SIZE


def read_bulk(seasons: set[int], group: str) -> list[tuple[int, int, dict]]:
    """(season, sportId, split) for every cached bulk row in the given seasons."""
    out = []
    for f in sorted(bulk_dir(group).glob("*.json")):
        season, sport_id, _ = (int(x) for x in f.stem.split("_"))
        if season in seasons and not is_skipped(season, sport_id):
            out += [(season, sport_id, sp) for sp in json.loads(f.read_text(encoding="utf-8"))["stats"][0]["splits"]]
    return out


def bulk_fallback(covered: set[tuple[int, int, int]], seasons: set[int], group: str) -> list[tuple[int, int, dict]]:
    """Bulk rows for player-levels missing from every roster.

    `covered` holds (season, sportId, playerId) already built from rosters. A
    single-team bulk row is already a correct stint and is returned; a
    multi-team one can't be split, so it's left out with a warning (the
    reconciliation then logs it, or stops).
    """
    extra, unfixable = [], set()
    for season, sport_id, sp in read_bulk(seasons, group):
        if (season, sport_id, sp["player"]["id"]) in covered:
            continue
        if sp.get("numTeams", 1) == 1:
            extra.append((season, sport_id, sp))
        else:
            unfixable.add((season, sport_id, sp["player"]["id"], sp["player"]["fullName"]))
    if unfixable:
        print(f"warning: {len(unfixable)} multi-team player-levels missing from all rosters: {sorted(unfixable)[:10]}")
    print(f"bulk fallback ({group}): {len(extra)} single-team rows for player-levels missing from rosters")
    return extra


def write_reconcile_log(log: pd.DataFrame, name: str, strict_season: int | None, what: str) -> None:
    """Write mismatches to data/processed/{name} (even when empty, so a stale log never
    lingers). Raise if strict_season is set and has any mismatch."""
    path = PROCESSED_DIR / name
    log.to_csv(path, index=False)
    print(f"{what} check: {len(log)} mismatches -> {path}")
    if len(log):
        print(log.groupby("season").size().rename("mismatches").to_string())
    if strict_season is not None and (log.season == strict_season).any():
        bad = log[log.season == strict_season]
        raise RuntimeError(f"{what} doesn't match bulk totals for {strict_season}:\n{bad.to_string(index=False)}")


def season_teams(season: int) -> pd.DataFrame:
    teams = pd.read_parquet(PROCESSED_DIR / "teams_season.parquet")
    out = teams[teams.season == season]
    if out.empty:
        raise RuntimeError(f"no teams for {season} in teams_season.parquet; run pull_teams.py {season}")
    return out


def pull_roster(season: int, sport_id: int, team_id: int, group: str) -> bool:
    """Fetch (or reuse cached) one team's roster with `group` stats. Returns True if fetched."""
    _, fetched = fetch_cached(
        f"/teams/{team_id}/roster",
        roster_dir(group) / f"{season}_{sport_id}_{team_id}.json",
        params={"rosterType": "fullSeason", "season": season,
                "hydrate": f"person(stats(type=season,season={season},group={group},sportId={sport_id}))"},
    )
    return fetched


def pull_rosters(season: int, group: str) -> Iterator[tuple[int, int, int]]:
    """Pull every in-scope team's roster for a season.
    Yields (sportId, teams, api_calls) per level."""
    teams = season_teams(season)
    for sport_id in levels_for(season):
        level_teams = teams[teams.sportId == sport_id].teamId
        calls = sum(pull_roster(season, sport_id, t, group) for t in level_teams)
        yield sport_id, len(level_teams), calls


def iter_stint_splits(group: str) -> Iterator[tuple[int, int, dict]]:
    """Yield (season, sportId, split) for every per-team regular-season split in the
    cached rosters. Combined rows and skipped levels are dropped. A traded player
    appears on each of his teams' rosters, so the same split can be yielded more
    than once; callers dedupe."""
    for f in sorted(roster_dir(group).glob("*.json")):
        season, sport_id, _ = (int(x) for x in f.stem.split("_"))
        if is_skipped(season, sport_id):
            continue
        for entry in json.loads(f.read_text(encoding="utf-8")).get("roster", []):
            for st in entry["person"].get("stats", []):
                for sp in st["splits"]:
                    if "team" not in sp or sp.get("numTeams", 1) > 1:
                        continue  # combined/total row: never a stint
                    if sp.get("gameType", "R") != "R" or int(sp["season"]) != season:
                        continue
                    if sp["sport"]["id"] != sport_id:
                        raise RuntimeError(f"{f.name}: split at sportId {sp['sport']['id']}")
                    yield season, sport_id, sp

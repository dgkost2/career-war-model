"""Shared roster-based stint pulling (used by pull_hitting.py and pull_fielding.py).

Stints come from each team's fullSeason roster with season stats hydrated: it
lists every player who was on the team and gives his per-team splits, plus
combined rows (no team, numTeams > 1) that are never stints. See CLAUDE.md
"Data decisions" for why the bulk /stats endpoint isn't used for stints.

Raw files: data/raw/{group}_roster/{season}_{sportId}_{teamId}.json
"""
import json
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, is_skipped, levels_for


def roster_dir(group: str) -> Path:
    return RAW_DIR / f"{group}_roster"


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

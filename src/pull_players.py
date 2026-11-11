"""Step 1.2: pull player lists per level and season.

Usage:
  python src/pull_players.py 2025
  python src/pull_players.py 2023 2024 2025
  python src/pull_players.py 2005-2025

Caches each response to data/raw/players/{season}_{sportId}.json and skips
files that already exist.
"""
import argparse

from mlb_api import LEVEL_IDS, RAW_DIR, fetch_cached, parse_seasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="+", help="season(s), e.g. 2025, or a range like 2005-2025")
    args = parser.parse_args()

    for season in parse_seasons(args.seasons):
        for sport_id in LEVEL_IDS:
            cache_file = RAW_DIR / "players" / f"{season}_{sport_id}.json"
            data, fetched = fetch_cached(
                f"/sports/{sport_id}/players", cache_file, params={"season": season}
            )
            status = "downloaded" if fetched else "cached"
            print(f"{season} sportId={sport_id:>2}: {len(data.get('people', [])):>5} players ({status})")


if __name__ == "__main__":
    main()

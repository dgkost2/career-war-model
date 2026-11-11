"""Step 1.3: pull biographical data for every player in the cached players files.

Usage:
  python src/pull_people.py            # IDs from every cached players file
  python src/pull_people.py 2025       # IDs from 2025 players files only
  python src/pull_people.py 2005-2025

Reads player IDs from data/raw/players/{season}_{sportId}.json, skips IDs
already present in any cached people file, and fetches the rest from
/people?personIds=... in batches of 100. Each batch is cached to
data/raw/people/batch_{hash}.json, named by a hash of its IDs so batches
stay valid when more seasons are added later.

Then rebuilds data/processed/people.parquet from every cached people file.
"""
import argparse
import hashlib
import json
import re

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons

PLAYERS_RAW = RAW_DIR / "players"
PEOPLE_RAW = RAW_DIR / "people"
BATCH_SIZE = 100


def player_ids(seasons: list[int] | None) -> set[int]:
    files = sorted(PLAYERS_RAW.glob("*_*.json"))
    if seasons is not None:
        files = [f for f in files if int(f.stem.split("_")[0]) in seasons]
    if not files:
        raise RuntimeError("no cached players files found; run pull_players.py first")
    ids: set[int] = set()
    for f in files:
        ids.update(p["id"] for p in json.loads(f.read_text(encoding="utf-8"))["people"])
    return ids


def cached_people() -> list[dict]:
    people = []
    for f in sorted(PEOPLE_RAW.glob("batch_*.json")):
        people.extend(json.loads(f.read_text(encoding="utf-8"))["people"])
    return people


def pull(ids: set[int]) -> None:
    have = {p["id"] for p in cached_people()}
    todo = sorted(ids - have)
    print(f"{len(ids)} player IDs, {len(ids) - len(todo)} already cached, {len(todo)} to fetch")
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        key = hashlib.sha1(",".join(map(str, batch)).encode()).hexdigest()[:12]
        data, _ = fetch_cached(
            "/people",
            PEOPLE_RAW / f"batch_{key}.json",
            params={"personIds": ",".join(map(str, batch))},
        )
        got = {p["id"] for p in data["people"]}
        if missing := set(batch) - got:
            print(f"  warning: API returned no record for {sorted(missing)}")
        print(f"  batch {i // BATCH_SIZE + 1}/{-(-len(todo) // BATCH_SIZE)}: {len(got)} people")


def height_inches(h: str | None) -> int | None:
    """'6\\' 2"' -> 74"""
    m = re.match(r"(\d+)'\s*(\d+)", h or "")
    return int(m[1]) * 12 + int(m[2]) if m else None


def build_table() -> pd.DataFrame:
    rows = {}
    for p in cached_people():
        rows[p["id"]] = {  # dict keyed by id, so a person in two batches appears once
            "id": p["id"],
            "fullName": p.get("fullName"),
            "birthDate": p.get("birthDate"),
            "birthCountry": p.get("birthCountry"),
            "batSide": p.get("batSide", {}).get("code"),
            "pitchHand": p.get("pitchHand", {}).get("code"),
            "height": p.get("height"),
            "heightInches": height_inches(p.get("height")),
            "weight": p.get("weight"),
            "primaryPosition": p.get("primaryPosition", {}).get("abbreviation"),
            "draftYear": p.get("draftYear"),
            "mlbDebutDate": p.get("mlbDebutDate"),
        }
    df = pd.DataFrame(rows.values())
    df["birthDate"] = pd.to_datetime(df["birthDate"])
    df["mlbDebutDate"] = pd.to_datetime(df["mlbDebutDate"])
    df = df.astype({"heightInches": "Int64", "weight": "Int64", "draftYear": "Int64"})
    return df.sort_values("id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("seasons", nargs="*", help="limit to players from these season(s); default all cached")
    args = parser.parse_args()

    pull(player_ids(parse_seasons(args.seasons) if args.seasons else None))

    df = build_table()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "people.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

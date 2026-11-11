"""Shared helpers for the MLB Stats API pull scripts."""
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

# sportIds in scope for v1, highest level first.
LEVEL_IDS = [1, 11, 12, 13, 14, 16]

SLEEP_SECONDS = 0.5

_session = requests.Session()
_session.headers["User-Agent"] = "career-war-model/0.1"


def parse_seasons(tokens: list[str]) -> list[int]:
    """Parse CLI season args like ['2025'] or ['2005-2010', '2015'] into a sorted list."""
    seasons: set[int] = set()
    for tok in tokens:
        if "-" in tok:
            start, end = (int(x) for x in tok.split("-", 1))
            seasons.update(range(start, end + 1))
        else:
            seasons.add(int(tok))
    return sorted(seasons)


def fetch_json(path: str, params: dict | None = None) -> dict:
    """GET {BASE_URL}{path} and return parsed JSON. Raises on HTTP errors."""
    resp = _session.get(f"{BASE_URL}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_cached(path: str, cache_file: Path, params: dict | None = None) -> tuple[dict, bool]:
    """Return JSON from cache_file if it exists; otherwise fetch, save, and sleep.

    Returns (data, fetched) where fetched is True if an API call was made.
    """
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8")), False
    data = fetch_json(path, params)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(cache_file)  # atomic, so a crash never leaves a half-written cache file
    time.sleep(SLEEP_SECONDS)
    return data, True

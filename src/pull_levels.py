"""Step 1.1: pull /sports and build the levels lookup.

Outputs:
  data/raw/sports.json            raw API response (active sports only)
  data/raw/sports_{id}.json       inactive in-scope levels, e.g. 15 Short-Season A
  data/processed/levels.parquet   sportId, name, abbreviation for in-scope levels
"""
import pandas as pd

from mlb_api import LEVEL_IDS, PROCESSED_DIR, RAW_DIR, fetch_cached


def main() -> None:
    data, fetched = fetch_cached("/sports", RAW_DIR / "sports.json")
    print(f"sports.json: {'downloaded' if fetched else 'cached'}")

    sports = data["sports"]
    # /sports lists only active sports; fetch retired in-scope levels (e.g. 15) by id.
    for sport_id in sorted(set(LEVEL_IDS) - {s["id"] for s in sports}):
        extra, fetched = fetch_cached(f"/sports/{sport_id}", RAW_DIR / f"sports_{sport_id}.json")
        print(f"sports_{sport_id}.json: {'downloaded' if fetched else 'cached'} (inactive level)")
        sports += extra["sports"]

    sports = pd.DataFrame(sports)
    levels = (
        sports.loc[sports["id"].isin(LEVEL_IDS), ["id", "name", "abbreviation"]]
        .rename(columns={"id": "sportId"})
        .set_index("sportId")
        .loc[LEVEL_IDS]  # keep MLB -> Rookie order; raises KeyError if a level is missing
        .reset_index()
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "levels.parquet"
    levels.to_parquet(out, index=False)
    print(f"wrote {out} ({len(levels)} rows)")
    print(levels.to_string(index=False))


if __name__ == "__main__":
    main()

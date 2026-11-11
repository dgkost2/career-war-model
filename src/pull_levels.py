"""Step 1.1: pull /sports and build the levels lookup.

Outputs:
  data/raw/sports.json            raw API response
  data/processed/levels.parquet   sportId, name, abbreviation for in-scope levels
"""
import pandas as pd

from mlb_api import LEVEL_IDS, PROCESSED_DIR, RAW_DIR, fetch_cached


def main() -> None:
    data, fetched = fetch_cached("/sports", RAW_DIR / "sports.json")
    print(f"sports.json: {'downloaded' if fetched else 'cached'}")

    sports = pd.DataFrame(data["sports"])
    levels = (
        sports.loc[sports["id"].isin(LEVEL_IDS), ["id", "name", "abbreviation"]]
        .rename(columns={"id": "sportId"})
        .set_index("sportId")
        .loc[LEVEL_IDS]  # keep MLB -> Rookie order
        .reset_index()
    )

    missing = set(LEVEL_IDS) - set(levels["sportId"])
    if missing:
        raise RuntimeError(f"sportIds missing from /sports response: {sorted(missing)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "levels.parquet"
    levels.to_parquet(out, index=False)
    print(f"wrote {out} ({len(levels)} rows)")
    print(levels.to_string(index=False))


if __name__ == "__main__":
    main()

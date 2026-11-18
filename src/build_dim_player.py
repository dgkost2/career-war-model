"""Step 2.1: build dim_player (one row per player) from people, draft, hitting_stints
and fielding_season.

Usage:
  python src/build_dim_player.py      (build_warehouse.py also runs it)

Writes data/processed/dim_player.parquet; build_warehouse.py loads it into
warehouse.duckdb as dim_player. Every player in people gets a row.

Columns:
  playerId, fullName, birthDate, birthCountry, batSide, pitchHand, heightInches, weight
  birthCountry      full names; the 3-letter codes the API sometimes returns
                    (DOM, VEN, ...) and a few name variants are mapped
  acquisition_type  "draft"          he has a signing draft (draft.is_signing_draft)
                    "international"  born outside USA / Canada / Puerto Rico, no signing draft
                    "other/unknown"  everything else (undrafted US/CAN/PR free agents,
                                     unknown birthCountry, drafted pitchers with no
                                     hitting stints, whose signing draft can't be set)
  draft_year, draft_round, draft_pick (overall), draft_school_type   from the signing draft
  first_pro_season  first season he appears in hitting_stints or fielding_season
                    (fielding covers pitchers, whose first hitting stint can come
                    years after their debut)
  first_pro_level   lowest level he appeared at in that season
  first_pro_age     age on July 1 of first_pro_season, in years (1 decimal)
  first_pro_censored  True when his real pro debut is (or may be) before
                    first_pro_season: first_pro_season is 2005, the first season
                    pulled, or his MLB debut came before it (veterans, mostly
                    pitchers whose first hitting stint since 2005 came late)
  mlb_debut_season  year of people.mlbDebutDate, else his first MLB stint
Players with no hitting or fielding stints have null first_pro_* fields.
"""
import pandas as pd

from mlb_api import PROCESSED_DIR

FIRST_SEASON = 2005
DOMESTIC = {"USA", "Canada", "Puerto Rico"}
COUNTRY_NAMES = {
    "DOM": "Dominican Republic", "VEN": "Venezuela", "MEX": "Mexico", "CUB": "Cuba",
    "PAN": "Panama", "PUR": "Puerto Rico", "COL": "Colombia", "BAH": "Bahamas",
    "NCA": "Nicaragua", "CUW": "Curacao", "JPN": "Japan", "CAN": "Canada",
    "ESP": "Spain", "HKG": "Hong Kong", "KUW": "Kuwait", "NIR": "Northern Ireland",
    "RU1": "Russia",
    # name variants
    "United States of America": "USA", "Republic of Korea": "South Korea",
}
# Lowest to highest, for picking a player's first level in his first season.
LEVEL_ORDER = ["DSL", "VSL", "ROK", "A(Short)", "A", "A+", "AA", "AAA", "MEX", "MLB"]


def build() -> pd.DataFrame:
    people = pd.read_parquet(PROCESSED_DIR / "people.parquet")
    draft = pd.read_parquet(PROCESSED_DIR / "draft.parquet")
    cols = ["playerId", "season", "level"]
    stints = pd.concat([pd.read_parquet(PROCESSED_DIR / "hitting_stints.parquet", columns=cols),
                        pd.read_parquet(PROCESSED_DIR / "fielding_season.parquet", columns=cols)])

    df = people.rename(columns={"id": "playerId"})[
        ["playerId", "fullName", "birthDate", "birthCountry", "batSide", "pitchHand",
         "heightInches", "weight", "mlbDebutDate"]].copy()
    df["birthCountry"] = df.birthCountry.replace(COUNTRY_NAMES)

    signing = (draft[draft.is_signing_draft == True]
               .rename(columns={"year": "draft_year", "round": "draft_round", "pickNumber": "draft_pick",
                                "school_type": "draft_school_type"})
               [["playerId", "draft_year", "draft_round", "draft_pick", "draft_school_type"]])
    if signing.playerId.duplicated().any():
        raise RuntimeError("player with more than one signing draft")
    df = df.merge(signing, on="playerId", how="left")

    df["acquisition_type"] = "other/unknown"
    df.loc[df.birthCountry.notna() & ~df.birthCountry.isin(DOMESTIC), "acquisition_type"] = "international"
    df.loc[df.draft_year.notna(), "acquisition_type"] = "draft"

    rank = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
    s = stints.assign(level_rank=stints.level.map(rank))
    if s.level_rank.isna().any():
        raise RuntimeError(f"unknown levels: {set(s.level[s.level_rank.isna()])}")
    first = (s.sort_values(["playerId", "season", "level_rank"]).groupby("playerId").first()
             [["season", "level"]].rename(columns={"season": "first_pro_season", "level": "first_pro_level"}))
    df = df.merge(first, on="playerId", how="left")
    july1 = pd.to_datetime(df.first_pro_season.astype("Int64").astype("string") + "-07-01", errors="coerce")
    df["first_pro_age"] = ((july1 - df.birthDate).dt.days / 365.25).round(1)
    first_mlb = stints[stints.level == "MLB"].groupby("playerId").season.min()
    df["mlb_debut_season"] = df.mlbDebutDate.dt.year.astype("Int64").fillna(
        df.playerId.map(first_mlb).astype("Int64"))
    censored = (df.first_pro_season == FIRST_SEASON) | (df.mlb_debut_season < df.first_pro_season).fillna(False)
    df["first_pro_censored"] = censored.astype("boolean").where(df.first_pro_season.notna())

    df = df.astype({"draft_year": "Int64", "draft_round": "string", "draft_pick": "Int64",
                    "first_pro_season": "Int64"})
    cols = ["playerId", "fullName", "birthDate", "birthCountry", "batSide", "pitchHand", "heightInches",
            "weight", "acquisition_type", "draft_year", "draft_round", "draft_pick", "draft_school_type",
            "first_pro_season", "first_pro_level", "first_pro_age", "first_pro_censored", "mlb_debut_season"]
    if df.playerId.duplicated().any():
        raise RuntimeError("dim_player has duplicate playerIds")
    return df[cols].sort_values("playerId").reset_index(drop=True)


def main() -> None:
    df = build()
    out = PROCESSED_DIR / "dim_player.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

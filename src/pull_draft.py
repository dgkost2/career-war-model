"""Step 1.6: pull the Rule 4 (June amateur) draft for each year and build draft.parquet.

Usage:
  python src/pull_draft.py 2005-2026

Caches /draft/{year} to data/raw/draft/{year}.json (skipped if it exists), then
rebuilds data/processed/draft.parquet from every cached year.

Columns: playerId, year, round, pickNumber (overall), teamId, school,
school_class, school_type, school_type_source, signingBonus, signed,
draftType, isPass, is_signing_draft.
  school_class      API schoolClass (e.g. 'HS SR', 'JC J2', '4YR JR'); only in
                    2009 and 2018+
  school_type       HS / JUCO / 4YR. school_type_source says how it was set:
                    'api' (own schoolClass), 'lookup' (same school's type in
                    years that have schoolClass), 'name' (guessed from the name;
                    reliable for HS, can call a JUCO named 'X College' 4YR)
  signingBonus      dollars, from the API (2017+ only), else null
  signed            the API has no signed flag in any year, so always null
  is_signing_draft  the draft he signed from. If the API has no signed flag:
                    the latest draft at or before his first pro season in
                    hitting_season. Null when he has no pro hitting stints
                    (unsigned, or a pitcher), since it can't be determined.
A player can appear in several drafts (e.g. HS draftee redrafted out of
college); every draft row is kept.
"""
import argparse
import json
import re

import pandas as pd

from mlb_api import PROCESSED_DIR, RAW_DIR, fetch_cached, parse_seasons

DRAFT_RAW = RAW_DIR / "draft"

HS_PATTERN = re.compile(r"\bHS\b|High School|High Shool|\bH\.S\.|Academy|Prep\b|Secondary|\bSS\b|Collegiate|"
                        r"\bSchool\b", re.I)
JUCO_PATTERN = re.compile(r"\bJC\b|\bCC\b|Junior College|Community College|\bJ\.C\.|\bC\.C\.", re.I)


def class_type(school_class: str | None) -> str | None:
    """API schoolClass -> HS / JUCO / 4YR. Prefixed values ('HS SR', 'JC J2', '4YR JR')
    are explicit; bare college classes ('JR', 'SR', 'SO', '5S', 'FR', 'GR') are 4YR
    (checked: their schools are all colleges); bare J1-J3 are JUCO; 'NS' is no school."""
    if not isinstance(school_class, str) or not school_class:
        return None
    sc = school_class.upper()
    if sc.startswith("HS"):
        return "HS"
    if sc.startswith("JC") or sc[:2] in ("J1", "J2", "J3"):
        return "JUCO"
    if sc.startswith("4YR") or sc in ("JR", "SR", "SO", "5S", "FR", "GR"):
        return "4YR"
    return None


def name_type(name: str | None) -> str | None:
    """Guess from the school name alone. Reliable for HS; misses JUCOs named 'X College'."""
    if not isinstance(name, str) or not name or name == "No School":
        return None
    if HS_PATTERN.search(name):
        return "HS"
    if JUCO_PATTERN.search(name):
        return "JUCO"
    return "4YR"


def add_school_type(df: pd.DataFrame) -> pd.DataFrame:
    """school_type from, in order: the pick's own schoolClass ('api'); the type the
    same school name has in other years' schoolClass ('lookup'); the name ('name')."""
    api = df.school_class.map(class_type)
    labeled = df.assign(t=api).dropna(subset=["t", "school"])
    labeled = labeled[labeled.school != "No School"]  # not a school; its type varies by player
    lookup = labeled.groupby("school").t.agg(lambda s: s.mode().iloc[0])
    by_lookup = df.school.map(lookup)
    by_name = df.school.map(name_type)
    df["school_type"] = api.fillna(by_lookup).fillna(by_name)
    df["school_type_source"] = (pd.Series("name", index=df.index).where(by_name.notna())
                                .mask(by_lookup.notna(), "lookup").mask(api.notna(), "api"))
    return df


def pull(years: list[int]) -> None:
    for year in years:
        data, fetched = fetch_cached(f"/draft/{year}", DRAFT_RAW / f"{year}.json")
        n = sum(len(r["picks"]) for r in data["drafts"]["rounds"])
        print(f"{year}: {n:>5} picks ({'downloaded' if fetched else 'cached'})")


def build_table() -> pd.DataFrame:
    rows = []
    for f in sorted(DRAFT_RAW.glob("*.json")):
        for rnd in json.loads(f.read_text(encoding="utf-8"))["drafts"]["rounds"]:
            for p in rnd["picks"]:
                school = p.get("school", {})
                rows.append({
                    "playerId": p.get("person", {}).get("id"),
                    "year": int(p["year"]),
                    "round": p["pickRound"],
                    "pickNumber": p["pickNumber"],
                    "teamId": p.get("team", {}).get("id"),
                    "school": school.get("name"),
                    "school_class": school.get("schoolClass"),
                    "signingBonus": p.get("signingBonus"),
                    "signed": p.get("signed"),
                    "draftType": p.get("draftType", {}).get("code"),
                    "isPass": p.get("isPass"),
                })
    df = pd.DataFrame(rows).astype({"playerId": "Int64", "teamId": "Int64"})
    # The API lists at least one pick twice (2008, pick 127); keep one copy.
    n = len(df)
    df = df.drop_duplicates(["year", "pickNumber", "playerId"])
    if len(df) < n:
        print(f"dropped {n - len(df)} duplicate pick rows")
    if df.duplicated(["year", "pickNumber"]).any():
        raise RuntimeError(f"two players at one pick:\n{df[df.duplicated(['year', 'pickNumber'], keep=False)]}")
    df = add_school_type(df)
    df["signingBonus"] = pd.to_numeric(df["signingBonus"], errors="coerce")
    df["signed"] = df["signed"].astype("boolean")
    df["is_signing_draft"] = signing_draft(df)
    return df.sort_values(["year", "pickNumber"]).reset_index(drop=True)


def signing_draft(df: pd.DataFrame) -> pd.Series:
    """True for the draft each player signed from, False for his other drafts,
    null where it can't be determined."""
    out = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if df["signed"].notna().any():
        # API signed flag: a player signs at most once, so use it directly.
        known = df["signed"].notna()
        out[known] = df.loc[known, "signed"]
        return out

    hitting = pd.read_parquet(PROCESSED_DIR / "hitting_season.parquet")
    first_pro = hitting.groupby("playerId").season.min()
    d = df[df.playerId.notna()].assign(first_pro=lambda x: x.playerId.map(first_pro))
    has_pro = d.first_pro.notna()
    out[d.index[has_pro]] = False
    eligible = d[has_pro & (d.year <= d.first_pro)]
    latest = eligible.groupby("playerId").year.transform("max")
    out[eligible.index[eligible.year == latest]] = True
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("years", nargs="+", help="draft year(s), e.g. 2025 or 2005-2026")
    args = parser.parse_args()

    pull(parse_seasons(args.years))
    df = build_table()
    out = PROCESSED_DIR / "draft.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()

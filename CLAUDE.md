# Career WAR Projection Model

Context for Claude Code working in this repo. Read this before writing code.

## Goal
Project rest-of-career WAR for every active MLB and MiLB player, viewable per player on a website.
Two numbers per player:
- **Conditional WAR** — projected career WAR *if* he reaches and sticks in MLB.
- **Expected WAR** — P(reach/stick) × conditional WAR. This is the headline number.

## Scope (v1)
- Hitters (position players) only. Pitchers come later.
- No scout grades, contracts, or fantasy points yet.
- No pitch-level / Statcast data in v1.

## Data
- Source: official MLB Stats API (https://statsapi.mlb.com/api/v1/), no key needed.
- Start year: 2005 (pending spot-check of 2005 MiLB coverage).
- Levels (sportId): MLB=1, AAA=11, AA=12, High-A=13, Single-A=14, Rookie/Complex=16.
- Note: no MiLB season in 2020.

## Environment
- Windows, PowerShell, Python 3.14 in `.venv` (activate: `.venv\Scripts\Activate.ps1`).
- Packages: requests, pandas, duckdb, pyarrow, scikit-learn, jupyter, matplotlib (see requirements.txt).
- When writing requirements.txt from PowerShell, make sure it is saved as UTF-8 (PowerShell `>` writes UTF-16).

## Folder layout
- `data/raw/` — untouched API responses (JSON). Never edit. Git-ignored.
- `data/processed/` — Parquet files + `warehouse.duckdb`. Git-ignored.
- `src/` — pipeline scripts (pull, clean, model).
- `notebooks/` — exploration only; anything reused moves to `src/`.
- `models/` — saved model artifacts.

## Rules for code
- Cache API responses to `data/raw/` and skip re-downloading files that already exist.
- Sleep briefly between API calls; never hammer the API.
- Scripts should be re-runnable from scratch without manual steps.
- Validate against held-out cohorts before trusting any projection.

## Data decisions
- Stats are stored as stints: one row per player × season × team × level. Traded/promoted players have multiple stints. Drop any combined/total rows to avoid double counting.
- Org membership comes from a teams-by-season table (team → level, league, parentOrgId), since affiliations change year to year.
- Transactions are used for events only (rehab assignments, signings, releases, trades, IL, promotions), not to build stat stints. MLB players' MiLB rehab stints must be flagged.
- Hitter vs pitcher is decided per player-season from PA and batters faced, not primaryPosition. Two-way players go through both models later.
- currentAge and currentTeam are as-of-today fields; never use them for season data. Season age comes from birthDate.

## Build order (phases)
0 Setup → 1 Data pull → 2 Clean & build tables → 3 MiLB→MLB translations → 4 Talent estimate (weighting + regression) → 5 Aging curves → 6 Playing time & attrition → 7 WAR calc → 8 Reach-MLB model & expected WAR → 9 Validation → 10 Pitchers → 11 Storage (Supabase) & site (Next.js/Vercel) → 12 Scout grades, contracts, prospect ranks.

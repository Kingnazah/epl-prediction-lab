"""Data loading, cleaning and merging (football-data.co.uk format)."""
import numpy as np
import pandas as pd

CORE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
STATS = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HF", "AF", "HR", "AR"]
ODDS = ["AvgH", "AvgD", "AvgA", "AvgCH", "AvgCD", "AvgCA", "B365H", "B365D", "B365A"]
KEY = ["Date", "HomeTeam", "AwayTeam"]
OUT_COLS = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + STATS + ODDS


def season_of(ts) -> str:
    """EPL season label: a match in Aug 2026 belongs to '2026/27'."""
    y = ts.year if ts.month >= 7 else ts.year - 1
    return f"{y}/{str((y + 1) % 100).zfill(2)}"


def parse_dates(series: pd.Series) -> pd.Series:
    """ISO (YYYY-MM-DD) is parsed strictly; anything else is read day-first (dd/mm/yyyy or dd/mm/yy)."""
    s = series.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(s[~iso], dayfirst=True, errors="coerce")
    return out


def _prep_common(df: pd.DataFrame, required) -> pd.DataFrame:
    df = df.copy()
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["Date"] = parse_dates(df["Date"])
    for c in ("HomeTeam", "AwayTeam"):
        df[c] = df[c].astype(str).str.strip()
    return df


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a results table into the canonical layout. Rows without a final score are dropped."""
    df = _prep_common(df, CORE)
    for c in ("FTHG", "FTAG"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Date", "FTHG", "FTAG"])
    df = df[(df.HomeTeam != "") & (df.AwayTeam != "") & (df.HomeTeam != "nan") & (df.AwayTeam != "nan")]
    df["FTHG"], df["FTAG"] = df["FTHG"].astype(int), df["FTAG"].astype(int)
    df["FTR"] = np.where(df.FTHG > df.FTAG, "H", np.where(df.FTHG < df.FTAG, "A", "D"))
    for c in STATS + ODDS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    df["Season"] = df["Date"].map(season_of)
    return df[OUT_COLS].sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def standardise_fixtures(df: pd.DataFrame) -> pd.DataFrame:
    """Clean an upcoming-fixtures table (football-data 'fixtures.csv' layout or manual entry)."""
    df = _prep_common(df, KEY)
    df = df.dropna(subset=["Date"])
    df = df[(df.HomeTeam != "") & (df.AwayTeam != "") & (df.HomeTeam != "nan") & (df.AwayTeam != "nan")]
    for c in ODDS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return df[KEY + ODDS].sort_values(["Date", "HomeTeam"]).reset_index(drop=True)


def load_history(path) -> pd.DataFrame:
    return standardise(pd.read_csv(path))


def merge_results(history: pd.DataFrame, new: pd.DataFrame):
    """Append genuinely new results. Existing rows are never overwritten; score conflicts are reported."""
    new = standardise(new)
    existing_mask = new.set_index(KEY).index.isin(history.set_index(KEY).index)
    dup = new[existing_mask].merge(history[KEY + ["FTHG", "FTAG"]], on=KEY, suffixes=("", "_old"))
    conflicts = int(((dup.FTHG != dup.FTHG_old) | (dup.FTAG != dup.FTAG_old)).sum())
    added = new[~existing_mask]
    merged = (pd.concat([history, added], ignore_index=True)
                .sort_values(["Date", "HomeTeam"]).reset_index(drop=True))
    return merged, {"added": int(len(added)), "already_present": int(existing_mask.sum()), "conflicts": conflicts}


def unknown_teams(history: pd.DataFrame, names) -> list:
    known = set(history.HomeTeam) | set(history.AwayTeam)
    return sorted(set(names) - known)


def seasons_in(history: pd.DataFrame) -> list:
    return sorted(history.Season.unique())


def teams_in_season(history: pd.DataFrame, season: str) -> set:
    m = history.Season == season
    return set(history.loc[m, "HomeTeam"]) | set(history.loc[m, "AwayTeam"])


def promoted_in(history: pd.DataFrame, season: str, extra_teams=()) -> set:
    """Teams in `season` that were not in the previous season of the data (promoted / returning)."""
    earlier = [s for s in seasons_in(history) if s < season]
    if not earlier:
        return set()
    cur = teams_in_season(history, season) | set(extra_teams)
    return cur - teams_in_season(history, earlier[-1])

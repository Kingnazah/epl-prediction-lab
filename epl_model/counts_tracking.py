"""
Corners prediction log and scoring (same rules as tracking.py): append-only, tamper-evident hash chain,
the FIRST prediction logged for a fixture + model version is the one that counts.
Scoring compares the model with the league-average baseline (mu_clim) stored at prediction time.
"""
import hashlib

import numpy as np
import pandas as pd

from . import config
from .counts import K_POISSON, logpmf, over_prob
from .tracking import GENESIS, prediction_id

LINES = config.CORNERS["lines"]
OVER_COLS = [f"p_over_{L}" for L in LINES]
FLOAT_COLS = ["mu_home", "mu_away", "mu_total", "k", "mu_clim"] + OVER_COLS
LOG_COLUMNS = (["prediction_id", "logged_at_utc", "model_version", "Season", "Date", "HomeTeam", "AwayTeam"]
               + FLOAT_COLS + ["prev_hash", "row_hash"])
HASH_FIELDS = ["prediction_id", "logged_at_utc", "model_version", "Date", "HomeTeam", "AwayTeam"] + FLOAT_COLS


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def load_log(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return empty_log()
    for c in LOG_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Corners log file is missing column '{c}'")
    return df[LOG_COLUMNS]


def _fmt(field, value) -> str:
    if field in FLOAT_COLS:
        return "" if pd.isna(value) else f"{float(value):.5f}"
    if field == "Date":
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value)


def _row_hash(prev_hash: str, row) -> str:
    payload = prev_hash + "|" + "|".join(_fmt(f, row[f]) for f in HASH_FIELDS)
    return hashlib.sha256(payload.encode()).hexdigest()


def append_predictions(log: pd.DataFrame, preds: pd.DataFrame, now=None):
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    today = (now.tz_localize(None) if now.tzinfo is None else now.tz_convert("UTC").tz_localize(None)).normalize()
    log = log.copy()
    prev_hash = log["row_hash"].iloc[-1] if len(log) else GENESIS
    existing = set(log["prediction_id"])
    rows, dup, past = [], 0, 0
    for _, p in preds.iterrows():
        pid = prediction_id(p.Date, p.HomeTeam, p.AwayTeam, p.model_version)
        if pid in existing:
            dup += 1
            continue
        if pd.Timestamp(p.Date) < today:
            past += 1
            continue
        row = {"prediction_id": pid, "logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "model_version": p.model_version, "Season": p.Season,
               "Date": pd.Timestamp(p.Date).strftime("%Y-%m-%d"), "HomeTeam": p.HomeTeam, "AwayTeam": p.AwayTeam}
        for c in FLOAT_COLS:
            row[c] = round(float(p[c]), 5)
        row["prev_hash"] = prev_hash
        row["row_hash"] = _row_hash(prev_hash, row)
        prev_hash = row["row_hash"]
        rows.append(row)
        existing.add(pid)
    if rows:
        add = pd.DataFrame(rows)[LOG_COLUMNS]
        log = pd.concat([log, add], ignore_index=True)[LOG_COLUMNS] if len(log) else add
    return log, {"added": len(rows), "already_logged": dup, "kickoff_passed": past}


def verify_chain(log: pd.DataFrame):
    prev = GENESIS
    for i, (_, r) in enumerate(log.iterrows()):
        if r["prev_hash"] != prev:
            return False, f"Chain broken at row {i + 1}: a row was removed, reordered or edited."
        if _row_hash(prev, r) != r["row_hash"]:
            return False, f"Row {i + 1} was modified after it was logged."
        prev = r["row_hash"]
    return True, f"Chain intact: {len(log)} logged corner prediction(s), none altered."


def evaluate(log: pd.DataFrame, results: pd.DataFrame):
    """Score logged corner predictions against actual results. None if nothing is settled yet."""
    if log.empty:
        return None
    lg = log.copy()
    lg["Date"] = pd.to_datetime(lg["Date"])
    res = results.dropna(subset=["HC", "AC"])[["Date", "HomeTeam", "AwayTeam", "HC", "AC"]]
    m = lg.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    if m.empty:
        return None
    y = (m.HC + m.AC).values.astype(float)
    mu, k, muc = (m[c].values.astype(float) for c in ("mu_total", "k", "mu_clim"))
    ls_m, ls_c = logpmf(y, mu, k), logpmf(y, muc, K_POISSON)
    d = ls_m - ls_c
    out = {"n": len(m), "logscore": ls_m.mean(), "clim_logscore": ls_c.mean(),
           "diff": d.mean(), "diff_se": d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
           "mae": np.abs(y - mu).mean(), "clim_mae": np.abs(y - muc).mean(), "actual_mean": y.mean(),
           "pred_mean": mu.mean()}
    rows = []
    for L in LINES:
        o = (y > L).astype(float)
        p = np.clip(m[f"p_over_{L}"].values.astype(float), 1e-6, 1 - 1e-6)
        pc = np.clip(over_prob(L, muc, K_POISSON), 1e-6, 1 - 1e-6)
        ll = lambda q: -(o * np.log(q) + (1 - o) * np.log(1 - q)).mean()  # noqa: E731
        rows.append({"Line": f"Over {L}", "Matches over": int(o.sum()), "Model log loss": ll(p), "Baseline log loss": ll(pc)})
    out["lines"] = pd.DataFrame(rows)
    m["total"] = y.astype(int)
    out["table"] = m[["Date", "HomeTeam", "AwayTeam", "mu_total", "total"]]
    return out


def reliability_note(n: int) -> str:
    if n < 100:
        return (f"Only {n} settled corner prediction(s): far too few to judge. In testing this model beat the league "
                "average by a very small margin, which cannot be seen in so few matches.")
    if n < 380:
        return f"{n} settled corner predictions. Still too few to separate a small edge from noise."
    return f"{n} settled corner predictions. Enough to start drawing cautious conclusions."

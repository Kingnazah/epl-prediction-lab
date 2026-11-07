"""
Prediction log and live performance tracking.

The log is append-only and tamper-evident: every row stores a SHA-256 hash that covers the previous
row's hash, so editing or deleting an old row breaks the chain and is detected by verify_chain().
(Tamper-EVIDENT, not tamper-proof: keep your own downloaded copies.)

Rule: the FIRST prediction logged for a fixture + model version is the one that counts. It is never replaced.
"""
import hashlib

import numpy as np
import pandas as pd

from . import config

LOG_COLUMNS = ["prediction_id", "logged_at_utc", "model_version", "Season", "Date", "HomeTeam", "AwayTeam",
               "p_H", "p_D", "p_A", "elo_H", "elo_D", "elo_A", "dc_H", "dc_D", "dc_A",
               "xg_home", "xg_away", "p_over25", "p_btts", "mkt_H", "mkt_D", "mkt_A",
               "prev_hash", "row_hash"]
HASH_FIELDS = ["prediction_id", "logged_at_utc", "model_version", "Date", "HomeTeam", "AwayTeam",
               "p_H", "p_D", "p_A", "xg_home", "xg_away"]
FLOAT_COLS = ["p_H", "p_D", "p_A", "elo_H", "elo_D", "elo_A", "dc_H", "dc_D", "dc_A",
              "xg_home", "xg_away", "p_over25", "p_btts", "mkt_H", "mkt_D", "mkt_A"]
GENESIS = "0" * 64
_Y = {"H": 0, "D": 1, "A": 2}


def empty_log() -> pd.DataFrame:
    return pd.DataFrame(columns=LOG_COLUMNS)


def load_log(path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return empty_log()
    for c in LOG_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Log file is missing column '{c}'")
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


def prediction_id(date, home, away, version) -> str:
    return hashlib.sha1(f"{pd.Timestamp(date):%Y-%m-%d}|{home}|{away}|{version}".encode()).hexdigest()[:12]


def append_predictions(log: pd.DataFrame, preds: pd.DataFrame, now=None):
    """Add predictions for fixtures not yet logged. Returns (new_log, report)."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    today = now.tz_localize(None).normalize() if now.tzinfo is None else now.tz_convert("UTC").tz_localize(None).normalize()
    log = log.copy()
    prev_hash = log["row_hash"].iloc[-1] if len(log) else GENESIS
    existing = set(log["prediction_id"])
    rows, skipped_dup, skipped_past = [], 0, 0
    for _, p in preds.iterrows():
        pid = prediction_id(p.Date, p.HomeTeam, p.AwayTeam, p.model_version)
        if pid in existing:
            skipped_dup += 1
            continue
        if pd.Timestamp(p.Date) < today:
            skipped_past += 1          # kickoff date already passed: cannot be a genuine pre-match prediction
            continue
        row = {"prediction_id": pid, "logged_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
               "model_version": p.model_version, "Season": p.Season,
               "Date": pd.Timestamp(p.Date).strftime("%Y-%m-%d"), "HomeTeam": p.HomeTeam, "AwayTeam": p.AwayTeam}
        for c in FLOAT_COLS:
            row[c] = round(float(p[c]), 5) if c in p and pd.notna(p[c]) else np.nan
        row["prev_hash"] = prev_hash
        row["row_hash"] = _row_hash(prev_hash, row)
        prev_hash = row["row_hash"]
        rows.append(row)
        existing.add(pid)
    if rows:
        log = pd.concat([log, pd.DataFrame(rows)], ignore_index=True)[LOG_COLUMNS] if len(log) else pd.DataFrame(rows)[LOG_COLUMNS]
    return log, {"added": len(rows), "already_logged": skipped_dup, "kickoff_passed": skipped_past}


def verify_chain(log: pd.DataFrame):
    """Returns (ok, message). Recomputes every hash from the stored values."""
    prev = GENESIS
    for i, (_, r) in enumerate(log.iterrows()):
        if r["prev_hash"] != prev:
            return False, f"Chain broken at row {i + 1}: previous-hash mismatch (a row was removed, reordered or edited)."
        if _row_hash(prev, r) != r["row_hash"]:
            return False, f"Row {i + 1} was modified after it was logged."
        prev = r["row_hash"]
    return True, f"Chain intact: {len(log)} logged prediction(s), none altered."


def _logloss(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))


def _brier(P, y):
    return ((P - np.eye(3)[y]) ** 2).sum(1)


def evaluate(log: pd.DataFrame, results: pd.DataFrame):
    """Score logged predictions against actual results. Returns None if nothing is settled yet."""
    if log.empty:
        return None
    lg = log.copy()
    lg["Date"] = pd.to_datetime(lg["Date"])
    res = results[["Date", "HomeTeam", "AwayTeam", "FTR"]]
    m = lg.merge(res, on=["Date", "HomeTeam", "AwayTeam"], how="inner")
    if m.empty:
        return None
    y = m.FTR.map(_Y).values
    P = m[["p_H", "p_D", "p_A"]].values.astype(float)

    # naive baseline: league outcome frequencies from results strictly before each match
    r = results.sort_values("Date")
    dates = r.Date.values
    oh = np.eye(3)[r.FTR.map(_Y).values]
    cum = np.vstack([np.zeros(3), np.cumsum(oh, 0)])
    k = np.searchsorted(dates, m.Date.values, side="left")
    cnt = cum[k]
    Pn = (cnt + 1) / (cnt.sum(1, keepdims=True) + 3)

    out = {"n": len(m), "logloss": _logloss(P, y).mean(), "brier": _brier(P, y).mean(),
           "accuracy": float((P.argmax(1) == y).mean()),
           "naive_logloss": _logloss(Pn, y).mean(), "naive_accuracy": float((Pn.argmax(1) == y).mean())}
    ll_model, ll_naive = _logloss(P, y), _logloss(Pn, y)
    d = ll_model - ll_naive
    out["vs_naive_diff"] = d.mean()
    out["vs_naive_se"] = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan

    mk = m[["mkt_H", "mkt_D", "mkt_A"]].values.astype(float)
    has_mkt = ~np.isnan(mk).any(1)
    out["n_market"] = int(has_mkt.sum())
    if has_mkt.sum() >= 1:
        out["market_logloss"] = _logloss(mk[has_mkt], y[has_mkt]).mean()
        out["model_logloss_on_market_rows"] = ll_model[has_mkt].mean()
        out["market_accuracy"] = float((mk[has_mkt].argmax(1) == y[has_mkt]).mean())
        dm = ll_model[has_mkt] - _logloss(mk[has_mkt], y[has_mkt])
        out["vs_market_diff"] = dm.mean()
        out["vs_market_se"] = dm.std(ddof=1) / np.sqrt(len(dm)) if len(dm) > 1 else np.nan
    m["outcome"] = m.FTR
    out["table"] = m[["Date", "HomeTeam", "AwayTeam", "p_H", "p_D", "p_A", "outcome"]]
    return out


def reliability_note(n: int) -> str:
    if n < 100:
        return (f"Only {n} settled prediction(s). This is far too few to judge the model: normal luck alone can swing "
                "log loss by more than the differences we care about.")
    if n < 380:
        return (f"{n} settled predictions. Still too few for firm conclusions; differences between models of about "
                "0.02 log loss cannot yet be separated from noise.")
    return f"{n} settled predictions. Enough to start drawing cautious conclusions; keep tracking."

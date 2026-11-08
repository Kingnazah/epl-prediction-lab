"""Self-checks for the corners model and its log. Run: python test_corners.py"""
import numpy as np
import pandas as pd

from epl_model import config
from epl_model.counts import CornersEngine, over_prob, pmf
from epl_model.counts_tracking import (append_predictions, empty_log, evaluate, verify_chain, reliability_note, OVER_COLS)
from epl_model.data import load_history

hist = load_history("data/epl_history.csv")
bt = pd.read_csv("data/backtest_counts_v1.csv", parse_dates=["Date"])
eng = CornersEngine(hist)
passed = 0


def check(name, cond):
    global passed
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    passed += 1


# 1. reproduces the stored walk-forward backtest (final corners model = team model) on several blocks
worst_mu = worst_p = 0.0
for b in [0, 19, 38, 57, 76, 95, 114, 122]:
    blk = bt.iloc[10 * b:10 * b + 10]
    p = eng.predict(blk[["Date", "HomeTeam", "AwayTeam"]])
    worst_mu = max(worst_mu, np.abs(p.mu_total.values - blk.cor_mu_team.values).max())
    for L in config.CORNERS["lines"]:
        worst_p = max(worst_p, np.abs(p[f"p_over_{L}"].values - blk[f"cor_over{L}_final"].values).max())
print(f"      max |mean diff| {worst_mu:.5f} corners, max |probability diff| {worst_p:.5f}")
check("engine reproduces the stored corners backtest (means within 0.01, probabilities within 0.002)", worst_mu < 0.01 and worst_p < 0.002)

# 2. valid outputs
fx = pd.DataFrame({"Date": ["07/11/2026", "07/11/2026", "08/11/2026"],
                   "HomeTeam": ["Arsenal", "Coventry", "Newcastle"],
                   "AwayTeam": ["Chelsea", "Man City", "Atlantis FC"]})
p = eng.predict(fx)
check("dd/mm/yyyy dates parsed", p.Date.min() == pd.Timestamp("2026-11-07"))
check("expected corners are positive and sensible (6 to 15)", p.mu_total.between(6, 15).all())
check("home + away means add up to the total", np.allclose(p.mu_home + p.mu_away, p.mu_total))
lines = [f"p_over_{L}" for L in config.CORNERS["lines"]]
check("over probabilities lie in [0,1] and fall as the line rises", ((p[lines] >= 0) & (p[lines] <= 1)).all().all()
      and (p[lines].diff(axis=1).iloc[:, 1:] <= 1e-12).all().all())
xs = np.arange(0, 80)
check("probability distribution of the total sums to 1", np.allclose([pmf(xs, m, 80).sum() for m in p.mu_total], 1, atol=1e-6))
check("over probability matches the distribution", np.isclose(over_prob(9.5, 10.3, 80), 1 - pmf(np.arange(0, 10), 10.3, 80).sum()))
check("unknown team flagged, prediction still finite", "not in history" in p["warnings"].iloc[2] and np.isfinite(p.mu_total).all())

# 3. no leakage: wrecking every corner count from the fixture date onward must not change the prediction
fx_old = pd.DataFrame({"Date": ["2024-01-13"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Crystal Palace"]})
p1 = eng.predict(fx_old)
h2 = hist.copy()
m = h2.Date >= pd.Timestamp("2024-01-13")
h2.loc[m, ["HC", "AC"]] = h2.loc[m, ["AC", "HC"]].values * 3 + 5
p2 = CornersEngine(h2).predict(fx_old)
check("future corner counts cannot change a prediction (no leakage)", np.allclose(p1.mu_total, p2.mu_total, atol=1e-12))

# 4. rows without corner counts are ignored, not crashed on
h3 = hist.copy()
h3.loc[h3.index[-30:], ["HC", "AC"]] = np.nan
p3 = CornersEngine(h3).predict(fx)
check("matches with missing corner counts are skipped safely", np.isfinite(p3.mu_total).all())

try:
    CornersEngine(hist[hist.Date < pd.Timestamp("2021-10-01")]).predict(pd.DataFrame(
        {"Date": ["2021-10-02"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Chelsea"]}))
    ok = False
except ValueError:
    ok = True
check("raises ValueError when there is too little history", ok)

# 5. corners log: freeze, dedupe, skip past, tamper detection
now = pd.Timestamp("2026-11-05T10:00:00Z")
pr = eng.predict(pd.DataFrame({"Date": ["2026-11-07", "2026-11-08", "2026-11-03"],
                               "HomeTeam": ["Arsenal", "Chelsea", "Everton"],
                               "AwayTeam": ["Liverpool", "Brighton", "Fulham"]}))
log, rep = append_predictions(empty_log(), pr, now)
check("log adds future fixtures, skips passed kickoff", rep["added"] == 2 and rep["kickoff_passed"] == 1)
log2, rep2 = append_predictions(log, pr, now + pd.Timedelta(hours=5))
check("re-logging is ignored (first prediction stands)", rep2["added"] == 0 and rep2["already_logged"] == 2)
check("hash chain verifies on untouched log", verify_chain(log)[0])
t = log.copy(); t.loc[0, "mu_total"] = 15.0
check("editing a prediction is detected", not verify_chain(t)[0])
check("deleting an earlier row is detected", not verify_chain(log.iloc[1:].reset_index(drop=True))[0])
rt = pd.read_csv(pd.io.common.StringIO(log.to_csv(index=False)))
check("hash chain survives CSV save/load", verify_chain(rt)[0])

# 6. scoring against real results
blk = bt.iloc[:60][["Date", "HomeTeam", "AwayTeam"]]
lg = np.nan
pb = eng.predict(blk)
past, _ = append_predictions(empty_log(), pb, pd.Timestamp("2023-08-01T00:00:00Z"))
res = evaluate(past, hist)
check("evaluation joins predictions with results", res is not None and res["n"] == 60)
check("scores are finite and sane", np.isfinite(res["logscore"]) and -3.2 < res["logscore"] < -2.2 and 1.5 < res["mae"] < 4.5)
check("baseline scored too", np.isfinite(res["clim_logscore"]) and len(res["lines"]) == len(config.CORNERS["lines"]))
check("reliability note warns at small n", "far too few" in reliability_note(60))
check("evaluate returns None when nothing settled", evaluate(empty_log(), hist) is None)

print(f"\nAll {passed} checks passed.")

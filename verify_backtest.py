"""
Audit script: does the live engine reproduce the stored v3 backtest?

For several blocks of 10 test matches, ask the engine to predict them using only earlier results
and compare with data/backtest_predictions_v3.csv.

Expected: Dixon-Coles part matches to ~1e-3. The Elo part differs slightly because the backtest updated
Elo after each match inside a block, whereas live predictions for a whole matchweek are made before any of
its matches are played.

Run:  python verify_backtest.py
"""
import numpy as np
import pandas as pd

from epl_model.data import load_history
from epl_model.engine import Engine

hist = load_history("data/epl_history.csv")
bt = pd.read_csv("data/backtest_predictions_v3.csv", parse_dates=["Date"])
eng = Engine(hist)

blocks = [0, 19, 38, 57, 76, 95, 114, 122]          # block b = matches 10*b .. 10*b+9 of the test period
rows = []
for b in blocks:
    blk = bt.iloc[10 * b: 10 * b + 10]
    pred = eng.predict(blk[["Date", "HomeTeam", "AwayTeam"]])
    m = blk.merge(pred, on=["Date", "HomeTeam", "AwayTeam"], suffixes=("_bt", ""))
    dc_diff = np.abs(m[["dc_v3_H", "dc_v3_D", "dc_v3_A"]].values - m[["dc_H", "dc_D", "dc_A"]].values).max()
    bl_diff = np.abs(m[["blend_v3_H", "blend_v3_D", "blend_v3_A"]].values - m[["p_H", "p_D", "p_A"]].values).max()
    rows.append({"block": b, "first_date": blk.Date.min().date(), "n": len(m),
                 "max |DC diff|": round(dc_diff, 5), "max |blend diff|": round(bl_diff, 5)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
ok = (out["max |DC diff|"] < 0.005).all() and (out["max |blend diff|"] < 0.03).all()
print("\nRESULT:", "PASS - engine reproduces the backtest within tolerance" if ok else "FAIL - investigate")
raise SystemExit(0 if ok else 1)

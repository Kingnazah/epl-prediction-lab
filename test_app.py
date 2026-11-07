"""Headless smoke test of the Streamlit app (no browser needed). Run: python test_app.py"""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from epl_model.data import load_history
from epl_model.engine import Engine
from epl_model.tracking import append_predictions, empty_log

hist = load_history("data/epl_history.csv")

# 1. cold start renders every tab without errors
at = AppTest.from_file("app.py", default_timeout=180).run()
assert not at.exception, at.exception
print("PASS  app starts and all five tabs render")
assert len(at.tabs) == 5
print("PASS  five tabs present:", [t.label for t in at.tabs])

# 2. prediction display path (pre-load predictions as if the user pressed Predict), incl. odds comparison
fx = pd.DataFrame({"Date": ["2026-11-07", "2026-11-07", "2026-11-08"],
                   "HomeTeam": ["Arsenal", "Coventry", "Newcastle"],
                   "AwayTeam": ["Chelsea", "Man City", "Wolves"],
                   "AvgH": [2.1, 6.0, np.nan], "AvgD": [3.6, 4.6, np.nan], "AvgA": [3.5, 1.5, np.nan]})
preds = Engine(hist).predict(fx)
at.session_state["preds"] = preds
at.run()
assert not at.exception, at.exception
print("PASS  predictions render (cards, charts, market comparison, warnings)")

# 3. pressing "Log these predictions" appends to the log
btn = [b for b in at.button if b.key == "log_btn"]
assert btn, "log button not found"
btn[0].click().run()
assert not at.exception, at.exception
print("PASS  log button works; rows now in log:", len(at.session_state["log"]))

# 4. tracking tab with settled predictions (logged in the past) computes performance
bt = pd.read_csv("data/backtest_predictions_v3.csv", parse_dates=["Date"])
pb = Engine(hist).predict(bt.iloc[:30][["Date", "HomeTeam", "AwayTeam"]])
past_log, _ = append_predictions(empty_log(), pb, pd.Timestamp("2023-08-01T00:00:00Z"))
at.session_state["log"] = past_log
at.run()
assert not at.exception, at.exception
metrics = [m.label for m in at.metric]
assert "Log loss (lower is better)" in metrics, metrics
print("PASS  tracking tab scores settled predictions; metrics:", metrics)

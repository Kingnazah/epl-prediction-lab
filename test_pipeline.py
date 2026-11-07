"""Self-checks for data merging, prediction log integrity, scoring and edge cases. Run: python test_pipeline.py"""
import numpy as np
import pandas as pd

from epl_model.data import load_history, merge_results, standardise_fixtures, unknown_teams
from epl_model.engine import Engine
from epl_model.tracking import (append_predictions, empty_log, evaluate, verify_chain, reliability_note)

hist = load_history("data/epl_history.csv")
eng = Engine(hist)
passed = 0


def check(name, cond):
    global passed
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    passed += 1


# 1. football-data style dd/mm/yyyy fixtures + unknown team + promoted team flags
fx_raw = pd.DataFrame({"Date": ["07/11/2026", "07/11/2026", "08/11/2026"],
                       "HomeTeam": ["Arsenal", "Coventry", "Newcastle"],
                       "AwayTeam": ["Chelsea", "Man City", "Atlantis FC"],
                       "AvgH": [2.1, 6.0, np.nan], "AvgD": [3.6, 4.6, np.nan], "AvgA": [3.5, 1.5, np.nan]})
pred = eng.predict(fx_raw)
check("dd/mm/yyyy dates parsed", pred.Date.min() == pd.Timestamp("2026-11-07"))
check("probabilities sum to 1", np.allclose(pred[["p_H", "p_D", "p_A"]].sum(axis=1), 1))
check("unknown team flagged", "not in history" in pred["warnings"].iloc[2])
check("promoted team flagged", "promoted" in pred["warnings"].iloc[1])
check("market de-vig present only where odds given", pred.mkt_H.notna().tolist() == [True, True, False])
check("de-vigged market sums to 1", np.allclose(pred.loc[pred.mkt_H.notna(), ["mkt_H", "mkt_D", "mkt_A"]].sum(axis=1), 1))
check("top scores produced", all(isinstance(s, str) and "-" in s for s in pred.top_scores))
check("unknown_teams helper", unknown_teams(hist, ["Arsenal", "Atlantis FC"]) == ["Atlantis FC"])

# 2. no leakage: results on/after the fixture date must not influence the prediction
fx_old = pd.DataFrame({"Date": ["2024-01-13"], "HomeTeam": ["Arsenal"], "AwayTeam": ["Crystal Palace"]})
p1 = eng.predict(fx_old)
h2 = hist.copy()
future = h2.Date >= pd.Timestamp("2024-01-13")
h2.loc[future, ["FTHG", "FTAG"]] = h2.loc[future, ["FTAG", "FTHG"]].values     # scramble the future
h2["FTR"] = np.where(h2.FTHG > h2.FTAG, "H", np.where(h2.FTHG < h2.FTAG, "A", "D"))
p2 = Engine(h2).predict(fx_old)
check("future results cannot change a prediction (no leakage)", np.allclose(p1[["p_H", "p_D", "p_A"]], p2[["p_H", "p_D", "p_A"]]))

# 3. merge: new results appended, duplicates ignored, conflicts reported, nothing overwritten
new = pd.DataFrame({"Date": ["02/11/2026", "05/11/2026"], "HomeTeam": ["Tottenham", "Arsenal"],
                    "AwayTeam": ["Chelsea", "Wolves"], "FTHG": [9, 2], "FTAG": [9, 0]})
last_row = hist.iloc[-1]
conflict_row = pd.DataFrame({"Date": [last_row.Date.strftime("%d/%m/%Y")], "HomeTeam": [last_row.HomeTeam],
                             "AwayTeam": [last_row.AwayTeam], "FTHG": [last_row.FTHG + 3], "FTAG": [last_row.FTAG]})
merged, rep = merge_results(hist, pd.concat([new, conflict_row], ignore_index=True))
check("merge adds only unseen fixtures", rep["added"] >= 1 and len(merged) == len(hist) + rep["added"])
check("merge reports score conflict", rep["conflicts"] == 1)
check("existing result not overwritten", int(merged[(merged.Date == last_row.Date) & (merged.HomeTeam == last_row.HomeTeam)].FTHG.iloc[0]) == last_row.FTHG)

# 4. prediction log: freeze, dedupe, skip past kickoffs, tamper detection
now = pd.Timestamp("2026-11-05T10:00:00Z")
preds = eng.predict(pd.DataFrame({"Date": ["2026-11-07", "2026-11-08", "2026-11-03"],
                                  "HomeTeam": ["Arsenal", "Chelsea", "Everton"],
                                  "AwayTeam": ["Liverpool", "Brighton", "Fulham"]}))
log, rep = append_predictions(empty_log(), preds, now)
check("log adds future fixtures, skips passed kickoff", rep["added"] == 2 and rep["kickoff_passed"] == 1)
log2, rep2 = append_predictions(log, preds, now + pd.Timedelta(hours=5))
check("re-logging the same fixture is ignored (first prediction stands)", rep2["added"] == 0 and rep2["already_logged"] == 2)
ok, msg = verify_chain(log)
check("hash chain verifies on untouched log", ok)
tampered = log.copy()
tampered.loc[0, "p_H"] = 0.99
check("editing a probability is detected", not verify_chain(tampered)[0])
tampered2 = log.iloc[1:].reset_index(drop=True)
check("deleting an earlier row is detected", not verify_chain(tampered2)[0])
roundtrip = pd.read_csv(pd.io.common.StringIO(log.to_csv(index=False)))
check("hash chain survives CSV save/load", verify_chain(roundtrip)[0])

# 5. scoring: log first test block as if before kickoff, then score vs real results
bt = pd.read_csv("data/backtest_predictions_v3.csv", parse_dates=["Date"])
blk = bt.iloc[:40][["Date", "HomeTeam", "AwayTeam"]]
pb = eng.predict(blk)
lg, _ = append_predictions(empty_log(), pb, pd.Timestamp("2023-08-01T00:00:00Z"))
res = evaluate(lg, hist)
check("evaluation joins logged predictions with results", res is not None and res["n"] == 40)
check("log loss is a sane number", 0.5 < res["logloss"] < 1.6)
check("naive baseline computed", 0.9 < res["naive_logloss"] < 1.3)
check("reliability note warns at small n", "too few" in reliability_note(40))
check("evaluate returns None when nothing settled", evaluate(empty_log(), hist) is None)

print(f"\nAll {passed} checks passed.")

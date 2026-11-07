# EPL Prediction Lab (model v3.0)

Weekly 1X2 (home / draw / away) probabilities for the Premier League, with an honest, tamper-evident track record.
Research tool, not betting advice: in testing the model was clearly less accurate than bookmaker closing odds.

## What is inside
| Path | Purpose |
|---|---|
| `app.py` | The dashboard (Streamlit): Predict, Update results, Tracking, Ratings, About |
| `epl_model/config.py` | Frozen model settings. Change anything here = a NEW model version (bump `MODEL_VERSION`) |
| `epl_model/data.py` | Loading, cleaning, merging results |
| `epl_model/elo.py`, `poisson.py` | The two models |
| `epl_model/engine.py` | Blends them into the final probabilities |
| `epl_model/tracking.py` | Append-only prediction log with a hash chain, and live scoring |
| `data/epl_history.csv` | 1,990 matches, 2021/22 to 2 Nov 2026 (football-data.co.uk, downloaded 4 Nov 2026) |
| `data/predictions_log.csv` | Your track record (starts empty) |
| `data/backtest_predictions_v3.csv` | The 1,230 backtest predictions, used by the audit script |
| `verify_backtest.py`, `test_pipeline.py`, `test_app.py` | Self-checks (see bottom) |

## Weekly routine
1. **Update results** tab: upload the current-season `E0.csv` from football-data.co.uk (or type results in).
2. **Predict** tab: upload `fixtures.csv` from football-data.co.uk, or type the fixtures. Add odds if you want a market comparison.
3. Press **Predict**, read the cards, then press **Log these predictions** BEFORE kickoff.
4. **Tracking** tab: download the log. Do this every session.
5. Repeat after the next round. Judge the model only after hundreds of settled matches.

## Run it on a computer
```
pip install -r requirements.txt
streamlit run app.py
```

## Put it online for free (Streamlit Community Cloud)
1. Create a free GitHub account and a new repository (public is fine).
2. Upload every file and folder from this project into the repository (GitHub: Add file, Upload files). A computer makes this much easier than a phone.
3. Go to share.streamlit.io, sign in with GitHub, choose New app, pick your repository, branch `main`, main file `app.py`, then Deploy.
4. The online app forgets changes when it restarts. To keep updates, replace `data/epl_history.csv` and `data/predictions_log.csv` in GitHub with the files you download from the app.

## Self-checks
- `python verify_backtest.py`: confirms the engine reproduces the stored backtest (Dixon-Coles part to within 0.0001).
- `python test_pipeline.py`: 23 checks including no-leakage, log tamper detection and edge cases.
- `python test_app.py`: headless test of every tab.

## Known limits
- No injury, suspension or line-up information (the biggest known gap; likely needs a paid data source).
- Newly promoted teams and early-season matches are more uncertain.
- Over 2.5 and BTTS are shown but were no better than base rates in testing. Corners and cards models are not built yet.
- The log is tamper-evident, not tamper-proof: keep your own downloaded copies.

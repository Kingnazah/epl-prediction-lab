"""
EPL Prediction Lab: weekly 1X2 predictions with an honest, tamper-evident track record.
Run locally:  streamlit run app.py
"""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from epl_model import config
from epl_model.data import (load_history, merge_results, seasons_in, standardise_fixtures, unknown_teams, KEY)
from epl_model.engine import Engine
from epl_model.tracking import (append_predictions, evaluate, load_log, reliability_note, verify_chain)
from epl_model import counts as cn
from epl_model import counts_tracking as ct

ROOT = Path(__file__).parent
st.set_page_config(page_title="EPL Prediction Lab", page_icon="⚽", layout="wide")


# ------------------------------------------------------------------ state
@st.cache_data
def _default_history() -> pd.DataFrame:
    return load_history(ROOT / "data" / "epl_history.csv")


if "history" not in st.session_state:
    st.session_state.history = _default_history()
if "log" not in st.session_state:
    st.session_state.log = load_log(ROOT / "data" / "predictions_log.csv")
if "clog" not in st.session_state:
    st.session_state.clog = ct.load_log(ROOT / "data" / "corners_log.csv")

hist: pd.DataFrame = st.session_state.history


@st.cache_data(show_spinner="Fitting the model on all completed matches…")
def run_predict(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    return Engine(history).predict(fixtures)


@st.cache_data(show_spinner="Computing current team ratings…")
def run_ratings(history: pd.DataFrame) -> pd.DataFrame:
    return Engine(history).ratings()


@st.cache_data(show_spinner="Fitting the corners model…")
def run_corners(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    return cn.CornersEngine(history).predict(fixtures)


def pct(x) -> str:
    return "–" if pd.isna(x) else f"{x * 100:.1f}%"


# ------------------------------------------------------------------ header
st.title("⚽ EPL Prediction Lab")
st.caption(f"Model {config.MODEL_VERSION} (frozen 4 Nov 2026) · {len(hist):,} matches · "
           f"results through {hist.Date.max():%d %b %Y} · source: football-data.co.uk")
st.warning("Research tool, not betting advice. In testing, this model was clearly **less accurate than bookmaker "
           "closing odds**, so it has no proven edge. Every number below is a probability, never a certainty.",
           icon="⚠️")

tab_pred, tab_corn, tab_upd, tab_track, tab_rate, tab_about = st.tabs(
    ["Predict", "Corners (experimental)", "Update results", "Tracking", "Ratings", "About"])

recent = seasons_in(hist)[-2:]
TEAMS = sorted(set(hist[hist.Season.isin(recent)].HomeTeam) | set(hist[hist.Season.isin(recent)].AwayTeam))

# ================================================================== PREDICT
with tab_pred:
    st.subheader("1. Choose fixtures")
    mode = st.radio("How do you want to enter them?", ["Type them in", "Upload fixtures.csv"],
                    horizontal=True, key="fx_mode")
    fx_input = None
    if mode == "Type them in":
        default = pd.DataFrame({"Date": [dt.date.today()], "HomeTeam": [None], "AwayTeam": [None],
                                "AvgH": [np.nan], "AvgD": [np.nan], "AvgA": [np.nan]})
        fx_input = st.data_editor(
            default, num_rows="dynamic", hide_index=True, width="stretch", key="fx_editor",
            column_config={
                "Date": st.column_config.DateColumn("Kickoff date", format="DD MMM YYYY", required=True),
                "HomeTeam": st.column_config.SelectboxColumn("Home team", options=TEAMS, required=True),
                "AwayTeam": st.column_config.SelectboxColumn("Away team", options=TEAMS, required=True),
                "AvgH": st.column_config.NumberColumn("Home odds (optional)", min_value=1.01, format="%.2f"),
                "AvgD": st.column_config.NumberColumn("Draw odds (optional)", min_value=1.01, format="%.2f"),
                "AvgA": st.column_config.NumberColumn("Away odds (optional)", min_value=1.01, format="%.2f"),
            })
        st.caption("Odds are optional and used only to compare the model with the market. Decimal odds, e.g. 2.10.")
    else:
        up = st.file_uploader("football-data.co.uk fixtures file (fixtures.csv)", type="csv", key="fx_upload")
        st.caption("Download https://www.football-data.co.uk/fixtures.csv on your phone and upload it here. "
                   "It needs Date, HomeTeam and AwayTeam columns; Avg odds columns are used if present.")
        if up is not None:
            try:
                raw = pd.read_csv(up, encoding="latin-1")
                if "Div" in raw.columns:
                    raw = raw[raw["Div"] == "E0"]
                fx_input = raw
                st.write(f"{len(raw)} Premier League fixture(s) found.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that file: {e}")

    if st.button("Predict these fixtures", type="primary", key="predict_btn", width="stretch"):
        if fx_input is None or len(fx_input) == 0:
            st.error("Add at least one fixture first.")
        else:
            try:
                tmp = fx_input.dropna(subset=["HomeTeam", "AwayTeam"])
                fx = standardise_fixtures(tmp)
                same = fx.HomeTeam == fx.AwayTeam
                if same.any():
                    st.error("A team cannot play itself. Fix those rows.")
                    fx = fx[~same]
                already = fx.set_index(KEY).index.isin(hist.set_index(KEY).index)
                if already.any():
                    st.info(f"{int(already.sum())} fixture(s) already have a result in the data and were skipped.")
                    fx = fx[~already]
                if fx.empty:
                    st.error("Nothing left to predict.")
                else:
                    bad = unknown_teams(hist, set(fx.HomeTeam) | set(fx.AwayTeam))
                    if bad:
                        st.warning(f"Not in the history (check spelling, or newly promoted): {', '.join(bad)}. "
                                   "They are treated as average, with low reliability.")
                    st.session_state.preds = run_predict(hist, fx)
                    st.session_state.fx_last = fx
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not make predictions: {e}")

    preds = st.session_state.get("preds")
    if preds is not None and len(preds):
        st.subheader("2. Predictions")
        view = pd.DataFrame({
            "Date": preds.Date.dt.strftime("%a %d %b"),
            "Match": preds.HomeTeam + " v " + preds.AwayTeam,
            "Home win": (preds.p_H * 100).round(1),
            "Draw": (preds.p_D * 100).round(1),
            "Away win": (preds.p_A * 100).round(1),
            "Expected goals": preds.xg_home.round(2).astype(str) + " - " + preds.xg_away.round(2).astype(str),
            "Caution": np.where(preds["warnings"] != "", "⚠️", ""),
        })
        st.dataframe(view, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(c + " %", format="%.1f")
                                    for c in ["Home win", "Draw", "Away win"]})
        st.caption("Probabilities add up to 100% per match. A 60% chance still fails 4 times in 10.")

        for _, r in preds.iterrows():
            with st.expander(f"{r.HomeTeam} v {r.AwayTeam} · {r.Date:%a %d %b}"):
                c1, c2 = st.columns(2)
                chart = pd.DataFrame({"Probability (%)": [r.p_H * 100, r.p_D * 100, r.p_A * 100]},
                                     index=[f"{r.HomeTeam} win", "Draw", f"{r.AwayTeam} win"])
                c1.bar_chart(chart)
                spread = max(abs(r.elo_H - r.dc_H), abs(r.elo_D - r.dc_D), abs(r.elo_A - r.dc_A)) * 100
                uncertain = bool(r["warnings"]) or spread > 10
                c2.markdown(
                    f"**Most likely result:** {['Home win', 'Draw', 'Away win'][int(np.argmax([r.p_H, r.p_D, r.p_A]))]} "
                    f"({pct(max(r.p_H, r.p_D, r.p_A))})\n\n"
                    f"**Expected goals:** {r.xg_home:.2f} – {r.xg_away:.2f}\n\n"
                    f"**Likeliest scores:** {r.top_scores}\n\n"
                    f"**Elo says** {pct(r.elo_H)} / {pct(r.elo_D)} / {pct(r.elo_A)} · "
                    f"**Poisson says** {pct(r.dc_H)} / {pct(r.dc_D)} / {pct(r.dc_A)} "
                    f"(they differ by up to {spread:.0f} points)\n\n"
                    f"**Confidence:** {'⚠️ lower than usual' if uncertain else 'standard for this model'}")
                if r["warnings"]:
                    st.warning(r["warnings"])
                if pd.notna(r.mkt_H):
                    cmp_ = pd.DataFrame({"Model %": [r.p_H * 100, r.p_D * 100, r.p_A * 100],
                                         "Market % (margin removed)": [r.mkt_H * 100, r.mkt_D * 100, r.mkt_A * 100]},
                                        index=["Home", "Draw", "Away"])
                    cmp_["Model minus market (pts)"] = cmp_.iloc[:, 0] - cmp_.iloc[:, 1]
                    st.dataframe(cmp_.round(1), width="stretch")
                    st.caption("The market sees team news this model cannot. Large gaps are more likely our blind "
                               "spot than a bargain: in testing the market was more accurate.")
                    if st.checkbox("Show exploratory value calculation (not a recommendation)",
                                   key=f"ev_{r.HomeTeam}_{r.AwayTeam}_{r.Date:%Y%m%d}"):
                        ev = pd.DataFrame({"Odds": [r.AvgH, r.AvgD, r.AvgA],
                                           "Expected value per 1 staked": [r.p_H * r.AvgH - 1, r.p_D * r.AvgD - 1,
                                                                           r.p_A * r.AvgA - 1]},
                                          index=["Home", "Draw", "Away"])
                        st.dataframe(ev.round(3), width="stretch")
                        st.caption("EV = probability × odds − 1. This model has not beaten the market in testing, so "
                                   "positive values are far more likely to be model error than genuine value.")
                st.caption(f"Over 2.5 goals {pct(r.p_over25)} · Both teams score {pct(r.p_btts)} "
                           "(unvalidated: in testing these were no better than base rates).")

        st.subheader("3. Freeze them in the track record")
        st.write("Logging locks in today's prediction before kickoff so performance can be judged honestly later. "
                 "The first prediction logged for a match is the one that counts; it cannot be replaced.")
        if st.button("Log these predictions", key="log_btn", width="stretch"):
            new_log, rep = append_predictions(st.session_state.log, preds)
            st.session_state.log = new_log
            msg = f"Logged {rep['added']} new prediction(s)."
            if rep["already_logged"]:
                msg += f" {rep['already_logged']} already logged earlier (kept as they were)."
            if rep["kickoff_passed"]:
                msg += f" {rep['kickoff_passed']} skipped because the kickoff date has passed."
            st.success(msg + " Download the log in the Tracking tab to keep it.")
        st.download_button("Download these predictions (CSV)", preds.drop(columns=["top_scores"]).to_csv(index=False),
                           "predictions.csv", "text/csv", key="dl_preds", width="stretch")

# ================================================================== CORNERS
with tab_corn:
    st.subheader("Total corners per match (experimental)")
    st.warning("**Weak signal.** In testing on 1,230 matches this model beat a league-average guess by only about "
               "0.03 corners per match: small, but statistically detectable. No bookmaker corner odds were available "
               "to test against, so it has **no proven betting value**. Any single over/under line stays close to a "
               "coin flip.", icon="⚠️")
    fxc = st.session_state.get("fx_last")
    LINES = config.CORNERS["lines"]
    if fxc is None or len(fxc) == 0:
        st.info("Make predictions in the Predict tab first. The same fixtures are used here.")
    else:
        try:
            cp = run_corners(hist, fxc)
        except Exception as e:  # noqa: BLE001
            cp = None
            st.error(f"Could not make corner predictions: {e}")
        if cp is not None and len(cp):
            view = pd.DataFrame({
                "Date": cp.Date.dt.strftime("%a %d %b"),
                "Match": cp.HomeTeam + " v " + cp.AwayTeam,
                "Expected total": cp.mu_total.round(1),
                "Home (rough split)": cp.mu_home.round(1),
                "Away (rough split)": cp.mu_away.round(1),
                "League average": cp.mu_clim.round(1),
                **{f"Over {L}": (cp[f"p_over_{L}"] * 100).round(0) for L in LINES}})
            st.dataframe(view, hide_index=True, width="stretch",
                         column_config={f"Over {L}": st.column_config.NumberColumn(f"Over {L} %", format="%.0f")
                                        for L in LINES})
            st.caption("“Over 9.5” means 10 or more corners in the match. The home/away split is a rough model "
                       "estimate and has not been tested separately.")
            if (cp["warnings"] != "").any():
                st.warning("  \n".join(f"{m}: {w}" for m, w in zip(view["Match"], cp["warnings"]) if w))

            pick = st.selectbox("See the full distribution for", list(range(len(cp))), key="corner_pick",
                                format_func=lambda i: f"{cp.HomeTeam.iloc[i]} v {cp.AwayTeam.iloc[i]}")
            r = cp.iloc[pick]
            xs = np.arange(0, 27)
            st.bar_chart(pd.DataFrame({"Probability (%)": cn.pmf(xs, r.mu_total, r.k) * 100}, index=xs))
            sd = float(np.sqrt(r.mu_total + r.mu_total ** 2 / r.k))
            st.caption(f"Expected {r.mu_total:.1f} corners, but the typical spread is about ±{sd:.1f}. "
                       "That is why one line is rarely a confident call.")

            if st.button("Log these corner predictions", key="clog_btn", width="stretch"):
                new_log, rep = ct.append_predictions(st.session_state.clog, cp)
                st.session_state.clog = new_log
                msg = f"Logged {rep['added']} new corner prediction(s)."
                if rep["already_logged"]:
                    msg += f" {rep['already_logged']} already logged earlier (kept as they were)."
                if rep["kickoff_passed"]:
                    msg += f" {rep['kickoff_passed']} skipped because the kickoff date has passed."
                st.success(msg)

    st.markdown("---")
    st.subheader("Corners track record")
    st.caption("Predictions are scored against the league-average baseline stored with each prediction. "
               "Matches need their corner counts entered (Update results tab) to be scored.")
    cup = st.file_uploader("Restore a saved corners log (CSV you downloaded earlier)", type="csv", key="clog_upload")
    if cup is not None and st.button("Use this corners log", key="use_clog"):
        try:
            cand = ct.load_log(cup)
            ok, msg = ct.verify_chain(cand)
            if ok:
                st.session_state.clog = cand
                st.success("Corners log restored. " + msg)
            else:
                st.error("This file failed the integrity check and was not loaded. " + msg)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the log: {e}")
    clog = st.session_state.clog
    if clog.empty:
        st.info("Nothing logged yet.")
    else:
        ok, msg = ct.verify_chain(clog)
        (st.success if ok else st.error)(("🔒 " if ok else "🚨 ") + msg)
        cres = ct.evaluate(clog, hist)
        if cres is None:
            st.write(f"{len(clog)} logged, none settled yet.")
        else:
            st.write(ct.reliability_note(cres["n"]))
            a1, a2, a3 = st.columns(3)
            a1.metric("Settled", cres["n"])
            a2.metric("Average miss (corners)", f"{cres['mae']:.2f}", f"{cres['mae'] - cres['clim_mae']:+.2f} vs league avg",
                      delta_color="inverse")
            a3.metric("Log score (higher is better)", f"{cres['logscore']:.3f}",
                      f"{cres['diff']:+.3f} vs league avg", delta_color="normal")
            st.write(f"Average actual total {cres['actual_mean']:.2f} vs average predicted {cres['pred_mean']:.2f}.")
            st.dataframe(cres["lines"].round(4), hide_index=True, width="stretch")
        st.download_button("Download the corners log (keep a copy!)", clog.to_csv(index=False),
                           "corners_log.csv", "text/csv", key="dl_clog", width="stretch")
        st.caption("The free app forgets the log when it restarts. Download it after each session and upload it above "
                   "next time (or commit it to GitHub as data/corners_log.csv).")

# ================================================================== UPDATE RESULTS
with tab_upd:
    st.subheader("Add new results")
    st.write("After each round, add the finished matches. The model refits on everything the next time you predict.")
    up = st.file_uploader("Option A: upload the current-season results file (football-data.co.uk E0.csv)",
                          type="csv", key="res_upload")
    if up is not None:
        try:
            new = pd.read_csv(up, encoding="latin-1")
            merged, rep = merge_results(hist, new)
            st.write(f"**{rep['added']}** new result(s) found · {rep['already_present']} already in the data · "
                     f"{rep['conflicts']} with a different score than the one stored (stored score is kept).")
            names = set(new.get("HomeTeam", [])) | set(new.get("AwayTeam", []))
            bad = unknown_teams(hist, {str(n).strip() for n in names if pd.notna(n)})
            if bad:
                st.warning(f"Team names not seen before (check spelling, or newly promoted): {', '.join(bad)}")
            if rep["added"] and st.button("Add these results", key="apply_upload", type="primary"):
                st.session_state.history = merged
                st.success(f"Added {rep['added']} result(s). Data now runs to {merged.Date.max():%d %b %Y}.")
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {e}")

    st.markdown("---")
    st.write("Option B: type results in by hand")
    dflt = pd.DataFrame({"Date": [dt.date.today()], "HomeTeam": [None], "AwayTeam": [None],
                         "FTHG": [None], "FTAG": [None], "HS": [np.nan], "AS": [np.nan],
                         "HST": [np.nan], "AST": [np.nan], "HC": [np.nan], "AC": [np.nan]})
    typed = st.data_editor(
        dflt, num_rows="dynamic", hide_index=True, width="stretch", key="res_editor",
        column_config={
            "Date": st.column_config.DateColumn("Match date", format="DD MMM YYYY", required=True),
            "HomeTeam": st.column_config.SelectboxColumn("Home team", options=TEAMS, required=True),
            "AwayTeam": st.column_config.SelectboxColumn("Away team", options=TEAMS, required=True),
            "FTHG": st.column_config.NumberColumn("Home goals", min_value=0, step=1, required=True),
            "FTAG": st.column_config.NumberColumn("Away goals", min_value=0, step=1, required=True),
            "HS": st.column_config.NumberColumn("Home shots", min_value=0, step=1),
            "AS": st.column_config.NumberColumn("Away shots", min_value=0, step=1),
            "HST": st.column_config.NumberColumn("Home on target", min_value=0, step=1),
            "AST": st.column_config.NumberColumn("Away on target", min_value=0, step=1),
            "HC": st.column_config.NumberColumn("Home corners", min_value=0, step=1),
            "AC": st.column_config.NumberColumn("Away corners", min_value=0, step=1),
        })
    st.caption("Shots and corners are optional. Without shots the model falls back to goals; matches without "
               "corner counts are simply skipped by the corners model, so enter them to keep it current.")
    if st.button("Add typed results", key="apply_typed"):
        try:
            tmp = typed.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
            if tmp.empty:
                st.error("Fill in at least one complete row.")
            else:
                merged, rep = merge_results(hist, tmp)
                st.session_state.history = merged
                st.success(f"Added {rep['added']} result(s); {rep['already_present']} were already there; "
                           f"{rep['conflicts']} score conflict(s) ignored.")
                st.rerun()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not add results: {e}")

    st.markdown("---")
    st.download_button("Download the updated results file", hist.to_csv(index=False), "epl_history_updated.csv",
                       "text/csv", key="dl_hist", width="stretch")
    st.caption("This app forgets changes when it restarts. To keep updates permanently, replace "
               "data/epl_history.csv in your GitHub repository with the downloaded file.")

# ================================================================== TRACKING
with tab_track:
    st.subheader("Prediction track record")
    st.caption("This is the only evidence that counts: predictions made before kickoff, scored afterwards, never edited.")
    upl = st.file_uploader("Restore a saved log (CSV you downloaded earlier)", type="csv", key="log_upload")
    if upl is not None and st.button("Use this log", key="use_log"):
        try:
            cand = load_log(upl)
            ok, msg = verify_chain(cand)
            if ok:
                st.session_state.log = cand
                st.success("Log restored. " + msg)
            else:
                st.error("This file failed the integrity check and was not loaded. " + msg)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the log: {e}")

    log = st.session_state.log
    if log.empty:
        st.info("Nothing logged yet. Make predictions in the Predict tab and press “Log these predictions”.")
    else:
        ok, msg = verify_chain(log)
        (st.success if ok else st.error)(("🔒 " if ok else "🚨 ") + msg)
        res = evaluate(log, hist)
        settled = 0 if res is None else res["n"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Logged", len(log))
        c2.metric("Settled", settled)
        c3.metric("Waiting for result", len(log) - settled)
        if res is not None:
            st.write(reliability_note(res["n"]))
            m1, m2, m3 = st.columns(3)
            m1.metric("Log loss (lower is better)", f"{res['logloss']:.4f}",
                      f"{res['logloss'] - res['naive_logloss']:+.4f} vs naive", delta_color="inverse")
            m2.metric("Brier score", f"{res['brier']:.4f}")
            m3.metric("Accuracy", f"{res['accuracy'] * 100:.1f}%", f"naive {res['naive_accuracy'] * 100:.1f}%",
                      delta_color="off")
            if res["n_market"] >= 1:
                st.write(f"Against the bookmaker market on the {res['n_market']} match(es) where you entered odds: "
                         f"market log loss {res['market_logloss']:.4f} vs model {res['model_logloss_on_market_rows']:.4f}.")
            st.dataframe(res["table"].assign(Date=lambda d: d.Date.dt.strftime("%d %b %Y"))
                         .round({"p_H": 3, "p_D": 3, "p_A": 3}), hide_index=True, width="stretch")
        st.download_button("Download the log (keep a copy!)", log.to_csv(index=False), "predictions_log.csv",
                           "text/csv", key="dl_log", width="stretch")
        st.caption("This app forgets the log when it restarts. Download it after each session, and upload it above "
                   "next time (or commit it to your GitHub repository as data/predictions_log.csv).")

# ================================================================== RATINGS
with tab_rate:
    st.subheader("Current team strengths")
    rt = run_ratings(hist)
    show = rt[["Team", "Elo", "Attack", "Defence", "Strength (attack - defence)", "Promoted"]].round(2)
    st.dataframe(show, hide_index=True, width="stretch")
    st.bar_chart(rt.set_index("Team")["Elo"])
    st.caption("Elo: higher is stronger. Attack: higher means scores more. Defence: LOWER means concedes less. "
               "Ratings are model estimates from results, not a verdict on quality.")

# ================================================================== ABOUT
with tab_about:
    st.subheader("What this model is")
    st.markdown(f"""
**Version {config.MODEL_VERSION}**, frozen 4 Nov 2026. Final probability = 50% **Elo rating** model + 50% **Poisson goals model**
with Dixon-Coles correction (team attack/defence strengths, home advantage, recent matches weighted more, newly promoted teams
start from how earlier promoted teams performed). Any change to the model becomes a new version.

**How it was tested.** Strictly chronological: 2022/23 used only for tuning; 2023/24 to 2 Nov 2026 (1,230 matches)
predicted match-by-match using only earlier results.
""")
    st.table(pd.DataFrame({
        "Model": ["Naive (league averages)", "v1 blend (older)", "v3 blend (this app)", "Bookmakers, odds a few hours earlier",
                  "Bookmakers, closing odds"],
        "Log loss (lower is better)": [1.0750, 0.9885, 0.9837, 0.9690, 0.9641],
        "Right outcome picked": ["42.9%", "53.2%", "52.9%", "53.7%", "54.5%"]}))
    st.markdown("""
**Honest limits**
- It is clearly less accurate than the bookmakers' closing odds (statistically solid gap). **No betting edge has been shown.**
- It knows nothing about injuries, suspensions, line-ups or tactics, which is the largest known gap.
- Newly promoted teams and early-season matches are more uncertain.
- Over 2.5 goals and Both Teams To Score are shown for interest only; in testing they were no better than base rates.
- **Corners (experimental tab):** total corners per match, model corners-v1.0 (frozen 8 Nov 2026). Team attack/defence
  effects with a negative-binomial total. On the same 1,230 test matches it beat a league-average guess by only a small
  margin (average miss 2.71 vs 2.74 corners; probability quality +0.016, statistically detectable). No bookmaker corner odds
  were available, so there is no market comparison and no proven betting value.
- **Yellow cards:** tested, no reliable improvement over the league average (referee effect did not hold up), so they are not shown.
- The exploratory promoted-team rule was designed after looking at test results; the live track record is the real check.
- Bookmaker odds include a margin (about 4.7% on 1X2); even a perfect copy of the market loses that on average.

**Data:** football-data.co.uk Premier League files (2021/22 to 2026/27), downloaded 4 Nov 2026. One known data error
(Newcastle v West Ham, Aug 2021, shots on target) is handled by falling back to goals.
""")

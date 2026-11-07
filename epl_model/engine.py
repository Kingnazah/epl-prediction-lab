"""Prediction engine: frozen v3 model = 50/50 blend of Elo and Dixon-Coles."""
import numpy as np
import pandas as pd

from . import config
from .data import standardise_fixtures, season_of, unknown_teams, promoted_in, teams_in_season
from .elo import EloModel
from .poisson import PoissonDC


def devig(odds: pd.DataFrame) -> np.ndarray:
    """Proportional removal of the bookmaker margin. Rows with missing odds return NaN."""
    inv = 1.0 / odds.values.astype(float)
    return inv / inv.sum(1, keepdims=True)


class Engine:
    def __init__(self, history: pd.DataFrame):
        self.history = history.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    def _fit(self, fixtures: pd.DataFrame):
        ref = fixtures.Date.min()
        season = season_of(ref)
        past = self.history[self.history.Date < ref]
        if past.empty:
            raise ValueError("No completed matches before the first fixture date.")
        extra = set(fixtures.HomeTeam) | set(fixtures.AwayTeam)
        elo = EloModel(**config.ELO).fit(past, season)
        dc = PoissonDC(max_goals=config.MAX_GOALS, **config.POISSON).fit(past, ref, season, extra_teams=extra)
        return elo, dc, past, season

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        fx = standardise_fixtures(fixtures)
        if fx.empty:
            raise ValueError("No valid fixtures supplied.")
        elo, dc, past, season = self._fit(fx)
        pe = elo.predict(fx)
        pdc = dc.predict(fx)
        w = config.BLEND_WEIGHT_ELO
        p = w * pe + (1 - w) * pdc["p1x2"]
        assert np.allclose(p.sum(1), 1.0, atol=1e-6) and not np.isnan(p).any()

        known = set(self.history.HomeTeam) | set(self.history.AwayTeam)
        played = past[past.Season == season]
        counts = pd.concat([played.HomeTeam, played.AwayTeam]).value_counts()
        prom = promoted_in(past, season, set(fx.HomeTeam) | set(fx.AwayTeam))

        flags = []
        for h, a in zip(fx.HomeTeam, fx.AwayTeam):
            f = []
            for role, t in (("home", h), ("away", a)):
                if t not in known:
                    f.append(f"{t}: not in history (treated as average, low reliability)")
                elif t in prom:
                    f.append(f"{t}: promoted/returning (extra uncertainty)")
                n = int(counts.get(t, 0))
                if n < 5 and t in known:
                    f.append(f"{t}: only {n} match(es) played this season")
            flags.append("; ".join(f))

        out = fx[["Date", "HomeTeam", "AwayTeam"]].copy()
        out["p_H"], out["p_D"], out["p_A"] = p[:, 0], p[:, 1], p[:, 2]
        out["elo_H"], out["elo_D"], out["elo_A"] = pe[:, 0], pe[:, 1], pe[:, 2]
        out["dc_H"], out["dc_D"], out["dc_A"] = pdc["p1x2"][:, 0], pdc["p1x2"][:, 1], pdc["p1x2"][:, 2]
        out["xg_home"], out["xg_away"] = pdc["xg_home"], pdc["xg_away"]
        out["p_over25"], out["p_btts"], out["top_scores"] = pdc["over25"], pdc["btts"], pdc["top_scores"]
        out["warnings"] = flags
        out["Season"] = season
        out["model_version"] = config.MODEL_VERSION
        # market odds at prediction time (optional, for comparison only)
        for c in ("AvgH", "AvgD", "AvgA"):
            out[c] = fx[c].values
        mk = devig(fx[["AvgH", "AvgD", "AvgA"]])
        out["mkt_H"], out["mkt_D"], out["mkt_A"] = mk[:, 0], mk[:, 1], mk[:, 2]
        out.attrs["info"] = {"n_train": dc.n_train, "rho": dc.rho, "cohort": dc.cohort, "cohort_n": dc.cohort_n}
        return out

    def ratings(self) -> pd.DataFrame:
        """Current Elo and Dixon-Coles strengths for teams in the latest season of the data."""
        last = self.history.Date.max()
        season = season_of(last)
        ref = last + pd.Timedelta(days=1)
        elo = EloModel(**config.ELO).fit(self.history, season)
        dc = PoissonDC(max_goals=config.MAX_GOALS, **config.POISSON).fit(self.history, ref, season)
        cur = teams_in_season(self.history, season)
        t = elo.ratings_table().merge(dc.strengths_table(), on="Team", how="left")
        t = t[t.Team.isin(cur)].copy()
        t["Promoted"] = t.Team.isin(dc.promoted)
        t["Strength (attack - defence)"] = t["Attack"] - t["Defence"]
        return t.sort_values("Elo", ascending=False).reset_index(drop=True)

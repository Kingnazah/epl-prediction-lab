"""
Corners model (experimental), frozen as config.CORNERS_VERSION.

For each side: log(mean corners) = intercept + home advantage (home side only) + team attack + opponent defence.
Fitted with time-decay weights and ridge shrinkage (quasi-Poisson). The match TOTAL is the sum of the two
means and is modelled as negative binomial: home and away corners are negatively correlated, so treating the
sides as independent would overstate the spread of the total.

Judged against a baseline: the time-decayed league-average total (mu_clim), stored with every prediction.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

from . import config
from .data import season_of, standardise_fixtures

K_POISSON = 1e6          # dispersion value that means "plain Poisson"


# ------------------------------------------------------------------ distribution helpers (array-safe)
def logpmf(y, mu, k):
    y, mu, k = (np.asarray(v, float) for v in (y, mu, k))
    kk = np.minimum(k, 1e5)
    return np.where(k >= 1e5, poisson.logpmf(y, mu), nbinom.logpmf(y, kk, kk / (kk + mu)))


def pmf(y, mu, k):
    return np.exp(logpmf(y, mu, k))


def over_prob(line, mu, k):
    """P(total corners > line), e.g. line 9.5 means 10 or more."""
    thr = np.floor(line)
    mu, k = np.asarray(mu, float), np.asarray(k, float)
    kk = np.minimum(k, 1e5)
    return np.where(k >= 1e5, poisson.sf(thr, mu), nbinom.sf(thr, kk, kk / (kk + mu)))


# ------------------------------------------------------------------ model
class CornersModel:
    def __init__(self, hl, ridge, k, clim_hl):
        self.hl, self.ridge, self.k, self.clim_hl = hl, ridge, k, clim_hl

    def fit(self, history: pd.DataFrame, ref_date, extra_teams=()):
        """Fit on completed matches strictly before ref_date (rows without corner counts are ignored)."""
        ref_date = pd.Timestamp(ref_date)
        tr = history[history.Date < ref_date].dropna(subset=["HC", "AC"])
        if len(tr) < 200:
            raise ValueError("Not enough completed matches with corner data to fit the corners model.")
        self.known = set(history.HomeTeam) | set(history.AwayTeam)
        self.teams = sorted(self.known | set(extra_teams))
        self.tid = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        h = tr.HomeTeam.map(self.tid).values
        a = tr.AwayTeam.map(self.tid).values
        yh, ya = tr.HC.values.astype(float), tr.AC.values.astype(float)
        days = tr.Date.values.astype("datetime64[D]").astype(int)
        ref_day = int(np.datetime64(ref_date, "D").astype(int))
        w = 0.5 ** ((ref_day - days) / self.hl)
        ridge = self.ridge

        def f(v):
            c, ha = v[0], v[1]
            att, dfn = v[2:2 + T], v[2 + T:]
            eh, ea = c + ha + att[h] + dfn[a], c + att[a] + dfn[h]
            lh, la = np.exp(eh), np.exp(ea)
            nll = (w * (lh - yh * eh + la - ya * ea)).sum() + ridge * ((att ** 2).sum() + (dfn ** 2).sum())
            rh, ra = w * (lh - yh), w * (la - ya)
            g = np.zeros(2 + 2 * T)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            g[2:2 + T] = np.bincount(h, rh, T) + np.bincount(a, ra, T) + 2 * ridge * att
            g[2 + T:] = np.bincount(a, rh, T) + np.bincount(h, ra, T) + 2 * ridge * dfn
            return nll, g

        x0 = np.zeros(2 + 2 * T)
        x0[0] = np.log((yh + ya).mean() / 2)
        th = minimize(f, x0, jac=True, method="L-BFGS-B").x
        self.c, self.home, self.att, self.dfn = th[0], th[1], th[2:2 + T], th[2 + T:]
        wc = 0.5 ** ((ref_day - days) / self.clim_hl)
        self.mu_clim = float((wc * (yh + ya)).sum() / wc.sum())
        self.n_train = len(tr)
        return self

    def _par(self, team):
        i = self.tid.get(team)
        return (0.0, 0.0) if i is None else (self.att[i], self.dfn[i])

    def predict(self, fx: pd.DataFrame) -> pd.DataFrame:
        mh, ma = [], []
        for h, a in zip(fx.HomeTeam, fx.AwayTeam):
            ah, dh = self._par(h)
            aa, da = self._par(a)
            mh.append(np.exp(self.c + self.home + ah + da))
            ma.append(np.exp(self.c + aa + dh))
        mh, ma = np.array(mh), np.array(ma)
        mu = mh + ma
        out = fx[["Date", "HomeTeam", "AwayTeam"]].copy().reset_index(drop=True)
        out["Season"] = [season_of(d) for d in out.Date]
        out["mu_home"], out["mu_away"], out["mu_total"] = mh, ma, mu
        out["k"] = float(self.k) if self.k else K_POISSON
        out["mu_clim"] = self.mu_clim
        for L in config.CORNERS["lines"]:
            out[f"p_over_{L}"] = over_prob(L, mu, out["k"].values)
        out["warnings"] = [
            "; ".join(f"{t}: not in history (treated as average)" for t in (h, a) if t not in self.known)
            for h, a in zip(fx.HomeTeam, fx.AwayTeam)]
        out["model_version"] = config.CORNERS_VERSION
        return out


class CornersEngine:
    def __init__(self, history: pd.DataFrame):
        self.history = history.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)

    def predict(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        fx = standardise_fixtures(fixtures)
        if fx.empty:
            raise ValueError("No valid fixtures supplied.")
        extra = set(fx.HomeTeam) | set(fx.AwayTeam)
        p = config.CORNERS
        m = CornersModel(p["hl"], p["ridge"], p["k"], p["clim_hl"]).fit(self.history, fx.Date.min(), extra)
        out = m.predict(fx)
        out.attrs["info"] = {"n_train": m.n_train}
        return out

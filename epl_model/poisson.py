"""
Time-decayed ridge Poisson model with Dixon-Coles correction.

Each team has an attack and a defence parameter (log scale) plus one league-wide home advantage.
Training target = alpha * actual goals + (1 - alpha) * shots-based goal proxy.
Newly promoted teams are shrunk toward the average of earlier promoted cohorts (untuned rule),
all other teams toward league average.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

from .data import seasons_in, teams_in_season, promoted_in


def cohort_prior(history: pd.DataFrame, target_season: str):
    """
    Average log(goals scored / league avg) and log(goals conceded / league avg) of promoted teams
    in COMPLETED seasons before target_season. Returns (att, def, n_team_seasons).
    """
    seasons = [s for s in seasons_in(history) if s < target_season]
    sa = sd = 0.0
    cnt = 0
    for j in range(1, len(seasons)):
        sj = seasons[j]
        sub = history[history.Season == sj]
        lg = (sub.FTHG.mean() + sub.FTAG.mean()) / 2
        for t in promoted_in(history, sj):
            scored = np.concatenate([sub.loc[sub.HomeTeam == t, "FTHG"], sub.loc[sub.AwayTeam == t, "FTAG"]]).mean()
            conceded = np.concatenate([sub.loc[sub.HomeTeam == t, "FTAG"], sub.loc[sub.AwayTeam == t, "FTHG"]]).mean()
            sa += np.log(scored / lg)
            sd += np.log(conceded / lg)
            cnt += 1
    return (sa / cnt, sd / cnt, cnt) if cnt else (0.0, 0.0, 0)


class PoissonDC:
    def __init__(self, hl, ridge, alpha, max_goals=11):
        self.hl, self.ridge, self.alpha, self.G = hl, ridge, alpha, max_goals

    # ------------------------------------------------------------------ fitting
    def fit(self, history: pd.DataFrame, ref_date, target_season: str, extra_teams=()):
        tr = history[history.Date < ref_date].reset_index(drop=True)
        self.teams = sorted(set(history.HomeTeam) | set(history.AwayTeam) | set(extra_teams))
        self.tid = {t: i for i, t in enumerate(self.teams)}
        T = len(self.teams)
        self.T = T
        hi = tr.HomeTeam.map(self.tid).values
        ai = tr.AwayTeam.map(self.tid).values
        HG, AG = tr.FTHG.values.astype(float), tr.FTAG.values.astype(float)
        SH, SA = tr["HS"].values.astype(float), tr["AS"].values.astype(float)
        SOTH, SOTA = tr["HST"].values.astype(float), tr["AST"].values.astype(float)
        bad = np.isnan(SH) | np.isnan(SA) | np.isnan(SOTH) | np.isnan(SOTA) | (SOTH > SH) | (SOTA > SA)
        days = tr.Date.values.astype("datetime64[D]").astype(int)
        ref_day = int(np.datetime64(pd.Timestamp(ref_date), "D").astype(int))
        w = np.ones(len(tr)) if self.hl is None else 0.5 ** ((ref_day - days) / self.hl)

        # shots-based goal proxy (coefficients estimated on training rows only)
        if self.alpha >= 1.0 or (~bad).sum() < 50:
            th, ta = HG, AG
        else:
            ok = ~bad
            X = np.vstack([np.column_stack([SOTH[ok], SH[ok] - SOTH[ok]]), np.column_stack([SOTA[ok], SA[ok] - SOTA[ok]])])
            c, *_ = np.linalg.lstsq(X, np.concatenate([HG[ok], AG[ok]]), rcond=None)
            c = np.clip(c, 0, None)
            with np.errstate(invalid="ignore"):
                xh = np.where(bad, HG, SOTH * c[0] + (SH - SOTH) * c[1])
                xa = np.where(bad, AG, SOTA * c[0] + (SA - SOTA) * c[1])
            th, ta = self.alpha * HG + (1 - self.alpha) * xh, self.alpha * AG + (1 - self.alpha) * xa

        # promoted-team prior means
        self.promoted = promoted_in(history, target_season, extra_teams)
        pa, pdf, self.cohort_n = cohort_prior(history, target_season)
        self.cohort = (pa, pdf)
        mu_att, mu_def = np.zeros(T), np.zeros(T)
        for t in self.promoted:
            if t in self.tid:
                mu_att[self.tid[t]], mu_def[self.tid[t]] = pa, pdf
        ridge = self.ridge

        def f(v):
            att, dfn, home = v[:T], v[T:2 * T], v[2 * T]
            eh, ea = home + att[hi] + dfn[ai], att[ai] + dfn[hi]
            lh, la = np.exp(eh), np.exp(ea)
            pa_, pd_ = att - mu_att, dfn - mu_def
            nll = (w * (lh - th * eh + la - ta * ea)).sum() + ridge * ((pa_ ** 2).sum() + (pd_ ** 2).sum())
            rh, ra = w * (lh - th), w * (la - ta)
            g_att = np.bincount(hi, rh, T) + np.bincount(ai, ra, T) + 2 * ridge * pa_
            g_def = np.bincount(ai, rh, T) + np.bincount(hi, ra, T) + 2 * ridge * pd_
            return nll, np.concatenate([g_att, g_def, [rh.sum()]])

        x0 = np.zeros(2 * T + 1)
        x0[2 * T] = 0.25
        theta = minimize(f, x0, jac=True, method="L-BFGS-B").x
        self.att, self.dfn, self.home = theta[:T], theta[T:2 * T], theta[2 * T]

        # Dixon-Coles rho on actual low scores
        lh = np.exp(self.home + self.att[hi] + self.dfn[ai])
        la = np.exp(self.att[ai] + self.dfn[hi])
        g0, g1 = HG, AG
        m00, m01, m10, m11 = (g0 == 0) & (g1 == 0), (g0 == 0) & (g1 == 1), (g0 == 1) & (g1 == 0), (g0 == 1) & (g1 == 1)

        def nll_rho(r):
            tau = np.ones(len(tr))
            tau[m00] = 1 - lh[m00] * la[m00] * r
            tau[m01] = 1 + lh[m01] * r
            tau[m10] = 1 + la[m10] * r
            tau[m11] = 1 - r
            return -(w * np.log(np.clip(tau, 1e-9, None))).sum()

        self.rho = float(minimize_scalar(nll_rho, bounds=(-0.3, 0.3), method="bounded").x)
        self.n_train = len(tr)
        return self

    # --------------------------------------------------------------- prediction
    def _param(self, team):
        i = self.tid.get(team)
        if i is None:
            return 0.0, 0.0
        return self.att[i], self.dfn[i]

    def predict(self, fixtures: pd.DataFrame):
        G = self.G
        lh, la = [], []
        for h, a in zip(fixtures.HomeTeam, fixtures.AwayTeam):
            ah, dh = self._param(h)
            aa, da = self._param(a)
            lh.append(np.exp(self.home + ah + da))
            la.append(np.exp(aa + dh))
        lh, la = np.array(lh), np.array(la)
        g = np.arange(G)
        M = poisson.pmf(g[None, :, None], lh[:, None, None]) * poisson.pmf(g[None, None, :], la[:, None, None])
        r = self.rho
        M[:, 0, 0] *= 1 - lh * la * r
        M[:, 0, 1] *= 1 + lh * r
        M[:, 1, 0] *= 1 + la * r
        M[:, 1, 1] *= 1 - r
        M /= M.sum((1, 2), keepdims=True)
        i, j = np.meshgrid(g, g, indexing="ij")
        p1x2 = np.column_stack([(M * (i > j)).sum((1, 2)), (M * (i == j)).sum((1, 2)), (M * (i < j)).sum((1, 2))])
        over25 = (M * ((i + j) >= 3)).sum((1, 2))
        btts = (M * ((i >= 1) & (j >= 1))).sum((1, 2))
        tops = []
        for k in range(len(lh)):
            flat = np.argsort(M[k].ravel())[::-1][:5]
            tops.append(", ".join(f"{x // G}-{x % G} ({M[k].ravel()[x]*100:.1f}%)" for x in flat))
        return {"p1x2": p1x2, "xg_home": lh, "xg_away": la, "over25": over25, "btts": btts, "top_scores": tops}

    def strengths_table(self) -> pd.DataFrame:
        return pd.DataFrame({"Team": self.teams, "Attack": self.att, "Defence": self.dfn})

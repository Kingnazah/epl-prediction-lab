"""Elo rating model with a logistic layer that turns a rating gap into H/D/A probabilities."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

_Y = {"H": 0, "D": 1, "A": 2}


class EloModel:
    def __init__(self, K, hfa, regress, newoff):
        self.K, self.hfa, self.regress, self.newoff = K, hfa, regress, newoff
        self.R = {}
        self.clf = None

    def _regress(self, R):
        mu = np.mean(list(R.values()))
        for t in R:
            R[t] = mu + (1 - self.regress) * (R[t] - mu)

    def fit(self, hist: pd.DataFrame, target_season: str):
        """
        Process completed matches in order, updating ratings after each one.
        The logistic layer is trained only on pre-match rating gaps from seasons BEFORE target_season.
        """
        h = hist.sort_values(["Date", "HomeTeam"]).reset_index(drop=True)
        seasons, HT, AT = h.Season.values, h.HomeTeam.values, h.AwayTeam.values
        HG, AG = h.FTHG.values, h.FTAG.values
        first = seasons[0]
        R = {t: 1500.0 for t in set(HT[seasons == first]) | set(AT[seasons == first])}
        diffs = np.empty(len(h))
        prev = None
        for i in range(len(h)):
            if seasons[i] != prev:
                if prev is not None:
                    self._regress(R)
                prev = seasons[i]
            for t in (HT[i], AT[i]):
                if t not in R:
                    R[t] = np.mean(list(R.values())) + self.newoff
            diffs[i] = R[HT[i]] - R[AT[i]]
            E = 1 / (1 + 10 ** (-(R[HT[i]] + self.hfa - R[AT[i]]) / 400))
            S = 1.0 if HG[i] > AG[i] else 0.5 if HG[i] == AG[i] else 0.0
            d = self.K * (np.log(abs(HG[i] - AG[i]) + 1) + 1) * (S - E)
            R[HT[i]] += d
            R[AT[i]] -= d
        if prev is not None and target_season != prev:      # new season with no results yet
            self._regress(R)
        self.R = R

        train = seasons < target_season
        if train.sum() < 100:                               # too little history: fall back to all rows
            train = np.ones(len(h), dtype=bool)
        y = h.FTR.map(_Y).values
        self.clf = LogisticRegression(C=10, max_iter=2000).fit(diffs[train, None], y[train])
        return self

    def rating(self, team):
        return self.R.get(team, np.mean(list(self.R.values())) + self.newoff)

    def predict(self, fixtures: pd.DataFrame) -> np.ndarray:
        d = np.array([self.rating(h) - self.rating(a) for h, a in zip(fixtures.HomeTeam, fixtures.AwayTeam)])
        return self.clf.predict_proba(d[:, None])

    def ratings_table(self) -> pd.DataFrame:
        return (pd.DataFrame({"Team": list(self.R), "Elo": list(self.R.values())})
                  .sort_values("Elo", ascending=False).reset_index(drop=True))

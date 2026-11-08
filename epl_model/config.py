"""
Frozen model configuration.

MODEL v3.0 was frozen on 4 Nov 2026. Changing ANY value below creates a NEW model
version: bump MODEL_VERSION so old and new predictions are never mixed in the log.
"""

MODEL_VERSION = "v3.0"

# Elo (tuned on the 2022/23 validation season only)
ELO = {"K": 25, "hfa": 80, "regress": 0.3, "newoff": -50}

# Time-decayed ridge Poisson with Dixon-Coles low-score correction
POISSON = {"hl": 120, "ridge": 3.0, "alpha": 0.5}   # half-life (days), ridge, weight on actual goals vs shot-based proxy

BLEND_WEIGHT_ELO = 0.5          # final 1X2 = 0.5 * Elo + 0.5 * Dixon-Coles
MAX_GOALS = 11                  # score grid 0..10

# ---------------------------------------------------------------------------------------------
# CORNERS model (experimental). Frozen 8 Nov 2026 from the validation-season tuning in
# run_counts_v1.py. Any change = a NEW version string, so old and new predictions are never mixed.
CORNERS_VERSION = "corners-v1.0"
CORNERS = {
    "hl": 120,                 # time-decay half-life (days)
    "ridge": 40.0,             # shrinkage of team attack/defence effects toward league average
    "k": 80,                   # negative-binomial dispersion of the match total (None-like = 1e6)
    "clim_hl": 365,            # half-life of the league-average baseline the model is judged against
    "lines": [8.5, 9.5, 10.5, 11.5, 12.5],
}

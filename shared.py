"""Shared utilities for World Cup predictor scripts."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

NAME_ALIASES = {
    "West Germany": "Germany",
    "East Germany": "Germany",
    "German DR": "Germany",
    "Soviet Union": "Russia",
    "USSR": "Russia",
    "Yugoslavia": "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Dutch East Indies": "Indonesia",
    "Dutch Guyana": "Suriname",
    "Republic of Ireland": "Ireland",
    "Burma": "Myanmar",
    "United Arab Republic": "Egypt",
    "Vietnam Republic": "Vietnam",
    "South Vietnam": "Vietnam",
    "Zaire": "DR Congo",
    "Ivory Coast": "Côte d'Ivoire",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Côte dIvoire": "Côte d'Ivoire",
    "Cote dIvoire": "Côte d'Ivoire",
    "South Korea": "Korea Republic",
    "Korea, South": "Korea Republic",
    "North Korea": "Korea DPR",
    "Korea, North": "Korea DPR",
    "Iran": "IR Iran",
    "United States": "USA",
    "United States of America": "USA",
    "UAE": "United Arab Emirates",
    "Cape Verde Islands": "Cape Verde",
    "Bosnia": "Bosnia and Herzegovina",
    "Korea South": "Korea Republic",
    "Korea North": "Korea DPR",
    "Curacao": "Curaçao",
}

COUNTRY_FEATURE_COLUMNS = [
    "gdp_per_capita",
    "population",
    "life_expectancy",
    "urbanization_pct",
    "health_spending_pct_gdp",
    "elo_rating",
    "fifa_rank",
    "football_power_index",
    "football_tradition",
]

# Match-level tradition features derived from ``football_tradition`` (see
# ``compute_match_features``). Used for ablation studies and feature analysis.
TRADITION_FEATURE_COLUMNS = [
    "football_tradition",
    "opp_football_tradition",
    "tradition_diff",
]

# Default gradient-boosted tree backend after backtest comparison (``xgb`` | ``lgbm``).
DEFAULT_GBT_MODEL = "lgbm"

# Hyperopt-tuned LightGBM params (100 TPE trials, chronological 80/20 holdout, log-loss).
LGBM_DEFAULT_PARAMS = {
    "n_estimators": 963,
    "max_depth": 5,
    "num_leaves": 57,
    "learning_rate": 0.013524545989201327,
    "min_child_samples": 96,
    "subsample": 0.7358169178252429,
    "colsample_bytree": 0.5051207842691607,
    "reg_alpha": 0.004018013300211406,
    "reg_lambda": 2.5790710672484894e-08,
    "min_split_gain": 0.3422570569407757,
}

# Betting-odds derived features merged onto matches that have bookmaker odds.
# Matches without odds get NaN for every column (XGBoost handles missing values
# natively, so no imputation is required). ``odds_overround`` is the bookmaker
# margin (sum of implied probabilities minus 1); a useful signal of how confident
# / liquid the market was on a given fixture.
ODDS_FEATURE_COLUMNS = [
    "implied_home_prob",
    "implied_draw_prob",
    "implied_away_prob",
    "odds_overround",
]

# Transfermarkt squad market-value features. Values span ~1M to ~1.8B EUR, so
# every level/diff is log-scaled (log1p) before it reaches the model. A team/year
# with no Transfermarkt coverage yields NaN for every column; XGBoost handles the
# missing split natively, so no imputation is performed for these columns.
SQUAD_VALUE_FEATURE_COLUMNS = [
    "squad_value",
    "opp_squad_value",
    "squad_value_diff",
    "squad_value_ratio",
]

# World Cup editions for which Transfermarkt squad values are available. Lookups
# use the most recent edition at or before the match year (a "year lag"), so a
# 2023 friendly is valued with the 2022 squad and a 2026 fixture with 2026 values.
SQUAD_VALUE_YEARS = (2014, 2018, 2022, 2026)

GROUP_2026_TEAMS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czech Republic"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Côte d'Ivoire", "Ecuador", "Curaçao"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "IR Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

WC_WINNERS = {
    1930: "Uruguay",
    1934: "Italy",
    1938: "Italy",
    1950: "Uruguay",
    1954: "Germany",
    1958: "Brazil",
    1962: "Brazil",
    1966: "England",
    1970: "Brazil",
    1974: "Germany",
    1978: "Argentina",
    1982: "Italy",
    1986: "Argentina",
    1990: "Germany",
    1994: "Brazil",
    1998: "France",
    2002: "Brazil",
    2006: "Italy",
    2010: "Spain",
    2014: "Germany",
    2018: "France",
    2022: "Argentina",
}

THIRD_SLOT_LABELS = {
    3: "M74",
    9: "M77",
    13: "M79",
    15: "M80",
    17: "M81",
    19: "M82",
    25: "M85",
    29: "M87",
}
THIRD_SLOT_ALLOWED_GROUPS = {
    3: set("ABCDFI"),  # 1E vs third from A/B/C/D/F/I
    9: set("CDFGH"),   # 1I vs third from C/D/F/G/H
    13: set("CEFHI"),  # 1A vs third from C/E/F/H/I
    15: set("EHIJKL"), # 1L vs third from E/H/I/J/K/L
    17: set("BEFIJ"),  # 1D vs third from B/E/F/I/J
    19: set("AEFHIJ"), # 1G vs third from A/E/F/H/I/J
    25: set("EFGIJ"),  # 1B vs third from E/F/G/I/J
    29: set("DEGIJL"), # 1K vs third from D/E/G/I/J/L
}
THIRD_SLOT_CANDIDATE_ORDER = {
    3: "DCABF",
    9: "HCDFG",
    13: "ECFHI",
    15: "EHIJK",
    17: "BEFIJ",
    19: "AHEIJ",
    25: "GEFIJ",
    29: "LDEIJ",
}
THIRD_SLOTS = [3, 9, 13, 15, 17, 19, 25, 29]
INITIAL_ELO = 1500
K_FACTOR = 32
STAGE_TO_INT = {"group": 0, "round_of_16": 1, "quarterfinal": 2, "semifinal": 3, "final": 4}
# Blend weight used by ``blend_probabilities`` for knockout matches. In this
# codebase alpha is the XGBoost weight, so 0.50 means a conservative 50/50
# XGBoost / Dixon-Coles blend after group-stage probabilities are unchanged.
KNOCKOUT_ALPHA = 0.50
H2H_YEARS_LIMIT = 15
H2H_HALF_LIFE_YEARS = 10
COUNTRY_FEATURE_STALE_YEARS = 8

# Rolling window length for form / goals (kept identical across every script so the
# features the model is trained on match the features used at prediction time).
FORM_WINDOW = 20

# Knockout stage codes used by the 2026 prediction/backtest pipeline. Historical
# training only ever sees stages 0..4 (see ``STAGE_TO_INT``), so the 2026 pipeline
# must collapse its richer R32/R16/QF/SF/Final scheme onto the same 0..4 range to
# avoid asking the model to extrapolate to a ``stage`` value it never observed.
WC2026_STAGE_TO_TRAIN = {
    "group": 0,
    "round_of_32": 1,
    "round_of_16": 1,
    "quarterfinal": 2,
    "semifinal": 3,
    "third_place": 3,
    "final": 4,
}


# FIFA Annex C third-place allocation table (495 combinations).
# Maps sorted combo string -> {group_letter: slot_index}.
# Slot indices: 3=M74(1E), 9=M77(1I), 13=M79(1A), 15=M80(1L),
#               17=M81(1D), 19=M82(1G), 25=M85(1B), 29=M87(1K)
# NOTE: "EFGHIJKL" entry only has 7 assignments (1A column is empty
# in FIFA's Annex C table for this combination).
FIFA_THIRD_PLACE_TABLE: Dict[str, Dict[str, int]] = {
    "ABCDEFGH": {"A": 19, "B": 17, "C": 3, "D": 29, "E": 15, "F": 9, "G": 25, "H": 13},
    "ABCDEFGI": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "I": 15},
    "ABCDEFGJ": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "J": 15},
    "ABCDEFGK": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "K": 15},
    "ABCDEFGL": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 15, "F": 9, "G": 25, "L": 29},
    "ABCDEFHI": {"A": 19, "B": 17, "C": 3, "D": 29, "E": 25, "F": 9, "H": 13, "I": 15},
    "ABCDEFHJ": {"A": 19, "B": 17, "C": 3, "D": 29, "E": 15, "F": 9, "H": 13, "J": 25},
    "ABCDEFHK": {"A": 19, "B": 17, "C": 3, "D": 29, "E": 25, "F": 9, "H": 13, "K": 15},
    "ABCDEFHL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 15, "F": 25, "H": 13, "L": 29},
    "ABCDEFIJ": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "I": 15, "J": 25},
    "ABCDEFIK": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "I": 29, "K": 15},
    "ABCDEFIL": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "I": 15, "L": 29},
    "ABCDEFJK": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "J": 25, "K": 15},
    "ABCDEFJL": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 15, "F": 9, "J": 25, "L": 29},
    "ABCDEFKL": {"A": 19, "B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "K": 15, "L": 29},
    "ABCDEGHI": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "I": 15},
    "ABCDEGHJ": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "J": 15},
    "ABCDEGHK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "K": 15},
    "ABCDEGHL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 15, "G": 25, "H": 13, "L": 29},
    "ABCDEGIJ": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 29, "J": 15},
    "ABCDEGIK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 29, "K": 15},
    "ABCDEGIL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 15, "L": 29},
    "ABCDEGJK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "J": 29, "K": 15},
    "ABCDEGJL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "J": 15, "L": 29},
    "ABCDEGKL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "K": 15, "L": 29},
    "ABCDEHIJ": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 29, "H": 13, "I": 15, "J": 25},
    "ABCDEHIK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 25, "H": 13, "I": 29, "K": 15},
    "ABCDEHIL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 25, "H": 13, "I": 15, "L": 29},
    "ABCDEHJK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 29, "H": 13, "J": 25, "K": 15},
    "ABCDEHJL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 15, "H": 13, "J": 25, "L": 29},
    "ABCDEHKL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 25, "H": 13, "K": 15, "L": 29},
    "ABCDEIJK": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "I": 29, "J": 25, "K": 15},
    "ABCDEIJL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "I": 15, "J": 25, "L": 29},
    "ABCDEIKL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "I": 25, "K": 15, "L": 29},
    "ABCDEJKL": {"A": 19, "B": 17, "C": 3, "D": 9, "E": 13, "J": 25, "K": 15, "L": 29},
    "ABCDFGHI": {"A": 19, "B": 17, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "I": 15},
    "ABCDFGHJ": {"A": 19, "B": 17, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "J": 15},
    "ABCDFGHK": {"A": 19, "B": 17, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "K": 15},
    "ABCDFGHL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "H": 15, "L": 29},
    "ABCDFGIJ": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 29, "J": 15},
    "ABCDFGIK": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 29, "K": 15},
    "ABCDFGIL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 15, "L": 29},
    "ABCDFGJK": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "J": 29, "K": 15},
    "ABCDFGJL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "J": 15, "L": 29},
    "ABCDFGKL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "K": 15, "L": 29},
    "ABCDFHIJ": {"A": 19, "B": 17, "C": 3, "D": 29, "F": 9, "H": 13, "I": 15, "J": 25},
    "ABCDFHIK": {"A": 19, "B": 17, "C": 3, "D": 9, "F": 25, "H": 13, "I": 29, "K": 15},
    "ABCDFHIL": {"A": 19, "B": 17, "C": 3, "D": 9, "F": 25, "H": 13, "I": 15, "L": 29},
    "ABCDFHJK": {"A": 19, "B": 17, "C": 3, "D": 29, "F": 9, "H": 13, "J": 25, "K": 15},
    "ABCDFHJL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "H": 15, "J": 25, "L": 29},
    "ABCDFHKL": {"A": 19, "B": 17, "C": 3, "D": 9, "F": 25, "H": 13, "K": 15, "L": 29},
    "ABCDFIJK": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "I": 29, "J": 25, "K": 15},
    "ABCDFIJL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "I": 15, "J": 25, "L": 29},
    "ABCDFIKL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "I": 25, "K": 15, "L": 29},
    "ABCDFJKL": {"A": 19, "B": 17, "C": 13, "D": 3, "F": 9, "J": 25, "K": 15, "L": 29},
    "ABCDGHIJ": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 29, "J": 15},
    "ABCDGHIK": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 29, "K": 15},
    "ABCDGHIL": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 15, "L": 29},
    "ABCDGHJK": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "J": 29, "K": 15},
    "ABCDGHJL": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "J": 15, "L": 29},
    "ABCDGHKL": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "K": 15, "L": 29},
    "ABCDGIJK": {"A": 19, "B": 17, "C": 13, "D": 3, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABCDGIJL": {"A": 19, "B": 17, "C": 13, "D": 3, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABCDGIKL": {"A": 19, "B": 17, "C": 3, "D": 9, "G": 25, "I": 13, "K": 15, "L": 29},
    "ABCDGJKL": {"A": 19, "B": 17, "C": 13, "D": 3, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABCDHIJK": {"A": 19, "B": 17, "C": 3, "D": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABCDHIJL": {"A": 19, "B": 17, "C": 3, "D": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABCDHIKL": {"A": 19, "B": 17, "C": 3, "D": 9, "H": 13, "I": 25, "K": 15, "L": 29},
    "ABCDHJKL": {"A": 19, "B": 17, "C": 3, "D": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABCDIJKL": {"A": 19, "B": 17, "C": 3, "D": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABCEFGHI": {"A": 19, "B": 17, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15},
    "ABCEFGHJ": {"A": 19, "B": 17, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 15},
    "ABCEFGHK": {"A": 19, "B": 17, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "K": 15},
    "ABCEFGHL": {"A": 19, "B": 17, "C": 3, "E": 15, "F": 9, "G": 25, "H": 13, "L": 29},
    "ABCEFGIJ": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 15},
    "ABCEFGIK": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 29, "K": 15},
    "ABCEFGIL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 15, "L": 29},
    "ABCEFGJK": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "J": 29, "K": 15},
    "ABCEFGJL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "J": 15, "L": 29},
    "ABCEFGKL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "K": 15, "L": 29},
    "ABCEFHIJ": {"A": 19, "B": 17, "C": 3, "E": 29, "F": 9, "H": 13, "I": 15, "J": 25},
    "ABCEFHIK": {"A": 19, "B": 17, "C": 3, "E": 25, "F": 9, "H": 13, "I": 29, "K": 15},
    "ABCEFHIL": {"A": 19, "B": 17, "C": 3, "E": 25, "F": 9, "H": 13, "I": 15, "L": 29},
    "ABCEFHJK": {"A": 19, "B": 17, "C": 3, "E": 29, "F": 9, "H": 13, "J": 25, "K": 15},
    "ABCEFHJL": {"A": 19, "B": 17, "C": 3, "E": 15, "F": 9, "H": 13, "J": 25, "L": 29},
    "ABCEFHKL": {"A": 19, "B": 17, "C": 3, "E": 25, "F": 9, "H": 13, "K": 15, "L": 29},
    "ABCEFIJK": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "I": 29, "J": 25, "K": 15},
    "ABCEFIJL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "I": 15, "J": 25, "L": 29},
    "ABCEFIKL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "I": 25, "K": 15, "L": 29},
    "ABCEFJKL": {"A": 19, "B": 17, "C": 3, "E": 13, "F": 9, "J": 25, "K": 15, "L": 29},
    "ABCEGHIJ": {"A": 19, "B": 17, "C": 3, "E": 29, "G": 9, "H": 13, "I": 15, "J": 25},
    "ABCEGHIK": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 25, "H": 9, "I": 29, "K": 15},
    "ABCEGHIL": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 25, "H": 9, "I": 15, "L": 29},
    "ABCEGHJK": {"A": 19, "B": 17, "C": 3, "E": 29, "G": 9, "H": 13, "J": 25, "K": 15},
    "ABCEGHJL": {"A": 19, "B": 17, "C": 3, "E": 15, "G": 9, "H": 13, "J": 25, "L": 29},
    "ABCEGHKL": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 25, "H": 9, "K": 15, "L": 29},
    "ABCEGIJK": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABCEGIJL": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABCEGIKL": {"A": 3, "B": 17, "C": 9, "E": 13, "G": 25, "I": 19, "K": 15, "L": 29},
    "ABCEGJKL": {"A": 19, "B": 17, "C": 3, "E": 13, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABCEHIJK": {"A": 19, "B": 17, "C": 3, "E": 13, "H": 9, "I": 29, "J": 25, "K": 15},
    "ABCEHIJL": {"A": 19, "B": 17, "C": 3, "E": 13, "H": 9, "I": 15, "J": 25, "L": 29},
    "ABCEHIKL": {"A": 19, "B": 17, "C": 3, "E": 13, "H": 9, "I": 25, "K": 15, "L": 29},
    "ABCEHJKL": {"A": 19, "B": 17, "C": 3, "E": 13, "H": 9, "J": 25, "K": 15, "L": 29},
    "ABCEIJKL": {"A": 3, "B": 17, "C": 9, "E": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABCFGHIJ": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 15},
    "ABCFGHIK": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 29, "K": 15},
    "ABCFGHIL": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 15, "L": 29},
    "ABCFGHJK": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "J": 29, "K": 15},
    "ABCFGHJL": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "J": 15, "L": 29},
    "ABCFGHKL": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "K": 15, "L": 29},
    "ABCFGIJK": {"A": 19, "B": 17, "C": 13, "F": 3, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABCFGIJL": {"A": 19, "B": 17, "C": 13, "F": 3, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABCFGIKL": {"A": 19, "B": 17, "C": 3, "F": 9, "G": 25, "I": 13, "K": 15, "L": 29},
    "ABCFGJKL": {"A": 19, "B": 17, "C": 13, "F": 3, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABCFHIJK": {"A": 19, "B": 17, "C": 3, "F": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABCFHIJL": {"A": 19, "B": 17, "C": 3, "F": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABCFHIKL": {"A": 19, "B": 17, "C": 3, "F": 9, "H": 13, "I": 25, "K": 15, "L": 29},
    "ABCFHJKL": {"A": 19, "B": 17, "C": 3, "F": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABCFIJKL": {"A": 19, "B": 17, "C": 3, "F": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABCGHIJK": {"A": 19, "B": 17, "C": 3, "G": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABCGHIJL": {"A": 19, "B": 17, "C": 3, "G": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABCGHIKL": {"A": 19, "B": 17, "C": 3, "G": 25, "H": 9, "I": 13, "K": 15, "L": 29},
    "ABCGHJKL": {"A": 19, "B": 17, "C": 3, "G": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABCGIJKL": {"A": 19, "B": 17, "C": 3, "G": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABCHIJKL": {"A": 19, "B": 17, "C": 3, "H": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABDEFGHI": {"A": 19, "B": 17, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15},
    "ABDEFGHJ": {"A": 19, "B": 17, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 15},
    "ABDEFGHK": {"A": 19, "B": 17, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "K": 15},
    "ABDEFGHL": {"A": 19, "B": 17, "D": 3, "E": 15, "F": 9, "G": 25, "H": 13, "L": 29},
    "ABDEFGIJ": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 15},
    "ABDEFGIK": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 29, "K": 15},
    "ABDEFGIL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 15, "L": 29},
    "ABDEFGJK": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "J": 29, "K": 15},
    "ABDEFGJL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "J": 15, "L": 29},
    "ABDEFGKL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "K": 15, "L": 29},
    "ABDEFHIJ": {"A": 19, "B": 17, "D": 3, "E": 29, "F": 9, "H": 13, "I": 15, "J": 25},
    "ABDEFHIK": {"A": 19, "B": 17, "D": 3, "E": 25, "F": 9, "H": 13, "I": 29, "K": 15},
    "ABDEFHIL": {"A": 19, "B": 17, "D": 3, "E": 25, "F": 9, "H": 13, "I": 15, "L": 29},
    "ABDEFHJK": {"A": 19, "B": 17, "D": 3, "E": 29, "F": 9, "H": 13, "J": 25, "K": 15},
    "ABDEFHJL": {"A": 19, "B": 17, "D": 3, "E": 15, "F": 9, "H": 13, "J": 25, "L": 29},
    "ABDEFHKL": {"A": 19, "B": 17, "D": 3, "E": 25, "F": 9, "H": 13, "K": 15, "L": 29},
    "ABDEFIJK": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "I": 29, "J": 25, "K": 15},
    "ABDEFIJL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "I": 15, "J": 25, "L": 29},
    "ABDEFIKL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "I": 25, "K": 15, "L": 29},
    "ABDEFJKL": {"A": 19, "B": 17, "D": 3, "E": 13, "F": 9, "J": 25, "K": 15, "L": 29},
    "ABDEGHIJ": {"A": 19, "B": 17, "D": 3, "E": 29, "G": 9, "H": 13, "I": 15, "J": 25},
    "ABDEGHIK": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 25, "H": 9, "I": 29, "K": 15},
    "ABDEGHIL": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 25, "H": 9, "I": 15, "L": 29},
    "ABDEGHJK": {"A": 19, "B": 17, "D": 3, "E": 29, "G": 9, "H": 13, "J": 25, "K": 15},
    "ABDEGHJL": {"A": 19, "B": 17, "D": 3, "E": 15, "G": 9, "H": 13, "J": 25, "L": 29},
    "ABDEGHKL": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 25, "H": 9, "K": 15, "L": 29},
    "ABDEGIJK": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABDEGIJL": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABDEGIKL": {"A": 3, "B": 17, "D": 9, "E": 13, "G": 25, "I": 19, "K": 15, "L": 29},
    "ABDEGJKL": {"A": 19, "B": 17, "D": 3, "E": 13, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABDEHIJK": {"A": 19, "B": 17, "D": 3, "E": 13, "H": 9, "I": 29, "J": 25, "K": 15},
    "ABDEHIJL": {"A": 19, "B": 17, "D": 3, "E": 13, "H": 9, "I": 15, "J": 25, "L": 29},
    "ABDEHIKL": {"A": 19, "B": 17, "D": 3, "E": 13, "H": 9, "I": 25, "K": 15, "L": 29},
    "ABDEHJKL": {"A": 19, "B": 17, "D": 3, "E": 13, "H": 9, "J": 25, "K": 15, "L": 29},
    "ABDEIJKL": {"A": 3, "B": 17, "D": 9, "E": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABDFGHIJ": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 15},
    "ABDFGHIK": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 29, "K": 15},
    "ABDFGHIL": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 15, "L": 29},
    "ABDFGHJK": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "J": 29, "K": 15},
    "ABDFGHJL": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "J": 15, "L": 29},
    "ABDFGHKL": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "K": 15, "L": 29},
    "ABDFGIJK": {"A": 19, "B": 17, "D": 3, "F": 13, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABDFGIJL": {"A": 19, "B": 17, "D": 3, "F": 13, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABDFGIKL": {"A": 19, "B": 17, "D": 3, "F": 9, "G": 25, "I": 13, "K": 15, "L": 29},
    "ABDFGJKL": {"A": 19, "B": 17, "D": 3, "F": 13, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABDFHIJK": {"A": 19, "B": 17, "D": 3, "F": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABDFHIJL": {"A": 19, "B": 17, "D": 3, "F": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABDFHIKL": {"A": 19, "B": 17, "D": 3, "F": 9, "H": 13, "I": 25, "K": 15, "L": 29},
    "ABDFHJKL": {"A": 19, "B": 17, "D": 3, "F": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABDFIJKL": {"A": 19, "B": 17, "D": 3, "F": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABDGHIJK": {"A": 19, "B": 17, "D": 3, "G": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABDGHIJL": {"A": 19, "B": 17, "D": 3, "G": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABDGHIKL": {"A": 19, "B": 17, "D": 3, "G": 25, "H": 9, "I": 13, "K": 15, "L": 29},
    "ABDGHJKL": {"A": 19, "B": 17, "D": 3, "G": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABDGIJKL": {"A": 19, "B": 17, "D": 3, "G": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABDHIJKL": {"A": 19, "B": 17, "D": 3, "H": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABEFGHIJ": {"A": 19, "B": 17, "E": 29, "F": 3, "G": 9, "H": 13, "I": 15, "J": 25},
    "ABEFGHIK": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 25, "H": 9, "I": 29, "K": 15},
    "ABEFGHIL": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 25, "H": 9, "I": 15, "L": 29},
    "ABEFGHJK": {"A": 19, "B": 17, "E": 29, "F": 3, "G": 9, "H": 13, "J": 25, "K": 15},
    "ABEFGHJL": {"A": 19, "B": 17, "E": 15, "F": 3, "G": 9, "H": 13, "J": 25, "L": 29},
    "ABEFGHKL": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 25, "H": 9, "K": 15, "L": 29},
    "ABEFGIJK": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 9, "I": 29, "J": 25, "K": 15},
    "ABEFGIJL": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 9, "I": 15, "J": 25, "L": 29},
    "ABEFGIKL": {"A": 3, "B": 17, "E": 13, "F": 9, "G": 25, "I": 19, "K": 15, "L": 29},
    "ABEFGJKL": {"A": 19, "B": 17, "E": 13, "F": 3, "G": 9, "J": 25, "K": 15, "L": 29},
    "ABEFHIJK": {"A": 19, "B": 17, "E": 13, "F": 3, "H": 9, "I": 29, "J": 25, "K": 15},
    "ABEFHIJL": {"A": 19, "B": 17, "E": 13, "F": 3, "H": 9, "I": 15, "J": 25, "L": 29},
    "ABEFHIKL": {"A": 19, "B": 17, "E": 13, "F": 3, "H": 9, "I": 25, "K": 15, "L": 29},
    "ABEFHJKL": {"A": 19, "B": 17, "E": 13, "F": 3, "H": 9, "J": 25, "K": 15, "L": 29},
    "ABEFIJKL": {"A": 3, "B": 17, "E": 13, "F": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABEGHIJK": {"A": 3, "B": 17, "E": 13, "G": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "ABEGHIJL": {"A": 3, "B": 17, "E": 13, "G": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "ABEGHIKL": {"A": 3, "B": 17, "E": 13, "G": 25, "H": 9, "I": 19, "K": 15, "L": 29},
    "ABEGHJKL": {"A": 3, "B": 17, "E": 13, "G": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "ABEGIJKL": {"A": 3, "B": 17, "E": 13, "G": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABEHIJKL": {"A": 3, "B": 17, "E": 13, "H": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABFGHIJK": {"A": 19, "B": 17, "F": 3, "G": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ABFGHIJL": {"A": 19, "B": 17, "F": 3, "G": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ABFGHIKL": {"A": 3, "B": 17, "F": 9, "G": 25, "H": 13, "I": 19, "K": 15, "L": 29},
    "ABFGHJKL": {"A": 19, "B": 17, "F": 3, "G": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ABFGIJKL": {"A": 19, "B": 17, "F": 3, "G": 9, "I": 13, "J": 25, "K": 15, "L": 29},
    "ABFHIJKL": {"A": 3, "B": 17, "F": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "ABGHIJKL": {"A": 3, "B": 17, "G": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "ACDEFGHI": {"A": 19, "C": 3, "D": 29, "E": 17, "F": 9, "G": 25, "H": 13, "I": 15},
    "ACDEFGHJ": {"A": 19, "C": 3, "D": 29, "E": 15, "F": 9, "G": 25, "H": 13, "J": 17},
    "ACDEFGHK": {"A": 19, "C": 3, "D": 29, "E": 17, "F": 9, "G": 25, "H": 13, "K": 15},
    "ACDEFGHL": {"A": 19, "C": 3, "D": 9, "E": 15, "F": 17, "G": 25, "H": 13, "L": 29},
    "ACDEFGIJ": {"A": 19, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "I": 15, "J": 17},
    "ACDEFGIK": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "I": 29, "K": 15},
    "ACDEFGIL": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "I": 15, "L": 29},
    "ACDEFGJK": {"A": 19, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "J": 17, "K": 15},
    "ACDEFGJL": {"A": 19, "C": 13, "D": 3, "E": 15, "F": 9, "G": 25, "J": 17, "L": 29},
    "ACDEFGKL": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "K": 15, "L": 29},
    "ACDEFHIJ": {"A": 19, "C": 3, "D": 29, "E": 17, "F": 9, "H": 13, "I": 15, "J": 25},
    "ACDEFHIK": {"A": 19, "C": 3, "D": 9, "E": 25, "F": 17, "H": 13, "I": 29, "K": 15},
    "ACDEFHIL": {"A": 19, "C": 3, "D": 9, "E": 25, "F": 17, "H": 13, "I": 15, "L": 29},
    "ACDEFHJK": {"A": 19, "C": 3, "D": 29, "E": 17, "F": 9, "H": 13, "J": 25, "K": 15},
    "ACDEFHJL": {"A": 19, "C": 3, "D": 9, "E": 15, "F": 17, "H": 13, "J": 25, "L": 29},
    "ACDEFHKL": {"A": 19, "C": 3, "D": 9, "E": 25, "F": 17, "H": 13, "K": 15, "L": 29},
    "ACDEFIJK": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "I": 29, "J": 25, "K": 15},
    "ACDEFIJL": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "I": 15, "J": 25, "L": 29},
    "ACDEFIKL": {"A": 19, "C": 13, "D": 3, "E": 25, "F": 9, "I": 17, "K": 15, "L": 29},
    "ACDEFJKL": {"A": 19, "C": 13, "D": 3, "E": 17, "F": 9, "J": 25, "K": 15, "L": 29},
    "ACDEGHIJ": {"A": 19, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "I": 15, "J": 17},
    "ACDEGHIK": {"A": 19, "C": 3, "D": 9, "E": 17, "G": 25, "H": 13, "I": 29, "K": 15},
    "ACDEGHIL": {"A": 19, "C": 3, "D": 9, "E": 17, "G": 25, "H": 13, "I": 15, "L": 29},
    "ACDEGHJK": {"A": 19, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "J": 17, "K": 15},
    "ACDEGHJL": {"A": 19, "C": 3, "D": 9, "E": 15, "G": 25, "H": 13, "J": 17, "L": 29},
    "ACDEGHKL": {"A": 19, "C": 3, "D": 9, "E": 17, "G": 25, "H": 13, "K": 15, "L": 29},
    "ACDEGIJK": {"A": 19, "C": 3, "D": 9, "E": 13, "G": 25, "I": 29, "J": 17, "K": 15},
    "ACDEGIJL": {"A": 19, "C": 3, "D": 9, "E": 13, "G": 25, "I": 15, "J": 17, "L": 29},
    "ACDEGIKL": {"A": 19, "C": 3, "D": 9, "E": 13, "G": 25, "I": 17, "K": 15, "L": 29},
    "ACDEGJKL": {"A": 19, "C": 3, "D": 9, "E": 13, "G": 25, "J": 17, "K": 15, "L": 29},
    "ACDEHIJK": {"A": 19, "C": 3, "D": 9, "E": 17, "H": 13, "I": 29, "J": 25, "K": 15},
    "ACDEHIJL": {"A": 19, "C": 3, "D": 9, "E": 17, "H": 13, "I": 15, "J": 25, "L": 29},
    "ACDEHIKL": {"A": 19, "C": 3, "D": 9, "E": 25, "H": 13, "I": 17, "K": 15, "L": 29},
    "ACDEHJKL": {"A": 19, "C": 3, "D": 9, "E": 17, "H": 13, "J": 25, "K": 15, "L": 29},
    "ACDEIJKL": {"A": 19, "C": 3, "D": 9, "E": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACDFGHIJ": {"A": 19, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 17},
    "ACDFGHIK": {"A": 19, "C": 3, "D": 9, "F": 17, "G": 25, "H": 13, "I": 29, "K": 15},
    "ACDFGHIL": {"A": 19, "C": 3, "D": 9, "F": 17, "G": 25, "H": 13, "I": 15, "L": 29},
    "ACDFGHJK": {"A": 19, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "J": 17, "K": 15},
    "ACDFGHJL": {"A": 19, "C": 13, "D": 3, "F": 9, "G": 25, "H": 15, "J": 17, "L": 29},
    "ACDFGHKL": {"A": 19, "C": 3, "D": 9, "F": 17, "G": 25, "H": 13, "K": 15, "L": 29},
    "ACDFGIJK": {"A": 19, "C": 13, "D": 3, "F": 9, "G": 25, "I": 29, "J": 17, "K": 15},
    "ACDFGIJL": {"A": 19, "C": 13, "D": 3, "F": 9, "G": 25, "I": 15, "J": 17, "L": 29},
    "ACDFGIKL": {"A": 19, "C": 13, "D": 3, "F": 9, "G": 25, "I": 17, "K": 15, "L": 29},
    "ACDFGJKL": {"A": 19, "C": 13, "D": 3, "F": 9, "G": 25, "J": 17, "K": 15, "L": 29},
    "ACDFHIJK": {"A": 19, "C": 3, "D": 9, "F": 17, "H": 13, "I": 29, "J": 25, "K": 15},
    "ACDFHIJL": {"A": 19, "C": 3, "D": 9, "F": 17, "H": 13, "I": 15, "J": 25, "L": 29},
    "ACDFHIKL": {"A": 19, "C": 3, "D": 9, "F": 25, "H": 13, "I": 17, "K": 15, "L": 29},
    "ACDFHJKL": {"A": 19, "C": 3, "D": 9, "F": 17, "H": 13, "J": 25, "K": 15, "L": 29},
    "ACDFIJKL": {"A": 19, "C": 13, "D": 3, "F": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACDGHIJK": {"A": 19, "C": 3, "D": 9, "G": 25, "H": 13, "I": 29, "J": 17, "K": 15},
    "ACDGHIJL": {"A": 19, "C": 3, "D": 9, "G": 25, "H": 13, "I": 15, "J": 17, "L": 29},
    "ACDGHIKL": {"A": 19, "C": 3, "D": 9, "G": 25, "H": 13, "I": 17, "K": 15, "L": 29},
    "ACDGHJKL": {"A": 19, "C": 3, "D": 9, "G": 25, "H": 13, "J": 17, "K": 15, "L": 29},
    "ACDGIJKL": {"A": 19, "C": 3, "D": 9, "G": 25, "I": 13, "J": 17, "K": 15, "L": 29},
    "ACDHIJKL": {"A": 19, "C": 3, "D": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACEFGHIJ": {"A": 19, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 17},
    "ACEFGHIK": {"A": 19, "C": 3, "E": 17, "F": 9, "G": 25, "H": 13, "I": 29, "K": 15},
    "ACEFGHIL": {"A": 19, "C": 3, "E": 17, "F": 9, "G": 25, "H": 13, "I": 15, "L": 29},
    "ACEFGHJK": {"A": 19, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 17, "K": 15},
    "ACEFGHJL": {"A": 19, "C": 3, "E": 15, "F": 9, "G": 25, "H": 13, "J": 17, "L": 29},
    "ACEFGHKL": {"A": 19, "C": 3, "E": 17, "F": 9, "G": 25, "H": 13, "K": 15, "L": 29},
    "ACEFGIJK": {"A": 19, "C": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 17, "K": 15},
    "ACEFGIJL": {"A": 19, "C": 3, "E": 13, "F": 9, "G": 25, "I": 15, "J": 17, "L": 29},
    "ACEFGIKL": {"A": 19, "C": 3, "E": 13, "F": 9, "G": 25, "I": 17, "K": 15, "L": 29},
    "ACEFGJKL": {"A": 19, "C": 3, "E": 13, "F": 9, "G": 25, "J": 17, "K": 15, "L": 29},
    "ACEFHIJK": {"A": 19, "C": 3, "E": 17, "F": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ACEFHIJL": {"A": 19, "C": 3, "E": 17, "F": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ACEFHIKL": {"A": 19, "C": 3, "E": 25, "F": 9, "H": 13, "I": 17, "K": 15, "L": 29},
    "ACEFHJKL": {"A": 19, "C": 3, "E": 17, "F": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ACEFIJKL": {"A": 19, "C": 3, "E": 13, "F": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACEGHIJK": {"A": 19, "C": 3, "E": 13, "G": 25, "H": 9, "I": 29, "J": 17, "K": 15},
    "ACEGHIJL": {"A": 19, "C": 3, "E": 13, "G": 25, "H": 9, "I": 15, "J": 17, "L": 29},
    "ACEGHIKL": {"A": 19, "C": 3, "E": 13, "G": 25, "H": 9, "I": 17, "K": 15, "L": 29},
    "ACEGHJKL": {"A": 19, "C": 3, "E": 13, "G": 25, "H": 9, "J": 17, "K": 15, "L": 29},
    "ACEGIJKL": {"A": 19, "C": 3, "E": 13, "G": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACEHIJKL": {"A": 19, "C": 3, "E": 13, "H": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACFGHIJK": {"A": 19, "C": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 17, "K": 15},
    "ACFGHIJL": {"A": 19, "C": 3, "F": 9, "G": 25, "H": 13, "I": 15, "J": 17, "L": 29},
    "ACFGHIKL": {"A": 19, "C": 3, "F": 9, "G": 25, "H": 13, "I": 17, "K": 15, "L": 29},
    "ACFGHJKL": {"A": 19, "C": 3, "F": 9, "G": 25, "H": 13, "J": 17, "K": 15, "L": 29},
    "ACFGIJKL": {"A": 19, "C": 3, "F": 9, "G": 25, "I": 13, "J": 17, "K": 15, "L": 29},
    "ACFHIJKL": {"A": 19, "C": 3, "F": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "ACGHIJKL": {"A": 19, "C": 3, "G": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "ADEFGHIJ": {"A": 19, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 17},
    "ADEFGHIK": {"A": 19, "D": 3, "E": 17, "F": 9, "G": 25, "H": 13, "I": 29, "K": 15},
    "ADEFGHIL": {"A": 19, "D": 3, "E": 17, "F": 9, "G": 25, "H": 13, "I": 15, "L": 29},
    "ADEFGHJK": {"A": 19, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 17, "K": 15},
    "ADEFGHJL": {"A": 19, "D": 3, "E": 15, "F": 9, "G": 25, "H": 13, "J": 17, "L": 29},
    "ADEFGHKL": {"A": 19, "D": 3, "E": 17, "F": 9, "G": 25, "H": 13, "K": 15, "L": 29},
    "ADEFGIJK": {"A": 19, "D": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 17, "K": 15},
    "ADEFGIJL": {"A": 19, "D": 3, "E": 13, "F": 9, "G": 25, "I": 15, "J": 17, "L": 29},
    "ADEFGIKL": {"A": 19, "D": 3, "E": 13, "F": 9, "G": 25, "I": 17, "K": 15, "L": 29},
    "ADEFGJKL": {"A": 19, "D": 3, "E": 13, "F": 9, "G": 25, "J": 17, "K": 15, "L": 29},
    "ADEFHIJK": {"A": 19, "D": 3, "E": 17, "F": 9, "H": 13, "I": 29, "J": 25, "K": 15},
    "ADEFHIJL": {"A": 19, "D": 3, "E": 17, "F": 9, "H": 13, "I": 15, "J": 25, "L": 29},
    "ADEFHIKL": {"A": 19, "D": 3, "E": 25, "F": 9, "H": 13, "I": 17, "K": 15, "L": 29},
    "ADEFHJKL": {"A": 19, "D": 3, "E": 17, "F": 9, "H": 13, "J": 25, "K": 15, "L": 29},
    "ADEFIJKL": {"A": 19, "D": 3, "E": 13, "F": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ADEGHIJK": {"A": 19, "D": 3, "E": 13, "G": 25, "H": 9, "I": 29, "J": 17, "K": 15},
    "ADEGHIJL": {"A": 19, "D": 3, "E": 13, "G": 25, "H": 9, "I": 15, "J": 17, "L": 29},
    "ADEGHIKL": {"A": 19, "D": 3, "E": 13, "G": 25, "H": 9, "I": 17, "K": 15, "L": 29},
    "ADEGHJKL": {"A": 19, "D": 3, "E": 13, "G": 25, "H": 9, "J": 17, "K": 15, "L": 29},
    "ADEGIJKL": {"A": 19, "D": 3, "E": 13, "G": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ADEHIJKL": {"A": 19, "D": 3, "E": 13, "H": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "ADFGHIJK": {"A": 19, "D": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 17, "K": 15},
    "ADFGHIJL": {"A": 19, "D": 3, "F": 9, "G": 25, "H": 13, "I": 15, "J": 17, "L": 29},
    "ADFGHIKL": {"A": 19, "D": 3, "F": 9, "G": 25, "H": 13, "I": 17, "K": 15, "L": 29},
    "ADFGHJKL": {"A": 19, "D": 3, "F": 9, "G": 25, "H": 13, "J": 17, "K": 15, "L": 29},
    "ADFGIJKL": {"A": 19, "D": 3, "F": 9, "G": 25, "I": 13, "J": 17, "K": 15, "L": 29},
    "ADFHIJKL": {"A": 19, "D": 3, "F": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "ADGHIJKL": {"A": 19, "D": 3, "G": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "AEFGHIJK": {"A": 19, "E": 13, "F": 3, "G": 25, "H": 9, "I": 29, "J": 17, "K": 15},
    "AEFGHIJL": {"A": 19, "E": 13, "F": 3, "G": 25, "H": 9, "I": 15, "J": 17, "L": 29},
    "AEFGHIKL": {"A": 19, "E": 13, "F": 3, "G": 25, "H": 9, "I": 17, "K": 15, "L": 29},
    "AEFGHJKL": {"A": 19, "E": 13, "F": 3, "G": 25, "H": 9, "J": 17, "K": 15, "L": 29},
    "AEFGIJKL": {"A": 19, "E": 13, "F": 3, "G": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "AEFHIJKL": {"A": 19, "E": 13, "F": 3, "H": 9, "I": 17, "J": 25, "K": 15, "L": 29},
    "AEGHIJKL": {"A": 3, "E": 13, "G": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "AFGHIJKL": {"A": 19, "F": 3, "G": 9, "H": 13, "I": 17, "J": 25, "K": 15, "L": 29},
    "BCDEFGHI": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "H": 19, "I": 15},
    "BCDEFGHJ": {"B": 17, "C": 3, "D": 29, "E": 15, "F": 9, "G": 25, "H": 13, "J": 19},
    "BCDEFGHK": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "H": 19, "K": 15},
    "BCDEFGHL": {"B": 17, "C": 13, "D": 3, "E": 15, "F": 9, "G": 25, "H": 19, "L": 29},
    "BCDEFGIJ": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "I": 15, "J": 19},
    "BCDEFGIK": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "G": 25, "I": 29, "K": 15},
    "BCDEFGIL": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "G": 25, "I": 15, "L": 29},
    "BCDEFGJK": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "J": 19, "K": 15},
    "BCDEFGJL": {"B": 17, "C": 13, "D": 3, "E": 15, "F": 9, "G": 25, "J": 19, "L": 29},
    "BCDEFGKL": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "G": 25, "K": 15, "L": 29},
    "BCDEFHIJ": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "H": 19, "I": 15, "J": 25},
    "BCDEFHIK": {"B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "H": 19, "I": 29, "K": 15},
    "BCDEFHIL": {"B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "H": 19, "I": 15, "L": 29},
    "BCDEFHJK": {"B": 17, "C": 13, "D": 3, "E": 29, "F": 9, "H": 19, "J": 25, "K": 15},
    "BCDEFHJL": {"B": 17, "C": 13, "D": 3, "E": 15, "F": 9, "H": 19, "J": 25, "L": 29},
    "BCDEFHKL": {"B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "H": 19, "K": 15, "L": 29},
    "BCDEFIJK": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "I": 29, "J": 25, "K": 15},
    "BCDEFIJL": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "I": 15, "J": 25, "L": 29},
    "BCDEFIKL": {"B": 17, "C": 13, "D": 3, "E": 25, "F": 9, "I": 19, "K": 15, "L": 29},
    "BCDEFJKL": {"B": 17, "C": 13, "D": 3, "E": 19, "F": 9, "J": 25, "K": 15, "L": 29},
    "BCDEGHIJ": {"B": 17, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "I": 15, "J": 19},
    "BCDEGHIK": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "I": 29, "K": 15},
    "BCDEGHIL": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "I": 15, "L": 29},
    "BCDEGHJK": {"B": 17, "C": 3, "D": 9, "E": 29, "G": 25, "H": 13, "J": 19, "K": 15},
    "BCDEGHJL": {"B": 17, "C": 3, "D": 9, "E": 15, "G": 25, "H": 13, "J": 19, "L": 29},
    "BCDEGHKL": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "K": 15, "L": 29},
    "BCDEGIJK": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 29, "J": 19, "K": 15},
    "BCDEGIJL": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 15, "J": 19, "L": 29},
    "BCDEGIKL": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "I": 19, "K": 15, "L": 29},
    "BCDEGJKL": {"B": 17, "C": 3, "D": 9, "E": 13, "G": 25, "J": 19, "K": 15, "L": 29},
    "BCDEHIJK": {"B": 17, "C": 3, "D": 9, "E": 13, "H": 19, "I": 29, "J": 25, "K": 15},
    "BCDEHIJL": {"B": 17, "C": 3, "D": 9, "E": 13, "H": 19, "I": 15, "J": 25, "L": 29},
    "BCDEHIKL": {"B": 17, "C": 3, "D": 9, "E": 13, "H": 19, "I": 25, "K": 15, "L": 29},
    "BCDEHJKL": {"B": 17, "C": 3, "D": 9, "E": 13, "H": 19, "J": 25, "K": 15, "L": 29},
    "BCDEIJKL": {"B": 17, "C": 3, "D": 9, "E": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCDFGHIJ": {"B": 17, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 19},
    "BCDFGHIK": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "I": 29, "K": 15},
    "BCDFGHIL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "I": 15, "L": 29},
    "BCDFGHJK": {"B": 17, "C": 3, "D": 29, "F": 9, "G": 25, "H": 13, "J": 19, "K": 15},
    "BCDFGHJL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "J": 15, "L": 29},
    "BCDFGHKL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "K": 15, "L": 29},
    "BCDFGIJK": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 29, "J": 19, "K": 15},
    "BCDFGIJL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 15, "J": 19, "L": 29},
    "BCDFGIKL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "I": 19, "K": 15, "L": 29},
    "BCDFGJKL": {"B": 17, "C": 13, "D": 3, "F": 9, "G": 25, "J": 19, "K": 15, "L": 29},
    "BCDFHIJK": {"B": 17, "C": 13, "D": 3, "F": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BCDFHIJL": {"B": 17, "C": 13, "D": 3, "F": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BCDFHIKL": {"B": 17, "C": 13, "D": 3, "F": 9, "H": 19, "I": 25, "K": 15, "L": 29},
    "BCDFHJKL": {"B": 17, "C": 13, "D": 3, "F": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BCDFIJKL": {"B": 17, "C": 13, "D": 3, "F": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCDGHIJK": {"B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 29, "J": 19, "K": 15},
    "BCDGHIJL": {"B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 15, "J": 19, "L": 29},
    "BCDGHIKL": {"B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "I": 19, "K": 15, "L": 29},
    "BCDGHJKL": {"B": 17, "C": 3, "D": 9, "G": 25, "H": 13, "J": 19, "K": 15, "L": 29},
    "BCDGIJKL": {"B": 17, "C": 3, "D": 9, "G": 25, "I": 13, "J": 19, "K": 15, "L": 29},
    "BCDHIJKL": {"B": 17, "C": 3, "D": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCEFGHIJ": {"B": 17, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 19},
    "BCEFGHIK": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 29, "K": 15},
    "BCEFGHIL": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 15, "L": 29},
    "BCEFGHJK": {"B": 17, "C": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 19, "K": 15},
    "BCEFGHJL": {"B": 17, "C": 3, "E": 15, "F": 9, "G": 25, "H": 13, "J": 19, "L": 29},
    "BCEFGHKL": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "K": 15, "L": 29},
    "BCEFGIJK": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 19, "K": 15},
    "BCEFGIJL": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 15, "J": 19, "L": 29},
    "BCEFGIKL": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "I": 19, "K": 15, "L": 29},
    "BCEFGJKL": {"B": 17, "C": 3, "E": 13, "F": 9, "G": 25, "J": 19, "K": 15, "L": 29},
    "BCEFHIJK": {"B": 17, "C": 3, "E": 13, "F": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BCEFHIJL": {"B": 17, "C": 3, "E": 13, "F": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BCEFHIKL": {"B": 17, "C": 3, "E": 13, "F": 9, "H": 19, "I": 25, "K": 15, "L": 29},
    "BCEFHJKL": {"B": 17, "C": 3, "E": 13, "F": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BCEFIJKL": {"B": 17, "C": 3, "E": 13, "F": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCEGHIJK": {"B": 17, "C": 3, "E": 13, "G": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BCEGHIJL": {"B": 17, "C": 3, "E": 13, "G": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BCEGHIKL": {"B": 17, "C": 3, "E": 13, "G": 25, "H": 9, "I": 19, "K": 15, "L": 29},
    "BCEGHJKL": {"B": 17, "C": 3, "E": 13, "G": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BCEGIJKL": {"B": 17, "C": 3, "E": 13, "G": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCEHIJKL": {"B": 17, "C": 3, "E": 13, "H": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCFGHIJK": {"B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 19, "K": 15},
    "BCFGHIJL": {"B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 15, "J": 19, "L": 29},
    "BCFGHIKL": {"B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "I": 19, "K": 15, "L": 29},
    "BCFGHJKL": {"B": 17, "C": 3, "F": 9, "G": 25, "H": 13, "J": 19, "K": 15, "L": 29},
    "BCFGIJKL": {"B": 17, "C": 3, "F": 9, "G": 25, "I": 13, "J": 19, "K": 15, "L": 29},
    "BCFHIJKL": {"B": 17, "C": 3, "F": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BCGHIJKL": {"B": 17, "C": 3, "G": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BDEFGHIJ": {"B": 17, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "I": 15, "J": 19},
    "BDEFGHIK": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 29, "K": 15},
    "BDEFGHIL": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 15, "L": 29},
    "BDEFGHJK": {"B": 17, "D": 3, "E": 29, "F": 9, "G": 25, "H": 13, "J": 19, "K": 15},
    "BDEFGHJL": {"B": 17, "D": 3, "E": 15, "F": 9, "G": 25, "H": 13, "J": 19, "L": 29},
    "BDEFGHKL": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "K": 15, "L": 29},
    "BDEFGIJK": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 29, "J": 19, "K": 15},
    "BDEFGIJL": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 15, "J": 19, "L": 29},
    "BDEFGIKL": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "I": 19, "K": 15, "L": 29},
    "BDEFGJKL": {"B": 17, "D": 3, "E": 13, "F": 9, "G": 25, "J": 19, "K": 15, "L": 29},
    "BDEFHIJK": {"B": 17, "D": 3, "E": 13, "F": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BDEFHIJL": {"B": 17, "D": 3, "E": 13, "F": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BDEFHIKL": {"B": 17, "D": 3, "E": 13, "F": 9, "H": 19, "I": 25, "K": 15, "L": 29},
    "BDEFHJKL": {"B": 17, "D": 3, "E": 13, "F": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BDEFIJKL": {"B": 17, "D": 3, "E": 13, "F": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BDEGHIJK": {"B": 17, "D": 3, "E": 13, "G": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BDEGHIJL": {"B": 17, "D": 3, "E": 13, "G": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BDEGHIKL": {"B": 17, "D": 3, "E": 13, "G": 25, "H": 9, "I": 19, "K": 15, "L": 29},
    "BDEGHJKL": {"B": 17, "D": 3, "E": 13, "G": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BDEGIJKL": {"B": 17, "D": 3, "E": 13, "G": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BDEHIJKL": {"B": 17, "D": 3, "E": 13, "H": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BDFGHIJK": {"B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 29, "J": 19, "K": 15},
    "BDFGHIJL": {"B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 15, "J": 19, "L": 29},
    "BDFGHIKL": {"B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "I": 19, "K": 15, "L": 29},
    "BDFGHJKL": {"B": 17, "D": 3, "F": 9, "G": 25, "H": 13, "J": 19, "K": 15, "L": 29},
    "BDFGIJKL": {"B": 17, "D": 3, "F": 9, "G": 25, "I": 13, "J": 19, "K": 15, "L": 29},
    "BDFHIJKL": {"B": 17, "D": 3, "F": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BDGHIJKL": {"B": 17, "D": 3, "G": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "BEFGHIJK": {"B": 17, "E": 13, "F": 3, "G": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "BEFGHIJL": {"B": 17, "E": 13, "F": 3, "G": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "BEFGHIKL": {"B": 17, "E": 13, "F": 3, "G": 25, "H": 9, "I": 19, "K": 15, "L": 29},
    "BEFGHJKL": {"B": 17, "E": 13, "F": 3, "G": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "BEFGIJKL": {"B": 17, "E": 13, "F": 3, "G": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BEFHIJKL": {"B": 17, "E": 13, "F": 3, "H": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "BEGHIJKL": {"B": 3, "E": 13, "G": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "BFGHIJKL": {"B": 17, "F": 3, "G": 9, "H": 13, "I": 19, "J": 25, "K": 15, "L": 29},
    "CDEFGHIJ": {"C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "H": 19, "I": 15, "J": 17},
    "CDEFGHIK": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "H": 19, "I": 29, "K": 15},
    "CDEFGHIL": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "H": 19, "I": 15, "L": 29},
    "CDEFGHJK": {"C": 13, "D": 3, "E": 29, "F": 9, "G": 25, "H": 19, "J": 17, "K": 15},
    "CDEFGHJL": {"C": 13, "D": 3, "E": 15, "F": 9, "G": 25, "H": 19, "J": 17, "L": 29},
    "CDEFGHKL": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "H": 19, "K": 15, "L": 29},
    "CDEFGIJK": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "I": 29, "J": 19, "K": 15},
    "CDEFGIJL": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "I": 15, "J": 19, "L": 29},
    "CDEFGIKL": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "I": 19, "K": 15, "L": 29},
    "CDEFGJKL": {"C": 13, "D": 3, "E": 17, "F": 9, "G": 25, "J": 19, "K": 15, "L": 29},
    "CDEFHIJK": {"C": 13, "D": 3, "E": 17, "F": 9, "H": 19, "I": 29, "J": 25, "K": 15},
    "CDEFHIJL": {"C": 13, "D": 3, "E": 17, "F": 9, "H": 19, "I": 15, "J": 25, "L": 29},
    "CDEFHIKL": {"C": 13, "D": 3, "E": 25, "F": 9, "H": 19, "I": 17, "K": 15, "L": 29},
    "CDEFHJKL": {"C": 13, "D": 3, "E": 17, "F": 9, "H": 19, "J": 25, "K": 15, "L": 29},
    "CDEFIJKL": {"C": 13, "D": 3, "E": 17, "F": 9, "I": 19, "J": 25, "K": 15, "L": 29},
    "CDEGHIJK": {"C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "I": 29, "J": 17, "K": 15},
    "CDEGHIJL": {"C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "I": 15, "J": 17, "L": 29},
    "CDEGHIKL": {"C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "I": 17, "K": 15, "L": 29},
    "CDEGHJKL": {"C": 3, "D": 9, "E": 13, "G": 25, "H": 19, "J": 17, "K": 15, "L": 29},
    "CDEGIJKL": {"C": 3, "D": 9, "E": 13, "G": 25, "I": 17, "J": 19, "K": 15, "L": 29},
    "CDEHIJKL": {"C": 3, "D": 9, "E": 13, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "CDFGHIJK": {"C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "I": 29, "J": 17, "K": 15},
    "CDFGHIJL": {"C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "I": 15, "J": 17, "L": 29},
    "CDFGHIKL": {"C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "I": 17, "K": 15, "L": 29},
    "CDFGHJKL": {"C": 13, "D": 3, "F": 9, "G": 25, "H": 19, "J": 17, "K": 15, "L": 29},
    "CDFGIJKL": {"C": 13, "D": 3, "F": 9, "G": 25, "I": 17, "J": 19, "K": 15, "L": 29},
    "CDFHIJKL": {"C": 13, "D": 3, "F": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "CDGHIJKL": {"C": 3, "D": 9, "G": 25, "H": 13, "I": 17, "J": 19, "K": 15, "L": 29},
    "CEFGHIJK": {"C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 29, "J": 17, "K": 15},
    "CEFGHIJL": {"C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 15, "J": 17, "L": 29},
    "CEFGHIKL": {"C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 17, "K": 15, "L": 29},
    "CEFGHJKL": {"C": 3, "E": 13, "F": 9, "G": 25, "H": 19, "J": 17, "K": 15, "L": 29},
    "CEFGIJKL": {"C": 3, "E": 13, "F": 9, "G": 25, "I": 17, "J": 19, "K": 15, "L": 29},
    "CEFHIJKL": {"C": 3, "E": 13, "F": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "CEGHIJKL": {"C": 3, "E": 13, "G": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "CFGHIJKL": {"C": 3, "F": 9, "G": 25, "H": 13, "I": 17, "J": 19, "K": 15, "L": 29},
    "DEFGHIJK": {"D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 29, "J": 17, "K": 15},
    "DEFGHIJL": {"D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 15, "J": 17, "L": 29},
    "DEFGHIKL": {"D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "I": 17, "K": 15, "L": 29},
    "DEFGHJKL": {"D": 3, "E": 13, "F": 9, "G": 25, "H": 19, "J": 17, "K": 15, "L": 29},
    "DEFGIJKL": {"D": 3, "E": 13, "F": 9, "G": 25, "I": 17, "J": 19, "K": 15, "L": 29},
    "DEFHIJKL": {"D": 3, "E": 13, "F": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "DEGHIJKL": {"D": 3, "E": 13, "G": 9, "H": 19, "I": 17, "J": 25, "K": 15, "L": 29},
    "DFGHIJKL": {"D": 3, "F": 9, "G": 25, "H": 13, "I": 17, "J": 19, "K": 15, "L": 29},
    "EFGHIJKL": {"E": 25, "F": 19, "G": 29, "H": 9, "I": 3, "J": 17, "L": 15},
}


def _third_place_assignment_for_combo(groups: Sequence[str]) -> Dict[str, int]:
    """Assign third-place groups to FIFA R32 slots for one qualifying combination."""
    key = "".join(sorted(groups))
    try:
        return dict(FIFA_THIRD_PLACE_TABLE[key])
    except KeyError:
        raise ValueError(f"No FIFA third-place allocation found for groups: {','.join(groups)}")


THIRD_PLACE_ALLOCATION_MATRIX = {
    combo: _third_place_assignment_for_combo(combo)
    for combo in combinations("ABCDEFGHIJKL", 8)
}


def harmonize_country(name: object) -> object:
    """Return the canonical project country/team name."""
    if pd.isna(name):
        return name
    text = str(name).strip()
    return NAME_ALIASES.get(text, text)


# Actual penalty winners for completed WC 2026 knockout draws that are not
# represented in results.csv. Keep this in shared code so backtests and bracket
# predictions evaluate knockout winners consistently.
WC2026_PENALTY_WINNERS = {
    frozenset(("Germany", "Paraguay")): "Paraguay",
    frozenset(("Netherlands", "Morocco")): "Morocco",
}


def wc2026_penalty_winner(home: str, away: str) -> Optional[str]:
    """Return the known WC 2026 shootout winner for a tied knockout, if any."""
    key = frozenset((harmonize_country(home), harmonize_country(away)))
    winner = WC2026_PENALTY_WINNERS.get(key)
    return harmonize_country(winner) if winner else None


def harmonize_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(harmonize_country)
    return out


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def update_elo(elo_a: float, elo_b: float, score_a: int, score_b: int, neutral: bool = True):
    home_advantage = 0 if neutral else 50
    ea = expected_score(elo_a + home_advantage, elo_b)
    sa = 1 if score_a > score_b else (0.5 if score_a == score_b else 0)
    margin = abs(score_a - score_b)
    multiplier = np.log(max(margin, 1) + 1)
    return (
        elo_a + K_FACTOR * multiplier * (sa - ea),
        elo_b + K_FACTOR * multiplier * ((1 - sa) - (1 - ea)),
    )


def make_team_state() -> Dict[str, object]:
    """Create the mutable rolling state expected by match feature builders."""
    return {
        "elo": INITIAL_ELO,
        "form": [],
        "goals_for": [],
        "goals_against": [],
        "last_match": None,
        # New rolling history needed for momentum / opponent-weighted / fatigue features.
        "elo_history": [],          # Elo value AFTER each match (chronological)
        "opp_elo": [],              # opponent Elo BEFORE each match (chronological)
        "match_dates": [],          # date of each rolling match (chronological)
        "h2h": defaultdict(lambda: {
            "matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0,
            "entries": [],
        }),
        "wc_participations": 0,
        "wc_titles": 0,
        "wc_wins": 0,
        "wc_matches": 0,
    }


def _trim(seq: list, n: int = FORM_WINDOW) -> list:
    """Return the last ``n`` items of ``seq`` (helper to keep windows consistent)."""
    return seq[-n:]


def update_team_state(
    state,
    team: str,
    opponent: str,
    gf: int,
    ga: int,
    match_date,
    *,
    neutral: bool = True,
    is_world_cup: bool = False,
) -> None:
    """Apply one match result to ONE team's rolling state.

    Centralizing this here guarantees that every script (predict_2026, backtest,
    monte_carlo) updates Elo, form, goals, H2H and the new rolling histories in
    exactly the same way, which previously diverged (different form windows and a
    broken H2H key in the backtest). Call once per team, per match.
    """
    s, o = state[team], state[opponent]
    opp_elo_before = o["elo"]

    # Form / goals rolling windows.
    result = 1 if gf > ga else (0.5 if gf == ga else 0)
    s["form"] = _trim(s["form"] + [result])
    s["goals_for"] = _trim(s["goals_for"] + [gf])
    s["goals_against"] = _trim(s["goals_against"] + [ga])
    s["opp_elo"] = _trim(s["opp_elo"] + [opp_elo_before])
    s["match_dates"] = _trim(s["match_dates"] + [match_date])
    s["last_match"] = match_date

    # H2H (always keyed by the sorted pair so lookups are symmetric).
    key = tuple(sorted([team, opponent]))
    rec = s["h2h"][key]
    rec["matches"] += 1
    if gf > ga:
        rec["wins"] += 1
    elif gf == ga:
        rec["draws"] += 1
    else:
        rec["losses"] += 1
    rec["gf"] += gf
    rec["ga"] += ga
    rec.setdefault("entries", []).append({
        "date": match_date,
        "result": result,
        "gf": gf,
        "ga": ga,
    })

    if is_world_cup:
        s["wc_matches"] += 1
        if gf > ga:
            s["wc_wins"] += 1


def apply_match_to_state(
    state,
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    match_date,
    *,
    neutral: bool = True,
    is_world_cup: bool = False,
) -> None:
    """Apply a full match (both sides + Elo) to the shared rolling state.

    Elo must be updated AFTER both teams' opponent-Elo snapshots are recorded, so
    that ``opp_elo`` reflects the pre-match strength of the opponent.
    """
    home = harmonize_country(home)
    away = harmonize_country(away)
    pre_home, pre_away = state[home]["elo"], state[away]["elo"]

    update_team_state(state, home, away, home_score, away_score, match_date,
                      neutral=neutral, is_world_cup=is_world_cup)
    update_team_state(state, away, home, away_score, home_score, match_date,
                      neutral=neutral, is_world_cup=is_world_cup)

    new_home, new_away = update_elo(pre_home, pre_away, home_score, away_score, neutral)
    state[home]["elo"] = new_home
    state[away]["elo"] = new_away
    state[home]["elo_history"] = _trim(state[home]["elo_history"] + [new_home])
    state[away]["elo_history"] = _trim(state[away]["elo_history"] + [new_away])


def finalize_world_cup_history(state, wc_year: int, participants: Iterable[str]) -> None:
    """Update state once a World Cup has completed."""
    teams = {harmonize_country(team) for team in participants}
    for team in teams:
        state[team]["wc_participations"] += 1

    winner = WC_WINNERS.get(int(wc_year))
    if winner in teams:
        state[winner]["wc_titles"] += 1


def _elo_momentum(state_entry, lookback: int) -> float:
    """Elo gained/lost over the last ``lookback`` matches (0 if not enough history)."""
    hist = state_entry.get("elo_history", [])
    if len(hist) < 2:
        return 0.0
    window = hist[-(lookback + 1):]
    return float(window[-1] - window[0])


def _form_score(form, window=30):
    """Sum of win(+1)/draw(0)/loss(-1) over last ``window`` matches."""
    recent = form[-window:] if form else []
    if not recent:
        return 0
    return sum(1 if f == 1 else (-1 if f == 0 else 0) for f in recent)


def _current_streak(form, target):
    """Count consecutive ``target`` values from the end of form list."""
    streak = 0
    for f in reversed(form):
        if f == target:
            streak += 1
        else:
            break
    return streak


def _unbeaten_streak(form) -> int:
    """Count consecutive matches without a loss (form != 0) from the end."""
    streak = 0
    for f in reversed(form):
        if f != 0:
            streak += 1
        else:
            break
    return streak


def _opp_weighted_form(state_entry) -> float:
    """Recent win/draw points weighted by the strength of the opponent faced.

    A win against a 2000-Elo side counts far more than a win against a 1300-Elo
    side. Returns a strength-weighted average of recent results in [0, 1]-ish range.
    """
    form = state_entry.get("form", [])
    opp = state_entry.get("opp_elo", [])
    n = min(len(form), len(opp))
    if n == 0:
        return 0.5
    form, opp = form[-n:], opp[-n:]
    # Weight = opponent strength relative to 1500 baseline (clipped to stay positive).
    weights = [max(0.25, e / 1500.0) for e in opp]
    total_w = sum(weights)
    if total_w == 0:
        return float(np.mean(form))
    return float(sum(f * w for f, w in zip(form, weights)) / total_w)


def _matches_in_window(state_entry, match_date, days: int) -> int:
    """Number of matches played in the ``days`` leading up to ``match_date`` (fatigue)."""
    dates = state_entry.get("match_dates", [])
    if not dates or match_date is None:
        return 0
    count = 0
    for d in dates:
        if d is None:
            continue
        try:
            delta = (match_date - d).days
        except TypeError:
            continue
        if 0 <= delta <= days:
            count += 1
    return count


def _weighted_h2h_record(record: dict, match_date) -> dict:
    """Return recency-weighted H2H stats from one team's perspective.

    The historical aggregate is retained for backward compatibility, but when
    per-match entries are available we only use the last ``H2H_YEARS_LIMIT``
    years and exponentially decay them with a 10-year half-life.
    """
    entries = record.get("entries") or []
    if not entries:
        return dict(record)
    try:
        current_date = pd.to_datetime(match_date)
    except Exception:
        current_date = pd.Timestamp.max
    cutoff = current_date - pd.DateOffset(years=H2H_YEARS_LIMIT)
    weighted = {"matches": 0.0, "wins": 0.0, "draws": 0.0, "losses": 0.0, "gf": 0.0, "ga": 0.0}
    for item in entries:
        try:
            item_date = pd.to_datetime(item.get("date"))
        except Exception:
            item_date = pd.NaT
        if pd.isna(item_date) or item_date < cutoff or item_date >= current_date:
            continue
        age_years = max((current_date - item_date).days / 365.25, 0.0)
        weight = 0.5 ** (age_years / H2H_HALF_LIFE_YEARS)
        result = float(item.get("result", 0.5))
        weighted["matches"] += weight
        weighted["wins"] += weight if result == 1 else 0.0
        weighted["draws"] += weight if result == 0.5 else 0.0
        weighted["losses"] += weight if result == 0 else 0.0
        weighted["gf"] += weight * float(item.get("gf", 0.0))
        weighted["ga"] += weight * float(item.get("ga", 0.0))
    return weighted


def compute_match_features(team, opponent, state, country_features, stage_num, match_date, neutral=True, is_home=False, odds_row=None, squad_values=None):
    s, o = state[team], state[opponent]
    form = s["form"][-10:] if s["form"] else [0.5]
    opp_form = o["form"][-10:] if o["form"] else [0.5]
    gf5 = s["goals_for"][-5:] or [0]
    ga5 = s["goals_against"][-5:] or [0]
    gf10 = s["goals_for"][-10:] or [0]
    ga10 = s["goals_against"][-10:] or [0]
    gf3 = s["goals_for"][-3:] or [0]
    ga3 = s["goals_against"][-3:] or [0]
    opp_gf5 = o["goals_for"][-5:] or [0]
    opp_ga5 = o["goals_against"][-5:] or [0]
    team_form = s["form"] or [0.5]
    opp_form_full = o["form"] or [0.5]
    rest = min((match_date - s["last_match"]).days if s["last_match"] else 30, 60)
    h2h_key = tuple(sorted([team, opponent]))
    h = s["h2h"].get(h2h_key, {"matches": 0, "wins": 0, "draws": 0, "losses": 0, "gf": 0, "ga": 0})
    h = _weighted_h2h_record(h, match_date)
    hm = max(h["matches"], 1)
    cf = get_effective_country_features(team, country_features, state)
    oc = get_effective_country_features(opponent, country_features, state)

    # ── Existing derived quantities ──
    form_win_rate = sum(1 for f in form if f == 1) / len(form)
    form_draw_rate = sum(1 for f in form if f == 0.5) / len(form)
    opp_form_draw_rate = sum(1 for f in opp_form if f == 0.5) / len(opp_form)
    elo_diff = s["elo"] - o["elo"]

    # ── NEW FEATURES ──
    # 1. Elo momentum: is the team trending up or down recently?
    elo_momentum_5 = _elo_momentum(s, 5)
    elo_momentum_10 = _elo_momentum(s, 10)
    opp_elo_momentum_5 = _elo_momentum(o, 5)
    elo_momentum_diff = elo_momentum_5 - opp_elo_momentum_5

    # 2. Attacking / defensive trend: recent 3 vs baseline 10.
    attack_trend = float(np.mean(gf3) - np.mean(gf10))
    defense_trend = float(np.mean(ga3) - np.mean(ga10))  # positive = conceding more lately

    # 3. Opponent-strength-weighted form (quality of recent results).
    weighted_form = _opp_weighted_form(s)
    opp_weighted_form = _opp_weighted_form(o)
    weighted_form_diff = weighted_form - opp_weighted_form

    # 4. Fatigue / congestion: matches in the last 30 and 90 days.
    fatigue_30 = _matches_in_window(s, match_date, 30)
    fatigue_90 = _matches_in_window(s, match_date, 90)
    opp_fatigue_30 = _matches_in_window(o, match_date, 30)
    fatigue_diff_30 = fatigue_30 - opp_fatigue_30

    # 5. Draw-propensity signals (the model's documented weakness is missing draws).
    #    Closely-matched, low-scoring, draw-prone sides are the classic draw setup.
    elo_parity = 1.0 / (1.0 + abs(elo_diff) / 100.0)  # ~1 when evenly matched, ->0 apart
    combined_draw_rate = (form_draw_rate + opp_form_draw_rate) / 2.0
    expected_total_goals = float(np.mean(gf5) + np.mean(ga5)
                                 + np.mean(o["goals_for"][-5:] or [0])
                                 + np.mean(o["goals_against"][-5:] or [0])) / 2.0
    low_scoring_indicator = 1.0 / (1.0 + expected_total_goals)

    # Form score (last 30) and win/loss streaks.
    form_score_30 = _form_score(team_form, 30)
    opp_form_score_30 = _form_score(opp_form_full, 30)
    form_score_diff = form_score_30 - opp_form_score_30
    win_streak = _current_streak(team_form, 1)
    opp_win_streak = _current_streak(opp_form_full, 1)
    unbeaten_streak = _unbeaten_streak(team_form)
    loss_streak = _current_streak(team_form, 0)
    clean_sheets_5 = sum(1 for g in ga5 if g == 0)
    clean_sheets_10 = sum(1 for g in ga10 if g == 0)
    opp_clean_sheets_5 = sum(1 for g in opp_ga5 if g == 0)
    gd_5 = float(np.mean(gf5) - np.mean(ga5))
    gd_10 = float(np.mean(gf10) - np.mean(ga10))
    opp_gd_5 = float(np.mean(opp_gf5) - np.mean(opp_ga5))
    goal_diff_diff = gd_5 - opp_gd_5
    scoring_rate_5 = sum(1 for g in gf5 if g > 0) / max(len(gf5), 1)
    conceding_rate_5 = sum(1 for g in ga5 if g > 0) / max(len(ga5), 1)

    # 6. Bookmaker implied probabilities (single best predictor in the football
    #    literature). NaN when the match has no odds; XGBoost handles missing.
    if odds_row is None:
        odds_row = {col: np.nan for col in ODDS_FEATURE_COLUMNS}

    # 7. Transfermarkt squad market values (log-scaled). Uses the most recent WC
    #    edition's values at or before the match year. Missing teams/years stay
    #    NaN so XGBoost can learn a dedicated "no valuation" split rather than
    #    treating an absent value as 0 EUR.
    home_val = squad_value_for_team(squad_values, team, match_date)["total_squad_value_eur"]
    away_val = squad_value_for_team(squad_values, opponent, match_date)["total_squad_value_eur"]
    squad_value = np.log1p(home_val) if np.isfinite(home_val) else np.nan
    opp_squad_value = np.log1p(away_val) if np.isfinite(away_val) else np.nan
    if np.isfinite(home_val) and np.isfinite(away_val):
        squad_value_diff = np.log1p(home_val) - np.log1p(away_val)
        # Ratio of raw values (home / away); +1 guards against a zero-value squad.
        squad_value_ratio = (home_val + 1.0) / (away_val + 1.0)
    else:
        squad_value_diff = np.nan
        squad_value_ratio = np.nan

    return {
        "elo": s["elo"], "elo_opponent": o["elo"],
        "elo_diff": elo_diff, "elo_sum": s["elo"] + o["elo"],
        "form_win_rate": form_win_rate,
        "form_draw_rate": form_draw_rate,
        "form_loss_rate": sum(1 for f in form if f == 0) / len(form),
        "avg_goals_scored_5": np.mean(gf5), "avg_goals_conceded_5": np.mean(ga5),
        "avg_goals_scored_10": np.mean(gf10), "avg_goals_conceded_10": np.mean(ga10),
        "rest_days": rest,
        "h2h_matches": h["matches"], "h2h_win_rate": h["wins"] / hm,
        "h2h_draw_rate": h["draws"] / hm, "h2h_avg_goals_for": h["gf"] / hm,
        "h2h_avg_goals_against": h["ga"] / hm,
        "wc_participations": s["wc_participations"], "wc_titles": s["wc_titles"],
        "wc_win_rate": s["wc_wins"] / max(s["wc_matches"], 1),
        "stage": stage_num, "neutral": int(bool(neutral)), "is_home": int(bool(is_home)),
        "gdp_per_capita": cf.get("gdp_per_capita", np.nan),
        "population": cf.get("population", np.nan),
        "life_expectancy": cf.get("life_expectancy", np.nan),
        "urbanization_pct": cf.get("urbanization_pct", np.nan),
        "health_spending_pct_gdp": cf.get("health_spending_pct_gdp", np.nan),
        "elo_pre_tournament": cf.get("elo_rating", s["elo"]),
        "fifa_rank": cf.get("fifa_rank", 100),
        "football_power_index": cf.get("football_power_index", 0),
        "football_tradition": cf.get("football_tradition", 0),
        "opp_elo_pre_tournament": oc.get("elo_rating", o["elo"]),
        "opp_football_power_index": oc.get("football_power_index", 0),
        "opp_football_tradition": oc.get("football_tradition", 0),
        "elo_diff_pre": cf.get("elo_rating", s["elo"]) - oc.get("elo_rating", o["elo"]),
        "power_diff": cf.get("football_power_index", 0) - oc.get("football_power_index", 0),
        "tradition_diff": cf.get("football_tradition", 0) - oc.get("football_tradition", 0),
        # ── NEW engineered features ──
        "elo_momentum_5": elo_momentum_5,
        "elo_momentum_10": elo_momentum_10,
        "elo_momentum_diff": elo_momentum_diff,
        "attack_trend": attack_trend,
        "defense_trend": defense_trend,
        "weighted_form": weighted_form,
        "weighted_form_diff": weighted_form_diff,
        "fatigue_30": fatigue_30,
        "fatigue_90": fatigue_90,
        "fatigue_diff_30": fatigue_diff_30,
        "elo_parity": elo_parity,
        "combined_draw_rate": combined_draw_rate,
        "expected_total_goals": expected_total_goals,
        "low_scoring_indicator": low_scoring_indicator,
        "form_score_30": form_score_30,
        "opp_form_score_30": opp_form_score_30,
        "form_score_diff": form_score_diff,
        "win_streak": win_streak,
        "opp_win_streak": opp_win_streak,
        "unbeaten_streak": unbeaten_streak,
        "loss_streak": loss_streak,
        "clean_sheets_5": clean_sheets_5,
        "clean_sheets_10": clean_sheets_10,
        "opp_clean_sheets_5": opp_clean_sheets_5,
        "goal_diff_5": gd_5,
        "goal_diff_10": gd_10,
        "goal_diff_diff": goal_diff_diff,
        "scoring_rate_5": scoring_rate_5,
        "conceding_rate_5": conceding_rate_5,
        # ── Bookmaker implied probabilities (NaN when no odds available) ──
        "implied_home_prob": odds_row.get("implied_home_prob", np.nan),
        "implied_draw_prob": odds_row.get("implied_draw_prob", np.nan),
        "implied_away_prob": odds_row.get("implied_away_prob", np.nan),
        "odds_overround": odds_row.get("odds_overround", np.nan),
        # ── Transfermarkt squad market values (log-scaled; NaN when unavailable) ──
        "squad_value": squad_value,
        "opp_squad_value": opp_squad_value,
        "squad_value_diff": squad_value_diff,
        "squad_value_ratio": squad_value_ratio,
    }


def finalize_feature_frame(rows: Sequence[dict]) -> pd.DataFrame:
    """Turn feature dicts into a model-ready frame.

    Most columns are filled with 0 (the historical behaviour), but the
    bookmaker-odds and squad-value columns are *deliberately left as NaN* when
    missing so that XGBoost can learn dedicated "missing odds" / "no valuation"
    splits instead of treating an absent market as a 0% implied probability or an
    unvalued squad as 0 EUR. ``predict_proba`` at inference time must use the same
    convention (see ``prepare_prediction_frame``).
    """
    X = pd.DataFrame(rows)
    keep_nan = set(ODDS_FEATURE_COLUMNS) | set(SQUAD_VALUE_FEATURE_COLUMNS)
    non_nan = [c for c in X.columns if c not in keep_nan]
    if non_nan:
        X[non_nan] = X[non_nan].fillna(0)
    return X


def prepare_prediction_frame(feat: dict, feature_names: Sequence[str]) -> pd.DataFrame:
    """Build a single-row frame aligned to ``feature_names`` for prediction.

    Mirrors :func:`finalize_feature_frame`: odds and squad-value columns keep
    their NaN so the model sees the same missing-value encoding it was trained on.
    """
    X = pd.DataFrame([feat])
    for name in feature_names:
        if name not in X.columns:
            X[name] = np.nan
    X = X[list(feature_names)]
    keep_nan = set(ODDS_FEATURE_COLUMNS) | set(SQUAD_VALUE_FEATURE_COLUMNS)
    non_nan = [c for c in X.columns if c not in keep_nan]
    if non_nan:
        X[non_nan] = X[non_nan].fillna(0)
    return X


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def parse_neutral_flag(value: object, default: bool = True) -> bool:
    """Parse the results.csv neutral flag, defaulting missing values to neutral."""
    if pd.isna(value):
        return bool(default)
    return parse_bool(value)


def _odds_key(date, home: str, away: str) -> Tuple[str, str, str]:
    """Build the merge key used to attach bookmaker odds to a match.

    Keyed by (ISO date string, canonical home, canonical away) so the lookup is
    robust to the various name spellings used in the odds source.
    """
    try:
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    except Exception:
        date_str = str(date)
    return (date_str, harmonize_country(home), harmonize_country(away))


def load_betting_odds(
    path: Path = DATA_DIR / "betting_odds.csv",
) -> Dict[Tuple[str, str, str], Dict[str, float]]:
    """Load bookmaker implied probabilities keyed by (date, home, away).

    Returns a mapping from ``(date, canonical_home, canonical_away)`` to the
    ``ODDS_FEATURE_COLUMNS`` values. ``odds_overround`` is derived from the three
    implied probabilities (their sum minus 1). The orientation-swapped key is also
    stored (with home/away implied probs flipped) so a match recorded in the
    opposite orientation still finds its odds.

    Missing/unparseable files yield an empty mapping, so callers degrade
    gracefully to "no odds" (all-NaN) rather than crashing.
    """
    odds: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    if not Path(path).exists():
        return odds
    try:
        df = pd.read_csv(path)
    except Exception:
        return odds

    required = {"date", "home_team", "away_team",
                "implied_home_prob", "implied_draw_prob", "implied_away_prob"}
    if not required.issubset(df.columns):
        return odds

    for _, row in df.iterrows():
        try:
            ph = float(row["implied_home_prob"])
            pd_ = float(row["implied_draw_prob"])
            pa = float(row["implied_away_prob"])
        except (TypeError, ValueError):
            continue
        if not all(np.isfinite([ph, pd_, pa])):
            continue
        overround = ph + pd_ + pa - 1.0
        forward = {
            "implied_home_prob": ph,
            "implied_draw_prob": pd_,
            "implied_away_prob": pa,
            "odds_overround": overround,
        }
        reverse = {
            "implied_home_prob": pa,
            "implied_draw_prob": pd_,
            "implied_away_prob": ph,
            "odds_overround": overround,
        }
        key = _odds_key(row["date"], row["home_team"], row["away_team"])
        # Forward orientation takes precedence; only add the swapped orientation
        # if that exact (date, home, away) pair is not already populated.
        odds.setdefault(key, forward)
        swapped = (key[0], key[2], key[1])
        odds.setdefault(swapped, reverse)
    return odds


def odds_features_for_match(
    odds: Optional[Dict[Tuple[str, str, str], Dict[str, float]]],
    date,
    home: str,
    away: str,
) -> Dict[str, float]:
    """Return the odds feature dict for a match, or all-NaN when unavailable."""
    nan_row = {col: np.nan for col in ODDS_FEATURE_COLUMNS}
    if not odds:
        return nan_row
    return dict(odds.get(_odds_key(date, home, away), nan_row))


def load_squad_values(
    path: Path = DATA_DIR / "squad_values.csv",
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Load Transfermarkt squad values keyed by (canonical_team, year).

    The CSV uses the same raw team spellings as ``results.csv`` (e.g. "Iran",
    "South Korea", "Ivory Coast"), so each team is harmonized through
    :data:`NAME_ALIASES` on load to match the canonical names used in the rolling
    state dict ("IR Iran", "Korea Republic", "Cote d'Ivoire" -> "Côte d'Ivoire").

    Returns a mapping ``(team, year) -> {total_squad_value_eur,
    avg_player_value_eur}``. Missing/unparseable files yield an empty mapping so
    callers degrade gracefully to all-NaN squad-value features.
    """
    values: Dict[Tuple[str, int], Dict[str, float]] = {}
    if not Path(path).exists():
        return values
    try:
        df = pd.read_csv(path)
    except Exception:
        return values

    required = {"team", "year", "total_squad_value_eur", "avg_player_value_eur"}
    if not required.issubset(df.columns):
        return values

    for _, row in df.iterrows():
        try:
            year = int(row["year"])
        except (TypeError, ValueError):
            continue
        team = harmonize_country(row["team"])
        if pd.isna(team):
            continue

        def _num(col: str) -> float:
            try:
                v = float(row[col])
            except (TypeError, ValueError):
                return np.nan
            return v if np.isfinite(v) else np.nan

        values[(str(team), year)] = {
            "total_squad_value_eur": _num("total_squad_value_eur"),
            "avg_player_value_eur": _num("avg_player_value_eur"),
        }
    return values


def _squad_value_year(match_year: int) -> Optional[int]:
    """Most recent World Cup edition with squad values at or before ``match_year``.

    Implements the documented year lag: matches in 2023 look up 2022 squads, 2026
    fixtures look up 2026 values, etc. Returns ``None`` when the match predates the
    earliest available edition.
    """
    usable = [y for y in SQUAD_VALUE_YEARS if y <= match_year]
    return max(usable) if usable else None


def squad_value_for_team(
    squad_values: Optional[Dict[Tuple[str, int], Dict[str, float]]],
    team: str,
    match_date,
) -> Dict[str, float]:
    """Return the squad-value entry for ``team`` at the lagged WC year, or NaNs.

    ``team`` must already be the canonical (harmonized) name. ``match_date`` may be
    anything ``pd.to_datetime`` understands; its calendar year drives the year-lag
    lookup via :func:`_squad_value_year`.
    """
    nan_entry = {"total_squad_value_eur": np.nan, "avg_player_value_eur": np.nan}
    if not squad_values:
        return dict(nan_entry)
    try:
        match_year = int(pd.to_datetime(match_date).year)
    except Exception:
        return dict(nan_entry)
    lookup_year = _squad_value_year(match_year)
    if lookup_year is None:
        return dict(nan_entry)
    return dict(squad_values.get((harmonize_country(team), lookup_year), nan_entry))


def load_country_feature_history(
    path: Path = DATA_DIR / "world_cup_predictors_dataset.csv",
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Load country features keyed by canonical team and data year.

    The source column is named ``wc_year`` because the original dataset was
    World-Cup-edition based, but callers should treat it as a feature vintage.
    If newer non-WC rows are added later (for example 2023/2024 country data),
    they will be eligible for 2026 lookups automatically.
    """
    df = pd.read_csv(path)
    history: Dict[str, Dict[int, Dict[str, float]]] = {}
    for _, row in df.iterrows():
        team = harmonize_country(row["country"])
        year = int(row["wc_year"])
        entry = {
            col: row.get(col, np.nan) for col in COUNTRY_FEATURE_COLUMNS
        }
        entry["feature_year"] = year
        history.setdefault(team, {})[year] = entry
    return history


def country_features_for_year(
    history: Dict[str, Dict[int, Dict[str, float]]], year: int
) -> Dict[str, Dict[str, float]]:
    """Return latest known country features at or before ``year`` for each team.

    Feature vintages older than ``COUNTRY_FEATURE_STALE_YEARS`` are retained but
    explicitly flagged via ``country_features_stale`` and ``feature_age_years``.
    That keeps predictions possible while making cases such as 2026 Norway using
    1998 country data visible to reports and CLIs.
    """
    features: Dict[str, Dict[str, float]] = {}
    for team, by_year in history.items():
        usable = [wc_year for wc_year in by_year if wc_year <= year]
        if not usable:
            continue
        feature_year = max(usable)
        entry = dict(by_year[feature_year])
        age = int(year) - int(feature_year)
        entry["feature_year"] = feature_year
        entry["feature_age_years"] = age
        entry["country_features_stale"] = int(age > COUNTRY_FEATURE_STALE_YEARS)
        features[team] = entry
    return features


def get_effective_country_features(team, country_features, state, year=2026):
    """Return country features, substituting Elo-based proxies for stale data."""
    cf = dict(country_features.get(team, {}))
    age = cf.get("feature_age_years", 99)

    if age > 10:
        elo = state.get(team, {}).get("elo", 1500)
        cf["football_power_index"] = max(0, (elo - 1000) / 5.5)

        wc_matches = state.get(team, {}).get("wc_matches", 0)
        wc_participations = state.get(team, {}).get("wc_participations", 0)
        cf["football_tradition"] = min(100, wc_participations * 5 + wc_matches * 0.5)

        cf["feature_year"] = year
        cf["feature_age_years"] = 0
        cf["country_features_stale"] = 0

    return cf


def country_feature_staleness_warnings(
    features: Dict[str, Dict[str, float]],
    teams: Iterable[str],
    target_year: int,
) -> List[str]:
    """Human-readable warnings for stale country feature vintages."""
    warnings: List[str] = []
    for team in sorted({harmonize_country(t) for t in teams}):
        entry = features.get(team)
        if not entry:
            warnings.append(f"{team}: no country feature row available for {target_year}.")
            continue
        if int(entry.get("country_features_stale", 0)):
            fy = entry.get("feature_year", "unknown")
            age = entry.get("feature_age_years", "unknown")
            warnings.append(f"{team}: country features are from {fy} ({age} years old for {target_year}).")
    return warnings


def wc_calibration_buckets(
    path: Path = ROOT / "backtest_walkforward_results.csv",
    bucket_size: float = 0.10,
) -> List[dict]:
    """Compute World-Cup-only top-prediction calibration buckets.

    Returns one dict per confidence bucket with ``n``, average confidence,
    empirical accuracy and smoothed accuracy. The smoothed value is conservative:
    sparse buckets are pulled toward their own average confidence instead of
    overreacting to one or two historical examples.
    """
    path = Path(path)
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    required = {"tournament", "confidence", "correct"}
    if not required.issubset(df.columns):
        return []
    wc = df[df["tournament"].astype(str).eq("FIFA World Cup")].copy()
    wc["confidence"] = pd.to_numeric(wc["confidence"], errors="coerce")
    wc = wc.dropna(subset=["confidence"])
    if wc.empty:
        return []
    buckets: List[dict] = []
    edges = np.arange(0.0, 1.0 + bucket_size, bucket_size)
    prior_strength = 12.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            part = wc[(wc["confidence"] >= lo) & (wc["confidence"] <= hi)]
        else:
            part = wc[(wc["confidence"] >= lo) & (wc["confidence"] < hi)]
        if part.empty:
            continue
        correct = part["correct"].map(parse_bool).astype(float)
        avg_conf = float(part["confidence"].mean())
        accuracy = float(correct.mean())
        smoothed = float((correct.sum() + prior_strength * avg_conf) / (len(part) + prior_strength))
        buckets.append({
            "lo": float(lo),
            "hi": float(hi),
            "n": int(len(part)),
            "avg_confidence": avg_conf,
            "accuracy": accuracy,
            "smoothed_accuracy": smoothed,
        })
    return buckets


def apply_wc_knockout_calibration(
    p_home: float,
    p_away: float,
    buckets: Sequence[dict],
    *,
    min_bucket_n: int = 5,
) -> Tuple[float, float, str]:
    """Calibrate a two-way knockout favorite probability using WC buckets."""
    total = float(p_home) + float(p_away)
    if total <= 0 or not np.isfinite(total):
        return 0.5, 0.5, "WC calibration skipped: invalid knockout probabilities."
    p_home = float(p_home) / total
    p_away = float(p_away) / total
    home_favored = p_home >= p_away
    favorite_prob = p_home if home_favored else p_away
    bucket = None
    for candidate in buckets or []:
        lo, hi = float(candidate.get("lo", 0.0)), float(candidate.get("hi", 1.0))
        if lo <= favorite_prob < hi or (hi >= 1.0 and favorite_prob <= hi):
            bucket = candidate
            break
    if not bucket or int(bucket.get("n", 0)) < min_bucket_n:
        return p_home, p_away, "WC calibration skipped: no sufficiently populated bucket."
    calibrated_favorite = float(bucket.get("smoothed_accuracy", favorite_prob))
    calibrated_favorite = float(np.clip(calibrated_favorite, 0.50, 0.98))
    calibrated_underdog = 1.0 - calibrated_favorite
    note = (
        f"WC bucket {bucket['lo']:.0%}-{bucket['hi']:.0%}: n={bucket['n']}, "
        f"avg_conf={bucket['avg_confidence']:.1%}, acc={bucket['accuracy']:.1%}, "
        f"smoothed={calibrated_favorite:.1%}"
    )
    if home_favored:
        return calibrated_favorite, calibrated_underdog, note
    return calibrated_underdog, calibrated_favorite, note


def stage_counts_for_year(year: int, total_matches: int) -> Tuple[int, int, int, int]:
    """Return counts for group, R16, QF, SF before final/tail handling."""
    if year == 1930:
        return (15, 0, 0, 2)
    if year == 1934:
        return (0, 9, 4, 2)
    if year == 1938:
        return (0, 10, 4, 2)
    if year == 1950:
        return (16, 0, 0, 0)
    if year == 1954:
        return (18, 0, 4, 2)
    if year == 1958:
        return (27, 0, 4, 2)
    if year in {1962, 1966, 1970}:
        return (24, 0, 4, 2)
    if year in {1974, 1978}:
        return (24, 12, 0, 0)
    if year == 1982:
        return (36, 12, 0, 2)
    if year in {1986, 1990, 1994}:
        return (36, 8, 4, 2)
    if total_matches >= 64:
        return (48, 8, 4, 2)
    if total_matches >= 52:
        return (36, 8, 4, 2)
    if total_matches >= 32:
        return (24, 0, 4, 2)
    return (max(total_matches - 3, 0), 0, 0, 2)


def select_final_match_index(wc_matches: pd.DataFrame, year: int) -> int:
    """Select the likely final row within a canonicalized World Cup match frame."""
    winner = harmonize_country(WC_WINNERS.get(int(year), ""))
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index()
    if df.empty:
        return -1
    latest_date = df["date"].max()
    same_day = df[df["date"] == latest_date]
    involving_winner = same_day[(same_day["home_team"] == winner) | (same_day["away_team"] == winner)]
    if not involving_winner.empty:
        return int(involving_winner.index[0])
    if int(year) == 1950:
        end_slice = df.tail(min(6, len(df)))
        candidates = end_slice[(end_slice["home_team"] == winner) | (end_slice["away_team"] == winner)]
        if not candidates.empty:
            return int(candidates.index[-1])
    return int(df.index[-1])


def assign_wc_stage_map(wc_matches: pd.DataFrame, year: int) -> Dict[int, int]:
    """Assign stage code per original dataframe index using historical WC schedule shapes."""
    if wc_matches.empty:
        return {}
    df = wc_matches.sort_values(["date", "home_team", "away_team"]).reset_index()
    total = len(df)
    stage_labels = ["group"] * total
    group_count, r16_count, qf_count, sf_count = stage_counts_for_year(year, total)
    idx = 0
    for label, count in (
        ("group", group_count),
        ("round_of_16", r16_count),
        ("quarterfinal", qf_count),
        ("semifinal", sf_count),
    ):
        for _ in range(count):
            if idx < total:
                stage_labels[idx] = label
                idx += 1
    while idx < total:
        stage_labels[idx] = "semifinal"
        idx += 1
    final_idx = select_final_match_index(wc_matches, year)
    if 0 <= final_idx < total:
        stage_labels[final_idx] = "final"
    return {int(df.iloc[i]["index"]): STAGE_TO_INT[stage_labels[i]] for i in range(total)}


def infer_world_cup_stage_map(results_df: pd.DataFrame) -> Dict[int, int]:
    """Infer historical World Cup stage labels for rows in a results dataframe."""
    if results_df.empty or "tournament" not in results_df.columns or "date" not in results_df.columns:
        return {}
    df = harmonize_columns(results_df.copy(), ["home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wc = df[df["tournament"].astype(str).eq("FIFA World Cup")].copy()
    if wc.empty:
        return {}
    stage_by_index: Dict[int, int] = {}
    for year, part in wc.groupby(wc["date"].dt.year):
        if pd.isna(year):
            continue
        stage_by_index.update(assign_wc_stage_map(part, int(year)))
    return stage_by_index


# Half-life (in days) for recency weighting of training rows. A 4-year half-life
# means a friendly from 4 years ago counts half as much as a match today, so the
# model leans on recent squad/era information without discarding deep history.
TIME_DECAY_HALF_LIFE_DAYS = 365 * 4

# Multiplier applied to draw rows (label == 1) so the softmax stops collapsing
# draws into home wins (the model's documented weakness). Combined with isotonic
# calibration, this targets log-loss/Brier on the under-predicted draw class.
DRAW_CLASS_WEIGHT = 1.6


def time_decay_weights(dates: Sequence, half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS) -> np.ndarray:
    """Recency weights ``0.5 ** (age_days / half_life_days)`` for each row."""
    d = pd.to_datetime(pd.Series(list(dates)), errors="coerce")
    max_date = d.max()
    age_days = (max_date - d).dt.days.fillna(0).clip(lower=0)
    return np.asarray(0.5 ** (age_days / float(half_life_days)), dtype=float)


def sample_weights(
    y: np.ndarray,
    dates: Sequence | None = None,
    *,
    time_decay: bool = True,
    draw_weight: float = DRAW_CLASS_WEIGHT,
) -> np.ndarray:
    """Combine recency decay and draw up-weighting into one sample-weight vector."""
    y = np.asarray(y)
    w = np.ones(len(y), dtype=float)
    if time_decay and dates is not None and len(dates) == len(y):
        w = w * time_decay_weights(dates)
    if draw_weight and draw_weight != 1.0:
        w = w * np.where(y == 1, draw_weight, 1.0)
    return w


class IsotonicProbabilityCalibrator:
    """Per-class isotonic recalibration of a fitted multiclass classifier.

    ``CalibratedClassifierCV(method="isotonic", cv="prefit")`` is the canonical
    tool, but it refits on the whole validation set and does not always preserve
    the exact 3-class ordering the rest of the pipeline assumes. This thin wrapper
    fits one ``IsotonicRegression`` per class on the chronological holdout, then
    renormalizes, which keeps the [home, draw, away] column order intact and lets
    callers keep using ``predict_proba``/``predict`` unchanged.
    """

    def __init__(self, base_model, classes):
        self.base_model = base_model
        self.classes_ = np.asarray(classes)
        self._iso = {}

    def fit(self, X_val, y_val, sample_weight=None):
        from sklearn.isotonic import IsotonicRegression

        probs = self.base_model.predict_proba(X_val)
        y_val = np.asarray(y_val)
        for j, cls in enumerate(self.classes_):
            target = (y_val == cls).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            try:
                iso.fit(probs[:, j], target, sample_weight=sample_weight)
                self._iso[j] = iso
            except Exception:
                self._iso[j] = None
        return self

    def predict_proba(self, X):
        probs = np.asarray(self.base_model.predict_proba(X), dtype=float)
        out = np.empty_like(probs)
        for j in range(probs.shape[1]):
            iso = self._iso.get(j)
            out[:, j] = iso.predict(probs[:, j]) if iso is not None else probs[:, j]
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return out / row_sums

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def drop_feature_columns(X: pd.DataFrame, exclude: Sequence[str] | None) -> pd.DataFrame:
    """Return a copy of ``X`` without the listed columns (no-op when empty)."""
    if not exclude:
        return X
    drop = [c for c in exclude if c in X.columns]
    return X.drop(columns=drop) if drop else X


def analyze_tradition_correlation(X: pd.DataFrame) -> dict[str, float]:
    """Pearson and Spearman correlation between ``tradition_diff`` and ``elo_diff``."""
    if "tradition_diff" not in X.columns or "elo_diff" not in X.columns:
        return {"pearson": float("nan"), "spearman": float("nan")}
    td = X["tradition_diff"].astype(float)
    ed = X["elo_diff"].astype(float)
    pearson = float(td.corr(ed, method="pearson"))
    spearman = float(td.corr(ed, method="spearman"))
    return {"pearson": pearson, "spearman": spearman}


def get_gbt_feature_importance(model, feature_names: Sequence[str], *, top_n: int = 15) -> pd.DataFrame:
    """Extract gain-based feature importances from XGBoost or LightGBM."""
    names = list(feature_names)
    importances: np.ndarray | None = None
    base = getattr(model, "base_model", model)
    if hasattr(base, "feature_importances_"):
        importances = np.asarray(base.feature_importances_, dtype=float)
    elif hasattr(base, "get_booster"):
        try:
            score = base.get_booster().get_score(importance_type="gain")
            importances = np.array([score.get(f, score.get(f"f{i}", 0.0)) for i, f in enumerate(names)], dtype=float)
        except Exception:
            pass
    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])
    total = importances.sum()
    if total > 0:
        importances = importances / total
    df = pd.DataFrame({"feature": names, "importance": importances})
    return df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def chronological_train_val_split(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    *,
    val_fraction: float = 0.2,
):
    """Chronological 80/20 split when dates are available, else stratified random holdout."""
    sw = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None
    if dates is not None and len(dates) == len(X):
        order = pd.Series(pd.to_datetime(dates, errors="coerce")).sort_values().index.to_numpy()
        split = max(1, int(len(order) * (1.0 - val_fraction)))
        if split >= len(order):
            split = len(order) - 1
        train_idx, val_idx = order[:split], order[split:]
        split_label = "chronological holdout"
    else:
        stratify = y if min(np.bincount(y.astype(int))) >= 2 else None
        idx = np.arange(len(X))
        train_idx, val_idx = train_test_split(
            idx, test_size=val_fraction, random_state=42, stratify=stratify
        )
        split_label = "random holdout"
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    sw_train = sw[train_idx] if sw is not None else None
    sw_val = sw[val_idx] if sw is not None else None
    return X_train, X_val, y_train, y_val, sw_train, sw_val, split_label, train_idx, val_idx


def _wrap_with_isotonic(model, X_val, y_val, classes, sample_weight_val, label: str):
    try:
        return IsotonicProbabilityCalibrator(model, classes).fit(
            X_val, y_val, sample_weight=sample_weight_val
        )
    except Exception as exc:  # pragma: no cover
        print(f"  {label} calibration failed ({exc}); using uncalibrated model")
        return model


def _report_validation_metrics(estimator, X_val, y_val, label: str, split_label: str, calibrate: bool):
    val_probs = estimator.predict_proba(X_val)
    val_pred = np.argmax(val_probs, axis=1)
    labels = sorted(np.unique(y_val).astype(int).tolist())
    acc = accuracy_score(y_val, val_pred)
    loss = log_loss(y_val, val_probs, labels=labels)
    cal_tag = " (isotonic-calibrated)" if calibrate else ""
    print(f"  {label} {split_label}{cal_tag} accuracy={acc:.3f}, log-loss={loss:.3f} ({len(X_val)} matches)")
    return {"accuracy": float(acc), "log_loss": float(loss), "n_val": int(len(X_val)), "split": split_label}


def create_xgb_classifier(**overrides):
    import xgboost as xgb

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "verbosity": 0,
    }
    params.update(overrides)
    return xgb.XGBClassifier(**params)


def create_lgbm_classifier(**overrides):
    from lightgbm import LGBMClassifier

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "random_state": 42,
        "verbosity": -1,
        **LGBM_DEFAULT_PARAMS,
    }
    params.update(overrides)
    return LGBMClassifier(**params)


def tune_lgbm_hyperopt(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    n_trials: int = 100,
    label: str = "LightGBM",
) -> dict:
    """Bayesian hyperparameter search (Hyperopt TPE) minimizing validation log-loss."""
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    from lightgbm import LGBMClassifier, early_stopping

    X_train, X_val, y_train, y_val, sw_train, sw_val, split_label, _, _ = chronological_train_val_split(
        X, y, dates, sample_weight
    )
    space = {
        "n_estimators": hp.qloguniform("n_estimators", np.log(100), np.log(1000), 1),
        "max_depth": hp.quniform("max_depth", 3, 12, 1),
        "num_leaves": hp.quniform("num_leaves", 15, 255, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        "min_child_samples": hp.quniform("min_child_samples", 5, 100, 1),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-8), np.log(10.0)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-8), np.log(10.0)),
        "min_split_gain": hp.uniform("min_split_gain", 0.0, 1.0),
    }

    def objective(raw_params):
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 42,
            "verbosity": -1,
            "n_estimators": int(raw_params["n_estimators"]),
            "max_depth": int(raw_params["max_depth"]),
            "num_leaves": int(raw_params["num_leaves"]),
            "learning_rate": float(raw_params["learning_rate"]),
            "min_child_samples": int(raw_params["min_child_samples"]),
            "subsample": float(raw_params["subsample"]),
            "colsample_bytree": float(raw_params["colsample_bytree"]),
            "reg_alpha": float(raw_params["reg_alpha"]),
            "reg_lambda": float(raw_params["reg_lambda"]),
            "min_split_gain": float(raw_params["min_split_gain"]),
        }
        model = LGBMClassifier(**params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "multi_logloss",
            "callbacks": [early_stopping(30, verbose=False)],
        }
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
            fit_kwargs["eval_sample_weight"] = [sw_val]
        try:
            model.fit(X_train, y_train, **fit_kwargs)
        except Exception:
            return {"loss": float("inf"), "status": STATUS_OK}
        probs = model.predict_proba(X_val)
        loss = log_loss(y_val, probs, labels=[0, 1, 2])
        return {"loss": loss, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(42),
        show_progressbar=False,
    )
    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "num_leaves": int(best["num_leaves"]),
        "learning_rate": float(best["learning_rate"]),
        "min_child_samples": int(best["min_child_samples"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "reg_alpha": float(best["reg_alpha"]),
        "reg_lambda": float(best["reg_lambda"]),
        "min_split_gain": float(best["min_split_gain"]),
    }
    best_loss = float(trials.best_trial["result"]["loss"])
    print(f"  {label} Hyperopt ({n_trials} trials, {split_label}): best log-loss={best_loss:.4f}")
    return best_params


def fit_lgbm_with_validation(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    label: str = "LightGBM",
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    calibrate: bool = False,
):
    """Fit a LightGBM classifier with chronological holdout and optional isotonic calibration."""
    from lightgbm import early_stopping

    X_train, X_val, y_train, y_val, sw_train, sw_val, split_label, _, _ = chronological_train_val_split(
        X, y, dates, sample_weight
    )
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "eval_metric": "multi_logloss",
        "callbacks": [early_stopping(30, verbose=False)],
    }
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train
        fit_kwargs["eval_sample_weight"] = [sw_val]
    try:
        model.fit(X_train, y_train, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("callbacks", None)
        if sw_train is not None:
            model.fit(X_train, y_train, sample_weight=sw_train, **{k: v for k, v in fit_kwargs.items() if k != "eval_sample_weight"})
        else:
            model.fit(X_train, y_train, **fit_kwargs)

    estimator = model
    if calibrate:
        classes = getattr(model, "classes_", np.array(sorted(np.unique(y).astype(int).tolist())))
        estimator = _wrap_with_isotonic(model, X_val, y_val, classes, sw_val, label)

    metrics = _report_validation_metrics(estimator, X_val, y_val, label, split_label, calibrate)
    return estimator, metrics


def fit_xgb_with_validation(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    label: str = "model",
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    calibrate: bool = False,
):
    """Fit an XGBoost classifier and report a chronological holdout when dates are available.

    When ``sample_weight`` is provided it is applied to the training split (and
    the early-stopping eval set), so time-decay and draw up-weighting flow into
    the fit. When ``calibrate=True`` the fitted model is wrapped in an isotonic
    calibrator fitted on the chronological holdout; the returned object still
    exposes ``predict``/``predict_proba`` and the original estimator stays
    available as ``.base_model``.
    """
    X_train, X_val, y_train, y_val, sw_train, sw_val, split_label, _, _ = chronological_train_val_split(
        X, y, dates, sample_weight
    )

    fit_kwargs = {"eval_set": [(X_val, y_val)], "verbose": False}
    if sw_train is not None:
        fit_kwargs["sample_weight"] = sw_train
    try:
        model.fit(X_train, y_train, early_stopping_rounds=30, **fit_kwargs)
    except TypeError:
        try:
            model.set_params(early_stopping_rounds=30)
            model.fit(X_train, y_train, **fit_kwargs)
        except TypeError:
            if sw_train is not None:
                model.fit(X_train, y_train, sample_weight=sw_train)
            else:
                model.fit(X_train, y_train)

    estimator = model
    if calibrate:
        classes = getattr(model, "classes_", np.array(sorted(np.unique(y).astype(int).tolist())))
        estimator = _wrap_with_isotonic(model, X_val, y_val, classes, sw_val, label)

    metrics = _report_validation_metrics(estimator, X_val, y_val, label, split_label, calibrate)
    return estimator, metrics


def fit_gbt_with_validation(
    model_type: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    dates: Sequence | None = None,
    sample_weight: Sequence | None = None,
    calibrate: bool = True,
    lgbm_params: dict | None = None,
    hyperopt_trials: int = 0,
    label: str | None = None,
):
    """Train XGBoost or Hyperopt-tuned LightGBM with shared validation protocol."""
    model_type = model_type.lower()
    if model_type == "lgbm":
        params = dict(LGBM_DEFAULT_PARAMS)
        if lgbm_params:
            params.update(lgbm_params)
        if hyperopt_trials > 0:
            tuned = tune_lgbm_hyperopt(
                X, y, dates=dates, sample_weight=sample_weight, n_trials=hyperopt_trials,
                label=label or "LightGBM",
            )
            params.update(tuned)
        model = create_lgbm_classifier(**params)
        return fit_lgbm_with_validation(
            model, X, y, label=label or "LightGBM", dates=dates,
            sample_weight=sample_weight, calibrate=calibrate,
        )
    model = create_xgb_classifier()
    return fit_xgb_with_validation(
        model, X, y, label=label or "XGBoost", dates=dates,
        sample_weight=sample_weight, calibrate=calibrate,
    )


class DixonColesModel:
    """Dixon-Coles bivariate-Poisson goal model.

    Estimates per-team attack/defense strengths plus a global home advantage and
    a low-score dependence parameter ``rho``. W/D/L probabilities come from a
    truncated scoreline grid, which naturally produces realistic *draw* rates
    (the XGBoost model's documented weakness). Strengths are fit by weighted
    Poisson MLE; recent matches are up-weighted via the shared time-decay.
    """

    def __init__(self, attack, defense, home_adv, rho, base, teams):
        self.attack = attack
        self.defense = defense
        self.home_adv = float(home_adv)
        self.rho = float(rho)
        self.base = float(base)
        self.teams = set(teams)

    def _lambdas(self, home: str, away: str, neutral: bool = True):
        home = harmonize_country(home)
        away = harmonize_country(away)
        ah = self.attack.get(home, 0.0)
        dh = self.defense.get(home, 0.0)
        aa = self.attack.get(away, 0.0)
        da = self.defense.get(away, 0.0)
        adv = 0.0 if neutral else self.home_adv
        lam_home = np.exp(self.base + adv + ah - da)
        lam_away = np.exp(self.base + aa - dh)
        # Guard against pathological strengths producing huge expected goals.
        return float(np.clip(lam_home, 1e-3, 8.0)), float(np.clip(lam_away, 1e-3, 8.0))

    @staticmethod
    def _tau(i, j, lam, mu, rho):
        if i == 0 and j == 0:
            return 1.0 - lam * mu * rho
        if i == 0 and j == 1:
            return 1.0 + lam * rho
        if i == 1 and j == 0:
            return 1.0 + mu * rho
        if i == 1 and j == 1:
            return 1.0 - rho
        return 1.0

    def outcome_probs(self, home: str, away: str, neutral: bool = True, max_goals: int = 10) -> np.ndarray:
        """Return [P(home win), P(draw), P(away win)] from the scoreline grid."""
        lam, mu = self._lambdas(home, away, neutral)
        i = np.arange(0, max_goals + 1)
        # Poisson PMFs.
        from math import lgamma

        log_fact = np.array([lgamma(k + 1) for k in i])
        ph = np.exp(i * np.log(lam) - lam - log_fact)
        pa = np.exp(i * np.log(mu) - mu - log_fact)
        grid = np.outer(ph, pa)
        # Dixon-Coles low-score correction on the four 0/1 cells.
        for a in (0, 1):
            for b in (0, 1):
                grid[a, b] *= self._tau(a, b, lam, mu, self.rho)
        grid = np.clip(grid, 0.0, None)
        total = grid.sum()
        if total <= 0:
            return np.array([1 / 3, 1 / 3, 1 / 3])
        grid /= total
        p_home = np.tril(grid, -1).sum()   # home goals > away goals
        p_away = np.triu(grid, 1).sum()    # away goals > home goals
        p_draw = np.trace(grid)
        return np.array([p_home, p_draw, p_away])


def fit_dixon_coles(
    results_df: pd.DataFrame,
    *,
    max_year: int | None = None,
    half_life_days: int = TIME_DECAY_HALF_LIFE_DAYS,
    min_matches: int = 8,
    exclude_2026_wc: bool = True,
) -> "DixonColesModel":
    """Fit a Dixon-Coles model by weighted Poisson MLE on historical results.

    Falls back to a closed-form attack/defense estimate (no ``rho``) when SciPy
    is unavailable, so the ensemble still works without the optional dependency.
    """
    try:
        from scipy.optimize import minimize
        _have_scipy = True
    except Exception:
        minimize = None
        _have_scipy = False

    df = harmonize_columns(results_df.copy(), ["home_team", "away_team"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if exclude_2026_wc:
        df = df[~((df["date"].dt.year == 2026) & (df["tournament"] == "FIFA World Cup"))]
    df = df.dropna(subset=["home_score", "away_score", "date"])
    if max_year is not None:
        df = df[df["date"].dt.year <= max_year]
    # Keep recent history meaningful; ~last 12 years covers several WC cycles
    # while keeping the parameter count (2 * n_teams + 3) tractable for L-BFGS-B.
    cutoff = df["date"].max() - pd.Timedelta(days=365 * 12)
    df = df[df["date"] >= cutoff].copy()

    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    teams = sorted(counts[counts >= min_matches].index.tolist())
    df = df[df["home_team"].isin(teams) & df["away_team"].isin(teams)].reset_index(drop=True)
    if df.empty or len(teams) < 2:
        return DixonColesModel({}, {}, 0.0, 0.0, 0.0, teams)

    idx = {t: k for k, t in enumerate(teams)}
    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hg = df["home_score"].to_numpy(dtype=float)
    ag = df["away_score"].to_numpy(dtype=float)
    neutral = df.get("neutral")
    is_neutral = (neutral.map(parse_bool).to_numpy() if neutral is not None
                  else np.ones(len(df), dtype=bool))
    w = time_decay_weights(df["date"], half_life_days)

    n = len(teams)

    if not _have_scipy:
        # Closed-form weighted estimate: attack ~ log(weighted GF / mean), and
        # defense ~ -log(weighted GA / mean). No low-score (rho) correction.
        gf_sum = np.zeros(n); ga_sum = np.zeros(n); wt = np.zeros(n)
        for k in range(len(df)):
            gf_sum[hi[k]] += w[k] * hg[k]; ga_sum[hi[k]] += w[k] * ag[k]; wt[hi[k]] += w[k]
            gf_sum[ai[k]] += w[k] * ag[k]; ga_sum[ai[k]] += w[k] * hg[k]; wt[ai[k]] += w[k]
        wt = np.clip(wt, 1e-9, None)
        gf_rate = gf_sum / wt
        ga_rate = ga_sum / wt
        base = float(np.log(np.clip(np.average(gf_rate, weights=wt), 1e-3, None)))
        attack = np.log(np.clip(gf_rate, 1e-3, None)) - base
        attack = attack - attack.mean()
        defense = -(np.log(np.clip(ga_rate, 1e-3, None)) - base)
        defense = defense - defense.mean()
        home_adv = float(np.log(np.clip(hg.mean(), 1e-3, None) / np.clip(ag.mean(), 1e-3, None)))
        return DixonColesModel(
            attack={t: float(attack[idx[t]]) for t in teams},
            defense={t: float(defense[idx[t]]) for t in teams},
            home_adv=max(0.0, home_adv), rho=0.0, base=base, teams=teams,
        )
    # Params: attack[n], defense[n], home_adv, rho, base. Identifiability: mean
    # attack fixed to 0 via a soft penalty.
    def unpack(p):
        return p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1], p[2 * n + 2]

    def neg_log_like(p):
        attack, defense, home_adv, rho, base = unpack(p)
        adv = np.where(is_neutral, 0.0, home_adv)
        log_lh = base + adv + attack[hi] - defense[ai]
        log_la = base + attack[ai] - defense[hi]
        lam = np.exp(np.clip(log_lh, -4, 3))
        mu = np.exp(np.clip(log_la, -4, 3))
        ll = hg * np.log(lam) - lam + ag * np.log(mu) - mu
        # Dixon-Coles low-score correction (vectorized over the 0/1 cells).
        tau = np.ones(len(df))
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
        tau[m01] = 1.0 + lam[m01] * rho
        tau[m10] = 1.0 + mu[m10] * rho
        tau[m11] = 1.0 - rho
        tau = np.clip(tau, 1e-6, None)
        ll = ll + np.log(tau)
        penalty = 1e3 * (attack.mean() ** 2)  # anchor mean attack at 0
        return -np.sum(w * ll) + penalty

    p0 = np.zeros(2 * n + 3)
    p0[2 * n] = 0.25   # home advantage
    p0[2 * n + 1] = -0.05  # rho
    p0[2 * n + 2] = 0.0    # base
    bounds = [(-3, 3)] * (2 * n) + [(-1.0, 1.0), (-0.2, 0.2), (-2.0, 2.0)]
    try:
        res = minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200})
        p = res.x
    except Exception:
        p = p0
    attack, defense, home_adv, rho, base = unpack(p)
    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defense={t: float(defense[idx[t]]) for t in teams},
        home_adv=float(home_adv), rho=float(rho), base=float(base), teams=teams,
    )


def blend_probabilities(p_xgb: np.ndarray, p_poisson: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend ``alpha * xgb + (1 - alpha) * poisson``, renormalized."""
    p = alpha * np.asarray(p_xgb, dtype=float) + (1.0 - alpha) * np.asarray(p_poisson, dtype=float)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def empty_group_table() -> Dict[str, Dict[str, int]]:
    return {"pts": 0, "gd": 0, "gf": 0, "ga": 0, "played": 0}


def group_for_match(home: str, away: str) -> Optional[str]:
    home = harmonize_country(home)
    away = harmonize_country(away)
    for group, teams in GROUP_2026_TEAMS.items():
        if home in teams and away in teams:
            return group
    return None


def wc2026_stage_for_match(match_date) -> int:
    """Map a 2026 World Cup fixture date onto the training stage code."""
    date = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(date):
        return WC2026_STAGE_TO_TRAIN["group"]
    if date < pd.Timestamp("2026-06-28"):
        return WC2026_STAGE_TO_TRAIN["group"]
    if date <= pd.Timestamp("2026-07-03"):
        return WC2026_STAGE_TO_TRAIN["round_of_32"]
    if date <= pd.Timestamp("2026-07-07"):
        return WC2026_STAGE_TO_TRAIN["round_of_16"]
    if date <= pd.Timestamp("2026-07-11"):
        return WC2026_STAGE_TO_TRAIN["quarterfinal"]
    if date <= pd.Timestamp("2026-07-15"):
        return WC2026_STAGE_TO_TRAIN["semifinal"]
    if date == pd.Timestamp("2026-07-18"):
        return WC2026_STAGE_TO_TRAIN["third_place"]
    if date == pd.Timestamp("2026-07-19"):
        return WC2026_STAGE_TO_TRAIN["final"]
    return WC2026_STAGE_TO_TRAIN["group"]


def apply_group_result(groups, group: str, home: str, away: str, home_score: int, away_score: int) -> None:
    home = harmonize_country(home)
    away = harmonize_country(away)
    for team, gf, ga in [(home, home_score, away_score), (away, away_score, home_score)]:
        groups[group][team]["played"] += 1
        groups[group][team]["gf"] += gf
        groups[group][team]["ga"] += ga
        groups[group][team]["gd"] += gf - ga
    if home_score > away_score:
        groups[group][home]["pts"] += 3
    elif home_score < away_score:
        groups[group][away]["pts"] += 3
    else:
        groups[group][home]["pts"] += 1
        groups[group][away]["pts"] += 1


def build_2026_group_state(results: pd.DataFrame):
    """Build 2026 tables from completed CSV rows and list only unplayed fixtures."""
    groups = {
        group: {team: empty_group_table() for team in teams}
        for group, teams in GROUP_2026_TEAMS.items()
    }
    wc26 = results.copy()
    wc26["date"] = pd.to_datetime(wc26["date"])
    wc26 = wc26[(wc26["tournament"] == "FIFA World Cup") & (wc26["date"].dt.year == 2026)].copy()
    wc26 = harmonize_columns(wc26, ["home_team", "away_team"])
    remaining = []
    for _, row in wc26.sort_values("date").iterrows():
        group = group_for_match(row["home_team"], row["away_team"])
        if group is None:
            continue
        if pd.notna(row["home_score"]) and pd.notna(row["away_score"]):
            apply_group_result(groups, group, row["home_team"], row["away_team"], int(row["home_score"]), int(row["away_score"]))
        else:
            remaining.append((row["date"], row["home_team"], row["away_team"], group))
    return groups, remaining


def sorted_group_standings(group_data: Dict[str, Dict[str, int]]) -> List[Tuple[str, int, int, int]]:
    return sorted(
        [(team, data["pts"], data["gd"], data["gf"]) for team, data in group_data.items()],
        key=lambda item: (-item[1], -item[2], -item[3], item[0]),
    )


def rank_third_place_teams(standings_by_group: Dict[str, List[Tuple[str, int, int, int]]]):
    thirds = [
        (group, table[2][0], table[2][1], table[2][2], table[2][3])
        for group, table in standings_by_group.items()
    ]
    return sorted(thirds, key=lambda item: (-item[2], -item[3], -item[4], item[0]))


def build_round_of_32(gw: Dict[str, str], gr: Dict[str, str], best_thirds: Sequence[Tuple[str, str, int, int, int]]):
    """Build the 2026 R32, replacing all third-place placeholders with actual teams."""
    r32_base = [
        gr["A"], gr["B"],
        gw["E"], None,
        gw["F"], gr["C"],
        gw["C"], gr["F"],
        gw["I"], None,
        gr["E"], gr["I"],
        gw["A"], None,
        gw["L"], None,
        gw["D"], None,
        gw["G"], None,
        gr["K"], gr["L"],
        gw["H"], gr["J"],
        gw["B"], None,
        gw["J"], gr["H"],
        gw["K"], None,
        gr["D"], gr["G"],
    ]

    third_by_group = {group: team for group, team, *_ in best_thirds}
    combo = tuple(sorted(third_by_group))
    try:
        slot_by_group = THIRD_PLACE_ALLOCATION_MATRIX[combo]
    except KeyError as exc:
        raise ValueError(f"Unsupported third-place group combination: {','.join(combo)}") from exc

    for group, slot in slot_by_group.items():
        r32_base[slot] = third_by_group[group]

    missing = [slot for slot in THIRD_SLOTS if r32_base[slot] is None]
    if missing:
        raise ValueError(f"Could not assign third-place teams to R32 slots: {missing}")

    labels = [f"Match {match_num}" for match_num in range(73, 89)]
    return [
        (labels[i // 2], r32_base[i], r32_base[i + 1])
        for i in range(0, len(r32_base), 2)
    ]

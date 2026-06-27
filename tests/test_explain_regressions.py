import unittest
from collections import defaultdict

import pandas as pd

import explain_match
import incremental_predictor
import shared


class ExplainRegressionTests(unittest.TestCase):
    def test_world_cup_history_uses_final_tournament_counts(self):
        brazil = explain_match.world_cup_history("Brazil", through_year=2022)
        self.assertEqual(brazil["participations"], 22)
        self.assertEqual(brazil["titles"], 5)

    def test_shared_feature_row_is_not_zero_vector(self):
        state = defaultdict(shared.make_team_state)
        state["Brazil"]["elo"] = 1850
        state["Japan"]["elo"] = 1650
        key = tuple(sorted(["Brazil", "Japan"]))
        state["Brazil"]["h2h"][key].update({"matches": 14, "wins": 11, "draws": 2, "losses": 1, "gf": 35, "ga": 10})
        history = shared.load_country_feature_history()
        country_features = shared.country_features_for_year(history, 2022)

        row = shared.compute_match_features(
            "Brazil",
            "Japan",
            state,
            country_features,
            1,
            pd.Timestamp("2026-06-29"),
        )

        self.assertGreater(row["elo"], 1000)
        self.assertGreater(row["elo_opponent"], 1000)
        self.assertGreater(row["elo_sum"], 2500)
        self.assertEqual(row["h2h_matches"], 14)
        self.assertEqual(row["stage"], 1)

    def test_incremental_country_features_drop_leakage_columns(self):
        country_df = pd.read_csv(shared.DATA_DIR / "world_cup_predictors_dataset.csv")
        _, feature_cols = incremental_predictor.build_country_feature_lookup(country_df)
        forbidden = {
            "gdp_per_capita_vs_winner",
            "population_vs_winner",
            "total_goals_in_tournament",
            "avg_goals_per_match",
        }
        self.assertTrue(forbidden.isdisjoint(feature_cols))


if __name__ == "__main__":
    unittest.main()

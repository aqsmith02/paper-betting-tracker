"""
test_data_processing.py

Unit tests for data_processing.py module.

UNEXPECTED BEHAVIOR: Prettify headers makes every column lowercase except the first letter of a word, changing
bookmaker names like "GTbets" to "Gtbets". This is unintended, but does not seem to have any negative impact
on functionality.

Author: Andrew Smith
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch

from codebase.find_bets.data_processing import (
    _find_bookmaker_columns,
    _remove_non_target_bookmakers,
    _add_metadata,
    _clean_odds_data,
    _min_bookmaker_filter,
    _max_odds_filter,
    _all_outcomes_present_filter,
    _prettify_column_headers,
    process_target_odds_data,
    calculate_vigfree_probabilities,
)


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Data from an API pull on 2025-09-25
        self.unprocessed = pd.read_csv("testing/find_bets/data_processing/unprocessed.csv")
        self.pre_metadata_check = pd.read_csv("testing/find_bets/data_processing/pre_metadata.csv")
        self.pre_clean = pd.read_csv("testing/find_bets/data_processing/pre_clean.csv")
        self.pre_min_bookmaker_check = pd.read_csv(
            "testing/find_bets/data_processing/pre_min_bookmaker_check.csv"
        )
        self.pre_max_odds_check = pd.read_csv(
            "testing/find_bets/data_processing/pre_max_odds_check.csv"
        )
        self.pre_outcome_check = pd.read_csv("testing/find_bets/data_processing/pre_outcome_check.csv")
        self.pre_prettify = pd.read_csv("testing/find_bets/data_processing/pre_prettify.csv")
        self.processed = pd.read_csv("testing/find_bets/data_processing/processed.csv")

    def test_find_bookmaker_columns(self):
        """Test _find_bookmaker_columns function."""
        df = self.unprocessed

        # Test basic functionality
        bookmaker_cols = _find_bookmaker_columns(df)
        expected_cols = [
            "GTbets",
            "Bovada",
            "PointsBet (AU)",
            "DraftKings",
            "BetMGM",
            "Nordic Bet",
            "Betsson",
            "Caesars",
            "FanDuel",
            "Paddy Power",
            "Smarkets",
            "LeoVegas",
            "Unibet",
            "BetRivers",
            "Unibet (SE)",
            "Unibet (NL)",
            "Unibet (IT)",
            "Pinnacle",
            "Matchbook",
            "Betway",
            "Betfair Sportsbook",
            "Betclic (FR)",
            "TAB",
            "Tipico",
            "Bet Victor",
            "Betfair",
            "Coolbet",
            "Fanatics",
            "888sport",
            "MyBookie.ag",
            "Winamax (FR)",
            "Winamax (DE)",
            "Virgin Bet",
            "Casumo",
            "LiveScore Bet",
            "Grosvenor",
            "LeoVegas (SE)",
            "TABtouch",
            "Ladbrokes",
            "Marathon Bet",
            "PlayUp",
            "LowVig.ag",
            "Coral",
            "Neds",
            "1xBet",
            "BetOnline.ag",
            "William Hill",
            "Betr",
            "SportsBet",
            "BetUS",
            "BoyleSports",
            "Everygame",
        ]
        self.assertEqual(set(bookmaker_cols), set(expected_cols))

    def test_find_bookmaker_columns_exclude(self):
        """Test _find_bookmaker_columns with exclude_columns parameter."""
        df = self.unprocessed

        # Test with exclude_columns
        bookmaker_cols_excluded = _find_bookmaker_columns(
            df, exclude_columns=["GTbets"]
        )
        expected_excluded = [
            "Bovada",
            "PointsBet (AU)",
            "DraftKings",
            "BetMGM",
            "Nordic Bet",
            "Betsson",
            "Caesars",
            "FanDuel",
            "Paddy Power",
            "Smarkets",
            "LeoVegas",
            "Unibet",
            "BetRivers",
            "Unibet (SE)",
            "Unibet (NL)",
            "Unibet (IT)",
            "Pinnacle",
            "Matchbook",
            "Betway",
            "Betfair Sportsbook",
            "Betclic (FR)",
            "TAB",
            "Tipico",
            "Bet Victor",
            "Betfair",
            "Coolbet",
            "Fanatics",
            "888sport",
            "MyBookie.ag",
            "Winamax (FR)",
            "Winamax (DE)",
            "Virgin Bet",
            "Casumo",
            "LiveScore Bet",
            "Grosvenor",
            "LeoVegas (SE)",
            "TABtouch",
            "Ladbrokes",
            "Marathon Bet",
            "PlayUp",
            "LowVig.ag",
            "Coral",
            "Neds",
            "1xBet",
            "BetOnline.ag",
            "William Hill",
            "Betr",
            "SportsBet",
            "BetUS",
            "BoyleSports",
            "Everygame",
        ]
        self.assertEqual(set(bookmaker_cols_excluded), set(expected_excluded))

    def test_remove_non_target_bookmakers(self):
        """Test _remove_non_target_bookmakers function."""
        df = self.unprocessed
        result_df = _remove_non_target_bookmakers(df)

        # Check that only target bookmaker columns remain
        remaining_bms = _find_bookmaker_columns(result_df)
        expected_bms = [
            "DraftKings",
            "BetMGM",
            "Caesars",
            "FanDuel",
            "Pinnacle",
            "Fanatics",
            "BetOnline.ag",
        ]
        self.assertEqual(set(remaining_bms), set(expected_bms))

    def test_add_metadata(self):
        """Test _add_metadata function."""
        df = self.pre_metadata_check
        result_df = _add_metadata(df)

        # Check that metadata columns are added
        self.assertIn("Best Odds", result_df.columns)
        self.assertIn("Best Bookmaker", result_df.columns)
        self.assertIn("Result", result_df.columns)
        self.assertIn("Outcomes", result_df.columns)

        # Check Best Odds calculation
        self.assertEqual(result_df["Best Odds"].iloc[0], 101.0)
        self.assertEqual(result_df["Best Odds"].iloc[4], 3.35)
        self.assertEqual(result_df["Best Odds"].iloc[9], 60.0)
        self.assertEqual(result_df["Best Odds"].iloc[14], 2.2)

        # Check Best Bookmaker identification
        self.assertEqual(result_df["Best Bookmaker"].iloc[0], "PointsBet (AU)")
        self.assertEqual(result_df["Best Bookmaker"].iloc[4], "Bovada")
        self.assertEqual(result_df["Best Bookmaker"].iloc[9], "Nordic Bet")
        self.assertEqual(result_df["Best Bookmaker"].iloc[14], "Pinnacle")

        # Check Result initialization
        self.assertTrue(all(result_df["Result"] == "Not Found"))

        # Check Outcomes calculation
        self.assertEqual(result_df["Outcomes"].iloc[0], 2)
        self.assertEqual(result_df["Outcomes"].iloc[1], 2)
        self.assertEqual(result_df["Outcomes"].iloc[4], 3)
        self.assertEqual(result_df["Outcomes"].iloc[5], 3)
        self.assertEqual(result_df["Outcomes"].iloc[6], 3)

    def test_add_metadata_with_best_odds_bms(self):
        """Test _add_metadata function with best_odds_bms parameter."""
        df = self.pre_metadata_check
        best_odds_bms = ["DraftKings", "BetMGM", "Caesars", "FanDuel","Fanatics"]
        result_df = _add_metadata(df, best_odds_bms=best_odds_bms)

        # Check that metadata columns are added
        self.assertIn("Best Odds", result_df.columns)
        self.assertIn("Best Bookmaker", result_df.columns)
        self.assertIn("Result", result_df.columns)
        self.assertIn("Outcomes", result_df.columns)

        # Check Best Odds calculation
        self.assertTrue(np.isnan(result_df["Best Odds"].iloc[0]))  # No valid bookmaker in list
        self.assertEqual(result_df["Best Odds"].iloc[2], 8.5)  # From Fanduel
        self.assertEqual(result_df["Best Odds"].iloc[9], 36)  # From Fanduel
        self.assertEqual(result_df["Best Odds"].iloc[14], 2.15)  # From Caesars

        # Check Best Bookmaker identification
        self.assertTrue(np.isnan(result_df["Best Bookmaker"].iloc[0]))  
        self.assertEqual(result_df["Best Bookmaker"].iloc[2], "FanDuel")
        self.assertEqual(result_df["Best Bookmaker"].iloc[9], "FanDuel")
        self.assertEqual(result_df["Best Bookmaker"].iloc[14], "Caesars")

    def test_clean_odds_data(self):
        """Test _clean_odds_data function."""
        df = self.pre_clean
        result_df = _clean_odds_data(df)

        # Check that odds equal to 1.0 are replaced with NaN
        self.assertTrue(pd.isna(result_df["PointsBet (AU)"].iloc[1]))
        self.assertTrue(pd.isna(result_df["PointsBet (AU)"].iloc[10]))
        self.assertTrue(pd.isna(result_df["Nordic Bet"].iloc[10]))
        self.assertTrue(pd.isna(result_df["Betsson"].iloc[10]))
        self.assertTrue(pd.isna(result_df["FanDuel"].iloc[10]))
        self.assertTrue(pd.isna(result_df["Paddy Power"].iloc[10]))
        self.assertTrue(pd.isna(result_df["Tipico"].iloc[10]))

        # Check that valid odds remain unchanged
        self.assertEqual(result_df["GTbets"].iloc[1], 1.01)
        self.assertEqual(result_df["Bovada"].iloc[1], 1.48)
        self.assertEqual(result_df["GTbets"].iloc[10], 1.03)
        self.assertEqual(result_df["Bovada"].iloc[10], 1.44)
        self.assertEqual(result_df["BetMGM"].iloc[10], 1.01)
        self.assertEqual(result_df["Caesars"].iloc[10], 1.01)
        self.assertEqual(result_df["Pinnacle"].iloc[10], 1.07)
        self.assertEqual(result_df["Betway"].iloc[10], 1.03)
        self.assertEqual(result_df["Bet Victor"].iloc[10], 1.01)
        self.assertEqual(result_df["Coolbet"].iloc[10], 1.01)
        self.assertEqual(result_df["MyBookie.ag"].iloc[10], 1.03)
        self.assertEqual(result_df["Winamax (FR)"].iloc[10], 1.01)
        self.assertEqual(result_df["Winamax (DE)"].iloc[10], 1.01)

    def test_min_bookmaker_filter(self):
        """Test _min_bookmaker_filter function."""
        df = self.pre_min_bookmaker_check
        result_df = _min_bookmaker_filter(df)

        filtered_out_matches = [
            "Army Black Knights @ East Carolina Pirates",
            "Avai @ Chapecoense",
        ]

        # Confirm the filtered out matches are actually filtered
        self.assertTrue(
            all(match not in filtered_out_matches for match in result_df["match"])
        )

        # Confirm only no other rows are removed
        self.assertEqual(len(result_df), 22)


    def test_max_odds_filter(self):
        """Test _max_odds_filter function."""
        df = self.pre_max_odds_check
        df = _add_metadata(df)
        result_df = _max_odds_filter(df)

        # Confirm that the 5th row is gone
        self.assertNotIn(
            "Colorado Rockies", result_df["team"].values
        )

        # Confirm rest of rows remain
        self.assertEqual(len(result_df), 21)

    def test_all_outcomes_present_filter(self):
        """Test _all_outcomes_present_filter function."""
        df = self.pre_outcome_check
        result_df = _all_outcomes_present_filter(df)

        # Confirm that the incomplete match is gone
        self.assertNotIn(
            "Colorado Rockies @ Seattle Mariners", result_df["match"].values
        )

        # Confirm no other rows are removed
        self.assertEqual(len(result_df), 20)

    def test_prettify_column_headers(self):
        """Test _prettify_column_headers function."""
        df = self.pre_prettify
        result_df = _prettify_column_headers(df)

        expected_cols = [
            "Match",
            "League",
            "Start Time",
            "Team",
            "Gtbets",
            "Bovada",
            "Pointsbet (Au)",
            "Draftkings",
            "Betmgm",
            "Nordic Bet",
            "Betsson",
            "Caesars",
            "Fanduel",
            "Paddy Power",
            "Leovegas",
            "Unibet",
            "Betrivers",
            "Unibet (Se)",
            "Unibet (Nl)",
            "Unibet (It)",
            "Pinnacle",
            "Betway",
            "Betclic (Fr)",
            "Tab",
            "Tipico",
            "Bet Victor",
            "Coolbet",
            "Fanatics",
            "888Sport",
            "Mybookie.Ag",
            "Winamax (Fr)",
            "Winamax (De)",
            "Virgin Bet",
            "Casumo",
            "Livescore Bet",
            "Grosvenor",
            "Leovegas (Se)",
            "Tabtouch",
            "Ladbrokes",
            "Marathon Bet",
            "Playup",
            "Lowvig.Ag",
            "Coral",
            "Neds",
            "1Xbet",
            "Betonline.Ag",
            "William Hill",
            "Betr",
            "Sportsbet",
            "Betus",
            "Boylesports",
            "Everygame",
            "Best Odds",
            "Best Bookmaker",
            "Result",
            "Outcomes",
        ]
        self.assertEqual(list(result_df.columns), expected_cols)

    def test_process_target_odds_data_integration(self):
        """Test the complete process_odds_data pipeline with target bookmakers only."""
        df = self.unprocessed
        result_df = process_target_odds_data(df)

        # Check that all processing steps were applied
        # 1. Non-target bookmakers should be removed
        remaining_bms = _find_bookmaker_columns(result_df)
        expected_bms = [
            "Draftkings",
            "Betmgm",
            "Caesars",
            "Fanduel",
            "Pinnacle",
            "Fanatics",
            "Betonline.Ag",
        ]
        self.assertEqual(set(remaining_bms), set(expected_bms))

        # 2. Metadata should be added
        self.assertIn("Best Odds", result_df.columns)
        self.assertIn("Best Bookmaker", result_df.columns)
        self.assertIn("Result", result_df.columns)
        self.assertIn("Outcomes", result_df.columns)

        # 3. Check data integrity
        self.assertEqual(len(result_df), 20)

        # 4. Column headers should be prettified
        self.assertIn("Match", result_df.columns)
        self.assertIn("League", result_df.columns)
        self.assertIn("Start Time", result_df.columns)
        self.assertIn("Team", result_df.columns)

    def test_calculate_vigfree_probabilities(self):
        """Test calculate_vigfree_probabilities function."""
        df = self.processed
        result_df = calculate_vigfree_probabilities(df)

        expected_cols = [
            "Vigfree Gtbets",
            "Vigfree Bovada",
            "Vigfree Pointsbet (Au)",
            "Vigfree Draftkings",
            "Vigfree Betmgm",
            "Vigfree Nordic Bet",
            "Vigfree Betsson",
            "Vigfree Caesars",
            "Vigfree Fanduel",
            "Vigfree Paddy Power",
            "Vigfree Leovegas",
            "Vigfree Unibet",
            "Vigfree Betrivers",
            "Vigfree Unibet (Se)",
            "Vigfree Unibet (Nl)",
            "Vigfree Unibet (It)",
            "Vigfree Pinnacle",
            "Vigfree Betway",
            "Vigfree Betclic (Fr)",
            "Vigfree Tab",
            "Vigfree Tipico",
            "Vigfree Bet Victor",
            "Vigfree Coolbet",
            "Vigfree Fanatics",
            "Vigfree 888Sport",
            "Vigfree Mybookie.Ag",
            "Vigfree Winamax (Fr)",
            "Vigfree Winamax (De)",
            "Vigfree Virgin Bet",
            "Vigfree Casumo",
            "Vigfree Livescore Bet",
            "Vigfree Grosvenor",
            "Vigfree Leovegas (Se)",
            "Vigfree Tabtouch",
            "Vigfree Ladbrokes",
            "Vigfree Marathon Bet",
            "Vigfree Playup",
            "Vigfree Lowvig.Ag",
            "Vigfree Coral",
            "Vigfree Neds",
            "Vigfree 1Xbet",
            "Vigfree Betonline.Ag",
            "Vigfree William Hill",
            "Vigfree Betr",
            "Vigfree Sportsbet",
            "Vigfree Betus",
            "Vigfree Boylesports",
            "Vigfree Everygame",
        ]
        missing_cols = [col for col in expected_cols if col not in result_df.columns]
        if missing_cols:
            print(f"Missing expected columns: {missing_cols}")
        self.assertTrue(all(col in result_df.columns for col in expected_cols))

        # Test Gtbets, Seattle Seahawks @ Arizona Cardinals
        vf1 = result_df["Vigfree Gtbets"].iloc[0]
        vf2 = result_df["Vigfree Gtbets"].iloc[1]
        self.assertAlmostEqual(vf1, 0.1114583333, places=3)
        self.assertAlmostEqual(vf2, 0.8885416667, places=3)

        # Test Bovada, Kansas City Royals @ Los Angeles Angels
        vf3 = result_df["Vigfree Bovada"].iloc[2]
        vf4 = result_df["Vigfree Bovada"].iloc[3]
        self.assertAlmostEqual(vf3, 0.5, places=3)
        self.assertAlmostEqual(vf4, 0.5, places=3)

        # Create test data for 3 game outcomes
        three_outcome_df = pd.DataFrame(
            {
                "Match": ["Team1 @ Team2", "Team1 @ Team2", "Team1 @ Team2"],
                "Team": ["Team1", "Team2", "Draw"],
                "Bookmaker1": [3.0, 2.8, 2.9],
                "Best Odds": [3.0, 2.8, 2.9],
                "Best Bookmaker": ["Bookmaker1", "Bookmaker1", "Bookmaker1"],
                "Result": ["Not Found", "Not Found", "Not Found"],
                "Outcomes": [3, 3, 3],
            }
        )
        three_outcome_df = calculate_vigfree_probabilities(three_outcome_df)
        vf5 = three_outcome_df["Vigfree Bookmaker1"].iloc[0]
        vf6 = three_outcome_df["Vigfree Bookmaker1"].iloc[1]
        vf7 = three_outcome_df["Vigfree Bookmaker1"].iloc[2]
        self.assertAlmostEqual(vf5, 0.3219666931, places=3)
        self.assertAlmostEqual(vf6, 0.344964314, places=3)
        self.assertAlmostEqual(vf7, 0.3330689929, places=3)

    def test_calculate_vigfree_probabilities_missing_data(self):
        """Test calculate_vigfree_probabilities with missing data."""
        test_data = pd.DataFrame(
            {
                "Match": ["Team1 vs Team2", "Team1 vs Team2"],
                "Team": ["Team1", "Team2"],
                "Bookmaker1": [2.0, np.nan],
                "Best Odds": [2.0, 2.0],
                "Best Bookmaker": ["Bookmaker1", "Bookmaker2"],
                "Result": ["Not Found", "Not Found"],
                "Outcomes": [2, 2],
            }
        )

        result_df = calculate_vigfree_probabilities(test_data)

        # Should have NaN for both outcomes when one is missing
        vigfree_bm1 = result_df["Vigfree Bookmaker1"].values
        self.assertTrue(all(pd.isna(vigfree_bm1)))

    def test_all_nan_odds(self):
        """Test behavior when all odds are NaN."""
        test_data = pd.DataFrame(
            {
                "match": ["Match1"],
                "team": ["Team1"],
                "Book1": [np.nan],
                "Book2": [np.nan],
            }
        )

        result_df = _add_metadata(test_data)
        result_df = _clean_odds_data(result_df)
        self.assertTrue(pd.isna(result_df["Best Odds"].iloc[0]))
        self.assertTrue(pd.isna(result_df["Best Bookmaker"].iloc[0]))


if __name__ == "__main__":
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataProcessing))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
    )
    print(f"{'='*50}")

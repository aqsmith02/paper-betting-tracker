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

from src.find_bets.data_processing import (
    _find_bookmaker_columns,
    _remove_non_target_bookmakers,
    _add_metadata,
    _clean_odds_data,
    _min_bookmaker_filter,
    _max_odds_filter,
    _all_outcomes_present_filter,
    _prettify_column_headers,
    process_target_odds_data,
    calculate_market_margin,
    remove_margin_proportional_to_odds,
    calculate_vigfree_probabilities,
)


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Data from an API pull on 2025-09-25
        self.unprocessed = pd.read_csv("tests/find_bets/data_processing/test_data/unprocessed.csv")
        self.pre_metadata_check = pd.read_csv("tests/find_bets/data_processing/test_data/pre_metadata.csv")
        self.pre_clean = pd.read_csv("tests/find_bets/data_processing/test_data/pre_clean.csv")
        self.pre_min_bookmaker_check = pd.read_csv(
            "tests/find_bets/data_processing/test_data/pre_min_bookmaker_check.csv"
        )
        self.pre_max_odds_check = pd.read_csv(
            "tests/find_bets/data_processing/test_data/pre_max_odds_check.csv"
        )
        self.pre_outcome_check = pd.read_csv("tests/find_bets/data_processing/test_data/pre_outcome_check.csv")
        self.pre_prettify = pd.read_csv("tests/find_bets/data_processing/test_data/pre_prettify.csv")
        self.processed = pd.read_csv("tests/find_bets/data_processing/test_data/processed.csv")

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
        self.assertEqual(result_df["Best Odds"].iloc[2], 8.5)  # From Fanduel
        self.assertEqual(result_df["Best Odds"].iloc[9], 36)  # From Fanduel
        self.assertEqual(result_df["Best Odds"].iloc[14], 2.15)  # From Caesars

        # Check Best Bookmaker identification
        self.assertTrue(pd.isna(result_df["Best Bookmaker"].iloc[0])) 
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

    def test_calculate_market_margin(self):
        """Test calculate_market_margin function."""
        # Test with 2-outcome market (no margin)
        odds_fair = [2.0, 2.0]
        margin = calculate_market_margin(odds_fair)
        self.assertAlmostEqual(margin, 0.0, places=5)
        
        # Test with 2-outcome market with 5% margin
        # Implied probs: 0.52 + 0.52 = 1.04, so margin = 0.04 (4%)
        odds_with_margin = [1.92, 1.92]
        margin = calculate_market_margin(odds_with_margin)
        self.assertAlmostEqual(margin, 0.0417, places=3)
        
        # Test with 2-outcome market with typical 8% margin
        odds_with_8pct = [1.87, 1.95]
        margin = calculate_market_margin(odds_with_8pct)
        self.assertAlmostEqual(margin, 0.0476, places=3)
        
        # Test with 3-outcome market (soccer)
        # Fair odds might be: home 2.0, draw 3.5, away 4.0
        # With margin applied proportionally
        odds_3way = [1.90, 3.30, 3.75]
        margin = calculate_market_margin(odds_3way)
        self.assertGreater(margin, 0.05)  # Should have some margin
        
        # Test with empty list
        margin_empty = calculate_market_margin([])
        self.assertEqual(margin_empty, 0)
        
        # Test with single odds (invalid market)
        margin_single = calculate_market_margin([2.0])
        self.assertAlmostEqual(margin_single, 0.0, places=5)  # 1/2 - 1 = -0.5, but max(0, -0.5) = 0
        
        # Test with very low margin (Pinnacle-like)
        odds_low_margin = [1.98, 1.98]
        margin = calculate_market_margin(odds_low_margin)
        self.assertAlmostEqual(margin, 0.0101, places=3)

    def test_remove_margin_proportional_to_odds(self):
        """Test remove_margin_proportional_to_odds function."""
        # Test with 2-outcome market
        # Example: odds of 2.0 in a market with 5% margin
        all_market_odds = [1.92, 1.92]
        bookmaker_odds = 1.92
        n_outcomes = 2
        
        fair_odds = remove_margin_proportional_to_odds(bookmaker_odds, all_market_odds, n_outcomes)
        # Should return odds closer to 2.0
        self.assertIsNotNone(fair_odds)
        self.assertGreater(fair_odds, bookmaker_odds)
        self.assertAlmostEqual(fair_odds, 2.0, places=1)
        
        # Test with longshot in 2-outcome market
        all_market_odds_longshot = [1.30, 4.0]
        bookmaker_odds_longshot = 4.0
        fair_odds_longshot = remove_margin_proportional_to_odds(
            bookmaker_odds_longshot, all_market_odds_longshot, 2
        )
        # Longshot should have larger margin removed
        self.assertIsNotNone(fair_odds_longshot)
        self.assertGreater(fair_odds_longshot, bookmaker_odds_longshot)
        
        # Test with 3-outcome market (soccer)
        all_market_odds_3way = [1.90, 3.30, 3.75]
        bookmaker_odds_home = 1.90
        fair_odds_home = remove_margin_proportional_to_odds(
            bookmaker_odds_home, all_market_odds_3way, 3
        )
        self.assertIsNotNone(fair_odds_home)
        self.assertGreater(fair_odds_home, bookmaker_odds_home)
        
        # Test with draw odds (middle value)
        bookmaker_odds_draw = 3.30
        fair_odds_draw = remove_margin_proportional_to_odds(
            bookmaker_odds_draw, all_market_odds_3way, 3
        )
        self.assertIsNotNone(fair_odds_draw)
        self.assertGreater(fair_odds_draw, bookmaker_odds_draw)
        
        # Test with away odds (longshot)
        bookmaker_odds_away = 3.75
        fair_odds_away = remove_margin_proportional_to_odds(
            bookmaker_odds_away, all_market_odds_3way, 3
        )
        self.assertIsNotNone(fair_odds_away)
        self.assertGreater(fair_odds_away, bookmaker_odds_away)
        
        # Test edge case: zero or negative odds
        fair_odds_zero = remove_margin_proportional_to_odds(0, [2.0, 2.0], 2)
        self.assertIsNone(fair_odds_zero)
        
        fair_odds_negative = remove_margin_proportional_to_odds(-2.0, [2.0, 2.0], 2)
        self.assertIsNone(fair_odds_negative)
        
        # Test edge case: odds that would create negative denominator
        # This happens when M × O >= n
        # For n=2, M=0.5, O=5.0: denominator = 2 - 0.5*5 = 2 - 2.5 = -0.5
        all_market_odds_extreme = [1.20, 5.0]
        bookmaker_odds_extreme = 5.0
        fair_odds_extreme = remove_margin_proportional_to_odds(
            bookmaker_odds_extreme, all_market_odds_extreme, 2
        )
        # Should handle gracefully (return None for invalid case)
        if fair_odds_extreme is not None:
            self.assertGreater(fair_odds_extreme, 0)

    def test_remove_margin_proportional_to_odds_real_examples(self):
        """Test remove_margin_proportional_to_odds with real-world examples."""
        # Example from the documentation: fair odds [1.5, 5, 7.5] with 8% margin
        # Should produce bookmaker odds [1.44, 4.42, 6.25]
        
        # Test reverse: given bookmaker odds, should recover fair odds
        all_market_odds = [1.44, 4.42, 6.25]
        
        fair_odds_1 = remove_margin_proportional_to_odds(1.44, all_market_odds, 3)
        fair_odds_2 = remove_margin_proportional_to_odds(4.42, all_market_odds, 3)
        fair_odds_3 = remove_margin_proportional_to_odds(6.25, all_market_odds, 3)
        
        # Should approximately recover [1.5, 5, 7.5]
        self.assertAlmostEqual(fair_odds_1, 1.5, places=1)
        self.assertAlmostEqual(fair_odds_2, 5.0, places=1)
        self.assertAlmostEqual(fair_odds_3, 7.5, places=1)

    def test_vigfree_two_outcome_even_odds(self):
        """Test vigfree calculation for 2-outcome market with even odds."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.95, 1.95],
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Check vigfree column exists
        self.assertIn("Vigfree Bookmaker1", result.columns)
        
        # Check probabilities are valid
        vf1 = result["Vigfree Bookmaker1"].iloc[0]
        vf2 = result["Vigfree Bookmaker1"].iloc[1]
        
        self.assertIsNotNone(vf1)
        self.assertIsNotNone(vf2)
        self.assertGreater(vf1, 0)
        self.assertGreater(vf2, 0)
        self.assertLess(vf1, 1)
        self.assertLess(vf2, 1)
        
        # Should be approximately equal for even odds
        self.assertAlmostEqual(vf1, vf2, places=5)
        
        # Should be close to 0.5 each (true probability)
        self.assertAlmostEqual(vf1, 0.5, places=2)
        self.assertAlmostEqual(vf2, 0.5, places=2)

    def test_vigfree_two_outcome_uneven_odds(self):
        """Test vigfree calculation for 2-outcome market with uneven odds (favorite vs underdog)."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.2, 4.5],  # Favorite vs underdog
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        vf_favorite = result["Vigfree Bookmaker1"].iloc[0]
        vf_underdog = result["Vigfree Bookmaker1"].iloc[1]
        
        self.assertAlmostEqual(vf_favorite, 0.8055558333, places=3)
        self.assertAlmostEqual(vf_underdog, 0.1944447222, places=3)

    def test_vigfree_three_outcome_market(self):
        """Test vigfree calculation for 3-outcome market (soccer-style)."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Draw", "Team B"],
            "Bookmaker1": [2.20, 3.40, 3.20],
            "Outcomes": [3, 3, 3],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        vf_home = result["Vigfree Bookmaker1"].iloc[0]
        vf_draw = result["Vigfree Bookmaker1"].iloc[1]
        vf_away = result["Vigfree Bookmaker1"].iloc[2]
        
        # All probabilities should be valid
        self.assertIsNotNone(vf_home)
        self.assertIsNotNone(vf_draw)
        self.assertIsNotNone(vf_away)
        
        for prob in [vf_home, vf_draw, vf_away]:
            self.assertGreater(prob, 0)
            self.assertLess(prob, 1)
        
        # Home should have highest probability (lowest odds)
        self.assertGreater(vf_home, vf_draw)
        self.assertGreater(vf_home, vf_away)

    def test_vigfree_multiple_bookmakers(self):
        """Test vigfree calculation with multiple bookmakers."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.90, 2.00],
            "Bookmaker2": [1.85, 2.10],
            "Bookmaker3": [1.95, 1.95],
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Check all vigfree columns exist
        self.assertIn("Vigfree Bookmaker1", result.columns)
        self.assertIn("Vigfree Bookmaker2", result.columns)
        self.assertIn("Vigfree Bookmaker3", result.columns)
        
        # Check each bookmaker has valid probabilities
        for bookmaker in ["Bookmaker1", "Bookmaker2", "Bookmaker3"]:
            vf_col = f"Vigfree {bookmaker}"
            vf1 = result[vf_col].iloc[0]
            vf2 = result[vf_col].iloc[1]
            
            self.assertIsNotNone(vf1)
            self.assertIsNotNone(vf2)
            self.assertGreater(vf1, 0)
            self.assertGreater(vf2, 0)

    def test_vigfree_missing_odds_single_bookmaker(self):
        """Test vigfree when one outcome is missing for a bookmaker."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.90, np.nan],  # Missing second outcome
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Should have NaN for both since we need all outcomes
        vf1 = result["Vigfree Bookmaker1"].iloc[0]
        vf2 = result["Vigfree Bookmaker1"].iloc[1]
        
        self.assertTrue(pd.isna(vf1))
        self.assertTrue(pd.isna(vf2))

    def test_vigfree_missing_odds_one_bookmaker(self):
        """Test vigfree when one bookmaker has missing data but others don't."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.90, 2.00],
            "Bookmaker2": [1.85, np.nan],  # Missing for Bookmaker2
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Bookmaker1 should have valid vigfree
        vf1_bm1 = result["Vigfree Bookmaker1"].iloc[0]
        vf2_bm1 = result["Vigfree Bookmaker1"].iloc[1]
        self.assertIsNotNone(vf1_bm1)
        self.assertIsNotNone(vf2_bm1)
        
        # Bookmaker2 should have NaN
        vf1_bm2 = result["Vigfree Bookmaker2"].iloc[0]
        vf2_bm2 = result["Vigfree Bookmaker2"].iloc[1]
        self.assertTrue(pd.isna(vf1_bm2))
        self.assertTrue(pd.isna(vf2_bm2))

    def test_vigfree_multiple_matches(self):
        """Test vigfree calculation across multiple matches."""
        df = pd.DataFrame({
            "Match": [
                "Match1", "Match1",
                "Match2", "Match2",
                "Match3", "Match3"
            ],
            "Team": [
                "Team A", "Team B",
                "Team C", "Team D",
                "Team E", "Team F"
            ],
            "Bookmaker1": [1.80, 2.20, 1.50, 3.00, 2.00, 2.00],
            "Outcomes": [2, 2, 2, 2, 2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Check all matches have vigfree calculated
        for i in range(6):
            vf = result["Vigfree Bookmaker1"].iloc[i]
            self.assertIsNotNone(vf)
            self.assertGreater(vf, 0)
            self.assertLess(vf, 1)
        
        # Check probabilities sum to 1.0 for each match
        match1_sum = result["Vigfree Bookmaker1"].iloc[0] + result["Vigfree Bookmaker1"].iloc[1]
        match2_sum = result["Vigfree Bookmaker1"].iloc[2] + result["Vigfree Bookmaker1"].iloc[3]
        match3_sum = result["Vigfree Bookmaker1"].iloc[4] + result["Vigfree Bookmaker1"].iloc[5]
        
        self.assertAlmostEqual(match1_sum, 1.0, places=5)
        self.assertAlmostEqual(match2_sum, 1.0, places=5)
        self.assertAlmostEqual(match3_sum, 1.0, places=5)

    def test_vigfree_mixed_outcome_counts(self):
        """Test vigfree with different outcome counts in different matches."""
        df = pd.DataFrame({
            "Match": [
                "Match1", "Match1",  # 2-way
                "Match2", "Match2", "Match2"  # 3-way
            ],
            "Team": [
                "Team A", "Team B",
                "Team C", "Draw", "Team D"
            ],
            "Bookmaker1": [1.90, 2.00, 2.20, 3.40, 3.20],
            "Outcomes": [2, 2, 3, 3, 3],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Check 2-way match
        vf_2way_1 = result["Vigfree Bookmaker1"].iloc[0]
        vf_2way_2 = result["Vigfree Bookmaker1"].iloc[1]
        self.assertIsNotNone(vf_2way_1)
        self.assertIsNotNone(vf_2way_2)
        self.assertAlmostEqual(vf_2way_1 + vf_2way_2, 1.0, places=5)
        
        # Check 3-way match
        vf_3way_1 = result["Vigfree Bookmaker1"].iloc[2]
        vf_3way_2 = result["Vigfree Bookmaker1"].iloc[3]
        vf_3way_3 = result["Vigfree Bookmaker1"].iloc[4]
        self.assertIsNotNone(vf_3way_1)
        self.assertIsNotNone(vf_3way_2)
        self.assertIsNotNone(vf_3way_3)
        self.assertAlmostEqual(vf_3way_1 + vf_3way_2 + vf_3way_3, 1.0, places=5)

    def test_vigfree_insufficient_outcomes(self):
        """Test vigfree when fewer outcomes present than required."""
        df = pd.DataFrame({
            "Match": ["Match1", "Match1", "Match1"],
            "Team": ["Team A", "Draw", "Team B"],
            "Bookmaker1": [2.20, np.nan, 3.20],  # Missing draw odds
            "Outcomes": [3, 3, 3],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Should have NaN for all since we need 3 outcomes but only have 2
        for i in range(3):
            vf = result["Vigfree Bookmaker1"].iloc[i]
            self.assertTrue(pd.isna(vf))

    def test_vigfree_very_low_margin(self):
        """Test vigfree with very low margin (Pinnacle-like)."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.98, 1.98],  # Very low margin
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        vf1 = result["Vigfree Bookmaker1"].iloc[0]
        vf2 = result["Vigfree Bookmaker1"].iloc[1]
        
        # Should still calculate valid probabilities
        self.assertIsNotNone(vf1)
        self.assertIsNotNone(vf2)
        
        # Should be very close to 0.5 each
        self.assertAlmostEqual(vf1, 0.5, places=2)
        self.assertAlmostEqual(vf2, 0.5, places=2)

    def test_vigfree_high_margin(self):
        """Test vigfree with high margin (retail bookmaker)."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.80, 1.80],  # Higher margin
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        vf1 = result["Vigfree Bookmaker1"].iloc[0]
        vf2 = result["Vigfree Bookmaker1"].iloc[1]
        
        # Should still calculate valid probabilities
        self.assertIsNotNone(vf1)
        self.assertIsNotNone(vf2)
        
        # Should be approximately 0.5 each after margin removal
        self.assertAlmostEqual(vf1, 0.5, places=1)
        self.assertAlmostEqual(vf2, 0.5, places=1)

    def test_vigfree_extreme_longshot(self):
        """Test vigfree with extreme longshot odds."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.10, 15.0],  # Heavy favorite vs extreme longshot
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        vf_favorite = result["Vigfree Bookmaker1"].iloc[0]
        vf_longshot = result["Vigfree Bookmaker1"].iloc[1]
        
        # Should still have valid probabilities
        self.assertIsNotNone(vf_favorite)
        self.assertIsNotNone(vf_longshot)
        
        # Favorite should have very high probability
        self.assertGreater(vf_favorite, 0.85)
        
        # Longshot should have very low probability
        self.assertLess(vf_longshot, 0.15)

    def test_vigfree_empty_dataframe(self):
        """Test vigfree with empty DataFrame."""
        df = pd.DataFrame({
            "Match": [],
            "Team": [],
            "Bookmaker1": [],
            "Outcomes": [],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Should return empty DataFrame with vigfree column
        self.assertEqual(len(result), 0)
        self.assertIn("Vigfree Bookmaker1", result.columns)

    def test_vigfree_custom_bookmaker_columns(self):
        """Test vigfree with custom bookmaker_columns parameter."""
        df = pd.DataFrame({
            "Match": ["Team A @ Team B", "Team A @ Team B"],
            "Team": ["Team A", "Team B"],
            "Bookmaker1": [1.90, 2.00],
            "Bookmaker2": [1.85, 2.10],
            "OtherColumn": [100, 200],  # Not a bookmaker
            "Outcomes": [2, 2],
        })
        
        # Only process Bookmaker1
        result = calculate_vigfree_probabilities(df, bookmaker_columns=["Bookmaker1"])
        
        # Should have vigfree for Bookmaker1 but not Bookmaker2
        self.assertIn("Vigfree Bookmaker1", result.columns)
        self.assertNotIn("Vigfree Bookmaker2", result.columns)

    def test_vigfree_probabilities_sum_to_one(self):
        """Test that vigfree probabilities always sum to 1.0 for each match."""
        # Test various scenarios
        test_cases = [
            # 2-way markets
            ([1.90, 2.00], 2),
            ([1.50, 3.00], 2),
            ([1.10, 10.0], 2),
            # 3-way markets
            ([2.20, 3.40, 3.20], 3),
            ([1.50, 4.00, 8.00], 3),
        ]
        
        for odds, n_outcomes in test_cases:
            match_names = [f"Match{i}" for i in range(n_outcomes)]
            df = pd.DataFrame({
                "Match": ["TestMatch"] * n_outcomes,
                "Team": match_names,
                "Bookmaker1": odds,
                "Outcomes": [n_outcomes] * n_outcomes,
            })
            
            result = calculate_vigfree_probabilities(df)
            prob_sum = result["Vigfree Bookmaker1"].sum()
            
            self.assertAlmostEqual(
                prob_sum, 1.0, places=5,
                msg=f"Probabilities don't sum to 1.0 for odds {odds}"
            )

    def test_vigfree_real_world_example(self):
        """Test with real-world odds example."""
        # Real example: NFL game with typical bookmaker margins
        df = pd.DataFrame({
            "Match": ["Patriots @ Bills", "Patriots @ Bills"],
            "Team": ["Patriots", "Bills"],
            "DraftKings": [2.10, 1.83],
            "FanDuel": [2.08, 1.85],
            "Caesars": [2.15, 1.80],
            "Outcomes": [2, 2],
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Check all bookmakers processed
        for bm in ["DraftKings", "FanDuel", "Caesars"]:
            vf_col = f"Vigfree {bm}"
            self.assertIn(vf_col, result.columns)
            
            # Check valid probabilities
            vf1 = result[vf_col].iloc[0]
            vf2 = result[vf_col].iloc[1]
            
            self.assertIsNotNone(vf1)
            self.assertIsNotNone(vf2)
            self.assertAlmostEqual(vf1 + vf2, 1.0, places=5)
            
            # Bills (lower odds) should have higher probability
            self.assertGreater(vf2, vf1)


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

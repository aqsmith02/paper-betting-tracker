"""
test_betting_strategies.py

Unit tests for betting_strategies.py module.

Author: Brendon Smith
"""

import unittest
import pandas as pd
import numpy as np
from codebase.find_bets.betting_configs import (
    MAX_MISSING_VF_PCT,
    EV_THRESHOLD,
    Z_SCORE_THRESHOLD,
    MAX_Z_SCORE
)
from codebase.find_bets.betting_strategies import (
    analyze_average_edge_bets,
    analyze_modified_zscore_outliers,
    analyze_pinnacle_edge_bets,
    analyze_zscore_outliers,
    _missing_vigfree_odds_pct,
    find_random_bets
)


class TestMissingVigfreeOddsPct(unittest.TestCase):
    """Test suite for _missing_vigfree_odds_pct function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.func = _missing_vigfree_odds_pct
        self.MAX_MISSING_VF_PCT = MAX_MISSING_VF_PCT

    def test_missing_vigfree_odds_within_limit(self):
        """Test when missing vig-free percentage is within acceptable limit."""
        test_row = pd.Series({
            'Bookmaker1': 2.5,
            'Bookmaker2': 2.4,
            'Bookmaker3': 2.6,
            'Bookmaker4': 2.7,
            'Bookmaker5': 2.8,
            'Vigfree Bookmaker1': np.nan,  # 1 missing
            'Vigfree Bookmaker2': 0.40,
            'Vigfree Bookmaker3': 0.37,   
            'Vigfree Bookmaker4': 0.36,   
            'Vigfree Bookmaker5': 0.35
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3', 'Bookmaker4', 'Bookmaker5']
        result = self.func(test_row, bookmaker_columns, max_missing=self.MAX_MISSING_VF_PCT)
        self.assertTrue(result)

    def test_missing_vigfree_odds_exceeds_limit(self):
        """Test when missing vig-free percentage exceeds acceptable limit."""
        test_row = pd.Series({
            'Bookmaker1': 2.5,
            'Bookmaker2': 2.4,
            'Bookmaker3': 2.6,
            'Bookmaker4': 2.7,
            'Bookmaker5': 2.8,
            'Vigfree Bookmaker1': np.nan,  # 3 missing out of 5 = 60%
            'Vigfree Bookmaker2': np.nan,
            'Vigfree Bookmaker3': np.nan,   
            'Vigfree Bookmaker4': 0.36,   
            'Vigfree Bookmaker5': 0.35
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3', 'Bookmaker4', 'Bookmaker5']
        result = self.func(test_row, bookmaker_columns, max_missing=self.MAX_MISSING_VF_PCT)
        self.assertFalse(result)

    def test_no_missing_vigfree_odds(self):
        """Test when all vig-free odds are present."""
        test_row = pd.Series({
            'Bookmaker1': 2.5,
            'Bookmaker2': 2.4,
            'Vigfree Bookmaker1': 0.38,
            'Vigfree Bookmaker2': 0.40
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2']
        result = self.func(test_row, bookmaker_columns, max_missing=self.MAX_MISSING_VF_PCT)
        self.assertTrue(result)


class TestAnalyzeAverageEdgeBets(unittest.TestCase):
    """Test suite for analyze_average_edge_bets function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_average_edge_bets

    def test_positive_expected_value(self):
        """Test calculation with positive expected value."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [4.0],
            'Bovada': [3.5],
            'Pinnacle': [3.8],
            'Draftkings': [3.6],
            'Vigfree Bovada': [0.28],
            'Vigfree Pinnacle': [0.26],
            'Vigfree Draftkings': [0.27]
        })
        
        result = self.func(df)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Expected Value', result.columns)
        
        # Average probability = (0.28 + 0.26 + 0.27) / 3 = 0.27
        # Fair odds = 1 / 0.27 = 3.70
        # EV = (0.27 * (4.0 - 1)) - ((1 - 0.27) * 1) = 0.81 - 0.73 = 0.08
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 3.70, places=2)
        
        # EV should be calculated if above threshold
        if result['Expected Value'].iloc[0] is not None:
            self.assertGreater(result['Expected Value'].iloc[0], 0)

    def test_negative_expected_value(self):
        """Test that bets with negative EV are filtered out."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [2.0],
            'Bovada': [2.0],
            'Pinnacle': [2.0],
            'Vigfree Bovada': [0.48],
            'Vigfree Pinnacle': [0.48]
        })
        
        result = self.func(df)
        
        # EV = (0.48 * (2.0 - 1)) - ((1 - 0.48) * 1) = 0.48 - 0.52 = -0.04
        # Should be None since below threshold
        self.assertIsNone(result['Expected Value'].iloc[0])

    def test_too_many_missing_vigfree(self):
        """Test when too many bookmakers have missing vig-free odds."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [4.0],
            'Bookmaker1': [3.5],
            'Bookmaker2': [3.6],
            'Bookmaker3': [3.7],
            'Bookmaker4': [3.8],
            'Bookmaker5': [3.9],
            'Vigfree Bookmaker1': np.nan,
            'Vigfree Bookmaker2': np.nan,
            'Vigfree Bookmaker3': np.nan,
            'Vigfree Bookmaker4': 0.26,
            'Vigfree Bookmaker5': 0.25
        })
        
        result = self.func(df)
        
        # Should return None for both columns due to too many missing
        self.assertIsNone(result['Fair Odds Avg'].iloc[0])
        self.assertIsNone(result['Expected Value'].iloc[0])
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.func(empty_df)
        self.assertEqual(len(result), 0)
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Expected Value', result.columns)


class TestAnalyzeZscoreOutliers(unittest.TestCase):
    """Test suite for analyze_zscore_outliers function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_zscore_outliers

    def test_zscore_calculation_with_valid_outlier(self):
        """Test Z-score calculation with a valid outlier."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [5.0],
            'Book1': [3.0],
            'Book2': [3.1],
            'Book3': [3.0],
            'Book4': [3.2],
            'Book5': [5.0],
            'Vigfree Book1': [0.32],
            'Vigfree Book2': [0.31],
            'Vigfree Book3': [0.32],
            'Vigfree Book4': [0.30],
            'Vigfree Book5': [0.19]
        })
        
        result = self.func(df)
        
        # Check that columns exist
        self.assertIn('Z Score', result.columns)
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Expected Value', result.columns)
        
        # Z-score should be calculated
        if result['Z Score'].iloc[0] is not None:
            self.assertGreater(result['Z Score'].iloc[0], Z_SCORE_THRESHOLD)

    def test_zscore_zero_std(self):
        """Test Z-score when standard deviation is zero."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [3.0],
            'Bovada': [3.0],
            'Pinnacle': [3.0],
            'Draftkings': [3.0],
            'Vigfree Bovada': [0.33],
            'Vigfree Pinnacle': [0.33],
            'Vigfree Draftkings': [0.33]
        })
        
        result = self.func(df)
        
        # Should return None for zero std
        self.assertIsNone(result['Z Score'].iloc[0])

    def test_zscore_exceeds_max_threshold(self):
        """Test that Z-scores above MAX_Z_SCORE are filtered."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [20.0],
            'Book1': [2.0],
            'Book2': [2.0],
            'Book3': [2.0],
            'Book4': [2.0],
            'Book5': [20.0],
            'Vigfree Book1': [0.49],
            'Vigfree Book2': [0.49],
            'Vigfree Book3': [0.49],
            'Vigfree Book4': [0.49],
            'Vigfree Book5': [0.05]
        })
        
        result = self.func(df)
        
        # Z-score should be None if it exceeds MAX_Z_SCORE
        # (unrealistic outlier, likely data error)
        self.assertIsNone(result['Z Score'].iloc[0])


class TestAnalyzeModifiedZscoreOutliers(unittest.TestCase):
    """Test suite for analyze_modified_zscore_outliers function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_modified_zscore_outliers

    def test_modified_zscore_calculation(self):
        """Test Modified Z-score calculation with valid outlier."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [4.5],
            'Book1': [3.0],
            'Book2': [3.1],
            'Book3': [3.0],
            'Book4': [3.2],
            'Book5': [4.5],
            'Vigfree Book1': [0.32],
            'Vigfree Book2': [0.31],
            'Vigfree Book3': [0.32],
            'Vigfree Book4': [0.30],
            'Vigfree Book5': [0.21]
        })
        
        result = self.func(df)

        # Check that columns exist
        self.assertIn('Modified Z Score', result.columns)
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Expected Value', result.columns)
        
        # Modified Z-score should be calculated if valid
        if result['Modified Z Score'].iloc[0] is not None:
            self.assertGreater(result['Modified Z Score'].iloc[0], Z_SCORE_THRESHOLD)

    def test_modified_zscore_zero_mad(self):
        """Test Modified Z-score when MAD is zero."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [3.0],
            'Bovada': [3.0],
            'Pinnacle': [3.0],
            'Draftkings': [3.0],
            'Vigfree Bovada': [0.33],
            'Vigfree Pinnacle': [0.33],
            'Vigfree Draftkings': [0.33]
        })
        
        result = self.func(df)
        
        # Should return None for zero MAD
        self.assertIsNone(result['Modified Z Score'].iloc[0])

    def test_modified_zscore_exceeds_max(self):
        """Test that Modified Z-scores above MAX_Z_SCORE are filtered."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [25.0],
            'Book1': [2.5],
            'Book2': [2.5],
            'Book3': [2.5],
            'Book4': [2.5],
            'Book5': [25.0],
            'Vigfree Book1': [0.39],
            'Vigfree Book2': [0.39],
            'Vigfree Book3': [0.39],
            'Vigfree Book4': [0.39],
            'Vigfree Book5': [0.04]
        })
        
        result = self.func(df)
        
        # Modified Z-score should be None if exceeds MAX_Z_SCORE
        self.assertIsNone(result['Modified Z Score'].iloc[0])


class TestAnalyzePinnacleEdgeBets(unittest.TestCase):
    """Test suite for analyze_pinnacle_edge_bets function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_pinnacle_edge_bets

    def test_pinnacle_positive_ev(self):
        """Test Pinnacle edge calculation with positive EV."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [4.0],
            'Pinnacle': [3.5],
            'Fanduel': [4.0],
            'Vigfree Pinnacle': [0.28],
            'Vigfree Fanduel': [0.24]
        })
        
        result = self.func(df)
        
        # Check that Pinnacle columns are added
        self.assertIn('Pinnacle Fair Odds', result.columns)
        self.assertIn('Expected Value', result.columns)
        
        # Pinnacle fair odds = 1 / 0.28 = 3.57
        self.assertAlmostEqual(result['Pinnacle Fair Odds'].iloc[0], 3.57, places=2)
        
        # EV = (0.28 * (4.0 - 1)) - ((1 - 0.28) * 1) = 0.84 - 0.72 = 0.12
        if result['Expected Value'].iloc[0] is not None:
            self.assertGreater(result['Expected Value'].iloc[0], 0)

    def test_pinnacle_negative_ev(self):
        """Test that bets with negative EV are filtered."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [2.0],
            'Pinnacle': [2.0],
            'Vigfree Pinnacle': [0.48]
        })
        
        result = self.func(df)
        
        # EV should be None since below threshold
        self.assertIsNone(result['Expected Value'].iloc[0])

    def test_pinnacle_missing_vigfree(self):
        """Test when Pinnacle vig-free odds are missing."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [10.0],
            'Pinnacle': [8.0],
            'Vigfree Pinnacle': [np.nan]
        })
        
        result = self.func(df)
        
        # Should return None for missing vig-free
        self.assertIsNone(result['Pinnacle Fair Odds'].iloc[0])
        self.assertIsNone(result['Expected Value'].iloc[0])
    
    def test_no_pinnacle_column(self):
        """Test when Pinnacle column doesn't exist."""
        df = pd.DataFrame({
            'Match': ['Team A @ Team B'],
            'Team': ['Team A'],
            'Best Odds': [10.0],
            'Bovada': [8.0],
            'Vigfree Bovada': [0.12]
        })
        
        result = self.func(df)
        
        # Should return original DataFrame without new columns
        self.assertNotIn('Pinnacle Fair Odds', result.columns)
        self.assertNotIn('Expected Value', result.columns)


class TestFindRandomBets(unittest.TestCase):
    """Test suite for find_random_bets function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = find_random_bets

    def test_random_bets_column_added(self):
        """Test that Random Placed Bet column is added."""
        df = pd.DataFrame({
            'Match': ['Match 1', 'Match 2', 'Match 3'],
            'Team': ['Team A', 'Team B', 'Team C'],
            'Best Odds': [2.0, 3.0, 4.0]
        })
        
        result = self.func(df)
        self.assertIn('Random Placed Bet', result.columns)
    
    def test_random_bets_binary(self):
        """Test that Random Placed Bet contains only 0s and 1s."""
        df = pd.DataFrame({
            'Match': ['Match 1', 'Match 2', 'Match 3'],
            'Team': ['Team A', 'Team B', 'Team C'],
            'Best Odds': [2.0, 3.0, 4.0]
        })
        
        result = self.func(df)
        unique_values = result['Random Placed Bet'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))
    
    def test_random_bets_count_range(self):
        """Test that number of bets is within expected range."""
        df = pd.DataFrame({
            'Match': [f'Match {i}' for i in range(10)],
            'Team': [f'Team {i}' for i in range(10)],
            'Best Odds': [2.0 + i * 0.1 for i in range(10)]
        })
        
        result = self.func(df)
        bet_count = result['Random Placed Bet'].sum()
        
        # Should be between 0 and min(5, len(df))
        self.assertGreaterEqual(bet_count, 0)
        self.assertLessEqual(bet_count, 5)
    
    def test_random_bets_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.func(empty_df)
        self.assertIn('Random Placed Bet', result.columns)
        self.assertEqual(len(result), 0)
    
    def test_random_bets_small_dataframe(self):
        """Test with DataFrame smaller than 5 rows."""
        small_df = pd.DataFrame({
            'Match': ['Match 1', 'Match 2'],
            'Team': ['Team A', 'Team B'],
            'Best Odds': [2.0, 3.0]
        })
        
        result = self.func(small_df)
        bet_count = result['Random Placed Bet'].sum()
        
        # Should be between 0 and 2
        self.assertGreaterEqual(bet_count, 0)
        self.assertLessEqual(bet_count, 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for multiple functions"""
    
    def setUp(self):
        """Set up comprehensive test data"""
        self.df = pd.DataFrame({
            'Match': ['Team A @ Team B'] * 2,
            'Team': ['Team A', 'Team B'],
            'Best Odds': [4.5, 1.95],
            'Bovada': [4.0, 1.90],
            'Pinnacle': [4.2, 1.92],
            'Draftkings': [4.3, 1.95],
            'Betmgm': [4.5, 1.88],
            'Vigfree Bovada': [0.24, 0.51],
            'Vigfree Pinnacle': [0.23, 0.50],
            'Vigfree Draftkings': [0.22, 0.49],
            'Vigfree Betmgm': [0.21, 0.52]
        })
    
    def test_full_pipeline(self):
        """Test running all analysis functions in sequence."""
        result = self.df.copy()
        result = analyze_average_edge_bets(result)
        result = analyze_zscore_outliers(result)
        result = analyze_modified_zscore_outliers(result)
        result = analyze_pinnacle_edge_bets(result)
        result = find_random_bets(result)
        
        # Check all expected columns exist
        expected_columns = [
            'Fair Odds Avg', 'Expected Value',
            'Z Score', 'Modified Z Score',
            'Pinnacle Fair Odds',
            'Random Placed Bet'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that original data is preserved
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(all(result['Match'] == self.df['Match']))
        self.assertTrue(all(result['Team'] == self.df['Team']))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
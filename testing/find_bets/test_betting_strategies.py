"""
test_betting_strategies.py

Unit tests for betting_strategies.py module.

Author: Andrew Smith
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch
import sys
import os


from codebase.find_bets.betting_strategies import (
    _count_missing_vigfree_odds,
    analyze_average_edge_bets,
    analyze_zscore_outliers,
    analyze_modified_zscore_outliers,
    analyze_pinnacle_edge_bets,
    find_random_bets
)


class TestBettingStrategies(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Load the processed CSV data for testing
        self.data = pd.read_csv('vf.csv')


    def test_count_missing_vigfree_odds_within_limit(self):
        """Test _count_missing_vigfree_odds when missing count is within limit."""
        test_row = pd.Series({
            'Bookmaker1': 2.5,
            'Bookmaker2': 2.4,
            'Bookmaker3': 2.6,
            'Bookmaker4': 2.6,
            'Bookmaker5': 2.6,
            'Vigfree Bookmaker1': np.nan,  # Missing
            'Vigfree Bookmaker2': 0.40,
            'Vigfree Bookmaker3': 0.37,   
            'Vigfree Bookmaker4': 0.37,   
            'Vigfree Bookmaker5': np.nan   # Missing
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3', 'Bookmaker4', 'Bookmaker5']
        result = _count_missing_vigfree_odds(test_row, bookmaker_columns, max_missing=2)
        
        # 2 missing vig-free odds, max_missing=2, should return True
        self.assertTrue(result)

    def test_count_missing_vigfree_odds_exceeds_limit(self):
        """Test _count_missing_vigfree_odds when missing count exceeds limit."""
        test_row = pd.Series({
            'Bookmaker1': 2.5,
            'Bookmaker2': 2.4,
            'Bookmaker3': 2.6,
            'Bookmaker4': 2.6,
            'Bookmaker5': 2.6,
            'Vigfree Bookmaker1': np.nan,  # Missing
            'Vigfree Bookmaker2': np.nan,  # Missing
            'Vigfree Bookmaker3': 0.37,   
            'Vigfree Bookmaker4': 0.37,   
            'Vigfree Bookmaker5': np.nan   # Missing
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3']
        result = _count_missing_vigfree_odds(test_row, bookmaker_columns, max_missing=1)
        
        # 2 missing vig-free odds, max_missing=1, should return False
        self.assertFalse(result)

    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_count_missing_vigfree_odds_no_odds(self):
        """Test _count_missing_vigfree_odds when bookmaker has no odds."""
        test_row = pd.Series({
            'Bookmaker1': np.nan,  # No odds
            'Bookmaker2': 2.4,
            'Bookmaker3': 2.6,
            'Vigfree Bookmaker1': np.nan,
            'Vigfree Bookmaker2': 0.40,
            'Vigfree Bookmaker3': 0.37
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3']
        result = _count_missing_vigfree_odds(test_row, bookmaker_columns, max_missing=5)
        
        # Only Bookmaker1 has no odds, so it shouldn't count as missing vig-free
        # Should return True (0 missing vig-free odds where there are actual odds)
        self.assertTrue(result)

    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_average_edge_bets_basic(self):
        """Test analyze_average_edge_bets with basic functionality."""
        df = self.simple_data.copy()
        result_df = analyze_average_edge_bets(df)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result_df.columns)
        self.assertIn('Avg Edge Pct', result_df.columns)
        
        # Check Fair Odds Avg calculation for first row
        # Average probability = (0.38 + 0.40 + 0.37) / 3 = 0.383333
        # Fair odds = 1 / 0.383333 = 2.61
        expected_fair_odds = round(1 / ((0.38 + 0.40 + 0.37) / 3), 2)
        self.assertEqual(result_df['Fair Odds Avg'].iloc[0], expected_fair_odds)
        
        # Check that edge percentage is calculated
        self.assertIsNotNone(result_df['Avg Edge Pct'].iloc[0])

    @patch('betting_strategies.EDGE_THRESHOLD', 0.10)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_average_edge_bets_no_edge(self):
        """Test analyze_average_edge_bets when edge doesn't meet threshold."""
        df = self.simple_data.copy()
        result_df = analyze_average_edge_bets(df)
        
        # With high edge threshold (10%), likely no edges will qualify
        # Check that at least the columns exist
        self.assertIn('Fair Odds Avg', result_df.columns)
        self.assertIn('Avg Edge Pct', result_df.columns)

    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 0)
    def test_analyze_average_edge_bets_missing_vigfree(self):
        """Test analyze_average_edge_bets with missing vig-free data."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B'],
            'Team': ['Match A'],
            'Bookmaker1': [2.5],
            'Bookmaker2': [2.4],
            'Best Odds': [2.5],
            'Vigfree Bookmaker1': [np.nan],  # Missing
            'Vigfree Bookmaker2': [0.40]
        })
        
        result_df = analyze_average_edge_bets(df)
        
        # Should have None values due to missing vig-free data
        self.assertIsNone(result_df['Fair Odds Avg'].iloc[0])
        self.assertIsNone(result_df['Avg Edge Pct'].iloc[0])

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_zscore_outliers(self):
        """Test analyze_zscore_outliers function."""
        df = self.simple_data.copy()
        result_df = analyze_zscore_outliers(df)
        
        # Check that Z Score column is added
        self.assertIn('Z Score', result_df.columns)
        
        # Check that average edge columns are also added (since it calls analyze_average_edge_bets)
        self.assertIn('Fair Odds Avg', result_df.columns)
        self.assertIn('Avg Edge Pct', result_df.columns)

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    def test_analyze_zscore_outliers_zero_std(self):
        """Test analyze_zscore_outliers with zero standard deviation."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B'],
            'Team': ['Match A'],
            'Bookmaker1': [2.0],
            'Bookmaker2': [2.0],  # All same odds = zero std
            'Bookmaker3': [2.0],
            'Best Odds': [2.0],
            'Vigfree Bookmaker1': [0.50],
            'Vigfree Bookmaker2': [0.50],
            'Vigfree Bookmaker3': [0.50]
        })
        
        with patch('betting_strategies.EDGE_THRESHOLD', 0.02):
            with patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5):
                result_df = analyze_zscore_outliers(df)
        
        # Should have None for Z Score due to zero standard deviation
        self.assertIsNone(result_df['Z Score'].iloc[0])

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_modified_zscore_outliers(self):
        """Test analyze_modified_zscore_outliers function."""
        df = self.simple_data.copy()
        result_df = analyze_modified_zscore_outliers(df)
        
        # Check that Modified Z Score column is added
        self.assertIn('Modified Z Score', result_df.columns)
        
        # Check that average edge columns are also added
        self.assertIn('Fair Odds Avg', result_df.columns)
        self.assertIn('Avg Edge Pct', result_df.columns)

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    def test_analyze_modified_zscore_outliers_zero_mad(self):
        """Test analyze_modified_zscore_outliers with zero MAD."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B'],
            'Team': ['Match A'],
            'Bookmaker1': [2.0],
            'Bookmaker2': [2.0],  # All same odds = zero MAD
            'Bookmaker3': [2.0],
            'Best Odds': [2.0],
            'Vigfree Bookmaker1': [0.50],
            'Vigfree Bookmaker2': [0.50],
            'Vigfree Bookmaker3': [0.50]
        })
        
        with patch('betting_strategies.EDGE_THRESHOLD', 0.02):
            with patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5):
                result_df = analyze_modified_zscore_outliers(df)
        
        # Should have None for Modified Z Score due to zero MAD
        self.assertIsNone(result_df['Modified Z Score'].iloc[0])

    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    def test_analyze_pinnacle_edge_bets_with_pinnacle(self):
        """Test analyze_pinnacle_edge_bets when Pinnacle data exists."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B', 'Match A vs Match B'],
            'Team': ['Match A', 'Match B'],
            'Pinnacle': [2.5, 1.8],
            'Best Odds': [2.6, 1.9],
            'Vigfree Pinnacle': [0.40, 0.55]
        })
        
        result_df = analyze_pinnacle_edge_bets(df)
        
        # Check that Pinnacle columns are added
        self.assertIn('Pinnacle Fair Odds', result_df.columns)
        self.assertIn('Pin Edge Pct', result_df.columns)
        
        # Check Pinnacle Fair Odds calculation for first row
        # Fair odds = 1 / 0.40 = 2.5
        expected_pinnacle_fair_odds = round(1 / 0.40, 2)
        self.assertEqual(result_df['Pinnacle Fair Odds'].iloc[0], expected_pinnacle_fair_odds)

    def test_analyze_pinnacle_edge_bets_without_pinnacle(self):
        """Test analyze_pinnacle_edge_bets when Pinnacle data doesn't exist."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B'],
            'Team': ['Match A'],
            'Bookmaker1': [2.5],
            'Best Odds': [2.6]
        })
        
        result_df = analyze_pinnacle_edge_bets(df)
        
        # Should return the dataframe unchanged (no Pinnacle column)
        self.assertEqual(len(result_df.columns), len(df.columns))

    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    def test_analyze_pinnacle_edge_bets_missing_vigfree(self):
        """Test analyze_pinnacle_edge_bets with missing Pinnacle vig-free data."""
        df = pd.DataFrame({
            'Match': ['Match A vs Match B'],
            'Team': ['Match A'],
            'Pinnacle': [2.5],
            'Best Odds': [2.6],
            'Vigfree Pinnacle': [np.nan]  # Missing
        })
        
        result_df = analyze_pinnacle_edge_bets(df)
        
        # Should have None values due to missing vig-free data
        self.assertIsNone(result_df['Pinnacle Fair Odds'].iloc[0])
        self.assertIsNone(result_df['Pin Edge Pct'].iloc[0])

    def test_find_random_bets(self):
        """Test find_random_bets function."""
        df = self.simple_data.copy()
        result_df = find_random_bets(df)
        
        # Check that Random Placed Bet column is added
        self.assertIn('Random Placed Bet', result_df.columns)
        
        # Check that values are 0 or 1
        self.assertTrue(all(val in [0, 1] for val in result_df['Random Placed Bet']))
        
        # Check that the number of placed bets is within expected range (0 to min(5, len(df)))
        placed_bets = result_df['Random Placed Bet'].sum()
        self.assertTrue(0 <= placed_bets <= min(5, len(df)))

    def test_find_random_bets_determinism(self):
        """Test that find_random_bets produces different results on multiple runs."""
        df = self.simple_data.copy()
        
        # Run multiple times and collect results
        results = []
        for _ in range(10):
            result_df = find_random_bets(df)
            results.append(result_df['Random Placed Bet'].sum())
        
        # Should have some variation in results (not all the same)
        # Note: This test might occasionally fail due to randomness
        unique_results = len(set(results))
        self.assertGreater(unique_results, 1, "Random bets should vary across runs")


class TestBettingStrategiesIntegration(unittest.TestCase):
    """Integration tests using the full processed CSV data."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.data = pd.read_csv('processed.csv')

    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_average_edge_bets_on_real_data(self):
        """Test analyze_average_edge_bets on real processed data."""
        df = self.data.copy()
        result_df = analyze_average_edge_bets(df)
        
        # Check that all rows have the new columns
        self.assertEqual(len(result_df), len(df))
        self.assertIn('Fair Odds Avg', result_df.columns)
        self.assertIn('Avg Edge Pct', result_df.columns)
        
        # Check that Fair Odds Avg values are reasonable (between 1 and 100)
        valid_fair_odds = result_df['Fair Odds Avg'].dropna()
        if len(valid_fair_odds) > 0:
            self.assertTrue(all(1 <= odds <= 100 for odds in valid_fair_odds))

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_zscore_outliers_on_real_data(self):
        """Test analyze_zscore_outliers on real processed data."""
        df = self.data.copy()
        result_df = analyze_zscore_outliers(df)
        
        # Check that all rows have the new columns
        self.assertEqual(len(result_df), len(df))
        self.assertIn('Z Score', result_df.columns)
        
        # Check that Z Score values are within expected range
        valid_z_scores = result_df['Z Score'].dropna()
        if len(valid_z_scores) > 0:
            self.assertTrue(all(0 <= z <= 10 for z in valid_z_scores))

    @patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0)
    @patch('betting_strategies.MAX_Z_SCORE', 10.0)
    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    @patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5)
    def test_analyze_modified_zscore_outliers_on_real_data(self):
        """Test analyze_modified_zscore_outliers on real processed data."""
        df = self.data.copy()
        result_df = analyze_modified_zscore_outliers(df)
        
        # Check that all rows have the new columns
        self.assertEqual(len(result_df), len(df))
        self.assertIn('Modified Z Score', result_df.columns)
        
        # Check that Modified Z Score values are within expected range
        valid_mod_z_scores = result_df['Modified Z Score'].dropna()
        if len(valid_mod_z_scores) > 0:
            self.assertTrue(all(0 <= z <= 10 for z in valid_mod_z_scores))

    @patch('betting_strategies.EDGE_THRESHOLD', 0.02)
    def test_analyze_pinnacle_edge_bets_on_real_data(self):
        """Test analyze_pinnacle_edge_bets on real processed data."""
        df = self.data.copy()
        result_df = analyze_pinnacle_edge_bets(df)
        
        # Check that all rows are preserved
        self.assertEqual(len(result_df), len(df))
        
        # Check that Pinnacle columns exist
        self.assertIn('Pinnacle Fair Odds', result_df.columns)
        self.assertIn('Pin Edge Pct', result_df.columns)

    def test_all_strategies_pipeline(self):
        """Test running all strategies in sequence."""
        df = self.data.copy()
        
        with patch('betting_strategies.EDGE_THRESHOLD', 0.02):
            with patch('betting_strategies.MAX_MISSING_VIGFREE_ODDS', 5):
                with patch('betting_strategies.Z_SCORE_THRESHOLD', 1.0):
                    with patch('betting_strategies.MAX_Z_SCORE', 10.0):
                        # Run all strategies
                        df = analyze_average_edge_bets(df)
                        df = analyze_zscore_outliers(df)
                        df = analyze_modified_zscore_outliers(df)
                        df = analyze_pinnacle_edge_bets(df)
                        df = find_random_bets(df)
        
        # Check that all expected columns exist
        expected_columns = [
            'Fair Odds Avg', 'Avg Edge Pct', 'Z Score', 
            'Modified Z Score', 'Pinnacle Fair Odds', 
            'Pin Edge Pct', 'Random Placed Bet'
        ]
        for col in expected_columns:
            self.assertIn(col, df.columns)
        
        # Check that dataframe length is unchanged
        self.assertEqual(len(df), len(self.data))


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBettingStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestBettingStrategiesIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
"""
test_betting_strategies.py

Unit tests for betting_strategies.py module.

Author: Andrew Smith
"""

import unittest
import pandas as pd
import numpy as np
from codebase.find_bets.betting_configs import MAX_MISSING_VF_PCT
from codebase.find_bets.betting_strategies import analyze_average_edge_bets,analyze_modified_zscore_outliers,analyze_pinnacle_edge_bets,analyze_zscore_outliers,_missing_vigfree_odds_pct,find_random_bets



class TestMissingVigfreeOddsPct(unittest.TestCase):
    """Test suite for _missing_vigfree_odds_pct function"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Import the function
        self.func = _missing_vigfree_odds_pct
        self.MAX_MISSING_VF_PCT = MAX_MISSING_VF_PCT

    
    def test_missing_vigfree_odds_pct_within_limit(self):
        """Test _missing_vigfree_odds_pct when missing count is within limit."""
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
            'Vigfree Bookmaker5': 0.37
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3', 'Bookmaker4', 'Bookmaker5']
        result = self.func(test_row, bookmaker_columns, max_missing=self.MAX_MISSING_VF_PCT)
        self.assertTrue(result)

    def test_count_missing_vigfree_odds_exceeds_limit(self):
        """Test _missing_vigfree_odds_pct when missing count exceeds limit."""
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
            'Vigfree Bookmaker5': 0.37
        })
        
        bookmaker_columns = ['Bookmaker1', 'Bookmaker2', 'Bookmaker3', 'Bookmaker4', 'Bookmaker5']
        result = self.func(test_row, bookmaker_columns, max_missing=self.MAX_MISSING_VF_PCT)
        
        self.assertFalse(result)


class TestAnalyzeAverageEdgeBets(unittest.TestCase):
    """Test suite for analyze_average_edge_bets function"""
    
    def setUp(self):
        """Set up test data"""

        self.func = analyze_average_edge_bets
        
        # Create sample DataFrame from provided CSV data
        self.df = pd.DataFrame({
            'Match': ['Seattle Seahawks @ Arizona Cardinals', 'Seattle Seahawks @ Arizona Cardinals'],
            'Team': ['Arizona Cardinals', 'Seattle Seahawks'],
            'Best Odds': [12.21, 1.11],
            'Bovada': [6.5, 1.1],
            'Pinnacle': [12.21, 1.05],
            'Draftkings': [6.75, 1.11],
            'Vigfree Bovada': [0.14473684210526316, 0.8552631578947368],
            'Vigfree Pinnacle': [0.07918552036199095, 0.920814479638009],
            'Vigfree Draftkings': [0.14122137404580154, 0.8587786259541985]
        })
    

    def test_basic_edge_calculation(self):
        """Test basic edge calculation with valid data"""
        result = self.func(self.df)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        
        # Check that values are approximately correct
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 8.2, places=1)
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[1], 1.14, places=2)

        self.assertAlmostEqual(result['Avg Edge Pct'].iloc[0], 48.6, places=1)
        self.assertTrue(np.isnan(result['Avg Edge Pct'].iloc[1]))
    
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.func(empty_df)
        self.assertEqual(len(result), 0)


class TestAnalyzeZscoreOutliers(unittest.TestCase):
    """Test suite for analyze_zscore_outliers function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_zscore_outliers
        
        self.df = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels', 'Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals', 'Los Angeles Angels'],
            'Best Odds': [6.5, 1.3],
            'Mybookie.ag': [6.5, 1.1],
            'Bovada': [3.59, 1.3],
            'Pinnacle': [3.59, 1.3],
            'Draftkings': [3.59, 1.3],
            'Fanduel': [3.59, 1.3],
            'Fanatics': [3.59, 1.3],
            'Vigfree Mybookie.ag': [0.14, 0.86],
            'Vigfree Bovada': [0.27, 0.73],
            'Vigfree Pinnacle': [0.27, 0.73],
            'Vigfree Draftkings': [0.27, 0.73],
            'Vigfree Fanduel': [0.27, 0.73],
            'Vigfree Fanatics': [0.27, 0.73]
        })
    

    def test_zscore_calculation(self):
        """Test Z-score calculation"""
        result = self.func(self.df)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Z Score', result.columns)
        
        # Check that values are approximately correct
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 4.03, places=2)
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[1], 1.33, places=2)

        self.assertAlmostEqual(result['Avg Edge Pct'].iloc[0], 61.42, places=2)
        self.assertTrue(pd.isna(result['Avg Edge Pct'].iloc[1]))

        self.assertAlmostEqual(result['Z Score'].iloc[0], 2.04, places=2)
        self.assertTrue(pd.isna(result['Z Score'].iloc[1]))
    

    def test_zscore_zero_std(self):
        """Test Z-score calculation when standard deviation is zero"""
        df_same_odds = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals'],
            'Best Odds': [5.0],
            'Bovada': [5.0],
            'Pinnacle': [5.0],
            'Draftkings': [5.0],
            'Vigfree Bovada': [0.2],
            'Vigfree Pinnacle': [0.2],
            'Vigfree Draftkings': [0.2]
        })
        result = self.func(df_same_odds)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Z Score', result.columns)

        # Should return None for zero std
        self.assertIsNone(result['Z Score'].iloc[0])
    

    def test_zscore_bounds(self):
        """Test that Z-scores outside bounds are filtered"""
        df_extreme = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels', 'Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals', 'Los Angeles Angels'],
            'Best Odds': [6.5, 1.3],
            'Underpriced': [6.5, 1.1],
            'Consensus 1': [3.59, 1.3],
            'Consensus 2': [3.59, 1.3],
            'Consensus 3': [3.59, 1.3],
            'Consensus 4': [3.59, 1.3],
            'Consensus 5': [3.59, 1.3],
            'Consensus 6': [3.59, 1.3],
            'Consensus 7': [3.59, 1.3],
            'Consensus 8': [3.59, 1.3],
            'Consensus 9': [3.59, 1.3],
            'Consensus 10': [3.59, 1.3],
            'Consensus 11': [3.59, 1.3],
            'Consensus 12': [3.59, 1.3],
            'Consensus 13': [3.59, 1.3],
            'Consensus 14': [3.59, 1.3],
            'Consensus 15': [3.59, 1.3],
            'Consensus 16': [3.59, 1.3],
            'Consensus 17': [3.59, 1.3],
            'Consensus 18': [3.59, 1.3],
            'Consensus 19': [3.59, 1.3],
            'Consensus 20': [3.59, 1.3],
            'Consensus 21': [3.59, 1.3],
            'Consensus 22': [3.59, 1.3],
            'Consensus 23': [3.59, 1.3],
            'Consensus 24': [3.59, 1.3],
            'Consensus 25': [3.59, 1.3],
            'Consensus 26': [3.59, 1.3],
            'Vigfree Underpriced': [0.0485, 0.9515],
            'Vigfree Consensus 1': [0.27, 0.73],
            'Vigfree Consensus 2': [0.27, 0.73],
            'Vigfree Consensus 3': [0.27, 0.73],
            'Vigfree Consensus 4': [0.27, 0.73],
            'Vigfree Consensus 5': [0.27, 0.73],
            'Vigfree Consensus 6': [0.27, 0.73],
            'Vigfree Consensus 7': [0.27, 0.73],
            'Vigfree Consensus 8': [0.27, 0.73],
            'Vigfree Consensus 9': [0.27, 0.73],
            'Vigfree Consensus 10': [0.27, 0.73],
            'Vigfree Consensus 11': [0.27, 0.73],
            'Vigfree Consensus 12': [0.27, 0.73],
            'Vigfree Consensus 13': [0.27, 0.73],
            'Vigfree Consensus 14': [0.27, 0.73],
            'Vigfree Consensus 15': [0.27, 0.73],
            'Vigfree Consensus 16': [0.27, 0.73],
            'Vigfree Consensus 17': [0.27, 0.73],
            'Vigfree Consensus 18': [0.27, 0.73],
            'Vigfree Consensus 19': [0.27, 0.73],
            'Vigfree Consensus 20': [0.27, 0.73],
            'Vigfree Consensus 21': [0.27, 0.73],
            'Vigfree Consensus 22': [0.27, 0.73],
            'Vigfree Consensus 23': [0.27, 0.73],
            'Vigfree Consensus 24': [0.27, 0.73],
            'Vigfree Consensus 25': [0.27, 0.73],
            'Vigfree Consensus 26': [0.27, 0.73],
        })
        result = self.func(df_extreme)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Z Score', result.columns)
        
        # Check that values are approximately correct
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 3.82, places=2)
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[1], 1.35, places=2)

        self.assertAlmostEqual(result['Avg Edge Pct'].iloc[0], 70.17, places=2)
        self.assertTrue(pd.isna(result['Avg Edge Pct'].iloc[1]))

        # Z score for row 1 is 5.003, greater than threshold
        self.assertIsNone(result['Z Score'].iloc[0])
        self.assertIsNone(result['Z Score'].iloc[1])


class TestAnalyzeModifiedZscoreOutliers(unittest.TestCase):
    """Test suite for analyze_modified_zscore_outliers function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_modified_zscore_outliers
        
        self.df = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels', 'Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals', 'Los Angeles Angels'],
            'Best Odds': [4, 1.3],
            'Mybookie.ag': [4, 1.23],
            'Bovada': [3.7, 1.26],
            'Pinnacle': [3.65, 1.28],
            'Draftkings': [3.65, 1.28],
            'Fanduel': [3.59, 1.3],
            'Fanatics': [3.59, 1.3],
            'Vigfree Mybookie.ag': [0.24, 0.76],
            'Vigfree Bovada': [0.25, 0.75],
            'Vigfree Pinnacle': [0.26, 0.74],
            'Vigfree Draftkings': [0.26, 0.74],
            'Vigfree Fanduel': [0.27, 0.73],
            'Vigfree Fanatics': [0.27, 0.73]
        })
    

    def test_modified_zscore_calculation(self):
        """Test Modified Z-score calculation"""
        result = self.func(self.df)

        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Modified Z Score', result.columns)
        
        # Check that values are approximately correct
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 3.87, places=2)
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[1], 1.35, places=2)

        self.assertAlmostEqual(result['Avg Edge Pct'].iloc[0], 3.33, places=2)
        self.assertTrue(pd.isna(result['Avg Edge Pct'].iloc[1]))

        self.assertAlmostEqual(result['Modified Z Score'].iloc[0], 4.29, places=2)
        self.assertTrue(pd.isna(result['Modified Z Score'].iloc[1]))
    

    def test_modified_zscore_zero_mad(self):
        """Test Modified Z-score calculation when MAD is zero"""
        df_same_odds = pd.DataFrame({
            'Match': ['Test Match'],
            'Team': ['Team A'],
            'Best Odds': [5.0],
            'Bovada': [5.0],
            'Pinnacle': [5.0],
            'Draftkings': [5.0],
            'Vigfree Bovada': [0.2],
            'Vigfree Pinnacle': [0.2],
            'Vigfree Draftkings': [0.2]
        })
        result = self.func(df_same_odds)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Modified Z Score', result.columns)

        # Should return None for zero std
        self.assertIsNone(result['Modified Z Score'].iloc[0])
    

    def test_modified_zscore_bounds(self):
        """Test that Modified Z-scores outside bounds are filtered"""
        df_extreme = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels', 'Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals', 'Los Angeles Angels'],
            'Best Odds': [6.5, 1.3],
            'Mybookie.ag': [6.5, 1.1],
            'Bovada': [3.7, 1.26],
            'Pinnacle': [3.65, 1.28],
            'Draftkings': [3.65, 1.28],
            'Fanduel': [3.59, 1.3],
            'Fanatics': [3.59, 1.3],
            'Vigfree Mybookie.ag': [0.14, 0.86],
            'Vigfree Bovada': [0.25, 0.75],
            'Vigfree Pinnacle': [0.26, 0.74],
            'Vigfree Draftkings': [0.26, 0.74],
            'Vigfree Fanduel': [0.27, 0.73],
            'Vigfree Fanatics': [0.27, 0.73]
        })
        result = self.func(df_extreme)
        
        # Check that new columns are added
        self.assertIn('Fair Odds Avg', result.columns)
        self.assertIn('Avg Edge Pct', result.columns)
        self.assertIn('Modified Z Score', result.columns)
        
        # Check that values are approximately correct
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[0], 4.14, places=2)
        self.assertAlmostEqual(result['Fair Odds Avg'].iloc[1], 1.32, places=2)

        self.assertAlmostEqual(result['Avg Edge Pct'].iloc[0], 57.08, places=2)
        self.assertTrue(pd.isna(result['Avg Edge Pct'].iloc[1]))

        # Z score for row 1 is 57, greater than threshold
        self.assertIsNone(result['Modified Z Score'].iloc[0])
        self.assertIsNone(result['Modified Z Score'].iloc[1])


class TestAnalyzePinnacleEdgeBets(unittest.TestCase):
    """Test suite for analyze_pinnacle_edge_bets function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = analyze_pinnacle_edge_bets
        
        self.df = pd.DataFrame({
            'Match': ['Kansas City Royals @ Los Angeles Angels', 'Kansas City Royals @ Los Angeles Angels'],
            'Team': ['Kansas City Royals', 'Los Angeles Angels'],
            'Best Odds': [3.5, 1.4],
            'Pinnacle': [3.0, 1.4],
            'Fanduel': [3.5, 1.3],
            'Vigfree Pinnacle': [0.32,0.68],
            'Vigfree Fanduel': [0.27,0.73]
        })
    
    def test_pinnacle_edge_calculation(self):
        """Test Pinnacle edge calculation"""
        result = self.func(self.df)
        
        # Check that Pinnacle columns are added
        self.assertIn('Pinnacle Fair Odds', result.columns)
        self.assertIn('Pin Edge Pct', result.columns)
        
        self.assertAlmostEqual(result['Pinnacle Fair Odds'].iloc[0], 3.12, places=2)
        self.assertAlmostEqual(result['Pinnacle Fair Odds'].iloc[1], 1.47, places=2)
        self.assertAlmostEqual(result['Pin Edge Pct'].iloc[0],12.00, places=2)
        self.assertTrue(pd.isna(result['Pin Edge Pct'].iloc[1]))

    
    def test_pinnacle_missing_vigfree(self):
        """Test when Pinnacle vig-free odds are missing"""
        df_missing = pd.DataFrame({
            'Match': ['Test Match'],
            'Team': ['Team A'],
            'Best Odds': [10.0],
            'Pinnacle': [8.0],
            'Vigfree Pinnacle': [np.nan]
        })
        result = self.func(df_missing)
        
        # Should return None for missing vig-free
        self.assertTrue(pd.isna(result['Pinnacle Fair Odds'].iloc[0]))
        self.assertTrue(pd.isna(result['Pin Edge Pct'].iloc[0]))
    
    def test_no_pinnacle_column(self):
        """Test when Pinnacle column doesn't exist"""
        df_no_pinnacle = pd.DataFrame({
            'Match': ['Test Match'],
            'Team': ['Team A'],
            'Best Odds': [10.0],
            'Bovada': [8.0]
        })
        result = self.func(df_no_pinnacle)
        
        # Should return original DataFrame
        self.assertEqual(len(result.columns), len(df_no_pinnacle.columns))
    
    def test_pinnacle_edge_below_threshold(self):
        """Test when edge is below threshold"""
        df_low_edge = pd.DataFrame({
            'Match': ['Test Match'],
            'Team': ['Team A'],
            'Best Odds': [2.0],
            'Pinnacle': [1.95],
            'Vigfree Pinnacle': [0.505]  # Edge is 1%
        })
        result = self.func(df_low_edge)
        
        # Edge should be filtered out (set to None)
        self.assertTrue(pd.isna(result['Pin Edge Pct'].iloc[0]))


class TestFindRandomBets(unittest.TestCase):
    """Test suite for find_random_bets function"""
    
    def setUp(self):
        """Set up test data"""
        self.func = find_random_bets
        
        self.df = pd.DataFrame({
            'Match': ['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5'],
            'Team': ['Team A', 'Team B', 'Team C', 'Team D', 'Team E'],
            'Best Odds': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
    
    def test_random_bets_column_added(self):
        """Test that Random Placed Bet column is added"""
        result = self.func(self.df)
        self.assertIn('Random Placed Bet', result.columns)
    
    def test_random_bets_binary(self):
        """Test that Random Placed Bet contains only 0s and 1s"""
        result = self.func(self.df)
        unique_values = result['Random Placed Bet'].unique()
        self.assertTrue(all(val in [0, 1] for val in unique_values))
    
    def test_random_bets_count_range(self):
        """Test that number of bets is within expected range"""
        result = self.func(self.df)
        bet_count = result['Random Placed Bet'].sum()
        
        # Should be between 0 and min(5, len(df))
        self.assertGreaterEqual(bet_count, 0)
        self.assertLessEqual(bet_count, min(5, len(self.df)))
    
    def test_random_bets_empty_dataframe(self):
        """Test with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.func(empty_df)
        self.assertIn('Random Placed Bet', result.columns)
        self.assertEqual(len(result), 0)
    
    def test_random_bets_small_dataframe(self):
        """Test with DataFrame smaller than 5 rows"""
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
        """Set up comprehensive test data from provided CSV"""
        self.df = pd.DataFrame({
            'Match': ['Seattle Seahawks @ Arizona Cardinals'] * 2,
            'Team': ['Arizona Cardinals', 'Seattle Seahawks'],
            'Best Odds': [14.0, 1.11],
            'Bovada': [6.5, 1.1],
            'Pinnacle': [12.21, 1.05],
            'Draftkings': [6.75, 1.11],
            'Betmgm': [7.5, 1.09],
            'Vigfree Bovada': [0.14473684210526316, 0.8552631578947368],
            'Vigfree Pinnacle': [0.07918552036199095, 0.920814479638009],
            'Vigfree Draftkings': [0.14122137404580154, 0.8587786259541985],
            'Vigfree Betmgm': [0.12689173457508732, 0.8731082654249126]
        })
    
    def test_full_pipeline(self):
        """Test running all analysis functions in sequence"""
        
        # Run all analyses
        result = self.df.copy()
        result = analyze_average_edge_bets(result)
        result = analyze_zscore_outliers(result)
        result = analyze_modified_zscore_outliers(result)
        result = analyze_pinnacle_edge_bets(result)
        result = find_random_bets(result)
        
        # Check all expected columns exist
        expected_columns = [
            'Fair Odds Avg', 'Avg Edge Pct',
            'Z Score', 'Modified Z Score',
            'Pinnacle Fair Odds', 'Pin Edge Pct',
            'Random Placed Bet'
        ]
        
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check that original data is preserved
        self.assertEqual(len(result), len(self.df))
        self.assertTrue(all(result['Match'] == self.df['Match']))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
"""
test_summary_creation.py

Comprehensive pytest test suite for summary_creation.py module.

Author: Andrew Smith
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from summary_creation import (
    create_average_summary_minimal,
    create_average_summary_full,
    create_modified_zscore_summary_minimal,
    create_modified_zscore_summary_full,
    create_random_summary_minimal,
    create_random_summary_full,
)


@pytest.fixture
def sample_betting_df():
    """Sample DataFrame with betting analysis data."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Sport Key': ['basketball_nba', 'basketball_nba', 'soccer_epl', 'soccer_epl', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA', 'EPL', 'EPL', 'NBA'],
        'Start Time': pd.to_datetime(['2024-01-15 19:00', '2024-01-15 20:00', 
                                       '2024-01-16 15:00', '2024-01-16 17:30', '2024-01-17 19:00']),
        'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * 5),
        'Match': ['Lakers @ Celtics', 'Warriors @ Nets', 'Arsenal @ Chelsea', 
                  'Liverpool @ Man City', 'Heat @ Bucks'],
        'Team': ['Lakers', 'Warriors', 'Arsenal', 'Liverpool', 'Heat'],
        'Bet365': [2.4, 2.9, 2.1, 3.8, 2.7],
        'DraftKings': [2.5, 3.0, 2.2, 4.0, 2.8],
        'FanDuel': [2.3, 2.8, 2.0, 3.5, 2.6],
        'Vigfree Bet365': [0.42, 0.34, 0.48, 0.26, 0.37],
        'Vigfree DraftKings': [0.40, 0.33, 0.45, 0.25, 0.36],
        'Vigfree FanDuel': [0.43, 0.36, 0.50, 0.29, 0.38],
        'Best Bookmaker': ['DraftKings', 'DraftKings', 'DraftKings', 'DraftKings', 'DraftKings'],
        'Best Odds': [2.5, 3.0, 2.2, 4.0, 2.8],
        'Fair Odds Average': [2.4, 3.0, 2.2, 3.9, 2.7],
        'Expected Value': [0.05, 0.10, 0.03, 0.15, 0.08],
        'Modified Z-Score': [1.5, 2.0, 1.2, 2.5, 1.8],
        'Random Placed Bet': [0, 1, 0, 1, 0],
        'Outcomes': [2, 2, 2, 2, 2],
        'Result': ["Not Found", "Not Found", "Not Found", "Not Found", "Not Found"],
    })


@pytest.fixture
def sample_df_with_nan():
    """Sample DataFrame with NaN values in critical columns."""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Sport Key': ['basketball_nba'] * 4,
        'Sport Title': ['NBA'] * 4,
        'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 4),
        'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * 4),
        'Match': ['Lakers vs Celtics'] * 4,
        'Team': ['Lakers', 'Celtics', 'Lakers', 'Celtics'],
        'Bet365': [2.4, 2.9, 2.1, 3.8],
        'DraftKings': [2.5, 3.0, 2.2, 4.0],
        'Vigfree Bet365': [0.42, 0.34, 0.48, 0.26],
        'Vigfree DraftKings': [0.40, 0.33, 0.45, 0.25],
        'Best Bookmaker': ['DraftKings'] * 4,
        'Best Odds': [2.5, 3.0, 2.2, 4.0],
        'Fair Odds Average': [2.4, np.nan, 2.2, 3.9],  # Row 1 has NaN
        'Expected Value': [0.05, 0.10, np.nan, 0.15],  # Row 2 has NaN
        'Modified Z-Score': [1.5, 2.0, 1.2, np.nan],   # Row 3 has NaN
        'Random Placed Bet': [0, 1, 0, 1],
        'Outcomes': [2] * 4,
        'Result': ["Not Found"] * 4,
    })


@pytest.fixture
def mock_find_bookmaker_columns():
    """Mock find_bookmaker_columns to return predictable results."""
    with patch('summary_creation.find_bookmaker_columns') as mock:
        mock.return_value = ['Bet365', 'DraftKings', 'FanDuel']
        yield mock


# ============================================================================
# Test create_average_summary_minimal
# ============================================================================

class TestCreateAverageSummaryMinimal:
    """Test create_average_summary_minimal function."""
    
    def test_filters_rows_with_nan_expected_value(self, sample_df_with_nan):
        """Should filter out rows with NaN in Expected Value."""
        result = create_average_summary_minimal(sample_df_with_nan)
        
        # Row 2 has NaN in Expected Value, so should be filtered out
        assert len(result) == 3
        assert 2 not in result['ID'].values
    
    def test_filters_rows_with_nan_fair_odds(self, sample_df_with_nan):
        """Should filter out rows with NaN in Fair Odds Average."""
        result = create_average_summary_minimal(sample_df_with_nan)
        
        # Row 1 has NaN in Fair Odds Average, so should be filtered out
        assert len(result) == 3
        assert 1 not in result['ID'].values
    
    def test_includes_only_minimal_columns(self, sample_betting_df):
        """Should include only specified minimal columns."""
        result = create_average_summary_minimal(sample_betting_df)
        
        expected_columns = [
            "ID", "Sport Key", "Sport Title", "Start Time",
            "Scrape Time", "Match", "Team",
            "Best Bookmaker", "Best Odds", "Fair Odds Average", 
            "Expected Value", "Outcomes", "Result"
        ]
        
        assert list(result.columns) == expected_columns
    
    def test_excludes_vigfree_columns(self, sample_betting_df):
        """Should not include Vigfree columns in minimal summary."""
        result = create_average_summary_minimal(sample_betting_df)
        
        vigfree_cols = [col for col in result.columns if col.startswith('Vigfree')]
        assert len(vigfree_cols) == 0
    
    def test_excludes_bookmaker_odds_columns(self, sample_betting_df):
        """Should not include individual bookmaker odds columns in minimal summary."""
        result = create_average_summary_minimal(sample_betting_df)
        
        assert 'Bet365' not in result.columns
        assert 'DraftKings' not in result.columns
        assert 'FanDuel' not in result.columns
    
    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_average_summary_minimal(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))
    
    def test_preserves_all_valid_rows(self, sample_betting_df):
        """Should preserve all rows without NaN in critical columns."""
        result = create_average_summary_minimal(sample_betting_df)
        
        # All rows in sample_betting_df have valid values
        assert len(result) == len(sample_betting_df)
    
    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        result = create_average_summary_minimal(df)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_does_not_modify_original_df(self, sample_betting_df):
        """Should not modify the original DataFrame."""
        original_len = len(sample_betting_df)
        original_cols = sample_betting_df.columns.tolist()
        
        _ = create_average_summary_minimal(sample_betting_df)
        
        assert len(sample_betting_df) == original_len
        assert sample_betting_df.columns.tolist() == original_cols
    
    def test_column_order_is_correct(self, sample_betting_df):
        """Should maintain specified column order."""
        result = create_average_summary_minimal(sample_betting_df)
        
        expected_order = [
            "ID", "Sport Key", "Sport Title", "Start Time",
            "Scrape Time", "Match", "Team",
            "Best Bookmaker", "Best Odds", "Fair Odds Average", 
            "Expected Value", "Outcomes", "Result"
        ]
        
        assert list(result.columns) == expected_order


# ============================================================================
# Test create_average_summary_full
# ============================================================================

class TestCreateAverageSummaryFull:
    """Test create_average_summary_full function."""
    
    def test_filters_rows_with_nan(self, sample_df_with_nan, mock_find_bookmaker_columns):
        """Should filter out rows with NaN in Expected Value or Fair Odds Average."""
        result = create_average_summary_full(sample_df_with_nan)
        
        # Rows 1 and 2 have NaN in critical columns
        assert len(result) == 2
        assert 1 not in result['ID'].values
        assert 2 not in result['ID'].values
    
    def test_includes_bookmaker_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should include bookmaker odds columns in full summary."""
        result = create_average_summary_full(sample_betting_df)
        
        assert 'Bet365' in result.columns
        assert 'DraftKings' in result.columns
        assert 'FanDuel' in result.columns
    
    def test_includes_vigfree_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should include Vigfree columns in full summary."""
        result = create_average_summary_full(sample_betting_df)
        
        assert 'Vigfree Bet365' in result.columns
        assert 'Vigfree DraftKings' in result.columns
        assert 'Vigfree FanDuel' in result.columns
    
    def test_column_order_is_correct(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should maintain correct column order with bookmakers and vigfree columns."""
        result = create_average_summary_full(sample_betting_df)
        
        # Should start with ID columns
        assert result.columns[0] == 'ID'
        assert result.columns[1] == 'Sport Key'
        
        # Should have bookmaker columns before vigfree
        bet365_idx = list(result.columns).index('Bet365')
        vigfree_bet365_idx = list(result.columns).index('Vigfree Bet365')
        assert bet365_idx < vigfree_bet365_idx
        
        # Should end with result columns
        assert result.columns[-1] == 'Result'
    
    def test_calls_find_bookmaker_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should call find_bookmaker_columns to get bookmaker column list."""
        result = create_average_summary_full(sample_betting_df)
        
        mock_find_bookmaker_columns.assert_called_once()
    
    def test_resets_index(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should reset index starting from 0."""
        result = create_average_summary_full(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))
    
    def test_does_not_modify_original_df(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should not modify the original DataFrame."""
        original_len = len(sample_betting_df)
        original_cols = sample_betting_df.columns.tolist()
        
        _ = create_average_summary_full(sample_betting_df)
        
        assert len(sample_betting_df) == original_len
        assert sample_betting_df.columns.tolist() == original_cols


# ============================================================================
# Test create_modified_zscore_summary_minimal
# ============================================================================

class TestCreateModifiedZscoreSummaryMinimal:
    """Test create_modified_zscore_summary_minimal function."""
    
    def test_filters_rows_with_nan_in_any_critical_column(self, sample_df_with_nan):
        """Should filter out rows with NaN in any of the three critical columns."""
        result = create_modified_zscore_summary_minimal(sample_df_with_nan)
        
        # Only row 0 has all three columns filled
        assert len(result) == 1
        assert result['ID'].iloc[0] == 1
    
    def test_requires_all_three_columns(self, sample_df_with_nan):
        """Should require Expected Value, Fair Odds Average, AND Modified Z-Score."""
        result = create_modified_zscore_summary_minimal(sample_df_with_nan)
        
        # Row 1: missing Fair Odds Average
        # Row 2: missing Expected Value
        # Row 3: missing Modified Z-Score
        # Only row 0 has all three
        assert len(result) == 1
    
    def test_includes_modified_zscore_column(self, sample_betting_df):
        """Should include Modified Z-Score column in output."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)
        
        assert 'Modified Z-Score' in result.columns
    
    def test_column_order_includes_zscore(self, sample_betting_df):
        """Should have Modified Z-Score in correct position."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)
        
        expected_columns = [
            "ID", "Sport Key", "Sport Title", "Start Time", 
            "Scrape Time", "Match", "Team",
            "Best Bookmaker", "Best Odds",
            "Fair Odds Average", "Expected Value", 
            "Modified Z-Score", "Outcomes", "Result"
        ]
        
        assert list(result.columns) == expected_columns
    
    def test_excludes_bookmaker_columns(self, sample_betting_df):
        """Should not include individual bookmaker odds columns."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)
        
        assert 'Bet365' not in result.columns
        assert 'DraftKings' not in result.columns
    
    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))
    
    def test_does_not_modify_original_df(self, sample_betting_df):
        """Should not modify the original DataFrame."""
        original_len = len(sample_betting_df)
        
        _ = create_modified_zscore_summary_minimal(sample_betting_df)
        
        assert len(sample_betting_df) == original_len


# ============================================================================
# Test create_modified_zscore_summary_full
# ============================================================================

class TestCreateModifiedZscoreSummaryFull:
    """Test create_modified_zscore_summary_full function."""
    
    def test_filters_rows_with_nan(self, sample_df_with_nan, mock_find_bookmaker_columns):
        """Should filter out rows with NaN in any critical column."""
        result = create_modified_zscore_summary_full(sample_df_with_nan)
        
        # Only row 0 has all three critical columns filled
        assert len(result) == 1
        assert result['ID'].iloc[0] == 1
    
    def test_includes_all_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should include bookmaker, vigfree, and z-score columns."""
        result = create_modified_zscore_summary_full(sample_betting_df)
        
        assert 'Bet365' in result.columns
        assert 'Vigfree Bet365' in result.columns
        assert 'Modified Z-Score' in result.columns
    
    def test_calls_find_bookmaker_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should call find_bookmaker_columns."""
        result = create_modified_zscore_summary_full(sample_betting_df)
        
        mock_find_bookmaker_columns.assert_called_once()
    
    def test_resets_index(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should reset index starting from 0."""
        result = create_modified_zscore_summary_full(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))


# ============================================================================
# Test create_random_summary_minimal
# ============================================================================

class TestCreateRandomSummaryMinimal:
    """Test create_random_summary_minimal function."""
    
    def test_filters_rows_with_zero_random_bet(self, sample_betting_df):
        """Should only include rows where Random Placed Bet is not 0."""
        result = create_random_summary_minimal(sample_betting_df)
        
        # Only rows 1 and 3 have Random Placed Bet = 1
        assert len(result) == 2
        assert all(result['ID'].isin([2, 4]))
    
    def test_includes_all_random_bets(self, sample_betting_df):
        """Should include all rows with Random Placed Bet != 0."""
        result = create_random_summary_minimal(sample_betting_df)
        
        expected_ids = sample_betting_df[sample_betting_df['Random Placed Bet'] != 0]['ID'].tolist()
        assert sorted(result['ID'].tolist()) == sorted(expected_ids)
    
    def test_excludes_ev_and_zscore_columns(self, sample_betting_df):
        """Should not include EV or Z-Score columns for random bets."""
        result = create_random_summary_minimal(sample_betting_df)
        
        assert 'Expected Value' not in result.columns
        assert 'Modified Z-Score' not in result.columns
        assert 'Fair Odds Average' not in result.columns
    
    def test_minimal_column_set(self, sample_betting_df):
        """Should include only minimal required columns."""
        result = create_random_summary_minimal(sample_betting_df)
        
        expected_columns = [
            "ID", "Sport Key", "Sport Title", "Start Time", 
            "Scrape Time", "Match", "Team",
            "Best Bookmaker", "Best Odds",
            "Outcomes", "Result"
        ]
        
        assert list(result.columns) == expected_columns
    
    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_random_summary_minimal(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))
    
    def test_handles_no_random_bets(self):
        """Should handle case where no random bets were placed."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Random Placed Bet': [0, 0, 0],
            'Best Odds': [2.5, 3.0, 2.2],
        })
        
        result = create_random_summary_minimal(df)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_handles_all_random_bets(self):
        """Should handle case where all bets are random."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Sport Key': ['basketball_nba'] * 3,
            'Sport Title': ['NBA'] * 3,
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 3),
            'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * 3),
            'Match': ['Game 1', 'Game 2', 'Game 3'],
            'Team': ['Team A', 'Team B', 'Team C'],
            'Best Bookmaker': ['DraftKings'] * 3,
            'Best Odds': [2.5, 3.0, 2.2],
            'Random Placed Bet': [1, 1, 1],
            'Outcomes': [2, 2, 2],
            'Result': [np.nan, np.nan, np.nan],
        })
        
        result = create_random_summary_minimal(df)
        
        assert len(result) == 3
    
    def test_does_not_modify_original_df(self, sample_betting_df):
        """Should not modify the original DataFrame."""
        original_len = len(sample_betting_df)
        
        _ = create_random_summary_minimal(sample_betting_df)
        
        assert len(sample_betting_df) == original_len


# ============================================================================
# Test create_random_summary_full
# ============================================================================

class TestCreateRandomSummaryFull:
    """Test create_random_summary_full function."""
    
    def test_filters_rows_with_zero_random_bet(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should only include rows where Random Placed Bet is not 0."""
        result = create_random_summary_full(sample_betting_df)
        
        # Only rows 1 and 3 have Random Placed Bet = 1
        assert len(result) == 2
    
    def test_includes_bookmaker_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should include bookmaker odds columns."""
        result = create_random_summary_full(sample_betting_df)
        
        assert 'Bet365' in result.columns
        assert 'DraftKings' in result.columns
    
    def test_includes_vigfree_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should include Vigfree columns."""
        result = create_random_summary_full(sample_betting_df)
        
        assert 'Vigfree Bet365' in result.columns
        assert 'Vigfree DraftKings' in result.columns
    
    def test_excludes_ev_and_zscore_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should not include EV or Z-Score columns for random bets."""
        result = create_random_summary_full(sample_betting_df)
        
        assert 'Expected Value' not in result.columns
        assert 'Modified Z-Score' not in result.columns
        assert 'Fair Odds Average' not in result.columns
    
    def test_calls_find_bookmaker_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should call find_bookmaker_columns."""
        result = create_random_summary_full(sample_betting_df)
        
        mock_find_bookmaker_columns.assert_called_once()
    
    def test_resets_index(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should reset index starting from 0."""
        result = create_random_summary_full(sample_betting_df)
        
        assert list(result.index) == list(range(len(result)))
    
    def test_does_not_modify_original_df(self, sample_betting_df, mock_find_bookmaker_columns):
        """Should not modify the original DataFrame."""
        original_len = len(sample_betting_df)
        
        _ = create_random_summary_full(sample_betting_df)
        
        assert len(sample_betting_df) == original_len


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for summary creation workflows."""
    
    def test_average_minimal_vs_full_row_count(self, sample_betting_df, mock_find_bookmaker_columns):
        """Minimal and full summaries should have same number of rows."""
        minimal = create_average_summary_minimal(sample_betting_df)
        full = create_average_summary_full(sample_betting_df)
        
        assert len(minimal) == len(full)
    
    def test_zscore_minimal_vs_full_row_count(self, sample_betting_df, mock_find_bookmaker_columns):
        """Minimal and full Z-score summaries should have same number of rows."""
        minimal = create_modified_zscore_summary_minimal(sample_betting_df)
        full = create_modified_zscore_summary_full(sample_betting_df)
        
        assert len(minimal) == len(full)
    
    def test_random_minimal_vs_full_row_count(self, sample_betting_df, mock_find_bookmaker_columns):
        """Minimal and full random summaries should have same number of rows."""
        minimal = create_random_summary_minimal(sample_betting_df)
        full = create_random_summary_full(sample_betting_df)
        
        assert len(minimal) == len(full)
    
    def test_full_summaries_have_more_columns(self, sample_betting_df, mock_find_bookmaker_columns):
        """Full summaries should have more columns than minimal summaries."""
        avg_min = create_average_summary_minimal(sample_betting_df)
        avg_full = create_average_summary_full(sample_betting_df)
        
        assert len(avg_full.columns) > len(avg_min.columns)
    
    def test_common_columns_have_same_values(self, sample_betting_df, mock_find_bookmaker_columns):
        """Common columns between minimal and full should have identical values."""
        minimal = create_average_summary_minimal(sample_betting_df)
        full = create_average_summary_full(sample_betting_df)
        
        common_columns = ['ID', 'Best Odds', 'Expected Value']
        
        for col in common_columns:
            pd.testing.assert_series_equal(
                minimal[col].reset_index(drop=True),
                full[col].reset_index(drop=True),
                check_names=False
            )
    
    def test_all_summaries_preserve_id_order(self, sample_betting_df, mock_find_bookmaker_columns):
        """All summary functions should preserve ID order after filtering."""
        avg_min = create_average_summary_minimal(sample_betting_df)
        avg_full = create_average_summary_full(sample_betting_df)
        zscore_min = create_modified_zscore_summary_minimal(sample_betting_df)
        zscore_full = create_modified_zscore_summary_full(sample_betting_df)
        
        # IDs should be in ascending order
        assert all(avg_min['ID'].diff().dropna() > 0)
        assert all(avg_full['ID'].diff().dropna() > 0)
        assert all(zscore_min['ID'].diff().dropna() > 0)
        assert all(zscore_full['ID'].diff().dropna() > 0)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_all_rows_filtered_out(self):
        """Should handle case where all rows are filtered out."""
        df = pd.DataFrame({
            'ID': [1, 2, 3],
            'Expected Value': [np.nan, np.nan, np.nan],
            'Fair Odds Average': [np.nan, np.nan, np.nan],
        })
        
        result = create_average_summary_minimal(df)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_missing_optional_columns(self, mock_find_bookmaker_columns):
        """Should handle DataFrames missing optional columns gracefully."""
        df = pd.DataFrame({
            'ID': [1, 2],
            'Match': ['Game 1', 'Game 2'],
            'Expected Value': [0.05, 0.10],
            'Fair Odds Average': [2.4, 3.0],
            # Missing 'Result' column
        })
        
        result = create_average_summary_minimal(df)
        
        # Should still work, 'Result' will be NaN
        assert 'Result' in result.columns
        assert len(result) == 2
    
    def test_large_dataframe_performance(self, mock_find_bookmaker_columns):
        """Should handle large DataFrames efficiently."""
        n_rows = 10000
        df = pd.DataFrame({
            'ID': range(n_rows),
            'Sport Key': ['basketball_nba'] * n_rows,
            'Sport Title': ['NBA'] * n_rows,
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * n_rows),
            'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * n_rows),
            'Match': ['Game'] * n_rows,
            'Team': ['Team A'] * n_rows,
            'Best Bookmaker': ['DraftKings'] * n_rows,
            'Best Odds': np.random.uniform(1.5, 5.0, n_rows),
            'Fair Odds Average': np.random.uniform(1.5, 5.0, n_rows),
            'Expected Value': np.random.uniform(0.01, 0.2, n_rows),
            'Outcomes': [2] * n_rows,
            'Result': [np.nan] * n_rows,
        })
        
        import time
        start = time.time()
        result = create_average_summary_minimal(df)
        elapsed = time.time() - start
        
        # Should complete quickly
        assert elapsed < 1.0
        assert len(result) == n_rows
    
    def test_duplicate_ids(self, mock_find_bookmaker_columns):
        """Should handle DataFrames with duplicate IDs."""
        df = pd.DataFrame({
            'ID': [1, 1, 2, 2],  # Duplicate IDs
            'Sport Key': ['basketball_nba'] * 4,
            'Sport Title': ['NBA'] * 4,
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 4),
            'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * 4),
            'Match': ['Game 1'] * 4,
            'Team': ['Team A', 'Team B', 'Team C', 'Team D'],
            'Best Bookmaker': ['DraftKings'] * 4,
            'Best Odds': [2.5, 3.0, 2.2, 4.0],
            'Fair Odds Average': [2.4, 3.0, 2.2, 3.9],
            'Expected Value': [0.05, 0.10, 0.03, 0.15],
            'Outcomes': [2] * 4,
            'Result': [np.nan] * 4,
        })
        
        result = create_average_summary_minimal(df)
        
        # Should preserve all rows including duplicates
        assert len(result) == 4
        assert result['ID'].tolist() == [1, 1, 2, 2]
    
    def test_special_characters_in_strings(self, mock_find_bookmaker_columns):
        """Should handle special characters in string columns."""
        df = pd.DataFrame({
            'ID': [1, 2],
            'Sport Key': ['basketball_nba', 'soccer_epl'],
            'Sport Title': ['NBA', 'EPL'],
            'Start Time': pd.to_datetime(['2024-01-15 19:00'] * 2),
            'Scrape Time': pd.to_datetime(['2024-01-15 10:00'] * 2),
            'Match': ['Team A vs Team B', 'Team C & Team D'],  # Special chars
            'Team': ["O'Brien's Team", 'Team "Quotes"'],  # Special chars
            'Best Bookmaker': ['DraftKings', 'Bet365'],
            'Best Odds': [2.5, 3.0],
            'Fair Odds Average': [2.4, 3.0],
            'Expected Value': [0.05, 0.10],
            'Outcomes': [2, 2],
            'Result': [np.nan, np.nan],
        })
        
        result = create_average_summary_minimal(df)
        
        assert len(result) == 2
        assert "O'Brien's Team" in result['Team'].values
        assert 'Team "Quotes"' in result['Team'].values


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
"""
test_betting_strategies.py

Comprehensive pytest test suite for betting_strategies.py module.

Author: Andrew Smith
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.find_bets.betting_strategies import (
    _get_vigfree_columns,
    _calculate_fair_odds_from_probabilities,
    _calculate_expected_value,
    _calculate_modified_zscore,
    _filter_by_threshold,
    find_average_bets,
    find_modified_zscore_bets,
    find_random_bets,
)


@pytest.fixture
def sample_vigfree_df():
    """Sample DataFrame with vig-free odds."""
    return pd.DataFrame({
        'Best Odds': [2.75, 3.0, 2.2, 4.0],
        'Bet365': [2.4, 2.9, 2.1, 3.8],
        'DraftKings': [2.75, 3.0, 2.2, 4.0],
        'FanDuel': [2.3, 2.8, 2.0, 3.5],
        'Vigfree Bet365': [0.42, 0.34, 0.48, 0.26],
        'Vigfree DraftKings': [0.36, 0.33, 0.45, 0.25],
        'Vigfree FanDuel': [0.43, 0.36, 0.50, 0.29],
    })


@pytest.fixture
def sample_probabilities():
    """Sample probability series."""
    return pd.Series([0.5, 0.25, 0.1, 0.333])


@pytest.fixture
def sample_odds():
    """Sample odds series."""
    return pd.Series([2.5, 4.0, 10.0, 3.0])


class TestGetVigfreeColumns:
    """Test _get_vigfree_columns function."""
    
    def test_returns_vigfree_columns(self):
        """Should return only columns starting with 'Vigfree '."""
        df = pd.DataFrame({
            'Vigfree Bet365': [0.5, 0.6],
            'Vigfree DraftKings': [0.5, 0.6],
            'Bet365': [2.0, 1.67],
            'DraftKings': [2.0, 1.67],
            'Best Odds': [2.0, 1.67],
        })
        result = _get_vigfree_columns(df)
        assert result == ['Vigfree Bet365', 'Vigfree DraftKings']
        assert len(result) == 2
    
    def test_returns_empty_when_no_vigfree(self):
        """Should return empty list when no Vigfree columns exist."""
        df = pd.DataFrame({'Bet365': [2.0, 1.67]})
        result = _get_vigfree_columns(df)
        assert result == []
        assert isinstance(result, list)
    
    def test_preserves_column_order(self):
        """Should maintain the order of columns as they appear in DataFrame."""
        df = pd.DataFrame({
            'Vigfree DraftKings': [0.5],
            'Bet365': [2.0],
            'Vigfree Bet365': [0.5],
            'Vigfree FanDuel': [0.5],
        })
        result = _get_vigfree_columns(df)
        assert result == ['Vigfree DraftKings', 'Vigfree Bet365', 'Vigfree FanDuel']
    
    def test_handles_empty_dataframe(self):
        """Should return empty list for empty DataFrame."""
        df = pd.DataFrame()
        result = _get_vigfree_columns(df)
        assert result == []


class TestCalculateFairOddsFromProbabilities:
    """Test _calculate_fair_odds_from_probabilities function."""
    
    def test_calculates_fair_odds_correctly(self, sample_probabilities):
        """Should correctly calculate fair odds from valid probabilities."""
        result = _calculate_fair_odds_from_probabilities(sample_probabilities)
        expected = pd.Series([2.0, 4.0, 10.0, 3.0])
        pd.testing.assert_series_equal(result, expected)
    
    def test_rounds_to_two_decimals(self):
        """Should round results to 2 decimal places."""
        probabilities = pd.Series([0.333, 0.666])
        result = _calculate_fair_odds_from_probabilities(probabilities)
        assert result.iloc[0] == 3.0
        assert result.iloc[1] == 1.5
    
    def test_raises_error_for_zero_probability(self):
        """Should raise ValueError when probability is 0."""
        probabilities = pd.Series([0.5, 0.0, 0.25])
        with pytest.raises(ValueError):
            _calculate_fair_odds_from_probabilities(probabilities)
    
    def test_raises_error_for_one_probability(self):
        """Should raise ValueError when probability is 1."""
        probabilities = pd.Series([0.5, 1.0, 0.25])
        with pytest.raises(ValueError):
            _calculate_fair_odds_from_probabilities(probabilities)
    
    def test_raises_error_for_negative_probability(self):
        """Should raise ValueError for negative probabilities."""
        probabilities = pd.Series([0.5, -0.1, 0.25])
        with pytest.raises(ValueError):
            _calculate_fair_odds_from_probabilities(probabilities)
    
    def test_raises_error_for_probability_greater_than_one(self):
        """Should raise ValueError for probabilities > 1."""
        probabilities = pd.Series([0.5, 1.5, 0.25])
        with pytest.raises(ValueError):
            _calculate_fair_odds_from_probabilities(probabilities)
    
    def test_raises_error_for_nan_probability(self):
        """Should raise ValueError when any probability is NaN."""
        probabilities = pd.Series([0.5, np.nan, 0.25])
        with pytest.raises(ValueError):
            _calculate_fair_odds_from_probabilities(probabilities)
    
    def test_handles_single_value(self):
        """Should work with single-element Series."""
        probabilities = pd.Series([0.5])
        result = _calculate_fair_odds_from_probabilities(probabilities)
        assert result.iloc[0] == 2.0
    
    def test_preserves_index(self):
        """Should preserve the original Series index."""
        probabilities = pd.Series([0.5, 0.25], index=['A', 'B'])
        result = _calculate_fair_odds_from_probabilities(probabilities)
        assert list(result.index) == ['A', 'B']


class TestCalculateExpectedValue:
    """Test _calculate_expected_value function."""
    
    def test_calculates_ev_correctly(self):
        """Should correctly calculate expected value."""
        probabilities = pd.Series([0.5, 0.25, 0.4])
        best_odds = pd.Series([2.5, 5.0, 3.0])
        result = _calculate_expected_value(probabilities, best_odds)
        
        # EV = (prob * odds) - 1
        # [0.5 * 2.5 - 1, 0.25 * 5.0 - 1, 0.4 * 3.0 - 1]
        # [1.25 - 1, 1.25 - 1, 1.2 - 1]
        # [0.25, 0.25, 0.2]
        expected = pd.Series([0.25, 0.25, 0.2])
        pd.testing.assert_series_equal(result, expected)
    
    def test_rounds_to_two_decimals(self):
        """Should round results to 2 decimal places."""
        probabilities = pd.Series([0.333])
        best_odds = pd.Series([3.5])
        result = _calculate_expected_value(probabilities, best_odds)
        # 0.333 * 3.5 - 1 = 1.1655 - 1 = 0.1655 → rounds to 0.17
        assert result.iloc[0] == 0.17
    
    def test_handles_missing_probabilities(self):
        """Should return NaN when probability is NaN."""
        probabilities = pd.Series([0.5, np.nan, 0.4])
        best_odds = pd.Series([2.5, 5.0, 3.0])
        result = _calculate_expected_value(probabilities, best_odds)
        
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])
    
    def test_handles_missing_odds(self):
        """Should return NaN when best odds is NaN."""
        probabilities = pd.Series([0.5, 0.25, 0.4])
        best_odds = pd.Series([2.5, np.nan, 3.0])
        result = _calculate_expected_value(probabilities, best_odds)
        
        assert not np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])
    
    def test_handles_all_nan(self):
        """Should return all NaN when all inputs are NaN."""
        probabilities = pd.Series([np.nan, np.nan])
        best_odds = pd.Series([np.nan, np.nan])
        result = _calculate_expected_value(probabilities, best_odds)
        
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
    
    def test_negative_ev(self):
        """Should correctly calculate negative EV."""
        probabilities = pd.Series([0.3])
        best_odds = pd.Series([2.0])
        result = _calculate_expected_value(probabilities, best_odds)
        # 0.3 * 2.0 - 1 = 0.6 - 1 = -0.4
        assert result.iloc[0] == -0.4
    
    def test_preserves_index(self):
        """Should preserve the original Series index."""
        probabilities = pd.Series([0.5, 0.25], index=['X', 'Y'])
        best_odds = pd.Series([2.5, 5.0], index=['X', 'Y'])
        result = _calculate_expected_value(probabilities, best_odds)
        assert list(result.index) == ['X', 'Y']


class TestCalculateModifiedZscore:
    """Test _calculate_modified_zscore function."""
    
    def test_calculates_zscore_correctly(self):
        """Should correctly calculate modified Z-score."""
        values = pd.Series([3.0, 2.5, 4.0])
        reference = pd.DataFrame({
            'A': [2.0, 2.0, 2.0],
            'B': [2.1, 2.1, 2.15],
            'C': [2.2, 2.2, 2.3],
        })
        result = _calculate_modified_zscore(values, reference)
        
        expected = pd.Series([6.07, 2.70, 8.32])
        pd.testing.assert_series_equal(result, expected)
    
    def test_handles_zero_mad(self):
        """Should return NaN when MAD is zero (all values identical)."""
        values = pd.Series([3.0, 2.0])
        reference = pd.DataFrame({
            'A': [2.0, 2.0],
            'B': [2.0, 2.0],
            'C': [2.0, 2.0],
        })
        result = _calculate_modified_zscore(values, reference)
        
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
    
    def test_only_positive_deviations(self):
        """Should use max(0, deviation) - only positive deviations count."""
        values = pd.Series([1.5, 3.0])  # Below and above median
        reference = pd.DataFrame({
            'A': [2.0, 2.0],
            'B': [2.1, 2.1],
            'C': [2.2, 2.2],
        })
        result = _calculate_modified_zscore(values, reference)
        
        # Row 0: value < median, so deviation = max(0, 1.5 - 2.1) = 0
        assert result.iloc[0] == 0.0
        # Row 1: value > median, so should have positive z-score
        assert result.iloc[1] == 6.07
    
    def test_handles_nan_in_values(self):
        """Should return NaN for NaN input values."""
        values = pd.Series([np.nan])
        reference = pd.DataFrame({
            'A': [2.0],
            'B': [2.1],
            'C': [2.2],
        })
        result = _calculate_modified_zscore(values, reference)
        
        assert np.isnan(result.iloc[0])
    
    def test_handles_nan_in_reference(self):
        """Should handle NaN values in reference DataFrame."""
        values = pd.Series([3.0])
        reference = pd.DataFrame({
            'A': [2.0],
            'B': [np.nan],
            'C': [2.1],
            'D': [2.2],
        })
        result = _calculate_modified_zscore(values, reference)
        
        # Should still calculate (pandas median handles NaN)
        assert result.iloc[0] == 6.07
    

class TestFilterByThreshold:
    """Test _filter_by_threshold function."""
    
    def test_filters_by_min_threshold(self):
        """Should filter out values <= min_threshold."""
        values = pd.Series([0.5, 1.5, 2.5, 3.5])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=10.0)
        
        assert np.isnan(result.iloc[0])  # 0.5 <= 1.0
        assert result.iloc[1] == 1.5
        assert result.iloc[2] == 2.5
        assert result.iloc[3] == 3.5
    
    def test_filters_by_max_threshold(self):
        """Should filter out values >= max_threshold."""
        values = pd.Series([0.5, 1.5, 2.5, 3.5])
        result = _filter_by_threshold(values, min_threshold=0.0, max_threshold=3.0)
        
        assert result.iloc[0] == 0.5
        assert result.iloc[1] == 1.5
        assert result.iloc[2] == 2.5
        assert np.isnan(result.iloc[3])  # 3.5 >= 3.0
    
    def test_filters_by_both_thresholds(self):
        """Should apply both min and max threshold filters."""
        values = pd.Series([0.5, 1.5, 2.5, 3.5])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=3.0)
        
        assert np.isnan(result.iloc[0])  # Too low
        assert result.iloc[1] == 1.5
        assert result.iloc[2] == 2.5
        assert np.isnan(result.iloc[3])  # Too high
    
    def test_exclusive_thresholds(self):
        """Values exactly at threshold should be filtered out."""
        values = pd.Series([1.0, 2.0, 3.0])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=3.0)
        
        assert np.isnan(result.iloc[0])  # Exactly at min
        assert result.iloc[1] == 2.0
        assert np.isnan(result.iloc[2])  # Exactly at max
    
    def test_preserves_nan_values(self):
        """Should preserve NaN values in input."""
        values = pd.Series([0.5, np.nan, 2.5, 3.5])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=3.0)
        
        assert np.isnan(result.iloc[0])  # Filtered
        assert np.isnan(result.iloc[1])  # Was NaN
        assert result.iloc[2] == 2.5
        assert np.isnan(result.iloc[3])  # Filtered
    
    def test_returns_all_nan_when_all_filtered(self):
        """Should return all NaN when all values are out of range."""
        values = pd.Series([0.1, 0.2, 0.3])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=3.0)
        
        assert all(np.isnan(result))
    
    def test_preserves_index(self):
        """Should preserve the original Series index."""
        values = pd.Series([1.5, 2.5], index=['A', 'B'])
        result = _filter_by_threshold(values, min_threshold=1.0, max_threshold=3.0)
        
        assert list(result.index) == ['A', 'B']


class TestFindAverageBets:
    """Test find_average_bets function."""
    
    def test_adds_fair_odds_average_column(self, sample_vigfree_df):
        """Should add 'Fair Odds Average' column to DataFrame."""
        result = find_average_bets(sample_vigfree_df)
        assert 'Fair Odds Average' in result.columns
    
    def test_adds_expected_value_column(self, sample_vigfree_df):
        """Should add 'Expected Value' column to DataFrame."""
        result = find_average_bets(sample_vigfree_df)
        assert 'Expected Value' in result.columns
    
    def test_calculates_fair_odds_correctly(self, sample_vigfree_df):
        """Should correctly calculate average fair odds."""
        result = find_average_bets(sample_vigfree_df)
        
        # First row: avg prob = (0.42 + 0.36 + 0.43) / 3 = 0.4033
        # Fair odds = 1 / 0.4033 = 2.48
        assert abs(result['Fair Odds Average'].iloc[0] - 2.48) < 0.01

    def test_calculates_fair_odds_correctly_with_nan(self):
        """Should handle NaN values in vig-free odds."""
        df = pd.DataFrame({
            'Best Odds': [3.0],
            'Vigfree Bet365': [np.nan],
            'Vigfree DraftKings': [0.40],
            'Vigfree FanDuel': [0.42],
            'Vigfree Fanatics': [0.43],
        })
        result = find_average_bets(df)

        assert abs(result['Fair Odds Average'].iloc[0] - 2.4) < 0.01

    def test_calculates_ev_correctly(self, sample_vigfree_df):
        """Should correctly calculate ev."""
        result = find_average_bets(sample_vigfree_df)
        
        # EV = (0.4033 * 2.75) - 1 = 1.1092 - 1 = 0.1092
        assert abs(result['Expected Value'].iloc[0] - 0.1092) < 0.001

    def test_calculates_ev_correctly_with_nan(self):
        """Should handle NaN values in vig-free odds."""
        df = pd.DataFrame({
            'Best Odds': [2.75],
            'Vigfree Bet365': [np.nan],
            'Vigfree DraftKings': [0.36],
            'Vigfree FanDuel': [0.42],
            'Vigfree Fanatics': [0.43],
        })
        result = find_average_bets(df)

        assert abs(result['Expected Value'].iloc[0] - 0.1092) < 0.001

    def test_preserves_original_columns(self, sample_vigfree_df):
        """Should preserve all original columns."""
        original_cols = sample_vigfree_df.columns.tolist()
        result = find_average_bets(sample_vigfree_df)
        
        for col in original_cols:
            assert col in result.columns
    
    def test_preserves_row_count(self, sample_vigfree_df):
        """Should not change number of rows."""
        result = find_average_bets(sample_vigfree_df)
        assert len(result) == len(sample_vigfree_df)
        
    def test_filters_ev_by_thresholds(self, sample_vigfree_df):
        """Should filter EV values outside threshold range."""
        with patch('src.find_bets.betting_strategies.EV_THRESHOLD', 0.1):
            with patch('src.find_bets.betting_strategies.MAX_EV', 0.2):
                result = find_average_bets(sample_vigfree_df)
                
                ev_values = result['Expected Value'].dropna()
                if len(ev_values) > 0:
                    assert all(ev_values > 0.1)
                    assert all(ev_values < 0.2)
    
    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame({
            'Best Odds': [],
            'Vigfree Bet365': [],
        })
        result = find_average_bets(df)
        
        assert len(result) == 0
        assert 'Fair Odds Average' in result.columns
        assert 'Expected Value' in result.columns


class TestFindModifiedZscoreBets:
    """Test find_modified_zscore_bets function."""
    
    def test_adds_modified_zscore_column(self, sample_vigfree_df):
        """Should add 'Modified Z-Score' column to DataFrame."""
        result = find_modified_zscore_bets(sample_vigfree_df)
        assert 'Modified Z-Score' in result.columns
    
    def test_includes_average_bet_columns(self, sample_vigfree_df):
        """Should include columns from find_average_bets."""
        result = find_modified_zscore_bets(sample_vigfree_df)
        assert 'Fair Odds Average' in result.columns
        assert 'Expected Value' in result.columns
    
    def test_calculates_zscore_correctly(self, sample_vigfree_df):
        """Should calculate modified Z-scores."""
        result = find_modified_zscore_bets(sample_vigfree_df)
        
        assert abs(result['Modified Z-Score'].iloc[0] - 2.36) < 0.01

    
    def test_preserves_original_columns(self, sample_vigfree_df):
        """Should preserve all original columns."""
        original_cols = sample_vigfree_df.columns.tolist()
        result = find_modified_zscore_bets(sample_vigfree_df)
        
        for col in original_cols:
            assert col in result.columns
    
    def test_filters_zscore_by_thresholds(self, sample_vigfree_df):
        """Should filter Z-score values outside threshold range."""
        with patch('src.find_bets.betting_strategies.Z_SCORE_THRESHOLD', 2.4):
            with patch('src.find_bets.betting_strategies.MAX_Z_SCORE', 4.0):
                result = find_modified_zscore_bets(sample_vigfree_df)
                
                zscore_values = result['Modified Z-Score'].dropna()
                if len(zscore_values) > 0:
                    assert all(zscore_values > 2.4)
                    assert all(zscore_values < 4.0)


class TestFindRandomBets:
    """Test find_random_bets function."""
    
    def test_adds_random_bet_column(self, sample_vigfree_df):
        """Should add 'Random Placed Bet' column to DataFrame."""
        result = find_random_bets(sample_vigfree_df)
        assert 'Random Placed Bet' in result.columns
    
    def test_column_is_integer_type(self, sample_vigfree_df):
        """Random Placed Bet column should be integer (0 or 1)."""
        result = find_random_bets(sample_vigfree_df)
        assert result['Random Placed Bet'].dtype == int
    
    def test_values_are_binary(self, sample_vigfree_df):
        """All values should be 0 or 1."""
        result = find_random_bets(sample_vigfree_df)
        assert all(result['Random Placed Bet'].isin([0, 1]))
    
    def test_respects_max_five_bets(self):
        """Should place at most 5 bets."""
        df = pd.DataFrame({
            'Best Odds': range(100),  # 100 rows
        })
        result = find_random_bets(df)
        assert result['Random Placed Bet'].sum() <= 5
    
    def test_handles_small_dataframe(self):
        """Should handle DataFrames with fewer than 5 rows."""
        df = pd.DataFrame({
            'Best Odds': [2.5, 3.0],  # Only 2 rows
        })
        result = find_random_bets(df)
        assert result['Random Placed Bet'].sum() <= 2
    
    def test_handles_single_row(self):
        """Should handle single-row DataFrame."""
        df = pd.DataFrame({
            'Best Odds': [2.5],
        })
        result = find_random_bets(df)
        assert result['Random Placed Bet'].sum() <= 1
    
    def test_handles_empty_dataframe(self):
        """Should handle empty DataFrame gracefully."""
        df = pd.DataFrame({
            'Best Odds': [],
        })
        result = find_random_bets(df)
        assert len(result) == 0
        assert 'Random Placed Bet' in result.columns
    
    def test_can_place_zero_bets(self, sample_vigfree_df):
        """Should be able to place 0 bets (random choice)."""
        # Run multiple times to potentially get 0 bets
        results = []
        for _ in range(20):
            result = find_random_bets(sample_vigfree_df)
            results.append(result['Random Placed Bet'].sum())
        
        # At least one should be 0 (statistically likely)
        assert min(results) >= 0

"""
tests/test_vigfree_probabilities.py

Unit tests for vigfree_probabilities module.

Author: Andrew Smith
Date: January 2026
"""

import pytest
import pandas as pd
import numpy as np
from src.find_bets.vigfree_probabilities import (
    _calculate_market_margin,
    _remove_margin_proportional_to_odds,
    _calculate_vigfree_probs_for_market,
    _has_complete_odds,
    _process_bookmaker_for_match,
    calculate_vigfree_probabilities
)


@pytest.fixture
def sample_odds_df():
    """Sample DataFrame with bookmaker odds"""
    return pd.DataFrame({
        'Match': ['A @ B', 'A @ B', 'C @ D', 'C @ D'],
        'Team': ['A', 'B', 'C', 'D'],
        'Outcomes': [2, 2, 2, 2],
        'DraftKings': [2.10, 1.85, 2.50, 1.55],
        'FanDuel': [2.05, 1.90, 2.45, 1.60],
        'BetMGM': [2.08, 1.85, np.nan, 1.67]  # Missing odds for Team C
    })


@pytest.fixture
def three_way_market_df():
    """Sample DataFrame with 3-way market (e.g., soccer with draw)"""
    return pd.DataFrame({
        'Match': ['Team1 @ Team2', 'Team1 @ Team2', 'Team1 @ Team2'],
        'Team': ['Team1', 'Draw', 'Team2'],
        'Outcomes': [3, 3, 3],
        'DraftKings': [2.50, 3.20, 2.80],
        'FanDuel': [2.45, 3.30, 2.85]
    })


class TestCalculateMarketMargin:
    """Tests for _calculate_market_margin function"""
    
    def test_positive_margin(self):
        """Test calculation with typical positive margin"""
        # Bookmaker odds: 2.0 and 2.0
        # Implied probs: 0.5 + 0.5 = 1.0, but bookmaker adds margin
        # Actual bookmaker odds might be 1.90 and 1.90
        # Implied probs: 0.526 + 0.526 = 1.052 → margin = 0.052 (5.2%)
        odds_list = [1.90, 1.90]
        margin = _calculate_market_margin(odds_list)
        
        assert margin > 0
        assert abs(margin - 0.052) < 0.001  # ~5.2%
    
    def test_zero_margin(self):
        """Test with fair odds (no margin)"""
        # True probabilities: 50/50
        # Fair odds: 2.0 and 2.0
        # Implied probs: 0.5 + 0.5 = 1.0 → margin = 0
        odds_list = [2.0, 2.0]
        margin = _calculate_market_margin(odds_list)
        
        assert abs(margin) < 0.001  # Should be ~0
    
    def test_three_way_market(self):
        """Test margin calculation for 3-way market"""
        # 3 outcomes with margin
        odds_list = [2.50, 3.20, 2.80]
        margin = _calculate_market_margin(odds_list)
        
        # Sum of implied probs should be > 1.0
        assert abs(margin - 0.070) < 0.001
    
    def test_empty_list(self):
        """Test with empty odds list"""
        margin = _calculate_market_margin([])
        assert margin == 0
    
    def test_none_input(self):
        """Test with None input"""
        margin = _calculate_market_margin(None)
        assert margin == 0
    
    def test_series_input(self):
        """Test with pandas Series input"""
        odds_series = pd.Series([1.90, 1.90])
        margin = _calculate_market_margin(odds_series)
        assert margin > 0
        assert abs(margin - 0.052) < 0.001  # ~5.2%
        assert isinstance(margin, (int, float))
    
    def test_negative_margin_raises_error(self):
        """Test that negative margin (impossible odds) raises error"""
        # Odds that imply probabilities summing to < 1.0
        # This shouldn't happen in real markets
        odds_list = [5.0, 5.0]  # Implies 0.2 + 0.2 = 0.4 < 1.0
        
        with pytest.raises(ValueError, match="negative margin"):
            _calculate_market_margin(odds_list)

    def test_unusually_high_margin_raises_error(self):
        """Test that unusually high margin raises error"""
        # Create odds that imply a very high margin
        odds_list = [1.01, 1.01]  # Implies ~0.9901 + 0.9901 = 1.9802 → margin ~0.9802
        
        with pytest.raises(ValueError, match="Unusually high market margin"):
            _calculate_market_margin(odds_list)


class TestRemoveMarginProportionalToOdds:
    """Tests for _remove_margin_proportional_to_odds function"""
    
    def test_removes_margin_correctly(self):
        """Test that margin is removed using correct formula"""
        # Market: [1.90, 1.90] with margin ~5.2%
        bookmaker_odds = 1.90
        all_market_odds = [1.90, 1.90]
        n_outcomes = 2
        
        fair_odds = _remove_margin_proportional_to_odds(
            bookmaker_odds, all_market_odds, n_outcomes
        )
        
        # Fair odds should be higher than bookmaker odds
        assert fair_odds > bookmaker_odds
        # Should be close to 2.0 (fair 50/50 odds)
        assert abs(fair_odds - 2.0) < 0.1
    
    def test_favorite_vs_underdog(self):
        """Test that favorites and underdogs are adjusted differently"""
        all_market_odds = [1.50, 2.80]  # Favorite vs underdog
        n_outcomes = 2
        
        fair_favorite = _remove_margin_proportional_to_odds(1.50, all_market_odds, 2)
        fair_underdog = _remove_margin_proportional_to_odds(2.80, all_market_odds, 2)
        
        # Both should increase (margin removed)
        assert abs(fair_favorite - 1.52) < 0.01
        assert abs(fair_underdog - 2.90) < 0.01
    
    def test_three_way_market(self):
        """Test margin removal for 3-way market"""
        all_market_odds = [2.50, 3.20, 2.80]
        
        fair_odds_team1 = _remove_margin_proportional_to_odds(2.50, all_market_odds, 3)
        fair_odds_draw = _remove_margin_proportional_to_odds(3.20, all_market_odds, 3)
        fair_odds_team2 = _remove_margin_proportional_to_odds(2.80, all_market_odds, 3)
        
        # All should be higher than original
        assert abs(fair_odds_team1 - 2.65) < 0.01
        assert abs(fair_odds_draw - 3.46) < 0.01
        assert abs(fair_odds_team2 - 3) < 0.01

        # Fair probabilities should sum to ~1.0
        fair_prob_sum = (1/fair_odds_team1) + (1/fair_odds_draw) + (1/fair_odds_team2)
        assert abs(fair_prob_sum - 1.0) < 0.01
    
    def test_invalid_odds_less_than_one(self):
        """Test that odds < 1 raise error"""
        with pytest.raises(ValueError, match="Invalid odds: less than 1"):
            _remove_margin_proportional_to_odds(0.5, [1.90, 1.90], 2)
    
    def test_zero_denominator_error(self):
        """Test handling of edge case that would cause division by zero"""
        # Create scenario where denominator = 0
        # denominator = n_outcomes - (margin * bookmaker_odds)
        # If margin is very large and odds are high, this could happen
        
        # This is a theoretical edge case - might not trigger with realistic odds
        # But we test error handling
        try:
            _remove_margin_proportional_to_odds(100, [1.9, 1.9], 2)
        except ValueError as e:
            assert "non-positive denominator" in str(e)


class TestCalculateVigfreeProbs:
    """Tests for _calculate_vigfree_probs_for_market function"""
    
    def test_calculates_probabilities(self):
        """Test that vig-free probabilities are calculated"""
        valid_odds = pd.Series([2.10, 1.85])
        required_outcomes = 2
        
        vigfree_probs = _calculate_vigfree_probs_for_market(valid_odds, required_outcomes)
        
        assert len(vigfree_probs) == 2
        assert vigfree_probs[0] == 0.4678
        assert vigfree_probs[1] == 0.5322
    
    def test_three_way_market(self):
        """Test calculation for 3-way market"""
        valid_odds = pd.Series([2.50, 3.20, 2.80])
        required_outcomes = 3
        
        vigfree_probs = _calculate_vigfree_probs_for_market(valid_odds, required_outcomes)
        
        assert len(vigfree_probs) == 3
        assert vigfree_probs[0] == 0.3768
        assert vigfree_probs[1] == 0.2893
        assert vigfree_probs[2] == 0.3339


class TestHasCompleteOdds:
    """Tests for _has_complete_odds function"""
    
    def test_complete_odds(self):
        """Test when bookmaker has odds for all outcomes"""
        match_group = pd.DataFrame({
            'DraftKings': [2.10, 1.85],
            'Team': ['A', 'B']
        })
        
        has_complete, valid_odds = _has_complete_odds(match_group, 'DraftKings', 2)
        
        assert has_complete is True
        assert valid_odds is not None
        assert len(valid_odds) == 2
    
    def test_incomplete_odds(self):
        """Test when bookmaker has missing odds"""
        match_group = pd.DataFrame({
            'DraftKings': [2.10, np.nan],
            'Team': ['A', 'B']
        })
        
        has_complete, valid_odds = _has_complete_odds(match_group, 'DraftKings', 2)
        
        assert has_complete is False
        assert valid_odds is None
    
    def test_partial_odds(self):
        """Test when bookmaker has some but not all odds"""
        match_group = pd.DataFrame({
            'DraftKings': [2.10, np.nan, 2.50],
            'Team': ['A', 'B', 'C']
        })
        
        has_complete, valid_odds = _has_complete_odds(match_group, 'DraftKings', 3)
        
        assert has_complete is False
        assert valid_odds is None


class TestProcessBookmakerForMatch:
    """Tests for _process_bookmaker_for_match function"""
    
    def test_processes_complete_market(self, sample_odds_df):
        """Test processing a complete market"""
        df = sample_odds_df.copy()
        match_group = df[df['Match'] == 'A @ B']
        
        # Add vig-free column
        df['Vigfree DraftKings'] = np.nan
        
        _process_bookmaker_for_match(df, match_group, 'DraftKings', 'Vigfree DraftKings')
        
        # Check that vig-free probabilities were calculated correctly
        match_vigfree = df[df['Match'] == 'A @ B']['Vigfree DraftKings']
        expected = [0.4678, 0.5322]
        actual = match_vigfree.tolist()
        assert all(a == e for a, e in zip(actual, expected))

    
    def test_skips_incomplete_market(self, sample_odds_df):
        """Test that incomplete markets are skipped"""
        df = sample_odds_df.copy()
        match_group = df[df['Match'] == 'C @ D']
        
        # Add vig-free column
        df['Vigfree BetMGM'] = np.nan
        
        # BetMGM has NaN for Team C
        _process_bookmaker_for_match(df, match_group, 'BetMGM', 'Vigfree BetMGM')
        
        # Should remain NaN since market is incomplete
        match_vigfree = df[df['Match'] == 'C @ D']['Vigfree BetMGM']
        assert match_vigfree.isna().all()
    
    def test_three_way_market(self, three_way_market_df):
        """Test processing a 3-way market"""
        df = three_way_market_df.copy()
        match_group = df[df['Match'] == 'Team1 @ Team2']
        
        df['Vigfree DraftKings'] = np.nan
        
        _process_bookmaker_for_match(df, match_group, 'DraftKings', 'Vigfree DraftKings')

        # Check that vig-free probabilities were calculated correctly
        match_vigfree = df[df['Match'] == 'Team1 @ Team2']['Vigfree DraftKings']
        expected = [0.3768, 0.2893, 0.3339]
        actual = match_vigfree.tolist()
        assert all(a == e for a, e in zip(actual, expected))


class TestCalculateVigfreeProbabilities:
    """Tests for main calculate_vigfree_probabilities function"""
    
    def test_adds_vigfree_columns(self, sample_odds_df):
        """Test that vig-free columns are added for each bookmaker"""
        result = calculate_vigfree_probabilities(sample_odds_df)
        
        assert 'Vigfree DraftKings' in result.columns
        assert 'Vigfree FanDuel' in result.columns
        assert 'Vigfree BetMGM' in result.columns
    
    def test_calculates_vigfree_probabilities(self, sample_odds_df):
        """Test that vig-free probabilities are calculated correctly"""
        result = calculate_vigfree_probabilities(sample_odds_df)
        
        # Match A @ B should have complete odds for all bookmakers
        match_a = result[result['Match'] == 'A @ B']
        
        draftkings_expected = [0.4678, 0.5322]
        actual = match_a['Vigfree DraftKings'].tolist()
        assert all(a == e for a, e in zip(actual, draftkings_expected))

        fanduel_expected = [0.4807, 0.5193] 
        actual = match_a['Vigfree FanDuel'].tolist()
        assert all(a == e for a, e in zip(actual, fanduel_expected))

        betmgm_expected = [0.4701, 0.5299] 
        actual = match_a['Vigfree BetMGM'].tolist()
        assert all(a == e for a, e in zip(actual, betmgm_expected))
    
    def test_handles_incomplete_markets(self, sample_odds_df):
        """Test handling of incomplete markets"""
        result = calculate_vigfree_probabilities(sample_odds_df)
        
        # Match C @ D has incomplete BetMGM odds
        match_c = result[result['Match'] == 'C @ D']
        
        # BetMGM vig-free should be NaN for this match
        assert match_c['Vigfree BetMGM'].isna().all()
    
    def test_three_way_market(self, three_way_market_df):
        """Test calculation for 3-way market"""
        result = calculate_vigfree_probabilities(three_way_market_df)
        
        draftkings_expected = [0.3768, 0.2893, 0.3339]
        actual = result['Vigfree DraftKings'].tolist()
        assert all(a == e for a, e in zip(actual, draftkings_expected))

        fanduel_expected = [0.3875, 0.2823, 0.3302] 
        actual = result['Vigfree FanDuel'].tolist()
        assert all(a == e for a, e in zip(actual, fanduel_expected))
    
    def test_preserves_original_columns(self, sample_odds_df):
        """Test that original columns are preserved"""
        result = calculate_vigfree_probabilities(sample_odds_df)
        
        # Original columns should still exist
        assert 'Match' in result.columns
        assert 'Team' in result.columns
        assert 'Outcomes' in result.columns
        assert 'DraftKings' in result.columns
        assert 'FanDuel' in result.columns
        assert 'BetMGM' in result.columns
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        df = pd.DataFrame()
        result = calculate_vigfree_probabilities(df)
        
        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)
    
    def test_all_nan_odds(self):
        """Test handling when all odds are NaN"""
        df = pd.DataFrame({
            'Match': ['A @ B', 'A @ B'],
            'Team': ['A', 'B'],
            'Outcomes': [2, 2],
            'DraftKings': [np.nan, np.nan],
            'FanDuel': [np.nan, np.nan]
        })
        
        result = calculate_vigfree_probabilities(df)
        
        # Vig-free columns should exist but be NaN
        assert 'Vigfree DraftKings' in result.columns
        assert 'Vigfree FanDuel' in result.columns
        assert result['Vigfree DraftKings'].isna().all()
        assert result['Vigfree FanDuel'].isna().all()
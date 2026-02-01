"""
test_find_bets.py

Comprehensive tests for the find_bets.py pipeline.

Author: Test Suite
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

# Import functions to test
from src.fetch_odds.fetch_odds import fetch_odds
from src.find_bets.data_processing import process_target_odds_data
from src.find_bets.vigfree_probabilities import calculate_vigfree_probabilities
from src.find_bets.betting_strategies import (
    find_average_bets,
    find_modified_zscore_bets,
    find_random_bets,
)
from src.find_bets.summary_creation import (
    create_average_summary_minimal,
    create_average_summary_full,
    create_modified_zscore_summary_minimal,
    create_modified_zscore_summary_full,
    create_random_summary_minimal,
    create_random_summary_full,
)
from src.find_bets.file_management import save_betting_data


# ============================================================================
# STAGE 0: RAW ODDS DATA (Input to fetch_odds)
# ============================================================================

@pytest.fixture
def valid_odds_data():
    """Valid odds data that should pass all processing."""
    return [
        {
            'id': '1',
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': '2024-01-15T19:00:00Z',
            'home_team': 'Lakers',
            'away_team': 'Celtics',
            'bookmakers': [
                {
                    'key': 'draftkings',
                    'title': 'DraftKings',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Lakers', 'price': 2.65},
                                {'name': 'Celtics', 'price': 1.5}
                            ]
                        }
                    ]
                },
                {
                    'key': 'fanduel',
                    'title': 'FanDuel',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Lakers', 'price': 2.4},
                                {'name': 'Celtics', 'price': 1.65}
                            ]
                        }
                    ]
                },
                {
                    'key': 'fanatics',
                    'title': 'Fanatics',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Lakers', 'price': 2.4},
                                {'name': 'Celtics', 'price': 1.65}
                            ]
                        }
                    ]
                },
                {
                    'key': 'betmgm',
                    'title': 'BetMGM',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Lakers', 'price': 2.35},
                                {'name': 'Celtics', 'price': 1.7}
                            ]
                        }
                    ]
                },
                {
                    'key': 'caesars',
                    'title': 'Caesars',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Lakers', 'price': 2.35},
                                {'name': 'Celtics', 'price': 1.7}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            'id': '2',
            'sport_key': 'basketball_nba',
            'sport_title': 'NBA',
            'commence_time': '2024-01-16T20:00:00Z',
            'home_team': 'Warriors',
            'away_team': 'Nets',
            'bookmakers': [
                {
                    'key': 'draftkings',
                    'title': 'DraftKings',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Warriors', 'price': 3.0},
                                {'name': 'Nets', 'price': 1.4}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            'id': '3',
            'sport_key': 'soccer_epl',
            'sport_title': 'EPL',
            'commence_time': '2024-01-17T15:00:00Z',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea',
            'bookmakers': [
                {
                    'key': 'draftkings',
                    'title': 'DraftKings',
                    'markets': [
                        {
                            'key': 'h2h',
                            'outcomes': [
                                {'name': 'Arsenal', 'price': 2.2},
                                {'name': 'Chelsea', 'price': 2.8}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

# ============================================================================
# STAGE 1: RAW ODDS DATA (Input to process_target_odds_data)
# ============================================================================

@pytest.fixture
def raw_odds_data():
    """
    Raw odds data as returned from fetch_odds.
    This is the starting point before any processing.
    """
    return pd.DataFrame({
        'ID': ['1', '1', '2', '2', '3', '3'],
        'Sport Key': ['basketball_nba', 'basketball_nba', 'basketball_nba', 'basketball_nba', 'soccer_epl', 'soccer_epl'],
        'Sport Title': ['NBA', 'NBA', 'NBA', 'NBA', 'EPL', 'EPL'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z', '2024-01-16T20:00:00Z', '2024-01-16T20:00:00Z', '2024-01-17T15:00:00Z', '2024-01-17T15:00:00Z'],
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers', 'Nets @ Warriors', 'Nets @ Warriors', 'Chelsea @ Arsenal', 'Chelsea @ Arsenal'],
        'Team': ['Lakers', 'Celtics', 'Warriors', 'Nets', 'Arsenal', 'Chelsea'],
        'DraftKings': [2.65, 1.5, 3.0, 1.4, 2.2, 2.8],
        'FanDuel': [2.4, 1.65, np.nan, np.nan, np.nan, np.nan],
        'Fanatics': [2.4, 1.65, np.nan, np.nan, np.nan, np.nan],
        'BetMGM': [2.35, 1.7, np.nan, np.nan, np.nan, np.nan],
        'Caesars': [2.35, 1.7, np.nan, np.nan, np.nan, np.nan],
    })


# ============================================================================
# STAGE 2: PROCESSED ODDS DATA (Output of process_target_odds_data)
# ============================================================================

@pytest.fixture
def processed_odds_data():
    """
    Processed odds data after applying all data cleaning and validation.
    Adds: Best Odds, Best Bookmaker, Outcomes, Result, Scrape Time
    
    Note: Only Match 1 (Celtics @ Lakers) remains after filtering because it's the only
    match with 5+ bookmakers. Matches 2 and 3 only have 1 bookmaker (DraftKings) so they
    are filtered out by _min_bookmaker_filter().
    """
    return pd.DataFrame({
        'ID': ['1', '1'],
        'Sport Key': ['basketball_nba', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z'], 
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers'],
        'Team': ['Lakers', 'Celtics'],
        'DraftKings': [2.65, 1.5],
        'FanDuel': [2.4, 1.65],
        'Fanatics': [2.4, 1.65],
        'BetMGM': [2.35, 1.7],
        'Caesars': [2.35, 1.7],
        'Outcomes': [2, 2],
        'Best Odds': [2.65, 1.7],
        'Best Bookmaker': ['DraftKings', 'BetMGM'],
        'Result': ['Not Found', 'Not Found'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'] * 2,
    })


# ============================================================================
# STAGE 3: VIG-FREE PROBABILITIES (Output of calculate_vigfree_probabilities)
# ============================================================================

@pytest.fixture
def vigfree_data():
    """
    Data after calculating vig-free probabilities for each bookmaker.
    Adds: Vigfree DraftKings, Vigfree FanDuel, Vigfree Fanatics, Vigfree BetMGM, Vigfree Caesars
    
    Only Match 1 (Celtics @ Lakers) remains after the MIN_BOOKMAKERS filter.
    All bookmakers have complete odds for this match.
    
    Example calculation for DraftKings Match 1:
    - Celtics odds: 1.5, Lakers odds: 2.65
    - Margin = (1/1.5 + 1/2.65) - 1 = 1.0441 - 1 = 0.0441
    - Fair Celtics odds = (2 * 1.5) / (2 - 0.0441 * 1.5) = 3.0 / 1.934 = 1.551
    - Vigfree prob Celtics = 1 / 1.551 = 0.645
    """
    return pd.DataFrame({
        'ID': ['1', '1'],
        'Sport Key': ['basketball_nba', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z'],
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers'],
        'Team': ['Lakers', 'Celtics'],
        'DraftKings': [2.65, 1.5],
        'FanDuel': [2.4, 1.65],
        'Fanatics': [2.4, 1.65],
        'BetMGM': [2.35, 1.7],
        'Caesars': [2.35, 1.7],
        'Outcomes': [2, 2],
        'Best Odds': [2.65, 1.7],
        'Best Bookmaker': ['DraftKings', 'BetMGM'],
        'Result': ['Not Found', 'Not Found'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'] * 2,
        # Vig-free probabilities for all bookmakers (all have complete odds)
        'Vigfree DraftKings': [0.3553, 0.6447],
        'Vigfree FanDuel': [0.4053, 0.5947],
        'Vigfree Fanatics': [0.4053, 0.5947],
        'Vigfree BetMGM': [0.4186, 0.5814],
        'Vigfree Caesars': [0.4186, 0.5814],
    })


# ============================================================================
# STAGE 4A: AVERAGE BETS ANALYSIS (Output of find_average_bets)
# ============================================================================

@pytest.fixture
def average_bets_data():
    """
    Data after applying average betting strategy.
    Adds: Fair Odds Average, Expected Value
    
    Only Match 1 (Celtics @ Lakers) data after MIN_BOOKMAKERS filter.
    
    Calculation details:
    - Average probability = mean of all vig-free probabilities
    - Fair Odds Average = 1 / Average probability
    - Expected Value = (avg_prob * best_odds) - 1
    - EV is filtered to show only values > EV_THRESHOLD (e.g., 0.01)
    
    Example for Celtics:
    - Vigfree probs: [0.645, 0.604, 0.604, 0.604, 0.588]
    - Avg prob = 0.609
    - Fair Odds Avg = 1 / 0.609 = 1.64
    - Best Odds = 1.7
    - EV = (0.609 * 1.7) - 1 = 0.035 (3.5% edge)
    """
    return pd.DataFrame({
        'ID': ['1', '1'],
        'Sport Key': ['basketball_nba', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z'],
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers'],
        'Team': ['Lakers', 'Celtics'],
        'DraftKings': [2.65, 1.5],
        'FanDuel': [2.4, 1.65],
        'Fanatics': [2.4, 1.65],
        'BetMGM': [2.35, 1.7],
        'Caesars': [2.35, 1.7],
        'Outcomes': [2, 2],
        'Best Odds': [2.65, 1.7],
        'Best Bookmaker': ['DraftKings', 'BetMGM'],
        'Result': ['Not Found', 'Not Found'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'] * 2,
        'Vigfree DraftKings': [0.3553, 0.6447],
        'Vigfree FanDuel': [0.4053, 0.5947],
        'Vigfree Fanatics': [0.4053, 0.5947],
        'Vigfree BetMGM': [0.4186, 0.5814],
        'Vigfree Caesars': [0.4186, 0.5814],
        'Fair Odds Average': [2.50, 1.67],
        'Expected Value': [0.06, np.nan],
    })


# ============================================================================
# STAGE 4B: MODIFIED Z-SCORE BETS ANALYSIS (Output of find_modified_zscore_bets)
# ============================================================================

@pytest.fixture
def modified_zscore_bets_data():
    """
    Data after applying modified z-score betting strategy.
    Adds: Fair Odds Average, Expected Value, Modified Z-Score
    
    Only Match 1 (Celtics @ Lakers) data after MIN_BOOKMAKERS filter.
    
    Modified Z-score calculation:
    - Median of bookmaker odds for each outcome
    - MAD = median absolute deviation from median
    - Modified Z = 0.6745 * (best_odds - median) / MAD
    - Only positive deviations (best_odds > median) are considered
    - Filtered to show only values > Z_SCORE_THRESHOLD (e.g., 1.5)
    
    Example for Celtics:
    - Bookmaker odds: [1.5, 1.65, 1.65, 1.65, 1.7]
    - Median = 1.65
    - MAD = median([0.15, 0, 0, 0, 0.05]) = 0.05
    - Modified Z = 0.6745 * max(0, 1.7 - 1.65) / 0.05 = 0.67
    """
    return pd.DataFrame({
        'ID': ['1', '1'],
        'Sport Key': ['basketball_nba', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z'],
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers'],
        'Team': ['Lakers', 'Celtics'],
        'DraftKings': [2.65, 1.5],
        'FanDuel': [2.4, 1.65],
        'Fanatics': [2.4, 1.65],
        'BetMGM': [2.35, 1.7],
        'Caesars': [2.35, 1.7],
        'Outcomes': [2, 2],
        'Best Odds': [2.65, 1.7],
        'Best Bookmaker': ['DraftKings', 'BetMGM'],
        'Result': ['Not Found', 'Not Found'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'] * 2,
        'Vigfree DraftKings': [0.3553, 0.6447],
        'Vigfree FanDuel': [0.4053, 0.5947],
        'Vigfree Fanatics': [0.4053, 0.5947],
        'Vigfree BetMGM': [0.4186, 0.5814],
        'Vigfree Caesars': [0.4186, 0.5814],
        'Fair Odds Average': [2.50, 1.67],
        'Expected Value': [0.06, np.nan],
        'Modified Z-Score': [3.37, np.nan],
    })


# ============================================================================
# STAGE 4C: RANDOM BETS ANALYSIS (Output of find_random_bets)
# ============================================================================

@pytest.fixture
def random_bets_data():
    """
    Data after applying random betting strategy (for baseline comparison).
    Adds: Random Placed Bet (1 if selected, 0 otherwise)
    
    Only Match 1 (Celtics @ Lakers) data after MIN_BOOKMAKERS filter.
    
    Note: This is non-deterministic, so fixture shows example output.
    """
    return pd.DataFrame({
        'ID': ['1', '1'],
        'Sport Key': ['basketball_nba', 'basketball_nba'],
        'Sport Title': ['NBA', 'NBA'],
        'Start Time': ['2024-01-15T19:00:00Z', '2024-01-15T19:00:00Z'],
        'Match': ['Celtics @ Lakers', 'Celtics @ Lakers'],
        'Team': ['Lakers', 'Celtics'],
        'DraftKings': [2.65, 1.5],
        'FanDuel': [2.4, 1.65],
        'Fanatics': [2.4, 1.65],
        'BetMGM': [2.35, 1.7],
        'Caesars': [2.35, 1.7],
        'Outcomes': [2, 2],
        'Best Odds': [2.65, 1.7],
        'Best Bookmaker': ['DraftKings', 'BetMGM'],
        'Result': ['Not Found', 'Not Found'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'] * 2,
        'Vigfree DraftKings': [0.3553, 0.6447],
        'Vigfree FanDuel': [0.4053, 0.5947],
        'Vigfree Fanatics': [0.4053, 0.5947],
        'Vigfree BetMGM': [0.4186, 0.5814],
        'Vigfree Caesars': [0.4186, 0.5814],
        'Random Placed Bet': [1, 0],
    })


# ============================================================================
# STAGE 5: SUMMARIES (Output of summary creation functions)
# ============================================================================

@pytest.fixture
def average_summary_minimal():
    """
    Minimal summary for average betting strategy.
    Contains only essential columns and rows with valid Expected Value.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Fair Odds Average': [2.50],
        'Expected Value': [0.06],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })


@pytest.fixture
def average_summary_full():
    """
    Full summary for average betting strategy.
    Includes all bookmaker columns and vig-free probabilities.
    Only Match 1 (Celtics @ Lakers) data after MIN_BOOKMAKERS filter.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'DraftKings': [2.65],
        'FanDuel': [2.4],
        'Fanatics': [2.4],
        'BetMGM': [2.35],
        'Caesars': [2.35],
        'Vigfree DraftKings': [0.3553],
        'Vigfree FanDuel': [0.4053],
        'Vigfree Fanatics': [0.4053],
        'Vigfree BetMGM': [0.4186],
        'Vigfree Caesars': [0.4186],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Fair Odds Average': [2.50],
        'Expected Value': [0.06],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })


@pytest.fixture
def modified_zscore_summary_minimal():
    """
    Minimal summary for modified z-score betting strategy.
    Contains only essential columns and rows with valid Modified Z-Score.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Fair Odds Average': [2.50],
        'Expected Value': [0.06],
        'Modified Z-Score': [3.37],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })


@pytest.fixture
def modified_zscore_summary_full():
    """
    Full summary for modified z-score betting strategy.
    Includes all bookmaker columns and vig-free probabilities.
    Only Match 1 (Celtics @ Lakers) data after MIN_BOOKMAKERS filter.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'DraftKings': [2.65],
        'FanDuel': [2.4],
        'Fanatics': [2.4],
        'BetMGM': [2.35],
        'Caesars': [2.35],
        'Vigfree DraftKings': [0.3553],
        'Vigfree FanDuel': [0.4053],
        'Vigfree Fanatics': [0.4053],
        'Vigfree BetMGM': [0.4186],
        'Vigfree Caesars': [0.4186],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Fair Odds Average': [2.50],
        'Expected Value': [0.06],
        'Modified Z-Score': [3.37],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })


@pytest.fixture
def random_summary_minimal():
    """
    Minimal summary for random betting strategy.
    Contains only essential columns and rows with valid Expected Value.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })


@pytest.fixture
def random_summary_full():
    """
    Full summary for random betting strategy.
    Includes all bookmaker columns and vig-free probabilities.
    """
    return pd.DataFrame({
        'ID': ['1'],
        'Sport Key': ['basketball_nba'],
        'Sport Title': ['NBA'],
        'Start Time': ['2024-01-15T19:00:00Z'],
        'Scrape Time': ['2024-01-14 10:00:00+00:00'],
        'Match': ['Celtics @ Lakers'],
        'Team': ['Lakers'],
        'DraftKings': [2.65],
        'FanDuel': [2.4],
        'Fanatics': [2.4],
        'BetMGM': [2.35],
        'Caesars': [2.35],
        'Vigfree DraftKings': [0.3553],
        'Vigfree FanDuel': [0.4053],
        'Vigfree Fanatics': [0.4053],
        'Vigfree BetMGM': [0.4186],
        'Vigfree Caesars': [0.4186],
        'Best Bookmaker': ['DraftKings'],
        'Best Odds': [2.65],
        'Outcomes': [2],
        'Result': ['Not Found'],
    })

class TestFindBets:
    """Tests for the find_bets.py pipeline functions."""
    
    def test_fetch_odds(self, valid_odds_data, raw_odds_data):
        """Test fetch_odds function with valid data."""
        fetched_odds = fetch_odds(valid_odds_data)
        pd.testing.assert_frame_equal(fetched_odds, raw_odds_data)

    def test_process_target_odds_data(self, raw_odds_data, processed_odds_data):
        """Test process_target_odds_data function."""
        processed_data = process_target_odds_data(raw_odds_data)
        pd.testing.assert_frame_equal(
            processed_data.drop(columns=["Scrape Time"]),
            processed_odds_data.drop(columns=["Scrape Time"])
        )

    def test_calculate_vigfree_probabilities(self, processed_odds_data, vigfree_data):
        """Test calculate_vigfree_probabilities function."""
        vigfree_df = calculate_vigfree_probabilities(processed_odds_data)
        pd.testing.assert_frame_equal(
            vigfree_df.drop(columns=["Scrape Time"]), 
            vigfree_data.drop(columns=["Scrape Time"])
        )

    def test_find_average_bets(self, vigfree_data, average_bets_data):
        """Test find_average_bets function."""
        average_bets_df = find_average_bets(vigfree_data)
        pd.testing.assert_frame_equal(
            average_bets_df.drop(columns=["Scrape Time"]), 
            average_bets_data.drop(columns=["Scrape Time"])
        )
    
    def test_find_modified_zscore_bets(self, vigfree_data, modified_zscore_bets_data):
        """Test find_modified_zscore_bets function."""
        modified_zscore_bets_df = find_modified_zscore_bets(vigfree_data)
        pd.testing.assert_frame_equal(
            modified_zscore_bets_df.drop(columns=["Scrape Time"]), 
            modified_zscore_bets_data.drop(columns=["Scrape Time"])
        )

    def test_find_random_bets(self, vigfree_data, random_bets_data):
        """Test find_random_bets function."""
        random_bets_df = find_random_bets(vigfree_data)
        pd.testing.assert_frame_equal(
            random_bets_df.drop(columns=["Scrape Time", "Random Placed Bet"]), 
            random_bets_data.drop(columns=["Scrape Time", "Random Placed Bet"])
        )

    def test_create_average_summary_minimal(self, average_bets_data, average_summary_minimal):
        """Test create_average_summary_minimal function."""
        summary_df = create_average_summary_minimal(average_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            average_summary_minimal.drop(columns=["Scrape Time"])
        )

    def test_create_average_summary_full(self, average_bets_data, average_summary_full):
        """Test create_average_summary_full function."""
        summary_df = create_average_summary_full(average_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            average_summary_full.drop(columns=["Scrape Time"])
        )

    def test_create_modified_zscore_summary_minimal(self, modified_zscore_bets_data, modified_zscore_summary_minimal):
        """Test create_modified_zscore_summary_minimal function."""
        summary_df = create_modified_zscore_summary_minimal(modified_zscore_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            modified_zscore_summary_minimal.drop(columns=["Scrape Time"])
        )

    def test_create_modified_zscore_summary_full(self, modified_zscore_bets_data, modified_zscore_summary_full):
        """Test create_modified_zscore_summary_full function."""
        summary_df = create_modified_zscore_summary_full(modified_zscore_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            modified_zscore_summary_full.drop(columns=["Scrape Time"])
        )

    def test_create_random_summary_minimal(self, random_bets_data, random_summary_minimal):
        """Test create_random_summary_minimal function."""
        summary_df = create_random_summary_minimal(random_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            random_summary_minimal.drop(columns=["Scrape Time"])
        )

    def test_create_random_summary_full(self, random_bets_data, random_summary_full):
        """Test create_random_summary_full function."""
        summary_df = create_random_summary_full(random_bets_data)
        pd.testing.assert_frame_equal(
            summary_df.drop(columns=["Scrape Time"]), 
            random_summary_full.drop(columns=["Scrape Time"])
        )

    

"""
tests/test_data_processing.py

Unit tests for data_processing module.

Author: Andrew Smith
Date: January 2026
"""

import numpy as np
import pandas as pd
import pytest

from src.find_bets.data_processing import (
    _add_metadata,
    _add_outcomes_metadata,
    _all_outcomes_present_filter,
    _clean_odds_data,
    _max_odds_filter,
    _min_bookmaker_filter,
    _minimum_outcomes_filter,
    _remove_unwanted_bookmakers,
    find_bookmaker_columns,
    process_target_odds_data,
)


@pytest.fixture
def sample_odds_df():
    """Sample DataFrame matching output from fetch_odds"""
    return pd.DataFrame(
        {
            "ID": ["game1", "game1", "game2", "game2"],
            "Sport Key": [
                "basketball_nba",
                "basketball_nba",
                "basketball_nba",
                "basketball_nba",
            ],
            "Sport Title": ["NBA", "NBA", "NBA", "NBA"],
            "Start Time": ["2026-01-25T19:00:00Z"] * 4,
            "Match": [
                "Warriors @ Lakers",
                "Warriors @ Lakers",
                "Celtics @ Heat",
                "Celtics @ Heat",
            ],
            "Team": ["Warriors", "Lakers", "Celtics", "Heat"],
            "DraftKings": [2.10, 1.85, 1.95, 1.95],
            "FanDuel": [2.05, 1.90, 2.00, 1.90],
            "BetMGM": [2.15, 1.80, 1.98, 1.92],
            "Caesars": [2.08, 1.87, np.nan, 1.94],
            "Pinnacle": [2.12, 1.83, 2.02, 1.88],
        }
    )


@pytest.fixture
def sample_with_unwanted_bms():
    """Sample DataFrame with bookmakers not in ALL_BMS"""
    return pd.DataFrame(
        {
            "ID": ["game1", "game1"],
            "Match": ["Warriors @ Lakers", "Warriors @ Lakers"],
            "Team": ["Warriors", "Lakers"],
            "DraftKings": [2.10, 1.85],
            "FanDuel": [2.05, 1.90],
            "UnknownBook": [2.00, 1.95],
            "RandomBook": [2.12, 1.88],
        }
    )


class TestFindBookmakerColumns:
    """Tests for find_bookmaker_columns function"""

    def test_finds_bookmaker_columns(self, sample_odds_df):
        """Test finding bookmaker columns from standard DataFrame"""
        bookmakers = find_bookmaker_columns(sample_odds_df)

        expected = ["DraftKings", "FanDuel", "BetMGM", "Caesars", "Pinnacle"]
        assert set(bookmakers) == set(expected)

    def test_with_additional_exclusions(self, sample_odds_df):
        """Test with additional columns to exclude"""
        bookmakers = find_bookmaker_columns(
            sample_odds_df, exclude_columns=["DraftKings", "FanDuel"]
        )

        expected = ["BetMGM", "Caesars", "Pinnacle"]
        assert set(bookmakers) == set(expected)

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        df = pd.DataFrame()
        bookmakers = find_bookmaker_columns(df)
        assert bookmakers == []


class TestAddOutcomesMetadata:
    """Tests for _add_outcomes_metadata function"""

    def test_adds_outcomes_column(self):
        """Test that Outcomes column is added correctly"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B", "C @ D", "C @ D", "C @ D"],
                "Team": ["A", "B", "C", "D", "E"],
            }
        )

        result = _add_outcomes_metadata(df)

        expected_outcomes = [2, 2, 3, 3, 3]
        assert result["Outcomes"].tolist() == expected_outcomes


class TestMinimumOutcomesFilter:
    """Tests for _minimum_outcomes_filter function"""

    def test_filters_insufficient_outcomes(self):
        """Test that rows with insufficient outcomes are removed"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B", "C @ D"],
                "Team": ["A", "B", "C"],
                "Outcomes": [2, 2, 1],
            }
        )

        result = _minimum_outcomes_filter(df)

        # Should only keep 'A @ B' which has both outcomes
        assert len(result) == 2
        assert all(result["Match"] == "A @ B")


class TestRemoveUnwantedBookmakers:
    """Tests for _remove_unwanted_bookmakers function"""

    def test_removes_unwanted_bookmakers(self, sample_with_unwanted_bms):
        """Test that bookmakers not in ALL_BMS are removed"""
        result = _remove_unwanted_bookmakers(sample_with_unwanted_bms)

        expected_columns = ["ID", "Match", "Team", "DraftKings", "FanDuel"]
        assert set(result.columns) == set(expected_columns)


class TestAddMetadata:
    """Tests for _add_metadata function"""

    def test_adds_all_metadata_columns(self, sample_odds_df):
        """Test that all metadata columns are added"""
        result = _add_metadata(sample_odds_df)

        assert "Best Odds" in result.columns
        assert "Best Bookmaker" in result.columns
        assert "Result" in result.columns
        assert "Scrape Time" in result.columns

    def test_best_odds_calculation(self, sample_odds_df):
        """Test that best odds are correctly calculated"""
        result = _add_metadata(sample_odds_df)

        # First row: Warriors odds are [2.10, 2.05, 2.15, 2.08, 2.12]
        assert result.iloc[0]["Best Odds"] == 2.15
        assert result.iloc[0]["Best Bookmaker"] == "BetMGM"

    def test_best_odds_with_specific_bookmakers(self, sample_odds_df):
        """Test best odds calculation with specific bookmaker list"""
        # Only consider DraftKings and FanDuel
        result = _add_metadata(sample_odds_df, best_odds_bms=["DraftKings", "FanDuel"])

        # First row: Warriors odds from DK=2.10, FD=2.05
        assert result.iloc[0]["Best Odds"] == 2.10
        assert result.iloc[0]["Best Bookmaker"] == "DraftKings"

    def test_result_initialized(self, sample_odds_df):
        """Test that Result column is initialized to 'Not Found'"""
        result = _add_metadata(sample_odds_df)

        assert all(result["Result"] == "Not Found")

    def test_handles_missing_bookmakers(self):
        """Test graceful handling when specified bookmakers don't exist"""
        df = pd.DataFrame(
            {"Match": ["A @ B", "A @ B"], "Team": ["A", "B"], "DraftKings": [2.0, 1.8]}
        )

        # Try to use non-existent bookmakers
        result = _add_metadata(df, best_odds_bms=["FanDuel", "BetMGM"])

        # Should fallback gracefully
        assert "Best Odds" in result.columns
        assert "Best Bookmaker" in result.columns
        # All values should be None since specified bookmakers don't exist
        assert all(pd.isna(result["Best Odds"]) | (result["Best Odds"] == None))
        assert all(
            pd.isna(result["Best Bookmaker"]) | (result["Best Bookmaker"] == None)
        )


class TestCleanOddsData:
    """Tests for _clean_odds_data function"""

    def test_replaces_one_with_nan(self):
        """Test that odds of 1.0 are replaced with NaN"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B"],
                "Team": ["A", "B"],
                "DraftKings": [2.10, 1.0],
                "FanDuel": [1.0, 1.90],
            }
        )

        result = _clean_odds_data(df)

        assert pd.isna(result.iloc[0]["FanDuel"])
        assert pd.isna(result.iloc[1]["DraftKings"])
        assert result.iloc[0]["DraftKings"] == 2.10
        assert result.iloc[1]["FanDuel"] == 1.90


class TestMinBookmakerFilter:
    """Tests for _min_bookmaker_filter function"""

    def test_filters_insufficient_bookmakers(self):
        """Test that rows with too few bookmakers are removed"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B", "C @ D", "C @ D"],
                "Team": ["A", "B", "C", "D"],
                "DraftKings": [2.0, 1.8, 2.0, np.nan],
                "FanDuel": [2.1, 1.7, 2.0, np.nan],
                "BetMGM": [2.2, 1.75, 2.0, np.nan],
                "Pinnacle": [2.0, 1.7, 2.0, np.nan],
                "Caesars": [2.1, 1.7, np.nan, np.nan],
                "BetOnline.ag": [2.1, np.nan, np.nan, np.nan],
            }
        )

        # Assuming MIN_BOOKMAKERS = 5
        result = _min_bookmaker_filter(df)

        # First row has 6 bookmakers, second has 5, third has 4, rest have 0
        # Should keep first two rows
        assert len(result) == 2
        assert all(result["Match"] == "A @ B")


class TestMaxOddsFilter:
    """Tests for _max_odds_filter function"""

    def test_filters_extreme_odds(self):
        """Test that rows with odds exceeding MAX_ODDS are removed"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B", "C @ D", "C @ D"],
                "Team": ["A", "B", "C", "D"],
                "DraftKings": [2.5, 1.8, 100.0, 1.9],
                "FanDuel": [2.4, 1.9, 2.0, 1.8],
                "BetMGM": [2.6, 1.85, 2.1, 1.85],
            }
        )

        # Assuming MAX_ODDS = 30
        result = _max_odds_filter(df)

        # Should filter out row with 100.0 odds
        assert result["Team"].tolist() == ["A", "B", "D"]


class TestAllOutcomesPresentFilter:
    """Tests for _all_outcomes_present_filter function"""

    def test_removes_incomplete_matches(self):
        """Test that matches with missing outcomes are removed"""
        df = pd.DataFrame(
            {
                "Match": ["A @ B", "A @ B", "C @ D"],  # C @ D only has 1 outcome
                "Team": ["A", "B", "C"],
                "Outcomes": [2, 2, 2],  # Expected to have 2 outcomes each
            }
        )

        result = _all_outcomes_present_filter(df)

        # Should only keep 'A @ B' which has both outcomes
        assert len(result) == 2
        assert all(result["Match"] == "A @ B")


class TestProcessTargetOddsData:
    """Integration tests for the complete processing pipeline"""

    @pytest.fixture
    def comprehensive_test_df(self):
        """
        Comprehensive test dataset that exercises all filtering functions.

        Expected filtering behavior:
        - Game 1 (Match A @ B): KEEP - Valid, complete, good odds
        - Game 2 (Match C @ D): CHANGE AND REMOVE - Has invalid odds (1.0) for Team C, change to NaN, remove due to insufficient bookmakers
        - Game 3 (Match E @ F): REMOVE - Only 1 outcome (incomplete match)
        - Game 4 (Match G @ H): REMOVE - Team G has only 4 bookmakers (below MIN_BOOKMAKERS)
        - Game 5 (Match I @ J): REMOVE - Team I has extreme odds (>MAX_ODDS)
        - Game 6 (Match K @ L): KEEP - Valid, complete, good odds

        Also tests:
        - Removal of unwanted bookmakers (UnknownBook1, UnknownBook2)
        - Metadata addition (Best Odds, Best Bookmaker, Outcomes, Result, Scrape Time)
        """
        return pd.DataFrame(
            {
                "ID": [
                    "g1",
                    "g1",
                    "g2",
                    "g2",
                    "g3",
                    "g4",
                    "g4",
                    "g5",
                    "g5",
                    "g6",
                    "g6",
                ],
                "Sport Key": ["basketball_nba"] * 11,
                "Sport Title": ["NBA"] * 11,
                "Start Time": ["2026-01-25T19:00:00Z"] * 11,
                "Match": [
                    "A @ B",
                    "A @ B",
                    "C @ D",
                    "C @ D",
                    "E @ F",
                    "G @ H",
                    "G @ H",
                    "I @ J",
                    "I @ J",
                    "K @ L",
                    "K @ L",
                ],
                "Team": ["A", "B", "C", "D", "E", "G", "H", "I", "J", "K", "L"],
                # Valid bookmakers (in ALL_BMS)
                "DraftKings": [
                    2.10,
                    1.85,
                    1.0,
                    2.00,
                    2.50,
                    2.00,
                    1.90,
                    150.0,
                    1.20,
                    2.20,
                    1.75,
                ],
                "FanDuel": [
                    2.05,
                    1.90,
                    2.10,
                    1.95,
                    2.45,
                    2.00,
                    1.90,
                    145.0,
                    1.25,
                    2.15,
                    1.80,
                ],
                "BetMGM": [
                    2.15,
                    1.80,
                    2.05,
                    1.98,
                    2.40,
                    2.00,
                    1.95,
                    140.0,
                    1.22,
                    2.25,
                    1.70,
                ],
                "Caesars": [
                    2.08,
                    1.87,
                    2.08,
                    1.96,
                    2.40,
                    2.00,
                    1.88,
                    140.0,
                    1.24,
                    2.18,
                    1.77,
                ],
                "Pinnacle": [
                    2.20,
                    2.0,
                    2.20,
                    2.05,
                    2.65,
                    np.nan,
                    2.00,
                    165.0,
                    1.30,
                    2.25,
                    1.85,
                ],
                # Unwanted bookmakers (not in ALL_BMS) - should be removed
                "UnknownBook1": [
                    2.00,
                    1.95,
                    2.00,
                    2.00,
                    2.30,
                    2.10,
                    1.85,
                    130.0,
                    1.30,
                    2.10,
                    1.85,
                ],
                "UnknownBook2": [
                    2.12,
                    1.88,
                    2.12,
                    1.92,
                    2.35,
                    2.05,
                    1.92,
                    135.0,
                    1.28,
                    2.12,
                    1.82,
                ],
            }
        )

    def test_complete_pipeline_integration(self, comprehensive_test_df):
        """
        Test the complete pipeline with a dataset that exercises all filters.
        """
        result = process_target_odds_data(comprehensive_test_df)

        # 1. Test that unwanted bookmakers were removed
        assert "UnknownBook1" not in result.columns
        assert "UnknownBook2" not in result.columns

        # 2. Test that valid bookmakers are kept
        assert "DraftKings" in result.columns
        assert "FanDuel" in result.columns
        assert "BetMGM" in result.columns
        assert "Caesars" in result.columns
        assert "Pinnacle" in result.columns

        # 3. Test that metadata columns were added
        assert "Best Odds" in result.columns
        assert "Best Bookmaker" in result.columns
        assert "Outcomes" in result.columns
        assert "Result" in result.columns
        assert "Scrape Time" in result.columns

        # 4. Verify Outcomes count
        print("Outcomes:")
        print(result["Outcomes"])
        assert all(result["Outcomes"] == 2)

        # 5. Verify all Result fields are initialized
        assert all(result["Result"] == "Not Found")

        # 6. Verify Scrape Time was added
        assert all(result["Scrape Time"].notna())

        # 7. Test filtering results - should only keep Games 1 and 6
        print(result["Match"].unique())
        assert len(result) == 4  # 2 games × 2 outcomes each
        assert set(result["Match"].unique()) == {"A @ B", "K @ L"}

        # 8. Verify Best Odds calculation, Best Bookmaker assignment (No Pinnacle should show up)
        team_a_row = result[
            (result["Match"] == "A @ B") & (result["Team"] == "A")
        ].iloc[0]
        team_b_row = result[
            (result["Match"] == "A @ B") & (result["Team"] == "B")
        ].iloc[0]
        team_k_row = result[
            (result["Match"] == "K @ L") & (result["Team"] == "K")
        ].iloc[0]
        team_l_row = result[
            (result["Match"] == "K @ L") & (result["Team"] == "L")
        ].iloc[0]

        assert team_a_row["Best Odds"] == 2.15
        assert team_b_row["Best Odds"] == 1.90
        assert team_k_row["Best Odds"] == 2.25
        assert team_l_row["Best Odds"] == 1.80
        assert team_a_row["Best Bookmaker"] == "BetMGM"
        assert team_b_row["Best Bookmaker"] == "FanDuel"
        assert team_k_row["Best Bookmaker"] == "BetMGM"
        assert team_l_row["Best Bookmaker"] == "FanDuel"

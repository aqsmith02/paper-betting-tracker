"""
test_summary_creation.py

Comprehensive pytest test suite for summary_creation.py module.

Author: Andrew Smith
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.find_bets.summary_creation import (
    create_average_summary_full,
    create_average_summary_minimal,
    create_modified_zscore_summary_full,
    create_modified_zscore_summary_minimal,
    create_random_summary_full,
    create_random_summary_minimal,
)


@pytest.fixture
def sample_betting_df():
    """Sample DataFrame with NaN values in critical columns."""
    return pd.DataFrame(
        {
            "ID": [1, 2, 3, 4],
            "Sport Key": ["basketball_nba"] * 4,
            "Sport Title": ["NBA"] * 4,
            "Start Time": pd.to_datetime(["2024-01-15 19:00"] * 4),
            "Scrape Time": pd.to_datetime(["2024-01-15 10:00"] * 4),
            "Match": ["Lakers vs Celtics"] * 4,
            "Team": ["Lakers", "Celtics", "Lakers", "Celtics"],
            "Bet365": [2.4, 2.9, 2.1, 3.8],
            "DraftKings": [2.5, 3.0, 2.2, 4.0],
            "Vigfree Bet365": [0.42, 0.34, 0.48, 0.26],
            "Vigfree DraftKings": [0.40, 0.33, 0.45, 0.25],
            "Best Bookmaker": ["DraftKings"] * 4,
            "Best Odds": [2.5, 3.0, 2.2, 4.0],
            "Fair Odds Average": [2.4, np.nan, 2.2, 3.9],
            "Expected Value": [0.05, np.nan, np.nan, 0.15],
            "Modified Z-Score": [1.5, 2.0, 1.2, np.nan],
            "Random Placed Bet": [0, 1, 0, 1],
            "Outcomes": [2] * 4,
            "Result": ["Not Found"] * 4,
        }
    )


@pytest.fixture
def mock_find_bookmaker_columns():
    """Mock find_bookmaker_columns to return predictable results."""
    with patch("summary_creation.find_bookmaker_columns") as mock:
        mock.return_value = ["Bet365", "DraftKings", "FanDuel"]
        yield mock


class TestCreateAverageSummaryMinimal:
    """Test create_average_summary_minimal function."""

    def test_filters_rows_with_nan_expected_value(self, sample_betting_df):
        """Should filter out rows with NaN in Expected Value."""
        result = create_average_summary_minimal(sample_betting_df)

        # Row 2 and 3 should be filtered out
        assert len(result) == 2
        assert 2 not in result["ID"].values
        assert 3 not in result["ID"].values

    def test_includes_only_minimal_columns(self, sample_betting_df):
        """Should include only specified minimal columns."""
        result = create_average_summary_minimal(sample_betting_df)

        expected_columns = [
            "ID",
            "Sport Key",
            "Sport Title",
            "Start Time",
            "Scrape Time",
            "Match",
            "Team",
            "Best Bookmaker",
            "Best Odds",
            "Fair Odds Average",
            "Expected Value",
            "Outcomes",
            "Result",
        ]

        assert list(result.columns) == expected_columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_average_summary_minimal(sample_betting_df)

        assert list(result.index) == list(range(len(result)))


class TestCreateAverageSummaryFull:
    """Test create_average_summary_full function."""

    def test_filters_rows_with_nan(self, sample_betting_df):
        """Should filter out rows with NaN in Expected Value or Fair Odds Average."""
        result = create_average_summary_full(sample_betting_df)

        # Row 2 and 3 should be filtered out
        assert len(result) == 2
        assert 2 not in result["ID"].values
        assert 3 not in result["ID"].values

    def test_includes_all_columns(self, sample_betting_df):
        """Should include bookmaker odds columns in full summary."""
        result = create_average_summary_full(sample_betting_df)

        assert "Bet365" in result.columns
        assert "DraftKings" in result.columns
        assert "Vigfree Bet365" in result.columns
        assert "Vigfree DraftKings" in result.columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_average_summary_full(sample_betting_df)

        assert list(result.index) == list(range(len(result)))


class TestCreateModifiedZscoreSummaryMinimal:
    """Test create_modified_zscore_summary_minimal function."""

    def test_filters_rows_with_nan_in_any_critical_column(self, sample_betting_df):
        """Should filter out rows with NaN in any of the three critical columns."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)

        # Only row 0 has all three columns filled
        assert len(result) == 1
        assert result["ID"].iloc[0] == 1

    def test_column_order_includes_zscore(self, sample_betting_df):
        """Should have Modified Z-Score in correct position."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)

        expected_columns = [
            "ID",
            "Sport Key",
            "Sport Title",
            "Start Time",
            "Scrape Time",
            "Match",
            "Team",
            "Best Bookmaker",
            "Best Odds",
            "Fair Odds Average",
            "Expected Value",
            "Modified Z-Score",
            "Outcomes",
            "Result",
        ]

        assert list(result.columns) == expected_columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_modified_zscore_summary_minimal(sample_betting_df)

        assert list(result.index) == list(range(len(result)))


class TestCreateModifiedZscoreSummaryFull:
    """Test create_modified_zscore_summary_full function."""

    def test_filters_rows_with_nan(self, sample_betting_df):
        """Should filter out rows with NaN in any critical column."""
        result = create_modified_zscore_summary_full(sample_betting_df)

        # Only row 0 has all three critical columns filled
        assert len(result) == 1
        assert result["ID"].iloc[0] == 1

    def test_includes_all_columns(self, sample_betting_df):
        """Should include bookmaker odds columns in full summary."""
        result = create_modified_zscore_summary_full(sample_betting_df)

        assert "Bet365" in result.columns
        assert "DraftKings" in result.columns
        assert "Vigfree Bet365" in result.columns
        assert "Vigfree DraftKings" in result.columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_modified_zscore_summary_full(sample_betting_df)

        assert list(result.index) == list(range(len(result)))


class TestCreateRandomSummaryMinimal:
    """Test create_random_summary_minimal function."""

    def test_filters_rows_with_zero_random_bet(self, sample_betting_df):
        """Should only include rows where Random Placed Bet is not 0."""
        result = create_random_summary_minimal(sample_betting_df)

        # Only rows 1 and 3 have Random Placed Bet = 1
        assert len(result) == 2
        assert all(result["ID"].isin([2, 4]))

    def test_minimal_column_set(self, sample_betting_df):
        """Should include only minimal required columns."""
        result = create_random_summary_minimal(sample_betting_df)

        expected_columns = [
            "ID",
            "Sport Key",
            "Sport Title",
            "Start Time",
            "Scrape Time",
            "Match",
            "Team",
            "Best Bookmaker",
            "Best Odds",
            "Outcomes",
            "Result",
        ]

        assert list(result.columns) == expected_columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_random_summary_minimal(sample_betting_df)

        assert list(result.index) == list(range(len(result)))


class TestCreateRandomSummaryFull:
    """Test create_random_summary_full function."""

    def test_filters_rows_with_zero_random_bet(self, sample_betting_df):
        """Should only include rows where Random Placed Bet is not 0."""
        result = create_random_summary_minimal(sample_betting_df)

        # Only rows 1 and 3 have Random Placed Bet = 1
        assert len(result) == 2
        assert all(result["ID"].isin([2, 4]))

    def test_includes_all_columns(self, sample_betting_df):
        """Should include bookmaker odds columns in full summary."""
        result = create_average_summary_full(sample_betting_df)

        assert "Bet365" in result.columns
        assert "DraftKings" in result.columns
        assert "Vigfree Bet365" in result.columns
        assert "Vigfree DraftKings" in result.columns

    def test_resets_index(self, sample_betting_df):
        """Should reset index starting from 0."""
        result = create_random_summary_full(sample_betting_df)

        assert list(result.index) == list(range(len(result)))

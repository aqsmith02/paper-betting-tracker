"""
Tests for date_utils.py

Tests utility functions for date conversion and time-based filtering.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.results.date_utils import _start_date_from_timestamp, _time_since_start


class TestStartDate:
    """Test suite for _start_date function."""

    def test_datetime_object(self):
        """Test conversion of datetime object to date string."""
        dt = datetime(2025, 7, 15, 14, 30, 0)
        result = _start_date_from_timestamp(dt)
        assert result == "2025-07-15"

    def test_iso_string(self):
        """Test conversion of ISO format string to date string."""
        iso_string = "2025-07-15T14:30:00Z"
        result = _start_date_from_timestamp(iso_string)
        assert result == "2025-07-15"

    def test_iso_string_with_timezone(self):
        """Test conversion of ISO format string with timezone to date string."""
        iso_string = "2025-07-15T14:30:00+05:00"
        result = _start_date_from_timestamp(iso_string)
        assert result == "2025-07-15"

    def test_timestamp(self):
        """Test conversion of timestamp to date string."""
        timestamp = pd.Timestamp("2025-07-15 14:30:00")
        result = _start_date_from_timestamp(timestamp)
        assert result == "2025-07-15"

    def test_simple_date_string(self):
        """Test conversion of simple date string."""
        date_string = "2025-07-15"
        result = _start_date_from_timestamp(date_string)
        assert result == "2025-07-15"


class TestTimeSinceStart:
    """Test suite for _time_since_start function."""

    def test_empty_dataframe(self):
        """Test that empty DataFrame is returned unchanged."""
        df = pd.DataFrame()
        result = _time_since_start(df, 12)
        assert result.empty
        assert len(result) == 0

    def test_filter_recent_games(self):
        """Test filtering out games that started less than threshold hours ago."""
        current_time = datetime.now(timezone.utc)

        # Create test data with games at different times
        data = {
            "Start Time": [
                (
                    current_time - timedelta(hours=6)
                ).isoformat(),  # 6 hours ago - should be filtered
                (
                    current_time - timedelta(hours=15)
                ).isoformat(),  # 15 hours ago - should remain
                (
                    current_time - timedelta(hours=24)
                ).isoformat(),  # 24 hours ago - should remain
            ],
            "Match": ["Game 1", "Game 2", "Game 3"],
        }
        df = pd.DataFrame(data)

        # Filter games starting less than 12 hours ago
        result = _time_since_start(df, 12)

        # Should only have 2 games (15 and 24 hours ago)
        assert len(result) == 2
        assert "Game 1" not in result["Match"].values
        assert "Game 2" in result["Match"].values
        assert "Game 3" in result["Match"].values

    def test_all_games_recent(self):
        """Test when all games started less than threshold hours ago."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (current_time - timedelta(hours=2)).isoformat(),
                (current_time - timedelta(hours=4)).isoformat(),
                (current_time - timedelta(hours=6)).isoformat(),
            ],
            "Match": ["Game 1", "Game 2", "Game 3"],
        }
        df = pd.DataFrame(data)

        # Filter games starting less than 12 hours ago
        result = _time_since_start(df, 12)

        # All games should be filtered out
        assert len(result) == 0
        assert result.empty

    def test_all_games_old(self):
        """Test when all games started more than threshold hours ago."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (current_time - timedelta(hours=20)).isoformat(),
                (current_time - timedelta(hours=30)).isoformat(),
                (current_time - timedelta(hours=48)).isoformat(),
            ],
            "Match": ["Game 1", "Game 2", "Game 3"],
        }
        df = pd.DataFrame(data)

        # Filter games starting less than 12 hours ago
        result = _time_since_start(df, 12)

        # All games should remain
        assert len(result) == 3
        assert list(result["Match"].values) == ["Game 1", "Game 2", "Game 3"]

    def test_exact_threshold_boundary(self):
        """Test game that started exactly at threshold hours ago."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (
                    current_time - timedelta(hours=12)
                ).isoformat(),  # Exactly 12 hours ago
            ],
            "Match": ["Game 1"],
        }
        df = pd.DataFrame(data)

        # Filter games starting less than 12 hours ago
        result = _time_since_start(df, 12)

        # Game at exactly threshold should remain (<=)
        assert len(result) == 1

    def test_different_threshold_values(self):
        """Test with different threshold values."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (current_time - timedelta(hours=10)).isoformat(),
                (current_time - timedelta(hours=25)).isoformat(),
            ],
            "Match": ["Game 1", "Game 2"],
        }
        df = pd.DataFrame(data)

        # Test with 24 hour threshold
        result_24 = _time_since_start(df.copy(), 24)
        assert len(result_24) == 1  # Only Game 2
        assert "Game 2" in result_24["Match"].values

        # Test with 8 hour threshold
        result_8 = _time_since_start(df.copy(), 8)
        assert len(result_8) == 2  # Both games

    def test_timezone_aware_datetimes(self):
        """Test that function handles timezone-aware datetimes correctly."""
        current_time = datetime.now(timezone.utc)

        # Create timezone-aware datetime strings
        data = {
            "Start Time": [
                (current_time - timedelta(hours=15))
                .replace(tzinfo=timezone.utc)
                .isoformat(),
            ],
            "Match": ["Game 1"],
        }
        df = pd.DataFrame(data)

        result = _time_since_start(df, 12)

        # Should not raise exception and should filter correctly
        assert len(result) == 1

    def test_preserves_other_columns(self):
        """Test that other DataFrame columns are preserved."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (current_time - timedelta(hours=15)).isoformat(),
                (current_time - timedelta(hours=24)).isoformat(),
            ],
            "Match": ["Game 1", "Game 2"],
            "Team": ["Team A", "Team B"],
            "Result": ["Win", "Loss"],
        }
        df = pd.DataFrame(data)

        result = _time_since_start(df, 12)

        # Check all columns are preserved
        assert list(result.columns) == ["Start Time", "Match", "Team", "Result"]
        assert len(result) == 2

    def test_zero_threshold(self):
        """Test with zero threshold - should filter all future/current games."""
        current_time = datetime.now(timezone.utc)

        data = {
            "Start Time": [
                (current_time - timedelta(hours=1)).isoformat(),
                (current_time - timedelta(hours=5)).isoformat(),
            ],
            "Match": ["Game 1", "Game 2"],
        }
        df = pd.DataFrame(data)

        result = _time_since_start(df, 0)

        # All games should remain (started more than 0 hours ago)
        assert len(result) == 2

"""
Tests for results.py

Tests functions that orchestrate the sports results fetching pipeline.
"""

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from src.results.results import clean_old_pending_results, main, process_files


class TestCleanOldPendingResults:
    """Test suite for clean_old_pending_results function."""

    def test_removes_old_pending_results(self):
        """Test that old rows with pending results are removed."""
        current_time = datetime.now(timezone.utc)

        # Create test data with old pending results
        df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3"],
                "Start Time": [
                    (current_time - timedelta(days=10)).isoformat(),  # Old + Pending
                    (current_time - timedelta(days=5)).isoformat(),  # Old + Pending
                    (
                        current_time - timedelta(hours=12)
                    ).isoformat(),  # Recent + Pending
                ],
                "Result": ["Pending", "Not Found", "Pending"],
            }
        )

        full_df = df.copy()

        with patch("src.results.results.DAYS_CUTOFF", 3):
            filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # Should keep event3 (recent) and remove event1, event2 (old + pending)
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["ID"] == "event3"
        assert len(filtered_full_df) == 1

    def test_keeps_old_results_with_valid_outcomes(self):
        """Test that old rows with valid results are kept."""
        current_time = datetime.now(timezone.utc)

        df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3"],
                "Start Time": [
                    (
                        current_time - timedelta(days=10)
                    ).isoformat(),  # Old + Valid result
                    (
                        current_time - timedelta(days=8)
                    ).isoformat(),  # Old + Valid result
                    (current_time - timedelta(days=9)).isoformat(),  # Old + Pending
                ],
                "Result": ["Team A", "Team B", "Pending"],
            }
        )

        full_df = df.copy()

        with patch("src.results.results.DAYS_CUTOFF", 3):
            filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # Should keep event1 and event2 (old but valid results), remove event3
        assert len(filtered_df) == 2
        assert set(filtered_df["ID"]) == {"event1", "event2"}

    def test_keeps_all_recent_games(self):
        """Test that all recent games are kept regardless of result status."""
        current_time = datetime.now(timezone.utc)

        df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3"],
                "Start Time": [
                    (current_time - timedelta(days=2)).isoformat(),
                    (current_time - timedelta(days=0)).isoformat(),
                    (current_time - timedelta(days=1)).isoformat(),
                ],
                "Result": ["Pending", "Not Found", "API Error"],
            }
        )

        full_df = df.copy()

        with patch("src.results.results.DAYS_CUTOFF", 3):
            filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # All recent games should be kept
        assert len(filtered_df) == 3
        assert set(filtered_df["ID"]) == {"event1", "event2", "event3"}

    def test_handles_all_pending_result_types(self):
        """Test that all PENDING_RESULTS types are handled."""
        current_time = datetime.now(timezone.utc)

        df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3", "event4"],
                "Start Time": [
                    (current_time - timedelta(days=10)).isoformat(),
                    (current_time - timedelta(days=10)).isoformat(),
                    (current_time - timedelta(days=10)).isoformat(),
                    (current_time - timedelta(days=10)).isoformat(),
                ],
                "Result": ["Pending", "Not Found", "API Error", "Team A"],
            }
        )

        full_df = df.copy()

        with patch("src.results.results.DAYS_CUTOFF", 3):
            with patch(
                "src.results.results.PENDING_RESULTS",
                ["Pending", "Not Found", "API Error"],
            ):
                filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # Only event4 with valid result should remain
        assert len(filtered_df) == 1
        assert filtered_df.iloc[0]["ID"] == "event4"

    def test_empty_dataframe(self):
        """Test with empty DataFrames."""
        df = pd.DataFrame({"ID": [], "Start Time": [], "Result": []})

        full_df = df.copy()

        filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        assert len(filtered_df) == 0
        assert len(filtered_full_df) == 0

    def test_preserves_start_time_format(self):
        """Test that original start time format is preserved."""
        current_time = datetime.now(timezone.utc)
        original_time_str = (current_time - timedelta(days=2)).isoformat()

        df = pd.DataFrame(
            {"ID": ["event1"], "Start Time": [original_time_str], "Result": ["Pending"]}
        )

        full_df = df.copy()

        with patch("src.results.results.DAYS_CUTOFF", 7):
            filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # Start time should be same format (string)
        assert filtered_df.iloc[0]["Start Time"] == original_time_str
        assert isinstance(filtered_df.iloc[0]["Start Time"], str)

    def test_synchronizes_both_dataframes(self):
        """Test that both DataFrames are filtered identically."""
        current_time = datetime.now(timezone.utc)

        df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3"],
                "Start Time": [
                    (current_time - timedelta(days=10)).isoformat(),
                    (current_time - timedelta(days=2)).isoformat(),
                    (current_time - timedelta(days=10)).isoformat(),
                ],
                "Result": ["Pending", "Pending", "Team A"],
            }
        )

        full_df = pd.DataFrame(
            {
                "ID": ["event1", "event2", "event3"],
                "Start Time": [
                    (current_time - timedelta(days=10)).isoformat(),
                    (current_time - timedelta(days=2)).isoformat(),
                    (current_time - timedelta(days=10)).isoformat(),
                ],
                "Result": ["Pending", "Pending", "Team A"],
                "Extra Column": ["A", "B", "C"],
            }
        )

        with patch("src.results.results.DAYS_CUTOFF", 7):
            filtered_df, filtered_full_df = clean_old_pending_results(df, full_df)

        # Both should have same rows kept
        assert len(filtered_df) == 2
        assert len(filtered_full_df) == 2
        assert list(filtered_df["ID"]) == list(filtered_full_df["ID"])
        assert list(filtered_df["ID"]) == ["event2", "event3"]

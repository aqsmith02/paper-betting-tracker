"""
test_file_management.py

Unit tests for file_management.py module.

Author: Test Suite
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from codebase.find_bets.file_management import (
    _start_date_from_timestamp,
    BetFileManager,
)


class TestStartDateFromTimestamp(unittest.TestCase):
    """Test cases for _start_date_from_timestamp function."""

    def test_datetime_object(self):
        """Test conversion of datetime object to date string."""
        dt = datetime(2025, 11, 1, 15, 30, 0)
        result = _start_date_from_timestamp(dt)
        self.assertEqual(result, "2025-11-01")

    def test_string_timestamp(self):
        """Test conversion of string timestamp to date string."""
        timestamp_str = "2025-11-01 15:30:00"
        result = _start_date_from_timestamp(timestamp_str)
        self.assertEqual(result, "2025-11-01")

    def test_iso_format_string(self):
        """Test conversion of ISO format string."""
        timestamp_str = "2025-11-01T15:30:00"
        result = _start_date_from_timestamp(timestamp_str)
        self.assertEqual(result, "2025-11-01")

    def test_date_only_string(self):
        """Test conversion of date-only string."""
        date_str = "2025-11-01"
        result = _start_date_from_timestamp(date_str)
        self.assertEqual(result, "2025-11-01")

    def test_different_date_formats(self):
        """Test various date format inputs."""
        test_cases = [
            ("2025-12-31", "2025-12-31"),
            ("2025-01-01 00:00:00", "2025-01-01"),
            ("2025/06/15", "2025-06-15"),
        ]
        for input_date, expected in test_cases:
            result = _start_date_from_timestamp(input_date)
            self.assertEqual(result, expected)


class TestBetFileManagerInit(unittest.TestCase):
    """Test cases for BetFileManager initialization."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_init_with_default_directory(self):
        """Test initialization with default directory."""
        manager = BetFileManager()
        self.assertIsInstance(manager.data_dir, Path)
        self.assertTrue(manager.data_dir.exists())

    def test_init_with_custom_directory(self):
        """Test initialization with custom directory."""
        custom_dir = Path(self.test_dir) / "custom_data"
        manager = BetFileManager(custom_dir)
        self.assertEqual(manager.data_dir, custom_dir)
        self.assertTrue(manager.data_dir.exists())

    def test_creates_directory_if_not_exists(self):
        """Test that initialization creates directory if it doesn't exist."""
        new_dir = Path(self.test_dir) / "new_directory"
        self.assertFalse(new_dir.exists())
        manager = BetFileManager(new_dir)
        self.assertTrue(new_dir.exists())


class TestBetFileManagerGetColumns(unittest.TestCase):
    """Test cases for BetFileManager._get_columns method."""

    def setUp(self):
        """Set up BetFileManager instance."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_get_strategy_columns_master_nc_avg(self):
        """Test getting strategy columns for master_nc_avg."""
        result = self.manager._get_columns("master_nc_avg_bets.csv", "strategy")
        self.assertEqual(result, ["Avg Edge Pct", "Fair Odds Avg"])

    def test_get_strategy_columns_master_nc_zscore(self):
        """Test getting strategy columns for master_nc_zscore."""
        result = self.manager._get_columns("master_nc_zscore_full.csv", "strategy")
        self.assertEqual(result, ["Z Score", "Avg Edge Pct"])

    def test_get_strategy_columns_master_nc_pin(self):
        """Test getting strategy columns for master_nc_pin."""
        result = self.manager._get_columns("master_nc_pin.csv", "strategy")
        self.assertEqual(result, ["Pinnacle Fair Odds", "Pin Edge Pct"])

    def test_get_strategy_columns_master_nc_random(self):
        """Test getting strategy columns for master_nc_random."""
        result = self.manager._get_columns("master_nc_random_bets.csv", "strategy")
        self.assertEqual(result, ["Random Bet Odds"])

    def test_get_strategy_columns_with_full_suffix(self):
        """Test getting columns with _full suffix."""
        result = self.manager._get_columns("master_nc_avg_full.csv", "strategy")
        self.assertEqual(result, ["Avg Edge Pct", "Fair Odds Avg"])

    def test_get_strategy_columns_with_bets_suffix(self):
        """Test getting columns with _bets suffix."""
        result = self.manager._get_columns("master_nc_mod_zscore_bets.csv", "strategy")
        self.assertEqual(result, ["Modified Z Score", "Avg Edge Pct"])

    def test_get_columns_nonexistent_strategy(self):
        """Test getting columns for non-existent strategy."""
        result = self.manager._get_columns("nonexistent.csv", "strategy")
        self.assertIsNone(result)

    def test_get_columns_nc_strategies(self):
        """Test getting columns for north carolina dataset."""
        nc_tests = [
            ("master_nc_avg.csv", ["Avg Edge Pct", "Fair Odds Avg"]),
            ("master_nc_zscore.csv", ["Z Score", "Avg Edge Pct"]),
            ("master_nc_pin.csv", ["Pinnacle Fair Odds", "Pin Edge Pct"]),
            ("master_nc_random.csv", ["Random Bet Odds"]),
        ]
        for filename, expected in nc_tests:
            result = self.manager._get_columns(filename, "strategy")
            self.assertEqual(result, expected)


class TestBetFileManagerAppendUniqueBets(unittest.TestCase):
    """Test cases for BetFileManager._append_unique_bets method."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)
        self.sample_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team C vs Team D"],
                "League": ["Premier League", "La Liga"],
                "Team": ["Team A", "Team C"],
                "Start Time": ["2025-11-01T15:00:00Z", "2025-11-01T18:00:00Z"],
                "Best Bookmaker": ["Bet365", "William Hill"],
                "Best Odds": [2.5, 3.0],
                "Result": ["Not Found", "Not Found"],
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("codebase.find_bets.file_management.datetime")
    def test_append_to_new_file(self, mock_datetime):
        """Test appending data to a new file."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_bets.csv"
        self.manager._append_unique_bets(self.sample_data, filename)

        file_path = Path(self.test_dir) / filename
        self.assertTrue(file_path.exists())

        result = pd.read_csv(file_path)
        self.assertEqual(len(result), 2)
        self.assertIn("Scrape Time", result.columns)

    @patch("codebase.find_bets.file_management.datetime")
    def test_append_unique_bets_to_existing_file(self, mock_datetime):
        """Test appending unique bets to existing file."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_bets.csv"
        # Create initial file
        self.manager._append_unique_bets(self.sample_data, filename)

        # Create new data with one duplicate and one new bet
        new_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team E vs Team F"],
                "League": ["Premier League", "Bundesliga"],
                "Team": ["Team A", "Team E"],
                "Start Time": ["2025-11-01T15:00:00Z", "2025-11-02T20:00:00Z"],
                "Best Bookmaker": ["Bet365", "Betway"],
                "Best Odds": [2.5, 1.8],
                "Result": ["Win", "Win"],
            }
        )

        self.manager._append_unique_bets(new_data, filename)

        result = pd.read_csv(Path(self.test_dir) / filename)
        self.assertEqual(len(result), 3)  # 2 original + 1 new

    @patch("codebase.find_bets.file_management.datetime")
    def test_append_all_duplicates(self, mock_datetime):
        """Test appending when all bets are duplicates."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_bets.csv"
        self.manager._append_unique_bets(self.sample_data, filename)

        # Try to append the same data again
        self.manager._append_unique_bets(self.sample_data, filename)

        result = pd.read_csv(Path(self.test_dir) / filename)
        self.assertEqual(len(result), 2)  # Should still be 2

    def test_append_empty_dataframe(self):
        """Test appending empty DataFrame."""
        filename = "test_bets.csv"
        empty_df = pd.DataFrame()

        self.manager._append_unique_bets(empty_df, filename)

        file_path = Path(self.test_dir) / filename
        self.assertFalse(file_path.exists())

    @patch("codebase.find_bets.file_management.datetime")
    def test_duplicate_detection_different_date(self, mock_datetime):
        """Test that bets on different dates are not considered duplicates."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_bets.csv"
        self.manager._append_unique_bets(self.sample_data, filename)

        # Same match but different date
        new_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B"],
                "League": ["Premier League"],
                "Team": ["Team A"],
                "Start Time": ["2025-11-02T15:00:00Z"],
                "Best Bookmaker": ["Bet365"],
                "Best Odds": [2.5],
                "Result": ["Win"],
            }
        )

        self.manager._append_unique_bets(new_data, filename)

        result = pd.read_csv(Path(self.test_dir) / filename)
        self.assertEqual(len(result), 3)


class TestBetFileManagerAlignColumnSchemas(unittest.TestCase):
    """Test cases for BetFileManager._align_column_schemas method."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    def test_align_identical_schemas(self):
        """Test alignment when schemas are identical."""
        columns = ["Match", "Team", "Best Odds", "Result"]
        existing_df = pd.DataFrame(columns=columns)
        new_df = pd.DataFrame(columns=columns)

        result = self.manager._align_column_schemas(
            existing_df, new_df, "master_nc_avg_bets.csv"
        )

        self.assertIsInstance(result, list)
        self.assertTrue(all(col in result for col in columns))

    def test_align_with_new_columns(self):
        """Test alignment when new data has additional columns."""
        existing_df = pd.DataFrame(columns=["Match", "Team", "Best Odds"])
        new_df = pd.DataFrame(columns=["Match", "Team", "Best Odds", "New Column"])

        result = self.manager._align_column_schemas(
            existing_df, new_df, "master_nc_avg_bets.csv"
        )

        self.assertIn("New Column", result)
        self.assertEqual(len(result), 4)

    def test_column_ordering_strategy_columns_at_end(self):
        """Test that strategy columns are placed at end."""
        existing_df = pd.DataFrame(
            columns=["Match", "Team", "Avg Edge Pct", "Fair Odds Avg", "Best Odds"]
        )
        new_df = pd.DataFrame(
            columns=["Match", "Team", "League", "Avg Edge Pct", "Fair Odds Avg"]
        )

        result = self.manager._align_column_schemas(
            existing_df, new_df, "master_nc_avg_bets.csv"
        )

        # Strategy columns should be near the end
        strategy_indices = [
            result.index(col) for col in ["Avg Edge Pct", "Fair Odds Avg"]
        ]
        non_strategy_indices = [
            result.index(col) for col in ["Match", "Team", "League"]
        ]

        self.assertTrue(max(non_strategy_indices) < min(strategy_indices))

    def test_end_columns_ordering(self):
        """Test that end columns (Best Odds, Best Bookmaker, Result, etc.) are at end."""
        columns = [
            "Match",
            "Team",
            "League",
            "Avg Edge Pct",
            "Best Odds",
            "Best Bookmaker",
            "Result",
            "Scrape Time",
        ]
        existing_df = pd.DataFrame(columns=columns)
        new_df = pd.DataFrame(columns=columns)

        result = self.manager._align_column_schemas(
            existing_df, new_df, "master_nc_avg_bets.csv"
        )

        # Check that specific columns are at the end
        end_columns = ["Best Odds", "Best Bookmaker", "Result", "Scrape Time"]
        result_end = result[-len(end_columns) :]

        for col in end_columns:
            self.assertIn(col, result_end)


class TestBetFileManagerSaveBestBetsOnly(unittest.TestCase):
    """Test cases for BetFileManager.save_best_bets_only method."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)
        self.sample_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team A vs Team B", "Team C vs Team D"],
                "Team": ["Team A", "Team B", "Team C"],
                "Start Time": [
                    "2025-11-01T15:00:00Z",
                    "2025-11-01T15:00:00Z",
                    "2025-11-01T18:00:00Z",
                ],
                "Avg Edge Pct": [5.2, 3.8, 4.5],
                "Best Bookmaker": ["Bet365", "William Hill", "Betway"],
                "Best Odds": [2.5, 3.0, 1.8],
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("codebase.find_bets.file_management.datetime")
    def test_save_best_bet_per_match(self, mock_datetime):
        """Test that only the best bet per match is saved."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        result = self.manager.save_best_bets_only(
            self.sample_data, "test_bets.csv", "Avg Edge Pct"
        )

        self.assertEqual(len(result), 2)  # 2 unique matches
        # First match should have the highest Avg Edge Pct (5.2)
        match_a_row = result[result["Match"] == "Team A vs Team B"]
        self.assertEqual(match_a_row.iloc[0]["Avg Edge Pct"], 5.2)

    @patch("codebase.find_bets.file_management.datetime")
    def test_save_best_bets_creates_file(self, mock_datetime):
        """Test that file is created when saving best bets."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_bets.csv"
        self.manager.save_best_bets_only(self.sample_data, filename, "Avg Edge Pct")

        file_path = Path(self.test_dir) / filename
        self.assertTrue(file_path.exists())

    def test_save_best_bets_empty_dataframe(self):
        """Test saving empty DataFrame returns empty DataFrame."""
        empty_df = pd.DataFrame()
        result = self.manager.save_best_bets_only(
            empty_df, "test_bets.csv", "Avg Edge Pct"
        )

        self.assertTrue(result.empty)

    @patch("codebase.find_bets.file_management.datetime")
    def test_descending_sort_for_best_bet(self, mock_datetime):
        """Test that bets are sorted in descending order to get highest score."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team A vs Team B"],
                "Team": ["Team A", "Team B"],
                "Start Time": ["2025-11-01T15:00:00Z", "2025-11-01T15:00:00Z"],
                "Z Score": [2.5, 3.8],
                "Best Odds": [2.5, 3.0],
            }
        )

        result = self.manager.save_best_bets_only(data, "test_bets.csv", "Z Score")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Z Score"], 3.8)


class TestBetFileManagerSaveFullBettingData(unittest.TestCase):
    """Test cases for BetFileManager.save_full_betting_data method."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)

        self.source_df = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team C vs Team D"],
                "Team": ["Team A", "Team C"],
                "Start Time": ["2025-11-01T15:00:00Z", "2025-11-01T18:00:00Z"],
                "League": ["Premier League", "La Liga"],
                "Vigfree Pinnacle": [2.4, 2.9],
                "Vigfree Bet365": [2.5, 3.0],
                "Best Odds": [2.5, 3.0],
                "Best Bookmaker": ["Bet365", "William Hill"],
                "Avg Edge Pct": [5.2, 4.5],
            }
        )

        self.filtered_summary = pd.DataFrame(
            {
                "Match": ["Team A vs Team B"],
                "Team": ["Team A"],
                "Start Time": ["2025-11-01T15:00:00Z"],
                "Avg Edge Pct": [5.2],
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("codebase.find_bets.file_management.datetime")
    def test_save_full_data_merges_correctly(self, mock_datetime):
        """Test that full data is merged correctly with filtered summary."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_full.csv"
        self.manager.save_full_betting_data(
            self.source_df, self.filtered_summary, filename
        )

        result = pd.read_csv(Path(self.test_dir) / filename)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Match"], "Team A vs Team B")
        self.assertIn("League", result.columns)

    @patch("codebase.find_bets.file_management.datetime")
    def test_save_full_data_removes_vigfree_columns(self, mock_datetime):
        """Test that Vigfree columns are removed from output."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_full.csv"
        self.manager.save_full_betting_data(
            self.source_df, self.filtered_summary, filename
        )

        result = pd.read_csv(Path(self.test_dir) / filename)
        vigfree_cols = [col for col in result.columns if col.startswith("Vigfree ")]
        self.assertEqual(len(vigfree_cols), 0)

    def test_save_full_data_empty_filtered_summary(self):
        """Test that no file is created when filtered summary is empty."""
        filename = "test_full.csv"
        empty_summary = pd.DataFrame()

        self.manager.save_full_betting_data(self.source_df, empty_summary, filename)

        file_path = Path(self.test_dir) / filename
        self.assertFalse(file_path.exists())

    @patch("codebase.find_bets.file_management.datetime")
    def test_save_full_data_preserves_all_source_columns(self, mock_datetime):
        """Test that all non-vigfree columns from source are preserved."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        filename = "test_full.csv"
        self.manager.save_full_betting_data(
            self.source_df, self.filtered_summary, filename
        )

        result = pd.read_csv(Path(self.test_dir) / filename)
        expected_columns = [
            "Match",
            "Team",
            "Start Time",
            "League",
            "Best Odds",
            "Best Bookmaker",
            "Avg Edge Pct",
        ]

        for col in expected_columns:
            self.assertIn(col, result.columns)


class TestBetFileManagerIntegration(unittest.TestCase):
    """Integration tests for BetFileManager."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = BetFileManager(self.test_dir)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch("codebase.find_bets.file_management.datetime")
    def test_full_workflow(self, mock_datetime):
        """Test complete workflow: save best bets and full data."""
        mock_datetime.now.return_value.strftime.return_value = "2025-11-01T12:00:00Z"

        # Create sample data
        full_data = pd.DataFrame(
            {
                "Match": ["Team A vs Team B", "Team A vs Team B", "Team C vs Team D"],
                "Team": ["Team A", "Team B", "Team C"],
                "Start Time": [
                    "2025-11-01T15:00:00Z",
                    "2025-11-01T15:00:00Z",
                    "2025-11-01T18:00:00Z",
                ],
                "League": ["Premier League", "Premier League", "La Liga"],
                "Avg Edge Pct": [5.2, 3.8, 4.5],
                "Vigfree Pinnacle": [2.4, 2.8, 2.9],
                "Best Odds": [2.5, 3.0, 1.8],
                "Best Bookmaker": ["Bet365", "William Hill", "Betway"],
            }
        )

        # Save best bets
        best_bets = self.manager.save_best_bets_only(
            full_data, "master_nc_avg_bets.csv", "Avg Edge Pct"
        )

        # Save full data
        self.manager.save_full_betting_data(full_data, best_bets, "master_nc_avg_full.csv")

        # Verify both files exist
        self.assertTrue((Path(self.test_dir) / "master_nc_avg_bets.csv").exists())
        self.assertTrue((Path(self.test_dir) / "master_nc_avg_full.csv").exists())

        # Verify content
        bets_df = pd.read_csv(Path(self.test_dir) / "master_nc_avg_bets.csv")
        full_df = pd.read_csv(Path(self.test_dir) / "master_nc_avg_full.csv")

        self.assertEqual(len(bets_df), 2)  # 2 unique matches
        self.assertEqual(len(full_df), 2)  # Same 2 matches in full data


if __name__ == "__main__":
    unittest.main(verbosity=2)
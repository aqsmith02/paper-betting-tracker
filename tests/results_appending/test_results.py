"""
test_results.py

Unit tests for results.py module.

Author: Test Suite
"""

import unittest
from unittest.mock import patch
import pandas as pd
from datetime import datetime, timezone, timedelta
import tempfile
import shutil
from pathlib import Path
from src.results.results import (
    filter_rows_to_search,
    fetch_results_from_theodds,
    fetch_results_from_sportsdb,
    clean_old_pending_results,
    process_file_pair,
    main,
)


class TestFilterRowsToSearch(unittest.TestCase):
    """Test cases for filter_rows_to_search function."""

    def test_filter_pending_results(self):
        """Test filtering rows with pending results."""
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2", "Game 3"],
            "Result": ["Not Found", "Lakers", "Pending"],
        })

        result = filter_rows_to_search(df)
        
        self.assertEqual(len(result), 2)
        self.assertIn("Not Found", result["Result"].values)
        self.assertIn("Pending", result["Result"].values)

    def test_no_pending_results(self):
        """Test when no rows have pending results."""
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2"],
            "Result": ["Lakers", "Warriors"],
        })

        result = filter_rows_to_search(df)
        
        self.assertEqual(len(result), 0)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["Match", "Result"])
        result = filter_rows_to_search(df)
        self.assertEqual(len(result), 0)


class TestFetchResultsFromTheodds(unittest.TestCase):
    """Test cases for fetch_results_from_theodds function."""

    @patch('codebase.results_appending.results.get_finished_games_from_theodds')
    @patch('codebase.results_appending.results.map_league_to_key')
    def test_fetch_with_multiple_leagues(self, mock_map, mock_get_games):
        """Test fetching results for multiple leagues."""
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2"],
            "League": ["NFL", "MLB"],
            "Result": ["Not Found", "Pending"],
        })

        mock_map.return_value = ["americanfootball_nfl", "baseball_mlb"]
        mock_get_games.return_value = df

        result = fetch_results_from_theodds(df)
        
        # Should call get_finished_games_from_theodds twice (once per league)
        self.assertEqual(mock_get_games.call_count, 2)

    @patch('codebase.results_appending.results.get_finished_games_from_theodds')
    @patch('codebase.results_appending.results.map_league_to_key')
    def test_no_pending_results(self, mock_map, mock_get_games):
        """Test when no rows need checking."""
        df = pd.DataFrame({
            "Match": ["Game 1"],
            "League": ["NFL"],
            "Result": ["Patriots"],
        })

        result = fetch_results_from_theodds(df)
        
        # Should not call get_finished_games_from_theodds
        mock_get_games.assert_not_called()


class TestFetchResultsFromSportsdb(unittest.TestCase):
    """Test cases for fetch_results_from_sportsdb function."""

    @patch('codebase.results_appending.results.get_finished_games_from_thesportsdb')
    def test_fetch_results(self, mock_get_games):
        """Test fetching results from SportsDB."""
        df = pd.DataFrame({
            "Match": ["Game 1"],
            "Result": ["Not Found"],
        })

        mock_get_games.return_value = df

        result = fetch_results_from_sportsdb(df)
        
        mock_get_games.assert_called_once_with(df)


class TestCleanOldPendingResults(unittest.TestCase):
    """Test cases for clean_old_pending_results function."""

    @patch('codebase.results_appending.results.DAYS_CUTOFF', 7)
    def test_remove_old_pending_results(self):
        """Test removing old pending results."""
        current_time = datetime.now(timezone.utc)
        
        bet_df = pd.DataFrame({
            "Match": ["Old Game", "Recent Game", "Completed Game"],
            "Start Time": [
                (current_time - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "Result": ["Not Found", "Pending", "Lakers"],
        })

        full_df = bet_df.copy()

        result_bet, result_full = clean_old_pending_results(bet_df, full_df)
        
        # Old Game should be removed (old + pending)
        # Recent Game should remain (recent + pending)
        # Completed Game should remain (old but completed)
        self.assertEqual(len(result_bet), 2)
        self.assertIn("Recent Game", result_bet["Match"].values)
        self.assertIn("Completed Game", result_bet["Match"].values)

    @patch('codebase.results_appending.results.DAYS_CUTOFF', 7)
    def test_no_rows_removed(self):
        """Test when no rows should be removed."""
        current_time = datetime.now(timezone.utc)
        
        bet_df = pd.DataFrame({
            "Match": ["Game 1", "Game 2"],
            "Start Time": [
                (current_time - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "Result": ["Pending", "Lakers"],
        })

        full_df = bet_df.copy()

        result_bet, result_full = clean_old_pending_results(bet_df, full_df)
        
        self.assertEqual(len(result_bet), 2)

    @patch('codebase.results_appending.results.DAYS_CUTOFF', 7)
    def test_preserve_original_format(self):
        """Test that original timestamp format is preserved."""
        current_time = datetime.now(timezone.utc)
        
        original_timestamp = (current_time - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        bet_df = pd.DataFrame({
            "Match": ["Game 1"],
            "Start Time": [original_timestamp],
            "Result": ["Pending"],
        })

        full_df = bet_df.copy()

        result_bet, result_full = clean_old_pending_results(bet_df, full_df)
        
        # Verify format is preserved
        self.assertEqual(result_bet.iloc[0]["Start Time"], original_timestamp)


class TestProcessFilePair(unittest.TestCase):
    """Test cases for process_file_pair function."""

    def setUp(self):
        """Set up temporary directory for testing."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir)

    @patch('codebase.results_appending.results.DATA_DIR')
    @patch('codebase.results_appending.results.fetch_results_from_theodds')
    @patch('codebase.results_appending.results.fetch_results_from_sportsdb')
    @patch('codebase.results_appending.results.clean_old_pending_results')
    def test_process_file_pair(self, mock_clean, mock_sportsdb, mock_theodds, mock_data_dir):
        """Test processing a file pair."""
        mock_data_dir.__truediv__ = lambda self, x: self.test_dir / x
        
        # Create test files
        bet_df = pd.DataFrame({
            "Match": ["Game 1"],
            "Result": ["Not Found"],
        })
        full_df = pd.DataFrame({
            "Match": ["Game 1"],
            "Result": ["Not Found"],
        })

        bet_file = self.test_dir / "test_bets.csv"
        full_file = self.test_dir / "test_full.csv"
        
        bet_df.to_csv(bet_file, index=False)
        full_df.to_csv(full_file, index=False)

        # Mock returns
        mock_theodds.return_value = bet_df
        mock_sportsdb.return_value = bet_df
        mock_clean.return_value = (bet_df, full_df)

        with patch('codebase.results_appending.results.DATA_DIR', self.test_dir):
            process_file_pair("test_bets.csv", "test_full.csv")

        # Verify API functions were called
        mock_theodds.assert_called_once()
        mock_sportsdb.assert_called_once()
        mock_clean.assert_called_once()


class TestMain(unittest.TestCase):
    """Test cases for main function."""

    @patch('codebase.results_appending.results.process_file_pair')
    @patch('codebase.results_appending.results.FILE_CONFIGS', [
        ("bet1.csv", "full1.csv"),
        ("bet2.csv", "full2.csv"),
    ])
    @patch('codebase.results_appending.results.SLEEP_DURATION', 0)
    def test_main_success(self, mock_process):
        """Test successful execution of main pipeline."""
        main()

        # Should process both file pairs
        self.assertEqual(mock_process.call_count, 2)

    @patch('codebase.results_appending.results.process_file_pair')
    @patch('codebase.results_appending.results.FILE_CONFIGS', [
        ("bet1.csv", "full1.csv"),
        ("bet2.csv", "full2.csv"),
    ])
    @patch('codebase.results_appending.results.SLEEP_DURATION', 0)
    def test_main_handles_exceptions(self, mock_process):
        """Test that main continues after exceptions."""
        # First file fails, second succeeds
        mock_process.side_effect = [Exception("File error"), None]

        # Should not raise exception
        main()

        # Should have attempted both files
        self.assertEqual(mock_process.call_count, 2)

    @patch('codebase.results_appending.results.process_file_pair')
    @patch('codebase.results_appending.results.time.sleep')
    @patch('codebase.results_appending.results.FILE_CONFIGS', [
        ("bet1.csv", "full1.csv"),
        ("bet2.csv", "full2.csv"),
        ("bet3.csv", "full3.csv"),
    ])
    @patch('codebase.results_appending.results.SLEEP_DURATION', 5)
    def test_main_sleeps_between_files(self, mock_sleep, mock_process):
        """Test that main sleeps between file processing."""
        main()

        # Should sleep 2 times (between 3 files, not after the last one)
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_called_with(5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
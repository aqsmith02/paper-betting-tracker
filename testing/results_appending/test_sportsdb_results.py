"""
test_sportsdb_results.py

Unit tests for sportsdb_results.py module.

Author: Test Suite
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
from datetime import datetime, timezone, timedelta
from codebase.results_appending.sportsdb_results import (
    _start_date,
    _format_match_for_thesportsdb,
    _time_since_start,
    _get_results,
    get_finished_games_from_thesportsdb,
)


class TestStartDate(unittest.TestCase):
    """Test cases for _start_date function."""

    def test_iso_format_timestamp(self):
        """Test conversion of ISO format timestamp."""
        timestamp = "2025-11-04T19:00:00Z"
        result = _start_date(timestamp)
        self.assertEqual(result, "2025-11-04")

    def test_datetime_object(self):
        """Test conversion of datetime object."""
        dt = datetime(2025, 11, 4, 19, 0, 0)
        result = _start_date(dt)
        self.assertEqual(result, "2025-11-04")

    def test_string_date(self):
        """Test conversion of string date."""
        date_str = "2025-11-04"
        result = _start_date(date_str)
        self.assertEqual(result, "2025-11-04")


class TestFormatMatchForThesportsdb(unittest.TestCase):
    """Test cases for _format_match_for_thesportsdb function."""

    def test_format_at_symbol(self):
        """Test formatting match with @ symbol."""
        match = "Warriors @ Lakers"
        result = _format_match_for_thesportsdb(match)
        self.assertEqual(result, "Lakers_vs_Warriors")

    def test_format_vs(self):
        """Test formatting match with vs."""
        match = "Lakers vs Warriors"
        result = _format_match_for_thesportsdb(match)
        self.assertEqual(result, "Lakers_vs_Warriors")

    def test_format_plain(self):
        """Test formatting plain match string."""
        match = "Lakers Warriors"
        result = _format_match_for_thesportsdb(match)
        self.assertEqual(result, "Lakers_Warriors")

    def test_format_with_extra_spaces(self):
        """Test formatting with extra spaces."""
        match = "Golden State Warriors @ Los Angeles Lakers"
        result = _format_match_for_thesportsdb(match)
        self.assertEqual(result, "Los_Angeles_Lakers_vs_Golden_State_Warriors")


class TestTimeSinceStart(unittest.TestCase):
    """Test cases for _time_since_start function."""

    def test_filter_recent_games(self):
        """Test that games starting less than threshold hours ago are filtered out."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2", "Game 3"],
            "Start Time": [
                (current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(hours=10)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
        })

        result = _time_since_start(df, 12)
        
        # Only Game 1 (15 hours ago) should remain
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["Match"], "Game 1")

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["Match", "Start Time"])
        result = _time_since_start(df, 12)
        self.assertEqual(len(result), 0)


class TestGetResults(unittest.TestCase):
    """Test cases for _get_results function."""

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_home_team_wins(self, mock_get):
        """Test getting result when home team wins."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "event": [{
                "strHomeTeam": "Lakers",
                "strAwayTeam": "Warriors",
                "intHomeScore": "110",
                "intAwayScore": "105",
            }]
        }
        mock_get.return_value = mock_response

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "Lakers")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_away_team_wins(self, mock_get):
        """Test getting result when away team wins."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "event": [{
                "strHomeTeam": "Lakers",
                "strAwayTeam": "Warriors",
                "intHomeScore": "105",
                "intAwayScore": "110",
            }]
        }
        mock_get.return_value = mock_response

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "Warriors")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_draw(self, mock_get):
        """Test getting result when game is a draw."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "event": [{
                "strHomeTeam": "Team A",
                "strAwayTeam": "Team B",
                "intHomeScore": "2",
                "intAwayScore": "2",
            }]
        }
        mock_get.return_value = mock_response

        result = _get_results("Team_A_vs_Team_B", "2025-11-04")
        self.assertEqual(result, "Draw")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_pending_result(self, mock_get):
        """Test getting result when game is pending (no scores yet)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "event": [{
                "strHomeTeam": "Lakers",
                "strAwayTeam": "Warriors",
                "intHomeScore": None,
                "intAwayScore": None,
            }]
        }
        mock_get.return_value = mock_response

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "Pending")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_not_found(self, mock_get):
        """Test when event is not found."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"event": None}
        mock_get.return_value = mock_response

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "Not Found")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_api_error(self, mock_get):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "API Error")

    @patch('codebase.results_appending.sportsdb_results.requests.get')
    def test_exception_handling(self, mock_get):
        """Test exception handling."""
        mock_get.side_effect = Exception("Network error")

        result = _get_results("Lakers_vs_Warriors", "2025-11-04")
        self.assertEqual(result, "Error")


class TestGetFinishedGamesFromThesportsdb(unittest.TestCase):
    """Test cases for get_finished_games_from_thesportsdb function."""

    @patch('codebase.results_appending.sportsdb_results._get_results')
    @patch('codebase.results_appending.sportsdb_results._time_since_start')
    def test_update_results(self, mock_time_filter, mock_get_results):
        """Test updating results for games."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Warriors @ Lakers"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "Result": ["Not Found"],
        })

        mock_time_filter.return_value = df
        mock_get_results.return_value = "Lakers"

        result = get_finished_games_from_thesportsdb(df)
        
        self.assertEqual(result.iloc[0]["Result"], "Lakers")

    @patch('codebase.results_appending.sportsdb_results._get_results')
    @patch('codebase.results_appending.sportsdb_results._time_since_start')
    def test_skip_existing_results(self, mock_time_filter, mock_get_results):
        """Test that existing valid results are not overwritten."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Warriors @ Lakers"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "Result": ["Lakers"],
        })

        mock_time_filter.return_value = df
        mock_get_results.return_value = "Warriors"

        result = get_finished_games_from_thesportsdb(df)
        
        # Result should remain unchanged (not overwritten)
        self.assertEqual(result.iloc[0]["Result"], "Lakers")

    @patch('codebase.results_appending.sportsdb_results._get_results')
    @patch('codebase.results_appending.sportsdb_results._time_since_start')
    @patch('codebase.results_appending.sportsdb_results.time.sleep')
    def test_rate_limiting(self, mock_sleep, mock_time_filter, mock_get_results):
        """Test rate limiting after 30 requests."""
        current_time = datetime.now(timezone.utc)
        
        # Create 31 games to trigger rate limiting
        df = pd.DataFrame({
            "Match": [f"Team A @ Team B {i}" for i in range(31)],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")] * 31,
            "Result": ["Not Found"] * 31,
        })

        mock_time_filter.return_value = df
        mock_get_results.return_value = "Team B"

        get_finished_games_from_thesportsdb(df)
        
        # Should sleep once after 30 requests
        mock_sleep.assert_called_once_with(60)

    @patch('codebase.results_appending.sportsdb_results._get_results')
    @patch('codebase.results_appending.sportsdb_results._time_since_start')
    def test_multiple_games(self, mock_time_filter, mock_get_results):
        """Test processing multiple games."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2", "Game 3"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")] * 3,
            "Result": ["Not Found", "Pending", "Not Found"],
        })

        mock_time_filter.return_value = df
        mock_get_results.side_effect = ["Winner 1", "Winner 2", "Winner 3"]

        result = get_finished_games_from_thesportsdb(df)
        
        # All three should be updated
        self.assertEqual(result.iloc[0]["Result"], "Winner 1")
        self.assertEqual(result.iloc[1]["Result"], "Winner 2")
        self.assertEqual(result.iloc[2]["Result"], "Winner 3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
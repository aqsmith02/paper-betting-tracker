"""
test_theodds_results.py

Unit tests for theodds_results.py module.

Author: Test Suite
"""

import unittest
from unittest.mock import patch, Mock
import pandas as pd
from datetime import datetime, timezone, timedelta
from src.results.theodds_results import (
    _start_date,
    _parse_match_teams,
    _time_since_start,
    _get_scores_from_api,
    _filter,
    _get_winner,
    get_finished_games_from_theodds,
    map_league_to_key,
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

    def test_utc_timestamp_with_time(self):
        """Test that time component is ignored, only date is returned."""
        timestamp = "2025-11-04T23:59:59Z"
        result = _start_date(timestamp)
        self.assertEqual(result, "2025-11-04")


class TestParseMatchTeams(unittest.TestCase):
    """Test cases for _parse_match_teams function."""

    def test_standard_format(self):
        """Test parsing standard match format."""
        match = "Team A @ Team B"
        result = _parse_match_teams(match)
        self.assertEqual(result, ["Team A", "Team B"])

    def test_no_extra_spaces(self):
        """Test parsing when there are no extra spaces."""
        match = "Lakers@Warriors"
        result = _parse_match_teams(match)
        self.assertEqual(result, ["Lakers", "Warriors"])

    def test_extra_spaces(self):
        """Test parsing with extra spaces."""
        match = "New York Knicks  @  Los Angeles Lakers"
        result = _parse_match_teams(match)
        self.assertEqual(result, ["New York Knicks", "Los Angeles Lakers"])


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

    def test_no_games_filtered(self):
        """Test when no games should be filtered."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2"],
            "Start Time": [
                (current_time - timedelta(hours=20)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
        })

        result = _time_since_start(df, 12)
        
        # Both games should remain
        self.assertEqual(len(result), 2)

    def test_all_games_filtered(self):
        """Test when all games should be filtered."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Game 1", "Game 2"],
            "Start Time": [
                (current_time - timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                (current_time - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
        })

        result = _time_since_start(df, 12)
        
        # No games should remain
        self.assertEqual(len(result), 0)


class TestGetScoresFromApi(unittest.TestCase):
    """Test cases for _get_scores_from_api function."""

    @patch('codebase.results_appending.theodds_results.requests.get')
    def test_successful_api_call(self, mock_get):
        """Test successful API call."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": "1", "home_team": "Team A", "away_team": "Team B"}
        ]
        mock_get.return_value = mock_response

        result = _get_scores_from_api("basketball_nba")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["home_team"], "Team A")

    @patch('codebase.results_appending.theodds_results.requests.get')
    def test_api_error(self, mock_get):
        """Test API error handling."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = _get_scores_from_api("basketball_nba")
        
        self.assertEqual(result, [])

    @patch('codebase.results_appending.theodds_results.requests.get')
    def test_api_with_days_from(self, mock_get):
        """Test API call with custom days_from parameter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        _get_scores_from_api("basketball_nba", days_from=5)
        
        # Verify the URL contains daysFrom=5
        call_args = mock_get.call_args
        self.assertIn("daysFrom=5", call_args[0][0])


class TestFilter(unittest.TestCase):
    """Test cases for _filter function."""

    def test_filter_matching_game(self):
        """Test filtering to find matching game."""
        scores = [
            {
                "commence_time": "2025-11-04T19:00:00Z",
                "home_team": "Lakers",
                "away_team": "Warriors",
            },
            {
                "commence_time": "2025-11-04T20:00:00Z",
                "home_team": "Celtics",
                "away_team": "Heat",
            },
        ]

        result = _filter(scores, "2025-11-04", "Lakers", "Warriors")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["home_team"], "Lakers")

    def test_filter_no_matches(self):
        """Test filtering when no games match."""
        scores = [
            {
                "commence_time": "2025-11-04T19:00:00Z",
                "home_team": "Lakers",
                "away_team": "Warriors",
            },
        ]

        result = _filter(scores, "2025-11-05", "Lakers", "Warriors")
        
        self.assertEqual(len(result), 0)

    def test_filter_multiple_matches(self):
        """Test filtering when multiple games match (edge case)."""
        scores = [
            {
                "commence_time": "2025-11-04T19:00:00Z",
                "home_team": "Lakers",
                "away_team": "Warriors",
            },
            {
                "commence_time": "2025-11-04T21:00:00Z",
                "home_team": "Lakers",
                "away_team": "Warriors",
            },
        ]

        result = _filter(scores, "2025-11-04", "Lakers", "Warriors")
        
        self.assertEqual(len(result), 2)


class TestGetWinner(unittest.TestCase):
    """Test cases for _get_winner function."""

    def test_home_team_wins(self):
        """Test determining winner when home team wins."""
        game = {
            "completed": True,
            "home_team": "Lakers",
            "away_team": "Warriors",
            "scores": [
                {"name": "Lakers", "score": "110"},
                {"name": "Warriors", "score": "105"},
            ],
        }

        result = _get_winner(game)
        self.assertEqual(result, "Lakers")

    def test_away_team_wins(self):
        """Test determining winner when away team wins."""
        game = {
            "completed": True,
            "home_team": "Lakers",
            "away_team": "Warriors",
            "scores": [
                {"name": "Lakers", "score": "105"},
                {"name": "Warriors", "score": "110"},
            ],
        }

        result = _get_winner(game)
        self.assertEqual(result, "Warriors")

    def test_draw(self):
        """Test determining result when game is a draw."""
        game = {
            "completed": True,
            "home_team": "Team A",
            "away_team": "Team B",
            "scores": [
                {"name": "Team A", "score": "2"},
                {"name": "Team B", "score": "2"},
            ],
        }

        result = _get_winner(game)
        self.assertEqual(result, "Draw")

    def test_game_not_completed(self):
        """Test result when game is not completed."""
        game = {
            "completed": False,
            "home_team": "Lakers",
            "away_team": "Warriors",
        }

        result = _get_winner(game)
        self.assertEqual(result, "Pending")


class TestGetFinishedGamesFromTheodds(unittest.TestCase):
    """Test cases for get_finished_games_from_theodds function."""

    @patch('codebase.results_appending.theodds_results._get_scores_from_api')
    @patch('codebase.results_appending.theodds_results._time_since_start')
    def test_update_results(self, mock_time_filter, mock_get_scores):
        """Test updating results for games."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Warriors @ Lakers"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "Result": ["Not Found"],
        })

        mock_time_filter.return_value = df
        mock_get_scores.return_value = [
            {
                "commence_time": (current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "home_team": "Lakers",
                "away_team": "Warriors",
                "completed": True,
                "scores": [
                    {"name": "Lakers", "score": "110"},
                    {"name": "Warriors", "score": "105"},
                ],
            }
        ]

        result = get_finished_games_from_theodds(df, "basketball_nba")
        
        self.assertEqual(result.iloc[0]["Result"], "Lakers")

    @patch('codebase.results_appending.theodds_results._get_scores_from_api')
    @patch('codebase.results_appending.theodds_results._time_since_start')
    def test_skip_existing_results(self, mock_time_filter, mock_get_scores):
        """Test that existing results are not overwritten."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Warriors @ Lakers"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "Result": ["Lakers"],
        })

        mock_time_filter.return_value = df
        mock_get_scores.return_value = []

        result = get_finished_games_from_theodds(df, "basketball_nba")
        
        # Result should remain unchanged
        self.assertEqual(result.iloc[0]["Result"], "Lakers")

    @patch('codebase.results_appending.theodds_results._get_scores_from_api')
    @patch('codebase.results_appending.theodds_results._time_since_start')
    def test_game_not_found(self, mock_time_filter, mock_get_scores):
        """Test handling when game is not found in API results."""
        current_time = datetime.now(timezone.utc)
        
        df = pd.DataFrame({
            "Match": ["Warriors @ Lakers"],
            "Start Time": [(current_time - timedelta(hours=15)).strftime("%Y-%m-%dT%H:%M:%SZ")],
            "Result": ["Not Found"],
        })

        mock_time_filter.return_value = df
        mock_get_scores.return_value = []

        result = get_finished_games_from_theodds(df, "basketball_nba")
        
        self.assertEqual(result.iloc[0]["Result"], "Not Found")


class TestMapLeagueToKey(unittest.TestCase):
    """Test cases for map_league_to_key function."""

    def test_single_league(self):
        """Test mapping a single league."""
        df = pd.DataFrame({
            "League": ["NBA"],
        })

        result = map_league_to_key(df)
        
        # NBA is not in the mapping, so result should be empty
        self.assertEqual(len(result), 0)

    def test_multiple_leagues(self):
        """Test mapping multiple leagues."""
        df = pd.DataFrame({
            "League": ["NFL", "MLB", "NHL"],
        })

        result = map_league_to_key(df)
        
        self.assertEqual(len(result), 3)
        self.assertIn("americanfootball_nfl", result)
        self.assertIn("baseball_mlb", result)
        self.assertIn("icehockey_nhl", result)

    def test_duplicate_leagues(self):
        """Test that duplicate leagues result in unique keys."""
        df = pd.DataFrame({
            "League": ["NFL", "NFL", "MLB"],
        })

        result = map_league_to_key(df)
        
        self.assertEqual(len(result), 2)
        self.assertIn("americanfootball_nfl", result)
        self.assertIn("baseball_mlb", result)

    def test_unknown_league(self):
        """Test handling of leagues not in mapping."""
        df = pd.DataFrame({
            "League": ["Unknown League", "NFL"],
        })

        result = map_league_to_key(df)
        
        # Should only include NFL, unknown league ignored
        self.assertEqual(len(result), 1)
        self.assertIn("americanfootball_nfl", result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
"""
Tests for theodds_results.py

Tests functions that fetch and process sports game results from The-Odds-API.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
from src.results.theodds_results import (
    _get_scores_from_theodds,
    _get_pending_event_ids,
    _append_results,
    get_finished_games_from_theodds
)


class TestGetScoresFromTheOdds:
    """Test suite for _get_scores_from_theodds function."""

    @patch('src.results.theodds_results.requests.get')
    def test_successful_api_call(self, mock_get):
        """Test successful API call returns JSON data."""
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": "event123",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "3"},
                    {"name": "Team B", "score": "2"}
                ]
            }
        ]
        mock_get.return_value = mock_response
        
        result = _get_scores_from_theodds("baseball_mlb", "event123,event456", 3)
        
        assert len(result) == 1
        assert result[0]["id"] == "event123"
        assert result[0]["completed"] is True

    @patch('src.results.theodds_results.requests.get')
    def test_api_connection_error(self, mock_get):
        """Test API connection error returns empty list."""
        mock_get.side_effect = Exception("Connection error")
        
        result = _get_scores_from_theodds("baseball_mlb", "event123", 3)
        
        assert result == []


class TestGetPendingEventIds:
    """Test suite for _get_pending_event_ids function."""

    def test_single_pending_event(self):
        """Test with single pending event."""
        df = pd.DataFrame({
            "ID": ["event123"],
            "Result": ["Pending"]
        })
        
        result = _get_pending_event_ids(df)
        assert result == "event123"

    def test_no_pending_events(self):
        """Test with no pending events."""
        df = pd.DataFrame({
            "ID": ["event1", "event2"],
            "Result": ["Team A", "Team B"]
        })
        
        result = _get_pending_event_ids(df)
        assert result == ""

    def test_all_pending_results_types(self):
        """Test with all types of pending results."""
        df = pd.DataFrame({
            "ID": ["event1", "event2", "event3", "event4"],
            "Result": ["Pending", "Not Found", "API Error", "Team A"]
        })
        
        result = _get_pending_event_ids(df)
        # Should include all PENDING_RESULTS: ["Not Found", "Pending", "API Error"]
        assert "event1" in result
        assert "event2" in result
        assert "event3" in result
        assert "event4" not in result

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame({
            "ID": [],
            "Result": []
        })
        
        result = _get_pending_event_ids(df)
        assert result == ""


class TestAppendResults:
    """Test suite for _append_results function."""

    def test_single_completed_game_home_win(self):
        """Test appending result for single completed game where home team wins."""
        df = pd.DataFrame({
            "ID": ["event123"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event123",
                "completed": True,
                "scores": [
                    {"name": "Home Team", "score": "5"},
                    {"name": "Away Team", "score": "3"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Home Team"

    def test_single_completed_game_away_win(self):
        """Test appending result for single completed game where away team wins."""
        df = pd.DataFrame({
            "ID": ["event456"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event456",
                "completed": True,
                "scores": [
                    {"name": "Home Team", "score": "2"},
                    {"name": "Away Team", "score": "4"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Away Team"

    def test_draw_game(self):
        """Test appending result for game that ended in a draw."""
        df = pd.DataFrame({
            "ID": ["event789"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event789",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "3"},
                    {"name": "Team B", "score": "3"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Draw"

    def test_incomplete_game(self):
        """Test that incomplete games don't update results."""
        df = pd.DataFrame({
            "ID": ["event111"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event111",
                "completed": False,
                "scores": [
                    {"name": "Team A", "score": "1"},
                    {"name": "Team B", "score": "0"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        # Should remain Pending since game not completed
        assert df.loc[0, "Result"] == "Pending"

    def test_multiple_games(self):
        """Test appending results for multiple games."""
        df = pd.DataFrame({
            "ID": ["event1", "event2", "event3"],
            "Result": ["Pending", "Not Found", "Pending"]
        })
        
        game_dicts = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "5"},
                    {"name": "Team B", "score": "3"}
                ]
            },
            {
                "id": "event2",
                "completed": True,
                "scores": [
                    {"name": "Team C", "score": "2"},
                    {"name": "Team D", "score": "2"}
                ]
            },
            {
                "id": "event3",
                "completed": True,
                "scores": [
                    {"name": "Team E", "score": "1"},
                    {"name": "Team F", "score": "4"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Team A"
        assert df.loc[1, "Result"] == "Draw"
        assert df.loc[2, "Result"] == "Team F"

    def test_missing_event_in_api_response(self):
        """Test that events not in API response keep their original result."""
        df = pd.DataFrame({
            "ID": ["event1", "event2"],
            "Result": ["Pending", "Not Found"]
        })
        
        game_dicts = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "3"},
                    {"name": "Team B", "score": "2"}
                ]
            }
            # event2 not in response
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Team A"
        assert df.loc[1, "Result"] == "Not Found"  # Should remain unchanged

    def test_insufficient_scores(self):
        """Test game with insufficient score data."""
        df = pd.DataFrame({
            "ID": ["event1"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "3"}
                    # Missing second team
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        # Should remain Pending due to insufficient data
        assert df.loc[0, "Result"] == "Pending"

    def test_empty_game_dicts(self):
        """Test with empty game_dicts list."""
        df = pd.DataFrame({
            "ID": ["event1"],
            "Result": ["Pending"]
        })
        
        _append_results(df, [])
        
        # Should remain unchanged
        assert df.loc[0, "Result"] == "Pending"

    def test_zero_scores(self):
        """Test game with zero scores resulting in draw."""
        df = pd.DataFrame({
            "ID": ["event1"],
            "Result": ["Pending"]
        })
        
        game_dicts = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "0"},
                    {"name": "Team B", "score": "0"}
                ]
            }
        ]
        
        _append_results(df, game_dicts)
        
        assert df.loc[0, "Result"] == "Draw"


class TestGetFinishedGamesFromTheOdds:
    """Test suite for get_finished_games_from_theodds function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = get_finished_games_from_theodds(df)
        assert result.empty

    def test_no_games_to_check(self):
        """Test when no games meet time threshold."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ID": ["event1"],
            "Sport Key": ["baseball_mlb"],
            "Start Time": [(current_time - timedelta(hours=6)).isoformat()],
            "Result": ["Pending"]
        })
        
        result = get_finished_games_from_theodds(df)
        assert result.equals(df)

    @patch('src.results.theodds_results._get_scores_from_theodds')
    def test_single_sport_key_with_results(self, mock_get_scores):
        """Test fetching results for single sport key."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ID": ["event1", "event2"],
            "Sport Key": ["baseball_mlb", "baseball_mlb"],
            "Start Time": [
                (current_time - timedelta(hours=24)).isoformat(),
                (current_time - timedelta(hours=30)).isoformat()
            ],
            "Result": ["Pending", "Not Found"]
        })
        
        # Mock API response
        mock_get_scores.return_value = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Red Sox", "score": "5"},
                    {"name": "Yankees", "score": "3"}
                ]
            },
            {
                "id": "event2",
                "completed": True,
                "scores": [
                    {"name": "Dodgers", "score": "2"},
                    {"name": "Giants", "score": "4"}
                ]
            }
        ]
        
        result = get_finished_games_from_theodds(df)
        
        # Verify results were updated
        assert result.loc[0, "Result"] == "Red Sox"
        assert result.loc[1, "Result"] == "Giants"

    @patch('src.results.theodds_results._get_scores_from_theodds')
    def test_multiple_sport_keys(self, mock_get_scores):
        """Test fetching results for multiple sport keys."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ID": ["event1", "event2", "event3"],
            "Sport Key": ["baseball_mlb", "basketball_nba", "soccer_epl"],
            "Start Time": [
                (current_time - timedelta(hours=24)).isoformat(),
                (current_time - timedelta(hours=30)).isoformat(),
                (current_time - timedelta(hours=48)).isoformat()
            ],
            "Result": ["Pending", "Pending", "Not Found"]
        })
        
        # Mock API responses for different sports
        def mock_api_response(sport_key, event_ids, days=3):
            if "baseball_mlb" in sport_key:
                return [{"id": "event1", "completed": True, "scores": [
                    {"name": "Team A", "score": "5"}, {"name": "Team B", "score": "3"}
                ]}]
            elif "basketball_nba" in sport_key:
                return [{"id": "event2", "completed": True, "scores": [
                    {"name": "Team C", "score": "100"}, {"name": "Team D", "score": "98"}
                ]}]
            elif "soccer_epl" in sport_key:
                return [{"id": "event3", "completed": True, "scores": [
                    {"name": "Team E", "score": "2"}, {"name": "Team F", "score": "2"}
                ]}]
            return []
        
        mock_get_scores.side_effect = mock_api_response
        
        result = get_finished_games_from_theodds(df)
        
        # Verify results
        assert result.loc[0, "Result"] == "Team A"
        assert result.loc[1, "Result"] == "Team C"
        assert result.loc[2, "Result"] == "Draw"

    def test_sport_key_without_results_support(self):
        """Test that sport keys not in SPORT_KEYS_WITH_RESULTS are skipped."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ID": ["event1"],
            "Sport Key": ["unsupported_sport"],  # Not in SPORT_KEYS_WITH_RESULTS
            "Start Time": [(current_time - timedelta(hours=24)).isoformat()],
            "Result": ["Pending"]
        })
        
        result = get_finished_games_from_theodds(df)
        
        # Result should remain unchanged
        assert result.loc[0, "Result"] == "Pending"

    @patch('src.results.theodds_results._get_scores_from_theodds')
    def test_mixed_supported_and_unsupported_sports(self, mock_get_scores):
        """Test with mix of supported and unsupported sport keys."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame({
            "ID": ["event1", "event2"],
            "Sport Key": ["baseball_mlb", "unsupported_sport"],
            "Start Time": [
                (current_time - timedelta(hours=24)).isoformat(),
                (current_time - timedelta(hours=30)).isoformat()
            ],
            "Result": ["Pending", "Pending"]
        })
        
        mock_get_scores.return_value = [
            {
                "id": "event1",
                "completed": True,
                "scores": [
                    {"name": "Team A", "score": "5"},
                    {"name": "Team B", "score": "3"}
                ]
            }
        ]
        
        result = get_finished_games_from_theodds(df)
        
        # Only first event should be updated
        assert result.loc[0, "Result"] == "Team A"
        assert result.loc[1, "Result"] == "Pending"
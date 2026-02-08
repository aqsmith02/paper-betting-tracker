"""
Tests for sportsdb_results.py

Tests functions that fetch and process sports game results from TheSportsDB API.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.results.sportsdb_results import (
    _format_match_for_thesportsdb,
    _get_score_from_thesportsdb,
    _process_individual_result,
    get_finished_games_from_thesportsdb,
)


class TestFormatMatchForTheSportsDB:
    """Test suite for _format_match_for_thesportsdb function."""

    def test_format_away_at_home_match(self):
        """Test formatting of 'Team @ Team' format."""
        match = "Boston Red Sox @ New York Yankees"
        result = _format_match_for_thesportsdb(match)
        expected = "New_York_Yankees_vs_Boston_Red_Sox"
        assert result == expected

    def test_format_match_with_extra_spaces(self):
        """Test formatting with extra spaces around '@'."""
        match = "Team A  @  Team B"
        result = _format_match_for_thesportsdb(match)
        expected = "Team_B_vs_Team_A"
        assert result == expected

    def test_format_match_error(self):
        """Test match that's in an unrecognized format."""
        match = "Some Random String"
        with pytest.raises(ValueError):
            _format_match_for_thesportsdb(match)


class TestGetScoreFromTheSportsDB:
    """Test suite for _get_score_from_thesportsdb function."""

    @patch("src.results.sportsdb_results.requests.get")
    def test_successful_api_call(self, mock_get):
        """Test successful API call returns JSON data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "3",
                    "intAwayScore": "2",
                }
            ]
        }
        mock_get.return_value = mock_response

        result = _get_score_from_thesportsdb("Team_A_vs_Team_B", "2025-07-15")

        assert result["event"][0]["strHomeTeam"] == "Home Team"
        assert result["event"][0]["intHomeScore"] == "3"

    @patch("src.results.sportsdb_results.requests.get")
    def test_api_connection_error(self, mock_get):
        """Test API connection error returns empty list."""
        mock_get.side_effect = Exception("Connection error")

        result = _get_score_from_thesportsdb("Team_A_vs_Team_B", "2025-07-15")

        assert result == []


class TestProcessIndividualResult:
    """Test suite for _process_individual_result function."""

    def test_none_game_dict(self):
        """Test when game_dict is None."""
        result = _process_individual_result(None)
        assert result == "Not Found"

    def test_empty_game_dict(self):
        """Test when game_dict is empty."""
        result = _process_individual_result({})
        assert result == "Not Found"

    def test_no_event_key(self):
        """Test when game_dict has no 'event' key."""
        game_dict = {"other_key": "value"}
        result = _process_individual_result(game_dict)
        assert result == "Not Found"

    def test_empty_event_list(self):
        """Test when event list is empty."""
        game_dict = {"event": []}
        result = _process_individual_result(game_dict)
        assert result == "Not Found"

    def test_home_team_wins(self):
        """Test when home team wins."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "5",
                    "intAwayScore": "3",
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Home Team"

    def test_away_team_wins(self):
        """Test when away team wins."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "2",
                    "intAwayScore": "4",
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Away Team"

    def test_draw(self):
        """Test when game is a draw."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "3",
                    "intAwayScore": "3",
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Draw"

    def test_missing_home_score(self):
        """Test when home score is missing (game not finished)."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": None,
                    "intAwayScore": "3",
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Pending"

    def test_missing_away_score(self):
        """Test when away score is missing (game not finished)."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "3",
                    "intAwayScore": None,
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Pending"

    def test_both_scores_missing(self):
        """Test when both scores are missing."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": None,
                    "intAwayScore": None,
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Pending"

    def test_zero_score_game(self):
        """Test game with zero scores."""
        game_dict = {
            "event": [
                {
                    "strHomeTeam": "Home Team",
                    "strAwayTeam": "Away Team",
                    "intHomeScore": "0",
                    "intAwayScore": "0",
                }
            ]
        }
        result = _process_individual_result(game_dict)
        assert result == "Draw"


class TestGetFinishedGamesFromTheSportsDB:
    """Test suite for get_finished_games_from_thesportsdb function."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = get_finished_games_from_thesportsdb(df)
        assert result.empty

    def test_no_pending_games(self):
        """Test when no games have pending results."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "Match": ["Team A @ Team B"],
                "Start Time": [(current_time - timedelta(hours=24)).isoformat()],
                "Result": ["Team B"],  # Already has result
            }
        )

        result = get_finished_games_from_thesportsdb(df)
        assert result.equals(df)

    @patch("src.results.sportsdb_results._get_score_from_thesportsdb")
    def test_fetch_results_for_pending_games(self, mock_get_score):
        """Test fetching results for games with pending status."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "Match": ["Team A @ Team B", "Team C @ Team D"],
                "Start Time": [
                    (current_time - timedelta(hours=24)).isoformat(),
                    (current_time - timedelta(hours=30)).isoformat(),
                ],
                "Result": ["Pending", "Not Found"],
            }
        )

        # Mock API responses
        mock_get_score.side_effect = [
            {
                "event": [
                    {
                        "strHomeTeam": "Team B",
                        "strAwayTeam": "Team A",
                        "intHomeScore": "3",
                        "intAwayScore": "2",
                    }
                ]
            },
            {
                "event": [
                    {
                        "strHomeTeam": "Team D",
                        "strAwayTeam": "Team C",
                        "intHomeScore": "1",
                        "intAwayScore": "4",
                    }
                ]
            },
        ]

        result = get_finished_games_from_thesportsdb(df)

        # Verify results were updated
        assert result.loc[0, "Result"] == "Team B"
        assert result.loc[1, "Result"] == "Team C"

    @patch("src.results.sportsdb_results._get_score_from_thesportsdb")
    @patch("src.results.sportsdb_results._time_since_start")
    @patch("src.results.sportsdb_results.time.sleep")
    def test_rate_limiting(self, mock_sleep, mock_time_filter, mock_get_score):
        """Test that rate limiting is applied after batch of requests."""
        current_time = datetime.now(timezone.utc)

        # Create 35 games to trigger rate limiting (batch size is 30)
        games = []
        for i in range(35):
            games.append(
                {
                    "Match": f"Team {i} @ Team {i+1}",
                    "Start Time": (current_time - timedelta(hours=24)).isoformat(),
                    "Result": "Pending",
                }
            )

        df = pd.DataFrame(games)
        mock_time_filter.return_value = df

        # Mock API to return pending results
        mock_get_score.return_value = {
            "event": [
                {
                    "strHomeTeam": "Team",
                    "strAwayTeam": "Team",
                    "intHomeScore": None,
                    "intAwayScore": None,
                }
            ]
        }

        result = get_finished_games_from_thesportsdb(df)

        # Verify sleep was called once (after 30 requests)
        assert mock_sleep.call_count == 1
        assert mock_sleep.call_args[0][0] == 60  # SPORTSDB_RATE_LIMIT_WAIT

    @patch("src.results.sportsdb_results._get_score_from_thesportsdb")
    @patch("src.results.sportsdb_results._time_since_start")
    def test_mixed_results(self, mock_time_filter, mock_get_score):
        """Test with mix of pending and completed results."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "Match": ["Game 1 @ Game 2", "Game 3 @ Game 4", "Game 5 @ Game 6"],
                "Start Time": [
                    (current_time - timedelta(hours=24)).isoformat(),
                    (current_time - timedelta(hours=30)).isoformat(),
                    (current_time - timedelta(hours=48)).isoformat(),
                ],
                "Result": [
                    "Pending",
                    "Game 4",
                    "Not Found",
                ],  # Only first and third should be fetched
            }
        )

        # Filter should return only pending games
        pending_df = df[df["Result"].isin(["Not Found", "Pending"])]
        mock_time_filter.return_value = pending_df

        # Mock API responses
        mock_get_score.side_effect = [
            {
                "event": [
                    {
                        "strHomeTeam": "Game 2",
                        "strAwayTeam": "Game 1",
                        "intHomeScore": "5",
                        "intAwayScore": "3",
                    }
                ]
            },
            {
                "event": [
                    {
                        "strHomeTeam": "Game 6",
                        "strAwayTeam": "Game 5",
                        "intHomeScore": "2",
                        "intAwayScore": "2",
                    }
                ]
            },
        ]

        result = get_finished_games_from_thesportsdb(df)

        # First and third should be updated, second should remain unchanged
        assert result.loc[0, "Result"] == "Game 2"
        assert result.loc[1, "Result"] == "Game 4"
        assert result.loc[2, "Result"] == "Draw"

    @patch("src.results.sportsdb_results._get_score_from_thesportsdb")
    @patch("src.results.sportsdb_results._time_since_start")
    def test_api_not_found_result(self, mock_time_filter, mock_get_score):
        """Test when API returns no event (not found)."""
        current_time = datetime.now(timezone.utc)
        df = pd.DataFrame(
            {
                "Match": ["Team A @ Team B"],
                "Start Time": [(current_time - timedelta(hours=24)).isoformat()],
                "Result": ["Pending"],
            }
        )

        mock_time_filter.return_value = df
        mock_get_score.return_value = {"event": None}  # API returns no event

        result = get_finished_games_from_thesportsdb(df)

        # Result should be "Not Found"
        assert result.loc[0, "Result"] == "Not Found"

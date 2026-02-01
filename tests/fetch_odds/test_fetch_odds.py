"""
tests/test_fetch_odds.py

Unit tests for fetch_odds module with mocked API responses.

Author: Andrew Smith
Date: January 2026
"""

import pytest
from unittest.mock import patch, Mock
import pandas as pd
from src.fetch_odds.fetch_odds import (
    fetch_odds,
    _process_game,
    _create_bm_dict_list,
    _get_outcomes_list
)


@pytest.fixture
def sample_game():
    """Sample game data matching actual The-Odds-API response format"""
    return {
        "id": "a474a3086c863280dfd2b7be0ce30120",
        "sport_key": "basketball_wncaab",
        "sport_title": "WNCAAB",
        "commence_time": "2026-01-18T18:05:37Z",
        "home_team": "NC State Wolfpack",
        "away_team": "Louisville Cardinals",
        "bookmakers": [
            {
                "key": "betrivers",
                "title": "BetRivers",
                "last_update": "2026-01-18T20:16:55Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-01-18T20:16:55Z",
                        "outcomes": [
                            {"name": "Louisville Cardinals", "price": 1.19},
                            {"name": "NC State Wolfpack", "price": 4.1}
                        ]
                    }
                ]
            },
            {
                "key": "williamhill_us",
                "title": "Caesars",
                "last_update": "2026-01-18T20:13:39Z",
                "markets": [
                    {
                        "key": "h2h",
                        "last_update": "2026-01-18T20:13:39Z",
                        "outcomes": [
                            {"name": "Louisville Cardinals", "price": 1.62},
                            {"name": "NC State Wolfpack", "price": 2.25}
                        ]
                    }
                ]
            }
        ]
    }





class TestGetOutcomesList:
    """Tests for _get_outcomes_list function"""
    
    def test_normal_case(self):
        """Test extracting outcomes from bookmaker dictionary"""
        bm_dicts = [
            {"BetRivers": {"Louisville Cardinals": 1.19, "NC State Wolfpack": 4.1}}
        ]
        result = _get_outcomes_list(bm_dicts)
        assert set(result) == {"Louisville Cardinals", "NC State Wolfpack"}
        assert len(result) == 2
    
    def test_empty_list(self):
        """Test with no bookmakers"""
        assert _get_outcomes_list([]) == []
    
    def test_multiple_bookmakers(self):
        """Test that it uses only the first bookmaker"""
        bm_dicts = [
            {"BetRivers": {"TeamA": 1.85, "TeamB": 2.10}},
            {"Caesars": {"TeamA": 1.90, "TeamB": 2.05, "TeamC": 3.0}}
        ]
        result = _get_outcomes_list(bm_dicts)
        # Should use first bookmaker only (BetRivers)
        assert result == ["TeamA", "TeamB"]
        assert "TeamC" not in result


class TestCreateBmDictList:
    """Tests for _create_bm_dict_list function"""
    
    def test_normal_case(self, sample_game):
        """Test creating bookmaker dictionaries from game data"""
        result = _create_bm_dict_list(sample_game)
        
        assert len(result) == 2
        assert "BetRivers" in result[0]
        assert "Caesars" in result[1]
        
        # Check BetRivers odds
        assert result[0]["BetRivers"]["Louisville Cardinals"] == 1.19
        assert result[0]["BetRivers"]["NC State Wolfpack"] == 4.1
        
        # Check Caesars odds
        assert result[1]["Caesars"]["Louisville Cardinals"] == 1.62
        assert result[1]["Caesars"]["NC State Wolfpack"] == 2.25
    
    def test_no_bookmakers(self):
        """Test game with no bookmakers"""
        game = {
            "id": "123",
            "sport_key": "basketball_nba",
            "bookmakers": []
        }
        result = _create_bm_dict_list(game)
        assert result == []
    
    def test_no_h2h_market(self):
        """Test bookmaker with no h2h market (only spreads)"""
        game = {
            "id": "123",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "TeamA", "price": 1.91}
                            ]
                        }
                    ]
                }
            ]
        }
        result = _create_bm_dict_list(game)
        assert result == []
    
    def test_missing_markets_key(self):
        """Test bookmaker with no markets key"""
        game = {
            "id": "123",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings"
                }
            ]
        }
        result = _create_bm_dict_list(game)
        assert result == []
    
    def test_empty_outcomes(self):
        """Test market with no outcomes"""
        game = {
            "id": "123",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": []
                        }
                    ]
                }
            ]
        }
        result = _create_bm_dict_list(game)
        # Should create dict but with empty outcomes
        assert len(result) == 1
        assert result[0]["DraftKings"] == {}


class TestProcessGame:
    """Tests for _process_game function"""
    
    def test_creates_correct_rows(self, sample_game):
        """Test that game is processed into correct row format"""
        rows = _process_game(sample_game)
        
        # Should create 2 rows (one per outcome)
        assert len(rows) == 2
        
        # Check Louisville row
        louisville_row = next(r for r in rows if r["Team"] == "Louisville Cardinals")
        assert louisville_row["ID"] == "a474a3086c863280dfd2b7be0ce30120"
        assert louisville_row["Sport Key"] == "basketball_wncaab"
        assert louisville_row["Sport Title"] == "WNCAAB"
        assert louisville_row["Match"] == "Louisville Cardinals @ NC State Wolfpack"
        assert louisville_row["Team"] == "Louisville Cardinals"
        assert louisville_row["Start Time"] == "2026-01-18T18:05:37Z"
        assert louisville_row["BetRivers"] == 1.19
        assert louisville_row["Caesars"] == 1.62
        
        # Check NC State row
        ncstate_row = next(r for r in rows if r["Team"] == "NC State Wolfpack")
        assert ncstate_row["ID"] == "a474a3086c863280dfd2b7be0ce30120"
        assert ncstate_row["Sport Key"] == "basketball_wncaab"
        assert ncstate_row["Sport Title"] == "WNCAAB"
        assert ncstate_row["Match"] == "Louisville Cardinals @ NC State Wolfpack"
        assert ncstate_row["Team"] == "NC State Wolfpack"
        assert ncstate_row["Start Time"] == "2026-01-18T18:05:37Z"
        assert ncstate_row["BetRivers"] == 4.1
        assert ncstate_row["Caesars"] == 2.25
    
    def test_handles_missing_bookmaker_odds(self):
        """Test when one bookmaker doesn't have odds for all outcomes"""
        game = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2026-01-25T19:00:00Z",
            "home_team": "Lakers",
            "away_team": "Warriors",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 1.85}
                                # Warriors missing - bookmaker suspended odds
                            ]
                        }
                    ]
                }
            ]
        }
        rows = _process_game(game)
        
        # Should still create row for Lakers
        assert len(rows) == 1
        lakers_row = rows[0]
        assert lakers_row["Team"] == "Lakers"
        assert lakers_row["DraftKings"] == 1.85
    
    def test_none_for_missing_odds(self):
        """Test that None is used when bookmaker doesn't have odds for outcome"""
        game = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2026-01-25T19:00:00Z",
            "home_team": "Lakers",
            "away_team": "Warriors",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 1.85},
                                {"name": "Warriors", "price": 2.10}
                            ]
                        }
                    ]
                },
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": 1.90}
                                # Warriors not offered
                            ]
                        }
                    ]
                }
            ]
        }
        rows = _process_game(game)
        
        warriors_row = next(r for r in rows if r["Team"] == "Warriors")
        assert warriors_row["DraftKings"] == 2.10
        assert warriors_row["FanDuel"] is None


class TestFetchOdds:
    """Tests for main fetch_odds function"""
    
    def test_successful_fetch(self, sample_game):
        """Test successful API call and data processing"""
        df = fetch_odds([sample_game])
        
        # Verify DataFrame structure
        assert not df.empty
        assert len(df) == 2  # Two outcomes
        
        # Verify columns exist
        expected_columns = ['ID', 'Sport Key', 'Sport Title', 'Start Time', 
                          'Match', 'Team', 'BetRivers', 'Caesars']
        for col in expected_columns:
            assert col in df.columns
        
        # Verify first row (Louisville Cardinals)
        louisville_row = df[df['Team'] == 'Louisville Cardinals'].iloc[0]
        assert louisville_row['ID'] == 'a474a3086c863280dfd2b7be0ce30120'
        assert louisville_row['Sport Key'] == 'basketball_wncaab'
        assert louisville_row['Sport Title'] == 'WNCAAB'
        assert louisville_row['Start Time'] == '2026-01-18T18:05:37Z'
        assert louisville_row['Match'] == 'Louisville Cardinals @ NC State Wolfpack'
        assert louisville_row['Team'] == 'Louisville Cardinals'
        assert louisville_row['BetRivers'] == 1.19
        assert louisville_row['Caesars'] == 1.62
        
        # Verify second row (NC State Wolfpack)
        ncstate_row = df[df['Team'] == 'NC State Wolfpack'].iloc[0]
        assert ncstate_row['ID'] == 'a474a3086c863280dfd2b7be0ce30120'
        assert ncstate_row['Sport Key'] == 'basketball_wncaab'
        assert ncstate_row['Sport Title'] == 'WNCAAB'
        assert ncstate_row['Start Time'] == '2026-01-18T18:05:37Z'
        assert ncstate_row['Match'] == 'Louisville Cardinals @ NC State Wolfpack'
        assert ncstate_row['Team'] == 'NC State Wolfpack'
        assert ncstate_row['BetRivers'] == 4.1
        assert ncstate_row['Caesars'] == 2.25
    
    def test_multiple_games(self, sample_game):
        """Test processing multiple games"""
        game2 = sample_game.copy()
        game2["id"] = "different_id"
        game2["home_team"] = "Duke Blue Devils"
        game2["away_team"] = "UNC Tar Heels"
        
        df = fetch_odds([sample_game, game2])
        
        # Should have 4 rows (2 outcomes per game)
        assert len(df) == 4
        assert df["ID"].nunique() == 2
    
    def test_empty_games_list(self):
        """Test API returning no games"""
        df = fetch_odds([])
        
        assert df.empty
    
    def test_game_processing_error_continues(self, sample_game):
        """Test that errors in individual games don't stop processing"""
        bad_game = {"id": "bad"}  # Missing required fields
        
        df = fetch_odds([sample_game, bad_game])
        
        # Should still process the good game despite bad game error
        assert len(df) == 2
        assert df.iloc[0]["ID"] == "a474a3086c863280dfd2b7be0ce30120"
    

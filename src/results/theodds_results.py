"""
theodds_results.py

The file fetches sports game results using The-Odds-API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""

import requests
import pandas as pd
from typing import List, Dict
from src.constants import PENDING_RESULTS
from src.results.date_utils import _start_date, _time_since_start
from config.api_config import THE_ODDS_API_KEY


def _parse_match_teams(match: str) -> List[str]:
    """
    Convert a match string to a list containing the individual teams.

    Args:
        match (str): Match string in format "Team1 @ Team2".

    Returns:
        List[str]: List containing individual team names [away_team, home_team].
    """
    return [t.strip() for t in match.split("@")]


def _get_scores_from_api(sports_key: str, days_from: int = 3) -> List[Dict]:
    """
    Fetch game outcomes from The-Odds-API for the last specified number of days.

    Args:
        sports_key (str): Sport key to fetch games for (maximum 3 days lookback).
        days_from (int): Number of days back to look for completed games.

    Returns:
        List[Dict]: List of completed game dictionaries from the API response.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sports_key}/scores/?daysFrom={days_from}&apiKey={THE_ODDS_API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error fetching scores from Odds API: {resp.status_code}")
        return []
    return resp.json()


def _filter(
    scores: List[Dict], start_date: str, home_team: str, away_team: str
) -> List[Dict]:
    """
    Filter games list to find matches with specific date and teams.

    Args:
        scores (List[Dict]): List of game dictionaries from _get_scores_from_api().
        start_date (str): Start date of the desired game in YYYY-MM-DD format.
        home_team (str): Home team name from the desired game.
        away_team (str): Away team name from the desired game.

    Returns:
        List[Dict]: List containing the game(s) matching the submitted criteria.
    """
    return [
        game
        for game in scores
        if _start_date(game.get("commence_time")) == start_date
        and game.get("home_team") == home_team
        and game.get("away_team") == away_team
    ]


def _get_winner(game: Dict) -> str:
    """
    Determine the game result by comparing scores from a completed game.

    Args:
        game (Dict): Game dictionary containing score and completion information.

    Returns:
        str: Winning team name or "Pending" if game has not completed.
    """
    if not game.get("completed"):
        return "Pending"

    home_team = game["home_team"]
    away_team = game["away_team"]

    home_score = None
    away_score = None

    for item in game["scores"]:
        if item["name"] == home_team:
            home_score = int(item["score"])
        elif item["name"] == away_team:
            away_score = int(item["score"])

    print(f"{home_team} (H) vs {away_team} (A): {home_score}-{away_score}")

    if home_score > away_score:
        return home_team
    elif away_score > home_score:
        return away_team
    else:
        return "Draw"


def get_finished_games_from_theodds(df: pd.DataFrame, sports_key: str) -> pd.DataFrame:
    """
    Add game results to DataFrame by fetching data from The-Odds-API.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with columns "Match", "Start Time", and optionally "Result".
        sports_key (str): Sport key for The-Odds-API to specify which sport's results to fetch.

    Returns:
        pd.DataFrame: Updated DataFrame with "Result" column populated from API calls.
    """
    # Get a list of the games from the past 3 days in the specified sport
    scores = _get_scores_from_api(sports_key)

    # Filter out games that started less than 12 hours ago
    indices = _time_since_start(df, 12).index.tolist()

    for i in indices:
        row = df.iloc[i]
        existing_result = row.get("Result")

        # Skip rows that already have a valid result
        if existing_result not in PENDING_RESULTS:
            continue

        # Note the necessary args for the filter() function
        start_date = _start_date(row["Start Time"])
        teams = _parse_match_teams(row["Match"])
        away_team, home_team = teams[0], teams[1]

        matches = _filter(scores, start_date, home_team, away_team)

        # With the scores list filtered to match the game at this row, find the result
        if matches:
            result = _get_winner(matches[0])
        else:
            result = "Not Found"

        df.at[i, "Result"] = result

    return df
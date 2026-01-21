"""
sportsdb_results.py

The file fetches sports game results using TheSportsDB API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""

import requests
import pandas as pd
import time
from src.constants import PENDING_RESULTS, API_REQUEST_THRESHOLD_HOURS, SPORTSDB_RATE_LIMIT_BATCH, SPORTSDB_RATE_LIMIT_WAIT
from src.results.date_utils import _start_date, _time_since_start
from config.api_config import THE_SPORTS_DB_API_KEY


def _format_match_for_thesportsdb(match: str) -> str:
    """
    Convert a match string to TheSportsDB API format.

    Args:
        match (str): Match string in format "Team1 @ Team2" or "Team1 vs Team2".

    Returns:
        str: Formatted match string with underscores replacing spaces (e.g., "Team1_vs_Team2").
    """
    if "@" in match:
        teams = [t.strip() for t in match.split("@")]
        formatted = f"{teams[1]} vs {teams[0]}"
    elif "vs" in match.lower():
        formatted = match
    else:
        return match.replace(" ", "_")
    return formatted.replace(" ", "_")


def _get_results(match: str, date: str) -> str:
    """
    Fetch the results of a game from TheSportsDB API.

    Args:
        match (str): Formatted match name for API query.
        date (str): Date of the match in YYYY-MM-DD format.

    Returns:
        str: Game outcome - winning team name, "Draw", "Pending", "Not Found", or "API Error".
    """
    url = f"https://www.thesportsdb.com/api/v1/json/{THE_SPORTS_DB_API_KEY}/searchevents.php?e={match}&d={date}"
    try:
        # Request url
        resp = requests.get(url)
        if resp.status_code != 200:
            return "API Error"

        # Store data and check if it does not exist
        data = resp.json()
        if not data or "event" not in data or not data["event"]:
            return "Not Found"

        # Store event
        event = data["event"][0]

        # Store teams
        home = event.get("strHomeTeam", "Home")
        away = event.get("strAwayTeam", "Away")

        # Store scores
        home_score = event.get("intHomeScore")
        away_score = event.get("intAwayScore")

        print(f"{home} (H) vs {away} (A): {home_score}-{away_score}")

        if home_score is None or away_score is None:
            return "Pending"

        # Results
        home_score, away_score = int(home_score), int(away_score)
        if home_score > away_score:
            return home
        elif away_score > home_score:
            return away
        else:
            return "Draw"

    except Exception as e:
        print(f"Error fetching match: {e}")
        return "Error"


def get_finished_games_from_thesportsdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch and update game results from TheSportsDB API for games in DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing game data with columns "Match", "Start Time", and optionally "Result".

    Returns:
        pd.DataFrame: Updated DataFrame with "Result" column populated from API calls.
    """
    # Only loop through games that started more than API_REQUEST_THRESHOLD_HOURS hours ago and have pending results
    indices = _time_since_start(df, API_REQUEST_THRESHOLD_HOURS).index.tolist()
    indices = [i for i in indices if df.at[i, "Result"] in PENDING_RESULTS]

    # Track API requests to respect rate limits
    fetches = 0

    for i in indices:
        row = df.iloc[i]

        match = _format_match_for_thesportsdb(row["Match"])
        date = _start_date(row["Start Time"])
        result = _get_results(match, date)
        fetches += 1
        df.at[i, "Result"] = result

        if fetches % SPORTSDB_RATE_LIMIT_BATCH == 0:
            # Every SPORTSDB_RATE_LIMIT_BATCH requests, wait SPORTSDB_RATE_LIMIT_WAIT
            print(f"\nPausing for {SPORTSDB_RATE_LIMIT_WAIT} seconds to respect SportsDB API rate limits...\n")
            time.sleep(SPORTSDB_RATE_LIMIT_WAIT)

    return df
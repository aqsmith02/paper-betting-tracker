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
from typing import List, Dict
from src.constants import PENDING_RESULTS, API_REQUEST_THRESHOLD_HOURS, SPORTSDB_RATE_LIMIT_BATCH, SPORTSDB_RATE_LIMIT_WAIT, RESULT_COLUMN, START_TIME_COLUMN, MATCH_COLUMN
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


def _get_score_from_thesportsdb(match: str, date: str) -> List[Dict]:
    """
    Fetch game outcome from TheSportsDB for the event specified.

    Args:
        match (str): Formatted match name for API query.
        date (str): Date of the match in YYYY-MM-DD format.

    Returns:
        List[Dict]: List of completed game dictionaries from the API response.
    """
    url = f"https://www.thesportsdb.com/api/v1/json/{THE_SPORTS_DB_API_KEY}/searchevents.php?e={match}&d={date}"

    try:
        resp = requests.get(url)
        return resp.json()
    except:
        print("Error connecting to TheSportsDB API.")
        return []
    

def _process_individual_result(game_dict: List[Dict]) -> None:
    """
    Append game result from API data to the DataFrame.

    Args:
        game_dicts (List[Dict]): List of game dictionaries from _get_scores_from_api().

    Returns:
        None: The function modifies the DataFrame in place.
    """
    if game_dict is None or "event" not in game_dict or not game_dict["event"]:
        return "Not Found"
    
    # Store event
    event = game_dict["event"][0]

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


def get_finished_games_from_thesportsdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch and update game results from TheSportsDB API for games in DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing game data with columns "Match", "Start Time", and optionally "Result".

    Returns:
        pd.DataFrame: Updated DataFrame with "Result" column populated from API calls.
    """
    # Only use games that finished more than API_REQUEST_THRESHOLD_HOURS days ago and result is pending
    filtered_df = _time_since_start(df, API_REQUEST_THRESHOLD_HOURS)
    filtered_df = filtered_df[filtered_df[RESULT_COLUMN].isin(PENDING_RESULTS)]

    # If no games to check, return original DataFrame
    if filtered_df.empty:
        print("No games to check from TheSportsDB")
        return df

    # Track API requests to respect rate limits
    fetches = 0

    # Loop through filtered games and fetch results
    for idx in filtered_df.index:
        row = df.loc[idx]
        
        # Format match and get date
        match = _format_match_for_thesportsdb(row[MATCH_COLUMN])
        date = _start_date(row[START_TIME_COLUMN])
        
        # Fetch result from API
        game_dict = _get_score_from_thesportsdb(match, date)
        result = _process_individual_result(df, game_dict)
        
        # Update the original DataFrame
        df.at[idx, RESULT_COLUMN] = result
        fetches += 1

        # Rate limiting: pause every SPORTSDB_RATE_LIMIT_BATCH requests
        if fetches % SPORTSDB_RATE_LIMIT_BATCH == 0:
            print(f"\nPausing for {SPORTSDB_RATE_LIMIT_WAIT} seconds to respect SportsDB API rate limits...\n")
            time.sleep(SPORTSDB_RATE_LIMIT_WAIT)

    return df
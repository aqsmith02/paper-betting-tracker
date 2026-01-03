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
import yaml
from datetime import datetime, timedelta, timezone
from typing import Any
from src.constants import PENDING_RESULTS, CONFIG_DIR

# Load config

config_path = CONFIG_DIR / "api_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# API Keys
THE_SPORTS_DB_API_KEY = config["api"]["the_sports_db_api_key"]


def _start_date(ts: Any) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to "YYYY-MM-DD".

    Args:
        ts (Any): Timestamp to convert to date (UTC).

    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    dt = pd.to_datetime(ts)
    return dt.strftime("%Y-%m-%d")


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


def _time_since_start(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Filter out games that started less than threshold hours ago.

    Args:
        df (pd.DataFrame): DataFrame containing game data with "Start Time" column.
        thresh (float): Threshold in hours - games starting less than this many hours ago are filtered out.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only games that started more than thresh hours ago.
    """
    # Handle empty DataFrame
    if df.empty:
        return df
    
    # Get the current time in UTC
    current_time = datetime.now(timezone.utc)

    # Convert "Start Time" column to datetime objects and make timezone-aware (UTC)
    # Use format='ISO8601' to handle both ISO format strings and other formats
    df["Start Time"] = pd.to_datetime(df["Start Time"], format='ISO8601', utc=True)

    # Create conditions for removal
    cutoff = current_time - timedelta(hours=thresh)

    # Filter out games that started less than threshold hours ago
    mask = df["Start Time"] <= cutoff
    df = df[mask]

    return df


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
    # Only loop through games that started more than 12 hours ago
    indices = _time_since_start(df, 12).index.tolist()

    # Track API requests to respect rate limits
    fetches = 0

    for i in indices:
        row = df.iloc[i]
        existing_result = row.get("Result")

        # Skip rows that already have a result other than "Not Found"
        if existing_result not in PENDING_RESULTS:
            continue

        match = _format_match_for_thesportsdb(row["Match"])
        date = _start_date(row["Start Time"])
        result = _get_results(match, date)
        fetches += 1
        df.at[i, "Result"] = result

        if fetches % 30 == 0:
            # Every 30 requests, wait 60 seconds
            print("\nPausing for 60 seconds to respect SportsDB API rate limits...\n")
            time.sleep(60)

    return df
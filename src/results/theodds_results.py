"""
theodds_results.py

The file fetches sports game results using The-Odds-API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores/?daysFrom=3&apiKey=f59d869954199512fe61d505fbf60fb8
import requests
import pandas as pd
from typing import List, Dict
from src.constants import PENDING_RESULTS, API_REQUEST_THRESHOLD_HOURS, DAYS_FROM_SCORE_FETCHING
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


def _get_scores_from_api(sports_key: str, event_ids: str, days_from: int = DAYS_FROM_SCORE_FETCHING) -> List[Dict]:
    """
    Fetch game outcomes from The-Odds-API for the event IDs specified.

    Args:
        sports_key (str): Sport key to fetch games for (maximum 3 days lookback).
        event_ids (str): Comma-separated string of event IDs to filter the scores.
        days_from (int): Number of days back to look for completed games.

    Returns:
        List[Dict]: List of completed game dictionaries from the API response.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sports_key}/scores/?daysFrom={days_from}&apiKey={THE_ODDS_API_KEY}&eventIds={event_ids}"
    print(url)

    try:
        resp = requests.get(url)
        return resp.json()
    except:
        print("Error connecting to Odds API.")
        return []
    

def _get_pending_event_ids(df: pd.DataFrame) -> str:
    """
    Get a comma-separated string of event IDs for games with pending results.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with "Event ID" and "Result" columns.

    Returns:
        str: Comma-separated string of event IDs with pending results.
    """
    pending_event_ids = df[df["Result"].isin(PENDING_RESULTS)]["Event ID"]
    return ",".join(pending_event_ids)


def _append_results(df: pd.DataFrame, game_dicts: List[Dict]) -> None:
    """
    Append game results from API data to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with "ID" and "Result" columns.
        game_dicts (List[Dict]): List of game dictionaries from _get_scores_from_api().

    Returns:
        None: The function modifies the DataFrame in place.
    """
    # Create a mapping of game ID to winner
    id_to_winner = {}
    
    for game in game_dicts:
        if not game.get('completed'):
            continue
        
        game_id = game.get('id')
        scores = game.get('scores', [])
        
        if len(scores) < 2:
            continue
        
        # Determine winner
        team1_score = int(scores[0]['score'])
        team2_score = int(scores[1]['score'])
        
        if team1_score > team2_score:
            winner = scores[0]['name']
        elif team2_score > team1_score:
            winner = scores[1]['name']
        else:
            winner = "Draw"
        
        id_to_winner[game_id] = winner
    
    # Update results, fill na with existing results (e.g., "Not Found")
    df['Result'] = df['ID'].map(id_to_winner).fillna(df['Result'])


def get_finished_games_from_theodds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add game results to DataFrame by fetching data from The-Odds-API.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with columns "Match", "Start Time", and optionally "Result".
        sports_key (str): Sport key for The-Odds-API to specify which sport's results to fetch.

    Returns:
        pd.DataFrame: Updated DataFrame with "Result" column populated from API calls.
    """
    # Only use games that finished more than API_REQUEST_THRESHOLD_HOURS days ago
    filtered_df = _time_since_start(df, API_REQUEST_THRESHOLD_HOURS)

    # Get a list of game dictionaries based on pending event IDs
    event_ids = _get_pending_event_ids(filtered_df)

    # Group by Sport Key and combine event IDs
    grouped = filtered_df.groupby('Sport Key')['ID'].apply(lambda x: ','.join(x)).reset_index()

    # Loop through each sport key and fetch scores
    for sport_key, event_ids in grouped.items():
        game_dicts = _get_scores_from_api(sport_key, event_ids)
        _append_results(df, game_dicts)

    return df
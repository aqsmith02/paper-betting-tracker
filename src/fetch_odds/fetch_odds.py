"""
fetch_odds.py

The file fetches sports betting odds using The-Odds-API. Pulls in odds from any designated sport,
region, or market. Organizes a DataFrame with odds to contain one row per outcome, with columns being
bookmakers and essential information.

Author: Andrew Smith
Date: July 2025
"""

import requests
import pandas as pd
import json
from typing import List, Dict
from .fetch_configs import SPORT, SPORT_KEY, REGIONS, MARKETS, ODDS_FORMAT
from src.constants import THEODDS_API_KEY


def fetch_odds() -> pd.DataFrame:
    """
    Fetches head-to-head (h2h) betting odds from The Odds API.

    Returns:
        pd.DataFrame: DataFrame with columns: match, league, start time, team, bookmaker, odds, last update.
                    Each row represents one team's odds from one bookmaker.
    """
    # Build API request
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": THEODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }

    print(f"Fetching odds for sport: {SPORT}")
    
    try:
        response = requests.get(url, params=params)
    except Exception as e:
        print(f"Network error during API request: {e}")
        return pd.DataFrame()

    # Log API usage
    print("Requests Remaining:", response.headers.get("x-requests-remaining"))
    print("Requests Used:", response.headers.get("x-requests-used"))

    if response.status_code != 200:
        print(f"API request failed: {response.status_code} - {response.text}")
        return pd.DataFrame()

    games_data = response.json()
    print(json.dumps(games_data, indent=2))
    print(f"Retrieved {len(games_data)} games")

    # Process each game into rows
    rows = []
    for game in games_data:
        try:
            # After processing the game, a list of dicts are returned, which are added individually as rows here
            rows.extend(_process_game(game))
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    return pd.DataFrame(rows)


def _process_game(game: Dict) -> List[Dict]:
    """
    Process a single game from the API response into one row per outcome (row includes match data
    and bookmakers).

    Args:
        game (Dict): Game data dictionary from The Odds API response.

    Returns:
        List[Dict]: List of dictionaries, where each dictionary is an outcome.
    """
    home_team = game["home_team"]
    away_team = game["away_team"]
    league = game["sport_title"]
    start_time = game["commence_time"]  # Use UTC time directly
    bm_dicts = _create_bm_dict_list(game)
    rows = []

    # (Assuming all bookmakers offer odds for the same outcomes)
    # Get the list of outcomes from the first bookmaker
    if not bm_dicts:
        outcomes_list = []
    else:
        first_bm_name = list(bm_dicts[0].keys())[0]
        outcomes_list = list(bm_dicts[0][first_bm_name].keys())

    # Create a row for each outcome including match data and all bookmakers
    for outcome_team in outcomes_list:
        row = {
            "match": f"{away_team} @ {home_team}",
            "league": league,
            "start_time": start_time,
            "team": outcome_team,
        }

        # Add each bookmaker's odds for this outcome
        for bm in bm_dicts:
            for bm_name, odds_dict in bm.items():
                row[bm_name] = odds_dict.get(outcome_team, None)  # None if outcome not found


        rows.append(row)

    return rows


def _create_bm_dict_list(game: Dict) -> List[Dict]:
    """
    Processes a single game dictionary and builds a list of bookmaker dictionaries, each mapping the
    bookmaker name to its odds for all outcomes in the game.

    Args:
        game (Dict): Game data dictionary from The Odds API response.

    Returns:
        List[Dict]: List of dictionaries, where each dictionary is a bookmaker and it's odds for different
                    outcomes.
    """
    bm_dicts_list = []
    bookmakers = game.get("bookmakers", [])
    
    for bm in bookmakers:
        bookmaker_name = bm["title"]
        markets = bm.get("markets", [])
        
        # Find the h2h market specifically
        h2h_market = None
        for market in markets:
            if market.get("key") == "h2h":
                h2h_market = market
                break
        
        # Skip this bookmaker if no h2h market found
        if h2h_market is None:
            continue
            
        # Create a dict for each outcome
        outcomes = {o["name"]: o["price"] for o in h2h_market.get("outcomes", [])}
        bm_dict = {bookmaker_name: outcomes}
        bm_dicts_list.append(bm_dict)

    return bm_dicts_list


def main() -> None:
    """
    Main pipeline for fetching, organizing, and saving sports betting odds.

    Args:
        None

    Returns:
        None
    """
    # Fetch and organize odds data
    print("Starting odds fetch...")
    raw_odds = fetch_odds()

    if not raw_odds.empty:
        output_file = "odds.csv"
        raw_odds.to_csv(output_file, index=False)
        print(f"Saved {len(raw_odds)} outcomes to {output_file}")
    else:
        print("No odds data retrieved")


if __name__ == "__main__":
    main()
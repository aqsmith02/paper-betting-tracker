"""
fetch_odds.py

The file fetches sports betting odds using The-Odds-API. Pulls in odds from any designated sport,
region, or market. Organizes a DataFrame with odds to contain one row per outcome of a game, with columns being
bookmakers and essential information.

Author: Andrew Smith
Date: July 2025
"""

from typing import Dict, List, Optional

import pandas as pd
import requests

from config.api_config import THE_ODDS_API_KEY
from config.fetch_config import MARKETS, ODDS_FORMAT, REGIONS, SPORT, SPORT_KEY


def _get_outcomes_list(bm_dicts: List[Dict]) -> List[str]:
    """
    Extract the list of outcome names from bookmaker dictionaries.

    Assumes all bookmakers offer odds for the same outcomes and uses the first
    bookmaker to determine what outcomes are available.

    Args:
        bm_dicts: List of bookmaker dictionaries, each mapping bookmaker name
                  to outcome odds (e.g., [{"DraftKings": {"TeamA": 2.10, "TeamB": 1.85}}])

    Returns:
        List of outcome names (e.g., ["TeamA", "TeamB"]).
        Returns empty list if no bookmakers available.
    """
    if not bm_dicts:
        return []

    # Get first bookmaker's name and extract its outcome names
    first_bm_name = list(bm_dicts[0].keys())[0]
    return list(bm_dicts[0][first_bm_name].keys())


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


def _process_game(game: Dict) -> List[Dict]:
    """
    Process a single game from the API response into one row per outcome (row includes match data
    and bookmakers).

    Args:
        game (Dict): Game data dictionary from The Odds API response.

    Returns:
        List[Dict]: List of dictionaries, where each dictionary is an outcome.
    """
    id = game["id"]
    sport_key = game["sport_key"]
    sport_title = game["sport_title"]
    start_time = game["commence_time"]
    home_team = game["home_team"]
    away_team = game["away_team"]
    bm_dicts = _create_bm_dict_list(game)
    rows = []

    outcomes_list = _get_outcomes_list(bm_dicts)

    # Create a row for each outcome including match data and all bookmakers
    for outcome_team in outcomes_list:
        row = {
            "ID": id,
            "Sport Key": sport_key,
            "Sport Title": sport_title,
            "Start Time": start_time,
            "Match": f"{away_team} @ {home_team}",
            "Team": outcome_team,
        }

        # Add each bookmaker's odds for this outcome
        for bm in bm_dicts:
            for bm_name, odds_dict in bm.items():
                row[bm_name] = odds_dict.get(
                    outcome_team, None
                )  # None if outcome not found

        rows.append(row)

    return rows


def _get_json_response() -> Dict:
    """
    Helper function to get JSON response from The Odds API.

    Returns:
        Dict: JSON response from the API.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }
    try:
        response = requests.get(url, params=params)
        return response.json()
    except Exception as e:
        raise Exception(f"Error while fetching API response: {e}")


def fetch_odds(games_data: Optional[List[Dict]] = None) -> pd.DataFrame:
    """
    Fetches head-to-head (h2h) betting odds from The Odds API.

    Args:
        games_data (Optional[List[Dict]]): Pre-fetched game data for testing purposes. If None, data will be fetched from the API.

    Returns:
        pd.DataFrame: DataFrame with columns: match, league, start time, team, bookmaker, odds, last update.
                    Each row represents one team's odds from one bookmaker.
    """
    print("----------------------------------------------------")
    print(f"Fetching odds for sport: {SPORT}")

    if games_data is None:
        games_data = _get_json_response()

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

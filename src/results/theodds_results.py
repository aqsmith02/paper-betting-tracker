"""
theodds_results.py

The file fetches sports game results using The-Odds-API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""

import requests
import pandas as pd
import yaml
from datetime import datetime, timedelta, timezone
from typing import Any, List, Dict
from src.constants import PENDING_RESULTS, CONFIG_DIR

# Load config

config_path = CONFIG_DIR / "api_config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# API Keys
THE_ODDS_API_KEY = config["api"]["the_odds_api_key"]


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


def _parse_match_teams(match: str) -> List[str]:
    """
    Convert a match string to a list containing the individual teams.

    Args:
        match (str): Match string in format "Team1 @ Team2".

    Returns:
        List[str]: List containing individual team names [away_team, home_team].
    """
    return [t.strip() for t in match.split("@")]


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


def map_league_to_key(df: pd.DataFrame) -> List[str]:
    """
    Map league names in DataFrame to corresponding The-Odds-API sport keys.

    Args:
        df (pd.DataFrame): DataFrame containing a "League" column with league names.

    Returns:
        List[str]: List of unique sport keys corresponding to leagues found in the DataFrame.
    """
    league_to_key = {
        "CFL": "americanfootball_cfl",
        "NCAAF": "americanfootball_ncaaf",
        "NCAAF Championship Winner": "americanfootball_ncaaf_championship_winner",
        "NFL": "americanfootball_nfl",
        "NFL Preseason": "americanfootball_nfl_preseason",
        "NFL Super Bowl Winner": "americanfootball_nfl_super_bowl_winner",
        "AFL": "aussierules_afl",
        "KBO": "baseball_kbo",
        "MLB": "baseball_mlb",
        "MLB World Series Winner": "baseball_mlb_world_series_winner",
        "NBA Championship Winner": "basketball_nba_championship_winner",
        "NCAAB": "basketball_ncaab",
        "NCAAB Championship Winner": "basketball_ncaab_championship_winner",
        "WNBA": "basketball_wnba",
        "Boxing": "boxing_boxing",
        "International Twenty20": "cricket_international_t20",
        "Test Matches": "cricket_test_match",
        "Masters Tournament Winner": "golf_masters_tournament_winner",
        "NHL": "icehockey_nhl",
        "NHL Championship Winner": "icehockey_nhl_championship_winner",
        "PLL": "lacrosse_pll",
        "MMA": "mma_mixed_martial_arts",
        "NRL": "rugbyleague_nrl",
        "Primera División - Argentina": "soccer_argentina_primera_division",
        "Austrian Football Bundesliga": "soccer_austria_bundesliga",
        "Belgium First Div": "soccer_belgium_first_div",
        "Brazil Série A": "soccer_brazil_campeonato",
        "Brazil Série B": "soccer_brazil_serie_b",
        "Primera División - Chile": "soccer_chile_campeonato",
        "Super League - China": "soccer_china_superleague",
        "Copa Libertadores": "soccer_conmebol_copa_libertadores",
        "Copa Sudamericana": "soccer_conmebol_copa_sudamericana",
        "Denmark Superliga": "soccer_denmark_superliga",
        "Championship": "soccer_efl_champ",
        "EFL Cup": "soccer_england_efl_cup",
        "League 1": "soccer_england_league1",
        "League 2": "soccer_england_league2",
        "EPL": "soccer_epl",
        "FIFA World Cup Qualifiers - Europe": "soccer_fifa_world_cup_qualifiers_europe",
        "FIFA World Cup Winner": "soccer_fifa_world_cup_winner",
        "Veikkausliiga - Finland": "soccer_finland_veikkausliiga",
        "Ligue 1 - France": "soccer_france_ligue_one",
        "Ligue 2 - France": "soccer_france_ligue_two",
        "Bundesliga - Germany": "soccer_germany_bundesliga",
        "Bundesliga 2 - Germany": "soccer_germany_bundesliga2",
        "Super League - Greece": "soccer_greece_super_league",
        "Serie A - Italy": "soccer_italy_serie_a",
        "J League": "soccer_japan_j_league",
        "K League 1": "soccer_korea_kleague1",
        "League of Ireland": "soccer_league_of_ireland",
        "Liga MX": "soccer_mexico_ligamx",
        "Dutch Eredivisie": "soccer_netherlands_eredivisie",
        "Eliteserien - Norway": "soccer_norway_eliteserien",
        "Ekstraklasa - Poland": "soccer_poland_ekstraklasa",
        "La Liga - Spain": "soccer_spain_la_liga",
        "Premiership - Scotland": "soccer_spl",
        "Allsvenskan - Sweden": "soccer_sweden_allsvenskan",
        "Superettan - Sweden": "soccer_sweden_superettan",
        "Swiss Superleague": "soccer_switzerland_superleague",
        "Turkey Super League": "soccer_turkey_super_league",
        "UEFA Champions League Qualification": "soccer_uefa_champs_league_qualification",
        "MLS": "soccer_usa_mls",
    }

    key_list = df["League"].map(league_to_key)
    unique_keys = key_list.dropna().unique().tolist()
    return unique_keys
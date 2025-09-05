"""
theodds_results.py

The file fetches sports game results using The-Odds-API. Results are appended to DataFrames containing paper
bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables --------------------------------------------
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Any, List, Dict

sports= [
    "americanfootball_cfl",
    "americanfootball_ncaaf",
    "americanfootball_ncaaf_championship_winner",
    "americanfootball_nfl",
    "americanfootball_nfl_preseason",
    "americanfootball_nfl_super_bowl_winner",
    "aussierules_afl",
    "baseball_kbo",
    "baseball_mlb",
    "baseball_mlb_world_series_winner",
    "basketball_nba_championship_winner",
    "basketball_ncaab_championship_winner",
    "basketball_wnba",
    "boxing_boxing",
    "cricket_international_t20",
    "cricket_test_match",
    "golf_masters_tournament_winner",
    "icehockey_nhl",
    "icehockey_nhl_championship_winner",
    "lacrosse_pll",
    "mma_mixed_martial_arts",
    "rugbyleague_nrl",
    "soccer_argentina_primera_division",
    "soccer_austria_bundesliga",
    "soccer_belgium_first_div",
    "soccer_brazil_campeonato",
    "soccer_brazil_serie_b",
    "soccer_chile_campeonato",
    "soccer_china_superleague",
    "soccer_conmebol_copa_libertadores",
    "soccer_conmebol_copa_sudamericana",
    "soccer_denmark_superliga",
    "soccer_efl_champ",
    "soccer_england_efl_cup",
    "soccer_england_league1",
    "soccer_england_league2",
    "soccer_epl",
    "soccer_fifa_world_cup_qualifiers_europe",
    "soccer_fifa_world_cup_winner",
    "soccer_finland_veikkausliiga",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_greece_super_league",
    "soccer_italy_serie_a",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_league_of_ireland",
    "soccer_mexico_ligamx",
    "soccer_netherlands_eredivisie",
    "soccer_norway_eliteserien",
    "soccer_poland_ekstraklasa",
    "soccer_spain_la_liga",
    "soccer_spl",
    "soccer_sweden_allsvenskan",
    "soccer_sweden_superettan",
    "soccer_switzerland_superleague",
    "soccer_turkey_super_league",
    "soccer_uefa_champs_league_qualification",
    "soccer_usa_mls"
]

sports_with_results = [
    "americanfootball_cfl",
    "americanfootball_ncaaf",
    "americanfootball_nfl",
    "americanfootball_nfl_preseason",
    "americanfootball_ufl",
    "aussierules_afl",
    "baseball_mlb",
    "basketball_euroleague",
    "basketball_nba",
    "basketball_nba_preseason",
    "basketball_wnba",
    "basketball_ncaab",
    "icehockey_nhl",
    "rugbyleague_nrl",
    "soccer_argentina_primera_division",
    "soccer_australia_aleague",
    "soccer_austria_bundesliga",
    "soccer_belgium_first_div",
    "soccer_brazil_campeonato",
    "soccer_brazil_serie_b",
    "soccer_chile_campeonato",
    "soccer_china_superleague",
    "soccer_denmark_superliga",
    "soccer_efl_champ",
    "soccer_england_efl_cup",
    "soccer_england_league1",
    "soccer_england_league2",
    "soccer_epl",
    "soccer_fa_cup",
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_womens",
    "soccer_fifa_club_world_cup",
    "soccer_finland_veikkausliiga",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_germany_liga3",
    "soccer_greece_super_league",
    "soccer_italy_serie_a",
    "soccer_italy_serie_b",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_league_of_ireland",
    "soccer_mexico_ligamx",
    "soccer_netherlands_eredivisie",
    "soccer_norway_eliteserien",
    "soccer_poland_ekstraklasa",
    "soccer_portugal_primeira_liga",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_spl",
    "soccer_sweden_allsvenskan",
    "soccer_sweden_superettan",
    "soccer_switzerland_superleague",
    "soccer_turkey_super_league",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_champs_league",
    "soccer_uefa_champs_league_qualification",
    "soccer_uefa_europa_league",
    "soccer_uefa_european_championship",
    "soccer_uefa_euro_qualification",
    "soccer_uefa_nations_league",
    "soccer_conmebol_copa_america",
    "soccer_conmebol_copa_libertadores",
    "soccer_usa_mls"
]

API_KEY = "7ca177e18aa6a5230dddc27a238e3f73"


# ------------------------------------------ Formatting Helpers ---------------------------------------------
def _start_date(ts: Any) -> str:
    """
    Convert a timestamp / datetime-like / ISO string to "YYYY-MM-DD".
    
    Args:
        ts (Any): Timestamp to convert to date (adjusted 4 hours forward from UTC to EDT).
        
    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    dt = pd.to_datetime(ts) + pd.Timedelta(hours=4)
    return dt.strftime("%Y-%m-%d")


def _parse_match_teams(match: str) -> List[str]:
    """
    Convert a match string to a list containing the individual teams.
    
    Args:
        match (str): Match string in format "Team1 @ Team2" or "Team1 vs Team2".
        
    Returns:
        List[str]: List containing individual team names [away_team, home_team].
    """
    if "@" in match:
        teams = [t.strip() for t in match.split("@")]
    elif "vs" in match.lower():
        teams = [t.strip() for t in match.lower().split("vs")]
    else:
        teams = [match.strip()]
    return teams


# -------------------------------------- Time Since Start Filter --------------------------------------------
def _time_since_start(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Filter out games that started less than threshold hours ago.
    
    Args:
        df (pd.DataFrame): DataFrame containing game data with "Start Time" column.
        thresh (float): Threshold in hours - games starting less than this many hours ago are filtered out.
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only games that started more than thresh hours ago.
    """
    # Get the current time
    current_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    current_time_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")

    # Convert "Start Time" column to datetime objects
    df['Start Time'] = pd.to_datetime(df['Start Time'], format="%Y-%m-%d %H:%M:%S")

    # Create conditions for removal
    cutoff = current_time_obj - timedelta(hours=thresh)

    # Filter out games that started less than threshold hours ago
    mask = (df["Start Time"] <= cutoff)
    df = df[mask]

    return df


# ------------------------------------------- Results Fetcher -----------------------------------------------
def _get_scores_from_api(sports_key: str, days_from: int = 3) -> List[Dict]:
    """
    Fetch game outcomes from The-Odds-API for the last specified number of days.
    
    Args:
        sports_key (str): Sport key to fetch games for (maximum 3 days lookback).
        days_from (int): Number of days back to look for completed games.
        
    Returns:
        List[Dict]: List of completed game dictionaries from the API response.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sports_key}/scores/?daysFrom={days_from}&apiKey={API_KEY}"
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Error fetching scores from Odds API: {resp.status_code}")
        return []
    return resp.json()


# ------------------------------------------ Results Filter -------------------------------------------------
def _filter(scores: List[Dict], start_date: str, home_team: str, away_team: str) -> List[Dict]:
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
        game for game in scores
        if _start_date(game.get("commence_time")) == start_date and
           game.get("home_team") == home_team and
           game.get("away_team") == away_team
    ]


# --------------------------------------- Determine Winner of Game ------------------------------------------
def _get_winner(game: Dict, home: str, away: str) -> str:
    """
    Determine the game result by comparing scores from a completed game.
    
    Args:
        game (Dict): Game dictionary containing score and completion information.
        home (str): Home team name.
        away (str): Away team name.
        
    Returns:
        str: Winner team name, "Draw", "Pending", or "invalid" if scores cannot be parsed.
    """
    # Check if the game has completed
    if not game.get("completed"):
        return "Pending"
    
    # Pull the scores
    home_score = game.get("scores", [{}])[0].get("score", 0)
    away_score = game.get("scores", [{}])[1].get("score", 0)
    print(f"{home} (H) vs {away} (A): {home_score}-{away_score}")

    # Force scores to int type
    try:
        home_score = int(home_score)
        away_score = int(away_score)
    except (ValueError, TypeError):
        return "invalid"
    
    # Compare
    if home_score > away_score:
        return home
    elif away_score > home_score:
        return away
    else:
        return "Draw"


# --------------------------------------- Results Column Function -------------------------------------------
def get_finished_games_from_theodds(df: pd.DataFrame, sports_key: str) -> pd.DataFrame:
    """
    Add game results to DataFrame by fetching data from The-Odds-API.
    
    Args:
        df (pd.DataFrame): DataFrame containing betting data with columns "Match", "Start Time", and optionally "Result".
        sports_key (str): Sport key for The-Odds-API to specify which sport's results to fetch.
        
    Returns:
        pd.DataFrame: Updated DataFrame with "Result" column populated from API calls.
    """
    if "Result" not in df.columns:
        df["Result"] = "Not Found"

    # Get a list of the games from the past 3 days in the specified sport
    scores = _get_scores_from_api(sports_key)

    # Filter out games that started less than 12 hours ago
    indices = _time_since_start(df, 12).index.tolist()

    for i in indices:
        row = df.iloc[i]
        existing_result = row.get("Result")

        # Skip rows that already have a result other than "Not Found"
        if existing_result not in ["Not Found", "Pending", "API Error"]:
            continue

        # Note the necessary args for the filter() function
        start_date = _start_date(row["Start Time"])
        teams = _parse_match_teams(row["Match"])
        away_team, home_team = teams[0], teams[1]

        matches = _filter(scores, start_date, home_team, away_team)

        # With the scores list filtered to match the game at this row, find the result
        if matches:
            result = _get_winner(matches[0], home_team, away_team)
        else:
            result = "Not Found"

        df.at[i, "Result"] = result

    return df


# ---------------------------------------------- Key Finder -------------------------------------------------
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
        "MLS": "soccer_usa_mls"
    }

    key_list = df["League"].map(league_to_key)
    unique_keys = key_list.dropna().unique().tolist()
    return unique_keys


# -------------------------------------------- Main Pipeline ------------------------------------------------
def main() -> None:
    """
    Main pipeline for fetching and updating game results from The-Odds-API.
    
    Args:
        None
        
    Returns:
        None
    """
    input_csv = "master_avg_bets.csv"
    output_csv = "results.csv"
    df = pd.read_csv(input_csv)

    # Find keys
    keys = map_league_to_key(df)

    # Loop through keys
    for key in keys:
        df = get_finished_games_from_theodds(df, key)

    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
"""
fetch_odds.py

The file fetches sports betting odds using The-Odds-API. Pulls in odds from any designated sport,
region, or market. Organizes a DataFrame with odds to contain one row per outcome, with columns being
bookmakers and essential information.

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports ------------------------------------------ #
import requests
import pandas as pd
from datetime import datetime
from dateutil import parser
import pytz
from typing import List, Dict, Optional

# ---------------------------------------- Configuration ------------------------------------------ #

# API Configuration
API_KEY = "7ca177e18aa6a5230dddc27a238e3f73"
REGIONS = "us,uk,eu,au"
MARKETS = "h2h"
ODDS_FORMAT = "decimal"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Current sport to fetch (change this to switch sports)
CURRENT_SPORT = "upcoming"

# Available Sports Keys
SPORTS = {
    "upcoming": "upcoming",
    "kbo": "baseball_kbo", 
    "mlb": "baseball_mlb",
    "ncaa_baseball": "baseball_ncaa",
    "wnba": "basketball_wnba",
    "brazil_serie_a": "soccer_brazil_campeonato",
    "brazil_serie_b": "soccer_brazil_serie_b",
    "super_league_china": "soccer_china_superleague",
    "japan_league": "soccer_japan_j_league",
    "mls": "soccer_usa_mls",
    "cfl": "americanfootball_cfl",
    "aussie": "aussierules_afl",
    "npb": "baseball_npb",
    "boxing": "boxing_boxing",
    "cricket": "cricket_t20_blast",
    "lacrosse": "lacrosse_pll",
    "rugby": "rugbyleague_nrl",
    "mma": "mma_mixed_martial_arts",
    "euroleague": "soccer_uefa_european_championship",
    "finland": "soccer_finland_veikkausliiga",
    "nhl": "icehockey_nhl",
    "sweden_hockey": "icehockey_sweden_hockey_league",
    "mexico": "soccer_mexico_ligamx",
    "ireland": "soccer_league_of_ireland"
}

# Betting exchanges to exclude (they work differently than traditional bookmakers)
EXCHANGE_BLOCKLIST = {
    "Smarkets",
    "Betfair", 
    "Matchbook",
    "Betfair Sportsbook"
}


# -------------------------- Helper Functions ---------------------- #
def _convert_to_eastern_time(utc_time_str: str) -> str:
    """Convert UTC time string to Eastern time."""
    eastern = pytz.timezone("US/Eastern")
    utc_dt = datetime.fromisoformat(utc_time_str[:-1]).replace(tzinfo=pytz.utc)
    local_dt = utc_dt.astimezone(eastern)
    return local_dt.strftime(DATE_FORMAT)


def _convert_iso_to_eastern_time(iso_time_str: str) -> str:
    """Convert ISO time string (with timezone info) to Eastern time."""
    eastern = pytz.timezone("US/Eastern")
    utc_dt = parser.isoparse(iso_time_str)
    local_dt = utc_dt.astimezone(eastern)
    return local_dt.strftime(DATE_FORMAT)


def _is_exchange(bookmaker_name: str) -> bool:
    """Check if bookmaker is a betting exchange (to be excluded)."""
    return any(exchange.lower() in bookmaker_name.lower() for exchange in EXCHANGE_BLOCKLIST)


# -------------------------- Main Functions ---------------------- #
def fetch_odds(sport_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetches head-to-head (h2h) betting odds from The Odds API.
    
    Args:
        sport_key: Optional sport key to fetch. If None, uses CURRENT_SPORT.
    
    Returns:
        pd.DataFrame: Each row represents one team's odds from one bookmaker.
        Columns: match, league, start time, team, bookmaker, odds, last update
    """
    sport = sport_key or CURRENT_SPORT
    
    # Build API request
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGIONS, 
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT
    }
    
    print(f"Fetching odds for sport: {sport}")
    response = requests.get(url, params=params)
    
    # Log API usage
    print("Requests Remaining:", response.headers.get("x-requests-remaining"))
    print("Requests Used:", response.headers.get("x-requests-used"))
    
    if response.status_code != 200:
        print(f"API request failed: {response.status_code} - {response.text}")
        return pd.DataFrame()
    
    games_data = response.json()
    print(f"Retrieved {len(games_data)} games")
    
    # Process each game into rows
    rows = []
    for game in games_data:
        try:
            rows.extend(_process_game(game))
        except Exception as e:
            print(f"Error processing game: {e}")
            continue
    
    return pd.DataFrame(rows)


def _process_game(game: Dict) -> List[Dict]:
    """Process a single game from the API response into multiple rows: each bookmaker,
       market (should only be H2H), and outcome will be a separate row."""
    home_team = game["home_team"]
    away_team = game["away_team"]
    league = game["sport_title"]
    start_time = _convert_to_eastern_time(game["commence_time"])
    
    rows = []
    
    for bookmaker in game.get("bookmakers", []):
        if _is_exchange(bookmaker["title"]):
            continue
            
        # Convert bookmaker's last update time
        last_update_str = _convert_iso_to_eastern_time(bookmaker["last_update"])
        
        # Get h2h market outcomes (should only be one market since we specify h2h in API call)
        markets = bookmaker.get("markets", [])
        if markets:
            h2h_market = markets[0]  # Should be h2h market since that's all we requested
            for outcome in h2h_market.get("outcomes", []):
                rows.append({
                    "match": f"{away_team} @ {home_team}",
                    "league": league,
                    "start time": start_time,
                    "team": outcome["name"],
                    "bookmaker": bookmaker["title"],
                    "odds": outcome["price"],
                    "last update": last_update_str
                })
    
    return rows


def organize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the odds data so each row is an outcome with all bookmaker odds as columns.
    
    Args:
        df: DataFrame from fetch_odds() with one row per bookmaker per outcome
        
    Returns:
        pd.DataFrame: Each row is one team outcome, columns are bookmakers plus metadata
    """
    if df.empty:
        return pd.DataFrame()
    
    # Get unique bookmakers for columns
    bookmakers = sorted(df["bookmaker"].unique())
    
    # Prepare columns for the organized dataframe
    base_columns = ["match", "league", "start time", "team", "last update", "result"]
    all_columns = base_columns + bookmakers + ["best odds", "best bookmaker"]
    
    rows = []
    
    # Group by match and team to create one row per outcome
    for (match, team), group in df.groupby(["match", "team"]):
        # Get static information (same for all bookmakers)
        first_row = group.iloc[0]
        
        # Create bookmaker odds dictionary
        bookmaker_odds = {}
        best_odds = None
        best_bookmaker = None
        
        for _, row in group.iterrows():
            bookmaker = row["bookmaker"]
            odds = row["odds"]
            bookmaker_odds[bookmaker] = odds
            
            # Track best odds
            if best_odds is None or odds > best_odds:
                best_odds = odds
                best_bookmaker = bookmaker
        
        # Build the final row
        organized_row = {
            "match": match,
            "league": first_row["league"], 
            "start time": first_row["start time"],
            "team": team,
            "last update": first_row["last update"],
            "result": "Not Found",
            **{bm: bookmaker_odds.get(bm) for bm in bookmakers},
            "best odds": best_odds,
            "best bookmaker": best_bookmaker
        }
        
        rows.append(organized_row)
    
    return pd.DataFrame(rows, columns=all_columns)


# ------------------------------------------ Main Pipeline ------------------------------------------ #
if __name__ == "__main__":
    # Fetch and organize odds data
    print("Starting odds fetch...")
    raw_odds = fetch_odds()
    
    if not raw_odds.empty:
        print("Organizing odds data...")
        organized_odds = organize(raw_odds)
        
        # Save to CSV
        output_file = "odds.csv"
        organized_odds.to_csv(output_file, index=False)
        print(f"Saved {len(organized_odds)} outcomes to {output_file}")
    else:
        print("No odds data retrieved")
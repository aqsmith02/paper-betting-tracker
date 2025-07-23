"""
results.py

The file fetches sports game results using both The-Odds-API and TheSportsDB API. Results are appended to 
DataFrames containing paper bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables --------------------------------------------
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from theodds_results import get_finished_games, map_league_to_key
from sportsdb_results import get_finished_games_from_thesportsdb

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


# ------------------------------------------- Main Pipeline -----------------------------------------------
if __name__ == "__main__":
    # Create file paths
    files = [
        "master_avg_bets.csv",
        "master_avg_full.csv",
        "master_mod_zscore_bets.csv",
        "master_mod_zscore_full.csv",
        "master_pin_bets.csv",
        "master_pin_full.csv",
        "master_zscore_bets.csv",
        "master_zscore_full.csv",
    ]

    # Loop through files
    for file in files:
        print(f"Starting {file}.")

        # Load in file -------------------------------------------------------------------------------------
        df = pd.read_csv(file)
        rows_to_search = df[df["Result"].isin(["Not Found", "Pending"])]

        # Pull results from The-Odds-Api --------------------------------------------------------------------
        keys = map_league_to_key(rows_to_search)

        # Loop through keys
        for key in keys:
            df = get_finished_games(df, key)

        print("\nCompleted The-Odds-API pull, now pulling from TheSportsDB.\n")

        # Pull remaining results from SportsDB -------------------------------------------------------------
        df = get_finished_games_from_thesportsdb(df)

        # Remove rows older than 4 days old ----------------------------------------------------------------
        # Get the current time
        current_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
        current_time_obj = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")

        # Convert "Start Time" column to datetime objects
        df['Start Time'] = pd.to_datetime(df['Start Time'], format="%Y-%m-%d %H:%M:%S")

        # Create conditions for removal
        cutoff = current_time_obj - timedelta(days=3)
        unwanted_results = ["Pending", "Not Found", "API Error"]

        # Filter
        mask = ~((df["Start Time"] < cutoff) & (df["Result"].isin(unwanted_results)))
        df = df[mask]

        # Write to .csv
        df.to_csv(file,index=False)
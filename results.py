"""
results.py

The file fetches sports game results using both The-Odds-API and TheSportsDB API. Results are appended to 
DataFrames containing paper bets in a column called "Result".

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports and Variables --------------------------------------------
import pandas as pd
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from theodds_results import get_finished_games_from_theodds, map_league_to_key, sports_with_results
from sportsdb_results import get_finished_games_from_thesportsdb


# ------------------------------------------- Main Pipeline -----------------------------------------------
if __name__ == "__main__":
    # Create file paths
    bet_files = [
        "data/master_avg_bets.csv",
        "data/master_mod_zscore_bets.csv",
        "data/master_pin_bets.csv",
        "data/master_zscore_bets.csv",
    ]

    full_files = [
        "data/master_avg_full.csv",
        "data/master_mod_zscore_full.csv",
        "data/master_pin_full.csv",
        "data/master_zscore_full.csv",
    ]

    # Loop through files
    for i in range(len(bet_files)):
        bet_file = bet_files[i]
        full_file = full_files[i]
        print(f"\nStarting {bet_file} and {full_file}.\n")

        # Load in file -------------------------------------------------------------------------------------
        df = pd.read_csv(bet_file)
        full_df = pd.read_csv(full_file)

        # Filter out rows that do not need to be checked
        rows_to_search = df[df["Result"].isin(["Not Found", "Pending", "API Error"])]
        keys = map_league_to_key(rows_to_search)

        # Pull results from The-Odds-Api --------------------------------------------------------------------
        # Loop through keys
        for key in keys:
            df = get_finished_games_from_theodds(df, key)

        print("\nCompleted The-Odds-API pull, now pulling from TheSportsDB.\n")

        # Pull remaining results from SportsDB -------------------------------------------------------------
        df = get_finished_games_from_thesportsdb(df)

        full_df["Result"] = df["Result"]

        # Remove rows older than 3 days old that do not have a valid result --------------------------------
        # Get the current time
        current_time = datetime.now(ZoneInfo("America/New_York"))

        # Convert "Start Time" column to datetime objects, store original "Start Time" column
        temp_start_time = df["Start Time"].copy()
        df['Start Time'] = pd.to_datetime(df['Start Time'], format="%Y-%m-%d %H:%M:%S")
        df["Start Time"] = df["Start Time"].dt.tz_localize("America/New_York")

        # Create conditions for removal
        cutoff = current_time - timedelta(days=3)
        unwanted_results = ["Pending", "Not Found", "API Error"]

        # Filter
        mask = ~((df["Start Time"] < cutoff) & (df["Result"].isin(unwanted_results)))
        df = df[mask]
        full_df = full_df[mask]

        # Replace "Start Time" column with the original state
        df["Start Time"] = temp_start_time

        # Write to .csv
        df.to_csv(bet_file,index=False)
        full_df.to_csv(full_file,index=False)

        time.sleep(60)
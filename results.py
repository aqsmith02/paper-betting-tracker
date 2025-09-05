"""
results.py

Fetches sports game results using both The-Odds-API and TheSportsDB API.
Results are appended to DataFrames containing paper bets in a "Result" column.

Author: Andrew Smith
Date: July 2025
"""

import pandas as pd
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo
from theodds_results import get_finished_games_from_theodds, map_league_to_key
from sportsdb_results import get_finished_games_from_thesportsdb


# Configuration
TIMEZONE = ZoneInfo("America/New_York")
DAYS_CUTOFF = 3
SLEEP_DURATION = 60
DATA_DIR = Path("data")

# File configurations
FILE_CONFIGS = [
    ("master_avg_bets.csv", "master_avg_full.csv"),
    ("master_mod_zscore_bets.csv", "master_mod_zscore_full.csv"),
    ("master_pin_bets.csv", "master_pin_full.csv"),
    ("master_zscore_bets.csv", "master_zscore_full.csv"),
]

PENDING_RESULTS = ["Not Found", "Pending", "API Error"]


def load_dataframes(bet_file: Path, full_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load bet and full DataFrames from CSV files."""
    try:
        bet_df = pd.read_csv(bet_file)
        full_df = pd.read_csv(full_file)
        return bet_df, full_df
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        print(f"Empty data file encountered: {e}")
        raise


def filter_rows_to_search(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to only include rows that need result checking."""
    return df[df["Result"].isin(PENDING_RESULTS)]


def fetch_results_from_theodds(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch results from The-Odds-API for all relevant leagues."""
    rows_to_search = filter_rows_to_search(df)
    
    if rows_to_search.empty:
        print("No rows need checking from The-Odds-API")
        return df
    
    keys = map_league_to_key(rows_to_search)
    
    for key in keys:
        df = get_finished_games_from_theodds(df, key)
    
    print("Completed The-Odds-API pull")
    return df


def fetch_results_from_sportsdb(df: pd.DataFrame) -> pd.DataFrame:
    """Fetch remaining results from TheSportsDB API."""
    print("Pulling remaining results from TheSportsDB")
    return get_finished_games_from_thesportsdb(df)


def clean_old_pending_results(df: pd.DataFrame, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows older than DAYS_CUTOFF that still have pending results."""
    current_time = datetime.now(TIMEZONE)
    cutoff_time = current_time - timedelta(days=DAYS_CUTOFF)
    
    # Store original start time column
    original_start_time = df["Start Time"].copy()
    
    try:
        # Convert to datetime with timezone
        df_temp = df.copy()
        df_temp["Start Time"] = pd.to_datetime(df_temp["Start Time"], format="%Y-%m-%d %H:%M:%S")
        df_temp["Start Time"] = df_temp["Start Time"].dt.tz_localize(TIMEZONE)
        
        # Create filter mask - keep rows that are either recent OR have valid results
        mask = ~((df_temp["Start Time"] < cutoff_time) & (df_temp["Result"].isin(PENDING_RESULTS)))
        
        # Apply filter to both DataFrames
        filtered_df = df[mask].copy()
        filtered_full_df = full_df[mask].copy()
        
        # Restore original start time format
        filtered_df["Start Time"] = original_start_time[mask].values
        
        removed_count = len(df) - len(filtered_df)
        if removed_count > 0:
            print(f"Removed {removed_count} old rows with pending results")
        
        return filtered_df, filtered_full_df
        
    except Exception as e:
        print(f"Error cleaning old results: {e}")
        # Return original DataFrames if cleaning fails
        return df, full_df


def save_dataframes(df: pd.DataFrame, full_df: pd.DataFrame, bet_file: Path, full_file: Path) -> None:
    """Save DataFrames to CSV files."""
    try:
        df.to_csv(bet_file, index=False)
        full_df.to_csv(full_file, index=False)
        print(f"Saved results to {bet_file.name} and {full_file.name}")
    except Exception as e:
        print(f"Error saving files: {e}")
        raise


def process_file_pair(bet_filename: str, full_filename: str) -> None:
    """Process a single pair of bet and full files."""
    bet_file = DATA_DIR / bet_filename
    full_file = DATA_DIR / full_filename
    
    print(f"\nProcessing {bet_filename} and {full_filename}")
    
    try:
        # Load data
        bet_df, full_df = load_dataframes(bet_file, full_file)
        
        # Fetch results from APIs
        bet_df = fetch_results_from_theodds(bet_df)
        bet_df = fetch_results_from_sportsdb(bet_df)
        
        # Update full DataFrame with results
        full_df["Result"] = bet_df["Result"]
        
        # Clean old pending results
        bet_df, full_df = clean_old_pending_results(bet_df, full_df)
        
        # Save results
        save_dataframes(bet_df, full_df, bet_file, full_file)
        
    except Exception as e:
        print(f"Error processing {bet_filename}: {e}")
        raise


def main() -> None:
    """Main pipeline for processing all file pairs."""
    print("Starting sports results pipeline")
    
    for i, (bet_filename, full_filename) in enumerate(FILE_CONFIGS):
        try:
            process_file_pair(bet_filename, full_filename)
            
            # Sleep between files (except after the last one)
            if i < len(FILE_CONFIGS) - 1:
                print(f"Sleeping for {SLEEP_DURATION} seconds...")
                time.sleep(SLEEP_DURATION)
                
        except Exception as e:
            print(f"Failed to process {bet_filename}: {e}")
            # Continue with next file pair instead of stopping entire pipeline
            continue
    
    print("\nCompleted sports results pipeline")


if __name__ == "__main__":
    main()
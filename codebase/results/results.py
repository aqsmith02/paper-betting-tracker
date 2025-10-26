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
from typing import Tuple
from .theodds_results import get_finished_games_from_theodds, map_league_to_key
from .sportsdb_results import get_finished_games_from_thesportsdb
from .results_configs import (
    PENDING_RESULTS,
    TIMEZONE_UTC,
    TIMEZONE_EST,
    DAYS_CUTOFF,
    FILE_CONFIGS,
    SLEEP_DURATION,
)
from codebase.constants import DATA_DIR


def filter_rows_to_search(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows that need result checking.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with "Result" column.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with pending results.
    """
    return df[df["Result"].isin(PENDING_RESULTS)]


def fetch_results_from_theodds(df: pd.DataFrame, start_time_col: str = "Start Time UTC") -> pd.DataFrame:
    """
    Fetch results from The-Odds-API for all relevant leagues.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with "League" and "Result" columns.
        start_time_col (str): Column name for start time (default "Start Time UTC").

    Returns:
        pd.DataFrame: Updated DataFrame with results fetched from The-Odds-API.
    """
    rows_to_search = filter_rows_to_search(df)

    if rows_to_search.empty:
        print("No rows need checking from The-Odds-API")
        return df

    keys = map_league_to_key(rows_to_search)

    for key in keys:
        df = get_finished_games_from_theodds(df, key, start_time_col)

    print("Completed The-Odds-API pull")
    return df


def fetch_results_from_sportsdb(df: pd.DataFrame, start_time_col: str = "Start Time UTC") -> pd.DataFrame:
    """
    Fetch remaining results from TheSportsDB API.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with pending results.
        start_time_col (str): Column name for start time (default "Start Time UTC").

    Returns:
        pd.DataFrame: Updated DataFrame with additional results fetched from TheSportsDB API.
    """
    df = get_finished_games_from_thesportsdb(df, start_time_col)
    print("Completed TheSportsDB pull")
    return df


def clean_old_pending_results(
    df: pd.DataFrame, full_df: pd.DataFrame, start_time_col: str = "Start Time UTC"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows older than DAYS_CUTOFF that still have pending results.

    Args:
        df (pd.DataFrame): Betting summary DataFrame to clean.
        full_df (pd.DataFrame): Full betting data DataFrame to clean.
        start_time_col (str): Column name for start time (default "Start Time UTC").

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of cleaned DataFrames (filtered_df, filtered_full_df).
    """
    current_time = datetime.now(TIMEZONE_EST)
    cutoff_time = current_time - timedelta(days=DAYS_CUTOFF)

    # Store original start time column
    original_start_time = df[start_time_col].copy()

    # Convert to datetime with timezone
    df_temp = df.copy()
    df_temp[start_time_col] = pd.to_datetime(df_temp[start_time_col])
    
    # Localize to timezone if naive
    if df_temp[start_time_col].dt.tz is None:
        df_temp[start_time_col] = df_temp[start_time_col].dt.tz_localize(TIMEZONE_UTC)

    # Create filter mask - keep rows that are either recent OR have valid results
    mask = ~(
        (df_temp[start_time_col] < cutoff_time)
        & (df_temp["Result"].isin(PENDING_RESULTS))
    )

    # Apply filter to both DataFrames
    filtered_df = df[mask].copy()
    filtered_full_df = full_df[mask].copy()

    # Restore original start time format
    filtered_df[start_time_col] = original_start_time[mask].values

    removed_count = len(df) - len(filtered_df)
    if removed_count > 0:
        print(f"Removed {removed_count} old rows with pending results")

    return filtered_df, filtered_full_df


def process_file_pair(bet_filename: str, full_filename: str, start_time_col: str = "Start Time UTC") -> None:
    """
    Process a single pair of bet and full files through the complete results pipeline.

    Args:
        bet_filename (str): Name of the betting summary CSV file.
        full_filename (str): Name of the full betting data CSV file.
        start_time_col (str): Column name for start time (default "Start Time UTC").

    Returns:
        None
    """
    bet_file = DATA_DIR / bet_filename
    full_file = DATA_DIR / full_filename

    print(f"\nProcessing {bet_filename} and {full_filename}")

    # Load data
    bet_df = pd.read_csv(bet_file)
    full_df = pd.read_csv(full_file)

    # Fetch results from APIs
    bet_df = fetch_results_from_theodds(bet_df, start_time_col)
    bet_df = fetch_results_from_sportsdb(bet_df, start_time_col)

    # Update full DataFrame with results
    full_df["Result"] = bet_df["Result"]

    # Clean old pending results
    bet_df, full_df = clean_old_pending_results(bet_df, full_df, start_time_col)

    # Save results
    bet_df.to_csv(bet_file, index=False)
    full_df.to_csv(full_file, index=False)


def main(start_time_col: str = "Start Time UTC") -> None:
    """
    Main pipeline for processing all file pairs to fetch and update sports results.

    Args:
        start_time_col (str): Column name for start time (default "Start Time UTC").

    Returns:
        None
    """
    print("Starting sports results pipeline")

    for i, (bet_filename, full_filename) in enumerate(FILE_CONFIGS):
        try:
            process_file_pair(bet_filename, full_filename, start_time_col)

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
"""
results.py

Fetches sports game results using both The-Odds-API and TheSportsDB API.
Results are appended to DataFrames containing paper bets in a "Result" column.

Author: Andrew Smith
Date: July 2025
"""

import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from typing import Tuple
from src.results.theodds_results import get_finished_games_from_theodds
from src.results.sportsdb_results import get_finished_games_from_thesportsdb
from src.constants import (
    PENDING_RESULTS,
    DAYS_CUTOFF,
    FILE_NAMES,
    SLEEP_DURATION,
    DATA_DIR,
)
from config.results_config import map_league_to_key


def fetch_results_from_theodds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch results from The-Odds-API for all relevant leagues.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with "League" and "Result" columns.

    Returns:
        pd.DataFrame: Updated DataFrame with results fetched from The-Odds-API.
    """
    keys = map_league_to_key(df)

    for key in keys:
        df = get_finished_games_from_theodds(df, key)

    print("Completed The-Odds-API pull")
    return df


def fetch_results_from_sportsdb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch remaining results from TheSportsDB API.

    Args:
        df (pd.DataFrame): DataFrame containing betting data with pending results.

    Returns:
        pd.DataFrame: Updated DataFrame with additional results fetched from TheSportsDB API.
    """
    df = get_finished_games_from_thesportsdb(df)
    print("Completed TheSportsDB pull")
    return df


def clean_old_pending_results(
    df: pd.DataFrame, full_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove rows older than DAYS_CUTOFF that still have pending results.

    Args:
        df (pd.DataFrame): Betting summary DataFrame to clean.
        full_df (pd.DataFrame): Full betting data DataFrame to clean.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Tuple of cleaned DataFrames (filtered_df, filtered_full_df).
    """
    current_time = datetime.now(timezone.utc)
    cutoff_time = current_time - timedelta(days=DAYS_CUTOFF)

    # Store original start time column
    original_start_time = df["Start Time"].copy()

    # Convert to datetime
    df_temp = df.copy()
    df_temp["Start Time"] = pd.to_datetime(df_temp["Start Time"])

    # Create filter mask - keep rows that are either recent OR have valid results
    mask = ~(
        (df_temp["Start Time"] < cutoff_time)
        & (df_temp["Result"].isin(PENDING_RESULTS))
    )

    # Apply filter to both DataFrames
    filtered_df = df[mask].copy()
    filtered_full_df = full_df[mask].copy()

    # Restore original start time format
    filtered_df["Start Time"] = original_start_time[mask].values

    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} old rows with pending results")

    return filtered_df, filtered_full_df


def process_files(bet_filename: str, full_filename: str) -> None:
    """
    Process files through the complete results pipeline.

    Args:
        bet_filename (str): Name of the betting summary CSV file.
        full_filename (str): Name of the full betting data CSV file.

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
    bet_df = fetch_results_from_theodds(bet_df)
    bet_df = fetch_results_from_sportsdb(bet_df)

    # Update full DataFrame with results
    full_df["Result"] = bet_df["Result"]

    # Clean old pending results
    bet_df, full_df = clean_old_pending_results(bet_df, full_df)

    # Save results
    bet_df.to_csv(bet_file, index=False)
    full_df.to_csv(full_file, index=False)


def main() -> None:
    """
    Main pipeline for processing all file pairs to fetch and update sports results.

    Args:
        None

    Returns:
        None
    """
    print("Starting sports results pipeline")

    for i, (bet_filename, full_filename) in enumerate(FILE_NAMES):
        try:
            process_files(bet_filename, full_filename)

            # Sleep between files (except after the last one)
            if i < len(FILE_NAMES) - 1:
                print(f"Sleeping for {SLEEP_DURATION} seconds...")
                time.sleep(SLEEP_DURATION)

        except Exception as e:
            print(f"Failed to process {bet_filename}: {e}")
            # Continue with next file pair instead of stopping entire pipeline
            continue

    print("\nCompleted sports results pipeline")


if __name__ == "__main__":
    main()
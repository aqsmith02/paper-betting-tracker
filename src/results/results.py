"""
results.py

Fetches sports game results using both The-Odds-API and TheSportsDB API.
Results are appended to DataFrames containing paper bets in a "Result" column.

Author: Andrew Smith
Date: July 2025
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Tuple

import pandas as pd

from src.constants import (
    DIRECTORIES,
    DAYS_CUTOFF,
    FILE_NAMES,
    PENDING_RESULTS,
    SLEEP_DURATION,
)
from src.results.theodds_results import get_finished_games_from_theodds
from src.results.thesportsdb_results import get_finished_games_from_thesportsdb


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
    if df.empty:
        print(f"\n")
        print(f"Removed 0 old rows with pending results")
        return df, full_df

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
    print(f"\n")
    print(f"Removed {removed_count} old rows with pending results")

    return filtered_df, filtered_full_df


def process_files(bet_filename: str, full_filename: str, dir: str) -> None:
    """
    Process files through the complete results pipeline.

    Args:
        bet_filename (str): Name of the betting summary CSV file.
        full_filename (str): Name of the full betting data CSV file.
        dir (str): The directory path where the files are located.
    Returns:
        None
    """
    bet_file = dir / bet_filename
    full_file = dir / full_filename

    print("----------------------------------------------------")
    print(f"Processing {bet_filename} and {full_filename}")

    # Load data
    bet_df = pd.read_csv(bet_file)
    full_df = pd.read_csv(full_file)

    # Fetch results from APIs
    bet_df = get_finished_games_from_theodds(bet_df)
    bet_df = get_finished_games_from_thesportsdb(bet_df)

    # Update full DataFrame with results
    full_df["Result"] = bet_df["Result"]

    # Clean old pending results
    bet_df, full_df = clean_old_pending_results(bet_df, full_df)

    # Save results
    bet_df.to_csv(bet_file, index=False)
    full_df.to_csv(full_file, index=False)

    print(f"Done processing {bet_filename} and {full_filename}")
    print("----------------------------------------------------")


def main() -> None:
    """
    Main pipeline for processing all file pairs to fetch and update sports results.

    Args:
        None

    Returns:
        None
    """
    print("----------------------------------------------------")
    print("Starting sports results pipeline")

    for i, (bet_filename, full_filename) in enumerate(FILE_NAMES):
        try:
            start_time = time.perf_counter()  # Start timing
            process_files(bet_filename, full_filename, DIRECTORIES[i])
            elapsed = time.perf_counter() - start_time  # Calculate elapsed time
            remaining_sleep = SLEEP_DURATION - elapsed

            # Sleep only if there is remaining time and it's not the last file
            if i < len(FILE_NAMES) - 1 and remaining_sleep > 0:
                print(
                    f"Processing took {elapsed:.2f}s, sleeping for {remaining_sleep:.2f}s..."
                )
                time.sleep(remaining_sleep)
            elif i < len(FILE_NAMES) - 1:
                print(f"Processing took {elapsed:.2f}s, no need to sleep.")

        except Exception as e:
            print(f"Failed to process {bet_filename}: {e}")
            print("----------------------------------------------------")
            # Continue with next file pair instead of stopping entire pipeline
            continue

    print("Completed sports results pipeline")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()

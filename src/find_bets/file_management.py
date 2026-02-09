"""
file_management.py

Logic for appending bet summaries to existing bet .csv files.

Author: Andrew Smith
Date: July 2025
"""

from typing import Any, List

import pandas as pd
import requests
from datetime import datetime

from src.constants import DATE_FORMAT, INSERT_BEFORE_COLUMN


def _start_date_from_timestamp(timestamp: Any) -> str:
    """
    Convert a timestamp to YYYY-MM-DD format.

    Args:
        timestamp (Any): Timestamp value to convert (can be string, datetime, etc.).

    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    return pd.to_datetime(timestamp).strftime(DATE_FORMAT)


def _filter_best_bets_only(summary_df: pd.DataFrame, score_column: str) -> pd.DataFrame:
    """
    Save only the best bet per match (highest scoring bet).

    Args:
        summary_df (pd.DataFrame): Summary DataFrame containing all potential bets.
        score_column (str): Column name to use for sorting bet values.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only the best bet per match.
    """
    if summary_df.empty:
        return pd.DataFrame()

    best_bets = summary_df.sort_values(score_column, ascending=False).drop_duplicates(
        subset=["Match", "Start Time"], keep="first"
    )

    return best_bets


def _remove_duplicates(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify non-duplicate rows in new DataFrame based on (Match, Start Date) key.
    Use Start Date to avoid issues with exact timestamp mismatches from API.

    Args:
        existing_df (pd.DataFrame): Existing data from file.
        new_df (pd.DataFrame): New data to be added.

    Returns:
        pd.DataFrame: DataFrame containing only non-duplicate rows from new_df.
    """
    # If no existing data, all new data is unique
    if existing_df.empty:
        return new_df

    # Add date keys
    existing_df = existing_df.copy()
    new_df = new_df.copy()

    existing_df["Start Date"] = existing_df["Start Time"].apply(
        _start_date_from_timestamp
    )
    new_df["Start Date"] = new_df["Start Time"].apply(_start_date_from_timestamp)

    # Use merge to find duplicates
    existing_df["Exists"] = True
    merged = new_df.merge(
        existing_df[["Match", "Start Date", "Exists"]],
        on=["Match", "Start Date"],
        how="left",
    )

    # Keep only rows that didn't match (NaN in Exists column)
    # Use the merged DataFrame directly since it contains all the original new_df data
    result = merged[merged["Exists"].isna()].drop(columns=["Exists"])

    return result


def _notify_user_of_new_bets(new_bets_df: pd.DataFrame) -> None:
    """
    Send a Discord notification about new bets found.

    Args:
        new_bets_df (pd.DataFrame): DataFrame containing new bets identified.
        filename (str): Name of the file for which new bets were found.

    Returns:
        None
    """
    if new_bets_df.empty:
        return

    # Send Discord notification for each bet
    for _, bet in new_bets_df.iterrows():
        # Full Kelly = (bp - q) / b, where b = decimal_odds - 1, p = fair_prob, q = 1 - p
        b = bet["Best Odds"] - 1
        p = 1/bet["Fair Odds Average"]
        q = 1 - p
        kelly = min(((b * p - q) / b)/2, 0.025)  # Cap at 2.5% to avoid overbetting
        
        embed = {
            "title": f"New Bet Found",
            "color": 0x00ff00,  # Green
            "fields": [
                {"name": "Match", "value": bet['Match'], "inline": False},
                {"name": "Team", "value": bet['Team'], "inline": True},
                {"name": "Bookmaker", "value": bet.get('Best Bookmaker', 'N/A'), "inline": True},
                {"name": "Odds", "value": str(bet.get('Best Odds', 'N/A')), "inline": True},
                {"name": "EV%", "value": f"{bet.get('EV%', 'N/A')}", "inline": True},
                {"name": "Bet Size", "value": f"{kelly:.2%}", "inline": True},
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": f"Total bets found: {len(new_bets_df)}"}
        }
        
        data = {"embeds": [embed]}
        
        try:
            from config.discord_config import DISCORD_WEBHOOK
            response = requests.post(DISCORD_WEBHOOK, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send Discord notification: {e}")


def _align_column_schemas(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> List[str]:
    """
    Create unified column schema for existing and new data.

    Args:
        existing_df (pd.DataFrame): Existing data from file.
        new_df (pd.DataFrame): New data to be added.

    Returns:
        List[str]: Unified list of column names in proper order.
    """
    # If either DataFrame is empty, return columns from the other
    if existing_df.empty:
        return list(new_df.columns)
    if new_df.empty:
        return list(existing_df.columns)

    existing_columns = list(existing_df.columns)
    new_columns = [c for c in new_df.columns if c not in existing_df.columns]

    insert_before = INSERT_BEFORE_COLUMN
    idx = existing_columns.index(insert_before)

    final_columns = existing_columns[:idx] + new_columns + existing_columns[idx:]

    return final_columns


def save_betting_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    filename: str,
    score_column: str,
    print_bets: bool = False,
) -> None:
    """
    Save complete betting data for bets identified in filtered summary.

    Args:
        source_df (pd.DataFrame): Full DataFrame containing all betting analysis data.
        filtered_summary_df (pd.DataFrame): Filtered summary containing only best bets.
        filename (str): Path to save the full betting data.

    Returns:
        None
    """
    new_df = new_df.copy()
    existing_df = existing_df.copy()

    # If no new data, nothing to do
    if new_df.empty:
        return

    # Filter new_df to only include best bets
    filtered_new_df = _filter_best_bets_only(new_df, score_column)

    # Remove duplicates of new_df based on existing_df
    unique_new_df = _remove_duplicates(existing_df, filtered_new_df)

    if print_bets:
        if filename == "nc_avg_minimal.csv":
            _notify_user_of_new_bets(unique_new_df, filename)

        pd.set_option("display.max_rows", None)
        print("----------------------------------------------------")
        if len(unique_new_df) == 1:
            print(f"{len(unique_new_df)} bet found for {filename}:")
        else:
            print(f"{len(unique_new_df)} bets found for {filename}:")
        print(f"{unique_new_df[['Match', 'Team']]}")
        pd.reset_option("display.max_rows")
        print("----------------------------------------------------")

    # Get list of merged column schemas
    column_schema = _align_column_schemas(existing_df, unique_new_df)

    # Reindex both DataFrames to the unified schema
    existing_aligned = existing_df.reindex(columns=column_schema)
    new_aligned = unique_new_df.reindex(columns=column_schema)

    # Combine and save
    combined = pd.concat([existing_aligned, new_aligned], ignore_index=True)
    combined.to_csv(filename, index=False)

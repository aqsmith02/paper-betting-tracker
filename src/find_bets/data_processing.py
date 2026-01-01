"""
data_processing.py

Cleans and validates data from fetch_odds.py.

Author: Andrew Smith
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from src.constants import (
    MIN_BOOKMAKERS,
    MAX_ODDS,
    TARGET_BMS,
    NC_BMS,
)


def _find_bookmaker_columns(
    df: pd.DataFrame, exclude_columns: Optional[List[str]] = None
) -> List[str]:
    """
    Find columns that contain bookmaker odds (numeric columns, excluding metadata).

    Args:
        df (pd.DataFrame): DataFrame to search for bookmaker columns.
        exclude_columns (Optional[List[str]]): Additional columns to exclude from search.

    Returns:
        List[str]: List of column names that contain bookmaker odds.
    """
    # Exclude any columns that will be an int or float and are not bookmakers
    excluded = {"Best Odds", "Start Time", "Outcomes", "Event ID"}
    if exclude_columns:
        excluded.update(exclude_columns)

    return [
        col
        for col in df.select_dtypes(include=["float", "int"]).columns
        if col not in excluded
    ]


def _remove_non_target_bookmakers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove bookmakers that are not in the target list.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        df (pd.DataFrame): DataFrame containing odds data without non-no-commission bookmaker columns.
    """
    df = df.copy()
    bookmakers = _find_bookmaker_columns(df)
    cols_to_drop = [bm for bm in bookmakers if bm not in TARGET_BMS]
    df = df.drop(columns=cols_to_drop)
    return df


def _add_metadata(
    df: pd.DataFrame, best_odds_bms: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add Best Odds, Best Bookmaker, Result, and Outcomes columns.
    Handles missing bookmaker columns gracefully.
    """
    df = df.copy()
    bms = _find_bookmaker_columns(df)

    if best_odds_bms:
        # Only include bookmaker columns that exist
        existing_bms = [bm for bm in best_odds_bms if bm in df.columns]

        if existing_bms:
            df["Best Odds"] = df[existing_bms].max(axis=1)
            df["Best Bookmaker"] = df[existing_bms].apply(lambda row: row.idxmax() if row.notna().any() else None, axis=1)

        else:
            # Fallback if none exist
            df["Best Odds"] = None
            df["Best Bookmaker"] = None
    else:
        if bms:
            df["Best Odds"] = df[bms].max(axis=1)
            df["Best Bookmaker"] = df[bms].idxmax(axis=1)
        else:
            df["Best Odds"] = None
            df["Best Bookmaker"] = None

    df["Result"] = "Not Found"
    df["Outcomes"] = df.groupby("match")["team"].transform("count")
    df["Event ID"] = "Not Found"

    return df



def _clean_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace odds equal to 1.0 with NaN (invalid odds).

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with invalid odds (1.0) replaced with NaN.
    """
    df = df.copy()
    bookmaker_columns = _find_bookmaker_columns(df)
    df[bookmaker_columns] = df[bookmaker_columns].where(
        df[bookmaker_columns] != 1, np.nan
    )
    return df


def _min_bookmaker_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with less than MIN_BOOKMAKERS bookmaker columns.

    Args:
        df (pd.DataFrame): DataFrame containing odds data without exchange columns.

    Returns:
        pd.DataFrame: DataFrame with only rows that contain sufficient bookmaker counts.
    """
    df = df.copy()
    bookmaker_columns = _find_bookmaker_columns(df)
    num_bookmakers = df[bookmaker_columns].notna().sum(axis=1)
    df = df[num_bookmakers >= MIN_BOOKMAKERS]
    return df


def _max_odds_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with best odds greater than MAX_ODDS.

    Args:
        df (pd.DataFrame): DataFrame containing odds data without exchange columns, and with metadata.

    Returns:
        pd.DataFrame: DataFrame with only rows that contain odds that are not extreme.
    """
    df = df.copy()
    mask = df["Best Odds"] <= MAX_ODDS
    df = df[mask]
    return df


def _all_outcomes_present_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where not all outcomes are present.

    Args:
        df (pd.DataFrame): DataFrame containing odds data without exchange columns, with metadata, and
                            and with other data processing filters applied.

    Returns:
        pd.DataFrame: DataFrame with only rows that contain all outcomes.
    """
    df = df.copy()
    mask = df["Outcomes"] == df.groupby("match")["team"].transform("count")
    df = df[mask]
    return df


def _prettify_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to Title Case and replace underscores with spaces.

    Args:
        df (pd.DataFrame): DataFrame with columns to prettify.

    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    column_mapping = {col: col.replace("_", " ").title() for col in df.columns}
    return df.rename(columns=column_mapping)


def process_target_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform fetch_odds df into cleaned df for target bookmakers.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
    """
    df = _remove_non_target_bookmakers(df)
    df = _add_metadata(df, best_odds_bms=NC_BMS)
    df = _clean_odds_data(df)
    df = _min_bookmaker_filter(df)
    df = _max_odds_filter(df)
    df = _all_outcomes_present_filter(df)
    df = _prettify_column_headers(df)
    return df


def calculate_vigfree_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vig-free implied probability columns for each bookmaker.

    Args:
        df (pd.DataFrame): Processed DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with additional vig-free probability columns for each bookmaker.
    """
    df = df.copy()
    bookmaker_columns = _find_bookmaker_columns(df)

    # Add vig-free columns for each bookmaker
    for bookmaker in bookmaker_columns:
        vigfree_column = f"Vigfree {bookmaker}"
        df[vigfree_column] = np.nan

        # Process each match separately
        for match_name, match_group in df.groupby("Match", sort=False):
            required_outcomes = match_group["Outcomes"].iloc[0]

            # Get valid odds for this bookmaker in this match
            valid_odds = match_group[bookmaker].dropna()
            if len(valid_odds) < required_outcomes:
                continue

            # Calculate vig-free probabilities
            implied_probs = 1 / valid_odds
            normalized_probs = implied_probs / implied_probs.sum()

            # Update DataFrame with vig-free probabilities
            df.loc[valid_odds.index, vigfree_column] = normalized_probs.values
    return df

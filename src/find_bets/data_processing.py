"""
data_processing.py

Cleans and validates data from fetch_odds.py.

Author: Andrew Smith
"""

from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from src.constants import (
    ALL_BMS,
    MAX_ODDS,
    MIN_BOOKMAKERS,
    MIN_OUTCOMES,
    NC_BMS,
    NON_BM_COLUMNS,
    TIMESTAMP_FORMAT,
)


def find_bookmaker_columns(
    df: pd.DataFrame, exclude_columns: Optional[List[str]] = None
) -> List[str]:
    """
    Find columns that contain bookmaker odds.

    Args:
        df (pd.DataFrame): DataFrame to search for bookmaker columns.
        exclude_columns (Optional[List[str]]): Additional columns to exclude from search.

    Returns:
        List[str]: List of column names that contain bookmaker odds.
    """
    excluded = NON_BM_COLUMNS.copy()
    if exclude_columns:
        excluded.update(exclude_columns)

    return [col for col in df.columns if col not in excluded]


def _add_outcomes_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Outcomes column.
    """
    df = df.copy()
    df["Outcomes"] = df.groupby("Match")["Team"].transform("count")
    return df


def _minimum_outcomes_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where outcomes are less than minimum required.

    Args:
        df (pd.DataFrame): DataFrame containing odds data with outcomes metadata.

    Returns:
        pd.DataFrame: DataFrame with only rows that contain sufficient outcomes.
    """
    df = df.copy()
    mask = df["Outcomes"] >= MIN_OUTCOMES
    df = df[mask]
    return df


def _remove_unwanted_bookmakers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove bookmaker columns that are not in ALL_BMS.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        df (pd.DataFrame): DataFrame containing odds data with only ALL_BMS bookmaker columns.
    """
    df = df.copy()
    bookmakers = find_bookmaker_columns(df)
    cols_to_drop = [bm for bm in bookmakers if bm not in ALL_BMS]
    df = df.drop(columns=cols_to_drop)
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
    bookmaker_columns = find_bookmaker_columns(df)
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
    bookmaker_columns = find_bookmaker_columns(df)
    num_bookmakers = df[bookmaker_columns].notna().sum(axis=1)
    df = df[num_bookmakers >= MIN_BOOKMAKERS]
    return df


def _max_odds_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with best odds greater than MAX_ODDS.

    Args: df (pd.DataFrame): DataFrame containing odds data without exchange columns, and with metadata.

    Returns: pd.DataFrame: DataFrame with only rows that contain odds that are not extreme.
    """
    df = df.copy()
    bms = find_bookmaker_columns(df)
    mask = (df[bms] <= MAX_ODDS).all(axis=1)
    return df[mask]


def _add_metadata(
    df: pd.DataFrame, best_odds_bms: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Add Best Odds, Best Bookmaker, Outcomes, Result, and Scrape Time columns.
    Handles missing bookmaker columns gracefully.
    """
    df = df.copy()
    bms = find_bookmaker_columns(df)

    if best_odds_bms:
        # Only include bookmaker columns that exist
        existing_bms = [bm for bm in best_odds_bms if bm in df.columns]

        if existing_bms:
            df["Best Odds"] = df[existing_bms].max(axis=1)
            df["Best Bookmaker"] = df[existing_bms].apply(
                lambda row: row.idxmax() if row.notna().any() else None, axis=1
            )

        else:
            # Fallback if none exist
            df["Best Odds"] = None
            df["Best Bookmaker"] = None
    else:
        if bms:
            df["Best Odds"] = df[bms].max(axis=1)
            df["Best Bookmaker"] = df[bms].apply(
                lambda row: row.idxmax() if row.notna().any() else None, axis=1
            )
        else:
            df["Best Odds"] = None
            df["Best Bookmaker"] = None

    df["Result"] = "Not Found"
    df["Scrape Time"] = datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
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
    mask = df["Outcomes"] == df.groupby("Match")["Team"].transform("count")
    df = df[mask]
    return df


def process_target_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform fetch_odds df into cleaned df for target bookmakers.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: Cleaned and validated DataFrame.
    """
    df = df.copy()
    df = _add_outcomes_metadata(df)
    df = _minimum_outcomes_filter(df)
    df = _remove_unwanted_bookmakers(df)
    df = _clean_odds_data(df)
    df = _min_bookmaker_filter(df)
    df = _max_odds_filter(df)
    df = _add_metadata(df, best_odds_bms=NC_BMS)
    df = _all_outcomes_present_filter(df)
    return df

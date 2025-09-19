import numpy as np
import pandas as pd
from typing import List, Optional
from betting_configs import DATE_FORMAT, MIN_BOOKMAKERS, MAX_ODDS, EXCHANGE_BLOCKLIST


def _remove_exchanges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_to_drop = [col for col in EXCHANGE_BLOCKLIST if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    return df


def _find_bookmaker_columns(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
    """
    Find columns that contain bookmaker odds (numeric columns, excluding metadata).
    
    Args:
        df (pd.DataFrame): DataFrame to search for bookmaker columns.
        exclude_columns (Optional[List[str]]): Additional columns to exclude from search.
        
    Returns:
        List[str]: List of column names that contain bookmaker odds.
    """
    excluded = {"Best Odds", "Start Time"}
    if exclude_columns:
        excluded.update(exclude_columns)
    
    return [col for col in df.select_dtypes(include=["float", "int"]).columns 
            if col not in excluded]


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
    df[bookmaker_columns] = df[bookmaker_columns].where(df[bookmaker_columns] != 1, np.nan)
    return df


def _min_bookmaker_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bookmaker_columns = _find_bookmaker_columns(df)
    num_bookmakers = df[bookmaker_columns].notna().sum(axis=1)
    df = df[num_bookmakers > MIN_BOOKMAKERS]
    
    return df


def _max_odds_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bookmaker_columns = _find_bookmaker_columns(df)
    mask = (df[bookmaker_columns] <= MAX_ODDS).all(axis=1)
    df = df[mask]

    return df


def _add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bms = _find_bookmaker_columns(df)

    df["Best Odds"] = df[bms].max(axis=1)
    df["Best Bookmaker"] = df[bms].idxmax(axis=1)
    df["Result"] = "Not Found"

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


def process_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    df = _remove_exchanges(df)
    df = _clean_odds_data(df)
    df = _min_bookmaker_filter(df)
    df = _max_odds_filter(df)
    df = _add_metadata(df)
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
            required_outcomes = len(match_group)
            
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


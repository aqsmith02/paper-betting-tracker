"""
summary_creation.py

Logic for creating bet summary DataFrames given betting analysis data.

Author: Andrew Smith
"""
from src.find_bets.data_processing import find_bookmaker_columns
import pandas as pd


def create_average_summary_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create minimalsummary of profitable average edge bets.

    Args:
        df (pd.DataFrame): DataFrame containing average edge columns.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable average edge bets.
    """
    # Filter out rows with NaN in critical columns
    filtered_df = df.dropna(subset=["Expected Value", "Fair Odds Average"])

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time",
        "Scrape Time", "Match", "Team",
        "Best Bookmaker", "Best Odds", "Fair Odds Average", 
        "Expected Value", "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = filtered_df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)


def create_average_summary_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create full summary of profitable average edge bets.

    Args:
        df (pd.DataFrame): DataFrame containing average edge columns.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable average edge bets.
    """
    # Filter out rows with NaN in critical columns
    filtered_df = df.dropna(subset=["Expected Value", "Fair Odds Average"])

    # Find vf columns and bookmaker columns
    vigfree_columns = [col for col in filtered_df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = find_bookmaker_columns(filtered_df, vigfree_columns)

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time",
        "Scrape Time", "Match", "Team"
    ] + bookmaker_columns + vigfree_columns + [
        "Best Bookmaker", "Best Odds", "Fair Odds Average", 
        "Expected Value", "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = filtered_df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)



def create_modified_zscore_summary_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create minimalsummary of profitable Modified Z-score outlier bets.

    Args:
        df (pd.DataFrame): DataFrame containing Modified Z-score column.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Modified Z-score outlier bets.
    """
    # Filter out rows with NaN in critical columns
    filtered_df = df.dropna(subset=["Expected Value", "Fair Odds Average", "Modified Z-Score"])

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time", 
        "Scrape Time", "Match", "Team",
        "Best Bookmaker", "Best Odds",
        "Fair Odds Average", "Expected Value", 
        "Modified Z-Score", "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = filtered_df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)


def create_modified_zscore_summary_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create full summary of profitable Modified Z-score outlier bets.

    Args:
        df (pd.DataFrame): DataFrame containing Modified Z-score column.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Modified Z-score outlier bets.
    """
    # Filter out rows with NaN in critical columns
    filtered_df = df.dropna(subset=["Expected Value", "Fair Odds Average", "Modified Z-Score"])

    # Find vf columns and bookmaker columns
    vigfree_columns = [col for col in filtered_df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = find_bookmaker_columns(filtered_df, vigfree_columns)

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time", 
        "Scrape Time", "Match", "Team"
    ] + bookmaker_columns + vigfree_columns + [
        "Best Bookmaker", "Best Odds",
        "Fair Odds Average", "Expected Value", 
        "Modified Z-Score", "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = filtered_df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)


def create_random_summary_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create minimal summary of randomly selected bets.

    Args:
        df (pd.DataFrame): DataFrame containing random bet column.

    Returns:
        pd.DataFrame: Summary DataFrame with only random bets.
    """
    df = df[df['Random Placed Bet'] != 0].reset_index(drop=True)

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time", 
        "Scrape Time", "Match", "Team",
        "Best Bookmaker", "Best Odds",
        "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)


def create_random_summary_full(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create full summary of randomly selected bets.

    Args:
        df (pd.DataFrame): DataFrame containing random bet column.

    Returns:
        pd.DataFrame: Summary DataFrame with only random bets.
    """
    df = df[df['Random Placed Bet'] != 0].reset_index(drop=True)

    # Find vf columns and bookmaker columns
    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = find_bookmaker_columns(df, vigfree_columns)

    # Define column order
    columns_to_keep = [
        "ID", "Sport Key", "Sport Title", "Start Time", 
        "Scrape Time", "Match", "Team"
    ] + bookmaker_columns + vigfree_columns + [
        "Best Bookmaker", "Best Odds",
        "Outcomes", "Result"
    ]

    # Reindex to enforce order, ignore missing columns if needed
    summary_df = df.reindex(columns=columns_to_keep)

    return summary_df.reset_index(drop=True)

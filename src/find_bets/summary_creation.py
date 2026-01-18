"""
summary_creation.py

Logic for creating bet summary DataFrames given betting analysis data.

Author: Andrew Smith
"""

import pandas as pd

def create_average_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of profitable average edge bets.

    Args:
        df (pd.DataFrame): DataFrame containing average edge columns.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable average edge bets.
    """
    summary_rows = []
    for _, row in df.iterrows():
        # Do not include rows with NaN in Expected Value or Fair Odds Average
        if pd.isna(row.get("Expected Value")) or pd.isna(row.get("Fair Odds Average")):
            continue

        summary_rows.append(
            {
                "ID": row["ID"],
                "Sport Key": row["Sport Key"],
                "Sport Title": row["Sport Title"],
                "Start Time": row["Start Time"],
                "Scrape Time": row["Scrape Time"],
                "Match": row["Match"],
                "Team": row["Team"],
                "Best Bookmaker": row["Best Bookmaker"],
                "Best Odds": row["Best Odds"],
                "Fair Odds Average": row["Fair Odds Average"],
                "Expected Value": row["Expected Value"],
                "Outcomes": row["Outcomes"],
                "Result": row["Result"],
            }
        )
    return pd.DataFrame(summary_rows)


def create_modified_zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of profitable Modified Z-score outlier bets.

    Args:
        df (pd.DataFrame): DataFrame containing Modified Z-score column.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Modified Z-score outlier bets.
    """
    summary_rows = []
    for _, row in df.iterrows():
        # Do not include rows with NaN in Modified Z Score or Expected Value
        if pd.isna(row.get("Modified Z Score")) or pd.isna(row.get("Expected Value")) or pd.isna(row.get("Fair Odds Average")):
            continue

        summary_rows.append(
            {
                "ID": row["ID"],
                "Sport Key": row["Sport Key"],
                "Sport Title": row["Sport Title"],
                "Start Time": row["Start Time"],
                "Scrape Time": row["Scrape Time"],
                "Match": row["Match"],
                "Team": row["Team"],
                "Best Bookmaker": row["Best Bookmaker"],
                "Best Odds": row["Best Odds"],
                "Fair Odds Average": row["Fair Odds Average"],
                "Expected Value": row["Expected Value"],
                "Modified Z Score": row["Modified Z Score"],
                "Outcomes": row["Outcomes"],
                "Result": row["Result"],
            }
        )

    return pd.DataFrame(summary_rows)


def create_random_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of randomly selected bets.

    Args:
        df (pd.DataFrame): DataFrame containing random bet column.

    Returns:
        pd.DataFrame: Summary DataFrame with only random bets.
    """
    summary_rows = []
    for _, row in df.iterrows():
        if row["Random Placed Bet"] == 0:
            continue

        summary_rows.append(
            {
                "ID": row["ID"],
                "Sport Key": row["Sport Key"],
                "Sport Title": row["Sport Title"],
                "Start Time": row["Start Time"],
                "Scrape Time": row["Scrape Time"],
                "Match": row["Match"],
                "Team": row["Team"],
                "Best Bookmaker": row["Best Bookmaker"],
                "Best Odds": row["Best Odds"],
                "Outcomes": row["Outcomes"],
                "Result": row["Result"],
            }
        )

    return pd.DataFrame(summary_rows)

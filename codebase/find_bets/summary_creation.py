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
        if pd.isna(row.get("Avg Edge Pct")) or pd.isna(row.get("Fair Odds Avg")):
            continue

        summary_rows.append(
            {
                "Match": row["Match"],
                "League": row["League"],
                "Team": row["Team"],
                "Start Time": row["Start Time"],
                "Avg Edge Book": row["Best Bookmaker"],
                "Avg Edge Odds": row["Best Odds"],
                "Expected Value": row["Expected Value"],
                "Result": row["Result"],
            }
        )
    return pd.DataFrame(summary_rows)


def create_zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of profitable Z-score outlier bets.

    Args:
        df (pd.DataFrame): DataFrame containing Z-score column.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Z-score outlier bets.
    """
    summary_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Z Score")) or pd.isna(row.get("Avg Edge Pct")):
            continue

        summary_rows.append(
            {
                "Match": row["Match"],
                "League": row["League"],
                "Team": row["Team"],
                "Start Time": row["Start Time"],
                "Outlier Book": row["Best Bookmaker"],
                "Outlier Odds": row["Best Odds"],
                "Z Score": row["Z Score"],
                "Expected Value": row["Expected Value"],
                "Result": row.get("Result", "Not Found"),
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
        if pd.isna(row.get("Modified Z Score")) or pd.isna(row.get("Avg Edge Pct")):
            continue

        summary_rows.append(
            {
                "Match": row["Match"],
                "League": row["League"],
                "Team": row["Team"],
                "Start Time": row["Start Time"],
                "Outlier Book": row["Best Bookmaker"],
                "Outlier Odds": row["Best Odds"],
                "Modified Z Score": row["Modified Z Score"],
                "Expected Value": row["Expected Value"],
                "Result": row.get("Result", "Not Found"),
            }
        )

    return pd.DataFrame(summary_rows)


def create_pinnacle_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of profitable Pinnacle edge bets.

    Args:
        df (pd.DataFrame): DataFrame containing Pinnacle edge column.

    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Pinnacle edge bets.
    """
    summary_rows = []
    vigfree_pinnacle = f"Vigfree Pinnacle"
    if vigfree_pinnacle not in df.columns:
        return pd.DataFrame(summary_rows)

    for _, row in df.iterrows():
        if pd.isna(row.get("Pinnacle Fair Odds")) or pd.isna(row.get("Pin Edge Pct")):
            continue

        summary_rows.append(
            {
                "Match": row["Match"],
                "League": row["League"],
                "Team": row["Team"],
                "Start Time": row["Start Time"],
                "Pinnacle Edge Book": row["Best Bookmaker"],
                "Pinnacle Edge Odds": row["Best Odds"],
                "Expected Value": row["Expected Value"],
                "Pinnacle Fair Odds": row["Pinnacle Fair Odds"],
                "Result": row.get("Result", "Not Found"),
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
                "Match": row["Match"],
                "League": row["League"],
                "Team": row["Team"],
                "Start Time": row["Start Time"],
                "Random Bet Book": row["Best Bookmaker"],
                "Random Bet Odds": row["Best Odds"],
                "Result": row.get("Result", "Not Found"),
            }
        )

    return pd.DataFrame(summary_rows)

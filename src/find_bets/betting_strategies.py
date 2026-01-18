"""
betting_strategies.py

Logic for 3 different betting strategy methods. Functions append results columns to input pd.DataFrame.

Author: Andrew Smith
"""

from typing import List
import pandas as pd
from src.constants import (
    EV_THRESHOLD,
    Z_SCORE_THRESHOLD,
    MAX_Z_SCORE,
)
from src.find_bets.data_processing import find_bookmaker_columns
import random


def _missing_vigfree_odds(row: pd.Series, bookmaker_columns: List[str]) -> bool:
    """
    Check if row has any bookmakers with odds but no vig-free probability.

    Args:
        row (pd.Series): Single row to check.
        bookmaker_columns (List[str]): List of bookmaker column names.

    Returns:
        bool: True if all bookmakers with odds have vig-free probabilities, False if any are missing.
    """
    for bookmaker in bookmaker_columns:
        if pd.notna(row[bookmaker]):  # Has odds
            vigfree_col = f"Vigfree {bookmaker}"
            if vigfree_col in row and pd.isna(row[vigfree_col]):  # Missing vig-free odds
                return True
    
    return False


def analyze_average_edge_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds exceed average fair odds by threshold.

    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.

    Returns:
        pd.DataFrame: DataFrame with additional columns for fair odds average and edge percentage.
    """
    df = df.copy()
    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = find_bookmaker_columns(df, vigfree_columns)

    ev_list = []
    fair_odds_averages = []

    for _, row in df.iterrows():
        # Check if row has sufficient vig-free data
        if _missing_vigfree_odds(row, bookmaker_columns):
            ev_list.append(None)
            fair_odds_averages.append(None)
            continue

        # Calculate average fair odds
        average_probability = row[vigfree_columns].mean()
        fair_odds = 1 / average_probability
        fair_odds_averages.append(round(fair_odds, 2))

        # Validate average probability
        if average_probability <= 0 or average_probability >= 1:
            raise ValueError("Average probability out of bounds for EV calculation.")

        # Calculate EV
        best_odds = row["Best Odds"]
        ev = (average_probability * (best_odds - 1)) - ((1 - average_probability) * 1)

        if ev > EV_THRESHOLD:
            ev_list.append(ev)
        else:
            ev_list.append(None)

    df["Fair Odds Average"] = fair_odds_averages
    df["Expected Value"] = ev_list
    return df


def analyze_modified_zscore_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds are outliers using Modified Z-score (more robust to outliers) and
    average edge.

    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.

    Returns:
        pd.DataFrame: DataFrame with additional Modified Z-score column and average edge columns.
    """
    df = df.copy()
    df = analyze_average_edge_bets(df)

    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    cols_to_exclude = vigfree_columns + ["Fair Odds Average", "Expected Value"]
    bookmaker_columns = find_bookmaker_columns(df, cols_to_exclude)
    modified_z_scores = []

    for _, row in df.iterrows():
        best_odds = row["Best Odds"]

        # Calculate Modified Z-score using median and MAD
        median_odds = row[bookmaker_columns].median()
        mad = (row[bookmaker_columns] - median_odds).abs().median()

        if mad == 0:  # Avoid division by zero
            modified_z_scores.append(None)
            continue

        modified_z = 0.6745 * max(0, best_odds - median_odds) / mad

        # Only include if within reasonable bounds
        if Z_SCORE_THRESHOLD < modified_z < MAX_Z_SCORE:
            modified_z_scores.append(round(modified_z, 2))
        else:
            modified_z_scores.append(None)

    df["Modified Z Score"] = modified_z_scores
    return df


def find_random_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place random bets.

    Args:
        df (pd.DataFrame): DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with additional column for placed random bets.
    """
    df = df.copy()
    rand_amount = random.randint(0, min(5, len(df)))
    df_sample = df.sample(n=rand_amount)
    df["Random Placed Bet"] = df.index.isin(df_sample.index).astype(int)

    return df
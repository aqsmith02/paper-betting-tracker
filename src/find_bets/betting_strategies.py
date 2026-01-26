"""
betting_strategies.py

Vectorized logic for 3 different betting strategy methods. 
Functions append results columns to input pd.DataFrame.

Author: Andrew Smith
Refactored for vectorization and modularity
"""
from typing import List
import pandas as pd
import numpy as np
from src.constants import (
    EV_THRESHOLD,
    Z_SCORE_THRESHOLD,
    MAX_EV,
    MAX_Z_SCORE,
)
from src.find_bets.data_processing import find_bookmaker_columns
import random


def _get_vigfree_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract vig-free column names from DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        List[str]: List of vig-free column names.
    """
    return [col for col in df.columns if col.startswith("Vigfree ")]

def _calculate_fair_odds_from_probabilities(probabilities: pd.Series) -> pd.Series:
    """
    Calculate fair odds from probabilities.
    
    Args:
        probabilities (pd.Series): Series of probabilities.
        
    Returns:
        pd.Series: Series of fair odds rounded to 2 decimal places.
    """
    # Check for invalid probabilities (including NaN)
    if not probabilities.between(0, 1, inclusive='neither').all():
        raise ValueError("Probabilities must be between 0 and 1 (exclusive).")
    
    # Calculate fair odds
    fair_odds = 1 / probabilities
    return fair_odds.round(2)


def _calculate_expected_value(
    probabilities: pd.Series, 
    best_odds: pd.Series
) -> pd.Series:
    """
    Calculate expected value for bets.
    
    EV = (prob * (odds - 1)) - ((1 - prob) * 1) = (prob * odds) - 1
    
    Args:
        probabilities (pd.Series): Fair probabilities.
        best_odds (pd.Series): Best available odds.
        
    Returns:
        pd.Series: Expected value for each bet rounded to 2 decimal places.
    """
    ev = pd.Series(np.nan, index=probabilities.index)
    
    mask = probabilities.notna() & best_odds.notna()
    ev[mask] = (
        (probabilities[mask] * (best_odds[mask])) - 1
    )
    
    return ev.round(2)


def _calculate_modified_zscore(
    values: pd.Series,
    reference_values: pd.DataFrame
) -> pd.Series:
    """
    Calculate Modified Z-score using median and MAD.
    
    Modified Z-score = 0.6745 * (value - median) / MAD
    
    Args:
        values (pd.Series): Values to calculate Modified Z-score for.
        reference_values (pd.DataFrame): Reference values for median/MAD calculation.
        
    Returns:
        pd.Series: Modified Z-scores rounded to 2 decimal places.
    """
    median_vals = reference_values.median(axis=1)
    mad = (reference_values.sub(median_vals, axis=0)).abs().median(axis=1)
    
    # Initialize with NaN
    modified_z = pd.Series(np.nan, index=values.index)
    
    # Only calculate where MAD is not zero
    valid_mask = (mad != 0) & values.notna()
    
    # Calculate modified z-score (only for positive deviations)
    deviations = np.maximum(0, values[valid_mask] - median_vals[valid_mask])
    modified_z[valid_mask] = 0.6745 * deviations / mad[valid_mask]
    
    return modified_z.round(2)


def _filter_by_threshold(
    values: pd.Series,
    min_threshold: float,
    max_threshold: float
) -> pd.Series:
    """
    Filter values by threshold range.
    
    Values must be > min_threshold and < max_threshold.
    
    Args:
        values (pd.Series): Values to filter.
        min_threshold (float): Minimum threshold (exclusive).
        max_threshold (float): Maximum threshold (exclusive).
        
    Returns:
        pd.Series: Filtered values (NaN where conditions not met).
    """
    filtered = values.copy()
    
    # Keep only values greater than min threshold
    filtered[values <= min_threshold] = np.nan
    
    # Keep only values less than max threshold
    filtered[values >= max_threshold] = np.nan
    
    return filtered


def find_average_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds exceed average fair odds by threshold (vectorized).
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for fair odds average and edge percentage.
    """
    df = df.copy()
    
    # Get relevant columns
    vigfree_columns = _get_vigfree_columns(df)
    
    # Calculate average probability across vig-free columns
    average_probability = df[vigfree_columns].mean(axis=1)
    
    # Calculate fair odds
    fair_odds = _calculate_fair_odds_from_probabilities(average_probability)
    
    # Calculate expected value
    ev = _calculate_expected_value(average_probability, df["Best Odds"])
    
    # Apply threshold filters (EV_THRESHOLD < ev < MAX_EV)
    ev_filtered = _filter_by_threshold(ev, EV_THRESHOLD, MAX_EV)
    
    # Add columns to dataframe
    df["Fair Odds Average"] = fair_odds
    df["Expected Value"] = ev_filtered
    
    return df


def find_modified_zscore_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds are outliers using Modified Z-score (vectorized).
    
    More robust to outliers than standard Z-score.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional Modified Z-score and average edge columns.
    """
    df = df.copy()
    
    # First calculate average bets (includes Fair Odds Average and Expected Value)
    df = find_average_bets(df)
    
    # Get bookmaker columns (excluding vig-free and calculated columns)
    vigfree_columns = _get_vigfree_columns(df)
    bookmaker_columns = find_bookmaker_columns(df, vigfree_columns)
    
    # Calculate modified z-scores
    modified_z = _calculate_modified_zscore(
        df["Best Odds"],
        df[bookmaker_columns]
    )
    
    # Apply threshold filters
    modified_z_filtered = _filter_by_threshold(
        modified_z,
        Z_SCORE_THRESHOLD,
        MAX_Z_SCORE
    )
    
    df["Modified Z-Score"] = modified_z_filtered
    
    return df


def find_random_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place random bets (for baseline comparison).
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional column for placed random bets.
    """
    df = df.copy()
    
    # Randomly select bets
    rand_amount = random.randint(0, min(5, len(df)))
    
    if rand_amount > 0:
        df_sample = df.sample(n=rand_amount)
        df["Random Placed Bet"] = df.index.isin(df_sample.index).astype(int)
    else:
        df["Random Placed Bet"] = 0
    
    return df
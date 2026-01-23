"""
vigfree_probabilities.py

Calculates vig-free probabilities from odds data.

Author: Andrew Smith
"""
import pandas as pd
import numpy as np
from src.find_bets.data_processing import find_bookmaker_columns

def _calculate_market_margin(odds_list):
    """
    Calculate the total market margin (overround) from a list of odds.
    
    Args:
        odds_list: List of decimal odds for all outcomes in the market
    
    Returns:
        Margin as a decimal (e.g., 0.08 for 8% margin)
    """
    # Handle None
    if odds_list is None:
        return 0
    
    # Convert Series to list if needed
    if hasattr(odds_list, "tolist"):
        odds_list = odds_list.tolist()
    
    # Check empty (handles both empty list and empty Series after conversion)
    if not odds_list:
        return 0
    
    # Sum of implied probabilities
    implied_prob_sum = sum(1/odds for odds in odds_list if odds > 0)
    
    # Margin is the amount over 1.0 (100%)
    margin = implied_prob_sum - 1.0

    if margin < 0:
        raise ValueError(
            f"Invalid market: negative margin ({margin:.4f}). "
            f"Implied probabilities sum to {implied_prob_sum:.4f}. "
            f"Odds: {odds_list}"
        )
    
    return margin


def _remove_margin_proportional_to_odds(bookmaker_odds, all_market_odds, n_outcomes):
    """
    Remove bookmaker margin using the "proportional to odds" method.
    
    This method assumes bookmakers apply larger margins to longshots and smaller
    margins to favorites, proportional to the odds size.
    
    Formula for fair odds:
        Fair_Odds = (n × Bookmaker_Odds) / (n - M × Bookmaker_Odds)
    
    Where:
        n = number of outcomes in the market
        M = total market margin
    
    Args:
        bookmaker_odds: The specific odds you're evaluating
        all_market_odds: List of all odds in the market (to calculate margin)
        n_outcomes: Number of possible outcomes
    
    Returns:
        Fair odds with margin removed
    """
    if bookmaker_odds < 1:
        raise ValueError(
            f"Invalid odds: less than 1 ({bookmaker_odds}). "
        )
    
    # Calculate market margin
    margin = _calculate_market_margin(all_market_odds)
    
    # Apply formula: Fair_Odds = (n × O) / (n - M × O)
    denominator = n_outcomes - (margin * bookmaker_odds)

    # Avoid division by zero or negative denominators
    if denominator <= 0:
        raise ValueError(
            f"Margin removal error: non-positive denominator ({denominator}).\n"
            f"Components: n_outcomes={n_outcomes}, "
            f"bookmaker_odds={bookmaker_odds}, margin={margin}."
        )
    
    fair_odds = (n_outcomes * bookmaker_odds) / denominator
    return fair_odds


def _calculate_vigfree_probs_for_market(valid_odds, required_outcomes):
    """
    Calculate vig-free probabilities for all outcomes in a market.
    
    Args:
        valid_odds: Series of odds for all outcomes
        required_outcomes: Number of outcomes in the market
    
    Returns:
        list: Vig-free probabilities for each outcome (empty list if calculation fails)
    """
    vigfree_probs = []
    
    for odds_value in valid_odds:
        fair_odds = _remove_margin_proportional_to_odds(
            odds_value, 
            valid_odds, 
            int(required_outcomes)
        )
        
        # Skip if fair_odds calculation produces invalid result
        if fair_odds is None or fair_odds <= 1:
            continue
        
        vigfree_prob = 1 / fair_odds
        vigfree_probs.append(vigfree_prob)
    
    return vigfree_probs


def _has_complete_odds(match_group, bookmaker, required_outcomes):
    """
    Check if a bookmaker has odds for all required outcomes in a match.
    
    Args:
        match_group: DataFrame group for a single match
        bookmaker: Name of the bookmaker column
        required_outcomes: Number of outcomes required for a complete market
    
    Returns:
        tuple: (has_complete_odds: bool, valid_odds: Series or None)
    """
    valid_odds = match_group[bookmaker].dropna()
    
    if len(valid_odds) < required_outcomes:
        return False, None
    
    return True, valid_odds


def _process_bookmaker_for_match(df, match_group, bookmaker, vigfree_column):
    """
    Calculate and update vig-free probabilities for a single bookmaker in a single match.
    
    Args:
        df: Main DataFrame to update
        match_group: DataFrame group for a single match
        bookmaker: Name of the bookmaker column
        vigfree_column: Name of the column to store vig-free probabilities
    
    Returns:
        None (modifies df in place)
    """
    required_outcomes = match_group["Outcomes"].iloc[0]
    
    # Check if bookmaker has complete odds
    has_complete, valid_odds = _has_complete_odds(match_group, bookmaker, required_outcomes)
    
    if not has_complete:
        # Leave vig-free probabilities as NaN if not all outcomes are present
        return
    
    # Calculate vig-free probabilities for this market
    vigfree_probs = _calculate_vigfree_probs_for_market(valid_odds, required_outcomes)
    
    # Only update if we got valid probabilities for all outcomes
    if len(vigfree_probs) == len(valid_odds):
        df.loc[valid_odds.index, vigfree_column] = vigfree_probs


def calculate_vigfree_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add vig-free implied probability columns for each bookmaker using margin proportional to odds method.

    Args:
        df (pd.DataFrame): Processed DataFrame containing odds data.

    Returns:
        pd.DataFrame: DataFrame with additional vig-free probability columns for each bookmaker.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)

    # Add vig-free columns for each bookmaker
    for bookmaker in bookmaker_columns:
        vigfree_column = f"Vigfree {bookmaker}"
        df[vigfree_column] = np.nan

        # Process each match separately
        for match_name, match_group in df.groupby("Match", sort=False):
            _process_bookmaker_for_match(df, match_group, bookmaker, vigfree_column)
            
    return df
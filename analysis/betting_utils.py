"""
Shared betting calculation utilities.

This module contains common betting functions used across multiple analysis files.
All Kelly criterion calculations should use the kelly_bet function from this module
to ensure consistency.

Configuration is loaded from config.py.
"""

import pandas as pd
from config.analysis_config import (
    KELLY_FRACTION,
    MIN_EV_THRESHOLD,
    MAX_EV_THRESHOLD,
    ZSCORE_MAX_BET_THRESHOLD,
    MAX_BET_MULTIPLIER,
    NULL_EV,
    RESULT_COLUMN,
    TEAM_COLUMN,
)


def kelly_bet(odds, fair_odds, ev=None, zscore=None, max_multiplier=None):
    """
    Calculate Kelly criterion bet size with parameter uncertainty adjustment.
    
    This is the single source of truth for Kelly betting calculations across
    all analysis modules. Uses fractional Kelly (configured in config.py) to
    reduce variance while maintaining long-term growth.
    
    Args:
        odds: Decimal odds for the bet (e.g., 2.5 for +150 American odds)
        fair_odds: Fair odds calculated from estimated probability
        ev: Expected value as decimal (optional, used for filtering)
            If provided, bets are filtered based on MIN/MAX_EV_THRESHOLD from config
        zscore: Z-score value (optional, triggers max bet if > threshold)
            If provided and >= ZSCORE_MAX_BET_THRESHOLD from config, returns max bet
        max_multiplier: Maximum bet size as percentage of bankroll 
            If None, uses MAX_BET_MULTIPLIER from config
    
    Returns:
        Bet size as percentage of bankroll (0-100)
        Returns 0 if:
        - Odds are invalid (NaN or <= 1)
        - EV is outside threshold range
        - Kelly criterion suggests no bet (negative edge)
    
    Examples:
        >>> kelly_bet(2.5, 2.0)  # Good value, no filters
        12.5
        >>> kelly_bet(2.5, 2.0, ev=0.02)  # Low EV, filtered out
        0.0
        >>> kelly_bet(2.5, 2.0, ev=0.10, zscore=4.0)  # High z-score
        2.5
    """
    # Use config value if max_multiplier not provided
    if max_multiplier is None:
        max_multiplier = MAX_BET_MULTIPLIER
    
    # Validate odds
    if pd.isna(odds) or odds <= 1:
        return 0
    
    # Check EV threshold if provided (uses config thresholds)
    if ev is not None and not pd.isna(ev):
        if ev < MIN_EV_THRESHOLD:
            return 0
        if ev > MAX_EV_THRESHOLD:
            return 0

    # Calculate win probability from fair odds
    p = 1 / fair_odds
    # Clamp probability to avoid edge cases
    p = max(min(p, 0.9999), 0.0001)
    
    # Calculate net odds (profit per unit bet)
    b = odds - 1

    # Calculate raw Kelly fraction: f* = (bp - q) / b = p - q/b
    # where q = 1 - p (probability of losing)
    f_kelly = p - ((1 - p) / b)

    # Only bet if Kelly is positive (we have an edge)
    if f_kelly <= 0:
        return 0

    # Check if Z-score triggers max bet (uses config threshold)
    # High z-score indicates high confidence in the edge
    if zscore is not None and not pd.isna(zscore):
        if zscore >= ZSCORE_MAX_BET_THRESHOLD:
            return max_multiplier

    # Apply fractional Kelly for risk management (uses config fraction)
    # Full Kelly can be too aggressive, fractional Kelly reduces variance
    f_adjusted = f_kelly * KELLY_FRACTION

    # Cap at max multiplier to limit exposure
    return min(f_adjusted * 100, max_multiplier)


def calculate_null_probability(odds, null_ev=None):
    """
    Calculate win probability under null hypothesis of specified EV.
    
    Used in Monte Carlo simulations to test whether observed results are
    statistically significant or could be explained by random chance with
    a specific expected value (typically negative, representing vig).
    
    Mathematical derivation:
        EV = prob * (odds - 1) - (1 - prob)
        EV = prob * odds - prob - 1 + prob
        EV = prob * odds - 1
        prob = (EV + 1) / odds
    
    Args:
        odds: Decimal odds for the bet
        null_ev: Expected value under null hypothesis
                 If None, uses NULL_EV from config (typically -0.05 = -5% representing bookmaker edge)
    
    Returns:
        Win probability under null hypothesis, clamped to [0, 1]
    
    Examples:
        >>> calculate_null_probability(2.0, null_ev=-0.05)  # 2.0 odds, -5% EV
        0.475
        >>> calculate_null_probability(3.0, null_ev=-0.05)  # 3.0 odds, -5% EV
        0.3167
    """
    # Use config value if null_ev not provided
    if null_ev is None:
        null_ev = NULL_EV
    
    # Using derived formula: prob = (EV + 1) / odds
    probability = (null_ev + 1) / odds
    
    # Clamp probability to valid range [0, 1]
    # Can go outside range with extreme odds or EV values
    return max(0, min(1, probability))


def calculate_time_spent(df):
    """
    Calculate time interval over which data was collected.
    
    Args:
        df: DataFrame with 'Scrape Time' column containing datetime strings
    
    Returns:
        Formatted string: "Xd Yh Zm Zs" (e.g., "45d 3h 27m 12s")
    
    Examples:
        >>> df = pd.DataFrame({'Scrape Time': ['2024-01-01 10:00:00', '2024-01-15 14:30:45']})
        >>> calculate_time_spent(df)
        '14d 4h 30m 45s'
    """
    # Parse first and last scrape times
    time_start = pd.to_datetime(df.loc[0, "Scrape Time"])
    time_end = pd.to_datetime(df.loc[len(df) - 1, "Scrape Time"])
    time_spent = time_end - time_start

    # Break down into days, hours, minutes, seconds
    days = time_spent.days
    hours, remainder = divmod(time_spent.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    return f"{days}d {hours}h {minutes}m {seconds}s"


def filter_pending_results(df, pending_results):
    """
    Filter out bets with pending results.
    
    Args:
        df: DataFrame with 'Result' column
        pending_results: List of result values considered pending (from constants)
    
    Returns:
        Filtered DataFrame with pending results removed
    """
    return df[~df["Result"].isin(pending_results)].copy()


def calculate_flat_bet_profit(df, odds_col, result_col=None, team_col=None, bet_size=1.0):
    """
    Calculate profit from flat betting strategy.
    
    Args:
        df: DataFrame with betting data (already filtered for pending results)
        odds_col: Column name containing odds
        result_col: Column name containing actual results (default from config)
        team_col: Column name containing team bet on (default from config)
        bet_size: Size of each flat bet (default 1.0 unit)
    
    Returns:
        DataFrame with added columns: 'Bet_Size', 'Profit'
    """
    # Use config defaults if not provided
    if result_col is None:
        result_col = RESULT_COLUMN
    if team_col is None:
        team_col = TEAM_COLUMN
    
    df = df.copy()
    df['Bet_Size'] = bet_size
    df['Profit'] = df.apply(
        lambda row: bet_size * (row[odds_col] - 1) if row[team_col] == row[result_col] else -bet_size,
        axis=1
    )
    return df


def calculate_kelly_bet_profit(df, odds_col, fair_odds_col, ev_col=None, zscore_col=None,
                                result_col=None, team_col=None):
    """
    Calculate profit from Kelly criterion betting strategy.
    
    Args:
        df: DataFrame with betting data (already filtered for pending results)
        odds_col: Column name containing odds
        fair_odds_col: Column name containing fair odds
        ev_col: Column name containing expected value (optional)
        zscore_col: Column name containing z-scores (optional)
        result_col: Column name containing actual results (default from config)
        team_col: Column name containing team bet on (default from config)
    
    Returns:
        DataFrame with added columns: 'Bet_Size', 'Profit'
        Only includes rows where Bet_Size > 0 (Kelly says to bet)
    """
    # Use config defaults if not provided
    if result_col is None:
        result_col = RESULT_COLUMN
    if team_col is None:
        team_col = TEAM_COLUMN
    
    df = df.copy()
    
    # Calculate Kelly bet size for each bet
    df['Bet_Size'] = df.apply(
        lambda row: kelly_bet(
            odds=row[odds_col],
            fair_odds=row[fair_odds_col],
            ev=row.get(ev_col) if ev_col else None,
            zscore=row.get(zscore_col) if zscore_col else None,
        ),
        axis=1,
    )
    
    # Calculate profit for each bet
    df['Profit'] = df.apply(
        lambda row: row['Bet_Size'] * (row[odds_col] - 1) if row[team_col] == row[result_col] 
                    else -row['Bet_Size'],
        axis=1
    )
    
    # Filter to only bets where Kelly says to bet
    return df[df['Bet_Size'] > 0].copy()
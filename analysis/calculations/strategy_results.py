"""
ROI Calculator

Calculate return on investment (ROI) for betting strategies using both
flat betting and Kelly criterion approaches.
"""

import pandas as pd
from src.constants import PENDING_RESULTS
from config.analysis_config import (
    KELLY_FRACTION,
    MIN_EV_THRESHOLD,
    MAX_EV_THRESHOLD,
    ZSCORE_MAX_BET_THRESHOLD,
)
from analysis.betting_utils import kelly_bet, calculate_time_spent
from analysis.strategy_definitions import ALL_STRATEGIES


# ====================================
# === ROI Calculations ===============
# ====================================


def flat_betting_results(df, odds_col):
    """
    Calculate ROI along with other metrics using flat betting (1 unit per bet).
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        
    Returns:
        Dictionary with total wagered, winnings, profit, ROI
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)]
    total_wagered = len(df)
    total_winnings = (df[odds_col] * (df["Team"] == df["Result"])).sum()
    net_profit = total_winnings - total_wagered
    return {
        "Total Wagered": total_wagered,
        "Total Winnings": total_winnings,
        "Net Profit": net_profit,
        "ROI": net_profit / total_wagered * 100 if total_wagered > 0 else 0,
    }


def kelly_betting_results(df, odds_col, fair_col, ev_col=None, zscore_col=None):
    """
    Calculate ROI along with other metrics using Kelly criterion with EV and Z-score considerations.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
        
    Returns:
        Dictionary with total wagered, winnings, profit, ROI, and bets placed
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()

    df["Bet"] = df.apply(
        lambda row: kelly_bet(
            odds=row[odds_col],
            fair_odds=row[fair_col],
            ev=row.get(ev_col) if ev_col else None,
            zscore=row.get(zscore_col) if zscore_col else None,
        ),
        axis=1,
    )

    total_wagered = df["Bet"].sum()
    total_winnings = (df["Bet"] * (df[odds_col] * (df["Team"] == df["Result"]))).sum()
    net_profit = total_winnings - total_wagered
    
    # Calculate number of bets actually placed
    bets_placed = (df["Bet"] > 0).sum()
    
    return {
        "Total Wagered": total_wagered,
        "Total Winnings": total_winnings,
        "Net Profit": net_profit,
        "ROI": net_profit / total_wagered * 100 if total_wagered > 0 else 0,
        "Bets Placed": bets_placed,
    }


# ====================================
# === Comparison & Reporting =========
# ====================================


def compare_methods(strategy):
    """
    Compare flat betting vs Kelly betting for a strategy.
    
    Args:
        strategy: BettingStrategy object to analyze
    """
    df = pd.read_csv(strategy.path)
    time_spent = calculate_time_spent(df)

    print(f"\n=== {strategy.name} ===")
    print(f"Time Spent: {time_spent}")
    print(f"Total Opportunities: {len(df[~df['Result'].isin(PENDING_RESULTS)])}")
    print(
        f"{'Method':<15} {'Wagered':>10} {'Winnings':>10} {'Net Profit':>12} {'ROI %':>8} {'Bets':>6}"
    )
    print("-" * 70)

    # Flat betting
    flat = flat_betting_results(df, strategy.odds_column)
    print(
        f"{'Flat':<15} {flat['Total Wagered']:>10.2f} {flat['Total Winnings']:>10.2f} "
        f"{flat['Net Profit']:>12.2f} {flat['ROI']:>8.2f} {flat['Total Wagered']:>6.0f}"
    )

    # Kelly betting (only if strategy has EV column)
    if strategy.ev_column:
        kelly = kelly_betting_results(
            df,
            strategy.odds_column,
            strategy.fair_odds_column,
            strategy.ev_column,
            strategy.zscore_column,
        )

        print(
            f"{'Kelly':<15} {kelly['Total Wagered']:>10.2f} {kelly['Total Winnings']:>10.2f} "
            f"{kelly['Net Profit']:>12.2f} {kelly['ROI']:>8.2f} {kelly['Bets Placed']:>6.0f}"
        )


def print_config():
    """Print current configuration parameters."""
    print("=" * 70)
    print("CONFIGURATION")
    print("=" * 70)
    print(f"Kelly Fraction: {KELLY_FRACTION}")
    print(f"Minimum EV Threshold: {MIN_EV_THRESHOLD:.2%}")
    print(f"Maximum EV Threshold: {MAX_EV_THRESHOLD:.2%}")
    print(f"Z-Score Max Bet Threshold: {ZSCORE_MAX_BET_THRESHOLD}")
    print("=" * 70)


def main():
    """Run ROI analysis for all strategies."""
    print_config()
    print("\n" + "=" * 70)
    print("COMPARISON: Flat vs Kelly")
    print("=" * 70)
    for strat in ALL_STRATEGIES:
        compare_methods(strat)


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from codebase.results_appending.results_configs import PENDING_RESULTS


# --- Data class for strategies ---
@dataclass
class BettingStrategy:
    name: str
    path: str
    odds_column: str
    fair_odds_column: str = None
    ev_column: str = None
    zscore_column: str = None


STRATEGIES = [
    BettingStrategy(
        "Average NC",
        "codebase/data/master_nc_avg_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
    ),
    BettingStrategy(
        "Modified Zscore NC",
        "codebase/data/master_nc_mod_zscore_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
        "Modified Z Score",
    ),
    BettingStrategy(
        "Pinnacle NC",
        "codebase/data/master_nc_pin_full.csv",
        "Best Odds",
        "Pinnacle Fair Odds",
        "Expected Value",
    ),
    BettingStrategy(
        "Zscore NC",
        "codebase/data/master_nc_zscore_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
        "Z Score",
    ),
    BettingStrategy(
        "Random NC", "codebase/data/master_nc_random_full.csv", "Best Odds"
    ),
]

# Kelly fraction
KELLY_FRACTION = 0.75

# EV threshold - only place bets with EV above this percentage
MIN_EV_THRESHOLD = 0.05

# Z-score threshold for max bet
EV_MAX_BET_THRESHOLD = 0.2
ZSCORE_MAX_BET_THRESHOLD = 3.5



def kelly_bet(odds, fair_odds, ev=None, zscore=None, max_multiplier=5.0):
    """
    Kelly bet adjusted for parameter uncertainty via shrinkage coefficient.
    
    Args:
        odds: Decimal odds for the bet
        fair_odds: Fair odds calculated from probability
        ev: Expected value (optional, used for filtering)
        zscore: Z-score value (optional, triggers max bet if > threshold)
        max_multiplier: Maximum bet size as percentage of bankroll
    
    Returns:
        Bet size as percentage of bankroll
    """
    if pd.isna(odds) or odds <= 1:
        return 0
    
    # Check EV threshold if provided
    if ev is not None and not pd.isna(ev):
        if ev < MIN_EV_THRESHOLD:
            return 0
        if ev > EV_MAX_BET_THRESHOLD:
            return max_multiplier

    p = 1 / fair_odds
    p = max(min(p, 0.9999), 0.0001)
    b = odds - 1

    # Raw Kelly fraction
    f_kelly = p - ((1 - p) / b)

    # Only bet if Kelly is positive (positive edge)
    if f_kelly <= 0:
        return 0

    # Check if Z-score triggers max bet
    if zscore is not None and not pd.isna(zscore):
        if zscore >= ZSCORE_MAX_BET_THRESHOLD:
            return max_multiplier

    # Apply shrinkage coefficient
    shrink = KELLY_FRACTION
    f_adjusted = f_kelly * shrink

    return min(f_adjusted * 100, max_multiplier)


# ====================================
# === ROI Calculations ===============
# ====================================


def roi_flat(df, odds_col):
    """Calculate ROI using flat betting ($1 per bet)."""
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


def roi_ev(df, odds_col, fair_col, ev_col=None, zscore_col=None):
    """
    Calculate ROI using Kelly criterion with EV and Z-score considerations.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
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


def calculate_time_spent(df):
    """Calculate time interval data has been collected."""
    # Use pandas to_datetime for flexible parsing
    time_start = pd.to_datetime(df.loc[0, "Scrape Time"])
    time_end = pd.to_datetime(df.loc[len(df) - 1, "Scrape Time"])
    time_spent = time_end - time_start

    days = time_spent.days
    hours, remainder = divmod(time_spent.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


# ====================================
# === Comparison & Reporting =========
# ====================================


def compare_methods(strategy):
    """Compare flat betting vs Kelly betting for a strategy."""
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
    flat = roi_flat(df, strategy.odds_column)
    print(
        f"{'Flat':<15} {flat['Total Wagered']:>10.2f} {flat['Total Winnings']:>10.2f} "
        f"{flat['Net Profit']:>12.2f} {flat['ROI']:>8.2f} {flat['Total Wagered']:>6.0f}"
    )

    # Kelly betting
    if strategy.ev_column:
        kelly = roi_ev(
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
    print(f"Z-Score Max Bet Threshold: {ZSCORE_MAX_BET_THRESHOLD}")
    print("=" * 70)


def main():
    print_config()
    print("\n" + "=" * 70)
    print("COMPARISON: Flat vs Kelly")
    print("=" * 70)
    for strat in STRATEGIES:
        compare_methods(strat)


if __name__ == "__main__":
    main()
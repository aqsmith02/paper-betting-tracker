import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from codebase.results.results_configs import PENDING_RESULTS

# --- Data class for strategies ---
@dataclass
class BettingStrategy:
    name: str
    path: str
    odds_column: str
    edge_column: str = None


STRATEGIES = [
    BettingStrategy(
        "Average", "codebase/data/master_avg_bets.csv", "Avg Edge Odds", "Avg Edge Pct"
    ),
    BettingStrategy(
        "Modified Zscore",
        "codebase/data/master_mod_zscore_bets.csv",
        "Outlier Odds",
        "Avg Edge Pct",
    ),
    BettingStrategy(
        "Pinnacle",
        "codebase/data/master_pin_bets.csv",
        "Pinnacle Edge Odds",
        "Pin Edge Pct",
    ),
    BettingStrategy(
        "Zscore", "codebase/data/master_zscore_bets.csv", "Outlier Odds", "Avg Edge Pct"
    ),
    BettingStrategy(
        "Random", "codebase/data/master_random_bets.csv", "Random Bet Odds"
    ),
    BettingStrategy(
        "Average NC", "codebase/data/master_nc_avg_bets.csv", "Avg Edge Odds", "Avg Edge Pct"
    ),
    BettingStrategy(
        "Modified Zscore NC",
        "codebase/data/master_nc_mod_zscore_bets.csv",
        "Outlier Odds",
        "Avg Edge Pct",
    ),
    BettingStrategy(
        "Pinnacle NC",
        "codebase/data/master_nc_pin_bets.csv",
        "Pinnacle Edge Odds",
        "Pin Edge Pct",
    ),
    BettingStrategy(
        "Zscore NC", "codebase/data/master_nc_zscore_bets.csv", "Outlier Odds", "Avg Edge Pct"
    ),
    BettingStrategy(
        "Random NC", "codebase/data/master_nc_random_bets.csv", "Random Bet Odds"
    ),
]

# From rmse.py
MSE = 0.04

# ============================================================
# === Shrinkage coefficient based on Baker & McHale (2013) ===
# ============================================================


def shrinking_coefficient(p_hat, var_p_hat):
    """
    Compute Baker & McHale's shrinkage coefficient for adjusting Kelly bet
    based on uncertainty (variance) in estimated probabilities.

    Parameters
    ----------
    p_hat : float
        Estimated win probability.
    var_p_hat : float
        Variance of estimated probability.

    Returns
    -------
    float
        Shrinking coefficient between 0 and 1.
    """
    if (
        pd.isna(p_hat)
        or pd.isna(var_p_hat)
        or var_p_hat <= 0
        or p_hat <= 0
        or p_hat >= 1
    ):
        return 1.0
    shrink = 1 / (1 + var_p_hat / (p_hat**2 * (1 - p_hat) ** 2))
    return max(min(shrink, 1.0), 0.0)


# ====================================
# === Bet sizing functions ===========
# ====================================


def linear_bet(edge_pct, base_unit=1.0, max_multiplier=5.0):
    """Linear scaling of bet size by edge percentage."""
    if pd.isna(edge_pct) or edge_pct <= 0:
        return base_unit
    multiplier = min(1 + (edge_pct / 20) * (max_multiplier - 1), max_multiplier)
    return base_unit * multiplier


def kelly_bet(edge_pct, odds, max_multiplier=5.0):
    """
    Kelly bet adjusted for parameter uncertainty via shrinkage coefficient.
    """
    if pd.isna(edge_pct) or pd.isna(odds) or edge_pct <= 0 or odds <= 1:
        return 1.0

    # Derived probability from implied odds and edge
    p = 1 / odds + edge_pct / 100
    p = max(min(p, 0.9999), 0.0001)
    b = odds - 1

    # Raw Kelly fraction
    f_kelly = (b * p - (1 - p)) / b

    # Apply shrinkage coefficient
    # shrink = shrinking_coefficient(p, MSE)
    shrink = 0.25
    f_adjusted = max(f_kelly * shrink, 0)

    return min(f_adjusted * 100, max_multiplier)


# ====================================
# === ROI Calculations ===============
# ====================================


def roi_flat(df, odds_col):
    df = df[~df["Result"].isin(PENDING_RESULTS)]
    total_wagered = len(df)
    total_winnings = (df[odds_col] * (df["Team"] == df["Result"])).sum()
    net_profit = total_winnings - total_wagered
    return {
        "Total Wagered": total_wagered,
        "Total Winnings": total_winnings,
        "Net Profit": net_profit,
        "ROI": net_profit / total_wagered * 100,
    }


def roi_edge(df, odds_col, edge_col, method="linear"):
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()

    if method == "linear":
        df["Bet"] = df[edge_col].apply(linear_bet)

    elif method == "kelly":
        df["Bet"] = df.apply(
            lambda row: kelly_bet(
                edge_pct=row[edge_col],
                odds=row[odds_col],
            ),
            axis=1,
        )

    total_wagered = df["Bet"].sum()
    total_winnings = (df["Bet"] * (df[odds_col] * (df["Team"] == df["Result"]))).sum()
    net_profit = total_winnings - total_wagered
    return {
        "Total Wagered": total_wagered,
        "Total Winnings": total_winnings,
        "Net Profit": net_profit,
        "ROI": net_profit / total_wagered * 100,
    }


def calculate_time_spent(df):
    """Calculate time interval data has been collected."""
    time_start = datetime.strptime(df.loc[0, "Scrape Time"], "%Y-%m-%d %H:%M:%S")
    time_end = datetime.strptime(df.loc[len(df)-1, "Scrape Time"], "%Y-%m-%d %H:%M:%S")
    time_spent = time_end - time_start

    days = time_spent.days
    hours, remainder = divmod(time_spent.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


# ====================================
# === Comparison & Reporting =========
# ====================================


def compare_methods(strategy):
    df = pd.read_csv(strategy.path)
    time_spent = calculate_time_spent(df)
    
    print(f"\n=== {strategy.name} ===")
    print(f"Time Spent: {time_spent}")
    print(
        f"{'Method':<15} {'Wagered':>10} {'Winnings':>10} {'Net Profit':>12} {'ROI %':>8}"
    )
    print("-" * 60)

    # Flat betting
    flat = roi_flat(df, strategy.odds_column)
    print(
        f"{'Flat':<15} {flat['Total Wagered']:>10.2f} {flat['Total Winnings']:>10.2f} "
        f"{flat['Net Profit']:>12.2f} {flat['ROI']:>8.2f}"
    )

    # Edge-based methods
    if strategy.edge_column:
        linear = roi_edge(
            df, strategy.odds_column, strategy.edge_column, method="linear"
        )
        kelly = roi_edge(
            df,
            strategy.odds_column,
            strategy.edge_column,
            method="kelly",
        )

        print(
            f"{'Linear':<15} {linear['Total Wagered']:>10.2f} {linear['Total Winnings']:>10.2f} "
            f"{linear['Net Profit']:>12.2f} {linear['ROI']:>8.2f}"
        )

        print(
            f"{'Kelly':<15} {kelly['Total Wagered']:>10.2f} {kelly['Total Winnings']:>10.2f} "
            f"{kelly['Net Profit']:>12.2f} {kelly['ROI']:>8.2f}"
        )


def main():
    print("=" * 60)
    print("COMPARISON: Flat vs Linear vs Kelly (with Shrinkage)")
    print("=" * 60)
    for strat in STRATEGIES:
        compare_methods(strat)


if __name__ == "__main__":
    main()
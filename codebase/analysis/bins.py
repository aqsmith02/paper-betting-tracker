import pandas as pd
import numpy as np
from dataclasses import dataclass
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

# Bin definitions for analysis
EV_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.0]
EV_LABELS = ['0-2%', '2-5%', '5-10%', '10-15%', '15-20%', '20%+']

ZSCORE_BINS = [0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 10.0]
ZSCORE_LABELS = ['0-1.5', '1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-3.5', '3.5+']


def analyze_ev_bins(df, odds_col, ev_col):
    """
    Analyze ROI across different EV bins using flat betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        ev_col: Column name for expected value
    
    Returns:
        DataFrame with bin analysis results
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    df = df[df[ev_col].notna()].copy()
    
    # Create bins
    df['EV_Bin'] = pd.cut(df[ev_col], bins=EV_BINS, labels=EV_LABELS, include_lowest=True)
    
    results = []
    for bin_label in EV_LABELS:
        bin_df = df[df['EV_Bin'] == bin_label]
        
        if len(bin_df) == 0:
            continue
        
        total_wagered = len(bin_df)
        total_winnings = (bin_df[odds_col] * (bin_df["Team"] == bin_df["Result"])).sum()
        net_profit = total_winnings - total_wagered
        wins = (bin_df["Team"] == bin_df["Result"]).sum()
        
        results.append({
            'Bin': bin_label,
            'Count': total_wagered,
            'Wins': wins,
            'Win %': wins / total_wagered * 100,
            'Net Profit': net_profit,
            'ROI %': net_profit / total_wagered * 100,
            'Avg EV': bin_df[ev_col].mean() * 100,
        })
    
    return pd.DataFrame(results)


def analyze_zscore_bins(df, odds_col, zscore_col):
    """
    Analyze ROI across different Z-score bins using flat betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        zscore_col: Column name for z-score
    
    Returns:
        DataFrame with bin analysis results
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    df = df[df[zscore_col].notna()].copy()
    
    # Create bins
    df['ZScore_Bin'] = pd.cut(df[zscore_col], bins=ZSCORE_BINS, labels=ZSCORE_LABELS, include_lowest=True)
    
    results = []
    for bin_label in ZSCORE_LABELS:
        bin_df = df[df['ZScore_Bin'] == bin_label]
        
        if len(bin_df) == 0:
            continue
        
        total_wagered = len(bin_df)
        total_winnings = (bin_df[odds_col] * (bin_df["Team"] == bin_df["Result"])).sum()
        net_profit = total_winnings - total_wagered
        wins = (bin_df["Team"] == bin_df["Result"]).sum()
        
        results.append({
            'Bin': bin_label,
            'Count': total_wagered,
            'Wins': wins,
            'Win %': wins / total_wagered * 100,
            'Net Profit': net_profit,
            'ROI %': net_profit / total_wagered * 100,
            'Avg Z-Score': bin_df[zscore_col].mean(),
        })
    
    return pd.DataFrame(results)


def analyze_bins(strategy):
    """
    Perform bin analysis for EV and/or Z-score.
    
    Args:
        strategy: BettingStrategy object to analyze
    """
    df = pd.read_csv(strategy.path)
    
    print(f"\n{'='*70}")
    print(f"BIN ANALYSIS: {strategy.name}")
    print(f"{'='*70}")
    
    # EV Bin Analysis
    if strategy.ev_column:
        print(f"\n--- Expected Value Bins ---")
        ev_results = analyze_ev_bins(df, strategy.odds_column, strategy.ev_column)
        if not ev_results.empty:
            print(f"{'Bin':<10} {'Count':>7} {'Wins':>6} {'Win %':>7} {'Net Profit':>12} {'ROI %':>8} {'Avg EV %':>10}")
            print("-" * 70)
            for _, row in ev_results.iterrows():
                print(
                    f"{row['Bin']:<10} {row['Count']:>7.0f} {row['Wins']:>6.0f} "
                    f"{row['Win %']:>7.1f} {row['Net Profit']:>12.2f} "
                    f"{row['ROI %']:>8.2f} {row['Avg EV']:>10.2f}"
                )
        else:
            print("No data available for EV bin analysis")
    
    # Z-Score Bin Analysis
    if strategy.zscore_column:
        print(f"\n--- Z-Score Bins ---")
        zscore_results = analyze_zscore_bins(df, strategy.odds_column, strategy.zscore_column)
        if not zscore_results.empty:
            print(f"{'Bin':<10} {'Count':>7} {'Wins':>6} {'Win %':>7} {'Net Profit':>12} {'ROI %':>8} {'Avg Z':>8}")
            print("-" * 70)
            for _, row in zscore_results.iterrows():
                print(
                    f"{row['Bin']:<10} {row['Count']:>7.0f} {row['Wins']:>6.0f} "
                    f"{row['Win %']:>7.1f} {row['Net Profit']:>12.2f} "
                    f"{row['ROI %']:>8.2f} {row['Avg Z-Score']:>8.2f}"
                )
        else:
            print("No data available for Z-Score bin analysis")


def print_config():
    """Print current bin configuration."""
    print("=" * 70)
    print("BIN CONFIGURATION")
    print("=" * 70)
    print(f"EV Bins: {EV_LABELS}")
    print(f"Z-Score Bins: {ZSCORE_LABELS}")
    print("=" * 70)


def main():
    print_config()
    
    print("\n" + "=" * 70)
    print("DETAILED BIN ANALYSIS")
    print("=" * 70)
    
    for strat in STRATEGIES:
        if strat.ev_column or strat.zscore_column:
            analyze_bins(strat)


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
MIN_EV_THRESHOLD = 0.05  # 2% minimum EV to place bet

# Z-score threshold for max bet
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


def calculate_cumulative_profit_flat(df, odds_col):
    """
    Calculate cumulative profit over time using flat betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
    
    Returns:
        DataFrame with cumulative profit information
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Sort by start time
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df = df.sort_values('Start Time').reset_index(drop=True)
    
    # Calculate profit for each bet (flat $1 betting)
    df['Bet_Size'] = 1.0
    df['Profit'] = df.apply(
        lambda row: (row[odds_col] - 1) if row['Team'] == row['Result'] else -1,
        axis=1
    )
    
    # Calculate cumulative profit
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    df['Bet_Number'] = range(1, len(df) + 1)
    
    return df[['Start Time', 'Bet_Number', 'Bet_Size', 'Profit', 'Cumulative_Profit']]


def calculate_cumulative_profit_kelly(df, odds_col, fair_col, ev_col=None, zscore_col=None):
    """
    Calculate cumulative profit over time using Kelly betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
    
    Returns:
        DataFrame with cumulative profit information
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Sort by start time
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df = df.sort_values('Start Time').reset_index(drop=True)
    
    # Calculate Kelly bet size for each bet
    df['Bet_Size'] = df.apply(
        lambda row: kelly_bet(
            odds=row[odds_col],
            fair_odds=row[fair_col],
            ev=row.get(ev_col) if ev_col else None,
            zscore=row.get(zscore_col) if zscore_col else None,
        ),
        axis=1,
    )
    
    # Calculate profit for each bet
    df['Profit'] = df.apply(
        lambda row: row['Bet_Size'] * (row[odds_col] - 1) if row['Team'] == row['Result'] 
                    else -row['Bet_Size'],
        axis=1
    )
    
    # Calculate cumulative profit
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    
    # Only count bets where Kelly says to bet
    df_bets = df[df['Bet_Size'] > 0].copy()
    df_bets['Bet_Number'] = range(1, len(df_bets) + 1)
    
    return df_bets[['Start Time', 'Bet_Number', 'Bet_Size', 'Profit', 'Cumulative_Profit']]


def plot_profit_over_time(strategy, save_fig=False):
    """
    Create profit over time visualization for a strategy.
    
    Args:
        strategy: BettingStrategy object
        save_fig: Whether to save the figure to disk
    """
    df = pd.read_csv(strategy.path)
    
    # Calculate profits for both methods
    flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'Profit Over Time: {strategy.name}', fontsize=16, fontweight='bold')
    
    # Plot 1: Flat Betting
    axes[0].plot(flat_profit['Bet_Number'], flat_profit['Cumulative_Profit'], 
                 linewidth=2, color='#2E86AB', label='Flat Betting')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Bet Number', fontsize=12)
    axes[0].set_ylabel('Cumulative Profit ($)', fontsize=12)
    axes[0].set_title('Flat Betting ($1 per bet)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)
    
    # Add summary statistics
    final_profit_flat = flat_profit['Cumulative_Profit'].iloc[-1]
    total_bets_flat = len(flat_profit)
    roi_flat = (final_profit_flat / total_bets_flat) * 100
    
    axes[0].text(0.02, 0.98, 
                f'Final Profit: ${final_profit_flat:.2f}\n'
                f'Total Bets: {total_bets_flat}\n'
                f'ROI: {roi_flat:.2f}%',
                transform=axes[0].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    
    # Plot 2: Kelly Betting (if applicable)
    if strategy.ev_column:
        kelly_profit = calculate_cumulative_profit_kelly(
            df, 
            strategy.odds_column, 
            strategy.fair_odds_column,
            strategy.ev_column,
            strategy.zscore_column
        )
        
        if not kelly_profit.empty:
            axes[1].plot(kelly_profit['Bet_Number'], kelly_profit['Cumulative_Profit'], 
                        linewidth=2, color='#A23B72', label='Kelly Betting')
            axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[1].set_xlabel('Bet Number', fontsize=12)
            axes[1].set_ylabel('Cumulative Profit ($)', fontsize=12)
            axes[1].set_title('Kelly Betting (Variable Bet Sizing)', fontsize=13, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(fontsize=11)
            
            # Add summary statistics
            final_profit_kelly = kelly_profit['Cumulative_Profit'].iloc[-1]
            total_wagered_kelly = kelly_profit['Bet_Size'].sum()
            total_bets_kelly = len(kelly_profit)
            roi_kelly = (final_profit_kelly / total_wagered_kelly) * 100
            
            axes[1].text(0.02, 0.98, 
                        f'Final Profit: ${final_profit_kelly:.2f}\n'
                        f'Total Wagered: ${total_wagered_kelly:.2f}\n'
                        f'Bets Placed: {total_bets_kelly}\n'
                        f'ROI: {roi_kelly:.2f}%',
                        transform=axes[1].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                        fontsize=10)
        else:
            axes[1].text(0.5, 0.5, 'No Kelly bets placed (all filtered out)',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[1].transAxes,
                        fontsize=14)
            axes[1].set_xlabel('Bet Number', fontsize=12)
            axes[1].set_ylabel('Cumulative Profit ($)', fontsize=12)
            axes[1].set_title('Kelly Betting (Variable Bet Sizing)', fontsize=13, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Kelly betting not available (no EV column)',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1].transAxes,
                    fontsize=14)
        axes[1].set_xlabel('Bet Number', fontsize=12)
        axes[1].set_ylabel('Cumulative Profit ($)', fontsize=12)
        axes[1].set_title('Kelly Betting (Variable Bet Sizing)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"codebase/analysis/profit_over_time_{strategy.name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
    
    # plt.show()


def plot_comparison_all_strategies(save_fig=False):
    """
    Create a comparison plot showing all strategies on the same chart.
    
    Args:
        save_fig: Whether to save the figure to disk
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Profit Over Time: All Strategies Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot 1: Flat Betting Comparison
    for i, strategy in enumerate(STRATEGIES):
        df = pd.read_csv(strategy.path)
        flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
        
        axes[0].plot(flat_profit['Bet_Number'], flat_profit['Cumulative_Profit'], 
                    linewidth=2, color=colors[i % len(colors)], label=strategy.name, alpha=0.8)
    
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_xlabel('Bet Number', fontsize=12)
    axes[0].set_ylabel('Cumulative Profit ($)', fontsize=12)
    axes[0].set_title('Flat Betting Comparison', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    
    # Plot 2: Kelly Betting Comparison
    for i, strategy in enumerate(STRATEGIES):
        if strategy.ev_column:
            df = pd.read_csv(strategy.path)
            kelly_profit = calculate_cumulative_profit_kelly(
                df, 
                strategy.odds_column, 
                strategy.fair_odds_column,
                strategy.ev_column,
                strategy.zscore_column
            )
            
            if not kelly_profit.empty:
                axes[1].plot(kelly_profit['Bet_Number'], kelly_profit['Cumulative_Profit'], 
                            linewidth=2, color=colors[i % len(colors)], label=strategy.name, alpha=0.8)
    
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_xlabel('Bet Number', fontsize=12)
    axes[1].set_ylabel('Cumulative Profit ($)', fontsize=12)
    axes[1].set_title('Kelly Betting Comparison', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        filename = "codebase/analysis/profit_over_time_all_strategies.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
    
    # plt.show()


def print_config():
    """Print current configuration parameters."""
    print("=" * 70)
    print("PROFIT OVER TIME VISUALIZATION - CONFIGURATION")
    print("=" * 70)
    print(f"Kelly Fraction: {KELLY_FRACTION}")
    print(f"Minimum EV Threshold: {MIN_EV_THRESHOLD:.2%}")
    print(f"Z-Score Max Bet Threshold: {ZSCORE_MAX_BET_THRESHOLD}")
    print("=" * 70)


def main():
    print_config()
    
    print("\nGenerating profit over time visualizations...")
    print("=" * 70)
    
    # Option 1: Plot each strategy individually
    for strategy in STRATEGIES:
        print(f"\nPlotting: {strategy.name}")
        plot_profit_over_time(strategy, save_fig=True)
    
    # Option 2: Plot all strategies together for comparison
    print("\nPlotting comparison of all strategies...")
    plot_comparison_all_strategies(save_fig=True)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
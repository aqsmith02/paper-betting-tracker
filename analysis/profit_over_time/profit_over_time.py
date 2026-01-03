import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dataclasses import dataclass
from datetime import datetime
from src.constants import PENDING_RESULTS


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
        "Average",
        "data/master_nc_avg_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
    ),
    BettingStrategy(
        "Average With Modified Zscore Constraint",
        "data/master_nc_mod_zscore_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
        "Modified Z Score",
    ),
    BettingStrategy(
        "Pinnacle",
        "data/master_nc_pin_full.csv",
        "Best Odds",
        "Pinnacle Fair Odds",
        "Expected Value",
    ),
    BettingStrategy(
        "Average With Zscore Constraint",
        "data/master_nc_zscore_full.csv",
        "Best Odds",
        "Fair Odds Avg",
        "Expected Value",
        "Z Score",
    ),
    BettingStrategy(
        "Random Strategy", "data/master_nc_random_full.csv", "Best Odds"
    ),
]

# Kelly fraction
KELLY_FRACTION = 0.5

# EV threshold - only place bets with EV between these percentages
MIN_EV_THRESHOLD = 0.05
MAX_EV_THRESHOLD = 0.35

# Z-score threshold for max bet
ZSCORE_MAX_BET_THRESHOLD = 3.5


def kelly_bet(odds, fair_odds, ev=None, zscore=None, max_multiplier=2.5):
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
        if ev > MAX_EV_THRESHOLD:
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
    For Random strategy: shows flat betting
    For others: shows Kelly betting only
    
    Args:
        strategy: BettingStrategy object
        save_fig: Whether to save the figure to disk
    """
    df = pd.read_csv(strategy.path)
    
    # Get current date
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Get first bet date (data collection start date)
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    first_bet_date = df['Start Time'].min().strftime('%B %d, %Y')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Check if this is the Random strategy (no EV column)
    if not strategy.ev_column:
        # Plot flat betting for Random strategy
        fig.suptitle(f'Flat Betting Profit Over Time: {strategy.name}', 
                     fontsize=16, fontweight='bold')
        
        flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
        
        ax.plot(flat_profit['Start Time'], flat_profit['Cumulative_Profit'], 
                linewidth=2.5, color='#2E86AB', label='Flat Betting')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
        ax.set_xlabel('Date', fontsize=13)
        ax.set_ylabel('Cumulative Profit (Units)', fontsize=13)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11)
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add summary statistics
        final_profit_flat = flat_profit['Cumulative_Profit'].iloc[-1]
        total_bets_flat = len(flat_profit)
        roi_flat = (final_profit_flat / total_bets_flat) * 100
        
        ax.text(0.02, 0.98, 
                f'Final Profit: {final_profit_flat:.2f} units\n'
                f'Total Bets: {total_bets_flat}\n'
                f'ROI: {roi_flat:.2f}%\n'
                f'Data Collection Started: {first_bet_date}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10)
    else:
        # Plot Kelly betting for other strategies
        fig.suptitle(f'1/2 Kelly Criterion Profit Over Time: {strategy.name}', 
                     fontsize=16, fontweight='bold')
        
        kelly_profit = calculate_cumulative_profit_kelly(
            df, 
            strategy.odds_column, 
            strategy.fair_odds_column,
            strategy.ev_column,
            strategy.zscore_column
        )
        
        if not kelly_profit.empty:
            ax.plot(kelly_profit['Start Time'], kelly_profit['Cumulative_Profit'], 
                    linewidth=2.5, color='#A23B72', label='Kelly Betting')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
            ax.set_xlabel('Date', fontsize=13)
            ax.set_ylabel('Cumulative Profit (Units)', fontsize=13)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=11)
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
            # Add summary statistics
            final_profit_kelly = kelly_profit['Cumulative_Profit'].iloc[-1]
            total_wagered_kelly = kelly_profit['Bet_Size'].sum()
            total_bets_kelly = len(kelly_profit)
            roi_kelly = (final_profit_kelly / total_wagered_kelly) * 100
            
            ax.text(0.02, 0.98, 
                    f'Final Profit: {final_profit_kelly:.2f} units\n'
                    f'Total Wagered: {total_wagered_kelly:.2f} units\n'
                    f'Bets Placed: {total_bets_kelly}\n'
                    f'ROI: {roi_kelly:.2f}%\n'
                    f'Data Collection Started: {first_bet_date}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No Kelly bets placed (all filtered out)',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14)
            ax.set_xlabel('Date', fontsize=13)
            ax.set_ylabel('Cumulative Profit (Units)', fontsize=13)
    
    # Add date stamp in bottom right corner (LARGER)
    ax.text(0.98, 0.02, 
            f'Generated: {current_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=14,
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = f"analysis/profit_over_time/profit_over_time_{strategy.name.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {filename}")
    
    # plt.show()


def plot_comparison_all_strategies(save_fig=False):
    """
    Create a comparison plot showing all strategies (Kelly for most, flat for Random).
    
    Args:
        save_fig: Whether to save the figure to disk
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get current date
    current_date = datetime.now().strftime('%B %d, %Y')
    
    # Get earliest start date across all strategies
    earliest_date = None
    for strategy in STRATEGIES:
        df = pd.read_csv(strategy.path)
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        strategy_start = df['Start Time'].min()
        if earliest_date is None or strategy_start < earliest_date:
            earliest_date = strategy_start
    
    first_bet_date = earliest_date.strftime('%B %d, %Y')
    
    fig.suptitle(f'Profit Over Time - All Strategies', 
                 fontsize=16, fontweight='bold')
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # Plot all strategies
    for i, strategy in enumerate(STRATEGIES):
        df = pd.read_csv(strategy.path)
        
        if strategy.ev_column:
            # Plot Kelly betting for strategies with EV column
            kelly_profit = calculate_cumulative_profit_kelly(
                df, 
                strategy.odds_column, 
                strategy.fair_odds_column,
                strategy.ev_column,
                strategy.zscore_column
            )
            
            if not kelly_profit.empty:
                ax.plot(kelly_profit['Start Time'], kelly_profit['Cumulative_Profit'], 
                        linewidth=2.5, color=colors[i % len(colors)], label=strategy.name, alpha=0.85)
        else:
            # Plot flat betting for Random strategy
            flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
            
            if not flat_profit.empty:
                ax.plot(flat_profit['Start Time'], flat_profit['Cumulative_Profit'], 
                        linewidth=2.5, color=colors[i % len(colors)], label=strategy.name, alpha=0.85, linestyle='--')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Cumulative Profit (Units)', fontsize=13)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Move legend to upper left to avoid overlapping with date boxes
    ax.legend(fontsize=11, loc='upper left')
    
    # Add data collection start date in bottom left corner
    ax.text(0.02, 0.02, 
            f'Data Collection Started: {first_bet_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=12,
            fontweight='bold')
    
    # Add date stamp in bottom right corner (LARGER)
    ax.text(0.98, 0.02, 
            f'Generated: {current_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=14,
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = "analysis/profit_over_time/profit_over_time_all_strategies.png"
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
    # Kelly betting for strategies with EV, flat betting for Random
    for strategy in STRATEGIES:
        print(f"\nPlotting: {strategy.name}")
        plot_profit_over_time(strategy, save_fig=True)
    
    # Option 2: Plot all strategies together for comparison (Kelly + flat for Random)
    print("\nPlotting comparison of all strategies...")
    plot_comparison_all_strategies(save_fig=True)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
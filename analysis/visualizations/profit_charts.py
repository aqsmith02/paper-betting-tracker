"""
Profit Over Time Visualizations

Create charts showing cumulative profit over time for betting strategies.
Uses calculation functions from calculations/profit_calculator.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
from config.analysis_config import (
    KELLY_FRACTION,
    MIN_EV_THRESHOLD,
    MAX_EV_THRESHOLD,
    ZSCORE_MAX_BET_THRESHOLD,
    FIGURE_SIZE_DEFAULT,
    DPI,
    COLORS,
    DATE_FORMAT,
    FONT_SIZE_TITLE,
    FONT_SIZE_AXIS_LABEL,
    FONT_SIZE_LEGEND,
    FONT_SIZE_ANNOTATION,
    FONT_SIZE_DATE_STAMP,
    get_output_path,
    format_output_filename,
)
from analysis.strategy_definitions import ALL_STRATEGIES
from analysis.calculations.profit_over_time import (
    calculate_cumulative_profit_flat,
    calculate_cumulative_profit_kelly,
    calculate_profit_summary,
)

EASTERN_TZ = pytz.timezone('America/New_York')


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
    current_date = datetime.now(EASTERN_TZ).strftime('%B %d, %Y')
    
    # Get first bet date (data collection start date)
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    first_bet_date = df['Start Time'].min().strftime('%B %d, %Y')
    
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_DEFAULT)
    
    # Check if this is the Random strategy (no EV column)
    if not strategy.ev_column:
        # Plot flat betting for Random strategy
        fig.suptitle(f'Flat Betting Profit Over Time: {strategy.name}', 
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
        
        flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
        summary = calculate_profit_summary(flat_profit, 'flat')
        
        ax.plot(flat_profit['Start Time'], flat_profit['Cumulative_Profit'], 
                linewidth=2.5, color=COLORS[0], label='Flat Betting')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
        ax.set_xlabel('Date', fontsize=FONT_SIZE_AXIS_LABEL)
        ax.set_ylabel('Cumulative Profit (Units)', fontsize=FONT_SIZE_AXIS_LABEL)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=FONT_SIZE_LEGEND)
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()
        
        # Add summary statistics
        ax.text(0.02, 0.98, 
                f'Final Profit: {summary["final_profit"]:.2f} units\n'
                f'Total Bets: {summary["total_bets"]}\n'
                f'ROI: {summary["roi"]:.2f}%\n'
                f'Data Collection Started: {first_bet_date}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=FONT_SIZE_ANNOTATION)
    else:
        # Plot Kelly betting for other strategies
        fig.suptitle(f'Kelly Criterion Profit Over Time: {strategy.name}', 
                     fontsize=FONT_SIZE_TITLE, fontweight='bold')
        
        kelly_profit = calculate_cumulative_profit_kelly(
            df, 
            strategy.odds_column, 
            strategy.fair_odds_column,
            strategy.ev_column,
            strategy.zscore_column
        )
        
        if not kelly_profit.empty:
            summary = calculate_profit_summary(kelly_profit, 'kelly')
            
            ax.plot(kelly_profit['Start Time'], kelly_profit['Cumulative_Profit'], 
                    linewidth=2.5, color=COLORS[1], label='Kelly Betting')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
            ax.set_xlabel('Date', fontsize=FONT_SIZE_AXIS_LABEL)
            ax.set_ylabel('Cumulative Profit (Units)', fontsize=FONT_SIZE_AXIS_LABEL)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(fontsize=FONT_SIZE_LEGEND)
            
            # Format x-axis for dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()
            
            # Add summary statistics
            ax.text(0.02, 0.98, 
                    f'Final Profit: {summary["final_profit"]:.2f} units\n'
                    f'Total Wagered: {summary["total_wagered"]:.2f} units\n'
                    f'Bets Placed: {summary["total_bets"]}\n'
                    f'ROI: {summary["roi"]:.2f}%\n'
                    f'Data Collection Started: {first_bet_date}',
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    fontsize=FONT_SIZE_ANNOTATION)
        else:
            ax.text(0.5, 0.5, 'No Kelly bets placed (all filtered out)',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax.transAxes,
                    fontsize=14)
            ax.set_xlabel('Date', fontsize=FONT_SIZE_AXIS_LABEL)
            ax.set_ylabel('Cumulative Profit (Units)', fontsize=FONT_SIZE_AXIS_LABEL)
    
    # Add date stamp in bottom right corner
    ax.text(0.98, 0.02, 
            f'Generated: {current_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=FONT_SIZE_DATE_STAMP,
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = format_output_filename('profit_chart', strategy_name=strategy.name)
        filepath = get_output_path('profit_charts', filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
    
    plt.close()


def plot_comparison_all_strategies(strategies=None, save_fig=False):
    """
    Create a comparison plot showing all strategies (Kelly for most, flat for Random).
    
    Args:
        strategies: List of BettingStrategy objects (default: ALL_STRATEGIES)
        save_fig: Whether to save the figure to disk
    """
    if strategies is None:
        strategies = ALL_STRATEGIES
    
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_DEFAULT)
    
    # Get current date
    current_date = datetime.now(EASTERN_TZ).strftime('%B %d, %Y')
    
    # Get earliest start date across all strategies
    earliest_date = None
    for strategy in strategies:
        df = pd.read_csv(strategy.path)
        df['Start Time'] = pd.to_datetime(df['Start Time'])
        strategy_start = df['Start Time'].min()
        if earliest_date is None or strategy_start < earliest_date:
            earliest_date = strategy_start
    
    first_bet_date = earliest_date.strftime('%B %d, %Y')
    
    fig.suptitle(f'Profit Over Time - All Strategies', 
                 fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    # Plot all strategies
    for i, strategy in enumerate(strategies):
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
                        linewidth=2.5, color=COLORS[i % len(COLORS)], label=strategy.name, alpha=0.85)
        else:
            # Plot flat betting for Random strategy
            flat_profit = calculate_cumulative_profit_flat(df, strategy.odds_column)
            
            if not flat_profit.empty:
                ax.plot(flat_profit['Start Time'], flat_profit['Cumulative_Profit'], 
                        linewidth=2.5, color=COLORS[i % len(COLORS)], label=strategy.name, alpha=0.85, linestyle='--')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Break-even')
    ax.set_xlabel('Date', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Cumulative Profit (Units)', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMAT))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    
    # Move legend to upper left to avoid overlapping with date boxes
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left')
    
    # Add data collection start date in bottom left corner
    ax.text(0.02, 0.02, 
            f'Data Collection Started: {first_bet_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=12,
            fontweight='bold')
    
    # Add date stamp in bottom right corner
    ax.text(0.98, 0.02, 
            f'Generated: {current_date}',
            transform=ax.transAxes,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=1.5),
            fontsize=FONT_SIZE_DATE_STAMP,
            fontweight='bold')
    
    plt.tight_layout()
    
    if save_fig:
        filename = format_output_filename('comparison_chart')
        filepath = get_output_path('profit_charts', filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
    
    plt.close()


def print_config():
    """Print current configuration parameters."""
    print("=" * 70)
    print("PROFIT OVER TIME VISUALIZATION - CONFIGURATION")
    print("=" * 70)
    print(f"Kelly Fraction: {KELLY_FRACTION}")
    print(f"EV Thresholds: {MIN_EV_THRESHOLD:.2%} - {MAX_EV_THRESHOLD:.2%}")
    print(f"Z-Score Max Bet Threshold: {ZSCORE_MAX_BET_THRESHOLD}")
    print(f"Figure Size: {FIGURE_SIZE_DEFAULT}")
    print(f"DPI: {DPI}")
    print("=" * 70)


def main():
    """Generate all profit over time visualizations."""
    print_config()
    
    print("\nGenerating profit over time visualizations...")
    print("=" * 70)
    
    # Plot each strategy individually
    for strategy in ALL_STRATEGIES:
        print(f"\nPlotting: {strategy.name}")
        plot_profit_over_time(strategy, save_fig=True)
    
    # Plot all strategies together for comparison
    print("\nPlotting comparison of all strategies...")
    plot_comparison_all_strategies(save_fig=True)
    
    print("\n" + "=" * 70)
    print("Visualization complete!")


if __name__ == "__main__":
    main()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from codebase.results.results_configs import PENDING_RESULTS
from codebase.analysis.roi import (
    STRATEGIES,
    kelly_bet,
)


def calculate_cumulative_profit(df, odds_col, edge_col=None):
    """
    Calculate cumulative profit over time for a given betting method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Betting data with results
    odds_col : str
        Column name for odds
    edge_col : str, optional
        Column name for edge percentage
    
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime and cumulative profit
    """
    # Filter out pending results
    df_settled = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Convert scrape time to datetime
    df_settled["DateTime"] = pd.to_datetime(df_settled["Scrape Time"])
    df_settled = df_settled.sort_values("DateTime")
    
    # Calculate bet sizes
    df_settled["Bet"] = df_settled.apply(
        lambda row: kelly_bet(
            edge_pct=row[edge_col],
            odds=row[odds_col],
        ),
        axis=1,
    )
    
    # Calculate profit for each bet
    df_settled["Profit"] = df_settled.apply(
        lambda row: row["Bet"] * row[odds_col] - row["Bet"] if row["Team"] == row["Result"] else -row["Bet"],
        axis=1
    )
    
    # Calculate cumulative profit
    df_settled["Cumulative_Profit"] = df_settled["Profit"].cumsum()
    
    return df_settled[["DateTime", "Cumulative_Profit"]]


def plot_profit_over_time(strategy, save_path=None):
    """
    Plot cumulative profit over time using Kelly criterion.
    
    Parameters
    ----------
    strategy : BettingStrategy
        Strategy to analyze
    save_path : str, optional
        Path to save the plot
    """
    df = pd.read_csv(strategy.path)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Kelly criterion only if edge column exists
    if strategy.edge_column:
        kelly_profit = calculate_cumulative_profit(
            df, strategy.odds_column, strategy.edge_column
        )
        ax.plot(kelly_profit["DateTime"], kelly_profit["Cumulative_Profit"], 
                label="Kelly Criterion", linewidth=2.5, alpha=0.8, color='#2E86AB')
    else:
        print(f"Skipping {strategy.name} - no edge column available")
        plt.close()
        return
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Profit (Units)", fontsize=12, fontweight='bold')
    ax.set_title(f"Kelly Criterion Profit Over Time: {strategy.name}", fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_all_strategies_comparison(strategies, save_path=None):
    """
    Create a multi-panel plot comparing all strategies using Kelly criterion.
    
    Parameters
    ----------
    strategies : list
        List of BettingStrategy objects
    save_path : str, optional
        Path to save the plot
    """
    # Filter to only strategies with edge columns
    valid_strategies = [s for s in strategies if s.edge_column]
    
    n_strategies = len(valid_strategies)
    n_cols = 2
    n_rows = (n_strategies + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    for idx, strategy in enumerate(valid_strategies):
        ax = axes[idx]
        df = pd.read_csv(strategy.path)
        
        # Plot Kelly criterion only
        kelly_profit = calculate_cumulative_profit(
            df, strategy.odds_column, strategy.edge_column
        )
        ax.plot(kelly_profit["DateTime"], kelly_profit["Cumulative_Profit"], 
                label="Kelly", linewidth=2.5, alpha=0.8, color='#2E86AB')
        
        # Formatting
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Cumulative Profit (Units)", fontsize=10)
        ax.set_title(strategy.name, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    print("=" * 60)
    print("GENERATING KELLY CRITERION PROFIT PLOTS...")
    print("=" * 60)
    
    # Generate individual plots for each strategy (Kelly only)
    for strat in STRATEGIES:
        if strat.edge_column:  # Only plot strategies with edge columns
            safe_name = strat.name.replace(" ", "_").replace("/", "_")
            print(f"\nGenerating Kelly plot for {strat.name}...")
            plot_profit_over_time(strat)
    
    # Generate comparison plot (all strategies, Kelly only)
    # print("\nGenerating multi-panel Kelly comparison plot...")
    # plot_all_strategies_comparison(STRATEGIES)
    
    print("\n" + "=" * 60)
    print("All Kelly criterion plots generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
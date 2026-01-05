"""
Monte Carlo Visualization

Create histogram plots showing distribution of simulated results compared
to actual results for Monte Carlo simulations.

Uses calculation functions from calculations/monte_carlo_simulator.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from config.analysis_config import (
    KELLY_FRACTION,
    MIN_EV_THRESHOLD,
    MAX_EV_THRESHOLD,
    N_SIMULATIONS,
    NULL_EV,
    FIGURE_SIZE_MONTE_CARLO,
    DPI,
    get_output_path,
    format_output_filename,
)
from analysis.strategy_definitions import NON_RANDOM_STRATEGIES
from analysis.calculations.monte_carlo_sim import (
    monte_carlo_simulation,
    print_simulation_results,
)


def plot_simulation_results(results, strategy_name, save_fig=False):
    """
    Plot histogram of simulated profits with actual result marked.
    
    Args:
        results: Dictionary returned from monte_carlo_simulation()
        strategy_name: Name of the strategy for title
        save_fig: Whether to save the figure to disk
    """
    if results is None:
        print(f"No bets placed for {strategy_name}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_MONTE_CARLO)
    
    # Plot profit distribution
    ax.hist(results["simulated_profits"], bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(results["actual_profit"], color='red', linestyle='--', linewidth=2, 
                label=f'Actual: ${results["actual_profit"]:.2f}')
    ax.axvline(results["simulated_mean_profit"], color='blue', linestyle='--', linewidth=2,
                label=f'Expected: ${results["simulated_mean_profit"]:.2f}')
    ax.set_xlabel('Total Profit ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{strategy_name} Strategy\nNull Hypothesis (-5% EV) Profit Distribution (p={results["p_value"]:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f'n = {results["n_bets"]} bets\n'
        f'Wagered: ${results["total_wagered"]:.2f}\n'
        f'ROI: {results["actual_roi"]:.2f}%'
    )
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    if save_fig:
        filename = format_output_filename('monte_carlo_chart', strategy_name=strategy_name)
        filepath = get_output_path('monte_carlo', filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"Saved figure: {filepath}")
    
    plt.close()


def print_config():
    """Print current configuration parameters."""
    print("=" * 70)
    print("MONTE CARLO SIMULATION - CONFIGURATION")
    print("=" * 70)
    print(f"Null Hypothesis: Expected Value = {NULL_EV:.1%}")
    print(f"Number of Simulations: {N_SIMULATIONS:,}")
    print(f"Kelly Fraction: {KELLY_FRACTION}")
    print(f"EV Threshold: [{MIN_EV_THRESHOLD:.1%}, {MAX_EV_THRESHOLD:.1%}]")
    print("=" * 70)


def main():
    """Run Monte Carlo simulation for all strategies with visualization."""
    print_config()
    
    for strategy in NON_RANDOM_STRATEGIES:
        try:
            df = pd.read_csv(strategy.path)
            results = monte_carlo_simulation(
                df,
                strategy.odds_column,
                strategy.fair_odds_column,
                strategy.ev_column,
                strategy.zscore_column,
            )
            
            # Print results to console
            print_simulation_results(strategy.name, results)
            
            # Generate plots
            if results:
                plot_simulation_results(results, strategy.name, save_fig=True)
            
        except FileNotFoundError:
            print(f"\n=== {strategy.name} ===")
            print(f"File not found: {strategy.path}")
        except Exception as e:
            print(f"\n=== {strategy.name} ===")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
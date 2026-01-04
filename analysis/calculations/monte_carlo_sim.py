"""
Monte Carlo Simulator

Run Monte Carlo simulations to test null hypothesis of betting strategies.
Tests whether observed results are statistically significant or could be
explained by random chance with a specific expected value (bookmaker edge).

This module contains pure calculation logic. For visualization, see
visualizations/monte_carlo_plots.py
"""

import numpy as np
from src.constants import PENDING_RESULTS
from config.analysis_config import (
    N_SIMULATIONS,
    NULL_EV,
    RANDOM_SEED,
    ALPHA_LEVELS,
)
from analysis.betting_utils import kelly_bet, calculate_null_probability


def monte_carlo_simulation(df, odds_col, fair_col, ev_col=None, zscore_col=None, 
                           n_simulations=None, null_ev=None, random_seed=None):
    """
    Run Monte Carlo simulation to test null hypothesis using calculated probabilities.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
        n_simulations: Number of simulations to run (default from config)
        null_ev: Expected value under null hypothesis (default from config)
        random_seed: Random seed for reproducibility (default from config)
    
    Returns:
        Dictionary with simulation results and statistical tests, or None if no bets
    """
    # Use config defaults if not provided
    if n_simulations is None:
        n_simulations = N_SIMULATIONS
    if null_ev is None:
        null_ev = NULL_EV
    if random_seed is None:
        random_seed = RANDOM_SEED
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Filter out pending results
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Calculate bet sizes using Kelly criterion
    df["Bet_Size"] = df.apply(
        lambda row: kelly_bet(
            odds=row[odds_col],
            fair_odds=row[fair_col],
            ev=row.get(ev_col) if ev_col else None,
            zscore=row.get(zscore_col) if zscore_col else None,
        ),
        axis=1,
    )
    
    # Filter to only bets that would be placed (Bet_Size > 0)
    df_bets = df[df["Bet_Size"] > 0].copy()
    
    if len(df_bets) == 0:
        return None
    
    # Calculate null hypothesis probability for each bet
    df_bets["Null_Win_Prob"] = df_bets[odds_col].apply(
        lambda odds: calculate_null_probability(odds, null_ev=null_ev)
    )
    
    # Calculate actual profit
    df_bets["Actual_Win"] = (df_bets["Team"] == df_bets["Result"]).astype(int)
    df_bets["Actual_Profit"] = df_bets.apply(
        lambda row: row["Bet_Size"] * (row[odds_col] - 1) if row["Actual_Win"] 
                    else -row["Bet_Size"],
        axis=1
    )
    
    actual_total_profit = df_bets["Actual_Profit"].sum()
    actual_total_wagered = df_bets["Bet_Size"].sum()
    actual_roi = (actual_total_profit / actual_total_wagered * 100) if actual_total_wagered > 0 else 0
    
    # Run Monte Carlo simulations
    simulated_profits = []
    
    for _ in range(n_simulations):
        # For each bet, randomly determine win/loss based on null probability
        random_outcomes = np.random.random(len(df_bets)) < df_bets["Null_Win_Prob"].values
        
        # Calculate profit for this simulation
        sim_profits = np.where(
            random_outcomes,
            df_bets["Bet_Size"].values * (df_bets[odds_col].values - 1),
            -df_bets["Bet_Size"].values
        )
        
        total_sim_profit = sim_profits.sum()
        simulated_profits.append(total_sim_profit)
    
    simulated_profits = np.array(simulated_profits)
    
    # Calculate p-value (one-tailed test - are actual results better than expected?)
    # What proportion of simulations are as extreme or more extreme than actual?
    p_value = np.mean(simulated_profits >= actual_total_profit)
    
    # Calculate confidence intervals
    ci_95_profit = np.percentile(simulated_profits, [2.5, 97.5])
    
    # Test rejection at different alpha levels
    rejection_tests = {}
    for alpha in ALPHA_LEVELS:
        rejection_tests[f'reject_null_{int(alpha * 100):02d}'] = p_value < alpha
    
    return {
        "n_bets": len(df_bets),
        "total_wagered": actual_total_wagered,
        "actual_profit": actual_total_profit,
        "actual_roi": actual_roi,
        "simulated_mean_profit": np.mean(simulated_profits),
        "simulated_std_profit": np.std(simulated_profits),
        "ci_95_profit": ci_95_profit,
        "p_value": p_value,
        **rejection_tests,  # reject_null_05, reject_null_01, etc.
        "simulated_profits": simulated_profits,
        "null_ev": null_ev,
        "n_simulations": n_simulations,
    }


def print_simulation_results(strategy_name, results):
    """
    Print formatted simulation results to console.
    
    Args:
        strategy_name: Name of the strategy
        results: Dictionary returned from monte_carlo_simulation()
    """
    if results is None:
        print(f"\n=== {strategy_name} ===")
        print("No bets placed with current criteria")
        return
    
    print(f"\n=== {strategy_name} ===")
    print(f"Number of Bets: {results['n_bets']}")
    print(f"Total Wagered: ${results['total_wagered']:.2f}")
    print(f"\nActual Results:")
    print(f"  Profit: ${results['actual_profit']:.2f}")
    print(f"  ROI: {results['actual_roi']:.2f}%")
    print(f"\nNull Hypothesis (EV = {results['null_ev']:.1%}):")
    print(f"  Expected Profit: ${results['simulated_mean_profit']:.2f} ± ${results['simulated_std_profit']:.2f}")
    print(f"  95% CI Profit: [${results['ci_95_profit'][0]:.2f}, ${results['ci_95_profit'][1]:.2f}]")
    print(f"\nStatistical Test:")
    print(f"  P-value: {results['p_value']:.4f}")
    
    # Print rejection tests for all alpha levels
    for alpha in ALPHA_LEVELS:
        key = f'reject_null_{int(alpha * 100):02d}'
        status = 'YES ✓' if results[key] else 'NO ✗'
        print(f"  Reject Null (α={alpha:.2f}): {status}")
    
    if results.get('reject_null_05', False):
        print(f"\n  → Strategy performance is statistically significant!")
    else:
        print(f"\n  → Strategy performance not statistically distinguishable from random.")
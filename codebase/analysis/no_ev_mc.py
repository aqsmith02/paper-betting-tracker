import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from codebase.analysis.roi import STRATEGIES, roi_edge, roi_flat
from codebase.results_appending.results_configs import PENDING_RESULTS

# Central lookup table for both strategy + scoring columns
STRATEGY_INFO: Dict[str, Dict[str, List[str]]] = {
    "codebase/data/master_avg_bets.csv": {
        "odds": "Avg Edge Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_mod_zscore_bets.csv": {
        "odds": "Outlier Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_pin_bets.csv": {
        "odds": "Pinnacle Edge Odds",
        "edge": "Pin Edge Pct",
    },
    "codebase/data/master_zscore_bets.csv": {
        "odds": "Outlier Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_random_bets.csv": {
        "odds": "Random Bet Odds",
        "edge": "",
    },
    "codebase/data/master_nc_avg_bets.csv": {
        "odds": "Avg Edge Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_nc_mod_zscore_bets.csv": {
        "odds": "Outlier Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_nc_pin_bets.csv": {
        "odds": "Pinnacle Edge Odds",
        "edge": "Pin Edge Pct",
    },
    "codebase/data/master_nc_zscore_bets.csv": {
        "odds": "Outlier Odds",
        "edge": "Avg Edge Pct",
    },
    "codebase/data/master_nc_random_bets.csv": {
        "odds": "Random Bet Odds",
        "edge": "",
    },
}


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

    shrink = 1/4
    f_adjusted = max(f_kelly * shrink, 0)

    return min(f_adjusted * 100, max_multiplier)


def get_observed_results(file_path, strat):
    """
    Get the observed ROI and profit from actual betting results.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing bet data.
    strat : BettingStrategy
        Strategy object with odds and edge column info.
        
    Returns
    -------
    dict
        Dictionary containing observed metrics.
    """
    df = pd.read_csv(file_path)
    
    if strat.edge_column:
        # Use Kelly method for strategies with edge
        results = roi_edge(df, strat.odds_column, strat.edge_column, method="kelly")
    else:
        # Use flat betting for random strategy
        results = roi_flat(df, strat.odds_column)
    
    return {
        'observed_profit': results['Net Profit'],
        'observed_roi': results['ROI'] / 100,  # Convert to decimal
        'total_wagered': results['Total Wagered'],
        'total_winnings': results['Total Winnings']
    }


def simulate_profit_distribution(odds_list, stakes_list, vig_adjustment=-0.05, n_sims=100000, seed=1):
    """
    Simulate total profitability distribution across the full dataset of bets.

    Parameters
    ----------
    odds_list : list or np.ndarray
        Decimal odds for each bet.
    stakes_list : list or np.ndarray
        Stake size for each bet.
    vig_adjustment : float
        Expected value adjustment due to vigorish (default -0.05 for -5% EV).
        Set to 0 for strategies with actual edge, negative for random betting.
    n_sims : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Simulated distribution with total_profit and roi columns.
    """
    np.random.seed(seed)

    # Convert to arrays
    odds = np.array(odds_list, dtype=float)
    stakes = np.array(stakes_list, dtype=float)
    n_bets = len(odds)

    # Adjust win probability to account for vig
    if vig_adjustment < 0:
        true_p = (1 + vig_adjustment) / odds
    else:
        true_p = 1 / odds  # Use implied probability for strategies with edge
    
    # Clip probabilities to valid range
    true_p = np.clip(true_p, 0, 1)
    
    # Print any probabilities exceeding 1 before clipping
    if np.any((1 + vig_adjustment) / odds > 1):
        print("Warning: Some win probabilities exceed 1 (before clipping):")
        bad_probs = ((1 + vig_adjustment) / odds > 1)
        for idx in np.where(bad_probs)[0][:5]:  # Show first 5
            print(f"  Bet index {idx}: p = {(1 + vig_adjustment) / odds[idx]:.4f}, odds = {odds[idx]}")

    # Simulate outcomes: shape (n_sims, n_bets)
    wins = np.random.rand(n_sims, n_bets) < true_p
    
    # Calculate profit/loss with actual stakes
    profit_if_win = stakes * (odds - 1)
    loss_if_lose = -stakes
    returns = np.where(wins, profit_if_win, loss_if_lose)

    # Total profit per simulation
    total_profit = returns.sum(axis=1)
    
    # ROI
    total_staked = stakes.sum()
    roi = total_profit / total_staked if total_staked > 0 else np.zeros(n_sims)
    
    return pd.DataFrame({
        'total_profit': total_profit,
        'roi': roi
    })


def print_detailed_stats(sim_results, observed_results, stakes):
    """
    Print comprehensive statistics about profitability and ROI distributions.
    
    Parameters
    ----------
    sim_results : pd.DataFrame
        Simulated distribution data.
    observed_results : dict
        Observed results from actual betting.
    stakes : pd.Series
        Stake sizes for each bet.
    """
    print(f"\n{'='*70}")
    print("BETTING PARAMETERS")
    print(f"{'='*70}")
    print(f"Total stakes: {stakes.sum():.2f} units")
    print(f"Number of bets: {len(stakes)}")
    print(f"Avg stake per bet: {stakes.mean():.2f} units")
    print(f"Median stake: {stakes.median():.2f} units")
    print(f"Min/Max stake: {stakes.min():.2f} / {stakes.max():.2f} units")
    print(f"Std dev of stakes: {stakes.std():.2f} units")
    
    print(f"\n{'='*70}")
    print("OBSERVED RESULTS (Actual Performance)")
    print(f"{'='*70}")
    print(f"Total Wagered: {observed_results['total_wagered']:.2f} units")
    print(f"Total Winnings: {observed_results['total_winnings']:.2f} units")
    print(f"Net Profit: {observed_results['observed_profit']:.2f} units")
    print(f"ROI: {observed_results['observed_roi']:.4f} ({observed_results['observed_roi']*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print("SIMULATED PROFIT DISTRIBUTION")
    print(f"{'='*70}")
    profit_mean = sim_results['total_profit'].mean()
    profit_std = sim_results['total_profit'].std()
    profit_median = sim_results['total_profit'].median()
    
    print(f"Mean: {profit_mean:.2f} units")
    print(f"Median: {profit_median:.2f} units")
    print(f"Std dev: {profit_std:.2f} units")
    print(f"Min: {sim_results['total_profit'].min():.2f} units")
    print(f"Max: {sim_results['total_profit'].max():.2f} units")
    print(f"Range: {sim_results['total_profit'].max() - sim_results['total_profit'].min():.2f} units")
    
    # Percentiles
    print(f"\nProfit Percentiles:")
    profit_percentiles = sim_results['total_profit'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for p, val in profit_percentiles.items():
        print(f"  {int(p*100):>3d}th percentile: {val:>10.2f} units")
    
    # Probability metrics
    print(f"\nProfitability Probabilities:")
    print(f"  P(Profit > 0): {(sim_results['total_profit'] > 0).mean():.4f}")
    print(f"  P(Profit > 10): {(sim_results['total_profit'] > 10).mean():.4f}")
    print(f"  P(Profit > 50): {(sim_results['total_profit'] > 50).mean():.4f}")
    print(f"  P(Profit > 100): {(sim_results['total_profit'] > 100).mean():.4f}")
    print(f"  P(Loss > 10): {(sim_results['total_profit'] < -10).mean():.4f}")
    print(f"  P(Loss > 50): {(sim_results['total_profit'] < -50).mean():.4f}")
    print(f"  P(Loss > 100): {(sim_results['total_profit'] < -100).mean():.4f}")
    
    # Observed vs Expected
    print(f"\nObserved vs Simulated Profit:")
    profit_percentile = (sim_results['total_profit'] < observed_results['observed_profit']).mean()
    print(f"  Observed profit: {observed_results['observed_profit']:.2f} units")
    print(f"  Expected profit: {profit_mean:.2f} units")
    print(f"  Difference: {observed_results['observed_profit'] - profit_mean:.2f} units")
    print(f"  Observed profit percentile: {profit_percentile:.4f} ({profit_percentile*100:.2f}th)")
    if profit_std > 0:
        z_score = (observed_results['observed_profit'] - profit_mean) / profit_std
        print(f"  Z-score: {z_score:.2f}")
    
    print(f"\n{'='*70}")
    print("SIMULATED ROI DISTRIBUTION")
    print(f"{'='*70}")
    roi_mean = sim_results['roi'].mean()
    roi_std = sim_results['roi'].std()
    roi_median = sim_results['roi'].median()
    
    print(f"Mean ROI: {roi_mean:.4f} ({roi_mean*100:.2f}%)")
    print(f"Median ROI: {roi_median:.4f} ({roi_median*100:.2f}%)")
    print(f"Std dev: {roi_std:.4f} ({roi_std*100:.2f}%)")
    print(f"Min ROI: {sim_results['roi'].min():.4f} ({sim_results['roi'].min()*100:.2f}%)")
    print(f"Max ROI: {sim_results['roi'].max():.4f} ({sim_results['roi'].max()*100:.2f}%)")
    
    # ROI Percentiles
    print(f"\nROI Percentiles:")
    roi_percentiles = sim_results['roi'].quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    for p, val in roi_percentiles.items():
        print(f"  {int(p*100):>3d}th percentile: {val:>8.4f} ({val*100:>7.2f}%)")
    
    # ROI probability metrics
    print(f"\nROI Probabilities:")
    print(f"  P(ROI > 0%): {(sim_results['roi'] > 0).mean():.4f}")
    print(f"  P(ROI > 5%): {(sim_results['roi'] > 0.05).mean():.4f}")
    print(f"  P(ROI > 10%): {(sim_results['roi'] > 0.10).mean():.4f}")
    print(f"  P(ROI < -5%): {(sim_results['roi'] < -0.05).mean():.4f}")
    print(f"  P(ROI < -10%): {(sim_results['roi'] < -0.10).mean():.4f}")
    
    # Observed vs Expected ROI
    print(f"\nObserved vs Simulated ROI:")
    roi_percentile = (sim_results['roi'] < observed_results['observed_roi']).mean()
    print(f"  Observed ROI: {observed_results['observed_roi']:.4f} ({observed_results['observed_roi']*100:.2f}%)")
    print(f"  Expected ROI: {roi_mean:.4f} ({roi_mean*100:.2f}%)")
    print(f"  Difference: {observed_results['observed_roi'] - roi_mean:.4f} ({(observed_results['observed_roi'] - roi_mean)*100:.2f}%)")
    print(f"  Observed ROI percentile: {roi_percentile:.4f} ({roi_percentile*100:.2f}th)")
    if roi_std > 0:
        z_score_roi = (observed_results['observed_roi'] - roi_mean) / roi_std
        print(f"  Z-score: {z_score_roi:.2f}")


def simulate_all_files():
    """
    Simulate all betting strategies and compare with observed results.
    """
    files = [
        "codebase/data/master_avg_bets.csv",
        "codebase/data/master_nc_avg_bets.csv",
        "codebase/data/master_mod_zscore_bets.csv",
        "codebase/data/master_nc_mod_zscore_bets.csv",
        "codebase/data/master_pin_bets.csv",
        "codebase/data/master_nc_pin_bets.csv",
        "codebase/data/master_random_bets.csv",
        "codebase/data/master_nc_random_bets.csv",
        "codebase/data/master_zscore_bets.csv",
        "codebase/data/master_nc_zscore_bets.csv",
    ]
    
    for file in files:
        print(f"\n\n{'#'*70}")
        print(f"# {file}")
        print(f"{'#'*70}")
        
        df = pd.read_csv(file)
        odds_col_name = STRATEGY_INFO[file]["odds"]
        edge_col_name = STRATEGY_INFO[file]["edge"]
        
        odds = df[odds_col_name]
        
        # Get strategy object
        strat = next((s for s in STRATEGIES if s.path == file), None)
        if not strat:
            print(f"Warning: No strategy found for {file}")
            continue
        
        # Get observed results
        observed_results = get_observed_results(file, strat)
        
        # Calculate stakes using Kelly criterion
        if edge_col_name and edge_col_name in df.columns:
            # Has edge column - use Kelly
            stakes = df.apply(
                lambda row: kelly_bet(row[edge_col_name], row[odds_col_name]),
                axis=1
            )
        else:
            # Random strategy - use flat betting
            stakes = pd.Series([1.0] * len(df))
        
        vig_adjustment = -.05
        sim_results = simulate_profit_distribution(
            odds, stakes, vig_adjustment=vig_adjustment, n_sims=100000
        )

        # Print detailed statistics
        print_detailed_stats(sim_results, observed_results, stakes)
        
        strat_name = strat.name if strat else file.split('/')[-1].replace('.csv', '')

        # Plot distributions with observed values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Total Profit Distribution
        ax1.hist(sim_results['total_profit'], bins=60, alpha=0.7, color="skyblue", edgecolor="gray")
        ax1.axvline(0, color="red", linestyle="--", label="Break-even", linewidth=2)
        ax1.axvline(sim_results['total_profit'].mean(), color="green", linestyle="--", 
                   label=f"Mean: {sim_results['total_profit'].mean():.2f}", linewidth=2)
        ax1.axvline(observed_results['observed_profit'], color="purple", linestyle="-", 
                   label=f"Observed: {observed_results['observed_profit']:.2f}", linewidth=2)
        ax1.set_title(f"Total Profit Distribution: {strat_name}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Total Profit (units)", fontsize=11)
        ax1.set_ylabel("Frequency", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Add text with key stats
        textstr = f'P(Profit > 0) = {(sim_results["total_profit"] > 0).mean():.3f}\n'
        textstr += f'Median = {sim_results["total_profit"].median():.2f}\n'
        textstr += f'Std Dev = {sim_results["total_profit"].std():.2f}'
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # ROI Distribution
        ax2.hist(sim_results['roi'], bins=60, alpha=0.7, color="lightcoral", edgecolor="gray")
        ax2.axvline(0, color="red", linestyle="--", label="Break-even", linewidth=2)
        ax2.axvline(sim_results['roi'].mean(), color="green", linestyle="--", 
                   label=f"Mean ROI: {sim_results['roi'].mean():.4f}", linewidth=2)
        ax2.axvline(observed_results['observed_roi'], color="purple", linestyle="-", 
                   label=f"Observed: {observed_results['observed_roi']:.4f}", linewidth=2)
        ax2.set_title(f"ROI Distribution: {strat_name}", fontsize=12, fontweight='bold')
        ax2.set_xlabel("ROI", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Add text with key stats
        textstr = f'P(ROI > 0) = {(sim_results["roi"] > 0).mean():.3f}\n'
        textstr += f'Median = {sim_results["roi"].median():.4f}\n'
        textstr += f'Std Dev = {sim_results["roi"].std():.4f}'
        ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def main():
    simulate_all_files()


if __name__ == "__main__":
    main()
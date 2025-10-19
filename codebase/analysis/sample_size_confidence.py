import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from codebase.analysis.roi import print_strategy_results_and_return_roi, STRATEGIES

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

def simulate_profit_distribution(odds_list, edge_list, n_sims=100000, seed=1):
    """
    Simulate total profitability distribution across the full dataset of bets.

    Parameters
    ----------
    odds_list : list or np.ndarray
        Decimal odds for each bet.
    edge_list : list or np.ndarray
        Expected ROI (edge) for each bet, e.g. 0.05 = +5%.
    n_sims : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.Series
        Simulated total ROI distribution across all bets.
    """
    np.random.seed(seed)

    # Convert to arrays
    odds = np.array(odds_list, dtype=float)
    edge = np.array(edge_list, dtype=float) / 100.0
    n_bets = len(odds)

    # Compute implied win probability per bet
    p = (1 + edge) / odds
    # Print any probabilities exceeding 1
    if np.any(p > 1):
        print("Warning: Some implied probabilities exceed 1:")
        for idx, val in enumerate(p):
            if val > 1:
                print(f"  Bet index {idx}: p = {val:.4f}, odds = {odds[idx]}, edge = {edge[idx]}")

    # Simulate outcomes: shape (n_sims, n_bets)
    wins = np.random.rand(n_sims, n_bets) < p  # True = win
    returns = np.where(wins, odds - 1, -1)  # profit or loss per bet

    # Compute total ROI per simulation
    total_returns = returns.sum(axis=1)  # total profit (in stake units)
    roi = total_returns / n_bets  # average profit per bet
    return pd.Series(roi, name="ROI")


def simulate_all_files():
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
        df = pd.read_csv(file)
        odds_col_name = STRATEGY_INFO[file]["odds"]
        odds = df[odds_col_name]
        edge_col_name = STRATEGY_INFO[file]["edge"]
        edge = df[edge_col_name]
        roi_dist = simulate_profit_distribution(odds, edge, n_sims=100000)

        # Summary stats
        print("File: ", file)
        print("Mean ROI:", roi_dist.mean())
        print("Std dev:", roi_dist.std())
        print("P(ROI > 0):", (roi_dist > 0).mean())
        percentiles = roi_dist.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        for p, val in percentiles.items():
            print(f"{int(p*100):>3d}th percentile: {val:.4f}")
        print("\n")

        strat = next((s for s in STRATEGIES if s.path == file), None)
        observed_roi = print_strategy_results_and_return_roi(strat) / 100

        # Plot distribution
        plt.figure(figsize=(8, 5))
        plt.hist(roi_dist, bins=60, alpha=0.7, color="skyblue", edgecolor="gray")
        plt.axvline(0, color="red", linestyle="--", label="Break-even")
        plt.axvline(observed_roi, color="green", linestyle="--", label="Observed ROI")
        plt.title(f"Simulated ROI Distribution: {strat.name}")
        plt.xlabel("ROI")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    simulate_all_files()


if __name__ == "__main__":
    main()

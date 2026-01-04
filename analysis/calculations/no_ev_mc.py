import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from src.results.results_configs import PENDING_RESULTS


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

# Configuration
KELLY_FRACTION = 0.5
MIN_EV_THRESHOLD = 0.05
MAX_EV_THRESHOLD = 0.35
ZSCORE_MAX_BET_THRESHOLD = 3.5
NULL_EV = -0.05  # -5% expected value under null hypothesis
N_SIMULATIONS = 10000


def kelly_bet(odds, fair_odds, ev=None, zscore=None, max_multiplier=2.5):
    """Kelly bet adjusted for parameter uncertainty via shrinkage coefficient."""
    if pd.isna(odds) or odds <= 1:
        return 0
    
    if ev is not None and not pd.isna(ev):
        if ev < MIN_EV_THRESHOLD or ev > MAX_EV_THRESHOLD:
            return 0

    p = 1 / fair_odds
    p = max(min(p, 0.9999), 0.0001)
    b = odds - 1

    f_kelly = p - ((1 - p) / b)

    if f_kelly <= 0:
        return 0

    if zscore is not None and not pd.isna(zscore):
        if zscore >= ZSCORE_MAX_BET_THRESHOLD:
            return max_multiplier

    shrink = KELLY_FRACTION
    f_adjusted = f_kelly * shrink

    return min(f_adjusted * 100, max_multiplier)


def calculate_null_probability(odds, null_ev=NULL_EV):
    """
    Calculate win probability under null hypothesis of expected EV.
    
    Given:
    - EV = prob * (payout + 1) - 1
    - Payout = odds - 1
    
    Solving for probability:
    - prob = (EV + 1) / (payout + 1)
    - prob = (EV + 1) / odds
    
    Args:
        odds: Decimal odds for the bet
        null_ev: Expected value under null hypothesis
    
    Returns:
        Win probability under null hypothesis
    """
    payout = odds - 1
    probability = (null_ev + 1) / (payout + 1)
    
    # Clamp probability between 0 and 1
    return max(0, min(1, probability))


def monte_carlo_simulation(df, odds_col, fair_col, ev_col=None, zscore_col=None, 
                           n_simulations=N_SIMULATIONS, null_ev=NULL_EV):
    """
    Run Monte Carlo simulation to test null hypothesis using calculated probabilities.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
        n_simulations: Number of simulations to run
        null_ev: Expected value under null hypothesis
    
    Returns:
        Dictionary with simulation results and statistical tests
    """
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
    simulated_rois = []
    
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
        total_sim_wagered = df_bets["Bet_Size"].sum()
        sim_roi = (total_sim_profit / total_sim_wagered * 100) if total_sim_wagered > 0 else 0
        
        simulated_profits.append(total_sim_profit)
        simulated_rois.append(sim_roi)
    
    simulated_profits = np.array(simulated_profits)
    simulated_rois = np.array(simulated_rois)
    
    # Calculate p-value (two-tailed test)
    # What proportion of simulations are as extreme or more extreme than actual?
    p_value_profit = np.mean(simulated_profits >= actual_total_profit)
    p_value_roi = np.mean(simulated_rois >= actual_roi)
    
    # Calculate confidence intervals
    ci_95_profit = np.percentile(simulated_profits, [2.5, 97.5])
    ci_95_roi = np.percentile(simulated_rois, [2.5, 97.5])
    
    return {
        "n_bets": len(df_bets),
        "total_wagered": actual_total_wagered,
        "actual_profit": actual_total_profit,
        "actual_roi": actual_roi,
        "simulated_mean_profit": np.mean(simulated_profits),
        "simulated_std_profit": np.std(simulated_profits),
        "simulated_mean_roi": np.mean(simulated_rois),
        "simulated_std_roi": np.std(simulated_rois),
        "ci_95_profit": ci_95_profit,
        "ci_95_roi": ci_95_roi,
        "p_value_profit": p_value_profit,
        "p_value_roi": p_value_roi,
        "reject_null_05": p_value_roi < 0.05,
        "reject_null_01": p_value_roi < 0.01,
        "simulated_profits": simulated_profits,
        "simulated_rois": simulated_rois,
    }


def plot_simulation_results(results, strategy_name, save_path=None):
    """Plot histogram of simulated profits/ROIs with actual result marked."""
    if results is None:
        print(f"No bets placed for {strategy_name}")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot profit distribution
    ax1.hist(results["simulated_profits"], bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(results["actual_profit"], color='red', linestyle='--', linewidth=2, 
                label=f'Actual: {results["actual_profit"]:.2f}')
    ax1.axvline(results["simulated_mean_profit"], color='blue', linestyle='--', linewidth=2,
                label=f'Expected: {results["simulated_mean_profit"]:.2f}')
    ax1.set_xlabel('Total Profit')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'{strategy_name}\nProfit Distribution (p={results["p_value_profit"]:.4f})')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot ROI distribution
    ax2.hist(results["simulated_rois"], bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(results["actual_roi"], color='red', linestyle='--', linewidth=2,
                label=f'Actual: {results["actual_roi"]:.2f}%')
    ax2.axvline(results["simulated_mean_roi"], color='blue', linestyle='--', linewidth=2,
                label=f'Expected: {results["simulated_mean_roi"]:.2f}%')
    ax2.set_xlabel('ROI (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{strategy_name}\nROI Distribution (p={results["p_value_roi"]:.4f})')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_simulation_results(strategy_name, results):
    """Print formatted simulation results."""
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
    print(f"\nNull Hypothesis (EV = {NULL_EV:.1%}):")
    print(f"  Expected Profit: ${results['simulated_mean_profit']:.2f} ± ${results['simulated_std_profit']:.2f}")
    print(f"  Expected ROI: {results['simulated_mean_roi']:.2f}% ± {results['simulated_std_roi']:.2f}%")
    print(f"  95% CI Profit: [${results['ci_95_profit'][0]:.2f}, ${results['ci_95_profit'][1]:.2f}]")
    print(f"  95% CI ROI: [{results['ci_95_roi'][0]:.2f}%, {results['ci_95_roi'][1]:.2f}%]")
    print(f"\nStatistical Test:")
    print(f"  P-value (ROI): {results['p_value_roi']:.4f}")
    print(f"  Reject Null (α=0.05): {'YES ✓' if results['reject_null_05'] else 'NO ✗'}")
    print(f"  Reject Null (α=0.01): {'YES ✓' if results['reject_null_01'] else 'NO ✗'}")
    
    if results['reject_null_05']:
        print(f"\n  → Strategy performance is statistically significant!")
    else:
        print(f"\n  → Strategy performance not statistically distinguishable from random.")


def main():
    """Run Monte Carlo simulation for all strategies."""
    print("=" * 70)
    print("MONTE CARLO SIMULATION - NULL HYPOTHESIS TESTING")
    print("=" * 70)
    print(f"Null Hypothesis: Expected Value = {NULL_EV:.1%}")
    print(f"Number of Simulations: {N_SIMULATIONS:,}")
    print(f"Kelly Fraction: {KELLY_FRACTION}")
    print(f"EV Threshold: [{MIN_EV_THRESHOLD:.1%}, {MAX_EV_THRESHOLD:.1%}]")
    print("=" * 70)
    
    for strategy in STRATEGIES:
        try:
            df = pd.read_csv(strategy.path)
            results = monte_carlo_simulation(
                df,
                strategy.odds_column,
                strategy.fair_odds_column,
                strategy.ev_column,
                strategy.zscore_column,
            )
            print_simulation_results(strategy.name, results)
            
            # Generate plots for each strategy
            if results:
                plot_simulation_results(results, strategy.name, 
                                      save_path=f"codebase/analysis/no_ev_mc/monte_carlo_{strategy.name.replace(' ', '_').lower()}.png")
            
        except FileNotFoundError:
            print(f"\n=== {strategy.name} ===")
            print(f"File not found: {strategy.path}")
        except Exception as e:
            print(f"\n=== {strategy.name} ===")
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
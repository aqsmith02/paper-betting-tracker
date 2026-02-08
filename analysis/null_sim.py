"""
null_simulation.py

Run simulation under the null hypothesis that bets have -5% EV.
Tests whether observed results are statistically significantly better than
the null hypothesis of -5% expected value.

Author: Andrew Smith
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from scipy import stats


class NullHypothesisSimulator:
    """Simulate betting performance under null hypothesis of -5% EV."""
    
    def __init__(self, df: pd.DataFrame, starting_bankroll: float = 10000.0,
                 null_ev: float = -0.05, n_simulations: int = 10000):
        """
        Initialize the simulator.
        
        Args:
            df: DataFrame with betting data
            starting_bankroll: Initial bankroll
            null_ev: Null hypothesis EV (default: -0.05 for -5%)
            n_simulations: Number of Monte Carlo simulations
        """
        self.df = df.copy()
        self.starting_bankroll = starting_bankroll
        self.null_ev = null_ev
        self.n_simulations = n_simulations
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data with necessary calculations."""
        # Filter for completed bets only
        pending_statuses = ['Pending', 'Not Found', 'API Error']
        self.df = self.df[~self.df['Result'].isin(pending_statuses)].copy()
        
        if len(self.df) == 0:
            print("Warning: No completed bets found")
            return
        
        # Determine if bet won
        self.df['Bet_Won'] = self.df.apply(
            lambda row: row['Result'] == row['Team'],
            axis=1
        )
        
        # Calculate Kelly and stake (same as actual strategy)
        self.df['Fair_Probability'] = 1 / self.df['Fair Odds Average']
        self.df['Implied_Probability'] = 1 / self.df['Best Odds']
        
        # 1/2 Kelly sizing
        b = self.df['Best Odds'] - 1
        p = self.df['Fair_Probability']
        q = 1 - p
        kelly = ((b * p - q) / b).clip(lower=0)
        self.df['Stake_Pct'] = (kelly / 2).clip(upper=0.05)
        
        # Calculate actual profit
        stake = self.df['Stake_Pct'] * self.starting_bankroll
        self.df['Actual_Profit'] = np.where(
            self.df['Bet_Won'],
            stake * (self.df['Best Odds'] - 1),
            -stake
        )
        
        # Calculate null hypothesis win probability
        # Under null: prob = (null_ev + 1) / odds
        self.df['Null_Win_Prob'] = (self.null_ev + 1) / self.df['Best Odds']
        self.df['Null_Win_Prob'] = self.df['Null_Win_Prob'].clip(0, 1)
    
    def run_simulation(self) -> Dict:
        """
        Run Monte Carlo simulation under null hypothesis.
        
        Returns:
            Dictionary with simulation results
        """
        if len(self.df) == 0:
            return {}
        
        print(f"\nRunning {self.n_simulations:,} simulations under null hypothesis (EV = {self.null_ev*100}%)...")
        
        # Pre-calculate stakes for efficiency
        stakes = (self.df['Stake_Pct'] * self.starting_bankroll).values
        odds = self.df['Best Odds'].values
        null_probs = self.df['Null_Win_Prob'].values
        n_bets = len(self.df)
        
        # Run simulations
        simulated_profits = np.zeros(self.n_simulations)
        
        for i in range(self.n_simulations):
            # Randomly determine wins/losses based on null hypothesis probabilities
            wins = np.random.random(n_bets) < null_probs
            
            # Calculate profit for this simulation
            profits = np.where(wins, stakes * (odds - 1), -stakes)
            simulated_profits[i] = profits.sum()
            
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1:,} / {self.n_simulations:,} simulations")
        
        # Calculate actual observed profit
        actual_profit = self.df['Actual_Profit'].sum()
        
        # Calculate p-value (one-tailed test: observed > null)
        p_value = (simulated_profits >= actual_profit).sum() / self.n_simulations
        
        # Calculate statistics on simulated distribution
        results = {
            'actual_profit': actual_profit,
            'simulated_profits': simulated_profits,
            'null_ev': self.null_ev,
            'n_simulations': self.n_simulations,
            'n_bets': n_bets,
            'mean_simulated_profit': simulated_profits.mean(),
            'std_simulated_profit': simulated_profits.std(),
            'median_simulated_profit': np.median(simulated_profits),
            'min_simulated_profit': simulated_profits.min(),
            'max_simulated_profit': simulated_profits.max(),
            'p_value': p_value,
            'percentile_rank': stats.percentileofscore(simulated_profits, actual_profit),
            'z_score': (actual_profit - simulated_profits.mean()) / simulated_profits.std(),
            'ci_lower': np.percentile(simulated_profits, 2.5),
            'ci_upper': np.percentile(simulated_profits, 97.5)
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print formatted simulation results."""
        if not results:
            print("No results to display")
            return
        
        print("\n" + "="*80)
        print("NULL HYPOTHESIS SIMULATION RESULTS")
        print("="*80)
        
        print(f"\nNULL HYPOTHESIS")
        print(f"   EV: {results['null_ev']*100}%")
        print(f"   Number of Bets: {results['n_bets']}")
        print(f"   Simulations: {results['n_simulations']:,}")
        
        print(f"\nACTUAL PERFORMANCE")
        print(f"   Observed Profit: ${results['actual_profit']:,.2f}")
        
        print(f"\nSIMULATED DISTRIBUTION (UNDER NULL)")
        print(f"   Mean: ${results['mean_simulated_profit']:,.2f}")
        print(f"   Std Dev: ${results['std_simulated_profit']:,.2f}")
        print(f"   Median: ${results['median_simulated_profit']:,.2f}")
        print(f"   95% CI: [${results['ci_lower']:,.2f}, ${results['ci_upper']:,.2f}]")
        print(f"   Range: [${results['min_simulated_profit']:,.2f}, ${results['max_simulated_profit']:,.2f}]")
        
        print(f"\nSTATISTICAL SIGNIFICANCE")
        print(f"   Z-Score: {results['z_score']:.2f}")
        print(f"   Percentile Rank: {results['percentile_rank']:.2f}%")
        print(f"   P-Value: {results['p_value']:.4f}")
        
        # Interpretation
        print(f"\nINTERPRETATION")
        if results['p_value'] < 0.001:
            significance = "highly significant (p < 0.001)"
        elif results['p_value'] < 0.01:
            significance = "very significant (p < 0.01)"
        elif results['p_value'] < 0.05:
            significance = "significant (p < 0.05)"
        elif results['p_value'] < 0.10:
            significance = "marginally significant (p < 0.10)"
        else:
            significance = "not significant (p >= 0.10)"
        
        print(f"   Statistical Significance: {significance}")
        
        if results['actual_profit'] > results['ci_upper']:
            print(f"   Your profit (${results['actual_profit']:,.2f}) is ABOVE the 95% confidence interval")
            print(f"   under the null hypothesis of {results['null_ev']*100}% EV.")
        elif results['actual_profit'] > results['mean_simulated_profit']:
            print(f"   Your profit (${results['actual_profit']:,.2f}) is above the expected value")
            print(f"   under the null hypothesis, but within the 95% confidence interval.")
        else:
            print(f"   Your profit (${results['actual_profit']:,.2f}) is below or at the expected value")
            print(f"   under the null hypothesis.")
        
        print("\n" + "="*80 + "\n")
    
    def plot_simulation_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Create visualization of simulation results.
        
        Args:
            results: Results dictionary from run_simulation()
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        simulated_profits = results['simulated_profits']
        actual_profit = results['actual_profit']
        
        # 1. Histogram of simulated profits
        ax1 = axes[0, 0]
        ax1.hist(simulated_profits, bins=50, alpha=0.7, color='steelblue', 
                edgecolor='black')
        ax1.axvline(actual_profit, color='red', linestyle='--', linewidth=2,
                   label=f'Actual: ${actual_profit:,.0f}')
        ax1.axvline(results['mean_simulated_profit'], color='green', 
                   linestyle='--', linewidth=2,
                   label=f'Null Mean: ${results["mean_simulated_profit"]:,.0f}')
        ax1.set_xlabel('Total Profit ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Distribution of Profits Under Null (EV = {results["null_ev"]*100}%)')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Cumulative distribution
        ax2 = axes[0, 1]
        sorted_profits = np.sort(simulated_profits)
        cumulative = np.arange(1, len(sorted_profits) + 1) / len(sorted_profits)
        ax2.plot(sorted_profits, cumulative * 100, color='steelblue', linewidth=2)
        ax2.axvline(actual_profit, color='red', linestyle='--', linewidth=2,
                   label=f'Actual (Percentile: {results["percentile_rank"]:.1f}%)')
        ax2.set_xlabel('Total Profit ($)')
        ax2.set_ylabel('Cumulative Probability (%)')
        ax2.set_title('Cumulative Distribution Function')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Box plot comparison
        ax3 = axes[1, 0]
        box_data = [simulated_profits]
        bp = ax3.boxplot(box_data, labels=['Simulated'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.scatter([1], [actual_profit], color='red', s=200, zorder=3,
                   label='Actual Profit', marker='D')
        ax3.set_ylabel('Total Profit ($)')
        ax3.set_title('Profit Distribution: Actual vs Null Hypothesis')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Z-score visualization
        ax4 = axes[1, 1]
        
        # Create normal distribution overlay
        x = np.linspace(simulated_profits.min(), simulated_profits.max(), 1000)
        normal_dist = stats.norm.pdf(x, results['mean_simulated_profit'], 
                                     results['std_simulated_profit'])
        
        # Normalize to match histogram scale
        ax4_2 = ax4.twinx()
        ax4.hist(simulated_profits, bins=50, alpha=0.5, color='steelblue',
                density=True, label='Simulated')
        ax4_2.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Fit')
        
        ax4.axvline(actual_profit, color='darkred', linestyle='--', linewidth=2,
                   label=f'Actual (Z={results["z_score"]:.2f})')
        
        ax4.set_xlabel('Total Profit ($)')
        ax4.set_ylabel('Density', color='steelblue')
        ax4_2.set_ylabel('Normal Density', color='red')
        ax4.set_title('Actual vs Normal Distribution Under Null')
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_2.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
    
    def get_simulation_summary(self, results: Dict) -> pd.DataFrame:
        """
        Get summary statistics as a DataFrame.
        
        Args:
            results: Results dictionary from run_simulation()
        
        Returns:
            DataFrame with summary statistics
        """
        if not results:
            return pd.DataFrame()
        
        summary = pd.DataFrame({
            'Metric': [
                'Null Hypothesis EV',
                'Number of Bets',
                'Number of Simulations',
                'Actual Profit',
                'Expected Profit (Null)',
                'Std Dev (Null)',
                'Min Profit (Null)',
                'Max Profit (Null)',
                '95% CI Lower',
                '95% CI Upper',
                'Z-Score',
                'Percentile Rank',
                'P-Value'
            ],
            'Value': [
                f"{results['null_ev']*100}%",
                results['n_bets'],
                f"{results['n_simulations']:,}",
                f"${results['actual_profit']:,.2f}",
                f"${results['mean_simulated_profit']:,.2f}",
                f"${results['std_simulated_profit']:,.2f}",
                f"${results['min_simulated_profit']:,.2f}",
                f"${results['max_simulated_profit']:,.2f}",
                f"${results['ci_lower']:,.2f}",
                f"${results['ci_upper']:,.2f}",
                f"{results['z_score']:.3f}",
                f"{results['percentile_rank']:.2f}%",
                f"{results['p_value']:.4f}"
            ]
        })
        
        return summary


def run_null_hypothesis_test(csv_path: str, starting_bankroll: float = 10000.0,
                             null_ev: float = -0.05, n_simulations: int = 10000,
                             plot: bool = True, save_plot: Optional[str] = None):
    """
    Run null hypothesis simulation from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        starting_bankroll: Starting bankroll
        null_ev: Null hypothesis EV (default: -0.05 for -5%)
        n_simulations: Number of simulations
        plot: Whether to create plots
        save_plot: Optional path to save plot
    
    Returns:
        Tuple of (simulator instance, results dictionary)
    """
    df = pd.read_csv(csv_path)
    simulator = NullHypothesisSimulator(df, starting_bankroll, null_ev, n_simulations)
    
    results = simulator.run_simulation()
    simulator.print_results(results)
    
    if plot and results:
        simulator.plot_simulation_results(results, save_plot)
    
    return simulator, results


if __name__ == "__main__":
    csv_paths = ["data/nc_avg_minimal.csv", "data/nc_mod_zscore_minimal.csv", "data/nc_random_minimal.csv"]
    
    for csv_path in csv_paths:
        simulator, results = run_null_hypothesis_test(
            csv_path, starting_bankroll=100.0, null_ev=-0.05, n_simulations=10000,
            plot=True)
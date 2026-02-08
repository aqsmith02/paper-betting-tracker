"""
standard_metrics.py

Calculate standard betting performance metrics including profit, ROI, win rate, etc.
Uses 1/2 Kelly criterion for bet sizing.

Author: Andrew Smith
Date: February 2026
"""

import pandas as pd
import numpy as np
from typing import Dict


class StandardMetricsCalculator:
    """Calculate standard betting performance metrics."""
    
    def __init__(self, df: pd.DataFrame, starting_bankroll: float = 100.0):
        """
        Initialize the calculator.
        
        Args:
            df: DataFrame with betting data
            starting_bankroll: Initial bankroll in dollars
        """
        self.df = df.copy()
        self.starting_bankroll = starting_bankroll
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data by adding calculated columns."""
        # Filter for completed bets only
        pending_statuses = ['Pending', 'Not Found', 'API Error']
        self.df = self.df[~self.df['Result'].isin(pending_statuses)].copy()
        
        if len(self.df) == 0:
            print("Warning: No completed bets found")
            return
        
        # Determine if bet won
        self.df['Bet_Won'] = self.df.apply(self._check_if_won, axis=1)
        
        # Calculate Kelly criterion stake
        self.df['Fair_Probability'] = 1 / self.df['Fair Odds Average']
        self.df['Implied_Probability'] = 1 / self.df['Best Odds']
        
        # Full Kelly = (bp - q) / b, where b = decimal_odds - 1, p = fair_prob, q = 1 - p
        # 1/2 Kelly = Full Kelly / 2
        self.df['Kelly_Fraction'] = self.df.apply(self._calculate_kelly, axis=1)
        
        # Calculate stake as percentage of bankroll
        self.df['Stake_Pct'] = self.df['Kelly_Fraction'] / 2  # Half Kelly
        self.df['Stake_Pct'] = self.df['Stake_Pct'].clip(lower=0, upper=0.025)  # Cap at 2.5%
        
        # Calculate profit/loss for each bet
        self.df['Profit_Loss'] = self.df.apply(self._calculate_profit_loss, axis=1)
        
        # Calculate cumulative metrics
        self.df = self.df.sort_values('Scrape Time').reset_index(drop=True)
        self.df['Cumulative_Profit'] = self.df['Profit_Loss'].cumsum()
        self.df['Running_Bankroll'] = self.starting_bankroll + self.df['Cumulative_Profit']
        
        # Calculate ROI for each bet
        self.df['ROI'] = (self.df['Profit_Loss'] / 
                          (self.df['Stake_Pct'] * self.starting_bankroll)) * 100
    
    def _check_if_won(self, row) -> bool:
        """Check if a bet won based on result."""
        if pd.isna(row['Result']):
            return False
        
        # The result should match the team we bet on
        return row['Result'] == row['Team']
    
    def _calculate_kelly(self, row) -> float:
        """Calculate Kelly criterion fraction."""
        b = row['Best Odds'] - 1  # Net odds (profit per unit stake)
        p = row['Fair_Probability']
        q = 1 - p
        
        # Kelly formula: (bp - q) / b
        kelly = (b * p - q) / b
        
        return max(kelly, 0)  # Don't bet if negative
    
    def _calculate_profit_loss(self, row) -> float:
        """Calculate profit/loss for a bet using dynamic bankroll."""
        stake = row['Stake_Pct'] * self.starting_bankroll
        
        if row['Bet_Won']:
            # Profit = stake * (decimal_odds - 1)
            return stake * (row['Best Odds'] - 1)
        else:
            # Loss = stake
            return -stake
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all standard metrics."""
        if len(self.df) == 0:
            return self._empty_metrics()
        
        total_bets = len(self.df)
        wins = self.df['Bet_Won'].sum()
        losses = total_bets - wins
        
        total_profit = self.df['Profit_Loss'].sum()
        total_staked = (self.df['Stake_Pct'] * self.starting_bankroll).sum()
        
        # Win rate
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # ROI
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        # Average metrics
        avg_win = self.df[self.df['Bet_Won']]['Profit_Loss'].mean() if wins > 0 else 0
        avg_loss = self.df[~self.df['Bet_Won']]['Profit_Loss'].mean() if losses > 0 else 0
        avg_odds = self.df['Best Odds'].mean()
        avg_ev = self.df['Expected Value'].mean()
        avg_stake_pct = self.df['Stake_Pct'].mean()
        
        # Bankroll metrics
        final_bankroll = self.starting_bankroll + total_profit
        bankroll_growth_pct = (total_profit / self.starting_bankroll) * 100
        max_bankroll = self.df['Running_Bankroll'].max()
        min_bankroll = self.df['Running_Bankroll'].min()
        
        # Drawdown
        running_max = self.df['Running_Bankroll'].cummax()
        drawdown = self.df['Running_Bankroll'] - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max[drawdown.idxmin()]) * 100 if len(drawdown) > 0 else 0
        
        # Risk metrics
        std_dev = self.df['ROI'].std()
        sharpe_ratio = (roi / std_dev) if std_dev > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate * 100, 2),
            'total_profit': round(total_profit, 2),
            'total_staked': round(total_staked, 2),
            'roi': round(roi, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_odds': round(avg_odds, 2),
            'avg_ev': round(avg_ev * 100, 2),
            'avg_stake_pct': round(avg_stake_pct * 100, 2),
            'starting_bankroll': self.starting_bankroll,
            'final_bankroll': round(final_bankroll, 2),
            'bankroll_growth_pct': round(bankroll_growth_pct, 2),
            'max_bankroll': round(max_bankroll, 2),
            'min_bankroll': round(min_bankroll, 2),
            'max_drawdown': round(max_drawdown, 2),
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'std_dev_roi': round(std_dev, 2),
            'sharpe_ratio': round(sharpe_ratio, 2)
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when no data."""
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'total_profit': 0,
            'total_staked': 0,
            'roi': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_odds': 0,
            'avg_ev': 0,
            'avg_stake_pct': 0,
            'starting_bankroll': self.starting_bankroll,
            'final_bankroll': self.starting_bankroll,
            'bankroll_growth_pct': 0,
            'max_bankroll': self.starting_bankroll,
            'min_bankroll': self.starting_bankroll,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'std_dev_roi': 0,
            'sharpe_ratio': 0
        }
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        metrics = self.calculate_all_metrics()
        
        print("\n" + "="*60)
        print("BETTING PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\n OVERALL STATISTICS")
        print(f"   Total Bets: {metrics['total_bets']}")
        print(f"   Wins: {metrics['wins']} | Losses: {metrics['losses']}")
        print(f"   Win Rate: {metrics['win_rate']}%")
        
        print(f"\n PROFITABILITY")
        print(f"   Total Profit: ${metrics['total_profit']:,.2f}")
        print(f"   Total Staked: ${metrics['total_staked']:,.2f}")
        print(f"   ROI: {metrics['roi']}%")
        print(f"   Average EV: {metrics['avg_ev']}%")
        
        print(f"\n BANKROLL")
        print(f"   Starting: ${metrics['starting_bankroll']:,.2f}")
        print(f"   Final: ${metrics['final_bankroll']:,.2f}")
        print(f"   Growth: {metrics['bankroll_growth_pct']}%")
        print(f"   Peak: ${metrics['max_bankroll']:,.2f}")
        print(f"   Valley: ${metrics['min_bankroll']:,.2f}")
        
        print(f"\n RISK METRICS")
        print(f"   Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_pct']}%)")
        print(f"   Std Dev (ROI): {metrics['std_dev_roi']}%")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']}")
        
        print(f"\n BET CHARACTERISTICS")
        print(f"   Average Odds: {metrics['avg_odds']}")
        print(f"   Average Win: ${metrics['avg_win']:,.2f}")
        print(f"   Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"   Average Stake: {metrics['avg_stake_pct']}% of bankroll")
        
        print(f"\n STREAKS")
        print(f"   Longest Win Streak: {metrics['longest_win_streak']}")
        print(f"   Longest Loss Streak: {metrics['longest_loss_streak']}")
        
        print("\n" + "="*60 + "\n")
    
    def get_processed_dataframe(self) -> pd.DataFrame:
        """Return the processed DataFrame with all calculated columns."""
        return self.df


def analyze_betting_data(csv_path: str, starting_bankroll: float = 100.0):
    """
    Analyze betting data from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        starting_bankroll: Starting bankroll amount
    
    Returns:
        Tuple of (metrics dict, processed DataFrame)
    """
    df = pd.read_csv(csv_path)
    calculator = StandardMetricsCalculator(df, starting_bankroll)
    
    metrics = calculator.calculate_all_metrics()
    calculator.print_summary()
    
    return metrics, calculator.get_processed_dataframe()


if __name__ == "__main__":
    csv_paths = ["data/nc_avg_minimal.csv", "data/nc_mod_zscore_minimal.csv", "data/nc_random_minimal.csv"]
    
    for csv_path in csv_paths:
        metrics, df = analyze_betting_data(csv_path, starting_bankroll=100.0)
        
        output_path = csv_path.replace('.csv', '_with_metrics.csv')
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")
"""
segmentation_analysis.py

Analyze betting performance across multiple segments:
- Sportsbook
- Sport/League
- Time from game start
- Odds ranges

Author: Andrew Smith
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class SegmentationAnalyzer:
    """Analyze betting performance across multiple segments."""
    
    def __init__(self, df: pd.DataFrame, starting_bankroll: float = 10000.0):
        """
        Initialize the segmentation analyzer.
        
        Args:
            df: DataFrame with betting data
            starting_bankroll: Initial bankroll for calculations
        """
        self.df = df.copy()
        self.starting_bankroll = starting_bankroll
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data with necessary calculations and derived columns."""
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
        
        # Calculate Kelly and stake
        self.df['Fair_Probability'] = 1 / self.df['Fair Odds Average']
        self.df['Implied_Probability'] = 1 / self.df['Best Odds']
        
        # 1/2 Kelly sizing
        b = self.df['Best Odds'] - 1
        p = self.df['Fair_Probability']
        q = 1 - p
        kelly = ((b * p - q) / b).clip(lower=0)
        self.df['Stake_Pct'] = (kelly / 2).clip(upper=0.05)
        
        # Calculate profit/loss and ROI
        stake = self.df['Stake_Pct'] * self.starting_bankroll
        self.df['Profit_Loss'] = np.where(
            self.df['Bet_Won'],
            stake * (self.df['Best Odds'] - 1),
            -stake
        )
        self.df['ROI'] = (self.df['Profit_Loss'] / stake) * 100
        
        # Add time-based features
        self._add_time_features()
        
        # Add odds-based features
        self._add_odds_features()
        
        # Add market features
        self._add_market_features()
    
    def _add_time_features(self):
        """Add time-based segmentation features."""
        # Convert to datetime
        self.df['Scrape_Time_dt'] = pd.to_datetime(self.df['Scrape Time'])
        self.df['Start_Time_dt'] = pd.to_datetime(self.df['Start Time'])
        
        # Time until game starts (lead time)
        self.df['Lead_Time_Hours'] = (
            (self.df['Start_Time_dt'] - self.df['Scrape_Time_dt'])
            .dt.total_seconds() / 3600
        )
        
        # Categorize lead time
        self.df['Lead_Time_Category'] = pd.cut(
            self.df['Lead_Time_Hours'],
            bins=[-1, 2, 6, 24, 72, 168, 1000],
            labels=['<2h', '2-6h', '6-24h', '1-3d', '3-7d', '7d+']
        )
        
        # Day of week
        self.df['Day_of_Week'] = self.df['Start_Time_dt'].dt.day_name()
        
        # Hour of day (when game starts)
        self.df['Hour_of_Day'] = self.df['Start_Time_dt'].dt.hour
        
        # Time of day category
        self.df['Time_of_Day'] = pd.cut(
            self.df['Hour_of_Day'],
            bins=[-1, 6, 12, 17, 21, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening', 'Late']
        )
        
        # Month
        self.df['Month'] = self.df['Start_Time_dt'].dt.month_name()
    
    def _add_odds_features(self):
        """Add odds-based segmentation features."""
        # Favorite vs Underdog
        self.df['Bet_Type'] = pd.cut(
            self.df['Best Odds'],
            bins=[0, 1.5, 2.0, 3.0, 100],
            labels=['Heavy Favorite', 'Slight Favorite', 'Underdog', 'Longshot']
        )
        
        # Odds range (continuous)
        self.df['Odds_Rounded'] = (self.df['Best Odds'] // 0.5) * 0.5
    
    def segment_by_bookmaker(self) -> pd.DataFrame:
        """Analyze performance by bookmaker."""
        return self._segment_analysis('Best Bookmaker', 'Bookmaker')
    
    def segment_by_sport(self) -> pd.DataFrame:
        """Analyze performance by sport/league."""
        return self._segment_analysis('Sport Title', 'Sport')
    
    def segment_by_lead_time(self) -> pd.DataFrame:
        """Analyze performance by lead time (time until game starts)."""
        return self._segment_analysis('Lead_Time_Category', 'Lead Time')
    
    def segment_by_bet_type(self) -> pd.DataFrame:
        """Analyze performance by bet type (favorite vs underdog)."""
        return self._segment_analysis('Bet_Type', 'Bet Type')
    
    def _segment_analysis(self, column: str) -> pd.DataFrame:
        """
        Generic segmentation analysis.
        
        Args:
            column: Column to segment by
            segment_name: Name for the segment
        
        Returns:
            DataFrame with analysis results
        """
        if column not in self.df.columns or len(self.df) == 0:
            return pd.DataFrame()
        
        analysis = self.df.groupby(column, observed=True).agg({
            'Bet_Won': ['sum', 'count', 'mean'],
            'ROI': ['mean', 'std'],
            'Profit_Loss': 'sum',
            'Expected Value': 'mean',
            'Best Odds': 'mean'
        }).round(4)
        
        # Flatten column names
        analysis.columns = ['_'.join(col).strip() for col in analysis.columns.values]
        analysis = analysis.rename(columns={
            'Bet_Won_sum': 'Wins',
            'Bet_Won_count': 'Total_Bets',
            'Bet_Won_mean': 'Win_Rate',
            'ROI_mean': 'Avg_ROI',
            'ROI_std': 'ROI_StdDev',
            'Profit_Loss_sum': 'Total_Profit',
            'Expected Value_mean': 'Avg_EV',
            'Best Odds_mean': 'Avg_Odds'
        })
        
        # Convert to percentages
        analysis['Win_Rate'] = (analysis['Win_Rate'] * 100).round(2)
        analysis['Avg_ROI'] = analysis['Avg_ROI'].round(2)
        analysis['Avg_EV'] = (analysis['Avg_EV'] * 100).round(2)
        
        # Calculate Sharpe ratio (risk-adjusted return)
        analysis['Sharpe_Ratio'] = (
            analysis['Avg_ROI'] / analysis['ROI_StdDev']
        ).round(3)
        
        # Sort by total profit
        analysis = analysis.sort_values('Total_Profit', ascending=False)
        
        return analysis
    
    def print_segmentation_analysis(self):
        """Print comprehensive segmentation analysis."""
        print("\n" + "="*100)
        print("SEGMENTATION ANALYSIS")
        print("="*100)
        
        segments = [
            ('Bookmaker', self.segment_by_bookmaker()),
            ('Sport/League', self.segment_by_sport()),
            ('Lead Time', self.segment_by_lead_time()),
            ('Bet Type', self.segment_by_bet_type())
        ]
        
        for name, analysis in segments:
            if not analysis.empty:
                print(f"\nBY {name.upper()}")
                print("-" * 100)
                print(analysis.to_string())
        
        print("\n" + "="*100 + "\n")
    
    def plot_segmentation_overview(self, save_path: Optional[str] = None):
        """
        Create visualization of key segments.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.df) == 0:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. ROI by Bookmaker
        ax1 = axes[0, 0]
        bookmaker_data = self.segment_by_bookmaker()
        if not bookmaker_data.empty:
            bookmaker_data.head(10).plot(kind='barh', y='Avg_ROI', ax=ax1, 
                                         color='steelblue', alpha=0.8)
            ax1.set_title('ROI by Bookmaker (Top 10)')
            ax1.set_xlabel('Average ROI (%)')
            ax1.axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # 2. ROI by Sport
        ax2 = axes[0, 1]
        sport_data = self.segment_by_sport()
        if not sport_data.empty:
            sport_data.plot(kind='barh', y='Avg_ROI', ax=ax2, 
                           color='coral', alpha=0.8)
            ax2.set_title('ROI by Sport')
            ax2.set_xlabel('Average ROI (%)')
            ax2.axvline(x=0, color='red', linestyle='--', linewidth=1)
        
        # 3. ROI by Lead Time
        ax3 = axes[0, 2]
        lead_time_data = self.segment_by_lead_time()
        if not lead_time_data.empty:
            lead_time_data.plot(kind='bar', y='Avg_ROI', ax=ax3, 
                               color='mediumseagreen', alpha=0.8, legend=False)
            ax3.set_title('ROI by Lead Time')
            ax3.set_ylabel('Average ROI (%)')
            ax3.set_xticklabels(lead_time_data.index, rotation=45)
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        # 4. Win Rate by Bet Type
        ax4 = axes[1, 0]
        bet_type_data = self.segment_by_bet_type()
        if not bet_type_data.empty:
            bet_type_data.plot(kind='bar', y='Win_Rate', ax=ax4, 
                              color='mediumpurple', alpha=0.8, legend=False)
            ax4.set_title('Win Rate by Bet Type')
            ax4.set_ylabel('Win Rate (%)')
            ax4.set_xticklabels(bet_type_data.index, rotation=45)
            ax4.axhline(y=50, color='red', linestyle='--', linewidth=1)
        
        # 5. Sample Size by Sport
        ax5 = axes[1, 1]
        if not sport_data.empty:
            sport_data.plot(kind='barh', y='Total_Bets', ax=ax5, 
                           color='gold', alpha=0.8)
            ax5.set_title('Sample Size by Sport')
            ax5.set_xlabel('Number of Bets')
        
        # 6. Lead Time vs Sample Size
        ax6 = axes[1, 2]
        lead_time_data = self.segment_by_lead_time()
        if not lead_time_data.empty:
            colors = ['lightcoral' if x < 30 else 'lightgreen' 
                     for x in lead_time_data['Total_Bets']]
            lead_time_data.plot(kind='bar', y='Total_Bets', ax=ax6, 
                               color=colors, alpha=0.8, legend=False)
            ax6.set_title('Sample Size by Lead Time')
            ax6.set_ylabel('Number of Bets')
            ax6.set_xticklabels(lead_time_data.index, rotation=45)
            ax6.axhline(y=30, color='orange', linestyle='--', 
                       label='Minimum for significance', linewidth=1)
            ax6.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()


def analyze_segments(csv_path: str, starting_bankroll: float = 10000.0,
                    plot: bool = True, save_plot: Optional[str] = None):
    """
    Analyze betting performance across segments from a CSV file.
    
    Args:
        csv_path: Path to the CSV file
        starting_bankroll: Starting bankroll
        plot: Whether to create plots
        save_plot: Optional path to save plot
    
    Returns:
        SegmentationAnalyzer instance
    """
    df = pd.read_csv(csv_path)
    analyzer = SegmentationAnalyzer(df, starting_bankroll)
    
    analyzer.print_segmentation_analysis()
    
    if plot:
        analyzer.plot_segmentation_overview(save_plot)
    
    return analyzer


if __name__ == "__main__":
    csv_paths = ["data/nc_avg_minimal.csv", "data/nc_mod_zscore_minimal.csv", "data/nc_random_minimal.csv"]
    
    for csv_path in csv_paths:
        analyzer = analyze_segments(csv_path, starting_bankroll=100.0, plot=True)
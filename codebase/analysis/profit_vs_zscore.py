import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from codebase.results.results_configs import PENDING_RESULTS

# Read your CSV
df = pd.read_csv('codebase/data/master_nc_mod_zscore_full.csv')

# Filter out pending results - only consider bets that have been placed/settled
df = df[~df['Result'].isin(PENDING_RESULTS)]

SHRINK_FACTOR = 0.8

def calculate_kelly_bet(row):
    """
    Calculate Kelly bet size with shrinkage factor
    Kelly = p - ((1 - p) / b)
    where b = decimal odds - 1, p = win probability
    """
    if pd.isna(row['Best Odds']) or pd.isna(row['Fair Odds Avg']):
        return np.nan
    
    if row["Modified Z Score"] > 4:
        return 5
    elif row["Modified Z Score"] > 3:
        return 5
    
    # Calculate implied probability from fair odds
    p = 1 / row['Fair Odds Avg']
    p = max(min(p, 0.9999), 0.0001)
    
    # Best odds available
    best_odds = row['Best Odds']
    b = best_odds - 1  # Profit
    
    f_kelly = p - ((1 - p) / b)
    
    # Only bet if Kelly is positive (positive edge)
    if f_kelly <= 0:
        return 0
    
    # Apply shrinkage factor
    f_adjusted = f_kelly * SHRINK_FACTOR
    
    return min(f_adjusted * 100, 5)

def calculate_profit_with_kelly(row):
    """Calculate profit based on Kelly-sized bet and result"""
    bet_size = calculate_kelly_bet(row)
    
    if pd.isna(bet_size) or bet_size == 0:
        return np.nan
    
    if row['Result'] == row['Team']:  # Win
        return bet_size * (row['Best Odds'] - 1)
    else:  # Loss
        return -bet_size

# Calculate bet sizes and profits
df['Kelly_Bet_Size'] = df.apply(lambda row: calculate_kelly_bet(row), axis=1)
df['Profit'] = df.apply(lambda row: calculate_profit_with_kelly(row), axis=1)
df['ROI'] = (df['Profit'] / df['Kelly_Bet_Size']) * 100

# Remove rows with missing data
df_clean = df.dropna(subset=['Modified Z Score', 'Profit', 'Kelly_Bet_Size'])

print(f"Total rows in CSV: {len(pd.read_csv('codebase/data/master_nc_mod_zscore_full.csv'))}")
print(f"Rows after filtering pending results: {len(df)}")
print(f"Rows with complete data for analysis: {len(df_clean)}")

# Create bins for Modified Z Score
df_clean['Z_Score_Bin'] = pd.cut(df_clean['Modified Z Score'], 
                                   bins=[-np.inf, 1, 2, 3, 4, np.inf],
                                   labels=['<1', '1-2', '2-3', '3-4', '4+'])

# Comprehensive analysis by Z-Score bins
print("\n" + "=" * 80)
print(f"PROFITABILITY ANALYSIS BY MODIFIED Z-SCORE ({SHRINK_FACTOR} Kelly Sizing)")
print("=" * 80)

profitability_stats = df_clean.groupby('Z_Score_Bin').agg({
    'Profit': ['sum', 'mean', 'std'],
    'Kelly_Bet_Size': ['mean', 'sum'],
    'ROI': 'mean',
    'Modified Z Score': 'mean'
}).round(2)

profitability_stats.columns = ['Total_Profit', 'Avg_Profit', 'Std_Profit', 
                                'Avg_Bet_Size', 'Total_Wagered', 
                                'Avg_ROI', 'Avg_Z_Score']

print("\nProfitability Statistics by Z-Score Range:")
print(profitability_stats)

# Calculate win rate and count by Z-Score bin
def calculate_stats(group):
    wins = (group['Result'] == group['Team']).sum()
    total = len(group)
    win_rate = wins / total if total > 0 else 0
    
    return pd.Series({
        'Count': total,
        'Wins': wins,
        'Win_Rate': win_rate,
        'Expected_Win_Rate': (1 / group['Fair Odds Avg']).mean()
    })

additional_stats = df_clean.groupby('Z_Score_Bin').apply(calculate_stats)
print("\nWin Rate Statistics by Z-Score Range:")
print(additional_stats.round(3))

# Overall statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)
print(f"Total Bets: {len(df_clean)}")
print(f"Total Profit: ${df_clean['Profit'].sum():,.2f}")
print(f"Total Wagered: ${df_clean['Kelly_Bet_Size'].sum():,.2f}")
print(f"Overall ROI: {(df_clean['Profit'].sum() / df_clean['Kelly_Bet_Size'].sum() * 100):.2f}%")
print(f"Average Bet Size: ${df_clean['Kelly_Bet_Size'].mean():.2f}")
print(f"Win Rate: {((df_clean['Result'] == df_clean['Team']).sum() / len(df_clean) * 100):.2f}%")

# Correlation analysis
correlation = df_clean['Modified Z Score'].corr(df_clean['Profit'])
print(f"\nCorrelation (Z-Score vs Profit): {correlation:.3f}")

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_clean['Modified Z Score'], 
    df_clean['Profit']
)
print(f"Linear regression p-value: {p_value:.4f}")
print(f"R-squared: {r_value**2:.3f}")
print(f"Slope: ${slope:.2f} profit per unit increase in Z-Score")

# Visualizations with improved layout
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 1. Scatter plot: Profit vs Z-Score
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(df_clean['Modified Z Score'], df_clean['Profit'], alpha=0.6)
ax1.plot(df_clean['Modified Z Score'], 
         intercept + slope * df_clean['Modified Z Score'], 
         'r-', label=f'Fit (R²={r_value**2:.3f})')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Modified Z-Score', fontsize=10)
ax1.set_ylabel('Profit ($)', fontsize=10)
ax1.set_title(f'Profit vs Modified Z-Score ({SHRINK_FACTOR} Kelly)', fontsize=10, pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Average profit by Z-Score bin
ax2 = fig.add_subplot(gs[0, 1])
profit_by_bin = df_clean.groupby('Z_Score_Bin')['Profit'].mean()
colors = ['red' if x < 0 else 'green' for x in profit_by_bin.values]
ax2.bar(range(len(profit_by_bin)), profit_by_bin.values, color=colors, alpha=0.7)
ax2.set_xticks(range(len(profit_by_bin)))
ax2.set_xticklabels(profit_by_bin.index, fontsize=9)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Modified Z-Score Range', fontsize=10)
ax2.set_ylabel('Average Profit per Bet ($)', fontsize=10)
ax2.set_title('Average Profit by Z-Score Range', fontsize=10, pad=10)
ax2.grid(True, alpha=0.3)

# 3. Total profit by Z-Score bin
ax3 = fig.add_subplot(gs[0, 2])
cumulative_profit = df_clean.groupby('Z_Score_Bin')['Profit'].sum()
colors = ['red' if x < 0 else 'green' for x in cumulative_profit.values]
ax3.bar(range(len(cumulative_profit)), cumulative_profit.values, color=colors, alpha=0.7)
ax3.set_xticks(range(len(cumulative_profit)))
ax3.set_xticklabels(cumulative_profit.index, fontsize=9)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('Modified Z-Score Range', fontsize=10)
ax3.set_ylabel('Total Profit ($)', fontsize=10)
ax3.set_title('Cumulative Profit by Z-Score Range', fontsize=10, pad=10)
ax3.grid(True, alpha=0.3)

# 4. Win rate by Z-Score bin
ax4 = fig.add_subplot(gs[1, 0])
win_rates = additional_stats['Win_Rate']
expected_rates = additional_stats['Expected_Win_Rate']
x = np.arange(len(win_rates))
width = 0.35
ax4.bar(x - width/2, win_rates.values, width, label='Actual', alpha=0.7)
ax4.bar(x + width/2, expected_rates.values, width, label='Expected', alpha=0.7)
ax4.set_xticks(x)
ax4.set_xticklabels(win_rates.index, fontsize=9)
ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Break-even')
ax4.set_xlabel('Modified Z-Score Range', fontsize=10)
ax4.set_ylabel('Win Rate', fontsize=10)
ax4.set_title('Win Rate by Z-Score Range', fontsize=10, pad=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. ROI by Z-Score bin
ax5 = fig.add_subplot(gs[1, 1])
roi_by_bin = df_clean.groupby('Z_Score_Bin')['ROI'].mean()
colors = ['red' if x < 0 else 'green' for x in roi_by_bin.values]
ax5.bar(range(len(roi_by_bin)), roi_by_bin.values, color=colors, alpha=0.7)
ax5.set_xticks(range(len(roi_by_bin)))
ax5.set_xticklabels(roi_by_bin.index, fontsize=9)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Modified Z-Score Range', fontsize=10)
ax5.set_ylabel('ROI (%)', fontsize=10)
ax5.set_title('Average ROI by Z-Score Range', fontsize=10, pad=10)
ax5.grid(True, alpha=0.3)

# 6. Bet count and average bet size by Z-Score bin
ax6 = fig.add_subplot(gs[1, 2])
bet_counts = df_clean.groupby('Z_Score_Bin').size()
avg_bet_sizes = df_clean.groupby('Z_Score_Bin')['Kelly_Bet_Size'].mean()
ax6_twin = ax6.twinx()
ax6.bar(range(len(bet_counts)), bet_counts.values, alpha=0.7, color='steelblue', label='Count')
ax6_twin.plot(range(len(avg_bet_sizes)), avg_bet_sizes.values, 'ro-', linewidth=2, markersize=8, label='Avg Bet Size')
ax6.set_xticks(range(len(bet_counts)))
ax6.set_xticklabels(bet_counts.index, fontsize=9)
ax6.set_xlabel('Modified Z-Score Range', fontsize=10)
ax6.set_ylabel('Number of Bets', color='steelblue', fontsize=10)
ax6_twin.set_ylabel('Average Bet Size ($)', color='red', fontsize=10)
ax6.set_title('Bet Count and Average Size by Z-Score', fontsize=10, pad=10)
ax6.tick_params(axis='y', labelcolor='steelblue')
ax6_twin.tick_params(axis='y', labelcolor='red')
ax6.grid(True, alpha=0.3)

plt.savefig('kelly_zscore_profitability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nAnalysis complete! Chart saved as 'kelly_zscore_profitability_analysis.png'")

# Export detailed results
results_df = df_clean[['Match', 'Team', 'Modified Z Score', 'Best Odds', 
                        'Fair Odds Avg', 'Kelly_Bet_Size', 'Result', 'Profit', 'ROI']]
results_df.to_csv('kelly_betting_results_analysis.csv', index=False)
print("Detailed results exported to 'kelly_betting_results_analysis.csv'")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from codebase.results_appending.results_configs import PENDING_RESULTS

# Read the CSV
df = pd.read_csv('codebase/data/master_nc_pin_full.csv')

# Filter out pending results - only consider bets that have been placed/settled
df = df[~df['Result'].isin(PENDING_RESULTS)]

# Kelly criterion parameters
SHRINK_FACTOR = 0.8  # Fractional Kelly for risk management
MAX_BET_PERCENT = 5.0  # Maximum bet as % of bankroll

def calculate_expected_value(row):
    """
    Calculate Expected Value (EV) of a bet
    EV = (Probability of Win × Profit if Win) - (Probability of Loss × Loss if Loss)
    EV = (p × (odds - 1)) - ((1 - p) × 1)
    """
    if pd.isna(row['Best Odds']) or pd.isna(row['Pinnacle Fair Odds']):
        return np.nan
    
    # Implied probability from fair odds
    p = 1 / row['Pinnacle Fair Odds']
    p = max(min(p, 0.9999), 0.0001)  # Clamp between 0.01% and 99.99%
    
    # Best available odds
    best_odds = row['Best Odds']
    
    # EV calculation
    ev = (p * (best_odds - 1)) - ((1 - p) * 1)
    
    return ev

def calculate_kelly_bet_size(row):
    """
    Calculate Kelly Criterion bet size
    Kelly = (p × b - (1 - p)) / b = p - ((1 - p) / b)
    where b = decimal odds - 1, p = win probability
    """
    if pd.isna(row['Best Odds']) or pd.isna(row['Pinnacle Fair Odds']):
        return np.nan
    
    if row['Expected_Value'] < .05:
        return 0
    
    # Implied probability from fair odds
    p = 1 / row['Pinnacle Fair Odds']
    p = max(min(p, 0.9999), 0.0001)
    
    # Best odds available
    best_odds = row['Best Odds']
    b = best_odds - 1  # Net profit per unit staked
    
    # Kelly formula
    f_kelly = p - ((1 - p) / b)
    
    # Only bet if Kelly is positive (positive edge exists)
    if f_kelly <= 0:
        return 0
    
    # Apply fractional Kelly for risk management
    f_adjusted = f_kelly * SHRINK_FACTOR
    
    # Cap at maximum bet percentage
    return min(f_adjusted * 100, MAX_BET_PERCENT)

def calculate_profit(row):
    """Calculate actual profit from a bet"""
    bet_size = row['Kelly_Bet_Size']
    
    if pd.isna(bet_size) or bet_size == 0:
        return np.nan
    
    # Check if bet won
    if row['Result'] == row['Team']:  # Win
        return bet_size * (row['Best Odds'] - 1)
    else:  # Loss
        return -bet_size

# Calculate Expected Value
df['Expected_Value'] = df.apply(calculate_expected_value, axis=1)
df['EV_Percent'] = (df['Expected_Value'] / 1) * 100  # EV as percentage

# Calculate Kelly bet size
df['Kelly_Bet_Size'] = df.apply(calculate_kelly_bet_size, axis=1)

# Calculate actual profit
df['Profit'] = df.apply(calculate_profit, axis=1)

# Calculate ROI
df['ROI'] = (df['Profit'] / df['Kelly_Bet_Size']) * 100

# Remove rows with missing data for analysis
df_clean = df.dropna(subset=['Expected_Value', 'Kelly_Bet_Size', 'Profit']).copy()

print(f"\nRows with complete data for analysis: {len(df_clean)}")

# Create EV bins for analysis
df_clean['EV_Bin'] = pd.cut(df_clean['EV_Percent'], 
                             bins=[-np.inf, 0, 2, 5, 10, np.inf],
                             labels=['Negative', '0-2%', '2-5%', '5-10%', '10%+'])

# ============================================================================
# ANALYSIS SECTION
# ============================================================================

print("\n" + "=" * 80)
print(f"EXPECTED VALUE ANALYSIS (Kelly Criterion with {SHRINK_FACTOR} shrinkage)")
print("=" * 80)

# Overall statistics
print(f"\nOVERALL STATISTICS:")
print(f"Total Bets: {len(df_clean)}")
print(f"Total Wagered: ${df_clean['Kelly_Bet_Size'].sum():,.2f}")
print(f"Total Profit: ${df_clean['Profit'].sum():,.2f}")
print(f"Overall ROI: {(df_clean['Profit'].sum() / df_clean['Kelly_Bet_Size'].sum() * 100):.2f}%")
print(f"Average Bet Size: ${df_clean['Kelly_Bet_Size'].mean():.2f}")
print(f"Average EV%: {df_clean['EV_Percent'].mean():.2f}%")
print(f"Win Rate: {((df_clean['Result'] == df_clean['Team']).sum() / len(df_clean) * 100):.2f}%")

# Analysis by EV bins
print("\n" + "=" * 80)
print("PROFITABILITY BY EXPECTED VALUE RANGE")
print("=" * 80)

ev_stats = df_clean.groupby('EV_Bin').agg({
    'Profit': ['sum', 'mean', 'std'],
    'Kelly_Bet_Size': ['mean', 'sum'],
    'ROI': 'mean',
    'Expected_Value': 'mean',
    'EV_Percent': 'mean'
}).round(2)

ev_stats.columns = ['Total_Profit', 'Avg_Profit', 'Std_Profit',
                    'Avg_Bet_Size', 'Total_Wagered',
                    'Avg_ROI', 'Avg_EV', 'Avg_EV_Pct']

print("\n" + str(ev_stats))

# Win rate by EV bin
def calculate_win_stats(group):
    wins = (group['Result'] == group['Team']).sum()
    total = len(group)
    win_rate = wins / total if total > 0 else 0
    expected_win_rate = (1 / group['Pinnacle Fair Odds']).mean()
    
    return pd.Series({
        'Count': total,
        'Wins': wins,
        'Win_Rate': win_rate,
        'Expected_Win_Rate': expected_win_rate
    })

win_stats = df_clean.groupby('EV_Bin').apply(calculate_win_stats)
print("\n\nWIN RATE BY EV RANGE:")
print(win_stats.round(3))

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

corr_ev_profit = df_clean['EV_Percent'].corr(df_clean['Profit'])
corr_ev_roi = df_clean['EV_Percent'].corr(df_clean['ROI'])

print(f"Correlation (EV% vs Profit): {corr_ev_profit:.3f}")
print(f"Correlation (EV% vs ROI): {corr_ev_roi:.3f}")

# Linear regression: EV vs Profit
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df_clean['EV_Percent'], 
    df_clean['Profit']
)
print(f"\nLinear Regression (EV% vs Profit):")
print(f"  R-squared: {r_value**2:.3f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Slope: ${slope:.2f} profit per 1% increase in EV")

# Compare theoretical EV to actual results
print("\n" + "=" * 80)
print("THEORETICAL EV vs ACTUAL RESULTS")
print("=" * 80)

total_theoretical_ev = (df_clean['Expected_Value'] * df_clean['Kelly_Bet_Size']).sum()
total_actual_profit = df_clean['Profit'].sum()

print(f"Total Theoretical EV: ${total_theoretical_ev:,.2f}")
print(f"Total Actual Profit: ${total_actual_profit:,.2f}")
print(f"Difference: ${(total_actual_profit - total_theoretical_ev):,.2f}")
print(f"Actual/Expected Ratio: {(total_actual_profit / total_theoretical_ev):.2f}x")

# ============================================================================
# VISUALIZATION SECTION
# ============================================================================

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Scatter plot: Profit vs EV%
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(df_clean['EV_Percent'], df_clean['Profit'], alpha=0.5)
ax1.plot(df_clean['EV_Percent'], 
         intercept + slope * df_clean['EV_Percent'], 
         'r-', linewidth=2, label=f'Fit (R²={r_value**2:.3f})')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('Expected Value (%)', fontsize=10)
ax1.set_ylabel('Actual Profit ($)', fontsize=10)
ax1.set_title('Actual Profit vs Expected Value', fontsize=11, pad=10)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Average profit by EV bin
ax2 = fig.add_subplot(gs[0, 1])
profit_by_ev = df_clean.groupby('EV_Bin')['Profit'].mean()
colors = ['red' if x < 0 else 'green' for x in profit_by_ev.values]
ax2.bar(range(len(profit_by_ev)), profit_by_ev.values, color=colors, alpha=0.7)
ax2.set_xticks(range(len(profit_by_ev)))
ax2.set_xticklabels(profit_by_ev.index, fontsize=9, rotation=15)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Expected Value Range', fontsize=10)
ax2.set_ylabel('Average Profit per Bet ($)', fontsize=10)
ax2.set_title('Average Profit by EV Range', fontsize=11, pad=10)
ax2.grid(True, alpha=0.3)

# 3. Total profit by EV bin
ax3 = fig.add_subplot(gs[0, 2])
cumulative_profit_ev = df_clean.groupby('EV_Bin')['Profit'].sum()
colors = ['red' if x < 0 else 'green' for x in cumulative_profit_ev.values]
ax3.bar(range(len(cumulative_profit_ev)), cumulative_profit_ev.values, color=colors, alpha=0.7)
ax3.set_xticks(range(len(cumulative_profit_ev)))
ax3.set_xticklabels(cumulative_profit_ev.index, fontsize=9, rotation=15)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('Expected Value Range', fontsize=10)
ax3.set_ylabel('Total Profit ($)', fontsize=10)
ax3.set_title('Cumulative Profit by EV Range', fontsize=11, pad=10)
ax3.grid(True, alpha=0.3)

# 4. Win rate by EV bin
ax4 = fig.add_subplot(gs[1, 0])
win_rates_ev = win_stats['Win_Rate']
expected_rates_ev = win_stats['Expected_Win_Rate']
x = np.arange(len(win_rates_ev))
width = 0.35
ax4.bar(x - width/2, win_rates_ev.values, width, label='Actual', alpha=0.7, color='steelblue')
ax4.bar(x + width/2, expected_rates_ev.values, width, label='Expected', alpha=0.7, color='orange')
ax4.set_xticks(x)
ax4.set_xticklabels(win_rates_ev.index, fontsize=9, rotation=15)
ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Break-even')
ax4.set_xlabel('Expected Value Range', fontsize=10)
ax4.set_ylabel('Win Rate', fontsize=10)
ax4.set_title('Win Rate by EV Range', fontsize=11, pad=10)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5. ROI by EV bin
ax5 = fig.add_subplot(gs[1, 1])
roi_by_ev = df_clean.groupby('EV_Bin')['ROI'].mean()
colors = ['red' if x < 0 else 'green' for x in roi_by_ev.values]
ax5.bar(range(len(roi_by_ev)), roi_by_ev.values, color=colors, alpha=0.7)
ax5.set_xticks(range(len(roi_by_ev)))
ax5.set_xticklabels(roi_by_ev.index, fontsize=9, rotation=15)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_xlabel('Expected Value Range', fontsize=10)
ax5.set_ylabel('ROI (%)', fontsize=10)
ax5.set_title('Average ROI by EV Range', fontsize=11, pad=10)
ax5.grid(True, alpha=0.3)

# 6. Bet count and average bet size by EV bin
ax6 = fig.add_subplot(gs[1, 2])
bet_counts_ev = df_clean.groupby('EV_Bin').size()
avg_bet_sizes_ev = df_clean.groupby('EV_Bin')['Kelly_Bet_Size'].mean()
ax6_twin = ax6.twinx()
ax6.bar(range(len(bet_counts_ev)), bet_counts_ev.values, alpha=0.7, color='steelblue')
ax6_twin.plot(range(len(avg_bet_sizes_ev)), avg_bet_sizes_ev.values, 'ro-', 
              linewidth=2, markersize=8, label='Avg Bet Size')
ax6.set_xticks(range(len(bet_counts_ev)))
ax6.set_xticklabels(bet_counts_ev.index, fontsize=9, rotation=15)
ax6.set_xlabel('Expected Value Range', fontsize=10)
ax6.set_ylabel('Number of Bets', color='steelblue', fontsize=10)
ax6_twin.set_ylabel('Average Bet Size ($)', color='red', fontsize=10)
ax6.set_title('Bet Count and Size by EV Range', fontsize=11, pad=10)
ax6.tick_params(axis='y', labelcolor='steelblue')
ax6_twin.tick_params(axis='y', labelcolor='red')
ax6.grid(True, alpha=0.3)

# 7. Distribution of Expected Value
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(df_clean['EV_Percent'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax7.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero EV')
ax7.axvline(x=df_clean['EV_Percent'].mean(), color='orange', linestyle='--', 
            linewidth=2, label=f'Mean: {df_clean["EV_Percent"].mean():.2f}%')
ax7.set_xlabel('Expected Value (%)', fontsize=10)
ax7.set_ylabel('Frequency', fontsize=10)
ax7.set_title('Distribution of Expected Value', fontsize=11, pad=10)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3)

# 8. Theoretical EV vs Actual Profit by EV bin
ax8 = fig.add_subplot(gs[2, 1])
theoretical_by_bin = df_clean.groupby('EV_Bin').apply(
    lambda x: (x['Expected_Value'] * x['Kelly_Bet_Size']).sum()
)
actual_by_bin = df_clean.groupby('EV_Bin')['Profit'].sum()
x = np.arange(len(theoretical_by_bin))
width = 0.35
ax8.bar(x - width/2, theoretical_by_bin.values, width, label='Theoretical EV', alpha=0.7, color='orange')
ax8.bar(x + width/2, actual_by_bin.values, width, label='Actual Profit', alpha=0.7, color='steelblue')
ax8.set_xticks(x)
ax8.set_xticklabels(theoretical_by_bin.index, fontsize=9, rotation=15)
ax8.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax8.set_xlabel('Expected Value Range', fontsize=10)
ax8.set_ylabel('Total Value ($)', fontsize=10)
ax8.set_title('Theoretical EV vs Actual Profit', fontsize=11, pad=10)
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Cumulative profit over time (if possible, otherwise EV vs Kelly size)
ax9 = fig.add_subplot(gs[2, 2])
ax9.scatter(df_clean['EV_Percent'], df_clean['Kelly_Bet_Size'], alpha=0.5, color='purple')
ax9.set_xlabel('Expected Value (%)', fontsize=10)
ax9.set_ylabel('Kelly Bet Size (% of bankroll)', fontsize=10)
ax9.set_title('Kelly Bet Size vs Expected Value', fontsize=11, pad=10)
ax9.axhline(y=MAX_BET_PERCENT, color='r', linestyle='--', alpha=0.5, 
            label=f'Max Bet: {MAX_BET_PERCENT}%')
ax9.legend(fontsize=9)
ax9.grid(True, alpha=0.3)

plt.savefig('ev_kelly_profitability_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nVisualization saved as 'ev_kelly_profitability_analysis.png'")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
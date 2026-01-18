import pandas as pd
from src.constants import PENDING_RESULTS
from analysis.betting_utils import calculate_kelly_bet_profit

# Read the CSV file
df = pd.read_csv('data/master_nc_mod_zscore_full.csv')
df = df[~df["Result"].isin(PENDING_RESULTS)].copy()

# Calculate profit using the shared utility function
# This adds 'Bet_Size' and 'Profit' columns
df = calculate_kelly_bet_profit(
    df=df,
    odds_col='Best Odds',
    fair_odds_col='Fair Odds Avg',
    ev_col='Expected Value',
    zscore_col='Modified Z Score',
    result_col='Result',
    team_col='Team'
)

# Calculate cumulative profit
df['Cumulative Profit'] = df['Profit'].cumsum()

# Save to new CSV file
df.to_csv('betting_results_with_profit.csv', index=False)

print("Profit calculation complete!")
print(f"\nTotal Profit: {df['Profit'].sum():.2f} units")
print(f"Final Cumulative Profit: {df['Cumulative Profit'].iloc[-1]:.2f} units")
print(f"Win Rate: {(df['Profit'] > 0).sum() / len(df):.2%}")

df2 = pd.read_csv('kelly_debug.csv')
df2 = df2[df2["Bet_Size"] != 0]
print(len(df2))
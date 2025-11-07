import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class BettingStrategy:
    name: str
    path: str
    fair_odds_column: str

STRATEGIES = [
    BettingStrategy("Average", "codebase/data/master_avg_full.csv", "Fair Odds Avg"),
    BettingStrategy("Modified Zscore", "codebase/data/master_mod_zscore_full.csv",  "Fair Odds Avg"),
    BettingStrategy("Pinnacle", "codebase/data/master_pin_full.csv", "Pinnacle Fair Odds"),
    BettingStrategy("Zscore", "codebase/data/master_zscore_full.csv",  "Fair Odds Avg"),
    BettingStrategy("Average NC", "codebase/data/master_nc_avg_full.csv", "Fair Odds Avg"),
    BettingStrategy("Modified Zscore NC", "codebase/data/master_nc_mod_zscore_full.csv",  "Fair Odds Avg"),
    BettingStrategy("Pinnacle NC", "codebase/data/master_nc_pin_full.csv", "Pinnacle Fair Odds"),
    BettingStrategy("Zscore NC", "codebase/data/master_nc_zscore_full.csv",  "Fair Odds Avg"),
]

# Strategy to test
strat = STRATEGIES[4]

# Path to df
path = strat.path

# Fair odds column
fair_odds_col = strat.fair_odds_column

# === Load your dataset ===
df = pd.read_csv(path)

# --- Clean ---
# Drop rows with missing or "Not Found" results
df = df[df["Result"].notna()]
df = df[df["Result"].str.lower() != "not found"]

# Predicted probability = 1 / Fair Odds Avg
df[fair_odds_col] = pd.to_numeric(df[fair_odds_col], errors="coerce")
df["p_hat"] = 1.0 / df[fair_odds_col]

# Outcome: 1 if Team won, else 0
df["win"] = (df["Result"] == df["Team"]).astype(int)

# Drop any rows missing predictions or outcomes
df = df.dropna(subset=["p_hat", "win"])

# --- Calibration-based variance estimation ---

# Bin predictions into quantiles (e.g., deciles)
n_bins = 10
df["p_bin"] = pd.qcut(df["p_hat"], q=n_bins, duplicates="drop")

# Compute observed win rates per bin
bin_stats = df.groupby("p_bin").agg(
    avg_pred=("p_hat", "mean"),
    obs_win_rate=("win", "mean"),
    n=("win", "size")
).reset_index()

# Within-bin variance of actual outcomes (empirical)
bin_stats["var_obs"] = bin_stats["obs_win_rate"] * (1 - bin_stats["obs_win_rate"]) / bin_stats["n"]

# Deviation between predicted and observed probabilities per bin
bin_stats["calibration_error"] = (bin_stats["obs_win_rate"] - bin_stats["avg_pred"])**2
mse = bin_stats["calibration_error"].mean()

print("=== Calibration-based variance estimation ===")
print(f"Estimated Var(MSE): {mse:.6f}")
print(f"Estimated Std(RMSE): {np.sqrt(mse):.6f}")
print("\nBin summary:")
print(bin_stats[["avg_pred", "obs_win_rate", "n", "calibration_error"]])

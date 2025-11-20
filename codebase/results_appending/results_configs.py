"""
results_configs.py

Centralized variables for results folder.

Author: Andrew Smith
"""

from zoneinfo import ZoneInfo

# Local timezone for measuring how long it has been since each bet
TIMEZONE = ZoneInfo("America/New_York")

# How many days back in The-Odds_API to look back for results (max of 3 days)
DAYS_CUTOFF = 3

# How long to sleep after TheSportsDB request limit (30 requests/min)
SLEEP_DURATION = 60

# File names
FILE_CONFIGS = [
    ("master_avg_bets.csv", "master_avg_full.csv"),
    ("master_nc_avg_bets.csv", "master_nc_avg_full.csv"),
    ("master_mod_zscore_bets.csv", "master_mod_zscore_full.csv"),
    ("master_nc_mod_zscore_bets.csv", "master_nc_mod_zscore_full.csv"),
    ("master_pin_bets.csv", "master_pin_full.csv"),
    ("master_nc_pin_bets.csv", "master_nc_pin_full.csv"),
    ("master_zscore_bets.csv", "master_zscore_full.csv"),
    ("master_nc_zscore_bets.csv", "master_nc_zscore_full.csv"),
    ("master_random_bets.csv", "master_random_full.csv"),
    ("master_nc_random_bets.csv", "master_nc_random_full.csv"),
]

# Unfinished results
PENDING_RESULTS = ["Not Found", "Pending", "API Error"]

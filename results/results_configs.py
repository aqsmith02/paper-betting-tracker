from zoneinfo import ZoneInfo

# Configuration
TIMEZONE = ZoneInfo("America/New_York")
DAYS_CUTOFF = 3
SLEEP_DURATION = 60

# File configurations
FILE_CONFIGS = [
    ("master_avg_bets.csv", "master_avg_full.csv"),
    ("master_mod_zscore_bets.csv", "master_mod_zscore_full.csv"),
    ("master_pin_bets.csv", "master_pin_full.csv"),
    ("master_zscore_bets.csv", "master_zscore_full.csv"),
    ("master_random_bets.csv", "master_random_full.csv"),
]

PENDING_RESULTS = ["Not Found", "Pending", "API Error"]
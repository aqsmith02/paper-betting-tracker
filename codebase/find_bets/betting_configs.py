"""
betting_configs.py

Centralized variables for find_bets folder.

Author: Andrew Smith
"""

# Betting thresholds
EDGE_THRESHOLD = 0.05
Z_SCORE_THRESHOLD = 2.0
MAX_Z_SCORE = 6.0
MIN_BOOKMAKERS = 5
MAX_ODDS = 50
MAX_MISSING_VF_PCT = 0.2

# Formatting
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Betting exchanges to exclude (they work differently than traditional bookmakers)
EXCHANGE_BLOCKLIST = {"Smarkets", "Betfair", "Matchbook", "Betfair Sportsbook"}

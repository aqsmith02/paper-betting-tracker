"""
Fetch Odds Configuration

Configuration for fetching odds from The Odds API.
"""

# ============================================================================
# Fetch Odds Parameters
# ============================================================================

# Sport to fetch odds for
# Options: "upcoming", or specific sport keys like "mlb", "kbo", etc.
SPORT = "upcoming"

# Sport key
SPORT_KEY = "upcoming"

# Regions to get odds from
# Options: "us", "uk", "eu", etc.
REGIONS = "us,us_ex,eu"

# Markets to fetch
# Options: "h2h" (moneyline), "spreads", etc.
MARKETS = "h2h"

# Odds format
# Options: "decimal", "american", etc.
ODDS_FORMAT = "decimal"

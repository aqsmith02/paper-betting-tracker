from pathlib import Path

# Data directory
CONFIG_DIR = Path("config")
DATA_DIR = Path("data")

# Available Sports Keys
THEODDS_SPORTS_DICT = {
    "upcoming": "upcoming",
    "kbo": "baseball_kbo",
    "mlb": "baseball_mlb",
    "ncaa_baseball": "baseball_ncaa",
    "wnba": "basketball_wnba",
    "brazil_serie_a": "soccer_brazil_campeonato",
    "brazil_serie_b": "soccer_brazil_serie_b",
    "super_league_china": "soccer_china_superleague",
    "japan_league": "soccer_japan_j_league",
    "mls": "soccer_usa_mls",
    "cfl": "americanfootball_cfl",
    "aussie": "aussierules_afl",
    "npb": "baseball_npb",
    "boxing": "boxing_boxing",
    "cricket": "cricket_t20_blast",
    "lacrosse": "lacrosse_pll",
    "rugby": "rugbyleague_nrl",
    "mma": "mma_mixed_martial_arts",
    "euroleague": "soccer_uefa_european_championship",
    "finland": "soccer_finland_veikkausliiga",
    "nhl": "icehockey_nhl",
    "sweden_hockey": "icehockey_sweden_hockey_league",
    "mexico": "soccer_mexico_ligamx",
    "ireland": "soccer_league_of_ireland",
}

# Betting thresholds
EV_THRESHOLD = 0.05
Z_SCORE_THRESHOLD = 2.0
MAX_Z_SCORE = 5.0
MIN_BOOKMAKERS = 5
MAX_ODDS = 50
MAX_MISSING_VF_PCT = 0.2
TARGET_BMS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics", "Pinnacle", "BetOnline.ag"]
NC_BMS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics"]

# Formatting
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Betting exchanges to exclude (they work differently than traditional bookmakers)
EXCHANGE_BLOCKLIST = {"Smarkets", "Betfair", "Matchbook", "Betfair Sportsbook"}

# How many days back in The-Odds_API to look back for results (max of 3 days)
DAYS_CUTOFF = 3

# How long to sleep after TheSportsDB request limit (30 requests/min)
SLEEP_DURATION = 60

# File names
FILE_NAMES = [
    ("master_nc_avg_bets.csv", "master_nc_avg_full.csv"),
    ("master_nc_mod_zscore_bets.csv", "master_nc_mod_zscore_full.csv"),
    ("master_nc_random_bets.csv", "master_nc_random_full.csv"),
]

# Unfinished results
PENDING_RESULTS = ["Not Found", "Pending", "API Error"]
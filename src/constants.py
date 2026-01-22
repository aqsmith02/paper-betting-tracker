from pathlib import Path

# Data directory
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
MAX_ODDS = 30
TARGET_BMS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics", "Pinnacle", "BetOnline.ag"]
NC_BMS = ["FanDuel", "DraftKings", "BetMGM", "Caesars", "Fanatics"]

# Formatting
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# How many days back in The-Odds_API to look back for results (max of 3 days)
DAYS_CUTOFF = 3

# How long to sleep after TheSportsDB request limit (30 requests/min)
SLEEP_DURATION = 60

# File names
FILE_NAMES = [
    ("nc_avg_minimal.csv", "nc_avg_full.csv"),
    ("nc_mod_zscore_minimal.csv", "nc_mod_zscore_full.csv"),
    ("nc_random_minimal.csv", "nc_random_full.csv"),
]

# Unfinished results
PENDING_RESULTS = ["Not Found", "Pending", "API Error"]

# Non Bookmaker Columns
NON_BM_COLUMNS = {"ID","Sport Key","Sport Title","Start Time","Scrape Time","Match","Team","Best Odds","Best Bookmaker","Outcomes","Result", "Fair Odds Average", "Expected Value", "Modified Z-Score", "Random Placed Bet"}

# Column to add new columns before when merging DataFrames
INSERT_BEFORE_COLUMN = "Best Bookmaker"

# API Request Threshold Hours
API_REQUEST_THRESHOLD_HOURS = 12

# TheSportsDB
SPORTSDB_RATE_LIMIT_BATCH = 30
SPORTSDB_RATE_LIMIT_WAIT = 60

# The-Odds-API
DAYS_FROM_SCORE_FETCHING = 3
SPORT_KEY_COLUMN = "Sport Key"

# Column names
START_TIME_COLUMN = "Start Time"
RESULT_COLUMN = "Result"
ID_COLUMN = "ID"
SPORT_KEY_COLUMN = "Sport Key"
MATCH_COLUMN = "Match"

# Sports with results available on The-Odds-API
SPORT_KEYS_WITH_RESULTS = [
    "americanfootball_cfl",
    "americanfootball_ncaaf",
    "americanfootball_nfl",
    "americanfootball_nfl_preseason",
    "americanfootball_ufl",
    "aussierules_afl",
    "baseball_mlb",
    "basketball_euroleague",
    "basketball_nba",
    "basketball_nba_preseason",
    "basketball_nba_summer_league",
    "basketball_wnba",
    "basketball_ncaab",
    "basketball_nbl",
    "handball_germany_bundesliga",
    "icehockey_nhl",
    "icehockey_nhl_preseason",
    "icehockey_sweden_hockey_league",
    "icehockey_sweden_allsvenskan",
    "rugbyleague_nrl",
    "soccer_argentina_primera_division",
    "soccer_australia_aleague",
    "soccer_austria_bundesliga",
    "soccer_belgium_first_div",
    "soccer_brazil_campeonato",
    "soccer_brazil_serie_b",
    "soccer_chile_campeonato",
    "soccer_china_superleague",
    "soccer_denmark_superliga",
    "soccer_efl_champ",
    "soccer_england_efl_cup",
    "soccer_england_league1",
    "soccer_england_league2",
    "soccer_epl",
    "soccer_fa_cup",
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_womens",
    "soccer_fifa_club_world_cup",
    "soccer_finland_veikkausliiga",
    "soccer_france_ligue_one",
    "soccer_france_ligue_two",
    "soccer_germany_bundesliga",
    "soccer_germany_bundesliga2",
    "soccer_germany_liga3",
    "soccer_greece_super_league",
    "soccer_italy_serie_a",
    "soccer_italy_serie_b",
    "soccer_japan_j_league",
    "soccer_korea_kleague1",
    "soccer_league_of_ireland",
    "soccer_mexico_ligamx",
    "soccer_netherlands_eredivisie",
    "soccer_norway_eliteserien",
    "soccer_poland_ekstraklasa",
    "soccer_portugal_primeira_liga",
    "soccer_russia_premier_league",
    "soccer_spain_la_liga",
    "soccer_spain_segunda_division",
    "soccer_spl",
    "soccer_sweden_allsvenskan",
    "soccer_sweden_superettan",
    "soccer_switzerland_superleague",
    "soccer_turkey_super_league",
    "soccer_uefa_europa_conference_league",
    "soccer_uefa_champs_league",
    "soccer_uefa_champs_league_qualification",
    "soccer_uefa_champs_league_women",
    "soccer_uefa_europa_league",
    "soccer_uefa_european_championship",
    "soccer_uefa_euro_qualification",
    "soccer_uefa_nations_league",
    "soccer_concacaf_leagues_cup",
    "soccer_conmebol_copa_america",
    "soccer_conmebol_copa_libertadores",
    "soccer_conmebol_copa_sudamericana",
    "soccer_usa_mls",
]

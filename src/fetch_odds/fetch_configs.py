"""
fetch_configs.py

Centralized variables for fetch_odds folder.

Author: Andrew Smith
"""

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

# API Configuration
SPORT = "upcoming"
SPORT_KEY = THEODDS_SPORTS_DICT[SPORT]
REGIONS = "us"
MARKETS = "h2h"
ODDS_FORMAT = "decimal"

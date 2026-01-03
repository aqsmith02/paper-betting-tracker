TEST_GAME_DATA = {
    "id": "c38b5a8fed92009138c3abbfecdb2bb8",
    "sport_key": "baseball_mlb",
    "sport_title": "MLB",
    "commence_time": "2025-09-24T01:39:00Z",
    "home_team": "Los Angeles Angels",
    "away_team": "Kansas City Royals",
    "bookmakers": [
        {
            "key": "fanduel",
            "title": "FanDuel",
            "last_update": "2025-09-24T04:21:31Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2025-09-24T04:21:31Z",
                    "outcomes": [
                        {"name": "Arizona Diamondbacks", "price": 1.67},
                        {"name": "Los Angeles Dodgers", "price": 2.18},
                    ],
                }
            ],
        },
        {
            "key": "sport888",
            "title": "888sport",
            "last_update": "2025-09-24T04:17:30Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2025-09-24T04:17:30Z",
                    "outcomes": [
                        {"name": "Arizona Diamondbacks", "price": 1.95},
                        {"name": "Los Angeles Dodgers", "price": 1.73},
                    ],
                }
            ],
        },
    ],
}

TEST_GAME_DATA_NO_BM = {
    "id": "c38b5a8fed92009138c3abbfecdb2bb8",
    "sport_key": "baseball_mlb",
    "sport_title": "MLB",
    "commence_time": "2025-09-24T01:39:00Z",
    "home_team": "Los Angeles Angels",
    "away_team": "Kansas City Royals",
    "bookmakers": [],
}

TEST_GAME_DATA_MULT_MARKETS = {
    "id": "c38b5a8fed92009138c3abbfecdb2bb8",
    "sport_key": "baseball_mlb",
    "sport_title": "MLB",
    "commence_time": "2025-09-24T01:39:00Z",
    "home_team": "Los Angeles Angels",
    "away_team": "Kansas City Royals",
    "bookmakers": [
        {
            "key": "smarkets1",
            "title": "Smarkets1",
            "last_update": "2025-09-24T04:20:12Z",
            "markets": [
                {
                    "key": "h2h",
                    "last_update": "2025-09-24T04:20:12Z",
                    "outcomes": [
                        {"name": "Kansas City Royals", "price": 1.01},
                        {"name": "Los Angeles Angels", "price": 17.99},
                    ],
                },
                {
                    "key": "h2h_lay",
                    "last_update": "2025-09-24T04:20:12Z",
                    "outcomes": [
                        {"name": "Kansas City Royals", "price": 1.11},
                        {"name": "Los Angeles Angels", "price": 100.0},
                    ],
                },
            ],
        },
        {
            "key": "smarkets2",
            "title": "Smarkets2",
            "last_update": "2025-09-24T04:20:12Z",
            "markets": [
                {
                    "key": "h2h_lay",
                    "last_update": "2025-09-24T04:20:12Z",
                    "outcomes": [
                        {"name": "Kansas City Royals", "price": 1.11},
                        {"name": "Los Angeles Angels", "price": 100.0},
                    ],
                },
                {
                    "key": "h2h",
                    "last_update": "2025-09-24T04:20:12Z",
                    "outcomes": [
                        {"name": "Kansas City Royals", "price": 1.01},
                        {"name": "Los Angeles Angels", "price": 17.99},
                    ],
                },
            ],
        },
    ],
}

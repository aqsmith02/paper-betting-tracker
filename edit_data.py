# import pandas as pd
# from src.constants import FILE_NAMES_BETS, FILE_NAMES_FULL

# for bets,full in zip(FILE_NAMES_BETS,FILE_NAMES_FULL):
#     bets_df = pd.read_csv(bets)
#     full_df = pd.read_csv(full)
#     bets_df["Start Time Accuracy"] = "Approximate"
#     bets_df["Event Id"] = ""
#     bets_df["Expected Value Method"] = "Old"
#     full_df["Start Time Accuracy"] = "Approximate"
#     full_df["Expected Value Method"] = "Old"

import pandas as pd
from src.constants import FILE_NAMES_BETS, FILE_NAMES_FULL

bets = "data/nc_random_bets.csv"
full = "data/nc_random_full.csv"

bets_df = pd.read_csv(bets)
full_df = pd.read_csv(full)
bets_df["Start Time Accuracy"] = "Approximate"
bets_df["Event Id"] = ""
bets_df["Expected Value Method"] = "Old"
full_df["Start Time Accuracy"] = "Approximate"
full_df["Expected Value Method"] = "Old"

bets_df = bets_df.rename(columns={
    "Random Bet Book": "Sportsbook Placed",
    "Random Bet Odds": "Odds Placed",
})

full_df = full_df.rename(columns={
    "Best Bookmaker": "Sportsbook Placed",
    "Best Odds": "Odds Placed",
    "Fair Odds Avg": "Fair Odds Average"
})

full_df = full_df.drop(columns=["Random Placed Bet"])

# ORDER OF COLS: Match, Team, Start Time, Start Time Accuracy, Event Id, League, Outcomes (if full), Scrape Time, Books (if full), Fair Odds Average (if full), Sportsbook Placed, Odds Placed, EV, EV Method, 

bets_order = [
    "Match",
    "Team",
    "Start Time",
    "Start Time Accuracy",
    "Event Id",
    "League",
    "Scrape Time",
    "Sportsbook Placed",
    "Odds Placed",
    "Result",
]

full_order = [
    "Match",
    "Team",
    "Start Time",
    "Start Time Accuracy",
    "Event Id",
    "League",
    "Outcomes",
    "Scrape Time",
    "Caesars",
    "Draftkings",
    "Fanduel",
    "Pinnacle",
    "Fanatics",
    "Betmgm",
    "Betonline.Ag",
    "Sportsbook Placed",
    "Odds Placed",
    "Result",
]

bets_df = bets_df[bets_order]
full_df = full_df[full_order]

bets_df.to_csv(bets, index=False)
full_df.to_csv(full, index=False)
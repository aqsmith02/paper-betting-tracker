import pandas as pd
from dataclasses import dataclass
from codebase.results.results_configs import PENDING_RESULTS
from datetime import datetime


@dataclass
class BettingStrategy:
    name: str
    path: str
    odds_column: str
    edge_column: str

s1 = BettingStrategy(
    name="Average",
    path="codebase/data/master_avg_bets.csv",
    odds_column="Avg Edge Odds",
    edge_column="Avg Edge Pct",
)
s2 = BettingStrategy(
    name="Modified Zscore",
    path="codebase/data/master_mod_zscore_bets.csv",
    odds_column="Outlier Odds",
    edge_column="Avg Edge Pct",
)
s3 = BettingStrategy(
    name="Pinnacle",
    path="codebase/data/master_pin_bets.csv",
    odds_column="Pinnacle Edge Odds",
    edge_column="Pin Edge Pct",
)
s4 = BettingStrategy(
    name="Zscore",
    path="codebase/data/master_zscore_bets.csv",
    odds_column="Outlier Odds",
    edge_column="Avg Edge Pct",
)
s5 = BettingStrategy(
    name="Random",
    path="codebase/data/master_random_bets.csv",
    odds_column="Random Bet Odds",
    edge_column=None,
)
s6 = BettingStrategy(
    name="Average NC",
    path="codebase/data/master_nc_avg_bets.csv",
    odds_column="Avg Edge Odds",
    edge_column="Avg Edge Pct",
)
s7 = BettingStrategy(
    name="Modified Zscore NC",
    path="codebase/data/master_nc_mod_zscore_bets.csv",
    odds_column="Outlier Odds",
    edge_column="Avg Edge Pct",
)
s8 = BettingStrategy(
    name="Pinnacle NC",
    path="codebase/data/master_nc_pin_bets.csv",
    odds_column="Pinnacle Edge Odds",
    edge_column="Pin Edge Pct",
)
s9 = BettingStrategy(
    name="Zscore NC",
    path="codebase/data/master_nc_zscore_bets.csv",
    odds_column="Outlier Odds",
    edge_column="Avg Edge Pct",
)
s10 = BettingStrategy(
    name="Random NC",
    path="codebase/data/master_nc_random_bets.csv",
    odds_column="Random Bet Odds",
    edge_column=None,
)

STRATEGIES = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]


def calculate_betting_roi(df, odds_col):
    """
    Calculate ROI for betting strategy where $1 is bet on each row.

    Parameters:
    df (pandas.DataFrame): DataFrame containing betting data
    odds_col (str): Name of the column containing decimal odds

    Required columns in df:
    - Team: The team being bet on
    - Result: The actual winning team
    - [odds_col]: Column with decimal odds

    Returns:
    float: Total ROI as a percentage
    """

    # Validate required columns exist
    required_cols = ["Team", "Result", odds_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Calculate total amount wagered ($1 per bet)
    finalized_bets = df[~df["Result"].isin(PENDING_RESULTS)]
    total_wagered = len(finalized_bets)

    winning_bets = finalized_bets["Team"] == finalized_bets["Result"]
    filtered_df = finalized_bets[winning_bets]
    total_winnings = (filtered_df[odds_col]).sum()  # Total payout
    net_profit = total_winnings - total_wagered
    roi_percentage = (net_profit / total_wagered) * 100

    # Calculate how long of a time interval data has been collected
    time_start = datetime.strptime(df.loc[0, "Scrape Time"], "%Y-%m-%d %H:%M:%S")
    time_end = datetime.strptime(df.loc[len(df)-1, "Scrape Time"], "%Y-%m-%d %H:%M:%S")
    time_spent = time_end - time_start

    # Format nicely
    days = time_spent.days
    hours, remainder = divmod(time_spent.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    readable_time = f"{days}d {hours}h {minutes}m {seconds}s"

    output = {
        "Total Bets Placed": round(float(total_wagered), 2),
        "Total Units Wagered": round(float(total_wagered), 2),
        "Unit Winnings": round(float(total_winnings), 2),
        "Net Unit Profit": round(float(net_profit), 2),
        "ROI": round(float(roi_percentage), 2),
        "Time Spent": readable_time,
    }
    return output


def print_strategy_results(strategy):
    df = pd.read_csv(strategy.path)
    results = calculate_betting_roi(df, strategy.odds_column)

    print(f"\n=== {strategy.name} Strategy Results ===")
    for k, v in results.items():
        print(f"{k:20}: {v}")


def print_strategy_results_and_return_roi(strategy):
    df = pd.read_csv(strategy.path)
    results = calculate_betting_roi(df, strategy.odds_column)

    print(f"\n=== {strategy.name} Strategy Results ===")
    for k, v in results.items():
        print(f"{k:20}: {v}")

    return results["ROI"]


def all_strategy_results(strategies):
    for strats in strategies:
        print_strategy_results(strats)


def main():
    all_strategy_results(STRATEGIES)


if __name__ == "__main__":
    main()

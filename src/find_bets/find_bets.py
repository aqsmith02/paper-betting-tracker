"""
find_bets.py

The file fetches odds using a separate file called fetch_odds.py, then identifies profitable bets from them
using two different strategies. The first strategy is comparing all available odds to the average fair odds of an outcome.
The second strategy is the same as the first with an additional modified Z-score constraint. 
A random strategy is also conducted as a control. Profitable bets are then saved into a master .csv file.

Author: Andrew Smith
"""

from src.fetch_odds.fetch_odds import fetch_odds
from src.find_bets.file_management import BetFileManager
from src.find_bets.betting_strategies import (
    analyze_average_edge_bets,
    analyze_modified_zscore_outliers,
    find_random_bets,
)
from src.find_bets.summary_creation import (
    create_random_summary,
    create_average_edge_summary,
    create_modified_zscore_summary,
)
from src.find_bets.data_processing import (
    process_target_odds_data,
)
from src.find_bets.vigfree_probabilities import (
    calculate_vigfree_probabilities,
)
import pandas as pd
from dataclasses import dataclass


@dataclass
class BettingStrategy:
    name: str
    nc_summary_file: str
    nc_full_file: str
    score_column: str
    summary_func: callable
    analysis_func: callable


def run_betting_strategy(
    strategy: BettingStrategy,
    nc_df: pd.DataFrame,
    file_manager: BetFileManager,
) -> None:
    """
    Execute a complete betting strategy analysis and save results.

    Args:
        strategy (BettingStrategy): Betting strategy configuration object.
        vigfree_data (pd.DataFrame): Betting data with vig-free probability calculations.
        file_manager (BetFileManager): File manager for saving results.

    Returns:
        None
    """
    print(f"\nRunning {strategy.name} analysis...")

    try:
        # Run the NC analysis
        nc_analysis_result = strategy.analysis_func(nc_df)
        nc_summary = strategy.summary_func(nc_analysis_result)

        if nc_summary.empty:
            print(f"No NC profitable bets found for {strategy.name}")
        else:
            # Save summary (best bets only) and get the filtered data back
            nc_filtered_summary = file_manager.save_best_bets_only(
                nc_summary, strategy.nc_summary_file, strategy.score_column
            )

            # Save full data using the same filtered summary
            file_manager.save_full_betting_data(
                nc_analysis_result, nc_filtered_summary, strategy.nc_full_file
            )

    except Exception as e:
        print(f"Error running NC {strategy.name}: {e}")


def main():
    """
    Main betting analysis pipeline.

    Args:
        None

    Returns:
        None
    """
    # Initialize file manager
    file_manager = BetFileManager()

    # Define betting strategies
    strategies = [
        BettingStrategy(
            name="Average Edge",
            nc_summary_file="master_nc_avg_bets.csv",
            nc_full_file="master_nc_avg_full.csv",
            score_column="Expected Value",
            summary_func=create_average_edge_summary,
            analysis_func=analyze_average_edge_bets,
        ),
        BettingStrategy(
            name="Modified Z-Score Outliers",
            nc_summary_file="master_nc_mod_zscore_bets.csv",
            nc_full_file="master_nc_mod_zscore_full.csv",
            score_column="Modified Z Score",
            summary_func=create_modified_zscore_summary,
            analysis_func=analyze_modified_zscore_outliers,
        ),
        BettingStrategy(
            name="Random Bets",
            nc_summary_file="master_nc_random_bets.csv",
            nc_full_file="master_nc_random_full.csv",
            score_column="Random Bet Odds",
            summary_func=create_random_summary,
            analysis_func=find_random_bets,
        ),
    ]

    # Run pipeline
    try:
        # Step 1: Fetch and prepare data
        raw_odds = fetch_odds()
        if raw_odds.empty:
            print("No odds data available")
            return

        nc_processed_odds = process_target_odds_data(raw_odds)

        if nc_processed_odds.empty:
            print("No data passed cleaning requirements")
            return

        nc_df = calculate_vigfree_probabilities(nc_processed_odds)

        # Step 2: Run each betting strategy
        for strategy in strategies:
            run_betting_strategy(strategy, nc_df, file_manager)

        print("\nBetting analysis pipeline completed successfully")

    except Exception as e:
        print(f"Betting pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

"""
find_bets.py

The file fetches odds using a separate file called fetch_odds.py, then identifies profitable bets from them
using four different strategies. The strategies are comparing odds to the average fair odds of an outcome,
computing the Z-score and modified Z-score of the odds of an outcome, and comparing odds to the fair odds of
Pinnacle sportsbook (a known "sharp" sportsbook). Profitable bets are then saved into a master .csv file.

Author: Andrew Smith
"""

from .file_management import BetFileManager
from .betting_strategies import (
    analyze_average_edge_bets,
    analyze_modified_zscore_outliers,
    analyze_pinnacle_edge_bets,
    analyze_zscore_outliers,
    find_random_bets,
)
from .summary_creation import (
    create_random_summary,
    create_zscore_summary,
    create_average_edge_summary,
    create_pinnacle_edge_summary,
    create_modified_zscore_summary,
)
from .data_processing import (
    process_odds_data,
    process_target_odds_data,
    calculate_vigfree_probabilities,
)
import pandas as pd
from codebase.fetch_odds.fetch_odds import fetch_odds
from dataclasses import dataclass


@dataclass
class BettingStrategy:
    name: str
    standard_summary_file: str
    standard_full_file: str
    nc_summary_file: str
    nc_full_file: str
    score_column: str
    summary_func: callable
    analysis_func: callable


def run_betting_strategy(
    strategy: BettingStrategy,
    standard_df: pd.DataFrame,
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
        # Run the standard analysis
        standard_analysis_result = strategy.analysis_func(standard_df)
        standard_summary = strategy.summary_func(standard_analysis_result)

        if standard_summary.empty:
            print(f"No standard profitable bets found for {strategy.name}")

        # Save summary (best bets only) and get the filtered data back)
        standard_filtered_summary = file_manager.save_best_bets_only(
            standard_summary, strategy.standard_summary_file, strategy.score_column
        )

        # Save full data using the same filtered summary
        file_manager.save_full_betting_data(
            standard_analysis_result, standard_filtered_summary, strategy.standard_full_file
        )

    except Exception as e:
        print(f"Error running standard {strategy.name}: {e}")

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
            standard_summary_file="master_avg_bets.csv",
            standard_full_file="master_avg_full.csv",
            nc_summary_file="master_nc_avg_bets.csv",
            nc_full_file="master_nc_avg_full.csv",
            score_column="Avg Edge Pct",
            summary_func=create_average_edge_summary,
            analysis_func=analyze_average_edge_bets,
        ),
        BettingStrategy(
            name="Z-Score Outliers",
            standard_summary_file="master_zscore_bets.csv",
            standard_full_file="master_zscore_full.csv",
            nc_summary_file="master_nc_zscore_bets.csv",
            nc_full_file="master_nc_zscore_full.csv",
            score_column="Z Score",
            summary_func=create_zscore_summary,
            analysis_func=analyze_zscore_outliers,
        ),
        BettingStrategy(
            name="Modified Z-Score Outliers",
            standard_summary_file="master_mod_zscore_bets.csv",
            standard_full_file="master_mod_zscore_full.csv",
            nc_summary_file="master_nc_mod_zscore_bets.csv",
            nc_full_file="master_nc_mod_zscore_full.csv",
            score_column="Modified Z Score",
            summary_func=create_modified_zscore_summary,
            analysis_func=analyze_modified_zscore_outliers,
        ),
        BettingStrategy(
            name="Pinnacle Edge",
            standard_summary_file="master_pin_bets.csv",
            standard_full_file="master_pin_full.csv",
            nc_summary_file="master_nc_pin_bets.csv",
            nc_full_file="master_nc_pin_full.csv",
            score_column="Pin Edge Pct",
            summary_func=create_pinnacle_edge_summary,
            analysis_func=analyze_pinnacle_edge_bets,
        ),
        BettingStrategy(
            name="Random Bets",
            standard_summary_file="master_random_bets.csv",
            standard_full_file="master_random_full.csv",
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

        standard_processed_odds = process_odds_data(raw_odds)
        nc_processed_odds = process_target_odds_data(raw_odds)

        if standard_processed_odds.empty and nc_processed_odds.empty:
            print("No data passed cleaning requirements")
            return

        standard_df = calculate_vigfree_probabilities(standard_processed_odds)
        nc_df = calculate_vigfree_probabilities(nc_processed_odds)

        # Step 2: Run each betting strategy
        for strategy in strategies:
            run_betting_strategy(strategy, standard_df, nc_df, file_manager)

        print("\nBetting analysis pipeline completed successfully")

    except Exception as e:
        print(f"Betting pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

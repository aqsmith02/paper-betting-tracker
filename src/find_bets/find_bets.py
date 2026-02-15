"""
find_bets.py

Main pipeline for finding and saving betting opportunities. Starts by fetching odds data,
processing it, calculating vig-free probabilities, and then applying various betting strategies.
Profitable bets are stored, then appended to existing records.

Author: Andrew Smith
Date: July 2025
"""

from dataclasses import dataclass

import pandas as pd

from src.fetch_odds.fetch_odds import fetch_odds
from src.find_bets.betting_strategies import (
    find_average_bets,
    find_modified_zscore_bets,
    find_random_bets,
)
from src.find_bets.data_processing import (
    process_target_odds_data,
)
from src.find_bets.file_management import save_betting_data
from src.find_bets.summary_creation import (
    create_average_summary_full,
    create_average_summary_minimal,
    create_modified_zscore_summary_full,
    create_modified_zscore_summary_minimal,
    create_random_summary_full,
    create_random_summary_minimal,
)
from src.find_bets.vigfree_probabilities import (
    calculate_vigfree_probabilities,
)


@dataclass
class BettingStrategy:
    name: str
    minimal_file_path: str
    full_file_path: str
    score_column: str
    analysis_func: callable
    minimal_summary_func: callable
    full_summary_func: callable


strategies = [
    BettingStrategy(
        name="Average",
        minimal_file_path="data/nc_avg_minimal.csv",
        full_file_path="data/nc_avg_full.csv",
        score_column="Expected Value",
        analysis_func=find_average_bets,
        minimal_summary_func=create_average_summary_minimal,
        full_summary_func=create_average_summary_full,
    ),
    BettingStrategy(
        name="Modified Z-Score",
        minimal_file_path="data/nc_mod_zscore_minimal.csv",
        full_file_path="data/nc_mod_zscore_full.csv",
        score_column="Modified Z-Score",
        analysis_func=find_modified_zscore_bets,
        minimal_summary_func=create_modified_zscore_summary_minimal,
        full_summary_func=create_modified_zscore_summary_full,
    ),
    BettingStrategy(
        name="Random",
        minimal_file_path="data/nc_random_minimal.csv",
        full_file_path="data/nc_random_full.csv",
        score_column="Best Odds",
        analysis_func=find_random_bets,
        minimal_summary_func=create_random_summary_minimal,
        full_summary_func=create_random_summary_full,
    ),
]


def main():
    """
    Main betting analysis pipeline.

    Args:
        None

    Returns:
        None
    """
    # Run pipeline
    try:
        print("----------------------------------------------------")
        print("Starting betting analysis pipeline")

        # Fetch data
        raw_odds = fetch_odds()
        if raw_odds.empty:
            print("No odds data available")
            return

        # Process data
        processed_odds = process_target_odds_data(raw_odds, best_odds_bms="Kalshi")

        if processed_odds.empty:
            print("No data passed cleaning requirements")
            return

        # Calculate vig-free probabilities
        vf = calculate_vigfree_probabilities(processed_odds)

        for strategy in strategies:
            # Load existing data, if file not found, initialize empty DataFrame
            try:
                minimal_existing = pd.read_csv(strategy.minimal_file_path)
                full_existing = pd.read_csv(strategy.full_file_path)
            except FileNotFoundError:
                minimal_existing = pd.DataFrame()
                full_existing = pd.DataFrame()

            # Analyze and summarize
            analyzed = strategy.analysis_func(vf)
            minimal_summary = strategy.minimal_summary_func(analyzed)
            full_summary = strategy.full_summary_func(analyzed)

            # Save updated data
            save_betting_data(
                existing_df=minimal_existing,
                new_df=minimal_summary,
                filename=strategy.minimal_file_path,
                score_column=strategy.score_column,
                print_bets=True,
            )
            save_betting_data(
                existing_df=full_existing,
                new_df=full_summary,
                filename=strategy.full_file_path,
                score_column=strategy.score_column,
                print_bets=False,
            )

        print("Completed betting analysis pipeline")
        print("----------------------------------------------------")

    except Exception as e:
        print(f"Betting pipeline failed with error: {e}")
        print("----------------------------------------------------")
        raise


if __name__ == "__main__":
    main()

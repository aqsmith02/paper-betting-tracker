"""
file_management.py

Logic for appending bet summaries to existing bet .csv files.

Author: Andrew Smith
"""

from .betting_configs import DATE_FORMAT, TIMESTAMP_FORMAT
from codebase.constants import DATA_DIR
from typing import Any, Optional, List, Dict
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from pathlib import Path


def _start_date_from_timestamp(timestamp: Any) -> str:
    """
    Convert a timestamp to YYYY-MM-DD format.

    Args:
        timestamp (Any): Timestamp value to convert (can be string, datetime, etc.).

    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    return pd.to_datetime(timestamp).strftime(DATE_FORMAT)


class BetFileManager:
    """
    Manages CSV file operations for betting data.
    """

    # Central lookup table for both strategy columns
    STRATEGY_INFO: Dict[str, Dict[str, List[str]]] = {
        "master_avg": {
            "strategy": ["Fair Odds Avg", "Avg Edge Pct"],
        },
        "master_mod_zscore": {
            "strategy": ["Modified Z Score", "Avg Edge Pct"],
        },
        "master_pin": {
            "strategy": ["Pinnacle Fair Odds", "Pin Edge Pct"],
        },
        "master_zscore": {
            "strategy": ["Z Score", "Avg Edge Pct"],
        },
        "master_random": {
            "strategy": ["Random Bet Odds"],
        },
        "master_nc_avg": {
            "strategy": ["Avg Edge Pct", "Fair Odds Avg"],
        },
        "master_nc_mod_zscore": {
            "strategy": ["Modified Z Score", "Avg Edge Pct"],
        },
        "master_nc_pin": {
            "strategy": ["Pinnacle Fair Odds", "Pin Edge Pct"],
        },
        "master_nc_zscore": {
            "strategy": ["Z Score", "Avg Edge Pct"],
        },
        "master_nc_random": {
            "strategy": ["Random Bet Odds"],
        },
    }

    def __init__(self, data_directory: Path = DATA_DIR):
        """
        Initialize BetFileManager with specified data directory.

        Args:
            data_directory (Path): Directory path for storing betting data files.

        Returns:
            None
        """
        self.data_dir = Path(data_directory)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_columns(
        self, filename: str, col_type: str = "strategy"
    ) -> Optional[List[str]]:
        """
        Generalized column lookup for strategy or scoring columns.

        Args:
            filename (str): Name of the strategy file.
            col_type (str): Either 'strategy' or 'scoring'.

        Returns:
            Optional[List[str]]: Matching column names, or None if not found.
        """
        name = Path(filename).stem  # e.g., "master_avg_full"
        # strip known suffixes
        for suffix in ("_full", "_bets"):
            if name.endswith(suffix):
                name = name.replace(suffix, "")
                break

        return self.STRATEGY_INFO.get(name, {}).get(col_type)

    def _append_unique_bets(self, new_data: pd.DataFrame, filename: str) -> None:
        """
        Append new betting data, avoiding duplicates based on (Match, Date) key.

        Args:
            new_data (pd.DataFrame): New betting data to append.
            filename (str): Name of the CSV file to update.

        Returns:
            None
        """
        # Handle empty data
        if new_data.empty:
            print(f"No data to append to {filename}")
            return

        new_data = new_data.copy()
        # Append Scrape Time column for bets
        new_data["Scrape Time"] = datetime.now(ZoneInfo("America/New_York")).strftime(
            TIMESTAMP_FORMAT
        )

        # Check if data file exists, if not, create it
        full_path = self.data_dir / filename
        if not full_path.exists():
            new_data.to_csv(full_path, index=False)
            print(f"Created {filename} with {len(new_data)} rows")
            return

        # Align column schemas
        existing_data = pd.read_csv(full_path)
        all_columns = self._align_column_schemas(existing_data, new_data, filename)
        existing_data = existing_data.reindex(columns=all_columns, fill_value=np.nan)
        new_data = new_data.reindex(columns=all_columns, fill_value=np.nan)

        # Create existing keys to find non-duplicate bets
        existing_keys = {
            (row["Match"], _start_date_from_timestamp(row["Start Time"]))
            for _, row in existing_data.iterrows()
        }
        is_new_row = new_data.apply(
            lambda row: (
                row["Match"],
                _start_date_from_timestamp(row["Start Time"]),
            )
            not in existing_keys,
            axis=1,
        )
        new_rows = new_data[is_new_row]

        # Handle if only duplicates are found
        if new_rows.empty:
            print(f"No new rows to add to {filename} - all were duplicates")
            return

        # Append the new bets and save
        combined_data = pd.concat([existing_data, new_rows], ignore_index=True)
        combined_data.to_csv(full_path, index=False)
        print(f"Added {len(new_rows)} new rows to {filename}")

    def _align_column_schemas(
        self, existing_df: pd.DataFrame, new_df: pd.DataFrame, filename: str
    ) -> List[str]:
        """
        Create unified column schema for existing and new data.

        Args:
            existing_df (pd.DataFrame): Existing data from file.
            new_df (pd.DataFrame): New data to be added.
            filename (str): Name of file for strategy-specific column ordering.

        Returns:
            List[str]: Unified list of column names in proper order.
        """
        all_columns = list(existing_df.columns)

        for column in new_df.columns:
            if column not in all_columns:
                all_columns.append(column)

        strategy_columns = self._get_columns(filename, "strategy") or []
        end_columns = strategy_columns + [
            "Best Odds",
            "Best Bookmaker",
            "Result",
            "Outcomes",
            "Event ID",
            "Scrape Time",
        ]

        reordered_columns = [col for col in all_columns if col not in end_columns]
        reordered_columns.extend([col for col in end_columns if col in all_columns])

        return reordered_columns

    def save_best_bets_only(
        self, summary_df: pd.DataFrame, filename: str, score_column: str
    ) -> pd.DataFrame:
        """
        Save only the best bet per match (highest scoring bet).

        Args:
            summary_df (pd.DataFrame): Summary DataFrame containing all potential bets.
            filename (str): Path to save the filtered best bets.

        Returns:
            pd.DataFrame: Filtered DataFrame containing only the best bet per match. Returned so it can
                be used in save_full_betting_data().
        """
        if summary_df.empty:
            return pd.DataFrame()

        best_bets = summary_df.sort_values(
            score_column, ascending=False
        ).drop_duplicates(subset=["Match", "Start Time"], keep="first")

        self._append_unique_bets(best_bets, filename)

        return best_bets

    def save_full_betting_data(
        self, source_df: pd.DataFrame, filtered_summary_df: pd.DataFrame, filename: str
    ) -> None:
        """
        Save complete betting data for bets identified in filtered summary.

        Args:
            source_df (pd.DataFrame): Full DataFrame containing all betting analysis data.
            filtered_summary_df (pd.DataFrame): Filtered summary containing only best bets.
            filename (str): Path to save the full betting data.

        Returns:
            None
        """
        if filtered_summary_df.empty:
            return

        # Merge full data file on the key columns from filtered_summary_df.
        key_columns = ["Match", "Team", "Start Time"]
        merged_data = pd.merge(
            filtered_summary_df[key_columns], source_df, on=key_columns, how="left"
        )

        # Keep all columns except Vigfree probabilities.
        vigfree_columns = [
            col for col in merged_data.columns if col.startswith("Vigfree ")
        ]
        output_data = merged_data.drop(columns=vigfree_columns)

        self._append_unique_bets(output_data, filename)

from betting_configs import DATE_FORMAT, DATA_DIR, TIMESTAMP_FORMAT
import os
from typing import Any, Optional, List
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo


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
    
    Args:
        data_directory (str): Directory path for storing betting data files.
    """
    
    def __init__(self, data_directory: str = DATA_DIR):
        """
        Initialize BetFileManager with specified data directory.
        
        Args:
            data_directory (str): Directory path for storing betting data files.
            
        Returns:
            None
        """
        self.data_dir = data_directory
        os.makedirs(data_directory, exist_ok=True)
    
    def get_strategy_columns(self, filename: str) -> Optional[List[str]]:
        """
        Get unique columns for different betting strategies.
        
        Args:
            filename (str): Name of the strategy file.
            
        Returns:
            Optional[List[str]]: List of strategy-specific column names, None if not found.
        """
        strategy_columns = {
            "master_avg_full.csv": ["Avg Edge Pct", "Fair Odds Avg"],
            "master_mod_zscore_full.csv": ["Modified Z Score"],
            "master_pin_full.csv": ["Pinnacle Fair Odds", "Pin Edge Pct"],
            "master_zscore_full.csv": ["Z Score"],
            "master_random_full.csv": ["Random Bet Odds"]
        }
        return strategy_columns.get(filename)
    
    def append_unique_bets(self, new_data: pd.DataFrame, filepath: str) -> None:
        """
        Append new betting data, avoiding duplicates based on (Match, Date) key.
        
        Args:
            new_data (pd.DataFrame): New betting data to append.
            filepath (str): Path to the CSV file relative to data directory.
            
        Returns:
            None
        """
        if new_data.empty:
            print(f"No data to append to {filepath}")
            return
        
        new_data = new_data.copy()
        new_data["Scrape Time"] = datetime.now(ZoneInfo("America/New_York")).strftime(TIMESTAMP_FORMAT)
        
        full_path = os.path.join(self.data_dir, filepath)
        
        # Create new file if doesn't exist
        if not os.path.exists(full_path):
            new_data.to_csv(full_path, index=False)
            print(f"Created {filepath} with {len(new_data)} rows")
            return
        
        # Load existing data and merge schemas
        existing_data = pd.read_csv(full_path)
        
        # Align column schemas
        all_columns = self._align_column_schemas(existing_data, new_data, filepath)
        existing_data = existing_data.reindex(columns=all_columns, fill_value=np.nan)
        new_data = new_data.reindex(columns=all_columns, fill_value=np.nan)
        
        # Fill Result column appropriately
        existing_data["Result"] = existing_data["Result"].fillna("Not Found") 
        new_data["Result"] = new_data["Result"].fillna("Not Found")
        
        # Find truly new rows (not duplicates)
        existing_keys = {
            (row["Match"], _start_date_from_timestamp(row["Start Time"]))
            for _, row in existing_data.iterrows()
        }
        
        is_new_row = new_data.apply(
            lambda row: (row["Match"], _start_date_from_timestamp(row["Start Time"])) not in existing_keys,
            axis=1
        )
        
        new_rows = new_data[is_new_row]
        
        if new_rows.empty:
            print(f"No new rows to add to {filepath} - all were duplicates")
            return
        
        # Combine and save
        combined_data = pd.concat([existing_data, new_rows], ignore_index=True)
        combined_data.to_csv(full_path, index=False)
        print(f"Added {len(new_rows)} new rows to {filepath}")
    
    def _align_column_schemas(self, existing_df: pd.DataFrame, new_df: pd.DataFrame, 
                            filename: str) -> List[str]:
        """
        Create unified column schema for existing and new data.
        
        Args:
            existing_df (pd.DataFrame): Existing data from file.
            new_df (pd.DataFrame): New data to be added.
            filename (str): Name of file for strategy-specific column ordering.
            
        Returns:
            List[str]: Unified list of column names in proper order.
        """
        # Start with existing columns to preserve order
        all_columns = list(existing_df.columns)
        
        # Add any new columns from new data
        for column in new_df.columns:
            if column not in all_columns:
                all_columns.append(column)
        
        # Move strategy-specific columns to end for better organization
        strategy_columns = self.get_strategy_columns(filename) or []
        end_columns = strategy_columns + ["Best Odds", "Best Bookmaker", "Result", "Scrape Time"]
        
        # Reorder: base columns first, then end columns
        reordered_columns = [col for col in all_columns if col not in end_columns]
        reordered_columns.extend([col for col in end_columns if col in all_columns])
        
        return reordered_columns
    
    def save_best_bets_only(self, summary_df: pd.DataFrame, filepath: str) -> pd.DataFrame:
        """
        Save only the best bet per match (highest scoring bet).
        
        Args:
            summary_df (pd.DataFrame): Summary DataFrame containing all potential bets.
            filepath (str): Path to save the filtered best bets.
            
        Returns:
            pd.DataFrame: Filtered DataFrame containing only the best bet per match.
        """
        if summary_df.empty:
            return pd.DataFrame()
        
        # Find the score column for this strategy
        score_columns = ["Avg Edge Pct", "Z Score", "Modified Z Score", "Pin Edge Pct", "Random Bet Odds"]
        score_column = None
        
        for column in score_columns:
            if column in summary_df.columns:
                score_column = column
                break
        
        if not score_column:
            raise ValueError("No recognizable score column found")
        
        # Keep only the best bet per match
        best_bets = (
            summary_df.sort_values(score_column, ascending=False)
                    .drop_duplicates(subset=["Match", "Start Time"], keep="first")
        )
        
        self.append_unique_bets(best_bets, filepath)
        return best_bets  # Return the filtered data for use in save_full_betting_data

    def save_full_betting_data(self, source_df: pd.DataFrame, filtered_summary_df: pd.DataFrame, 
                            filepath: str) -> None:
        """
        Save complete betting data for bets identified in filtered summary.
        
        Args:
            source_df (pd.DataFrame): Full DataFrame containing all betting analysis data.
            filtered_summary_df (pd.DataFrame): Filtered summary containing only best bets.
            filepath (str): Path to save the full betting data.
            
        Returns:
            None
        """
        if filtered_summary_df.empty:
            return
        
        # Match FILTERED summary bets with full data
        key_columns = ["Match", "Team", "Start Time"]
        merged_data = pd.merge(filtered_summary_df[key_columns], source_df, on=key_columns, how="left")
        
        # Remove vig-free columns (not needed in final output)
        vigfree_columns = [col for col in merged_data.columns if col.startswith("Vigfree ")]
        output_data = merged_data.drop(columns=vigfree_columns)
        
        self.append_unique_bets(output_data, filepath)


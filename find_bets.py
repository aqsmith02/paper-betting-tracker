"""
find_bets.py

The file fetches odds using a separate file called fetch_odds.py, then identifies profitable bets from them
using four different strategies. The strategies are comparing odds to the average fair odds of an outcome,
computing the Z-score and modified Z-score of the odds of an outcome, and comparing odds to the fair odds of 
Pinnacle sportsbook (a known "sharp" sportsbook). Profitable bets are then saved into a master .csv file.

Author: Andrew Smith
Date: July 2025
"""
# ---------------------------------------- Imports ---------------------------------------- #
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from fetch_odds import fetch_odds, organize


# ---------------------------------------- Configuration ---------------------------------------- #
# Betting thresholds
EDGE_THRESHOLD = 0.05
Z_SCORE_THRESHOLD = 2.0
MAX_Z_SCORE = 6.0
MIN_BOOKMAKERS = 5
MAX_ODDS = 50
MAX_MISSING_VIGFREE_ODDS = 2

# File paths
DATA_DIR = "data"
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Betting strategy definitions
@dataclass
class BettingStrategy:
    name: str
    summary_file: str
    full_file: str
    score_column: str
    summary_func: callable
    analysis_func: callable


# ---------------------------------------- Utility Functions ---------------------------------------- #
def start_date_from_timestamp(timestamp: Any) -> str:
    """
    Convert a timestamp to YYYY-MM-DD format.
    
    Args:
        timestamp (Any): Timestamp value to convert (can be string, datetime, etc.).
        
    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    return pd.to_datetime(timestamp).strftime(DATE_FORMAT)


def find_bookmaker_columns(df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> List[str]:
    """
    Find columns that contain bookmaker odds (numeric columns, excluding metadata).
    
    Args:
        df (pd.DataFrame): DataFrame to search for bookmaker columns.
        exclude_columns (Optional[List[str]]): Additional columns to exclude from search.
        
    Returns:
        List[str]: List of column names that contain bookmaker odds.
    """
    excluded = {"Best Odds", "Start Time", "Last Update"}
    if exclude_columns:
        excluded.update(exclude_columns)
    
    return [col for col in df.select_dtypes(include=["float", "int"]).columns 
            if col not in excluded]


def safe_float_conversion(value: Any) -> float:
    """
    Convert value to float, returning -inf if conversion fails.
    
    Args:
        value (Any): Value to convert to float.
        
    Returns:
        float: Converted float value or -inf if conversion fails.
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return -np.inf


def validate_dataframe(df: pd.DataFrame, required_columns: List[str], name: str = "DataFrame") -> None:
    """
    Validate DataFrame meets basic requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_columns (List[str]): List of columns that must be present.
        name (str): Name of DataFrame for error messages.
        
    Returns:
        None: Raises ValueError if validation fails.
    """
    if df.empty:
        raise ValueError(f"{name} cannot be empty")
    
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"{name} missing required columns: {missing_columns}")


# ---------------------------------------- Data Cleaning ---------------------------------------- #
def clean_odds_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace odds equal to 1.0 with NaN (invalid odds).
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        
    Returns:
        pd.DataFrame: DataFrame with invalid odds (1.0) replaced with NaN.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)
    df[bookmaker_columns] = df[bookmaker_columns].where(df[bookmaker_columns] != 1, np.nan)
    return df


def prettify_column_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names to Title Case and replace underscores with spaces.
    
    Args:
        df (pd.DataFrame): DataFrame with columns to prettify.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned column names.
    """
    column_mapping = {col: col.replace("_", " ").title() for col in df.columns}
    return df.rename(columns=column_mapping)


def validate_betting_requirements(row: pd.Series, bookmaker_columns: List[str]) -> bool:
    """
    Check if row meets minimum requirements for betting analysis.
    
    Args:
        row (pd.Series): Single row from DataFrame to validate.
        bookmaker_columns (List[str]): List of bookmaker column names.
        
    Returns:
        bool: True if row meets betting requirements, False otherwise.
    """
    # Count valid bookmaker odds
    valid_odds_count = sum(
        1 for bm in bookmaker_columns
        if pd.notna(row[bm]) and isinstance(safe_float_conversion(row[bm]), float) and row[bm] > 0
    )
    
    # Check requirements
    return (valid_odds_count >= MIN_BOOKMAKERS and 
            row.get("Best Odds", 0) <= MAX_ODDS)


def filter_valid_betting_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows that don't meet betting analysis requirements.
    
    Args:
        df (pd.DataFrame): DataFrame to filter.
        
    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid betting rows.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)
    
    valid_mask = df.apply(lambda row: validate_betting_requirements(row, bookmaker_columns), axis=1)
    filtered_df = df[valid_mask]
    
    print(f"Filtered betting data: {len(df)} → {len(filtered_df)} rows")
    return filtered_df


def clean_betting_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete data cleaning pipeline for betting analysis.
    
    Args:
        df (pd.DataFrame): Raw betting data to clean.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame ready for betting analysis.
    """
    validate_dataframe(df, ["best odds"], "Input DataFrame")
    
    df = clean_odds_data(df)
    df = prettify_column_headers(df) 
    df = filter_valid_betting_rows(df)
    return df


# ---------------------------------------- Vig-Free Probability Calculation ---------------------------------------- #
def calculate_vigfree_probabilities(df: pd.DataFrame, min_outcomes: Optional[int] = None) -> pd.DataFrame:
    """
    Add vig-free implied probability columns for each bookmaker.
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        min_outcomes (Optional[int]): Minimum number of outcomes required per match. 
                                    If None, uses all outcomes in each match.
        
    Returns:
        pd.DataFrame: DataFrame with additional vig-free probability columns for each bookmaker.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)
    
    # Add vig-free columns for each bookmaker
    for bookmaker in bookmaker_columns:
        vigfree_column = f"Vigfree {bookmaker}"
        df[vigfree_column] = np.nan
        
        # Process each match separately
        for match_name, match_group in df.groupby("Match", sort=False):
            required_outcomes = len(match_group) if min_outcomes is None else min_outcomes
            
            # Get valid odds for this bookmaker in this match
            valid_odds = match_group[bookmaker].where(match_group[bookmaker] > 0).dropna()
            
            if len(valid_odds) < required_outcomes:
                continue
            
            # Calculate vig-free probabilities
            implied_probs = 1 / valid_odds
            normalized_probs = implied_probs / implied_probs.sum()
            
            # Update DataFrame with vig-free probabilities
            df.loc[valid_odds.index, vigfree_column] = normalized_probs.values
    
    return df


# ---------------------------------------- Betting Strategy Analysis ---------------------------------------- #
def count_missing_vigfree_odds(bookmaker_columns: List[str], row: pd.Series, max_missing: int) -> bool:
    """
    Check if row has too many bookmakers with odds but no vig-free probability.
    
    Args:
        bookmaker_columns (List[str]): List of bookmaker column names.
        row (pd.Series): Single row to check.
        max_missing (int): Maximum number of missing vig-free odds allowed.
        
    Returns:
        bool: True if missing vig-free count is within limit, False otherwise.
    """
    missing_vigfree_count = 0
    
    for bookmaker in bookmaker_columns:
        if pd.notna(row[bookmaker]):  # Has odds
            vigfree_col = f"Vigfree {bookmaker}"
            if vigfree_col in row and pd.isna(row[vigfree_col]):  # Missing vig-free odds
                missing_vigfree_count += 1
    
    return missing_vigfree_count <= max_missing


def analyze_average_edge_bets(df: pd.DataFrame, edge_threshold: float = EDGE_THRESHOLD) -> pd.DataFrame:
    """
    Find bets where best odds exceed average fair odds by threshold percentage.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free probability data.
        edge_threshold (float): Minimum edge percentage required for profitable bet.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for fair odds average and edge percentage.
    """
    df = df.copy()
    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = find_bookmaker_columns(df, vigfree_columns)
    
    edge_percentages = []
    fair_odds_averages = []
    
    for _, row in df.iterrows():
        # Check if row has sufficient vig-free data
        if not count_missing_vigfree_odds(bookmaker_columns, row, MAX_MISSING_VIGFREE_ODDS):
            edge_percentages.append(None)
            fair_odds_averages.append(None)
            continue
        
        # Collect vig-free probabilities
        valid_probabilities = [row[col] for col in vigfree_columns if pd.notnull(row[col])]
        if not valid_probabilities:
            edge_percentages.append(None)
            fair_odds_averages.append(None)
            continue
        
        # Calculate average fair odds
        average_probability = np.mean(valid_probabilities)
        fair_odds = 1 / average_probability
        fair_odds_averages.append(round(fair_odds, 3))
        
        # Calculate edge percentage
        best_odds = row["Best Odds"]
        edge = (best_odds / fair_odds) - 1
        
        if edge > edge_threshold:
            edge_percentages.append(round(edge * 100, 2))
        else:
            edge_percentages.append(None)
    
    df["Fair Odds Avg"] = fair_odds_averages
    df["Avg Edge Pct"] = edge_percentages
    return df


def analyze_zscore_outliers(df: pd.DataFrame, z_threshold: float = Z_SCORE_THRESHOLD) -> pd.DataFrame:
    """
    Find bets where best odds are statistical outliers using Z-score.
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        z_threshold (float): Minimum Z-score required to identify outlier.
        
    Returns:
        pd.DataFrame: DataFrame with additional Z-score column for outlier bets.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)
    z_scores = []
    
    for _, row in df.iterrows():
        # Collect all valid odds for this outcome
        valid_odds = [row[col] for col in bookmaker_columns if pd.notnull(row[col])]
        if not valid_odds:
            z_scores.append(None)
            continue
        
        best_odds = row["Best Odds"]
        
        # Calculate Z-score
        mean_odds = np.mean(valid_odds)
        std_odds = np.std(valid_odds, ddof=1)
        
        if std_odds == 0:  # Avoid division by zero
            z_scores.append(None)
            continue
        
        z_score = max(0, best_odds - mean_odds) / std_odds
        
        # Only include if within reasonable bounds
        if z_threshold < z_score < MAX_Z_SCORE:
            z_scores.append(round(z_score, 2))
        else:
            z_scores.append(None)
    
    df["Z Score"] = z_scores
    return df


def analyze_modified_zscore_outliers(df: pd.DataFrame, z_threshold: float = Z_SCORE_THRESHOLD) -> pd.DataFrame:
    """
    Find bets where best odds are outliers using Modified Z-score (more robust to outliers).
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        z_threshold (float): Minimum Modified Z-score required to identify outlier.
        
    Returns:
        pd.DataFrame: DataFrame with additional Modified Z-score column for outlier bets.
    """
    df = df.copy()
    bookmaker_columns = find_bookmaker_columns(df)
    modified_z_scores = []
    
    for _, row in df.iterrows():
        # Collect all valid odds
        valid_odds = [row[col] for col in bookmaker_columns if pd.notnull(row[col])]
        if not valid_odds:
            modified_z_scores.append(None)
            continue
        
        best_odds = row["Best Odds"]
        
        # Calculate Modified Z-score using median and MAD
        median_odds = np.median(valid_odds)
        mad = np.median(np.abs(valid_odds - median_odds))
        
        if mad == 0:  # Avoid division by zero
            modified_z_scores.append(None)
            continue
        
        modified_z = 0.6745 * max(0, best_odds - median_odds) / mad
        
        # Only include if within reasonable bounds
        if z_threshold < modified_z < MAX_Z_SCORE:
            modified_z_scores.append(round(modified_z, 2))
        else:
            modified_z_scores.append(None)
    
    df["Modified Z Score"] = modified_z_scores
    return df


def analyze_pinnacle_edge_bets(df: pd.DataFrame, pinnacle_column: str = "Pinnacle", 
                              edge_threshold: float = EDGE_THRESHOLD) -> pd.DataFrame:
    """
    Find bets where best odds exceed Pinnacle's fair odds by threshold percentage.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free probability data including Pinnacle.
        pinnacle_column (str): Name of the Pinnacle bookmaker column.
        edge_threshold (float): Minimum edge percentage required for profitable bet.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for Pinnacle fair odds and edge percentage.
    """
    df = df.copy()
    vigfree_pinnacle = f"Vigfree {pinnacle_column}"
    
    if vigfree_pinnacle not in df.columns:
        raise ValueError(f"Missing {vigfree_pinnacle} column - run vig-free calculation first")
    
    pinnacle_fair_odds = []
    edge_percentages = []
    
    for _, row in df.iterrows():
        pinnacle_probability = row[vigfree_pinnacle]
        
        if pd.isna(pinnacle_probability) or pinnacle_probability <= 0:
            pinnacle_fair_odds.append(None)
            edge_percentages.append(None)
            continue
        
        # Calculate Pinnacle's fair odds
        fair_odds = 1 / pinnacle_probability
        pinnacle_fair_odds.append(round(fair_odds, 3))
        
        # Calculate edge vs Pinnacle
        best_odds = row["Best Odds"]
        edge = (best_odds / fair_odds) - 1
        
        if edge > edge_threshold:
            edge_percentages.append(round(edge * 100, 2))
        else:
            edge_percentages.append(None)
    
    df["Pinnacle Fair Odds"] = pinnacle_fair_odds
    df["Pin Edge Pct"] = edge_percentages
    return df


# ---------------------------------------- Summary Creation ---------------------------------------- #
def create_average_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of profitable average-edge bets.
    
    Args:
        df (pd.DataFrame): DataFrame containing average edge analysis results.
        
    Returns:
        pd.DataFrame: Summary DataFrame with only profitable average-edge bets.
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Avg Edge Pct")) or pd.isna(row.get("Fair Odds Avg")):
            continue
        
        summary_rows.append({
            "Match": row["Match"],
            "League": row["League"], 
            "Team": row["Team"],
            "Start Time": row["Start Time"],
            "Avg Edge Book": row["Best Bookmaker"],
            "Avg Edge Odds": row["Best Odds"],
            "Avg Edge Pct": row["Avg Edge Pct"],
            "Result": row.get("Result", "Not Found"),
        })
    
    return pd.DataFrame(summary_rows)


def create_zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of Z-score outlier bets.
    
    Args:
        df (pd.DataFrame): DataFrame containing Z-score analysis results.
        
    Returns:
        pd.DataFrame: Summary DataFrame with only Z-score outlier bets.
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Z Score")):
            continue
        
        summary_rows.append({
            "Match": row["Match"],
            "League": row["League"],
            "Team": row["Team"], 
            "Start Time": row["Start Time"],
            "Outlier Book": row["Best Bookmaker"],
            "Outlier Odds": row["Best Odds"],
            "Z Score": row["Z Score"],
            "Result": row.get("Result", "Not Found"),
        })
    
    return pd.DataFrame(summary_rows)


def create_modified_zscore_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of Modified Z-score outlier bets.
    
    Args:
        df (pd.DataFrame): DataFrame containing Modified Z-score analysis results.
        
    Returns:
        pd.DataFrame: Summary DataFrame with only Modified Z-score outlier bets.
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Modified Z Score")):
            continue
        
        summary_rows.append({
            "Match": row["Match"],
            "League": row["League"],
            "Team": row["Team"],
            "Start Time": row["Start Time"],
            "Outlier Book": row["Best Bookmaker"],
            "Outlier Odds": row["Best Odds"],
            "Modified Z Score": row["Modified Z Score"],
            "Result": row.get("Result", "Not Found"),
        })
    
    return pd.DataFrame(summary_rows)


def create_pinnacle_edge_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create summary of Pinnacle edge bets.
    
    Args:
        df (pd.DataFrame): DataFrame containing Pinnacle edge analysis results.
        
    Returns:
        pd.DataFrame: Summary DataFrame with only profitable Pinnacle edge bets.
    """
    if df.empty:
        return pd.DataFrame()
    
    summary_rows = []
    for _, row in df.iterrows():
        if pd.isna(row.get("Pinnacle Fair Odds")) or pd.isna(row.get("Pin Edge Pct")):
            continue
        
        summary_rows.append({
            "Match": row["Match"],
            "League": row["League"],
            "Team": row["Team"],
            "Start Time": row["Start Time"],
            "Pinnacle Edge Book": row["Best Bookmaker"],
            "Pinnacle Edge Odds": row["Best Odds"],
            "Pin Edge Pct": row["Pin Edge Pct"],
            "Pinnacle Fair Odds": row["Pinnacle Fair Odds"],
            "Result": row.get("Result", "Not Found"),
        })
    
    return pd.DataFrame(summary_rows)


# ---------------------------------------- File Management ---------------------------------------- #
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
            "master_zscore_full.csv": ["Z Score"]
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
            (row["Match"], start_date_from_timestamp(row["Start Time"]))
            for _, row in existing_data.iterrows()
        }
        
        is_new_row = new_data.apply(
            lambda row: (row["Match"], start_date_from_timestamp(row["Start Time"])) not in existing_keys,
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
        score_columns = ["Avg Edge Pct", "Z Score", "Modified Z Score", "Pin Edge Pct"]
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


# ---------------------------------------- Strategy Execution ---------------------------------------- #
def run_betting_strategy(strategy: BettingStrategy, cleaned_data: pd.DataFrame, 
                        vigfree_data: pd.DataFrame, file_manager: BetFileManager) -> None:
    """
    Execute a complete betting strategy analysis and save results.
    
    Args:
        strategy (BettingStrategy): Betting strategy configuration object.
        cleaned_data (pd.DataFrame): Cleaned betting data without vig-free calculations.
        vigfree_data (pd.DataFrame): Betting data with vig-free probability calculations.
        file_manager (BetFileManager): File manager for saving results.
        
    Returns:
        None
    """
    print(f"\nRunning {strategy.name} analysis...")
    
    try:
        # Run the analysis
        analysis_result = strategy.analysis_func(vigfree_data if "Pinnacle" in strategy.name or "Average" in strategy.name else cleaned_data)
        summary = strategy.summary_func(analysis_result)
        
        if summary.empty:
            print(f"No profitable bets found for {strategy.name}")
            return
        
        # Save summary (best bets only) and get the filtered data back
        filtered_summary = file_manager.save_best_bets_only(summary, strategy.summary_file)
        
        # Save full data using the SAME filtered summary (not the original summary)
        file_manager.save_full_betting_data(analysis_result, filtered_summary, strategy.full_file)
        
    except Exception as e:
        print(f"Error running {strategy.name}: {e}")


# ---------------------------------------- Main Pipeline ---------------------------------------- #
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
            summary_file="master_avg_bets.csv",
            full_file="master_avg_full.csv", 
            score_column="Avg Edge Pct",
            summary_func=create_average_edge_summary,
            analysis_func=analyze_average_edge_bets
        ),
        BettingStrategy(
            name="Z-Score Outliers",
            summary_file="master_zscore_bets.csv",
            full_file="master_zscore_full.csv",
            score_column="Z Score", 
            summary_func=create_zscore_summary,
            analysis_func=analyze_zscore_outliers
        ),
        BettingStrategy(
            name="Modified Z-Score Outliers",
            summary_file="master_mod_zscore_bets.csv",
            full_file="master_mod_zscore_full.csv",
            score_column="Modified Z Score",
            summary_func=create_modified_zscore_summary,
            analysis_func=analyze_modified_zscore_outliers
        )
    ]
    
    try:
        # Step 1: Fetch and prepare data
        raw_odds = fetch_odds()
        if raw_odds.empty:
            print("No odds data available")
            return
        
        organized_odds = organize(raw_odds)
        cleaned_data = clean_betting_data(organized_odds)
        
        if cleaned_data.empty:
            print("No data passed cleaning requirements")
            return
        
        # Step 2: Calculate vig-free probabilities (needed for some strategies)
        vigfree_data = calculate_vigfree_probabilities(cleaned_data)
        
        # Step 3: Run each betting strategy
        for strategy in strategies:
            run_betting_strategy(strategy, cleaned_data, vigfree_data, file_manager)
        
        # Step 4: Run Pinnacle strategy if Pinnacle data exists
        if "Pinnacle" in cleaned_data.columns and cleaned_data["Pinnacle"].notna().any():
            pinnacle_strategy = BettingStrategy(
                name="Pinnacle Edge",
                summary_file="master_pin_bets.csv",
                full_file="master_pin_full.csv",
                score_column="Pin Edge Pct",
                summary_func=create_pinnacle_edge_summary,
                analysis_func=analyze_pinnacle_edge_bets
            )
            run_betting_strategy(pinnacle_strategy, cleaned_data, vigfree_data, file_manager)
        else:
            print("No Pinnacle odds available - skipping Pinnacle edge analysis")
        
        print("\nBetting analysis pipeline completed successfully")
        
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
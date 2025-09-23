from typing import List
import pandas as pd
from .betting_configs import EDGE_THRESHOLD, MAX_MISSING_VIGFREE_ODDS,Z_SCORE_THRESHOLD, MAX_Z_SCORE
from .data_processing import _find_bookmaker_columns
import random


def _count_missing_vigfree_odds(row: pd.Series, bookmaker_columns: List[str], max_missing: int) -> bool:
    """
    Check if row has too many bookmakers with odds but no vig-free probability (then average of vf
    probabilities is not representative of all odds).
    
    Args:
        row (pd.Series): Single row to check.
        bookmaker_columns (List[str]): List of bookmaker column names.
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


def analyze_average_edge_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds exceed average fair odds by threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for fair odds average and edge percentage.
    """
    df = df.copy()
    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = _find_bookmaker_columns(df, vigfree_columns)
    
    edge_percentages = []
    fair_odds_averages = []
    
    for _, row in df.iterrows():
        # Check if row has sufficient vig-free data
        if not _count_missing_vigfree_odds(row, bookmaker_columns, MAX_MISSING_VIGFREE_ODDS):
            edge_percentages.append(None)
            fair_odds_averages.append(None)
            continue
        
        # Calculate average fair odds
        average_probability = row[vigfree_columns].mean()
        fair_odds = 1 / average_probability
        fair_odds_averages.append(round(fair_odds, 2))
        
        # Calculate edge percentage
        best_odds = row["Best Odds"]
        edge = (best_odds / fair_odds) - 1
        
        if edge > EDGE_THRESHOLD:
            edge_percentages.append(round(edge * 100, 2))
        else:
            edge_percentages.append(None)
    
    df["Fair Odds Avg"] = fair_odds_averages
    df["Avg Edge Pct"] = edge_percentages
    return df


def analyze_zscore_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds are statistical outliers using Z-score and average edge.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional Z-score column and average edge columns.
    """
    df = df.copy()
    df = analyze_average_edge_bets(df)
    
    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = _find_bookmaker_columns(df, vigfree_columns)
    z_scores = []
    
    for _, row in df.iterrows():
        best_odds = row["Best Odds"]
        
        # Calculate Z-score
        mean_odds = row[bookmaker_columns].mean()
        std_odds = row[bookmaker_columns].std()
        
        if std_odds == 0:  # Avoid division by zero
            z_scores.append(None)
            continue
        
        z_score = max(0, best_odds - mean_odds) / std_odds
        
        # Only include if within reasonable bounds
        if Z_SCORE_THRESHOLD < z_score < MAX_Z_SCORE:
            z_scores.append(round(z_score, 2))
        else:
            z_scores.append(None)
    
    df["Z Score"] = z_scores
    return df


def analyze_modified_zscore_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds are outliers using Modified Z-score (more robust to outliers) and
    average edge.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional Modified Z-score column and average edge columns.
    """
    df = df.copy()
    df = analyze_average_edge_bets(df)

    vigfree_columns = [col for col in df.columns if col.startswith("Vigfree ")]
    bookmaker_columns = _find_bookmaker_columns(df, vigfree_columns)
    modified_z_scores = []
    
    for _, row in df.iterrows():   
        best_odds = row["Best Odds"]
        
        # Calculate Modified Z-score using median and MAD
        median_odds = row[bookmaker_columns].median()
        mad = (row[bookmaker_columns] - median_odds).abs().median()
        
        if mad == 0:  # Avoid division by zero
            modified_z_scores.append(None)
            continue
        
        modified_z = 0.6745 * max(0, best_odds - median_odds) / mad
        
        # Only include if within reasonable bounds
        if Z_SCORE_THRESHOLD < modified_z < MAX_Z_SCORE:
            modified_z_scores.append(round(modified_z, 2))
        else:
            modified_z_scores.append(None)
    
    df["Modified Z Score"] = modified_z_scores
    return df



def analyze_pinnacle_edge_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find bets where best odds exceed Pinnacle's fair odds by threshold.
    
    Args:
        df (pd.DataFrame): DataFrame containing vig-free probability data including Pinnacle.
        pinnacle_column (str): Name of the Pinnacle bookmaker column.
        edge_threshold (float): Minimum edge percentage required for profitable bet.
        
    Returns:
        pd.DataFrame: DataFrame with additional columns for Pinnacle fair odds and edge percentage.
    """
    df = df.copy()
    vigfree_pinnacle = f"Vigfree Pinnacle"
    if vigfree_pinnacle not in df.columns:
        return df
    
    pinnacle_fair_odds = []
    edge_percentages = []
    
    for _, row in df.iterrows():
        pinnacle_probability = row[vigfree_pinnacle]
        
        if pd.isna(pinnacle_probability):
            pinnacle_fair_odds.append(None)
            edge_percentages.append(None)
            continue
        
        # Calculate Pinnacle's fair odds
        fair_odds = 1 / pinnacle_probability
        pinnacle_fair_odds.append(round(fair_odds, 2))
        
        # Calculate edge vs Pinnacle
        best_odds = row["Best Odds"]
        edge = (best_odds / fair_odds) - 1
        
        if edge > EDGE_THRESHOLD:
            edge_percentages.append(round(edge * 100, 2))
        else:
            edge_percentages.append(None)
    
    df["Pinnacle Fair Odds"] = pinnacle_fair_odds
    df["Pin Edge Pct"] = edge_percentages
    return df


def find_random_bets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Place random bets.
    
    Args:
        df (pd.DataFrame): DataFrame containing odds data.
        
    Returns:
        pd.DataFrame: DataFrame with additional column for placed random bets.
    """
    df = df.copy()
    rand_amount = random.randint(0, min(5, len(df)))
    df_sample = df.sample(n=rand_amount)
    df["Random Placed Bet"] = df.index.isin(df_sample.index).astype(int)

    return df
"""
Profit Calculator

Calculate cumulative profit over time for betting strategies using both
flat betting and Kelly criterion approaches.

This module contains pure calculation logic. For visualization, see
visualizations/profit_charts.py
"""

import pandas as pd
from src.constants import PENDING_RESULTS
from analysis.betting_utils import kelly_bet


def calculate_cumulative_profit_flat(df, odds_col):
    """
    Calculate cumulative profit over time using flat betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
    
    Returns:
        DataFrame with cumulative profit information including:
        - Start Time: datetime of bet
        - Bet_Number: sequential bet number
        - Bet_Size: size of bet (always 1.0 for flat)
        - Profit: profit/loss for this bet
        - Cumulative_Profit: running total profit
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Sort by start time
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df = df.sort_values('Start Time').reset_index(drop=True)
    
    # Calculate profit for each bet (flat $1 betting)
    df['Bet_Size'] = 1.0
    df['Profit'] = df.apply(
        lambda row: (row[odds_col] - 1) if row['Team'] == row['Result'] else -1,
        axis=1
    )
    
    # Calculate cumulative profit
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    df['Bet_Number'] = range(1, len(df) + 1)
    
    return df[['Start Time', 'Bet_Number', 'Bet_Size', 'Profit', 'Cumulative_Profit']]


def calculate_cumulative_profit_kelly(df, odds_col, fair_col, ev_col=None, zscore_col=None):
    """
    Calculate cumulative profit over time using Kelly betting.
    
    Args:
        df: DataFrame with betting data
        odds_col: Column name for odds
        fair_col: Column name for fair odds
        ev_col: Column name for expected value (optional)
        zscore_col: Column name for z-score (optional)
    
    Returns:
        DataFrame with cumulative profit information for bets where Kelly > 0:
        - Start Time: datetime of bet
        - Bet_Number: sequential bet number (only for placed bets)
        - Bet_Size: Kelly bet size as % of bankroll
        - Profit: profit/loss for this bet
        - Cumulative_Profit: running total profit
    """
    df = df[~df["Result"].isin(PENDING_RESULTS)].copy()
    
    # Sort by start time
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df = df.sort_values('Start Time').reset_index(drop=True)
    
    # Calculate Kelly bet size for each bet
    df['Bet_Size'] = df.apply(
        lambda row: kelly_bet(
            odds=row[odds_col],
            fair_odds=row[fair_col],
            ev=row.get(ev_col) if ev_col else None,
            zscore=row.get(zscore_col) if zscore_col else None,
        ),
        axis=1,
    )
    
    # Calculate profit for each bet
    df['Profit'] = df.apply(
        lambda row: row['Bet_Size'] * (row[odds_col] - 1) if row['Team'] == row['Result'] 
                    else -row['Bet_Size'],
        axis=1
    )
    
    # Calculate cumulative profit
    df['Cumulative_Profit'] = df['Profit'].cumsum()
    
    # Only count bets where Kelly says to bet
    df_bets = df[df['Bet_Size'] > 0].copy()
    df_bets['Bet_Number'] = range(1, len(df_bets) + 1)
    
    return df_bets[['Start Time', 'Bet_Number', 'Bet_Size', 'Profit', 'Cumulative_Profit']]


def calculate_profit_summary(profit_df, betting_method='flat'):
    """
    Calculate summary statistics from profit DataFrame.
    
    Args:
        profit_df: DataFrame from calculate_cumulative_profit_flat/kelly
        betting_method: 'flat' or 'kelly'
        
    Returns:
        Dictionary with summary statistics
    """
    if profit_df.empty:
        return {
            'final_profit': 0,
            'total_bets': 0,
            'total_wagered': 0,
            'roi': 0,
        }
    
    final_profit = profit_df['Cumulative_Profit'].iloc[-1]
    total_bets = len(profit_df)
    
    if betting_method == 'flat':
        total_wagered = total_bets  # $1 per bet
    else:  # kelly
        total_wagered = profit_df['Bet_Size'].sum()
    
    roi = (final_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    return {
        'final_profit': final_profit,
        'total_bets': total_bets,
        'total_wagered': total_wagered,
        'roi': roi,
    }
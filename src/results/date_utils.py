from datetime import datetime, timedelta, timezone
from typing import Any
from src.constants import DATE_FORMAT

import pandas as pd


def _start_date_from_timestamp(timestamp: Any) -> str:
    """
    Convert a timestamp to YYYY-MM-DD format.

    Args:
        timestamp (Any): Timestamp value to convert (can be string, datetime, etc.).

    Returns:
        str: Date string in YYYY-MM-DD format.
    """
    return pd.to_datetime(timestamp).strftime(DATE_FORMAT)


def _time_since_start(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    """
    Filter out games that started less than threshold hours ago.

    Args:
        df (pd.DataFrame): DataFrame containing game data with "Start Time" column.
        thresh (float): Threshold in hours - games starting less than this many hours ago are filtered out.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only games that started more than thresh hours ago.
    """
    # Handle empty DataFrame
    if df.empty:
        return df

    # Get the current time in UTC
    current_time = datetime.now(timezone.utc)

    # Convert "Start Time" column to datetime objects and make timezone-aware (UTC)
    # Use format='ISO8601' to handle both ISO format strings and other formats
    df["Start Time"] = pd.to_datetime(df["Start Time"], format="ISO8601", utc=True)

    # Create conditions for removal
    cutoff = current_time - timedelta(hours=thresh)

    # Filter out games that started less than threshold hours ago
    mask = df["Start Time"] <= cutoff
    df = df[mask]

    return df

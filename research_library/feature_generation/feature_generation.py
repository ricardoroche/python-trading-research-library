import pandas as pd
import numpy as np


def calculate_timestamp_features(df, column, hours_in_trading_day):
    """
    Calculate sine and cosine transformations of normalized microseconds since open.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column to use for calculation.

    Returns:
        pd.DataFrame: DataFrame with added timestamp feature columns.
    """
    out = pd.DataFrame()
    microseconds_in_trading_day = hours_in_trading_day * 60 * 60 * 1_000_000
    out["normalized_time"] = df[column] / microseconds_in_trading_day
    out["time_sin"] = np.sin(2 * np.pi * out["normalized_time"])
    out["time_cos"] = np.cos(2 * np.pi * out["normalized_time"])
    return pd.concat([df, out], axis=1)


def calculate_differenced_features(df, columns, prefix="diff"):
    """
    Calculate first-order differences for specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): List of columns to calculate differences for.
        prefix (str): Prefix for the new columns. Defaults to "diff".

    Returns:
        pd.DataFrame: DataFrame with added differenced columns.
    """
    out = pd.DataFrame()
    prefix = prefix if prefix.startswith("feat_") else f"feat_{prefix}"
    for column in columns:
        diff_column = f"{prefix}_{column}"
        out[diff_column] = df[column].diff()
    return pd.concat([df, out], axis=1)


def calculate_rolling_mean_features(df, column, windows, prefix=None):
    """
    Generate rolling window means and their differences for a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column for which to calculate rolling statistics.
        windows (list of int): List of window sizes (e.g., [5, 10, 30]).
        prefix (str, optional): Prefix for the new feature columns. Defaults to the column name.

    Returns:
        pd.DataFrame: DataFrame with added rolling statistics and differences.
    """
    out = pd.DataFrame()
    if prefix is None:
        prefix = column if column.startswith("feat_") else f"feat_{column}"

    for window in windows:
        # Rolling mean
        rolling_mean_col = f"{prefix}_rolling_mean_{window}"
        out[rolling_mean_col] = df[column].rolling(window=window, min_periods=1).mean()

    # Calculate difference of rolling windows
    for window_a in windows:
        for window_b in windows:
            if window_b >= window_a:
                continue
            rolling_mean_col = f"{prefix}_rolling_mean_{window_a}_minus_{window_b}"
            out[rolling_mean_col] = (
                out[f"{prefix}_rolling_mean_{window_a}"]
                - out[f"{prefix}_rolling_mean_{window_b}"]
            )

    return pd.concat([df, out], axis=1)


def calculate_rolling_sum_features(df, column, windows, prefix=None):
    """
    Generate rolling window stds and their differences for a specified column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        column (str): Column for which to calculate rolling statistics.
        windows (list of int): List of window sizes (e.g., [5, 10, 30]).
        prefix (str, optional): Prefix for the new feature columns. Defaults to the column name.

    Returns:
        pd.DataFrame: DataFrame with added rolling statistics and differences.
    """
    out = pd.DataFrame()
    if prefix is None:
        prefix = column if column.startswith("feat_") else f"feat_{column}"

    for window in windows:
        # Rolling sum
        rolling_sum_col = f"{prefix}_rolling_sum_{window}"
        out[rolling_sum_col] = df[column].rolling(window=window, min_periods=1).sum()

    # Calculate difference of rolling windows
    for window_a in windows:
        for window_b in windows:
            if window_b >= window_a:
                continue
            rolling_sum_col = f"{prefix}_rolling_sum_{window_a}_minus_{window_b}"
            out[rolling_sum_col] = (
                out[f"{prefix}_rolling_sum_{window_a}"]
                - out[f"{prefix}_rolling_sum_{window_b}"]
            )

    return pd.concat([df, out], axis=1)

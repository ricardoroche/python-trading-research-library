import logging
import pandas as pd
import numpy as np


from research_library.feature_generation.feature_generation import (
    calculate_rolling_sum_features,
    calculate_differenced_features,
    calculate_rolling_mean_features,
    calculate_timestamp_features,
)
from config import (
    PROCESSED_DATA_DIR,
    FEATURES_DATA_DIR,
    HOURS_IN_TRADING_DAY,
    WINDOWS,
    ORDER_BOOK_LEVELS,
)

logger = logging.getLogger(__name__)


def calculate_features(df):
    """
    Add predictive features to the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added features.
    """
    out = pd.DataFrame()
    out["sided_quantity"] = df["quantity"] * np.where(df["side"] == "a", -1, 1)
    # Mid-Price
    out["midprice"] = (df["ap0"] + df["bp0"]) / 2
    # VWAP
    out["vwap"] = (df["price"] * (df["bq0"] + df["aq0"])).cumsum() / (
        df["bq0"] + df["aq0"]
    ).cumsum()

    # Spread
    out["spread"] = df["ap0"] - df["bp0"]
    # Relative Spread
    out["relative_spread"] = out["spread"] / out["midprice"]

    # Depth
    out["depth"] = df[
        [f"bq{i}" for i in range(ORDER_BOOK_LEVELS)]
        + [f"aq{i}" for i in range(ORDER_BOOK_LEVELS)]
    ].sum(axis=1)
    out["bid_depth"] = df[[f"bq{i}" for i in range(ORDER_BOOK_LEVELS)]].sum(axis=1)
    out["ask_depth"] = df[[f"aq{i}" for i in range(ORDER_BOOK_LEVELS)]].sum(axis=1)

    # Imbalance
    out["top_imbalance"] = df["bq0"] - df["aq0"]
    out["imbalance"] = (out["bid_depth"] - out["ask_depth"]) / (
        out["bid_depth"] + out["ask_depth"]
    )

    out["bid_improved"] = df["bp0"] > df["bp0"].shift()
    out["ask_improved"] = df["ap0"] < df["ap0"].shift()
    out["bid_widened"] = df["bp0"] < df["bp0"].shift()
    out["ask_widened"] = df["ap0"] > df["ap0"].shift()
    return pd.concat([df, out], axis=1)


def calculate_vwap_change_target(
    df,
    price_column="price",
    quantity_column="quantity",
    timestamp_column="timestamp",
    target_column="vwap_change",
    time_bin="1s",
    shift_seconds=2,
):
    """
    Calculate VWAP or mid-price change after an interval of time as a target.

    Args:
        df (pd.DataFrame): Input DataFrame with order updates.
        price_column (str): Name of the price column.
        quantity_column (str): Name of the quantity column.
        timestamp_column (str): Name of the timestamp column.
        target_column (str): Name of the target column to create.
        time_bin (str): Time binning interval (e.g., "1s" for 1 second).
        shift_seconds (int): Seconds to look ahead for the target.

    Returns:
        pd.DataFrame: DataFrame with the calculated target column.
    """
    # Ensure the timestamp column is in datetime format
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])

    # Bin the data into 1-second intervals
    df["time_bin"] = df[timestamp_column].dt.floor(time_bin)

    # Calculate VWAP for each time bin
    vwap = df.groupby("time_bin").apply(
        lambda g: (g[price_column] * g[quantity_column]).sum()
        / g[quantity_column].sum()
    )
    vwap = vwap.rename("vwap")

    # Create a DataFrame with VWAP and calculate the change
    vwap_df = vwap.reset_index()
    vwap_df[target_column] = vwap_df["vwap"].shift(-shift_seconds) - vwap_df["vwap"]

    # Merge the target back into the original DataFrame
    df = df.merge(vwap_df[["time_bin", target_column]], on="time_bin", how="left")

    return df.drop(columns=["time_bin"])


def run_feature_generation():
    """
    Reads processed files, calculates predictive features, generates a target column,
    and saves the features for modeling.
    """
    logger.info("Starting feature generation...")
    processed_files = list(PROCESSED_DATA_DIR.glob("*.csv"))

    all_features = []
    for file in processed_files:
        logger.debug(f"Calculating features for {file.name}...")
        processed_data = pd.read_csv(file)
        features_data = (
            processed_data.pipe(
                calculate_vwap_change_target,
                price_column="price",
                quantity_column="quantity",
                timestamp_column="timestamp",
            )
            .set_index("timestamp")
            .pipe(calculate_features)
            .pipe(
                calculate_timestamp_features,
                "microseconds_since_open",
                HOURS_IN_TRADING_DAY,
            )
            .pipe(
                calculate_differenced_features,
                columns=[
                    "vwap",
                    "relative_spread",
                    "imbalance",
                    "top_imbalance",
                    "depth",
                    "ap0",
                    "bp0",
                ],
            )
        )
        for column in [
            "feat_diff_vwap",
            "feat_diff_ap0",
            "feat_diff_bp0",
            "relative_spread",
            "imbalance",
            "top_imbalance",
            "depth",
        ]:
            features_data = features_data.pipe(
                calculate_rolling_mean_features,
                column=column,
                windows=WINDOWS,
            )
        for column in [
            "crosses_spread",
            "ask_improved",
            "bid_improved",
            "ask_widened",
            "bid_widened",
        ]:
            features_data = features_data.pipe(
                calculate_rolling_sum_features,
                column=column,
                windows=WINDOWS,
            )
        features_data = features_data.reset_index()

        logger.debug(f"Shape: {features_data.shape}")

        features_data.dropna(inplace=True)  # Drop rows with NaN targets
        all_features.append(features_data)

    # Combine all features into a single DataFrame
    logger.info("Combining features...")
    combined_features = pd.concat(all_features, ignore_index=True)
    features_output_path = FEATURES_DATA_DIR / "features_with_target.csv"
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined_features.to_csv(features_output_path, index=False)
    logger.info(
        f"Feature generation completed. Features saved to {features_output_path}."
    )

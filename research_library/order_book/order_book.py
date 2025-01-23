import logging
import pandas as pd
import datetime as dt

from collections import defaultdict
from itertools import islice
from pathlib import Path

logger = logging.getLogger(__name__)


def get_top_levels(order_book: dict, side: str, levels: int = 5):
    """
    Get the top N levels of the order book for a given side.

    Args:
        order_book (dict): The current state of the order book with "a" and "b" sides.
        side (str): The side of the book to query ("a" for ask, "b" for bid).
        levels (int): Number of levels to retrieve.

    Returns:
        list: Top N levels as tuples (price, quantity).
    """
    book_side = order_book[side]
    # Sort prices ascending for "a", descending for "b"
    sorted_prices = sorted(book_side.keys(), reverse=(side == "b"))
    # Retrieve the top levels
    top_levels = [
        (price, book_side[price]) for price in sorted_prices if book_side[price] > 0
    ]
    # Pad with zeros for missing levels
    return list(islice(top_levels + [(0, 0)] * levels, levels))


def recalculate_order_book(order_record):
    order_book = {"b": defaultdict(int), "a": defaultdict(int)}
    for order_id in order_record.keys():
        side = order_record[order_id]["side"]
        price = order_record[order_id]["price"]
        quantity = order_record[order_id]["quantity"]
        order_book[side][price] = quantity
    return order_book


def process_order_book(input_path: Path, output_path: Path, levels: int = 5):
    """
    Process a single daily CSV file and save the processed output.

    Args:
        input_path (Path): Path to the input CSV file.
        output_path (Path): Path to save the processed CSV file.
        levels (int): Number of levels to retrieve.
    """
    # Read input data
    df = pd.read_csv(input_path)
    date = pd.to_datetime(input_path.stem.split("_")[-1])

    # Initialize the order book as dictionaries
    order_book = {"b": defaultdict(int), "a": defaultdict(int)}
    order_record = {}
    results = []

    # Process each row in the input file
    for _, row in df.iterrows():
        timestamp = pd.to_datetime(
            date + dt.timedelta(microseconds=int(row["timestamp"]))
        )
        order_id = row["id"]
        microseconds_since_open = row["timestamp"]
        side = row["side"]
        opp_side = "a" if side == "b" else "b"
        action = row["action"]
        price = float(row["price"])
        quantity = int(row["quantity"])
        crosses_spread = False

        assert side in {"a", "b"}
        assert action in {"a", "d", "m"}
        assert int(price) == float(price)

        # Update the order book based on the action
        if action == "a":  # Add
            assert quantity > 0
            order_record[order_id] = {
                "side": side,
                "price": price,
                "quantity": quantity,
            }
            order_book[side][price] += quantity
            if price in order_book[opp_side]:
                crosses_spread = True

        elif action == "d":  # Delete
            del order_record[order_id]
            order_book[side][price] -= quantity

        elif action == "m":  # Modify
            assert quantity > 0
            order_record[order_id] = {
                "side": side,
                "price": price,
                "quantity": quantity,
            }
            order_book = recalculate_order_book(order_record)

        # Get the top 5 levels for bids and asks
        best_bids = get_top_levels(order_book, "b", levels)
        best_asks = get_top_levels(order_book, "a", levels)

        # Construct the output row
        output_row = {
            "timestamp": timestamp,
            "microseconds_since_open": microseconds_since_open,
            "id": row["id"],
            "quantity": quantity,
            "price": price,
            "side": side,
            "action": action,
            "crosses_spread": crosses_spread,
            **{f"bp{i}": best_bids[i][0] for i in range(levels)},
            **{f"bq{i}": best_bids[i][1] for i in range(levels)},
            **{f"ap{i}": best_asks[i][0] for i in range(levels)},
            **{f"aq{i}": best_asks[i][1] for i in range(levels)},
        }
        results.append(output_row)

    # Save the results to the output file
    result_df = pd.DataFrame(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False)


def process_all_order_books(raw_data_dir: Path, processed_data_dir: Path, levels: int):
    """
    Process all daily CSV files in the raw data directory.

    Args:
        raw_data_dir (Path): Directory containing raw daily CSV files.
        processed_data_dir (Path): Directory to save processed CSV files.
    """
    raw_files = list(raw_data_dir.glob("*.csv"))
    if not raw_files:
        logger.error("No raw data files found in the directory.")
        return

    for raw_file in raw_files:
        output_file = processed_data_dir / f"{raw_file.stem}_processed.csv"
        logger.debug(f"Processing {raw_file.name}...")
        process_order_book(raw_file, output_file, levels)
        logger.debug(f"Processed file saved to {output_file.name}")

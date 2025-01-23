import pandas as pd
from collections import defaultdict
from research_library.order_book.order_book import process_order_book


# Helper function to initialize an order book
def create_empty_order_book():
    return {"b": defaultdict(int), "a": defaultdict(int)}


def test_example_scenario(tmp_path):
    # Define the input data for the scenario
    input_data = [
        {
            "timestamp": 0,
            "id": 1,
            "side": "b",
            "action": "a",
            "price": 15,
            "quantity": 3,
        },
        {
            "timestamp": 1,
            "id": 2,
            "side": "b",
            "action": "a",
            "price": 15,
            "quantity": 5,
        },
        {
            "timestamp": 2,
            "id": 1,
            "side": "b",
            "action": "m",
            "price": 20,
            "quantity": 3,
        },
        {
            "timestamp": 3,
            "id": 3,
            "side": "a",
            "action": "a",
            "price": 30,
            "quantity": 1,
        },
        {
            "timestamp": 4,
            "id": 2,
            "side": "b",
            "action": "d",
            "price": 15,
            "quantity": 5,
        },
    ]

    # Create a temporary input file
    input_file = tmp_path / "res_20240101.csv"
    pd.DataFrame(input_data).to_csv(input_file, index=False)

    # Define the expected output data
    expected_output = [
        {
            "id": 1,
            "microseconds_since_open": 0,
            "quantity": 3,
            "price": 15.0,
            "side": "b",
            "action": "a",
            "bp0": 15.0,
            "bq0": 3,
            "bp1": 0.0,
            "bq1": 0,
            "ap0": 0.0,
            "aq0": 0,
        },
        {
            "id": 2,
            "microseconds_since_open": 1,
            "quantity": 5,
            "price": 15.0,
            "side": "b",
            "action": "a",
            "bp0": 15.0,
            "bq0": 8,
            "bp1": 0.0,
            "bq1": 0,
            "ap0": 0.0,
            "aq0": 0,
        },
        {
            "id": 1,
            "microseconds_since_open": 2,
            "quantity": 3,
            "price": 20.0,
            "side": "b",
            "action": "m",
            "bp0": 20.0,
            "bq0": 3,
            "bp1": 15.0,
            "bq1": 5,
            "ap0": 0.0,
            "aq0": 0,
        },
        {
            "id": 3,
            "microseconds_since_open": 3,
            "quantity": 1,
            "price": 30.0,
            "side": "a",
            "action": "a",
            "bp0": 20.0,
            "bq0": 3,
            "bp1": 15.0,
            "bq1": 5,
            "ap0": 30.0,
            "aq0": 1,
        },
        {
            "id": 2,
            "microseconds_since_open": 4,
            "quantity": 5,
            "price": 15.0,
            "side": "b",
            "action": "d",
            "bp0": 20.0,
            "bq0": 3,
            "bp1": 0.0,
            "bq1": 0,
            "ap0": 30.0,
            "aq0": 1,
        },
    ]

    # Create a temporary output file
    output_file = tmp_path / "output.csv"

    # Process the order book
    process_order_book(input_file, output_file, levels=5)

    # Load the result and compare with the expected output
    result = pd.read_csv(output_file)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True)[
            [
                "id",
                "microseconds_since_open",
                "quantity",
                "price",
                "side",
                "action",
                "bp0",
                "bq0",
                "bp1",
                "bq1",
                "ap0",
                "aq0",
            ]
        ],
        pd.DataFrame(expected_output).reset_index(drop=True),
        check_like=True,  # Ignore column order
    )

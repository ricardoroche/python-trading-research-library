import logging

from research_library.order_book.order_book import process_all_order_books
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, ORDER_BOOK_LEVELS

logger = logging.getLogger(__name__)

def run_datagen():
    """
    Reads raw CSV files, processes the order book updates, and saves the processed files.
    """
    logger.info("Starting data generation from raw order book files...")
    process_all_order_books(RAW_DATA_DIR, PROCESSED_DATA_DIR, ORDER_BOOK_LEVELS)
    logger.info("Data generation completed. Processed files saved in the processed data directory.")

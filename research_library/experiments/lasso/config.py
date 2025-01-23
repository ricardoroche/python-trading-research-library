from pathlib import Path

DATA_DIR: Path = Path("data")
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
FEATURES_DATA_DIR: Path = DATA_DIR / "features"

ORDER_BOOK_LEVELS: int = 5
WINDOWS: list[int] = [1, 15, 60]
HOURS_IN_TRADING_DAY: int = 10

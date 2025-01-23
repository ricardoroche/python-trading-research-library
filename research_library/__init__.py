import logging
from rich.logging import RichHandler

# Configure logging with RichHandler
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",  # RichHandler manages formatting
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)

# Log a message to confirm initialization
logging.getLogger(__name__).info("Rich logging has been configured for the research_library module.")


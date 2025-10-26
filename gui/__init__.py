import logging

from utils.logging import setup_logging

# * Initialize console logging once when GUI package is imported
setup_logging()
logging.getLogger(__name__).info("GUI package import: initializing logging")

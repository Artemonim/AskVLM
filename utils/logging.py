import logging

# * Configure application logging


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging format and level."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(module)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# * Get named logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name."""
    return logging.getLogger(name)

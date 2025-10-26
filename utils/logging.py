import logging

# * Configure application logging


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging format and level.

    Ensures a stream handler to console with a concise, high-level format.
    """
    root = logging.getLogger()
    # Avoid duplicate handlers on repeated setup
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        sh = logging.StreamHandler()
        fmt = logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        sh.setFormatter(fmt)
        root.addHandler(sh)
    root.setLevel(level)


# * Get named logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name."""
    return logging.getLogger(name)

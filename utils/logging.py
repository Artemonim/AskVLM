import logging
from pathlib import Path

# * Configure application logging

_LOG_FILE_NAME = "log.log"


def _repo_root() -> Path:
    """Return repository root directory (parent of ``utils/``)."""
    return Path(__file__).resolve().parent.parent


def _has_console_stream_handler(root: logging.Logger) -> bool:
    """Return whether a non-file ``StreamHandler`` is attached to the root logger."""
    return any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )


def _has_file_handler_for_path(root: logging.Logger, path: Path) -> bool:
    """Return whether a ``FileHandler`` for ``path`` is already attached."""
    target = path.resolve()
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            try:
                if Path(h.baseFilename).resolve() == target:
                    return True
            except (OSError, ValueError):
                continue
    return False


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging format and level.

    Ensures a stream handler to console with a concise, high-level format, and
    a UTF-8 file handler writing the same format to ``log.log`` at the repo root.
    """
    root = logging.getLogger()
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # * Avoid duplicate handlers on repeated setup
    if not _has_console_stream_handler(root):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    log_path = _repo_root() / _LOG_FILE_NAME
    if not _has_file_handler_for_path(root, log_path):
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    root.setLevel(level)


# * Get named logger


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name."""
    return logging.getLogger(name)

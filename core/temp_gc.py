"""Garbage collection for orphaned AskVLM temporary directories."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

ASKVLM_TEMP_PREFIXES: tuple[str, ...] = (
    "askvlm-cli-",
    "askvlm-ext-ipc-",
    "askvlm-daemon-",
    "askvlm-gigaam-",
    "askvlm-gigaam-reencode-",
)


@dataclass(frozen=True, slots=True)
class TempGcSummary:
    """Result of an AskVLM temp directory GC pass.

    Attributes:
        deleted_dirs: Number of stale directories fully removed.
        skipped_dirs: Number of candidate directories left untouched.
        freed_bytes: Best-effort bytes freed by removed directories.

    """

    deleted_dirs: int = 0
    skipped_dirs: int = 0
    freed_bytes: int = 0


def _is_candidate_dir_name(name: str) -> bool:
    """Return whether *name* matches one of known AskVLM temp prefixes."""
    return any(name.startswith(prefix) for prefix in ASKVLM_TEMP_PREFIXES)


def _latest_payload_mtime_seconds(directory: Path) -> float | None:
    """Return freshest payload mtime with conservative error handling.

    The function prefers payload file mtimes whenever at least one file stat
    succeeds. If traversal is incomplete and no file mtime is available, the
    directory is treated as unknown and skipped by GC.

    """
    latest_file_mtime: float | None = None
    had_walk_error = False

    def _on_walk_error(exc: OSError) -> None:
        nonlocal had_walk_error
        had_walk_error = True
        logger.debug("askvlm temp GC walk error: path=%s error=%s", directory, exc)

    try:
        for root, _dirs, files in os.walk(directory, onerror=_on_walk_error):
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                try:
                    file_mtime = file_path.stat().st_mtime
                except OSError as exc:
                    had_walk_error = True
                    logger.debug(
                        "askvlm temp GC cannot stat file: path=%s error=%s",
                        file_path,
                        exc,
                    )
                    continue
                if latest_file_mtime is None or file_mtime > latest_file_mtime:
                    latest_file_mtime = file_mtime
    except OSError as exc:
        had_walk_error = True
        logger.debug(
            "askvlm temp GC cannot walk directory: path=%s error=%s", directory, exc
        )

    # * Any readable payload mtime is authoritative even if sibling stats fail.
    if latest_file_mtime is not None:
        return latest_file_mtime

    # ! Incomplete traversal with zero readable file mtimes is unsafe to delete.
    if had_walk_error:
        return None

    # * Directory mtime is used only for fully-walked trees with no files.
    try:
        return directory.stat().st_mtime
    except OSError as exc:
        logger.debug(
            "askvlm temp GC cannot stat directory: path=%s error=%s", directory, exc
        )
        return None


def _estimate_tree_size_bytes(directory: Path) -> int:
    """Return best-effort directory size in bytes."""
    total_bytes = 0

    def _on_walk_error(exc: OSError) -> None:
        logger.debug("askvlm temp GC size walk error: path=%s error=%s", directory, exc)

    try:
        for root, _dirs, files in os.walk(directory, onerror=_on_walk_error):
            root_path = Path(root)
            for file_name in files:
                file_path = root_path / file_name
                try:
                    total_bytes += file_path.stat().st_size
                except OSError as exc:
                    logger.debug(
                        "askvlm temp GC cannot read file size: path=%s error=%s",
                        file_path,
                        exc,
                    )
    except OSError as exc:
        logger.debug(
            "askvlm temp GC cannot walk for size: path=%s error=%s", directory, exc
        )

    return total_bytes


def _is_directory(entry: Path) -> bool:
    """Return whether *entry* is a directory without raising."""
    try:
        return entry.is_dir()
    except OSError as exc:
        logger.debug(
            "askvlm temp GC cannot inspect entry: path=%s error=%s", entry, exc
        )
        return False


def _safe_rmtree(directory: Path) -> bool:
    """Delete a directory best-effort and report whether it disappeared."""

    def _log_delete_error(path: str | Path, error: BaseException) -> None:
        logger.warning("askvlm temp GC delete failed: path=%s error=%s", path, error)

    def _on_delete_error(
        _func: object,
        path: str,
        exc_info: tuple[type[BaseException], BaseException, object],
    ) -> None:
        _log_delete_error(path, exc_info[1])

    def _on_delete_exc(_func: object, path: str, exc: BaseException) -> None:
        _log_delete_error(path, exc)

    rmtree = cast("Any", shutil.rmtree)
    try:
        try:
            rmtree(directory, ignore_errors=False, onexc=_on_delete_exc)
        except TypeError:
            rmtree(directory, ignore_errors=False, onerror=_on_delete_error)
    except OSError as exc:
        logger.warning("askvlm temp GC delete failed: path=%s error=%s", directory, exc)
    except Exception as exc:  # noqa: BLE001 - GC must never fail caller
        logger.warning(
            "askvlm temp GC unexpected delete error: path=%s error=%s",
            directory,
            exc,
        )

    try:
        return not directory.exists()
    except OSError as exc:
        logger.debug(
            "askvlm temp GC cannot confirm delete: path=%s error=%s",
            directory,
            exc,
        )
        return False


def _purge_candidate_dir(
    directory: Path, *, now: float, stale_after_s: float
) -> TempGcSummary:
    """Purge one candidate directory when it is stale."""
    latest_mtime = _latest_payload_mtime_seconds(directory)
    # ! Unknown age never qualifies for deletion.
    if latest_mtime is None:
        return TempGcSummary(skipped_dirs=1)
    # * Deletion is allowed only when the freshest payload is older than TTL.
    if (now - latest_mtime) <= stale_after_s:
        return TempGcSummary(skipped_dirs=1)

    size_bytes = _estimate_tree_size_bytes(directory)
    if _safe_rmtree(directory):
        return TempGcSummary(deleted_dirs=1, freed_bytes=size_bytes)

    logger.warning(
        "askvlm temp GC stale directory remains after delete attempt: path=%s",
        directory,
    )
    return TempGcSummary(skipped_dirs=1)


def purge_stale_askvlm_temp_dirs(
    temp_root: str | Path | None = None,
    *,
    stale_after_s: float = 3600.0,
    clock: Callable[[], float] = time,
) -> TempGcSummary:
    """Delete stale AskVLM temp directories from top-level temp root.

    The function is best-effort and never raises: all filesystem operations are
    protected and failures are logged.

    Args:
        temp_root: Root temporary directory to scan. Defaults to
            :func:`tempfile.gettempdir`.
        stale_after_s: Minimum age (seconds) required for deletion.
        clock: Clock function returning current timestamp in seconds.

    Returns:
        :class:`TempGcSummary` with removed and skipped directory counts plus
        best-effort freed bytes.

    """
    try:
        root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())

        try:
            now = float(clock())
        except Exception as exc:  # noqa: BLE001 - GC must never fail caller
            logger.warning("askvlm temp GC clock failed: error=%s", exc)
            return TempGcSummary()
        try:
            stale_threshold_s = float(stale_after_s)
        except Exception as exc:  # noqa: BLE001 - GC must never fail caller
            logger.warning("askvlm temp GC stale threshold is invalid: error=%s", exc)
            return TempGcSummary()

        deleted_dirs = 0
        skipped_dirs = 0
        freed_bytes = 0

        try:
            entries = list(root.iterdir())
        except OSError as exc:
            logger.debug("askvlm temp GC cannot list root: root=%s error=%s", root, exc)
            return TempGcSummary()

        for entry in entries:
            try:
                if not _is_directory(entry):
                    continue
                if not _is_candidate_dir_name(entry.name):
                    continue

                entry_summary = _purge_candidate_dir(
                    entry, now=now, stale_after_s=stale_threshold_s
                )
                deleted_dirs += entry_summary.deleted_dirs
                skipped_dirs += entry_summary.skipped_dirs
                freed_bytes += entry_summary.freed_bytes
            except Exception as exc:  # noqa: BLE001 - GC must never fail caller
                skipped_dirs += 1
                logger.warning(
                    "askvlm temp GC unexpected entry failure: path=%s error=%s",
                    entry,
                    exc,
                )

        return TempGcSummary(
            deleted_dirs=deleted_dirs,
            skipped_dirs=skipped_dirs,
            freed_bytes=freed_bytes,
        )
    except Exception as exc:  # noqa: BLE001 - GC must never fail caller
        logger.warning(
            "askvlm temp GC unexpected failure: root=%s error=%s",
            temp_root,
            exc,
        )
        return TempGcSummary()

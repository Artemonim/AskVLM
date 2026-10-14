"""Atomic and bounded JSON file helpers shared by the STT pool modules.

Writes go through a temporary file in the destination directory and an
:func:`os.replace`, so a reader never observes a partially written document.
Reads tolerate missing, partial, oversized and non-object files.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

logger = logging.getLogger(__name__)

# * Default upper bound for one JSON document read from disk.
DEFAULT_MAX_JSON_BYTES = 16 * 1024 * 1024

# * Windows sharing/access violations that are safe to retry on replace/link.
_WINDOWS_RETRY_WINERRORS = (5, 32)
_RETRY_ATTEMPTS = 6
_RETRY_BASE_DELAY_S = 0.05

# * Serializes the check-then-replace fallback of :func:`atomic_create_json`.
_CREATE_FALLBACK_LOCK = threading.Lock()


def _retry_windows_sharing(operation: Callable[[], object], target: Path) -> None:
    """Run *operation*, retrying transient Windows sharing violations.

    Readers on Windows may briefly hold the destination file. That surfaces as
    ``OSError`` with ``winerror`` 5 or 32. Other ``OSError`` values, including
    POSIX ``EACCES`` (no ``winerror``), are re-raised immediately.

    Args:
        operation: Filesystem operation to attempt.
        target: Destination path, used for diagnostics only.

    Raises:
        OSError: On a non-retryable failure, or after the last retry.

    """
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            operation()
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror not in _WINDOWS_RETRY_WINERRORS:
                raise
            if attempt >= _RETRY_ATTEMPTS - 1:
                raise
            delay_s = _RETRY_BASE_DELAY_S * (2**attempt)
            logger.debug(
                "atomic file operation retry: path=%s winerror=%s attempt=%s "
                "delay_s=%s",
                target,
                winerror,
                attempt + 1,
                delay_s,
            )
            sleep(delay_s)
        else:
            return


def replace_with_retry(tmp: Path, path: Path) -> None:
    """Replace *path* with *tmp* via :func:`os.replace`, retrying Windows shares.

    Args:
        tmp: Temporary file that already contains the new contents.
        path: Destination path to replace atomically.

    Raises:
        OSError: On a non-retryable failure, or after six transient Windows
            sharing/access violations.

    """
    # * Path.replace calls os.replace.
    _retry_windows_sharing(lambda: tmp.replace(path), path)


def dumps_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialize *payload* exactly as the atomic writers store it.

    Non-ASCII characters are kept verbatim as UTF-8. Text that UTF-8 cannot
    encode (lone surrogates from Windows paths or exception messages) falls back
    to ASCII-escaped serialization, which represents every code point and parses
    back to the same strings.

    Args:
        payload: JSON-serializable mapping.

    Returns:
        The encoded JSON document.

    """
    text = json.dumps(dict(payload), ensure_ascii=False)
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError:
        return json.dumps(dict(payload), ensure_ascii=True).encode("ascii")


def dumps_json(payload: Mapping[str, object]) -> str:
    """Serialize *payload* exactly as the atomic writers store it.

    Args:
        payload: JSON-serializable mapping.

    Returns:
        The JSON text; its UTF-8 encoding equals :func:`dumps_json_bytes`.

    """
    return dumps_json_bytes(payload).decode("utf-8")


def _discard_temp(tmp_path: Path) -> None:
    """Remove a temporary file on a best-effort basis.

    A failed removal never turns a finished (or already failing) write into a
    different exception; the stale file is reclaimed by the queue purge.

    Args:
        tmp_path: Temporary file to remove.

    """
    try:
        tmp_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.debug(
            "atomic write cannot remove temp file: path=%s error=%s", tmp_path, exc
        )


def _write_temp_json(path: Path, payload: Mapping[str, object]) -> Path:
    """Write *payload* to a durable temporary file next to *path*.

    Args:
        path: Final destination, used to place and name the temporary file.
        payload: JSON-serializable mapping.

    Returns:
        The temporary file path; the caller owns its removal.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dumps_json_bytes(payload)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        _discard_temp(tmp_path)
        raise
    return tmp_path


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write *payload* as JSON to *path* atomically, replacing existing content.

    Args:
        path: Destination file path; missing parent directories are created.
        payload: JSON-serializable mapping.

    """
    tmp_path = _write_temp_json(path, payload)
    try:
        replace_with_retry(tmp_path, path)
    finally:
        _discard_temp(tmp_path)


def atomic_create_json(path: Path, payload: Mapping[str, object]) -> bool:
    """Write *payload* as JSON to *path* atomically, never replacing a file.

    The document is written to a temporary file and published with a hard link,
    which fails when *path* exists; create-once then holds across processes.
    Filesystems without hard links fall back to an existence check followed by
    an atomic replace, serialized by a module-level lock. Create-once then holds
    within this process only; across processes it relies on the single-writer
    invariant of the queue (one supervisor per queue root).

    Args:
        path: Destination file path; missing parent directories are created.
        payload: JSON-serializable mapping.

    Returns:
        ``True`` when this call created *path*, ``False`` when it already existed.

    """
    if path.exists():
        return False
    tmp_path = _write_temp_json(path, payload)
    try:
        try:
            _retry_windows_sharing(lambda: os.link(tmp_path, path), path)
        except FileExistsError:
            return False
        except OSError:
            # ? Hard links are unavailable here; check and replace under a lock.
            with _CREATE_FALLBACK_LOCK:
                if path.exists():
                    return False
                replace_with_retry(tmp_path, path)
        return True
    finally:
        _discard_temp(tmp_path)


def read_json(
    path: Path, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES
) -> dict[str, object] | None:
    """Read a JSON object from *path*, tolerating missing or partial files.

    Args:
        path: Source file path.
        max_bytes: Largest accepted file size in bytes.

    Returns:
        The parsed object, or ``None`` when the file is absent, unreadable,
        larger than *max_bytes*, not valid UTF-8 JSON, or not a JSON object.

    """
    try:
        with path.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        return None
    if len(raw) > max_bytes:
        return None
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict) else None


def touch(path: Path) -> None:
    """Create an empty file at *path* if it does not exist.

    Args:
        path: File to create; missing parent directories are created.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

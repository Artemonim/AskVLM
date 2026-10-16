"""JSON-lines protocol between the STT supervisor and its worker processes.

One JSON object per line, UTF-8, ``"type"`` required, at most
:data:`MAX_LINE_BYTES` bytes per line.

Supervisor -> worker (worker stdin): ``spec`` (first line, sent only after the
process joined its Job Object), ``job``, ``cancel``, ``shutdown``.

Worker -> supervisor: ``hello``, ``loaded``, ``result``, ``fatal``, ``bye`` on a
private duplicate of the worker's original stdout (see :class:`WorkerChannel`);
file descriptor 1 and ``sys.stdout`` are redirected to stderr, so stray prints
of third-party libraries never corrupt the protocol stream.

Worker stdin EOF without ``shutdown`` means the supervisor is gone: the worker
exits with :data:`EXIT_PARENT_GONE`. The module imports no ML packages.

Every ``result`` of a worker carries an optional ``memory`` object, a memory
snapshot of the worker process taken when the result is sent (keys in
:data:`core.stt_pool.engine_api.MEMORY_SNAPSHOT_KEYS`; an unknown figure is
``null``). It is telemetry only: a reader ignores a missing or malformed
snapshot, so the field changes neither :data:`PROTOCOL_VERSION` nor the
handling of the result.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
from typing import IO, TYPE_CHECKING, NoReturn, TextIO

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 2
MAX_LINE_BYTES = 32 * 1024 * 1024
EXIT_PARENT_GONE = 3
EXIT_FATAL = 4

MSG_SPEC, MSG_JOB, MSG_CANCEL, MSG_SHUTDOWN = "spec", "job", "cancel", "shutdown"
MSG_HELLO, MSG_LOADED, MSG_RESULT, MSG_FATAL, MSG_BYE = (
    "hello",
    "loaded",
    "result",
    "fatal",
    "bye",
)

SUPERVISOR_TO_WORKER: frozenset[str] = frozenset(
    {MSG_SPEC, MSG_JOB, MSG_CANCEL, MSG_SHUTDOWN}
)
WORKER_TO_SUPERVISOR: frozenset[str] = frozenset(
    {MSG_HELLO, MSG_LOADED, MSG_RESULT, MSG_FATAL, MSG_BYE}
)
_ALL_MESSAGE_TYPES = SUPERVISOR_TO_WORKER | WORKER_TO_SUPERVISOR

# * Bytes per UTF-8 code point at most; a line with fewer characters than
# * MAX_LINE_BYTES / 4 cannot exceed the byte limit.
_MAX_UTF8_BYTES_PER_CHAR = 4
# * Chunk size used to skip the remainder of an oversized line.
_SKIP_CHUNK = 1024 * 1024
_COMPACT_SEPARATORS = (",", ":")


def encode_message(message: Mapping[str, object]) -> str:
    """Encode one protocol message as a compact, newline-terminated JSON line.

    Non-ASCII characters stay verbatim. Text that UTF-8 cannot encode (lone
    surrogates from Windows paths or exception messages) switches the whole
    line to ASCII-escaped JSON, which parses back to the same strings, the same
    fallback as :func:`core.stt_pool.fsio.dumps_json_bytes`. The result is
    therefore always encodable as UTF-8 and never contains a raw newline.

    Args:
        message: JSON-serializable mapping.

    Returns:
        The encoded line, ending with a single newline.

    Raises:
        TypeError: When *message* holds a value JSON cannot serialize.
        ValueError: When *message* contains a circular reference.

    """
    payload = dict(message)
    text = json.dumps(payload, ensure_ascii=False, separators=_COMPACT_SEPARATORS)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(payload, ensure_ascii=True, separators=_COMPACT_SEPARATORS)
    return text + "\n"


def _exceeds_line_limit(text: str) -> bool:
    """Return whether *text* is longer than :data:`MAX_LINE_BYTES` in UTF-8."""
    if len(text) > MAX_LINE_BYTES:
        return True
    if len(text) * _MAX_UTF8_BYTES_PER_CHAR <= MAX_LINE_BYTES:
        return False
    return len(text.encode("utf-8", errors="surrogatepass")) > MAX_LINE_BYTES


def decode_message(
    line: str, *, allowed: frozenset[str] | None = None
) -> dict[str, object] | None:
    """Decode one protocol line.

    Args:
        line: Raw line, with or without its line terminator.
        allowed: Accepted message types; ``None`` accepts every type of this
            protocol (both directions).

    Returns:
        The message object, or ``None`` for a blank, malformed, oversized or
        non-object line, or a missing or unknown ``"type"``.

    """
    text = line.strip()
    if not text.startswith("{") or _exceeds_line_limit(text):
        return None
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict):
        return None
    kind = data.get("type")
    accepted = _ALL_MESSAGE_TYPES if allowed is None else allowed
    if not isinstance(kind, str) or kind not in accepted:
        return None
    return data


def _ends_line(raw: bytes | str) -> bool:
    """Return whether a ``readline`` chunk ends with its line terminator."""
    if isinstance(raw, bytes):
        return raw.endswith(b"\n")
    return raw.endswith("\n")


def _skip_rest_of_line(stream: IO[bytes] | IO[str]) -> None:
    """Consume *stream* up to and including the next newline (or EOF)."""
    while True:
        chunk = stream.readline(_SKIP_CHUNK)
        if not chunk or _ends_line(chunk):
            return


def iter_lines(
    stream: IO[bytes] | IO[str], *, max_bytes: int = MAX_LINE_BYTES
) -> Iterator[str | None]:
    """Yield the lines of *stream* with a bounded memory footprint.

    Binary streams are decoded as UTF-8 (invalid bytes are replaced); for text
    streams the bound counts characters.

    Args:
        stream: Pipe or file opened for reading.
        max_bytes: Longest accepted line, without its terminator.

    Yields:
        Each line (with its terminator when present), or ``None`` for a line
        longer than *max_bytes*, whose remainder is skipped. Iteration stops at
        EOF.

    """
    while True:
        raw = stream.readline(max_bytes + 1)
        if not raw:
            return
        if len(raw) > max_bytes and not _ends_line(raw):
            _skip_rest_of_line(stream)
            yield None
            continue
        if isinstance(raw, bytes):
            yield raw.decode("utf-8", errors="replace")
        else:
            yield raw


def _describe_garbage(line: str) -> str:
    """Return a short, printable excerpt of an undecodable protocol line."""
    excerpt = line.strip()
    limit = 200
    if len(excerpt) > limit:
        excerpt = excerpt[:limit] + "..."
    return ascii(excerpt)


def iter_messages(
    stream: IO[bytes] | IO[str], *, allowed: frozenset[str] | None, source: str
) -> Iterator[dict[str, object]]:
    """Yield decoded protocol messages from *stream* until EOF.

    Oversized and undecodable lines are logged with *source* and skipped.

    Args:
        stream: Pipe opened for reading.
        allowed: Accepted message types (see :func:`decode_message`).
        source: Peer description used in log records.

    Yields:
        Decoded messages, in stream order.

    """
    for line in iter_lines(stream):
        if line is None:
            logger.warning(
                "protocol line over %d bytes skipped: source=%s", MAX_LINE_BYTES, source
            )
            continue
        message = decode_message(line, allowed=allowed)
        if message is None:
            if line.strip():
                logger.warning(
                    "undecodable protocol line skipped: source=%s line=%s",
                    source,
                    _describe_garbage(line),
                )
            continue
        yield message


def _exit_parent_gone() -> NoReturn:
    """Flush stderr and end the worker at once: the supervisor is gone."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        sys.stderr.flush()
    os._exit(EXIT_PARENT_GONE)


class WorkerChannel:
    """Worker-side protocol writer on a private duplicate of the original stdout.

    Create it with :meth:`install` as the first thing a worker does: from then
    on file descriptor 1 and ``sys.stdout`` point to stderr (the per-worker
    log), and only this channel writes to the supervisor's pipe. The duplicate
    descriptor is not inheritable, so processes the worker starts (ffmpeg)
    never hold the protocol pipe open.
    """

    def __init__(self, stream: TextIO) -> None:
        """Wrap an already-opened protocol stream; use :meth:`install` instead.

        Args:
            stream: Text stream to the supervisor.

        """
        self._stream = stream
        self._lock = threading.Lock()

    @classmethod
    def install(cls) -> WorkerChannel:
        """Move the protocol off file descriptor 1 and return its channel.

        Steps: flush ``sys.stdout``, duplicate descriptor 1 for the protocol,
        point descriptor 1 at stderr, and make ``sys.stdout`` an alias of
        ``sys.stderr``. The protocol stream is UTF-8 with LF line endings.

        Returns:
            The installed channel.

        """
        if sys.stdout is not None:
            with contextlib.suppress(OSError, ValueError):
                sys.stdout.flush()
        protocol_fd = os.dup(1)
        try:
            os.dup2(2, 1)
            stream = os.fdopen(protocol_fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(protocol_fd)
            raise
        sys.stdout = sys.stderr
        return cls(stream)

    def send(self, message: Mapping[str, object]) -> None:
        """Write one message and flush it (thread-safe).

        A broken pipe means the supervisor is gone: the process then ends at
        once with :data:`EXIT_PARENT_GONE`.

        Args:
            message: Protocol message.

        """
        line = encode_message(message)
        with self._lock:
            try:
                self._stream.write(line)
                self._stream.flush()
            except (OSError, ValueError):
                # ! The supervisor closed the pipe: never linger as an orphan.
                _exit_parent_gone()

    def iter_stdin(self) -> Iterator[dict[str, object]]:
        """Yield decoded supervisor messages from stdin until EOF.

        Lines that are not supervisor-to-worker messages are logged and
        skipped. The caller decides what EOF means (see :data:`EXIT_PARENT_GONE`).

        Yields:
            Messages in arrival order.

        """
        stdin = sys.stdin
        if stdin is None:
            return
        stream: IO[bytes] | IO[str] = getattr(stdin, "buffer", None) or stdin
        yield from iter_messages(
            stream, allowed=SUPERVISOR_TO_WORKER, source="supervisor"
        )

    def close(self) -> None:
        """Close the protocol stream (best effort, idempotent)."""
        with self._lock, contextlib.suppress(OSError, ValueError):
            self._stream.close()

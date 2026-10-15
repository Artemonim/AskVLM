"""Verified provisioning of the CPU speaker-diarization models (sherpa-onnx).

The diarization backend needs two ONNX networks, the same pair as the hardsub
skill: the pyannote segmentation 3.0 export (shipped inside a ``.tar.bz2``
release archive) and the WeSpeaker ResNet34-LM speaker embedding model. Both
are public GitHub release assets of ``k2-fsa/sherpa-onnx``; no token is
needed. :func:`ensure_diarization_models` makes both files available in a
cache folder (by default ``<project>/.cache/models/sherpa-diarization``) and
returns their paths.

The module depends on the standard library and :mod:`core.settings` only, so
the thin client, the supervisor and the worker processes can import it
without loading any ML package.

Security considerations:

* **Pinned hashes.** Every file (both models and the segmentation archive) is
  pinned by SHA-256 and exact size. A cached file is used only when its hash
  matches; otherwise it is downloaded again and replaced. Nothing is ever
  used or installed before its hash is verified.
* **HTTPS only.** The default fetcher accepts only ``https`` URLs, including
  every redirect target (GitHub redirects release assets to its CDN). The
  hash pin is the integrity guarantee; HTTPS adds transport confidentiality.
* **Size caps.** Downloads stop as soon as they exceed the pinned size, and
  the archive member must have exactly the pinned model size, so a hostile or
  broken server cannot fill the disk.
* **Single-member extraction.** The archive is opened only after its own hash
  is verified. Exactly one member with the pinned name is read, and only when
  it is a regular file (no links, devices or folders). It is streamed through
  ``tarfile.extractfile`` into a temporary file; nothing is written under a
  name taken from the archive, ``extractall`` is never used, and no shell or
  subprocess is involved, so path traversal and link tricks have no effect.
* **Atomic installation.** Data is written to ``<name>.<random>.part`` in the
  cache folder, flushed to disk, verified, and moved into place with
  :func:`os.replace`; any failure removes the ``.part`` file, so a partial or
  unverified file never appears under a model name. The downloaded archive is
  a ``.part`` file too and is deleted after extraction.
* **Cross-process lock.** The check and the download run under an exclusive
  OS lock on one byte of a lock file in the cache folder (``msvcrt.locking``
  on Windows, ``fcntl.flock`` elsewhere), with a re-check after the lock is
  acquired. Pool workers and one-shot CLI runs therefore never race; the
  operating system releases the lock when its holder exits, so a crashed
  process cannot leave a stale lock behind. Waiting is bounded by a timeout.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import secrets
import stat
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol

from core.settings import get_project_cache_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from http.client import HTTPMessage
    from typing import IO

logger = logging.getLogger(__name__)

# * Seconds to wait for another process that holds the cache lock (it may be
# * downloading ~33 MB on a slow link).
DEFAULT_LOCK_TIMEOUT_S: Final = 600.0
# * Socket timeout of one network operation of the default fetcher.
DOWNLOAD_TIMEOUT_S: Final = 60.0

_CHUNK_BYTES: Final = 1024 * 1024
_LOCK_POLL_S: Final = 0.2
_LOCK_FILE_NAME: Final = "diarization-models.lock"
_PART_SUFFIX: Final = ".part"
_PART_TOKEN_BYTES: Final = 8
# * Archive members inspected before the pinned member must have been found.
_MAX_ARCHIVE_MEMBERS: Final = 64
# * Windows sharing/access violations (an antivirus scan, a reader) that are
# * worth retrying when a file is renamed or removed.
_WINDOWS_RETRY_WINERRORS: Final = (5, 32)
_FS_RETRY_ATTEMPTS: Final = 6
_FS_RETRY_BASE_DELAY_S: Final = 0.05


class DiarizationModelError(RuntimeError):
    """A diarization model could not be downloaded, verified or installed."""


class ByteStream(Protocol):
    """Readable binary stream returned by a fetcher."""

    def read(self, size: int = -1, /) -> bytes:
        """Return up to *size* bytes; ``b""`` at the end of the stream."""
        ...

    def close(self) -> None:
        """Release the underlying connection or file."""
        ...


@dataclass(frozen=True)
class ArchiveSource:
    """A release archive that contains a model file.

    Attributes:
        url: Download URL of the archive.
        sha256: Pinned SHA-256 of the archive (lowercase hex).
        size: Pinned archive size in bytes.
        member: Exact member name of the model inside the archive.

    """

    url: str
    sha256: str
    size: int
    member: str


@dataclass(frozen=True)
class ModelSpec:
    """A pinned model file.

    Attributes:
        file_name: File name inside the cache folder.
        sha256: Pinned SHA-256 of the model file (lowercase hex).
        size: Pinned model size in bytes.
        url: Direct download URL; ``None`` when the model comes from *archive*.
        archive: Release archive that contains the model, if any.

    """

    file_name: str
    sha256: str
    size: int
    url: str | None = None
    archive: ArchiveSource | None = None

    @property
    def source_url(self) -> str:
        """Return the URL that is downloaded for this model."""
        if self.archive is not None:
            return self.archive.url
        if self.url is None:  # pragma: no cover - guarded by the pinned specs
            msg = f"model {self.file_name} has neither a URL nor an archive"
            raise DiarizationModelError(msg)
        return self.url


# * pyannote segmentation 3.0 (ONNX export by sherpa-onnx, MIT license).
SEGMENTATION_MODEL: Final = ModelSpec(
    file_name="sherpa-onnx-pyannote-segmentation-3-0.onnx",
    sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    size=5_992_913,
    archive=ArchiveSource(
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
            "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        ),
        sha256="24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
        size=6_958_444,
        member="sherpa-onnx-pyannote-segmentation-3-0/model.onnx",
    ),
)
# * WeSpeaker ResNet34-LM speaker embeddings (VoxCeleb). The release tag is
# * spelled "speaker-recongition-models" upstream.
EMBEDDING_MODEL: Final = ModelSpec(
    file_name="wespeaker_en_voxceleb_resnet34_LM.onnx",
    sha256="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
    size=26_530_550,
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
    ),
)
MODEL_SPECS: Final = (SEGMENTATION_MODEL, EMBEDDING_MODEL)


@dataclass(frozen=True)
class DiarizationModelPaths:
    """Paths of the verified diarization models.

    Attributes:
        segmentation: pyannote segmentation 3.0 ONNX model.
        embedding: WeSpeaker ResNet34-LM ONNX model.

    """

    segmentation: Path
    embedding: Path


def default_cache_dir() -> Path:
    """Return the default model folder.

    Returns:
        ``<project>/.cache/models/sherpa-diarization``.

    """
    return get_project_cache_dir() / "models" / "sherpa-diarization"


def model_paths(cache_dir: str | Path | None = None) -> DiarizationModelPaths:
    """Return where the models live in *cache_dir*, without checking them.

    Args:
        cache_dir: Model folder; ``None`` selects :func:`default_cache_dir`.

    Returns:
        The model paths inside the folder.

    """
    root = default_cache_dir() if cache_dir is None else Path(cache_dir)
    return DiarizationModelPaths(
        segmentation=root / SEGMENTATION_MODEL.file_name,
        embedding=root / EMBEDDING_MODEL.file_name,
    )


# * Network access.


class _HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to ``https`` URLs."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Return the redirected request, refusing non-https targets.

        Raises:
            DiarizationModelError: When the redirect leaves ``https``.

        """
        if urllib.parse.urlsplit(newurl).scheme.lower() != "https":
            error = (
                f"refusing the redirect from {req.full_url} to {newurl}: "
                "only https downloads are allowed"
            )
            raise DiarizationModelError(error)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_fetch(url: str) -> ByteStream:
    """Open an ``https`` URL for streaming (the default fetcher).

    Args:
        url: Absolute ``https`` URL.

    Returns:
        The response stream.

    Raises:
        DiarizationModelError: When *url* is not an ``https`` URL or a
            redirect leaves ``https``.
        OSError: When the connection or the HTTP request fails.

    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme.lower() != "https" or not parts.netloc:
        msg = f"refusing to download {url!r}: only https URLs are allowed"
        raise DiarizationModelError(msg)
    opener = urllib.request.build_opener(_HttpsOnlyRedirectHandler)
    response: ByteStream = opener.open(url, timeout=DOWNLOAD_TIMEOUT_S)
    return response


# * Filesystem helpers.


def _retry_windows_sharing(operation: Callable[[], object]) -> None:
    """Run *operation*, retrying transient Windows sharing violations.

    Raises:
        OSError: On a non-retryable failure, or after the last retry.

    """
    for attempt in range(_FS_RETRY_ATTEMPTS):
        try:
            operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _WINDOWS_RETRY_WINERRORS:
                raise
            if attempt >= _FS_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_FS_RETRY_BASE_DELAY_S * (2**attempt))
        else:
            return


def _remove_file(path: Path) -> None:
    """Delete *path* if it exists; a failure is logged, never raised."""
    try:
        _retry_windows_sharing(lambda: path.unlink(missing_ok=True))
    except OSError as exc:
        logger.warning("cannot remove the temporary file %s: %s", path, exc)


def _new_part_path(folder: Path, name: str) -> Path:
    """Return a fresh ``<name>.<random>.part`` path in *folder*."""
    token = secrets.token_hex(_PART_TOKEN_BYTES)
    return folder / f"{name}.{token}{_PART_SUFFIX}"


def _part_pattern(name: str) -> re.Pattern[str]:
    """Return the pattern of the ``.part`` files created for *name*."""
    token = rf"[0-9a-f]{{{_PART_TOKEN_BYTES * 2}}}"
    return re.compile(rf"{re.escape(name)}\.{token}{re.escape(_PART_SUFFIX)}")


def _remove_stale_parts(folder: Path, specs: Iterable[ModelSpec]) -> None:
    """Delete ``.part`` files of the known downloads left by a killed process.

    Called only while holding the cache lock, when no other process writes
    into the folder.

    Args:
        folder: Model folder.
        specs: Models whose file and archive names identify the ``.part`` files.

    """
    names: list[str] = []
    for spec in specs:
        names.append(spec.file_name)
        if spec.archive is not None:
            names.append(_archive_file_name(spec.archive))
    patterns = [_part_pattern(name) for name in names]
    with contextlib.suppress(OSError):
        for entry in folder.iterdir():
            if any(pattern.fullmatch(entry.name) for pattern in patterns):
                logger.info("removing a stale partial download: %s", entry)
                _remove_file(entry)


def _archive_file_name(archive: ArchiveSource) -> str:
    """Return the file name of *archive* taken from its URL path."""
    return Path(urllib.parse.urlsplit(archive.url).path).name


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 (lowercase hex) of *path*, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _file_matches(path: Path, sha256: str, size: int) -> bool:
    """Return whether *path* is a regular file with the pinned size and hash."""
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            return False
        return _sha256_file(path) == sha256
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("cannot verify %s: %s", path, exc)
        return False


def _stream_to_file(
    stream: ByteStream, out: IO[bytes], *, max_bytes: int, source: str
) -> tuple[int, str]:
    """Copy *stream* into *out* while hashing it, stopping past *max_bytes*.

    Args:
        stream: Source stream.
        out: Open binary destination.
        max_bytes: Largest accepted total size.
        source: Description of the source for the error message.

    Returns:
        ``(size, sha256)`` of the copied data.

    Raises:
        DiarizationModelError: When the data exceeds *max_bytes*.

    """
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            msg = f"{source} is larger than the pinned size of {max_bytes} bytes"
            raise DiarizationModelError(msg)
        digest.update(chunk)
        out.write(chunk)
    out.flush()
    os.fsync(out.fileno())
    return total, digest.hexdigest()


def _write_verified_part(
    open_source: Callable[[], ByteStream],
    part: Path,
    *,
    sha256: str,
    size: int,
    source: str,
) -> None:
    """Stream a source into *part* and verify its size and hash.

    The ``.part`` file is removed on any failure.

    Args:
        open_source: Opens the source stream.
        part: New temporary file to create.
        sha256: Pinned SHA-256 of the data.
        size: Pinned size of the data in bytes.
        source: Description of the source (URL or archive member).

    Raises:
        DiarizationModelError: When reading fails or the data does not match
            the pinned size and hash.

    """
    verified = False
    try:
        actual_size, actual_sha = _copy_to_part(
            open_source, part, max_bytes=size, source=source
        )
        verified = actual_size == size and actual_sha == sha256
    finally:
        if not verified:
            _remove_file(part)
    if not verified:
        msg = (
            f"{source} failed verification: expected {size} bytes with "
            f"sha256 {sha256}, got {actual_size} bytes with sha256 {actual_sha}"
        )
        raise DiarizationModelError(msg)


def _copy_to_part(
    open_source: Callable[[], ByteStream], part: Path, *, max_bytes: int, source: str
) -> tuple[int, str]:
    """Create *part* and stream the source into it.

    Returns:
        ``(size, sha256)`` of the written data.

    Raises:
        DiarizationModelError: When opening or reading the source or writing
            *part* fails, or the data exceeds *max_bytes*.

    """
    try:
        with part.open("xb") as out, contextlib.closing(open_source()) as stream:
            return _stream_to_file(stream, out, max_bytes=max_bytes, source=source)
    except DiarizationModelError:
        raise
    except Exception as exc:
        msg = f"cannot read {source} into {part}: {exc}"
        raise DiarizationModelError(msg) from exc


def _install(part: Path, destination: Path) -> None:
    """Move a verified ``.part`` file into place atomically.

    Raises:
        DiarizationModelError: When the rename fails; the ``.part`` is removed.

    """
    try:
        _retry_windows_sharing(lambda: part.replace(destination))
    except OSError as exc:
        _remove_file(part)
        msg = f"cannot install the verified model {part} as {destination}: {exc}"
        raise DiarizationModelError(msg) from exc


def _open_archive_member(
    archive_path: Path, archive: ArchiveSource, size: int
) -> ByteStream:
    """Open the pinned member of a verified archive for reading.

    Args:
        archive_path: Verified archive file.
        archive: Archive pin with the member name.
        size: Pinned size of the member in bytes.

    Returns:
        A stream of the member data; closing it closes the archive.

    Raises:
        DiarizationModelError: When the member is missing, is not a regular
            file, or has another size.

    """
    # * The archive stays open until the returned stream is closed.
    tar = tarfile.open(archive_path, mode="r:bz2")  # noqa: SIM115
    try:
        data = _member_data(tar, archive, size)
    except BaseException:
        tar.close()
        raise
    return _MemberStream(data, tar)


def _member_data(tar: tarfile.TarFile, archive: ArchiveSource, size: int) -> IO[bytes]:
    """Return the data stream of the pinned member of *tar*.

    Only the first :data:`_MAX_ARCHIVE_MEMBERS` members are inspected; the
    member must be a regular file (not a link, folder or device) with exactly
    the pinned size.

    Raises:
        DiarizationModelError: When the member is missing, is not a regular
            file, or has another size.

    """
    member: tarfile.TarInfo | None = None
    for index, candidate in enumerate(tar):
        if candidate.name == archive.member:
            member = candidate
            break
        if index + 1 >= _MAX_ARCHIVE_MEMBERS:
            break
    if member is None:
        msg = f"{archive.url} has no member {archive.member!r}"
        raise DiarizationModelError(msg)
    if not member.isreg():
        msg = f"{archive.url}: member {archive.member!r} is not a regular file"
        raise DiarizationModelError(msg)
    if member.size != size:
        msg = (
            f"{archive.url}: member {archive.member!r} has {member.size} bytes, "
            f"expected {size}"
        )
        raise DiarizationModelError(msg)
    data = tar.extractfile(member)
    if data is None:  # pragma: no cover - a regular member always has data
        msg = f"{archive.url}: member {archive.member!r} has no data"
        raise DiarizationModelError(msg)
    return data


class _MemberStream:
    """Archive member stream that also closes its archive."""

    def __init__(self, data: IO[bytes], tar: tarfile.TarFile) -> None:
        self._data = data
        self._tar = tar

    def read(self, size: int = -1, /) -> bytes:
        """Return up to *size* bytes of the member."""
        return self._data.read(size)

    def close(self) -> None:
        """Close the member stream and the archive."""
        try:
            self._data.close()
        finally:
            self._tar.close()


# * Provisioning.


def _provision(
    spec: ModelSpec, folder: Path, fetch: Callable[[str], ByteStream]
) -> None:
    """Download, verify and install one model into *folder*.

    Raises:
        DiarizationModelError: On any download, verification or install failure.

    """
    destination = folder / spec.file_name
    archive = spec.archive
    started = time.monotonic()
    if archive is None:
        url = spec.source_url
        logger.info("downloading %s (%d bytes) from %s", spec.file_name, spec.size, url)
        part = _new_part_path(folder, spec.file_name)
        _write_verified_part(
            lambda: fetch(url), part, sha256=spec.sha256, size=spec.size, source=url
        )
        _install(part, destination)
    else:
        logger.info(
            "downloading the archive of %s (%d bytes) from %s",
            spec.file_name,
            archive.size,
            archive.url,
        )
        archive_part = _new_part_path(folder, _archive_file_name(archive))
        _write_verified_part(
            lambda: fetch(archive.url),
            archive_part,
            sha256=archive.sha256,
            size=archive.size,
            source=archive.url,
        )
        try:
            part = _new_part_path(folder, spec.file_name)
            _write_verified_part(
                lambda: _open_archive_member(archive_part, archive, spec.size),
                part,
                sha256=spec.sha256,
                size=spec.size,
                source=f"{archive.url} member {archive.member}",
            )
            _install(part, destination)
        finally:
            _remove_file(archive_part)
    logger.info("installed %s in %.1f s", destination, time.monotonic() - started)


if sys.platform == "win32":
    import msvcrt

    def _try_lock(fd: int) -> bool:
        """Take a non-blocking exclusive lock on the first byte of *fd*."""
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the lock taken by :func:`_try_lock`."""
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            # * Closing the descriptor releases the lock regardless.
            logger.debug("model cache unlock failed: %s", exc)

else:
    import fcntl

    def _try_lock(fd: int) -> bool:
        """Take a non-blocking exclusive lock on *fd*."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        return True

    def _unlock(fd: int) -> None:
        """Release the lock taken by :func:`_try_lock`."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError as exc:
            # * Closing the descriptor releases the lock regardless.
            logger.debug("model cache unlock failed: %s", exc)


@contextlib.contextmanager
def _cache_lock(lock_path: Path, timeout_s: float) -> Iterator[None]:
    """Hold the exclusive cross-process lock of a model folder.

    Args:
        lock_path: Lock file (created when missing, never deleted).
        timeout_s: Seconds to wait for another holder.

    Yields:
        Nothing; the lock is held inside the ``with`` block.

    Raises:
        DiarizationModelError: When the lock file cannot be opened or the lock
            is not acquired in time.

    """
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666)
    except OSError as exc:
        msg = f"cannot open the model cache lock {lock_path}: {exc}"
        raise DiarizationModelError(msg) from exc
    try:
        deadline = time.monotonic() + timeout_s
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                msg = (
                    f"timed out after {timeout_s:g} s waiting for the model cache "
                    f"lock {lock_path}; another process may be downloading the "
                    "diarization models"
                )
                raise DiarizationModelError(msg)
            time.sleep(_LOCK_POLL_S)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def ensure_diarization_models(
    cache_dir: str | Path | None = None,
    *,
    fetch: Callable[[str], ByteStream] | None = None,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
) -> DiarizationModelPaths:
    """Return verified diarization models, downloading them when needed.

    A cached file is reused only when its SHA-256 matches the pin; a missing
    or mismatching file is downloaded and replaced atomically. The check and
    the download run under the cross-process lock of the folder, with a
    re-check after the lock is acquired, so concurrent callers download each
    file at most once.

    Args:
        cache_dir: Model folder; ``None`` selects :func:`default_cache_dir`.
        fetch: Opens a URL as a binary stream; defaults to :func:`https_fetch`.
        lock_timeout_s: Seconds to wait for another process holding the lock.

    Returns:
        Paths of the two verified models.

    Raises:
        DiarizationModelError: When a model cannot be downloaded, verified or
            installed, or the lock is not acquired in time.

    """
    paths = model_paths(cache_dir)
    folder = paths.segmentation.parent
    targets = (
        (SEGMENTATION_MODEL, paths.segmentation),
        (EMBEDDING_MODEL, paths.embedding),
    )
    if all(_file_matches(path, spec.sha256, spec.size) for spec, path in targets):
        return paths
    fetcher = https_fetch if fetch is None else fetch
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        msg = f"cannot create the model folder {folder}: {exc}"
        raise DiarizationModelError(msg) from exc
    with _cache_lock(folder / _LOCK_FILE_NAME, lock_timeout_s):
        _remove_stale_parts(folder, (spec for spec, _ in targets))
        for spec, path in targets:
            if _file_matches(path, spec.sha256, spec.size):
                continue
            if path.exists():
                logger.warning(
                    "%s does not match its pinned sha256; replacing it", path
                )
            _provision(spec, folder, fetcher)
    return paths

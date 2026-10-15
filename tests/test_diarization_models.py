"""Model provisioning scenarios with real files, archives and OS cache locks."""

from __future__ import annotations

import hashlib
import io
import json
import multiprocessing
import os
import tarfile
from dataclasses import dataclass
from http.client import HTTPMessage
from typing import TYPE_CHECKING
from urllib.request import Request

import pytest

from core import diarization_models as models
from core.diarization_models import ArchiveSource, DiarizationModelError, ModelSpec

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.synchronize import Event
    from pathlib import Path

_SEGMENTATION = b"verified segmentation bytes"
_EMBEDDING = b"verified embedding bytes"
_MEMBER = "release/model.onnx"
_WAIT_S = 15.0


@dataclass(frozen=True)
class Downloads:
    """Small independently pinned release assets at the external fetch boundary."""

    segmentation: ModelSpec
    embedding: ModelSpec
    payloads: dict[str, bytes]


def _archive(member: tarfile.TarInfo, content: bytes = b"") -> bytes:
    """Build a real compressed archive without extracting any member."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:bz2") as archive:
        archive.addfile(member, io.BytesIO(content) if member.isreg() else None)
    return stream.getvalue()


def _downloads(archive_bytes: bytes | None = None) -> Downloads:
    """Pin known-good fixture bytes using the standard SHA-256 implementation."""
    if archive_bytes is None:
        member = tarfile.TarInfo(_MEMBER)
        member.size = len(_SEGMENTATION)
        archive_bytes = _archive(member, _SEGMENTATION)
    archive_url = "https://example.invalid/segmentation.tar.bz2"
    embedding_url = "https://example.invalid/embedding.onnx"
    segmentation = ModelSpec(
        "segmentation.onnx",
        hashlib.sha256(_SEGMENTATION).hexdigest(),
        len(_SEGMENTATION),
        archive=ArchiveSource(
            archive_url,
            hashlib.sha256(archive_bytes).hexdigest(),
            len(archive_bytes),
            _MEMBER,
        ),
    )
    embedding = ModelSpec(
        "embedding.onnx",
        hashlib.sha256(_EMBEDDING).hexdigest(),
        len(_EMBEDDING),
        url=embedding_url,
    )
    return Downloads(
        segmentation,
        embedding,
        {archive_url: archive_bytes, embedding_url: _EMBEDDING},
    )


def _set_pins(downloads: Downloads, monkeypatch: pytest.MonkeyPatch) -> None:
    """Select tiny fixture assets while keeping the provisioning components real."""
    monkeypatch.setattr(models, "SEGMENTATION_MODEL", downloads.segmentation)
    monkeypatch.setattr(models, "EMBEDDING_MODEL", downloads.embedding)
    monkeypatch.setattr(
        models, "MODEL_SPECS", (downloads.segmentation, downloads.embedding)
    )


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch) -> Downloads:
    """Keep every test fetch offline and every model pin tied to fixture bytes."""
    assets = _downloads()
    _set_pins(assets, monkeypatch)
    return assets


@pytest.mark.integration
def test_provisioning_verifies_cache_repairs_corruption_and_reuses_valid_models(
    tmp_path: Path, downloads: Downloads
) -> None:
    """Catch unverified cache reuse, repeat downloads or corrupt replacement."""
    requested: list[str] = []
    responses: list[io.BytesIO] = []

    def fetch(url: str) -> io.BytesIO:
        requested.append(url)
        response = io.BytesIO(downloads.payloads[url])
        responses.append(response)
        return response

    paths = models.ensure_diarization_models(tmp_path, fetch=fetch)
    assert paths.segmentation.read_bytes() == _SEGMENTATION
    assert paths.embedding.read_bytes() == _EMBEDDING
    assert len(requested) == 2
    assert all(response.closed for response in responses)
    assert models.ensure_diarization_models(tmp_path, fetch=fetch) == paths
    assert len(requested) == 2
    paths.embedding.write_bytes(b"!" * len(_EMBEDDING))
    models.ensure_diarization_models(tmp_path, fetch=fetch)
    assert paths.embedding.read_bytes() == _EMBEDDING
    assert requested[2:] == [downloads.embedding.source_url]
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.integration
@pytest.mark.parametrize("fault", ["hash", "short", "oversize", "interrupted"])
def test_failed_model_download_never_replaces_a_destination_or_leaves_partial_files(
    tmp_path: Path, downloads: Downloads, fault: str
) -> None:
    """Catch corrupt, oversized or interrupted downloads becoming an installed model."""
    paths = models.model_paths(tmp_path)
    paths.segmentation.write_bytes(_SEGMENTATION)
    previous = b"old unverified file"
    paths.embedding.write_bytes(previous)
    payload = {
        "hash": b"!" * len(_EMBEDDING),
        "short": _EMBEDDING[:-1],
        "oversize": _EMBEDDING + b"!",
        "interrupted": _EMBEDDING,
    }[fault]

    class Response(io.BytesIO):
        """Fail reading after a partial response at the network boundary."""

        def read(self, size: int = -1, /) -> bytes:
            if fault == "interrupted":
                if self.tell() > 0:
                    msg = "connection interrupted"
                    raise OSError(msg)
                size = 4
            return super().read(size)

    response = Response(payload)
    with pytest.raises(DiarizationModelError) as error:
        models.ensure_diarization_models(tmp_path, fetch=lambda _url: response)
    assert downloads.embedding.source_url in str(error.value)
    assert response.closed
    assert paths.embedding.read_bytes() == previous
    assert paths.segmentation.read_bytes() == _SEGMENTATION
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.integration
def test_archive_hash_is_verified_before_the_archive_parser_is_used(
    tmp_path: Path, downloads: Downloads
) -> None:
    """Catch tar parsing an archive before checking its pinned hash."""
    source = downloads.segmentation.archive
    assert source is not None
    payload = b"!" * source.size
    with pytest.raises(DiarizationModelError) as error:
        models.ensure_diarization_models(
            tmp_path, fetch=lambda _url: io.BytesIO(payload)
        )
    detail = str(error.value)
    assert source.sha256 in detail
    assert hashlib.sha256(payload).hexdigest() in detail
    assert not models.model_paths(tmp_path).segmentation.exists()
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "directory", "traversal", "oversize", "missing"]
)
def test_unsafe_archive_members_cannot_install_models_or_escape_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Catch unsafe tar members bypassing verified single-file extraction."""
    member = tarfile.TarInfo(_MEMBER)
    content = _SEGMENTATION
    if kind in {"symlink", "hardlink"}:
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "../outside.onnx"
    elif kind == "directory":
        member.type = tarfile.DIRTYPE
    elif kind == "traversal":
        member.name = "../outside.onnx"
    elif kind == "oversize":
        content += b"!"
    elif kind == "missing":
        member.name = "release/other.onnx"
    if member.isreg():
        member.size = len(content)
    assets = _downloads(_archive(member, content))
    _set_pins(assets, monkeypatch)
    cache = tmp_path / "cache"
    with pytest.raises(DiarizationModelError):
        models.ensure_diarization_models(
            cache, fetch=lambda url: io.BytesIO(assets.payloads[url])
        )
    assert not models.model_paths(cache).segmentation.exists()
    assert not list(cache.glob("*.part"))
    assert not (tmp_path / "outside.onnx").exists()
    assert not (cache / "release").exists()


@pytest.mark.integration
def test_wrong_archive_model_hash_is_rejected_after_a_valid_archive_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch trusting an archive hash while neglecting its model's hash."""
    member = tarfile.TarInfo(_MEMBER)
    member.size = len(_SEGMENTATION)
    assets = _downloads(_archive(member, b"!" * len(_SEGMENTATION)))
    _set_pins(assets, monkeypatch)
    with pytest.raises(DiarizationModelError) as error:
        models.ensure_diarization_models(
            tmp_path, fetch=lambda url: io.BytesIO(assets.payloads[url])
        )
    assert assets.segmentation.sha256 in str(error.value)
    assert not models.model_paths(tmp_path).segmentation.exists()
    assert not list(tmp_path.glob("*.part"))


def _cache_peer(
    root: Path, assets: Downloads, ready: Event, release: Event, connection: Connection
) -> None:
    """Exercise the same cache from a distinct process with a gated external fetch."""
    try:
        with pytest.MonkeyPatch.context() as patch:
            _set_pins(assets, patch)

            def fetch(url: str) -> io.BytesIO:
                journal = root / f"fetch-{os.getpid()}.jsonl"
                with journal.open("a", encoding="utf-8") as log:
                    log.write(json.dumps(url) + "\n")
                if not release.wait(_WAIT_S):
                    msg = "test fetch gate was not released"
                    raise TimeoutError(msg)  # noqa: TRY301 - the except clause reports it to the parent
                return io.BytesIO(assets.payloads[url])

            ready.set()
            paths = models.ensure_diarization_models(
                root, fetch=fetch, lock_timeout_s=10
            )
            connection.send(
                (paths.segmentation.read_bytes(), paths.embedding.read_bytes())
            )
    except Exception as error:  # noqa: BLE001 - report child failure to its parent
        connection.send((type(error).__name__, str(error)))
    finally:
        connection.close()


@pytest.mark.integration
def test_concurrent_processes_recheck_the_locked_cache_and_download_each_model_once(
    tmp_path: Path, downloads: Downloads
) -> None:
    """Catch cache races, duplicate downloads or stale checks after locking."""
    context = multiprocessing.get_context("spawn")
    release = context.Event()
    peers = []
    connections = []
    try:
        for _ in range(2):
            parent, child = context.Pipe(duplex=False)
            ready = context.Event()
            peer = context.Process(
                target=_cache_peer, args=(tmp_path, downloads, ready, release, child)
            )
            peers.append(peer)
            connections.append(parent)
            peer.start()
            child.close()
            assert ready.wait(_WAIT_S)
        release.set()
        for peer, connection in zip(peers, connections, strict=True):
            assert connection.poll(_WAIT_S)
            assert connection.recv() == (_SEGMENTATION, _EMBEDDING)
            peer.join(_WAIT_S)
            assert peer.exitcode == 0
        requests = [
            json.loads(line)
            for path in tmp_path.glob("fetch-*.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        assert sorted(requests) == sorted(downloads.payloads)
        assert not list(tmp_path.glob("*.part"))
    finally:
        release.set()
        for peer in peers:
            if peer.is_alive():
                peer.kill()
            peer.join(_WAIT_S)
            peer.close()
        for connection in connections:
            connection.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://example.invalid/model",
        "file:///model",
        "ftp://example.invalid/model",
        "https:///model",
    ],
)
def test_default_fetcher_refuses_non_https_or_hostless_urls(url: str) -> None:
    """Catch insecure or local-file URLs reaching the default fetcher."""
    with pytest.raises(DiarizationModelError) as error:
        models.https_fetch(url)
    assert url in str(error.value)


@pytest.mark.parametrize("target", ["http://example.invalid/model", "file:///model"])
def test_https_redirect_handler_refuses_transport_downgrades(target: str) -> None:
    """Catch a secure release URL redirecting downloads to HTTP or a local file."""
    request = Request("https://example.invalid/release")
    with pytest.raises(DiarizationModelError) as error:
        # * The urllib adapter owns redirect scheme validation.
        models._HttpsOnlyRedirectHandler().redirect_request(  # noqa: SLF001
            request, io.BytesIO(), 302, "redirect", HTTPMessage(), target
        )
    assert target in str(error.value)

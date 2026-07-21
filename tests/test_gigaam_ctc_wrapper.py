"""Unit tests for the optional GigaAM CTC wrapper."""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING, Any

import pytest

from core.gigaam_ctc_wrapper import (
    _MISSING_EXTRA_MSG,
    GigaAMCtcWrapper,
    _raise_missing_extra,
)
from core.stt_providers import GIGAAM_HF_REPO_ID, GIGAAM_HF_REVISION

if TYPE_CHECKING:
    from pathlib import Path


def _write_silent_wav(
    path: Path, *, duration_s: float, sample_rate: int = 16000
) -> None:
    """Write a mono 16-bit silent WAV of the requested duration."""
    n_frames = int(duration_s * sample_rate)
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"\x00\x00" * n_frames)


def test_gigaam_rejects_cuda_before_load() -> None:
    """Non-CPU devices are rejected before any model import."""
    with pytest.raises(ValueError, match="CPU only"):
        GigaAMCtcWrapper(device="cuda")


def test_gigaam_auto_resolves_to_cpu() -> None:
    """``auto`` is accepted and forced onto CPU."""
    wrapper = GigaAMCtcWrapper(device="auto")
    assert wrapper.device == "cpu"


def test_gigaam_transcribe_normalizes_text_and_segments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Short-form results become ``{text, segments}`` and unload clears the model."""
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=1.0)
    calls: dict[str, Any] = {}

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            calls["path"] = path
            return "  hello world  "

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(repo_id: str, **kwargs: object) -> _FakeModel:
            calls["repo_id"] = repo_id
            calls["kwargs"] = kwargs
            return _FakeModel()

    def _fake_ensure(self: GigaAMCtcWrapper) -> tuple[Any, Any]:
        return _FakeAutoModel, object()

    monkeypatch.setattr(GigaAMCtcWrapper, "_ensure_deps", _fake_ensure)

    wrapper = GigaAMCtcWrapper(device="cpu")
    emitted: list[dict[str, object]] = []
    result = wrapper.transcribe(audio, on_segment=emitted.append)
    assert result["text"] == "hello world"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "hello world"
    assert emitted[0]["text"] == "hello world"
    assert calls["repo_id"] == GIGAAM_HF_REPO_ID
    assert calls["kwargs"]["revision"] == GIGAAM_HF_REVISION
    assert calls["kwargs"]["trust_remote_code"] is True
    assert wrapper._model is not None  # noqa: SLF001

    wrapper.close()
    assert wrapper._model is None  # noqa: SLF001


def test_gigaam_chunking_for_long_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audio longer than the short-form limit is split into timed segments."""
    audio = tmp_path / "long.wav"
    _write_silent_wav(audio, duration_s=45.0)
    pieces = ["one", "two", "three"]

    class _FakeModel:
        def __init__(self) -> None:
            self.n = 0

        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, _path: str) -> str:
            text = pieces[self.n]
            self.n += 1
            return text

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "one two three"
    assert len(result["segments"]) == 3
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][1]["start"] == 20.0
    assert result["segments"][2]["start"] == 40.0


def test_gigaam_missing_extra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing optional deps surface an install hint for ``.[gigaam]``."""

    def _ensure_raises(_self: GigaAMCtcWrapper) -> tuple[Any, Any]:
        try:
            msg = "No module named 'torch'"
            raise ModuleNotFoundError(msg, name="torch")
        except ImportError as exc:
            _raise_missing_extra(exc)

    monkeypatch.setattr(GigaAMCtcWrapper, "_ensure_deps", _ensure_raises)
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[gigaam\]\"") as ei:
        wrapper._load_model()  # noqa: SLF001
    assert "Missing module: torch" in str(ei.value)
    assert ei.value.__cause__ is not None
    assert getattr(ei.value.__cause__, "name", None) == "torch"


def test_gigaam_from_pretrained_import_error_without_module_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ImportError without ``name`` still gets the install hint (no Missing module)."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            msg = "remote code failed to import a dependency"
            raise ImportError(msg)

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[gigaam\]\"") as ei:
        wrapper._load_model()  # noqa: SLF001
    assert "Missing module:" not in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_gigaam_from_pretrained_remote_code_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote-code ImportError on load maps to the same ``.[gigaam]`` install hint."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            msg = "No module named 'sentencepiece'"
            raise ModuleNotFoundError(msg, name="sentencepiece")

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[gigaam\]\"") as ei:
        wrapper._load_model()  # noqa: SLF001
    msg = str(ei.value)
    assert "Missing module: sentencepiece" in msg
    assert "sentencepiece" in _MISSING_EXTRA_MSG
    assert isinstance(ei.value.__cause__, ModuleNotFoundError)
    assert ei.value.__cause__.name == "sentencepiece"


def test_gigaam_transcribe_remote_code_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy remote-code ImportError on first ``.transcribe`` keeps module context."""
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=1.0)

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, _path: str) -> str:
            msg = "No module named 'pyannote'"
            raise ModuleNotFoundError(msg, name="pyannote")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[gigaam\]\"") as ei:
        wrapper.transcribe(audio)
    assert "Missing module: pyannote" in str(ei.value)
    assert isinstance(ei.value.__cause__, ModuleNotFoundError)
    assert ei.value.__cause__.name == "pyannote"


def test_gigaam_from_pretrained_non_import_errors_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download / runtime failures from ``from_pretrained`` are not remapped."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            msg = "HF hub unreachable"
            raise OSError(msg)

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(OSError, match="HF hub unreachable"):
        wrapper._load_model()  # noqa: SLF001


def test_gigaam_transcribe_non_import_errors_pass_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model runtime errors from ``.transcribe`` are not remapped to install hints."""
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=1.0)

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, _path: str) -> str:
            msg = "cuFFT only supports powers of two"
            raise RuntimeError(msg)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match="cuFFT"):
        wrapper.transcribe(audio)

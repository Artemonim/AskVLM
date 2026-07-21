"""Unit tests for LocalPipeline STT provider dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from core import gigaam_ctc_wrapper as gigaam_mod
from core import pipelines as pl
from core.stt_providers import STT_PROVIDER_GIGAAM_CTC, STT_PROVIDER_WHISPER
from editing.text_model import Document

if TYPE_CHECKING:
    from pathlib import Path


class _StubWhisper:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "text": "whisper text",
            "segments": [{"start": 0.0, "end": 1.0, "text": "whisper text"}],
        }

    def unload(self, *, safe: bool = True) -> None:
        self.calls.append({"unload": safe})


class _StubGigaAM:
    def __init__(self, **_kwargs: object) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "text": "gigaam text",
            "segments": [{"start": 0.0, "end": 1.0, "text": "gigaam text"}],
        }

    def unload(self, *, safe: bool = True) -> None:
        self.calls.append({"unload": safe})


def test_local_pipeline_default_provider_is_whisper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing callers keep Whisper dispatch and Whisper-only ASR kwargs."""
    whisper = _StubWhisper()
    monkeypatch.setattr(pl, "load_env_file", lambda: None)
    monkeypatch.setattr(pl, "configure_ml_caches", lambda root: root)
    monkeypatch.setattr(pl, "get_project_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(pl, "WhisperXWrapper", lambda **_kwargs: whisper)
    monkeypatch.setattr(pl, "prepare_audio", lambda *_a, **_k: tmp_path / "a.wav")
    monkeypatch.setattr(pl, "get_media_duration_seconds", lambda _p: 1.0)
    monkeypatch.setattr(pl, "cleanup_intermediate_audio", lambda *_a, **_k: None)

    pipeline = pl.LocalPipeline(stt_provider=STT_PROVIDER_WHISPER, device="cpu")
    assert pipeline.stt_provider == STT_PROVIDER_WHISPER
    assert pipeline.whisperx is whisper
    assert pipeline.gigaam is None
    doc = pipeline.process(tmp_path / "in.wav", tmp_path / "work")
    assert isinstance(doc, Document)
    assert "whisper text" in doc.get_full_text()
    assert whisper.calls[0].get("beam_size") == 10
    assert whisper.calls[0].get("vad_filter") is True


def test_local_pipeline_dispatches_gigaam_without_whisper_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """GigaAM path does not receive Whisper beam/VAD kwargs."""
    created: list[_StubGigaAM] = []

    def _factory(**kwargs: object) -> _StubGigaAM:
        stub = _StubGigaAM(**kwargs)
        created.append(stub)
        return stub

    monkeypatch.setattr(pl, "load_env_file", lambda: None)
    monkeypatch.setattr(pl, "configure_ml_caches", lambda root: root)
    monkeypatch.setattr(pl, "get_project_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(gigaam_mod, "GigaAMCtcWrapper", _factory)
    monkeypatch.setattr(pl, "prepare_audio", lambda *_a, **_k: tmp_path / "a.wav")
    monkeypatch.setattr(pl, "get_media_duration_seconds", lambda _p: 1.0)
    monkeypatch.setattr(pl, "cleanup_intermediate_audio", lambda *_a, **_k: None)

    pipeline = pl.LocalPipeline(stt_provider=STT_PROVIDER_GIGAAM_CTC, device="auto")
    assert pipeline.whisperx is None
    assert pipeline.device == "cpu"
    assert created
    gigaam = created[0]
    assert pipeline.gigaam is gigaam
    doc = pipeline.process(tmp_path / "in.wav", tmp_path / "work")
    assert "gigaam text" in doc.get_full_text()
    call = gigaam.calls[0]
    assert "beam_size" not in call
    assert "vad_filter" not in call
    assert "word_timestamps" not in call
    assert "compute_type" not in call


def test_local_pipeline_rejects_gigaam_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CUDA is rejected before GigaAM construction."""
    monkeypatch.setattr(pl, "load_env_file", lambda: None)
    monkeypatch.setattr(pl, "configure_ml_caches", lambda root: root)
    monkeypatch.setattr(pl, "get_project_cache_dir", lambda: tmp_path)
    with pytest.raises(ValueError, match="CPU only"):
        pl.LocalPipeline(stt_provider=STT_PROVIDER_GIGAAM_CTC, device="cuda")

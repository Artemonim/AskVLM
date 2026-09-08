"""Unit tests for the optional GigaAM CTC wrapper."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import pytest

from core import gigaam_ctc_wrapper as gigaam_mod
from core.gigaam_ctc_wrapper import (
    _MIN_AUDIO_DURATION_S,
    _MISSING_EXTRA_MSG,
    _WINDOW_S,
    GigaAMCtcWrapper,
    _raise_missing_extra,
)
from core.stt_providers import GIGAAM_HF_REPO_ID, GIGAAM_HF_REVISION


def _write_pcm_silence(
    path: Path,
    *,
    n_frames: int,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write a 16-bit PCM WAV with the given frame count, rate, and channel count."""
    frame = b"\x00\x00" * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(frame * n_frames)


def _write_silent_wav(
    path: Path,
    *,
    duration_s: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> None:
    """Write a 16-bit silent WAV of the requested duration."""
    _write_pcm_silence(
        path,
        n_frames=int(duration_s * sample_rate),
        sample_rate=sample_rate,
        channels=channels,
    )


def _write_zero_rate_wav(path: Path, *, n_frames: int = 16000) -> None:
    """Write a PCM WAV whose header sample rate is 0 (unparseable for streaming)."""
    _write_pcm_silence(path, n_frames=n_frames, sample_rate=16000)
    data = bytearray(path.read_bytes())
    # * PCM WAV sample rate is a little-endian uint32 at offset 24.
    data[24:28] = b"\x00\x00\x00\x00"
    path.write_bytes(data)


def _wav_duration_s(path: str | Path) -> float:
    """Return PCM WAV duration in seconds via stdlib wave."""
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        frames = reader.getnframes()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def _patch_gigaam_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    texts: list[str] | None = None,
    transcribed_paths: list[str] | None = None,
    transcribed_durations: list[float] | None = None,
) -> None:
    """Install a fake GigaAM model that records paths and rejects long windows."""
    pieces = list(texts) if texts is not None else None

    class _FakeModel:
        def __init__(self) -> None:
            self.n = 0

        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            if transcribed_paths is not None:
                transcribed_paths.append(path)
            duration_s = _wav_duration_s(path)
            # * No streaming path may send the model more than one 20 s window.
            assert duration_s <= _WINDOW_S
            # * GigaAM STFT requires more than 320 samples @16 kHz after resample.
            assert duration_s >= _MIN_AUDIO_DURATION_S
            if transcribed_durations is not None:
                transcribed_durations.append(duration_s)
            text = f"chunk-{self.n}" if pieces is None else pieces[self.n]
            self.n += 1
            return text

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper,
        "_ensure_deps",
        lambda _self: (_FakeAutoModel, object()),
    )


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
    """Short-file results become ``{text, segments}`` and unload clears the model."""
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
    assert calls["path"] == str(audio)
    assert calls["repo_id"] == GIGAAM_HF_REPO_ID
    assert calls["kwargs"]["revision"] == GIGAAM_HF_REVISION
    assert calls["kwargs"]["trust_remote_code"] is True
    assert wrapper._model is not None  # noqa: SLF001

    wrapper.close()
    assert wrapper._model is None  # noqa: SLF001


@pytest.mark.parametrize("duration_s", [0.5, 19.9])
def test_gigaam_fast_path_uses_original_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duration_s: float
) -> None:
    """Files shorter than 20 s call the model once on the original path."""
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=duration_s)
    transcribed: list[str] = []
    _patch_gigaam_model(monkeypatch, texts=["hello"], transcribed_paths=transcribed)
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert transcribed == [str(audio)]
    assert result["text"] == "hello"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == pytest.approx(duration_s)
    assert result["segments"][0]["text"] == "hello"


@pytest.mark.parametrize(
    ("duration_s", "expected_starts"),
    [
        (20.0, [0.0]),
        (20.1, [0.0, 20.0]),
        (24.9, [0.0, 20.0]),
        (25.0, [0.0, 20.0]),
        (25.05, [0.0, 20.0]),
        (45.0, [0.0, 20.0, 40.0]),
    ],
)
def test_gigaam_chunked_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    duration_s: float,
    expected_starts: list[float],
) -> None:
    """Files at or above 20 s are split into ≤20 s windows with absolute offsets."""
    audio = tmp_path / "long.wav"
    _write_silent_wav(audio, duration_s=duration_s)
    texts = [f"p{idx}" for idx in range(len(expected_starts))]
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=texts,
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == " ".join(texts)
    assert [seg["start"] for seg in result["segments"]] == expected_starts
    assert result["segments"][-1]["end"] == pytest.approx(duration_s)
    assert [seg["text"] for seg in result["segments"]] == texts
    assert str(audio) not in transcribed
    assert len(transcribed) == len(expected_starts)
    assert len(durations) == len(expected_starts)
    for full_window_s in durations[:-1]:
        assert full_window_s == pytest.approx(_WINDOW_S)
    remainder_s = duration_s - _WINDOW_S * (len(durations) - 1)
    assert durations[-1] == pytest.approx(remainder_s)
    for chunk_path in transcribed:
        assert not Path(chunk_path).exists()


def test_gigaam_progress_uses_header_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Progress fraction is derived from the WAV header when it is available."""
    audio = tmp_path / "short.wav"
    _write_silent_wav(audio, duration_s=1.0)
    _patch_gigaam_model(monkeypatch, texts=["ok"])
    seen: list[tuple[float, str]] = []
    wrapper = GigaAMCtcWrapper(device="cpu")
    wrapper.transcribe(audio, progress=lambda frac, stage: seen.append((frac, stage)))
    assert seen
    assert seen[-1][0] == pytest.approx(1.0)
    assert seen[-1][1] == "transcribe"


def test_gigaam_unreadable_wav_reencodes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``wave.Error`` on the original file triggers one ffmpeg re-encode."""
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"not a wav")
    extract_calls: list[tuple[Path, Path, int, int]] = []

    def _fake_extract(
        src: str | Path,
        dst: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        extract_calls.append((Path(src), Path(dst), sample_rate, channels))
        _write_silent_wav(Path(dst), duration_s=1.0)

    monkeypatch.setattr("core.ffmpeg.extract_audio", _fake_extract)
    transcribed: list[str] = []
    _patch_gigaam_model(monkeypatch, texts=["reencoded"], transcribed_paths=transcribed)
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "reencoded"
    assert len(extract_calls) == 1
    assert extract_calls[0][0] == audio
    assert extract_calls[0][2] == 16000
    assert extract_calls[0][3] == 1
    assert transcribed == [str(extract_calls[0][1])]


def test_gigaam_reencode_still_unreadable_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second parse failure after re-encode becomes a clear RuntimeError."""
    audio = tmp_path / "bad.wav"
    audio.write_bytes(b"not a wav")

    def _fake_extract(
        src: str | Path,
        dst: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        Path(dst).write_bytes(b"still not a wav")

    monkeypatch.setattr("core.ffmpeg.extract_audio", _fake_extract)
    _patch_gigaam_model(monkeypatch, texts=["unused"])
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(
        RuntimeError, match="could not parse WAV after re-encoding"
    ) as ei:
        wrapper.transcribe(audio)
    assert "Too long wav file" not in str(ei.value)


@pytest.mark.parametrize(
    ("sample_rate", "channels"),
    [
        (16000, 2),
        (44100, 1),
        (48000, 1),
    ],
)
@pytest.mark.parametrize(
    ("duration_s", "expected_starts"),
    [
        (1.0, [0.0]),
        (21.0, [0.0, 20.0]),
    ],
)
def test_gigaam_nonstandard_wav_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: int,
    channels: int,
    duration_s: float,
    expected_starts: list[float],
) -> None:
    """Stereo 16 kHz and mono 44.1/48 kHz follow the same 20 s window rule."""
    audio = tmp_path / "nonstd.wav"
    _write_silent_wav(
        audio, duration_s=duration_s, sample_rate=sample_rate, channels=channels
    )
    texts = [f"p{idx}" for idx in range(len(expected_starts))]
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=texts,
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert [seg["start"] for seg in result["segments"]] == expected_starts
    assert result["segments"][-1]["end"] == pytest.approx(duration_s)
    assert len(transcribed) == len(expected_starts)
    if len(expected_starts) > 1:
        assert str(audio) not in transcribed
        assert durations[0] == pytest.approx(_WINDOW_S)
        assert durations[-1] == pytest.approx(duration_s - _WINDOW_S)
    else:
        assert transcribed == [str(audio)]
        assert durations == [pytest.approx(duration_s)]


@pytest.mark.parametrize("n_frames", [0, 1])
def test_gigaam_tiny_wav_fast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_frames: int
) -> None:
    """Empty and 1-frame WAVs are silence-padded before the model call."""
    audio = tmp_path / "tiny.wav"
    _write_pcm_silence(audio, n_frames=n_frames)
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["tiny"],
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert len(transcribed) == 1
    assert transcribed[0] != str(audio)
    assert Path(transcribed[0]).name.startswith("pad_")
    assert not Path(transcribed[0]).exists()
    assert list(tmp_path.glob("pad_*")) == []
    assert durations[0] >= _MIN_AUDIO_DURATION_S
    assert durations[0] == pytest.approx(_MIN_AUDIO_DURATION_S)
    assert result["text"] == "tiny"
    assert len(result["segments"]) == 1
    assert result["segments"][0]["start"] == 0.0
    assert result["segments"][0]["end"] == pytest.approx(n_frames / 16000.0)


def test_gigaam_stft_short_wav_is_padded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 216-sample 16 kHz clip (below n_fft=320) is padded to the duration floor."""
    audio = tmp_path / "stft_short.wav"
    _write_pcm_silence(audio, n_frames=216, sample_rate=16000)
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["ok"],
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "ok"
    assert len(transcribed) == 1
    assert transcribed[0] != str(audio)
    assert durations[0] >= _MIN_AUDIO_DURATION_S
    assert durations[0] == pytest.approx(_MIN_AUDIO_DURATION_S)
    assert not Path(transcribed[0]).exists()
    assert list(tmp_path.glob("pad_*")) == []


def test_gigaam_chunked_short_tail_is_padded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 20.0135 s file pads the ~216-sample tail window, not the first 20 s."""
    audio = tmp_path / "tail.wav"
    _write_pcm_silence(audio, n_frames=320000 + 216, sample_rate=16000)
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["head", ""],
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert len(transcribed) == 2
    assert str(audio) not in transcribed
    assert durations[0] == pytest.approx(_WINDOW_S)
    assert durations[1] >= _MIN_AUDIO_DURATION_S
    assert durations[1] == pytest.approx(_MIN_AUDIO_DURATION_S)
    assert result["text"] == "head"
    assert [seg["text"] for seg in result["segments"]] == ["head", ""]
    leftover = [path for path in tmp_path.iterdir() if path.name.startswith("pad_")]
    assert leftover == []


def test_gigaam_short_8khz_pads_by_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Padding uses duration, not a 16 kHz source-frame count (8 kHz short clip)."""
    audio = tmp_path / "short8k.wav"
    # * 160 frames @8 kHz = 20 ms, below 40 ms; 640 source frames would be 80 ms.
    _write_pcm_silence(audio, n_frames=160, sample_rate=8000)
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["ok"],
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "ok"
    assert len(transcribed) == 1
    assert transcribed[0] != str(audio)
    assert durations[0] >= _MIN_AUDIO_DURATION_S
    assert durations[0] == pytest.approx(_MIN_AUDIO_DURATION_S)
    assert not Path(transcribed[0]).exists()


def test_gigaam_unreadable_wav_skips_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A broken WAV header skips padding and the model still gets the original path."""
    audio = tmp_path / "broken.wav"
    audio.write_bytes(b"not a wav")
    transcribed: list[str] = []

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            transcribed.append(path)
            return "ok"

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with caplog.at_level("DEBUG", logger="core.gigaam_ctc_wrapper"):
        text = wrapper._transcribe_file(audio)  # noqa: SLF001
    assert text == "ok"
    assert transcribed == [str(audio)]
    assert list(tmp_path.glob("pad_*")) == []
    assert any(
        "min-duration check skipped" in record.getMessage() for record in caplog.records
    )


def test_gigaam_padded_temp_cleaned_on_model_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model RuntimeError still unlinks the silence-padded temp WAV."""
    audio = tmp_path / "tiny.wav"
    _write_pcm_silence(audio, n_frames=1)
    transcribed: list[str] = []

    class _FakeModel:
        def eval(self) -> _FakeModel:
            return self

        def transcribe(self, path: str) -> str:
            transcribed.append(path)
            msg = "model failed after open"
            raise RuntimeError(msg)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> _FakeModel:
            return _FakeModel()

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match="model failed after open"):
        wrapper.transcribe(audio)
    assert len(transcribed) == 1
    assert transcribed[0] != str(audio)
    assert Path(transcribed[0]).name.startswith("pad_")
    assert not Path(transcribed[0]).exists()
    assert list(tmp_path.glob("pad_*")) == []


def test_gigaam_threshold_duration_uses_original_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clip exactly at the STFT duration floor is not padded."""
    audio = tmp_path / "threshold.wav"
    _write_pcm_silence(audio, n_frames=640, sample_rate=16000)
    transcribed: list[str] = []
    durations: list[float] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["ok"],
        transcribed_paths=transcribed,
        transcribed_durations=durations,
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "ok"
    assert transcribed == [str(audio)]
    assert durations == [pytest.approx(_MIN_AUDIO_DURATION_S)]
    assert list(tmp_path.glob("pad_*")) == []


def test_gigaam_zero_rate_wav_reencodes_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``getframerate() == 0`` raises ``_UnparseableWavError`` and re-encodes once."""
    audio = tmp_path / "zero_rate.wav"
    _write_zero_rate_wav(audio)
    extract_calls: list[tuple[Path, Path, int, int]] = []

    def _fake_extract(
        src: str | Path,
        dst: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        extract_calls.append((Path(src), Path(dst), sample_rate, channels))
        _write_silent_wav(Path(dst), duration_s=1.0)

    monkeypatch.setattr("core.ffmpeg.extract_audio", _fake_extract)
    transcribed: list[str] = []
    _patch_gigaam_model(monkeypatch, texts=["reencoded"], transcribed_paths=transcribed)
    wrapper = GigaAMCtcWrapper(device="cpu")
    result = wrapper.transcribe(audio)
    assert result["text"] == "reencoded"
    assert len(extract_calls) == 1
    assert extract_calls[0][0] == audio
    assert extract_calls[0][2] == 16000
    assert extract_calls[0][3] == 1
    assert transcribed == [str(extract_calls[0][1])]


def test_gigaam_zero_rate_reencode_still_unparseable_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``_UnparseableWavError`` after re-encode becomes RuntimeError."""
    audio = tmp_path / "zero_rate.wav"
    _write_zero_rate_wav(audio)

    def _fake_extract(
        src: str | Path,
        dst: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> None:
        _write_zero_rate_wav(Path(dst))

    monkeypatch.setattr("core.ffmpeg.extract_audio", _fake_extract)
    _patch_gigaam_model(monkeypatch, texts=["unused"])
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(
        RuntimeError, match="could not parse WAV after re-encoding"
    ) as ei:
        wrapper.transcribe(audio)
    assert "Too long wav file" not in str(ei.value)


def test_gigaam_wrapper_import_does_not_bind_extract_audio() -> None:
    """Importing the wrapper does not require a module-level ffmpeg binding."""
    assert getattr(gigaam_mod, "extract_audio", None) is None


def test_gigaam_cancel_stops_before_third_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``on_segment`` raising on the second window skips the remaining windows."""
    audio = tmp_path / "long.wav"
    _write_silent_wav(audio, duration_s=45.0)
    transcribed: list[str] = []
    _patch_gigaam_model(
        monkeypatch,
        texts=["one", "two", "three"],
        transcribed_paths=transcribed,
    )

    class _CancelError(RuntimeError):
        """Simulates pipeline CancelledError between windows."""

    def _on_segment(_seg: dict[str, Any]) -> None:
        if len(transcribed) >= 2:
            msg = "Canceled"
            raise _CancelError(msg)

    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(_CancelError, match="Canceled"):
        wrapper.transcribe(audio, on_segment=_on_segment)
    assert len(transcribed) == 2


def test_gigaam_missing_extra_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ML deps surface an install hint for ``.[ml]``."""

    def _ensure_raises(_self: GigaAMCtcWrapper) -> tuple[Any, Any]:
        try:
            msg = "No module named 'torch'"
            raise ModuleNotFoundError(msg, name="torch")
        except ImportError as exc:
            _raise_missing_extra(exc)

    monkeypatch.setattr(GigaAMCtcWrapper, "_ensure_deps", _ensure_raises)
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[ml\]\"") as ei:
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
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[ml\]\"") as ei:
        wrapper._load_model()  # noqa: SLF001
    assert "Missing module:" not in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_gigaam_from_pretrained_remote_code_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote-code ImportError on load maps to the same ``.[ml]`` install hint."""

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*_args: object, **_kwargs: object) -> object:
            msg = "No module named 'sentencepiece'"
            raise ModuleNotFoundError(msg, name="sentencepiece")

    monkeypatch.setattr(
        GigaAMCtcWrapper, "_ensure_deps", lambda _self: (_FakeAutoModel, object())
    )
    wrapper = GigaAMCtcWrapper(device="cpu")
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[ml\]\"") as ei:
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
    with pytest.raises(RuntimeError, match=r"pip install -e \"\.\[ml\]\"") as ei:
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

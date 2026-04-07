from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from core.whisperx_wrapper import WhisperXWrapper

if TYPE_CHECKING:
    import pytest


class _FakeSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


def _fake_torch_with_cuda() -> SimpleNamespace:
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))


def test_whisperx_wrapper_falls_back_to_cpu_when_cuda_oom_happens_on_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading should retry on CPU when CUDA memory is exhausted."""
    init_calls: list[tuple[str, str]] = []

    class _FakeModel:
        def __init__(
            self,
            _model_name: str,
            *,
            device: str,
            compute_type: str,
            download_root: str | None = None,
        ) -> None:
            _ = download_root
            init_calls.append((device, compute_type))
            self.device = device
            if device == "cuda":
                msg = "CUDA out of memory"
                raise RuntimeError(msg)

        def transcribe(
            self, _audio_path: str, language: str | None = None, **kwargs: object
        ) -> tuple[list[_FakeSegment], dict[str, object]]:
            _ = language, kwargs
            return ([_FakeSegment(0.0, 1.0, "hello")], {})

    monkeypatch.setattr("core.whisperx_wrapper.fw_whisper_cls", _FakeModel)
    monkeypatch.setattr("core.whisperx_wrapper.torch_mod", _fake_torch_with_cuda())

    wrapper = WhisperXWrapper(model_name="small", device="cuda", compute_type="auto")
    result = wrapper.transcribe(Path("clip.wav"))

    assert result["text"] == "hello"
    assert init_calls == [("cuda", "float16"), ("cpu", "int8")]
    assert wrapper._active_device == "cpu"  # noqa: SLF001


def test_whisperx_wrapper_falls_back_to_cpu_when_cuda_oom_happens_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inference should retry on CPU when CUDA runs out of memory."""
    init_calls: list[tuple[str, str]] = []
    transcribe_calls: list[str] = []

    class _FakeModel:
        def __init__(
            self,
            _model_name: str,
            *,
            device: str,
            compute_type: str,
            download_root: str | None = None,
        ) -> None:
            _ = download_root
            init_calls.append((device, compute_type))
            self.device = device

        def transcribe(
            self, _audio_path: str, language: str | None = None, **kwargs: object
        ) -> tuple[list[_FakeSegment], dict[str, object]]:
            _ = language, kwargs
            transcribe_calls.append(self.device)
            if self.device == "cuda":
                msg = "CUDA out of memory"
                raise RuntimeError(msg)
            return ([_FakeSegment(0.0, 1.0, "cpu retry works")], {})

    monkeypatch.setattr("core.whisperx_wrapper.fw_whisper_cls", _FakeModel)
    monkeypatch.setattr("core.whisperx_wrapper.torch_mod", _fake_torch_with_cuda())

    wrapper = WhisperXWrapper(model_name="small", device="cuda", compute_type="auto")
    result = wrapper.transcribe(Path("clip.wav"))

    assert result["text"] == "cpu retry works"
    assert init_calls == [("cuda", "float16"), ("cpu", "int8")]
    assert transcribe_calls == ["cuda", "cpu"]
    assert wrapper._active_device == "cpu"  # noqa: SLF001

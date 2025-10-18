from pathlib import Path

from core.diarization import DiarizationPipeline
from core.llm_formatter import LLMFormatter
from core.whisperx_wrapper import WhisperXWrapper


def test_diarization_returns_list_without_pyannote() -> None:
    d = DiarizationPipeline(hf_token="")
    out = d.diarize(str(Path("missing.wav")))
    assert isinstance(out, list)


def test_llm_formatter_identity_when_no_model() -> None:
    fmt = LLMFormatter(model_path=None)
    text = "hello world"
    assert isinstance(fmt.format_text(text), str)


def test_whisperx_align_fallback_without_whisperx() -> None:
    wx = WhisperXWrapper(model_name="tiny", device="cpu", compute_type="auto")
    aligned = wx.align(Path("missing.wav"), {"segments": []}, language=None)
    assert isinstance(aligned, list)


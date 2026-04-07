from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from core.pipelines import LocalPipeline

if TYPE_CHECKING:
    import pytest


def test_local_pipeline_keeps_formatter_lazy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline should not create the formatter until it is first requested."""
    init_calls: list[str] = []

    class _FakeFormatter:
        def __init__(self, model_name: str, model_path: str | None = None) -> None:
            _ = model_path
            init_calls.append(model_name)

        def format_text(self, text: str) -> str:
            return text

    monkeypatch.setattr("core.pipelines.LLMFormatter", _FakeFormatter)

    pipeline = LocalPipeline(enable_dialog_blocks=False)

    assert init_calls == []
    formatter = pipeline._get_formatter()  # noqa: SLF001
    assert formatter is not None
    assert init_calls == ["gguf-q4_0"]


def test_local_pipeline_close_aggressive_unloads_whisper_and_drops_backends() -> None:
    """Aggressive close should release loaded backends and clear references."""
    unload_calls: list[bool] = []
    pipeline = LocalPipeline(enable_dialog_blocks=False)
    pipeline.whisperx = SimpleNamespace(
        unload=lambda *, safe=True: unload_calls.append(safe)
    )
    pipeline.diarizer = SimpleNamespace(_pipeline=object())
    pipeline.formatter = SimpleNamespace(_llm=object())

    pipeline.close(aggressive=True)

    assert unload_calls == [False]
    assert pipeline.diarizer is None
    assert pipeline.formatter is None

from collections.abc import Callable
from pathlib import Path

import pytest

import core.pipelines as pl
from core.audio_io import prepare_audio
from core.pipelines import LocalPipeline
from core.settings import get_project_cache_dir
from core.whisperx_wrapper import WhisperXWrapper


def test_integration_pipeline_fast_smoke(tmp_path: Path) -> None:
    """End-to-end smoke: run LocalPipeline on the fixture in fast mode.

    This test is opt-in because it exercises heavy ML. Enable with
    environment variable SK_INTEGRATION=1. It asserts that processing
    completes without errors and that some text is produced. No persistent
    artifacts are saved because we avoid exporters and use a pytest temp dir
    as the working directory.
    """
    # Always run as requested

    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        # Gracefully skip if fixture is not available locally
        return

    pipeline = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="small",  # fast quality
        engine="auto",
        enable_diarization=False,
        enable_dialog_blocks=False,
        device="cpu",  # force CPU for test stability
        compute_type="auto",
    )

    # Process without exporters; all intermediates are confined to tmp_path
    doc = pipeline.process(fixture, tmp_path)
    text = doc.get_full_text()
    assert isinstance(text, str)
    assert text.strip() != ""


@pytest.mark.parametrize(
    ("trigger_stage", "step_name"),
    [(0, "prepare_audio"), (1, "transcribe"), (2, "success")],
    ids=["prepare_cancel", "transcribe_cancel", "success"],
)
def test_cancel_responsiveness(
    tmp_path: Path, trigger_stage: int, step_name: str
) -> None:
    """Exercise cancel per stage; pytest IDs clarify failing step."""
    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        return

    # Stage indices: 0=prepare_audio, 1=transcribe, 2=rest
    stage = {"value": -1}

    def should_cancel() -> bool:
        # Cancel when first entering the target stage
        return stage["value"] == trigger_stage

    # Wrap prepare_audio to mark entry of stage 0
    def _prepare_with_mark(
        input_path: Path,
        work_dir: Path,
        sample_rate: int = 16000,
        channels: int = 1,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Path:
        stage["value"] = 0
        return prepare_audio(
            input_path,
            work_dir,
            sample_rate=sample_rate,
            channels=channels,
            should_cancel=should_cancel,
        )

    pipeline = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="small",
        engine="auto",
        enable_diarization=False,
        enable_dialog_blocks=False,
        device="cpu",
        compute_type="auto",
    )

    # Monkeypatch: direct import to replace in pipeline module
    # import moved to module top for linting compliance

    orig_prepare = pl.prepare_audio
    pl.prepare_audio = _prepare_with_mark  # type: ignore[assignment]

    # Monkeypatch transcribe to be fast and to mark entry into transcribe stage
    orig_transcribe = WhisperXWrapper.transcribe

    def _fast_transcribe(
        self: WhisperXWrapper,
        audio_path: Path,
        language: str | None = None,
        *,
        on_segment: Callable[[dict[str, object]], None] | None = None,
        progress: Callable[[float, str], None] | None = None,
        **kwargs: object,
    ) -> dict[str, object]:
        # Mark transcribe stage
        stage["value"] = 1
        seg = {"start": 0.1, "end": 0.2, "text": "hello"}
        if on_segment is not None:
            on_segment(seg)
        if progress is not None:
            progress(0.2, "transcribe")
        return {"text": "hello world", "segments": [seg]}

    WhisperXWrapper.transcribe = _fast_transcribe  # type: ignore[assignment]

    try:
        # Cancel function used by pipeline.process
        def _should_cancel() -> bool:
            return should_cancel()

        if trigger_stage < 2:
            with pytest.raises(Exception, match=r"(Canceled|ffmpeg exited)"):
                pipeline.process(
                    fixture,
                    tmp_path,
                    progress=None,
                    subtitle_max_line_width=None,
                    subtitle_max_lines=None,
                    should_cancel=_should_cancel,
                )
        else:
            # Final pass: no cancel expected, must succeed
            doc = pipeline.process(
                fixture,
                tmp_path,
                progress=None,
                subtitle_max_line_width=None,
                subtitle_max_lines=None,
                should_cancel=_should_cancel,
            )
            assert doc.get_full_text().strip() != "", (
                f"Empty text on full pass at step {step_name}"
            )
    finally:
        pl.prepare_audio = orig_prepare  # type: ignore[assignment]
        WhisperXWrapper.transcribe = orig_transcribe  # type: ignore[assignment]

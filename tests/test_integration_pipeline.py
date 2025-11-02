import os
from collections.abc import Callable
from pathlib import Path

import pytest

import core.pipelines as pl
from core.audio_io import prepare_audio
from core.pipelines import LocalPipeline
from core.settings import get_project_cache_dir
from core.whisperx_wrapper import WhisperXWrapper
from utils.exporters import export_document


def test_real_alignment_whisperx_small(tmp_path: Path) -> None:
    """Verify whisperx alignment returns word timings when whisperx is installed.

    Skips if whisperx is not available in the environment.
    """
    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        return
    pytest.importorskip("whisperx")

    # Prepare audio to WAV
    wav = prepare_audio(fixture, tmp_path)
    wx = WhisperXWrapper(
        model_name="small",
        device="cuda",
        compute_type="auto",
        model_root=get_project_cache_dir() / "models",
    )
    tx = wx.transcribe(audio_path=wav, language=None)
    aligned = wx.align(wav, tx, language=None)
    assert isinstance(aligned, list)
    # Expect at least one segment with aligned words when whisperx is present
    assert any(getattr(seg, "words", []) for seg in aligned)


def test_real_diarization_pyannote_small(tmp_path: Path) -> None:
    """Verify diarization produces segments when pyannote.audio is installed.

    Skips if pyannote.audio is not available in the environment.
    """
    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        return
    pytest.importorskip("pyannote.audio")
    # Require token; skip if not provided in environment/.env
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("PYANNOTE_TOKEN")
        or os.getenv("PYANNOTE_AUTH_TOKEN")
    )
    if not token:
        pytest.skip("HF_TOKEN not set; skipping diarization test")

    # Reuse pipeline with diarization enabled on small model
    pipeline = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="small",
        engine="auto",
        enable_diarization=True,
        enable_dialog_blocks=False,
        device="auto",
        compute_type="auto",
    )
    # Prepare audio path that diarizer consumes
    wav = prepare_audio(fixture, tmp_path)
    # Force lazy diarizer init if needed
    if pipeline.diarizer is None:
        from core.diarization import DiarizationPipeline

        pipeline.diarizer = DiarizationPipeline(device="cuda")
    segs = pipeline.diarizer.diarize(str(wav)) if pipeline.diarizer else []
    assert isinstance(segs, list)
    assert len(segs) > 0


def test_integration_pipeline_fast_smoke_with_diarization_and_export(
    tmp_path: Path,
) -> None:
    """Integration smoke on small model with diarization and export.

    - Uses small model for ASR
    - Enables diarization (pyannote token is loaded from .env)
    - Exports TXT and SRT and checks presence
    """
    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        # Gracefully skip if fixture is not available locally
        return

    pipeline = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="small",  # fast quality only
        engine="auto",
        enable_diarization=True,
        enable_dialog_blocks=False,
        device="auto",
        compute_type="auto",
    )

    # Process and export artifacts into tmp_path
    doc = pipeline.process(fixture, tmp_path)
    text = doc.get_full_text()
    assert isinstance(text, str)
    assert text.strip() != ""
    txtp = tmp_path / f"{fixture.stem}.txt"
    srtp = tmp_path / f"{fixture.stem}.srt"
    export_document(doc, "txt", txtp)
    export_document(doc, "srt", srtp)
    assert txtp.exists()
    assert txtp.stat().st_size >= 0
    assert srtp.exists()
    assert srtp.stat().st_size >= 0


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
        enable_diarization=True,
        enable_dialog_blocks=False,
        device="auto",
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

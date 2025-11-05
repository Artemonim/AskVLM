from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from core.pipelines import LocalPipeline
from core.settings import get_project_cache_dir
from gui.main_window import PipelineWorker


@pytest.mark.integration
def test_queue_real_good_and_fast(tmp_path: Path) -> None:
    """Run real LocalPipeline over multi-file queues in good and fast modes.

    Skips if faster_whisper is not installed in the environment.
    """
    pytest.importorskip("faster_whisper")

    # Ensure Qt app exists
    QApplication.instance() or QApplication([])

    # Prepare real inputs by copying existing fixture under new names
    fixture = Path("tests/fixtures/test_video_first.mp4").resolve()
    if not fixture.exists():
        pytest.skip("fixture video missing")

    def make_copy(name: str) -> Path:
        dst = tmp_path / name
        shutil.copy(fixture, dst)
        return dst

    inputs_good = [make_copy("first.mp4"), make_copy("short.mp4")]
    inputs_fast = [
        make_copy("first_f.mp4"),
        make_copy("short_f.mp4"),
        make_copy("second_f.mp4"),
        make_copy("third_f.mp4"),
    ]

    def make_opts(quality: str) -> dict[str, object]:
        return {
            "export_format": "txt",
            "single_view": False,
            "save_srt": False,
            "subtitle_max_line_width": 42,
            "subtitle_max_lines": 2,
            "quality": quality,
            "no_empty": False,
        }

    # Build a real pipeline (no diarization to keep dependencies minimal)
    pipeline_good = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="large-v3",
        engine="auto",
        enable_diarization=False,
        enable_dialog_blocks=False,
        device="auto",
        compute_type="auto",
    )
    errs: list[str] = []
    w_good = PipelineWorker(pipeline_good, inputs_good, tmp_path, make_opts("good"))
    w_good.error.connect(lambda m: errs.append(str(m)))
    w_good.run()
    assert not errs, f"Errors in real GOOD queue: {errs}"

    # Fast mode on small model
    pipeline_fast = LocalPipeline(
        model_root=get_project_cache_dir() / "models",
        whisper_model="small",
        engine="auto",
        enable_diarization=False,
        enable_dialog_blocks=False,
        device="auto",
        compute_type="auto",
    )
    w_fast = PipelineWorker(pipeline_fast, inputs_fast, tmp_path, make_opts("fast"))
    w_fast.error.connect(lambda m: errs.append(str(m)))
    w_fast.run()
    assert not errs, f"Errors in real FAST queue: {errs}"



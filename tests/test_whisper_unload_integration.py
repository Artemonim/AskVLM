"""Heavy integration probe for aggressive Whisper unload isolation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from core.ffmpeg import get_media_duration_seconds

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_SHORT_MP4 = (
    Path(__file__).resolve().parent / "fixtures" / "test_video_short.mp4"
)
_CHILD_HELPER = Path(__file__).resolve().parent / "whisper_unload_child.py"
_CHILD_TIMEOUT_S = 20 * 60


def _require_aggressive_unload_prereqs() -> None:
    if not _FIXTURE_SHORT_MP4.is_file():
        pytest.skip(f"Fixture not found: {_FIXTURE_SHORT_MP4}")
    if not _CHILD_HELPER.is_file():
        pytest.skip(f"Child helper not found: {_CHILD_HELPER}")
    pytest.importorskip("faster_whisper")
    torch = pytest.importorskip("torch")
    if getattr(torch, "cuda", None) is None or not torch.cuda.is_available():
        pytest.skip("Aggressive Whisper unload probe requires CUDA.")
    try:
        duration_s = float(get_media_duration_seconds(_FIXTURE_SHORT_MP4))
    except OSError:
        pytest.skip(
            "Aggressive Whisper unload probe requires ffmpeg/ffprobe "
            "for the short fixture."
        )
    if duration_s <= 0.0:
        pytest.skip(
            "Aggressive Whisper unload probe requires a non-zero fixture duration."
        )


def _format_child_output(stream: str | None) -> str:
    return stream if stream else "<empty>"


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.heavy_ml
@pytest.mark.xdist_group(name="ml_singleton")
def test_aggressive_whisper_unload_in_child_process(tmp_path: Path) -> None:
    """Run real Whisper transcribe + aggressive unload in a child Python process."""
    _require_aggressive_unload_prereqs()
    child_work_dir = tmp_path / "child_work"
    command = [
        sys.executable,
        str(_CHILD_HELPER),
        str(_FIXTURE_SHORT_MP4),
        str(child_work_dir),
    ]
    try:
        result = subprocess.run(  # noqa: S603
            command,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=_CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            "Aggressive Whisper unload child process timed out.\n"
            f"stdout:\n{_format_child_output(exc.stdout)}\n"
            f"stderr:\n{_format_child_output(exc.stderr)}"
        )
    if result.returncode != 0:
        pytest.fail(
            "Aggressive Whisper unload child process failed.\n"
            f"exit_code={result.returncode}\n"
            f"stdout:\n{_format_child_output(result.stdout)}\n"
            f"stderr:\n{_format_child_output(result.stderr)}"
        )

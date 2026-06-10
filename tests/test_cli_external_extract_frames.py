"""Tests for the external-extract-frames CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli import app

runner = CliRunner()


def _fake_duration(path: object, **_: object) -> float:
    return 60.0  # 60 seconds


def _fake_extract(
    video_file: object,
    start_s: float,
    end_s: float,
    output_pattern: Path,
    *,
    fps: float = 2.0,
) -> tuple[Path, ...]:
    # Create fake frame files matching the pattern
    out_dir = Path(output_pattern).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    t = 0.0
    idx = 1
    while t < (end_s - start_s):
        p = out_dir / f"frame-{idx:06d}.png"
        p.write_bytes(b"fake")
        frames.append(p)
        t += 1.0 / fps
        idx += 1
    return tuple(frames)


def test_external_extract_frames_uses_target_fps_when_under_budget(
    tmp_path: Path,
) -> None:
    """Target FPS is used when estimated frames <= frame_budget."""
    out_dir = tmp_path / "frames"
    # Create a stub video file so typer's exists=True check passes
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"stub")
    # 10s * 0.5 FPS = 5 frames → under budget of 20
    with (
        patch("core.ffmpeg.get_media_duration_seconds", return_value=10.0),
        patch("core.ffmpeg.extract_frames_for_span", wraps=_fake_extract) as mock_ex,
    ):
        result = runner.invoke(
            app,
            [
                "external-extract-frames",
                str(video_file),
                "--output-dir",
                str(out_dir),
                "--fps",
                "0.5",
                "--fps-fallback",
                "0.2",
                "--frame-budget",
                "20",
            ],
        )
    assert result.exit_code == 0, result.output
    # 10s * 0.5 FPS = 5 frames ≤ 20 → target FPS used
    call_kwargs = mock_ex.call_args
    assert abs(call_kwargs.kwargs["fps"] - 0.5) < 1e-6


def test_external_extract_frames_falls_back_when_over_budget(tmp_path: Path) -> None:
    """Fallback FPS is used when estimated frames > frame_budget."""
    out_dir = tmp_path / "frames"
    # Create a stub video file so typer's exists=True check passes
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"stub")
    with (
        patch("core.ffmpeg.get_media_duration_seconds", return_value=60.0),
        patch("core.ffmpeg.extract_frames_for_span", wraps=_fake_extract) as mock_ex,
    ):
        result = runner.invoke(
            app,
            [
                "external-extract-frames",
                str(video_file),
                "--output-dir",
                str(out_dir),
                "--fps",
                "0.5",
                "--fps-fallback",
                "0.2",
                "--frame-budget",
                "20",
            ],
        )
    assert result.exit_code == 0, result.output
    # 60s * 0.5 FPS = 30 frames > 20 → fallback 0.2 FPS used
    call_kwargs = mock_ex.call_args
    assert abs(call_kwargs.kwargs["fps"] - 0.2) < 1e-6


def test_external_extract_frames_json_output(tmp_path: Path) -> None:
    """--json flag produces parseable JSON with expected keys."""
    out_dir = tmp_path / "frames"
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"stub")
    with (
        patch("core.ffmpeg.get_media_duration_seconds", return_value=4.0),
        patch("core.ffmpeg.extract_frames_for_span", wraps=_fake_extract),
    ):
        result = runner.invoke(
            app,
            [
                "external-extract-frames",
                str(video_file),
                "--output-dir",
                str(out_dir),
                "--fps",
                "0.5",
                "--fps-fallback",
                "0.2",
                "--frame-budget",
                "20",
                "--json",
            ],
        )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output.strip())
    assert "frames" in data
    assert "fps_used" in data
    assert "duration_s" in data
    assert data["duration_s"] == pytest.approx(4.0)


def test_external_extract_frames_zero_duration_exits_clean(tmp_path: Path) -> None:
    """Zero/unknown duration exits cleanly with exit code 0 and no frame output."""
    out_dir = tmp_path / "frames"
    video_file = tmp_path / "video.mp4"
    video_file.write_bytes(b"stub")
    with patch("core.ffmpeg.get_media_duration_seconds", return_value=0.0):
        result = runner.invoke(
            app,
            [
                "external-extract-frames",
                str(video_file),
                "--output-dir",
                str(out_dir),
                "--fps",
                "0.5",
                "--fps-fallback",
                "0.2",
                "--frame-budget",
                "20",
                "--json",
            ],
        )
    assert result.exit_code == 0
    # * CliRunner merges stderr into output; find the JSON line by prefix
    json_line = next(
        (line for line in result.output.splitlines() if line.startswith("{")),
        None,
    )
    assert json_line is not None, f"No JSON line in output: {result.output!r}"
    data = json.loads(json_line)
    assert data["frames"] == []

from pathlib import Path

import pytest

from core import ffmpeg as ffm


def test_extract_frame_to_file_success(
    short_audio_fixture: Path, tmp_path: Path
) -> None:
    """extract_frame_to_file extracts a real frame from video fixture."""
    output_img = tmp_path / "frame.jpg"
    # Extract at 1.0 second
    res = ffm.extract_frame_to_file(short_audio_fixture, 1.0, output_img)

    assert res == output_img
    assert output_img.exists()
    assert output_img.stat().st_size > 0


def test_burn_subtitles_generates_file(
    short_audio_fixture: Path, tmp_path: Path
) -> None:
    """burn_subtitles runs ffmpeg and produces an output video."""
    # Create dummy SRT
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text(
        "1\n00:00:01,000 --> 00:00:02,000\nHello Test\n\n", encoding="utf-8"
    )

    output_video = tmp_path / "burned.mp4"

    # Run burn (synchronous wrapper)
    # We mock run_async inside start_burn_process or call burn_subtitles directly?
    # burn_subtitles uses .run(), so it blocks. That's fine for a small clip.

    # Force "autoscale=False" to simplify logic, or let it probe.
    # Since short_audio_fixture is real media, probing should work.

    ffm.burn_subtitles(
        video_file=short_audio_fixture,
        subtitle_file=srt_path,
        output_file=output_video,
        autoscale=True,
        normalize_audio=False,  # Disable normalization to speed up
    )

    assert output_video.exists()
    assert output_video.stat().st_size > 0


def test_extract_frames_for_span_returns_multiple_files(
    short_audio_fixture: Path, tmp_path: Path
) -> None:
    """extract_frames_for_span writes multiple sampled frames for a chunk."""
    outputs = ffm.extract_frames_for_span(
        short_audio_fixture,
        1.0,
        3.0,
        tmp_path / "chunk-%03d.png",
        fps=2.0,
    )

    assert len(outputs) >= 3
    assert all(path.exists() for path in outputs)
    assert all(path.suffix == ".png" for path in outputs)


def test_frame_extract_filtergraph_appends_colorspace_fix() -> None:
    """The filtergraph builder chains the colorspace fix after the fps filter."""
    build = ffm._frame_extract_filtergraph  # noqa: SLF001
    assert build(0.5, "") == "fps=0.5"
    assert build(0.5, "format=yuv420p") == "fps=0.5,format=yuv420p"


def test_extract_frames_for_span_recovers_from_invalid_color_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first 'Invalid color space' failure is recovered by the next strategy."""
    out_pattern = tmp_path / "chunk-%03d.png"
    used_filtergraphs: list[str] = []

    def _fake_run(
        _video_file: Path,
        _start_s: float,
        _duration_s: float,
        output_pattern: Path,
        filtergraph: str,
    ) -> None:
        used_filtergraphs.append(filtergraph)
        if len(used_filtergraphs) == 1:
            msg = "ffmpeg"
            raise ffm.ffmpeg.Error(msg, b"", b"[swscaler] Invalid color space")
        # * The colorspace-normalized retry succeeds and writes a frame.
        out_dir = Path(output_pattern).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "chunk-001.png").write_bytes(b"frame")

    monkeypatch.setattr(ffm, "_run_frame_extraction_once", _fake_run)

    frames = ffm.extract_frames_for_span(
        tmp_path / "video.mp4", 0.0, 2.0, out_pattern, fps=2.0
    )

    assert [p.name for p in frames] == ["chunk-001.png"]
    # * Plain attempt first, then at least one colorspace-fixing strategy.
    assert len(used_filtergraphs) >= 2
    assert used_filtergraphs[0] == "fps=2.0"
    assert "colorspace" in used_filtergraphs[1]


def test_extract_frames_for_span_raises_after_all_strategies_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When every colorspace strategy fails, the last ffmpeg error propagates."""
    out_pattern = tmp_path / "chunk-%03d.png"
    attempts: list[str] = []

    def _always_fail(
        _video_file: Path,
        _start_s: float,
        _duration_s: float,
        _output_pattern: Path,
        filtergraph: str,
    ) -> None:
        attempts.append(filtergraph)
        msg = "ffmpeg"
        raise ffm.ffmpeg.Error(msg, b"", b"Invalid color space")

    monkeypatch.setattr(ffm, "_run_frame_extraction_once", _always_fail)

    with pytest.raises(ffm.ffmpeg.Error):
        ffm.extract_frames_for_span(
            tmp_path / "video.mp4", 0.0, 2.0, out_pattern, fps=1.0
        )

    assert len(attempts) == len(ffm._FRAME_EXTRACT_VF_FALLBACKS)  # noqa: SLF001


def test_get_media_duration_real(short_audio_fixture: Path) -> None:
    """get_media_duration_seconds returns approx 16s for the short fixture."""
    dur = ffm.get_media_duration_seconds(short_audio_fixture)
    # The committed short fixture is about 16 seconds long.
    assert 15.0 <= dur <= 17.5


def test_start_burn_process_returns_popen(
    short_audio_fixture: Path, tmp_path: Path
) -> None:
    """start_burn_process returns a running Popen object."""
    srt_path = tmp_path / "subs.srt"
    srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nA\n\n", encoding="utf-8")
    out = tmp_path / "out.mp4"

    proc = ffm.start_burn_process(
        video_file=short_audio_fixture,
        subtitle_file=srt_path,
        output_file=out,
        normalize_audio=False,
    )

    try:
        assert proc.poll() is None  # Running
    finally:
        proc.kill()

from pathlib import Path

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


def test_get_media_duration_real(short_audio_fixture: Path) -> None:
    """get_media_duration_seconds returns approx 10s for the fixture."""
    dur = ffm.get_media_duration_seconds(short_audio_fixture)
    # The fixture is created with -t 10, so it should be close to 10s
    assert 9.0 <= dur <= 11.0


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

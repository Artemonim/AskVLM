import io
import os
import subprocess
import sys
import wave
from array import array
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core import audio_io
from core.stt_pool.engine_api import WorkItem
from tests.bounded_scenario_support import run_bounded_scenario
from tests.stt_pool_engine_support import EngineRuntime, spec
from utils import env as env_utils


def _sample_wave(path: Path, frames: int = 32000) -> bytes:
    """Write independently known PCM frames for real FFmpeg extraction."""
    samples = array("h", (index % 20000 - 10000 for index in range(frames)))
    if sys.byteorder != "little":
        samples.byteswap()
    pcm = samples.tobytes()
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        writer.writeframes(pcm)
    return pcm


@pytest.mark.integration
@pytest.mark.parametrize("cancellable", [False, True])
@pytest.mark.parametrize(
    ("start", "duration", "first_frame", "frames", "expected_args"),
    [
        (None, None, 0, 32000, (None, None)),
        (0.25, None, 4000, 28000, ("0.250", None)),
        (None, 0.5, 0, 8000, (None, "0.500")),
        (0.25, 0.5, 4000, 8000, ("0.250", "0.500")),
        (1.75, 1.0, 28000, 4000, ("1.750", "1.000")),
    ],
)
def test_real_audio_windows_preserve_exact_pcm_and_seek_before_input(
    tmp_path: Path,
    start: float | None,
    duration: float | None,
    first_frame: int,
    frames: int,
    expected_args: tuple[str | None, str | None],
    *,
    cancellable: bool,
) -> None:
    """Catch windows being ignored, output-side seeking or different cropping in cancellable extraction."""
    run_bounded_scenario(
        partial(
            _audio_window_scenario,
            start=start,
            duration=duration,
            first_frame=first_frame,
            frames=frames,
            expected_args=expected_args,
            cancellable=cancellable,
        ),
        tmp_path,
    )


def _audio_window_scenario(
    tmp_path: Path,
    *,
    start: float | None,
    duration: float | None,
    first_frame: int,
    frames: int,
    expected_args: tuple[str | None, str | None],
    cancellable: bool,
) -> None:
    """Keep real FFmpeg assertions inside the scenario's process deadline."""
    with pytest.MonkeyPatch.context() as patch:
        _check_audio_window(
            tmp_path,
            patch,
            start,
            duration,
            first_frame,
            frames,
            expected_args,
            cancellable=cancellable,
        )


def _check_audio_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: float | None,
    duration: float | None,
    first_frame: int,
    frames: int,
    expected_args: tuple[str | None, str | None],
    *,
    cancellable: bool,
) -> None:
    """Compare decoded PCM and process arguments to independently known windows."""
    source = tmp_path / "source.wav"
    pcm = _sample_wave(source)
    commands: list[list[str]] = []
    real_popen = subprocess.Popen

    def popen(argv: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:  # noqa: ANN401 - real subprocess boundary
        commands.append(argv)
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(audio_io.time, "sleep", lambda _seconds: None)
    prepared = audio_io.prepare_audio(
        source,
        tmp_path / "work",
        should_cancel=(lambda: False) if cancellable else None,
        start_s=start,
        duration_s=duration,
    )
    with wave.open(str(prepared), "rb") as reader:
        assert reader.getnframes() == frames
        assert (
            reader.readframes(frames)
            == pcm[2 * first_frame : 2 * (first_frame + frames)]
        )
    command = commands[0]
    if start is None:
        assert "-ss" not in command
    else:
        assert command.index("-ss") < command.index("-i")
        assert command[command.index("-ss") + 1] == expected_args[0]
    if duration is None:
        assert "-t" not in command
    else:
        assert command.index("-t") > command.index("-i")
        assert command[command.index("-t") + 1] == expected_args[1]


@pytest.mark.integration
@pytest.mark.parametrize("cancellable", [False, True])
def test_zero_frame_window_fails_and_removes_its_wav(
    tmp_path: Path, *, cancellable: bool
) -> None:
    """Catch an out-of-media window being mistaken for a successful empty transcript."""
    run_bounded_scenario(
        partial(_zero_frame_window_scenario, cancellable=cancellable), tmp_path
    )


def _zero_frame_window_scenario(tmp_path: Path, *, cancellable: bool) -> None:
    """Bound real decoding and cleanup of an out-of-media window."""
    with pytest.MonkeyPatch.context() as patch:
        _check_zero_frame_window(tmp_path, patch, cancellable=cancellable)


def _check_zero_frame_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cancellable: bool
) -> None:
    """Assert the input/window facts and absence of an unusable output WAV."""
    source = tmp_path / "source.wav"
    _sample_wave(source)
    monkeypatch.setattr(audio_io.time, "sleep", lambda _seconds: None)
    with pytest.raises(ValueError) as error:  # noqa: PT011 - inspect source/window facts, not prose
        audio_io.prepare_audio(
            source,
            tmp_path / "work",
            should_cancel=(lambda: False) if cancellable else None,
            start_s=3.0,
        )
    assert str(source) in str(error.value)
    assert "3" in str(error.value)
    assert not (tmp_path / "work" / "_work" / "source.wav").exists()


@pytest.mark.integration
@pytest.mark.parametrize("cancellable", [False, True])
def test_no_window_preserves_legacy_zero_frame_audio_behavior(
    tmp_path: Path, *, cancellable: bool
) -> None:
    """Catch the window-only no-audio check changing legacy extraction for an empty WAV."""
    run_bounded_scenario(
        partial(_legacy_zero_frame_scenario, cancellable=cancellable), tmp_path
    )


def _legacy_zero_frame_scenario(tmp_path: Path, *, cancellable: bool) -> None:
    """Bound real decoding of a legacy empty WAV."""
    with pytest.MonkeyPatch.context() as patch:
        _check_legacy_zero_frame_audio(tmp_path, patch, cancellable=cancellable)


def _check_legacy_zero_frame_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cancellable: bool
) -> None:
    """Assert the unchanged empty-output contract without a media window."""
    source = tmp_path / "empty.wav"
    _sample_wave(source, frames=0)
    monkeypatch.setattr(audio_io.time, "sleep", lambda _seconds: None)
    prepared = audio_io.prepare_audio(
        source,
        tmp_path / "work",
        should_cancel=(lambda: False) if cancellable else None,
    )
    with wave.open(str(prepared), "rb") as reader:
        assert reader.getnframes() == 0


def test_load_env_file_sets_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_env_file reads KEY=VALUE pairs and sets os.environ if absent."""
    content = "# comment\nA=1\nB='2'\nC=\"3\"\nNOEQUALS"
    env_path = tmp_path / ".env"
    env_path.write_text(content, encoding="utf-8")
    # Ensure no pre-existing vars
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)
    env_utils.load_env_file(env_path)

    assert os.environ.get("A") == "1"
    assert os.environ.get("B") == "2"
    assert os.environ.get("C") == "3"


class _AsyncFfmpeg:
    """Substitute only ffmpeg-python's external process chain and its exit."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        self.stderr = io.StringIO("progress=1\n")
        self.output_path: Path | None = None

    def input(self, *_args: object, **_kwargs: object) -> "_AsyncFfmpeg":
        """Accept the media path without decoding a real input."""
        return self

    def output(self, path: str, **_kwargs: object) -> "_AsyncFfmpeg":
        """Remember where the real caller requests its intermediate WAV."""
        self.output_path = Path(path)
        return self

    def overwrite_output(self) -> "_AsyncFfmpeg":
        """Keep the external builder chain intact."""
        return self

    def global_args(self, *_args: object) -> "_AsyncFfmpeg":
        """Accept the external progress arguments."""
        return self

    def run_async(self, **_kwargs: object) -> "_AsyncFfmpeg":
        """Leave a partial file like a failed external decoder."""
        if self.output_path is not None:
            self.output_path.write_bytes(b"partial WAV")
        return self

    def poll(self) -> int:
        """Return the OS status without timer-dependent progress."""
        return self.exit_code

    def terminate(self) -> None:
        """Satisfy the cancellable process boundary."""

    def kill(self) -> None:
        """Satisfy the cancellable process boundary."""


def test_prepare_audio_cancellable_path_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare_audio with should_cancel uses async ffmpeg path and succeeds."""
    monkeypatch.setattr(audio_io, "ffmpeg", _AsyncFfmpeg(0))

    inp = tmp_path / "in.wav"
    # No need to exist; pipeline does not read input when poll()=0 immediately
    out = audio_io.prepare_audio(inp, tmp_path, should_cancel=lambda: False)
    assert out.exists() is False or out.suffix == ".wav"


@pytest.mark.integration
@pytest.mark.parametrize(
    ("platform", "exit_code", "status"),
    [
        ("win32", 0xC000012D, "STATUS_COMMITMENT_LIMIT"),
        ("win32", 0xC0000017, "STATUS_NO_MEMORY"),
        ("win32", 0xFFFFFFF4, "ENOMEM"),
        ("win32", -1073741523, "STATUS_COMMITMENT_LIMIT"),
        ("win32", -1073741801, "STATUS_NO_MEMORY"),
        ("win32", -12, "ENOMEM"),
        ("win32", 1, None),
        ("linux", -12, None),
        ("linux", -9, None),
        ("linux", 137, None),
    ],
)
def test_ffmpeg_failure_reaches_pool_classification_and_removes_partial_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    exit_code: int,
    status: str | None,
) -> None:
    """Catch generic failure for loader OOM or OOM classification for a signal."""
    monkeypatch.setattr(audio_io, "sys", SimpleNamespace(platform=platform))
    ffmpeg = _AsyncFfmpeg(exit_code)
    monkeypatch.setattr(audio_io, "ffmpeg", ffmpeg)
    source = tmp_path / "incident.wav"
    item = WorkItem("job", "job:1", source, "ru", 1.0, None)
    runtime = EngineRuntime()
    runtime.install(monkeypatch, tmp_path)
    try:
        handle = runtime.open(spec("gigaam-e2e-ctc", "gpu"))
        handle.engine.submit(item)
        result = handle.result()
        assert result.status == "error"
        assert result.attempt_id == item.attempt_id
        assert result.error_kind == ("memory_limit" if status else "transcribe_failed")
        assert str(exit_code) in (result.error_detail or "")
        if status is not None:
            assert status in (result.error_detail or "")
        assert ffmpeg.output_path is not None
        assert not ffmpeg.output_path.exists()
    finally:
        runtime.release_and_close()
    assert not list(tmp_path.glob("askvlm-pool-*"))

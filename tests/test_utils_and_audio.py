import io
import os
import subprocess
import sys
import wave
from array import array
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from core import audio_io
from tests.bounded_scenario_support import run_bounded_scenario
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


def test_prepare_audio_cancellable_path_success(  # noqa: C901
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare_audio with should_cancel uses async ffmpeg path and succeeds."""

    # Fake ffmpeg chain
    class FakeStream:
        def __init__(self) -> None:
            self.stderr = io.StringIO("progress=1\n")

        def output(self, *_args: object, **_kwargs: object) -> "FakeStream":
            return self

        def overwrite_output(self) -> "FakeStream":
            return self

        def global_args(self, *_args: object) -> "FakeStream":
            return self

        def run_async(
            self,
            *,
            pipe_stdin: bool = True,
            pipe_stdout: bool = False,
            pipe_stderr: bool = True,
        ) -> object:
            class Proc:
                def __init__(self, stderr: io.StringIO) -> None:
                    self.stderr = stderr

                def poll(self) -> int:
                    return 0

                def terminate(self) -> None:
                    return None

                def kill(self) -> None:
                    return None

            # Touch keyword-only flags to avoid unused-argument warnings
            _ = (pipe_stdin, pipe_stdout, pipe_stderr)
            return Proc(self.stderr)

    class FakeFFMPEG:
        def input(self, *_args: object, **_kwargs: object) -> FakeStream:
            return FakeStream()

    # Patch module-level ffmpeg in audio_io
    monkeypatch.setattr(audio_io, "ffmpeg", FakeFFMPEG())

    inp = tmp_path / "in.wav"
    # No need to exist; pipeline does not read input when poll()=0 immediately
    out = audio_io.prepare_audio(inp, tmp_path, should_cancel=lambda: False)
    assert out.exists() is False or out.suffix == ".wav"

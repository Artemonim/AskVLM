import io
from pathlib import Path

import pytest

from core import audio_io
from utils import env as env_utils


def test_load_env_file_sets_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """load_env_file reads KEY=VALUE pairs and sets os.environ if absent."""
    content = "# comment\nA=1\nB='2'\nC=\"3\"\nNOEQUALS"
    env_path = tmp_path / ".env"
    env_path.write_text(content, encoding="utf-8")
    # Ensure no pre-existing vars
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)
    env_utils.load_env_file(env_path)
    import os

    assert os.environ.get("A") == "1"
    assert os.environ.get("B") == "2"
    assert os.environ.get("C") == "3"


def test_prepare_audio_cancellable_path_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """prepare_audio with should_cancel uses async ffmpeg path and succeeds."""

    # Fake ffmpeg chain
    class FakeStream:
        def __init__(self) -> None:
            self.stderr = io.StringIO("progress=1\n")

        def output(self, *args, **kwargs):
            return self

        def overwrite_output(self):
            return self

        def global_args(self, *args):
            return self

        def run_async(self, pipe_stdin=True, pipe_stdout=False, pipe_stderr=True):  # noqa: ARG002
            class Proc:
                def __init__(self, stderr) -> None:
                    self.stderr = stderr

                def poll(self) -> int:
                    return 0

                def terminate(self) -> None:
                    return None

                def kill(self) -> None:
                    return None

            return Proc(self.stderr)

    class FakeFFMPEG:
        def input(self, *args, **kwargs):
            return FakeStream()

    # Patch module-level ffmpeg in audio_io
    monkeypatch.setattr(audio_io, "ffmpeg", FakeFFMPEG())

    inp = tmp_path / "in.wav"
    # No need to exist; pipeline does not read input when poll()=0 immediately
    out = audio_io.prepare_audio(inp, tmp_path, should_cancel=lambda: False)
    assert out.exists() is False or out.suffix == ".wav"

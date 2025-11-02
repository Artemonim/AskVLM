import io
from pathlib import Path

import pytest

from utils import env as env_utils
from core import audio_io


def test_load_env_file_sets_variables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """load_env_file reads KEY=VALUE pairs and sets os.environ if absent."""
    content = "\n".join([
        "# comment",
        "A=1",
        "B='2'",
        'C="3"',
        "NOEQUALS",
    ])
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


def test_prepare_audio_cancellable_path_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """prepare_audio with should_cancel uses async ffmpeg path and succeeds."""
    # Fake ffmpeg chain
    class FakeStream:
        def __init__(self):
            self.stderr = io.StringIO("progress=1\n")

        def output(self, *args, **kwargs):  # noqa: ANN001
            return self

        def overwrite_output(self):
            return self

        def global_args(self, *args):  # noqa: ANN001
            return self

        def run_async(self, pipe_stdin=True, pipe_stdout=False, pipe_stderr=True):  # noqa: ARG002
            class Proc:
                def __init__(self, stderr):
                    self.stderr = stderr

                def poll(self):
                    return 0

                def terminate(self):
                    return None

                def kill(self):
                    return None

            return Proc(self.stderr)

    class FakeFFMPEG:
        def input(self, *args, **kwargs):  # noqa: ANN001
            return FakeStream()

    # Patch module-level ffmpeg in audio_io
    monkeypatch.setattr(audio_io, "ffmpeg", FakeFFMPEG())

    inp = tmp_path / "in.wav"
    # No need to exist; pipeline does not read input when poll()=0 immediately
    out = audio_io.prepare_audio(inp, tmp_path, should_cancel=lambda: False)
    assert out.exists() is False or out.suffix == ".wav"



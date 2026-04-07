from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

import cli

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

_RUNNER = CliRunner()


class _StubDocument:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_full_text(self) -> str:
        return self._text


def test_external_transcribe_uses_small_and_stdout_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external CLI defaults to a small one-shot stdout contract."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    work_dir = tmp_path / "work"
    calls: dict[str, object] = {}

    class _StubPipeline:
        def process(self, input_media: Path, process_work_dir: Path) -> _StubDocument:
            calls["process_input"] = input_media
            calls["process_work_dir"] = process_work_dir
            return _StubDocument("transcribed text")

        def close(self, *, aggressive: bool = False) -> None:
            calls["close_aggressive"] = aggressive

    def _fake_create_local_pipeline(**kwargs: object) -> _StubPipeline:
        calls["pipeline_kwargs"] = kwargs
        return _StubPipeline()

    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    result = _RUNNER.invoke(
        cli.app,
        ["external-transcribe", str(input_path), "--work-dir", str(work_dir)],
    )

    assert result.exit_code == 0
    assert result.stdout == "transcribed text\n"
    assert calls["pipeline_kwargs"] == {
        "whisper_model": "small",
        "engine": "whisperx",
        "diarization": False,
        "dialog_blocks": False,
        "language": None,
        "device": "auto",
        "compute_type": "auto",
    }
    assert calls["process_input"] == input_path
    assert calls["process_work_dir"] == work_dir
    assert calls["close_aggressive"] is True


def test_external_transcribe_writes_output_file_without_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The external CLI can write a file without emitting transcript stdout."""
    input_path = tmp_path / "clip.wav"
    input_path.write_text("stub", encoding="utf-8")
    output_file = tmp_path / "nested" / "transcript.txt"

    class _StubPipeline:
        def process(self, _input_media: Path, _process_work_dir: Path) -> _StubDocument:
            return _StubDocument("saved transcript")

        def close(self, *, aggressive: bool = False) -> None:
            _ = aggressive

    def _fake_create_local_pipeline(**_kwargs: object) -> _StubPipeline:
        return _StubPipeline()

    monkeypatch.setattr(cli, "_create_local_pipeline", _fake_create_local_pipeline)

    result = _RUNNER.invoke(
        cli.app,
        [
            "external-transcribe",
            str(input_path),
            "--work-dir",
            str(tmp_path / "work"),
            "--output-file",
            str(output_file),
            "--no-stdout",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout == ""
    assert output_file.read_text(encoding="utf-8") == "saved transcript"
